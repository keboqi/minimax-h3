"""Dub & Translate: turn a source into timed lines, then voice them with Drama 3 and remix.

Analysis extracts the soundtrack, optionally separates vocals from the background,
and transcribes with Gemini or locally: OmniVAD + Qwen3-ASR + ForcedAligner, or
MOSS-Transcribe-Diarize on SGLang-Omni, which also tells speakers apart; Hy-MT then
translates. Dubbing clones each line's voice from the original audio, fits the take to
its slot, lays the lines on the timeline and mixes them over the background. Takes
are content-keyed, so an unchanged line is never generated twice.

Each step is a generator of status strings that ends with one result object.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import contextvars
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator
import uuid
import wave

import numpy as np

from h3_app.catalog import VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.media_tools import probe_audio_seconds, run_media_process
from h3_app.provenance import read_snapshot, write_snapshot

from . import gemini, moss, refinement, segments as segment_tools, session as sessions, tools, tools_worker

RATE = 44100
PIPELINE_LOCAL = "Local: Qwen3-ASR + aligner + Hy-MT"
PIPELINE_MOSS = "Local: MOSS Transcribe + Diarize + Hy-MT"
PIPELINE_GEMINI = "Gemini (cloud)"
PIPELINES = (PIPELINE_LOCAL, PIPELINE_MOSS, PIPELINE_GEMINI)
# Transcriptions that tell speakers apart, so each speaker can get its own voice and volume.
SPEAKER_PIPELINES = frozenset({PIPELINE_MOSS, PIPELINE_GEMINI})
TRANSLATOR_HYMT = "Hy-MT 1.8B (local)"
TRANSLATOR_GEMINI = "Gemini"
TRANSLATORS = (TRANSLATOR_HYMT, TRANSLATOR_GEMINI)
SEPARATOR_MODELS = tuple(tools.layout.SEPARATOR_MODELS)
SOURCE_LANGUAGES = ("Auto-detect", "English", "Chinese", "Cantonese", "Japanese", "Korean", "French", "German",
                    "Spanish", "Portuguese", "Italian", "Russian", "Arabic", "Hindi", "Indonesian", "Thai",
                    "Vietnamese", "Turkish", "Malay", "Dutch")
TARGET_LANGUAGES = ("English", "Chinese", "Japanese", "Korean", "Spanish", "French", "German", "Portuguese",
                    "Italian", "Russian", "Arabic", "Indonesian", "Vietnamese", "Thai", "Cantonese")
VOICE_LINE = "Each line's own voice"
VOICE_SPEAKER = "One voice per speaker"
VOICE_MODES = (VOICE_LINE, VOICE_SPEAKER)
# Match stretches every take to its line's original length with ffmpeg, as index-tts-vllm's
# duration control does; Natural only speeds up long takes, by at most the chosen limit.
TIMING_MATCH = "Match the original timing"
TIMING_NATURAL = "Natural pace"
TIMINGS = (TIMING_MATCH, TIMING_NATURAL)
# Matched takes changed more than this are reported: their translation is much longer or shorter.
TEMPO_WARNING = (0.75, 1.6)
# Drama 3 sampling (temperature, top_p, top_k); Soundstage offers the same three.
VOICE_PRESETS = {"Strict": (0.7, 0.7, 30), "Balance": (1.0, 0.9, 0), "Expressive": (2.0, 1.0, 30)}
# A line's own clip shorter than this is a weak reference; its speaker's best clip is used instead.
MIN_REFERENCE_SECONDS = 2.0
REFERENCE_RANGE = (3.0, 15.0)
REFERENCE_PAD = 0.08
LINE_GAP = 0.05
GEMINI_TRANSLATE_BATCH = 60
# Each fitted take is one ffmpeg run; these many run at once while the dub is laid out.
TIMING_WORKERS = max(1, min(8, os.cpu_count() or 4))
# Hy-MT sees this many earlier lines of the dialogue as background for each line.
CONTEXT_LINES = 2
# A speaker's voice setting: empty clones it from the source, otherwise a saved Drama 3 speaker's key.
ORIGINAL_VOICE = ""
SPEAKER_VOLUME = (0.0, 200.0)
# The result player's separate tracks sit beside the outputs in this folder, which Media skips.
PLAYER_DIR = ".dub-player"
# Constant bitrate: browsers seek a long variable-bitrate MP3 by estimate and land seconds away
# from the reported time, so the dub would drift from its subtitles after every seek.
PLAYER_MP3 = ("-c:a", "libmp3lame", "-b:a", "128k")
# Session key for lines whose take failed its check or could not be generated, shown in Review lines.
TAKE_PROBLEMS = "take_problems"
OUTPUT_MP3 = ("-c:a", "libmp3lame", "-b:a", "192k")


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyzeRequest:
    source: str
    pipeline: str = PIPELINE_LOCAL
    target_language: str = "English"
    source_language: str = "Auto-detect"
    translate: bool = True
    translator: str = TRANSLATOR_HYMT
    separate: bool = True
    separator_model: str = "Balanced (BS-Roformer)"
    aligner: bool = True
    merge_gap: float = 0.35
    gemini_model: str = ""
    api_key: str = ""
    ignore_non_speech: bool = True
    notes: str = ""
    # Take the stems and transcript from an earlier analysis of the same file when its settings match.
    reuse: bool = True
    refine_timestamps: bool = True


@dataclass(frozen=True)
class DubRequest:
    session_id: str
    voice_mode: str = VOICE_LINE
    preset: str = "Balance"
    timing: str = TIMING_MATCH
    parallel: int = 16
    max_speed: float = 1.35
    speech_volume: float = 100.0
    background_volume: float = 100.0
    keep_original_track: bool = True
    regenerate: tuple[int, ...] = ()


@dataclass
class Analysis:
    session: sessions.Session
    message: str


@dataclass
class Player:
    """What the result player plays: the speech tracks and background apart, so either voice and the
    music can be switched while it plays, and each line's original and translated subtitle times."""

    video: Path | None
    dub: Path
    original: Path
    music: Path | None
    duration: float
    lines: list[dict]
    target_language: str = ""
    source_language: str = ""


@dataclass
class Dub:
    session: sessions.Session
    audio: Path
    video: Path | None
    subtitles: list[Path]
    message: str
    player: Player | None = None


@dataclass
class LinePreview:
    rate: int
    samples: np.ndarray
    message: str


@dataclass
class _Line:
    index: int
    segment: dict
    reference: Path
    transcript: str
    key: str = ""
    take: np.ndarray | None = None
    note: str = ""
    speed: float = 1.0


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def blocking(label: str, work: Callable[[Callable[[str], None]], Any]) -> Iterator[str]:
    """Drama 3's heartbeat runner as status strings; returns ``work(report)``'s result."""
    from h3_app.drama3.serving.speech import run_blocking

    updates = run_blocking(label, work)
    while True:
        try:
            update = next(updates)
        except StopIteration as done:
            return done.value
        yield update.status


def _ffmpeg(arguments: list[str], target: Path, *, timeout: float = 3600, action: str) -> Path:
    partial = target.with_name(f"{target.stem}.{uuid.uuid4().hex[:8]}.partial{target.suffix}")
    process = run_media_process(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments, str(partial)],
                                timeout=timeout, partial_outputs=(partial,))
    if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise H3Error(f"{action} failed: {process.stderr.strip()[-600:] or 'ffmpeg reported no output'}")
    partial.replace(target)
    return target


def _to_pcm(source: Path, target: Path, *, channels: int = 2, action: str) -> Path:
    return _ffmpeg(["-i", str(source), "-vn", "-ac", str(channels), "-ar", str(RATE), "-c:a", "pcm_s16le"],
                   target, action=action)


def read_slice(path: Path, start: float, end: float) -> np.ndarray:
    """Mono float32 samples of a 16-bit PCM WAV between two times, without reading the whole file."""
    with wave.open(str(path), "rb") as source:
        rate, channels, frames = source.getframerate(), source.getnchannels(), source.getnframes()
        first = max(0, min(frames, int(round(start * rate))))
        last = max(first, min(frames, int(round(end * rate))))
        source.setpos(first)
        pcm = np.frombuffer(source.readframes(last - first), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    return pcm if rate == RATE else _resample(pcm, rate, RATE)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    from h3_app.drama3.serving.audio import resample_audio

    return resample_audio(samples, source_rate, target_rate)


def _write_wav(samples: np.ndarray, rate: int, target: Path) -> Path:
    from h3_app.drama3.serving.audio import encode_wav

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode_wav(samples, rate))
    return target


def _language(value: str) -> str:
    return "" if not value or value == "Auto-detect" else value


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS | {".m4v"}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(request: AnalyzeRequest, *, runtime, free_gpu, hf_token=None) -> Iterator[str | Analysis]:
    if request.pipeline not in PIPELINES:
        raise H3Error(f"Unknown transcription pipeline: {request.pipeline}")
    target_language = str(request.target_language or "").strip()
    if request.translate and not target_language:
        raise H3Error("Choose the language to dub into.")
    if request.refine_timestamps and request.pipeline == PIPELINE_LOCAL and not request.aligner:
        raise H3Error("Enable Qwen3 word timing before refining it with Align.")
    session = sessions.create(runtime, request.source, video=is_video(request.source))
    session.data.update(pipeline=request.pipeline, target_language=target_language,
                        source_language=_language(request.source_language))
    yield "Extracting the soundtrack"
    audio = _to_pcm(session.path("source"), session.root / "audio.wav", action="Extracting the soundtrack")
    session.set_file("audio", audio)
    session.data["duration"] = round(probe_audio_seconds(audio) or 0.0, 3)
    session.save()
    earlier = sessions.same_source(session) if request.reuse else []
    reused = []

    if request.separate:
        donor = _earlier_stems(earlier, request.separator_model)
        if donor is not None:
            yield "Reusing the separated vocals from an earlier analysis of this file"
            for key in ("vocals", "backing"):
                session.set_file(key, _link_or_copy(donor.path(key), session.root / f"{key}.wav"))
            reused.append("separated vocals")
        else:
            reply = yield from blocking(
                "Separating vocals from the background",
                lambda report: tools.run("separate", {"input": str(audio), "output_dir": str(session.root / "stems"),
                                                      "model": request.separator_model},
                                         runtime, free_gpu=free_gpu, hf_token=hf_token, report=report))
            for key in ("vocals", "backing"):
                session.set_file(key, _to_pcm(Path(reply[key]), session.root / f"{key}.wav",
                                              action=f"Saving the {key} stem"))
            shutil.rmtree(session.root / "stems", ignore_errors=True)  # converted copies replace the raw stems
        session.data["separator_model"] = request.separator_model
        session.save()

    key = _transcript_key(request)
    transcript = _earlier_transcript(earlier, key)
    speech = session.speech_audio
    groups = []
    if transcript is not None:
        yield "Reusing the transcript from an earlier analysis of this file"
        reused.append("transcript")
    elif request.pipeline == PIPELINE_MOSS:
        duration = float(session.data.get("duration") or 0.0)
        found = yield from blocking(
            "Finding speakers and transcribing with MOSS",
            lambda report: moss.transcribe(runtime, speech, duration=duration, free_gpu=free_gpu,
                                           hf_token=hf_token, report=report))
        found = moss_lines(found)
        if not found:
            raise H3Error("No speech was found in this source.")
        transcript = {"segments": found, "speakers": {speaker: "" for speaker in
                                                      dict.fromkeys(s["speaker"] for s in found)}}
    elif request.pipeline == PIPELINE_GEMINI:
        upload = _ffmpeg(["-i", str(speech), "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k"],
                         session.root / "speech.mp3", action="Preparing audio for Gemini")
        found, speakers, _raw = yield from blocking(
            f"Transcribing{' and translating' if request.translate else ''} with {request.gemini_model}",
            lambda _report: gemini.transcribe(upload, model=request.gemini_model, api_key=request.api_key,
                                              target_language=target_language, translate=request.translate,
                                              ignore_non_speech=request.ignore_non_speech,
                                              custom_prompt=""))
        transcript = {"segments": found, "speakers": speakers}
    else:
        reply = yield from blocking(
            "Finding speech and transcribing with Qwen3-ASR",
            lambda report: tools.run("transcribe", {"audio": str(speech), "language": _language(
                request.source_language) or None, "aligner": bool(request.aligner),
                "merge_gap": float(request.merge_gap), "word_details": request.refine_timestamps},
                runtime, free_gpu=free_gpu, hf_token=hf_token,
                report=report))
        if not reply.get("segments"):
            raise H3Error("No speech was found in this source.")
        transcript = {
            "segments": segment_tools.normalize({"start": s["start"], "end": s["end"], "source": s["text"]}
                                                for s in reply["segments"]),
            "speakers": {segment_tools.DEFAULT_SPEAKER: "Single voice (local transcription)"},
            "language": reply.get("language") or "",
        }
        groups = reply.get("groups") or []
    if request.refine_timestamps and request.pipeline == PIPELINE_LOCAL and "refinement" not in transcript:
        transcript = yield from _refine_timestamps(
            session, transcript, groups, runtime=runtime)
    # Kept as transcribed, before edits and later translations, for the next analysis of this file.
    session.data["transcript"] = {**transcript, "key": key}
    session.segments, session.data["speakers"] = transcript["segments"], dict(transcript["speakers"])
    if transcript.get("language") and not session.data.get("source_language"):
        session.data["source_language"] = transcript["language"]
    session.save()
    if request.translate:
        # Gemini may already have translated; the translator fills every line still without one.
        missing = [i for i, s in enumerate(session.segments) if s["source"] and not s["translation"]]
        if missing:
            yield from _translate(session, missing, request, runtime=runtime, free_gpu=free_gpu,
                                  hf_token=hf_token)
    count = len(session.segments)
    speakers = len({s["speaker"] for s in session.segments})
    translated = sum(bool(s["translation"]) for s in session.segments)
    message = (f"Found {count} line(s) from {speakers} speaker(s); {translated} translated into "
               f"{target_language or 'nothing yet'}. Review the lines, then dub.")
    if reused:
        message += (f" Reused the {' and '.join(reused)} from an earlier analysis of this file; untick "
                    "Reuse earlier results to redo them.")
    if transcript.get("refinement"):
        message += " " + transcript["refinement"]["summary"]
    yield Analysis(session, message + length_advice(session.segments))


def _transcript_key(request: AnalyzeRequest) -> str:
    """The settings a transcript depends on, beyond the source file itself."""
    settings: dict[str, Any] = {"pipeline": request.pipeline,
                                "separator": request.separator_model if request.separate else ""}
    if request.pipeline == PIPELINE_LOCAL and request.refine_timestamps:
        settings.update(refinement=refinement.REVISION, language=_language(request.source_language))
    if request.pipeline == PIPELINE_MOSS:
        settings.update(model=f"{moss.MODEL_REPO}@{moss.MODEL_REVISION[:12]}")
    elif request.pipeline == PIPELINE_GEMINI:
        settings.update(model=request.gemini_model, ignore_non_speech=bool(request.ignore_non_speech),
                        target=str(request.target_language or "").strip() if request.translate else "")
    else:
        settings.update(language=_language(request.source_language), aligner=bool(request.aligner),
                        merge_gap=round(float(request.merge_gap), 3))
    return json.dumps(settings, sort_keys=True)


def _refine_timestamps(session, transcript, groups, *, runtime):
    """Preserve the baseline and per-word evidence; any failed group keeps its original line cuts."""
    baseline = transcript["segments"]
    audit = {"revision": refinement.REVISION, "original_segments": baseline, "groups": [
        {"start": group["start"], "end": group["end"], "language": group.get("language", ""),
         "original_words": group.get("words", []), "status": "original"} for group in groups]}
    try:
        refinement.check_runtime()
        if not groups:
            raise H3Error("The transcription returned no acoustic word timings to refine.")
        group_lines = segment_tools.normalize({**line, "source": line["text"]}
                                              for group in groups for line in group["lines"])
        if group_lines != baseline:
            raise H3Error("The word-timing groups do not match the original transcript lines.")
        for group in groups:
            group["language_code"] = refinement.language_code(group.get("language"))
        if not any(group["language_code"] and group.get("words") for group in groups):
            raise H3Error("No speech group has acoustic word timing in one of Align's supported languages.")
        results = yield from blocking(
            "Refining speech boundaries with Align",
            lambda report: refinement.refine(session.speech_audio, groups, tools_worker.models_root(
                runtime.comfy_dir / "models"), timeout=runtime.generation_timeout, report=report))
        if not isinstance(results, list) or len(results) != len(groups):
            raise H3Error("Align returned invalid speech groups.")
        lines, accepted, corrected = [], 0, 0
        for index, (group, result) in enumerate(zip(groups, results)):
            try:
                parts, evidence = refinement.apply_group(group, result, session.data["duration"])
                accepted += 1
                corrected += evidence["status"] == "refined"
            except (ValueError, KeyError, TypeError) as exc:
                parts = group["lines"]
                evidence = {"status": "original", "error": str(exc), "original_words": group.get("words", [])}
            lines += [{**part, "source": part["text"]} for part in parts]
            audit["groups"][index].update(evidence)
        transcript = {**transcript, "segments": segment_tools.normalize(lines)}
        audit["summary"] = (f"Used acoustic word timing for {accepted}/{len(groups)} speech groups "
                            f"({corrected} refined by Align); the others kept their original cuts.")
    except (H3Error, OSError, ValueError) as exc:
        audit["summary"] = f"Align refinement skipped; original timing kept. {exc}"
    return {**transcript, "refinement": audit}


def moss_lines(found: list[dict]) -> list[dict]:
    """MOSS's speaker turns as subtitle-sized lines: long turns are split at sentence ends."""
    lines = []
    for turn in found:
        start, end, text = float(turn["start"]), float(turn["end"]), str(turn.get("text") or "")
        parts = ([{"start": start, "end": end, "text": text}]
                 if end - start <= tools_worker.MAX_LINE_SECONDS and len(text) <= tools_worker.MAX_LINE_CHARS
                 else tools_worker.proportional_lines(start, end, text))
        lines += [{"start": part["start"], "end": part["end"], "speaker": turn.get("speaker"), "source": part["text"]}
                  for part in parts]
    return segment_tools.normalize(lines)


def _earlier_stems(earlier: list[sessions.Session], model: str) -> sessions.Session | None:
    """The latest earlier session whose stems came from ``model`` and are still on disk."""
    for candidate in earlier:
        paths = [candidate.path(key) for key in ("vocals", "backing")]
        if candidate.data.get("separator_model") == model and all(path and path.is_file() for path in paths):
            return candidate
    return None


def _earlier_transcript(earlier: list[sessions.Session], key: str) -> dict | None:
    for candidate in earlier:
        transcript = candidate.data.get("transcript")
        if isinstance(transcript, dict) and transcript.get("key") == key and transcript.get("segments"):
            return transcript
    return None


def _link_or_copy(source: Path, target: Path) -> Path:
    """Share a finished stem with a hard link (stems are never rewritten), else copy it."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _translate(session: sessions.Session, indices: list[int], request, *, runtime, free_gpu,
               hf_token=None) -> Iterator[str]:
    lines = session.segments
    texts = [lines[i]["source"] for i in indices]
    target = session.data.get("target_language") or request.target_language
    if request.translator == TRANSLATOR_GEMINI:
        # Each line's time on screen, so Gemini can word its translation to fit it.
        seconds = [lines[i]["end"] - lines[i]["start"] for i in indices]
        translated = []
        for first in range(0, len(texts), GEMINI_TRANSLATE_BATCH):
            batch = texts[first:first + GEMINI_TRANSLATE_BATCH]
            translated += yield from blocking(
                f"Translating lines {first + 1}-{first + len(batch)} of {len(texts)} with Gemini",
                lambda _report, batch=batch, first=first: gemini.translate(
                    batch, model=request.gemini_model, api_key=request.api_key, target_language=target,
                    source_language=session.data.get("source_language", ""), notes=request.notes,
                    seconds=seconds[first:first + len(batch)]))
    else:
        # Hy-MT translates one line at a time; the lines before it give it the conversation so far.
        contexts = ["\n".join(s["source"] for s in lines[max(0, i - CONTEXT_LINES):i] if s["source"])
                    for i in indices]
        reply = yield from blocking(
            f"Translating {len(texts)} line(s) into {target} with Hy-MT",
            lambda report: tools.run("translate", {"texts": texts, "contexts": contexts, "target_language": target,
                                                   "notes": request.notes},
                                     runtime, free_gpu=free_gpu, hf_token=hf_token, report=report))
        translated = reply["texts"]
    # The editor may have saved lines meanwhile: apply each translation to the line that still
    # has the same start and original text, on top of the latest saved lines.
    results = {(lines[i]["start"], lines[i]["source"]): text for i, text in zip(indices, translated)}

    def apply(latest: sessions.Session) -> None:
        updated = latest.segments
        for line in updated:
            text = results.get((line["start"], line["source"]))
            if text is not None:
                line["translation"] = text
        latest.segments = updated
        latest.data["target_language"] = target

    sessions.modify(session, apply)


def retranslate(session_id: str, indices: list[int] | None, request, *, runtime, free_gpu,
                hf_token=None) -> Iterator[str | Analysis]:
    """Translate the chosen lines again; without a choice, only lines that have no translation.

    A new target language makes every saved translation stale, so then every line is translated.
    """
    session = sessions.load(runtime, session_id)
    previous = session.data.get("target_language") or ""
    if request.target_language:
        session.data["target_language"] = request.target_language
    changed = bool(request.target_language) and request.target_language != previous
    lines = session.segments
    if indices is None:
        indices = [i for i, s in enumerate(lines) if s["source"] and (changed or not s["translation"])]
    indices = [i for i in indices if lines[i]["source"]]
    if not indices:
        target = session.data.get("target_language") or "the target language"
        yield Analysis(session, f"Every line already has a translation into {target}. "
                                "Select lines to translate them again.")
        return
    yield from _translate(session, indices, request, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token)
    yield Analysis(session, f"Translated {len(indices)} line(s) into {session.data['target_language']}."
                            + length_advice(session.segments))


def long_lines(lines: list[dict]) -> list[int]:
    """Lines to dub whose translation would need more than ``TEMPO_WARNING``'s speed-up to fit the line.

    Uses Drama 3's estimate of spoken length, so it can be checked before anything is voiced.
    """
    from h3_app.drama3.serving.planner import expected_seconds

    return [index for index, line in enumerate(lines)
            if line["translation"] and line.get("dub", True)
            and expected_seconds(line["translation"]) > (line["end"] - line["start"]) * TEMPO_WARNING[1]]


def length_advice(lines: list[dict]) -> str:
    """A sentence naming the translations that look too long to say in time; empty when none do."""
    found = long_lines(lines)
    if not found:
        return ""
    numbers = ", ".join(str(index + 1) for index in found[:20]) + (", …" if len(found) > 20 else "")
    return (f" {len(found)} translation(s) look too long to say in their line's time: {numbers}. Shorten them, "
            "or translate them again with Gemini, which is told how long each line lasts.")


# ---------------------------------------------------------------------------
# Dubbing
# ---------------------------------------------------------------------------


def _reference_clip(session: sessions.Session, segment: dict) -> Path:
    """The line's original audio, named by its times so an edited start or end cuts a new clip."""
    target = session.root / "references" / f"clip_{segment['start']:.3f}_{segment['end']:.3f}.wav"
    if not target.is_file():
        start = max(0.0, segment["start"] - REFERENCE_PAD)
        _write_wav(read_slice(session.speech_audio, start, segment["end"] + REFERENCE_PAD), RATE, target)
    return target


def speaker_reference(lines: list[dict], speaker: str) -> int | None:
    """The index of the speaker's best reference line: the longest within 3–15 s, else the closest to it."""
    own = [(i, s) for i, s in enumerate(lines) if s["speaker"] == speaker and s["source"]]
    if not own:
        return None
    low, high = REFERENCE_RANGE
    fitting = [(i, s) for i, s in own if low <= s["end"] - s["start"] <= high]
    if fitting:
        return max(fitting, key=lambda item: item[1]["end"] - item[1]["start"])[0]
    return min(own, key=lambda item: min(abs(item[1]["end"] - item[1]["start"] - low),
                                         abs(item[1]["end"] - item[1]["start"] - high)))[0]


def speaker_settings_available(session: sessions.Session) -> bool:
    """Per-speaker settings need real speakers, which only MOSS and Gemini tell apart."""
    return session.data.get("pipeline") in SPEAKER_PIPELINES


def _speaker_volume(value: Any) -> float:
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return 100.0
    return min(SPEAKER_VOLUME[1], max(SPEAKER_VOLUME[0], volume)) if volume == volume else 100.0


def speaker_settings(session: sessions.Session) -> dict[str, dict]:
    """``{speaker: {"voice": key or ORIGINAL_VOICE, "volume": percent}}``; empty without real speakers."""
    stored = session.data.get("speaker_settings")
    if not speaker_settings_available(session) or not isinstance(stored, dict):
        return {}
    return {str(speaker): {"voice": str(entry.get("voice") or ORIGINAL_VOICE),
                           "volume": _speaker_volume(entry.get("volume"))}
            for speaker, entry in stored.items() if isinstance(entry, dict)}


def session_speakers(session: sessions.Session) -> list[str]:
    """The speakers of the session's lines, in order of first appearance."""
    return list(dict.fromkeys(segment["speaker"] for segment in session.segments))


def set_speaker(runtime, session_id: str, speaker: str, *, voice: str, volume: float) -> sessions.Session:
    """Save one speaker's voice and volume; the default voice at 100% clears its entry."""
    from h3_app.drama3.speakers import load_speaker

    session = sessions.load(runtime, session_id)
    if not speaker_settings_available(session):
        raise H3Error("Per-speaker voices need a session transcribed with MOSS or Gemini, which tell speakers "
                      "apart.")
    speaker, voice = str(speaker or "").strip(), str(voice or ORIGINAL_VOICE)
    if speaker not in session_speakers(session):
        raise H3Error("Choose one of this session's speakers.")
    if voice != ORIGINAL_VOICE:
        load_speaker(voice, runtime)  # still saved
    entry = {"voice": voice, "volume": _speaker_volume(volume)}

    def change(latest: sessions.Session) -> None:
        settings = latest.data.setdefault("speaker_settings", {})
        if entry == {"voice": ORIGINAL_VOICE, "volume": 100.0}:
            settings.pop(speaker, None)
        else:
            settings[speaker] = entry

    return sessions.modify(session, change)


def _plan(session: sessions.Session, indices: list[int], voice_mode: str, runtime=None) -> list[_Line]:
    """Each line to voice with its reference: a saved voice chosen for its speaker, else a clip of the source."""
    from h3_app.drama3.speakers import load_speaker

    lines = session.segments
    planned = []
    speaker_refs: dict[str, tuple[Path, str]] = {}
    chosen_voices = {speaker: entry["voice"] for speaker, entry in speaker_settings(session).items()
                     if entry["voice"] != ORIGINAL_VOICE}
    saved_refs: dict[str, tuple[Path, str]] = {}
    for index in indices:
        segment = lines[index]
        if not segment["translation"]:
            continue
        own_length = segment["end"] - segment["start"]
        voice = chosen_voices.get(segment["speaker"])
        if voice:
            if voice not in saved_refs:
                try:
                    preset = load_speaker(voice, runtime)
                except H3Error as exc:
                    raise H3Error(f"{exc} Choose another voice for {segment['speaker']} under Speakers.") from exc
                saved_refs[voice] = (preset.audio, preset.transcript)
            reference = saved_refs[voice]
        elif voice_mode == VOICE_LINE and segment["source"] and own_length >= MIN_REFERENCE_SECONDS:
            reference = (_reference_clip(session, segment), segment["source"])
        else:
            speaker = segment["speaker"]
            if speaker not in speaker_refs:
                best = speaker_reference(lines, speaker)
                if best is None:
                    best = max(range(len(lines)), key=lambda i: lines[i]["end"] - lines[i]["start"]
                               if lines[i]["source"] else -1)
                speaker_refs[speaker] = (_reference_clip(session, lines[best]), lines[best]["source"])
            reference = speaker_refs[speaker]
        planned.append(_Line(index, segment, *reference))
    return planned


def slot_end(lines: list[dict], index: int, duration: float) -> float:
    """Where a line's take must end: just before the next line starts, never before its own end."""
    segment = lines[index]
    later = [s["start"] for s in lines[index + 1:] if s["start"] > segment["start"]]
    limit = (min(later) - LINE_GAP) if later else (duration or segment["end"] + 2.0)
    return max(segment["end"], limit)


def atempo_chain(tempo: float) -> str:
    """An atempo filter for any tempo: each stage stays within 0.5–2x, as in index-tts-vllm."""
    tempo = max(0.01, float(tempo))
    factors = []
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    factors.append(tempo)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _stretch(samples: np.ndarray, rate: int, tempo: float, cache: Path | None = None) -> np.ndarray:
    """Play speech ``tempo`` times faster (below 1: slower), pitch kept, with ffmpeg's atempo.

    With a ``cache`` folder the result is kept there under a hash of the take and the
    tempo, so a later dub that fits the same take the same way skips ffmpeg.
    """
    from h3_app.drama3.serving.audio import decode_wav

    cached = None
    if cache is not None:
        digest = hashlib.blake2b(np.ascontiguousarray(samples, dtype=np.float32).tobytes(), digest_size=16)
        digest.update(f"{rate}:{tempo:.6f}".encode())
        cached = cache / f"{digest.hexdigest()}.wav"
        try:
            timed = decode_wav(cached.read_bytes())[0]
            os.utime(cached)  # marks it as used by this dub
            return timed
        except (OSError, ValueError, EOFError, wave.Error):
            cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dub-stretch-", dir=cache) as directory:
        source = _write_wav(samples, rate, Path(directory) / "take.wav")
        target = _ffmpeg(["-i", str(source), "-af", atempo_chain(tempo), "-ar", str(rate), "-ac", "1"],
                         Path(directory) / "timed.wav", timeout=300, action="Fitting a line to its timing")
        timed = decode_wav(target.read_bytes())[0]
        if cached is not None:
            try:
                os.replace(target, cached)
            except OSError:  # another line with the same take and tempo is saving it right now
                pass
        return timed


def match(samples: np.ndarray, rate: int, target: float,
          cache: Path | None = None) -> tuple[np.ndarray, float, str]:
    """Stretch a take to exactly ``target`` seconds, then pad or trim the last samples."""
    seconds = len(samples) / rate
    if target <= 0 or not len(samples):
        return samples, 1.0, ""
    tempo = seconds / target
    timed = _stretch(samples, rate, tempo, cache) if abs(seconds - target) > 0.03 else samples
    size = int(round(target * rate))
    timed = timed[:size] if len(timed) >= size else np.concatenate(
        [timed, np.zeros(size - len(timed), dtype=np.float32)])
    note = ""
    if tempo > TEMPO_WARNING[1]:
        note = f"sped up {tempo:.2f}x to match; a shorter translation would sound more natural"
    elif tempo < TEMPO_WARNING[0]:
        note = f"slowed to {tempo:.2f}x to match; a longer translation would sound more natural"
    return timed, tempo, note


def fit(samples: np.ndarray, rate: int, available: float, max_speed: float,
        cache: Path | None = None) -> tuple[np.ndarray, float, str]:
    """Speed a take up to fit ``available`` seconds, by at most ``max_speed``; reports what it did."""
    seconds = len(samples) / rate
    if seconds <= available + 0.02 or available <= 0:
        return samples, 1.0, ""
    factor = min(seconds / available, max(1.0, float(max_speed)))
    fitted = _stretch(samples, rate, factor, cache) if factor > 1.01 else samples
    over = len(fitted) / rate - available
    note = f"{over:.1f}s longer than its slot" if over > 0.05 else ""
    return fitted, factor, note


def timed_take(take: np.ndarray, lines: list[dict], index: int, duration: float, request: DubRequest,
               cache: Path | None = None) -> tuple[np.ndarray, float, str, float]:
    """The take as it will play: (samples, tempo, note, seconds it was fitted to)."""
    segment = lines[index]
    if request.timing == TIMING_NATURAL:
        available = slot_end(lines, index, duration) - segment["start"]
        return (*fit(take, RATE, available, request.max_speed, cache), available)
    available = segment["end"] - segment["start"]
    return (*match(take, RATE, available, cache), available)


def _timing_cache(session: sessions.Session) -> Path:
    return session.root / "timed"


def _synthesize(planned: list[_Line], request: DubRequest, session: sessions.Session, *, runtime, free_gpu,
                hf_token=None) -> Iterator[str]:
    """Fill each line's ``take`` (mono at RATE), reusing saved takes unless regeneration was asked."""
    from h3_app.drama3.serving import audio, takes as take_store
    from h3_app.drama3.serving.speech import LineSpeaker, persist_caches

    temperature, top_p, top_k = VOICE_PRESETS.get(request.preset, VOICE_PRESETS["Balance"])
    sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    speaker = LineSpeaker(runtime)
    store = take_store.takes_root(runtime)
    forced = set(request.regenerate)
    occurrences: dict[str, int] = {}
    staged: dict[Path, Path] = {}
    for line in planned:
        staged.setdefault(line.reference, speaker.reference(line.reference))
        content = take_store.take_key(speaker=f"dub:{line.segment['speaker']}", text=line.segment["translation"],
                                      reference=staged[line.reference].stem, transcript=line.transcript,
                                      sampling=sampling, max_new_tokens=speaker.budget(line.segment["translation"]))
        occurrence = occurrences.get(content, 0)
        occurrences[content] = occurrence + 1
        line.key = take_store.occurrence_key(content, occurrence)
        saved = None if line.index in forced else take_store.load_take(store, line.key)
        if saved is not None:
            line.take = _resample(audio.tidy_take(*saved), saved[1], RATE)
    pending = [line for line in planned if line.take is None]
    reused = len(planned) - len(pending)
    if not pending:
        yield f"Reusing all {reused} saved take(s)"
        return
    yield f"Reusing {reused} saved take(s); generating {len(pending)}" if reused else f"Generating {len(pending)} line(s)"
    tools.stop()  # Drama 3 needs the VRAM the transcription models hold.
    for update in speaker.ready(free_gpu, hf_token):
        yield update.status

    started, done = time.monotonic(), 0
    outcomes: dict[int, tuple[dict, str]] = {}
    lines = [(line, line.segment["translation"], staged[line.reference], line.transcript) for line in pending]
    try:
        with closing(speaker.speak(lines, sampling=sampling, parallel=request.parallel,
                                   thread_prefix="dub-sglang", keep_going=True)) as batches:
            for batch in batches:
                for spoken in batch:
                    line = spoken.key
                    outcomes[line.index] = (line.segment, spoken.problem)
                    done += 1
                    if not spoken.samples.size:  # rejected: the line keeps no take and is marked for review
                        line.note = spoken.problem
                        continue
                    line.take = _resample(audio.tidy_take(spoken.samples, spoken.rate), spoken.rate, RATE)
                    line.note = f"kept its best take ({spoken.problem})" if spoken.problem else ""
                    try:
                        take_store.save_take(store, line.key, spoken.wav)
                    except OSError as exc:  # a full disk must not fail the dub
                        line.note = f"take not saved: {exc}"
                yield f"Voicing lines: {done}/{len(pending)} ({time.monotonic() - started:.0f}s)"
    finally:
        # Recorded even when interrupted, so the lines voiced so far keep their marks.
        if outcomes:
            _record_take_problems(session, outcomes)
    persist_caches()
    take_store.prune(store)


def take_problems(session: sessions.Session) -> dict[int, str]:
    """Lines whose current take failed its check or could not be generated, with why.

    A mark belongs to the line as it was voiced: once its start or translation changes,
    or lines are added or removed before it, the mark no longer matches and is ignored.
    """
    stored = session.data.get(TAKE_PROBLEMS)
    lines = session.segments
    found = {}
    for key, entry in (stored.items() if isinstance(stored, dict) else ()):
        try:
            index = int(key)
            matches = (isinstance(entry, dict) and 0 <= index < len(lines)
                       and lines[index]["translation"] == entry.get("translation")
                       and abs(lines[index]["start"] - float(entry.get("start"))) < 0.0005)
        except (TypeError, ValueError):
            continue
        if matches:
            found[index] = str(entry.get("reason") or "failed")
    return found


def _record_take_problems(session: sessions.Session, outcomes: dict[int, tuple[dict, str]]) -> None:
    """Mark lines whose new take failed, clear those that now have a good one, and drop stale marks."""

    def change(latest: sessions.Session) -> None:
        marks = {str(index): {"reason": reason, "translation": latest.segments[index]["translation"],
                              "start": latest.segments[index]["start"]}
                 for index, reason in take_problems(latest).items()}
        for index, (segment, problem) in outcomes.items():
            if problem:
                marks[str(index)] = {"reason": problem, "translation": segment["translation"],
                                     "start": segment["start"]}
            else:
                marks.pop(str(index), None)
        latest.data[TAKE_PROBLEMS] = marks

    sessions.modify(session, change)


def original_line(session_id: str, index: int, *, runtime) -> LinePreview:
    """The original soundtrack at the saved line's exact cuts; no padding or synthesis."""
    session = sessions.load(runtime, session_id)
    lines = session.segments
    if not 0 <= index < len(lines):
        raise H3Error(f"No line {index + 1}; the table has {len(lines)} line(s).")
    audio = session.path("audio")
    if audio is None or not audio.is_file():
        raise H3Error("This session's original soundtrack is missing. Analyze the source again.")
    line = lines[index]
    samples = read_slice(audio, line["start"], line["end"])
    if not samples.size:
        raise H3Error("This line's times are outside the original audio. Adjust its Start and End.")
    return LinePreview(RATE, samples, f"Original line {index + 1}: "
                       f"{segment_tools.format_time(line['start'])}–{segment_tools.format_time(line['end'])}.")


def preview_line(request: DubRequest, index: int, *, runtime, free_gpu, hf_token=None) -> Iterator[str | LinePreview]:
    session = sessions.load(runtime, request.session_id)
    lines = session.segments
    if not 0 <= index < len(lines):
        raise H3Error(f"No line {index + 1}; the table has {len(lines)} line(s).")
    planned = _plan(session, [index], request.voice_mode, runtime)
    if not planned:
        raise H3Error(f"Line {index + 1} has no translation to speak.")
    yield from _synthesize(planned, request, session, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token)
    line = planned[0]
    if line.take is None or not len(line.take):
        raise H3Error(f"Line {index + 1}: {line.note or 'no audio was generated'}. Try New take again, "
                      "or edit its translation.")
    fitted, speed, note, available = timed_take(line.take, lines, index, session.data.get("duration", 0.0),
                                                request, _timing_cache(session))
    volume = speaker_settings(session).get(line.segment["speaker"], {}).get("volume", 100.0)
    if volume != 100.0:
        fitted = fitted * np.float32(volume / 100.0)
    details = [f"Line {index + 1}: {len(fitted) / RATE:.1f}s for a {available:.1f}s slot"]
    if not note and abs(speed - 1.0) > 0.01:
        details.append(f"{'sped up' if speed > 1 else 'slowed to'} {speed:.2f}x")
    details += [part for part in (note, line.note) if part]
    yield LinePreview(RATE, fitted, " · ".join(details) + ".")


def _speech_track(session: sessions.Session, planned: list[_Line], request: DubRequest,
                  duration: float) -> tuple[np.ndarray, list[str], dict[int, float]]:
    """The dubbed speech on the source's timeline, the check notes, and where each voiced line ends.

    A last line still too long after the speed-up runs past the end of the source
    instead of being cut off; the timeline grows to hold it.
    """
    lines = session.segments
    notes = []
    placed = []
    gains = {speaker: entry["volume"] / 100.0 for speaker, entry in speaker_settings(session).items()
             if entry["volume"] != 100.0}
    with_audio = [line for line in planned if line.take is not None and len(line.take)]
    cache = _timing_cache(session)
    started = time.time()
    with ThreadPoolExecutor(max_workers=min(TIMING_WORKERS, max(1, len(with_audio))),
                            thread_name_prefix="dub-timing") as pool:
        # Each worker runs in this job's context, so Interrupt reaches its ffmpeg runs.
        timed = {line.index: pool.submit(contextvars.copy_context().run, timed_take, line.take, lines, line.index,
                                         duration, request, cache) for line in with_audio}
        timed = {index: future.result() for index, future in timed.items()}
    # Fitted takes this dub did not use belong to old takes or old timings.
    for path in cache.glob("*.wav") if cache.is_dir() else ():
        try:
            if path.stat().st_mtime < started - 10:  # margin for coarse file times
                path.unlink()
        except OSError:
            pass
    for line in planned:
        if line.index not in timed:
            notes.append(f"line {line.index + 1}: {line.note or 'no audio'}, original kept")
            continue
        fitted, speed, note, _available = timed[line.index]
        if line.segment["speaker"] in gains:
            fitted = fitted * np.float32(gains[line.segment["speaker"]])
        line.speed = speed
        if note or line.note:
            notes.append(f"line {line.index + 1}: " + "; ".join(part for part in (note, line.note) if part))
        placed.append((line.index, int(round(line.segment["start"] * RATE)), fitted))
    source_length = int(round(duration * RATE))
    track = np.zeros(max([source_length] + [start + len(fitted) for _, start, fitted in placed]), dtype=np.float32)
    spoken_ends = {}
    for index, start, fitted in placed:
        track[start:start + len(fitted)] += fitted
        spoken_ends[index] = (start + len(fitted)) / RATE
    if len(track) > source_length + RATE // 20:
        notes.append(f"the dub runs {(len(track) - source_length) / RATE:.1f}s past the end of the source")
    voiced = set(spoken_ends)
    if session.path("vocals") is not None:
        # Every line not voiced here (unticked, untranslated or failed) keeps its original voice.
        for index, segment in enumerate(lines):
            if index not in voiced:
                start = int(round(segment["start"] * RATE))
                original = read_slice(session.path("vocals"), segment["start"], segment["end"])
                if segment["speaker"] in gains:
                    original *= np.float32(gains[segment["speaker"]])
                end = min(len(track), start + len(original))
                track[start:end] += original[: end - start]
    return track, notes, spoken_ends


def subtitle_lines(lines: list[dict], spoken_ends: dict[int, float]) -> list[dict]:
    """Translated subtitles that stay up while a dubbed line is still being spoken.

    A line may extend to its spoken end but never past the next line's start, except the last.
    """
    shown = []
    for index, line in enumerate(lines):
        end = max(line["end"], spoken_ends.get(index, line["end"]))
        later = [s["start"] for s in lines[index + 1:] if s["start"] > line["start"]]
        if later:
            end = min(end, max(line["end"], min(later) - LINE_GAP))
        shown.append({**line, "end": round(end, 3)})
    return shown


def _filter_path(path: Path) -> str:
    """A path for an ffmpeg filter argument (drive colons and quotes escaped)."""
    return "'" + path.resolve().as_posix().replace("\\", "/").replace(":", r"\:").replace("'", r"\'") + "'"


def _language_code(language: str) -> str:
    codes = {"english": "eng", "chinese": "zho", "cantonese": "yue", "japanese": "jpn", "korean": "kor",
             "spanish": "spa", "french": "fra", "german": "deu", "portuguese": "por", "italian": "ita",
             "russian": "rus", "arabic": "ara", "indonesian": "ind", "vietnamese": "vie", "thai": "tha",
             "hindi": "hin", "turkish": "tur", "malay": "msa", "dutch": "nld"}
    return codes.get(str(language or "").strip().lower(), "und")


def _output_stem(session: sessions.Session, language: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", session.data.get("name") or "dub", flags=re.UNICODE).strip("_")[:60] or "dub"
    tag = re.sub(r"[^\w]+", "", language or "")[:16] or "dub"
    return f"{name}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _video_seconds(source: Path) -> float | None:
    """The first video stream's length, or None when it cannot be read."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
           "-of", "csv=p=0", str(source)]
    try:
        proc = run_media_process(cmd, capture_output=True, text=True, timeout=30)
        seconds = float(proc.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return seconds if proc.returncode == 0 and seconds > 0 else None


def export_subtitled_video(video: str, subtitles: str, *, runtime) -> Path:
    """Burn a finished dub's subtitles into a new export, preserving its audio tracks."""
    root = (runtime.outputs_dir / "dubbing").resolve()
    source, captions = Path(video).resolve(), Path(subtitles).resolve()
    if (source.parent != root or captions.parent != root or source.suffix != ".mp4"
            or captions.suffix != ".srt" or not captions.name.startswith(source.stem + ".")
            or captions.name.endswith(".original.srt")
            or not source.is_file() or not captions.is_file()):
        raise H3Error("Generate a video dub with translated subtitles first; its files must still exist.")
    duration = probe_audio_seconds(source)
    if not duration:
        raise H3Error("Could not read the dubbed video's duration.")
    target = root / f"{source.stem}_subtitled_{uuid.uuid4().hex[:8]}.mp4"
    # A dub longer than the picture holds its last frame: pad by the overrun (the whole duration
    # when the picture cannot be measured; -t trims it) before drawing captions, so the held
    # frame still shows the changing subtitles.
    picture = _video_seconds(source)
    overrun = duration - picture if picture else duration
    filters = f"subtitles={_filter_path(captions)}"
    if overrun > 0.05:
        filters = f"tpad=stop_mode=clone:stop_duration={overrun:.3f}," + filters
    # The selectable subtitle tracks are kept alongside the burned-in ones.
    result = _ffmpeg(["-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?",
                      "-vf", filters, "-t", f"{duration:.3f}", "-c:v", "libx264", "-crf", "18",
                      "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", "-c:s", "copy",
                      "-movflags", "+faststart"], target, timeout=runtime.generation_timeout,
                     action="Exporting video with burned-in subtitles")
    original = read_snapshot(source) or {}
    write_snapshot(result, {"family": "Dub & Translate", "source_asset_ids": [original["asset_id"]]
                           if original.get("asset_id") else [], "settings": {
        **original.get("settings", {}), "source": source.name, "burned_subtitles": True,
        "subtitle_file": captions.name}})
    return result


def _prune_players(output_dir: Path) -> None:
    """Drop player tracks whose dub was deleted from Media (each folder is named after its outputs)."""
    root = output_dir / PLAYER_DIR
    for folder in root.iterdir() if root.is_dir() else ():
        if folder.is_dir() and not (output_dir / f"{folder.name}.mp3").exists():
            shutil.rmtree(folder, ignore_errors=True)


def _player(session: sessions.Session, speech: Path, folder: Path, *, separated: bool, speech_gain: float,
            background_gain: float, video: Path | None, mixed: Path, lines: list[dict],
            spoken_ends: dict[int, float], duration: float, language: str) -> Player:
    """The dub's voice, the original voice and the background as separate tracks for the result player.

    Without separation the background still holds the original voices, so the player
    switches between the finished mix and the original soundtrack, with no music switch.
    """
    folder.mkdir(parents=True, exist_ok=True)

    def encode(source: Path, name: str, gain: float) -> Path:
        return _ffmpeg(["-i", str(source), "-af", f"volume={gain:.3f}", *PLAYER_MP3],
                       folder / name, action="Preparing the result player")

    if separated:
        dub_track = encode(speech, "dub.mp3", speech_gain)
        music = encode(session.path("backing"), "music.mp3", background_gain)
        original = encode(session.path("vocals"), "original.mp3", 1.0)
    else:
        dub_track, music = mixed, None
        original = encode(session.path("audio"), "original.mp3", 1.0)
    shown = subtitle_lines(lines, spoken_ends)
    cues = [{"start": line["start"], "end": line["end"], "shown_end": shown[index]["end"],
             "speaker": line["speaker"], "source": line["source"], "translation": line["translation"]}
            for index, line in enumerate(lines) if line["source"] or line["translation"]]
    return Player(video, dub_track, original, music, duration, cues, language,
                  session.data.get("source_language", ""))


def dub(request: DubRequest, *, runtime, free_gpu, hf_token=None) -> Iterator[str | Dub]:
    session = sessions.load(runtime, request.session_id)
    lines = session.segments
    chosen = [i for i, segment in enumerate(lines) if segment.get("dub", True) and segment["translation"]]
    if not chosen:
        raise H3Error("No line is ready to dub: give lines a translation and tick Dub.")
    duration = float(session.data.get("duration") or 0.0) or max(s["end"] for s in lines) + 1.0
    planned = _plan(session, chosen, request.voice_mode, runtime)
    yield from _synthesize(planned, request, session, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token)

    yield "Placing lines on the timeline"
    track, notes, spoken_ends = _speech_track(session, planned, request, duration)
    overrun = max(0.0, len(track) / RATE - duration)
    speech = _write_wav(track, RATE, session.root / "speech.wav")
    del track
    separated = session.path("backing") is not None
    background = session.path("backing") or session.path("audio")
    speech_gain = max(0.0, float(request.speech_volume)) / 100.0
    background_gain = max(0.0, float(request.background_volume)) / 100.0
    if separated:
        graph = (f"[1:a]volume={speech_gain:.3f},aformat=channel_layouts=stereo[sp];"
                 f"[0:a]volume={background_gain:.3f}[bg];"
                 "[bg][sp]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.97[out]")
    else:
        # The background still holds the original voices: duck it under the dub.
        graph = (f"[1:a]volume={speech_gain:.3f},aformat=channel_layouts=stereo,asplit=2[sp][key];"
                 f"[0:a]volume={background_gain:.3f}[bg];"
                 "[bg][key]sidechaincompress=threshold=0.02:ratio=12:attack=20:release=400[ducked];"
                 "[ducked][sp]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.97[out]")
    yield "Mixing the dub over the background"
    mix = _ffmpeg(["-i", str(background), "-i", str(speech), "-filter_complex", graph, "-map", "[out]",
                   "-ar", str(RATE), "-c:a", "pcm_s16le"], session.root / "mix.wav", action="Mixing the dub")

    language = session.data.get("target_language") or ""
    output_dir = runtime.outputs_dir / "dubbing"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(session, language)
    subtitles = []
    for field_name, suffix in (("translation", language or "dub"), ("source", "original")):
        shown = subtitle_lines(lines, spoken_ends) if field_name == "translation" else lines
        text = segment_tools.to_srt(shown, field_name)
        if text.strip():
            path = output_dir / f"{stem}.{re.sub(r'[^A-Za-z0-9]+', '', suffix).lower() or 'dub'}.srt"
            path.write_text(text, encoding="utf-8")
            subtitles.append(path)
    audio_out = _ffmpeg(["-i", str(mix), *OUTPUT_MP3], output_dir / f"{stem}.mp3",
                        action="Encoding the dubbed audio")
    settings = {
        "source": session.data.get("name"), "target_language": language,
        "source_language": session.data.get("source_language", ""), "pipeline": session.data.get("pipeline"),
        "separated": separated, "voice_mode": request.voice_mode, "preset": request.preset,
        "timing": request.timing, "max_speed": request.max_speed, "speech_volume": request.speech_volume,
        "background_volume": request.background_volume, "lines": len(lines), "dubbed_lines": len(planned),
        "sped_up_lines": sum(line.speed > 1.01 for line in planned), "session": session.id,
        "speakers": speaker_settings(session),
    }
    write_snapshot(audio_out, {"family": "Dub & Translate", "settings": settings})

    video_out = None
    if session.data.get("video"):
        yield "Writing the dubbed video"
        source = session.path("source")
        inputs = ["-i", str(source), "-i", str(mix)]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]
        metadata = ["-metadata:s:a:0", f"language={_language_code(language)}",
                    "-metadata:s:a:0", f"title=Dub ({language or 'translated'})", "-disposition:a:0", "default"]
        if request.keep_original_track:
            inputs += ["-i", str(session.path("audio"))]
            maps += ["-map", "2:a:0"]
            metadata += ["-metadata:s:a:1", f"language={_language_code(session.data.get('source_language', ''))}",
                         "-metadata:s:a:1", "title=Original", "-disposition:a:1", "0"]
        for number, path in enumerate(subtitles):
            inputs += ["-i", str(path)]
            maps += ["-map", f"{len(inputs) // 2 - 1}:s:0"]
            original = path.name.endswith(".original.srt")
            metadata += [f"-metadata:s:s:{number}",
                         f"language={_language_code(session.data.get('source_language', '') if original else language)}",
                         f"-metadata:s:s:{number}", f"title={'Original' if original else language or 'Translated'}"]
        # Keep the picture unchanged; subtitle burning is a separate export after generation.
        video_out = _ffmpeg([*inputs, *maps, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
                             *metadata, "-movflags", "+faststart"], output_dir / f"{stem}.mp4",
                            timeout=runtime.generation_timeout, action="Writing the dubbed video")
        write_snapshot(video_out, {"family": "Dub & Translate", "settings": {
            **settings, "original_track": request.keep_original_track, "burned_subtitles": False}})
    yield "Preparing the result player"
    _prune_players(output_dir)
    player = _player(session, speech, output_dir / PLAYER_DIR / stem, separated=separated, speech_gain=speech_gain,
                     background_gain=background_gain, video=video_out, mixed=audio_out, lines=lines,
                     spoken_ends=spoken_ends, duration=duration + overrun, language=language)
    for work_file in (speech, mix):
        work_file.unlink(missing_ok=True)
    # Only this field is the dub's: lines edited while it ran stay as saved.
    sessions.modify(session, lambda latest: latest.data.update(last_output=str(video_out or audio_out)))
    sped = sum(line.speed > 1.01 for line in planned)
    slowed = sum(line.speed < 0.99 for line in planned)
    message = f"Dubbed {len(planned)} line(s) into {language or 'the target language'}"
    if request.timing == TIMING_NATURAL:
        message += f"; {sped} sped up to fit" if sped else ""
    elif sped or slowed:
        message += f"; matched to the original timing ({sped} sped up, {slowed} slowed down)"
    if notes:
        message += ".\n\nCheck: " + "; ".join(notes[:12]) + ("; …" if len(notes) > 12 else "")
    yield Dub(session, audio_out, video_out, subtitles, message + ("" if notes else "."), player)
