"""Dub & Translate: turn a source into timed lines, then voice them with Drama 3 and remix.

Analysis extracts the soundtrack, optionally separates vocals from the background,
and transcribes with Gemini or locally: OmniVAD + Qwen3-ASR + ForcedAligner, or
MOSS-Transcribe-Diarize on SGLang-Omni, which also tells speakers apart, or Nemotron
Transcribe on vLLM, which also times each speaker's words through overlaps; Hy-MT then
translates. Dubbing clones each line's voice from the original audio, fits the take to
its slot, lays the lines on the timeline and mixes them over the background. Takes
are content-keyed, so an unchanged line is never generated twice.

Each step is a generator of status strings that ends with one result object.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import contextvars
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
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
from h3_app.jobs import JobCancelled
from h3_app.media_tools import probe_audio_seconds, run_media_process
from h3_app.provenance import read_snapshot, write_snapshot

from . import gemini, moss, nemotron, refinement, segments as segment_tools, session as sessions, tools, tools_worker

RATE = 44100
PIPELINE_LOCAL = "Local: Qwen3-ASR + aligner + Hy-MT"
PIPELINE_MOSS = "Local: MOSS Transcribe + Diarize + Hy-MT"
PIPELINE_NEMOTRON = "Local: Nemotron Transcribe + Hy-MT"
PIPELINE_GEMINI = "Gemini (cloud)"
PIPELINES = (PIPELINE_LOCAL, PIPELINE_MOSS, PIPELINE_NEMOTRON, PIPELINE_GEMINI)
# Transcriptions that tell speakers apart, so each speaker can get its own voice and volume.
SPEAKER_PIPELINES = frozenset({PIPELINE_MOSS, PIPELINE_NEMOTRON, PIPELINE_GEMINI})
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
VOICE_NEAREST = "Nearest clean voice"
VOICE_SPEAKER = "One voice per speaker"
VOICE_MODES = (VOICE_LINE, VOICE_NEAREST, VOICE_SPEAKER)
# Match stretches every take to its line's original length with ffmpeg, as index-tts-vllm's
# duration control does; Natural only speeds up long takes, by at most the chosen limit;
# Smart keeps each take's own pace and fits long ones into the silence around them first.
TIMING_SMART = "Smart timing"
TIMING_MATCH = "Match the original timing"
TIMING_NATURAL = "Natural pace"
TIMINGS = (TIMING_SMART, TIMING_MATCH, TIMING_NATURAL)
# Smart timing: the silence kept between dubbed lines, how much earlier a line may start and
# later a line may be pushed, the speed-up that still sounds natural, and the pauses inside a
# take it shortens (longer than the first, down to the second) before speeding anything up.
SMART_GAP = 0.1
SMART_LEAD = 0.3
SMART_DELAY = 0.5
SMART_SPEED = 1.25
SMART_PAUSE = (0.25, 0.15)
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
OUTPUT_MP3 = ("-c:a", "libmp3lame", "-b:a", "192k")
# Session keys for lines shown in Review lines: their take failed its check or could not be
# generated, or Smart timing found them still too long in the last dub.
TAKE_PROBLEMS = "take_problems"
TIMING_PROBLEMS = "timing_problems"


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
    # HY-MT retranslation can omit both earlier dialogue and translation notes.
    translation_context: bool = True


@dataclass(frozen=True)
class DubRequest:
    session_id: str
    voice_mode: str = VOICE_LINE
    preset: str = "Balance"
    timing: str = TIMING_SMART
    parallel: int = 16
    max_speed: float = 1.35
    speech_volume: float = 100.0
    background_volume: float = 100.0
    keep_original_track: bool = True
    regenerate: tuple[int, ...] = ()
    # Time every subtitle word with Qwen3's aligner; unchanged lines reuse their takes and timing.
    word_timing: bool = False


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
    word_timing: bool = False


@dataclass
class WordAlignment:
    player: Player
    message: str


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
    timing_problem: str = ""  # Smart timing still found it too long; it is marked for review


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
    elif request.pipeline == PIPELINE_NEMOTRON:
        duration = float(session.data.get("duration") or 0.0)
        upload = _ffmpeg(["-i", str(speech), "-ac", "1", "-ar", str(nemotron.UPLOAD_RATE), "-c:a", "pcm_s16le"],
                         session.root / "speech-16k.wav", action="Preparing audio for Nemotron")
        try:
            found = yield from blocking(
                "Finding speakers, words and overlaps with Nemotron",
                lambda report: nemotron.transcribe(runtime, upload, duration=duration, free_gpu=free_gpu,
                                                   hf_token=hf_token, report=report))
        finally:
            upload.unlink(missing_ok=True)
        # The words and who spoke when stay with the transcript: edits never change what was heard.
        transcript = {**found, "speakers": {speaker: "" for speaker in
                                            dict.fromkeys(s["speaker"] for s in found["segments"])}}
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
    yield Analysis(session, message + translation_advice(session.segments) + length_advice(session.segments))


def _transcript_key(request: AnalyzeRequest) -> str:
    """The settings a transcript depends on, beyond the source file itself."""
    settings: dict[str, Any] = {"pipeline": request.pipeline,
                                "separator": request.separator_model if request.separate else ""}
    if request.pipeline == PIPELINE_LOCAL and request.refine_timestamps:
        settings.update(refinement=refinement.REVISION, language=_language(request.source_language))
    if request.pipeline == PIPELINE_MOSS:
        settings.update(model=f"{moss.MODEL_REPO}@{moss.MODEL_REVISION[:12]}")
    elif request.pipeline == PIPELINE_NEMOTRON:
        settings.update(model=nemotron.MODEL_REPO)
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
                    if request.translation_context else ""
                    for i in indices]
        reply = yield from blocking(
            f"Translating {len(texts)} line(s) into {target} with Hy-MT",
            lambda report: tools.run("translate", {"texts": texts, "contexts": contexts, "target_language": target,
                                                   "notes": request.notes if request.translation_context else ""},
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
                            + translation_advice(session.segments) + length_advice(session.segments))


def translation_issues(lines: list[dict]) -> list[int]:
    """Recheck saved text so edits clear warnings and older sessions get them too."""
    return [index for index, line in enumerate(lines)
            if line.get("dub", True) and line["translation"]
            and tools_worker.invalid_translation(line["translation"], line["source"])]


def translation_advice(lines: list[dict]) -> str:
    found = translation_issues(lines)
    if not found:
        return ""
    numbers = ", ".join(str(index + 1) for index in found[:20]) + (", …" if len(found) > 20 else "")
    return (f" Review {len(found)} translation(s) that may contain explanations: {numbers}. "
            "Edit these lines or translate them again before dubbing.")


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


def default_voice_mode(pipeline: str | None) -> str:
    """Nemotron's speaker labels cover overlaps, so its sessions clone from clean speech by default;
    other transcribers can mislabel a line, which would clone the wrong person."""
    return VOICE_NEAREST if pipeline == PIPELINE_NEMOTRON else VOICE_LINE


def speaker_activity(session: sessions.Session) -> dict[str, list[tuple[float, float]]]:
    """When each speaker talks, for telling clean speech from cross-talk.

    Nemotron's words and diarization as transcribed, when the session has them; also every
    line as transcribed and as edited, so a moved or deleted line never hides speech that is
    still in the recording, and a line given to another speaker counts for both.
    """
    activity: dict[str, list[tuple[float, float]]] = {}
    transcript = session.data.get("transcript")
    transcript = transcript if isinstance(transcript, dict) else {}
    stored = transcript.get("activity")
    for speaker, spans in (stored.items() if isinstance(stored, dict) else ()):
        for span in spans if isinstance(spans, list) else ():
            try:
                activity.setdefault(str(speaker), []).append((float(span[0]), float(span[1])))
            except (IndexError, TypeError, ValueError):
                continue
    for line in [*(transcript.get("segments") or []), *session.segments]:
        try:
            activity.setdefault(segment_tools.speaker_id(line.get("speaker")), []).append(
                (float(line["start"]), float(line["end"])))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return activity


def _clean(activity: dict[str, list[tuple[float, float]]], speaker: str, start: float, end: float) -> bool:
    """No one else speaks during ``start``–``end``, padding included."""
    low, high = start - REFERENCE_PAD, end + REFERENCE_PAD
    return not any(other != speaker and span_start < high and span_end > low
                   for other, spans in activity.items() for span_start, span_end in spans)


def _native_words(session: sessions.Session, segment: dict) -> list[dict]:
    """The transcriber's own words for an unedited line (Nemotron), else nothing."""
    transcript = session.data.get("transcript")
    words = transcript.get("words") if isinstance(transcript, dict) else None
    if not isinstance(words, list):
        return []
    found = [word for word in words if isinstance(word, dict) and word.get("speaker") == segment["speaker"]
             and float(word["start"]) >= segment["start"] - 0.01 and float(word["end"]) <= segment["end"] + 0.01]
    if not found or nemotron.join_words([word["text"] for word in found]) != segment["source"].strip():
        return []
    return found


def _trimmed(session: sessions.Session, segment: dict) -> tuple[dict, str]:
    """A long clean clip cut to at most REFERENCE_RANGE's top at a word, ideally a sentence end;
    without native words the whole line is used."""
    if segment["end"] - segment["start"] <= REFERENCE_RANGE[1]:
        return segment, segment["source"]
    words = _native_words(session, segment)
    kept = [word for word in words if word["end"] - segment["start"] <= REFERENCE_RANGE[1]]
    if not kept:
        return segment, segment["source"]
    ends = [index for index, word in enumerate(kept)
            if nemotron.SENTENCE_END.search(word["text"]) and word["end"] - segment["start"] >= MIN_REFERENCE_SECONDS]
    kept = kept[:ends[-1] + 1] if ends else kept
    return ({"start": segment["start"], "end": kept[-1]["end"]},
            nemotron.join_words([word["text"] for word in kept]))


def clean_lines(lines: list[dict], activity: dict[str, list[tuple[float, float]]]) -> set[int]:
    """The lines usable as a clean reference: transcribed, at least MIN_REFERENCE_SECONDS, nobody else talking."""
    return {index for index, line in enumerate(lines)
            if line["source"] and line["end"] - line["start"] >= MIN_REFERENCE_SECONDS
            and _clean(activity, line["speaker"], line["start"], line["end"])}


def nearest_clean(lines: list[dict], index: int, clean: set[int]) -> int | None:
    """The line to clone line ``index`` from: itself when clean, else the same speaker's closest
    clean line, before or after, anywhere in the recording (ties: the earlier one); None when the
    speaker never speaks alone."""
    if index in clean:
        return index
    target = lines[index]

    def distance(line: dict) -> float:
        return max(0.0, target["start"] - line["end"], line["start"] - target["end"])

    found = [other for other in clean if lines[other]["speaker"] == target["speaker"]]
    return min(found, key=lambda other: (distance(lines[other]), lines[other]["start"])) if found else None


def speaker_settings_available(session: sessions.Session) -> bool:
    """Per-speaker settings need real speakers, which only MOSS, Nemotron and Gemini tell apart."""
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
        raise H3Error("Per-speaker voices need a session transcribed with MOSS, Nemotron or Gemini, which tell "
                      "speakers apart.")
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
    clean = clean_lines(lines, speaker_activity(session)) if voice_mode == VOICE_NEAREST else set()
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
        elif voice_mode == VOICE_NEAREST:
            best = nearest_clean(lines, index, clean)
            if best is None:
                # No clean speech from this speaker anywhere: the line's own audio, cross-talk and all.
                reference = (_reference_clip(session, segment), segment["source"] or segment["translation"])
            else:
                clip, transcript = _trimmed(session, lines[best])
                reference = (_reference_clip(session, clip), transcript)
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
    """Where a line's take must end: just before the next line starts, never before its own end.

    Lines that overlap it in the source (another speaker at the same time) do not cut it short.
    """
    segment = lines[index]
    later = [s["start"] for s in lines[index + 1:] if s["start"] >= segment["end"] - 1e-6]
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


@dataclass(frozen=True)
class SmartPlacement:
    start: float  # where the take starts on the timeline
    seconds: float  # how long it plays once fitted
    tighten: bool  # its inner pauses are shortened
    tempo: float
    room: float  # the time it could use without overlapping, borrowing included
    squeezed: bool  # still too long after the gentle speed-up: sped up to the limit, and marked


def smart_layout(lines: list[dict], natural: dict[int, float], tightened: Callable[[int], float],
                 max_speed: float) -> dict[int, SmartPlacement]:
    """Place each voiced line (``natural``: its take's own length) on the timeline, trying in turn:

    1. its own length at its own start, into the silence before the next line (less SMART_GAP);
    2. starting up to SMART_LEAD earlier, into silence the previous dubbed line left;
    3. its inner pauses shortened (``tightened`` gives that length);
    4. speeding up to SMART_SPEED;
    5. pushing the next line up to SMART_DELAY later, out of the time that line will not use,
       then speeding up to SMART_SPEED again;
    6. speeding up to ``max_speed`` and marking it; what still does not fit overlaps the next line.

    Lines not in ``natural`` keep their original voice at their original times and are not moved.
    Lines that overlap in the source (two speakers at once) do not hold each other back. A line is
    never started more than SMART_DELAY late, so an overlap cannot push every later line along.
    """
    gentle = min(SMART_SPEED, max(1.0, float(max_speed)))
    # The line each one must end before: the first later line that does not overlap it in the source.
    next_line = [next((later for later in range(index + 1, len(lines))
                       if lines[later]["start"] >= line["end"] - 1e-6), None)
                 for index, line in enumerate(lines)]

    def slack(index: int) -> float:
        """How far a dubbed line can be pushed and still play at its own length before its own next line."""
        if index not in natural:
            return 0.0
        if next_line[index] is None:
            return SMART_DELAY
        room = lines[next_line[index]]["start"] - SMART_GAP - lines[index]["start"]
        return min(SMART_DELAY, max(0.0, room - natural[index]))

    def fits(start: float, seconds: float, end_limit: float) -> bool:
        return start + seconds <= end_limit + 1e-6

    def sped(start: float, seconds: float, end_limit: float, cap: float) -> float | None:
        """The tempo that fits ``seconds`` between ``start`` and ``end_limit``, if it is at most ``cap``."""
        room = end_limit - start
        return seconds / room if room > 0 and seconds <= room * cap else None

    placements: dict[int, SmartPlacement] = {}
    finished: list[tuple[float, float]] = []  # (original end, placed end) of the lines before
    previous_end = None
    for index, line in enumerate(lines):
        while finished and finished[0][0] <= line["start"] + 1e-6:
            placed_end = heapq.heappop(finished)[1]
            previous_end = placed_end if previous_end is None else max(previous_end, placed_end)
        if index not in natural:
            heapq.heappush(finished, (line["end"], line["end"]))
            continue
        after = line["start"] if previous_end is None else min(previous_end + SMART_GAP,
                                                              line["start"] + SMART_DELAY)
        normal, earliest = max(line["start"], after), max(0.0, line["start"] - SMART_LEAD, after)
        following = next_line[index]
        limit = math.inf if following is None else lines[following]["start"] - SMART_GAP
        borrowed = limit if following is None else limit + slack(following)
        seconds, tighten, tempo, squeezed = natural[index], False, 1.0, False
        if not fits(earliest, seconds, limit):
            short = tightened(index)
            if short < seconds - 0.01:
                seconds, tighten = short, True
        if fits(earliest, seconds, limit):
            end_limit = limit
        elif (speed := sped(earliest, seconds, limit, gentle)) is not None:
            tempo, end_limit = speed, limit
        elif fits(earliest, seconds, borrowed):
            end_limit = borrowed
        elif (speed := sped(earliest, seconds, borrowed, gentle)) is not None:
            tempo, end_limit = speed, borrowed
        else:
            room = borrowed - earliest
            tempo = max(1.0, min(seconds / room if room > 0 else math.inf, float(max_speed)))
            end_limit, squeezed = borrowed, True
        length = seconds / tempo
        start = earliest if tempo > 1.0 else max(earliest, min(normal, end_limit - length))
        placements[index] = SmartPlacement(round(start, 3), length, tighten, tempo, end_limit - earliest, squeezed)
        heapq.heappush(finished, (line["end"], start + length))
    return placements


def smart_note(placement: SmartPlacement) -> str:
    """Why a squeezed line is marked: how fast it plays and whether it still overlaps."""
    over = placement.seconds - placement.room
    if over > 0.05:
        return f"too long even at {placement.tempo:.2f}x: overlaps the next line by {over:.1f}s"
    return f"too long for its slot: sped up {placement.tempo:.2f}x"


@dataclass(frozen=True)
class Timed:
    samples: np.ndarray
    start: float
    speed: float
    note: str
    available: float
    squeezed: bool = False


def place_takes(planned: list[_Line], lines: list[dict], duration: float, request: DubRequest,
                cache: Path | None = None, *, assume_others: bool = False) -> dict[int, Timed]:
    """Each voiced line's take as it will play and where it starts, for any timing.

    With ``assume_others`` (a single-line preview), Smart timing assumes every other line to dub
    fills exactly its own original time.
    """
    with_audio = [line for line in planned if line.take is not None and len(line.take)]

    def run(work: Callable[[_Line], Timed]) -> dict[int, Timed]:
        # Each worker runs in this job's context, so Interrupt reaches its ffmpeg runs.
        with ThreadPoolExecutor(max_workers=min(TIMING_WORKERS, max(1, len(with_audio))),
                                thread_name_prefix="dub-timing") as pool:
            futures = {line.index: pool.submit(contextvars.copy_context().run, work, line) for line in with_audio}
            return {index: future.result() for index, future in futures.items()}

    if request.timing != TIMING_SMART:
        def classic(line: _Line) -> Timed:
            fitted, speed, note, available = timed_take(line.take, lines, line.index, duration, request, cache)
            return Timed(fitted, line.segment["start"], speed, note, available)

        return run(classic)

    from h3_app.drama3.serving.audio import shorten_pauses

    by_index = {line.index: line for line in with_audio}
    natural = {index: len(line.take) / RATE for index, line in by_index.items()}
    if assume_others:
        natural.update({index: line["end"] - line["start"] for index, line in enumerate(lines)
                        if index not in natural and line["translation"] and line.get("dub", True)})
    shortened: dict[int, np.ndarray] = {}

    def tightened(index: int) -> float:
        if index not in by_index:
            return natural[index]
        if index not in shortened:
            longer_than, keep = SMART_PAUSE
            shortened[index] = shorten_pauses(by_index[index].take, RATE, longer_than=longer_than, keep=keep)
        return len(shortened[index]) / RATE

    layout = smart_layout(lines, natural, tightened, request.max_speed)

    def smart(line: _Line) -> Timed:
        placement = layout[line.index]
        samples = shortened[line.index] if placement.tighten else line.take
        if placement.tempo > 1.005:
            samples = _stretch(samples, RATE, placement.tempo, cache)
        # Never longer than planned, so the gap before the next line holds.
        samples = samples[:int(round(placement.seconds * RATE))]
        note = smart_note(placement) if placement.squeezed else ""
        return Timed(samples, placement.start, placement.tempo, note, placement.room, placement.squeezed)

    return run(smart)


def _timing_cache(session: sessions.Session) -> Path:
    return session.root / "timed"


def _synthesize(planned: list[_Line], request: DubRequest, session: sessions.Session, *, runtime, free_gpu,
                hf_token=None) -> Iterator[str]:
    """Fill each line's ``take`` (mono at RATE), reusing saved takes unless regeneration was asked."""
    from h3_app.drama3.serving import audio, takes as take_store
    from h3_app.drama3.serving.generate import FRAMES_PER_SECOND
    from h3_app.drama3.serving.planner import expected_seconds
    from h3_app.drama3.serving.speech import LineSpeaker, persist_caches

    temperature, top_p, top_k = VOICE_PRESETS.get(request.preset, VOICE_PRESETS["Balance"])
    sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    speaker = LineSpeaker(runtime)
    store = take_store.takes_root(runtime)
    forced = set(request.regenerate)
    previous_problems = take_problems(session)
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
        # Failed takes from older versions must not be replayed indefinitely.
        previous_problem = previous_problems.get(line.index)
        saved = None if line.index in forced or previous_problem else take_store.load_take(store, line.key)
        if saved is not None:
            try:
                check = audio.check_take(saved[0], saved[1], expected_seconds(line.segment["translation"]),
                                         max_seconds=speaker.budget(line.segment["translation"]) / FRAMES_PER_SECOND)
                if check.ok:
                    line.take = _resample(audio.tidy_take(*saved), saved[1], RATE)
            except ValueError:
                pass
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
                    if spoken.problem or not spoken.samples.size:
                        line.note = spoken.problem or "no translated audio returned"
                        line.take = None
                        # Remove a stale failed take at this identity; it must never
                        # masquerade as a successful translation in a later job.
                        try:
                            (store / f"{line.key}.wav").unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    line.take = _resample(audio.tidy_take(spoken.samples, spoken.rate), spoken.rate, RATE)
                    line.note = ""
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


def _take_mark(segment: dict) -> tuple[float, str, str]:
    """What a failed-take mark is filed under: the line as it was voiced, not its position,
    so adding or deleting other lines keeps it, and editing this line's start or text drops it.
    The speaker tells apart two people saying the same words at once (marks saved before it
    was added match any speaker's line)."""
    return round(float(segment["start"]), 3), str(segment.get("speaker") or ""), str(segment["translation"])


def _stored_marks(session: sessions.Session, key: str) -> dict[tuple[float, str, str], str]:
    stored = session.data.get(key)
    marks = {}
    for entry in stored if isinstance(stored, list) else ():
        try:
            marks[_take_mark(entry)] = str(entry.get("reason") or "failed")
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return marks


def _save_marks(latest: sessions.Session, key: str, marks: dict[tuple[float, str, str], str]) -> None:
    latest.data[key] = [{"start": start, "speaker": speaker, "translation": translation, "reason": reason}
                        for (start, speaker, translation), reason in marks.items()]


def _line_marks(session: sessions.Session, key: str) -> dict[int, str]:
    marks = _stored_marks(session, key)
    found = {}
    for index, line in enumerate(session.segments):
        start, _speaker, translation = mark = _take_mark(line)
        reason = marks.get(mark) or marks.get((start, "", translation))
        if line["translation"] and line.get("dub", True) and reason:
            found[index] = reason
    return found


def take_problems(session: sessions.Session) -> dict[int, str]:
    """The lines the next dub voices whose current take failed its check or could not be generated, with why."""
    return _line_marks(session, TAKE_PROBLEMS)


def timing_problems(session: sessions.Session) -> dict[int, str]:
    """The lines Smart timing found still too long in the last dub, with how fast they played."""
    return _line_marks(session, TIMING_PROBLEMS)


def _record_take_problems(session: sessions.Session, outcomes: dict[int, tuple[dict, str]]) -> None:
    """Mark lines whose new take failed, clear those that now have a good one, and drop stale marks."""

    def change(latest: sessions.Session) -> None:
        current = {_take_mark(line) for line in latest.segments}
        marks = {mark: reason for mark, reason in _stored_marks(latest, TAKE_PROBLEMS).items() if mark in current}
        for segment, problem in outcomes.values():
            if problem:
                marks[_take_mark(segment)] = problem
            else:
                marks.pop(_take_mark(segment), None)
        _save_marks(latest, TAKE_PROBLEMS, marks)

    sessions.modify(session, change)


def _record_timing_problems(session: sessions.Session, planned: list[_Line]) -> None:
    """Replace the timing marks with this dub's: they describe its layout, not the takes."""
    marks = {_take_mark(line.segment): line.timing_problem for line in planned if line.timing_problem}
    sessions.modify(session, lambda latest: _save_marks(latest, TIMING_PROBLEMS, marks))


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
    timed = place_takes(planned, lines, session.data.get("duration", 0.0), request, _timing_cache(session),
                        assume_others=True)[index]
    fitted, speed, note = timed.samples, timed.speed, timed.note
    volume = speaker_settings(session).get(line.segment["speaker"], {}).get("volume", 100.0)
    if volume != 100.0:
        fitted = fitted * np.float32(volume / 100.0)
    slot = "an open" if math.isinf(timed.available) else f"a {timed.available:.1f}s"
    details = [f"Line {index + 1}: {len(fitted) / RATE:.1f}s for {slot} slot",
               "fresh translated take" if index in request.regenerate else "saved translated take"]
    shift = timed.start - line.segment["start"]
    if abs(shift) >= 0.05:
        details.append(f"starts {abs(shift):.1f}s {'late' if shift > 0 else 'early'}")
    if not note and abs(speed - 1.0) > 0.01:
        details.append(f"{'sped up' if speed > 1 else 'slowed to'} {speed:.2f}x")
    details += [part for part in (note, line.note) if part]
    yield LinePreview(RATE, fitted, " · ".join(details) + ".")


def _speech_track(session: sessions.Session, planned: list[_Line], request: DubRequest,
                  duration: float, clips: dict[int, np.ndarray] | None = None,
                  ) -> tuple[np.ndarray, list[str], dict[int, tuple[float, float]]]:
    """The dubbed speech on the source's timeline, the check notes, and where each voiced line plays.

    A last line still too long after the speed-up runs past the end of the source
    instead of being cut off; the timeline grows to hold it. ``clips``, when given,
    receives each voiced line's fitted take as placed, keyed by line index.
    """
    lines = session.segments
    notes = []
    placed = []
    gains = {speaker: entry["volume"] / 100.0 for speaker, entry in speaker_settings(session).items()
             if entry["volume"] != 100.0}
    cache = _timing_cache(session)
    started = time.time()
    timed = place_takes(planned, lines, duration, request, cache)
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
        result = timed[line.index]
        fitted = result.samples
        if line.segment["speaker"] in gains:
            fitted = fitted * np.float32(gains[line.segment["speaker"]])
        line.speed = result.speed
        line.timing_problem = result.note if result.squeezed else ""
        if result.note or line.note:
            notes.append(f"line {line.index + 1}: " + "; ".join(part for part in (result.note, line.note) if part))
        placed.append((line.index, int(round(result.start * RATE)), fitted))
    source_length = int(round(duration * RATE))
    track = np.zeros(max([source_length] + [start + len(fitted) for _, start, fitted in placed]), dtype=np.float32)
    spoken = {}
    for index, start, fitted in placed:
        track[start:start + len(fitted)] += fitted
        spoken[index] = (start / RATE, (start + len(fitted)) / RATE)
        if clips is not None:
            clips[index] = fitted
    if len(track) > source_length + RATE // 20:
        notes.append(f"the dub runs {(len(track) - source_length) / RATE:.1f}s past the end of the source")
    voiced = set(spoken)
    if session.path("vocals") is not None:
        # Unticked or untranslated lines keep their original voice. The caller
        # rejects failed translated takes before reaching this mixing step.
        for first, last, speakers in kept_original(lines, voiced):
            start = int(round(first * RATE))
            original = read_slice(session.path("vocals"), first, last)
            # Overlapping speakers share these samples: only a single speaker's volume can apply.
            if len(speakers) == 1 and next(iter(speakers)) in gains:
                original *= np.float32(gains[next(iter(speakers))])
            end = min(len(track), start + len(original))
            track[start:end] += original[: end - start]
    return track, notes, spoken


def kept_original(lines: list[dict], voiced: set[int]) -> list[tuple[float, float, set[str]]]:
    """The original-voice stretches (start, end, speakers) of the lines not voiced, overlaps merged,
    so speech two such lines share is restored once instead of twice."""
    merged: list[list] = []
    for line in sorted((line for index, line in enumerate(lines) if index not in voiced),
                       key=lambda line: line["start"]):
        if merged and line["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], line["end"])
            merged[-1][2].add(line["speaker"])
        else:
            merged.append([line["start"], line["end"], {line["speaker"]}])
    return [(start, end, speakers) for start, end, speakers in merged]


def _validated_word_timing(text: str, words, duration: float) -> list[dict]:
    """Accept only complete timings that can actually play within this clip."""
    return tools_worker.aligned_clip_words(text, words, duration)


def _word_timings(session: sessions.Session, lines: list[dict], spoken: dict[int, tuple[float, float]],
                  clips: dict[int, np.ndarray], *, runtime, free_gpu, hf_token=None,
                  notes: list[str], existing: dict | None = None, align_source: bool = True) -> Iterator[str]:
    """Each line's ``{"source": words, "translation": words}`` on the player's timeline, where Qwen3 aligns them.

    The translation is timed on its fitted take as placed, the source on the original voice.
    Results are cached by clip audio and text, so a re-dub aligns only the lines that changed;
    a line that cannot be aligned keeps plain subtitles. Without ``align_source`` only the
    source words already in ``existing`` (the transcriber's own) are used: aligning edited
    lines against cross-talk would time them to the wrong voice.
    """
    from h3_app.drama3.serving.audio import encode_wav

    rate = tools_worker.ASR_RATE
    original = session.path("vocals") or session.path("audio")
    jobs = []  # (index, field, offset, wav, text)
    for index, line in enumerate(lines):
        known = "source" in (existing or {}).get(index, {})
        if ((align_source or known) and original is not None and line["source"].strip()
                and line["end"] > line["start"]):
            samples = read_slice(original, line["start"], line["end"])
            jobs.append((index, "source", line["start"], samples, line["source"]))
        if index in clips and line["translation"].strip():
            jobs.append((index, "translation", spoken[index][0], clips[index], line["translation"]))
    cache = session.root / "word_timing"
    found, pending = {}, []
    for index, field, offset, samples, text in jobs:
        duration = len(samples) / RATE
        previous = (existing or {}).get(index, {}).get(field)
        if previous:
            try:
                timed = _validated_word_timing(text, [
                    {"text": word["text"], "start": round(float(word["start"]) - offset, 3),
                     "end": round(float(word["end"]) - offset, 3)} for word in previous], duration)
                found[(index, field)] = (offset, timed)
                continue
            except (ValueError, TypeError, KeyError):
                pass
        wav = encode_wav(_resample(np.asarray(samples, dtype=np.float32), RATE, rate), rate)
        path = cache / f"{hashlib.sha256(wav + text.encode()).hexdigest()}.json"
        try:
            timed = json.loads(path.read_text(encoding="utf-8"))
            # Older caches include null/empty failures. Only complete, valid word
            # timings may suppress another alignment attempt.
            timed = _validated_word_timing(text, timed, duration)
            found[(index, field)] = (offset, timed)
        except (OSError, ValueError, TypeError, KeyError):
            pending.append((index, field, offset, duration, wav, text, path))
    if pending:
        try:
            with tempfile.TemporaryDirectory(prefix="dub-words-", dir=session.root) as directory:
                payload = []
                for number, (*_, wav, text, _path) in enumerate(pending):
                    clip = Path(directory) / f"{number}.wav"
                    clip.write_bytes(wav)
                    field = pending[number][1]
                    language = session.data.get("source_language" if field == "source" else "target_language", "")
                    payload.append({"audio": str(clip), "text": text, "language": language,
                                    "duration": pending[number][3]})
                reply = yield from blocking("Timing subtitle words with Qwen3", lambda report: tools.run(
                    "align_words", {"clips": payload}, runtime, free_gpu=free_gpu, hf_token=hf_token,
                    report=report))
            words = reply.get("words") or []
            if len(words) != len(pending):
                raise H3Error("The aligner returned word timing for the wrong number of lines.")
            cache.mkdir(parents=True, exist_ok=True)
            errors = reply.get("errors") or []
            for position, ((index, field, offset, duration, _wav, text, path), timed) in enumerate(zip(pending, words)):
                try:
                    timed = _validated_word_timing(text, timed, duration)
                except (ValueError, TypeError, KeyError) as exc:
                    reason = (str(errors[position])[:240] if position < len(errors) and errors[position]
                              else str(exc) if timed else "No valid word timings returned")
                    label = "original" if field == "source" else "translated"
                    notes.append(f"{label} line {index + 1}: {reason}")
                    continue
                found[(index, field)] = (offset, timed)
                try:
                    path.write_text(json.dumps(timed, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
        except JobCancelled:
            raise
        except Exception as exc:  # the dub is finished; only its word highlighting is lost
            notes.append(f"word-by-word subtitles unavailable for {len(pending)} line(s): {exc}")
    missing = []
    for field, label in (("source", "original"), ("translation", "translated")):
        indices = sorted({index + 1 for index, job_field, *_ in jobs
                          if job_field == field and (index, field) not in found})
        if indices:
            numbers = ", ".join(map(str, indices[:20])) + (", …" if len(indices) > 20 else "")
            missing.append(f"{label} lines {numbers}")
    if missing:
        notes.append("word-by-word subtitles unavailable for " + "; ".join(missing)
                     + ". Plain subtitles are kept. Use Generate / repair word alignment to retry.")
    timings: dict[int, dict[str, list[dict]]] = {}
    for (index, field), (offset, timed) in found.items():
        if timed:
            timings.setdefault(index, {})[field] = [
                {"text": word["text"], "start": round(offset + float(word["start"]), 3),
                 "end": round(offset + float(word["end"]), 3)} for word in timed]
    return timings


def subtitle_lines(lines: list[dict], spoken: dict[int, tuple[float, float]]) -> list[dict]:
    """Translated subtitles timed to the dub: each starts with its dubbed line and stays up while
    it is still being spoken, but never past the start of the next line after it, except the last.
    """
    starts = [spoken[index][0] if index in spoken else line["start"] for index, line in enumerate(lines)]
    shown = []
    for index, line in enumerate(lines):
        start, spoken_end = spoken.get(index, (line["start"], line["end"]))
        end = max(line["end"], spoken_end)
        # Another speaker talking over this line does not take its subtitle down.
        later = [starts[other] for other in range(index + 1, len(lines))
                 if lines[other]["start"] >= line["end"] - 1e-6]
        if later:
            end = min(end, max(min(line["end"], spoken_end), min(later) - LINE_GAP))
        shown.append({**line, "start": round(start, 3), "end": round(max(end, start + 0.05), 3)})
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
            spoken: dict[int, tuple[float, float]], duration: float, language: str,
            words: dict[int, dict[str, list[dict]]] | None = None) -> Player:
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
    shown = subtitle_lines(lines, spoken)
    cues = [{"start": line["start"], "end": line["end"], "shown_start": shown[index]["start"],
             "shown_end": shown[index]["end"],
             "speaker": line["speaker"], "source": line["source"], "translation": line["translation"],
             "translation_spoken": index in spoken,
             "source_words": (words or {}).get(index, {}).get("source", []),
             "translation_words": (words or {}).get(index, {}).get("translation", [])}
            for index, line in enumerate(lines) if line["source"] or line["translation"]]
    return Player(video, dub_track, original, music, duration, cues, language,
                  session.data.get("source_language", ""), word_timing=words is not None)


def dub(request: DubRequest, *, runtime, free_gpu, hf_token=None) -> Iterator[str | Dub]:
    session = sessions.load(runtime, request.session_id)
    lines = session.segments
    chosen = [i for i, segment in enumerate(lines) if segment.get("dub", True) and segment["translation"]]
    if not chosen:
        raise H3Error("No line is ready to dub: give lines a translation and tick Dub.")
    duration = float(session.data.get("duration") or 0.0) or max(s["end"] for s in lines) + 1.0
    planned = _plan(session, chosen, request.voice_mode, runtime)
    yield from _synthesize(planned, request, session, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token)

    failed = [line for line in planned if line.take is None or not len(line.take)]
    if failed:
        reasons = "; ".join(f"line {line.index + 1}: {line.note or 'no translated audio returned'}" for line in failed[:20])
        raise H3Error(f"Translated speech generation failed: {reasons}. "
                      "Use Preview to generate a fresh take, or untick these lines to keep their original voice.")
    yield "Placing lines on the timeline"
    clips: dict[int, np.ndarray] = {}
    track, notes, spoken = _speech_track(session, planned, request, duration, clips)
    _record_timing_problems(session, planned)
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
        shown = subtitle_lines(lines, spoken) if field_name == "translation" else lines
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
        "word_timing": bool(request.word_timing),
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
    words = None
    if request.word_timing:
        yield "Timing subtitle words"
        native = {index: {"source": found} for index, line in enumerate(lines)
                  if (found := _native_words(session, line))}
        words = yield from _word_timings(session, lines, spoken, clips, runtime=runtime, free_gpu=free_gpu,
                                         hf_token=hf_token, notes=notes, existing=native,
                                         align_source=session.data.get("pipeline") != PIPELINE_NEMOTRON)
    yield "Preparing the result player"
    _prune_players(output_dir)
    player = _player(session, speech, output_dir / PLAYER_DIR / stem, separated=separated, speech_gain=speech_gain,
                     background_gain=background_gain, video=video_out, mixed=audio_out, lines=lines,
                     spoken=spoken, duration=duration + overrun, language=language, words=words)
    # Freeze the exact fitted clips and dialogue for alignment without regenerating
    # voices or applying any edits made after this dub.
    folder = output_dir / PLAYER_DIR / stem
    try:
        np.savez_compressed(folder / "alignment.npz", **{str(index): clip for index, clip in clips.items()})
        (folder / "alignment.json").write_text(json.dumps({
            "player": asdict(player), "lines": lines, "spoken": spoken,
            "files": dict(session.data.get("files", {})),
        }, ensure_ascii=False, default=str), encoding="utf-8")
    except OSError as exc:
        notes.insert(0, f"Dub saved, but word-alignment repair audio could not be saved: {exc}. "
                     "The repair button is unavailable for this result; generate a dub again after resolving storage.")
        for name in ("alignment.npz", "alignment.json"):
            try:
                (folder / name).unlink(missing_ok=True)
            except OSError:
                pass
    clips.clear()
    for work_file in (speech, mix):
        work_file.unlink(missing_ok=True)
    # Only this field is the dub's: lines edited while it ran stay as saved.
    sessions.modify(session, lambda latest: latest.data.update(last_output=str(video_out or audio_out)))
    sped = sum(line.speed > 1.01 for line in planned)
    slowed = sum(line.speed < 0.99 for line in planned)
    message = f"Dubbed {len(planned)} line(s) into {language or 'the target language'}"
    if request.timing == TIMING_SMART:
        squeezed = sum(bool(line.timing_problem) for line in planned)
        message += f"; {sped} sped up to fit" if sped else ""
        message += f", {squeezed} still too long (marked in Review lines)" if squeezed else ""
    elif request.timing == TIMING_NATURAL:
        message += f"; {sped} sped up to fit" if sped else ""
    elif sped or slowed:
        message += f"; matched to the original timing ({sped} sped up, {slowed} slowed down)"
    if notes:
        message += ".\n\nCheck: " + "; ".join(notes[:12]) + ("; …" if len(notes) > 12 else "")
    yield Dub(session, audio_out, video_out, subtitles, message + ("" if notes else "."), player)


def align_dub_words(session_id: str, *, runtime, free_gpu, hf_token=None) -> Iterator[str | WordAlignment]:
    """Align the latest completed dub's frozen audio and text; reuse successful word caches."""
    session = sessions.load(runtime, session_id)
    output = Path(session.data.get("last_output") or "")
    output_dir = (runtime.outputs_dir / "dubbing").resolve()
    if output.resolve().parent != output_dir or not output.is_file():
        raise H3Error("Generate a dub first; its output files must still exist.")
    folder = output_dir / PLAYER_DIR / output.stem
    try:
        saved = json.loads((folder / "alignment.json").read_text(encoding="utf-8"))
        with np.load(folder / "alignment.npz", allow_pickle=False) as archive:
            clips = {int(index): archive[index] for index in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise H3Error("This dub has no saved alignment audio. Generate the dub once to enable this button.") from exc
    frozen = sessions.Session(session.root, {"files": saved["files"],
        "source_language": saved["player"].get("source_language", ""),
        "target_language": saved["player"].get("target_language", "")})
    if frozen.speech_audio is None or not frozen.speech_audio.is_file():
        raise H3Error("The original audio is no longer available. Analyze the source again.")
    lines = saved["lines"]
    spoken = {int(index): tuple(bounds) for index, bounds in saved["spoken"].items()}
    notes = []
    cue_indices = [index for index, line in enumerate(lines) if line["source"] or line["translation"]]
    existing = {}
    for index, cue in zip(cue_indices, saved["player"]["lines"]):
        for field in ("source", "translation"):
            timed = cue.get(f"{field}_words") or []
            try:
                offset = lines[index]["start"] if field == "source" else spoken[index][0]
                duration = (lines[index]["end"] - offset if field == "source" else len(clips[index]) / RATE)
                _validated_word_timing(lines[index][field], [
                    {"text": word["text"], "start": round(float(word["start"]) - offset, 3),
                     "end": round(float(word["end"]) - offset, 3)} for word in timed], duration)
            except (ValueError, TypeError, KeyError):
                continue
            existing.setdefault(index, {})[field] = timed
    yield "Generating missing word timings; reusing successful alignments"
    words = yield from _word_timings(frozen, lines, spoken, clips, runtime=runtime, free_gpu=free_gpu,
                                     hf_token=hf_token, notes=notes, existing=existing)
    data = saved["player"]
    for field in ("video", "dub", "original", "music"):
        data[field] = Path(data[field]) if data[field] else None
    player = Player(**data)
    for index, cue in zip(cue_indices, player.lines):
        cue["source_words"] = words.get(index, {}).get("source", [])
        cue["translation_words"] = words.get(index, {}).get("translation", [])
    player.word_timing = True
    expected = [(index, field) for index, line in enumerate(lines)
                for field in ("source", "translation")
                if line[field].strip() and (field == "source" or index in clips)]
    completed = sum(bool(words.get(index, {}).get(field)) for index, field in expected)
    before = sum(bool(existing.get(index, {}).get(field)) for index, field in expected)
    message = (f"All word timings generated: {completed}/{len(expected)} eligible original and translated tracks aligned."
               if completed == len(expected) else
               f"Word timings generated: {completed}/{len(expected)} line tracks aligned; "
               f"{len(expected) - completed} still missing. " + " ".join(notes))
    message += (f" Newly aligned this run: {max(0, completed - before)}; already aligned: {before}. "
                "The player below has been refreshed; replay it without generating the dub again."
                if completed == len(expected) else
                f" Newly aligned this run: {max(0, completed - before)}; already aligned: {before}. "
                "Missing tracks still use plain subtitles. Generating the dub again is not required to apply a successful repair.")
    saved["player"] = asdict(player)
    (folder / "alignment.json").write_text(json.dumps(saved, ensure_ascii=False, default=str), encoding="utf-8")
    yield WordAlignment(player, message)
