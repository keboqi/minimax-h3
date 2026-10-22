"""Dub & Translate: turn a source into timed lines, then voice them with Drama 3 and remix.

Analysis extracts the soundtrack, optionally separates vocals from the background,
and transcribes with Gemini or locally (OmniVAD + Qwen3-ASR + ForcedAligner, then
Hy-MT). Dubbing clones each line's voice from the original audio, fits the take to
its slot, lays the lines on the timeline and mixes them over the background. Takes
are content-keyed, so an unchanged line is never generated twice.

Each step is a generator of status strings that ends with one result object.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Iterator
import uuid
import wave

import numpy as np

from h3_app.catalog import VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.media_tools import probe_audio_seconds, run_media_process
from h3_app.provenance import write_snapshot

from . import gemini, segments as segment_tools, session as sessions, tools

RATE = 44100
PIPELINE_LOCAL = "Local: Qwen3-ASR + aligner + Hy-MT"
PIPELINE_GEMINI = "Gemini (cloud)"
PIPELINES = (PIPELINE_LOCAL, PIPELINE_GEMINI)
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


@dataclass(frozen=True)
class DubRequest:
    session_id: str
    voice_mode: str = VOICE_LINE
    preset: str = "Balance"
    timing: str = TIMING_MATCH
    parallel: int = 8
    max_speed: float = 1.35
    speech_volume: float = 100.0
    background_volume: float = 100.0
    keep_original_track: bool = True
    burn_subtitles: bool = False
    regenerate: tuple[int, ...] = ()


@dataclass
class Analysis:
    session: sessions.Session
    message: str


@dataclass
class Dub:
    session: sessions.Session
    audio: Path
    video: Path | None
    subtitles: list[Path]
    message: str


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
    session = sessions.create(runtime, request.source, video=is_video(request.source))
    session.data.update(pipeline=request.pipeline, target_language=target_language,
                        source_language=_language(request.source_language))
    yield "Extracting the soundtrack"
    audio = _to_pcm(session.path("source"), session.root / "audio.wav", action="Extracting the soundtrack")
    session.set_file("audio", audio)
    session.data["duration"] = round(probe_audio_seconds(audio) or 0.0, 3)
    session.save()

    if request.separate:
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

    speech = session.speech_audio
    if request.pipeline == PIPELINE_GEMINI:
        upload = _ffmpeg(["-i", str(speech), "-ac", "1", "-c:a", "libmp3lame", "-b:a", "96k"],
                         session.root / "speech.mp3", action="Preparing audio for Gemini")
        found, speakers, _raw = yield from blocking(
            f"Transcribing{' and translating' if request.translate else ''} with {request.gemini_model}",
            lambda _report: gemini.transcribe(upload, model=request.gemini_model, api_key=request.api_key,
                                              target_language=target_language, translate=request.translate,
                                              ignore_non_speech=request.ignore_non_speech,
                                              custom_prompt=""))
        session.segments, session.data["speakers"] = found, speakers
        session.save()
        if request.translate:
            # Gemini already translated; the translator only fills lines it left empty.
            missing = [i for i, s in enumerate(session.segments) if s["source"] and not s["translation"]]
            if missing:
                yield from _translate(session, missing, request, runtime=runtime, free_gpu=free_gpu,
                                      hf_token=hf_token)
    else:
        reply = yield from blocking(
            "Finding speech and transcribing with Qwen3-ASR",
            lambda report: tools.run("transcribe", {"audio": str(speech), "language": _language(
                request.source_language) or None, "aligner": bool(request.aligner),
                "merge_gap": float(request.merge_gap)}, runtime, free_gpu=free_gpu, hf_token=hf_token,
                report=report))
        if not reply.get("segments"):
            raise H3Error("No speech was found in this source.")
        session.segments = [{"start": s["start"], "end": s["end"], "source": s["text"]} for s in reply["segments"]]
        if reply.get("language") and not session.data.get("source_language"):
            session.data["source_language"] = reply["language"]
        session.data["speakers"] = {segment_tools.DEFAULT_SPEAKER: "Single voice (local transcription)"}
        session.save()
        if request.translate:
            yield from _translate(session, list(range(len(session.segments))), request, runtime=runtime,
                                  free_gpu=free_gpu, hf_token=hf_token)
    count = len(session.segments)
    speakers = len({s["speaker"] for s in session.segments})
    translated = sum(bool(s["translation"]) for s in session.segments)
    yield Analysis(session, f"Found {count} line(s) from {speakers} speaker(s); {translated} translated into "
                            f"{target_language or 'nothing yet'}. Review the lines, then dub.")


def _translate(session: sessions.Session, indices: list[int], request, *, runtime, free_gpu,
               hf_token=None) -> Iterator[str]:
    lines = session.segments
    texts = [lines[i]["source"] for i in indices]
    target = session.data.get("target_language") or request.target_language
    if request.translator == TRANSLATOR_GEMINI:
        translated = []
        for first in range(0, len(texts), GEMINI_TRANSLATE_BATCH):
            batch = texts[first:first + GEMINI_TRANSLATE_BATCH]
            translated += yield from blocking(
                f"Translating lines {first + 1}-{first + len(batch)} of {len(texts)} with Gemini",
                lambda _report, batch=batch: gemini.translate(
                    batch, model=request.gemini_model, api_key=request.api_key, target_language=target,
                    source_language=session.data.get("source_language", ""), notes=request.notes))
    else:
        reply = yield from blocking(
            f"Translating {len(texts)} line(s) into {target} with Hy-MT",
            lambda report: tools.run("translate", {"texts": texts, "target_language": target,
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
    yield Analysis(session, f"Translated {len(indices)} line(s) into {session.data['target_language']}.")


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


def _plan(session: sessions.Session, indices: list[int], voice_mode: str) -> list[_Line]:
    lines = session.segments
    planned = []
    speaker_refs: dict[str, tuple[Path, str]] = {}
    for index in indices:
        segment = lines[index]
        if not segment["translation"]:
            continue
        own_length = segment["end"] - segment["start"]
        if voice_mode == VOICE_LINE and segment["source"] and own_length >= MIN_REFERENCE_SECONDS:
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


def _stretch(samples: np.ndarray, rate: int, tempo: float) -> np.ndarray:
    """Play speech ``tempo`` times faster (below 1: slower), pitch kept, with ffmpeg's atempo."""
    from h3_app.drama3.serving.audio import decode_wav

    with tempfile.TemporaryDirectory(prefix="dub-stretch-") as directory:
        source = _write_wav(samples, rate, Path(directory) / "take.wav")
        target = _ffmpeg(["-i", str(source), "-af", atempo_chain(tempo), "-ar", str(rate), "-ac", "1"],
                         Path(directory) / "timed.wav", timeout=300, action="Fitting a line to its timing")
        return decode_wav(target.read_bytes())[0]


def match(samples: np.ndarray, rate: int, target: float) -> tuple[np.ndarray, float, str]:
    """Stretch a take to exactly ``target`` seconds, then pad or trim the last samples."""
    seconds = len(samples) / rate
    if target <= 0 or not len(samples):
        return samples, 1.0, ""
    tempo = seconds / target
    timed = _stretch(samples, rate, tempo) if abs(seconds - target) > 0.03 else samples
    size = int(round(target * rate))
    timed = timed[:size] if len(timed) >= size else np.concatenate(
        [timed, np.zeros(size - len(timed), dtype=np.float32)])
    note = ""
    if tempo > TEMPO_WARNING[1]:
        note = f"sped up {tempo:.2f}x to match; a shorter translation would sound more natural"
    elif tempo < TEMPO_WARNING[0]:
        note = f"slowed to {tempo:.2f}x to match; a longer translation would sound more natural"
    return timed, tempo, note


def fit(samples: np.ndarray, rate: int, available: float, max_speed: float) -> tuple[np.ndarray, float, str]:
    """Speed a take up to fit ``available`` seconds, by at most ``max_speed``; reports what it did."""
    seconds = len(samples) / rate
    if seconds <= available + 0.02 or available <= 0:
        return samples, 1.0, ""
    factor = min(seconds / available, max(1.0, float(max_speed)))
    fitted = _stretch(samples, rate, factor) if factor > 1.01 else samples
    over = len(fitted) / rate - available
    note = f"{over:.1f}s longer than its slot" if over > 0.05 else ""
    return fitted, factor, note


def timed_take(take: np.ndarray, lines: list[dict], index: int, duration: float,
               request: DubRequest) -> tuple[np.ndarray, float, str, float]:
    """The take as it will play: (samples, tempo, note, seconds it was fitted to)."""
    segment = lines[index]
    if request.timing == TIMING_NATURAL:
        available = slot_end(lines, index, duration) - segment["start"]
        return (*fit(take, RATE, available, request.max_speed), available)
    available = segment["end"] - segment["start"]
    return (*match(take, RATE, available), available)


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
    lines = [(line, line.segment["translation"], staged[line.reference], line.transcript) for line in pending]
    with closing(speaker.speak(lines, sampling=sampling, parallel=request.parallel,
                               thread_prefix="dub-sglang")) as batches:
        for batch in batches:
            for spoken in batch:
                line = spoken.key
                line.take = _resample(audio.tidy_take(spoken.samples, spoken.rate), spoken.rate, RATE)
                line.note = f"kept its best take ({spoken.problem})" if spoken.problem else ""
                try:
                    take_store.save_take(store, line.key, spoken.wav)
                except OSError as exc:  # a full disk must not fail the dub
                    line.note = f"take not saved: {exc}"
                done += 1
            yield f"Voicing lines: {done}/{len(pending)} ({time.monotonic() - started:.0f}s)"
    persist_caches()
    take_store.prune(store)


def preview_line(request: DubRequest, index: int, *, runtime, free_gpu, hf_token=None) -> Iterator[str | LinePreview]:
    session = sessions.load(runtime, request.session_id)
    lines = session.segments
    if not 0 <= index < len(lines):
        raise H3Error(f"No line {index + 1}; the table has {len(lines)} line(s).")
    planned = _plan(session, [index], request.voice_mode)
    if not planned:
        raise H3Error(f"Line {index + 1} has no translation to speak.")
    yield from _synthesize(planned, request, session, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token)
    line = planned[0]
    fitted, speed, note, available = timed_take(line.take, lines, index, session.data.get("duration", 0.0),
                                                request)
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
    for line in planned:
        if line.take is None or not len(line.take):
            notes.append(f"line {line.index + 1}: no audio, original kept")
            continue
        fitted, speed, note, _available = timed_take(line.take, lines, line.index, duration, request)
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


def dub(request: DubRequest, *, runtime, free_gpu, hf_token=None) -> Iterator[str | Dub]:
    session = sessions.load(runtime, request.session_id)
    lines = session.segments
    chosen = [i for i, segment in enumerate(lines) if segment.get("dub", True) and segment["translation"]]
    if not chosen:
        raise H3Error("No line is ready to dub: give lines a translation and tick Dub.")
    duration = float(session.data.get("duration") or 0.0) or max(s["end"] for s in lines) + 1.0
    planned = _plan(session, chosen, request.voice_mode)
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
    audio_out = _ffmpeg(["-i", str(mix), "-c:a", "libmp3lame", "-q:a", "2"], output_dir / f"{stem}.mp3",
                        action="Encoding the dubbed audio")
    settings = {
        "source": session.data.get("name"), "target_language": language,
        "source_language": session.data.get("source_language", ""), "pipeline": session.data.get("pipeline"),
        "separated": separated, "voice_mode": request.voice_mode, "preset": request.preset,
        "timing": request.timing, "max_speed": request.max_speed, "speech_volume": request.speech_volume,
        "background_volume": request.background_volume, "lines": len(lines), "dubbed_lines": len(planned),
        "sped_up_lines": sum(line.speed > 1.01 for line in planned), "session": session.id,
    }
    write_snapshot(audio_out, {"family": "Dub & Translate", "settings": settings})

    video_out = None
    if session.data.get("video"):
        yield "Writing the dubbed video" + (" with burned-in subtitles" if request.burn_subtitles else "")
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
        # A dub longer than the source keeps going over the last frame: re-encoding holds it
        # (tpad); a copied stream simply ends, and players show its last frame.
        filters = []
        if request.burn_subtitles and subtitles:
            filters.append(f"subtitles={_filter_path(subtitles[0])}")
        if filters and overrun > 0.05:
            filters.append(f"tpad=stop_mode=clone:stop_duration={overrun:.3f}")
        video_codec = ["-c:v", "copy"]
        if filters:
            video_codec = ["-vf", ",".join(filters), "-c:v", "libx264", "-crf", "18",
                           "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        video_out = _ffmpeg([*inputs, *maps, *video_codec, "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
                             *metadata, "-movflags", "+faststart"], output_dir / f"{stem}.mp4",
                            timeout=runtime.generation_timeout, action="Writing the dubbed video")
        write_snapshot(video_out, {"family": "Dub & Translate", "settings": {
            **settings, "original_track": request.keep_original_track, "burned_subtitles": request.burn_subtitles}})
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
    yield Dub(session, audio_out, video_out, subtitles, message + ("" if notes else "."))
