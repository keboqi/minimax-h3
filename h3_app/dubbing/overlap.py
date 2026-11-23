"""Isolate and identify local overlap windows for Drama 3 clone references."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
import logging
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import wave

import numpy as np

from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled, check_cancelled
from h3_requirements import SR_CORRNET_REVISION

from . import nemotron, tools, tools_worker
from .pcm import FITTED_AUDIO_RATE as RATE, read_slice, write_wav
from .voices import MIN_REFERENCE_SECONDS, REFERENCE_PAD

LOG = logging.getLogger(__name__)
REVISION = 3
MAX_WINDOW = 8.0
CONTEXT = 1.25
MIN_OVERLAP = .15
MIN_MATCHED_WORDS = 3
WORD_TOLERANCE = .40
WINDOW_BATCH = 3
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


@dataclass(frozen=True, order=True)
class Window:
    start: float
    end: float
    speakers: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceWindow:
    window: Window
    folder: Path
    expected: dict[str, list[dict]]


def _activity(transcript: dict) -> dict[str, list[tuple[float, float]]]:
    segments = transcript["segments"]
    activity = transcript.get("activity")
    spans = defaultdict(list)
    if activity:
        for speaker, bounds in activity.items():
            spans[speaker].extend((start, end) for start, end in bounds if end > start)
    else:
        for row in segments:
            spans[row["speaker"]].append((row["start"], row["end"]))
    for hint in transcript.get("overlaps", []):
        for speaker in hint["speakers"]:
            spans[speaker].append((hint["start"], hint["end"]))
    return {speaker: tools_worker.merge_spans(bounds, 0) for speaker, bounds in spans.items()}


def _intersections(activity: dict[str, list[tuple[float, float]]]) -> list[Window]:
    events = defaultdict(list)
    for speaker, bounds in activity.items():
        for start, end in bounds:
            events[start].append((speaker, 1))
            events[end].append((speaker, -1))
    active: Counter[str] = Counter()
    windows = []
    left = 0.0
    for right in sorted(events):
        speakers = tuple(sorted(speaker for speaker, count in active.items() if count > 0))
        if len(speakers) >= 2 and right - left >= MIN_OVERLAP:
            windows.append(Window(left, right, speakers))
        for speaker, delta in events[right]:
            active[speaker] += delta
        left = right
    return windows


def windows_for(transcript: dict, duration: float) -> tuple[list[Window], list[Window]]:
    """Find true overlap candidates, then add context without introducing another voice."""
    if not math.isfinite(duration) or duration <= 0:
        return [], []
    activity = _activity(transcript)
    hints = transcript.get("overlaps")
    overlaps = ([Window(hint["start"], hint["end"], tuple(sorted(set(hint["speakers"])))) for hint in hints
                 if hint["end"] - hint["start"] >= MIN_OVERLAP]
                if hints is not None else _intersections(activity))
    grouped_spans: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for overlap in overlaps:
        if len(overlap.speakers) not in tools_worker.OVERLAP_MODELS or nemotron.UNKNOWN_SPEAKER in overlap.speakers:
            continue
        outsiders = [span for speaker, spans in activity.items() if speaker not in overlap.speakers for span in spans]
        if any(start < overlap.end and end > overlap.start for start, end in outsiders):
            continue
        left = max([0.0, overlap.start - CONTEXT,
                    *(min(end, overlap.start) for start, end in outsiders if start < overlap.start)])
        right = min([duration, overlap.end + CONTEXT,
                     *(max(start, overlap.end) for start, end in outsiders if end > overlap.end)])
        if right > left:
            grouped_spans[overlap.speakers].append((left, right))
    windows = set()
    for speakers, spans in grouped_spans.items():
        for left, right in tools_worker.merge_spans(spans, 0):
            if right - left < MIN_REFERENCE_SECONDS:
                continue
            first = left
            while first + MAX_WINDOW < right:
                windows.add(Window(round(first, 3), round(first + MAX_WINDOW, 3), speakers))
                first += MAX_WINDOW - 2 * CONTEXT
            windows.add(Window(round(max(left, right - MAX_WINDOW), 3), round(right, 3), speakers))
    return overlaps, sorted(windows)


def has_overlap_candidates(session, lines: dict[int, dict]) -> bool:
    """Whether any candidate line intersects an overlap interval in ``session``'s transcript."""
    transcript = session.data.get("transcript")
    if not isinstance(transcript, dict) or not transcript.get("segments"):
        return False
    duration = float(session.data.get("duration") or max(row["end"] for row in transcript["segments"]))
    overlaps, _ = windows_for(transcript, duration)
    return any(line["start"] < overlap.end and line["end"] > overlap.start
               for line in lines.values() for overlap in overlaps)


def _tokens(word: dict) -> list[tuple[str, float, float]]:
    text = tools_worker.CJK.sub(r" \g<0> ", word["text"])
    return [(tools_worker.normalized_text(match[0]), word["start"], word["end"]) for match in TOKEN.finditer(text)]


def expected_words(transcript: dict, window: Window) -> dict[str, list[dict]]:
    native = transcript.get("words")
    rows = native or transcript["segments"]
    result = {speaker: [] for speaker in window.speakers}
    local = (row for row in rows if row["speaker"] in result
             and row["start"] < window.end and row["end"] > window.start)
    for row in sorted(local, key=lambda row: (row["start"], row["end"])):
        result[row["speaker"]].append({"text": row["text"] if native else row["source"],
                                       "start": row["start"], "end": row["end"]})
    return result


def _match_count(heard: list[tuple], expected: list[tuple]) -> int:
    """Longest ordered token match whose intervals agree on the source timeline."""
    counts = [0] * (len(expected) + 1)
    for token, start, end in heard:
        previous = 0
        for i, (other, low, high) in enumerate(expected, 1):
            before = counts[i]
            if token == other and start <= high + WORD_TOLERANCE and end >= low - WORD_TOLERANCE:
                counts[i] = previous + 1
            else:
                counts[i] = max(before, counts[i - 1])
            previous = before
    return counts[-1]


def attribute(window: Window, expected: dict[str, list[dict]], stems: list[list[dict]]) -> dict[str, int]:
    """Assign the anonymous stems together, requiring independent evidence for each speaker."""
    if len(stems) != len(window.speakers):
        return {}
    known = [[token for word in expected[speaker] for token in _tokens(word)] for speaker in window.speakers]
    heard = [[(token, start + window.start, end + window.start)
              for word in words for token, start, end in _tokens(word)] for words in stems]
    scores = [[_match_count(tokens, words) for words in known] for tokens in heard]
    coverage = [[count / max(1, len(tokens)) for count in row] for tokens, row in zip(heard, scores)]
    peaks = [max((row[speaker] for row in scores), default=0) for speaker in range(len(known))]
    evidence = [[(count / max(MIN_MATCHED_WORDS, len(tokens))) * (count / max(MIN_MATCHED_WORDS, peaks[speaker]))
                 for speaker, count in enumerate(row)] for tokens, row in zip(heard, scores)]
    ranked = sorted((sum(evidence[i][speaker] for i, speaker in enumerate(order)), order)
                    for order in permutations(range(len(known))))
    best, order = ranked[-1]
    assignment = {}
    for index, speaker in enumerate(order):
        alternative = next(score for score, other in reversed(ranked) if other[index] != speaker)
        rivals = [count for j, count in enumerate(scores[index]) if j != speaker]
        other_stems = [evidence[other][speaker] for other in range(len(stems)) if other != index]
        count = scores[index][speaker]
        if (best - alternative >= .18 and count >= MIN_MATCHED_WORDS and coverage[index][speaker] >= .75
                and count >= max(rivals) + 2 and max(rivals) <= .35 * count
                and count >= peaks[speaker] and max(other_stems, default=0.0) <= .65 * evidence[index][speaker]):
            assignment[window.speakers[speaker]] = index
    return assignment


def _read_cache(folder: Path) -> dict:
    try:
        stems = json.loads((folder / "stems.json").read_text(encoding="utf-8"))
        if isinstance(stems, dict) and all((folder / stem["file"]).is_file() for stem in stems.values()):
            return stems
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return {}


def _save_stems(folder: Path, scratch: Path, assignment: dict[str, int], paths: list[str],
                words: list[list[dict]]) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    data = {}
    for speaker, index in assignment.items():
        source = Path(paths[index])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
        target = folder / f"{digest}.wav"
        if not target.is_file():
            source.replace(target)
        data[speaker] = {"file": target.name, "words": words[index]}
    manifest = scratch / "stems.json"
    manifest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    manifest.replace(folder / manifest.name)
    return data


def _resolve_batch(windows: list[Window], transcript: dict, audio: Path, root: Path, language: str, run, failures,
                   *, skip_worker: bool = False) -> tuple[list[tuple[ReferenceWindow, dict]], bool]:
    count = len(windows[0].speakers)
    resolved, pending = [], []
    with TemporaryDirectory(dir=root) as tmp:
        scratch = Path(tmp)
        for window in windows:
            check_cancelled()
            try:
                samples = read_slice(audio, window.start, window.end)
                expected = expected_words(transcript, window)
                identity = json.dumps([REVISION, SR_CORRNET_REVISION, tools_worker.OVERLAP_MODELS[count],
                                       tools_worker.ASR_ID, tools_worker.ALIGNER_ID,
                                       vars(window), language, expected], sort_keys=True).encode()
                folder = root / hashlib.sha256(identity + samples.tobytes()).hexdigest()[:32]
                job = ReferenceWindow(window, folder, expected)
                cached = _read_cache(folder)
                if cached:
                    resolved.append((job, cached))
                elif skip_worker:
                    failures["worker errors"] += 1
                else:
                    path = write_wav(samples, RATE, scratch / f"window-{len(pending)}.wav")
                    pending.append((job, str(path)))
            except (OSError, ValueError, EOFError, wave.Error):
                LOG.exception("Could not read audio for an overlap window")
                failures["audio errors"] += 1
        if not pending:
            return resolved, skip_worker

        try:
            separated = run("separate_overlap", {"speaker_count": count, "windows": [path for _, path in pending]})["stems"]
            if len(separated) != len(pending) or any(stems and len(stems) != count for stems in separated):
                raise H3Error("Unexpected stem count from the overlap worker")
            clips = [path for stems in separated if stems for path in stems]
            recognized = run("transcribe_stems", {"clips": clips, "language": language})["words"] if clips else []
            if len(recognized) != len(clips):
                raise H3Error("Unexpected transcription count from the overlap worker")
        except JobCancelled:
            raise
        except (H3Error, OSError, RuntimeError, KeyError, TypeError):
            check_cancelled()
            LOG.exception("Overlap reference worker failed")
            failures["worker errors"] += len(pending)
            return resolved, True

        heard = iter(recognized)
        for (job, _), paths in zip(pending, separated, strict=True):
            check_cancelled()
            if not paths:
                failures["worker errors" if paths is None else "unverified stems"] += 1
                continue
            words = [next(heard) for _ in paths]
            if any(items is None for items in words):
                failures["stem transcription errors"] += 1
                continue
            assignment = attribute(job.window, job.expected, words)
            if assignment:
                try:
                    data = _save_stems(job.folder, scratch, assignment, paths, words)
                    resolved.append((job, data))
                except OSError:
                    LOG.exception("Could not cache an overlap window")
                    failures["audio errors"] += 1
            else:
                failures["unverified stems"] += 1
    return resolved, False


def _crop(window: Window, line: dict, entry: dict, folder: Path, expected: list[dict]):
    start = max(0.0, line["start"] - window.start)
    end = min(window.end - window.start, line["end"] - window.start)
    words = [word for word in entry["words"]
             if word["start"] >= start - .08 and word["end"] <= end + .08]
    if not words or words[-1]["end"] - words[0]["start"] < MIN_REFERENCE_SECONDS:
        return None
    if any(after["start"] - before["end"] > 1.25 for before, after in zip(words, words[1:])):
        return None
    wanted = [token for word in expected for token in _tokens(word)]
    actual = [(token, left + window.start, right + window.start)
              for word in words for token, left, right in _tokens(word)]
    matches = _match_count(actual, wanted)
    if matches < MIN_MATCHED_WORDS or matches < .85 * len(actual):
        return None
    left, right = max(0.0, words[0]["start"] - REFERENCE_PAD), words[-1]["end"] + REFERENCE_PAD
    transcript = nemotron.join_words([word["text"] for word in words])
    src = folder / entry["file"]
    identity = json.dumps([src.stem, left, right, transcript], ensure_ascii=False).encode()
    target = folder / f"reference-{hashlib.sha256(identity).hexdigest()[:24]}.wav"
    if not target.is_file():
        samples = read_slice(src, left, right)
        if len(samples) < RATE * MIN_REFERENCE_SECONDS or np.sqrt(np.mean(samples ** 2)) < .003:
            return None
        with TemporaryDirectory(dir=folder) as tmp:
            write_wav(samples, RATE, Path(tmp) / target.name).replace(target)
    return target, transcript


def _original_line(transcript: dict, line: dict) -> dict | None:
    # Speaker/time edits can change identity. Only accept an unambiguous source row.
    matches = [row for row in transcript["segments"] if row["speaker"] == line["speaker"]
               and abs(row["start"] - line["start"]) <= .35 and abs(row["end"] - line["end"]) <= .35
               and tools_worker.normalized_text(row["source"]) == tools_worker.normalized_text(line["source"])]
    return matches[0] if len(matches) == 1 else None


def prepare(session, lines: dict[int, dict], runtime, *, free_gpu, hf_token, report):
    """Return vetted references and a user-facing summary for requested overlap lines."""
    transcript = session.data["transcript"]
    duration = float(session.data.get("duration") or max(row["end"] for row in transcript["segments"]))
    overlaps, windows = windows_for(transcript, duration)
    candidates = {i: line for i, line in lines.items()
                  if any(line["start"] < overlap.end and line["end"] > overlap.start
                         for overlap in overlaps)}
    if not candidates:
        return {}, ""
    raw_language = session.data.get("source_language") or ""
    language = tools_worker.alignment_language(raw_language)
    if language is None:
        return {}, (f"Overlap references: 0 isolated, {len(candidates)} used the usual reference fallback "
                    f"(word timing does not support {raw_language}).")
    jobs = {}
    originals = {i: _original_line(transcript, line) for i, line in candidates.items()}
    for window in windows:
        indices = [i for i, line in candidates.items() if originals[i] is not None
                   and line["speaker"] in window.speakers
                   and min(line["end"], window.end) - max(line["start"], window.start) >= MIN_REFERENCE_SECONDS]
        if indices:
            jobs[window] = indices

    audio = session.speech_audio
    root = session.root / "references" / "overlap"
    found, ranks = {}, {}
    failures: Counter[str] = Counter()
    crop_errors = 0

    def run(op, payload):
        return tools.run(op, payload, runtime, free_gpu=free_gpu, hf_token=hf_token, report=report)

    if jobs and (audio is None or not audio.is_file()):
        failures["audio errors"] += len(jobs)
    elif jobs:
        root.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        for window in jobs:
            groups[len(window.speakers)].append(window)
        worker_failed = False
        for _, group in sorted(groups.items()):
            for first in range(0, len(group), WINDOW_BATCH):
                batch, worker_failed = _resolve_batch(
                    group[first:first + WINDOW_BATCH], transcript, audio, root, language, run, failures,
                    skip_worker=worker_failed)
                for job, data in batch:
                    for index in jobs[job.window]:
                        check_cancelled()
                        line = candidates[index]
                        speaker = line["speaker"]
                        if speaker not in data:
                            continue
                        # For Gemini, corroborate the final crop against the original utterance.
                        words = (job.expected[speaker] if transcript.get("words")
                                 else [{**originals[index], "text": line["source"]}])
                        try:
                            reference = _crop(job.window, line, data[speaker], job.folder, words)
                            if reference:
                                rank = (reference[0].stat().st_size, -job.window.start, -job.window.end)
                                if index not in ranks or rank > ranks[index]:
                                    found[index], ranks[index] = reference, rank
                        except (OSError, ValueError, EOFError, wave.Error):
                            LOG.exception("Could not crop an isolated voice reference")
                            crop_errors += 1
    fallback = len(candidates) - len(found)
    message = f"Overlap references: {len(found)} isolated"
    if fallback:
        message += f", {fallback} used the usual reference fallback"
        reasons = [f"{count} window{'' if count == 1 else 's'} with {reason}" for reason, count in failures.items()]
        if crop_errors:
            reasons.append(f"{crop_errors} reference crop error{'' if crop_errors == 1 else 's'}")
        if reasons:
            message += " (" + ", ".join(reasons) + ")"
        else:
            message += " (no suitable verified clip for those lines)"
    return found, message + "."
