"""Cached script alignment on tidied takes, before staging and vocal effects."""

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

from . import audio
from .planner import spoken_text


def _validate(lines, bounds, words, duration):
    from h3_app.dubbing.tools_worker import ALIGNMENT_TIME_TOLERANCE, aligned_line_words

    if len(bounds) != len(lines) or len(words) != len(lines):
        raise ValueError("Incomplete subtitle alignment")
    previous = 0.0
    for start, end in bounds:
        if not (math.isfinite(start) and math.isfinite(end)
                and previous <= start < duration and start < end <= duration + ALIGNMENT_TIME_TOLERANCE):
            raise ValueError("Subtitle alignment falls outside the take")
        previous = end
    checked = aligned_line_words(lines, [word for line in words for word in line])
    if [len(line) for line in checked] != [len(line) for line in words]:
        raise ValueError("Subtitle words cross a line boundary")
    for (start, end), line in zip(bounds, checked):
        if (not line or line[0]["start"] < start - ALIGNMENT_TIME_TOLERANCE
                or line[-1]["end"] > min(end, duration) + ALIGNMENT_TIME_TOLERANCE):
            raise ValueError("Subtitle words fall outside their line")
        if any(word["start"] >= duration for word in line):
            raise ValueError("Subtitle words fall outside the take")
    return checked


def take_audio_digest(samples, rate):
    return hashlib.sha256(audio.encode_wav(audio.tidy_take(samples, rate), rate)).hexdigest()


def align_take(take, root, runtime, *, free_gpu, hf_token=None):
    """(start, end) samples of each script line in the tidied take, and each line's words in seconds."""
    from h3_app.dubbing import tools

    lines = [spoken_text(line) for line in take.chunk.lines or (take.chunk.text,)]
    samples = audio.tidy_take(take.samples, take.rate)
    wav = audio.encode_wav(samples, take.rate)
    digest = hashlib.sha256(wav + json.dumps(lines, ensure_ascii=False).encode()).hexdigest()
    cache = Path(root) / f"{digest}.alignment.json"
    duration = len(samples) / take.rate
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        # Alignments cached before word timing hold only the line bounds: align those again.
        bounds, words = cached["bounds"], cached["words"]
        words = _validate(lines, bounds, words, duration)
        try:
            os.utime(cache)
        except OSError:
            pass
    except (OSError, ValueError, TypeError, KeyError):
        with tempfile.TemporaryDirectory(prefix="drama3-align-") as directory:
            source = Path(directory) / "take.wav"
            source.write_bytes(wav)
            reply = tools.run("align_lines", {"audio": str(source), "lines": lines}, runtime,
                              free_gpu=free_gpu, hf_token=hf_token)
        bounds, words = reply["bounds"], reply.get("words") or [[] for _ in lines]
        # Validate before caching: incomplete results must remain retryable.
        words = _validate(lines, bounds, words, duration)
        # Cache identity includes actual audio: regeneration invalidates it even with the same take key.
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"bounds": bounds, "words": words}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    timed = [[(str(word["text"]), float(word["start"]), min(duration, float(word["end"]))) for word in line]
             for line in words]
    return [(round(start * take.rate), min(len(samples), round(end * take.rate))) for start, end in bounds], timed


def repair_result(path, runtime, *, free_gpu, hf_token=None):
    """Repair a result's subtitles using its saved takes and frozen placement, without remixing."""
    from h3_app.errors import H3Error
    from h3_app.jobs import CURRENT_JOB, JobCancelled
    from h3_app.provenance import read_snapshot, write_snapshot
    from h3_app.dubbing.tools_worker import normalized_text
    from . import takes

    path = Path(path)
    snapshot = read_snapshot(path) if path.is_file() else None
    settings = (snapshot or {}).get("settings") or {}
    chunks = settings.get("chunks") or []
    if not chunks:
        raise H3Error("Generate speech first; the result and its settings must still exist.")
    duration = float(settings.get("duration") or 0)
    total = complete = already = 0
    problems = []
    root = takes.takes_root(runtime)

    def has_words(line):
        words = line.get("words") or []
        try:
            text = normalized_text(spoken_text(line["text"]))
            actual = "".join(normalized_text(word["text"]) for word in words)
            previous = float(line["start"]) - .02
            for word in words:
                start, end = float(word["start"]), float(word["end"])
                if not (math.isfinite(start) and math.isfinite(end)
                        and previous <= start <= end <= min(duration, float(line["end"])) + .02):
                    return False
                previous = end
            return bool(actual) and (text == actual or (float(line["end"]) >= duration and text.startswith(actual)))
        except (KeyError, TypeError, ValueError):
            return False

    for number, chunk in enumerate(chunks, 1):
        job = CURRENT_JOB.get()
        if job is not None:
            job.check()
        lines = chunk.get("lines") or []
        eligible = [line for line in lines if spoken_text(line.get("text", ""))
                    and float(line.get("start", duration)) < duration]
        total += len(eligible)
        already += sum(has_words(line) for line in eligible)
        if not eligible:
            continue
        if all(has_words(line) for line in eligible):
            complete += len(eligible)
            continue
        yield f"Aligning saved speech: chunk {number}/{len(chunks)}; successful timings are reused"
        try:
            placement = chunk.get("alignment")
            if not placement:
                raise ValueError("Remix once with saved takes to enable repair for this older result")
            saved = takes.load_take(root, str(chunk.get("take", "")))
            if saved is None:
                raise ValueError("saved take is missing; generate speech again")
            samples, rate = saved
            if take_audio_digest(samples, rate) != placement["audio"]:
                raise ValueError("saved take changed since this result; remix first")
            take = SimpleNamespace(samples=samples, rate=rate, chunk=SimpleNamespace(
                lines=tuple(line["text"] for line in lines), text=chunk["text"]))
            bounds, words = align_take(take, root, runtime, free_gpu=free_gpu, hf_token=hf_token)
            origin, scale = float(placement["start"]), float(placement["scale"])
            for line, (start, end), timed in zip(lines, bounds, words):
                line["start"] = round(min(duration, origin + start / rate * scale), 3)
                line["end"] = round(min(duration, origin + end / rate * scale), 3)
                line["words"] = [{"text": text, "start": round(origin + a * scale, 3),
                                  "end": round(min(duration, origin + b * scale), 3)}
                                 for text, a, b in timed if origin + a * scale < duration]
            chunk["subtitle_timing"] = "aligned"
        except JobCancelled:
            raise
        except Exception as exc:
            problems.append(f"chunk {number}: {exc}")
        complete += sum(has_words(line) for line in eligible)
    settings["word_timing"] = True
    settings["word_alignment"] = {"aligned": complete, "total": total, "problems": problems}
    repair_job = CURRENT_JOB.get()
    if repair_job is not None:
        settings["word_alignment"]["repair_job_id"] = repair_job.id
    write_snapshot(path, {key: value for key, value in snapshot.items() if key not in {"schema_version", "output"}},
                   record_job=False)
    yield ("No playable subtitle lines were found. Remix from saved takes to generate subtitle timing."
           if not total else f"All word timings generated: {complete}/{total} lines aligned." if complete == total else
           f"Word timings generated: {complete}/{total} lines aligned; {total - complete} still missing. "
           + "; ".join(problems)) + (
               f" Newly aligned this run: {max(0, complete - already)}; already aligned: {already}. "
               "Subtitles have been refreshed; replay the existing result without regenerating speech or remixing."
               if total and complete == total else
               f" Newly aligned this run: {max(0, complete - already)}; already aligned: {already}. "
               "Missing lines still use plain subtitles. Repeating repair may return the same alignment error.")
