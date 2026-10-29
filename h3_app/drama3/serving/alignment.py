"""Cached script alignment on tidied takes, before staging and vocal effects."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import audio
from .planner import spoken_text


def align_take(take, root, runtime, *, free_gpu, hf_token=None):
    """(start, end) samples of each script line in the tidied take, and each line's words in seconds."""
    from h3_app.dubbing import tools

    lines = [spoken_text(line) for line in take.chunk.lines or (take.chunk.text,)]
    samples = audio.tidy_take(take.samples, take.rate)
    wav = audio.encode_wav(samples, take.rate)
    digest = hashlib.sha256(wav + json.dumps(lines, ensure_ascii=False).encode()).hexdigest()
    cache = Path(root) / f"{digest}.alignment.json"
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        # Alignments cached before word timing hold only the line bounds: align those again.
        bounds, words = cached["bounds"], cached["words"]
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
        # Cache identity includes actual audio: regeneration invalidates it even with the same take key.
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"bounds": bounds, "words": words}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    if len(bounds) != len(lines) or len(words) != len(lines):
        raise ValueError("Incomplete subtitle alignment")
    duration = len(samples) / take.rate
    previous = 0.0
    for start, end in bounds:
        if not (previous <= start < duration and start < end <= duration + 0.02):
            raise ValueError("Subtitle alignment falls outside the take")
        previous = end
    timed = [[(str(word["text"]), float(word["start"]), min(duration, float(word["end"]))) for word in line]
             for line in words]
    return [(round(start * take.rate), min(len(samples), round(end * take.rate))) for start, end in bounds], timed
