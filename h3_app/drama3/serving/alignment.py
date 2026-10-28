"""Cached script alignment on tidied takes, before staging and vocal effects."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import audio
from .planner import spoken_text


def align_take(take, root, runtime, *, free_gpu, hf_token=None):
    from h3_app.dubbing import tools

    lines = [spoken_text(line) for line in take.chunk.lines]
    samples = audio.tidy_take(take.samples, take.rate)
    wav = audio.encode_wav(samples, take.rate)
    digest = hashlib.sha256(wav + json.dumps(lines, ensure_ascii=False).encode()).hexdigest()
    cache = Path(root) / f"{digest}.alignment.json"
    try:
        bounds = json.loads(cache.read_text(encoding="utf-8"))
        try:
            os.utime(cache)
        except OSError:
            pass
    except (OSError, ValueError):
        with tempfile.TemporaryDirectory(prefix="drama3-align-") as directory:
            source = Path(directory) / "take.wav"
            source.write_bytes(wav)
            bounds = tools.run("align_lines", {"audio": str(source), "lines": lines}, runtime,
                               free_gpu=free_gpu, hf_token=hf_token)["bounds"]
        # Cache identity includes actual audio: regeneration invalidates it even with the same take key.
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(bounds), encoding="utf-8")
        except OSError:
            pass
    if len(bounds) != len(lines):
        raise ValueError("Incomplete subtitle alignment")
    duration = len(samples) / take.rate
    previous = 0.0
    for start, end in bounds:
        if not (previous <= start < duration and start < end <= duration + 0.02):
            raise ValueError("Subtitle alignment falls outside the take")
        previous = end
    return [(round(start * take.rate), min(len(samples), round(end * take.rate))) for start, end in bounds]
