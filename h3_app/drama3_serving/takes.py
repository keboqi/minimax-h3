"""Content-keyed Drama 3 takes, so unchanged lines are not generated again.

A take is the server's WAV for one chunk, stored before tidying and speaker
effects, which are applied at mix time. Its key covers everything the server
saw for that request; changing any of it makes a new take.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import time
import uuid

import numpy as np

from . import audio

SCHEMA = 1
# Takes unused for this long are removed after a job.
KEEP_SECONDS = 30 * 24 * 3600


def takes_root(runtime) -> Path:
    """Private workspace folder, outside the served gallery outputs."""
    workspace = os.getenv("H3_WORKSPACE_DIR")
    base = Path(workspace) if workspace else Path(runtime.outputs_dir) / ".h3-workspace"
    return base.resolve() / "drama3_takes"


def take_key(*, speaker: int, text: str, reference: str, transcript: str, sampling: dict,
             max_new_tokens: int) -> str:
    """``reference`` identifies the voice clip's content (its staged digest)."""
    identity = {"schema": SCHEMA, "speaker": int(speaker), "text": text, "reference": reference,
                "transcript": transcript, "sampling": {name: sampling[name] for name in sorted(sampling)},
                "max_new_tokens": int(max_new_tokens)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


def _path(root: Path, key: str) -> Path:
    return root / f"{key}.wav"


def load_take(root: Path, key: str) -> tuple[np.ndarray, int] | None:
    """A saved take, or None when it is missing or unreadable (an unreadable file is removed)."""
    path = _path(root, key)
    try:
        samples, rate = audio.decode_wav(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, struct.error) as exc:
        logging.warning("Drama 3 take %s is unreadable; generating it again: %s", key, exc)
        path.unlink(missing_ok=True)
        return None
    if not samples.size or rate <= 0:
        path.unlink(missing_ok=True)
        return None
    try:
        os.utime(path)  # keeps a reused take from being pruned
    except OSError:
        pass
    return samples, rate


def save_take(root: Path, key: str, wav: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = _path(root, key)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(wav)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prune(root: Path, *, now: float | None = None) -> None:
    """Remove takes and leftover temporary files unused for ``KEEP_SECONDS``."""
    cutoff = (time.time() if now is None else now) - KEEP_SECONDS
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue
