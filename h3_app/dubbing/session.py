"""Dub & Translate sessions: a private folder per source with its stems, references and lines.

``session.json`` holds the source facts and the edited segments; audio work files
sit beside it. Sessions live in the private workspace, which is never served.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Callable
import uuid

from h3_app.errors import H3Error
from h3_app.video_download import workspace_dir

from . import segments as segment_tools

MANIFEST = "session.json"
# Sessions untouched for this long are removed when another one starts.
KEEP_SECONDS = 14 * 24 * 3600
_LOCK = threading.RLock()


def sessions_dir(runtime) -> Path:
    return workspace_dir(runtime) / "dubbing"


@dataclass
class Session:
    root: Path
    data: dict

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def segments(self) -> list[dict]:
        return self.data.get("segments") or []

    @segments.setter
    def segments(self, value) -> None:
        self.data["segments"] = segment_tools.normalize(value)

    def path(self, key: str) -> Path | None:
        """A work file named in the manifest (``audio``, ``vocals``, ``backing``, ``video``)."""
        name = self.data.get("files", {}).get(key)
        return self.root / name if name else None

    def set_file(self, key: str, path: Path | None) -> None:
        files = self.data.setdefault("files", {})
        if path is None:
            files.pop(key, None)
        else:
            files[key] = Path(path).relative_to(self.root).as_posix()

    @property
    def speech_audio(self) -> Path:
        """Separated vocals when available, otherwise the full soundtrack."""
        return self.path("vocals") or self.path("audio")

    def save(self) -> None:
        with _LOCK:
            self.data["updated"] = time.time()
            target = self.root / MANIFEST
            temporary = target.with_name(f"{MANIFEST}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def prune(runtime, *, now: float | None = None) -> None:
    """Remove sessions untouched for ``KEEP_SECONDS``: each holds a source copy and full-length WAVs."""
    root = sessions_dir(runtime)
    if not root.is_dir():
        return
    cutoff = (time.time() if now is None else now) - KEEP_SECONDS
    for folder in root.iterdir():
        manifest = folder / MANIFEST
        try:
            stale = folder.is_dir() and (manifest.stat().st_mtime if manifest.is_file()
                                         else folder.stat().st_mtime) < cutoff
        except OSError:
            continue
        if stale:
            shutil.rmtree(folder, ignore_errors=True)


def recent(runtime, limit: int = 30) -> list[dict]:
    """Manifests of analyzed sessions (those with lines), most recently changed first."""
    root = sessions_dir(runtime)
    found = []
    for manifest in root.glob(f"*/{MANIFEST}") if root.is_dir() else ():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("segments") and data.get("id") == manifest.parent.name:
            found.append(data)
    found.sort(key=lambda data: data.get("updated") or 0, reverse=True)
    return found[:limit]


def _copy_hashed(source: Path, target: Path) -> str:
    """Copy ``source`` to ``target`` in one pass, returning the SHA-256 of its bytes."""
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("wb") as writer:
        for block in iter(lambda: reader.read(1 << 20), b""):
            digest.update(block)
            writer.write(block)
    shutil.copystat(source, target)
    return digest.hexdigest()


def create(runtime, source: str | Path, *, video: bool) -> Session:
    """Copy the source in, so the session outlives the upload that started it.

    The copy's hash finds earlier sessions of the same file, whose stems and
    transcripts a new analysis can reuse.
    """
    source = Path(source)
    if not source.is_file():
        raise H3Error("Choose a video or audio file first.")
    prune(runtime)
    session_id = uuid.uuid4().hex
    root = sessions_dir(runtime) / session_id
    root.mkdir(parents=True)
    copied = root / f"source{source.suffix.lower()}"
    source_hash = _copy_hashed(source, copied)
    session = Session(root, {
        "id": session_id, "created": time.time(), "name": source.stem, "source_hash": source_hash,
        "video": bool(video), "files": {"source": copied.name}, "segments": [], "speakers": {},
    })
    session.save()
    return session


def same_source(session: Session) -> list[Session]:
    """Other sessions of the same source file, most recently changed first."""
    source_hash = session.data.get("source_hash")
    found = []
    for manifest in session.root.parent.glob(f"*/{MANIFEST}") if source_hash else ():
        if manifest.parent == session.root:
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("source_hash") == source_hash and data.get("id") == manifest.parent.name:
            found.append(Session(manifest.parent, data))
    found.sort(key=lambda earlier: earlier.data.get("updated") or 0, reverse=True)
    return found


def load(runtime, session_id: str) -> Session:
    session_id = str(session_id or "").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise H3Error("Analyze a source first.")
    root = sessions_dir(runtime) / session_id
    try:
        data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise H3Error("This dubbing session is no longer available; analyze the source again.") from exc
    return Session(root, data)


def modify(session: Session, change: Callable[[Session], None]) -> Session:
    """Apply ``change`` to the latest saved manifest under the lock.

    Jobs run for minutes while the editor keeps saving lines; a job writes only
    what it owns, onto whatever was saved last, so those edits survive.
    """
    with _LOCK:
        latest = Session(session.root, json.loads((session.root / MANIFEST).read_text(encoding="utf-8")))
        change(latest)
        latest.save()
        session.data = latest.data
        return latest


def update_segments(runtime, session_id: str, segments: list[dict]) -> Session:
    """Replace the lines under the manifest lock (editor saves race with running jobs)."""
    with _LOCK:
        session = load(runtime, session_id)
        session.segments = segments
        session.save()
        return session
