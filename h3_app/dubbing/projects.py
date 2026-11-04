"""Named dubbing projects shared by Workshop and Studio, independent of upload caches."""
import hashlib
import json
from pathlib import Path
import threading

from h3_app.errors import H3Error
from h3_models import write_json_atomic
from . import session as sessions

_LOCK = threading.RLock()


def _root(runtime):
    return sessions.sessions_dir(runtime).parent / "dubbing-projects"


def _path(runtime, name):
    name = str(name or "").strip()
    if not name:
        raise H3Error("Enter a project name.")
    return _root(runtime) / (hashlib.sha256(name.encode()).hexdigest() + ".json")


def _entries(runtime):
    for path in _root(runtime).glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(data, dict) and isinstance(data.get("name"), str)
                    and isinstance(data.get("session"), str) and isinstance(data.get("settings"), dict)):
                yield path, data
        except (OSError, ValueError, KeyError, TypeError):
            continue


def names(runtime):
    found = [(path.stat().st_mtime, data["name"]) for path, data in _entries(runtime)]
    return [name for _, name in sorted(found, reverse=True)]


def retained_sessions(runtime):
    """Only sessions still referenced by named projects are exempt from cleanup."""
    return {data["session"] for _, data in _entries(runtime)}


def load(runtime, name):
    try:
        data = json.loads(_path(runtime, name).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or not isinstance(data.get("session"), str)
                or not isinstance(data.get("settings"), dict)):
            raise ValueError("Invalid dubbing project")
    except (OSError, ValueError) as exc:
        raise H3Error("Choose a saved dubbing project.") from exc
    return data, sessions.load(runtime, data["session"])


def _same_source(session, chosen):
    source = session.path("source")
    if source.resolve() == chosen.resolve():
        return True
    # Gradio returns a cached copy of a File output on the next save. Compare bytes
    # so saving that copy preserves the analyzed lines, takes and result pointer.
    if not chosen.is_file() or source.stat().st_size != chosen.stat().st_size:
        return False
    digest = hashlib.sha256()
    with chosen.open("rb") as reader:
        for block in iter(lambda: reader.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == session.data.get("source_hash")


def save(runtime, name, session_id, source, settings):
    """Keep source bytes before Studio analyzes them; retain analyzed sessions and their takes."""
    target = _path(runtime, name)
    with _LOCK:
        session = sessions.load(runtime, session_id) if session_id else None
        chosen = Path(source) if source else None
        if session is None or (chosen and not _same_source(session, chosen)):
            if chosen is None:
                raise H3Error("Upload a source or choose one from Media before saving.")
            from h3_app.catalog import VIDEO_EXTENSIONS
            session = sessions.create(runtime, chosen, video=chosen.suffix.lower() in VIDEO_EXTENSIONS | {".m4v"})
        write_json_atomic(target, {"name": str(name).strip(), "session": session.id, "settings": settings})
        return session


def attach(runtime, name, session_id, *, expected_session=None):
    """Attach completed analysis to the project captured when the job started.

    A project saved by another browser while analysis runs keeps its newer session.
    """
    if not name or not session_id:
        return False
    with _LOCK:
        try:
            data, previous = load(runtime, name)
        except H3Error:
            return False
        if expected_session is not None and data["session"] != expected_session:
            return False
        session = sessions.load(runtime, session_id)
        if session.data.get("source_hash") == previous.data.get("source_hash"):
            sessions.modify(session, lambda current: current.data.update(name=previous.data["name"]))
        data["session"] = session.id
        write_json_atomic(_path(runtime, name), data)
        return True
