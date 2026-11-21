"""Named dubbing projects shared by Workshop and Studio, independent of upload caches."""
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

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
    found = [
        (max(path.stat().st_mtime_ns, _current_revision(path, data)[1]), data["name"])
        for path, data in _entries(runtime)
    ]
    return [name for _, name in sorted(found, reverse=True)]


def retained_sessions(runtime):
    """Only sessions still referenced by named projects are exempt from cleanup."""
    return {_current_session(path, data) for path, data in _entries(runtime)}


def _read_project(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or not isinstance(data.get("session"), str)
                or not isinstance(data.get("settings"), dict)):
            raise ValueError("Invalid dubbing project")
    except (OSError, ValueError) as exc:
        raise H3Error("Choose a saved dubbing project.") from exc
    return data


def _generation(data):
    # Old projects have no generation. Their starting session already identifies it.
    return data.get("generation") or data["session"]


def _revisions(path, data):
    return path.with_suffix("") / _generation(data)


def _current_revision(path, data):
    current = data["session"]
    published_at = 0
    children = {}
    for revision in _revisions(path, data).glob("*.json"):
        try:
            entry = json.loads(revision.read_text(encoding="utf-8"))
            parent = entry["previous"]
            published = entry["published"]
            if isinstance(parent, str) and type(published) is int:
                choice = (published, revision.stem)
                if parent not in children or choice < children[parent]:
                    children[parent] = choice
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # Different containers can publish against the same parent before either sees
    # the other's commit. Take the first published successor consistently.
    visited = {current}
    while (revision := children.get(current)) is not None and revision[1] not in visited:
        published_at, current = revision
        visited.add(current)
    return current, published_at


def _current_session(path, data):
    return _current_revision(path, data)[0]


def load(runtime, name):
    path = _path(runtime, name)
    data = _read_project(path)
    current = _current_session(path, data)
    return {**data, "session": current}, sessions.load(runtime, current)


def analysis_target(runtime, selection, source):
    """Capture the selected project, or identify its exact private source copy."""
    if isinstance(selection, dict) and selection.get("name"):
        name = str(selection["name"]).strip()
        target = {"name": name, "session": selection.get("session") or "", "generation": ""}
        try:
            path = _path(runtime, name)
            data = _read_project(path)
        except H3Error:
            return target
        if _current_session(path, data) == target["session"]:
            target["generation"] = _generation(data)
        return target

    chosen = Path(source).resolve()
    if not chosen.is_relative_to(sessions.sessions_dir(runtime)):
        return None
    matches = []
    for path, data in _entries(runtime):
        try:
            current = sessions.load(runtime, _current_session(path, data))
        except H3Error:
            continue
        original = current.path("source")
        if original is not None and original.resolve() == chosen:
            matches.append({"name": data["name"], "session": current.id, "generation": _generation(data)})
    return matches[0] if len(matches) == 1 else None


def reopen(runtime, name):
    """Recover a draft whose matching Studio dub was saved without a project link."""
    with _LOCK:
        data, draft = load(runtime, name)
        target = str(data["settings"].get("target_language") or "").strip().casefold()
        if draft.segments or draft.data.get("last_output") or not target or not draft.data.get("source_hash"):
            return data, draft
        rewrite = bool(data["settings"].get("rewrite"))
        projects = [{**project, "session": _current_session(path, project)} for path, project in _entries(runtime)]
        linked = {project["session"] for project in projects}
        output_root = (runtime.outputs_dir / "dubbing").resolve()
        matches = []
        for session in sessions.same_source(draft):
            if (
                session.id in linked
                or not session.segments
                or session.data.get("created", 0) < draft.data.get("created", 0)
                or session.data.get("video") != draft.data.get("video")
                or str(session.data.get("target_language") or "").strip().casefold() != target
                or bool(session.data.get("rewrite")) != rewrite
                or not session.data.get("last_output")
            ):
                continue
            result = Path(session.data["last_output"]).resolve()
            if result.parent == output_root and any(
                result.with_suffix(suffix).is_file() for suffix in (".mp3", ".mp4")
            ):
                matches.append(session)
        if len(matches) != 1:
            return data, draft
        # The same source and settings can be used by more than one named draft.
        # In that case the owner must choose which project should use the dub.
        for project in projects:
            settings = project["settings"]
            if (
                project["name"] == data["name"]
                or str(settings.get("target_language") or "").strip().casefold() != target
                or bool(settings.get("rewrite")) != rewrite
            ):
                continue
            try:
                other = sessions.load(runtime, project["session"])
            except H3Error:
                continue
            if not other.segments and other.data.get("source_hash") == draft.data["source_hash"]:
                return data, draft
        session = matches[0]
        if attach(runtime, name, session.id, expected_session=draft.id):
            return {**data, "session": session.id}, session
        return load(runtime, name)


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


def save(runtime, name, session_id, source, settings, *, expected_session=None):
    """Keep source bytes before Studio analyzes them; retain analyzed sessions and their takes."""
    target = _path(runtime, name)
    with _LOCK:
        previous = _read_project(target) if target.is_file() else None
        current = _current_session(target, previous) if previous is not None else None
        if expected_session is not None and current != expected_session:
            raise H3Error("This dubbing project changed. Load it again before saving.")
        session = sessions.load(runtime, session_id) if session_id else None
        chosen = Path(source) if source else None
        if session is None or (chosen and not _same_source(session, chosen)):
            if chosen is None:
                raise H3Error("Upload a source or choose one from Media before saving.")
            from h3_app.catalog import VIDEO_EXTENSIONS
            session = sessions.create(runtime, chosen, video=chosen.suffix.lower() in VIDEO_EXTENSIONS | {".m4v"})
        if current == session.id:
            data = {**previous, "settings": settings}
        else:
            data = {"name": str(name).strip(), "session": session.id, "settings": settings,
                    "generation": uuid.uuid4().hex}
        write_json_atomic(target, data)
        return session


def attach(runtime, name, session_id, *, expected_session=None, expected_generation=None):
    """Attach completed analysis to the project captured when the job started.

    A project saved by another browser while analysis runs keeps its newer session.
    """
    if not name or not session_id:
        return False
    with _LOCK:
        path = _path(runtime, name)
        try:
            data = _read_project(path)
            previous = sessions.load(runtime, _current_session(path, data))
        except H3Error:
            return False
        if expected_generation is not None and _generation(data) != expected_generation:
            return False
        if session_id == previous.id:
            return True
        if expected_session is not None and previous.id != expected_session:
            return False
        session = sessions.load(runtime, session_id)
        if session.data.get("source_hash") == previous.data.get("source_hash"):
            sessions.modify(session, lambda current: current.data.update(name=previous.data["name"]))
        revision = _revisions(path, data) / f"{session.id}.json"
        if revision.exists():
            return False
        write_json_atomic(revision, {"previous": previous.id, "published": time.time_ns()})
        return True
