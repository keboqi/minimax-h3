"""Portable cross-server project export, import, and remote transfer client."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from itertools import count
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx

from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, snapshot_path
from h3_models import write_json_atomic

TRANSFER_SCHEMA_VERSION = 1
PROJECT_KINDS = ("soundstage", "storyboard", "dubbing", "remake")
KIND_LABELS = {
    "soundstage": "Soundstage",
    "storyboard": "Storyboard",
    "dubbing": "Dubbing",
    "remake": "Remake",
}
CONFLICT_MODES = ("rename", "skip", "overwrite")

_APP_PAGE_SUFFIXES = (
    "/soundstage",
    "/dubbing",
    "/storyboard",
    "/remake",
    "/media",
    "/h3",
    "/image",
    "/ltx",
    "/song",
    "/comfyui",
)
_STORYBOARD_TAKE_PATH_KEYS = ("video", "latent", "stage_latent", "tail")
_STORYBOARD_SINGLE_MEDIA_KEYS = ("first_image", "last_image", "video")
_STORYBOARD_LIST_MEDIA_KEYS = ("ref_images", "ref_videos", "ref_audios")
_SYMLINK_MODE = 0o120000
_FILE_TYPE_MASK = 0o170000
_PRECOMPRESSED_SUFFIXES = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".webm",
        ".flac",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".safetensors",
        ".npz",
    }
)


def normalize_server_url(url: str) -> str:
    """Normalize a Studio or Workshop URL down to its base app origin/path."""
    raw = str(url or "").strip()
    if not raw:
        raise H3Error("Enter a Prism Studio or Workshop URL.")
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise H3Error("Enter a valid http:// or https:// server URL.")
    path = parsed.path.rstrip("/")
    for suffix in _APP_PAGE_SUFFIXES:
        if path == suffix or path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((scheme, parsed.netloc, path.rstrip("/"), "", ""))


def build_auth_headers(modal_key: str = "", modal_secret: str = "") -> dict[str, str]:
    """Build optional Modal proxy auth headers from user inputs."""
    key = str(modal_key or "").strip()
    secret = str(modal_secret or "").strip()
    if bool(key) != bool(secret):
        raise H3Error("Provide both Modal-Key and Modal-Secret, or leave both blank.")
    return {"Modal-Key": key, "Modal-Secret": secret} if key else {}


def _unique_name(base: str, existing: set[str], *, style: str = "paren") -> str:
    if base not in existing:
        return base
    if style == "stem":
        candidate = f"{base}_imported"
        index = 2
        while candidate in existing:
            candidate = f"{base}_imported_{index}"
            index += 1
        return candidate
    candidate = f"{base} (imported)"
    index = 2
    while candidate in existing:
        candidate = f"{base} (imported {index})"
        index += 1
    return candidate


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _extract_member_atomic(archive: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        with archive.open(member, "r") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 20)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _add_zip_file(archive: zipfile.ZipFile, source: Path, arc_name: str, added: set[str]) -> bool:
    if arc_name in added or not source.is_file() or source.is_symlink():
        return False
    compress = zipfile.ZIP_STORED if source.suffix.lower() in _PRECOMPRESSED_SUFFIXES else zipfile.ZIP_DEFLATED
    archive.write(source, arc_name, compress_type=compress)
    added.add(arc_name)
    return True


def _add_zip_dir(archive: zipfile.ZipFile, root_dir: Path, prefix: str, added: set[str]) -> None:
    if not root_dir.is_dir():
        return
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name.endswith(".tmp"):
            continue
        rel = path.relative_to(root_dir).as_posix()
        _add_zip_file(archive, path, f"{prefix}/{rel}", added)


def _validate_archive(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or ".." in PurePosixPath(name).parts
            or ((info.external_attr >> 16) & _FILE_TYPE_MASK) == _SYMLINK_MODE
        ):
            raise H3Error(f"Unsafe archive entry: {name!r}")


def _extract_prefix(archive: zipfile.ZipFile, prefix: str, dest_dir: Path, *, overwrite: bool = True) -> dict[str, Path]:
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    head = prefix.rstrip("/") + "/"
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(head):
            continue
        rel = info.filename[len(head) :]
        if not rel:
            continue
        target = (dest_root / rel).resolve()
        if not target.is_relative_to(dest_root):
            raise H3Error(f"Unsafe archive path: {info.filename!r}")
        if target.exists() and not overwrite:
            extracted[info.filename] = target
            continue
        _extract_member_atomic(archive, info.filename, target)
        extracted[info.filename] = target
    return extracted


def _update_snapshot_file(path: Path, *, output_name: str, session_id: str | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data["output"] = output_name
    target_media = path.with_name(output_name)
    data["asset_id"] = uuid.uuid5(uuid.NAMESPACE_URL, target_media.resolve().as_uri()).hex
    if session_id is not None and isinstance(data.get("settings"), dict):
        data["settings"]["session"] = session_id
    write_json_atomic(path, data)
    return data


def _register_imported_media(media_path: Path, metadata: dict[str, Any] | None, *, store, runtime) -> None:
    """Index an imported output file into the Media library catalog immediately."""
    if store is None or not media_path.is_file():
        return
    from h3_app.media_catalog import MediaCatalog

    MediaCatalog(store, lambda: runtime).register(media_path, metadata or {})


def _collect_referenced_speakers(keys: Iterable[str], runtime) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    from h3_app.drama3 import speakers as drama3_speakers

    metadata: list[dict[str, Any]] = []
    files: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for index, raw_key in enumerate(keys):
        key = str(raw_key or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            preset = drama3_speakers.load_speaker(key, runtime)
        except H3Error:
            continue
        if not preset.audio.is_file():
            continue
        arc_name = f"speakers/{index}{preset.audio.suffix.lower()}"
        metadata.append(
            {
                "key": preset.key,
                "name": preset.name,
                "group": preset.group,
                "transcript": preset.transcript,
                "description": preset.description,
                "audio_file": arc_name,
            }
        )
        files.append((preset.audio, arc_name))
    return metadata, files


def _transfer_result(
    kind: str,
    source_id: str,
    source_name: str,
    target_id: str,
    target_name: str,
    status: str,
    message: str | None = None,
) -> dict[str, Any]:
    label = KIND_LABELS.get(kind, kind)
    if message is None:
        message = (
            f'Skipped {label} "{source_name}" (already exists).'
            if status == "skipped"
            else f'{status.capitalize()} {label} "{target_name}".'
        )
    return {
        "kind": kind,
        "source_id": source_id,
        "source_name": source_name,
        "target_id": target_id,
        "target_name": target_name,
        "status": status,
        "message": message,
    }


def _resolve_named_conflict(
    kind: str, source_id: str, orig_name: str, existing: set[str], conflict: str
) -> tuple[str, str, dict[str, Any] | None]:
    if orig_name not in existing:
        return orig_name, "imported", None
    if conflict == "skip":
        return orig_name, "skipped", _transfer_result(kind, source_id, orig_name, orig_name, orig_name, "skipped")
    if conflict == "overwrite":
        return orig_name, "overwritten", None
    return _unique_name(orig_name, existing), "renamed", None


def _update_stills_source(stills_dir: Path, source_name: str, record_filename: str) -> None:
    stills_record = stills_dir / record_filename
    if not stills_record.is_file():
        return
    try:
        rec = json.loads(stills_record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(rec, dict):
        rec["source"] = source_name
        write_json_atomic(stills_record, rec)


def _restore_referenced_speakers(entries: list[dict[str, Any]], archive: zipfile.ZipFile, runtime) -> None:
    from h3_app.drama3 import speakers as drama3_speakers

    names_in_archive = set(archive.namelist())
    for entry in entries:
        key = str(entry.get("key") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not key or audio_file not in names_in_archive:
            continue
        try:
            drama3_speakers.load_speaker(key, runtime)
            continue
        except H3Error:
            pass
        suffix = Path(audio_file).suffix.lower() or ".wav"
        with TemporaryDirectory(prefix="h3-import-speaker-") as tmp:
            clip = Path(tmp) / f"voice{suffix}"
            _extract_member_atomic(archive, audio_file, clip)
            drama3_speakers.save_speaker(
                str(entry["name"]),
                clip,
                str(entry["transcript"]),
                runtime,
                str(entry.get("description") or ""),
                str(entry.get("group") or "Global"),
            )


def _soundstage_clips_dir(runtime) -> Path:
    return Path(runtime.output_dir) / "h3" / ".soundstage" / "clips"


def _soundstage_plan_audio(runtime, project_name: str) -> Path:
    digest = hashlib.sha256(project_name.encode()).hexdigest()
    return Path(runtime.output_dir) / ".soundstage" / "plans" / f"{digest}.wav"


def _list_soundstage(store) -> list[dict[str, Any]]:
    if store is None:
        return []
    with store.connect() as db:
        rows = db.execute(
            "SELECT name, result, updated, payload FROM soundstage_projects ORDER BY updated DESC"
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        lines = sum(1 for line in str(payload.get("text") or "").splitlines() if line.strip())
        state = "Generated" if row["result"] else "Draft"
        items.append(
            {
                "kind": "soundstage",
                "id": str(row["name"]),
                "name": str(row["name"]),
                "updated": float(row["updated"] or 0),
                "summary": f"{state} · {_plural(lines, 'line')}",
            }
        )
    return items


def _list_storyboard(store) -> list[dict[str, Any]]:
    if store is None:
        return []
    items: list[dict[str, Any]] = []
    for row in store.storyboards():
        board_id = str(row["id"])
        try:
            board = store.storyboard(None, board_id)
        except ValueError:
            continue
        scenes = len(board.get("scenes") or [])
        takes = len(board.get("takes") or {})
        items.append(
            {
                "kind": "storyboard",
                "id": board_id,
                "name": str(row["name"]),
                "updated": float(row["updated"] or 0),
                "summary": f"{_plural(scenes, 'scene')} · {_plural(takes, 'take')}",
            }
        )
    return items


def _list_dubbing(runtime) -> list[dict[str, Any]]:
    from h3_app.dubbing import projects as dubbing_projects

    items: list[dict[str, Any]] = []
    for name in dubbing_projects.names(runtime):
        try:
            data, session = dubbing_projects.load(runtime, name)
        except (H3Error, OSError, ValueError):
            continue
        segments = len(session.segments)
        src = str(session.data.get("source_language") or "").strip()
        tgt = str((data.get("settings") or {}).get("target_language") or session.data.get("target_language") or "").strip()
        if segments:
            lang = f"{src} → {tgt}" if src and tgt else tgt or src
            summary = f"{_plural(segments, 'line')} · {lang}" if lang else _plural(segments, "line")
        else:
            summary = "Source only"
        items.append(
            {
                "kind": "dubbing",
                "id": name,
                "name": name,
                "updated": float(session.data.get("updated") or session.data.get("created") or 0),
                "summary": summary,
            }
        )
    return items


def _list_remake(runtime) -> list[dict[str, Any]]:
    from h3_app.remake.project import Projects as RemakeProjects

    items: list[dict[str, Any]] = []
    for project in RemakeProjects(runtime.output_dir).list():
        windows = project.get("windows") or []
        done = sum(1 for window in windows if window.get("take"))
        key = str(project.get("key") or "")
        name = str(project.get("name") or Path(str(project.get("source") or key)).stem)
        items.append(
            {
                "kind": "remake",
                "id": key,
                "name": name,
                "updated": float(project.get("updated") or 0),
                "summary": f"{_plural(len(windows), 'window')} · {done} done",
            }
        )
    return items


def list_local_projects(store, runtime) -> list[dict[str, Any]]:
    """Return every local Soundstage, Storyboard, Dubbing, and Remake project, newest first."""
    items = [
        *_list_soundstage(store),
        *_list_storyboard(store),
        *_list_dubbing(runtime),
        *_list_remake(runtime),
    ]
    items.sort(key=lambda item: item.get("updated") or 0, reverse=True)
    return items


def _export_soundstage(archive: zipfile.ZipFile, project_id: str, *, store, runtime, added: set[str]) -> str:
    from h3_app.drama3 import sounds as drama3_sounds
    from h3_app.drama3.results import result_settings
    from h3_app.drama3.serving.takes import takes_root
    from h3_ui.drama3 import visuals
    from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND, UPLOAD_VOICE

    try:
        raw_payload, result = store.soundstage_project(project_id)
    except ValueError as exc:
        raise H3Error(f'Soundstage project "{project_id}" no longer exists.') from exc

    payload = copy.deepcopy(raw_payload)
    clips_dir = _soundstage_clips_dir(runtime)
    clip_map: dict[Path, str] = {}
    used_clip_names: set[str] = set()

    def archive_clip(raw_value: Any) -> str | None:
        if not raw_value:
            return None
        name = Path(str(raw_value)).name
        candidate = clips_dir / name
        if not candidate.is_file():
            fallback = Path(str(raw_value))
            if fallback.is_file():
                candidate = fallback
            else:
                return str(raw_value)
        resolved = candidate.resolve()
        if resolved in clip_map:
            return clip_map[resolved]
        stored_name = name
        idx = 2
        while stored_name in used_clip_names:
            stored_name = f"{candidate.stem}_{idx}{candidate.suffix.lower()}"
            idx += 1
        used_clip_names.add(stored_name)
        _add_zip_file(archive, resolved, f"clips/{stored_name}", added)
        clip_map[resolved] = stored_name
        return stored_name

    speaker_keys: list[str] = []
    for slot in range(DRAMA3_VOICE_SLOTS):
        field = f"voice_{slot}_audio"
        if payload.get(field):
            payload[field] = archive_clip(payload[field])
        preset = payload.get(f"voice_{slot}_preset")
        if preset and preset != UPLOAD_VOICE:
            speaker_keys.append(str(preset))
    referenced_speakers, speaker_files = _collect_referenced_speakers(speaker_keys, runtime)
    for src, arc in speaker_files:
        _add_zip_file(archive, src, arc, added)

    referenced_sounds: list[dict[str, Any]] = []
    seen_sounds: set[str] = set()
    for slot in range(DRAMA3_SOUND_SLOTS):
        field = f"sound_{slot}_audio"
        if payload.get(field):
            payload[field] = archive_clip(payload[field])
        preset = payload.get(f"sound_{slot}_preset")
        if not preset or preset == UPLOAD_SOUND:
            continue
        sound_name = str(preset)
        if sound_name in seen_sounds:
            continue
        seen_sounds.add(sound_name)
        try:
            saved = drama3_sounds.load_sound(sound_name, runtime)
        except H3Error:
            continue
        if not saved.audio.is_file():
            continue
        arc_name = f"sounds/{len(referenced_sounds)}{saved.audio.suffix.lower()}"
        _add_zip_file(archive, saved.audio, arc_name, added)
        referenced_sounds.append(
            {
                "name": saved.name,
                "description": saved.description,
                "duration": int(saved.duration),
                "audio_file": arc_name,
            }
        )

    if payload.get("background_music"):
        payload["background_music"] = archive_clip(payload["background_music"])
    if isinstance(payload.get("applied_music"), dict) and payload["applied_music"].get("background_music"):
        payload["applied_music"]["background_music"] = archive_clip(payload["applied_music"]["background_music"])

    result_name: str | None = None
    result_path = Path(result) if result else None
    if result_path is not None and not result_path.is_file():
        candidate = Path(runtime.output_dir) / "h3" / result_path.name
        result_path = candidate if candidate.is_file() else None
    if result_path is not None:
        result_name = result_path.name
        _add_zip_file(archive, result_path, f"result/{result_name}", added)
        snap = snapshot_path(result_path)
        _add_zip_file(archive, snap, f"result/{snap.name}", added)
        plan = visuals.plan_path(result_path)
        _add_zip_file(archive, plan, f"result/{plan.name}", added)
        _add_zip_dir(archive, visuals.stills_dir(result_path, runtime.output_dir), "stills", added)
        takes_dir = takes_root(runtime)
        for chunk in result_settings(result_path).get("chunks") or []:
            take_key = chunk.get("take")
            if take_key and re.fullmatch(r"[A-Za-z0-9_-]+", str(take_key)):
                _add_zip_file(archive, takes_dir / f"{take_key}.wav", f"takes/{take_key}.wav", added)
    else:
        plan_audio = _soundstage_plan_audio(runtime, project_id)
        _add_zip_file(archive, visuals.plan_path(plan_audio), "plan/visual_plan.json", added)
        _add_zip_dir(archive, visuals.stills_dir(plan_audio, runtime.output_dir), "plan_stills", added)

    archive.writestr(
        "project.json",
        json.dumps(
            {
                "name": project_id,
                "payload": payload,
                "result_name": result_name,
                "referenced_speakers": referenced_speakers,
                "referenced_sounds": referenced_sounds,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return project_id


def _export_storyboard(archive: zipfile.ZipFile, project_id: str, *, store, runtime, added: set[str]) -> str:
    try:
        board = copy.deepcopy(store.storyboard(None, project_id))
    except ValueError as exc:
        raise H3Error(f'Storyboard "{project_id}" no longer exists.') from exc

    output_dir = Path(runtime.output_dir)
    media_counter = count(1)
    cast_counter = count(1)
    media_map: dict[Path, str] = {}
    raw_to_arc: dict[str, str] = {}
    used_take_names: set[str] = set()
    take_map: dict[Path, str] = {}

    def resolve_path(raw: Any) -> Path | None:
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_absolute() else (output_dir / path)

    def archive_asset(raw: Any, prefix: str, counter: count, cache: dict[Path, str]) -> str | None:
        if not raw:
            return None
        raw_str = str(raw)
        path = resolve_path(raw_str)
        if path is None or not path.is_file() or path.is_symlink():
            return None
        resolved = path.resolve()
        if resolved in cache:
            raw_to_arc[raw_str] = cache[resolved]
            return cache[resolved]
        arc = f"{prefix}/{next(counter)}_{path.name}"
        _add_zip_file(archive, resolved, arc, added)
        cache[resolved] = arc
        raw_to_arc[raw_str] = arc
        return arc

    def rewrite_snapshot_lists(snapshot: dict[str, Any]) -> None:
        for member in snapshot.get("cast") or []:
            if isinstance(member, dict) and member.get("image"):
                archived = archive_asset(member["image"], "cast", cast_counter, media_map)
                if archived is not None:
                    member["image"] = archived
        for scene in snapshot.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for key in _STORYBOARD_SINGLE_MEDIA_KEYS:
                if scene.get(key):
                    scene[key] = archive_asset(scene[key], "media", media_counter, media_map)
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if scene.get(key):
                    items = [archive_asset(item, "media", media_counter, media_map) for item in scene[key]]
                    scene[key] = [item for item in items if item is not None]

    rewrite_snapshot_lists(board)
    for version in board.get("versions") or []:
        if isinstance(version, dict):
            rewrite_snapshot_lists(version)

    for take in (board.get("takes") or {}).values():
        if not isinstance(take, dict):
            continue
        for key in _STORYBOARD_TAKE_PATH_KEYS:
            val = take.get(key)
            if not val:
                continue
            path = resolve_path(val)
            if path is None or not path.is_file() or path.is_symlink():
                take[key] = None
                continue
            resolved = path.resolve()
            if resolved in take_map:
                take[key] = take_map[resolved]
                continue
            filename = path.name
            idx = 2
            while filename in used_take_names:
                filename = f"{path.stem}_{idx}{path.suffix}"
                idx += 1
            used_take_names.add(filename)
            arc = f"takes/{filename}"
            _add_zip_file(archive, resolved, arc, added)
            if key == "video":
                snap = snapshot_path(resolved)
                if snap.is_file():
                    _add_zip_file(archive, snap, f"takes/{filename}.settings.json", added)
            take_map[resolved] = arc
            take[key] = arc
        inputs = take.get("inputs")
        if isinstance(inputs, dict):
            for key in ("first_image", "last_image"):
                if inputs.get(key) and str(inputs[key]) in raw_to_arc:
                    inputs[key] = raw_to_arc[str(inputs[key])]
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if isinstance(inputs.get(key), list):
                    inputs[key] = [raw_to_arc.get(str(item), item) for item in inputs[key]]

    archive.writestr("board.json", json.dumps({"id": project_id, "board": board}, ensure_ascii=False, indent=2))
    return str(board.get("name") or project_id)


def _export_dubbing(archive: zipfile.ZipFile, project_id: str, *, runtime, added: set[str]) -> str:
    from h3_app.dubbing import projects as dubbing_projects, results as dub_results

    data, session = dubbing_projects.load(runtime, project_id)
    _add_zip_dir(archive, session.root, "session", added)

    speaker_keys = [
        entry.get("voice")
        for entry in (session.data.get("speaker_settings") or {}).values()
        if isinstance(entry, dict) and entry.get("voice")
    ]
    referenced_speakers, speaker_files = _collect_referenced_speakers(speaker_keys, runtime)
    for src, arc in speaker_files:
        _add_zip_file(archive, src, arc, added)

    last_output = session.data.get("last_output")
    last_output_name: str | None = None
    last_output_stem: str | None = None
    result = dub_results.load_result(last_output) if last_output else None
    if result is not None and last_output:
        last_path = Path(last_output)
        last_output_name = last_path.name
        last_output_stem = last_path.stem
        candidates = [result.audio, result.video, *result.subtitles]
        if result.video is not None:
            lipsync = result.video.with_name(f"{result.video.stem}.lipsync.mp4")
            if lipsync.is_file():
                candidates.append(lipsync)
        for file_path in candidates:
            if file_path is None:
                continue
            _add_zip_file(archive, file_path, f"result/{file_path.name}", added)
            snap = snapshot_path(file_path)
            _add_zip_file(archive, snap, f"result/{snap.name}", added)
        _add_zip_dir(archive, dub_results.player_folder(last_path), "result_player", added)

    archive.writestr(
        "project.json",
        json.dumps(
            {
                "name": project_id,
                "settings": data["settings"],
                "session_id": session.id,
                "last_output_name": last_output_name,
                "last_output_stem": last_output_stem,
                "referenced_speakers": referenced_speakers,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return project_id


def _export_remake(archive: zipfile.ZipFile, project_id: str, *, runtime, added: set[str]) -> str:
    from h3_app.remake.project import Projects as RemakeProjects

    project = RemakeProjects(runtime.output_dir).load(None, project_id)
    if project is None:
        raise H3Error(f'Remake project "{project_id}" no longer exists.')
    project = copy.deepcopy(project)
    folder = Path(project["folder"]).resolve()
    skip_files: set[Path] = set()
    used_take_names: set[str] = set()
    extra_counter = count(1)

    for index, window in enumerate(project.get("windows") or []):
        take = window.get("take")
        if not take:
            continue
        take_path = Path(take)
        if not take_path.is_file() or take_path.is_symlink():
            window["take"] = None
            continue
        resolved = take_path.resolve()
        skip_files.add(resolved)
        snap = snapshot_path(resolved)
        if snap.is_file():
            skip_files.add(snap.resolve())
        arc_name = f"w{index}_{resolved.name}"
        suffix_idx = 2
        while arc_name in used_take_names:
            arc_name = f"w{index}_{resolved.stem}_{suffix_idx}{resolved.suffix}"
            suffix_idx += 1
        used_take_names.add(arc_name)
        _add_zip_file(archive, resolved, f"takes/{arc_name}", added)
        if snap.is_file():
            _add_zip_file(archive, snap, f"takes/{arc_name}.settings.json", added)
        window["take"] = f"takes/{arc_name}"

    if folder.is_dir():
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name == "project.json" or path.name.endswith(".tmp"):
                continue
            if path.resolve() in skip_files:
                continue
            rel = path.relative_to(folder).as_posix()
            _add_zip_file(archive, path, f"folder/{rel}", added)

    def archive_folder_ref(raw: Any, default_prefix: str) -> Any:
        if not raw:
            return raw
        path = Path(str(raw))
        if not path.is_file() or path.is_symlink():
            return raw
        resolved = path.resolve()
        if resolved.is_relative_to(folder):
            return f"folder/{resolved.relative_to(folder).as_posix()}"
        arc = f"{default_prefix}/{next(extra_counter)}_{path.name}"
        _add_zip_file(archive, resolved, arc, added)
        return arc

    if project.get("source"):
        project["source"] = archive_folder_ref(project["source"], "folder")
    for window in project.get("windows") or []:
        if window.get("pictures"):
            window["pictures"] = [archive_folder_ref(pic, "folder/pictures") for pic in window["pictures"]]
        if window.get("carried"):
            window["carried"] = archive_folder_ref(window["carried"], "folder/pictures")

    cleaned = {k: v for k, v in project.items() if k not in {"folder", "updated"}}
    archive.writestr("project.json", json.dumps(cleaned, ensure_ascii=False, indent=2))
    return str(cleaned.get("name") or project_id)


def export_project_archive(kind: str, project_id: str, destination: Path, *, store, runtime) -> Path:
    """Export one project and all of its media, sidecars, and presets into a portable zip archive."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
    added: set[str] = set()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if kind == "soundstage":
                display_name = _export_soundstage(archive, project_id, store=store, runtime=runtime, added=added)
            elif kind == "storyboard":
                display_name = _export_storyboard(archive, project_id, store=store, runtime=runtime, added=added)
            elif kind == "dubbing":
                display_name = _export_dubbing(archive, project_id, runtime=runtime, added=added)
            else:
                display_name = _export_remake(archive, project_id, runtime=runtime, added=added)
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "schema_version": TRANSFER_SCHEMA_VERSION,
                        "kind": kind,
                        "id": project_id,
                        "name": display_name,
                        "exported_at": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _import_soundstage(
    archive: zipfile.ZipFile, *, store, runtime, conflict: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.drama3 import sounds as drama3_sounds
    from h3_app.drama3.serving.takes import takes_root
    from h3_ui.drama3 import visuals

    data = json.loads(archive.read("project.json"))
    orig_name = str(data.get("name") or source_name or source_id)
    payload = data["payload"]
    result_name = data.get("result_name")

    existing = {name for name, _ in store.soundstage_projects()}
    target_name, status, skipped = _resolve_named_conflict("soundstage", source_id, orig_name, existing, conflict)
    if skipped is not None:
        return skipped

    _extract_prefix(archive, "clips", _soundstage_clips_dir(runtime), overwrite=True)
    _restore_referenced_speakers(data.get("referenced_speakers") or [], archive, runtime)

    names_in_archive = set(archive.namelist())
    for entry in data.get("referenced_sounds") or []:
        sound_name = str(entry.get("name") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not sound_name or audio_file not in names_in_archive:
            continue
        if drama3_sounds.find_sound(sound_name, runtime) is not None:
            continue
        suffix = Path(audio_file).suffix.lower() or ".wav"
        with TemporaryDirectory(prefix="h3-import-sound-") as tmp:
            clip = Path(tmp) / f"sound{suffix}"
            _extract_member_atomic(archive, audio_file, clip)
            drama3_sounds.save_sound(
                sound_name,
                clip,
                str(entry.get("description") or ""),
                int(entry.get("duration") or 5),
                runtime,
            )

    _extract_prefix(archive, "takes", takes_root(runtime), overwrite=False)

    target_result: Path | None = None
    if result_name and f"result/{result_name}" in names_in_archive:
        h3_dir = Path(runtime.output_dir) / "h3"
        h3_dir.mkdir(parents=True, exist_ok=True)
        result_path_obj = Path(result_name)
        target_result_name = result_name
        if conflict == "rename" and (h3_dir / result_name).exists():
            used_stems = {p.stem for p in h3_dir.iterdir() if p.is_file()}
            new_stem = _unique_name(result_path_obj.stem, used_stems, style="stem")
            target_result_name = f"{new_stem}{result_path_obj.suffix}"
        target_result = h3_dir / target_result_name
        _extract_member_atomic(archive, f"result/{result_name}", target_result)

        snap_data = None
        snap_arc = f"result/{result_name}.settings.json"
        if snap_arc in names_in_archive:
            target_snap = snapshot_path(target_result)
            _extract_member_atomic(archive, snap_arc, target_snap)
            snap_data = _update_snapshot_file(target_snap, output_name=target_result.name)
        _register_imported_media(
            target_result, snap_data or {"family": "Drama 3"}, store=store, runtime=runtime
        )

        plan_arc = f"result/.{result_path_obj.stem}.visual_plan.json"
        if plan_arc in names_in_archive:
            _extract_member_atomic(archive, plan_arc, visuals.plan_path(target_result))

        stills_dir = visuals.stills_dir(target_result, runtime.output_dir)
        _extract_prefix(archive, "stills", stills_dir, overwrite=True)
        if target_result.name != result_name:
            _update_stills_source(stills_dir, target_result.name, visuals.STILLS_RECORD)
    else:
        plan_audio = _soundstage_plan_audio(runtime, target_name)
        if "plan/visual_plan.json" in names_in_archive:
            _extract_member_atomic(archive, "plan/visual_plan.json", visuals.plan_path(plan_audio))
        plan_stills_dir = visuals.stills_dir(plan_audio, runtime.output_dir)
        if _extract_prefix(archive, "plan_stills", plan_stills_dir, overwrite=True):
            _update_stills_source(plan_stills_dir, plan_audio.name, visuals.STILLS_RECORD)

    store.save_soundstage_project(target_name, payload)
    if target_result is not None:
        store.mark_soundstage_generated(target_name, str(target_result))
    elif status == "overwritten":
        with store.connect() as db:
            db.execute("UPDATE soundstage_projects SET result=NULL WHERE name=?", (target_name,))

    return _transfer_result("soundstage", source_id, orig_name, target_name, target_name, status)


def _import_storyboard(
    archive: zipfile.ZipFile, *, store, runtime, conflict: str, owner: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.storyboard import model

    payload = json.loads(archive.read("board.json"))
    orig_id = str(payload.get("id") or source_id)
    board = payload["board"]
    orig_name = str(board.get("name") or source_name or "Untitled storyboard")

    existing = store.storyboards()
    by_id = {str(row["id"]): row for row in existing}
    by_name = {str(row["name"]): row for row in existing}
    existing_names = set(by_name)
    match = by_id.get(orig_id) or by_name.get(orig_name)
    if match is not None:
        if conflict == "skip":
            return _transfer_result(
                "storyboard", orig_id, orig_name, str(match["id"]), str(match["name"]), "skipped"
            )
        if conflict == "overwrite":
            target_id = str(match["id"])
            target_name = orig_name
            status = "overwritten"
        else:
            target_id = uuid.uuid4().hex
            while target_id in by_id:
                target_id = uuid.uuid4().hex
            target_name = _unique_name(orig_name, existing_names)
            status = "renamed"
    else:
        target_id = orig_id if re.fullmatch(r"[A-Za-z0-9_-]+", orig_id) else uuid.uuid4().hex
        target_name = orig_name
        status = "imported"

    cast_dir = store.root / "storyboards" / target_id / "cast"
    media_dir = Path(runtime.output_dir) / "h3" / ".storyboard" / target_id / "media"
    takes_dir = Path(runtime.output_dir) / "h3" / ".storyboard" / target_id

    extracted_cast = _extract_prefix(archive, "cast", cast_dir, overwrite=True)
    extracted_media = _extract_prefix(archive, "media", media_dir, overwrite=True)
    extracted_takes = _extract_prefix(archive, "takes", takes_dir, overwrite=True)

    arc_to_abs = {**{k: str(v) for k, v in extracted_cast.items()}, **{k: str(v) for k, v in extracted_media.items()}}
    take_abs = {k: str(v) for k, v in extracted_takes.items()}
    take_rel = {k: f"h3/.storyboard/{target_id}/{v.name}" for k, v in extracted_takes.items()}

    for arc, local_path in extracted_takes.items():
        if local_path.name.endswith(".settings.json"):
            _update_snapshot_file(local_path, output_name=local_path.name[: -len(".settings.json")])

    def rewrite_snapshot_lists(snapshot: dict[str, Any]) -> None:
        for member in snapshot.get("cast") or []:
            if isinstance(member, dict) and member.get("image"):
                member["image"] = arc_to_abs.get(str(member["image"]), member["image"])
        for scene in snapshot.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for key in _STORYBOARD_SINGLE_MEDIA_KEYS:
                if scene.get(key):
                    scene[key] = arc_to_abs.get(str(scene[key]), scene[key])
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if scene.get(key):
                    scene[key] = [arc_to_abs.get(str(item), item) for item in scene[key]]

    rewrite_snapshot_lists(board)
    for version in board.get("versions") or []:
        if isinstance(version, dict):
            rewrite_snapshot_lists(version)

    for take in (board.get("takes") or {}).values():
        if not isinstance(take, dict):
            continue
        for key in ("video", "tail"):
            if take.get(key):
                take[key] = take_abs.get(str(take[key]), take[key])
        for key in ("latent", "stage_latent"):
            if take.get(key):
                take[key] = take_rel.get(str(take[key]), take[key])
        inputs = take.get("inputs")
        if isinstance(inputs, dict):
            for key in ("first_image", "last_image"):
                if inputs.get(key):
                    inputs[key] = arc_to_abs.get(str(inputs[key]), inputs[key])
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if isinstance(inputs.get(key), list):
                    inputs[key] = [arc_to_abs.get(str(item), item) for item in inputs[key]]

    board["name"] = target_name
    for scene in board.get("scenes") or []:
        model.validate_scene(scene)
    store.save_storyboard(owner, target_id, board)
    return _transfer_result("storyboard", orig_id, orig_name, target_id, target_name, status)


def _import_dubbing(
    archive: zipfile.ZipFile, *, store, runtime, conflict: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.dubbing import projects as dubbing_projects, results as dub_results, session as sessions

    data = json.loads(archive.read("project.json"))
    orig_name = str(data.get("name") or source_name or source_id)
    existing = set(dubbing_projects.names(runtime))
    target_name, status, skipped = _resolve_named_conflict("dubbing", source_id, orig_name, existing, conflict)
    if skipped is not None:
        return skipped

    _restore_referenced_speakers(data.get("referenced_speakers") or [], archive, runtime)

    new_session_id = uuid.uuid4().hex
    session_root = sessions.sessions_dir(runtime) / new_session_id
    _extract_prefix(archive, "session", session_root, overwrite=True)

    manifest_path = session_root / sessions.MANIFEST
    try:
        session_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise H3Error("Imported dubbing archive is missing a valid session.json.") from exc
    session_data["id"] = new_session_id
    session_data["updated"] = time.time()

    names_in_archive = set(archive.namelist())
    last_output_name = data.get("last_output_name")
    result_entries = [name for name in names_in_archive if name.startswith("result/") and not name.endswith("/")]
    if last_output_name and result_entries:
        dub_out_dir = Path(runtime.outputs_dir) / "dubbing"
        dub_out_dir.mkdir(parents=True, exist_ok=True)
        orig_stem = str(data.get("last_output_stem") or Path(last_output_name).stem)
        target_stem = orig_stem
        if conflict == "rename" and any(
            (dub_out_dir / candidate).exists()
            for candidate in (last_output_name, f"{orig_stem}.mp3", f"{orig_stem}.mp4")
        ):
            used_stems = {p.stem for p in dub_out_dir.iterdir() if p.is_file()}
            target_stem = _unique_name(orig_stem, used_stems, style="stem")

        written_media: list[Path] = []
        for arc_name in result_entries:
            fname = arc_name[len("result/") :]
            if target_stem != orig_stem and fname.startswith(orig_stem):
                target_fname = target_stem + fname[len(orig_stem) :]
            else:
                target_fname = fname
            dest = dub_out_dir / target_fname
            _extract_member_atomic(archive, arc_name, dest)
            if dest.name.endswith(".settings.json"):
                _update_snapshot_file(
                    dest,
                    output_name=dest.name[: -len(".settings.json")],
                    session_id=new_session_id,
                )
            else:
                written_media.append(dest)
        for media_path in written_media:
            _register_imported_media(
                media_path,
                read_snapshot(media_path) or {"family": dub_results.DUB_FAMILY},
                store=store,
                runtime=runtime,
            )

        player_dir = dub_out_dir / dub_results.PLAYER_DIR / target_stem
        _extract_prefix(archive, "result_player", player_dir, overwrite=True)
        if target_stem != orig_stem:
            for meta_name in (dub_results.PLAYER_FILE, "alignment.json"):
                meta_path = player_dir / meta_name
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(meta, dict):
                    container = meta.get("player") if meta_name == "alignment.json" and isinstance(meta.get("player"), dict) else meta
                    for field in ("video", "dub"):
                        val = container.get(field)
                        if isinstance(val, str) and Path(val.replace("\\", "/")).name.startswith(orig_stem):
                            fname = Path(val.replace("\\", "/")).name
                            container[field] = str(dub_out_dir / (target_stem + fname[len(orig_stem) :]))
                    write_json_atomic(meta_path, meta)

        target_last_name = (
            target_stem + last_output_name[len(orig_stem) :]
            if target_stem != orig_stem and last_output_name.startswith(orig_stem)
            else last_output_name
        )
        session_data["last_output"] = str(dub_out_dir / target_last_name)
    else:
        session_data.pop("last_output", None)

    write_json_atomic(manifest_path, session_data)
    project_path = dubbing_projects._path(runtime, target_name)
    write_json_atomic(
        project_path,
        {
            "name": target_name,
            "session": new_session_id,
            "settings": data["settings"],
            "generation": uuid.uuid4().hex,
        },
    )
    return _transfer_result("dubbing", source_id, orig_name, target_name, target_name, status)


def _import_remake(
    archive: zipfile.ZipFile, *, runtime, conflict: str, owner: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.remake.project import Projects as RemakeProjects

    project = json.loads(archive.read("project.json"))
    orig_key = str(project.get("key") or source_id)
    orig_name = str(project.get("name") or source_name or orig_key)

    projects = RemakeProjects(runtime.output_dir)
    existing_list = projects.list()
    existing_keys = {str(item.get("key") or "") for item in existing_list}
    existing_names = {str(item.get("name") or "") for item in existing_list}
    existing = projects.load(None, orig_key)

    if existing is not None:
        if conflict == "skip":
            return _transfer_result(
                "remake", orig_key, orig_name, str(existing["key"]), str(existing.get("name") or orig_name), "skipped"
            )
        if conflict == "overwrite":
            target_key = str(existing["key"])
            target_name = orig_name
            target_owner = str(existing.get("owner") or owner)
            target_folder = Path(existing["folder"])
            if target_folder.exists():
                shutil.rmtree(target_folder)
            status = "overwritten"
        else:
            target_key = f"{orig_key[:12]}-{uuid.uuid4().hex[:8]}"
            while target_key in existing_keys:
                target_key = f"{orig_key[:12]}-{uuid.uuid4().hex[:8]}"
            target_name = _unique_name(orig_name, existing_names)
            target_owner = owner
            target_folder = projects.folder(target_owner, target_key)
            project.pop("aliases", None)
            status = "renamed"
    else:
        target_key = orig_key
        if conflict == "rename" and orig_name in existing_names:
            target_name = _unique_name(orig_name, existing_names)
            status = "renamed"
        else:
            target_name = orig_name
            status = "imported"
        target_owner = owner
        target_folder = projects.folder(target_owner, target_key)

    target_folder.mkdir(parents=True, exist_ok=True)
    extracted_folder = _extract_prefix(archive, "folder", target_folder, overwrite=True)
    extracted_takes = _extract_prefix(archive, "takes", target_folder / "takes", overwrite=True)
    arc_map = {**{k: str(v) for k, v in extracted_folder.items()}, **{k: str(v) for k, v in extracted_takes.items()}}

    for local_path in extracted_takes.values():
        if local_path.name.endswith(".settings.json"):
            _update_snapshot_file(local_path, output_name=local_path.name[: -len(".settings.json")])

    project["key"] = target_key
    project["name"] = target_name
    project["owner"] = target_owner
    project["folder"] = str(target_folder)
    if project.get("source"):
        project["source"] = arc_map.get(str(project["source"]), project["source"])
    for window in project.get("windows") or []:
        if window.get("pictures"):
            window["pictures"] = [arc_map.get(str(pic), pic) for pic in window["pictures"]]
        if window.get("carried"):
            window["carried"] = arc_map.get(str(window["carried"]), window["carried"])
        if window.get("take"):
            window["take"] = arc_map.get(str(window["take"]), window["take"])

    RemakeProjects._write(project)
    return _transfer_result("remake", orig_key, orig_name, target_key, target_name, status)


def import_project_archive(
    archive_path: Path, *, store, runtime, conflict: str = "rename", owner: str = "imported"
) -> dict[str, Any]:
    """Validate and import a project archive into the local store and runtime."""
    if conflict not in CONFLICT_MODES:
        raise H3Error(f"Unsupported conflict mode: {conflict!r}")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise H3Error("Project archive file was not found.")
    with zipfile.ZipFile(archive_path, "r") as archive:
        _validate_archive(archive)
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, ValueError) as exc:
            raise H3Error("Invalid project archive: missing manifest.json.") from exc
        if not isinstance(manifest, dict):
            raise H3Error("Invalid project archive manifest.")
        version = int(manifest.get("schema_version") or 0)
        if version <= 0 or version > TRANSFER_SCHEMA_VERSION:
            raise H3Error(f"Project archive schema version {version} requires a newer Prism version.")
        kind = str(manifest.get("kind") or "")
        if kind not in PROJECT_KINDS:
            raise H3Error(f"Unsupported project kind in archive: {kind!r}")
        source_id = str(manifest.get("id") or "")
        source_name = str(manifest.get("name") or source_id)
        if kind == "soundstage":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            return _import_soundstage(
                archive, store=store, runtime=runtime, conflict=conflict, source_id=source_id, source_name=source_name
            )
        if kind == "storyboard":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            return _import_storyboard(
                archive,
                store=store,
                runtime=runtime,
                conflict=conflict,
                owner=owner,
                source_id=source_id,
                source_name=source_name,
            )
        if kind == "dubbing":
            return _import_dubbing(
                archive, store=store, runtime=runtime, conflict=conflict, source_id=source_id, source_name=source_name
            )
        return _import_remake(
            archive,
            runtime=runtime,
            conflict=conflict,
            owner=owner,
            source_id=source_id,
            source_name=source_name,
        )


def _translate_http_error(response: httpx.Response, base_url: str) -> H3Error:
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            detail = str(payload["detail"])
    except (ValueError, TypeError):
        pass
    if response.status_code in (401, 403):
        return H3Error("Authentication failed on the remote server. Check Modal-Key and Modal-Secret.")
    if response.status_code == 404:
        return H3Error(
            detail or f"{base_url} does not support project transfer yet. Update and redeploy it first."
        )
    if detail:
        return H3Error(detail)
    return H3Error(f"Remote server returned HTTP {response.status_code}.")


def fetch_remote_projects(
    base_url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    """Query a remote Prism Studio or Workshop server for its exportable projects."""
    normalized = normalize_server_url(base_url)
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(f"{normalized}/api/projects/transfer/list", headers=headers or {})
    except httpx.RequestError as exc:
        raise H3Error(f"Could not reach {normalized}: {exc}") from exc
    if response.status_code >= 400:
        raise _translate_http_error(response, normalized)
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise H3Error("Remote server did not return a valid Prism project list.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        raise H3Error("Remote server returned an unexpected project list format.")
    return data


def import_remote_projects(
    base_url: str,
    selections: list[tuple[str, str]],
    *,
    store,
    runtime,
    conflict: str = "rename",
    headers: dict[str, str] | None = None,
    owner: str = "imported",
    progress: Callable[[float, str], None] | None = None,
    on_imported: Callable[[], None] | None = None,
    timeout: float = 300.0,
) -> list[dict[str, Any]]:
    """Download and import selected projects from a remote Prism Studio or Workshop URL."""
    if conflict not in CONFLICT_MODES:
        raise H3Error(f"Unsupported conflict mode: {conflict!r}")
    normalized = normalize_server_url(base_url)
    items = [(str(kind), str(project_id)) for kind, project_id in selections]
    if not items:
        raise H3Error("Select at least one project to import.")

    results: list[dict[str, Any]] = []
    changed = False
    with TemporaryDirectory(prefix="h3-project-import-") as tmp_dir, httpx.Client(
        follow_redirects=True, timeout=timeout
    ) as client:
        tmp_root = Path(tmp_dir)
        for index, (kind, project_id) in enumerate(items):
            label = KIND_LABELS.get(kind, kind)
            if progress is not None:
                progress(index / len(items), f"Downloading {label} project {project_id}…")
            archive_path = tmp_root / f"transfer_{index}.zip"
            try:
                with client.stream(
                    "GET",
                    f"{normalized}/api/projects/transfer/export",
                    params={"kind": kind, "id": project_id},
                    headers=headers or {},
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise _translate_http_error(response, normalized)
                    with archive_path.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                if progress is not None:
                    progress((index + 0.5) / len(items), f"Importing {label} project {project_id}…")
                result = import_project_archive(
                    archive_path, store=store, runtime=runtime, conflict=conflict, owner=owner
                )
                if result["status"] in {"imported", "renamed", "overwritten"}:
                    changed = True
                results.append(result)
            except (H3Error, httpx.RequestError, zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
                results.append(
                    {
                        "kind": kind,
                        "source_id": project_id,
                        "source_name": project_id,
                        "target_id": None,
                        "target_name": None,
                        "status": "failed",
                        "message": f'Failed {label} "{project_id}": {exc}',
                    }
                )
            finally:
                archive_path.unlink(missing_ok=True)
    if progress is not None:
        progress(1.0, "Import complete.")
    if changed and on_imported is not None:
        on_imported()
    return results
