"""Portable cross-server project export, import, and remote transfer client."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import count
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Sequence
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
    "/system",
    "/comfyui",
)
_STORYBOARD_TAKE_PATH_KEYS = ("video", "latent", "stage_latent", "tail")
_STORYBOARD_SINGLE_MEDIA_KEYS = ("first_image", "last_image", "video")
_STORYBOARD_LIST_MEDIA_KEYS = ("ref_images", "ref_videos", "ref_audios")
_SYMLINK_MODE = 0o120000
_FILE_TYPE_MASK = 0o170000
_IO_CHUNK_BYTES = 4 << 20
_HTTP_MAX_ATTEMPTS = 4
_MAX_PARALLEL_DOWNLOADS = 3
_MEDIA_PREPARE_WORKERS = 4
_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 502, 503, 504})
_PROJECT_TRANSFER_LOCK = threading.RLock()
_PRECOMPRESSED_SUFFIXES = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mov",
        ".mkv",
        ".avi",
        ".ts",
        ".m2ts",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".webm",
        ".flac",
        # Uncompressed PCM WAVs are high-entropy; storing avoids heavy zlib CPU burn on low-CPU hosts.
        ".wav",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".avif",
        ".heic",
        ".heif",
        ".safetensors",
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
        ".zip",
        ".gz",
        ".zst",
        ".bz2",
        ".xz",
    }
)
_DUBBING_SESSION_SKIP_FILES = frozenset({"speech.wav", "mix.wav", "speech-16k.wav", "speech.mp3"})
_DUBBING_SESSION_SKIP_DIRS = frozenset({"timed", "stems"})
_DUBBING_PLAYER_SKIP_DIRS = frozenset({"lip_sync_cache"})
_TRANSIENT_FS_PREFIXES = (".saving-", ".replacing-", ".importing-")
_TRANSIENT_FS_SUFFIXES = (".tmp", ".partial", ".importing")


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


def _is_transient_fs_name(name: str) -> bool:
    return name.startswith(_TRANSIENT_FS_PREFIXES) or name.endswith(_TRANSIENT_FS_SUFFIXES)


def _extract_member_direct(archive: zipfile.ZipFile, member: str | zipfile.ZipInfo, destination: Path) -> None:
    with archive.open(member, "r") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=_IO_CHUNK_BYTES)


def _extract_member_atomic(
    archive: zipfile.ZipFile, member: str | zipfile.ZipInfo, destination: Path, *, ensure_parent: bool = True
) -> None:
    if ensure_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        _extract_member_direct(archive, member, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _same_file_as_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, existing: Path) -> bool:
    """Check whether ``existing`` already matches ``info`` byte-for-byte without writing."""
    try:
        stat = existing.stat()
    except OSError:
        return False
    if stat.st_size != info.file_size:
        return False
    try:
        crc = 0
        with existing.open("rb") as local_handle:
            while chunk := local_handle.read(_IO_CHUNK_BYTES):
                crc = zlib.crc32(chunk, crc)
        if (crc & 0xFFFFFFFF) != (info.CRC & 0xFFFFFFFF):
            return False
        with existing.open("rb") as local_handle, archive.open(info, "r") as zip_handle:
            while True:
                left = local_handle.read(_IO_CHUNK_BYTES)
                right = zip_handle.read(len(left) if left else _IO_CHUNK_BYTES)
                if left != right:
                    return False
                if not left:
                    return True
    except OSError:
        return False


def _add_zip_file(archive: zipfile.ZipFile, source: Path, arc_name: str, added: set[str]) -> bool:
    if arc_name in added or not source.is_file() or source.is_symlink():
        return False
    zinfo = zipfile.ZipInfo.from_file(source, arc_name, strict_timestamps=False)
    if zinfo.is_dir():
        return False
    compress = zipfile.ZIP_STORED if source.suffix.lower() in _PRECOMPRESSED_SUFFIXES else zipfile.ZIP_DEFLATED
    zinfo.compress_type = compress
    if compress == zipfile.ZIP_DEFLATED:
        zinfo._compresslevel = 1
    with source.open("rb") as reader, archive.open(
        zinfo, "w", force_zip64=zinfo.file_size >= (1 << 31)
    ) as writer:
        shutil.copyfileobj(reader, writer, length=_IO_CHUNK_BYTES)
    added.add(arc_name)
    return True


def _iter_scandir_files(
    root_dir: Path,
    rel_prefix: str = "",
    *,
    skip_rel: Callable[[PurePosixPath], bool] | None = None,
) -> Iterable[tuple[Path, str]]:
    """Walk ``root_dir`` deterministically while pruning skipped directories before statting children."""
    try:
        with os.scandir(root_dir) as raw_entries:
            entries = sorted(raw_entries, key=lambda entry: entry.name)
    except OSError:
        return
    for entry in entries:
        name = entry.name
        if _is_transient_fs_name(name):
            continue
        try:
            if entry.is_symlink():
                continue
            rel = f"{rel_prefix}/{name}" if rel_prefix else name
            if skip_rel is not None and skip_rel(PurePosixPath(rel)):
                continue
            if entry.is_dir(follow_symlinks=False):
                yield from _iter_scandir_files(Path(entry.path), rel, skip_rel=skip_rel)
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path), rel
        except OSError:
            continue


def _add_zip_dir(
    archive: zipfile.ZipFile,
    root_dir: Path,
    prefix: str,
    added: set[str],
    *,
    skip_rel: Callable[[PurePosixPath], bool] | None = None,
) -> None:
    if not root_dir.is_dir():
        return
    for path, rel in _iter_scandir_files(root_dir, skip_rel=skip_rel):
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


def _extract_prefix(
    archive: zipfile.ZipFile, prefix: str, dest_dir: Path, *, overwrite: bool = True, atomic: bool = True
) -> dict[str, Path]:
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    created_dirs: set[Path] = {dest_root}
    extracted: dict[str, Path] = {}
    head = prefix.rstrip("/") + "/"
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(head):
            continue
        rel = info.filename[len(head) :]
        posix_rel = PurePosixPath(rel)
        if (
            not rel
            or rel.startswith("/")
            or "\\" in rel
            or ":" in rel
            or any(part in ("", ".", "..") for part in posix_rel.parts)
            or ((info.external_attr >> 16) & _FILE_TYPE_MASK) == _SYMLINK_MODE
        ):
            raise H3Error(f"Unsafe archive path: {info.filename!r}")
        target = dest_root.joinpath(*posix_rel.parts)
        if not Path(os.path.abspath(target)).is_relative_to(dest_root):
            raise H3Error(f"Unsafe archive path: {info.filename!r}")
        if not overwrite and target.exists():
            extracted[info.filename] = target
            continue
        parent = target.parent
        if parent not in created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(parent)
        if atomic:
            _extract_member_atomic(archive, info, target, ensure_parent=False)
        else:
            _extract_member_direct(archive, info, target)
        extracted[info.filename] = target
    return extracted


class _StagingTransaction:
    """Stage extracted files and folders beside their targets and swap atomically on commit."""

    def __init__(self) -> None:
        self._dir_swaps: list[tuple[Path, Path]] = []
        self._file_swaps: list[tuple[Path, Path]] = []
        self._tracked_dirs: list[Path] = []
        self._committed = False

    def __enter__(self) -> _StagingTransaction:
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if self._committed:
            return
        for staging, _target in self._file_swaps:
            staging.unlink(missing_ok=True)
        for staging, _target in self._dir_swaps:
            shutil.rmtree(staging, ignore_errors=True)
        for folder in self._tracked_dirs:
            shutil.rmtree(folder, ignore_errors=True)

    def stage_dir(self, target_dir: Path, *, create: bool = True) -> Path:
        target = target_dir.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".importing-{target.name}-{uuid.uuid4().hex[:8]}"
        if create:
            staging.mkdir(parents=True, exist_ok=True)
        self._dir_swaps.append((staging, target))
        return staging

    def stage_file(self, target_file: Path) -> Path:
        target = target_file.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.importing"
        self._file_swaps.append((staging, target))
        return staging

    def track_created_dir(self, folder: Path) -> Path:
        resolved = folder.resolve()
        self._tracked_dirs.append(resolved)
        return resolved

    def commit(self, finalize: Callable[[], None] | None = None) -> None:
        dir_backups: list[tuple[Path, Path, bool]] = []
        file_backups: list[tuple[Path, Path, bool]] = []
        try:
            for staging, target in self._dir_swaps:
                has_staging = staging.is_dir()
                had_target = target.exists()
                backup = target.parent / f".replacing-{target.name}-{uuid.uuid4().hex[:8]}"
                if had_target:
                    target.rename(backup)
                if has_staging:
                    staging.rename(target)
                dir_backups.append((backup, target, had_target))

            for staging, target in self._file_swaps:
                if not staging.is_file():
                    continue
                had_target = target.is_file()
                backup = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.replacing"
                if had_target:
                    os.replace(target, backup)
                os.replace(staging, target)
                file_backups.append((backup, target, had_target))

            if finalize is not None:
                finalize()
        except BaseException:
            for backup, target, had_target in reversed(file_backups):
                try:
                    target.unlink(missing_ok=True)
                    if had_target and backup.is_file():
                        os.replace(backup, target)
                except OSError:
                    pass
                finally:
                    backup.unlink(missing_ok=True)
            for backup, target, had_target in reversed(dir_backups):
                try:
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                    if had_target and backup.exists():
                        backup.rename(target)
                except OSError:
                    pass
                finally:
                    shutil.rmtree(backup, ignore_errors=True)
            raise

        for backup, _target, _had_target in file_backups:
            backup.unlink(missing_ok=True)
        for backup, _target, _had_target in dir_backups:
            shutil.rmtree(backup, ignore_errors=True)
        self._committed = True


def _retarget_extracted(extracted: dict[str, Path], staging_root: Path, final_root: Path) -> dict[str, Path]:
    return {arc: final_root.joinpath(*path.relative_to(staging_root).parts) for arc, path in extracted.items()}


def _update_snapshot_file(
    path: Path,
    *,
    output_name: str,
    target_media: Path | None = None,
    session_id: str | None = None,
    source_name: str | None = None,
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    data["output"] = output_name
    resolved_media = (target_media if target_media is not None else path.with_name(output_name)).resolve()
    data["asset_id"] = uuid.uuid5(uuid.NAMESPACE_URL, resolved_media.as_uri()).hex
    if isinstance(data.get("settings"), dict):
        if session_id is not None:
            data["settings"]["session"] = session_id
        if source_name is not None and "source" in data["settings"]:
            data["settings"]["source"] = source_name
    write_json_atomic(path, data)
    return data


def _register_imported_media_batch(
    entries: Iterable[tuple[Path, dict[str, Any] | None]], *, store, runtime
) -> None:
    """Index imported output files into the Media library catalog in one batch."""
    if store is None:
        return
    valid = [(media_path, metadata or {}) for media_path, metadata in entries if media_path.is_file()]
    if not valid:
        return
    from h3_app.media_catalog import MediaCatalog

    catalog = MediaCatalog(store, lambda: runtime)
    if len(valid) == 1:
        single = catalog.prepare(valid[0][0], valid[0][1])
        records = [single] if single is not None else []
    else:
        workers = min(_MEDIA_PREPARE_WORKERS, len(valid))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = list(pool.map(lambda item: catalog.prepare(item[0], item[1]), valid))
        records = [record for record in prepared if record is not None]
    if records:
        store.index_assets(records)


def _register_imported_media(media_path: Path, metadata: dict[str, Any] | None, *, store, runtime) -> None:
    """Index an imported output file into the Media library catalog immediately."""
    _register_imported_media_batch([(media_path, metadata)], store=store, runtime=runtime)


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
    if not entries:
        return
    from h3_app.drama3 import speakers as drama3_speakers

    names_in_archive = set(archive.namelist())
    pending: list[dict[str, Any]] = []
    for entry in entries:
        key = str(entry.get("key") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not key or audio_file not in names_in_archive:
            continue
        try:
            drama3_speakers.load_speaker(key, runtime)
            continue
        except H3Error:
            pending.append(entry)
    if not pending:
        return
    with TemporaryDirectory(prefix="h3-import-speakers-") as tmp:
        tmp_root = Path(tmp)
        for index, entry in enumerate(pending):
            audio_file = str(entry.get("audio_file") or "")
            suffix = Path(audio_file).suffix.lower() or ".wav"
            clip = tmp_root / f"voice_{index}{suffix}"
            try:
                _extract_member_direct(archive, audio_file, clip)
                drama3_speakers.save_speaker(
                    str(entry.get("name") or ""),
                    clip,
                    str(entry.get("transcript") or ""),
                    runtime,
                    str(entry.get("description") or ""),
                    str(entry.get("group") or "Global"),
                )
            except (H3Error, KeyError, OSError):
                continue
            finally:
                clip.unlink(missing_ok=True)


def _soundstage_clips_dir(runtime) -> Path:
    return Path(runtime.output_dir) / "h3" / ".soundstage" / "clips"


def _soundstage_plan_audio(runtime, project_name: str) -> Path:
    digest = hashlib.sha256(project_name.encode()).hexdigest()
    return Path(runtime.output_dir) / ".soundstage" / "plans" / f"{digest}.wav"


def _latest_soundstage_video(audio_path: Path) -> Path | None:
    videos: list[tuple[int, Path]] = []
    for candidate in audio_path.parent.glob(f"{glob.escape(audio_path.stem)}_illustrated_*.mp4"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                videos.append((candidate.stat().st_mtime_ns, candidate))
        except OSError:
            continue
    return max(videos, key=lambda item: item[0])[1] if videos else None


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
    with store.connect() as db:
        rows = db.execute(
            "SELECT id, name, updated, "
            "CASE WHEN json_valid(payload) AND json_type(payload, '$.scenes') = 'array' "
            "THEN json_array_length(payload, '$.scenes') ELSE 0 END AS scenes, "
            "CASE WHEN json_valid(payload) AND json_type(payload, '$.takes') = 'object' "
            "THEN (SELECT COUNT(*) FROM json_each(payload, '$.takes')) ELSE 0 END AS takes "
            "FROM storyboards ORDER BY updated DESC"
        ).fetchall()
    return [
        {
            "kind": "storyboard",
            "id": str(row["id"]),
            "name": str(row["name"]),
            "updated": float(row["updated"] or 0),
            "summary": f"{_plural(int(row['scenes'] or 0), 'scene')} · {_plural(int(row['takes'] or 0), 'take')}",
        }
        for row in rows
    ]


def _list_dubbing(runtime) -> list[dict[str, Any]]:
    from h3_app.dubbing import projects as dubbing_projects

    items: list[dict[str, Any]] = []
    for entry in dubbing_projects.summaries(runtime):
        segments = int(entry["segments"])
        src = entry["source_language"]
        tgt = entry["target_language"]
        if segments:
            lang = f"{src} → {tgt}" if src and tgt else tgt or src
            summary = f"{_plural(segments, 'line')} · {lang}" if lang else _plural(segments, "line")
        else:
            summary = "Source only"
        name = str(entry["name"])
        items.append(
            {
                "kind": "dubbing",
                "id": name,
                "name": name,
                "updated": float(entry["updated"] or 0),
                "summary": summary,
            }
        )
    return items


def _list_remake(runtime) -> list[dict[str, Any]]:
    from h3_app.remake.project import Projects as RemakeProjects

    items: list[dict[str, Any]] = []
    for project in RemakeProjects(runtime.output_dir).list():
        windows = project.get("windows") or []
        done = sum(1 for window in windows if isinstance(window, dict) and window.get("take"))
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


def _stat_stamp(path: Path | None) -> str:
    if path is None:
        return "none"
    try:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "missing"


def project_export_stamp(kind: str, project_id: str, *, store, runtime) -> str:
    """Compute a fast freshness fingerprint for export caching, raising H3Error if missing."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")
    if kind == "soundstage":
        if store is None:
            raise H3Error("Workspace store is unavailable.")
        with store.connect() as db:
            row = db.execute(
                "SELECT updated, result FROM soundstage_projects WHERE name=?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise H3Error(f'Soundstage project "{project_id}" no longer exists.')
        from h3_ui.drama3 import visuals

        result = row["result"]
        result_path = Path(result) if result else None
        if result_path is not None and not result_path.is_file():
            output_dir = Path(runtime.output_dir)
            result_path = next(
                (
                    candidate
                    for candidate in (
                        output_dir / "drama3" / result_path.name,
                        output_dir / "h3" / result_path.name,
                    )
                    if candidate.is_file()
                ),
                None,
            )
        if result_path is not None:
            stills = visuals.stills_dir(result_path, runtime.output_dir)
            latest_vid = _latest_soundstage_video(result_path)
            parts = [
                str(row["updated"]),
                str(result_path),
                _stat_stamp(result_path),
                _stat_stamp(snapshot_path(result_path)),
                _stat_stamp(visuals.plan_path(result_path)),
                _stat_stamp(stills / visuals.STILLS_RECORD),
                _stat_stamp(latest_vid),
            ]
        else:
            plan_audio = _soundstage_plan_audio(runtime, project_id)
            stills = visuals.stills_dir(plan_audio, runtime.output_dir)
            parts = [
                str(row["updated"]),
                _stat_stamp(visuals.plan_path(plan_audio)),
                _stat_stamp(stills / visuals.STILLS_RECORD),
            ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]

    if kind == "storyboard":
        if store is None:
            raise H3Error("Workspace store is unavailable.")
        with store.connect() as db:
            row = db.execute("SELECT updated FROM storyboards WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise H3Error(f'Storyboard "{project_id}" no longer exists.')
        return f"sb:{row['updated']}"

    if kind == "dubbing":
        from h3_app.dubbing import projects as dubbing_projects, results as dub_results

        mtimes = dubbing_projects.project_mtime_ns(runtime, project_id)
        if mtimes is None:
            raise H3Error("Choose a saved dubbing project.")
        data, session = dubbing_projects.load(runtime, project_id)
        last_output = session.data.get("last_output")
        last_path = Path(last_output) if last_output else None
        player_dir = dub_results.player_folder(last_path) if last_path is not None else None
        parts = [
            f"{mtimes[0]}:{mtimes[1]}",
            json.dumps(data.get("settings") or {}, sort_keys=True, ensure_ascii=False),
            _stat_stamp(last_path),
            _stat_stamp(last_path.with_name(f"{last_path.stem}.lipsync.mp4") if last_path else None),
            _stat_stamp(player_dir / dub_results.PLAYER_FILE if player_dir else None),
            _stat_stamp(player_dir / "lip_sync_plan.json" if player_dir else None),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]

    from h3_app.remake.project import Projects as RemakeProjects

    project = RemakeProjects(runtime.output_dir).load(None, project_id)
    if project is None:
        raise H3Error(f'Remake project "{project_id}" no longer exists.')
    take_stamps = [
        _stat_stamp(Path(str(win["take"])))
        for win in (project.get("windows") or ())
        if isinstance(win, dict) and win.get("take")
    ]
    raw = f"{project.get('updated')}|" + "|".join(take_stamps)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def rename_local_project(
    kind: str,
    project_id: str,
    new_name: str,
    *,
    store,
    runtime,
    on_changed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Rename a local Soundstage, Storyboard, Dubbing, or Remake project."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")
    cleaned = str(new_name or "").strip()
    if not cleaned:
        raise H3Error("Enter a project name.")

    with _PROJECT_TRANSFER_LOCK:
        if kind == "soundstage":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            try:
                store.rename_soundstage_project(project_id, cleaned)
            except ValueError as exc:
                raise H3Error(str(exc)) from exc
            if cleaned != project_id:
                from h3_ui.drama3 import visuals

                old_plan_audio = _soundstage_plan_audio(runtime, project_id)
                new_plan_audio = _soundstage_plan_audio(runtime, cleaned)
                old_plan = visuals.plan_path(old_plan_audio)
                new_plan = visuals.plan_path(new_plan_audio)
                if old_plan.is_file():
                    new_plan.parent.mkdir(parents=True, exist_ok=True)
                    old_plan.replace(new_plan)
                old_stills = visuals.stills_dir(old_plan_audio, runtime.output_dir)
                new_stills = visuals.stills_dir(new_plan_audio, runtime.output_dir)
                if old_stills.is_dir():
                    shutil.rmtree(new_stills, ignore_errors=True)
                    new_stills.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(old_stills, new_stills)
                    _update_stills_source(new_stills, new_plan_audio.name, visuals.STILLS_RECORD)
            outcome = {
                "kind": "soundstage",
                "old_id": project_id,
                "id": cleaned,
                "name": cleaned,
                "message": f'Renamed Soundstage "{project_id}" to "{cleaned}".',
            }
        elif kind == "storyboard":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            from h3_app.storyboard.service import StoryboardService

            try:
                existing = store.storyboard(None, project_id)
            except ValueError as exc:
                raise H3Error(str(exc)) from exc
            old_name = str(existing.get("name") or project_id)
            with store.connect() as db:
                row = db.execute("SELECT owner FROM storyboards WHERE id=?", (project_id,)).fetchone()
            sb_owner = str(row["owner"]) if row and row["owner"] else "local"
            board = StoryboardService(store, lambda: runtime).rename(sb_owner, project_id, cleaned)
            updated_name = str(board.get("name") or cleaned)
            outcome = {
                "kind": "storyboard",
                "old_id": project_id,
                "id": project_id,
                "name": updated_name,
                "message": f'Renamed Storyboard "{old_name}" to "{updated_name}".',
            }
        elif kind == "dubbing":
            from h3_app.dubbing import projects as dubbing_projects

            updated = dubbing_projects.rename(runtime, project_id, cleaned)
            outcome = {
                "kind": "dubbing",
                "old_id": project_id,
                "id": updated,
                "name": updated,
                "message": f'Renamed Dubbing "{project_id}" to "{updated}".',
            }
        else:
            from h3_app.remake.project import Projects as RemakeProjects

            projects = RemakeProjects(runtime.output_dir)
            project = projects.load(None, project_id)
            if project is None:
                raise H3Error(f'Remake project "{project_id}" no longer exists.')
            old_name = str(project.get("name") or project_id)
            projects.edit(project, lambda current: current.update(name=cleaned))
            outcome = {
                "kind": "remake",
                "old_id": project_id,
                "id": project_id,
                "name": cleaned,
                "message": f'Renamed Remake "{old_name}" to "{cleaned}".',
            }

        if on_changed is not None:
            on_changed()
        return outcome


def delete_local_project(
    kind: str,
    project_id: str,
    *,
    store,
    runtime,
    on_changed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Remove a local Soundstage, Storyboard, Dubbing, or Remake project."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")

    with _PROJECT_TRANSFER_LOCK:
        if kind == "soundstage":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            try:
                store.delete_soundstage_project(project_id)
            except ValueError as exc:
                raise H3Error(str(exc)) from exc
            from h3_ui.drama3 import visuals

            plan_audio = _soundstage_plan_audio(runtime, project_id)
            visuals.plan_path(plan_audio).unlink(missing_ok=True)
            shutil.rmtree(visuals.stills_dir(plan_audio, runtime.output_dir), ignore_errors=True)
            outcome = {
                "kind": "soundstage",
                "id": project_id,
                "name": project_id,
                "message": f'Removed Soundstage "{project_id}".',
            }
        elif kind == "storyboard":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            from h3_app.storyboard.service import StoryboardService

            try:
                existing = store.storyboard(None, project_id)
            except ValueError as exc:
                raise H3Error(str(exc)) from exc
            old_name = str(existing.get("name") or project_id)
            StoryboardService(store, lambda: runtime).delete(None, project_id)
            outcome = {
                "kind": "storyboard",
                "id": project_id,
                "name": old_name,
                "message": f'Removed Storyboard "{old_name}".',
            }
        elif kind == "dubbing":
            from h3_app.dubbing import projects as dubbing_projects

            dubbing_projects.delete(runtime, project_id)
            outcome = {
                "kind": "dubbing",
                "id": project_id,
                "name": project_id,
                "message": f'Removed Dubbing "{project_id}".',
            }
        else:
            from h3_app.remake.project import Projects as RemakeProjects

            projects = RemakeProjects(runtime.output_dir)
            project = projects.load(None, project_id)
            if project is None:
                raise H3Error(f'Remake project "{project_id}" no longer exists.')
            old_name = str(project.get("name") or project_id)
            projects.delete(project)
            outcome = {
                "kind": "remake",
                "id": project_id,
                "name": old_name,
                "message": f'Removed Remake "{old_name}".',
            }

        if on_changed is not None:
            on_changed()
        return outcome


def _write_canonical_effect_manifests(
    archive: zipfile.ZipFile,
    stills_folder: Path,
    *,
    audio_path: Path,
    output_dir: Path,
    duration: float,
    payload: dict[str, Any],
    added: set[str],
) -> None:
    """Emit current-key SAM 3.1 manifests during export so older servers also recognize them."""
    if not stills_folder.is_dir() or duration <= 0:
        return
    from h3_app.drama3 import visual_plan
    from h3_ui.drama3 import effect_regions, visuals

    saved_plan = visuals.read_plan(audio_path)
    record = visuals.read_stills(audio_path, output_dir)
    if not saved_plan or not record:
        return
    style = payload.get("video_style")
    if style is None:
        style = visuals.STYLE_PRESETS[visuals.DEFAULT_STYLE]
    orientation = visuals.orientation_of(payload.get("video_orientation"))
    try:
        shots = visuals.plan_shots_for(saved_plan, duration, style, orientation)
        plan = saved_plan["plan"]
        subjects_map = {
            key: {"kind": kind, "anchor": anchor}
            for key, (kind, anchor) in visual_plan.subjects(plan).items()
        }
        keys = visual_plan.mentions(plan)
    except H3Error:
        return
    for shot, still, row, mentioned in zip(shots, visuals.match_stills(shots, record), plan["shots"], keys):
        if still is None or not still.get("image"):
            continue
        image_name = str(still["image"])
        labels = effect_regions.concepts(
            {"prompt": row["prompt"], "beat": shot.beat, "subjects": list(mentioned)},
            subjects_map,
        )
        exact_name = effect_regions.manifest_path(stills_folder, image_name, labels).name
        arc_name = f"stills/{effect_regions.MASK_FOLDER}/{exact_name}"
        if arc_name in added:
            continue
        regions = effect_regions.read_regions(stills_folder, image_name, labels)
        if regions is not None:
            archive.writestr(arc_name, json.dumps(regions, ensure_ascii=False))
            added.add(arc_name)


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
    illustrated_video_name: str | None = None
    result_path = Path(result) if result else None
    if result_path is not None and not result_path.is_file():
        output_dir = Path(runtime.output_dir)
        result_path = next(
            (
                candidate
                for candidate in (
                    output_dir / "drama3" / result_path.name,
                    output_dir / "h3" / result_path.name,
                )
                if candidate.is_file()
            ),
            None,
        )
    if result_path is not None:
        result_name = result_path.name
        _add_zip_file(archive, result_path, f"result/{result_name}", added)
        snap = snapshot_path(result_path)
        _add_zip_file(archive, snap, f"result/{snap.name}", added)
        plan = visuals.plan_path(result_path)
        _add_zip_file(archive, plan, f"result/{plan.name}", added)
        latest_video = _latest_soundstage_video(result_path)
        if latest_video is not None:
            illustrated_video_name = latest_video.name
            _add_zip_file(archive, latest_video, f"result/{illustrated_video_name}", added)
            vsnap = snapshot_path(latest_video)
            _add_zip_file(archive, vsnap, f"result/{vsnap.name}", added)
        stills_folder = visuals.stills_dir(result_path, runtime.output_dir)
        _add_zip_dir(archive, stills_folder, "stills", added)
        settings = result_settings(result_path)
        _write_canonical_effect_manifests(
            archive,
            stills_folder,
            audio_path=result_path,
            output_dir=Path(runtime.output_dir),
            duration=float(settings.get("duration") or 0),
            payload=payload,
            added=added,
        )
        takes_dir = takes_root(runtime)
        for chunk in settings.get("chunks") or []:
            take_key = chunk.get("take")
            if take_key and re.fullmatch(r"[A-Za-z0-9_-]+", str(take_key)):
                _add_zip_file(archive, takes_dir / f"{take_key}.wav", f"takes/{take_key}.wav", added)
    else:
        plan_audio = _soundstage_plan_audio(runtime, project_id)
        _add_zip_file(archive, visuals.plan_path(plan_audio), "plan/visual_plan.json", added)
        _add_zip_dir(archive, visuals.stills_dir(plan_audio, runtime.output_dir), "plan_stills", added)

    project_record: dict[str, Any] = {
        "name": project_id,
        "payload": payload,
        "result_name": result_name,
        "referenced_speakers": referenced_speakers,
        "referenced_sounds": referenced_sounds,
    }
    if illustrated_video_name is not None:
        project_record["illustrated_video_name"] = illustrated_video_name
    archive.writestr("project.json", json.dumps(project_record, ensure_ascii=False, indent=2))
    return project_id


def _export_storyboard(archive: zipfile.ZipFile, project_id: str, *, store, runtime, added: set[str]) -> str:
    try:
        board = copy.deepcopy(store.storyboard(None, project_id))
    except ValueError as exc:
        raise H3Error(f'Storyboard "{project_id}" no longer exists.') from exc

    output_dir = Path(runtime.output_dir)
    used_cast_names: set[str] = set()
    used_media_names: set[str] = set()
    cast_map: dict[Path, str] = {}
    media_map: dict[Path, str] = {}
    raw_to_arc: dict[str, str] = {}
    cast_name_by_member: dict[str, str] = {}
    used_take_names: set[str] = set()
    take_map: dict[Path, str] = {}

    def resolve_path(raw: Any) -> Path | None:
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_absolute() else (output_dir / path)

    def archive_asset(raw: Any, prefix: str, used_names: set[str], cache: dict[Path, str]) -> str | None:
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
        filename = path.name
        idx = 2
        while filename in used_names:
            filename = f"{path.stem}_{idx}{path.suffix}"
            idx += 1
        used_names.add(filename)
        arc = f"{prefix}/{filename}"
        _add_zip_file(archive, resolved, arc, added)
        cache[resolved] = arc
        raw_to_arc[raw_str] = arc
        return arc

    def resolve_archived_media(raw: Any) -> Any:
        if not raw:
            return raw
        raw_str = str(raw)
        if raw_str in raw_to_arc:
            return raw_to_arc[raw_str]
        path = resolve_path(raw_str)
        if path is not None:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = None
            if resolved is not None and resolved in media_map:
                return media_map[resolved]
        return raw

    def rewrite_snapshot_lists(snapshot: dict[str, Any], *, record_active_cast: bool = False) -> None:
        for member in snapshot.get("cast") or []:
            if isinstance(member, dict) and member.get("image"):
                archived = archive_asset(member["image"], "cast", used_cast_names, cast_map)
                if archived is not None:
                    member["image"] = archived
                    if record_active_cast and member.get("name"):
                        cast_name_by_member[str(member["name"])] = PurePosixPath(archived).name
        for scene in snapshot.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for key in _STORYBOARD_SINGLE_MEDIA_KEYS:
                if scene.get(key):
                    scene[key] = archive_asset(scene[key], "media", used_media_names, media_map)
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if scene.get(key):
                    items = [archive_asset(item, "media", used_media_names, media_map) for item in scene[key]]
                    scene[key] = [item for item in items if item is not None]

    rewrite_snapshot_lists(board, record_active_cast=True)
    for version in board.get("versions") or []:
        if isinstance(version, dict):
            rewrite_snapshot_lists(version)

    scene_media_video_by_take: dict[str, str] = {}
    for scene in board.get("scenes") or []:
        if isinstance(scene, dict) and scene.get("in_use") and scene.get("video"):
            scene_media_video_by_take[str(scene["in_use"])] = PurePosixPath(str(scene["video"])).name

    for take_id, take in (board.get("takes") or {}).items():
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
                if inputs.get(key):
                    inputs[key] = resolve_archived_media(inputs[key])
            for key in _STORYBOARD_LIST_MEDIA_KEYS:
                if isinstance(inputs.get(key), list):
                    inputs[key] = [resolve_archived_media(item) for item in inputs[key]]
            if isinstance(inputs.get("cast"), list):
                for pair in inputs["cast"]:
                    if (
                        isinstance(pair, list)
                        and len(pair) == 2
                        and isinstance(pair[0], str)
                        and pair[0] in cast_name_by_member
                    ):
                        pair[1] = cast_name_by_member[pair[0]]
            if inputs.get("kind") == "media" and str(take_id) in scene_media_video_by_take:
                inputs["video"] = scene_media_video_by_take[str(take_id)]

    archive.writestr("board.json", json.dumps({"id": project_id, "board": board}, ensure_ascii=False, indent=2))
    return str(board.get("name") or project_id)


def _skip_dubbing_session_rel(rel: PurePosixPath) -> bool:
    parts = rel.parts
    if not parts:
        return False
    if len(parts) == 1 and parts[0] in _DUBBING_SESSION_SKIP_FILES:
        return True
    return parts[0] in _DUBBING_SESSION_SKIP_DIRS or parts[0].startswith("dub-words-")


def _skip_dubbing_player_rel(rel: PurePosixPath) -> bool:
    parts = rel.parts
    return bool(parts and parts[0] in _DUBBING_PLAYER_SKIP_DIRS)


def _export_dubbing(archive: zipfile.ZipFile, project_id: str, *, runtime, added: set[str]) -> str:
    from h3_app.dubbing import projects as dubbing_projects, results as dub_results

    data, session = dubbing_projects.load(runtime, project_id)
    _add_zip_dir(archive, session.root, "session", added, skip_rel=_skip_dubbing_session_rel)

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
        _add_zip_dir(archive, dub_results.player_folder(last_path), "result_player", added, skip_rel=_skip_dubbing_player_rel)

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
        for path, rel in _iter_scandir_files(folder):
            if path.name == "project.json" or path.resolve() in skip_files:
                continue
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
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as archive:
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _extract_soundstage_clips(
    archive: zipfile.ZipFile, clips_dir: Path, payload: dict[str, Any], *, conflict: str
) -> None:
    """Extract clips into the shared Soundstage folder without clobbering unrelated projects."""
    from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS

    clips_root = clips_dir.resolve()
    clips_root.mkdir(parents=True, exist_ok=True)
    clip_renames: dict[str, str] = {}
    used_names: set[str] | None = None

    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith("clips/"):
            continue
        rel = info.filename[len("clips/") :]
        posix_rel = PurePosixPath(rel)
        if (
            not rel
            or rel.startswith("/")
            or "\\" in rel
            or ":" in rel
            or any(part in ("", ".", "..") for part in posix_rel.parts)
            or ((info.external_attr >> 16) & _FILE_TYPE_MASK) == _SYMLINK_MODE
        ):
            raise H3Error(f"Unsafe archive path: {info.filename!r}")
        target = clips_root.joinpath(*posix_rel.parts)
        if not Path(os.path.abspath(target)).is_relative_to(clips_root):
            raise H3Error(f"Unsafe archive path: {info.filename!r}")
        if target.is_file() and _same_file_as_zip_member(archive, info, target):
            continue
        if target.exists() and conflict != "overwrite" and len(posix_rel.parts) == 1:
            if used_names is None:
                used_names = {entry.name for entry in clips_root.iterdir()}
            orig_file = posix_rel.name
            stem = Path(orig_file).stem
            suffix = Path(orig_file).suffix
            new_stem = _unique_name(stem, {Path(n).stem for n in used_names if Path(n).suffix == suffix}, style="stem")
            new_file = f"{new_stem}{suffix}"
            used_names.add(new_file)
            clip_renames[orig_file] = new_file
            target = clips_root / new_file
        _extract_member_atomic(archive, info, target)

    if clip_renames:
        for slot in range(DRAMA3_VOICE_SLOTS):
            field = f"voice_{slot}_audio"
            val = payload.get(field)
            if isinstance(val, str) and val in clip_renames:
                payload[field] = clip_renames[val]
        for slot in range(DRAMA3_SOUND_SLOTS):
            field = f"sound_{slot}_audio"
            val = payload.get(field)
            if isinstance(val, str) and val in clip_renames:
                payload[field] = clip_renames[val]
        bgm = payload.get("background_music")
        if isinstance(bgm, str) and bgm in clip_renames:
            payload["background_music"] = clip_renames[bgm]
        applied = payload.get("applied_music")
        if isinstance(applied, dict):
            abgm = applied.get("background_music")
            if isinstance(abgm, str) and abgm in clip_renames:
                applied["background_music"] = clip_renames[abgm]


def _restore_referenced_sounds(entries: list[dict[str, Any]], archive: zipfile.ZipFile, runtime) -> None:
    if not entries:
        return
    from h3_app.drama3 import sounds as drama3_sounds

    names_in_archive = set(archive.namelist())
    pending: list[dict[str, Any]] = []
    for entry in entries:
        sound_name = str(entry.get("name") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not sound_name or audio_file not in names_in_archive:
            continue
        if drama3_sounds.find_sound(sound_name, runtime) is not None:
            continue
        pending.append(entry)
    if not pending:
        return
    with TemporaryDirectory(prefix="h3-import-sounds-") as tmp:
        tmp_root = Path(tmp)
        for index, entry in enumerate(pending):
            audio_file = str(entry.get("audio_file") or "")
            suffix = Path(audio_file).suffix.lower() or ".wav"
            clip = tmp_root / f"sound_{index}{suffix}"
            try:
                _extract_member_direct(archive, audio_file, clip)
                drama3_sounds.save_sound(
                    str(entry.get("name") or ""),
                    clip,
                    str(entry.get("description") or ""),
                    int(entry.get("duration") or 5),
                    runtime,
                )
            except (H3Error, KeyError, OSError, ValueError):
                continue
            finally:
                clip.unlink(missing_ok=True)


def _import_soundstage(
    archive: zipfile.ZipFile,
    *,
    store,
    runtime,
    conflict: str,
    source_id: str,
    source_name: str,
    deferred_media: list[tuple[Path, dict[str, Any] | None]] | None = None,
) -> dict[str, Any]:
    from h3_app.drama3.serving.takes import takes_root
    from h3_ui.drama3 import visuals

    data = json.loads(archive.read("project.json"))
    orig_name = str(data.get("name") or source_name or source_id)
    payload = data["payload"]
    result_name = data.get("result_name")
    illustrated_name = data.get("illustrated_video_name")

    existing = {name for name, _ in store.soundstage_projects()}
    target_name, status, skipped = _resolve_named_conflict("soundstage", source_id, orig_name, existing, conflict)
    if skipped is not None:
        return skipped

    _extract_soundstage_clips(archive, _soundstage_clips_dir(runtime), payload, conflict=conflict)
    _restore_referenced_speakers(data.get("referenced_speakers") or [], archive, runtime)
    _restore_referenced_sounds(data.get("referenced_sounds") or [], archive, runtime)
    _extract_prefix(archive, "takes", takes_root(runtime), overwrite=False)

    names_in_archive = set(archive.namelist())
    media_to_register: list[tuple[Path, dict[str, Any] | None]] = []
    target_result: Path | None = None

    with _StagingTransaction() as tx:
        if result_name and f"result/{result_name}" in names_in_archive:
            output_dir = Path(runtime.output_dir)
            h3_dir = (output_dir / "h3").resolve()
            h3_dir.mkdir(parents=True, exist_ok=True)
            result_path_obj = Path(result_name)
            orig_stem = result_path_obj.stem
            target_stem = orig_stem
            target_result_name = result_name
            drama3_dir = output_dir / "drama3"
            stills_root = output_dir / ".soundstage" / "stills"
            if conflict == "rename" and (
                (h3_dir / result_name).exists()
                or (drama3_dir / result_name).exists()
                or visuals.stills_dir(h3_dir / result_name, runtime.output_dir).exists()
            ):
                used_stems = {p.stem for p in h3_dir.iterdir() if p.is_file()}
                if drama3_dir.is_dir():
                    used_stems.update(p.stem for p in drama3_dir.iterdir() if p.is_file())
                if stills_root.is_dir():
                    used_stems.update(p.name for p in stills_root.iterdir() if p.is_dir())
                target_stem = _unique_name(orig_stem, used_stems, style="stem")
                target_result_name = f"{target_stem}{result_path_obj.suffix}"
            target_result = h3_dir / target_result_name
            staged_result = tx.stage_file(target_result)
            _extract_member_direct(archive, f"result/{result_name}", staged_result)

            snap_data = None
            snap_arc = f"result/{result_name}.settings.json"
            if snap_arc in names_in_archive:
                target_snap = snapshot_path(target_result)
                staged_snap = tx.stage_file(target_snap)
                _extract_member_direct(archive, snap_arc, staged_snap)
                snap_data = _update_snapshot_file(
                    staged_snap, output_name=target_result.name, target_media=target_result
                )
            media_to_register.append((target_result, snap_data or {"family": "Drama 3"}))

            if illustrated_name and f"result/{illustrated_name}" in names_in_archive:
                target_vid_name = (
                    target_stem + illustrated_name[len(orig_stem) :]
                    if target_stem != orig_stem and illustrated_name.startswith(orig_stem)
                    else illustrated_name
                )
                target_vid = h3_dir / target_vid_name
                staged_vid = tx.stage_file(target_vid)
                _extract_member_direct(archive, f"result/{illustrated_name}", staged_vid)
                vsnap_data = None
                vsnap_arc = f"result/{illustrated_name}.settings.json"
                if vsnap_arc in names_in_archive:
                    target_vsnap = snapshot_path(target_vid)
                    staged_vsnap = tx.stage_file(target_vsnap)
                    _extract_member_direct(archive, vsnap_arc, staged_vsnap)
                    vsnap_data = _update_snapshot_file(
                        staged_vsnap,
                        output_name=target_vid.name,
                        target_media=target_vid,
                        source_name=target_result.name,
                    )
                media_to_register.append(
                    (target_vid, vsnap_data or {"family": "Sound Stage illustrated video"})
                )

            plan_arc = f"result/.{orig_stem}.visual_plan.json"
            if plan_arc in names_in_archive:
                staged_plan = tx.stage_file(visuals.plan_path(target_result))
                _extract_member_direct(archive, plan_arc, staged_plan)

            stills_dir = visuals.stills_dir(target_result, runtime.output_dir)
            has_stills = any(name.startswith("stills/") and not name.endswith("/") for name in names_in_archive)
            if has_stills or stills_dir.exists():
                staged_stills = tx.stage_dir(stills_dir, create=has_stills)
                if has_stills:
                    _extract_prefix(archive, "stills", staged_stills, overwrite=True, atomic=False)
                    if target_result.name != result_name:
                        _update_stills_source(staged_stills, target_result.name, visuals.STILLS_RECORD)
        else:
            plan_audio = _soundstage_plan_audio(runtime, target_name)
            if "plan/visual_plan.json" in names_in_archive:
                staged_plan = tx.stage_file(visuals.plan_path(plan_audio))
                _extract_member_direct(archive, "plan/visual_plan.json", staged_plan)
            plan_stills_dir = visuals.stills_dir(plan_audio, runtime.output_dir)
            has_plan_stills = any(
                name.startswith("plan_stills/") and not name.endswith("/") for name in names_in_archive
            )
            if has_plan_stills or (status == "overwritten" and plan_stills_dir.exists()):
                staged_plan_stills = tx.stage_dir(plan_stills_dir, create=has_plan_stills)
                if has_plan_stills:
                    _extract_prefix(archive, "plan_stills", staged_plan_stills, overwrite=True, atomic=False)
                    _update_stills_source(staged_plan_stills, plan_audio.name, visuals.STILLS_RECORD)

        def finalize_soundstage() -> None:
            store.save_soundstage_project(target_name, payload)
            if target_result is not None:
                store.mark_soundstage_generated(target_name, str(target_result))
            elif status == "overwritten":
                with store.connect() as db:
                    db.execute("UPDATE soundstage_projects SET result=NULL WHERE name=?", (target_name,))

        tx.commit(finalize_soundstage)

    if media_to_register:
        if deferred_media is not None:
            deferred_media.extend(media_to_register)
        else:
            _register_imported_media_batch(media_to_register, store=store, runtime=runtime)

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

    cast_dir = (store.root / "storyboards" / target_id / "cast").resolve()
    takes_dir = (Path(runtime.output_dir) / "h3" / ".storyboard" / target_id).resolve()
    media_dir = takes_dir / "media"

    with _StagingTransaction() as tx:
        staging_cast = tx.stage_dir(cast_dir)
        staging_board_dir = tx.stage_dir(takes_dir)
        staging_media = staging_board_dir / "media"
        staging_media.mkdir(parents=True, exist_ok=True)

        raw_cast = _extract_prefix(archive, "cast", staging_cast, overwrite=True, atomic=False)
        raw_media = _extract_prefix(archive, "media", staging_media, overwrite=True, atomic=False)
        raw_takes = _extract_prefix(archive, "takes", staging_board_dir, overwrite=True, atomic=False)

        extracted_cast = _retarget_extracted(raw_cast, staging_cast, cast_dir)
        extracted_media = _retarget_extracted(raw_media, staging_media, media_dir)
        extracted_takes = _retarget_extracted(raw_takes, staging_board_dir, takes_dir)

        for arc, staged_path in raw_takes.items():
            if staged_path.name.endswith(".settings.json"):
                media_name = staged_path.name[: -len(".settings.json")]
                _update_snapshot_file(
                    staged_path,
                    output_name=media_name,
                    target_media=extracted_takes[arc].with_name(media_name),
                )

        arc_to_abs = {
            **{k: str(v) for k, v in extracted_cast.items()},
            **{k: str(v) for k, v in extracted_media.items()},
        }
        take_abs = {k: str(v) for k, v in extracted_takes.items()}
        take_rel = {k: f"h3/.storyboard/{target_id}/{v.name}" for k, v in extracted_takes.items()}

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

        tx.commit(lambda: store.save_storyboard(owner, target_id, board))

    return _transfer_result("storyboard", orig_id, orig_name, target_id, target_name, status)


def _import_dubbing(
    archive: zipfile.ZipFile,
    *,
    store,
    runtime,
    conflict: str,
    source_id: str,
    source_name: str,
    deferred_media: list[tuple[Path, dict[str, Any] | None]] | None = None,
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
    media_to_register: list[tuple[Path, dict[str, Any] | None]] = []

    with _StagingTransaction() as tx:
        tx.track_created_dir(session_root)
        _extract_prefix(archive, "session", session_root, overwrite=True, atomic=False)

        manifest_path = session_root / sessions.MANIFEST
        try:
            session_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise H3Error("Imported dubbing archive is missing a valid session.json.") from exc
        if not isinstance(session_data, dict):
            raise H3Error("Imported dubbing archive is missing a valid session.json.")
        session_data["id"] = new_session_id
        session_data["updated"] = time.time()

        names_in_archive = set(archive.namelist())
        last_output_name = data.get("last_output_name")
        result_entries = sorted(
            name for name in names_in_archive if name.startswith("result/") and not name.endswith("/")
        )
        dub_video: Path | None = None
        player_dir: Path | None = None
        written_media: list[Path] = []
        snapshots_by_media: dict[Path, dict[str, Any]] = {}

        if last_output_name and result_entries:
            dub_out_dir = (Path(runtime.outputs_dir) / "dubbing").resolve()
            dub_out_dir.mkdir(parents=True, exist_ok=True)
            orig_stem = str(data.get("last_output_stem") or Path(last_output_name).stem)
            target_stem = orig_stem
            if conflict == "rename" and any(
                (dub_out_dir / candidate).exists()
                for candidate in (last_output_name, f"{orig_stem}.mp3", f"{orig_stem}.mp4", f"{orig_stem}.lipsync.mp4")
            ):
                used_stems = {p.stem for p in dub_out_dir.iterdir() if p.is_file()}
                target_stem = _unique_name(orig_stem, used_stems, style="stem")

            staged_by_dest: dict[Path, Path] = {}
            for arc_name in result_entries:
                fname = arc_name[len("result/") :]
                if not fname or "/" in fname or "\\" in fname or fname in {".", ".."}:
                    raise H3Error(f"Unsafe archive path: {arc_name!r}")
                if target_stem != orig_stem and fname.startswith(orig_stem):
                    target_fname = target_stem + fname[len(orig_stem) :]
                else:
                    target_fname = fname
                dest = (dub_out_dir / target_fname).resolve()
                staged = tx.stage_file(dest)
                _extract_member_direct(archive, arc_name, staged)
                staged_by_dest[dest] = staged
                if dest.name.endswith(".settings.json"):
                    media_name = dest.name[: -len(".settings.json")]
                    target_media = dub_out_dir / media_name
                    snap_data = _update_snapshot_file(
                        staged,
                        output_name=media_name,
                        target_media=target_media,
                        session_id=new_session_id,
                    )
                    if snap_data is not None:
                        snapshots_by_media[target_media] = snap_data
                else:
                    written_media.append(dest)

            player_dir = (dub_out_dir / dub_results.PLAYER_DIR / target_stem).resolve()
            has_player = any(name.startswith("result_player/") and not name.endswith("/") for name in names_in_archive)
            if has_player or player_dir.exists():
                staged_player = tx.stage_dir(player_dir, create=has_player)
                if has_player:
                    _extract_prefix(archive, "result_player", staged_player, overwrite=True, atomic=False)
                    if target_stem != orig_stem:
                        for meta_name in (dub_results.PLAYER_FILE, "alignment.json"):
                            meta_path = staged_player / meta_name
                            if not meta_path.is_file():
                                continue
                            try:
                                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError):
                                continue
                            if isinstance(meta, dict):
                                container = (
                                    meta.get("player")
                                    if meta_name == "alignment.json" and isinstance(meta.get("player"), dict)
                                    else meta
                                )
                                for field in ("video", "dub"):
                                    val = container.get(field)
                                    if isinstance(val, str) and Path(val.replace("\\", "/")).name.startswith(orig_stem):
                                        fname = Path(val.replace("\\", "/")).name
                                        container[field] = str(dub_out_dir / (target_stem + fname[len(orig_stem) :]))
                                write_json_atomic(meta_path, meta)

                    from h3_app.dubbing.lip_sync import REPORT_FILE

                    report_path = staged_player / REPORT_FILE
                    if target_stem != orig_stem and report_path.is_file():
                        try:
                            report = json.loads(report_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            report = None
                        if (
                            isinstance(report, dict)
                            and isinstance(report.get("output"), str)
                            and report["output"].startswith(orig_stem)
                        ):
                            report["output"] = target_stem + report["output"][len(orig_stem) :]
                            write_json_atomic(report_path, report)

            dub_video = dub_out_dir / f"{target_stem}.mp4"
            target_last_name = (
                target_stem + last_output_name[len(orig_stem) :]
                if target_stem != orig_stem and last_output_name.startswith(orig_stem)
                else last_output_name
            )
            session_data["last_output"] = str(dub_out_dir / target_last_name)
        else:
            session_data.pop("last_output", None)

        write_json_atomic(manifest_path, session_data)
        tx.commit(
            lambda: dubbing_projects.save_imported(
                runtime, target_name, new_session_id, data.get("settings") or {}
            )
        )

    if player_dir is not None and dub_video is not None and dub_video.is_file():
        from h3_app.dubbing.lip_sync import PLAN_FILE
        from h3_app.dubbing.lip_sync_match import video_fingerprint

        plan_path = player_dir / PLAN_FILE
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                plan = None
            if isinstance(plan, dict):
                stat = dub_video.stat()
                plan["video"] = dub_video.name
                plan["video_mtime_ns"] = stat.st_mtime_ns
                plan["video_size"] = stat.st_size
                plan["video_fingerprint"] = video_fingerprint(dub_video)
                write_json_atomic(plan_path, plan)

    for media_path in written_media:
        snap = snapshots_by_media.get(media_path) or read_snapshot(media_path) or {"family": dub_results.DUB_FAMILY}
        media_to_register.append((media_path, snap))
    if media_to_register:
        if deferred_media is not None:
            deferred_media.extend(media_to_register)
        else:
            _register_imported_media_batch(media_to_register, store=store, runtime=runtime)

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
            target_folder = Path(existing["folder"]).resolve()
            status = "overwritten"
        else:
            target_key = f"{orig_key[:12]}-{uuid.uuid4().hex[:8]}"
            while target_key in existing_keys:
                target_key = f"{orig_key[:12]}-{uuid.uuid4().hex[:8]}"
            target_name = _unique_name(orig_name, existing_names)
            target_owner = owner
            target_folder = projects.folder(target_owner, target_key).resolve()
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
        target_folder = projects.folder(target_owner, target_key).resolve()

    takes_final = target_folder / "takes"
    with _StagingTransaction() as tx:
        staging_folder = tx.stage_dir(target_folder)
        staging_takes = staging_folder / "takes"
        staging_takes.mkdir(parents=True, exist_ok=True)

        raw_folder = _extract_prefix(archive, "folder", staging_folder, overwrite=True, atomic=False)
        raw_takes = _extract_prefix(archive, "takes", staging_takes, overwrite=True, atomic=False)

        extracted_folder = _retarget_extracted(raw_folder, staging_folder, target_folder)
        extracted_takes = _retarget_extracted(raw_takes, staging_takes, takes_final)
        arc_map = {
            **{k: str(v) for k, v in extracted_folder.items()},
            **{k: str(v) for k, v in extracted_takes.items()},
        }

        for arc, staged_path in raw_takes.items():
            if staged_path.name.endswith(".settings.json"):
                media_name = staged_path.name[: -len(".settings.json")]
                _update_snapshot_file(
                    staged_path,
                    output_name=media_name,
                    target_media=extracted_takes[arc].with_name(media_name),
                )

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

        staged_project_record = {**project, "folder": str(staging_folder)}
        RemakeProjects.write(staged_project_record)
        tx.commit()

    return _transfer_result("remake", orig_key, orig_name, target_key, target_name, status)


def import_project_archive(
    archive_path: Path,
    *,
    store,
    runtime,
    conflict: str = "rename",
    owner: str = "imported",
    deferred_media: list[tuple[Path, dict[str, Any] | None]] | None = None,
) -> dict[str, Any]:
    """Validate and import a project archive into the local store and runtime."""
    if conflict not in CONFLICT_MODES:
        raise H3Error(f"Unsupported conflict mode: {conflict!r}")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise H3Error("Project archive file was not found.")
    with _PROJECT_TRANSFER_LOCK, zipfile.ZipFile(archive_path, "r") as archive:
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
                archive,
                store=store,
                runtime=runtime,
                conflict=conflict,
                source_id=source_id,
                source_name=source_name,
                deferred_media=deferred_media,
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
                archive,
                store=store,
                runtime=runtime,
                conflict=conflict,
                source_id=source_id,
                source_name=source_name,
                deferred_media=deferred_media,
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


def _retry_sleep_seconds(attempt: int, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                parsed = float(retry_after)
                if 0 < parsed <= 10.0:
                    return parsed
            except ValueError:
                pass
    return min(4.0, 0.25 * (2**attempt))


def fetch_remote_projects(
    base_url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    """Query a remote Prism Studio or Workshop server for its exportable projects."""
    normalized = normalize_server_url(base_url)
    response: httpx.Response | None = None
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        for attempt in range(_HTTP_MAX_ATTEMPTS):
            try:
                response = client.get(f"{normalized}/api/projects/transfer/list", headers=headers or {})
            except httpx.RequestError as exc:
                if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                    time.sleep(_retry_sleep_seconds(attempt))
                    continue
                raise H3Error(f"Could not reach {normalized}: {exc}") from exc
            if response.status_code in _TRANSIENT_HTTP_STATUSES and attempt + 1 < _HTTP_MAX_ATTEMPTS:
                time.sleep(_retry_sleep_seconds(attempt, response))
                continue
            break
    assert response is not None
    if response.status_code >= 400:
        raise _translate_http_error(response, normalized)
    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise H3Error("Remote server did not return a valid Prism project list.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        raise H3Error("Remote server returned an unexpected project list format.")
    return data


def _normalize_selection_item(raw: Any) -> tuple[str, str, str | None]:
    if isinstance(raw, dict):
        kind = str(raw.get("kind") or "")
        project_id = str(raw.get("id") or "")
        name = str(raw["name"]) if raw.get("name") else None
        return kind, project_id, name
    if isinstance(raw, (tuple, list)):
        if len(raw) >= 3:
            return str(raw[0]), str(raw[1]), (str(raw[2]) if raw[2] else None)
        if len(raw) == 2:
            return str(raw[0]), str(raw[1]), None
    raise H3Error("Each selection item must specify kind and id.")


def _precheck_skip_conflict(
    kind: str, project_id: str, remote_name: str | None, *, store, runtime
) -> dict[str, Any] | None:
    """Return an immediate skip result when ``conflict == 'skip'`` can be resolved without downloading."""
    with _PROJECT_TRANSFER_LOCK:
        if kind == "soundstage":
            if store is None:
                return None
            existing = {name for name, _ in store.soundstage_projects()}
            if project_id in existing:
                return _transfer_result("soundstage", project_id, project_id, project_id, project_id, "skipped")
            return None
        if kind == "dubbing":
            from h3_app.dubbing import projects as dubbing_projects

            if dubbing_projects.project_mtime_ns(runtime, project_id) is not None:
                return _transfer_result("dubbing", project_id, project_id, project_id, project_id, "skipped")
            return None
        if kind == "storyboard":
            if store is None:
                return None
            rows = store.storyboards()
            by_id = {str(row["id"]): row for row in rows}
            by_name = {str(row["name"]): row for row in rows}
            match = by_id.get(project_id) or (by_name.get(remote_name) if remote_name else None)
            if match is not None:
                orig_name = remote_name or str(match["name"])
                return _transfer_result(
                    "storyboard", project_id, orig_name, str(match["id"]), str(match["name"]), "skipped"
                )
            return None
        if kind == "remake":
            from h3_app.remake.project import Projects as RemakeProjects

            existing_remake = RemakeProjects(runtime.output_dir).load(None, project_id)
            if existing_remake is not None:
                target_name = str(existing_remake.get("name") or remote_name or project_id)
                orig_name = remote_name or target_name
                return _transfer_result(
                    "remake", project_id, orig_name, str(existing_remake["key"]), target_name, "skipped"
                )
            return None
    return None


def _parse_content_range_start(header_value: str | None) -> int | None:
    if not header_value:
        return None
    match = re.match(r"^bytes\s+(\d+)-\d+/(\d+|\*)$", header_value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _download_project_archive(
    client: httpx.Client,
    normalized_url: str,
    kind: str,
    project_id: str,
    archive_path: Path,
    *,
    headers: dict[str, str] | None = None,
    on_bytes: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream one exported project archive with transient retry and HTTP Range resume."""
    base_headers = dict(headers or {})
    downloaded = 0
    total_bytes = 0
    etag: str | None = None
    used_range_resume = False

    for attempt in range(_HTTP_MAX_ATTEMPTS):
        req_headers = dict(base_headers)
        if downloaded > 0 and archive_path.is_file():
            req_headers["Range"] = f"bytes={downloaded}-"
            if etag:
                req_headers["If-Range"] = etag

        try:
            with client.stream(
                "GET",
                f"{normalized_url}/api/projects/transfer/export",
                params={"kind": kind, "id": project_id},
                headers=req_headers,
            ) as response:
                status = response.status_code
                if status == 416 and downloaded > 0:
                    archive_path.unlink(missing_ok=True)
                    downloaded = 0
                    total_bytes = 0
                    etag = None
                    continue
                if status in _TRANSIENT_HTTP_STATUSES and attempt + 1 < _HTTP_MAX_ATTEMPTS:
                    response.read()
                    time.sleep(_retry_sleep_seconds(attempt, response))
                    continue
                if status >= 400:
                    response.read()
                    raise _translate_http_error(response, normalized_url)

                range_start = _parse_content_range_start(response.headers.get("content-range"))
                if status == 206 and downloaded > 0 and range_start == downloaded:
                    mode = "ab"
                    used_range_resume = True
                    if total_bytes <= 0:
                        try:
                            rem = int(response.headers.get("content-length") or 0)
                            if rem > 0:
                                total_bytes = downloaded + rem
                        except ValueError:
                            pass
                else:
                    mode = "wb"
                    downloaded = 0
                    etag = response.headers.get("etag") or response.headers.get("last-modified")
                    try:
                        total_bytes = int(response.headers.get("content-length") or 0)
                    except ValueError:
                        total_bytes = 0

                with archive_path.open(mode, buffering=_IO_CHUNK_BYTES) as handle:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if on_bytes is not None:
                            on_bytes(downloaded, total_bytes)

            if total_bytes > 0 and downloaded < total_bytes:
                if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                    time.sleep(_retry_sleep_seconds(attempt))
                    continue
                raise H3Error(
                    f"Incomplete archive download ({downloaded / (1 << 20):.1f} of {total_bytes / (1 << 20):.1f} MB)."
                )

            if not zipfile.is_zipfile(archive_path):
                archive_path.unlink(missing_ok=True)
                downloaded = 0
                total_bytes = 0
                etag = None
                if used_range_resume and attempt + 1 < _HTTP_MAX_ATTEMPTS:
                    used_range_resume = False
                    continue
                raise H3Error("Downloaded project archive is corrupted or incomplete.")

            return archive_path
        except httpx.RequestError as exc:
            if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                time.sleep(_retry_sleep_seconds(attempt))
                continue
            raise H3Error(f"Transfer connection error from {normalized_url}: {exc}") from exc

    raise H3Error(f"Could not download {KIND_LABELS.get(kind, kind)} project {project_id!r}.")


def import_remote_projects(
    base_url: str,
    selections: Sequence[tuple[str, str] | tuple[str, str, str | None] | dict[str, Any]],
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
    items = [_normalize_selection_item(entry) for entry in selections]
    if not items:
        raise H3Error("Select at least one project to import.")

    total_items = len(items)
    ordered_results: list[dict[str, Any] | None] = [None] * total_items
    pending_indices: list[int] = []

    if conflict == "skip":
        for index, (kind, project_id, remote_name) in enumerate(items):
            if kind in PROJECT_KINDS:
                skipped = _precheck_skip_conflict(kind, project_id, remote_name, store=store, runtime=runtime)
                if skipped is not None:
                    ordered_results[index] = skipped
                    continue
            pending_indices.append(index)
    else:
        pending_indices = list(range(total_items))

    if not pending_indices:
        if progress is not None:
            progress(1.0, "Import complete.")
        return [res for res in ordered_results if res is not None]

    progress_lock = threading.Lock()
    item_units: list[float] = [1.0 if ordered_results[i] is not None else 0.0 for i in range(total_items)]
    max_fraction = 0.0
    last_report_at = 0.0

    def emit_progress(desc: str, *, force: bool = False) -> None:
        nonlocal max_fraction, last_report_at
        if progress is None:
            return
        with progress_lock:
            now = time.monotonic()
            if not force and now - last_report_at < 0.25:
                return
            last_report_at = now
            raw = 0.94 * (sum(item_units) / total_items)
            max_fraction = max(max_fraction, min(0.94, raw))
            frac = max_fraction
        progress(frac, desc)

    changed = False
    deferred_media: list[tuple[Path, dict[str, Any] | None]] = []
    http_timeout = httpx.Timeout(connect=30.0, read=max(float(timeout), 600.0), write=60.0, pool=30.0)
    http_limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    workers = min(_MAX_PARALLEL_DOWNLOADS, len(pending_indices))

    try:
        with (
            TemporaryDirectory(prefix="h3-project-import-") as tmp_dir,
            httpx.Client(follow_redirects=True, timeout=http_timeout, limits=http_limits) as client,
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h3-transfer-dl") as pool,
        ):
            tmp_root = Path(tmp_dir)

            def download_task(item_idx: int) -> Path:
                kind, project_id, remote_name = items[item_idx]
                label = KIND_LABELS.get(kind, kind)
                display = remote_name or project_id
                archive_path = tmp_root / f"transfer_{item_idx}.zip"
                emit_progress(f"Downloading {label} project {display}…", force=item_idx == pending_indices[0])

                def on_bytes(done_bytes: int, expected_bytes: int) -> None:
                    frac = (done_bytes / expected_bytes) if expected_bytes > 0 else 0.35
                    with progress_lock:
                        item_units[item_idx] = max(item_units[item_idx], 0.5 * min(1.0, frac))
                    mb_done = done_bytes / (1 << 20)
                    size_note = (
                        f"{mb_done:.1f} / {expected_bytes / (1 << 20):.1f} MB"
                        if expected_bytes > 0
                        else f"{mb_done:.1f} MB"
                    )
                    emit_progress(f"Downloading {label} project {display}… ({size_note})")

                return _download_project_archive(
                    client,
                    normalized,
                    kind,
                    project_id,
                    archive_path,
                    headers=headers,
                    on_bytes=on_bytes,
                )

            in_flight: dict[int, Future[Path]] = {}
            next_submit_pos = 0

            def refill_window() -> None:
                nonlocal next_submit_pos
                while len(in_flight) < workers and next_submit_pos < len(pending_indices):
                    idx = pending_indices[next_submit_pos]
                    next_submit_pos += 1
                    in_flight[idx] = pool.submit(download_task, idx)

            refill_window()

            for item_idx in pending_indices:
                kind, project_id, remote_name = items[item_idx]
                label = KIND_LABELS.get(kind, kind)
                display = remote_name or project_id
                archive_path = tmp_root / f"transfer_{item_idx}.zip"
                future = in_flight.pop(item_idx)
                try:
                    downloaded_path = future.result()
                    with progress_lock:
                        item_units[item_idx] = max(item_units[item_idx], 0.55)
                    emit_progress(f"Importing {label} project {display}…", force=True)
                    result = import_project_archive(
                        downloaded_path,
                        store=store,
                        runtime=runtime,
                        conflict=conflict,
                        owner=owner,
                        deferred_media=deferred_media,
                    )
                    if result["status"] in {"imported", "renamed", "overwritten"}:
                        changed = True
                    ordered_results[item_idx] = result
                except (H3Error, httpx.RequestError, zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
                    ordered_results[item_idx] = {
                        "kind": kind,
                        "source_id": project_id,
                        "source_name": display,
                        "target_id": None,
                        "target_name": None,
                        "status": "failed",
                        "message": f'Failed {label} "{display}": {exc}',
                    }
                finally:
                    archive_path.unlink(missing_ok=True)
                    with progress_lock:
                        item_units[item_idx] = 1.0
                    refill_window()
    finally:
        if deferred_media:
            if progress is not None:
                progress(0.94, "Indexing imported media…")
            _register_imported_media_batch(deferred_media, store=store, runtime=runtime)
        if changed and on_imported is not None:
            if progress is not None:
                progress(0.97, "Saving imported projects to workspace…")
            on_imported()

    if progress is not None:
        progress(1.0, "Import complete.")
    return [res for res in ordered_results if res is not None]
