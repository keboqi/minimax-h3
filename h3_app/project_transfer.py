"""Portable cross-server project export, import, and remote transfer client."""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import re
import shutil
import stat as stat_mod
import threading
import time
import uuid
import zipfile
import zlib
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from itertools import count
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Mapping, Sequence
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
_TRANSIENT_FS_SUFFIXES = (".tmp", ".partial", ".importing", ".replacing")
_DIGEST_CACHE_MAX_ENTRIES = 16384
_DIGEST_WORKERS = 4
_INLINE_JSON_MAX_BYTES = 4 << 20
_MAX_METADATA_BYTES = 64 << 20
MAX_KNOWN_DIGESTS = 50000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRANSFER_HEADER = "X-Prism-Transfer"


class _MissingReusedFileError(H3Error):
    """Raised when a delta archive references a local file that vanished or mutated mid-transfer."""


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and bool(_SHA256_RE.fullmatch(value))


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= (1 << 30):
        return f"{num_bytes / (1 << 30):.2f} GB"
    if num_bytes >= (1 << 20):
        return f"{num_bytes / (1 << 20):.1f} MB"
    if num_bytes >= (1 << 10):
        return f"{num_bytes / (1 << 10):.0f} KB"
    return f"{num_bytes} B"


class _FileDigestCache:
    """Bounded thread-safe SHA-256 cache keyed by ``(resolved_path, mtime_ns, size)``."""

    def __init__(self, max_entries: int = _DIGEST_CACHE_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, int, int], str] = OrderedDict()

    def get_sha256(
        self,
        path: Path,
        *,
        mtime_ns: int | None = None,
        size: int | None = None,
    ) -> str | None:
        try:
            if path.is_symlink():
                return None
            resolved = path.resolve()
            if mtime_ns is None or size is None:
                st = os.lstat(resolved)
                if not stat_mod.S_ISREG(st.st_mode):
                    return None
                mtime_ns = st.st_mtime_ns
                size = st.st_size
        except OSError:
            return None

        key = (str(resolved), mtime_ns, size)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit

        for _attempt in range(2):
            try:
                before = os.lstat(resolved)
                if not stat_mod.S_ISREG(before.st_mode):
                    return None
                hasher = hashlib.sha256()
                read_total = 0
                with resolved.open("rb") as reader:
                    while chunk := reader.read(_IO_CHUNK_BYTES):
                        hasher.update(chunk)
                        read_total += len(chunk)
                after = os.lstat(resolved)
            except OSError:
                return None
            digest = hasher.hexdigest()
            if (
                before.st_mtime_ns == after.st_mtime_ns
                and before.st_size == after.st_size == read_total
            ):
                stable_key = (str(resolved), after.st_mtime_ns, after.st_size)
                with self._lock:
                    self._entries[stable_key] = digest
                    self._entries.move_to_end(stable_key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                return digest
        return None

    def get_many(self, items: Sequence[tuple[Path, int | None, int | None]]) -> dict[Path, str]:
        if not items:
            return {}
        if len(items) == 1:
            p, mt, sz = items[0]
            digest = self.get_sha256(p, mtime_ns=mt, size=sz)
            return {p: digest} if digest is not None else {}
        results: dict[Path, str] = {}
        pending: list[tuple[Path, int | None, int | None]] = []
        for p, mt, sz in items:
            try:
                if p.is_symlink():
                    continue
                resolved = p.resolve()
                if mt is None or sz is None:
                    st = os.lstat(resolved)
                    if not stat_mod.S_ISREG(st.st_mode):
                        continue
                    mt, sz = st.st_mtime_ns, st.st_size
            except OSError:
                continue
            key = (str(resolved), mt, sz)
            with self._lock:
                hit = self._entries.get(key)
                if hit is not None:
                    self._entries.move_to_end(key)
                    results[p] = hit
                    continue
            pending.append((p, mt, sz))
        if not pending:
            return results
        if len(pending) == 1:
            p, mt, sz = pending[0]
            digest = self.get_sha256(p, mtime_ns=mt, size=sz)
            if digest is not None:
                results[p] = digest
            return results
        workers = min(_DIGEST_WORKERS, len(pending))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h3-transfer-hash") as pool:
            digests = list(pool.map(lambda item: self.get_sha256(item[0], mtime_ns=item[1], size=item[2]), pending))
        for (p, _mt, _sz), digest in zip(pending, digests):
            if digest is not None:
                results[p] = digest
        return results


_DIGEST_CACHE = _FileDigestCache()


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


def _link_or_copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    """Materialize ``source`` at ``destination`` via hardlink when possible, else verified copy."""
    destination.unlink(missing_ok=True)
    try:
        if source.is_symlink():
            raise _MissingReusedFileError(str(source))
        resolved = source.resolve()
        st = os.lstat(resolved)
        if not stat_mod.S_ISREG(st.st_mode):
            raise _MissingReusedFileError(str(source))
        if expected_size is not None and st.st_size != expected_size:
            raise _MissingReusedFileError(str(source))
        if expected_sha256 is not None:
            actual = _DIGEST_CACHE.get_sha256(resolved, mtime_ns=st.st_mtime_ns, size=st.st_size)
            if actual != expected_sha256:
                raise _MissingReusedFileError(str(source))
    except OSError as exc:
        raise _MissingReusedFileError(str(source)) from exc

    try:
        os.link(resolved, destination)
        return
    except OSError:
        pass

    try:
        hasher = hashlib.sha256() if expected_sha256 is not None else None
        written = 0
        with resolved.open("rb") as reader, destination.open("wb") as writer:
            while chunk := reader.read(_IO_CHUNK_BYTES):
                if hasher is not None:
                    hasher.update(chunk)
                writer.write(chunk)
                written += len(chunk)
        if expected_size is not None and written != expected_size:
            destination.unlink(missing_ok=True)
            raise _MissingReusedFileError(str(source))
        if hasher is not None and hasher.hexdigest() != expected_sha256:
            destination.unlink(missing_ok=True)
            raise _MissingReusedFileError(str(source))
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise _MissingReusedFileError(str(source)) from exc


def _same_file_as_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, existing: Path) -> bool:
    """Check whether ``existing`` already matches ``info`` byte-for-byte without writing."""
    try:
        if existing.is_symlink():
            return False
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
            if os.name == "nt" and entry.stat(follow_symlinks=False).st_file_attributes & stat_mod.FILE_ATTRIBUTE_REPARSE_POINT:
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


def _archive_filename(value: str) -> str:
    if not isinstance(value, str) or "/" in value:
        raise H3Error(f"Unsafe archive filename: {value!r}")
    _validate_arc_entry_name(value)
    return value


def _contained_path(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    root = root.absolute()
    relative = path.absolute().relative_to(root)
    if not relative.parts or ".." in relative.parts or resolved != root.resolve() / relative:
        raise H3Error(f"Unsafe import destination: {path.name!r}")
    return resolved


def _validate_arc_entry_name(name: str) -> PurePosixPath:
    if any(ord(c) < 32 or c in '\\:' for c in name):
        raise H3Error(f"Unsafe archive entry: {name!r}")
    for part in name.split("/"):
        if part in {"", ".", ".."} or (os.name == "nt" and (
            part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved() or any(c in '<>"|?*' for c in part)
        )):
            raise H3Error(f"Unsafe archive entry: {name!r}")
    return PurePosixPath(name)


def _validate_archive(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        if ((info.external_attr >> 16) & _FILE_TYPE_MASK) == _SYMLINK_MODE:
            raise H3Error(f"Unsafe archive entry: {info.filename!r}")
        _validate_arc_entry_name(info.filename.removesuffix("/") if info.is_dir() else info.filename)


class _StagingTransaction:
    """Stage extracted files, obsolete deletions, and newly created directories with atomic rollback."""

    def __init__(self) -> None:
        self._file_swaps: list[tuple[Path, Path]] = []
        self._file_deletes: list[Path] = []
        self._tracked_dirs: list[Path] = []
        self._created_parents: list[Path] = []
        self._created_parent_set: set[Path] = set()
        self._prune_roots: list[tuple[Path, bool]] = []
        self._committed = False
        self.deleted_paths: list[Path] = []

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_paths)

    def __enter__(self) -> _StagingTransaction:
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if self._committed:
            return
        for staging, _target in self._file_swaps:
            staging.unlink(missing_ok=True)
        for folder in self._tracked_dirs:
            shutil.rmtree(folder, ignore_errors=True)
        for parent in reversed(self._created_parents):
            try:
                parent.rmdir()
            except OSError:
                pass

    def _ensure_parent_dir(self, directory: Path) -> None:
        resolved = directory.resolve()
        missing: list[Path] = []
        cur = resolved
        while not cur.exists():
            missing.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
        for folder in reversed(missing):
            folder.mkdir(parents=True, exist_ok=True)
            if folder not in self._created_parent_set:
                self._created_parent_set.add(folder)
                self._created_parents.append(folder)

    def stage_file(self, target_file: Path) -> Path:
        target = target_file.resolve()
        self._ensure_parent_dir(target.parent)
        staging = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.importing"
        self._file_swaps.append((staging, target))
        return staging

    def stage_delete_file(self, target_file: Path) -> None:
        self._file_deletes.append(_contained_path(target_file.parent, target_file))

    def prune_empty_dirs_under(self, root_dir: Path, *, remove_root: bool = False) -> None:
        self._prune_roots.append((root_dir.resolve(), remove_root))

    def track_created_dir(self, folder: Path) -> Path:
        resolved = folder.resolve()
        self._tracked_dirs.append(resolved)
        return resolved

    def commit(self, finalize: Callable[[], None] | None = None) -> None:
        file_backups: list[tuple[Path, Path, bool]] = []
        delete_backups: list[tuple[Path, Path]] = []
        swapped_targets = {target for _staging, target in self._file_swaps}
        seen_deletes: set[Path] = set()
        effective_deletes: list[Path] = []
        for candidate in self._file_deletes:
            if candidate in swapped_targets or candidate in seen_deletes:
                continue
            seen_deletes.add(candidate)
            effective_deletes.append(candidate)

        try:
            for staging, target in self._file_swaps:
                if not staging.is_file():
                    continue
                had_target = target.is_file()
                backup = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.replacing"
                if had_target:
                    os.replace(target, backup)
                os.replace(staging, target)
                file_backups.append((backup, target, had_target))

            for target in effective_deletes:
                if not target.is_file() and not target.is_symlink():
                    continue
                backup = target.parent / f".{target.name}.{uuid.uuid4().hex[:8]}.replacing"
                os.replace(target, backup)
                delete_backups.append((backup, target))

            if finalize is not None:
                finalize()
        except BaseException:
            for backup, target in reversed(delete_backups):
                try:
                    if backup.exists() and not target.exists():
                        os.replace(backup, target)
                except OSError:
                    pass
                finally:
                    backup.unlink(missing_ok=True)
            for backup, target, had_target in reversed(file_backups):
                try:
                    target.unlink(missing_ok=True)
                    if had_target and backup.is_file():
                        os.replace(backup, target)
                except OSError:
                    pass
                finally:
                    backup.unlink(missing_ok=True)
            raise

        self.deleted_paths = [target for _backup, target in delete_backups]
        for backup, _target in delete_backups:
            backup.unlink(missing_ok=True)
        for backup, _target, _had_target in file_backups:
            backup.unlink(missing_ok=True)
        for prune_root, remove_root in self._prune_roots:
            if not prune_root.is_dir():
                continue
            for dirpath, _dirnames, _filenames in os.walk(prune_root, topdown=False):
                current_dir = Path(dirpath)
                if current_dir.resolve() != current_dir or (current_dir == prune_root and not remove_root):
                    continue
                try:
                    current_dir.rmdir()
                except OSError:
                    pass
        self._committed = True


class _ArchiveSource:
    """Unified reader across full and delta project archives, reusing verified local files."""

    def __init__(
        self,
        archive: zipfile.ZipFile,
        manifest: dict[str, Any],
        local_files: dict[str, list[Path]] | None = None,
    ) -> None:
        self.archive = archive
        self.is_delta = bool(manifest.get("delta"))
        self.local_files = local_files or {}
        self._zip_infos: dict[str, zipfile.ZipInfo] = {
            info.filename: info for info in archive.infolist() if not info.is_dir()
        }
        self.reused_specs: dict[str, tuple[str, int]] = {}
        raw_reused = manifest.get("reused_files")
        if raw_reused is not None:
            if not isinstance(raw_reused, dict):
                raise H3Error("Invalid project archive delta manifest.")
            for raw_arc, spec in raw_reused.items():
                arc_name = str(raw_arc)
                _validate_arc_entry_name(arc_name)
                if not isinstance(spec, dict):
                    raise H3Error(f"Invalid delta entry specification for {arc_name!r}.")
                sha = str(spec.get("sha256") or "").lower()
                try:
                    size = int(spec.get("size", -1))
                except (TypeError, ValueError) as exc:
                    raise H3Error(f"Invalid delta entry size for {arc_name!r}.") from exc
                if not _is_sha256_hex(sha) or size < 0:
                    raise H3Error(f"Invalid delta entry digest for {arc_name!r}.")
                if arc_name not in self._zip_infos:
                    self.reused_specs[arc_name] = (sha, size)

        self.reused_count = 0
        self.reused_bytes = 0
        self.transferred_content_count = 0
        self.transferred_bytes = 0

    @staticmethod
    def _is_content_asset(arc_name: str, size: int) -> bool:
        return not (arc_name.lower().endswith(".json") and size <= _INLINE_JSON_MAX_BYTES)

    def _entry_size(self, arc_name: str) -> int:
        if arc_name in self._zip_infos:
            return self._zip_infos[arc_name].file_size
        return self.reused_specs[arc_name][1]

    def _record_reuse(self, arc_name: str, size: int) -> None:
        if self._is_content_asset(arc_name, size):
            self.reused_count += 1
            self.reused_bytes += size

    def _record_transfer(self, arc_name: str, size: int) -> None:
        self.transferred_bytes += size
        if self._is_content_asset(arc_name, size):
            self.transferred_content_count += 1

    def all_entries(self) -> set[str]:
        return set(self._zip_infos) | set(self.reused_specs)

    def has_entry(self, arc_name: str) -> bool:
        return arc_name in self._zip_infos or arc_name in self.reused_specs

    def prefix_entries(self, prefix: str) -> list[str]:
        head = prefix.rstrip("/") + "/"
        return sorted(name for name in self.all_entries() if name.startswith(head))

    def read_bytes(self, arc_name: str) -> bytes:
        if self._entry_size(arc_name) > _MAX_METADATA_BYTES:
            raise H3Error(f"Archive metadata is too large: {arc_name!r}")
        if arc_name in self._zip_infos:
            return self.archive.read(arc_name)
        if arc_name in self.reused_specs:
            sha, size = self.reused_specs[arc_name]
            donor = self._find_donor(sha, size)
            with donor.open("rb") as stream:
                data = stream.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise _MissingReusedFileError(arc_name)
            return data
        raise KeyError(arc_name)

    def _find_donor(self, expected_sha: str, expected_size: int, preferred: Path | None = None) -> Path:
        candidates: list[Path] = []
        if preferred is not None:
            candidates.append(preferred)
        candidates.extend(self.local_files.get(expected_sha, ()))
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                if candidate.is_symlink():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                st = os.lstat(resolved)
                if not stat_mod.S_ISREG(st.st_mode) or st.st_size != expected_size:
                    continue
                if _DIGEST_CACHE.get_sha256(resolved, mtime_ns=st.st_mtime_ns, size=st.st_size) == expected_sha:
                    return resolved
            except OSError:
                continue
        raise _MissingReusedFileError(expected_sha)

    def matches_existing(self, arc_name: str, target: Path) -> tuple[bool, int]:
        """Check whether ``target`` on disk already matches ``arc_name`` byte-for-byte."""
        if not target.is_file() or target.is_symlink():
            return False, 0
        if arc_name in self.reused_specs:
            sha, size = self.reused_specs[arc_name]
            try:
                st = os.lstat(target)
                if (
                    stat_mod.S_ISREG(st.st_mode)
                    and st.st_size == size
                    and _DIGEST_CACHE.get_sha256(target, mtime_ns=st.st_mtime_ns, size=st.st_size) == sha
                ):
                    return True, size
            except OSError:
                pass
            return False, size
        info = self._zip_infos[arc_name]
        return _same_file_as_zip_member(self.archive, info, target), info.file_size

    def materialize_direct(self, arc_name: str, destination: Path) -> bool:
        """Write ``arc_name`` directly to ``destination`` (inside a newly created/temp folder)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if arc_name in self.reused_specs:
            sha, size = self.reused_specs[arc_name]
            donor = self._find_donor(sha, size)
            _link_or_copy_verified(donor, destination, expected_sha256=sha, expected_size=size)
            self._record_reuse(arc_name, size)
            return True
        info = self._zip_infos[arc_name]
        _extract_member_direct(self.archive, info, destination)
        self._record_transfer(arc_name, info.file_size)
        return False

    def stage_entry(
        self,
        tx: _StagingTransaction,
        arc_name: str,
        destination: Path,
        *,
        allow_in_place: bool | None = None,
    ) -> tuple[Path, bool]:
        """Stage ``arc_name`` for ``destination``; returns ``(work_path, reused_unchanged)``."""
        target = _contained_path(destination.parent, destination)
        if allow_in_place is None:
            allow_in_place = target.suffix.lower() != ".json"

        if allow_in_place:
            same, size = self.matches_existing(arc_name, target)
            if same:
                self._record_reuse(arc_name, size)
                return target, True

        staged = tx.stage_file(target)
        if arc_name in self.reused_specs:
            sha, size = self.reused_specs[arc_name]
            donor = self._find_donor(sha, size, preferred=target if target.exists() else None)
            _link_or_copy_verified(donor, staged, expected_sha256=sha, expected_size=size)
            self._record_reuse(arc_name, size)
            return staged, True

        info = self._zip_infos[arc_name]
        _extract_member_direct(self.archive, info, staged)
        self._record_transfer(arc_name, info.file_size)
        return staged, False

    def stage_prefix(
        self,
        tx: _StagingTransaction,
        prefix: str,
        dest_dir: Path,
        *,
        overwrite: bool = True,
        direct_into_new_dir: bool = False,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Stage all entries under ``prefix/`` into ``dest_dir``.

        Returns ``(work_paths_by_arc, final_paths_by_arc)``.
        """
        dest_root = dest_dir.resolve()
        if direct_into_new_dir:
            dest_root.mkdir(parents=True, exist_ok=True)
        head = prefix.rstrip("/") + "/"
        work_paths: dict[str, Path] = {}
        final_paths: dict[str, Path] = {}
        for arc_name in self.prefix_entries(prefix):
            rel = arc_name[len(head) :]
            posix_rel = _validate_arc_entry_name(rel)
            target = _contained_path(dest_root, dest_root.joinpath(*posix_rel.parts))
            if not overwrite and target.exists():
                self._record_reuse(arc_name, self._entry_size(arc_name))
                work_paths[arc_name] = target
                final_paths[arc_name] = target
                continue
            if direct_into_new_dir:
                self.materialize_direct(arc_name, target)
                work_paths[arc_name] = target
            else:
                staged, _reused = self.stage_entry(tx, arc_name, target)
                work_paths[arc_name] = staged
            final_paths[arc_name] = target
        return work_paths, final_paths

    def extract_prefix_shared(self, prefix: str, dest_dir: Path, *, overwrite: bool = False) -> dict[str, Path]:
        """Extract entries under ``prefix/`` into a shared folder atomically without removing sibling files."""
        dest_root = dest_dir.resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        head = prefix.rstrip("/") + "/"
        extracted: dict[str, Path] = {}
        for arc_name in self.prefix_entries(prefix):
            rel = arc_name[len(head) :]
            posix_rel = _validate_arc_entry_name(rel)
            target = _contained_path(dest_root, dest_root.joinpath(*posix_rel.parts))
            if not overwrite and target.exists():
                self._record_reuse(arc_name, self._entry_size(arc_name))
                extracted[arc_name] = target
                continue
            same, size = self.matches_existing(arc_name, target)
            if same:
                self._record_reuse(arc_name, size)
                extracted[arc_name] = target
                continue
            if arc_name in self.reused_specs:
                sha, size = self.reused_specs[arc_name]
                donor = self._find_donor(sha, size)
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.partial")
                try:
                    _link_or_copy_verified(donor, temp, expected_sha256=sha, expected_size=size)
                    os.replace(temp, target)
                except BaseException:
                    temp.unlink(missing_ok=True)
                    raise
                self._record_reuse(arc_name, size)
            else:
                info = self._zip_infos[arc_name]
                _extract_member_atomic(self.archive, info, target)
                self._record_transfer(arc_name, info.file_size)
            extracted[arc_name] = target
        return extracted


def _schedule_obsolete_deletes(
    tx: _StagingTransaction,
    root_dir: Path,
    keep_files: set[Path],
    *,
    skip_rel: Callable[[PurePosixPath], bool] | None = None,
    remove_root_if_empty: bool = False,
) -> None:
    resolved_root = root_dir.resolve()
    if not resolved_root.is_dir():
        return
    resolved_keep = {path.resolve() for path in keep_files}
    for existing_path, _rel in _iter_scandir_files(resolved_root, skip_rel=skip_rel):
        resolved_existing = existing_path.resolve()
        if resolved_existing not in resolved_keep:
            tx.stage_delete_file(resolved_existing)
    tx.prune_empty_dirs_under(resolved_root, remove_root=remove_root_if_empty)


def _is_media_already_cataloged(media_path: Path, store) -> bool:
    if store is None or not media_path.is_file():
        return False
    try:
        resolved = media_path.resolve()
        asset_id = uuid.uuid5(uuid.NAMESPACE_URL, resolved.as_uri()).hex
        with store.connect() as db:
            row = db.execute(
                "SELECT 1 FROM assets a JOIN media_catalog c ON c.asset_id=a.id WHERE a.id=? AND a.available=1",
                (asset_id,),
            ).fetchone()
        return row is not None
    except OSError:
        return False


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
    *,
    src: _ArchiveSource | None = None,
    tx: _StagingTransaction | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = KIND_LABELS.get(kind, kind)
    reused_count = src.reused_count if src is not None else 0
    reused_bytes = src.reused_bytes if src is not None else 0
    transferred_count = src.transferred_content_count if src is not None else 0
    transferred_bytes = src.transferred_bytes if src is not None else 0
    removed_count = tx.deleted_count if tx is not None else 0

    if message is None:
        if status == "skipped":
            message = f'Skipped {label} "{source_name}" (already exists).'
        elif status == "up_to_date":
            message = f'{label} "{target_name}" is already up to date.'
        elif status == "overwritten" and (reused_count > 0 or (src is not None and src.is_delta and removed_count > 0)):
            if transferred_count > 0:
                detail = f"synced {_plural(transferred_count, 'changed file')}"
                if reused_count > 0:
                    detail += f", reused {_plural(reused_count, 'unchanged file')}"
            elif reused_count > 0:
                detail = f"metadata updated, reused {_plural(reused_count, 'unchanged file')}"
            else:
                detail = "metadata updated"
            if removed_count > 0:
                detail += f", removed {removed_count}"
            if reused_bytes >= (64 << 10):
                detail += f" · saved {_format_bytes(reused_bytes)}"
            message = f'Overwritten {label} "{target_name}" ({detail}).'
        else:
            message = f'{status.capitalize()} {label} "{target_name}".'

    payload: dict[str, Any] = {
        "kind": kind,
        "source_id": source_id,
        "source_name": source_name,
        "target_id": target_id,
        "target_name": target_name,
        "status": status,
        "message": message,
    }
    if status == "overwritten" and src is not None:
        payload.update(
            {
                "incremental": bool(src.is_delta or reused_count > 0),
                "transferred_files": transferred_count,
                "reused_files": reused_count,
                "removed_files": removed_count,
                "transferred_bytes": transferred_bytes,
                "reused_bytes": reused_bytes,
            }
        )
    if extra:
        payload.update(extra)
    return payload


def format_import_summary_headline(results: Sequence[Mapping[str, Any]]) -> str:
    """Format a concise status-count headline for a batch of project transfer results."""
    if not results:
        return "No projects were imported."
    counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "imported")
        counts[status] = counts.get(status, 0) + 1
    parts = [f"{count} {status.replace('_', ' ')}" for status, count in counts.items() if count]
    return f"Import finished ({', '.join(parts)})"


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


def _update_stills_record_file(stills_record: Path, source_name: str) -> None:
    if not stills_record.is_file():
        return
    try:
        rec = json.loads(stills_record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(rec, dict):
        rec["source"] = source_name
        write_json_atomic(stills_record, rec)


def _update_stills_source(stills_dir: Path, source_name: str, record_filename: str) -> None:
    _update_stills_record_file(stills_dir / record_filename, source_name)


def _restore_referenced_speakers(entries: list[dict[str, Any]], src: _ArchiveSource, runtime) -> None:
    if not entries:
        return
    from h3_app.drama3 import speakers as drama3_speakers

    pending: list[dict[str, Any]] = []
    for entry in entries:
        key = str(entry.get("key") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not key or not src.has_entry(audio_file):
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
                src.materialize_direct(audio_file, clip)
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


def serialize_project_transfer(func):
    @wraps(func)
    def locked(*args, **kwargs):
        with _PROJECT_TRANSFER_LOCK:
            return func(*args, **kwargs)
    return locked


@serialize_project_transfer
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


@dataclass
class _PlannedFile:
    arc_name: str
    source: Path
    size: int
    mtime_ns: int
    sha256: str = ""


@dataclass
class _PlannedBytes:
    arc_name: str
    data: bytes
    sha256: str


class _ExportPlan:
    """Deterministic virtual archive layout shared by full export, delta export, and local diffing."""

    def __init__(self, kind: str, project_id: str) -> None:
        self.kind = kind
        self.project_id = project_id
        self.display_name = project_id
        self.added: set[str] = set()
        self.files: list[_PlannedFile] = []
        self.inline: list[_PlannedBytes] = []
        self.portable_digest: str = ""

    def add_file(self, source: Path, arc_name: str) -> bool:
        if arc_name in self.added or not source.is_file() or source.is_symlink():
            return False
        try:
            resolved = source.resolve()
            st = os.lstat(resolved)
        except OSError:
            return False
        if not stat_mod.S_ISREG(st.st_mode):
            return False
        self.added.add(arc_name)
        self.files.append(
            _PlannedFile(
                arc_name=arc_name,
                source=resolved,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )
        )
        return True

    def add_dir(
        self,
        root_dir: Path,
        prefix: str,
        *,
        skip_rel: Callable[[PurePosixPath], bool] | None = None,
    ) -> None:
        if not root_dir.is_dir():
            return
        for path, rel in _iter_scandir_files(root_dir, skip_rel=skip_rel):
            self.add_file(path, f"{prefix}/{rel}")

    def writestr(self, arc_name: str, content: str | bytes) -> None:
        if arc_name in self.added:
            return
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        self.added.add(arc_name)
        self.inline.append(
            _PlannedBytes(
                arc_name=arc_name,
                data=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )

    @property
    def content_file_count(self) -> int:
        return sum(1 for entry in self.files if _ArchiveSource._is_content_asset(entry.arc_name, entry.size))

    @property
    def content_bytes(self) -> int:
        return sum(entry.size for entry in self.files if _ArchiveSource._is_content_asset(entry.arc_name, entry.size))

    @property
    def reusable_digests(self) -> set[str]:
        return {
            entry.sha256
            for entry in self.files
            if entry.sha256 and _ArchiveSource._is_content_asset(entry.arc_name, entry.size)
        }

    def populate_digests(self, precomputed: dict[Path, str] | None = None) -> None:
        digest_map = dict(precomputed or {})
        missing = [
            (entry.source, entry.mtime_ns, entry.size)
            for entry in self.files
            if entry.source not in digest_map
        ]
        if missing:
            digest_map.update(_DIGEST_CACHE.get_many(missing))
        pairs: list[tuple[str, str]] = []
        for entry in self.files:
            sha = digest_map.get(entry.source) or _DIGEST_CACHE.get_sha256(
                entry.source, mtime_ns=entry.mtime_ns, size=entry.size
            )
            if sha is None:
                raise H3Error(f"Project file disappeared during export: {entry.source.name}")
            entry.sha256 = sha
            if entry.arc_name.lower().endswith(".json") and entry.size <= _INLINE_JSON_MAX_BYTES:
                try:
                    raw = entry.source.read_bytes()
                except OSError as exc:
                    raise H3Error(f"Could not read {entry.source.name}: {exc}") from exc
                p_sha = _portable_json_digest(self.kind, entry.arc_name, raw, sha)
            else:
                p_sha = sha
            pairs.append((entry.arc_name, p_sha))
        for inline_entry in self.inline:
            if inline_entry.arc_name.lower().endswith(".json"):
                p_sha = _portable_json_digest(
                    self.kind, inline_entry.arc_name, inline_entry.data, inline_entry.sha256
                )
            else:
                p_sha = inline_entry.sha256
            pairs.append((inline_entry.arc_name, p_sha))
        pairs.sort(key=lambda item: item[0])
        joined = f"kind={self.kind}\n" + "\n".join(f"{arc}\t{digest}" for arc, digest in pairs)
        self.portable_digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _portable_json_digest(kind: str, arc_name: str, raw: bytes, fallback_sha256: str) -> str:
    """Normalize machine-local paths/IDs in small JSON metadata so identical projects compare equal."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return fallback_sha256
    if not isinstance(parsed, dict):
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    normalized = copy.deepcopy(parsed)
    lower_arc = arc_name.lower()
    if lower_arc.endswith(".settings.json"):
        normalized.pop("asset_id", None)
        if kind == "dubbing" and isinstance(normalized.get("settings"), dict):
            normalized["settings"].pop("session", None)
    elif kind == "storyboard" and arc_name == "board.json":
        normalized["id"] = ""
    elif kind == "remake" and arc_name == "project.json":
        normalized["owner"] = ""
    elif kind == "dubbing":
        if arc_name == "project.json":
            normalized["session_id"] = ""
        elif arc_name == "session/session.json":
            normalized["id"] = ""
            normalized["updated"] = 0
            last_out = normalized.get("last_output")
            if isinstance(last_out, str) and last_out:
                normalized["last_output"] = PurePosixPath(last_out.replace("\\", "/")).name
        elif arc_name in {"result_player/player.json", "result_player/alignment.json"}:
            container = (
                normalized.get("player")
                if arc_name.endswith("alignment.json") and isinstance(normalized.get("player"), dict)
                else normalized
            )
            if isinstance(container, dict):
                for field_name in ("video", "dub"):
                    val = container.get(field_name)
                    if isinstance(val, str) and val:
                        container[field_name] = PurePosixPath(val.replace("\\", "/")).name
        elif arc_name == "result_player/lip_sync_plan.json":
            normalized["video_mtime_ns"] = 0
    elif kind == "soundstage" and lower_arc.endswith("/.stills.json"):
        pass

    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class _LocalOverwriteState:
    local_id: str
    local_name: str
    portable_digest: str
    files_by_sha256: dict[str, list[Path]] = field(default_factory=dict)
    content_file_count: int = 0
    content_bytes: int = 0

    @property
    def known_digests(self) -> set[str]:
        return set(self.files_by_sha256)


def _write_canonical_effect_manifests(
    plan_out: _ExportPlan,
    stills_folder: Path,
    *,
    audio_path: Path,
    output_dir: Path,
    duration: float,
    payload: dict[str, Any],
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
        if arc_name in plan_out.added:
            continue
        regions = effect_regions.read_regions(stills_folder, image_name, labels)
        if regions is not None:
            plan_out.writestr(arc_name, json.dumps(regions, ensure_ascii=False))


def _export_soundstage(plan_out: _ExportPlan, project_id: str, *, store, runtime) -> str:
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
        plan_out.add_file(resolved, f"clips/{stored_name}")
        clip_map[resolved] = stored_name
        return stored_name

    speaker_keys: list[str] = []
    for slot in range(DRAMA3_VOICE_SLOTS):
        field_name = f"voice_{slot}_audio"
        if payload.get(field_name):
            payload[field_name] = archive_clip(payload[field_name])
        preset = payload.get(f"voice_{slot}_preset")
        if preset and preset != UPLOAD_VOICE:
            speaker_keys.append(str(preset))
    referenced_speakers, speaker_files = _collect_referenced_speakers(speaker_keys, runtime)
    for src, arc in speaker_files:
        plan_out.add_file(src, arc)

    referenced_sounds: list[dict[str, Any]] = []
    seen_sounds: set[str] = set()
    for slot in range(DRAMA3_SOUND_SLOTS):
        field_name = f"sound_{slot}_audio"
        if payload.get(field_name):
            payload[field_name] = archive_clip(payload[field_name])
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
        plan_out.add_file(saved.audio, arc_name)
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
        plan_out.add_file(result_path, f"result/{result_name}")
        snap = snapshot_path(result_path)
        plan_out.add_file(snap, f"result/{snap.name}")
        plan = visuals.plan_path(result_path)
        plan_out.add_file(plan, f"result/{plan.name}")
        latest_video = _latest_soundstage_video(result_path)
        if latest_video is not None:
            illustrated_video_name = latest_video.name
            plan_out.add_file(latest_video, f"result/{illustrated_video_name}")
            vsnap = snapshot_path(latest_video)
            plan_out.add_file(vsnap, f"result/{vsnap.name}")
        stills_folder = visuals.stills_dir(result_path, runtime.output_dir)
        plan_out.add_dir(stills_folder, "stills")
        settings = result_settings(result_path)
        _write_canonical_effect_manifests(
            plan_out,
            stills_folder,
            audio_path=result_path,
            output_dir=Path(runtime.output_dir),
            duration=float(settings.get("duration") or 0),
            payload=payload,
        )
        takes_dir = takes_root(runtime)
        for chunk in settings.get("chunks") or []:
            take_key = chunk.get("take")
            if take_key and re.fullmatch(r"[A-Za-z0-9_-]+", str(take_key)):
                plan_out.add_file(takes_dir / f"{take_key}.wav", f"takes/{take_key}.wav")
    else:
        plan_audio = _soundstage_plan_audio(runtime, project_id)
        plan_out.add_file(visuals.plan_path(plan_audio), "plan/visual_plan.json")
        plan_out.add_dir(visuals.stills_dir(plan_audio, runtime.output_dir), "plan_stills")

    project_record: dict[str, Any] = {
        "name": project_id,
        "payload": payload,
        "result_name": result_name,
        "referenced_speakers": referenced_speakers,
        "referenced_sounds": referenced_sounds,
    }
    if illustrated_video_name is not None:
        project_record["illustrated_video_name"] = illustrated_video_name
    plan_out.writestr("project.json", json.dumps(project_record, ensure_ascii=False, indent=2))
    return project_id


def _resolve_storyboard_posters(
    videos: Sequence[tuple[Path, str]], *, runtime
) -> list[tuple[Path, str]]:
    """Return ``(poster_path, archive_name)`` pairs for exported storyboard videos.

    Uses already-cached posters from ``runtime.gallery_thumbnails_dir`` first, and generates
    any missing posters via ``gallery_thumbnail`` when FFmpeg is available (e.g. on Studio).
    """
    thumb_dir = getattr(runtime, "gallery_thumbnails_dir", None)
    if not videos or thumb_dir is None:
        return []
    from h3_app.gallery_store import gallery_thumbnail, gallery_thumbnail_path

    ready: dict[str, Path] = {}
    uncached: list[tuple[Path, str]] = []
    for video_path, video_arc in videos:
        try:
            video_mtime = video_path.stat().st_mtime
            cached = gallery_thumbnail_path(video_path, runtime=runtime)
            if cached.is_file() and cached.stat().st_mtime >= video_mtime:
                ready[video_arc] = cached
                continue
        except OSError:
            continue
        uncached.append((video_path, video_arc))

    if uncached:
        if len(uncached) == 1 or shutil.which("ffmpeg") is None:
            for video_path, video_arc in uncached:
                poster = gallery_thumbnail(video_path, runtime=runtime)
                if poster is not None and poster.is_file():
                    ready[video_arc] = poster
        else:
            workers = min(_MEDIA_PREPARE_WORKERS, len(uncached))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="h3-transfer-poster") as pool:
                posters = list(pool.map(lambda item: gallery_thumbnail(item[0], runtime=runtime), uncached))
            for (_video_path, video_arc), poster in zip(uncached, posters):
                if poster is not None and poster.is_file():
                    ready[video_arc] = poster

    return [
        (ready[video_arc], f"posters/{video_arc}.jpg")
        for _video_path, video_arc in videos
        if video_arc in ready
    ]


def _export_storyboard(plan_out: _ExportPlan, project_id: str, *, store, runtime) -> str:
    from h3_app.catalog import VIDEO_EXTENSIONS

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
    cast_name_by_member: dict[str, tuple[str, str]] = {}
    used_take_names: set[str] = set()
    take_map: dict[Path, str] = {}
    exported_videos: list[tuple[Path, str]] = []
    seen_video_arcs: set[str] = set()

    def record_exported_video(resolved: Path, arc: str) -> None:
        if arc not in seen_video_arcs and resolved.suffix.lower() in VIDEO_EXTENSIONS:
            seen_video_arcs.add(arc)
            exported_videos.append((resolved, arc))

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
        plan_out.add_file(resolved, arc)
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
                original_name = PurePosixPath(str(member["image"]).replace("\\", "/")).name
                archived = archive_asset(member["image"], "cast", used_cast_names, cast_map)
                if archived is not None:
                    member["image"] = archived
                    if record_active_cast and member.get("name"):
                        cast_name_by_member[str(member["name"])] = (original_name, PurePosixPath(archived).name)
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

    original_videos = {str(s["in_use"]): PurePosixPath(str(s["video"]).replace("\\", "/")).name
                       for s in board.get("scenes") or [] if s.get("in_use") and s.get("video")}
    rewrite_snapshot_lists(board, record_active_cast=True)
    for version in board.get("versions") or []:
        if isinstance(version, dict):
            rewrite_snapshot_lists(version)
    for resolved_media, media_arc in media_map.items():
        record_exported_video(resolved_media, media_arc)

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
                if key == "video":
                    record_exported_video(resolved, take_map[resolved])
                continue
            filename = path.name
            idx = 2
            while filename in used_take_names:
                filename = f"{path.stem}_{idx}{path.suffix}"
                idx += 1
            used_take_names.add(filename)
            arc = f"takes/{filename}"
            plan_out.add_file(resolved, arc)
            if key == "video":
                record_exported_video(resolved, arc)
                snap = snapshot_path(resolved)
                if snap.is_file():
                    plan_out.add_file(snap, f"takes/{filename}.settings.json")
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
                        and pair[1] == cast_name_by_member[pair[0]][0]
                    ):
                        pair[1] = cast_name_by_member[pair[0]][1]
            if (inputs.get("kind") == "media" and str(take_id) in scene_media_video_by_take
                    and inputs.get("video") == original_videos.get(str(take_id))):
                inputs["video"] = scene_media_video_by_take[str(take_id)]

    for poster_path, poster_arc in _resolve_storyboard_posters(exported_videos, runtime=runtime):
        plan_out.add_file(poster_path, poster_arc)

    plan_out.writestr("board.json", json.dumps({"id": project_id, "board": board}, ensure_ascii=False, indent=2))
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


def _export_dubbing(plan_out: _ExportPlan, project_id: str, *, runtime) -> str:
    from h3_app.dubbing import projects as dubbing_projects, results as dub_results

    data, session = dubbing_projects.load(runtime, project_id)
    plan_out.add_dir(session.root, "session", skip_rel=_skip_dubbing_session_rel)

    speaker_keys = [
        entry.get("voice")
        for entry in (session.data.get("speaker_settings") or {}).values()
        if isinstance(entry, dict) and entry.get("voice")
    ]
    referenced_speakers, speaker_files = _collect_referenced_speakers(speaker_keys, runtime)
    for src, arc in speaker_files:
        plan_out.add_file(src, arc)

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
            plan_out.add_file(file_path, f"result/{file_path.name}")
            snap = snapshot_path(file_path)
            plan_out.add_file(snap, f"result/{snap.name}")
        plan_out.add_dir(
            dub_results.player_folder(last_path), "result_player", skip_rel=_skip_dubbing_player_rel
        )

    plan_out.writestr(
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


def _export_remake(plan_out: _ExportPlan, project_id: str, *, runtime) -> str:
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
        prefix_tag = f"w{index}_"
        base_stem = resolved.stem if resolved.name.startswith(prefix_tag) else f"{prefix_tag}{resolved.stem}"
        arc_name = f"{base_stem}{resolved.suffix}"
        suffix_idx = 2
        while arc_name in used_take_names:
            arc_name = f"{base_stem}_{suffix_idx}{resolved.suffix}"
            suffix_idx += 1
        used_take_names.add(arc_name)
        plan_out.add_file(resolved, f"takes/{arc_name}")
        if snap.is_file():
            plan_out.add_file(snap, f"takes/{arc_name}.settings.json")
        window["take"] = f"takes/{arc_name}"

    if folder.is_dir():
        for path, rel in _iter_scandir_files(folder):
            if path.name == "project.json" or path.resolve() in skip_files:
                continue
            plan_out.add_file(path, f"folder/{rel}")

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
        plan_out.add_file(resolved, arc)
        return arc

    if project.get("source"):
        project["source"] = archive_folder_ref(project["source"], "folder")
    for window in project.get("windows") or []:
        if window.get("pictures"):
            window["pictures"] = [archive_folder_ref(pic, "folder/pictures") for pic in window["pictures"]]
        if window.get("carried"):
            window["carried"] = archive_folder_ref(window["carried"], "folder/pictures")

    cleaned = {k: v for k, v in project.items() if k not in {"folder", "updated"}}
    plan_out.writestr("project.json", json.dumps(cleaned, ensure_ascii=False, indent=2))
    return str(cleaned.get("name") or project_id)


@serialize_project_transfer
def build_project_export_plan(
    kind: str,
    project_id: str,
    *,
    store,
    runtime,
    compute_digests: bool = False,
) -> _ExportPlan:
    """Build a project's deterministic virtual archive layout, optionally hashing its entries."""
    if kind not in PROJECT_KINDS:
        raise H3Error(f"Unsupported project kind: {kind!r}")
    plan = _ExportPlan(kind, project_id)
    if kind == "soundstage":
        if store is None:
            raise H3Error("Workspace store is unavailable.")
        plan.display_name = _export_soundstage(plan, project_id, store=store, runtime=runtime)
    elif kind == "storyboard":
        if store is None:
            raise H3Error("Workspace store is unavailable.")
        plan.display_name = _export_storyboard(plan, project_id, store=store, runtime=runtime)
    elif kind == "dubbing":
        plan.display_name = _export_dubbing(plan, project_id, runtime=runtime)
    else:
        plan.display_name = _export_remake(plan, project_id, runtime=runtime)
    if compute_digests:
        plan.populate_digests()
    return plan


def _write_planned_archive(
    plan: _ExportPlan,
    destination: Path,
    *,
    delta: bool = False,
    known_digests: set[str] | None = None,
) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
    added: set[str] = set()
    reused_files: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "kind": plan.kind,
        "id": plan.project_id,
        "name": plan.display_name,
        "exported_at": time.time(),
    }
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as archive:
            for entry in plan.files:
                if (
                    delta
                    and known_digests
                    and _ArchiveSource._is_content_asset(entry.arc_name, entry.size)
                    and entry.sha256 in known_digests
                ):
                    reused_files[entry.arc_name] = {"sha256": entry.sha256, "size": entry.size}
                else:
                    _add_zip_file(archive, entry.source, entry.arc_name, added)
            for inline_entry in plan.inline:
                archive.writestr(inline_entry.arc_name, inline_entry.data)
            if delta:
                manifest["delta"] = True
                manifest["tree_digest"] = plan.portable_digest
                manifest["reused_files"] = reused_files
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@serialize_project_transfer
def export_project_archive(kind: str, project_id: str, destination: Path, *, store, runtime) -> Path:
    """Export one project and all of its media, sidecars, and presets into a portable zip archive."""
    plan = build_project_export_plan(kind, project_id, store=store, runtime=runtime, compute_digests=False)
    return _write_planned_archive(plan, destination)


@serialize_project_transfer
def export_project_delta_archive(
    kind: str,
    project_id: str,
    destination: Path,
    *,
    store,
    runtime,
    known_digests: Iterable[str] = (),
    plan: _ExportPlan | None = None,
) -> Path:
    """Export an incremental delta archive omitting content assets already present in ``known_digests``."""
    if plan is None:
        plan = build_project_export_plan(kind, project_id, store=store, runtime=runtime, compute_digests=True)
    elif not plan.portable_digest:
        plan.populate_digests()

    known = {str(d).lower() for d in known_digests if _is_sha256_hex(str(d).lower())}
    return _write_planned_archive(plan, destination, delta=True, known_digests=known)


def _find_existing_local_project(
    kind: str,
    project_id: str,
    remote_name: str | None,
    *,
    store,
    runtime,
    match_named_by_remote_name: bool = False,
) -> tuple[str, str] | None:
    """Locate an existing local project's ``(local_id, local_name)`` while holding ``_PROJECT_TRANSFER_LOCK``."""
    if kind == "soundstage":
        if store is None:
            return None
        existing = {name for name, _ in store.soundstage_projects()}
        if project_id in existing:
            return project_id, project_id
        if match_named_by_remote_name and remote_name and remote_name in existing:
            return remote_name, remote_name
        return None
    if kind == "storyboard":
        if store is None:
            return None
        rows = store.storyboards()
        by_id = {str(row["id"]): row for row in rows}
        by_name = {str(row["name"]): row for row in rows}
        match = by_id.get(project_id) or (by_name.get(remote_name) if remote_name else None)
        if match is None:
            return None
        return str(match["id"]), str(match["name"])
    if kind == "dubbing":
        from h3_app.dubbing import projects as dubbing_projects

        if dubbing_projects.project_mtime_ns(runtime, project_id) is not None:
            return project_id, project_id
        if (
            match_named_by_remote_name
            and remote_name
            and dubbing_projects.project_mtime_ns(runtime, remote_name) is not None
        ):
            return remote_name, remote_name
        return None
    if kind == "remake":
        from h3_app.remake.project import Projects as RemakeProjects

        existing_remake = RemakeProjects(runtime.output_dir).load(None, project_id)
        if existing_remake is None:
            return None
        local_id = str(existing_remake["key"])
        local_name = str(existing_remake.get("name") or remote_name or local_id)
        return local_id, local_name
    return None


def inspect_local_project_for_delta(
    kind: str,
    project_id: str,
    remote_name: str | None = None,
    *,
    store,
    runtime,
) -> _LocalOverwriteState | None:
    """Collect local file digests and portable fingerprint when an overwrite target already exists locally."""
    if kind not in PROJECT_KINDS:
        return None

    extra_candidates: list[tuple[Path, int, int]] = []
    with _PROJECT_TRANSFER_LOCK:
        found = _find_existing_local_project(
            kind,
            project_id,
            remote_name,
            store=store,
            runtime=runtime,
            match_named_by_remote_name=True,
        )
        if found is None:
            return None
        local_id, local_name = found
        try:
            plan = build_project_export_plan(
                kind, local_id, store=store, runtime=runtime, compute_digests=False
            )
        except H3Error:
            return None

        planned_paths = {entry.source for entry in plan.files}

        def add_extra_dir(folder: Path, limit: int = 1000) -> None:
            if not folder.is_dir():
                return
            scanned = 0
            for fpath, _rel in _iter_scandir_files(folder):
                if scanned >= limit:
                    break
                try:
                    resolved = fpath.resolve()
                    if resolved in planned_paths:
                        continue
                    st = os.lstat(resolved)
                    if stat_mod.S_ISREG(st.st_mode):
                        planned_paths.add(resolved)
                        extra_candidates.append((resolved, st.st_mtime_ns, st.st_size))
                        scanned += 1
                except OSError:
                    continue

        if kind == "storyboard" and store is not None:
            add_extra_dir(store.root / "storyboards" / local_id / "cast")
            add_extra_dir(Path(runtime.output_dir) / "h3" / ".storyboard" / local_id)
        elif kind == "soundstage":
            from h3_app.drama3.serving.takes import takes_root

            add_extra_dir(takes_root(runtime), limit=500)

    all_to_hash = [(entry.source, entry.mtime_ns, entry.size) for entry in plan.files] + extra_candidates
    digest_map = _DIGEST_CACHE.get_many(all_to_hash)
    try:
        plan.populate_digests(precomputed=digest_map)
    except H3Error:
        return None

    files_by_sha256: dict[str, list[Path]] = {}
    for entry in plan.files:
        if entry.sha256:
            files_by_sha256.setdefault(entry.sha256, []).append(entry.source)
    for extra_path, _mt, _sz in extra_candidates:
        sha = digest_map.get(extra_path)
        if sha:
            bucket = files_by_sha256.setdefault(sha, [])
            if extra_path not in bucket:
                bucket.append(extra_path)

    return _LocalOverwriteState(
        local_id=local_id,
        local_name=local_name or local_id,
        portable_digest=plan.portable_digest,
        files_by_sha256=files_by_sha256,
        content_file_count=plan.content_file_count,
        content_bytes=plan.content_bytes,
    )


def _stage_soundstage_clips(
    tx: _StagingTransaction,
    src: _ArchiveSource,
    clips_dir: Path,
    payload: dict[str, Any],
    *,
    conflict: str,
) -> None:
    """Stage clips for the shared Soundstage folder without clobbering unrelated projects."""
    from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS

    clips_root = clips_dir.resolve()
    clip_renames: dict[str, str] = {}
    used_names: set[str] | None = None

    for arc_name in src.prefix_entries("clips"):
        rel = arc_name[len("clips/") :]
        posix_rel = _validate_arc_entry_name(rel)
        target = _contained_path(clips_root, clips_root.joinpath(*posix_rel.parts))

        same, size = src.matches_existing(arc_name, target)
        if same:
            src._record_reuse(arc_name, size)
            continue

        if conflict != "overwrite" and len(posix_rel.parts) == 1 and (
            target.exists() or (used_names is not None and posix_rel.name in used_names)
        ):
            if used_names is None:
                used_names = (
                    {entry.name for entry in clips_root.iterdir() if not _is_transient_fs_name(entry.name)}
                    if clips_root.is_dir()
                    else set()
                )
            orig_file = posix_rel.name
            stem = Path(orig_file).stem
            suffix = Path(orig_file).suffix
            new_stem = _unique_name(
                stem, {Path(n).stem for n in used_names if Path(n).suffix == suffix}, style="stem"
            )
            new_file = f"{new_stem}{suffix}"
            used_names.add(new_file)
            clip_renames[orig_file] = new_file
            target = clips_root / new_file

        src.stage_entry(tx, arc_name, target, allow_in_place=False)

    if clip_renames:
        for slot in range(DRAMA3_VOICE_SLOTS):
            field_name = f"voice_{slot}_audio"
            val = payload.get(field_name)
            if isinstance(val, str) and val in clip_renames:
                payload[field_name] = clip_renames[val]
        for slot in range(DRAMA3_SOUND_SLOTS):
            field_name = f"sound_{slot}_audio"
            val = payload.get(field_name)
            if isinstance(val, str) and val in clip_renames:
                payload[field_name] = clip_renames[val]
        bgm = payload.get("background_music")
        if isinstance(bgm, str) and bgm in clip_renames:
            payload["background_music"] = clip_renames[bgm]
        applied = payload.get("applied_music")
        if isinstance(applied, dict):
            abgm = applied.get("background_music")
            if isinstance(abgm, str) and abgm in clip_renames:
                applied["background_music"] = clip_renames[abgm]


def _restore_referenced_sounds(entries: list[dict[str, Any]], src: _ArchiveSource, runtime) -> None:
    if not entries:
        return
    from h3_app.drama3 import sounds as drama3_sounds

    pending: list[dict[str, Any]] = []
    for entry in entries:
        sound_name = str(entry.get("name") or "")
        audio_file = str(entry.get("audio_file") or "")
        if not sound_name or not src.has_entry(audio_file):
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
                src.materialize_direct(audio_file, clip)
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
    src: _ArchiveSource,
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

    data = json.loads(src.read_bytes("project.json"))
    orig_name = str(data.get("name") or source_name or source_id)
    payload = data["payload"]
    result_name = data.get("result_name")
    illustrated_name = data.get("illustrated_video_name")
    for filename in (result_name, illustrated_name):
        if filename:
            _archive_filename(filename)

    existing = {name for name, _ in store.soundstage_projects()}
    target_name, status, skipped = _resolve_named_conflict("soundstage", source_id, orig_name, existing, conflict)
    if skipped is not None:
        return skipped

    media_to_register: list[tuple[Path, dict[str, Any] | None]] = []
    target_result: Path | None = None

    with _StagingTransaction() as tx:
        output_dir = Path(runtime.output_dir)
        clips_dir = _contained_path(output_dir, _soundstage_clips_dir(runtime))
        _stage_soundstage_clips(tx, src, clips_dir, payload, conflict=conflict)
        if result_name and src.has_entry(f"result/{result_name}"):
            h3_dir = _contained_path(output_dir, output_dir / "h3")
            h3_dir.mkdir(parents=True, exist_ok=True)
            result_path_obj = Path(result_name)
            orig_stem = _archive_filename(result_path_obj.stem)
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
            _staged_result, result_reused = src.stage_entry(tx, f"result/{result_name}", target_result)

            snap_data = None
            snap_arc = f"result/{result_name}.settings.json"
            if src.has_entry(snap_arc):
                target_snap = snapshot_path(target_result)
                staged_snap, _ = src.stage_entry(tx, snap_arc, target_snap, allow_in_place=False)
                snap_data = _update_snapshot_file(
                    staged_snap, output_name=target_result.name, target_media=target_result
                )
            if not (result_reused and _is_media_already_cataloged(target_result, store)):
                media_to_register.append((target_result, snap_data or {"family": "Drama 3"}))

            target_vid = None
            if illustrated_name and src.has_entry(f"result/{illustrated_name}"):
                target_vid_name = (
                    target_stem + illustrated_name[len(orig_stem) :]
                    if target_stem != orig_stem and illustrated_name.startswith(orig_stem)
                    else illustrated_name
                )
                target_vid = _contained_path(h3_dir, h3_dir / target_vid_name)
                _staged_vid, vid_reused = src.stage_entry(tx, f"result/{illustrated_name}", target_vid)
                vsnap_data = None
                vsnap_arc = f"result/{illustrated_name}.settings.json"
                if src.has_entry(vsnap_arc):
                    target_vsnap = snapshot_path(target_vid)
                    staged_vsnap, _ = src.stage_entry(tx, vsnap_arc, target_vsnap, allow_in_place=False)
                    vsnap_data = _update_snapshot_file(
                        staged_vsnap,
                        output_name=target_vid.name,
                        target_media=target_vid,
                        source_name=target_result.name,
                    )
                if not (vid_reused and _is_media_already_cataloged(target_vid, store)):
                    media_to_register.append(
                        (target_vid, vsnap_data or {"family": "Sound Stage illustrated video"})
                    )

            if status == "overwritten":
                for old in h3_dir.glob(f"{glob.escape(target_stem)}_illustrated_*.mp4"):
                    if old != target_vid and old.is_file() and not old.is_symlink():
                        tx.stage_delete_file(old)
                        sidecar = snapshot_path(old)
                        if sidecar.is_file() and not sidecar.is_symlink():
                            tx.stage_delete_file(sidecar)

            plan_arc = f"result/.{orig_stem}.visual_plan.json"
            target_plan_path = visuals.plan_path(target_result)
            if src.has_entry(plan_arc):
                src.stage_entry(tx, plan_arc, target_plan_path, allow_in_place=False)
            elif status == "overwritten" and target_plan_path.is_file():
                tx.stage_delete_file(target_plan_path)

            stills_dir = _contained_path(output_dir, visuals.stills_dir(target_result, output_dir))
            stills_entries = src.prefix_entries("stills")
            has_stills = bool(stills_entries)
            if has_stills or stills_dir.exists():
                stills_new = not stills_dir.exists()
                if stills_new and has_stills:
                    tx.track_created_dir(stills_dir)
                work_stills, final_stills = src.stage_prefix(
                    tx, "stills", stills_dir, direct_into_new_dir=stills_new and has_stills
                )
                if has_stills and target_result.name != result_name:
                    rec_arc = f"stills/{visuals.STILLS_RECORD}"
                    if rec_arc in work_stills:
                        _update_stills_record_file(work_stills[rec_arc], target_result.name)
                if not stills_new:
                    _schedule_obsolete_deletes(
                        tx, stills_dir, set(final_stills.values()), remove_root_if_empty=not has_stills
                    )

            if status == "overwritten":
                draft_plan_audio = _soundstage_plan_audio(runtime, target_name)
                draft_plan_file = visuals.plan_path(draft_plan_audio)
                if draft_plan_file.is_file():
                    tx.stage_delete_file(draft_plan_file)
                draft_stills_dir = _contained_path(output_dir, visuals.stills_dir(draft_plan_audio, output_dir))
                if draft_stills_dir.is_dir():
                    _schedule_obsolete_deletes(tx, draft_stills_dir, set(), remove_root_if_empty=True)
        else:
            plan_audio = _soundstage_plan_audio(runtime, target_name)
            draft_plan_path = visuals.plan_path(plan_audio)
            if src.has_entry("plan/visual_plan.json"):
                src.stage_entry(tx, "plan/visual_plan.json", draft_plan_path, allow_in_place=False)
            elif status == "overwritten" and draft_plan_path.is_file():
                tx.stage_delete_file(draft_plan_path)
            plan_stills_dir = _contained_path(output_dir, visuals.stills_dir(plan_audio, output_dir))
            has_plan_stills = bool(src.prefix_entries("plan_stills"))
            if has_plan_stills or (status == "overwritten" and plan_stills_dir.exists()):
                pstills_new = not plan_stills_dir.exists()
                if pstills_new and has_plan_stills:
                    tx.track_created_dir(plan_stills_dir)
                work_pstills, final_pstills = src.stage_prefix(
                    tx, "plan_stills", plan_stills_dir, direct_into_new_dir=pstills_new and has_plan_stills
                )
                if has_plan_stills:
                    rec_arc = f"plan_stills/{visuals.STILLS_RECORD}"
                    if rec_arc in work_pstills:
                        _update_stills_record_file(work_pstills[rec_arc], plan_audio.name)
                if not pstills_new:
                    _schedule_obsolete_deletes(
                        tx, plan_stills_dir, set(final_pstills.values()), remove_root_if_empty=not has_plan_stills
                    )

        _restore_referenced_speakers(data.get("referenced_speakers") or [], src, runtime)
        _restore_referenced_sounds(data.get("referenced_sounds") or [], src, runtime)
        src.extract_prefix_shared("takes", takes_root(runtime), overwrite=False)

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

    for removed in tx.deleted_paths:
        if target_result and removed.parent == target_result.parent and removed.suffix.lower() == ".mp4":
            store.forget_asset(removed)
    return _transfer_result("soundstage", source_id, orig_name, target_name, target_name, status, src=src, tx=tx)


def _import_storyboard(
    src: _ArchiveSource, *, store, runtime, conflict: str, owner: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.storyboard import model

    payload = json.loads(src.read_bytes("board.json"))
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

    board_meta_dir = _contained_path(store.root, store.root / "storyboards" / target_id)
    cast_dir = _contained_path(board_meta_dir, board_meta_dir / "cast")
    output_root = Path(runtime.output_dir)
    takes_dir = _contained_path(output_root, output_root / "h3" / ".storyboard" / target_id)
    media_dir = _contained_path(takes_dir, takes_dir / "media")

    with _StagingTransaction() as tx:
        cast_is_new = not cast_dir.exists()
        takes_is_new = not takes_dir.exists()
        if cast_is_new:
            tx.track_created_dir(board_meta_dir if not board_meta_dir.exists() else cast_dir)
        if takes_is_new:
            tx.track_created_dir(takes_dir)

        _work_cast, extracted_cast = src.stage_prefix(tx, "cast", cast_dir, direct_into_new_dir=cast_is_new)
        _work_media, extracted_media = src.stage_prefix(tx, "media", media_dir, direct_into_new_dir=takes_is_new)
        work_takes, extracted_takes = src.stage_prefix(tx, "takes", takes_dir, direct_into_new_dir=takes_is_new)

        imported_posters: list[tuple[Path, Path]] = []
        thumb_dir = getattr(runtime, "gallery_thumbnails_dir", None)
        if thumb_dir is not None:
            from h3_app.gallery_store import gallery_thumbnail_path

            thumb_root = Path(thumb_dir)
            video_targets = {**extracted_media, **extracted_takes}
            seen_posters: set[Path] = set()
            for poster_arc in src.prefix_entries("posters"):
                if not poster_arc.endswith(".jpg"):
                    continue
                video_arc = poster_arc[len("posters/") : -len(".jpg")]
                target_video = video_targets.get(video_arc)
                if target_video is None:
                    continue
                dest_poster = _contained_path(
                    thumb_root,
                    gallery_thumbnail_path(target_video, runtime=runtime),
                )
                if dest_poster in seen_posters:
                    continue
                seen_posters.add(dest_poster)
                src.stage_entry(tx, poster_arc, dest_poster)
                imported_posters.append((dest_poster, target_video))

        if not cast_is_new:
            _schedule_obsolete_deletes(tx, cast_dir, set(extracted_cast.values()))
        if not takes_is_new:
            _schedule_obsolete_deletes(
                tx, takes_dir, set(extracted_media.values()) | set(extracted_takes.values())
            )

        for arc, staged_path in work_takes.items():
            final_path = extracted_takes[arc]
            if final_path.name.endswith(".settings.json"):
                media_name = final_path.name[: -len(".settings.json")]
                _update_snapshot_file(
                    staged_path,
                    output_name=media_name,
                    target_media=final_path.with_name(media_name),
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

    for dest_poster, target_video in imported_posters:
        try:
            if dest_poster.is_file() and target_video.is_file():
                video_mtime_ns = target_video.stat().st_mtime_ns
                if dest_poster.stat().st_mtime_ns < video_mtime_ns:
                    target_ns = max(time.time_ns(), video_mtime_ns)
                    os.utime(dest_poster, ns=(target_ns, target_ns))
        except OSError:
            pass

    return _transfer_result("storyboard", orig_id, orig_name, target_id, target_name, status, src=src, tx=tx)


def _import_dubbing(
    src: _ArchiveSource,
    *,
    store,
    runtime,
    conflict: str,
    source_id: str,
    source_name: str,
    deferred_media: list[tuple[Path, dict[str, Any] | None]] | None = None,
) -> dict[str, Any]:
    from h3_app.dubbing import projects as dubbing_projects, results as dub_results, session as sessions

    data = json.loads(src.read_bytes("project.json"))
    orig_name = str(data.get("name") or source_name or source_id)
    existing = set(dubbing_projects.names(runtime))
    target_name, status, skipped = _resolve_named_conflict("dubbing", source_id, orig_name, existing, conflict)
    if skipped is not None:
        return skipped

    last_output_name = data.get("last_output_name")
    if last_output_name:
        _archive_filename(last_output_name)
    orig_stem = data.get("last_output_stem")
    if orig_stem is not None:
        _archive_filename(orig_stem)
        if not last_output_name or orig_stem != Path(last_output_name).stem:
            raise H3Error("Inconsistent dubbing output filename and stem.")
    elif last_output_name:
        orig_stem = _archive_filename(Path(last_output_name).stem)
    if orig_stem and orig_stem.casefold() == dub_results.SCAN_CACHE_DIR.casefold():
        raise H3Error("Reserved dubbing output stem.")
    _restore_referenced_speakers(data.get("referenced_speakers") or [], src, runtime)

    existing_session = None
    reuse_session_in_place = False
    if status == "overwritten":
        try:
            _existing_proj_data, existing_session = dubbing_projects.load(runtime, target_name)
        except H3Error:
            existing_session = None
        if existing_session is not None and existing_session.root.is_dir():
            other_sessions = {
                dubbing_projects._current_session(p, d)
                for p, d in dubbing_projects._entries(runtime)
                if d.get("name") != target_name
            }
            reuse_session_in_place = existing_session.id not in other_sessions

    if reuse_session_in_place and existing_session is not None:
        active_session_id = existing_session.id
        session_root = _contained_path(sessions.sessions_dir(runtime), existing_session.root)
    else:
        active_session_id = uuid.uuid4().hex
        session_root = (sessions.sessions_dir(runtime) / active_session_id).resolve()

    media_to_register: list[tuple[Path, dict[str, Any] | None]] = []

    with _StagingTransaction() as tx:
        if reuse_session_in_place:
            work_session, final_session = src.stage_prefix(tx, "session", session_root, direct_into_new_dir=False)
            manifest_arc = f"session/{sessions.MANIFEST}"
            manifest_work_path = work_session.get(manifest_arc)
            if manifest_work_path is None:
                raise H3Error("Imported dubbing archive is missing a valid session.json.")
            _schedule_obsolete_deletes(tx, session_root, set(final_session.values()), skip_rel=_skip_dubbing_session_rel)
        else:
            tx.track_created_dir(session_root)
            src.stage_prefix(tx, "session", session_root, direct_into_new_dir=True)
            manifest_work_path = session_root / sessions.MANIFEST

        try:
            session_data = json.loads(manifest_work_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise H3Error("Imported dubbing archive is missing a valid session.json.") from exc
        if not isinstance(session_data, dict):
            raise H3Error("Imported dubbing archive is missing a valid session.json.")
        session_data["id"] = active_session_id
        session_data["updated"] = time.time()

        result_entries = src.prefix_entries("result")
        stale_lipsync: Path | None = None
        written_media: list[tuple[Path, bool]] = []
        snapshots_by_media: dict[Path, dict[str, Any]] = {}

        if last_output_name and result_entries:
            dub_out_dir = _contained_path(Path(runtime.outputs_dir), Path(runtime.outputs_dir) / "dubbing")
            dub_out_dir.mkdir(parents=True, exist_ok=True)
            target_stem = orig_stem
            if conflict == "rename" and any(
                (dub_out_dir / candidate).exists()
                for candidate in (last_output_name, f"{orig_stem}.mp3", f"{orig_stem}.mp4", f"{orig_stem}.lipsync.mp4")
            ):
                used_stems = {p.stem for p in dub_out_dir.iterdir() if p.is_file()}
                target_stem = _unique_name(orig_stem, used_stems, style="stem")

            for arc_name in result_entries:
                fname = arc_name[len("result/") :]
                if not fname or "/" in fname or "\\" in fname or fname in {".", ".."}:
                    raise H3Error(f"Unsafe archive path: {arc_name!r}")
                if target_stem != orig_stem and fname.startswith(orig_stem):
                    target_fname = target_stem + fname[len(orig_stem) :]
                else:
                    target_fname = fname
                dest = _contained_path(dub_out_dir, dub_out_dir / _archive_filename(target_fname))
                if dest.name.endswith(".settings.json"):
                    staged, _ = src.stage_entry(tx, arc_name, dest, allow_in_place=False)
                    media_name = dest.name[: -len(".settings.json")]
                    target_media = dub_out_dir / media_name
                    snap_data = _update_snapshot_file(
                        staged,
                        output_name=media_name,
                        target_media=target_media,
                        session_id=active_session_id,
                    )
                    if snap_data is not None:
                        snapshots_by_media[target_media] = snap_data
                else:
                    _staged, media_reused = src.stage_entry(tx, arc_name, dest)
                    written_media.append((dest, media_reused))

            if status == "overwritten":
                written_dest_set = {dest for dest, _reused in written_media}
                subtitles = [dub_out_dir / f"{target_stem}.srt", *dub_out_dir.glob(f"{glob.escape(target_stem)}.*.srt")]
                for subtitle in subtitles:
                    if subtitle.is_file() and subtitle not in written_dest_set and not subtitle.is_symlink():
                        tx.stage_delete_file(subtitle)
                        sidecar = snapshot_path(subtitle)
                        if sidecar.is_file() and not sidecar.is_symlink():
                            tx.stage_delete_file(sidecar)
                candidate_lipsync = _contained_path(dub_out_dir, dub_out_dir / f"{target_stem}.lipsync.mp4")
                if candidate_lipsync not in written_dest_set and candidate_lipsync.is_file():
                    stale_lipsync = candidate_lipsync
                    tx.stage_delete_file(stale_lipsync)
                    stale_snap = snapshot_path(stale_lipsync)
                    if stale_snap.is_file() and not stale_snap.is_symlink():
                        tx.stage_delete_file(stale_snap)

            player_dir = _contained_path(dub_out_dir, dub_out_dir / dub_results.PLAYER_DIR / target_stem)
            has_player = bool(src.prefix_entries("result_player"))
            if has_player or player_dir.exists():
                player_is_new = not player_dir.exists()
                if player_is_new and has_player:
                    tx.track_created_dir(player_dir)
                work_player, final_player = src.stage_prefix(
                    tx, "result_player", player_dir, direct_into_new_dir=player_is_new and has_player
                )
                if has_player and target_stem != orig_stem:
                    for meta_name in (dub_results.PLAYER_FILE, "alignment.json"):
                        meta_path = work_player.get(f"result_player/{meta_name}")
                        if meta_path is None or not meta_path.is_file():
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
                            for field_name in ("video", "dub"):
                                val = container.get(field_name)
                                if isinstance(val, str) and Path(val.replace("\\", "/")).name.startswith(orig_stem):
                                    fname = Path(val.replace("\\", "/")).name
                                    container[field_name] = str(
                                        dub_out_dir / (target_stem + fname[len(orig_stem) :])
                                    )
                            write_json_atomic(meta_path, meta)

                    from h3_app.dubbing.lip_sync import PLAN_FILE, REPORT_FILE

                    plan_path = work_player.get(f"result_player/{PLAN_FILE}")
                    if plan_path:
                        try:
                            plan = json.loads(plan_path.read_text(encoding="utf-8"))
                        except (ValueError, OSError):
                            plan = None
                        if isinstance(plan, dict) and plan.get("video") == f"{orig_stem}.mp4":
                            plan["video"] = f"{target_stem}.mp4"
                            write_json_atomic(plan_path, plan)


                    report_path = work_player.get(f"result_player/{REPORT_FILE}")
                    if report_path is not None and report_path.is_file():
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

                if not player_is_new:
                    _schedule_obsolete_deletes(
                        tx,
                        player_dir,
                        set(final_player.values()),
                        skip_rel=_skip_dubbing_player_rel,
                        remove_root_if_empty=not has_player,
                    )

            target_last_name = (
                target_stem + last_output_name[len(orig_stem) :]
                if target_stem != orig_stem and last_output_name.startswith(orig_stem)
                else last_output_name
            )
            session_data["last_output"] = str(dub_out_dir / target_last_name)
        else:
            session_data.pop("last_output", None)

        write_json_atomic(manifest_work_path, session_data)
        tx.commit(
            lambda: dubbing_projects.save_imported(
                runtime, target_name, active_session_id, data.get("settings") or {}
            )
        )

    if store is not None and stale_lipsync is not None and stale_lipsync in tx.deleted_paths:
        store.forget_asset(stale_lipsync)

    for media_path, media_reused in written_media:
        if media_reused and _is_media_already_cataloged(media_path, store):
            continue
        snap = snapshots_by_media.get(media_path) or read_snapshot(media_path) or {"family": dub_results.DUB_FAMILY}
        media_to_register.append((media_path, snap))
    if media_to_register:
        if deferred_media is not None:
            deferred_media.extend(media_to_register)
        else:
            _register_imported_media_batch(media_to_register, store=store, runtime=runtime)

    return _transfer_result("dubbing", source_id, orig_name, target_name, target_name, status, src=src, tx=tx)


def _import_remake(
    src: _ArchiveSource, *, runtime, conflict: str, owner: str, source_id: str, source_name: str
) -> dict[str, Any]:
    from h3_app.remake.project import Projects as RemakeProjects

    project = json.loads(src.read_bytes("project.json"))
    orig_key = str(project.get("key") or source_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", orig_key):
        raise H3Error("Invalid Remake project key.")
    _archive_filename(orig_key)
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

    expected_folder = projects.folder(target_owner, target_key)
    if target_folder != _contained_path(Path(runtime.output_dir), expected_folder):
        raise H3Error("Unsafe Remake project directory.")
    takes_final = _contained_path(target_folder, target_folder / "takes")
    with _StagingTransaction() as tx:
        folder_is_new = not target_folder.exists()
        if folder_is_new:
            tx.track_created_dir(target_folder)

        _work_folder, extracted_folder = src.stage_prefix(
            tx, "folder", target_folder, direct_into_new_dir=folder_is_new
        )
        work_takes, extracted_takes = src.stage_prefix(
            tx, "takes", takes_final, direct_into_new_dir=folder_is_new
        )
        arc_map = {
            **{k: str(v) for k, v in extracted_folder.items()},
            **{k: str(v) for k, v in extracted_takes.items()},
        }

        for arc, staged_path in work_takes.items():
            final_path = extracted_takes[arc]
            if final_path.name.endswith(".settings.json"):
                media_name = final_path.name[: -len(".settings.json")]
                _update_snapshot_file(
                    staged_path,
                    output_name=media_name,
                    target_media=final_path.with_name(media_name),
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

        project_json_dest = _contained_path(target_folder, target_folder / "project.json")
        project_json_payload = json.dumps(
            {k: v for k, v in project.items() if k not in {"folder", "updated"}}, indent=1
        )
        if folder_is_new:
            target_folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(project_json_dest, json.loads(project_json_payload))
        else:
            staged_project_json = tx.stage_file(project_json_dest)
            staged_project_json.write_text(project_json_payload, encoding="utf-8")
            _schedule_obsolete_deletes(
                tx,
                target_folder,
                {project_json_dest} | set(extracted_folder.values()) | set(extracted_takes.values()),
            )
        tx.commit()

    return _transfer_result("remake", orig_key, orig_name, target_key, target_name, status, src=src, tx=tx)


def import_project_archive(
    archive_path: Path,
    *,
    store,
    runtime,
    conflict: str = "rename",
    owner: str = "imported",
    deferred_media: list[tuple[Path, dict[str, Any] | None]] | None = None,
    local_files: dict[str, list[Path]] | None = None,
) -> dict[str, Any]:
    """Validate and import a full or incremental delta project archive into the local store and runtime."""
    if conflict not in CONFLICT_MODES:
        raise H3Error(f"Unsupported conflict mode: {conflict!r}")
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise H3Error("Project archive file was not found.")
    with _PROJECT_TRANSFER_LOCK, zipfile.ZipFile(archive_path, "r") as archive:
        _validate_archive(archive)
        try:
            if archive.getinfo("manifest.json").file_size > _MAX_METADATA_BYTES:
                raise H3Error("Archive manifest is too large.")
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
        src = _ArchiveSource(archive, manifest, local_files=local_files)
        if kind == "soundstage":
            if store is None:
                raise H3Error("Workspace store is unavailable.")
            return _import_soundstage(
                src,
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
                src,
                store=store,
                runtime=runtime,
                conflict=conflict,
                owner=owner,
                source_id=source_id,
                source_name=source_name,
            )
        if kind == "dubbing":
            from h3_app.dubbing import projects, session
            with projects._LOCK, session._LOCK:
                return _import_dubbing(
                    src,
                    store=store,
                    runtime=runtime,
                    conflict=conflict,
                    source_id=source_id,
                    source_name=source_name,
                    deferred_media=deferred_media,
                )
        from h3_app.remake import project
        with project._LOCK:
            return _import_remake(
                src,
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
        found = _find_existing_local_project(kind, project_id, remote_name, store=store, runtime=runtime)
    if found is None:
        return None
    local_id, local_name = found
    orig_name = (remote_name or local_name) if kind in {"storyboard", "remake"} else local_name
    return _transfer_result(kind, project_id, orig_name, local_id, local_name, "skipped")


def _parse_content_range_start(header_value: str | None) -> int | None:
    if not header_value:
        return None
    match = re.match(r"^bytes\s+(\d+)-\d+/(\d+|\*)$", header_value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _stream_archive_with_resume(
    client: httpx.Client,
    normalized_url: str,
    *,
    method: str,
    endpoint: str,
    kind: str,
    project_id: str,
    archive_path: Path,
    params: Mapping[str, str] | None = None,
    json_body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    on_bytes: Callable[[int, int], None] | None = None,
    allow_delta_responses: bool = False,
) -> tuple[str, Path | dict[str, Any] | None]:
    """Stream an exported or delta archive with transient retry and HTTP Range resume."""
    base_headers = dict(headers or {})
    downloaded = 0
    total_bytes = 0
    etag: str | None = None
    used_range_resume = False
    archive_noun = "delta archive" if allow_delta_responses else "archive"

    for attempt in range(_HTTP_MAX_ATTEMPTS):
        req_headers = dict(base_headers)
        if downloaded > 0 and archive_path.is_file():
            req_headers["Range"] = f"bytes={downloaded}-"
            if etag:
                req_headers["If-Range"] = etag

        try:
            with client.stream(
                method,
                f"{normalized_url}{endpoint}",
                params=params,
                json=json_body,
                headers=req_headers,
            ) as response:
                status = response.status_code
                if allow_delta_responses and status in (404, 405) and TRANSFER_HEADER.lower() not in response.headers:
                    response.read()
                    return "unsupported", None
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

                if allow_delta_responses:
                    content_type = (response.headers.get("content-type") or "").lower()
                    if status == 200 and content_type.startswith("application/json"):
                        raw_body = response.read()
                        try:
                            meta = json.loads(raw_body)
                        except (TypeError, ValueError) as exc:
                            raise H3Error("Invalid delta transfer response from remote server.") from exc
                        if isinstance(meta, dict) and meta.get("status") == "up_to_date":
                            archive_path.unlink(missing_ok=True)
                            return "up_to_date", meta
                        raise H3Error("Unexpected JSON response from delta transfer endpoint.")

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
                    f"Incomplete {archive_noun} download ({downloaded / (1 << 20):.1f} of {total_bytes / (1 << 20):.1f} MB)."
                )

            if not zipfile.is_zipfile(archive_path):
                archive_path.unlink(missing_ok=True)
                downloaded = 0
                total_bytes = 0
                etag = None
                if used_range_resume and attempt + 1 < _HTTP_MAX_ATTEMPTS:
                    used_range_resume = False
                    continue
                raise H3Error(f"Downloaded project {archive_noun} is corrupted or incomplete.")

            return "archive", archive_path
        except httpx.RequestError as exc:
            if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                time.sleep(_retry_sleep_seconds(attempt))
                continue
            raise H3Error(f"Transfer connection error from {normalized_url}: {exc}") from exc

    verb = "sync" if allow_delta_responses else "download"
    raise H3Error(f"Could not {verb} {KIND_LABELS.get(kind, kind)} project {project_id!r}.")


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
    _outcome, payload = _stream_archive_with_resume(
        client,
        normalized_url,
        method="GET",
        endpoint="/api/projects/transfer/export",
        kind=kind,
        project_id=project_id,
        archive_path=archive_path,
        params={"kind": kind, "id": project_id},
        headers=headers,
        on_bytes=on_bytes,
    )
    assert isinstance(payload, Path)
    return payload


def _download_project_delta(
    client: httpx.Client,
    normalized_url: str,
    kind: str,
    project_id: str,
    archive_path: Path,
    local_state: _LocalOverwriteState,
    *,
    headers: dict[str, str] | None = None,
    on_bytes: Callable[[int, int], None] | None = None,
) -> tuple[str, Path | dict[str, Any] | None]:
    """Request an incremental delta archive; returns ``('up_to_date', meta)``, ``('archive', path)``, or ``('unsupported', None)``."""
    req_body = {
        "kind": kind,
        "id": project_id,
        "known_digests": sorted(local_state.known_digests)[:MAX_KNOWN_DIGESTS],
        "expected_tree_digest": local_state.portable_digest,
    }
    return _stream_archive_with_resume(
        client,
        normalized_url,
        method="POST",
        endpoint="/api/projects/transfer/delta",
        kind=kind,
        project_id=project_id,
        archive_path=archive_path,
        json_body=req_body,
        headers=headers,
        on_bytes=on_bytes,
        allow_delta_responses=True,
    )


@dataclass
class _DownloadedItem:
    outcome: str  # "archive" | "up_to_date"
    archive_path: Path | None = None
    local_state: _LocalOverwriteState | None = None
    up_to_date_meta: dict[str, Any] | None = None


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
    delta_supported = True

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

            def download_task(item_idx: int) -> _DownloadedItem:
                nonlocal delta_supported
                kind, project_id, remote_name = items[item_idx]
                label = KIND_LABELS.get(kind, kind)
                display = remote_name or project_id
                archive_path = tmp_root / f"transfer_{item_idx}.zip"

                def on_bytes(done_bytes: int, expected_bytes: int, *, verb: str = "Downloading") -> None:
                    frac = (done_bytes / expected_bytes) if expected_bytes > 0 else 0.35
                    with progress_lock:
                        item_units[item_idx] = max(item_units[item_idx], 0.5 * min(1.0, frac))
                    mb_done = done_bytes / (1 << 20)
                    size_note = (
                        f"{mb_done:.1f} / {expected_bytes / (1 << 20):.1f} MB"
                        if expected_bytes > 0
                        else f"{mb_done:.1f} MB"
                    )
                    emit_progress(f"{verb} {label} project {display}… ({size_note})")

                local_state: _LocalOverwriteState | None = None
                if conflict == "overwrite" and kind in PROJECT_KINDS and delta_supported:
                    emit_progress(
                        f"Checking local {label} project {display} for incremental sync…",
                        force=item_idx == pending_indices[0],
                    )
                    local_state = inspect_local_project_for_delta(
                        kind, project_id, remote_name, store=store, runtime=runtime
                    )

                if local_state is not None and delta_supported:
                    outcome, payload = _download_project_delta(
                        client,
                        normalized,
                        kind,
                        project_id,
                        archive_path,
                        local_state,
                        headers=headers,
                        on_bytes=lambda done, exp: on_bytes(done, exp, verb="Syncing"),
                    )
                    if outcome == "up_to_date":
                        return _DownloadedItem(
                            outcome="up_to_date",
                            local_state=local_state,
                            up_to_date_meta=payload if isinstance(payload, dict) else {},
                        )
                    if outcome == "archive" and isinstance(payload, Path):
                        return _DownloadedItem(
                            outcome="archive",
                            archive_path=payload,
                            local_state=local_state,
                        )
                    delta_supported = False

                emit_progress(f"Downloading {label} project {display}…", force=item_idx == pending_indices[0])
                full_path = _download_project_archive(
                    client,
                    normalized,
                    kind,
                    project_id,
                    archive_path,
                    headers=headers,
                    on_bytes=on_bytes,
                )
                return _DownloadedItem(outcome="archive", archive_path=full_path, local_state=local_state)

            in_flight: dict[int, Future[_DownloadedItem]] = {}
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
                    downloaded = future.result()
                    if downloaded.outcome == "up_to_date" and downloaded.local_state is not None:
                        meta = downloaded.up_to_date_meta or {}
                        ls = downloaded.local_state
                        total_files = int(meta.get("total_files", ls.content_file_count) or 0)
                        reused_bytes = int(meta.get("total_bytes", ls.content_bytes) or 0)
                        remote_disp = str(meta.get("name") or remote_name or ls.local_name)
                        if total_files > 0:
                            detail = f"{_plural(total_files, 'file')} unchanged"
                            if reused_bytes >= (64 << 10):
                                detail += f" · {_format_bytes(reused_bytes)} kept"
                            up_msg = f'{label} "{ls.local_name}" is already up to date ({detail}).'
                        else:
                            up_msg = f'{label} "{ls.local_name}" is already up to date.'
                        ordered_results[item_idx] = _transfer_result(
                            kind,
                            project_id,
                            remote_disp,
                            ls.local_id,
                            ls.local_name,
                            "up_to_date",
                            message=up_msg,
                            extra={
                                "incremental": True,
                                "transferred_files": 0,
                                "reused_files": total_files,
                                "removed_files": 0,
                                "transferred_bytes": 0,
                                "reused_bytes": reused_bytes,
                            },
                        )
                        continue

                    assert downloaded.archive_path is not None
                    with progress_lock:
                        item_units[item_idx] = max(item_units[item_idx], 0.55)
                    emit_progress(f"Importing {label} project {display}…", force=True)
                    local_map = downloaded.local_state.files_by_sha256 if downloaded.local_state else None
                    try:
                        result = import_project_archive(
                            downloaded.archive_path,
                            store=store,
                            runtime=runtime,
                            conflict=conflict,
                            owner=owner,
                            deferred_media=deferred_media,
                            local_files=local_map,
                        )
                    except _MissingReusedFileError:
                        downloaded.archive_path.unlink(missing_ok=True)
                        emit_progress(f"Downloading full {label} project {display}…", force=True)
                        fallback_path = _download_project_archive(
                            client,
                            normalized,
                            kind,
                            project_id,
                            archive_path,
                            headers=headers,
                        )
                        result = import_project_archive(
                            fallback_path,
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
