"""Extracted gallery store boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from types import EllipsisType
from urllib.parse import quote

from h3_app.catalog import VIDEO_EXTENSIONS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.processes import run_media_process

_GALLERY_RESOLUTION_CACHE = {}


def video_download_path(video: str | Path, *, runtime: RuntimeConfig) -> str:
    """Return a safe, public route for a generated video on this server."""
    resolved = managed_video_path(video, require_file=False, runtime=runtime)
    for bucket, root in (
        ("comfy", runtime.output_dir.resolve()),
        ("gradio", runtime.outputs_dir.resolve()),
    ):
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            return f"/downloads/{bucket}/{quote(relative, safe='/')}"
    raise H3Error("Generated video is outside the configured output directories.")


def managed_video_path(
    video: str | Path, *, require_file: bool = True, runtime: RuntimeConfig
) -> Path:
    """Resolve a video only when it belongs to a managed output directory."""
    resolved = Path(video).resolve()
    roots = (runtime.output_dir.resolve(), runtime.outputs_dir.resolve())
    if (
        resolved.suffix.lower() not in VIDEO_EXTENSIONS
        or ".processing" in resolved.parts
        or not any(resolved.is_relative_to(root) for root in roots)
        or (require_file and not resolved.is_file())
    ):
        raise H3Error("Video is not a managed generated output.")
    return resolved


def gallery_video_paths(
    *, limit: int | None | EllipsisType = ..., runtime: RuntimeConfig
) -> list[Path]:
    """Return generated videos, optionally limited to the newest entries."""
    if limit is Ellipsis:
        limit = runtime.gallery_limit
    videos: dict[Path, Path] = {}
    for root in (runtime.output_dir, runtime.outputs_dir):
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for candidate in root.rglob("*"):
            if (
                not candidate.is_file()
                or candidate.suffix.lower() not in VIDEO_EXTENSIONS
                or ".processing" in candidate.parts
            ):
                continue
            try:
                resolved = candidate.resolve()
                if resolved.is_relative_to(resolved_root):
                    videos[resolved] = candidate
            except OSError:
                continue

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    ordered = sorted(videos.values(), key=modified, reverse=True)
    return ordered if limit is None else ordered[: max(0, int(limit))]


def gallery_thumbnail(video: Path, *, runtime: RuntimeConfig) -> Path | None:
    """Create a small cached poster image for a video."""
    temporary: Path | None = None
    try:
        video_mtime = video.stat().st_mtime
        thumbnail = gallery_thumbnail_path(video, runtime=runtime)
        cache_key = thumbnail.stem
        if thumbnail.is_file() and thumbnail.stat().st_mtime >= video_mtime:
            return thumbnail

        runtime.gallery_thumbnails_dir.mkdir(parents=True, exist_ok=True)
        temporary = (
            runtime.gallery_thumbnails_dir / f"{cache_key}.{uuid.uuid4().hex}.tmp.jpg"
        )
        temporary.unlink(missing_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.1",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2:force_original_aspect_ratio=decrease",
            str(temporary),
        ]
        proc = run_media_process(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            partial_outputs=(temporary,),
        )
        if proc.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(thumbnail)
        return thumbnail
    except (OSError, subprocess.TimeoutExpired):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return None


def gallery_thumbnail_path(video: str | Path, *, runtime: RuntimeConfig) -> Path:
    cache_key = hashlib.sha256(str(Path(video).resolve()).encode("utf-8")).hexdigest()[
        :24
    ]
    return runtime.gallery_thumbnails_dir / f"{cache_key}.jpg"


def gallery_video_resolution(
    video: Path, *, runtime: RuntimeConfig
) -> tuple[int, int] | None:
    """Return cached source dimensions without decoding the full video."""
    try:
        resolved = video.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    if cache_key in _GALLERY_RESOLUTION_CACHE:
        return _GALLERY_RESOLUTION_CACHE[cache_key]

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(resolved),
    ]
    resolution: tuple[int, int] | None = None
    try:
        proc = run_media_process(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            stream = json.loads(proc.stdout)["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            if width > 0 and height > 0:
                resolution = (width, height)
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ):
        pass

    if len(_GALLERY_RESOLUTION_CACHE) >= runtime.gallery_metadata_cache_limit:
        _GALLERY_RESOLUTION_CACHE.clear()
    _GALLERY_RESOLUTION_CACHE[cache_key] = resolution
    return resolution


def gallery_resolution_text(video: Path, *, runtime: RuntimeConfig) -> str:
    resolution = gallery_video_resolution(video, runtime=runtime)
    return (
        f"{resolution[0]}×{resolution[1]}" if resolution else "resolution unavailable"
    )


def generated_video_family(video: str | Path, *, runtime: RuntimeConfig) -> str:
    """Identify direct ComfyUI outputs while keeping all managed videos eligible."""
    resolved = Path(video).resolve()
    output_root = runtime.output_dir.resolve()
    if resolved.is_relative_to(output_root):
        relative = resolved.relative_to(output_root)
        top_level = relative.parts[0].lower() if relative.parts else ""
        if top_level == "h3":
            return "MiniMax H3"
        if top_level == "ltx25":
            return "LTX-2.5"
    return "Post-processed"


def forget_gallery_metadata(video: str | Path | None = None) -> None:
    if video is None:
        _GALLERY_RESOLUTION_CACHE.clear()
        return
    resolved = str(Path(video).resolve())
    for key in [key for key in _GALLERY_RESOLUTION_CACHE if key[0] == resolved]:
        _GALLERY_RESOLUTION_CACHE.pop(key, None)
