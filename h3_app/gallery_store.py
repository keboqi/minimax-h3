"""Extracted gallery store boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from threading import Lock
from types import EllipsisType
from urllib.parse import quote

from h3_app.catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.processes import run_media_process

_GALLERY_RESOLUTION_CACHE = {}
_GALLERY_RESOLUTION_CACHE_LOCK = Lock()


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


def image_download_path(image: str | Path, *, runtime: RuntimeConfig) -> str:
    """Return a safe, public route for a generated image on this server."""
    resolved = managed_image_path(image, require_file=False, runtime=runtime)
    for bucket, root in (
        ("comfy", runtime.output_dir.resolve()),
        ("gradio", runtime.outputs_dir.resolve()),
    ):
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            return f"/downloads/{bucket}/{quote(relative, safe='/')}"
    raise H3Error("Generated image is outside the configured output directories.")


def managed_image_path(
    image: str | Path, *, require_file: bool = True, runtime: RuntimeConfig
) -> Path:
    """Resolve an image only when it belongs to a managed output directory."""
    resolved = Path(image).resolve()
    roots = (runtime.output_dir.resolve(), runtime.outputs_dir.resolve())
    thumbnail_root = runtime.gallery_thumbnails_dir.resolve()
    if (
        resolved.suffix.lower() not in IMAGE_EXTENSIONS
        or ".processing" in resolved.parts
        or resolved.is_relative_to(thumbnail_root)
        or not any(resolved.is_relative_to(root) for root in roots)
        or (require_file and not resolved.is_file())
    ):
        raise H3Error("Image is not a managed generated output.")
    return resolved


def audio_download_path(audio: str | Path, *, runtime: RuntimeConfig) -> str:
    """Return a safe, public route for a generated audio file on this server."""
    resolved = managed_audio_path(audio, require_file=False, runtime=runtime)
    for bucket, root in (
        ("comfy", runtime.output_dir.resolve()),
        ("gradio", runtime.outputs_dir.resolve()),
    ):
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            return f"/downloads/{bucket}/{quote(relative, safe='/')}"
    raise H3Error("Generated audio is outside the configured output directories.")


def managed_audio_path(
    audio: str | Path, *, require_file: bool = True, runtime: RuntimeConfig
) -> Path:
    """Resolve audio only when it belongs to a managed output directory."""
    resolved = Path(audio).resolve()
    roots = (runtime.output_dir.resolve(), runtime.outputs_dir.resolve())
    if (
        resolved.suffix.lower() not in AUDIO_EXTENSIONS
        or ".processing" in resolved.parts
        or not any(resolved.is_relative_to(root) for root in roots)
        or (require_file and not resolved.is_file())
    ):
        raise H3Error("Audio is not a managed generated output.")
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


def gallery_image_paths(
    *, limit: int | None | EllipsisType = ..., runtime: RuntimeConfig
) -> list[Path]:
    """Return generated images, optionally limited to the newest entries."""
    if limit is Ellipsis:
        limit = runtime.gallery_limit
    images: dict[Path, Path] = {}
    thumbnail_root = runtime.gallery_thumbnails_dir.resolve()
    for root in (runtime.output_dir, runtime.outputs_dir):
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for candidate in root.rglob("*"):
            if (
                not candidate.is_file()
                or candidate.suffix.lower() not in IMAGE_EXTENSIONS
                or ".processing" in candidate.parts
            ):
                continue
            try:
                resolved = candidate.resolve()
                if (
                    resolved.is_relative_to(resolved_root)
                    and not resolved.is_relative_to(thumbnail_root)
                ):
                    images[resolved] = candidate
            except OSError:
                continue

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    ordered = sorted(images.values(), key=modified, reverse=True)
    return ordered if limit is None else ordered[: max(0, int(limit))]


def gallery_audio_paths(
    *, limit: int | None | EllipsisType = ..., runtime: RuntimeConfig
) -> list[Path]:
    """Return generated audio files, optionally limited to the newest entries."""
    if limit is Ellipsis:
        limit = runtime.gallery_limit
    audio_files: dict[Path, Path] = {}
    for root in (runtime.output_dir, runtime.outputs_dir):
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for candidate in root.rglob("*"):
            if (
                not candidate.is_file()
                or candidate.suffix.lower() not in AUDIO_EXTENSIONS
                or ".processing" in candidate.parts
            ):
                continue
            try:
                resolved = candidate.resolve()
                if resolved.is_relative_to(resolved_root):
                    audio_files[resolved] = candidate
            except OSError:
                continue

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    ordered = sorted(audio_files.values(), key=modified, reverse=True)
    return ordered if limit is None else ordered[: max(0, int(limit))]


def gallery_audio_thumbnail(audio: Path, *, runtime: RuntimeConfig) -> Path | None:
    """Create a cached visual card so audio entries fit the shared gallery grid."""
    try:
        from PIL import Image, ImageDraw

        source_mtime = audio.stat().st_mtime
        thumbnail = gallery_audio_thumbnail_path(audio, runtime=runtime)
        if thumbnail.is_file() and thumbnail.stat().st_mtime >= source_mtime:
            return thumbnail
        runtime.gallery_thumbnails_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (480, 270), (18, 24, 38))
        draw = ImageDraw.Draw(canvas)
        digest = hashlib.sha256(str(audio.resolve()).encode("utf-8")).digest()
        center_y = 132
        for index in range(48):
            height = 18 + digest[index % len(digest)] % 78
            x = 28 + index * 9
            color = (78, 170 + digest[index % len(digest)] % 60, 220)
            draw.rounded_rectangle(
                (x, center_y - height // 2, x + 5, center_y + height // 2),
                radius=2,
                fill=color,
            )
        draw.text((28, 28), "AUDIO", fill=(225, 235, 248))
        display_name = audio.name if len(audio.name) <= 52 else f"{audio.name[:49]}..."
        draw.text((28, 230), display_name, fill=(172, 188, 210))
        temporary = thumbnail.with_name(
            f"{thumbnail.stem}.{uuid.uuid4().hex}.tmp.png"
        )
        canvas.save(temporary, format="PNG")
        temporary.replace(thumbnail)
        return thumbnail
    except (OSError, TypeError, ValueError):
        return None


def gallery_audio_thumbnail_path(
    audio: str | Path, *, runtime: RuntimeConfig
) -> Path:
    cache_key = hashlib.sha256(str(Path(audio).resolve()).encode("utf-8")).hexdigest()[
        :24
    ]
    return runtime.gallery_thumbnails_dir / f"audio-{cache_key}.png"


def gallery_image_thumbnail(image: Path, *, runtime: RuntimeConfig) -> Path | None:
    """Create a cached, bounded JPEG preview without serving the source image."""
    temporary: Path | None = None
    try:
        from PIL import Image, ImageOps

        source_mtime = image.stat().st_mtime
        thumbnail = gallery_image_thumbnail_path(image, runtime=runtime)
        if thumbnail.is_file() and thumbnail.stat().st_mtime >= source_mtime:
            return thumbnail
        runtime.gallery_thumbnails_dir.mkdir(parents=True, exist_ok=True)
        temporary = thumbnail.with_name(
            f"{thumbnail.stem}.{uuid.uuid4().hex}.tmp.jpg"
        )
        with Image.open(image) as opened:
            opened.draft("RGB", (480, 480))
            preview = ImageOps.exif_transpose(opened)
            preview.thumbnail((480, 480), Image.Resampling.LANCZOS)
            if preview.mode != "RGB":
                canvas = Image.new("RGB", preview.size, (18, 24, 38))
                if "A" in preview.getbands():
                    canvas.paste(preview, mask=preview.getchannel("A"))
                else:
                    canvas.paste(preview)
                preview = canvas
            preview.save(temporary, format="JPEG", quality=82)
        temporary.replace(thumbnail)
        return thumbnail
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return None


def gallery_image_thumbnail_path(image: str | Path, *, runtime: RuntimeConfig) -> Path:
    cache_key = hashlib.sha256(str(Path(image).resolve()).encode("utf-8")).hexdigest()[:24]
    return runtime.gallery_thumbnails_dir / f"image-{cache_key}.jpg"


def gallery_placeholder(media: Path, *, kind: str, runtime: RuntimeConfig) -> Path | None:
    """Keep an unreadable item selectable without exposing its full media file."""
    try:
        from PIL import Image, ImageDraw

        cache_key = hashlib.sha256(str(media.resolve()).encode("utf-8")).hexdigest()[:24]
        thumbnail = runtime.gallery_thumbnails_dir / f"{kind.lower()}-{cache_key}-unavailable.png"
        if thumbnail.is_file():
            return thumbnail
        runtime.gallery_thumbnails_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (480, 270), (18, 24, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 28), kind.upper(), fill=(225, 235, 248))
        draw.text((28, 230), "Preview unavailable", fill=(172, 188, 210))
        temporary = thumbnail.with_name(f"{thumbnail.stem}.{uuid.uuid4().hex}.tmp.png")
        canvas.save(temporary, format="PNG")
        temporary.replace(thumbnail)
        return thumbnail
    except (OSError, TypeError, ValueError):
        return None


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
    with _GALLERY_RESOLUTION_CACHE_LOCK:
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

    with _GALLERY_RESOLUTION_CACHE_LOCK:
        if len(_GALLERY_RESOLUTION_CACHE) >= runtime.gallery_metadata_cache_limit:
            _GALLERY_RESOLUTION_CACHE.clear()
        _GALLERY_RESOLUTION_CACHE[cache_key] = resolution
    return resolution


def gallery_resolution_text(video: Path, *, runtime: RuntimeConfig) -> str:
    resolution = gallery_video_resolution(video, runtime=runtime)
    return (
        f"{resolution[0]}×{resolution[1]}" if resolution else "resolution unavailable"
    )


def gallery_image_resolution(image: Path) -> tuple[int, int] | None:
    """Read an image's display dimensions, honoring EXIF orientation."""
    try:
        from PIL import Image, ImageOps

        with Image.open(image) as opened:
            width, height = ImageOps.exif_transpose(opened).size
        if width > 0 and height > 0:
            return int(width), int(height)
    except (OSError, TypeError, ValueError):
        pass
    return None


def gallery_image_resolution_text(image: Path) -> str:
    resolution = gallery_image_resolution(image)
    return (
        f"{resolution[0]}×{resolution[1]}"
        if resolution
        else "resolution unavailable"
    )


def generated_image_family(image: str | Path, *, runtime: RuntimeConfig) -> str:
    """Identify common generated-image sources from their managed output path."""
    resolved = Path(image).resolve()
    output_root = runtime.output_dir.resolve()
    outputs_root = runtime.outputs_dir.resolve()
    if resolved.is_relative_to(output_root):
        relative = resolved.relative_to(output_root)
        lowered = "/".join(relative.parts).lower()
        if "qwen_image21" in resolved.name.lower():
            return "Qwen Image 2.1"
        if "/input_upscale/" in f"/{lowered}/":
            return "SeedVR2"
        if relative.parts and relative.parts[0].lower() == "h3":
            return "MiniMax H3"
    if resolved.is_relative_to(outputs_root):
        relative = resolved.relative_to(outputs_root)
        if relative.parts and relative.parts[0].lower() == "imports":
            return "Imported"
    return "Generated image"


def generated_audio_family(audio: str | Path, *, runtime: RuntimeConfig) -> str:
    """Identify generated audio families from their managed filename and path."""
    resolved = Path(audio).resolve()
    name = resolved.name.lower()
    if "h3_fl2va" in name or "h3_ref2va" in name:
        return "MiniMax H3"
    if "minimax_music3" in name:
        return "MiniMax Music 3"
    if "yue2" in name:
        return "YuE2"
    outputs_root = runtime.outputs_dir.resolve()
    if resolved.is_relative_to(outputs_root):
        relative = resolved.relative_to(outputs_root)
        if relative.parts and relative.parts[0].lower() == "imports":
            return "Imported"
    return "Generated audio"


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
    with _GALLERY_RESOLUTION_CACHE_LOCK:
        if video is None:
            _GALLERY_RESOLUTION_CACHE.clear()
            return
        resolved = str(Path(video).resolve())
        for key in [key for key in _GALLERY_RESOLUTION_CACHE if key[0] == resolved]:
            _GALLERY_RESOLUTION_CACHE.pop(key, None)
