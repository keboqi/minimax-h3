"""Extracted staging boundary."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from threading import Lock

from h3_app.catalog import (
    REFERENCE_VIDEO_TRANSCODE_CACHE_VERSION,
    STAGED_INPUT_HASH_CHUNK_BYTES,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.processes import run_media_process

_STAGED_INPUT_LOCK = Lock()


def file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(STAGED_INPUT_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise H3Error(f"Could not hash input file {path.name}: {exc}") from exc
    return digest.hexdigest()


def staged_input_is_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def materialize_staged_input(
    source: Path,
    destination: Path,
    *,
    transcode_video: bool,
) -> None:
    temporary = destination.with_name(
        f".{destination.stem}-{uuid.uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        if transcode_video:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-t",
                "15",
                "-vf",
                "fps=24",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(temporary),
            ]
            proc = run_media_process(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise H3Error(
                    f"Reference-video conversion failed: {proc.stderr.strip()}"
                )
        else:
            shutil.copy2(source, temporary)

        if not staged_input_is_ready(temporary):
            raise H3Error(f"Staging produced an empty input file: {source.name}")
        temporary.replace(destination)
    except OSError as exc:
        raise H3Error(f"Could not stage input file {source.name}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def stage_file(
    path: str,
    category: str,
    transcode_video: bool = False,
    reuse: bool = False,
    *,
    runtime: RuntimeConfig,
) -> str:
    src = Path(path)
    if not src.is_file():
        raise H3Error(f"Input file does not exist: {src}")
    target_dir = runtime.input_dir / "h3_gradio" / category
    target_dir.mkdir(parents=True, exist_ok=True)
    if reuse:
        source_digest = file_content_sha256(src)
        token = (
            f"{REFERENCE_VIDEO_TRANSCODE_CACHE_VERSION}-{source_digest}"
            if transcode_video
            else source_digest
        )
    else:
        token = uuid.uuid4().hex

    suffix = ".mp4" if transcode_video else (src.suffix.lower() or ".bin")
    dst = target_dir / f"{token}{suffix}"

    if reuse:
        with _STAGED_INPUT_LOCK:
            if staged_input_is_ready(dst):
                action = "Reusing"
            else:
                materialize_staged_input(src, dst, transcode_video=transcode_video)
                action = "Stored"
        print(f"[h3-input-cache] {action} {category}/{dst.name}", flush=True)
    else:
        materialize_staged_input(src, dst, transcode_video=transcode_video)

    return dst.relative_to(runtime.input_dir).as_posix()
