"""FFmpeg post-processing and media probing."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from h3_app.catalog import COMFY_UPSCALE_OPTIONS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.media_types import VideoMetadata
from h3_app.processes import run_media_process

from .jobs import check_cancelled
from .provenance import copy_snapshot


def has_encoder(name: str) -> bool:
    # A listed NVENC codec can still fail to open on GPUs without encoder
    # hardware (such as H200), or when the container lacks driver access.
    try:
        proc = run_media_process(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "color=c=black:s=64x64:r=1", "-frames:v", "1",
                "-c:v", name, "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def postprocess_video(source: Path, option: str, *, runtime: RuntimeConfig) -> Path:
    if option == "None":
        return source
    if option in COMFY_UPSCALE_OPTIONS:
        raise H3Error(f"{option} must run through the ComfyUI workflow.")
    if option != "48 fps interpolation":
        raise H3Error(f"Unsupported post-processing method: {option}")
    runtime.outputs_dir.mkdir(parents=True, exist_ok=True)
    target = runtime.outputs_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}.mp4"
    pending = target.with_suffix(target.suffix + ".partial")
    filters = ["minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"]
    encoder = "h264_nvenc" if has_encoder("h264_nvenc") else "libx264"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", encoder, "-preset", "p5", "-cq", "19"]
    else:
        cmd += ["-c:v", encoder, "-preset", "medium", "-crf", "18"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-f", "mp4", str(pending)]
    try:
        proc = run_media_process(
            cmd,
            capture_output=True,
            text=True,
            timeout=runtime.generation_timeout,
            partial_outputs=(pending,),
        )
        if proc.returncode != 0 or not pending.is_file():
            raise H3Error(f"Post-processing failed: {proc.stderr.strip()}")
        check_cancelled()
        pending.replace(target)
        copy_snapshot(source, target)
        return target
    finally:
        pending.unlink(missing_ok=True)


def probe_audio_seconds(source: Path) -> float | None:
    """Playback length of a generated audio file, or None when it cannot be read."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(source)]
    try:
        proc = run_media_process(cmd, capture_output=True, text=True, timeout=30)
        seconds = float(json.loads(proc.stdout)["format"]["duration"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None
    return seconds if proc.returncode == 0 and seconds > 0 else None


def probe_video_metadata(source: Path) -> VideoMetadata:
    """Return dimensions and frame rate needed by AI upscale workflows."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    proc = run_media_process(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise H3Error(f"Could not inspect selected video: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
        streams = payload["streams"]
        stream = next(
            item for item in streams if item.get("codec_type", "video") == "video"
        )
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        numerator, denominator = str(rate).split("/", 1)
        fps = float(numerator) / float(denominator)
        if fps <= 0:
            raise ValueError("non-positive frame rate")
        width = int(stream["width"])
        height = int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("non-positive dimensions")
        raw_duration = payload.get("format", {}).get("duration") or stream.get(
            "duration"
        )
        duration = float(raw_duration)
        if duration <= 0:
            raise ValueError("non-positive duration")
        raw_frames = stream.get("nb_frames")
        frame_count = (
            int(raw_frames) if str(raw_frames).isdigit() else round(duration * fps)
        )
        if frame_count <= 0:
            raise ValueError("non-positive frame count")
        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            duration=duration,
            frame_count=frame_count,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )
    except (
        KeyError,
        IndexError,
        StopIteration,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        raise H3Error("Could not determine the selected video's metadata.") from exc
