"""Extracted media tools boundary."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Iterable

from h3_app.catalog import COMFY_UPSCALE_OPTIONS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.policy import ltx25_frame_length
from h3_app.processes import run_media_process
from h3_app.staging import stage_file

from .jobs import check_cancelled
from .provenance import copy_snapshot


def has_encoder(name: str) -> bool:
    proc = run_media_process(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return name in proc.stdout


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
        copy_snapshot(source, target)
        pending.replace(target)
        return target
    finally:
        pending.unlink(missing_ok=True)


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


def prepare_upscale_clip_batch(
    source: Path,
    *,
    category: str,
    split_enabled: bool,
    split_seconds: float,
    metadata: VideoMetadata,
    runtime: RuntimeConfig,
) -> UpscaleClipBatch:
    """Stage one source, or exact-frame LTX-safe source clips, for ComfyUI."""
    if not split_enabled:
        return UpscaleClipBatch((stage_file(str(source), category, runtime=runtime),))
    seconds = float(split_seconds)
    if not 1.0 <= seconds <= 15.0:
        raise H3Error("Upscale clip length must be between 1 and 15 seconds.")
    if metadata.duration <= 0 or metadata.frame_count <= 0:
        raise H3Error("Could not determine the source duration for split upscaling.")

    # LTX video lengths are 8n+1. Every clip is padded to this exact size;
    # concat_upscaled_clips trims only the final padded tail back off.
    clip_frames = ltx25_frame_length(seconds, metadata.fps)
    clip_duration = clip_frames / metadata.fps
    clip_count = max(1, math.ceil(metadata.frame_count / clip_frames))
    directory = runtime.input_dir / "h3_gradio" / category / f"clips_{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    try:
        for index in range(clip_count):
            path = directory / f"clip_{index + 1:04d}.mp4"
            start = index * clip_frames / metadata.fps
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ss",
                f"{start:.9f}",
                "-map",
                "0:v:0",
            ]
            if metadata.has_audio:
                cmd += ["-map", "0:a:0?", "-af", "apad"]
            cmd += [
                "-vf",
                (
                    f"fps={metadata.fps:.9f},"
                    f"tpad=stop_mode=clone:stop_duration={clip_duration:.9f}"
                ),
                "-frames:v",
                str(clip_frames),
                "-t",
                f"{clip_duration:.9f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
            if metadata.has_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            cmd += ["-avoid_negative_ts", "make_zero", str(path)]
            proc = run_media_process(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                partial_outputs=(path,),
            )
            if proc.returncode != 0 or not path.is_file():
                raise H3Error(
                    f"Could not create upscale clip {index + 1}/{clip_count}: "
                    f"{proc.stderr.strip()}"
                )
            paths.append(path)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return UpscaleClipBatch(
        tuple(path.relative_to(runtime.input_dir).as_posix() for path in paths),
        tuple(paths),
        directory,
    )


def concat_upscaled_clips(
    source: Path,
    clips: list[Path],
    *,
    option: str,
    duration: float,
    frame_count: int,
    runtime: RuntimeConfig,
) -> Path:
    """Losslessly concatenate upscale video streams and restore source audio."""
    if not clips:
        raise H3Error("No upscaled clips were produced.")
    runtime.outputs_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    manifest = runtime.outputs_dir / f".upscale_concat_{token}.txt"
    target = runtime.outputs_dir / f"{source.stem}_{token}.mp4"

    pending = target.with_suffix(target.suffix + ".partial")

    def manifest_path(path: Path) -> str:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        return f"file '{escaped}'"

    manifest.write_text(
        "\n".join(manifest_path(path) for path in clips) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-t",
        f"{float(duration):.9f}",
        "-frames:v",
        str(int(frame_count)),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(pending),
    ]
    try:
        proc = run_media_process(
            cmd, capture_output=True, text=True, timeout=600, partial_outputs=(pending,)
        )
        if proc.returncode != 0 or not pending.is_file():
            pending.unlink(missing_ok=True)
            raise H3Error(
                f"Could not concatenate upscaled clips: {proc.stderr.strip()}"
            )
        check_cancelled()
        copy_snapshot(source, target)
        pending.replace(target)
        return target
    finally:
        pending.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)


def cleanup_upscale_clip_batch(
    batch: UpscaleClipBatch | None,
    outputs: Iterable[Path] = (),
) -> None:
    """Remove only the explicitly tracked inputs and intermediate outputs."""
    for output in outputs:
        output.unlink(missing_ok=True)
    if batch is None:
        return
    for path in batch.temporary_inputs:
        path.unlink(missing_ok=True)
    if batch.temporary_directory is not None:
        try:
            batch.temporary_directory.rmdir()
        except OSError:
            pass
