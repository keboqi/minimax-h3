"""Source trimming and frame extraction for Remake."""

from __future__ import annotations

from pathlib import Path

from h3_app.errors import H3Error
from h3_app.media_tools import probe_video_metadata
from h3_app.processes import run_media_process

from .model import FPS


def frame_seconds(start: float, frame: int) -> float:
    return float(start or 0) + int(frame) / FPS


def extract_frame(video: str | Path, seconds: float, output: Path) -> Path:
    """Save the source frame at ``seconds`` as a lossless PNG."""
    partial = output.with_suffix(".partial.png")
    run_media_process(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(seconds):.3f}", "-i", str(video),
         "-frames:v", "1", str(partial)],
        timeout=60, partial_outputs=(partial,),
    )
    if not partial.is_file():
        raise H3Error(f"No frame exists at {float(seconds):.2f} seconds of the source.")
    partial.replace(output)
    return output


def trim_source(video: str | Path, output: Path, *, start: float, frames: int, width: int, height: int) -> Path:
    """Write exactly ``frames`` at 24 fps, cover-cropped to the canvas, keeping any soundtrack.

    This is the resize Viggle's sampler does with ffmpeg, and it keeps ComfyUI from
    decoding full-size frames only to shrink them.
    """
    partial = output.with_suffix(".partial.mp4")
    run_media_process(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}", "-i", str(video),
         "-t", f"{(frames + 2) / FPS:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", f"fps={FPS},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
         "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "320k", str(partial)],
        timeout=600, partial_outputs=(partial,),
    )
    written = probe_video_metadata(partial).frame_count
    if written < frames:
        partial.unlink(missing_ok=True)
        raise H3Error(f"The source only has {written} frames after the start time; {frames} are needed.")
    partial.replace(output)
    return output
