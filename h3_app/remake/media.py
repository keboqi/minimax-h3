"""Source trimming and frame extraction for Remake."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

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


def filmstrip(video: str | Path, start: float, seconds: float, output: Path, count: int = 24) -> Path:
    """One image of ``count`` evenly spaced frames, side by side, for the page's timeline."""
    partial = output.with_suffix(".partial.jpg")
    rate = count / max(float(seconds), 0.04)
    run_media_process(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}", "-t", f"{float(seconds):.3f}", "-i", str(video),
         "-vf", f"fps={rate:.6f},scale=-2:72,tile={count}x1", "-frames:v", "1", "-q:v", "5", str(partial)],
        timeout=120, partial_outputs=(partial,),
    )
    if not partial.is_file():
        raise H3Error("Couldn't read frames from the source.")
    partial.replace(output)
    return output


def detect_cuts(video: str | Path, start: float, seconds: float) -> list[int]:
    """Frames of the selection that start a new shot, with the scene score Viggle uses."""
    with tempfile.TemporaryDirectory(prefix="h3-remake-cuts-") as folder:
        # A file, because a process's captured output keeps only its tail.
        log = Path(folder) / "cuts.txt"
        target = log.as_posix().replace(":", r"\:")
        process = run_media_process(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{float(start or 0):.3f}",
             "-t", f"{float(seconds):.3f}", "-i", str(video), "-map", "0:v:0",
             "-vf", f"scale=320:-2,select='gt(scene,0.3)',metadata=print:file='{target}'", "-f", "null", "-"],
            timeout=1800, capture_output=True, text=True,
        )
        if process.returncode != 0:
            raise H3Error("Couldn't scan the source for cuts.")
        times = re.findall(r"pts_time:([0-9.]+)", log.read_text("utf-8")) if log.is_file() else []
    return sorted({round(float(time) * FPS) for time in times})


def join_windows(pieces: Sequence[dict[str, Any]], audio_source: str | Path, *, start: float, frames: int,
                 mix: int, output: Path, has_audio: bool) -> Path:
    """Join each take's kept frames, crossfading continuing windows, under the original soundtrack.

    Each piece is ``{"path", "first", "frames", "continues"}``; ``first`` counts from the take's
    start. Using the source's audio leaves no seam there at all.
    """
    arguments: list[str] = []
    chains = []
    for index, piece in enumerate(pieces):
        arguments += ["-i", str(piece["path"])]
        first = int(piece["first"])
        chains.append(
            f"[{index}:v]trim=start_frame={first}:end_frame={first + int(piece['frames'])},"
            f"setpts=PTS-STARTPTS,fps={FPS},format=yuv420p,settb=AVTB[v{index}]"
        )
    joined, length = "v0", int(pieces[0]["frames"])
    for index, piece in enumerate(pieces[1:], 1):
        if piece["continues"] and mix:
            offset = (length - mix) / FPS
            chains.append(f"[{joined}][v{index}]xfade=transition=fade:duration={mix / FPS:.6f}:offset={offset:.6f}"
                          f"[j{index}]")
            length += int(piece["frames"]) - mix
        else:
            chains.append(f"[{joined}][v{index}]concat=n=2:v=1:a=0[j{index}]")
            length += int(piece["frames"])
        joined = f"j{index}"
    audio = ["-ss", f"{float(start or 0):.3f}", "-t", f"{frames / FPS:.3f}", "-i", str(audio_source)] if has_audio else []
    partial = output.with_suffix(".partial.mp4")
    process = run_media_process(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments, *audio,
         "-filter_complex", ";".join(chains), "-map", f"[{joined}]",
         *(["-map", f"{len(pieces)}:a:0", "-c:a", "aac", "-b:a", "320k", "-shortest"] if has_audio else []),
         "-frames:v", str(length), "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(partial)],
        timeout=7200, partial_outputs=(partial,), capture_output=True, text=True,
    )
    if process.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise H3Error("Couldn't join the windows: " + (process.stderr or "ffmpeg failed").strip()[-300:])
    partial.replace(output)
    return output


def held_clip(take: str | Path, output: Path, *, first: int, frames: int) -> Path:
    """Copy exactly these frames of a take, losslessly, for the next window to start from."""
    partial = output.with_suffix(".partial.mp4")
    run_media_process(
        ["ffmpeg", "-y", "-v", "error", "-i", str(take), "-map", "0:v:0",
         "-vf", f"trim=start_frame={int(first)}:end_frame={int(first) + int(frames)},setpts=PTS-STARTPTS",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(partial)],
        timeout=120, partial_outputs=(partial,),
    )
    if not partial.is_file() or probe_video_metadata(partial).frame_count < frames:
        partial.unlink(missing_ok=True)
        raise H3Error(f"The previous window's render doesn't reach frame {int(first) + int(frames)}.")
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
