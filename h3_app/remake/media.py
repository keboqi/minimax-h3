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

# A quarter second: more than a source's streams differ by, too little to hide a wrong selection.
MAX_HELD_LAST_FRAMES = 6


def _run_ffmpeg(
    cmd: Sequence[str],
    *,
    timeout: float,
    partial_outputs: Sequence[Path] = (),
    capture_output: bool = True,
    text: bool = True,
):
    try:
        return run_media_process(
            cmd,
            timeout=timeout,
            partial_outputs=partial_outputs,
            capture_output=capture_output,
            text=text,
        )
    except FileNotFoundError as exc:
        raise H3Error("FFmpeg isn't installed in this environment.") from exc


def frame_seconds(start: float, frame: int) -> float:
    return float(start or 0) + int(frame) / FPS


def extract_frame(video: str | Path, seconds: float, output: Path) -> Path:
    """Save the source frame at ``seconds`` as a lossless PNG."""
    partial = output.with_suffix(".partial.png")
    _run_ffmpeg(
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
    _run_ffmpeg(
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
        process = _run_ffmpeg(
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
    process = _run_ffmpeg(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments, *audio,
         "-filter_complex", ";".join(chains), "-map", f"[{joined}]",
         *(["-map", f"{len(pieces)}:a:0", "-c:a", "aac", "-b:a", "320k"] if has_audio else []),
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
    _run_ffmpeg(
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


def has_untouched_source(source_duration: float, start: float, frames: int) -> tuple[bool, bool]:
    """Whether ``source_duration`` has untouched frames before and/or after the remade span."""
    start_frame = max(0, int(round(float(start or 0) * FPS)))
    source_frames = int(float(source_duration) * FPS + 0.5)
    return start_frame > 0, start_frame + int(frames) < source_frames


def splice_into_source(
    source: str | Path,
    remade: str | Path,
    *,
    start: float,
    frames: int,
    output: Path,
    has_audio: bool,
) -> Path:
    """Splice ``remade`` into ``source`` at ``start``..``start + frames``, keeping the untouched source and soundtrack."""
    remade_meta = probe_video_metadata(Path(remade))
    remade_count = min(int(frames), remade_meta.frame_count)
    if remade_count < 1:
        raise H3Error("The remade clip has no frames.")
    source_meta = probe_video_metadata(Path(source))
    start_frame = max(0, int(round(float(start or 0) * FPS)))
    has_before, has_after = has_untouched_source(source_meta.duration, start_frame / FPS, remade_count)
    remade_end = (start_frame + remade_count) / FPS

    target_w, target_h = remade_meta.width, remade_meta.height
    fit = (
        f"fps={FPS},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},setsar=1,format=yuv420p,settb=AVTB"
    )
    inputs: list[str] = []
    chains: list[str] = []
    segments: list[str] = []

    if has_before:
        idx = sum(1 for item in inputs if item == "-i")
        inputs += ["-t", f"{(start_frame + 2) / FPS:.3f}", "-i", str(source)]
        chains.append(f"[{idx}:v]{fit},trim=start_frame=0:end_frame={start_frame},setpts=PTS-STARTPTS[seg_before]")
        segments.append("[seg_before]")

    remade_idx = sum(1 for item in inputs if item == "-i")
    inputs += ["-i", str(remade)]
    chains.append(
        f"[{remade_idx}:v]fps={FPS},trim=start_frame=0:end_frame={remade_count},"
        f"setpts=PTS-STARTPTS,setsar=1,format=yuv420p,settb=AVTB[seg_remade]"
    )
    segments.append("[seg_remade]")

    if has_after:
        after_idx = sum(1 for item in inputs if item == "-i")
        inputs += ["-ss", f"{max(0.0, remade_end - 0.001):.6f}", "-i", str(source)]
        chains.append(f"[{after_idx}:v]{fit},setpts=PTS-STARTPTS[seg_after]")
        segments.append("[seg_after]")

    if len(segments) > 1:
        chains.append(f"{''.join(segments)}concat=n={len(segments)}:v=1:a=0[vout]")
        video_map = "[vout]"
    else:
        video_map = "[seg_remade]"

    audio_args: list[str] = []
    if has_audio:
        audio_idx = sum(1 for item in inputs if item == "-i")
        inputs += ["-i", str(source)]
        audio_args = ["-map", f"{audio_idx}:a:0", "-c:a", "aac", "-b:a", "320k"]

    partial = output.with_suffix(".partial.mp4")
    process = _run_ffmpeg(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
         "-filter_complex", ";".join(chains), "-map", video_map, *audio_args,
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(partial)],
        timeout=7200, partial_outputs=(partial,), capture_output=True, text=True,
    )
    if process.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise H3Error("Couldn't splice the remade window into the source: "
                      + (process.stderr or "ffmpeg failed").strip()[-300:])
    partial.replace(output)
    return output


def trim_source(video: str | Path, output: Path, *, start: float, frames: int, width: int, height: int) -> Path:
    """Write exactly ``frames`` at 24 fps, cover-cropped to the canvas, keeping any soundtrack.

    This is the resize Viggle's sampler does with ffmpeg, and it keeps ComfyUI from
    decoding full-size frames only to shrink them. A selection is measured by the container's
    duration, which can outlast the video slightly, so a few missing frames repeat the last one.
    """
    partial = output.with_suffix(".partial.mp4")
    _run_ffmpeg(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{float(start):.3f}", "-i", str(video),
         "-t", f"{(frames + 2) / FPS:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", f"fps={FPS},tpad=stop_mode=clone:stop={MAX_HELD_LAST_FRAMES},"
                f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
         "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "320k", str(partial)],
        timeout=600, partial_outputs=(partial,),
    )
    written = probe_video_metadata(partial).frame_count
    if written < frames:
        partial.unlink(missing_ok=True)
        available = max(0, written - MAX_HELD_LAST_FRAMES)
        raise H3Error(f"The source only has {available} frames after the start time; {frames} are needed.")
    partial.replace(output)
    return output
