"""A video from Media as a storyboard scene: probing it, fitting it to the canvas, cutting its tail.

All of it is ffmpeg on the CPU. The tail clip is what the next scene's own graph encodes into
the AV latent it continues from, so nothing here needs the GPU or a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from h3_app.errors import H3Error
from h3_app.processes import run_media_process

from .model import CROP, FPS, RESIZE, TAIL_FRAMES

AUDIO_RATE = 48000


def probe(path: str | Path) -> dict[str, Any]:
    """Width, height, length in seconds and whether the file has a soundtrack."""
    process = run_media_process(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height:stream_side_data=rotation:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        data = json.loads(process.stdout or "{}")
        streams = data.get("streams") or []
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        width, height = int(video["width"]), int(video["height"])
        rotation = next((int(side.get("rotation", 0)) for side in video.get("side_data_list") or []
                         if "rotation" in side), 0)
        if abs(rotation) % 180 == 90:
            width, height = height, width
        seconds = float(data["format"]["duration"])
    except (StopIteration, KeyError, TypeError, ValueError) as exc:
        raise H3Error(f"{Path(path).name} isn't a video ffmpeg can read.") from exc
    if process.returncode != 0 or width <= 0 or height <= 0 or seconds <= 0:
        raise H3Error(f"{Path(path).name} isn't a video ffmpeg can read.")
    return {"width": width, "height": height, "seconds": round(seconds, 3),
            "audio": any(stream.get("codec_type") == "audio" for stream in streams)}


def fit_filter(fit: str, width: int, height: int) -> str:
    """Resize stretches the whole frame onto the canvas; Crop covers it and cuts the overflow, centred."""
    if fit == RESIZE:
        return f"scale={width}:{height}:flags=lanczos,setsar=1"
    if fit == CROP:
        return (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height},setsar=1")
    raise H3Error(f"Unknown fit {fit!r}.")


def frame_count(start: float, end: float) -> int:
    return int(round((float(end) - float(start)) * FPS))


def _ffmpeg(arguments: list[str], output: Path, failure: str, *, timeout: int) -> Path:
    """Run ffmpeg into a partial file beside ``output``; only a non-empty result replaces it."""
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    process = run_media_process(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments, str(partial)],
        timeout=timeout, partial_outputs=(partial,), capture_output=True, text=True,
    )
    if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise H3Error(f"{failure}: " + ((process.stderr or "").strip()[-300:] or "ffmpeg failed"))
    partial.replace(output)
    return output


def conform(source: str | Path, output: Path, *, start: float, end: float, fit: str, width: int, height: int,
            has_audio: bool) -> Path:
    """The In–Out range at 24 fps on the storyboard's canvas, always with a stereo soundtrack.

    Joining takes maps every clip's video and audio, so a silent source gets a silent track.
    """
    frames = frame_count(start, end)
    inputs = ["-ss", f"{float(start):.3f}", "-t", f"{float(end) - float(start) + 1:.3f}", "-i", str(source)]
    if has_audio:
        audio = ["-map", "0:a:0", "-af", f"aresample={AUDIO_RATE},apad", "-ac", "2"]
    else:
        inputs += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        audio = ["-map", "1:a:0"]
    return _ffmpeg(
        [*inputs, "-map", "0:v:0", "-vf", f"fps={FPS},{fit_filter(fit, width, height)}", *audio,
         "-frames:v", str(frames), "-t", f"{frames / FPS:.6f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
        output, "Couldn't prepare the clip", timeout=3600,
    )


def cut_tail(conformed: str | Path, output: Path, *, frames: int) -> Path:
    """The conformed clip's last ``TAIL_FRAMES`` frames and their sound, losslessly, for the next scene."""
    if frames < TAIL_FRAMES:
        raise H3Error(f"The clip needs at least {TAIL_FRAMES} frames for a scene to continue it.")
    first = frames - TAIL_FRAMES
    # Fresh timestamps at 24 fps: keeping the trimmed ones leaves the last frame without a duration.
    return _ffmpeg(
        ["-i", str(conformed),
         "-vf", f"trim=start_frame={first}:end_frame={frames},setpts=N/({FPS}*TB)", "-r", str(FPS),
         "-af", f"atrim=start={first / FPS:.6f}:end={frames / FPS:.6f},asetpts=PTS-STARTPTS",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le"],
        output, "Couldn't cut the clip's ending", timeout=600,
    )


def still(source: str | Path, seconds: float, output: Path, *, width: int = 640) -> Path:
    """One frame as a small JPEG, for the Fit preview."""
    return _ffmpeg(
        ["-ss", f"{max(0.0, float(seconds)):.3f}", "-i", str(source),
         "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", "3"],
        output, f"No frame at {float(seconds):.2f} seconds of the clip", timeout=60,
    )
