"""Lip-sync post-processing for completed video dubs (window planning, H3 render, assembly)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Sequence
from uuid import uuid4

import numpy as np

from h3_app.drama3.serving.audio import encode_wav
from h3_app.errors import H3Error
from h3_app.generation.services import GenerationServices
from h3_app.jobs import check_cancelled
from h3_app.media_tools import has_encoder, probe_video_metadata
from h3_app.processes import run_media_process
from h3_app.provenance import copy_snapshot
from h3_app.workflows.lip_sync import (
    DEFAULT_AUTO_STRENGTH,
    DEFAULT_CROP_SIZE,
    FPS,
    MAX_FRAMES,
    build_auto_lip_sync_graph,
    build_lip_sync_graph,
    snap_h3_frames,
    validate_lip_sync_window,
)

from .alignment import FITTED_AUDIO_RATE, open_alignment

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
DEFAULT_SHOT_THRESHOLD = 0.35
DEFAULT_CONTEXT_SECONDS = 0.25
DEFAULT_MAX_GAP_SECONDS = 0.8
DEFAULT_MAX_WINDOW_SECONDS = 12.0
_MIN_TURN_SECONDS = 1.0 / FPS


@dataclass(frozen=True)
class LipSyncWindow:
    """One single-shot speaking window aligned to H3's 24 fps 17k+5 frame grid."""

    speaker: str
    start: float
    end: float
    active_start: float
    active_end: float
    first_frame: int
    source_frames: int
    render_frames: int
    active_start_frame: int
    active_end_frame: int
    face_target: tuple[int, int, int] | None = None


def fitted_dialogue(video: Path, speaker: str, first_frame: int, frames: int, *,
                    allow_silence: bool = False) -> np.ndarray:
    """Rebuild only this speaker's fitted speech on a window of the dub timeline."""
    if first_frame < 0 or not 1 <= frames <= MAX_FRAMES:
        raise H3Error("Choose a valid lip-sync window on the 24 fps render timeline.")
    if video.suffix.lower() != ".mp4" or not video.is_file():
        raise H3Error("Choose a completed dubbed MP4.")
    samples_per_frame = Fraction(FITTED_AUDIO_RATE, FPS)
    first = round(first_frame * samples_per_frame)
    stop = round((first_frame + frames) * samples_per_frame)
    dialogue = np.zeros(stop - first, dtype=np.float32)
    try:
        with open_alignment(video) as (saved, clips):
            lines = saved["lines"]
            for key, (start, end) in saved["spoken"].items():
                index = int(key)
                if not 0 <= index < len(lines):
                    raise ValueError("Invalid fitted line index")
                if lines[index]["speaker"] != speaker:
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError("Invalid fitted line timing")
                clip_start, clip_stop = round(start * FITTED_AUDIO_RATE), round(end * FITTED_AUDIO_RATE)
                left, right = max(first, clip_start), min(stop, clip_stop)
                if left >= right:
                    continue
                clip = clips[key]
                if clip.ndim != 1 or len(clip) != clip_stop - clip_start or not np.isfinite(clip).all():
                    raise ValueError("Invalid fitted dialogue samples")
                dialogue[left - first:right - first] += clip[left - clip_start:right - clip_start]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise H3Error("Saved fitted dialogue is missing or unreadable. Generate this dub again.") from exc
    if not allow_silence and not dialogue.any():
        raise H3Error("This window contains no audible fitted dialogue for the selected speaker.")
    return dialogue


def detect_shot_cuts(video: Path, *, threshold: float = DEFAULT_SHOT_THRESHOLD, timeout: int = 120) -> list[float]:
    """Return sorted hard-cut timestamps (seconds) detected via FFmpeg's scene filter."""
    if not video.is_file():
        raise H3Error("Source video for shot detection does not exist.")
    if not 0.0 < threshold < 1.0:
        raise H3Error("Shot-cut threshold must be between 0 and 1.")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(video),
        "-vf", f"select='gt(scene,{threshold:.3f})',showinfo",
        "-f", "null", "-",
    ]
    proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise H3Error(f"Shot-cut detection failed: {proc.stderr.strip()[-300:]}")
    cuts: list[float] = []
    for match in _PTS_TIME_RE.finditer(proc.stderr or ""):
        ts = float(match.group(1))
        if ts > 0.05 and (not cuts or ts - cuts[-1] > 0.05):
            cuts.append(round(ts, 4))
    return cuts


@dataclass(frozen=True)
class _Turn:
    speaker: str
    start: float
    end: float
    shot_start: float
    shot_end: float
    visual_hint: dict[str, Any] | None = None


def _drop_cross_speaker_overlaps(turns: list[_Turn]) -> list[_Turn]:
    """Keep only intervals where a single speaker is talking, dropping ambiguous overlaps."""
    if len(turns) <= 1:
        return turns
    boundaries = sorted({t.start for t in turns} | {t.end for t in turns})
    clean: list[_Turn] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < _MIN_TURN_SECONDS:
            continue
        active = [t for t in turns if t.start <= left + 1e-6 and t.end >= right - 1e-6]
        speakers = {t.speaker for t in active}
        if len(speakers) != 1:
            continue
        chosen = active[0]
        hint = next((t.visual_hint for t in active if t.visual_hint is not None), None)
        if (clean and clean[-1].speaker == chosen.speaker
                and clean[-1].shot_start == chosen.shot_start
                and abs(clean[-1].end - left) < 1e-4):
            prev = clean[-1]
            clean[-1] = _Turn(
                speaker=prev.speaker, start=prev.start, end=right,
                shot_start=prev.shot_start, shot_end=prev.shot_end,
                visual_hint=prev.visual_hint or hint,
            )
        else:
            clean.append(_Turn(
                speaker=chosen.speaker, start=left, end=right,
                shot_start=chosen.shot_start, shot_end=chosen.shot_end,
                visual_hint=hint,
            ))
    return clean


def _hint_to_face_target(hint: dict[str, Any] | None, video_size: tuple[int, int] | None,
                         first_frame: int, active_start_frame: int, active_end_frame: int) -> tuple[int, int, int] | None:
    if not hint or video_size is None or hint.get("visibility") != "on_screen":
        return None
    center = hint.get("face_center")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    width, height = video_size
    x = max(0, min(width - 1, round(float(center[0]) * width)))
    y = max(0, min(height - 1, round(float(center[1]) * height)))
    hint_time = hint.get("time")
    if isinstance(hint_time, (int, float)) and math.isfinite(hint_time):
        frame = round(float(hint_time) * FPS) - first_frame
        frame = max(active_start_frame, min(active_end_frame - 1, frame))
    else:
        frame = (active_start_frame + active_end_frame) // 2
    return (x, y, frame)


def plan_lip_sync_windows(
    dub: Path,
    video_duration: float,
    *,
    video_size: tuple[int, int] | None = None,
    shot_cuts: Sequence[float] = (),
    speaker: str | None = None,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    max_window_seconds: float = DEFAULT_MAX_WINDOW_SECONDS,
    use_visual_hints: bool = True,
) -> list[LipSyncWindow]:
    """Group fitted speaking turns into single-shot, H3-grid render windows."""
    if not math.isfinite(video_duration) or video_duration <= _MIN_TURN_SECONDS:
        raise H3Error("Dubbed video duration is too short for lip sync.")
    max_allowed = (MAX_FRAMES - 1) / FPS
    max_window = min(float(max_window_seconds), max_allowed)
    if max_window < 1.0:
        raise H3Error("max_window_seconds must be at least 1 second.")

    try:
        raw_turns: list[tuple[float, float, str, dict[str, Any] | None]] = []
        with open_alignment(dub) as (saved, clips):
            lines = saved["lines"]
            clip_keys = set(clips.files)
            for key, (start, end) in saved["spoken"].items():
                index = int(key)
                if key not in clip_keys or not 0 <= index < len(lines):
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    continue
                line = lines[index]
                line_speaker = line["speaker"]
                if speaker is not None and line_speaker != speaker:
                    continue
                hint = line.get("visual_speaker") if use_visual_hints else None
                if isinstance(hint, dict) and hint.get("visibility") == "off_screen":
                    continue
                clamped_start = max(0.0, float(start))
                clamped_end = min(float(video_duration), float(end))
                if clamped_end - clamped_start < _MIN_TURN_SECONDS:
                    continue
                raw_turns.append((clamped_start, clamped_end, line_speaker, hint if isinstance(hint, dict) else None))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise H3Error("Saved fitted dialogue is missing or unreadable. Generate this dub again.") from exc

    if not raw_turns:
        return []

    cuts = sorted(c for c in shot_cuts if 0.0 < c < video_duration)
    shots = list(zip([0.0, *cuts], [*cuts, float(video_duration)]))

    # Split every turn cleanly at shot boundaries.
    shot_turns: list[_Turn] = []
    for start, end, line_speaker, hint in sorted(raw_turns, key=lambda item: (item[0], item[1])):
        for shot_start, shot_end in shots:
            left = max(start, shot_start)
            right = min(end, shot_end)
            if right - left >= _MIN_TURN_SECONDS:
                shot_turns.append(_Turn(
                    speaker=line_speaker, start=left, end=right,
                    shot_start=shot_start, shot_end=shot_end, visual_hint=hint,
                ))

    if speaker is None:
        shot_turns = _drop_cross_speaker_overlaps(shot_turns)
    if not shot_turns:
        return []

    # Group consecutive turns of the same speaker within the same shot, then split overly long groups.
    grouped: list[_Turn] = []
    for turn in shot_turns:
        if (grouped and grouped[-1].speaker == turn.speaker
                and grouped[-1].shot_start == turn.shot_start
                and turn.start - grouped[-1].end <= max_gap_seconds
                and turn.end - grouped[-1].start <= max_window):
            prev = grouped[-1]
            grouped[-1] = _Turn(
                speaker=prev.speaker, start=prev.start, end=max(prev.end, turn.end),
                shot_start=prev.shot_start, shot_end=prev.shot_end,
                visual_hint=prev.visual_hint or turn.visual_hint,
            )
        else:
            grouped.append(turn)

    chunks: list[_Turn] = []
    for turn in grouped:
        cursor = turn.start
        while turn.end - cursor > max_window + 1e-6:
            chunks.append(_Turn(
                speaker=turn.speaker, start=cursor, end=cursor + max_window,
                shot_start=turn.shot_start, shot_end=turn.shot_end, visual_hint=turn.visual_hint,
            ))
            cursor += max_window
        if turn.end - cursor >= _MIN_TURN_SECONDS:
            chunks.append(_Turn(
                speaker=turn.speaker, start=cursor, end=turn.end,
                shot_start=turn.shot_start, shot_end=turn.shot_end, visual_hint=turn.visual_hint,
            ))

    total_frames = max(1, math.ceil(video_duration * FPS))
    windows: list[LipSyncWindow] = []
    prev_last_frame = 0
    for idx, chunk in enumerate(chunks):
        next_active_start = chunks[idx + 1].start if idx + 1 < len(chunks) else video_duration
        win_start = max(chunk.shot_start, chunk.start - context_seconds)
        win_end = min(chunk.shot_end, chunk.end + context_seconds, next_active_start)
        if win_end - win_start > max_allowed:
            win_end = win_start + max_allowed

        first_frame = max(prev_last_frame, math.floor(win_start * FPS))
        last_frame = min(total_frames, max(first_frame + 1, math.ceil(win_end * FPS)))
        if last_frame - first_frame > MAX_FRAMES:
            last_frame = first_frame + MAX_FRAMES
        source_frames = last_frame - first_frame
        if source_frames <= 0:
            continue

        render_frames = snap_h3_frames(source_frames)
        active_start_frame = max(0, min(source_frames - 1, math.floor(chunk.start * FPS) - first_frame))
        active_end_frame = max(active_start_frame + 1, min(source_frames, math.ceil(chunk.end * FPS) - first_frame))
        face_target = _hint_to_face_target(
            chunk.visual_hint, video_size, first_frame, active_start_frame, active_end_frame,
        )
        windows.append(LipSyncWindow(
            speaker=chunk.speaker,
            start=first_frame / FPS,
            end=min(float(video_duration), last_frame / FPS),
            active_start=chunk.start,
            active_end=chunk.end,
            first_frame=first_frame,
            source_frames=source_frames,
            render_frames=render_frames,
            active_start_frame=active_start_frame,
            active_end_frame=active_end_frame,
            face_target=face_target,
        ))
        prev_last_frame = last_frame
    return windows


def _extract_window_slice(source: Path, window: LipSyncWindow, destination: Path, *, timeout: int = 180) -> Path:
    """Slice a single 24 fps CFR video window, cloning the last frame to match H3's 17k+5 grid."""
    pad_frames = max(0, window.render_frames - window.source_frames)
    duration = max(window.source_frames / FPS, window.end - window.start)
    filters = [f"fps={FPS}", "format=yuv420p"]
    if pad_frames > 0:
        filters.append(f"tpad=stop_mode=clone:stop={pad_frames}")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{window.start:.6f}", "-i", str(source),
        "-t", f"{duration:.6f}",
        "-an", "-sn",
        "-vf", ",".join(filters),
        "-frames:v", str(window.render_frames),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14",
        "-pix_fmt", "yuv420p",
        str(destination),
    ]
    proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout, partial_outputs=(destination,))
    if proc.returncode != 0 or not destination.is_file():
        raise H3Error(f"Could not slice lip-sync window: {proc.stderr.strip()}")
    return destination


def assemble_lip_sync_video(
    dub: Path,
    rendered_windows: Sequence[tuple[LipSyncWindow, Path]],
    destination: Path,
    *,
    timeout: int = 600,
) -> Path:
    """Splice rendered lip-sync windows onto the full video and copy audio/subtitle tracks from dub."""
    if not rendered_windows:
        raise H3Error("No rendered lip-sync windows to assemble.")
    meta = probe_video_metadata(dub)
    fps_str = f"{meta.fps:.6f}".rstrip("0").rstrip(".")
    min_gap = 0.5 / meta.fps

    ordered = sorted(rendered_windows, key=lambda item: item[0].start)
    inputs = ["-i", str(dub)]
    filter_parts: list[str] = []
    concat_labels: list[str] = []
    cursor = 0.0
    seg_index = 0

    def add_source_segment(seg_start: float, seg_end: float) -> None:
        nonlocal seg_index
        if seg_end - seg_start < min_gap:
            return
        label = f"v{seg_index}"
        seg_index += 1
        filter_parts.append(
            f"[0:v]trim=start={seg_start:.6f}:end={seg_end:.6f},setpts=PTS-STARTPTS,"
            f"fps={fps_str},format=yuv420p[{label}]"
        )
        concat_labels.append(f"[{label}]")

    for input_idx, (window, rendered_path) in enumerate(ordered, start=1):
        if not rendered_path.is_file():
            raise H3Error(f"Rendered lip-sync window is missing: {rendered_path.name}")
        win_start = max(cursor, window.start)
        win_end = min(meta.duration, window.end)
        if win_end - win_start < min_gap:
            continue
        add_source_segment(cursor, win_start)
        inputs += ["-i", str(rendered_path)]
        label = f"v{seg_index}"
        seg_index += 1
        win_duration = win_end - win_start
        filter_parts.append(
            f"[{input_idx}:v]trim=start=0:end={win_duration:.6f},setpts=PTS-STARTPTS,"
            f"scale={meta.width}:{meta.height}:flags=lanczos,fps={fps_str},format=yuv420p[{label}]"
        )
        concat_labels.append(f"[{label}]")
        cursor = win_end

    add_source_segment(cursor, meta.duration)
    if not concat_labels:
        raise H3Error("No video segments were produced for lip-sync assembly.")

    filter_parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=1:a=0[vout]")
    encoder = "h264_nvenc" if has_encoder("h264_nvenc") else "libx264"
    video_args = (
        ["-c:v", encoder, "-preset", "p5", "-cq", "19"]
        if encoder == "h264_nvenc"
        else ["-c:v", encoder, "-preset", "medium", "-crf", "18"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".partial")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "0:a?",
        "-map", "0:s?",
        *video_args,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-c:s", "copy",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(pending),
    ]
    try:
        proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout, partial_outputs=(pending,))
        if proc.returncode != 0 or not pending.is_file():
            raise H3Error(f"Could not assemble lip-synced video: {proc.stderr.strip()}")
        check_cancelled()
        pending.replace(destination)
        copy_snapshot(dub, destination)
        return destination
    finally:
        pending.unlink(missing_ok=True)


def render_dub_lip_sync(
    *,
    dub: Path,
    services: GenerationServices,
    speaker: str | None = None,
    refinement: dict[str, Any] | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    strength: float = DEFAULT_AUTO_STRENGTH,
    seed: int = 0,
    output: Path | None = None,
    shot_threshold: float = DEFAULT_SHOT_THRESHOLD,
) -> Iterator[str | Path]:
    """Automatically track faces, render H3 mouth patches per speaking window, and assemble a full MP4."""
    yield "Inspecting the dubbed video"
    check_cancelled()
    if dub.suffix.lower() != ".mp4" or not dub.is_file():
        raise H3Error("Choose a completed dubbed MP4.")
    meta = services.media.probe_video_metadata(dub)
    yield "Detecting shot cuts"
    check_cancelled()
    cuts = detect_shot_cuts(dub, threshold=shot_threshold)
    windows = plan_lip_sync_windows(
        dub, meta.duration, video_size=(meta.width, meta.height), shot_cuts=cuts, speaker=speaker,
    )
    if not windows:
        raise H3Error("No eligible speaking windows were found for lip sync.")

    models = services.models.load_model_config()
    rendered_windows: list[tuple[LipSyncWindow, Path]] = []
    with TemporaryDirectory(prefix="h3-lip-sync-auto-") as directory:
        work_dir = Path(directory)
        total = len(windows)
        for index, window in enumerate(windows, start=1):
            check_cancelled()
            yield f"Preparing lip-sync window {index}/{total}"
            dialogue = fitted_dialogue(dub, window.speaker, window.first_frame, window.render_frames)
            wav_path = work_dir / f"window_{index:03d}.wav"
            wav_path.write_bytes(encode_wav(dialogue, FITTED_AUDIO_RATE))
            slice_path = _extract_window_slice(dub, window, work_dir / f"window_{index:03d}.mp4")

            staged_video = services.media.stage_file(str(slice_path), "lip_sync")
            staged_dialogue = services.media.stage_file(str(wav_path), "lip_sync")
            client_id = uuid4().hex
            config = {
                "model_profile": models.default_profile,
                **(refinement or {}),
                "_source_duration": window.render_frames / FPS,
            }
            graph = build_auto_lip_sync_graph(
                video=staged_video,
                dialogue=staged_dialogue,
                frames=window.render_frames,
                models=models,
                refinement=config,
                output_prefix=f"dubbing/.lip-sync-window/{client_id}",
                crop_size=crop_size,
                seed=seed + index - 1,
                strength=strength,
                active_start=window.active_start_frame,
                active_end=window.active_end_frame,
                face_target=window.face_target,
            )
            yield f"Rendering lip-sync window {index}/{total}"
            check_cancelled()
            queued_at = time.time()
            prompt_id = services.execution.submit_prompt(graph, client_id)
            for stage, _completed, _total, _step, _step_total in services.execution.poll_comfy_progress(prompt_id, graph):
                check_cancelled()
                yield f"Window {index}/{total}: {stage}"
            rendered_path = services.media.resolve_output(services.execution.wait_for_history(prompt_id), queued_at)
            check_cancelled()
            if not rendered_path.name.startswith(f"{client_id}_"):
                raise H3Error(f"Window {index}/{total} finished, but its output could not be located.")
            rendered_windows.append((window, rendered_path))

        yield "Assembling the lip-synced video"
        check_cancelled()
        destination = output or dub.with_name(f"{dub.stem}.lipsync.mp4")
        assembled = assemble_lip_sync_video(dub, rendered_windows, destination)
    yield assembled


def render_lip_sync_preview(*, dub: Path, speaker: str, first_frame: int, crop: Path, mask: Path,
                            services: GenerationServices, refinement: dict[str, Any] | None = None,
                            strength: float = 0.4) -> Iterator[str | Path]:
    """Run the single-window prototype on a supplied 24 fps face crop and grayscale mask.

    The crop and mask start at ``first_frame`` on the dub's 24 fps timeline, share
    its H3-grid frame count, and use the same crop coordinates. The output is a
    crop preview with fitted speech.
    """
    yield "Checking the lip-sync window"
    check_cancelled()
    picture = services.media.probe_video_metadata(crop)
    matte = services.media.probe_video_metadata(mask)
    width, height, frames = picture.width, picture.height, picture.frame_count
    validate_lip_sync_window(width, height, frames, strength)
    if (picture.fps != FPS or matte.fps != FPS
            or (matte.width, matte.height, matte.frame_count) != (width, height, frames)):
        raise H3Error("The face crop and mouth mask must have matching dimensions and frame counts at 24 fps.")
    dialogue = fitted_dialogue(dub, speaker, first_frame, frames)
    models = services.models.load_model_config()
    config = {"model_profile": models.default_profile, **(refinement or {}), "_source_duration": frames / FPS}
    client_id = uuid4().hex
    with TemporaryDirectory(prefix="h3-lip-sync-") as directory:
        wav = Path(directory) / "dialogue.wav"
        wav.write_bytes(encode_wav(dialogue, FITTED_AUDIO_RATE))
        staged_dialogue = services.media.stage_file(str(wav), "lip_sync")
    graph = build_lip_sync_graph(
        crop=services.media.stage_file(str(crop), "lip_sync"),
        mask=services.media.stage_file(str(mask), "lip_sync"), dialogue=staged_dialogue,
        width=width, height=height, frames=frames, models=models, refinement=config,
        output_prefix=f"dubbing/.lip-sync-preview/{client_id}", strength=strength,
    )
    yield "Rendering the H3 mouth patch"
    check_cancelled()
    queued_at = time.time()
    prompt_id = services.execution.submit_prompt(graph, client_id)
    for stage, _completed, _total, _step, _step_total in services.execution.poll_comfy_progress(prompt_id, graph):
        check_cancelled()
        yield stage
    output = services.media.resolve_output(services.execution.wait_for_history(prompt_id), queued_at)
    check_cancelled()
    if not output.name.startswith(f"{client_id}_"):
        raise H3Error("The render finished, but its lip-sync preview could not be located.")
    yield output
