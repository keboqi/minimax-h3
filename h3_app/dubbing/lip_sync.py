"""Lip-sync post-processing for completed video dubs (window planning, H3 render, assembly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4

import numpy as np

from h3_models import H3_TEXT_ENCODER_CHOICES, MODEL_SPECS

from h3_app.drama3.serving.audio import encode_wav
from h3_app.errors import H3Error
from h3_app.generation.services import GenerationServices, ModelsServices
from h3_app.jobs import JobCancelled, check_cancelled
from h3_app.llm import decisions
from h3_app.media_tools import has_encoder, probe_video_metadata
from h3_app.processes import run_media_process
from h3_app.provenance import copy_snapshot
from h3_app.refinement import resolve_face_refinement
from h3_app.workflows.lip_sync import (
    DEFAULT_AUTO_FEATHER,
    DEFAULT_AUTO_STRENGTH,
    DEFAULT_CROP_SIZE,
    DEFAULT_LIP_SYNC_CROP_FACTOR,
    FPS,
    MAX_FRAMES,
    build_auto_lip_sync_graph,
    build_lip_sync_graph,
    snap_h3_frames,
    validate_lip_sync_window,
)

from . import lip_sync_runtime
from .alignment import FITTED_AUDIO_RATE, open_alignment
from .lip_sync_match import (
    ActiveSpeakerScanner,
    FaceTrack,
    Shot,
    ShotSpeakerAssignment,
    SourceScan,
    TrackObservation,
    assign_speakers_to_shots,
    get_or_scan_source,
    resolve_uncertain_with_gemini_decision,
    uncertain_assignments,
    video_fingerprint,
)
from .pipeline import blocking
from .results import PLAYER_DIR, SCAN_CACHE_DIR, load_player, player_folder

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
DEFAULT_SHOT_THRESHOLD = 0.35
DEFAULT_CONTEXT_SECONDS = 0.25
DEFAULT_MAX_GAP_SECONDS = 0.8
DEFAULT_MAX_WINDOW_SECONDS = 12.0
DEFAULT_MIN_FACE_SIZE = 0.0
DEFAULT_MAX_FACE_SIZE = 1.0
LIP_SYNC_ENGINES = ("h3", "musetalk", "latentsync")
DEFAULT_LIP_SYNC_ENGINE = "h3"
DEFAULT_MUSETALK_BBOX_SHIFT = 0
DEFAULT_LATENTSYNC_STEPS = 20
DEFAULT_LATENTSYNC_GUIDANCE = 1.5
_MIN_TURN_SECONDS = 1.0 / FPS
WINDOW_CACHE_DIR = "lip_sync_cache"
TAKES_DIR = "lip_sync_takes"
TAKES_FILE = "lip_sync_takes.json"
TAKE_ORIGINAL = "original"
REPORT_FILE = "lip_sync_report.json"
PLAN_FILE = "lip_sync_plan.json"
OVERRIDES_FILE = "lip_sync_overrides.json"
_NO_FACE_MARKERS = (
    "no detected face",
    "no face detected",
    "no faces detected",
    "face not detected",
    "no editable pixels",
)


def validate_lip_sync_engine(engine: str | None) -> str:
    """Validate and normalize a lip-sync rendering engine name."""
    normalized = DEFAULT_LIP_SYNC_ENGINE if engine is None else str(engine).strip().lower()
    if normalized not in LIP_SYNC_ENGINES:
        raise H3Error(f"Unsupported lip-sync engine: {engine!r}. Choose one of {', '.join(LIP_SYNC_ENGINES)}.")
    return normalized


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise H3Error(f"{name} must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise H3Error(f"{name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise H3Error(f"{name} must be a finite number.")
    return parsed


def validate_lip_sync_options(
    *,
    bbox_shift: Any = None,
    inference_steps: Any = None,
    guidance_scale: Any = None,
) -> dict[str, Any]:
    """Validate and clamp optional engine-specific lip-sync parameters."""
    options: dict[str, Any] = {}
    if bbox_shift is not None:
        options["bbox_shift"] = max(-15, min(15, int(round(_finite_number(bbox_shift, "Mouth box shift")))))
    if inference_steps is not None:
        options["inference_steps"] = max(5, min(50, int(round(_finite_number(inference_steps, "Diffusion steps")))))
    if guidance_scale is not None:
        options["guidance_scale"] = max(1.0, min(3.0, _finite_number(guidance_scale, "Guidance scale")))
    return options


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
    shot_index: int = 0
    face_size: float | None = None
    manual: bool = False


def _parse_face_size_bound(value: float | int | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise H3Error("Face size bounds must be numbers between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise H3Error("Face size bounds must be numbers between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise H3Error("Face size bounds must be between 0.0 and 1.0 (fraction of frame height).")
    return parsed


def validate_face_size_range(
    min_face_size: float | None,
    max_face_size: float | None,
) -> tuple[float, float]:
    """Validate normalized [0.0..1.0] face-height bounds for lip-sync window filtering."""
    low = _parse_face_size_bound(min_face_size, DEFAULT_MIN_FACE_SIZE)
    high = _parse_face_size_bound(max_face_size, DEFAULT_MAX_FACE_SIZE)
    if low > high:
        raise H3Error("Minimum face size cannot exceed maximum face size.")
    return low, high


def window_matches_face_size(
    window: LipSyncWindow,
    *,
    min_face_size: float = DEFAULT_MIN_FACE_SIZE,
    max_face_size: float = DEFAULT_MAX_FACE_SIZE,
) -> bool:
    """Return True when the window passes the face-size filter (manual selections always pass)."""
    if window.manual or window.face_size is None:
        return True
    return min_face_size - 1e-6 <= float(window.face_size) <= max_face_size + 1e-6


def filter_lip_sync_windows(
    windows: Sequence[LipSyncWindow],
    *,
    min_face_size: float | None = None,
    max_face_size: float | None = None,
) -> list[LipSyncWindow]:
    """Filter lip-sync windows to a face-size range while keeping all manual overrides."""
    low, high = validate_face_size_range(min_face_size, max_face_size)
    return [
        window for window in windows
        if window_matches_face_size(window, min_face_size=low, max_face_size=high)
    ]


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
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-threads", "2", "-filter_threads", "2", "-i", str(video),
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
    shot_index: int = 0
    visual_hint: dict[str, Any] | None = None
    visual_hints: tuple[dict[str, Any], ...] = ()


def _merge_hints(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in (*first, *second):
        marker = id(item)
        if marker not in seen:
            seen.add(marker)
            merged.append(item)
    return tuple(merged)


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
        hints = _merge_hints((), [h for t in active for h in t.visual_hints])
        if (clean and clean[-1].speaker == chosen.speaker
                and clean[-1].shot_index == chosen.shot_index
                and abs(clean[-1].end - left) < 1e-4):
            prev = clean[-1]
            clean[-1] = _Turn(
                speaker=prev.speaker, start=prev.start, end=right,
                shot_start=prev.shot_start, shot_end=prev.shot_end,
                shot_index=prev.shot_index, visual_hint=prev.visual_hint or hint,
                visual_hints=_merge_hints(prev.visual_hints, hints),
            )
        else:
            clean.append(_Turn(
                speaker=chosen.speaker, start=left, end=right,
                shot_start=chosen.shot_start, shot_end=chosen.shot_end,
                shot_index=chosen.shot_index, visual_hint=hint, visual_hints=hints,
            ))
    return clean


def _center_to_face_target(center: tuple[float, float] | Sequence[float] | None, ref_time: float | None,
                           video_size: tuple[int, int] | None, first_frame: int,
                           active_start_frame: int, active_end_frame: int) -> tuple[int, int, int] | None:
    if center is None or video_size is None or len(center) != 2:
        return None
    width, height = video_size
    x = max(0, min(width - 1, round(float(center[0]) * width)))
    y = max(0, min(height - 1, round(float(center[1]) * height)))
    if isinstance(ref_time, (int, float)) and math.isfinite(ref_time):
        frame = round(float(ref_time) * FPS) - first_frame
        frame = max(active_start_frame, min(active_end_frame - 1, frame))
    else:
        frame = (active_start_frame + active_end_frame) // 2
    return (x, y, frame)


def _hint_to_face_target(hint: dict[str, Any] | None, video_size: tuple[int, int] | None,
                         first_frame: int, active_start_frame: int, active_end_frame: int) -> tuple[int, int, int] | None:
    if not hint or hint.get("visibility") != "on_screen":
        return None
    return _center_to_face_target(
        hint.get("face_center"), hint.get("time"), video_size, first_frame, active_start_frame, active_end_frame,
    )


def _line_visual_hints(
    line: dict[str, Any],
    speaker: str,
    start: float,
    end: float,
    visual_speakers: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    collected: list[dict[str, Any]] = []
    hint = line.get("visual_speaker")
    if isinstance(hint, dict):
        collected.append(hint)
    if visual_speakers:
        line_start = float(line["start"]) if isinstance(line.get("start"), (int, float)) else float(start)
        line_end = float(line["end"]) if isinstance(line.get("end"), (int, float)) else float(end)
        for item in visual_speakers:
            if (isinstance(item, dict)
                    and item.get("speaker") == speaker
                    and isinstance(item.get("time"), (int, float))
                    and line_start - 0.1 <= float(item["time"]) <= line_end + 0.1):
                collected.append(item)
    if not collected:
        return None, ()
    primary = (
        hint if isinstance(hint, dict) else
        next((item for item in collected if item.get("visibility") == "on_screen" and item.get("face_center")),
             collected[0])
    )
    return primary, tuple(collected)


def _best_in_window_hint(
    hints: Sequence[dict[str, Any]],
    first_frame: int,
    last_frame: int,
    mid_time: float,
) -> dict[str, Any] | None:
    candidates = [
        item for item in hints
        if item.get("visibility") == "on_screen"
        and item.get("face_center")
        and isinstance(item.get("time"), (int, float))
        and math.isfinite(item["time"])
        and first_frame <= round(float(item["time"]) * FPS) < last_frame
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(float(item["time"]) - mid_time))


def _subtract_intervals(
    start: float,
    end: float,
    blocked: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for b_start, b_end in blocked:
        next_pieces: list[tuple[float, float]] = []
        for left, right in pieces:
            if b_end <= left + 1e-6 or b_start >= right - 1e-6:
                next_pieces.append((left, right))
            else:
                if b_start - left >= _MIN_TURN_SECONDS:
                    next_pieces.append((left, b_start))
                if right - b_end >= _MIN_TURN_SECONDS:
                    next_pieces.append((b_end, right))
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def resolve_lip_sync_crop_size(
    window: LipSyncWindow,
    video_size: tuple[int, int] | None,
    base_crop_size: int = DEFAULT_CROP_SIZE,
) -> int:
    """Scale H3 lip-sync crop canvas up for large close-ups unless a custom crop size was requested."""
    if (
        int(base_crop_size) != DEFAULT_CROP_SIZE
        or window.face_size is None
        or video_size is None
        or video_size[1] <= 0
    ):
        return int(base_crop_size)
    needed = float(window.face_size) * float(video_size[1]) * DEFAULT_LIP_SYNC_CROP_FACTOR
    if needed <= 640:
        return 640
    if needed <= 768:
        return 768
    return 960


def plan_lip_sync_windows(
    dub: Path,
    video_duration: float,
    *,
    video_size: tuple[int, int] | None = None,
    shot_cuts: Sequence[float] = (),
    speaker: str | None = None,
    assignments: dict[tuple[int, str], ShotSpeakerAssignment] | None = None,
    visual_speakers: Sequence[dict[str, Any]] | None = None,
    tracks: dict[tuple[int, int], FaceTrack] | None = None,
    require_assignment: bool = False,
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
        raw_turns: list[tuple[float, float, str, dict[str, Any] | None, tuple[dict[str, Any], ...]]] = []
        raw_silent: list[tuple[float, float, str, dict[str, Any] | None, tuple[dict[str, Any], ...]]] = []
        with open_alignment(dub) as (saved, clips):
            lines = saved["lines"]
            saved_hints = list(saved.get("visual_speakers") or ())
            effective_hints = saved_hints if saved_hints else list(visual_speakers or ())
            clip_keys = set(clips.files)
            for key, (start, end) in saved["spoken"].items():
                index = int(key)
                if key not in clip_keys or not 0 <= index < len(lines):
                    continue
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    continue
                line = lines[index]
                line_speaker = line["speaker"]
                hint, all_hints = (
                    _line_visual_hints(line, line_speaker, start, end, effective_hints)
                    if use_visual_hints else (None, ())
                )
                hint_dict = hint if isinstance(hint, dict) else None
                clamped_start = max(0.0, float(start))
                clamped_end = min(float(video_duration), float(end))
                if clamped_end - clamped_start < _MIN_TURN_SECONDS:
                    continue
                raw_turns.append((
                    clamped_start, clamped_end, line_speaker,
                    hint_dict, all_hints,
                ))
                orig_start, orig_end = line.get("start"), line.get("end")
                if (
                    not isinstance(orig_start, bool) and not isinstance(orig_end, bool)
                    and isinstance(orig_start, (int, float)) and isinstance(orig_end, (int, float))
                    and math.isfinite(orig_start) and math.isfinite(orig_end)
                    and 0.0 <= float(orig_start) < float(orig_end)
                ):
                    o_start = max(0.0, float(orig_start))
                    o_end = min(float(video_duration), float(orig_end))
                    if clamped_start - o_start >= _MIN_TURN_SECONDS:
                        raw_silent.append((o_start, clamped_start, line_speaker, hint_dict, all_hints))
                    if o_end - clamped_end >= _MIN_TURN_SECONDS:
                        raw_silent.append((clamped_end, o_end, line_speaker, hint_dict, all_hints))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise H3Error("Saved fitted dialogue is missing or unreadable. Generate this dub again.") from exc

    if not raw_turns:
        return []

    cuts = sorted(c for c in shot_cuts if 0.0 < c < video_duration)
    shots = list(enumerate(zip([0.0, *cuts], [*cuts, float(video_duration)])))

    # Split every turn cleanly at shot boundaries, applying shot-level assignments when supplied.
    shot_turns: list[_Turn] = []
    for start, end, line_speaker, hint, all_hints in sorted(raw_turns, key=lambda item: (item[0], item[1])):
        for shot_index, (shot_start, shot_end) in shots:
            left = max(start, shot_start)
            right = min(end, shot_end)
            if right - left < _MIN_TURN_SECONDS:
                continue
            shot_turns.append(_Turn(
                speaker=line_speaker, start=left, end=right,
                shot_start=shot_start, shot_end=shot_end, shot_index=shot_index,
                visual_hint=hint, visual_hints=all_hints,
            ))

    shot_turns = _drop_cross_speaker_overlaps(shot_turns)
    if not shot_turns:
        return []

    # Cover silent original-speech leads/tails where nobody else is audibly speaking in the dub,
    # so the source mouth does not flap silently on translated lines that start late or finish early.
    if raw_silent:
        audible_spans = [(left, right) for left, right, _, _, _ in raw_turns]
        silent_turns: list[_Turn] = []
        for s_start, s_end, s_speaker, hint, all_hints in raw_silent:
            for left, right in _subtract_intervals(s_start, s_end, audible_spans):
                for shot_index, (shot_start, shot_end) in shots:
                    s_left = max(left, shot_start)
                    s_right = min(right, shot_end)
                    if s_right - s_left < _MIN_TURN_SECONDS:
                        continue
                    silent_turns.append(_Turn(
                        speaker=s_speaker, start=s_left, end=s_right,
                        shot_start=shot_start, shot_end=shot_end, shot_index=shot_index,
                        visual_hint=hint, visual_hints=all_hints,
                    ))
        silent_turns = _drop_cross_speaker_overlaps(silent_turns)
        if silent_turns:
            shot_turns = sorted([*shot_turns, *silent_turns], key=lambda item: (item.start, item.end, item.speaker))

    # Group consecutive turns of the same speaker within the same shot, then split overly long groups.
    grouped: list[_Turn] = []
    for turn in shot_turns:
        assigned = assignments.get((turn.shot_index, turn.speaker)) if assignments is not None else None
        if speaker is not None and turn.speaker != speaker:
            continue
        if require_assignment and assigned is None:
            continue
        if assigned and assigned.status in ("uncertain", "off_screen"):
            continue
        if (turn.visual_hint and turn.visual_hint.get("visibility") == "off_screen"
                and (assigned is None or assigned.status != "manual")):
            continue
        if (grouped and grouped[-1].speaker == turn.speaker
                and grouped[-1].shot_index == turn.shot_index
                and turn.start - grouped[-1].end <= max_gap_seconds
                and turn.end - grouped[-1].start <= max_window
                and not any(other != turn.speaker and left < turn.start and right > grouped[-1].end
                            for left, right, other, _, _ in raw_turns)):
            prev = grouped[-1]
            grouped[-1] = _Turn(
                speaker=prev.speaker, start=prev.start, end=max(prev.end, turn.end),
                shot_start=prev.shot_start, shot_end=prev.shot_end,
                shot_index=prev.shot_index, visual_hint=prev.visual_hint or turn.visual_hint,
                visual_hints=_merge_hints(prev.visual_hints, turn.visual_hints),
            )
        else:
            grouped.append(turn)

    chunks: list[_Turn] = []
    for turn in grouped:
        cursor = turn.start
        while turn.end - cursor > max_window + 1e-6:
            chunks.append(_Turn(
                speaker=turn.speaker, start=cursor, end=cursor + max_window,
                shot_start=turn.shot_start, shot_end=turn.shot_end,
                shot_index=turn.shot_index, visual_hint=turn.visual_hint,
                visual_hints=turn.visual_hints,
            ))
            cursor += max_window
        if turn.end - cursor >= _MIN_TURN_SECONDS:
            chunks.append(_Turn(
                speaker=turn.speaker, start=cursor, end=turn.end,
                shot_start=turn.shot_start, shot_end=turn.shot_end,
                shot_index=turn.shot_index, visual_hint=turn.visual_hint,
                visual_hints=turn.visual_hints,
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
        mid = (chunk.start + chunk.end) / 2
        face_target = None
        face_size = None
        manual = False
        if assignments is not None:
            assigned = assignments.get((chunk.shot_index, chunk.speaker))
            manual = bool(assigned is not None and assigned.status == "manual")
            if assigned and assigned.track_id is not None and tracks is not None:
                track = tracks.get((chunk.shot_index, assigned.track_id))
                samples = [obs for obs in track.observations
                           if first_frame <= round(obs.time * FPS) < last_frame] if track else []
                observation = min(samples, key=lambda obs: abs(obs.time - mid), default=None)
                if observation is not None:
                    heights = sorted(obs.box[3] for obs in samples)
                    face_size = round(float(heights[len(heights) // 2]), 4)
                    face_target = _center_to_face_target(
                        observation.center, observation.time, video_size, first_frame, 0, source_frames,
                    )
                elif track and track.observations:
                    nearest = min(track.observations, key=lambda obs: abs(obs.time - mid))
                    gap = max(0.0, first_frame / FPS - nearest.time, nearest.time - last_frame / FPS)
                    if not manual and gap > 0.5:
                        continue
                    heights = sorted(obs.box[3] for obs in track.observations)
                    face_size = round(float(heights[len(heights) // 2]), 4)
                    face_target = _center_to_face_target(
                        nearest.center, None, video_size, first_frame, active_start_frame, active_end_frame,
                    )
                else:
                    continue
            elif assigned and assigned.face_center is not None:
                ref = assigned.reference_time
                if ref is not None and not first_frame <= round(ref * FPS) < last_frame:
                    in_window_hint = _best_in_window_hint(chunk.visual_hints, first_frame, last_frame, mid)
                    if in_window_hint is not None:
                        face_target = _hint_to_face_target(in_window_hint, video_size, first_frame, 0, source_frames)
                    else:
                        face_target = _center_to_face_target(
                            assigned.face_center, None, video_size, first_frame, active_start_frame, active_end_frame,
                        )
                else:
                    face_target = _center_to_face_target(
                        assigned.face_center, ref, video_size, first_frame, 0, source_frames,
                    )
        if face_target is None:
            in_window_hint = _best_in_window_hint(chunk.visual_hints, first_frame, last_frame, mid)
            face_target = _hint_to_face_target(
                in_window_hint or chunk.visual_hint, video_size, first_frame, active_start_frame, active_end_frame,
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
            shot_index=chunk.shot_index,
            face_size=face_size,
            manual=manual,
        ))
        prev_last_frame = last_frame
    return windows


def _extract_window_slice(
    source: Path,
    window: LipSyncWindow,
    destination: Path,
    *,
    pad: bool = True,
    timeout: int = 180,
) -> Path:
    """Slice a single 24 fps CFR video window, optionally cloning trailing frames to H3's grid."""
    target_frames = window.render_frames if pad else window.source_frames
    duration = max(window.source_frames / FPS, window.end - window.start)
    # Fast-seek 2s before the window, then frame-accurately trim inside the filtergraph
    # so the slice aligns to the exact same frame as assemble_lip_sync_video.
    fast_seek = max(0.0, window.start - 2.0)
    trim_offset = max(0.0, window.start - fast_seek)
    # Give trim a tiny safety margin so fps=24 never starves on the last real frame when
    # resampling fractional-fps sources; trim=end_frame then clamps to source_frames so we
    # never bleed into the next shot, and tpad clones the final frame so -frames:v always
    # produces exactly target_frames even when the video stream ends slightly early.
    trim_duration = duration + (2.0 / FPS)
    filters = [
        f"trim=start={trim_offset:.6f}:duration={trim_duration:.6f}",
        "setpts=PTS-STARTPTS",
        f"fps={FPS}",
        f"trim=end_frame={window.source_frames}",
        "format=yuv420p",
        f"tpad=stop_mode=clone:stop={target_frames}",
    ]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *(["-ss", f"{fast_seek:.6f}"] if fast_seek > 0.0 else []),
        "-i", str(source),
        "-an", "-sn",
        "-vf", ",".join(filters),
        "-frames:v", str(target_frames),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14",
        "-pix_fmt", "yuv420p",
        str(destination),
    ]
    proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout, partial_outputs=(destination,))
    if proc.returncode != 0 or not destination.is_file():
        raise H3Error(f"Could not slice lip-sync window: {proc.stderr.strip()}")
    actual_frames = probe_video_metadata(destination).frame_count
    if actual_frames != target_frames:
        raise H3Error(
            f"Lip-sync window slice produced {actual_frames} frame(s), expected {target_frames} "
            f"(shot {window.shot_index + 1}, {window.start:.2f}s–{window.end:.2f}s)."
        )
    return destination


def _build_splice_filter_graph(
    dub: Path,
    rendered_windows: Sequence[tuple[LipSyncWindow, Path]],
    meta: Any,
    start_frame: int,
    end_frame: int,
) -> tuple[list[str], list[str], str]:
    """Build ffmpeg inputs and filter_complex graph splicing rendered windows onto [start_frame, end_frame)."""
    rate = Fraction(meta.fps).limit_denominator(100000)
    fps_str = f"{rate.numerator}/{rate.denominator}"
    time_base = f"{rate.denominator}/{rate.numerator}"
    # Cover rounding at the splice boundary, including a one-frame patch downsampled below 24 fps.
    patch_padding = max(1, math.ceil(FPS / rate))

    ordered = sorted(rendered_windows, key=lambda item: item[0].start)
    inputs = ["-i", str(dub)]
    filter_parts: list[str] = []
    concat_labels: list[str] = []
    cursor = start_frame
    seg_index = 0

    def add_source_segment(seg_start: int, seg_end: int) -> None:
        nonlocal seg_index
        if seg_end <= seg_start:
            return
        label = f"v{seg_index}"
        seg_index += 1
        filter_parts.append(
            f"[0:v]fps={fps_str}:start_time=0,"
            f"trim=start_frame={seg_start}:end_frame={seg_end},setpts=PTS-STARTPTS,format=yuv420p[{label}]"
        )
        concat_labels.append(f"[{label}]")

    for window, rendered_path in ordered:
        if not rendered_path.is_file():
            raise H3Error(f"Rendered lip-sync window is missing: {rendered_path.name}")
        patch_start = round(window.start * rate)
        win_start = max(cursor, patch_start)
        win_end = min(end_frame, round(window.end * rate))
        if win_end <= win_start:
            continue
        add_source_segment(cursor, win_start)
        input_idx = len(inputs) // 2
        inputs += ["-i", str(rendered_path)]
        label = f"v{seg_index}"
        seg_index += 1
        patch_offset = win_start - patch_start
        patch_end = patch_offset + win_end - win_start
        filter_parts.append(
            f"[{input_idx}:v]trim=end_frame={window.source_frames},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop={patch_padding},fps={fps_str}:start_time=0,"
            f"trim=start_frame={patch_offset}:end_frame={patch_end},setpts=PTS-STARTPTS,"
            f"scale={meta.width}:{meta.height}:flags=lanczos,format=yuv420p[{label}]"
        )
        concat_labels.append(f"[{label}]")
        cursor = win_end

    add_source_segment(cursor, end_frame)
    if not concat_labels:
        raise H3Error("No video segments were produced for lip-sync assembly.")
    # Number every frame on the common output timeline so concat duration rounding cannot accumulate.
    filter_parts.append(
        f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=1:a=0,settb={time_base},setpts=N[vout]"
    )
    return inputs, filter_parts, fps_str


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
    rate = Fraction(meta.fps).limit_denominator(100000)
    total_frames = max(1, round(meta.duration * rate))
    inputs, filter_parts, fps_str = _build_splice_filter_graph(dub, rendered_windows, meta, 0, total_frames)
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
        "-r", fps_str,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-disposition:a:0", "default",
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


def window_cache_key(
    dub: Path,
    window: LipSyncWindow,
    dialogue: np.ndarray,
    *,
    crop_size: int,
    strength: float,
    seed: int,
    refinement: dict[str, Any],
    engine: str = DEFAULT_LIP_SYNC_ENGINE,
    engine_options: dict[str, Any] | None = None,
) -> str:
    """Deterministic key for caching one rendered window so retries/edits skip unchanged shots."""
    payload = {
        "version": {"h3": 7, "musetalk": 7}.get(engine, 5),
        "engine": str(engine),
        "dub": [dub.name, video_fingerprint(dub)],
        "window": [
            window.shot_index, window.speaker, window.first_frame, window.source_frames,
            window.render_frames, window.active_start_frame, window.active_end_frame,
            list(window.face_target) if window.face_target is not None else None,
        ],
        "render": [int(crop_size), round(float(strength), 4), int(seed)],
        "refinement": refinement,
        "dialogue_sha256": hashlib.sha256(np.ascontiguousarray(dialogue).tobytes()).hexdigest(),
    }
    if engine_options:
        payload["engine_options"] = engine_options
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _window_cache_dir(dub: Path) -> Path:
    return player_folder(dub) / WINDOW_CACHE_DIR


def _store_cached_window(source: Path, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, cache_path)
        return cache_path
    finally:
        temporary.unlink(missing_ok=True)


def load_manual_overrides(dub: Path) -> dict[tuple[int, str], int | None]:
    """Read saved shot-speaker face overrides for a dub, if any."""
    path = player_folder(dub) / OVERRIDES_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    result: dict[tuple[int, str], int | None] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "shot_index" in item and item.get("speaker"):
                track_id = item.get("track_id")
                result[(int(item["shot_index"]), str(item["speaker"]))] = (
                    None if track_id is None else int(track_id)
                )
    return result


def save_manual_overrides(dub: Path, overrides: dict[tuple[int, str], int | None]) -> None:
    """Persist shot-speaker face overrides beside the dub so later edits accumulate."""
    folder = player_folder(dub)
    if not folder.is_dir():
        return
    payload = [
        {"shot_index": shot_index, "speaker": speaker, "track_id": track_id}
        for (shot_index, speaker), track_id in sorted(overrides.items())
    ]
    destination = folder / OVERRIDES_FILE
    temporary = folder / f"{OVERRIDES_FILE}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def shot_key_str(shot_index: int, speaker: str) -> str:
    return f"{int(shot_index)}:{speaker}"


def parse_shot_key(raw: str | tuple[int, str]) -> tuple[int, str]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return int(raw[0]), str(raw[1])
    text = str(raw or "")
    if ":" not in text:
        raise H3Error(f"Invalid shot key: {raw!r}. Expected '<shot_index>:<speaker>'.")
    idx_str, speaker = text.split(":", 1)
    try:
        idx = int(idx_str)
    except ValueError as exc:
        raise H3Error(f"Invalid shot index in shot key: {raw!r}.") from exc
    if not speaker:
        raise H3Error(f"Missing speaker in shot key: {raw!r}.")
    return idx, speaker


def _takes_dir(dub: Path) -> Path:
    return player_folder(dub) / TAKES_DIR


def _empty_takes_registry(dub: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "dub": dub.name,
        "dub_fingerprint": video_fingerprint(dub) if dub.is_file() else None,
        "shots": {},
        "assembled": None,
    }


def load_lip_sync_takes(dub: Path) -> dict[str, Any]:
    """Read persisted per-shot lip-sync takes for a dub, pruning missing cache files."""
    base = _empty_takes_registry(dub)
    path = player_folder(dub) / TAKES_FILE
    if not path.is_file():
        return base
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return base
    if not isinstance(raw, dict):
        return base
    if raw.get("dub") != dub.name:
        return base
    fp = base["dub_fingerprint"]
    if fp and raw.get("dub_fingerprint") and raw.get("dub_fingerprint") != fp:
        return base

    cache_dir = _window_cache_dir(dub)
    takes_dir = _takes_dir(dub)
    cleaned_shots: dict[str, dict[str, Any]] = {}
    raw_shots = raw.get("shots")
    if isinstance(raw_shots, dict):
        for key, entry in raw_shots.items():
            if not isinstance(entry, dict):
                continue
            valid_takes: list[dict[str, Any]] = []
            for take in entry.get("takes") or ():
                if not isinstance(take, dict) or not take.get("id"):
                    continue
                win_entries = take.get("windows") or []
                if not isinstance(win_entries, list) or not win_entries:
                    continue
                all_present = True
                for item in win_entries:
                    cache_name = item.get("cache_file") if isinstance(item, dict) else None
                    if not cache_name or not (cache_dir / cache_name).is_file():
                        all_present = False
                        break
                if not all_present:
                    continue
                copied = dict(take)
                preview_name = copied.get("preview_file")
                if preview_name and not (takes_dir / preview_name).is_file():
                    copied["preview_file"] = None
                valid_takes.append(copied)
            active = str(entry.get("active_take_id") or TAKE_ORIGINAL)
            valid_ids = {tk["id"] for tk in valid_takes}
            if active != TAKE_ORIGINAL and active not in valid_ids:
                active = valid_takes[-1]["id"] if valid_takes else TAKE_ORIGINAL
            if valid_takes or active == TAKE_ORIGINAL:
                cleaned_shots[str(key)] = {
                    "active_take_id": active,
                    "takes": valid_takes,
                    "next_number": max(
                        int(entry.get("next_number") or 1),
                        max((int(tk.get("number") or 0) for tk in valid_takes), default=0) + 1,
                    ),
                }
    assembled = raw.get("assembled")
    if not isinstance(assembled, dict):
        assembled = None
    return {
        "version": 1,
        "dub": dub.name,
        "dub_fingerprint": fp,
        "shots": cleaned_shots,
        "assembled": assembled,
    }


def save_lip_sync_takes(dub: Path, registry: dict[str, Any]) -> None:
    """Persist per-shot lip-sync takes atomically beside the dub."""
    folder = player_folder(dub)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / TAKES_FILE
    temporary = folder / f"{TAKES_FILE}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def lip_sync_take_preview_path(dub: Path, shot_key: str, take_id: str, *, build_missing: bool = False) -> Path | None:
    """Return the validated local MP4 preview path for one shot take, optionally building it on demand."""
    registry = load_lip_sync_takes(dub)
    entry = registry.get("shots", {}).get(shot_key)
    if not isinstance(entry, dict):
        return None
    takes_root = _takes_dir(dub).resolve()
    cache_dir = _window_cache_dir(dub)
    for take in entry.get("takes") or ():
        if take.get("id") != take_id:
            continue
        name = take.get("preview_file")
        if name and "/" not in str(name) and "\\" not in str(name):
            candidate = (takes_root / str(name)).resolve()
            if candidate.parent == takes_root and candidate.is_file():
                return candidate
        if not build_missing:
            return None
        win_pairs: list[tuple[LipSyncWindow, Path]] = []
        for item in take.get("windows") or ():
            cache_name = item.get("cache_file") if isinstance(item, dict) else None
            win_data = item.get("window") if isinstance(item, dict) else None
            if not cache_name or not isinstance(win_data, dict):
                return None
            cpath = cache_dir / str(cache_name)
            if not cpath.is_file():
                return None
            win_pairs.append((LipSyncWindow(**win_data), cpath))
        if not win_pairs:
            return None
        plan = load_lip_sync_plan(dub)
        bounds = _plan_shot_bounds(plan).get(int(take.get("shot_index", 0))) if plan else None
        preview_name = f"shot_{int(take.get('shot_index', 0)):03d}_{take_id}_{uuid4().hex[:6]}.mp4"
        dest = _takes_dir(dub) / preview_name
        build_shot_preview_clip(
            dub, win_pairs, dest,
            shot_start=bounds[0] if bounds else None,
            shot_end=bounds[1] if bounds else None,
        )
        take["preview_file"] = preview_name
        save_lip_sync_takes(dub, registry)
        return dest
    return None


def build_shot_preview_clip(
    dub: Path,
    shot_windows: Sequence[tuple[LipSyncWindow, Path]],
    destination: Path,
    *,
    shot_start: float | None = None,
    shot_end: float | None = None,
    timeout: int = 180,
) -> Path:
    """Splice one shot's rendered window(s) onto that shot's clip range with dubbed audio."""
    if not shot_windows:
        raise H3Error("No rendered lip-sync windows for shot preview.")
    meta = probe_video_metadata(dub)
    rate = Fraction(meta.fps).limit_denominator(100000)
    total_frames = max(1, round(meta.duration * rate))

    win_min = min(w.start for w, _ in shot_windows)
    win_max = max(w.end for w, _ in shot_windows)
    clip_start = max(0.0, min(win_min, float(shot_start) if shot_start is not None else win_min))
    clip_end = min(meta.duration, max(win_max, float(shot_end) if shot_end is not None else win_max))
    start_frame = max(0, round(clip_start * rate))
    end_frame = min(total_frames, max(start_frame + 1, round(clip_end * rate)))
    actual_start = float(start_frame / rate)
    actual_dur = max(0.05, float((end_frame - start_frame) / rate))

    inputs, filter_parts, fps_str = _build_splice_filter_graph(dub, shot_windows, meta, start_frame, end_frame)
    audio_idx = len(inputs) // 2
    inputs += ["-ss", f"{actual_start:.6f}", "-t", f"{actual_dur:.6f}", "-i", str(dub)]
    encoders = (
        [("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"]),
         ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])]
        if has_encoder("h264_nvenc")
        else [("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".partial")
    last_error = ""
    try:
        for _enc_name, video_args in encoders:
            pending.unlink(missing_ok=True)
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                *inputs,
                "-filter_complex", ";".join(filter_parts),
                "-map", "[vout]",
                "-map", f"{audio_idx}:a?",
                *video_args,
                "-r", fps_str,
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                "-f", "mp4",
                str(pending),
            ]
            proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout, partial_outputs=(pending,))
            if proc.returncode == 0 and pending.is_file():
                check_cancelled()
                pending.replace(destination)
                return destination
            last_error = (proc.stderr or "").strip()
        raise H3Error(f"Could not build shot preview clip: {last_error}")
    finally:
        pending.unlink(missing_ok=True)


def _plan_shot_bounds(plan: dict[str, Any]) -> dict[int, tuple[float, float]]:
    return {
        int(s["index"]): (float(s["start"]), float(s["end"]))
        for s in (plan.get("shots") or ())
        if isinstance(s, dict) and "index" in s
    }


def _settings_snapshot(
    engine: str,
    *,
    strength: float,
    feather: int,
    bbox_shift: int,
    inference_steps: int,
    guidance_scale: float,
) -> dict[str, Any]:
    return {
        "strength": round(float(strength), 3) if engine == "h3" else None,
        "feather": int(feather),
        "bbox_shift": int(bbox_shift) if engine == "musetalk" else None,
        "inference_steps": int(inference_steps) if engine == "latentsync" else None,
        "guidance_scale": round(float(guidance_scale), 2) if engine == "latentsync" else None,
    }


def _record_shot_take(
    dub: Path,
    registry: dict[str, Any],
    *,
    shot_index: int,
    speaker: str,
    engine: str,
    settings: dict[str, Any],
    track_id: int | None,
    shot_windows: Sequence[tuple[LipSyncWindow, Path]],
    shot_start: float | None = None,
    shot_end: float | None = None,
    build_preview: bool = True,
    force_new: bool = False,
) -> dict[str, Any]:
    """Record a rendered shot take in the registry and optionally build its preview clip."""
    key = shot_key_str(shot_index, speaker)
    shots = registry.setdefault("shots", {})
    entry = shots.get(key)
    if not isinstance(entry, dict):
        entry = {"active_take_id": TAKE_ORIGINAL, "takes": [], "next_number": 1}
        shots[key] = entry
    takes: list[dict[str, Any]] = entry.setdefault("takes", [])
    cache_files = [path.name for _, path in shot_windows]
    if not force_new:
        for existing in reversed(takes):
            ex_files = [item.get("cache_file") for item in existing.get("windows") or ()]
            if ex_files == cache_files and existing.get("engine") == engine and existing.get("track_id") == track_id:
                if build_preview and not existing.get("preview_file"):
                    preview_name = f"shot_{shot_index:03d}_{existing['id']}_{uuid4().hex[:6]}.mp4"
                    try:
                        build_shot_preview_clip(
                            dub, shot_windows, _takes_dir(dub) / preview_name,
                            shot_start=shot_start, shot_end=shot_end,
                        )
                        existing["preview_file"] = preview_name
                    except Exception:
                        pass
                entry["active_take_id"] = existing["id"]
                return existing

    number = max(
        int(entry.get("next_number") or 1),
        max((int(tk.get("number") or 0) for tk in takes), default=0) + 1,
    )
    take_id = f"t{number}"
    entry["next_number"] = number + 1
    preview_file: str | None = None
    if build_preview:
        preview_name = f"shot_{shot_index:03d}_{take_id}_{uuid4().hex[:6]}.mp4"
        try:
            build_shot_preview_clip(
                dub, shot_windows, _takes_dir(dub) / preview_name,
                shot_start=shot_start, shot_end=shot_end,
            )
            preview_file = preview_name
        except Exception:
            preview_file = None

    first_win = shot_windows[0][0]
    take_record = {
        "id": take_id,
        "number": number,
        "shot_index": int(shot_index),
        "speaker": str(speaker),
        "created_at": time.time(),
        "engine": str(engine),
        "settings": dict(settings),
        "track_id": track_id,
        "face_target": list(first_win.face_target) if first_win.face_target is not None else None,
        "preview_file": preview_file,
        "windows": [
            {"window": asdict(win), "cache_file": path.name}
            for win, path in shot_windows
        ],
    }
    takes.append(take_record)
    entry["active_take_id"] = take_id
    return take_record


def select_lip_sync_shot_take(dub: Path, shot_key: str, take_id: str) -> dict[str, Any]:
    """Choose which saved take (or 'original') is active for a shot."""
    parse_shot_key(shot_key)
    registry = load_lip_sync_takes(dub)
    shots = registry.setdefault("shots", {})
    entry = shots.get(shot_key)
    if take_id == TAKE_ORIGINAL:
        if not isinstance(entry, dict):
            entry = {"active_take_id": TAKE_ORIGINAL, "takes": [], "next_number": 1}
            shots[shot_key] = entry
        else:
            entry["active_take_id"] = TAKE_ORIGINAL
        save_lip_sync_takes(dub, registry)
        return registry
    if not isinstance(entry, dict) or not any(tk.get("id") == take_id for tk in entry.get("takes") or ()):
        raise H3Error(f"Take {take_id!r} was not found for shot {shot_key}.")
    entry["active_take_id"] = take_id
    save_lip_sync_takes(dub, registry)
    return registry


def delete_lip_sync_shot_take(dub: Path, shot_key: str, take_id: str) -> dict[str, Any]:
    """Delete one saved take for a shot and update the active take if needed."""
    parse_shot_key(shot_key)
    registry = load_lip_sync_takes(dub)
    entry = registry.get("shots", {}).get(shot_key)
    if not isinstance(entry, dict):
        raise H3Error(f"No takes found for shot {shot_key}.")
    takes = list(entry.get("takes") or ())
    removed = [tk for tk in takes if tk.get("id") == take_id]
    if not removed:
        raise H3Error(f"Take {take_id!r} was not found for shot {shot_key}.")
    remaining = [tk for tk in takes if tk.get("id") != take_id]
    entry["takes"] = remaining
    if entry.get("active_take_id") == take_id:
        entry["active_take_id"] = remaining[-1]["id"] if remaining else TAKE_ORIGINAL
    preview_name = removed[0].get("preview_file")
    if preview_name:
        try:
            (_takes_dir(dub) / str(preview_name)).unlink(missing_ok=True)
        except OSError:
            pass
    save_lip_sync_takes(dub, registry)
    return registry


def _is_no_face_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _NO_FACE_MARKERS)


def _format_clock(seconds: float) -> str:
    s = max(0.0, float(seconds))
    minutes = int(s // 60)
    return f"{minutes}:{s - minutes * 60:04.1f}"


def _window_location(window: LipSyncWindow) -> str:
    return f"Shot {window.shot_index + 1}, {_format_clock(window.start)}–{_format_clock(window.end)}, {window.speaker}"


def _short_error(exc: Exception) -> str:
    text = " ".join(str(exc).split()).strip()
    return text[:240] if text else exc.__class__.__name__


def _save_lip_sync_report(
    dub: Path,
    *,
    output: Path,
    backend: str,
    engine: str = DEFAULT_LIP_SYNC_ENGINE,
    total_windows: int,
    cached_windows: int,
    skipped_windows: int,
    assignments: dict[tuple[int, str], ShotSpeakerAssignment],
    failed_windows: Sequence[dict[str, Any]] = (),
    filtered_windows: int = 0,
    face_size_range: tuple[float, float] = (DEFAULT_MIN_FACE_SIZE, DEFAULT_MAX_FACE_SIZE),
) -> None:
    folder = player_folder(dub)
    if not folder.is_dir():
        return
    failed_list = list(failed_windows)
    report = {
        "output": output.name,
        "backend": backend,
        "engine": engine,
        "total_windows": total_windows,
        "cached_windows": cached_windows,
        "rendered_windows": max(0, total_windows - cached_windows - skipped_windows - len(failed_list)),
        "skipped_windows": skipped_windows,
        "failed_windows": failed_list,
        "filtered_windows": filtered_windows,
        "face_size_range": [round(float(face_size_range[0]), 4), round(float(face_size_range[1]), 4)],
        "uncertain_count": len(uncertain_assignments(assignments)),
        "assignments": [
            item.to_dict()
            for item in sorted(assignments.values(), key=lambda a: (a.shot_index, a.speaker))
        ],
    }
    destination = folder / REPORT_FILE
    temporary = folder / f"{REPORT_FILE}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_plan_matches_dub(data: dict[str, Any], dub: Path) -> bool:
    """Verify legacy mtime-only plans against video dimensions/duration and alignment turns."""
    width = data.get("width")
    height = data.get("height")
    duration = data.get("duration")
    if (
        isinstance(width, bool) or isinstance(height, bool) or isinstance(duration, bool)
        or not isinstance(width, (int, float)) or not isinstance(height, (int, float))
        or not isinstance(duration, (int, float))
        or width <= 0 or height <= 0 or not math.isfinite(duration) or duration <= 0
    ):
        return False
    try:
        meta = probe_video_metadata(dub)
    except (H3Error, OSError, ValueError):
        return False
    if meta.width != int(width) or meta.height != int(height) or abs(meta.duration - float(duration)) > 0.35:
        return False
    alignment_path = player_folder(dub) / "alignment.json"
    if alignment_path.is_file() and isinstance(data.get("turns"), list):
        try:
            saved = json.loads(alignment_path.read_text(encoding="utf-8"))
            lines = saved.get("lines") or ()
            spoken = saved.get("spoken") or {}
            expected_turns: list[tuple[float, float]] = []
            for key, span in spoken.items():
                try:
                    idx = int(key)
                    start, end = float(span[0]), float(span[1])
                except (ValueError, TypeError, IndexError):
                    continue
                if 0 <= idx < len(lines):
                    expected_turns.append((round(start, 3), round(end, 3)))
            expected_turns.sort()
            plan_turns = [
                (round(float(t["start"]), 3), round(float(t["end"]), 3))
                for t in data["turns"]
            ]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if plan_turns != expected_turns:
            return False
    return True


def load_lip_sync_plan(dub: Path) -> dict[str, Any] | None:
    """Read the saved Stage 1 face/speaker detection plan for a dubbed video, if present."""
    path = player_folder(dub) / PLAN_FILE
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("video") != dub.name:
            return None
        stat = dub.stat()
        if "video_size" in data and data["video_size"] != stat.st_size:
            return None
        if "video_fingerprint" in data:
            if data["video_fingerprint"] != video_fingerprint(dub):
                return None
        elif "video_mtime_ns" in data and data["video_mtime_ns"] != stat.st_mtime_ns:
            if "video_size" not in data and not _legacy_plan_matches_dub(data, dub):
                return None
        if "revision" not in data:
            data["revision"] = hashlib.sha256(raw).hexdigest()
        if data.get("window_version") in (2, 3, 4):
            # Use the same window planner for review and render while retaining the saved revision.
            windows = _reviewed_windows(dub, data, _plan_assignments(data))
            data.update(window_version=5, windows_count=len(windows),
                        windows=[asdict(window) for window in windows])
        return data
    except (OSError, ValueError):
        return None


def _save_lip_sync_plan(dub: Path, plan: dict[str, Any]) -> None:
    folder = player_folder(dub)
    destination = folder / PLAN_FILE
    temporary = folder / f"{PLAN_FILE}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        raise H3Error("Could not save this dub's lip-sync plan.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_lip_sync_plan_payload(
    *,
    dub: Path,
    scan: SourceScan,
    lines: Sequence[dict[str, Any]],
    spoken: dict[str, Any],
    assignments: dict[tuple[int, str], ShotSpeakerAssignment],
    windows: Sequence[LipSyncWindow],
) -> dict[str, Any]:
    """Serialize Stage 1 detection results for live video-overlay review in Step 4."""
    shots_out = []
    for shot in scan.shots:
        tracks_out = []
        for track in shot.tracks:
            rep = track.representative_box()
            tracks_out.append({
                "track_id": track.track_id,
                "box": [round(v, 4) for v in rep] if rep is not None else None,
                "obs": [
                    [round(o.time, 3), round(o.box[0], 4), round(o.box[1], 4),
                     round(o.box[2], 4), round(o.box[3], 4), round(o.score, 3)]
                    for o in track.observations
                ],
            })
        shots_out.append({
            "index": shot.index,
            "start": round(shot.start, 3),
            "end": round(shot.end, 3),
            "tracks": tracks_out,
        })
    turns_out = []
    for key, span in spoken.items():
        try:
            idx = int(key)
            start, end = float(span[0]), float(span[1])
        except (ValueError, TypeError, IndexError):
            continue
        if not 0 <= idx < len(lines):
            continue
        line = lines[idx]
        turns_out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker": str(line.get("speaker") or ""),
            "text": str(line.get("translation") or line.get("source") or ""),
        })
    turns_out.sort(key=lambda item: (item["start"], item["end"]))
    stat = dub.stat()
    return {
        "video": dub.name,
        "video_mtime_ns": stat.st_mtime_ns,
        "video_size": stat.st_size,
        "video_fingerprint": video_fingerprint(dub),
        "revision": uuid4().hex,
        "backend": scan.backend,
        "duration": float(scan.duration),
        "cuts": list(scan.cuts),
        "width": int(scan.width),
        "height": int(scan.height),
        "fps": round(float(scan.fps), 3),
        "window_version": 5,
        "windows_count": len(windows),
        "windows": [asdict(window) for window in windows],
        "uncertain_count": len(uncertain_assignments(assignments)),
        "shots": shots_out,
        "assignments": [
            item.to_dict()
            for item in sorted(assignments.values(), key=lambda a: (a.shot_index, a.speaker))
        ],
        "turns": turns_out,
    }


def _assignment_from_dict(item: dict[str, Any]) -> ShotSpeakerAssignment:
    center = item.get("face_center")
    return ShotSpeakerAssignment(**{**item, "face_center": tuple(center) if center else None})


def _plan_assignments(plan: dict[str, Any]) -> dict[tuple[int, str], ShotSpeakerAssignment]:
    assignments = {}
    for item in plan["assignments"]:
        assignment = _assignment_from_dict(item)
        assignments[(assignment.shot_index, assignment.speaker)] = assignment
    return assignments


def _plan_cuts(plan: dict[str, Any]) -> Sequence[float]:
    # Plans saved before exact cut times were persisted still have shot boundaries.
    if "cuts" in plan:
        return plan["cuts"]
    return [shot["start"] for shot in plan["shots"][1:]]


def _plan_shots(plan):
    shots = []
    bounds = [0.0, *_plan_cuts(plan), plan["duration"]]
    for item in plan["shots"]:
        shot_index = item["index"]
        tracks = []
        for track in item["tracks"]:
            observations = tuple(
                TrackObservation(t, (x, y, w, h), score) for t, x, y, w, h, score in track["obs"]
            )
            if observations:
                tracks.append(FaceTrack(track["track_id"], shot_index, observations[0].time,
                                        observations[-1].time, observations))
        shots.append(Shot(shot_index, bounds[shot_index], bounds[shot_index + 1], tuple(tracks)))
    return shots


def revise_lip_sync_plan(
    dub: Path,
    plan: dict[str, Any],
    edits: dict[tuple[int, str], int | None],
    *,
    reset: set[tuple[int, str]],
    clear: bool = False,
    visual_speakers: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Update only the affected shot-speaker pairs in the reviewed plan."""
    overrides = load_manual_overrides(dub)
    changed = set(edits) | reset
    if clear:
        changed.update(overrides)
        overrides.clear()
    for key in reset:
        overrides.pop(key, None)
    overrides.update(edits)

    assignments = _plan_assignments(plan)
    shots = _plan_shots(plan)
    tracks_by_shot = {shot.index: {track.track_id for track in shot.tracks} for shot in shots}
    for key, track_id in edits.items():
        if key not in assignments or (track_id is not None and track_id not in tracks_by_shot[key[0]]):
            raise H3Error("Choose a tracked face or mark a speaker off-screen in the current plan.")
    with open_alignment(dub) as (saved, _clips):
        lines = saved["lines"]
        spoken = saved["spoken"]
        hints = saved.get("visual_speakers") or visual_speakers
    scan = SourceScan("", plan["duration"], plan["width"], plan["height"], plan["fps"],
                      tuple(_plan_cuts(plan)), tuple(shots), plan["backend"])
    revised = assign_speakers_to_shots(scan, lines, spoken=spoken, visual_speakers=hints, manual_overrides=overrides)
    automatic = dict(plan.get("automatic_assignments", {}))
    for key in changed & assignments.keys():
        name = f"{key[0]}:{key[1]}"
        if key in edits and assignments[key].source != "manual":
            automatic[name] = assignments[key].to_dict()
        if key not in overrides and name in automatic:
            assignments[key] = _assignment_from_dict(automatic[name])
        elif key in revised:
            assignments[key] = revised[key]
    windows = _reviewed_windows(dub, plan, assignments, visual_speakers=hints)
    _save_lip_sync_plan(dub, {
        **plan, "revision": uuid4().hex, "automatic_assignments": automatic,
        "window_version": 5, "windows_count": len(windows),
        "windows": [asdict(window) for window in windows],
        "uncertain_count": len(uncertain_assignments(assignments)),
        "assignments": [item.to_dict() for item in assignments.values()],
    })
    save_manual_overrides(dub, overrides)


def _reviewed_windows(dub, plan, assignments, *, visual_speakers=None, speaker=None):
    return plan_lip_sync_windows(
        dub, plan["duration"], video_size=(plan["width"], plan["height"]), shot_cuts=_plan_cuts(plan),
        speaker=speaker, assignments=assignments, visual_speakers=visual_speakers,
        require_assignment=True, tracks={(s.index, t.track_id): t for s in _plan_shots(plan) for t in s.tracks},
    )


def detect_lip_sync_plan(
    *,
    dub: Path,
    services: GenerationServices | None = None,
    shot_threshold: float = DEFAULT_SHOT_THRESHOLD,
    scanner: ActiveSpeakerScanner | None = None,
    manual_overrides: dict[tuple[int, str], int | None] | None = None,
    visual_speakers: Sequence[dict[str, Any]] | None = None,
    scan_cache_dir: Path | None = None,
    gemini_api_key: str | None = None,
    source_video: Path | None = None,
    force_rescan: bool = False,
) -> Iterator[str | dict[str, Any]]:
    """Stage 1 only: detect shot cuts, track faces (YOLO + MediaPipe MAR), align speakers, save plan."""
    yield "Inspecting the dubbed video"
    check_cancelled()
    if dub.suffix.lower() != ".mp4" or not dub.is_file():
        raise H3Error("Choose a completed dubbed MP4.")
    if services is not None:
        ensure_keys = getattr(services.models, "ensure_model_keys", None)
        if callable(ensure_keys):
            ensure_keys(("face_yolov8m",))
    meta = probe_video_metadata(dub)
    yield "Detecting shot cuts"
    check_cancelled()
    source = source_video if source_video and source_video.is_file() else dub
    cuts = detect_shot_cuts(source, threshold=shot_threshold)

    yield "Tracking faces and scoring mouth motion"
    check_cancelled()
    effective_scan_cache = scan_cache_dir if scan_cache_dir is not None else (dub.parent / PLAYER_DIR / SCAN_CACHE_DIR)
    scan = get_or_scan_source(source, cuts, cache_dir=effective_scan_cache, scanner=scanner, force_rescan=force_rescan)
    try:
        with open_alignment(dub) as (saved, _clips):
            saved_lines = list(saved.get("lines") or ())
            saved_spoken = dict(saved.get("spoken") or {})
            saved_visual_speakers = list(saved.get("visual_speakers") or ())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise H3Error("Saved fitted dialogue is missing or unreadable. Generate this dub again.") from exc
    effective_visual_speakers = saved_visual_speakers if saved_visual_speakers else list(visual_speakers or ())

    merged_overrides = load_manual_overrides(dub)
    if manual_overrides:
        merged_overrides.update(manual_overrides)
        save_manual_overrides(dub, merged_overrides)

    yield "Aligning speakers to tracked faces"
    assignments = assign_speakers_to_shots(
        scan, saved_lines, spoken=saved_spoken, visual_speakers=effective_visual_speakers or None,
    )
    if uncertain_assignments(assignments) and decisions.key_or_none(gemini_api_key):
        yield "Resolving uncertain speakers with Gemini Decision"
        check_cancelled()
        player = load_player(dub) if source == dub else None
        assignments = resolve_uncertain_with_gemini_decision(
            video=source, scan=scan, lines=saved_lines, assignments=assignments,
            original_audio=player.original if player is not None else None,
            api_key=gemini_api_key, cache_dir=effective_scan_cache,
        )
    automatic = {}
    if merged_overrides:
        manual = assign_speakers_to_shots(scan, saved_lines, spoken=saved_spoken, manual_overrides=merged_overrides)
        for key in merged_overrides.keys() & assignments.keys():
            automatic[f"{key[0]}:{key[1]}"] = assignments[key].to_dict()
            assignments[key] = manual[key]
    windows = plan_lip_sync_windows(
        dub, min(scan.duration, meta.duration), video_size=(meta.width, meta.height), shot_cuts=cuts,
        assignments=assignments, visual_speakers=effective_visual_speakers or None,
        require_assignment=True, tracks={(shot.index, t.track_id): t for shot in scan.shots for t in shot.tracks},
    )
    plan = build_lip_sync_plan_payload(
        dub=dub, scan=scan, lines=saved_lines, spoken=saved_spoken,
        assignments=assignments, windows=windows,
    )
    plan["automatic_assignments"] = automatic
    _save_lip_sync_plan(dub, plan)
    yield plan


def _engine_options(
    engine: str,
    *,
    feather: int,
    bbox_shift: int,
    inference_steps: int,
    guidance_scale: float,
) -> dict[str, Any]:
    if engine == "musetalk":
        return {"feather": int(feather), "bbox_shift": int(bbox_shift)}
    if engine == "latentsync":
        return {
            "feather": int(feather),
            "inference_steps": int(inference_steps),
            "guidance_scale": round(float(guidance_scale), 4),
        }
    return {}


def _render_window_via_worker(
    *,
    engine: str,
    window: LipSyncWindow,
    slice_path: Path,
    wav_path: Path,
    output_path: Path,
    feather: int,
    seed: int,
    bbox_shift: int,
    inference_steps: int,
    guidance_scale: float,
    runtime: Any,
    free_gpu: Any,
    hf_token: str | None,
    report: Callable[[str], None],
) -> Path:
    payload: dict[str, Any] = {
        "video_path": str(slice_path),
        "audio_path": str(wav_path),
        "output_path": str(output_path),
        "source_frames": window.source_frames,
        "fps": FPS,
        "active_start_frame": window.active_start_frame,
        "active_end_frame": window.active_end_frame,
        "face_target": list(window.face_target) if window.face_target is not None else None,
        "feather": int(feather),
    }
    if engine == "musetalk":
        payload["bbox_shift"] = int(bbox_shift)
    else:
        payload.update(
            seed=int(seed),
            inference_steps=int(inference_steps),
            guidance_scale=float(guidance_scale),
        )
    reply = lip_sync_runtime.run(
        f"render_{engine}",
        payload,
        runtime,
        free_gpu=free_gpu,
        hf_token=hf_token,
        report=report,
    )
    rendered = Path(reply.get("output_path") or output_path)
    if not rendered.is_file():
        raise H3Error(f"Lip-sync worker completed {engine}, but its output could not be located.")
    return rendered


def _h3_text_encoder_choice(models, refinement: dict[str, Any], filename: str) -> str | None:
    if "text_encoder" in refinement:
        return str(refinement["text_encoder"])
    configured = models.text_encoders or {}
    for label, key in H3_TEXT_ENCODER_CHOICES.items():
        if filename == (configured.get(label) or MODEL_SPECS[key].local_name):
            return label
    return None


def _prepare_h3_lip_sync_models(services: ModelsServices, models, refinement: dict[str, Any]) -> None:
    """Provision the H3 checkpoint, LoRA, text encoder, VAEs, and adapters used by lip sync."""
    config = resolve_face_refinement(refinement, models)
    settings = config["settings"]
    mode = settings["mode"]
    profile_key = models.profile_key(settings["model_profile"])
    services.ensure_profile_model(profile_key, models.profile(profile_key), mode)
    if config["lora"]:
        services.ensure_turbo_lora(models, config["variant"], mode)
    text_encoder_choice = _h3_text_encoder_choice(models, refinement, config["text_encoder"])
    if text_encoder_choice is not None:
        services.ensure_h3_text_encoder(models, text_encoder_choice)
    services.ensure_base_video_vae(models)
    services.ensure_audio_vae(models)
    encoder, decoder = settings["video_encoder"], settings["lightvae_decoder"]
    if encoder != "Original" or decoder != "none":
        services.ensure_lightvae(encoder, decoder)
    if settings["use_sol_vae"]:
        services.ensure_int8_video_vae(models, sol=True)
    elif settings["use_lynnreal_vae"]:
        services.ensure_int8_video_vae(models, lynnreal=True)
    elif settings["use_int8_vae"]:
        services.ensure_int8_video_vae(models)
    if settings["semantic_bridge"] and settings["semantic_bridge_alpha"]:
        services.ensure_h3_semantic_bridge()
    if str(settings["attention_mode"]).lower() == "veda":
        services.ensure_h3_veda_predictor(mode)


def _render_window_via_comfy(
    *,
    index: int,
    total: int,
    window: LipSyncWindow,
    slice_path: Path,
    wav_path: Path,
    services: GenerationServices,
    models: Any,
    config: dict[str, Any],
    crop_size: int,
    seed: int,
    strength: float,
    feather: int,
) -> Iterator[str]:
    staged_video = services.media.stage_file(str(slice_path), "lip_sync")
    staged_dialogue = services.media.stage_file(str(wav_path), "lip_sync")
    client_id = uuid4().hex
    graph = build_auto_lip_sync_graph(
        video=staged_video,
        dialogue=staged_dialogue,
        frames=window.render_frames,
        models=models,
        refinement=config,
        output_prefix=f"dubbing/.lip-sync-window/{client_id}",
        crop_size=crop_size,
        seed=seed,
        strength=strength,
        active_start=window.active_start_frame,
        active_end=window.active_end_frame,
        face_target=window.face_target,
        feather=feather,
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
    return rendered_path


def render_dub_lip_sync(
    *,
    dub: Path,
    services: GenerationServices | None,
    speaker: str | None = None,
    refinement: dict[str, Any] | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    strength: float = DEFAULT_AUTO_STRENGTH,
    feather: int = DEFAULT_AUTO_FEATHER,
    seed: int = 0,
    output: Path | None = None,
    shot_threshold: float = DEFAULT_SHOT_THRESHOLD,
    scanner: ActiveSpeakerScanner | None = None,
    manual_overrides: dict[tuple[int, str], int | None] | None = None,
    visual_speakers: Sequence[dict[str, Any]] | None = None,
    use_cache: bool = True,
    scan_cache_dir: Path | None = None,
    gemini_api_key: str | None = None,
    reviewed_plan: dict[str, Any] | None = None,
    source_video: Path | None = None,
    min_face_size: float | None = None,
    max_face_size: float | None = None,
    engine: str = DEFAULT_LIP_SYNC_ENGINE,
    bbox_shift: int = DEFAULT_MUSETALK_BBOX_SHIFT,
    inference_steps: int = DEFAULT_LATENTSYNC_STEPS,
    guidance_scale: float = DEFAULT_LATENTSYNC_GUIDANCE,
    runtime: Any = None,
    free_gpu: Any = None,
    hf_token: str | None = None,
) -> Iterator[str | Path]:
    """Automatically scan shots, match speakers, render lip-sync patches (cached per window), and assemble MP4."""
    check_cancelled()
    engine = validate_lip_sync_engine(engine)
    validated_options = validate_lip_sync_options(
        bbox_shift=bbox_shift,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
    )
    bbox_shift = validated_options["bbox_shift"]
    inference_steps = validated_options["inference_steps"]
    guidance_scale = validated_options["guidance_scale"]
    min_size, max_size = validate_face_size_range(min_face_size, max_face_size)
    if reviewed_plan is None:
        for update in detect_lip_sync_plan(
            dub=dub, services=services, shot_threshold=shot_threshold, scanner=scanner,
            manual_overrides=manual_overrides, visual_speakers=visual_speakers,
            scan_cache_dir=scan_cache_dir, gemini_api_key=gemini_api_key, source_video=source_video,
        ):
            if isinstance(update, dict):
                reviewed_plan = update
            else:
                yield update
    else:
        yield "Using reviewed face assignments"
    assignments = _plan_assignments(reviewed_plan)
    backend = reviewed_plan["backend"]
    if reviewed_plan.get("window_version") == 5 and "windows" in reviewed_plan:
        windows = [LipSyncWindow(**w) for w in reviewed_plan["windows"] if speaker is None or w["speaker"] == speaker]
    else:
        windows = _reviewed_windows(dub, reviewed_plan, assignments, visual_speakers=visual_speakers, speaker=speaker)
    if not windows:
        raise H3Error("No confident on-screen speaking windows were found for lip sync.")
    total_before_filter = len(windows)
    windows = filter_lip_sync_windows(windows, min_face_size=min_size, max_face_size=max_size)
    filtered_windows = total_before_filter - len(windows)
    if not windows:
        raise H3Error("No speaking windows fall within the selected face size range. Widen the range or manually select a face.")
    if filtered_windows > 0:
        yield f"Filtered {filtered_windows} window(s) outside face size range ({len(windows)} to render)"

    is_h3 = engine == "h3"
    models = None
    free_gpu_cb = None
    h3_models_ready = False
    if is_h3:
        if services is None:
            raise H3Error("Generation services are required for H3 lip sync.")
        lip_sync_runtime.stop()
        models = services.models.load_model_config()
    else:
        if runtime is None:
            raise H3Error("Lip sync runtime is not configured.")
        free_gpu_cb = free_gpu or (services.models.unload_comfy_models if services is not None else (lambda: None))

    rendered_windows: list[tuple[LipSyncWindow, Path]] = []
    cached_hits = 0
    skipped_windows = 0
    failed_windows: list[dict[str, Any]] = []
    cache_dir = _window_cache_dir(dub)
    plan_w, plan_h = reviewed_plan.get("width"), reviewed_plan.get("height")
    video_size = (int(plan_w), int(plan_h)) if plan_w and plan_h else None

    with TemporaryDirectory(prefix="h3-lip-sync-auto-") as directory:
        work_dir = Path(directory)
        total = len(windows)
        for index, window in enumerate(windows, start=1):
            check_cancelled()
            frame_count = window.render_frames if is_h3 else window.source_frames
            dialogue = fitted_dialogue(dub, window.speaker, window.first_frame, frame_count, allow_silence=True)
            config = {
                **(refinement or {}),
                "_source_duration": frame_count / FPS,
                "_feather": int(feather),
            }
            options = None if is_h3 else _engine_options(
                engine,
                feather=feather,
                bbox_shift=bbox_shift,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
            effective_crop_size = resolve_lip_sync_crop_size(window, video_size, crop_size) if is_h3 else 0
            window_seed = seed + window.first_frame
            cache_path = None
            if use_cache:
                key = window_cache_key(
                    dub,
                    window,
                    dialogue,
                    crop_size=effective_crop_size,
                    strength=strength if is_h3 else 0.0,
                    seed=window_seed,
                    refinement=config,
                    engine=engine,
                    engine_options=options,
                )
                cache_path = cache_dir / f"{key}.mp4"
                if cache_path.is_file() and cache_path.stat().st_size > 0:
                    cached_hits += 1
                    yield f"Reusing cached window {index}/{total}"
                    rendered_windows.append((window, cache_path))
                    continue

            yield f"Preparing lip-sync window {index}/{total} ({_window_location(window)})"
            if is_h3 and not h3_models_ready:
                assert services is not None
                yield "Preparing H3 lip-sync models"
                check_cancelled()
                _prepare_h3_lip_sync_models(services.models, models, config)
                services.models.ensure_model_keys(("face_yolov8m",))
                h3_models_ready = True

            try:
                wav_path = work_dir / f"window_{index:03d}.wav"
                wav_path.write_bytes(encode_wav(dialogue, FITTED_AUDIO_RATE))
                slice_path = _extract_window_slice(dub, window, work_dir / f"window_{index:03d}.mp4", pad=is_h3)
                if is_h3:
                    assert services is not None
                    rendered_path = yield from _render_window_via_comfy(
                        index=index,
                        total=total,
                        window=window,
                        slice_path=slice_path,
                        wav_path=wav_path,
                        services=services,
                        models=models,
                        config=config,
                        crop_size=effective_crop_size,
                        seed=window_seed,
                        strength=strength,
                        feather=feather,
                    )
                else:
                    status = f"Rendering lip-sync window {index}/{total} ({engine})"
                    yield status
                    check_cancelled()
                    rendered_path = yield from blocking(status, lambda report: _render_window_via_worker(
                        engine=engine,
                        window=window,
                        slice_path=slice_path,
                        wav_path=wav_path,
                        output_path=work_dir / f"rendered_{index:03d}.mp4",
                        feather=feather,
                        seed=window_seed,
                        bbox_shift=bbox_shift,
                        inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        runtime=runtime,
                        free_gpu=free_gpu_cb,
                        hf_token=hf_token,
                        report=report,
                    ))
                    check_cancelled()
            except JobCancelled:
                raise
            except Exception as exc:
                if _is_no_face_error(exc):
                    skipped_windows += 1
                    yield f"Window {index}/{total} ({_window_location(window)}): no face detected, keeping original frames"
                    continue
                reason = _short_error(exc)
                failed_windows.append({
                    "window_index": index,
                    "shot_index": window.shot_index,
                    "speaker": window.speaker,
                    "start": round(float(window.start), 3),
                    "end": round(float(window.end), 3),
                    "error": reason,
                })
                print(
                    f"[Lip Sync] Window {index}/{total} ({_window_location(window)}) failed: {reason}",
                    flush=True,
                )
                yield f"Window {index}/{total} ({_window_location(window)}) failed: {reason} — keeping original frames"
                continue

            if cache_path is not None:
                try:
                    rendered_path = _store_cached_window(rendered_path, cache_path)
                except OSError:
                    pass
            rendered_windows.append((window, rendered_path))

        if not rendered_windows:
            if failed_windows:
                first = failed_windows[0]
                loc = (
                    f"Shot {first['shot_index'] + 1} "
                    f"({_format_clock(first['start'])}–{_format_clock(first['end'])}, {first['speaker']})"
                )
                extra = f" (+{len(failed_windows) - 1} more)" if len(failed_windows) > 1 else ""
                raise H3Error(f"Lip sync failed on {loc}{extra}: {first['error']}")
            raise H3Error("No faces were detected in any speaking window for lip sync.")
        yield "Assembling the lip-synced video"
        check_cancelled()
        destination = output or dub.with_name(f"{dub.stem}.lipsync.mp4")
        assembled = assemble_lip_sync_video(dub, rendered_windows, destination)
        _save_lip_sync_report(
            dub, output=assembled, backend=backend, engine=engine,
            total_windows=len(windows), cached_windows=cached_hits,
            skipped_windows=skipped_windows, assignments=assignments,
            failed_windows=failed_windows, filtered_windows=filtered_windows,
            face_size_range=(min_size, max_size),
        )
        try:
            registry = load_lip_sync_takes(dub)
            shot_bounds = _plan_shot_bounds(reviewed_plan)
            grouped: dict[tuple[int, str], list[tuple[LipSyncWindow, Path]]] = {}
            for win, rpath in rendered_windows:
                if rpath.parent == cache_dir and rpath.is_file():
                    grouped.setdefault((win.shot_index, win.speaker), []).append((win, rpath))
            settings_snapshot = _settings_snapshot(
                engine,
                strength=strength,
                feather=feather,
                bbox_shift=bbox_shift,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
            shot_takes_map: dict[str, str] = {}
            for (s_idx, spk), win_list in grouped.items():
                assign = assignments.get((s_idx, spk))
                bounds = shot_bounds.get(s_idx)
                take = _record_shot_take(
                    dub,
                    registry,
                    shot_index=s_idx,
                    speaker=spk,
                    engine=engine,
                    settings=settings_snapshot,
                    track_id=assign.track_id if assign is not None else None,
                    shot_windows=win_list,
                    shot_start=bounds[0] if bounds else None,
                    shot_end=bounds[1] if bounds else None,
                    build_preview=True,
                    force_new=False,
                )
                shot_takes_map[shot_key_str(s_idx, spk)] = take["id"]
            registry["assembled"] = {
                "output": assembled.name,
                "at": time.time(),
                "shot_takes": shot_takes_map,
            }
            save_lip_sync_takes(dub, registry)
        except Exception:
            pass
    yield assembled


def render_lip_sync_shots(
    *,
    dub: Path,
    shot_keys: Sequence[str | tuple[int, str]],
    services: GenerationServices | None,
    reviewed_plan: dict[str, Any] | None = None,
    refinement: dict[str, Any] | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    strength: float = DEFAULT_AUTO_STRENGTH,
    feather: int = DEFAULT_AUTO_FEATHER,
    seed: int = 0,
    fresh: bool = True,
    engine: str = DEFAULT_LIP_SYNC_ENGINE,
    bbox_shift: int = DEFAULT_MUSETALK_BBOX_SHIFT,
    inference_steps: int = DEFAULT_LATENTSYNC_STEPS,
    guidance_scale: float = DEFAULT_LATENTSYNC_GUIDANCE,
    runtime: Any = None,
    free_gpu: Any = None,
    hf_token: str | None = None,
) -> Iterator[str | dict[str, Any]]:
    """Render lip-sync takes for only the selected shot(s), saving preview clips without full assembly."""
    check_cancelled()
    engine = validate_lip_sync_engine(engine)
    validated_options = validate_lip_sync_options(
        bbox_shift=bbox_shift,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
    )
    bbox_shift = validated_options["bbox_shift"]
    inference_steps = validated_options["inference_steps"]
    guidance_scale = validated_options["guidance_scale"]

    parsed_keys: list[tuple[int, str]] = []
    seen_keys: set[tuple[int, str]] = set()
    for raw in shot_keys or ():
        key = parse_shot_key(raw)
        if key not in seen_keys:
            seen_keys.add(key)
            parsed_keys.append(key)
    if not parsed_keys:
        raise H3Error("Select at least one shot to render.")

    plan = reviewed_plan or load_lip_sync_plan(dub)
    if plan is None:
        raise H3Error("Detect faces and speakers before rendering shot takes.")
    assignments = _plan_assignments(plan)
    if plan.get("window_version") == 5 and "windows" in plan:
        all_windows = [LipSyncWindow(**w) for w in plan["windows"]]
    else:
        all_windows = _reviewed_windows(dub, plan, assignments)

    windows_by_key: dict[tuple[int, str], list[LipSyncWindow]] = {}
    for win in all_windows:
        k = (win.shot_index, win.speaker)
        if k in seen_keys:
            windows_by_key.setdefault(k, []).append(win)

    ordered_shot_keys = [k for k in parsed_keys if windows_by_key.get(k)]
    if not ordered_shot_keys:
        if all(
            (a := assignments.get(k)) is not None
            and a.status not in ("uncertain", "off_screen")
            and (a.track_id is not None or a.face_center is not None)
            for k in parsed_keys
        ):
            raise H3Error("Selected shot(s) have no non-overlapping dubbed speaking frames to render in this shot.")
        raise H3Error("Selected shot(s) have no on-screen speaking windows. Assign a visible face first.")

    is_h3 = engine == "h3"
    models = None
    free_gpu_cb = None
    h3_models_ready = False
    if is_h3:
        if services is None:
            raise H3Error("Generation services are required for H3 lip sync.")
        lip_sync_runtime.stop()
        models = services.models.load_model_config()
    else:
        if runtime is None:
            raise H3Error("Lip sync runtime is not configured.")
        free_gpu_cb = free_gpu or (services.models.unload_comfy_models if services is not None else (lambda: None))

    registry = load_lip_sync_takes(dub)
    cache_dir = _window_cache_dir(dub)
    plan_w, plan_h = plan.get("width"), plan.get("height")
    video_size = (int(plan_w), int(plan_h)) if plan_w and plan_h else None
    shot_bounds = _plan_shot_bounds(plan)
    settings_snapshot = _settings_snapshot(
        engine,
        strength=strength,
        feather=feather,
        bbox_shift=bbox_shift,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
    )

    total_shots = len(ordered_shot_keys)
    rendered_shots: list[dict[str, Any]] = []
    failed_windows: list[dict[str, Any]] = []

    with TemporaryDirectory(prefix="h3-lip-sync-shots-") as directory:
        work_dir = Path(directory)
        for shot_pos, (s_idx, spk) in enumerate(ordered_shot_keys, start=1):
            check_cancelled()
            s_key = shot_key_str(s_idx, spk)
            existing_entry = registry.get("shots", {}).get(s_key) or {}
            next_num = int(existing_entry.get("next_number") or (len(existing_entry.get("takes") or ()) + 1))
            shot_wins = windows_by_key[(s_idx, spk)]
            shot_rendered: list[tuple[LipSyncWindow, Path]] = []

            for w_idx, window in enumerate(shot_wins, start=1):
                check_cancelled()
                frame_count = window.render_frames if is_h3 else window.source_frames
                dialogue = fitted_dialogue(dub, window.speaker, window.first_frame, frame_count, allow_silence=True)
                config = {
                    **(refinement or {}),
                    "_source_duration": frame_count / FPS,
                    "_feather": int(feather),
                }
                options = None if is_h3 else _engine_options(
                    engine,
                    feather=feather,
                    bbox_shift=bbox_shift,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                )
                effective_crop_size = resolve_lip_sync_crop_size(window, video_size, crop_size) if is_h3 else 0
                window_seed = seed + window.first_frame + (next_num * 1009 if fresh else 0)
                key = window_cache_key(
                    dub,
                    window,
                    dialogue,
                    crop_size=effective_crop_size,
                    strength=strength if is_h3 else 0.0,
                    seed=window_seed,
                    refinement=config,
                    engine=engine,
                    engine_options=options,
                )
                cache_path = cache_dir / f"{key}.mp4"
                if not fresh and cache_path.is_file() and cache_path.stat().st_size > 0:
                    yield f"Shot {s_idx + 1} ({shot_pos}/{total_shots}): reusing cached window"
                    shot_rendered.append((window, cache_path))
                    continue

                yield f"Rendering Shot {s_idx + 1} ({shot_pos}/{total_shots}, {_window_location(window)})"
                if is_h3 and not h3_models_ready:
                    assert services is not None
                    yield "Preparing H3 lip-sync models"
                    check_cancelled()
                    _prepare_h3_lip_sync_models(services.models, models, config)
                    services.models.ensure_model_keys(("face_yolov8m",))
                    h3_models_ready = True

                try:
                    wav_path = work_dir / f"shot_{s_idx}_{w_idx}.wav"
                    wav_path.write_bytes(encode_wav(dialogue, FITTED_AUDIO_RATE))
                    slice_path = _extract_window_slice(dub, window, work_dir / f"shot_{s_idx}_{w_idx}.mp4", pad=is_h3)
                    if is_h3:
                        assert services is not None
                        rendered_path = yield from _render_window_via_comfy(
                            index=shot_pos,
                            total=total_shots,
                            window=window,
                            slice_path=slice_path,
                            wav_path=wav_path,
                            services=services,
                            models=models,
                            config=config,
                            crop_size=effective_crop_size,
                            seed=window_seed,
                            strength=strength,
                            feather=feather,
                        )
                    else:
                        status = f"Rendering Shot {s_idx + 1} ({shot_pos}/{total_shots}, {engine})"
                        yield status
                        check_cancelled()
                        rendered_path = yield from blocking(status, lambda report: _render_window_via_worker(
                            engine=engine,
                            window=window,
                            slice_path=slice_path,
                            wav_path=wav_path,
                            output_path=work_dir / f"rendered_{s_idx}_{w_idx}.mp4",
                            feather=feather,
                            seed=window_seed,
                            bbox_shift=bbox_shift,
                            inference_steps=inference_steps,
                            guidance_scale=guidance_scale,
                            runtime=runtime,
                            free_gpu=free_gpu_cb,
                            hf_token=hf_token,
                            report=report,
                        ))
                        check_cancelled()
                except JobCancelled:
                    raise
                except Exception as exc:
                    reason = _short_error(exc)
                    failed_windows.append({
                        "shot_index": window.shot_index,
                        "speaker": window.speaker,
                        "start": round(float(window.start), 3),
                        "end": round(float(window.end), 3),
                        "error": reason,
                    })
                    yield f"Shot {s_idx + 1} ({_window_location(window)}) failed: {reason}"
                    continue

                rendered_path = _store_cached_window(rendered_path, cache_path)
                shot_rendered.append((window, rendered_path))

            if shot_rendered:
                yield f"Building preview clip for Shot {s_idx + 1} ({shot_pos}/{total_shots})"
                assign = assignments.get((s_idx, spk))
                bounds = shot_bounds.get(s_idx)
                take = _record_shot_take(
                    dub,
                    registry,
                    shot_index=s_idx,
                    speaker=spk,
                    engine=engine,
                    settings=settings_snapshot,
                    track_id=assign.track_id if assign is not None else None,
                    shot_windows=shot_rendered,
                    shot_start=bounds[0] if bounds else None,
                    shot_end=bounds[1] if bounds else None,
                    build_preview=True,
                    force_new=fresh,
                )
                save_lip_sync_takes(dub, registry)
                rendered_shots.append({"shot_key": s_key, "shot_index": s_idx, "speaker": spk, "take": take})

    if not rendered_shots:
        if failed_windows:
            first = failed_windows[0]
            raise H3Error(f"Shot {first['shot_index'] + 1} failed: {first['error']}")
        raise H3Error("No shot takes were rendered.")

    yield {"rendered_shots": rendered_shots, "failed_windows": failed_windows, "takes": registry}


def assemble_lip_sync_from_takes(
    *,
    dub: Path,
    output: Path | None = None,
) -> Iterator[str | Path]:
    """Fast-splice the full lip-synced MP4 using each shot's active take (no GPU rendering)."""
    check_cancelled()
    registry = load_lip_sync_takes(dub)
    plan = load_lip_sync_plan(dub)
    cache_dir = _window_cache_dir(dub)
    rendered_windows: list[tuple[LipSyncWindow, Path]] = []
    shot_takes_map: dict[str, str] = {}
    engines_used: set[str] = set()

    for s_key, entry in sorted((registry.get("shots") or {}).items()):
        if not isinstance(entry, dict):
            continue
        active_id = str(entry.get("active_take_id") or TAKE_ORIGINAL)
        if active_id == TAKE_ORIGINAL:
            continue
        take = next((tk for tk in entry.get("takes") or () if tk.get("id") == active_id), None)
        if not take:
            continue
        win_pairs: list[tuple[LipSyncWindow, Path]] = []
        valid = True
        for item in take.get("windows") or ():
            cache_name = item.get("cache_file")
            win_data = item.get("window")
            if not cache_name or not isinstance(win_data, dict):
                valid = False
                break
            cpath = cache_dir / str(cache_name)
            if not cpath.is_file():
                valid = False
                break
            win_pairs.append((LipSyncWindow(**win_data), cpath))
        if valid and win_pairs:
            rendered_windows.extend(win_pairs)
            shot_takes_map[s_key] = active_id
            if take.get("engine"):
                engines_used.add(str(take["engine"]))

    if not rendered_windows:
        raise H3Error("No active shot takes to assemble. Render at least one shot take first.")

    prev_map = (registry.get("assembled") or {}).get("shot_takes") or {}
    changed_count = sum(
        1 for k in (set(shot_takes_map) | set(prev_map))
        if shot_takes_map.get(k) != prev_map.get(k)
    )
    if prev_map and changed_count > 0:
        yield f"Assembling full video (patching {changed_count} changed shot(s), reusing {max(0, len(shot_takes_map) - changed_count)} cached)"
    else:
        yield f"Assembling full video from {len(shot_takes_map)} active shot take(s)"
    check_cancelled()
    destination = output or dub.with_name(f"{dub.stem}.lipsync.mp4")
    assembled = assemble_lip_sync_video(dub, rendered_windows, destination)
    registry["assembled"] = {
        "output": assembled.name,
        "at": time.time(),
        "shot_takes": shot_takes_map,
    }
    save_lip_sync_takes(dub, registry)
    assignments = _plan_assignments(plan) if plan else {}
    engine_label = next(iter(engines_used)) if len(engines_used) == 1 else "mixed"
    _save_lip_sync_report(
        dub,
        output=assembled,
        backend=(plan or {}).get("backend", "takes"),
        engine=engine_label,
        total_windows=len(rendered_windows),
        cached_windows=len(rendered_windows),
        skipped_windows=0,
        assignments=assignments,
        failed_windows=[],
    )
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
    config = {**(refinement or {}), "_source_duration": frames / FPS}
    yield "Preparing H3 lip-sync models"
    check_cancelled()
    _prepare_h3_lip_sync_models(services.models, models, config)
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
