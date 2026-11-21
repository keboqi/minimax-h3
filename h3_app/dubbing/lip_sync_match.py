"""Shot-level face tracking cache, LR-ASD adapter, and speaker-to-face assignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, Mapping, Sequence

from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled
from h3_app.llm import decisions
from h3_app.media_tools import probe_video_metadata
from h3_app.processes import run_media_process

SCAN_VERSION = 1
DEFAULT_MIN_OBSERVATIONS = 3
DEFAULT_MIN_SPEAKING_SCORE = 0.2
DEFAULT_MIN_MARGIN = 0.35
DEFAULT_OFF_SCREEN_SCORE = -0.5
DEFAULT_HINT_MAX_DISTANCE = 0.18


@dataclass(frozen=True)
class TrackObservation:
    """One timestamped face bounding box (normalized 0..1) and optional active-speaker score."""

    time: float
    box: tuple[float, float, float, float]  # (x, y, w, h) normalized [0..1]
    score: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + 0.5 * w, y + 0.5 * h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(float(self.time), 4),
            "box": [round(float(v), 5) for v in self.box],
            "score": round(float(self.score), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackObservation:
        box = data["box"]
        return cls(
            time=float(data["time"]),
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(data.get("score", 0.0)),
        )


@dataclass(frozen=True)
class FaceTrack:
    """One tracked face inside a single shot."""

    track_id: int
    shot_index: int
    start: float
    end: float
    observations: tuple[TrackObservation, ...]

    def center_at(self, time: float, *, max_gap: float = 0.75) -> tuple[float, float] | None:
        if not self.observations:
            return None
        nearest = min(self.observations, key=lambda obs: abs(obs.time - time))
        if abs(nearest.time - time) > max_gap:
            return None
        return nearest.center

    def representative_center(self) -> tuple[float, float] | None:
        if not self.observations:
            return None
        mid = self.observations[len(self.observations) // 2]
        return mid.center

    def representative_box(self) -> tuple[float, float, float, float] | None:
        if not self.observations:
            return None
        mid = self.observations[len(self.observations) // 2]
        return mid.box

    def score_in_intervals(self, intervals: Sequence[tuple[float, float]]) -> tuple[float, int]:
        """Return (mean_score, observation_count) for observations falling in any interval."""
        scores: list[float] = []
        for obs in self.observations:
            if any(start - 1e-4 <= obs.time <= end + 1e-4 for start, end in intervals):
                scores.append(obs.score)
        if not scores:
            return (float("-inf"), 0)
        return (sum(scores) / len(scores), len(scores))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "shot_index": int(self.shot_index),
            "start": round(float(self.start), 4),
            "end": round(float(self.end), 4),
            "observations": [obs.to_dict() for obs in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceTrack:
        return cls(
            track_id=int(data["track_id"]),
            shot_index=int(data["shot_index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            observations=tuple(TrackObservation.from_dict(item) for item in data.get("observations", ())),
        )


@dataclass(frozen=True)
class Shot:
    """One continuous camera shot and its tracked candidate faces."""

    index: int
    start: float
    end: float
    tracks: tuple[FaceTrack, ...] = ()

    def track_by_id(self, track_id: int) -> FaceTrack | None:
        return next((track for track in self.tracks if track.track_id == track_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "start": round(float(self.start), 4),
            "end": round(float(self.end), 4),
            "tracks": [track.to_dict() for track in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Shot:
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            tracks=tuple(FaceTrack.from_dict(item) for item in data.get("tracks", ())),
        )


@dataclass(frozen=True)
class SourceScan:
    """Cached shot cuts, face tracks, and active-speaker scores for one source video."""

    fingerprint: str
    duration: float
    width: int
    height: int
    fps: float
    cuts: tuple[float, ...]
    shots: tuple[Shot, ...]
    backend: str = "none"
    version: int = SCAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "fingerprint": self.fingerprint,
            "duration": round(float(self.duration), 4),
            "width": int(self.width),
            "height": int(self.height),
            "fps": round(float(self.fps), 4),
            "cuts": [round(float(c), 4) for c in self.cuts],
            "shots": [shot.to_dict() for shot in self.shots],
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceScan:
        if int(data.get("version", 0)) != SCAN_VERSION:
            raise ValueError("Unsupported source scan cache version")
        return cls(
            fingerprint=str(data["fingerprint"]),
            duration=float(data["duration"]),
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data["fps"]),
            cuts=tuple(float(c) for c in data.get("cuts", ())),
            shots=tuple(Shot.from_dict(item) for item in data.get("shots", ())),
            backend=str(data.get("backend", "none")),
            version=SCAN_VERSION,
        )


@dataclass(frozen=True)
class ShotSpeakerAssignment:
    """Resolved face assignment for one speaker within one camera shot."""

    shot_index: int
    speaker: str
    status: str  # "matched" | "manual" | "uncertain" | "off_screen"
    track_id: int | None
    face_center: tuple[float, float] | None
    reference_time: float | None
    source: str  # "manual" | "gemini" | "lr-asd" | "single_face" | "none"
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_index": self.shot_index,
            "speaker": self.speaker,
            "status": self.status,
            "track_id": self.track_id,
            "face_center": [round(float(v), 5) for v in self.face_center] if self.face_center else None,
            "reference_time": round(float(self.reference_time), 4) if self.reference_time is not None else None,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
        }


ActiveSpeakerScanner = Callable[[Path, Sequence[float]], SourceScan]


def video_fingerprint(video: Path) -> str:
    """Fast content-and-size fingerprint for caching source scans across language dubs."""
    stat = video.stat()
    hasher = hashlib.sha256()
    hasher.update(f"{stat.st_size}:".encode("ascii"))
    chunk = 65536
    with video.open("rb") as handle:
        head = handle.read(chunk)
        hasher.update(head)
        if stat.st_size > chunk:
            handle.seek(max(0, stat.st_size - chunk))
            hasher.update(handle.read(chunk))
    return hasher.hexdigest()[:24]


def load_source_scan(cache_dir: Path, fingerprint: str) -> SourceScan | None:
    path = Path(cache_dir) / f"{fingerprint}.json"
    if not path.is_file():
        return None
    try:
        return SourceScan.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_source_scan(cache_dir: Path, scan: SourceScan) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{scan.fingerprint}.json"
    temporary = cache_dir / f"{scan.fingerprint}.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(scan.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def build_shots_from_cuts(duration: float, cuts: Sequence[float]) -> tuple[Shot, ...]:
    valid_cuts = sorted(c for c in cuts if 0.0 < c < duration)
    boundaries = [0.0, *valid_cuts, float(duration)]
    return tuple(
        Shot(index=idx, start=start, end=end, tracks=())
        for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        if end - start > 1e-3
    )


def parse_lr_asd_tracks(
    raw_tracks: Sequence[dict[str, Any]],
    *,
    fingerprint: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
    cuts: Sequence[float] = (),
    backend: str = "lr-asd",
) -> SourceScan:
    """Convert raw LR-ASD per-track frame/bbox/score records into a shot-grouped SourceScan.

    Each item in ``raw_tracks`` has:
      - ``frames``: list of frame indices (at ``track_fps``, default 25.0 or video ``fps``)
      - ``bboxes``: list of ``[x1, y1, x2, y2]`` pixel boxes
      - ``scores``: list of float active-speaker scores (positive = speaking)
      - optional ``fps``: float sampling rate for frame indices
    """
    valid_cuts = tuple(sorted(c for c in cuts if 0.0 < c < duration))
    base_shots = build_shots_from_cuts(duration, valid_cuts)
    shot_tracks: dict[int, list[FaceTrack]] = {shot.index: [] for shot in base_shots}

    for raw in raw_tracks:
        frames = raw.get("frames") or ()
        bboxes = raw.get("bboxes") or ()
        scores = raw.get("scores") or ()
        track_fps = float(raw.get("fps") or fps or 25.0)
        count = min(len(frames), len(bboxes), len(scores))
        if count <= 0 or width <= 0 or height <= 0 or track_fps <= 0:
            continue

        # Partition observations by shot so tracks never cross a hard cut.
        per_shot: dict[int, list[TrackObservation]] = {}
        for idx in range(count):
            t = float(frames[idx]) / track_fps
            if t < 0.0 or t > duration + 1e-3:
                continue
            x1, y1, x2, y2 = (float(v) for v in bboxes[idx])
            bw = max(0.0, min(float(width), x2) - max(0.0, x1)) / float(width)
            bh = max(0.0, min(float(height), y2) - max(0.0, y1)) / float(height)
            if bw <= 1e-4 or bh <= 1e-4:
                continue
            bx = max(0.0, min(1.0, x1 / float(width)))
            by = max(0.0, min(1.0, y1 / float(height)))
            obs = TrackObservation(time=t, box=(bx, by, bw, bh), score=float(scores[idx]))
            shot = next((s for s in base_shots if s.start - 1e-4 <= t <= s.end + 1e-4), None)
            if shot is not None:
                per_shot.setdefault(shot.index, []).append(obs)

        for shot_index, obs_list in per_shot.items():
            obs_list.sort(key=lambda item: item.time)
            next_id = len(shot_tracks[shot_index])
            shot_tracks[shot_index].append(FaceTrack(
                track_id=next_id,
                shot_index=shot_index,
                start=obs_list[0].time,
                end=obs_list[-1].time,
                observations=tuple(obs_list),
            ))

    shots = tuple(
        Shot(index=s.index, start=s.start, end=s.end, tracks=tuple(shot_tracks[s.index]))
        for s in base_shots
    )
    return SourceScan(
        fingerprint=fingerprint,
        duration=float(duration),
        width=int(width),
        height=int(height),
        fps=float(fps),
        cuts=valid_cuts,
        shots=shots,
        backend=backend,
    )


def run_lr_asd_scan(
    video: Path,
    cuts: Sequence[float] = (),
    *,
    lr_asd_dir: Path | None = None,
    python_executable: str | None = None,
    allow_fallback: bool = True,
    timeout: int = 600,
) -> SourceScan:
    """Run local LR-ASD if configured (via ``lr_asd_dir`` or ``H3_LR_ASD_DIR``), else fall back."""
    meta = probe_video_metadata(video)
    fp = video_fingerprint(video)
    env_dir = os.getenv("H3_LR_ASD_DIR", "").strip()
    root = Path(lr_asd_dir) if lr_asd_dir else (Path(env_dir) if env_dir else None)
    runner = root / "h3_export_tracks.py" if root else None

    if runner is None or not runner.is_file():
        if not allow_fallback:
            raise H3Error("LR-ASD runner is not installed. Set H3_LR_ASD_DIR to enable local active-speaker detection.")
        valid_cuts = tuple(sorted(c for c in cuts if 0.0 < c < meta.duration))
        return SourceScan(
            fingerprint=fp,
            duration=meta.duration,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            cuts=valid_cuts,
            shots=build_shots_from_cuts(meta.duration, valid_cuts),
            backend="none",
        )

    py = python_executable or os.getenv("H3_LR_ASD_PYTHON") or "python"
    cmd = [py, str(runner), "--video", str(video)]
    try:
        proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        if allow_fallback:
            valid_cuts = tuple(sorted(c for c in cuts if 0.0 < c < meta.duration))
            return SourceScan(
                fingerprint=fp, duration=meta.duration, width=meta.width, height=meta.height,
                fps=meta.fps, cuts=valid_cuts, shots=build_shots_from_cuts(meta.duration, valid_cuts),
                backend="none",
            )
        raise H3Error(f"LR-ASD execution failed: {exc}") from exc

    if proc.returncode != 0:
        raise H3Error(f"LR-ASD exited with code {proc.returncode}: {proc.stderr.strip()[-300:]}")
    try:
        payload = json.loads(proc.stdout)
        raw_tracks = payload.get("tracks", payload) if isinstance(payload, dict) else payload
    except (ValueError, TypeError) as exc:
        raise H3Error("LR-ASD returned unreadable track JSON.") from exc
    return parse_lr_asd_tracks(
        raw_tracks, fingerprint=fp, duration=meta.duration, width=meta.width,
        height=meta.height, fps=meta.fps, cuts=cuts, backend="lr-asd",
    )


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 1e-9 else 0.0


def _find_yolo_face_model(yolo_model_path: Path | None = None) -> Path | None:
    if yolo_model_path is not None and Path(yolo_model_path).is_file():
        return Path(yolo_model_path)
    candidates = []
    for env_key in ("H3_COMFY_DIR", "COMFY_DIR"):
        base = os.getenv(env_key, "").strip()
        if base:
            candidates.append(Path(base) / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt")
    candidates.extend([
        Path("/opt/h3/ComfyUI/models/ultralytics/bbox/face_yolov8m.pt"),
        Path("ComfyUI/models/ultralytics/bbox/face_yolov8m.pt"),
    ])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _mouth_aspect_ratio(crop_rgb, face_mesh) -> float | None:
    """MediaPipe FaceMesh inner-lip openness ratio (MAR), or None if no face landmarks found."""
    result = face_mesh.process(crop_rgb)
    faces = getattr(result, "multi_face_landmarks", None)
    if not faces:
        return None
    best = min(faces, key=lambda f: (f.landmark[1].x - 0.5) ** 2 + (f.landmark[1].y - 0.5) ** 2)
    top, bot = best.landmark[13], best.landmark[14]
    left, right = best.landmark[78], best.landmark[308]
    width = math.hypot(right.x - left.x, right.y - left.y)
    if width < 1e-4:
        return None
    return math.hypot(bot.x - top.x, bot.y - top.y) / width


def _score_motion_series(
    values: Sequence[float | None],
    diff_values: Sequence[float | None] | None = None,
) -> list[float]:
    """Convert per-frame MAR positions and/or fallback mouth-motion diffs into speaking scores centered around 0."""
    n = len(values)
    if n == 0:
        return []
    deltas: list[float | None] = [None] * n
    prev_val: float | None = None
    for i, val in enumerate(values):
        if val is not None:
            if prev_val is not None:
                deltas[i] = abs(val - prev_val)
            prev_val = val
        else:
            prev_val = None
            if diff_values is not None and i < len(diff_values) and diff_values[i] is not None:
                deltas[i] = float(diff_values[i])
    scores: list[float] = []
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        window = [deltas[j] for j in range(lo, hi) if deltas[j] is not None]
        if not window:
            scores.append(0.0)
            continue
        activity = sum(window) / len(window)
        # Typical quiet MAR/mouth motion is ~0.005-0.015; active speaking is ~0.04-0.18+.
        scores.append(round(max(-1.0, min(2.0, (activity - 0.028) * 18.0)), 4))
    return scores


def run_yolo_mediapipe_scan(
    video: Path,
    cuts: Sequence[float] = (),
    *,
    yolo_model_path: Path | None = None,
    sample_fps: float = 5.0,
    confidence: float = 0.35,
    iou_threshold: float = 0.3,
) -> SourceScan:
    """Detect & IoU-track faces per shot with face_yolov8m.pt and score mouth motion via MediaPipe MAR."""
    meta = probe_video_metadata(video)
    fp = video_fingerprint(video)
    valid_cuts = tuple(sorted(c for c in cuts if 0.0 < c < meta.duration))
    model_path = _find_yolo_face_model(yolo_model_path)
    if model_path is None:
        return SourceScan(
            fingerprint=fp, duration=meta.duration, width=meta.width, height=meta.height,
            fps=meta.fps, cuts=valid_cuts, shots=build_shots_from_cuts(meta.duration, valid_cuts),
            backend="none",
        )
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError:
        return SourceScan(
            fingerprint=fp, duration=meta.duration, width=meta.width, height=meta.height,
            fps=meta.fps, cuts=valid_cuts, shots=build_shots_from_cuts(meta.duration, valid_cuts),
            backend="none",
        )

    face_mesh = None
    backend = "yolo-motion"
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.35,
            )
            backend = "yolo-mediapipe"
    except Exception:
        face_mesh = None

    detector = YOLO(str(model_path))
    shot_bounds = list(zip([0.0, *valid_cuts], [*valid_cuts, float(meta.duration)]))
    shot_tracks: list[list[dict[str, Any]]] = [[] for _ in shot_bounds]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        if face_mesh is not None:
            face_mesh.close()
        return SourceScan(
            fingerprint=fp, duration=meta.duration, width=meta.width, height=meta.height,
            fps=meta.fps, cuts=valid_cuts, shots=build_shots_from_cuts(meta.duration, valid_cuts),
            backend="none",
        )

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or meta.fps or 24.0)
        stride = max(1, int(round(src_fps / max(1.0, float(sample_fps)))))
        frame_idx = 0
        shot_cursor = 0
        next_track_id = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            check_cancelled()
            t = frame_idx / src_fps
            frame_idx += 1
            while shot_cursor + 1 < len(shot_bounds) and t >= shot_bounds[shot_cursor][1]:
                shot_cursor += 1
            fh, fw = frame_bgr.shape[:2]
            if fh <= 0 or fw <= 0:
                continue
            results = detector.predict(frame_bgr, conf=confidence, verbose=False)
            boxes_norm: list[tuple[float, float, float, float]] = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for xyxy in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
                    bx = max(0.0, min(1.0, x1 / fw))
                    by = max(0.0, min(1.0, y1 / fh))
                    bw = max(0.0, min(1.0 - bx, (x2 - x1) / fw))
                    bh = max(0.0, min(1.0 - by, (y2 - y1) / fh))
                    if bw >= 0.02 and bh >= 0.02:
                        boxes_norm.append((bx, by, bw, bh))
            if not boxes_norm:
                continue

            active = shot_tracks[shot_cursor]
            matched_tracks: set[int] = set()
            for box in boxes_norm:
                best_idx = -1
                best_iou = iou_threshold
                for idx, tr in enumerate(active):
                    if idx in matched_tracks or t - tr["last_t"] > 1.5:
                        continue
                    iou = _box_iou(tr["last_box"], box)
                    if iou >= best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0:
                    tr = active[best_idx]
                    matched_tracks.add(best_idx)
                else:
                    tr = {
                        "track_id": next_track_id, "obs": [], "mar": [], "gray_diff": [],
                        "prev_mouth": None, "last_t": t, "last_box": box,
                    }
                    next_track_id += 1
                    active.append(tr)
                    matched_tracks.add(len(active) - 1)
                tr["last_t"] = t
                tr["last_box"] = box

                # Extract face crop for MediaPipe MAR or fallback mouth-region motion.
                bx, by, bw, bh = box
                pad = 0.15
                cx1 = max(0, int((bx - bw * pad) * fw))
                cy1 = max(0, int((by - bh * pad) * fh))
                cx2 = min(fw, int((bx + bw * (1.0 + pad)) * fw))
                cy2 = min(fh, int((by + bh * (1.0 + pad)) * fh))
                mar_val: float | None = None
                gray_diff: float | None = None
                if cx2 > cx1 + 8 and cy2 > cy1 + 8:
                    crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
                    if face_mesh is not None:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        mar_val = _mouth_aspect_ratio(crop_rgb, face_mesh)
                    if mar_val is not None:
                        tr["prev_mouth"] = None
                    else:
                        # Lightweight lower-face grayscale mean diff fallback.
                        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                        lh = gray.shape[0]
                        mouth_patch = cv2.resize(gray[int(lh * 0.55):, :], (32, 16)).astype(np.float32) / 255.0
                        if tr["prev_mouth"] is not None:
                            gray_diff = float(np.mean(np.abs(mouth_patch - tr["prev_mouth"])))
                        tr["prev_mouth"] = mouth_patch
                else:
                    tr["prev_mouth"] = None
                tr["obs"].append((t, box))
                tr["mar"].append(mar_val)
                tr["gray_diff"].append(gray_diff)
    finally:
        cap.release()
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass

    shots: list[Shot] = []
    for shot_index, ((shot_start, shot_end), raw_tracks) in enumerate(zip(shot_bounds, shot_tracks)):
        built_tracks: list[FaceTrack] = []
        for tr in raw_tracks:
            if not tr["obs"]:
                continue
            scores = _score_motion_series(tr["mar"], tr["gray_diff"])
            observations = tuple(
                TrackObservation(time=t, box=box, score=score)
                for (t, box), score in zip(tr["obs"], scores)
            )
            built_tracks.append(FaceTrack(
                track_id=tr["track_id"], shot_index=shot_index,
                start=observations[0].time, end=observations[-1].time,
                observations=observations,
            ))
        built_tracks.sort(key=lambda item: item.track_id)
        shots.append(Shot(index=shot_index, start=shot_start, end=shot_end, tracks=tuple(built_tracks)))

    return SourceScan(
        fingerprint=fp, duration=meta.duration, width=meta.width, height=meta.height,
        fps=meta.fps, cuts=valid_cuts, shots=tuple(shots), backend=backend,
    )


def get_or_scan_source(
    video: Path,
    cuts: Sequence[float],
    *,
    cache_dir: Path | None = None,
    scanner: ActiveSpeakerScanner | None = None,
    yolo_model_path: Path | None = None,
    force_rescan: bool = False,
) -> SourceScan:
    """Return a cached SourceScan when available, otherwise scan and persist it."""
    fp = video_fingerprint(video)
    if cache_dir is not None and not force_rescan:
        cached = load_source_scan(cache_dir, fp)
        if cached is not None and cached.cuts == tuple(sorted(c for c in cuts if 0.0 < c < cached.duration)):
            if cached.backend != "none" or (scanner is None and _find_yolo_face_model(yolo_model_path) is None):
                return cached
    if scanner is not None:
        scan = scanner(video, cuts)
    elif os.getenv("H3_LR_ASD_DIR", "").strip():
        scan = run_lr_asd_scan(video, cuts, allow_fallback=True)
    else:
        scan = run_yolo_mediapipe_scan(video, cuts, yolo_model_path=yolo_model_path)
    if cache_dir is not None:
        try:
            save_source_scan(cache_dir, scan)
        except OSError:
            pass
    return scan


def _match_track_to_point(shot: Shot, point: tuple[float, float], ref_time: float,
                          *, max_distance: float = DEFAULT_HINT_MAX_DISTANCE) -> FaceTrack | None:
    best_track: FaceTrack | None = None
    best_dist = float("inf")
    px, py = point
    for track in shot.tracks:
        center = track.center_at(ref_time) or track.representative_center()
        if center is None:
            continue
        dist = math.hypot(center[0] - px, center[1] - py)
        if dist < best_dist:
            best_dist = dist
            best_track = track
    if best_track is not None and best_dist <= max_distance:
        return best_track
    return None


def assign_speakers_to_shots(
    scan: SourceScan,
    lines: Sequence[dict[str, Any]],
    *,
    visual_speakers: Sequence[dict[str, Any]] | None = None,
    manual_overrides: dict[tuple[int, str], int | None] | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    min_speaking_score: float = DEFAULT_MIN_SPEAKING_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    off_screen_score: float = DEFAULT_OFF_SCREEN_SCORE,
    use_visual_hints: bool = True,
) -> dict[tuple[int, str], ShotSpeakerAssignment]:
    """Conservatively match each speaker's original turns in each shot to a visible face track.

    Priority:
      1. Manual override ``(shot_index, speaker) -> track_id | None``
      2. Gemini ``visual_speaker`` hints on original lines or transcript ``visual_speakers`` in the shot
      3. Local LR-ASD speaking scores aggregated over the speaker's original turns
      4. Single-speaker fallback when no tracks/hints exist (uses largest face in Phase 1)
    """
    overrides = manual_overrides or {}
    assignments: dict[tuple[int, str], ShotSpeakerAssignment] = {}

    for shot in scan.shots:
        # Collect original-timeline intervals and Gemini hints per speaker inside this shot.
        speaker_turns: dict[str, list[tuple[float, float]]] = {}
        speaker_hints: dict[str, list[dict[str, Any]]] = {}
        for line in lines:
            if not isinstance(line, dict):
                continue
            start = line.get("start")
            end = line.get("end")
            spk = line.get("speaker")
            if not spk or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            left = max(float(start), shot.start)
            right = min(float(end), shot.end)
            if right - left < 0.04:
                continue
            speaker_turns.setdefault(spk, []).append((left, right))
            hint = line.get("visual_speaker") if use_visual_hints else None
            if isinstance(hint, dict):
                hint_time = hint.get("time")
                if not isinstance(hint_time, (int, float)) or shot.start - 0.1 <= float(hint_time) <= shot.end + 0.1:
                    speaker_hints.setdefault(spk, []).append(hint)
        if use_visual_hints and visual_speakers:
            for hint in visual_speakers:
                if not isinstance(hint, dict):
                    continue
                spk = hint.get("speaker")
                if not spk or spk not in speaker_turns:
                    continue
                hint_time = hint.get("time")
                if isinstance(hint_time, (int, float)) and shot.start - 0.1 <= float(hint_time) <= shot.end + 0.1:
                    speaker_hints.setdefault(spk, []).append(hint)

        unique_speakers = list(speaker_turns.keys())
        for spk in unique_speakers:
            key = (shot.index, spk)
            turns = speaker_turns[spk]
            mid_time = 0.5 * (turns[0][0] + turns[0][1])

            # 1. Manual override
            if key in overrides:
                chosen_id = overrides[key]
                if chosen_id is None:
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="off_screen", track_id=None,
                        face_center=None, reference_time=mid_time, source="manual", confidence=1.0,
                        reason="Marked off-screen by user.",
                    )
                else:
                    track = shot.track_by_id(chosen_id)
                    center = (track.center_at(mid_time) or track.representative_center()) if track else None
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="manual", track_id=chosen_id,
                        face_center=center, reference_time=mid_time, source="manual", confidence=1.0,
                        reason="Selected by user.",
                    )
                continue

            # 2. Gemini visual_speaker hints
            hints = speaker_hints.get(spk, [])
            on_screen_hints = [h for h in hints if h.get("visibility") == "on_screen" and h.get("face_center")]
            if on_screen_hints:
                hint = on_screen_hints[0]
                cx, cy = float(hint["face_center"][0]), float(hint["face_center"][1])
                ref_t = float(hint.get("time", mid_time))
                matched_track = _match_track_to_point(shot, (cx, cy), ref_t) if shot.tracks else None
                center = (
                    (matched_track.center_at(ref_t) or matched_track.representative_center() or (cx, cy))
                    if matched_track is not None else (cx, cy)
                )
                assignments[key] = ShotSpeakerAssignment(
                    shot_index=shot.index, speaker=spk, status="matched",
                    track_id=matched_track.track_id if matched_track is not None else None,
                    face_center=center, reference_time=ref_t, source="gemini", confidence=0.9,
                    reason="Matched from Gemini visual speaker hint.",
                )
                continue
            if hints and all(h.get("visibility") == "off_screen" for h in hints):
                assignments[key] = ShotSpeakerAssignment(
                    shot_index=shot.index, speaker=spk, status="off_screen", track_id=None,
                    face_center=None, reference_time=mid_time, source="gemini", confidence=0.9,
                    reason="Marked off-screen by Gemini visual analysis.",
                )
                continue

            # 3. Local LR-ASD scoring across candidate face tracks in this shot
            if shot.tracks and scan.backend != "none":
                scored = [
                    (track, *track.score_in_intervals(turns))
                    for track in shot.tracks
                ]
                eligible = [(t, score, count) for t, score, count in scored if count >= min_observations]
                if not eligible:
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="uncertain", track_id=None,
                        face_center=None, reference_time=mid_time, source=scan.backend, confidence=0.0,
                        reason="Not enough tracked face observations during speech.",
                    )
                    continue
                eligible.sort(key=lambda item: item[1], reverse=True)
                best_track, best_score, _best_count = eligible[0]
                second_score = eligible[1][1] if len(eligible) > 1 else float("-inf")
                margin = best_score - second_score if len(eligible) > 1 else float("inf")

                if best_score >= min_speaking_score and margin >= min_margin:
                    center = best_track.center_at(mid_time) or best_track.representative_center()
                    confidence = max(0.0, min(1.0, 0.5 + 0.25 * best_score))
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="matched", track_id=best_track.track_id,
                        face_center=center, reference_time=mid_time, source=scan.backend, confidence=confidence,
                        reason=f"Active speaker score {best_score:.2f}"
                               + (f" (margin {margin:.2f})." if len(eligible) > 1 else "."),
                    )
                elif best_score <= off_screen_score:
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="off_screen", track_id=None,
                        face_center=None, reference_time=mid_time, source=scan.backend, confidence=0.8,
                        reason=f"No visible face speaking (best score {best_score:.2f}).",
                    )
                else:
                    assignments[key] = ShotSpeakerAssignment(
                        shot_index=shot.index, speaker=spk, status="uncertain", track_id=None,
                        face_center=None, reference_time=mid_time, source=scan.backend, confidence=0.3,
                        reason=f"Ambiguous speaker match (best {best_score:.2f}"
                               + (f", runner-up {second_score:.2f})." if len(eligible) > 1 else ")."),
                    )
                continue

            # 4. Fallback when no local ASD tracks and no Gemini hints exist
            if len(unique_speakers) == 1:
                assignments[key] = ShotSpeakerAssignment(
                    shot_index=shot.index, speaker=spk, status="matched", track_id=None,
                    face_center=None, reference_time=mid_time, source="single_face", confidence=0.5,
                    reason="Single speaker in shot; defaulting to largest visible face.",
                )
            else:
                assignments[key] = ShotSpeakerAssignment(
                    shot_index=shot.index, speaker=spk, status="uncertain", track_id=None,
                    face_center=None, reference_time=mid_time, source="none", confidence=0.0,
                    reason="Multiple speakers in shot without visual hints or active-speaker scores.",
                )

    return assignments


def uncertain_assignments(
    assignments: dict[tuple[int, str], ShotSpeakerAssignment],
) -> list[ShotSpeakerAssignment]:
    """Return shot-speaker pairs that could not be matched confidently and can be reviewed by the user."""
    return sorted(
        (item for item in assignments.values() if item.status == "uncertain"),
        key=lambda item: (item.shot_index, item.speaker),
    )


# ---------------------------------------------------------------------------
# Budget-aware Gemini Decision tie-breaker (respects 50k input tokens/min)
# ---------------------------------------------------------------------------

DECISION_MAX_TOKENS_PER_BATCH = 40_000
DECISION_TOKENS_PER_SECOND = 180
DECISION_OVERHEAD_TOKENS = 250
DECISION_SNIPPET_SECONDS = 2.0
DECISION_SNIPPET_FPS = 2
DECISION_SNIPPET_WIDTH = 480
DECISION_MIN_PROB = 0.65
DECISION_MIN_MARGIN = 0.20
DECISION_PACE_SECONDS = 62.0
_BOX_COLORS = ("cyan", "yellow", "magenta", "lime", "orange", "red")


def estimate_decision_snippet_tokens(duration_seconds: float, track_count: int) -> int:
    """Conservative input-token estimate for one low-res snippet + choice question."""
    return int(round(max(0.5, float(duration_seconds)) * DECISION_TOKENS_PER_SECOND)) + 80 + 25 * max(1, track_count)


def _position_label(cx: float) -> str:
    if cx < 0.36:
        return "left"
    if cx > 0.64:
        return "right"
    return "center"


def _best_turn_for_speaker(shot: Shot, lines: Sequence[dict[str, Any]], speaker: str,
                           max_seconds: float = DECISION_SNIPPET_SECONDS) -> tuple[float, float, str] | None:
    """Pick the longest original turn for ``speaker`` inside ``shot``, trimmed to ``max_seconds``."""
    best: tuple[float, float, str] | None = None
    for line in lines:
        if not isinstance(line, dict) or line.get("speaker") != speaker:
            continue
        start, end = line.get("start"), line.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        left = max(float(start), shot.start)
        right = min(float(end), shot.end)
        if right - left < 0.15:
            continue
        if best is None or (right - left) > (best[1] - best[0]):
            text = str(line.get("source") or line.get("translation") or "").strip()
            best = (left, right, text)
    if best is None:
        return None
    left, right, text = best
    if right - left > max_seconds:
        mid = 0.5 * (left + right)
        left = max(shot.start, mid - 0.5 * max_seconds)
        right = min(shot.end, left + max_seconds)
    return (left, right, text)


def _extract_annotated_snippet(
    video: Path,
    start: float,
    end: float,
    tracks: Sequence[FaceTrack],
    destination: Path,
    *,
    original_audio: Path | None = None,
    timeout: int = 60,
) -> Path:
    """Slice a compact low-res clip with color-coded face boxes (no font deps) and original audio."""
    duration = max(0.25, end - start)
    mid = 0.5 * (start + end)
    filters = [
        f"scale='min({DECISION_SNIPPET_WIDTH},iw)':-2",
        f"fps={DECISION_SNIPPET_FPS}",
    ]
    for idx, track in enumerate(tracks[:len(_BOX_COLORS)]):
        box = track.representative_box()
        if box is None:
            continue
        bx, by, bw, bh = box
        # Re-center using the track's position near the snippet midpoint when available.
        center = track.center_at(mid)
        if center is not None:
            bx = max(0.0, min(1.0 - bw, center[0] - 0.5 * bw))
            by = max(0.0, min(1.0 - bh, center[1] - 0.5 * bh))
        color = _BOX_COLORS[idx]
        filters.append(
            f"drawbox=x=iw*{bx:.4f}:y=ih*{by:.4f}:w=iw*{bw:.4f}:h=ih*{bh:.4f}:color={color}@0.95:t=3"
        )
    use_external_audio = original_audio is not None and original_audio.is_file()
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.4f}", "-i", str(video),
    ]
    if use_external_audio:
        cmd += ["-ss", f"{start:.4f}", "-i", str(original_audio)]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    cmd += [
        "-t", f"{duration:.4f}",
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        "-movflags", "+faststart",
        str(destination),
    ]
    proc = run_media_process(cmd, capture_output=True, text=True, timeout=timeout, partial_outputs=(destination,))
    if proc.returncode != 0 or not destination.is_file():
        raise H3Error(f"Could not extract decision snippet: {proc.stderr.strip()[-200:]}")
    return destination


def _decision_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return Path(cache_dir) / f"{fingerprint}.decisions.json"


def _load_decision_cache(cache_dir: Path | None, fingerprint: str) -> dict[str, dict[str, Any]]:
    if cache_dir is None:
        return {}
    path = _decision_cache_path(cache_dir, fingerprint)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_decision_cache(cache_dir: Path | None, fingerprint: str, cache: dict[str, dict[str, Any]]) -> None:
    if cache_dir is None:
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = _decision_cache_path(cache_dir, fingerprint)
    temporary = cache_dir / f"{fingerprint}.decisions.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def _parse_choice_decision(
    answer: Mapping[str, Any] | None,
    *,
    shot: Shot,
    speaker: str,
    tracks_by_key: Mapping[str, FaceTrack],
    ref_time: float,
    min_prob: float,
    min_margin: float,
) -> ShotSpeakerAssignment | None:
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        return None
    probs_raw = answer.get("probabilities")
    if isinstance(probs_raw, Mapping) and probs_raw:
        ranked = sorted(
            ((str(k), float(v)) for k, v in probs_raw.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        picked = answer.get("choice")
        conf = answer.get("confidence", 0.0)
        ranked = [(str(picked), float(conf))] if picked else []
    if not ranked:
        return None

    best_opt, best_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_p - second_p
    if best_p < min_prob or margin < min_margin:
        return None

    if best_opt == "off_screen":
        return ShotSpeakerAssignment(
            shot_index=shot.index, speaker=speaker, status="off_screen", track_id=None,
            face_center=None, reference_time=ref_time, source="gemini-decision",
            confidence=best_p, reason=f"Gemini Decision marked off-screen (p={best_p:.2f}).",
        )
    track = tracks_by_key.get(best_opt)
    if track is not None:
        center = track.center_at(ref_time) or track.representative_center()
        return ShotSpeakerAssignment(
            shot_index=shot.index, speaker=speaker, status="matched", track_id=track.track_id,
            face_center=center, reference_time=ref_time, source="gemini-decision",
            confidence=best_p, reason=f"Matched by Gemini Decision ({best_opt}, p={best_p:.2f}, margin={margin:.2f}).",
        )
    return None


def resolve_uncertain_with_gemini_decision(
    *,
    video: Path,
    scan: SourceScan,
    lines: Sequence[dict[str, Any]],
    assignments: dict[tuple[int, str], ShotSpeakerAssignment],
    original_audio: Path | None = None,
    api_key: str | None = None,
    cache_dir: Path | None = None,
    max_tokens_per_batch: int = DECISION_MAX_TOKENS_PER_BATCH,
    max_batches: int = 1,
    pace_batches: bool = False,
    min_prob: float = DECISION_MIN_PROB,
    min_margin: float = DECISION_MIN_MARGIN,
    decide_fn: Callable[..., dict[str, dict[str, Any]]] | None = None,
) -> dict[tuple[int, str], ShotSpeakerAssignment]:
    """Use Gemini Decision (`gemini-decision-1`) as a token-budgeted tie-breaker for uncertain shots.

    Only uncertain ``(shot_index, speaker)`` pairs with candidate face tracks are queried.
    Snippets are trimmed to ~2 seconds at low resolution and packed under ``max_tokens_per_batch``
    (default 40,000 tokens, staying safely below the 50,000 input tokens/min free-tier ceiling).
    Resolved matches are cached beside the source scan so re-runs cost 0 tokens.
    """
    key = decisions.key_or_none(api_key)
    if not key:
        return dict(assignments)

    shots_by_index = {shot.index: shot for shot in scan.shots}
    updated = dict(assignments)
    cache = _load_decision_cache(cache_dir, scan.fingerprint)
    cache_dirty = False

    pending: list[tuple[Shot, str, float, float, str, list[FaceTrack], str]] = []
    for item in uncertain_assignments(updated):
        shot = shots_by_index.get(item.shot_index)
        if shot is None or not shot.tracks:
            continue
        tracks = list(shot.tracks[:len(_BOX_COLORS)])
        turn = _best_turn_for_speaker(shot, lines, item.speaker)
        if turn is None:
            continue
        start, end, spoken_text = turn
        track_ids = [t.track_id for t in tracks]
        cache_key = f"{shot.index}:{item.speaker}:{start:.2f}:{end:.2f}:{','.join(map(str, track_ids))}"
        cached_entry = cache.get(cache_key)
        if isinstance(cached_entry, dict):
            status = cached_entry.get("status")
            if status in ("matched", "off_screen"):
                center = tuple(cached_entry["face_center"]) if cached_entry.get("face_center") else None
                updated[(shot.index, item.speaker)] = ShotSpeakerAssignment(
                    shot_index=shot.index, speaker=item.speaker, status=status,
                    track_id=cached_entry.get("track_id"), face_center=center,
                    reference_time=float(cached_entry.get("reference_time", 0.5 * (start + end))),
                    source="gemini-decision", confidence=float(cached_entry.get("confidence", 0.8)),
                    reason=str(cached_entry.get("reason", "Cached Gemini Decision match.")),
                )
                continue
        pending.append((shot, item.speaker, start, end, spoken_text, tracks, cache_key))

    if not pending:
        return updated

    # Pack pending items into token-budgeted batches under max_tokens_per_batch.
    batches: list[list[tuple[Shot, str, float, float, str, list[FaceTrack], str]]] = []
    current_batch: list[tuple[Shot, str, float, float, str, list[FaceTrack], str]] = []
    current_tokens = DECISION_OVERHEAD_TOKENS
    for entry in pending:
        est = estimate_decision_snippet_tokens(entry[3] - entry[2], len(entry[5]))
        if current_batch and current_tokens + est > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = DECISION_OVERHEAD_TOKENS
        current_batch.append(entry)
        current_tokens += est
    if current_batch:
        batches.append(current_batch)

    ask = decide_fn or decisions.decide
    limit = max(1, int(max_batches)) if not pace_batches else len(batches)
    with TemporaryDirectory(prefix="h3-lip-sync-decision-") as directory:
        work_dir = Path(directory)
        for batch_idx, batch in enumerate(batches[:limit]):
            check_cancelled()
            if batch_idx > 0 and pace_batches:
                deadline = time.monotonic() + DECISION_PACE_SECONDS
                while time.monotonic() < deadline:
                    check_cancelled()
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

            blocks: list[dict[str, Any]] = [
                decisions.text(
                    "Each clip below shows a short video segment with its original audio and color-coded "
                    "bounding boxes around candidate faces. For each question, identify which boxed person "
                    "is visibly speaking the dialogue in that clip, or choose off_screen / uncertain."
                )
            ]
            questions: dict[str, dict[str, Any]] = {}
            meta_by_q: dict[str, tuple[Shot, str, dict[str, FaceTrack], float, str]] = {}

            for clip_idx, (shot, speaker, start, end, spoken_text, tracks, cache_key) in enumerate(batch):
                clip_label = f"Clip {clip_idx + 1}"
                clip_path = work_dir / f"b{batch_idx}_c{clip_idx}.mp4"
                try:
                    _extract_annotated_snippet(
                        video, start, end, tracks, clip_path, original_audio=original_audio,
                    )
                except H3Error:
                    continue
                blocks.append(decisions.text(f"{clip_label} (shot {shot.index + 1}, {start:.1f}s-{end:.1f}s):"))
                blocks.append(decisions.media(clip_path, resolution="low"))

                options: dict[str, str] = {}
                tracks_by_key: dict[str, FaceTrack] = {}
                mid = 0.5 * (start + end)
                for t_idx, track in enumerate(tracks):
                    opt_key = f"face_{track.track_id}"
                    center = track.center_at(mid) or track.representative_center() or (0.5, 0.5)
                    color = _BOX_COLORS[t_idx]
                    options[opt_key] = f"Person in the {color} box on the {_position_label(center[0])} side"
                    tracks_by_key[opt_key] = track
                options["off_screen"] = "The speaker is off-screen or none of the boxed faces are speaking"
                options["uncertain"] = "Cannot tell reliably which face is speaking"

                quote = f' saying "{spoken_text[:120]}"' if spoken_text else ""
                q_key = f"s{shot.index}_{speaker}"
                questions[q_key] = decisions.choice(
                    f"In {clip_label}, which boxed person is visibly speaking{quote}?",
                    options,
                )
                meta_by_q[q_key] = (shot, speaker, tracks_by_key, mid, cache_key)

            if not questions:
                continue
            try:
                answers = ask(questions, blocks, key=key, timeout=120)
            except decisions.DecisionError:
                break

            for q_key, (shot, speaker, tracks_by_key, mid, cache_key) in meta_by_q.items():
                resolved = _parse_choice_decision(
                    answers.get(q_key),
                    shot=shot, speaker=speaker, tracks_by_key=tracks_by_key,
                    ref_time=mid, min_prob=min_prob, min_margin=min_margin,
                )
                if resolved is not None:
                    updated[(shot.index, speaker)] = resolved
                    cache[cache_key] = resolved.to_dict()
                    cache_dirty = True

    if cache_dirty:
        _save_decision_cache(cache_dir, scan.fingerprint, cache)
    return updated
