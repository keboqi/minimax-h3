"""Standalone entry point: run only with the isolated lip-sync interpreter.

``serve`` keeps MuseTalk 1.5 and LatentSync 1.6 resident: each loads on its
first request and stays in VRAM until the parent stops the process or unloads
it. Both rely on diffusers 0.32.2 and upstream model code cloned beside the
venv, so they run outside ComfyUI's interpreter.

Requests arrive one JSON line at a time on stdin; replies go to the original
stdout and library output goes to stderr (the log). Weights download lazily into
``<models>/dubbing/lip_sync`` on first use.

The parent imports this module for its constants and pure helpers, so keep
heavy ML imports inside functions.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Sequence
import urllib.request

MUSETALK_REPO_ID = "TMElyralab/MuseTalk"
MUSETALK_BISET_REPO_ID = "ManyOtherFunctions/face-parse-bisent"
MUSETALK_RESNET_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
LATENTSYNC_REPO_ID = "ByteDance/LatentSync-1.6"
SD_VAE_REPO_ID = "stabilityai/sd-vae-ft-mse"
WHISPER_TINY_REPO_ID = "openai/whisper-tiny"
LATENTSYNC_FPS = 25


def models_root(models_dir: str | Path) -> Path:
    return Path(models_dir) / "dubbing" / "lip_sync"


# ---------------------------------------------------------------------------
# Pure helpers (also exercised by CPU unit tests)
# ---------------------------------------------------------------------------


def _normalize_box(box: Sequence[float | int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return (x1, y1, x2, y2)


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / max(1, union)


def _pick_continuous_box(
    candidates: Sequence[tuple[int, int, int, int]],
    prev_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if not candidates:
        return None
    px, py = _box_center(prev_box)
    scale = max(1.0, float(max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])))
    best_box = None
    best_score = float("-inf")
    for box in candidates:
        iou = _box_iou(prev_box, box)
        cx, cy = _box_center(box)
        dist = math.hypot(cx - px, cy - py)
        if iou == 0.0 and dist > 2.5 * scale:
            continue
        score = iou * 2.0 - (dist / scale)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box


def select_target_face(
    detections_per_frame: Sequence[Sequence[Sequence[float | int]]],
    face_target: Sequence[int] | None,
) -> list[tuple[int, int, int, int] | None]:
    """Track one target face across frames using ``face_target = (x, y, frame_index)``."""
    total = len(detections_per_frame)
    if total == 0:
        return []
    frames = [[_normalize_box(b) for b in frame] for frame in detections_per_frame]
    non_empty = [idx for idx, boxes in enumerate(frames) if boxes]
    if not non_empty:
        return [None] * total

    if face_target is not None:
        tx, ty, target_frame = float(face_target[0]), float(face_target[1]), int(face_target[2])
        target_frame = max(0, min(target_frame, total - 1))
        anchor = min(non_empty, key=lambda idx: abs(idx - target_frame))
        anchor_box = min(
            frames[anchor],
            key=lambda b: (
                math.hypot(_box_center(b)[0] - tx, _box_center(b)[1] - ty),
                -_box_area(b),
            ),
        )
    else:
        anchor = max(non_empty, key=lambda idx: max(_box_area(b) for b in frames[idx]))
        anchor_box = max(frames[anchor], key=_box_area)

    selected: list[tuple[int, int, int, int] | None] = [None] * total
    selected[anchor] = anchor_box

    last_box = anchor_box
    for idx in range(anchor + 1, total):
        picked = _pick_continuous_box(frames[idx], last_box)
        selected[idx] = picked
        if picked is not None:
            last_box = picked

    last_box = anchor_box
    for idx in range(anchor - 1, -1, -1):
        picked = _pick_continuous_box(frames[idx], last_box)
        selected[idx] = picked
        if picked is not None:
            last_box = picked

    return selected


def temporal_opacity_weights(
    total_frames: int,
    active_start: int,
    active_end: int,
    ramp_frames: int = 3,
) -> list[float]:
    """Per-frame blend opacity inside ``[active_start, active_end)``, 0.0 on inactive padding."""
    if total_frames <= 0:
        return []
    start = max(0, min(int(active_start), total_frames))
    end = total_frames if int(active_end) < 0 else max(start, min(int(active_end), total_frames))
    if start >= end:
        return [0.0] * total_frames
    eff_ramp = max(1, min(int(ramp_frames), (end - start + 1) // 2)) if ramp_frames > 0 else 0
    weights = [0.0] * total_frames
    for idx in range(start, end):
        if eff_ramp <= 0:
            weights[idx] = 1.0
            continue
        left = (idx - start + 1) if start > 0 else eff_ramp
        right = (end - idx) if end < total_frames else eff_ramp
        weights[idx] = round(min(1.0, min(left, right) / eff_ramp), 6)
    return weights


def resample_frame_indices(src_count: int, dst_count: int) -> list[int]:
    """Deterministic monotonic nearest-frame mapping from ``src_count`` to ``dst_count``."""
    if dst_count <= 0:
        return []
    if src_count <= 0:
        raise ValueError("src_count must be positive when dst_count > 0")
    if dst_count == 1 or src_count == 1:
        return [0] * dst_count
    return [min(src_count - 1, round(idx * (src_count - 1) / (dst_count - 1))) for idx in range(dst_count)]


# ---------------------------------------------------------------------------
# Runtime & upstream helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _allow_unsafe_torch_load():
    """PyTorch >=2.6 defaults weights_only=True; upstream checkpoints need weights_only=False."""
    import torch

    original = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = compat_load
    try:
        yield
    finally:
        torch.load = original


@contextlib.contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _src_root() -> Path | None:
    configured = os.environ.get("H3_LIP_SYNC_SRC_DIR")
    if configured:
        return Path(configured)
    venv = Path(sys.executable).resolve().parent.parent
    candidate = venv.with_name(venv.name + "-src")
    return candidate if candidate.is_dir() else None


def _ensure_src_on_path() -> Path | None:
    root = _src_root()
    if root is None:
        return None
    for name in ("latentsync", "musetalk"):
        pkg = root / name
        if pkg.is_dir() and str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
    return root


def _download_url_atomic(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp_name)
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return destination


def _read_video_frames(video_path: Path) -> tuple[list[Any], float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video slice: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Video slice contains no frames: {video_path}")
    return frames, fps


def _write_cfr_mp4(frames_bgr: Sequence[Any], output_path: Path, fps: float) -> Path:
    if not frames_bgr:
        raise RuntimeError("No frames to encode")
    h, w = frames_bgr[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for frame in frames_bgr:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if code != 0 or not output_path.is_file():
        raise RuntimeError(f"ffmpeg failed encoding lip-sync window: {stderr.strip()}")
    return output_path


def _detect_faces_per_frame(
    frames_bgr: Sequence[Any],
    *,
    get_analyzer=None,
) -> list[list[tuple[int, int, int, int]]]:
    import cv2

    h, w = frames_bgr[0].shape[:2]
    detections: list[list[tuple[int, int, int, int]]] = [[] for _ in frames_bgr]
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            with mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.35
            ) as detector:
                for idx, frame in enumerate(frames_bgr):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detector.process(rgb)
                    for det in (result.detections or ()):
                        rel = det.location_data.relative_bounding_box
                        x1 = max(0, int(round(rel.xmin * w)))
                        y1 = max(0, int(round(rel.ymin * h)))
                        x2 = min(w, int(round((rel.xmin + rel.width) * w)))
                        y2 = min(h, int(round((rel.ymin + rel.height) * h)))
                        if x2 > x1 + 4 and y2 > y1 + 4:
                            detections[idx].append((x1, y1, x2, y2))
            if any(detections):
                return detections
    except Exception:
        pass

    if get_analyzer is not None:
        analyzer = get_analyzer()
    else:
        from insightface.app import FaceAnalysis

        analyzer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
    for idx, frame in enumerate(frames_bgr):
        for face in analyzer.get(frame):
            x1, y1, x2, y2 = (int(round(float(v))) for v in face.bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 + 4 and y2 > y1 + 4:
                detections[idx].append((x1, y1, x2, y2))
    return detections


def _smooth_boxes(boxes: Sequence[tuple[int, int, int, int] | None], width: int, height: int) -> list[tuple[int, int, int, int] | None]:
    """Smooth box centers/sizes with a 5-frame boxcar over valid detections."""
    valid_indices = [idx for idx, box in enumerate(boxes) if box is not None]
    if not valid_indices:
        return list(boxes)
    out: list[tuple[int, int, int, int] | None] = [None] * len(boxes)
    for idx in valid_indices:
        neighbors = [boxes[j] for j in range(max(0, idx - 2), min(len(boxes), idx + 3)) if boxes[j] is not None]
        x1 = max(0, min(width - 2, round(sum(b[0] for b in neighbors) / len(neighbors))))
        y1 = max(0, min(height - 2, round(sum(b[1] for b in neighbors) / len(neighbors))))
        x2 = max(x1 + 2, min(width, round(sum(b[2] for b in neighbors) / len(neighbors))))
        y2 = max(y1 + 2, min(height, round(sum(b[3] for b in neighbors) / len(neighbors))))
        out[idx] = (x1, y1, x2, y2)
    return out


def _feathered_lower_face_mask(height: int, width: int, box: tuple[int, int, int, int], feather: int) -> Any:
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    mask = np.zeros((height, width), dtype=np.float32)
    cx = int(round(x1 + 0.5 * bw))
    cy = int(round(y1 + 0.72 * bh))
    rx = max(2, int(round(0.46 * bw)))
    ry = max(2, int(round(0.32 * bh)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    k = max(3, (int(feather) * 2 + 1) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)[..., None]


def _blend_frame(original: Any, rendered: Any, mask: Any, opacity: float) -> Any:
    import numpy as np

    if opacity <= 0.0:
        return original
    alpha = np.clip(mask * float(opacity), 0.0, 1.0)
    blended = rendered.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in params}
    return fn(**filtered)


# ---------------------------------------------------------------------------
# Lazy weight downloads
# ---------------------------------------------------------------------------


def _ensure_musetalk_weights(root: Path, progress) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    base = root / "musetalk_v15"
    models_dir = base / "models"
    unet_dir = models_dir / "musetalkV15"
    if not (unet_dir / "unet.pth").is_file() or not (unet_dir / "musetalk.json").is_file():
        progress("Downloading MuseTalk 1.5 weights")
        hf_hub_download(MUSETALK_REPO_ID, "musetalkV15/musetalk.json", local_dir=str(models_dir))
        hf_hub_download(MUSETALK_REPO_ID, "musetalkV15/unet.pth", local_dir=str(models_dir))

    vae_dir = models_dir / "sd-vae-ft-mse"
    if not (vae_dir / "config.json").is_file():
        progress("Downloading SD-VAE weights for MuseTalk 1.5")
        snapshot_download(
            SD_VAE_REPO_ID,
            local_dir=str(vae_dir),
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )

    whisper_dir = models_dir / "whisper"
    if not (whisper_dir / "config.json").is_file():
        progress("Downloading Whisper-tiny weights for MuseTalk 1.5")
        snapshot_download(
            WHISPER_TINY_REPO_ID,
            local_dir=str(whisper_dir),
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )

    biset_dir = models_dir / "face-parse-bisent"
    if not (biset_dir / "79999_iter.pth").is_file():
        progress("Downloading BiSeNet face-parsing weights")
        hf_hub_download(MUSETALK_BISET_REPO_ID, "79999_iter.pth", local_dir=str(biset_dir))
    if not (biset_dir / "resnet18-5c106cde.pth").is_file():
        progress("Downloading ResNet-18 backbone weights")
        _download_url_atomic(MUSETALK_RESNET_URL, biset_dir / "resnet18-5c106cde.pth")
    return base


def _ensure_latentsync_weights(root: Path, progress) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    base = root / "latentsync_16"
    if not (base / "latentsync_unet.pt").is_file():
        progress("Downloading LatentSync 1.6 U-Net weights")
        hf_hub_download(LATENTSYNC_REPO_ID, "latentsync_unet.pt", local_dir=str(base))
    if not (base / "whisper" / "tiny.pt").is_file():
        progress("Downloading LatentSync Whisper weights")
        hf_hub_download(LATENTSYNC_REPO_ID, "whisper/tiny.pt", local_dir=str(base))
    vae_dir = base / "sd-vae-ft-mse"
    if not (vae_dir / "config.json").is_file():
        progress("Downloading SD-VAE weights for LatentSync 1.6")
        snapshot_download(
            SD_VAE_REPO_ID,
            local_dir=str(vae_dir),
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
    return base


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


class Worker:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.root = models_root(self.models_dir)
        self._models: dict[str, Any] = {}
        self._face_analyzer: Any = None
        self._report = None

    @property
    def loaded(self) -> list[str]:
        return sorted(self._models)

    def progress(self, message: str) -> None:
        if self._report is not None:
            self._report(message)

    def _get_face_analyzer(self) -> Any:
        if self._face_analyzer is None:
            from insightface.app import FaceAnalysis

            analyzer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self._face_analyzer = analyzer
        return self._face_analyzer

    def _unload(self, model: str | None) -> dict:
        if model is None:
            self._models.clear()
            self._face_analyzer = None
        else:
            self._models.pop(model, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return {}

    def _load_musetalk(self) -> dict[str, Any]:
        if "musetalk_v15" in self._models:
            return self._models["musetalk_v15"]
        _ensure_src_on_path()
        base = _ensure_musetalk_weights(self.root, self.progress)
        models_dir = base / "models"
        import torch
        from transformers import WhisperModel
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.progress("Loading MuseTalk 1.5 models")
        with _cwd(base), _allow_unsafe_torch_load():
            vae = _call_with_supported_kwargs(VAE, model_path=str(models_dir / "sd-vae-ft-mse"), device=str(device))
            unet = _call_with_supported_kwargs(
                UNet,
                unet_config=str(models_dir / "musetalkV15" / "musetalk.json"),
                model_path=str(models_dir / "musetalkV15" / "unet.pth"),
                use_float16=(dtype == torch.float16),
                device=device,
            )
            unet.model = unet.model.to(device=device, dtype=dtype)
            unet.model.eval()
            pe = PositionalEncoding(d_model=384).to(device=device, dtype=dtype)
            whisper = WhisperModel.from_pretrained(str(models_dir / "whisper")).to(device=device, dtype=dtype)
            whisper.eval()
            audio_processor = AudioProcessor(feature_extractor_path=str(models_dir / "whisper"))
            face_parser = _call_with_supported_kwargs(
                FaceParsing,
                resnet_path=str(models_dir / "face-parse-bisent" / "resnet18-5c106cde.pth"),
                model_path=str(models_dir / "face-parse-bisent" / "79999_iter.pth"),
                model_pth=str(models_dir / "face-parse-bisent" / "79999_iter.pth"),
                device=str(device),
            )
        bundle = {
            "base": base,
            "device": device,
            "dtype": dtype,
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "whisper": whisper,
            "audio_processor": audio_processor,
            "face_parser": face_parser,
        }
        self._models["musetalk_v15"] = bundle
        return bundle

    def _load_latentsync(self) -> dict[str, Any]:
        if "latentsync_16" in self._models:
            return self._models["latentsync_16"]
        src_root = _ensure_src_on_path()
        if src_root is None or not (src_root / "latentsync").is_dir():
            raise RuntimeError("LatentSync source checkout was not found beside the lip-sync venv.")
        latentsync_dir = src_root / "latentsync"
        base = _ensure_latentsync_weights(self.root, self.progress)

        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        config = OmegaConf.load(str(latentsync_dir / "configs" / "unet" / "stage2_512.yaml"))

        self.progress("Loading LatentSync 1.6 models")
        with _cwd(latentsync_dir), _allow_unsafe_torch_load():
            scheduler = DDIMScheduler.from_pretrained(str(latentsync_dir / "configs"))
            vae = AutoencoderKL.from_pretrained(str(base / "sd-vae-ft-mse"), torch_dtype=dtype)
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0
            audio_encoder = _call_with_supported_kwargs(
                Audio2Feature,
                model_path=str(base / "whisper" / "tiny.pt"),
                device=str(device),
                num_frames=int(config.data.num_frames),
                audio_feat_length=list(config.data.audio_feat_length),
            )
            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(config.model),
                str(base / "latentsync_unet.pt"),
                device="cpu",
            )
            unet = unet.to(dtype=dtype)
            pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to(device)
        bundle = {
            "latentsync_dir": latentsync_dir,
            "config": config,
            "dtype": dtype,
            "pipeline": pipeline,
        }
        self._models["latentsync_16"] = bundle
        return bundle

    def _render_musetalk(self, request: dict) -> dict:
        import cv2
        import numpy as np
        import torch
        from musetalk.utils.blending import get_image_blending, get_image_prepare_material

        video_path = Path(request["video_path"])
        audio_path = Path(request["audio_path"])
        output_path = Path(request["output_path"])
        fps = float(request.get("fps", 24))
        active_start = int(request.get("active_start_frame", 0))
        active_end = int(request.get("active_end_frame", -1))
        face_target = request.get("face_target")
        feather = int(request.get("feather", 6))
        bbox_shift = int(request.get("bbox_shift", 0))

        frames_bgr, _ = _read_video_frames(video_path)
        height, width = frames_bgr[0].shape[:2]
        detections = _detect_faces_per_frame(frames_bgr, get_analyzer=self._get_face_analyzer)
        boxes = _smooth_boxes(select_target_face(detections, face_target), width, height)
        if not any(box is not None for box in boxes):
            raise RuntimeError("No detected face in lip-sync window.")

        bundle = self._load_musetalk()
        device, dtype = bundle["device"], bundle["dtype"]
        vae, unet, pe = bundle["vae"], bundle["unet"], bundle["pe"]
        whisper, audio_processor, face_parser = bundle["whisper"], bundle["audio_processor"], bundle["face_parser"]

        self.progress("Extracting Whisper audio features (MuseTalk 1.5)")
        audio_feat, librosa_len = audio_processor.get_audio_feature(str(audio_path))
        whisper_chunks = audio_processor.get_whisper_chunk(
            audio_feat, device, dtype, whisper, librosa_len, fps=fps,
        )
        if not whisper_chunks:
            raise RuntimeError("No audio features extracted for MuseTalk window.")
        if len(whisper_chunks) < len(frames_bgr):
            pad = [whisper_chunks[-1]] * (len(frames_bgr) - len(whisper_chunks))
            whisper_chunks = list(whisper_chunks) + pad

        weights = temporal_opacity_weights(len(frames_bgr), active_start, active_end)
        rendered_indices = [idx for idx, box in enumerate(boxes) if box is not None and weights[idx] > 0.0]
        if not rendered_indices:
            raise RuntimeError("No detected face in active lip-sync frames.")

        self.progress(f"Synthesizing {len(rendered_indices)} MuseTalk 1.5 frame(s)")
        batch_size = 8
        timesteps = torch.tensor([0], device=device)
        out_frames = [frame.copy() for frame in frames_bgr]
        with _cwd(bundle["base"]), torch.inference_mode():
            for start in range(0, len(rendered_indices), batch_size):
                batch_ids = rendered_indices[start:start + batch_size]
                latents_list = []
                audio_list = []
                prep_items = []
                for idx in batch_ids:
                    x1, y1, x2, y2 = boxes[idx]  # type: ignore[misc]
                    y_shift = bbox_shift
                    sy1 = max(0, min(height - 2, y1 + y_shift))
                    sy2 = max(sy1 + 2, min(height, y2 + y_shift))
                    face_box = [x1, sy1, x2, sy2]
                    crop = cv2.resize(frames_bgr[idx][sy1:sy2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latents_list.append(vae.get_latents_for_unet(crop))
                    audio_list.append(whisper_chunks[idx])
                    try:
                        mask, crop_box = get_image_prepare_material(
                            frames_bgr[idx], face_box, fp=face_parser, mode="jaw",
                        )
                    except Exception:
                        mask, crop_box = None, None
                    prep_items.append((idx, face_box, mask, crop_box))

                latent_batch = torch.cat(latents_list, dim=0).to(device=device, dtype=dtype)
                audio_tensor = torch.stack(
                    [a if torch.is_tensor(a) else torch.as_tensor(a) for a in audio_list]
                ).to(device=device, dtype=dtype)
                if audio_tensor.ndim == 4 and audio_tensor.shape[1] == 1:
                    audio_tensor = audio_tensor.squeeze(1)
                audio_batch = pe(audio_tensor)
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_batch).sample
                decoded = vae.decode_latents(pred_latents)

                for item, res_face in zip(prep_items, decoded):
                    idx, face_box, mask, crop_box = item
                    fx1, fy1, fx2, fy2 = face_box
                    face_uint8 = np.clip(res_face, 0, 255).astype(np.uint8)
                    resized_face = cv2.resize(face_uint8, (fx2 - fx1, fy2 - fy1))
                    blended = None
                    if mask is not None and crop_box is not None:
                        try:
                            blended = get_image_blending(
                                frames_bgr[idx].copy(), resized_face, face_box, mask, crop_box
                            )
                        except Exception:
                            blended = None
                    if blended is None:
                        blended = frames_bgr[idx].copy()
                        blended[fy1:fy2, fx1:fx2] = resized_face
                    lower_mask = _feathered_lower_face_mask(height, width, boxes[idx], feather)  # type: ignore[arg-type]
                    out_frames[idx] = _blend_frame(frames_bgr[idx], blended, lower_mask, weights[idx])

        _write_cfr_mp4(out_frames, output_path, fps)
        return {"output_path": str(output_path), "frames": len(out_frames)}

    def _render_latentsync(self, request: dict) -> dict:
        import cv2
        import torch

        video_path = Path(request["video_path"])
        audio_path = Path(request["audio_path"])
        output_path = Path(request["output_path"])
        src_fps = float(request.get("fps", 24))
        active_start = int(request.get("active_start_frame", 0))
        active_end = int(request.get("active_end_frame", -1))
        face_target = request.get("face_target")
        feather = int(request.get("feather", 6))
        seed = int(request.get("seed", 0))
        inference_steps = int(request.get("inference_steps", 20))
        guidance_scale = float(request.get("guidance_scale", 1.5))

        frames_24, _ = _read_video_frames(video_path)
        src_count = len(frames_24)
        height, width = frames_24[0].shape[:2]

        # LatentSync expects 25 fps (Whisper 50 Hz = 2 audio steps per video frame).
        dst_count = max(1, round(src_count * LATENTSYNC_FPS / src_fps))
        idx_24_to_25 = resample_frame_indices(src_count, dst_count)
        frames_25 = [frames_24[i] for i in idx_24_to_25]
        scaled_target = None
        if face_target is not None:
            tf = max(0, min(dst_count - 1, round(int(face_target[2]) * LATENTSYNC_FPS / src_fps)))
            scaled_target = (int(face_target[0]), int(face_target[1]), tf)

        detections_25 = _detect_faces_per_frame(frames_25, get_analyzer=self._get_face_analyzer)
        boxes_25 = _smooth_boxes(select_target_face(detections_25, scaled_target), width, height)
        valid_boxes = [box for box in boxes_25 if box is not None]
        if not valid_boxes:
            raise RuntimeError("No detected face in lip-sync window.")

        # Build a single generous ROI around the target speaker so LatentSync sees only this face.
        cx = int(round(sum((b[0] + b[2]) * 0.5 for b in valid_boxes) / len(valid_boxes)))
        cy = int(round(sum((b[1] + b[3]) * 0.5 for b in valid_boxes) / len(valid_boxes)))
        max_dim = max(max(b[2] - b[0], b[3] - b[1]) for b in valid_boxes)
        roi_half = max(128, min(max(width, height) // 2, int(round(max_dim * 1.4))))
        rx1 = max(0, cx - roi_half)
        ry1 = max(0, cy - roi_half)
        rx2 = min(width, cx + roi_half)
        ry2 = min(height, cy + roi_half)
        # Ensure even ROI dimensions for libx264 yuv420p.
        if (rx2 - rx1) % 2:
            rx2 = max(rx1 + 2, rx2 - 1)
        if (ry2 - ry1) % 2:
            ry2 = max(ry1 + 2, ry2 - 1)

        bundle = self._load_latentsync()
        pipeline = bundle["pipeline"]
        config = bundle["config"]
        latentsync_dir = bundle["latentsync_dir"]

        with tempfile.TemporaryDirectory(prefix="latentsync-roi-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            roi_in = tmp_path / "roi_in.mp4"
            roi_out = tmp_path / "roi_out.mp4"
            roi_frames = [frame[ry1:ry2, rx1:rx2] for frame in frames_25]
            _write_cfr_mp4(roi_frames, roi_in, LATENTSYNC_FPS)

            self.progress(f"Denoising LatentSync 1.6 ({inference_steps} steps, CFG {guidance_scale:.1f})")
            torch.manual_seed(seed)
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            with _cwd(latentsync_dir), torch.inference_mode():
                _call_with_supported_kwargs(
                    pipeline,
                    video_path=str(roi_in),
                    audio_path=str(audio_path),
                    video_out_path=str(roi_out),
                    num_frames=int(config.data.num_frames),
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    weight_dtype=bundle["dtype"],
                    width=int(config.data.resolution),
                    height=int(config.data.resolution),
                    mask_image_path=str(latentsync_dir / "latentsync" / "utils" / "mask.png"),
                    generator=generator,
                )
            synced_roi, _ = _read_video_frames(roi_out)

        synced_25 = []
        for idx, orig in enumerate(frames_25):
            canvas = orig.copy()
            if idx < len(synced_roi) and boxes_25[idx] is not None:
                patch = synced_roi[idx]
                if patch.shape[:2] != (ry2 - ry1, rx2 - rx1):
                    patch = cv2.resize(patch, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_LANCZOS4)
                canvas[ry1:ry2, rx1:rx2] = patch
            synced_25.append(canvas)

        idx_25_to_24 = resample_frame_indices(len(synced_25), src_count)
        idx_box_24 = resample_frame_indices(len(boxes_25), src_count)
        weights_24 = temporal_opacity_weights(src_count, active_start, active_end)
        out_frames = []
        for idx, orig in enumerate(frames_24):
            box = boxes_25[idx_box_24[idx]]
            if box is None or weights_24[idx] <= 0.0:
                out_frames.append(orig)
                continue
            cand = synced_25[idx_25_to_24[idx]]
            mask = _feathered_lower_face_mask(height, width, box, feather)
            out_frames.append(_blend_frame(orig, cand, mask, weights_24[idx]))

        _write_cfr_mp4(out_frames, output_path, src_fps)
        return {"output_path": str(output_path), "frames": len(out_frames)}

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "unload":
            return self._unload(request.get("model"))
        if op == "render_musetalk":
            return self._render_musetalk(request)
        if op == "render_latentsync":
            return self._render_latentsync(request)
        raise ValueError(f"Unknown lip-sync worker operation: {op}")


def serve(models_dir: str, requests=None, replies=None) -> None:
    if replies is None:
        replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        tempfile.tempdir = str(models_root(models_dir) / "tmp")
        Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)
    worker = Worker(Path(models_dir))
    for line in requests if requests is not None else sys.stdin.buffer:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line.strip():
            continue
        request = json.loads(line)

        def report(message: str) -> None:
            replies.write(
                json.dumps({"id": request.get("id"), "event": "progress", "status": message}, ensure_ascii=False)
                + "\n"
            )
            replies.flush()

        worker._report = report if request.get("progress") else None
        try:
            reply = {"ok": True, **worker.handle(request)}
        except Exception as exc:
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stderr.flush()
        replies.write(
            json.dumps({"id": request.get("id"), "loaded": worker.loaded, **reply}, ensure_ascii=False) + "\n"
        )
        replies.flush()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[2])
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")
