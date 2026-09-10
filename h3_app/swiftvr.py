"""Extracted swiftvr boundary."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.processes import run_media_process
from h3_models import resolve_hf_token
from h3_requirements import SWIFTVR_HF_REPO

from .jobs import check_cancelled
from .provenance import copy_snapshot

_SWIFTVR_PIPELINE = None
_SWIFTVR_LOCK = RLock()


def ensure_swiftvr_checkpoint(*, runtime: RuntimeConfig) -> tuple[Path, bool]:
    configured = os.getenv("SWIFTVR_CHECKPOINT_DIR")
    checkpoint = (
        Path(configured or runtime.comfy_dir / "models" / "swiftvr")
        .expanduser()
        .resolve()
    )
    required = (
        checkpoint / "reae.safetensors",
        checkpoint / "prompt_embedding.safetensors",
        checkpoint / "transformer" / "config.json",
        checkpoint / "transformer" / "diffusion_pytorch_model.safetensors",
    )
    if all(path.is_file() for path in required):
        return checkpoint, False
    if configured:
        missing = ", ".join(
            str(path.relative_to(checkpoint)) for path in required if not path.is_file()
        )
        raise H3Error(
            f"SWIFTVR_CHECKPOINT_DIR is incomplete ({checkpoint}). Missing: {missing}."
        )

    try:
        from huggingface_hub import snapshot_download

        checkpoint.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=SWIFTVR_HF_REPO,
            local_dir=str(checkpoint),
            token=resolve_hf_token(),
            allow_patterns=(
                "reae.safetensors",
                "prompt_embedding.safetensors",
                "transformer/*.json",
                "transformer/*.safetensors",
            ),
        )
    except Exception as exc:
        raise H3Error(
            "SwiftVR checkpoint download failed from "
            f"{SWIFTVR_HF_REPO}. Check network access and available disk "
            f"space, then retry. Details: {exc}"
        ) from exc

    missing = [path for path in required if not path.is_file()]
    if missing:
        raise H3Error(
            "SwiftVR download completed without the required files: "
            + ", ".join(str(path.relative_to(checkpoint)) for path in missing)
        )
    return checkpoint, True


def import_swiftvr_pipeline(*, runtime: RuntimeConfig) -> Any:
    source_checkout = runtime.comfy_dir.parent / "SwiftVR"
    if source_checkout.is_dir() and str(source_checkout) not in sys.path:
        sys.path.insert(0, str(source_checkout))
    try:
        from swiftvr import SwiftVRPipeline
    except Exception as exc:
        raise H3Error(
            "SwiftVR runtime is not installed. Re-run setup_h3.py without "
            "--skip-env, then restart the app."
        ) from exc
    return SwiftVRPipeline


def postprocess_swiftvr_video(
    source: Path,
    *,
    fps: float,
    target_width: int,
    target_height: int,
    runtime: RuntimeConfig,
) -> Path:
    global _SWIFTVR_PIPELINE
    with _SWIFTVR_LOCK:
        checkpoint, _ = ensure_swiftvr_checkpoint(runtime=runtime)
        if _SWIFTVR_PIPELINE is None:
            pipeline_class = import_swiftvr_pipeline(runtime=runtime)
            _SWIFTVR_PIPELINE = pipeline_class.from_pretrained(
                checkpoint,
                device="cuda",
                dtype="bfloat16",
            )
        token = uuid.uuid4().hex[:8]
        raw_directory = runtime.outputs_dir / ".processing" / token
        raw_directory.mkdir(parents=True, exist_ok=False)
        raw = raw_directory / "swiftvr.mp4"
        result = runtime.outputs_dir / "swiftvr" / f"upscale_{token}.mp4"
        result.parent.mkdir(parents=True, exist_ok=True)
        pending = result.with_suffix(result.suffix + ".partial")
        try:
            _SWIFTVR_PIPELINE.restore_video(
                source,
                raw,
                resolution=(int(target_width), int(target_height)),
                upscale=2,
                clip_len=24,
                fps=float(fps),
                quality=95,
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-f",
                "mp4",
                str(pending),
            ]
            proc = run_media_process(
                cmd,
                capture_output=True,
                text=True,
                timeout=runtime.request_timeout,
                partial_outputs=(pending,),
            )
            if proc.returncode != 0 or not pending.is_file():
                raise H3Error(f"SwiftVR output mux failed: {proc.stderr.strip()}")
            check_cancelled()
            copy_snapshot(source, result)
            pending.replace(result)
            return result
        finally:
            pending.unlink(missing_ok=True)
            raw.unlink(missing_ok=True)
            raw_directory.rmdir()
