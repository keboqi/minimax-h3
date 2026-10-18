"""Standalone entry point: run only with the isolated Stable Audio 3 Medium interpreter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _resolve_local_model_dirs(model_id: str, cache_dir: str) -> tuple[Path, Path | None]:
    from huggingface_hub import snapshot_download

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    local_medium = cache_root / "medium"
    if (local_medium / "model_config.json").is_file() and (local_medium / "model.safetensors").is_file():
        model_dir = local_medium
    else:
        downloaded = snapshot_download(
            model_id,
            cache_dir=str(cache_root),
            local_dir=str(local_medium),
        )
        model_dir = Path(downloaded)

    t5_dir = cache_root / "google--t5gemma-b-b-ul2"
    if not (t5_dir / "config.json").is_file():
        try:
            snapshot_download(
                "google/t5gemma-b-b-ul2",
                cache_dir=str(cache_root),
                local_dir=str(t5_dir),
            )
        except Exception:
            t5_dir = None
    return model_dir, t5_dir if (t5_dir and (t5_dir / "config.json").is_file()) else None


def _patch_model_config(model_config: dict, model_dir: Path, t5_dir: Path | None) -> dict:
    cond_configs = (
        model_config.get("model", {})
        .get("conditioning", {})
        .get("configs", [])
    )
    if t5_dir is not None:
        for cond in cond_configs:
            cfg = cond.get("config", {})
            if cfg.get("t5_model_name") == "google/t5gemma-b-b-ul2":
                cfg["t5_model_name"] = str(t5_dir)
            if cfg.get("tokenizer_name") == "google/t5gemma-b-b-ul2":
                cfg["tokenizer_name"] = str(t5_dir)

    def fix_paths(node):
        if isinstance(node, dict):
            for key, val in list(node.items()):
                if key in ("ckpt_path", "pretransform_ckpt_path") and isinstance(val, str):
                    candidate = model_dir / Path(val).name
                    if candidate.is_file():
                        node[key] = str(candidate)
                    else:
                        del node[key]
                else:
                    fix_paths(val)
        elif isinstance(node, list):
            for entry in node:
                fix_paths(entry)

    fix_paths(model_config)
    return model_config


def synthesize(request: dict) -> None:
    os.environ["HF_HUB_CACHE"] = request["cache_dir"]
    import numpy as np
    import soundfile as sf
    import torch
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

    if not torch.cuda.is_available():
        raise RuntimeError("Stable Audio 3 Medium needs an available CUDA GPU.")

    model_id = request.get("model", "stabilityai/stable-audio-3-medium")
    model_dir, t5_dir = _resolve_local_model_dirs(model_id, request["cache_dir"])

    config_path = model_dir / "model_config.json"
    ckpt_path = model_dir / "model.safetensors"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = _patch_model_config(model_config, model_dir, t5_dir)

    model = create_model_from_config(model_config)
    state_dict = load_ckpt_state_dict(str(ckpt_path))
    ema_prefix = "diffusion_ema.ema_model."
    ema_state = {
        key[len(ema_prefix) :]: val
        for key, val in state_dict.items()
        if key.startswith(ema_prefix)
    }
    if ema_state and hasattr(model, "model"):
        copy_state_dict(model.model, ema_state)
    else:
        copy_state_dict(model, state_dict)
    del state_dict, ema_state

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.to(device).eval().requires_grad_(False)
    model.to(dtype)
    if getattr(model, "pretransform", None) is not None and not getattr(model.pretransform, "enable_half", False):
        model.pretransform.to(torch.float32)

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

    items = request.get("items") or [request]
    with torch.inference_mode():
        for item in items:
            prompt = str(item["prompt"]).strip()
            duration = max(1, min(int(item.get("duration", 5)), 47))
            seed = int(item.get("seed", -1))
            steps = int(item.get("steps", 8))
            cfg_scale = float(item.get("cfg_scale", 1.0))

            conditioning = [{"prompt": prompt, "seconds_total": duration}]
            output = generate_diffusion_cond_inpaint(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_max=1.0,
                sampler_type="pingpong",
                device=str(device),
                seed=seed,
                apg_scale=1.0,
                cut_to_seconds_total=True,
            )
            output = output.squeeze(0).to(torch.float32)
            peak = torch.max(torch.abs(output))
            if peak > 0:
                output = output.div(peak).clamp(-1.0, 1.0)
            audio = output.cpu().numpy()
            if audio.ndim == 2:
                # Channels-first (C, N) -> mono (N,)
                audio = np.mean(audio, axis=0)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if not audio.size or not np.isfinite(audio).all():
                raise RuntimeError(f"Stable Audio 3 Medium returned empty or invalid audio for '{prompt}'.")
            sf.write(item["output"], audio, sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    synthesize(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
