#!/usr/bin/env python3
"""Shared MiniMax H3 model inventory and Hugging Face provisioning.

This module is intentionally independent of Gradio, ComfyUI, and Modal so the
local and Modal deployment paths use exactly the same model sources, manifest
rules, parallel download behavior, and generated h3_models.json schema.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_REPO = "lilcheaty/MiniMax-H3-NVFP4"
ORIGINAL_MODEL_REPO = "Comfy-Org/MiniMax-H3"
TURBO_REPO = "lightx2v/Minimax-h3-Turbo"
LARRY_TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
EXPERIMENTAL_MODEL_REPO = "Kijai/MiniMax-H3-experimental"
TEXT_ENCODER_REPO = "Comfy-Org/MiniMax-H3"
SEEDVR2_REPO = "Comfy-Org/SeedVR2"
LTX25_REPO = "Lightricks/LTX-2.5"
LTX25_NVFP4_COMFY_REPO = "BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy"
LTX25_PIXEL_UPSCALER_REPO = (
    "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
)
LTX23_REPO = "Lightricks/LTX-2.3"

HF_METADATA_WORKERS = 2
HF_DOWNLOAD_WORKERS = 6
MIN_VALID_MODEL_BYTES = 1024 * 1024


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve an explicit/Modal token, then the active Hugging Face CLI login."""
    configured = token or os.getenv("HF_TOKEN")
    if configured:
        return configured

    # ``hf auth login`` persists the active token under HF_HOME. Using the Hub
    # helper keeps standalone launches aligned with the CLI's path overrides
    # and multi-token selection instead of assuming ~/.cache/huggingface/token.
    from huggingface_hub import get_token

    return get_token()


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    folder: str
    filename: str
    source: str
    local_filename: str | None = None
    expected_sha256: str | None = None

    @property
    def local_name(self) -> str:
        return self.local_filename or Path(self.filename).name


MODEL_SPECS: dict[str, ModelSpec] = {
    "speed_fl2va": ModelSpec(
        MODEL_REPO,
        "diffusion_models",
        "minimax_h3_fl2va_pruned_nvfp4.safetensors",
        "Speed · single-pass NVFP4",
    ),
    "speed_ref2va": ModelSpec(
        MODEL_REPO,
        "diffusion_models",
        "minimax_h3_ref2va_pruned_nvfp4.safetensors",
        "Speed · single-pass NVFP4",
    ),
    "quality_fl2va": ModelSpec(
        MODEL_REPO,
        "diffusion_models",
        "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors",
        "Quality · mixed NVFP4 / FP8 / INT8 ConvRot",
    ),
    "quality_ref2va": ModelSpec(
        MODEL_REPO,
        "diffusion_models",
        "minimax_h3_ref2va_pruned_nvfp4_convrot_int8.safetensors",
        "Quality · mixed NVFP4 / FP8 / INT8 ConvRot",
    ),
    "original_fl2va": ModelSpec(
        ORIGINAL_MODEL_REPO,
        "diffusion_models",
        "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "Original · BF16",
    ),
    "original_ref2va": ModelSpec(
        ORIGINAL_MODEL_REPO,
        "diffusion_models",
        "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
        "Original · BF16",
    ),
    "text_encoder": ModelSpec(
        TEXT_ENCODER_REPO,
        "text_encoders",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "Official Qwen3-VL 32B NVFP4/AWQ text encoder",
    ),
    "video_vae": ModelSpec(
        MODEL_REPO,
        "vae",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "FP16 video VAE",
    ),
    "video_vae_int8": ModelSpec(
        EXPERIMENTAL_MODEL_REPO,
        "vae",
        "minimax_h3_video_vae_int8_convrot.safetensors",
        "Experimental INT8 ConvRot video VAE",
    ),
    "audio_vae": ModelSpec(
        MODEL_REPO,
        "vae",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "FP32 audio VAE",
    ),
    "turbo_lora": ModelSpec(
        TURBO_REPO,
        "loras",
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "LightX2V Turbo 4-step v1.0 · official 768p ComfyUI BF16",
    ),
    "turbo_ref_lora": ModelSpec(
        TURBO_REPO,
        "loras",
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "LightX2V Ref2V Turbo 4-step v0.1 · official ComfyUI BF16",
    ),
    "turbo_8step_lora": ModelSpec(
        TURBO_REPO,
        "loras",
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "LightX2V Turbo 8-step v1.0 · official 544p ComfyUI BF16",
    ),
    "larry_turbo_lora": ModelSpec(
        LARRY_TURBO_REPO,
        "loras",
        "minimax_h3_turbo_v4_step600_ema.safetensors",
        "Larry v4-600 EMA Turbo · recommended 6-step quality option",
    ),
    "seedvr2_3b_nvfp4": ModelSpec(
        SEEDVR2_REPO,
        "diffusion_models",
        "diffusion_models/seedvr2_3b_nvfp4.safetensors",
        "SeedVR2 3B NVFP4 · lazy native 2x upscale model",
    ),
    "seedvr2_3b_int8": ModelSpec(
        SEEDVR2_REPO,
        "diffusion_models",
        "diffusion_models/seedvr2_3b_int8_convrot.safetensors",
        "SeedVR2 3B INT8 ConvRot · lazy native 2x upscale model",
    ),
    "seedvr2_7b_nvfp4": ModelSpec(
        SEEDVR2_REPO,
        "diffusion_models",
        "diffusion_models/seedvr2_7b_nvfp4.safetensors",
        "SeedVR2 7B NVFP4 · lazy native 2x upscale model",
    ),
    "seedvr2_7b_sharp_nvfp4": ModelSpec(
        SEEDVR2_REPO,
        "diffusion_models",
        "diffusion_models/seedvr2_7b_sharp_nvfp4.safetensors",
        "SeedVR2 7B Sharp NVFP4 · lazy native 2x upscale model",
    ),
    "seedvr2_vae": ModelSpec(
        SEEDVR2_REPO,
        "vae",
        "vae/seedvr2_ema_vae_fp16.safetensors",
        "SeedVR2 FP16 VAE · lazy native upscale asset",
    ),
    "ltx25_distilled": ModelSpec(
        LTX25_REPO,
        "diffusion_models",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "LTX-2.5 22B distilled transformer; lazy 8-step model",
    ),
    "ltx25_distilled_nvfp4": ModelSpec(
        LTX25_NVFP4_COMFY_REPO,
        "diffusion_models",
        "ltx-2.5-22b-distilled-transformer-nvfp4-comfy.safetensors",
        "LTX-2.5 22B distilled NVFP4; official weights with Comfy quant markers",
        local_filename="ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
        expected_sha256=(
            "2f3599d1adf22fc4c4a5bb9328cb42d64f449ed78dce5f47a16f098481bdee74"
        ),
    ),
    "ltx25_distilled_int8": ModelSpec(
        LTX25_REPO,
        "diffusion_models",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "LTX-2.5 22B distilled Comfy INT8 ConvRot transformer",
    ),
    "ltx25_text_encoder": ModelSpec(
        LTX25_REPO,
        "text_encoders",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "Gemma 4 12B text encoder fine-tuned for LTX-2.5",
    ),
    "ltx25_video_vae": ModelSpec(
        LTX25_REPO,
        "vae",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "LTX-2.5 convolutional video VAE; lower-memory decoder",
    ),
    "ltx25_audio_vae": ModelSpec(
        LTX25_REPO,
        "vae",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "LTX-2.5 audio VAE",
    ),
    "ltx25_video_vae_full": ModelSpec(
        LTX25_REPO,
        "vae",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        "LTX-2.5 diffusion-decoder video VAE for official workflows",
    ),
    "ltx25_text_enhancer": ModelSpec(
        "Comfy-Org/gemma-4",
        "text_encoders",
        "text_encoders/gemma4_e2b_it_bf16.safetensors",
        "Optional Gemma 4 prompt enhancer used by official workflows",
    ),
    "ltx25_spatial_upscaler": ModelSpec(
        LTX23_REPO,
        "latent_upscale_models",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "LTX 2x latent upscaler for official two-stage workflows",
    ),
    "ltx25_pixel_upscaler_x2": ModelSpec(
        LTX25_PIXEL_UPSCALER_REPO,
        "loras",
        "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        "LTX-2.5 22B IC-LoRA generative pixel-space 2x video upscaler",
    ),
    "ltx25_iclora_ingredients": ModelSpec(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
        "loras",
        "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "LTX IC-LoRA ingredients/reference-sheet control",
    ),
    "ltx25_iclora_in_outpaint": ModelSpec(
        "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
        "loras",
        "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "LTX IC-LoRA video inpainting and outpainting",
    ),
    "ltx25_iclora_motion_track": ModelSpec(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        "loras",
        "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "LTX IC-LoRA sparse motion-track control",
    ),
    "ltx25_iclora_union_control": ModelSpec(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "loras",
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "LTX IC-LoRA pose, depth, and canny union control",
    ),
    "ltx25_iclora_instant_shave": ModelSpec(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave",
        "loras",
        "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors",
        "LTX IC-LoRA instant-shave video edit",
    ),
}

PROFILE_MODEL_KEYS = {
    "speed": ("speed_fl2va", "speed_ref2va"),
    "quality": ("quality_fl2va", "quality_ref2va"),
    "original": ("original_fl2va", "original_ref2va"),
}
PROFILE_LABELS = {
    "speed": "Speed",
    "quality": "Quality",
    "original": "Original",
}
PRELOAD_PROFILES = ("quality",)
PROFILE_MODEL_KEY_SET = frozenset(
    key for keys in PROFILE_MODEL_KEYS.values() for key in keys
)
SEEDVR2_MODEL_CHOICES = {
    "3B NVFP4": "seedvr2_3b_nvfp4",
    "3B INT8": "seedvr2_3b_int8",
    "7B NVFP4": "seedvr2_7b_nvfp4",
    "7B Sharp NVFP4": "seedvr2_7b_sharp_nvfp4",
}
DEFAULT_SEEDVR2_MODEL = "7B NVFP4"
SEEDVR2_UPSCALE_MODEL_KEYS = (
    *SEEDVR2_MODEL_CHOICES.values(),
    "seedvr2_vae",
)
LAZY_POSTPROCESS_MODEL_KEYS = (
    *SEEDVR2_UPSCALE_MODEL_KEYS,
    "ltx25_pixel_upscaler_x2",
)
LTX25_MODEL_CHOICES = {
    "NVFP4 Comfy-ready (Blackwell recommended)": "ltx25_distilled_nvfp4",
    "INT8 ConvRot": "ltx25_distilled_int8",
    "BF16": "ltx25_distilled",
}
DEFAULT_LTX25_MODEL = "NVFP4 Comfy-ready (Blackwell recommended)"
LTX25_SHARED_MODEL_KEYS = (
    "ltx25_text_encoder",
    "ltx25_video_vae",
    "ltx25_audio_vae",
)
LTX25_MODEL_KEYS = (
    *LTX25_MODEL_CHOICES.values(),
    *LTX25_SHARED_MODEL_KEYS,
)
LTX25_ICLORA_MODEL_KEYS = (
    "ltx25_iclora_ingredients",
    "ltx25_iclora_in_outpaint",
    "ltx25_iclora_motion_track",
    "ltx25_iclora_union_control",
    "ltx25_iclora_instant_shave",
)
LTX25_OFFICIAL_WORKFLOW_MODEL_KEYS = (
    "ltx25_video_vae_full",
    "ltx25_text_enhancer",
    "ltx25_spatial_upscaler",
    *LTX25_ICLORA_MODEL_KEYS,
)
LAZY_OPTIONAL_MODEL_KEYS = (
    "video_vae_int8",
    *LAZY_POSTPROCESS_MODEL_KEYS,
    *LTX25_MODEL_KEYS,
    *LTX25_OFFICIAL_WORKFLOW_MODEL_KEYS,
)
SHARED_MODEL_KEYS = tuple(
    key for key in MODEL_SPECS
    if key not in PROFILE_MODEL_KEY_SET and key not in LAZY_OPTIONAL_MODEL_KEYS
)
PRELOAD_MODEL_KEYS = (
    *(
        key
        for profile in PRELOAD_PROFILES
        for key in PROFILE_MODEL_KEYS[profile]
    ),
    *SHARED_MODEL_KEYS,
)


def read_json(path: Path, default: Any):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def model_manifest_key(spec: ModelSpec) -> str:
    return f"{spec.folder}/{spec.local_name}"


def model_file_matches_manifest(
    root: Path,
    manifest: dict[str, Any],
    spec: ModelSpec,
) -> bool:
    """Return whether a local model matches its recorded source and size."""
    path = Path(root) / spec.folder / spec.local_name
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= MIN_VALID_MODEL_BYTES:
        return False

    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return False
    entry = files.get(model_manifest_key(spec), {})
    if not isinstance(entry, dict):
        return False
    recorded_size = entry.get("size")
    return (
        entry.get("repo_id") == spec.repo_id
        and entry.get("filename") == spec.filename
        and isinstance(recorded_size, int)
        and recorded_size == size
        and (
            spec.expected_sha256 is None
            or entry.get("sha256") == spec.expected_sha256
        )
    )


def stale_model_keys(
    *,
    root: Path,
    manifest_path: Path,
    model_keys: Iterable[str],
) -> list[str]:
    """Return selected model keys missing or stale against the local manifest."""
    selected = tuple(model_keys)
    unknown = sorted(set(selected) - set(MODEL_SPECS))
    if unknown:
        raise KeyError("Unknown model keys: " + ", ".join(unknown))
    manifest = read_json(manifest_path, {"files": {}})
    if not isinstance(manifest, dict):
        manifest = {"files": {}}
    return [
        key
        for key in selected
        if not model_file_matches_manifest(root, manifest, MODEL_SPECS[key])
    ]


def metadata_value(value: Any, name: str):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _fetch_repo(repo_id: str, token: str | None):
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(
        repo_id,
        revision="main",
        files_metadata=True,
    )
    return info.sha, {item.rfilename: item for item in info.siblings}


def _fetch_repositories(
    repo_ids: list[str],
    token: str | None,
    metadata_workers: int,
) -> dict[str, tuple[str, dict[str, Any]] | Exception]:
    results: dict[str, tuple[str, dict[str, Any]] | Exception] = {}
    workers = max(1, min(metadata_workers, len(repo_ids)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_repo, repo_id, token): repo_id
            for repo_id in repo_ids
        }
        for future in as_completed(futures):
            repo_id = futures[future]
            try:
                results[repo_id] = future.result()
            except Exception as exc:
                results[repo_id] = exc
    return results


def _plan_model(
    *,
    root: Path,
    manifest: dict[str, Any],
    repo_results: dict[str, tuple[str, dict[str, Any]] | Exception],
    key: str,
    spec: ModelSpec,
    log_prefix: str,
) -> dict[str, Any]:
    dest_dir = root / spec.folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.local_name
    manifest_key = model_manifest_key(spec)
    old = manifest.setdefault("files", {}).get(manifest_key, {})

    repo_result = repo_results[spec.repo_id]
    if isinstance(repo_result, Exception):
        if dest.is_file() and dest.stat().st_size > MIN_VALID_MODEL_BYTES:
            print(
                f"{log_prefix} WARNING remote check failed for {key}; "
                f"reusing {dest}: {repo_result}",
                flush=True,
            )
            return {
                "key": key,
                "spec": spec,
                "dest": dest,
                "manifest_key": manifest_key,
                "remote_ok": False,
                "needs_download": False,
            }
        raise RuntimeError(
            f"Cannot check or download {key}: {repo_result}"
        ) from repo_result

    revision, files = repo_result
    sibling = files.get(spec.filename)
    if sibling is None:
        raise FileNotFoundError(
            f"{spec.repo_id}/{spec.filename} is not present"
        )

    lfs = getattr(sibling, "lfs", None)
    sha256 = metadata_value(lfs, "sha256")
    size = getattr(sibling, "size", None) or metadata_value(lfs, "size")
    blob_id = getattr(sibling, "blob_id", None)
    identity = sha256 or blob_id or f"{revision}:{spec.filename}:{size}"
    if spec.expected_sha256 and sha256 != spec.expected_sha256:
        raise RuntimeError(
            f"Unexpected SHA-256 for {spec.repo_id}/{spec.filename}: "
            f"{sha256 or 'missing'} != {spec.expected_sha256}"
        )

    valid = dest.is_file() and dest.stat().st_size > MIN_VALID_MODEL_BYTES
    size_ok = size is None or (valid and dest.stat().st_size == int(size))
    identity_ok = old.get("remote_identity") == identity
    needs_download = not (valid and size_ok and identity_ok)

    if needs_download:
        if not valid:
            reason = "missing"
        elif not old:
            reason = "untracked local file; refreshing once"
        elif not identity_ok:
            reason = "remote file changed"
        else:
            reason = "local size mismatch"
        print(f"{log_prefix} queued {dest.name} ({reason})", flush=True)
    else:
        print(
            f"{log_prefix} up to date {dest.name} ({str(identity)[:16]})",
            flush=True,
        )

    return {
        "key": key,
        "spec": spec,
        "dest": dest,
        "manifest_key": manifest_key,
        "remote_ok": True,
        "revision": revision,
        "sha256": sha256,
        "size": int(size) if size is not None else None,
        "blob_id": blob_id,
        "identity": identity,
        "needs_download": needs_download,
    }


def _download_model(
    plan: dict[str, Any],
    token: str | None,
    log_prefix: str,
) -> str:
    from huggingface_hub import hf_hub_download

    spec: ModelSpec = plan["spec"]
    dest: Path = plan["dest"]
    print(f"{log_prefix} downloading {dest.name}", flush=True)

    cached = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=plan["revision"],
            token=token,
        )
    )
    expected = plan["size"]
    if expected is not None and cached.stat().st_size != expected:
        raise RuntimeError(
            f"Downloaded size mismatch for {spec.filename}: "
            f"{cached.stat().st_size} != {expected}"
        )

    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copy2(cached, tmp)
    os.replace(tmp, dest)
    print(f"{log_prefix} ready {dest.name}", flush=True)
    return plan["key"]


def _profile_config(profile: str) -> dict[str, str]:
    fl2va_key, ref2va_key = PROFILE_MODEL_KEYS[profile]
    fl2va = MODEL_SPECS[fl2va_key]
    ref2va = MODEL_SPECS[ref2va_key]
    return {
        "label": PROFILE_LABELS[profile],
        "fl2va": fl2va.local_name,
        "ref2va": ref2va.local_name,
        "fl2va_source": fl2va.source,
        "ref2va_source": ref2va.source,
    }


def _build_config(manifest_name: str) -> dict[str, Any]:
    text = MODEL_SPECS["text_encoder"]
    video_vae = MODEL_SPECS["video_vae"]
    video_vae_int8 = MODEL_SPECS["video_vae_int8"]
    audio_vae = MODEL_SPECS["audio_vae"]
    turbo_lora = MODEL_SPECS["turbo_lora"]
    turbo_ref_lora = MODEL_SPECS["turbo_ref_lora"]
    turbo_8step_lora = MODEL_SPECS["turbo_8step_lora"]
    larry_turbo_lora = MODEL_SPECS["larry_turbo_lora"]
    seedvr2_vae = MODEL_SPECS["seedvr2_vae"]
    seedvr2_models = {
        label: MODEL_SPECS[key].local_name
        for label, key in SEEDVR2_MODEL_CHOICES.items()
    }

    return {
        "schema_version": 13,
        "default_profile": "quality",
        "profiles": {
            profile: _profile_config(profile)
            for profile in PROFILE_MODEL_KEYS
        },
        "text_encoder": text.local_name,
        "video_vae": video_vae.local_name,
        "video_vae_int8": video_vae_int8.local_name,
        "video_vae_int8_source": video_vae_int8.source,
        "audio_vae": audio_vae.local_name,
        "turbo_lora": turbo_lora.local_name,
        "turbo_source": turbo_lora.source,
        "turbo_ref_lora": turbo_ref_lora.local_name,
        "turbo_ref_source": turbo_ref_lora.source,
        "turbo_8step_lora": turbo_8step_lora.local_name,
        "turbo_8step_source": turbo_8step_lora.source,
        "turbo_8step_ref_lora": turbo_8step_lora.local_name,
        "turbo_8step_ref_source": turbo_8step_lora.source,
        "larry_turbo_lora": larry_turbo_lora.local_name,
        "larry_turbo_source": larry_turbo_lora.source,
        # Larry's FL2VA-trained LoRA is also exposed for experimental Ref2VA.
        "larry_turbo_ref_lora": larry_turbo_lora.local_name,
        "larry_turbo_ref_source": larry_turbo_lora.source,
        # Retain the original fields for compatibility with older app code.
        "seedvr2_dit": seedvr2_models[DEFAULT_SEEDVR2_MODEL],
        "seedvr2_dit_source": MODEL_SPECS[
            SEEDVR2_MODEL_CHOICES[DEFAULT_SEEDVR2_MODEL]
        ].source,
        "seedvr2_models": seedvr2_models,
        "seedvr2_vae": seedvr2_vae.local_name,
        "seedvr2_vae_source": seedvr2_vae.source,
        "ltx25_models": {
            "transformers": {
                label: MODEL_SPECS[key].local_name
                for label, key in LTX25_MODEL_CHOICES.items()
            },
            **{
                key.removeprefix("ltx25_"): MODEL_SPECS[key].local_name
                for key in LTX25_SHARED_MODEL_KEYS
            },
        },
        "turbo_supported_profiles": list(PROFILE_MODEL_KEYS),
        "turbo_supported_modes": ["fl2va", "ref2va"],
        "manifest": manifest_name,
    }


def sync_models(
    *,
    root: Path,
    manifest_path: Path,
    token: str | None = None,
    log_prefix: str = "[h3-models]",
    metadata_workers: int = HF_METADATA_WORKERS,
    download_workers: int = HF_DOWNLOAD_WORKERS,
    model_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Refresh selected model files and return the complete model catalog."""
    token = resolve_hf_token(token)
    root = Path(root)
    manifest_path = Path(manifest_path)
    manifest = read_json(
        manifest_path,
        {"schema_version": 1, "files": {}},
    )
    if not isinstance(manifest, dict):
        manifest = {"schema_version": 1, "files": {}}
    if not isinstance(manifest.get("files"), dict):
        manifest["files"] = {}

    selected_keys = tuple(MODEL_SPECS if model_keys is None else model_keys)
    unknown = sorted(set(selected_keys) - set(MODEL_SPECS))
    if unknown:
        raise KeyError("Unknown model keys: " + ", ".join(unknown))
    selected_specs = {key: MODEL_SPECS[key] for key in selected_keys}
    repo_ids = sorted({spec.repo_id for spec in selected_specs.values()})
    repo_results = _fetch_repositories(
        repo_ids,
        token,
        metadata_workers,
    )

    plans = [
        _plan_model(
            root=root,
            manifest=manifest,
            repo_results=repo_results,
            key=key,
            spec=spec,
            log_prefix=log_prefix,
        )
        for key, spec in selected_specs.items()
    ]

    stale = [plan for plan in plans if plan["needs_download"]]
    downloaded_keys: set[str] = set()
    download_failure: RuntimeError | None = None
    if stale:
        workers = max(1, min(download_workers, len(stale)))
        print(
            f"{log_prefix} downloading {len(stale)} model files "
            f"with {workers} parallel workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download_model, plan, token, log_prefix): plan
                for plan in stale
            }
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    downloaded_keys.add(future.result())
                except Exception as exc:
                    if download_failure is None:
                        download_failure = RuntimeError(
                            f"Parallel download failed for {plan['key']}: {exc}"
                        )
                        download_failure.__cause__ = exc

    files_manifest = manifest.setdefault("files", {})
    for plan in plans:
        if not plan["remote_ok"]:
            continue
        # Record successful workers even when a sibling worker failed. This
        # prevents a completed multi-gigabyte file from being downloaded again
        # merely because the parallel batch could not finish every model.
        if plan["needs_download"] and plan["key"] not in downloaded_keys:
            continue
        spec: ModelSpec = plan["spec"]
        dest: Path = plan["dest"]
        files_manifest[plan["manifest_key"]] = {
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "local_path": plan["manifest_key"],
            "source": spec.source,
            "repo_revision": plan["revision"],
            "blob_id": plan["blob_id"],
            "sha256": plan["sha256"],
            "size": (
                plan["size"]
                if plan["size"] is not None
                else dest.stat().st_size
            ),
            "remote_identity": plan["identity"],
        }

    manifest["schema_version"] = 1
    manifest["repo_id"] = MODEL_REPO
    manifest["checked_at_unix"] = int(time.time())
    write_json_atomic(manifest_path, manifest)

    if download_failure is not None:
        raise download_failure

    return _build_config(manifest_path.name)


def validate_config_files(
    root: Path,
    config: dict[str, Any],
    profiles: Iterable[str] = PRELOAD_PROFILES,
) -> list[str]:
    """Return missing/invalid local files referenced by h3_models.json."""
    root = Path(root)
    required = [
        ("text_encoders", config.get("text_encoder")),
        ("vae", config.get("video_vae")),
        ("vae", config.get("audio_vae")),
        ("loras", config.get("turbo_lora")),
        ("loras", config.get("turbo_ref_lora")),
        ("loras", config.get("turbo_8step_lora")),
        ("loras", config.get("larry_turbo_lora")),
    ]
    for profile in profiles:
        data = config.get("profiles", {}).get(profile, {})
        required.extend(
            [
                ("diffusion_models", data.get("fl2va")),
                ("diffusion_models", data.get("ref2va")),
            ]
        )

    return [
        f"{folder}/{name}"
        for folder, name in required
        if (
            not name
            or not (root / folder / name).is_file()
            or (root / folder / name).stat().st_size <= MIN_VALID_MODEL_BYTES
        )
    ]


def selftest() -> None:
    import tempfile
    from unittest.mock import patch

    assert resolve_hf_token("explicit-token") == "explicit-token"
    with patch.dict(os.environ, {"HF_TOKEN": "environment-token"}):
        assert resolve_hf_token() == "environment-token"
    with (
        patch.dict(os.environ, {"HF_TOKEN": ""}),
        patch("huggingface_hub.get_token", return_value="cli-token"),
    ):
        assert resolve_hf_token() == "cli-token"

    assert MODEL_SPECS["text_encoder"].repo_id == TEXT_ENCODER_REPO
    assert MODEL_SPECS["text_encoder"].local_name == (
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    assert set(MODEL_SPECS) == {
        "speed_fl2va",
        "speed_ref2va",
        "quality_fl2va",
        "quality_ref2va",
        "original_fl2va",
        "original_ref2va",
        "text_encoder",
        "video_vae",
        "video_vae_int8",
        "audio_vae",
        "turbo_lora",
        "turbo_ref_lora",
        "turbo_8step_lora",
        "larry_turbo_lora",
        "seedvr2_3b_nvfp4",
        "seedvr2_3b_int8",
        "seedvr2_7b_nvfp4",
        "seedvr2_7b_sharp_nvfp4",
        "seedvr2_vae",
        "ltx25_distilled",
        "ltx25_distilled_nvfp4",
        "ltx25_distilled_int8",
        "ltx25_text_encoder",
        "ltx25_video_vae",
        "ltx25_audio_vae",
        "ltx25_video_vae_full",
        "ltx25_text_enhancer",
        "ltx25_spatial_upscaler",
        "ltx25_pixel_upscaler_x2",
        "ltx25_iclora_ingredients",
        "ltx25_iclora_in_outpaint",
        "ltx25_iclora_motion_track",
        "ltx25_iclora_union_control",
        "ltx25_iclora_instant_shave",
    }

    cfg = _build_config("manifest.json")
    assert set(PRELOAD_MODEL_KEYS).isdisjoint(PROFILE_MODEL_KEYS["speed"])
    assert set(PRELOAD_MODEL_KEYS).isdisjoint(PROFILE_MODEL_KEYS["original"])
    assert set(PROFILE_MODEL_KEYS["quality"]).issubset(PRELOAD_MODEL_KEYS)
    assert set(SEEDVR2_UPSCALE_MODEL_KEYS).isdisjoint(PRELOAD_MODEL_KEYS)
    assert tuple(cfg["profiles"]) == tuple(PROFILE_MODEL_KEYS)
    assert cfg["schema_version"] == 13
    assert cfg["default_profile"] == "quality"
    assert cfg["profiles"]["quality"]["fl2va"] == (
        "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors"
    )
    assert cfg["profiles"]["original"]["fl2va"] == (
        "minimax_h3_fl2va_pruned_bf16.safetensors"
    )
    assert cfg["text_encoder"] == (
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    )
    assert cfg["video_vae_int8"] == (
        "minimax_h3_video_vae_int8_convrot.safetensors"
    )
    assert "video_vae_int8" not in PRELOAD_MODEL_KEYS
    assert cfg["turbo_lora"] == "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    assert cfg["turbo_ref_lora"] == (
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    assert cfg["turbo_8step_lora"] == "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    assert cfg["turbo_8step_ref_lora"] == cfg["turbo_8step_lora"]
    assert {"turbo_lora", "turbo_ref_lora", "turbo_8step_lora"}.issubset(
        PRELOAD_MODEL_KEYS
    )
    assert cfg["larry_turbo_lora"] == "minimax_h3_turbo_v4_step600_ema.safetensors"
    assert cfg["larry_turbo_ref_lora"] == cfg["larry_turbo_lora"]
    assert cfg["seedvr2_dit"] == "seedvr2_7b_nvfp4.safetensors"
    assert cfg["seedvr2_models"] == {
        "3B NVFP4": "seedvr2_3b_nvfp4.safetensors",
        "3B INT8": "seedvr2_3b_int8_convrot.safetensors",
        "7B NVFP4": "seedvr2_7b_nvfp4.safetensors",
        "7B Sharp NVFP4": "seedvr2_7b_sharp_nvfp4.safetensors",
    }
    assert cfg["seedvr2_vae"] == "seedvr2_ema_vae_fp16.safetensors"
    assert cfg["ltx25_models"] == {
        "transformers": {
            "NVFP4 Comfy-ready (Blackwell recommended)": (
                "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
            ),
            "INT8 ConvRot": (
                "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
            ),
            "BF16": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        },
        "text_encoder": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": "ltx-2.5-video-vae-conv-bf16.safetensors",
        "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    }
    assert DEFAULT_LTX25_MODEL == "NVFP4 Comfy-ready (Blackwell recommended)"
    nvfp4 = MODEL_SPECS["ltx25_distilled_nvfp4"]
    assert nvfp4.repo_id == LTX25_NVFP4_COMFY_REPO
    assert nvfp4.filename.endswith("-nvfp4-comfy.safetensors")
    assert nvfp4.expected_sha256 == (
        "2f3599d1adf22fc4c4a5bb9328cb42d64f449ed78dce5f47a16f098481bdee74"
    )
    assert MODEL_SPECS["ltx25_pixel_upscaler_x2"].repo_id == (
        LTX25_PIXEL_UPSCALER_REPO
    )
    with tempfile.TemporaryDirectory() as model_temp:
        root = Path(model_temp)
        path = root / nvfp4.folder / nvfp4.local_name
        path.parent.mkdir(parents=True)
        with path.open("wb") as handle:
            handle.seek(MIN_VALID_MODEL_BYTES)
            handle.write(b"x")
        manifest = {
            "files": {
                model_manifest_key(nvfp4): {
                    "repo_id": nvfp4.repo_id,
                    "filename": nvfp4.filename,
                    "sha256": nvfp4.expected_sha256,
                    "size": path.stat().st_size,
                }
            }
        }
        assert model_file_matches_manifest(root, manifest, nvfp4)
        manifest["files"][model_manifest_key(nvfp4)]["size"] += 1
        assert not model_file_matches_manifest(root, manifest, nvfp4)
    assert set(LTX25_MODEL_KEYS).isdisjoint(PRELOAD_MODEL_KEYS)
    assert len(LTX25_ICLORA_MODEL_KEYS) == 5
    assert all(MODEL_SPECS[key].folder == "loras" for key in LTX25_ICLORA_MODEL_KEYS)
    assert set(LTX25_OFFICIAL_WORKFLOW_MODEL_KEYS).isdisjoint(
        PRELOAD_MODEL_KEYS
    )
    assert cfg["turbo_supported_profiles"] == ["speed", "quality", "original"]
    assert cfg["turbo_supported_modes"] == ["fl2va", "ref2va"]
    print("h3_models selftest OK")


if __name__ == "__main__":
    selftest()
