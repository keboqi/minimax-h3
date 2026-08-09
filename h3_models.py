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
from typing import Any


MODEL_REPO = "lilcheaty/MiniMax-H3-NVFP4"
TURBO_REPO = "Kijai/MiniMax-H3_comfy"
LARRY_TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
TEXT_ENCODER_REPO = "sakamakismile/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4"

HF_METADATA_WORKERS = 2
HF_DOWNLOAD_WORKERS = 6
MIN_VALID_MODEL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    folder: str
    filename: str
    source: str

    @property
    def local_name(self) -> str:
        return Path(self.filename).name


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
    "text_encoder": ModelSpec(
        TEXT_ENCODER_REPO,
        "text_encoders",
        "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "Heretic NVFP4 text encoder",
    ),
    "video_vae": ModelSpec(
        MODEL_REPO,
        "vae",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "FP16 video VAE",
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
        "loras/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
        "LightX2V Turbo 4-step v0.1 · Kijai full Comfy conversion",
    ),
    "larry_turbo_lora": ModelSpec(
        LARRY_TURBO_REPO,
        "loras",
        "minimax_h3_turbo_v4_step600_ema.safetensors",
        "Larry v4-600 EMA Turbo · recommended 6-step quality option",
    ),
}


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
    manifest_key = f"{spec.folder}/{dest.name}"
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


def _build_config(
    installed: dict[str, tuple[str, str]],
    manifest_name: str,
) -> dict[str, Any]:
    speed_fl, speed_fl_source = installed["speed_fl2va"]
    speed_ref, speed_ref_source = installed["speed_ref2va"]
    quality_fl, quality_fl_source = installed["quality_fl2va"]
    quality_ref, quality_ref_source = installed["quality_ref2va"]
    text, _ = installed["text_encoder"]
    video_vae, _ = installed["video_vae"]
    audio_vae, _ = installed["audio_vae"]
    turbo_lora, turbo_source = installed["turbo_lora"]
    larry_turbo_lora, larry_turbo_source = installed["larry_turbo_lora"]

    return {
        "schema_version": 4,
        "default_profile": "speed",
        "profiles": {
            "speed": {
                "label": "Speed",
                "fl2va": speed_fl,
                "ref2va": speed_ref,
                "fl2va_source": speed_fl_source,
                "ref2va_source": speed_ref_source,
            },
            "quality": {
                "label": "Quality",
                "fl2va": quality_fl,
                "ref2va": quality_ref,
                "fl2va_source": quality_fl_source,
                "ref2va_source": quality_ref_source,
            },
        },
        "text_encoder": text,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "turbo_lora": turbo_lora,
        "turbo_source": turbo_source,
        # The current LightX2V file is shared temporarily. Keeping a distinct
        # config key makes a future Ref2VA-specific Turbo asset a data-only swap.
        "turbo_ref_lora": turbo_lora,
        "turbo_ref_source": turbo_source,
        "larry_turbo_lora": larry_turbo_lora,
        "larry_turbo_source": larry_turbo_source,
        # Larry's FL2VA-trained LoRA is also exposed for experimental Ref2VA.
        "larry_turbo_ref_lora": larry_turbo_lora,
        "larry_turbo_ref_source": larry_turbo_source,
        "turbo_supported_profiles": ["speed", "quality"],
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
) -> dict[str, Any]:
    """Refresh stale/missing H3 model files and return h3_models.json data."""
    root = Path(root)
    manifest_path = Path(manifest_path)
    manifest = read_json(
        manifest_path,
        {"schema_version": 1, "files": {}},
    )

    repo_ids = sorted({spec.repo_id for spec in MODEL_SPECS.values()})
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
        for key, spec in MODEL_SPECS.items()
    ]

    stale = [plan for plan in plans if plan["needs_download"]]
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
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel download failed for {plan['key']}: {exc}"
                    ) from exc

    files_manifest = manifest.setdefault("files", {})
    for plan in plans:
        if not plan["remote_ok"]:
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

    installed = {
        plan["key"]: (plan["dest"].name, plan["spec"].source)
        for plan in plans
    }
    return _build_config(installed, manifest_path.name)


def validate_config_files(root: Path, config: dict[str, Any]) -> list[str]:
    """Return missing/invalid local files referenced by h3_models.json."""
    root = Path(root)
    required = [
        ("text_encoders", config.get("text_encoder")),
        ("vae", config.get("video_vae")),
        ("vae", config.get("audio_vae")),
        ("loras", config.get("turbo_lora")),
        ("loras", config.get("larry_turbo_lora")),
    ]
    for profile in ("speed", "quality"):
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
    assert MODEL_SPECS["text_encoder"].repo_id == TEXT_ENCODER_REPO
    assert MODEL_SPECS["text_encoder"].local_name == (
        "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"
    )
    assert set(MODEL_SPECS) == {
        "speed_fl2va",
        "speed_ref2va",
        "quality_fl2va",
        "quality_ref2va",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "turbo_lora",
        "larry_turbo_lora",
    }

    fake = {
        key: (spec.local_name, spec.source)
        for key, spec in MODEL_SPECS.items()
    }
    cfg = _build_config(fake, "manifest.json")
    assert cfg["schema_version"] == 4
    assert cfg["profiles"]["quality"]["fl2va"] == (
        "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors"
    )
    assert cfg["text_encoder"] == (
        "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"
    )
    assert cfg["turbo_lora"] == "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"
    assert cfg["turbo_ref_lora"] == cfg["turbo_lora"]
    assert cfg["larry_turbo_lora"] == "minimax_h3_turbo_v4_step600_ema.safetensors"
    assert cfg["larry_turbo_ref_lora"] == cfg["larry_turbo_lora"]
    assert cfg["turbo_supported_profiles"] == ["speed", "quality"]
    assert cfg["turbo_supported_modes"] == ["fl2va", "ref2va"]
    print("h3_models selftest OK")


if __name__ == "__main__":
    selftest()
