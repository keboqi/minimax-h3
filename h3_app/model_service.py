"""Extracted model service boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from h3_app.catalog import LARRY_TURBO, LIGHTX2V_8STEP_TURBO
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.jobs import JOBS, check_cancelled
from h3_app.model_types import (
    ModelConfig,
    ModelProfile,
    h3_text_encoder_settings,
    ltx25_model_keys,
    model_file_is_ready,
    music3_model_keys,
    seedvr2_upscale_model_names,
    trt_vae_engine_name,
)
from h3_app.policy import h3_latent_upscaler_settings, normalize_turbo_variant
from h3_models import (
    DEFAULT_LTX25_MODEL,
    MODEL_SPECS,
    SEEDVR2_MODEL_CHOICES,
    TRT_VAE_ENGINE_BUILD_ID,
    TRT_VAE_ENGINE_MARKER,
    TRT_VAE_RUNTIME_MODEL_KEYS,
    resolve_hf_token,
    stale_model_keys,
    sync_models,
)

from .progress import no_progress

_TRT_VAE_COMPILE_LOCK = Lock()


def load_model_config(*, runtime: RuntimeConfig) -> ModelConfig:
    if not runtime.models_config.exists():
        raise H3Error(
            f"Missing model configuration: {runtime.models_config}. Run setup_h3.py first."
        )
    data = json.loads(runtime.models_config.read_text(encoding="utf-8"))

    if "profiles" not in data:
        legacy = ModelProfile(
            label="Speed",
            fl2va=data["fl2va"],
            ref2va=data["ref2va"],
            fl2va_source=data.get("fl2va_source", "legacy"),
            ref2va_source=data.get("ref2va_source", "legacy"),
        )
        profiles = {"speed": legacy}
        default_profile = "speed"
    else:
        profiles = {
            key.lower(): ModelProfile(**value)
            for key, value in data["profiles"].items()
        }
        default_profile = str(data.get("default_profile", "speed")).lower()

    return ModelConfig(
        profiles=profiles,
        default_profile=default_profile,
        text_encoder=data["text_encoder"],
        text_encoders=data.get("text_encoders"),
        video_vae=data["video_vae"],
        audio_vae=data["audio_vae"],
        video_vae_int8=data.get("video_vae_int8"),
        video_vae_trt_encoder=data.get("video_vae_trt_encoder"),
        video_vae_trt_decoder=data.get("video_vae_trt_decoder"),
        video_vae_trt_source=data.get("video_vae_trt_source", "unknown"),
        video_vae_int8_source=data.get("video_vae_int8_source", "unknown"),
        image_vae_500k=data.get("image_vae_500k"),
        image_vae_500k_source=data.get("image_vae_500k_source", "unknown"),
        turbo_lora=data.get("turbo_lora"),
        turbo_source=data.get("turbo_source", "unknown"),
        turbo_ref_lora=data.get("turbo_ref_lora", data.get("turbo_lora")),
        turbo_ref_source=data.get(
            "turbo_ref_source", data.get("turbo_source", "unknown")
        ),
        turbo_8step_lora=data.get("turbo_8step_lora"),
        turbo_8step_source=data.get("turbo_8step_source", "unknown"),
        turbo_8step_ref_lora=data.get(
            "turbo_8step_ref_lora", data.get("turbo_8step_lora")
        ),
        turbo_8step_ref_source=data.get(
            "turbo_8step_ref_source", data.get("turbo_8step_source", "unknown")
        ),
        larry_turbo_lora=data.get("larry_turbo_lora"),
        larry_turbo_source=data.get("larry_turbo_source", "unknown"),
        larry_turbo_ref_lora=data.get(
            "larry_turbo_ref_lora", data.get("larry_turbo_lora")
        ),
        larry_turbo_ref_source=data.get(
            "larry_turbo_ref_source", data.get("larry_turbo_source", "unknown")
        ),
        seedvr2_dit=data.get("seedvr2_dit"),
        seedvr2_dit_source=data.get("seedvr2_dit_source", "unknown"),
        seedvr2_models=data.get("seedvr2_models"),
        seedvr2_vae=data.get("seedvr2_vae"),
        seedvr2_vae_source=data.get("seedvr2_vae_source", "unknown"),
    )


def trt_vae_decoder_paths(
    models: ModelConfig, *, runtime: RuntimeConfig
) -> tuple[Path, Path, Path]:
    if not models.video_vae_trt_decoder:
        raise H3Error("TensorRT VAE is not configured. Re-run setup_h3.py.")
    vae_dir = runtime.comfy_dir / "models" / "vae"
    onnx_path = vae_dir / models.video_vae_trt_decoder
    engine_path = vae_dir / trt_vae_engine_name(models.video_vae_trt_decoder)
    return onnx_path, engine_path, vae_dir / TRT_VAE_ENGINE_MARKER


def _load_trt_vae_compiler(node_path: Path, *, runtime: RuntimeConfig) -> Any:
    """Import the installed compiler afresh so setup patches take effect."""
    import importlib.util

    module_name = "_h3_trt_vae_node"
    comfy_path = str(runtime.comfy_dir)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, node_path)
    if spec is None or spec.loader is None:
        raise H3Error(f"Could not load TensorRT VAE compiler: {node_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def trt_vae_runtime_fingerprint(
    models: ModelConfig | None = None, *, runtime: RuntimeConfig
) -> str:
    """Return an environment signature containing build profile, TRT version, and GPU arch."""
    parts = [TRT_VAE_ENGINE_BUILD_ID]
    trt_version = None
    try:
        import tensorrt as trt

        trt_version = getattr(trt, "__version__", None)
    except Exception:
        pass

    if trt_version is None:
        try:
            node_path = (
                runtime.comfy_dir
                / "custom_nodes"
                / "ComfyUI-H3VAE_TRT"
                / "minimax_trt_node.py"
            )
            if node_path.is_file():
                module = _load_trt_vae_compiler(node_path, runtime=runtime)
                if getattr(module, "HAS_TRT", False):
                    trt_version = getattr(module.trt, "__version__", None)
        except Exception:
            pass

    if trt_version:
        parts.append(f"trt_{trt_version}")

    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            parts.append(f"sm_{cap[0]}{cap[1]}")
    except Exception:
        pass

    return ":".join(parts)


def is_trt_engine_loadable(engine_path: Path, *, runtime: RuntimeConfig) -> bool:
    """Verify that the engine file can be deserialized by the active TensorRT runtime."""
    if not engine_path.is_file():
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return True
    except Exception:
        return True

    trt_mod = None
    try:
        import tensorrt as trt

        trt_mod = trt
    except Exception:
        try:
            node_path = (
                runtime.comfy_dir
                / "custom_nodes"
                / "ComfyUI-H3VAE_TRT"
                / "minimax_trt_node.py"
            )
            if node_path.is_file():
                module = _load_trt_vae_compiler(node_path, runtime=runtime)
                if getattr(module, "HAS_TRT", False):
                    trt_mod = module.trt
        except Exception:
            pass

    if trt_mod is None:
        return True

    try:
        logger = trt_mod.Logger(trt_mod.Logger.ERROR)
        runtime = trt_mod.Runtime(logger)
        with engine_path.open("rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            return False
        del engine
        del runtime
        return True
    except Exception:
        return False


def trt_vae_engine_is_current(models: ModelConfig, *, runtime: RuntimeConfig) -> bool:
    _, engine_path, marker_path = trt_vae_decoder_paths(models, runtime=runtime)
    if not engine_path.is_file() or not marker_path.is_file():
        return False
    try:
        expected = trt_vae_runtime_fingerprint(models, runtime=runtime)
        actual = marker_path.read_text(encoding="utf-8").strip()
        if actual != expected:
            return False
    except OSError:
        return False
    return is_trt_engine_loadable(engine_path, runtime=runtime)


def ensure_h3_text_encoder(
    models: ModelConfig, model_choice: str, *, runtime: RuntimeConfig
) -> tuple[str, bool]:
    """Download an optional H3 text encoder on first use."""
    model_key, filename, requires_offload = h3_text_encoder_settings(
        models, model_choice
    )
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        sync_models(
            root=runtime.comfy_dir / "models",
            manifest_path=manifest_path,
            token=resolve_hf_token(),
            log_prefix="[h3-text-encoder-on-demand]",
            model_keys=(model_key,),
            download_workers=1,
        )
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS[model_key].folder / filename
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand H3 text encoder download did not produce {filename}.")
    return filename, requires_offload


def ensure_h3_semantic_bridge(*, runtime: RuntimeConfig) -> None:
    """Fetch only the optional v1 adapter; never preload it with the base models."""
    key = "semantic_bridge_v1"
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(key,),
    ):
        sync_models(
            root=runtime.comfy_dir / "models",
            manifest_path=manifest_path,
            token=resolve_hf_token(),
            model_keys=(key,),
            download_workers=1,
            log_prefix="[h3-semantic-bridge-on-demand]",
        )
    spec = MODEL_SPECS[key]
    if not model_file_is_ready(
        runtime.comfy_dir / "models" / spec.folder / spec.local_name
    ):
        raise H3Error(
            "Semantic Bridge adapter download did not produce a valid model file."
        )


def ensure_h3_latent_upscaler_model(
    model_choice: str, *, runtime: RuntimeConfig
) -> bool:
    """Download the selected native H3 latent upscaler on first use."""
    model_key, filename, _precision = h3_latent_upscaler_settings(model_choice)
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-latent-upscaler-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS[model_key].folder / filename
    )
    if not model_file_is_ready(destination):
        raise H3Error(
            f"On-demand H3 latent upscaler download did not produce {filename}."
        )
    return True


def ensure_profile_model(
    profile_key: str, profile: ModelProfile, mode: str, *, runtime: RuntimeConfig
) -> bool:
    """Download a lazy profile checkpoint before submitting its workflow."""
    reference = str(mode).strip().lower() == "reference media"
    filename = profile.ref2va if reference else profile.fl2va
    model_key = f"{profile_key}_{'ref2va' if reference else 'fl2va'}"
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS[model_key].folder / filename
    )
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand model download did not produce {filename}.")
    return True


def ensure_turbo_lora(
    models: ModelConfig, turbo_variant: str, mode: str, *, runtime: RuntimeConfig
) -> bool:
    """Download a non-default Turbo LoRA only when its variant is selected."""
    variant = normalize_turbo_variant(turbo_variant)
    reference = str(mode).strip().lower() == "reference media"
    if variant == LARRY_TURBO:
        model_key = "larry_turbo_lora"
        filename = models.larry_turbo_ref_lora if reference else models.larry_turbo_lora
    elif variant == LIGHTX2V_8STEP_TURBO:
        model_key = "turbo_8step_ref_lora" if reference else "turbo_8step_lora"
        filename = models.turbo_8step_ref_lora if reference else models.turbo_8step_lora
    else:
        return False
    if not filename:
        raise H3Error(f"{variant} Turbo LoRA is not configured.")
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-turbo-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS[model_key].folder / filename
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand Turbo LoRA download did not produce {filename}.")
    return True


def ensure_int8_video_vae(models: ModelConfig, *, runtime: RuntimeConfig) -> bool:
    """Download the optional INT8 ConvRot video VAE on first use."""
    filename = models.video_vae_int8
    if not filename:
        raise H3Error(
            "The model configuration predates INT8 video VAE support. "
            "Re-run setup_h3.py before enabling it."
        )
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS["video_vae_int8"].folder / filename
    )
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=("video_vae_int8",),
    ):
        return False

    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-int8-vae-on-demand]",
        model_keys=("video_vae_int8",),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand INT8 VAE download did not produce {filename}.")
    return True


def ensure_trt_video_vae(
    models: ModelConfig, *, require_engine: bool = True, runtime: RuntimeConfig
) -> bool:
    """Provision the decoder ONNX source and require its local TensorRT engine."""
    trt_vae_decoder_paths(models, runtime=runtime)
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=TRT_VAE_RUNTIME_MODEL_KEYS,
    ):
        sync_models(
            root=runtime.comfy_dir / "models",
            manifest_path=manifest_path,
            token=resolve_hf_token(),
            log_prefix="[h3-trt-vae-on-demand]",
            model_keys=TRT_VAE_RUNTIME_MODEL_KEYS,
            download_workers=1,
        )

    if require_engine and not trt_vae_engine_is_current(models, runtime=runtime):
        _, engine_path, _ = trt_vae_decoder_paths(models, runtime=runtime)
        if not engine_path.is_file():
            detail = f" Missing: {engine_path.name}."
        else:
            detail = (
                " The existing engine uses an older quality profile or runtime version."
            )
        raise H3Error(
            "TensorRT VAE decoder is not compiled for the current profile. "
            "Click Compile TensorRT VAE engine once, then retry." + detail
        )
    return True


def _build_trt_video_vae_engine(
    models: ModelConfig,
    progress: Any,
    *,
    release_backend: Callable[[], None],
    runtime: RuntimeConfig,
) -> None:
    """Build the local TensorRT decoder engine without exposing partial output."""
    temporary_engine: Path | None = None
    temporary_marker: Path | None = None
    try:
        node_path = (
            runtime.comfy_dir
            / "custom_nodes"
            / "ComfyUI-H3VAE_TRT"
            / "minimax_trt_node.py"
        )
        if not node_path.is_file():
            raise H3Error(
                "ComfyUI-H3VAE_TRT is not installed. Run setup_h3.py and restart."
            )

        module = _load_trt_vae_compiler(node_path, runtime=runtime)
        if not module.HAS_TRT:
            raise H3Error(
                "TensorRT is not installed. Re-run setup_h3.py to install the "
                "CUDA 13 TensorRT builder/runtime, then restart the app."
            )

        decoder_onnx, engine_path, marker_path = trt_vae_decoder_paths(
            models, runtime=runtime
        )
        temporary_engine = engine_path.with_name(engine_path.name + ".h3-building")
        temporary_marker = marker_path.with_name(marker_path.name + ".h3-building")
        temporary_engine.unlink(missing_ok=True)
        temporary_marker.unlink(missing_ok=True)

        progress(0.1, desc="Preparing TensorRT VAE compilation")
        release_backend()
        check_cancelled()
        module.mm.unload_all_models()
        module.mm.soft_empty_cache()
        module.torch.cuda.empty_cache()
        module.MiniMaxH3TRTCompilerNode._build_engine(
            str(decoder_onnx),
            str(temporary_engine),
            is_decoder=True,
        )
        if not model_file_is_ready(temporary_engine):
            raise H3Error("TensorRT compiler did not produce a valid decoder engine.")

        progress(0.9, desc="Finalizing TensorRT VAE decoder")
        temporary_engine.replace(engine_path)
        temporary_engine = None
        temporary_marker.write_text(
            trt_vae_runtime_fingerprint(models, runtime=runtime) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(marker_path)
        temporary_marker = None

        progress(1.0, desc="TensorRT VAE decoder compiled")
    finally:
        if temporary_engine is not None:
            temporary_engine.unlink(missing_ok=True)
        if temporary_marker is not None:
            temporary_marker.unlink(missing_ok=True)


def ensure_trt_video_vae_engine(
    models: ModelConfig,
    *,
    force: bool = False,
    progress=no_progress,
    release_backend: Callable[[], None],
    runtime: RuntimeConfig,
) -> bool:
    """Provision and compile the TensorRT decoder engine only when required."""
    with JOBS.maintenance("trt-compile"), _TRT_VAE_COMPILE_LOCK:
        ensure_trt_video_vae(models, require_engine=False, runtime=runtime)
        if not force and trt_vae_engine_is_current(models, runtime=runtime):
            return False
        _build_trt_video_vae_engine(
            models, progress, release_backend=release_backend, runtime=runtime
        )
        ensure_trt_video_vae(models, runtime=runtime)
        return True


def ensure_single_frame_image_vae(
    models: ModelConfig, *, runtime: RuntimeConfig
) -> bool:
    """Download the optional 500K image-only decoder on first use."""
    filename = models.image_vae_500k
    if not filename:
        raise H3Error(
            "The model configuration predates single-frame image VAE support. "
            "Re-run setup_h3.py before selecting it."
        )
    model_key = "image_vae_500k"
    destination = (
        runtime.comfy_dir / "models" / MODEL_SPECS[model_key].folder / filename
    )
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-image-vae-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(
            f"On-demand single-frame image VAE download did not produce {filename}."
        )
    return True


def missing_ltx25_model_names(
    model_choice: str = DEFAULT_LTX25_MODEL, *, runtime: RuntimeConfig
) -> list[str]:
    stale = stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=runtime.models_config.parent / "h3_model_manifest.json",
        model_keys=ltx25_model_keys(model_choice).values(),
    )
    return [MODEL_SPECS[key].local_name for key in stale]


def ensure_ltx25_models(
    model_choice: str = DEFAULT_LTX25_MODEL, *, runtime: RuntimeConfig
) -> bool:
    """Download the gated LTX-2.5 model set only when its tab is used."""
    required_keys = tuple(ltx25_model_keys(model_choice).values())
    if not missing_ltx25_model_names(model_choice, runtime=runtime):
        return False
    token = resolve_hf_token()
    if token is None:
        raise H3Error(
            "No Hugging Face credential was found. On standalone, run "
            "`hf auth login` as the same user that launches run_h3.sh, or set "
            "HF_TOKEN. On Modal, attach a Secret containing HF_TOKEN to the "
            "serve function and redeploy."
        )
    try:
        sync_models(
            root=runtime.comfy_dir / "models",
            manifest_path=runtime.models_config.parent / "h3_model_manifest.json",
            token=token,
            log_prefix="[ltx25-on-demand]",
            model_keys=required_keys,
            download_workers=len(required_keys),
        )
    except Exception as exc:
        raise H3Error(
            "LTX-2.5 model download failed. Accept the Lightricks/LTX-2.5 "
            "Hugging Face license and authenticate with `hf auth login` or "
            "HF_TOKEN, then retry. "
            f"Details: {exc}"
        ) from exc
    for key in required_keys:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(
            runtime.comfy_dir / "models" / spec.folder / spec.local_name
        ):
            raise H3Error(
                f"On-demand LTX-2.5 download did not produce {spec.local_name}."
            )
    return True


def ensure_seedvr2_upscale_models(
    models: ModelConfig, model_choice: str, *, runtime: RuntimeConfig
) -> bool:
    configured = seedvr2_upscale_model_names(models, model_choice)
    selected_key = SEEDVR2_MODEL_CHOICES[str(model_choice)]
    assets = (
        (selected_key, configured["seedvr2_dit"]),
        ("seedvr2_vae", configured["seedvr2_vae"]),
    )
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    missing_files = stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(key for key, _filename in assets),
    )
    if not missing_files:
        return False

    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[seedvr2-on-demand]",
        model_keys=tuple(key for key, _filename in assets),
        download_workers=len(assets),
    )
    for key, filename in assets:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(
            runtime.comfy_dir / "models" / spec.folder / filename
        ):
            raise H3Error(f"On-demand SeedVR2 download did not produce {filename}.")
    return True


def ensure_ltx25_upscale_models(
    model_choice: str = DEFAULT_LTX25_MODEL, *, runtime: RuntimeConfig
) -> bool:
    """Lazily install the selected LTX base set and the 2x upscaler IC-LoRA."""
    base_downloaded = ensure_ltx25_models(model_choice, runtime=runtime)
    upscaler_key = "ltx25_pixel_upscaler_x2"
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=(upscaler_key,),
    ):
        return base_downloaded

    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[ltx25-upscale-on-demand]",
        model_keys=(upscaler_key,),
        download_workers=1,
    )
    spec = MODEL_SPECS[upscaler_key]
    if not model_file_is_ready(
        runtime.comfy_dir / "models" / spec.folder / spec.local_name
    ):
        raise H3Error(
            f"On-demand LTX-2.5 upscaler download did not produce {spec.local_name}."
        )
    return True


def missing_music3_model_names(
    model_choice: str, *, runtime: RuntimeConfig
) -> list[str]:
    missing = stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=runtime.models_config.parent / "h3_model_manifest.json",
        model_keys=music3_model_keys(model_choice),
    )
    return [MODEL_SPECS[key].local_name for key in missing]


def ensure_music3_models(model_choice: str, *, runtime: RuntimeConfig) -> bool:
    """Lazily install the selected Music 3 DiT and its shared model files."""
    keys = music3_model_keys(model_choice)
    manifest_path = runtime.models_config.parent / "h3_model_manifest.json"
    missing = stale_model_keys(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        model_keys=keys,
    )
    if not missing:
        return False
    sync_models(
        root=runtime.comfy_dir / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[music3-on-demand]",
        model_keys=keys,
        download_workers=len(keys),
    )
    return True
