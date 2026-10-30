"""On-demand model downloads and the TensorRT VAE engine lifecycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping

from h3_app.catalog import (
    LARRY_TURBO,
    STANDARD_TURBO_LORA_SETTINGS,
    LIGHTX2V_8STEP_TURBO,
    LIGHTX2V_4STEP_TURBO,
    LTX25_POSTPROCESS_MODELS,
    LTX25_SDR_TO_HDR,
    LTX25_UPSCALE,
    H3_REFINEMENT,
    H3_REGENERATE_MODELS,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.jobs import JOBS, JobCancelled, check_cancelled
from h3_app.model_types import (
    ModelConfig,
    ModelProfile,
    h3_text_encoder_settings,
    ltx25_model_keys,
    model_file_is_ready,
    music3_model_keys,
    yue2_model_keys,
    seedvr2_upscale_model_names,
    trt_vae_engine_name,
)
from h3_app.policy import h3_latent_upscaler_settings, normalize_turbo_variant
from h3_models import (
    CQ_IMAGE_MODEL_KEYS,
    DEFAULT_LTX25_MODEL,
    MODEL_SPECS,
    PROFILE_LABELS,
    PROFILE_MODEL_KEYS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
    QWEN_IMAGE21_TURBO_MODES,
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
            if key.lower() != "fasth3_8step_v2"
            and str(value.get("label", "")).strip().lower() != "fasth3 8-step v2"
        }
        default_profile = str(data.get("default_profile", "speed")).lower()

    # Older generated configs predate newly added lazy profiles. Merge only
    # missing catalog entries in memory so upgrades can select and provision
    # them immediately without replacing any user-configured filenames.
    for profile_key, (fl2va_key, ref2va_key) in PROFILE_MODEL_KEYS.items():
        if profile_key in profiles:
            continue
        fl2va = MODEL_SPECS[fl2va_key]
        ref2va = MODEL_SPECS[ref2va_key]
        profiles[profile_key] = ModelProfile(
            label=PROFILE_LABELS[profile_key],
            fl2va=fl2va.local_name,
            ref2va=ref2va.local_name,
            fl2va_source=fl2va.source,
            ref2va_source=ref2va.source,
        )

    if default_profile not in profiles:
        default_profile = "singularity"

    dmad_lora = data.get("dmad_4step_lora", MODEL_SPECS["dmad_4step_lora"].local_name)
    # Migrate the previously generated filename so on-demand provisioning and
    # the workflow both use the official adapter. Preserve custom filenames.
    if dmad_lora == "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors":
        dmad_lora = MODEL_SPECS["dmad_4step_lora"].local_name

    return ModelConfig(
        profiles=profiles,
        default_profile=default_profile,
        text_encoder=data["text_encoder"],
        text_encoders=data.get("text_encoders"),
        video_vae=data["video_vae"],
        audio_vae=data["audio_vae"],
        video_vae_sol_int8=data.get("video_vae_sol_int8", MODEL_SPECS["video_vae_sol_int8"].local_name),
        video_vae_int8=data.get("video_vae_int8"),
        video_vae_lynnreal_int8=data.get(
            "video_vae_lynnreal_int8", MODEL_SPECS["video_vae_lynnreal_int8"].local_name
        ),
        video_vae_lynnreal_int8_source=data.get(
            "video_vae_lynnreal_int8_source", MODEL_SPECS["video_vae_lynnreal_int8"].source
        ),
        video_vae_trt_encoder=data.get("video_vae_trt_encoder"),
        video_vae_trt_decoder=data.get("video_vae_trt_decoder"),
        video_vae_trt_source=data.get("video_vae_trt_source", "unknown"),
        video_vae_int8_source=data.get("video_vae_int8_source", "unknown"),
        image_vae_500k=data.get("image_vae_500k"),
        image_vae_500k_source=data.get("image_vae_500k_source", "unknown"),
        pdmd_4step_lora=data.get(
            "pdmd_4step_lora", MODEL_SPECS["pdmd_4step_lora"].local_name
        ),
        dmad_4step_lora=dmad_lora,
        pika_flex_fl2va_lora=data.get("pika_flex_fl2va_lora", MODEL_SPECS["pika_flex_fl2va_lora"].local_name),
        pika_flex_ref2va_lora=data.get("pika_flex_ref2va_lora", MODEL_SPECS["pika_flex_ref2va_lora"].local_name),
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


def _models_root(runtime: RuntimeConfig) -> Path:
    return runtime.comfy_dir / "models"


def _manifest_path(runtime: RuntimeConfig) -> Path:
    return runtime.models_config.parent / "h3_model_manifest.json"


def spec_files(keys: Iterable[str]) -> dict[str, str]:
    """Each key's file name from the model inventory."""
    return {key: MODEL_SPECS[key].local_name for key in keys}


def missing_models(keys: Iterable[str], *, runtime: RuntimeConfig) -> list[str]:
    """The keys among ``keys`` whose files are absent or stale."""
    return stale_model_keys(
        root=_models_root(runtime), manifest_path=_manifest_path(runtime), model_keys=tuple(keys),
    )


def missing_model_names(keys: Iterable[str], *, runtime: RuntimeConfig) -> list[str]:
    return [MODEL_SPECS[key].local_name for key in missing_models(keys, runtime=runtime)]


def _verify_files(files: Mapping[str, str], failure: str, *, runtime: RuntimeConfig) -> None:
    for key, filename in files.items():
        if not model_file_is_ready(_models_root(runtime) / MODEL_SPECS[key].folder / filename):
            raise H3Error(failure.format(name=filename))


def ensure_models(
    keys: Iterable[str],
    *,
    runtime: RuntimeConfig,
    log_prefix: str = "[engine-on-demand]",
    download_workers: int = 1,
    only_stale: bool = False,
    files: Mapping[str, str] | None = None,
    verify_always: bool = False,
    failure: str = "On-demand download did not produce {name}.",
) -> bool:
    """Download ``keys`` through the shared manifest when any is stale; True when a download ran.

    ``only_stale`` fetches just the stale keys. ``files`` maps keys to the file
    names to check afterwards (``failure`` names the missing one); with
    ``verify_always`` they are checked even when nothing was downloaded.
    """
    keys = tuple(keys)
    stale = missing_models(keys, runtime=runtime)
    if stale:
        sync_models(
            root=_models_root(runtime),
            manifest_path=_manifest_path(runtime),
            token=resolve_hf_token(),
            log_prefix=log_prefix,
            model_keys=tuple(stale) if only_stale else keys,
            download_workers=download_workers,
        )
    if files and (stale or verify_always):
        _verify_files(files, failure, runtime=runtime)
    return bool(stale)


def ensure_h3_text_encoder(
    models: ModelConfig, model_choice: str, *, runtime: RuntimeConfig
) -> tuple[str, bool]:
    """Download an optional H3 text encoder on first use."""
    model_key, filename, requires_offload = h3_text_encoder_settings(
        models, model_choice
    )
    ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-text-encoder-on-demand]",
        files={model_key: filename}, verify_always=True,
        failure="On-demand H3 text encoder download did not produce {name}.",
    )
    return filename, requires_offload


def ensure_h3_veda_predictor(mode: str, *, runtime: RuntimeConfig) -> None:
    """Download only the predictor for the selected H3 task, on first use."""
    key = "veda_r2va" if mode == "Reference media" else "veda_t2va"
    ensure_models(
        (key,), runtime=runtime, log_prefix="[h3-veda-on-demand]",
        files=spec_files((key,)), verify_always=True,
        failure="Veda predictor download did not produce a valid model file.",
    )


def ensure_h3_semantic_bridge(*, runtime: RuntimeConfig) -> None:
    """Fetch only the optional v1 adapter; never preload it with the base models."""
    key = "semantic_bridge_v1"
    ensure_models(
        (key,), runtime=runtime, log_prefix="[h3-semantic-bridge-on-demand]",
        files=spec_files((key,)), verify_always=True,
        failure="Semantic Bridge adapter download did not produce a valid model file.",
    )


def ensure_h3_latent_upscaler_model(
    model_choice: str, *, runtime: RuntimeConfig
) -> bool:
    """Download the selected native H3 latent upscaler on first use."""
    model_key, filename, _precision = h3_latent_upscaler_settings(model_choice)
    return ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-latent-upscaler-on-demand]",
        files={model_key: filename},
        failure="On-demand H3 latent upscaler download did not produce {name}.",
    )


def ensure_profile_model(
    profile_key: str, profile: ModelProfile, mode: str, *, runtime: RuntimeConfig
) -> bool:
    """Download a lazy profile checkpoint before submitting its workflow."""
    reference = str(mode).strip().lower() == "reference media"
    filename = profile.ref2va if reference else profile.fl2va
    model_key = PROFILE_MODEL_KEYS[profile_key][1 if reference else 0]
    return ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-on-demand]",
        files={model_key: filename},
        failure="On-demand model download did not produce {name}.",
    )


def ensure_turbo_lora(
    models: ModelConfig, turbo_variant: str, mode: str, *, runtime: RuntimeConfig
) -> bool:
    """Download a non-default Turbo LoRA only when its variant is selected."""
    variant = (
        turbo_variant if turbo_variant in STANDARD_TURBO_LORA_SETTINGS
        else normalize_turbo_variant(turbo_variant)
    )
    reference = str(mode).strip().lower() == "reference media"
    if variant in STANDARD_TURBO_LORA_SETTINGS:
        spec = STANDARD_TURBO_LORA_SETTINGS[variant]
        model_key = spec.ref_lora_attr if reference else spec.lora_attr
        filename = getattr(models, model_key)
    elif variant == LARRY_TURBO:
        model_key = "larry_turbo_lora"
        filename = models.larry_turbo_ref_lora if reference else models.larry_turbo_lora
    elif variant == LIGHTX2V_8STEP_TURBO:
        model_key = "turbo_8step_ref_lora" if reference else "turbo_8step_lora"
        filename = models.turbo_8step_ref_lora if reference else models.turbo_8step_lora
    elif variant == LIGHTX2V_4STEP_TURBO:
        model_key = "turbo_ref_lora" if reference else "turbo_lora"
        filename = models.turbo_ref_lora if reference else models.turbo_lora
    else:
        return False
    if not filename:
        raise H3Error(f"{variant} Turbo LoRA is not configured.")
    return ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-turbo-on-demand]",
        files={model_key: filename},
        failure="On-demand Turbo LoRA download did not produce {name}.",
    )


def ensure_lightvae(video_encoder: str, decoder: str, *, runtime: RuntimeConfig) -> bool:
    """Provision only selected LightVAE components through the shared manifest."""
    from h3_app.resources import resolve_decoders

    selected = resolve_decoders(
        "Video", "", use_trt_vae=False, use_int8_vae=False,
        video_encoder=video_encoder, lightvae_decoder=decoder,
    )
    keys = selected.optional_model_keys
    if not keys:
        return False
    return ensure_models(
        keys, runtime=runtime, log_prefix="[h3-lightvae-on-demand]", only_stale=True,
        files=spec_files(keys), failure="LightVAE download did not produce {name}.",
    )


def ensure_int8_video_vae(
    models: ModelConfig, *, runtime: RuntimeConfig, lynnreal: bool = False, sol: bool = False
) -> bool:
    """Download the selected standard, Sol or LynnReal INT8 VAE on first use."""
    if sol and lynnreal:
        raise H3Error("Select only one video decoder.")
    model_key = "video_vae_sol_int8" if sol else "video_vae_lynnreal_int8" if lynnreal else "video_vae_int8"
    filename = getattr(models, model_key)
    if not filename:
        raise H3Error(
            "The model configuration predates INT8 video VAE support. "
            "Re-run setup_h3.py before enabling it."
        )
    return ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-int8-vae-on-demand]",
        files={model_key: filename},
        failure="On-demand INT8 VAE download did not produce {name}.",
    )


def _ensure_shared_vae(models: ModelConfig, key: str, what: str, log_prefix: str, *,
                       runtime: RuntimeConfig) -> bool:
    filename = getattr(models, key)
    if model_file_is_ready(_models_root(runtime) / MODEL_SPECS[key].folder / filename):
        return False
    return ensure_models(
        (key,), runtime=runtime, log_prefix=log_prefix, files={key: filename},
        failure=f"On-demand {what} download did not produce {{name}}.",
    )


def ensure_base_video_vae(models: ModelConfig, *, runtime: RuntimeConfig) -> bool:
    """Download the base video VAE on first use."""
    return _ensure_shared_vae(models, "video_vae", "video VAE", "[h3-video-vae-on-demand]",
                              runtime=runtime)


def ensure_audio_vae(models: ModelConfig, *, runtime: RuntimeConfig) -> bool:
    """Download the audio VAE on first use."""
    return _ensure_shared_vae(models, "audio_vae", "audio VAE", "[h3-audio-vae-on-demand]",
                              runtime=runtime)


def ensure_trt_video_vae(
    models: ModelConfig, *, require_engine: bool = True, runtime: RuntimeConfig
) -> bool:
    """Provision the decoder ONNX source and require its local TensorRT engine."""
    trt_vae_decoder_paths(models, runtime=runtime)
    ensure_models(TRT_VAE_RUNTIME_MODEL_KEYS, runtime=runtime, log_prefix="[h3-trt-vae-on-demand]")

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
    return ensure_models(
        (model_key,), runtime=runtime, log_prefix="[h3-image-vae-on-demand]",
        files={model_key: filename},
        failure="On-demand single-frame image VAE download did not produce {name}.",
    )


def missing_ltx25_model_names(
    model_choice: str = DEFAULT_LTX25_MODEL, *, runtime: RuntimeConfig
) -> list[str]:
    return missing_model_names(ltx25_model_keys(model_choice).values(), runtime=runtime)


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
            root=_models_root(runtime),
            manifest_path=_manifest_path(runtime),
            token=token,
            log_prefix="[ltx25-on-demand]",
            model_keys=required_keys,
            download_workers=len(required_keys),
        )
    except Exception as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise H3Error(
                "LTX-2.5 download failed because a Hugging Face cache path "
                "disappeared during download. Retry; if it persists, check for "
                "another process pruning the shared Hugging Face cache. "
                f"Details: {exc}"
            ) from exc
        raise H3Error(
            "LTX-2.5 model download failed. If the details show HTTP 401 or "
            "403, check the model license and Hugging Face credentials; "
            "otherwise check the cache, disk space, and network before retrying. "
            f"Details: {exc}"
        ) from exc
    _verify_files(spec_files(required_keys),
                  "On-demand LTX-2.5 download did not produce {name}.", runtime=runtime)
    return True


def ensure_seedvr2_upscale_models(
    models: ModelConfig, model_choice: str, *, runtime: RuntimeConfig
) -> bool:
    configured = seedvr2_upscale_model_names(models, model_choice)
    files = {
        SEEDVR2_MODEL_CHOICES[str(model_choice)]: configured["seedvr2_dit"],
        "seedvr2_vae": configured["seedvr2_vae"],
    }
    return ensure_models(
        files, runtime=runtime, log_prefix="[seedvr2-on-demand]", download_workers=len(files),
        files=files, failure="On-demand SeedVR2 download did not produce {name}.",
    )


def ensure_cq_image_enhance_models(*, runtime: RuntimeConfig) -> bool:
    """Download only the assets used by the official CQ image recipe."""
    return ensure_models(
        CQ_IMAGE_MODEL_KEYS, runtime=runtime, log_prefix="[cq-image-on-demand]",
        download_workers=2, files=spec_files(CQ_IMAGE_MODEL_KEYS),
        failure="On-demand CQ image download did not produce {name}.",
    )


def ensure_h3_regenerate_models(models: ModelConfig, option: str, *, runtime: RuntimeConfig, refinement=None) -> bool:
    """Provision the selected H3 profile, upscaler and refinement adapters on demand."""
    config = None
    if option == H3_REFINEMENT:
        from h3_app.refinement import resolve_refinement
        config = resolve_refinement(refinement, models)
        keys = (config["upscaler_key"],)
        if config["variant"] in H3_REGENERATE_MODELS:
            keys += ("turbo_lora", H3_REGENERATE_MODELS[config["variant"]])
    elif option in H3_REGENERATE_MODELS:
        keys = ("h3_latent_upscaler_3d_bf16", "turbo_lora", H3_REGENERATE_MODELS[option])
    else:
        raise H3Error(f"Unknown H3 Regenerate option: {option}")
    files = spec_files(keys)
    # Reject missing custom encoders before downloading a potentially large base model.
    encoder_key = next((key for key, spec in MODEL_SPECS.items()
                        if spec.folder == "text_encoders" and spec.local_name == models.text_encoder), None)
    if encoder_key:
        keys += (encoder_key,)
        files[encoder_key] = models.text_encoder
    elif not model_file_is_ready(_models_root(runtime) / "text_encoders" / models.text_encoder):
        raise H3Error(f"Configured H3 text encoder is missing: {models.text_encoder}")
    profile_key = models.profile_key(config["settings"]["model_profile"] if config else models.default_profile)
    downloaded = [
        ensure_profile_model(profile_key, models.profile(profile_key), "Text to video", runtime=runtime),
        ensure_base_video_vae(models, runtime=runtime),
        ensure_audio_vae(models, runtime=runtime),
        ensure_models(
            keys,
            runtime=runtime, log_prefix="[h3-regenerate-on-demand]",
            files=files,
        ),
    ]
    if config and config["lora"]:
        downloaded.append(ensure_turbo_lora(models, config["variant"], "Text to video", runtime=runtime))
    return any(downloaded)


def ensure_ltx25_upscale_models(
    model_choice: str = DEFAULT_LTX25_MODEL,
    *,
    runtime: RuntimeConfig,
    option: str = LTX25_UPSCALE,
) -> bool:
    """Lazily install the selected LTX base set and post-processing IC-LoRA."""
    if option not in LTX25_POSTPROCESS_MODELS:
        raise H3Error(f"Unknown LTX-2.5 post-processing method: {option}")
    if option == LTX25_SDR_TO_HDR:
        # HDR uses fixed conditioning and the full float32-capable DiffVAE;
        # no Gemma text encoder or audio generator is needed.
        keys = (
            ltx25_model_keys(model_choice)["distilled"],
            "ltx25_video_vae_full",
            "ltx25_sdr_to_hdr",
            "ltx25_sdr_to_hdr_conditioning",
        )
        try:
            return ensure_models(
                keys, runtime=runtime, log_prefix="[ltx25-hdr-on-demand]", download_workers=2,
                files=spec_files(keys), failure="On-demand HDR download did not produce {name}.",
            )
        except (JobCancelled, H3Error):
            raise
        except Exception as exc:
            raise H3Error(
                "SDR-to-HDR model download failed. For HTTP 401/403, accept access "
                "to Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR and Lightricks/LTX-2.5 "
                "on Hugging Face, then authenticate with hf auth login or HF_TOKEN. "
                f"Details: {exc}"
            ) from exc
    upscaler_key = LTX25_POSTPROCESS_MODELS[option]
    base_downloaded = ensure_ltx25_models(model_choice, runtime=runtime)
    return ensure_models(
        (upscaler_key,), runtime=runtime, log_prefix="[ltx25-postprocess-on-demand]",
        files=spec_files((upscaler_key,)),
        failure="On-demand LTX-2.5 post-processing download did not produce {name}.",
    ) or base_downloaded


def ensure_ltx25_ingredients_model(*, runtime: RuntimeConfig) -> bool:
    """Install the Ingredients adapter after the selected base model is ready."""
    key = "ltx25_iclora_ingredients"
    return ensure_models(
        (key,), runtime=runtime, log_prefix="[ltx25-ingredients-on-demand]",
        files=spec_files((key,)), failure="Ingredients download did not produce {name}.",
    )


def missing_music3_model_names(
    model_choice: str, *, runtime: RuntimeConfig
) -> list[str]:
    return missing_model_names(music3_model_keys(model_choice), runtime=runtime)


def ensure_music3_models(model_choice: str, *, runtime: RuntimeConfig) -> bool:
    """Lazily install the selected Music 3 DiT and its shared model files."""
    keys = music3_model_keys(model_choice)
    return ensure_models(keys, runtime=runtime, log_prefix="[music3-on-demand]",
                         download_workers=len(keys))


def qwen_image21_model_keys(
    model_choice: str, text_encoder_choice: str, turbo_variant: str = "Off"
) -> tuple[str, ...]:
    try:
        model_key = QWEN_IMAGE21_MODEL_CHOICES[str(model_choice)]
        encoder_key = QWEN_IMAGE21_TEXT_ENCODER_CHOICES[str(text_encoder_choice)]
    except KeyError as exc:
        raise H3Error(f"Unknown Qwen Image 2.1 model choice: {exc.args[0]}") from exc
    keys = (model_key, encoder_key, "qwen_image21_vae")
    if turbo_variant in QWEN_IMAGE21_TURBO_MODES:
        return (*keys, QWEN_IMAGE21_TURBO_MODES[turbo_variant][0])
    if turbo_variant != "Off":
        raise H3Error(f"Unknown Qwen Image 2.1 Turbo variant: {turbo_variant}")
    return keys


def missing_qwen_image21_model_names(
    model_choice: str,
    text_encoder_choice: str,
    turbo_variant: str = "Off",
    *,
    runtime: RuntimeConfig,
) -> list[str]:
    return missing_model_names(
        qwen_image21_model_keys(model_choice, text_encoder_choice, turbo_variant), runtime=runtime
    )


def ensure_qwen_image21_models(
    model_choice: str,
    text_encoder_choice: str,
    turbo_variant: str = "Off",
    *,
    runtime: RuntimeConfig,
) -> bool:
    """Lazily install the selected Qwen Image 2.1 DiT, encoder, and VAE."""
    keys = qwen_image21_model_keys(model_choice, text_encoder_choice, turbo_variant)
    return ensure_models(keys, runtime=runtime, log_prefix="[qwen-image21-on-demand]",
                         download_workers=len(keys))


def ensure_model_keys(keys: tuple[str, ...], *, runtime: RuntimeConfig) -> bool:
    """Provision selected optional engine assets using the shared manifest lock."""
    return ensure_models(keys, runtime=runtime)


def missing_yue2_model_names(model_choice: str, *, runtime: RuntimeConfig) -> list[str]:
    return missing_model_names(yue2_model_keys(model_choice), runtime=runtime)


def ensure_yue2_models(model_choice: str, *, runtime: RuntimeConfig) -> bool:
    """Lazily install the selected YuE2 checkpoint from the official Comfy-Org repo."""
    return ensure_models(yue2_model_keys(model_choice), runtime=runtime, log_prefix="[yue2-on-demand]")
