"""Validate and provision the effective H3 request before graph construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Generator

from h3_app.catalog import (
    COMFY_UPSCALE_OPTIONS,
    DEFAULT_SLA_PRESET,
    GENERATION_POSTPROCESS_OPTIONS,
    H3_LATENT_UPSCALE_STANDARD,
    H3_SEMANTIC_BRIDGE_NODE,
    LARRY_TURBO,
    LTX25_UPSCALE,
    SEEDVR2_UPSCALE,
    SINGLE_FRAME_IMAGE_VAE,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.model_types import (
    ModelConfig,
    ltx25_model_keys,
    seedvr2_upscale_model_names,
)
from h3_app.policy import (
    H3SplitUpscaleConfig,
    active_fl2va_voice_references,
    collect_reference_slots,
    frame_length,
    h3_latent_upscale_dimensions,
    h3_latent_upscaler_settings,
    normalize_image_vae,
    normalize_result_format,
    normalize_turbo_variant,
    resolve_cache_policy,
    resolve_h3_latent_upscale_method,
    resolve_h3_split_upscale_config,
    resolve_sla_preset,
    selected_image_sampling_length,
    turbo_strength_for,
    validate_image_frame_count,
)
from h3_app.progress import ProgressCallback
from h3_app.settings import ResolvedSettings
from h3_app.status import progress_status
from h3_app.workflows.h3 import required_nodes_for
from h3_app.workflows.upscale import required_upscale_nodes
from h3_models import MODEL_SPECS

from .requests import H3Request
from .results import GenerationUpdate
from .services import GenerationServices


@dataclass(frozen=True)
class PreparedH3:
    actual_seed: int
    available: set[str]
    cache_note: str | None
    effective_cache_mode: str
    effective_sage: bool
    effective_scheduler: str
    effective_sla: bool
    effective_sla_inputs: dict[str, Any]
    effective_sla_preset: str
    effective_sol: bool
    effective_stage_offload: bool
    effective_steps: int
    generation_frames: int
    generation_note: str | None
    latent_source_height: int
    latent_source_width: int
    latent_split_config: H3SplitUpscaleConfig | None
    latent_upscale_model_name: str | None
    latent_upscale_precision: str
    models: ModelConfig
    packed_tokens: int
    plan: ResolvedSettings
    refs_a: list[str]
    refs_i: list[str]
    refs_v: list[str]
    requested_image_frames: int
    resolved_height: int
    resolved_latent_upscale_method: str
    resolved_width: int
    selected_image_vae: str
    selected_label: str
    selected_model: str
    selected_text_encoder: str
    selected_turbo: str
    smart_stage_offload: bool
    sol_reason: str
    turbo_lora_name: str | None
    turbo_strength: float
    voice_refs: list[str]


def prepare_h3(
    request: H3Request,
    services: GenerationServices,
    runtime: RuntimeConfig,
    requested_values: dict[str, Any],
    started: float,
    progress: ProgressCallback,
) -> Generator[GenerationUpdate, None, PreparedH3]:
    voice_refs = active_fl2va_voice_references(
        request.media.mode,
        request.media.fl2va_audio_1,
        request.media.fl2va_audio_2,
        request.media.fl2va_audio_3,
    )
    plan = services.policy.resolve_request_settings(requested_values)
    if plan.issues:
        raise H3Error(" ".join(plan.issues))
    effective = plan.effective
    request.output.use_trt_vae = effective.use_trt_vae
    request.output.use_int8_vae = effective.use_int8_vae
    request.sampling.semantic_bridge = (
        effective.semantic_bridge and effective.semantic_bridge_alpha != 0
    )
    request.sampling.semantic_bridge_alpha = effective.semantic_bridge_alpha
    request.output.result_format = normalize_result_format(
        effective.output.result_format
    )
    selected_image_vae = normalize_image_vae(request.output.image_vae)
    requested_image_frames = validate_image_frame_count(effective.output.image_frames)
    if request.finishing.postprocess not in GENERATION_POSTPROCESS_OPTIONS:
        raise H3Error("Unsupported post-processing method.")
    if not request.media.prompt.strip():
        raise H3Error("Prompt is required.")
    request.finishing.postprocess = effective.finishing.postprocess
    request.finishing.latent_upscale = effective.finishing.latent_upscale
    generation_frames = (
        selected_image_sampling_length(
            requested_image_frames,
            selected_image_vae,
        )
        if request.output.result_format == "Image"
        else frame_length(request.output.duration)
    )
    policy_duration = generation_frames / 24.0
    resolved_width, resolved_height = effective.output.width, effective.output.height
    actual_seed = (
        random.randrange(0, 2**63 - 1)
        if int(request.output.seed) < 0
        else int(request.output.seed)
    )
    models = services.models.load_model_config()
    text_encoder_key, selected_text_encoder, bf16_text_encoder = (
        services.models.h3_text_encoder_settings(models, request.sampling.text_encoder)
    )
    effective_stage_offload = effective.sampling.stage_model_offload
    smart_stage_offload = bf16_text_encoder and bool(
        request.media.reuse_unchanged_inputs
    )
    if effective_stage_offload and runtime.memory_profile == "gpu-only":
        raise H3Error(
            "H3 stage offload cannot release VRAM while ComfyUI is running "
            "with --gpu-only because every model's offload device is CUDA. "
            "Update and restart run_h3.sh/Modal so ComfyUI uses Dynamic VRAM."
        )
    text_encoder_path = (
        runtime.comfy_dir
        / "models"
        / MODEL_SPECS[text_encoder_key].folder
        / selected_text_encoder
    )
    if not services.models.model_file_is_ready(text_encoder_path):
        progress(0, desc=f"Downloading {request.sampling.text_encoder} text encoder")
        yield GenerationUpdate(
            None,
            progress_status(
                f"Downloading the {request.sampling.text_encoder} H3 text encoder on demand",
                started=started,
                detail=(
                    "The BF16 checkpoint is approximately 51.5 GB."
                    if bf16_text_encoder
                    else None
                ),
            ),
        )
    selected_text_encoder, bf16_text_encoder = services.models.ensure_h3_text_encoder(
        models, request.sampling.text_encoder
    )
    if request.output.use_int8_vae:
        progress(0, desc="Preparing INT8 video VAE")
        services.models.ensure_int8_video_vae(models)
    if request.output.use_trt_vae:
        progress(0, desc="Preparing TensorRT video VAE")
        services.models.ensure_trt_video_vae_engine(models, progress=progress)
    if (
        request.output.result_format == "Image"
        and selected_image_vae == SINGLE_FRAME_IMAGE_VAE
    ):
        decoder_ready = models.image_vae_500k and services.models.model_file_is_ready(
            runtime.comfy_dir / "models" / "vae" / models.image_vae_500k
        )
        if not decoder_ready:
            progress(0, desc="Downloading single-frame image VAE")
            yield GenerationUpdate(
                None,
                progress_status(
                    "Downloading the 500K single-frame image VAE on demand",
                    started=started,
                ),
            )
        services.models.ensure_single_frame_image_vae(models)
    if request.finishing.postprocess == SEEDVR2_UPSCALE:
        seedvr2_upscale_model_names(models, request.finishing.seedvr2_model)
    elif request.finishing.postprocess == LTX25_UPSCALE:
        ltx25_model_keys(request.finishing.ltx25_model)
    profile_key = models.profile_key(request.sampling.model_profile)
    profile = models.profiles[profile_key]

    selected_model = (
        profile.ref2va if request.media.mode == "Reference media" else profile.fl2va
    )
    selected_path = runtime.comfy_dir / "models" / "diffusion_models" / selected_model
    if not services.models.model_file_is_ready(selected_path):
        progress(0, desc=f"Downloading {profile.label} model")
        yield GenerationUpdate(
            None,
            progress_status(
                f"Downloading {profile.label} model on demand",
                started=started,
            ),
        )
    services.models.ensure_profile_model(profile_key, profile, request.media.mode)

    requested_generation = str(request.sampling.generation_mode).strip().lower()
    use_turbo = requested_generation == "turbo"
    selected_turbo = normalize_turbo_variant(request.sampling.turbo_variant)
    generation_note = (
        f"Reference Turbo is experimental and currently uses the "
        f"FL2VA-trained {selected_turbo} LoRA."
        if (
            request.media.mode == "Reference media"
            and use_turbo
            and selected_turbo == LARRY_TURBO
        )
        else None
    )

    if use_turbo:
        turbo_lora_name = models.turbo_lora_for(request.media.mode, selected_turbo)
        if not turbo_lora_name:
            raise H3Error(
                f"{selected_turbo} Turbo LoRA is not provisioned. "
                "Re-run setup/provisioning."
            )
        if services.models.ensure_turbo_lora(
            models, selected_turbo, request.media.mode
        ):
            progress(0, desc=f"Downloaded {selected_turbo} Turbo LoRA")
        selected_label = f"{profile.label} · Turbo · {selected_turbo}"
        turbo_strength = turbo_strength_for(selected_turbo)
    else:
        selected_label = f"{profile.label} · Normal"
        turbo_lora_name = None
        turbo_strength = 1.0
    selected_label += (
        " · INT8 ConvRot VAE" if request.output.use_int8_vae else " · FP16 VAE"
    )
    selected_label += (
        f" · text encoder {request.sampling.text_encoder} · stage offload "
        f"{'on' if effective_stage_offload else 'off'}"
    )
    if request.output.result_format == "Image":
        selected_label += f" · image decoder {selected_image_vae}"

    # Variant defaults update outside the generation queue, while this
    # request deliberately honors any subsequent manual step adjustment.
    effective_steps = int(request.sampling.steps)
    effective_scheduler = str(request.sampling.scheduler)

    if use_turbo:
        if effective_steps < 4:
            raise H3Error("Turbo requires at least 4 steps.")
    elif effective_steps < 10:
        raise H3Error(
            "Normal H3 generation requires at least 10 steps. "
            "Use Generation=Turbo for lower-step generation."
        )

    latent_upscale_model_name: str | None = None
    latent_upscale_precision = "bf16"
    resolved_latent_upscale_method = H3_LATENT_UPSCALE_STANDARD
    latent_split_config: H3SplitUpscaleConfig | None = None
    latent_source_width, latent_source_height = resolved_width, resolved_height
    if request.finishing.latent_upscale:
        resolved_latent_upscale_method = resolve_h3_latent_upscale_method(
            request.finishing.latent_upscale_method
        )
        latent_split_config = resolve_h3_split_upscale_config(
            resolved_latent_upscale_method,
            tile_width=request.finishing.latent_split_tile_width,
            tile_height=request.finishing.latent_split_tile_height,
            overlap_ratio=request.finishing.latent_split_overlap_ratio,
            fade_ratio=request.finishing.latent_split_fade_ratio,
            chunk_frames=request.finishing.latent_split_chunk_frames,
            temporal_overlap_frames=request.finishing.latent_split_temporal_overlap_frames,
            seam_denoise=request.finishing.latent_split_seam_denoise,
            seam_polish=request.finishing.latent_split_seam_polish,
        )
        (
            latent_source_width,
            latent_source_height,
            resolved_width,
            resolved_height,
        ) = h3_latent_upscale_dimensions(resolved_width, resolved_height)
        refine_steps = int(request.finishing.latent_upscale_refine_steps)
        if refine_steps < 1 or refine_steps >= effective_steps:
            raise H3Error(
                "H3 latent upscale refinement steps must be at least 1 and smaller "
                f"than the total {effective_steps} sampling steps."
            )
        (
            _latent_upscale_key,
            latent_upscale_model_name,
            latent_upscale_precision,
        ) = h3_latent_upscaler_settings(request.finishing.latent_upscaler_model)
        destination = (
            runtime.comfy_dir
            / "models"
            / "latent_upscale_models"
            / latent_upscale_model_name
        )
        if not services.models.model_file_is_ready(destination):
            progress(0, desc="Downloading H3 latent upscaler")
            yield GenerationUpdate(
                None,
                progress_status(
                    f"Downloading {request.finishing.latent_upscaler_model} H3 latent upscaler",
                    started=started,
                ),
            )
        services.models.ensure_h3_latent_upscaler_model(
            request.finishing.latent_upscaler_model
        )

    info = services.execution.object_info()
    available = set(info)
    if request.sampling.semantic_bridge:
        if H3_SEMANTIC_BRIDGE_NODE not in available:
            raise H3Error(
                "Missing H3SemanticBridge node. Update provisioning and restart ComfyUI."
            )
        progress(0, desc="Preparing Semantic Bridge v1")
        yield GenerationUpdate(
            None,
            progress_status(
                "Preparing Semantic Bridge v1 (download on first use)", started=started
            ),
        )
        services.models.ensure_h3_semantic_bridge()
    if request.finishing.postprocess in COMFY_UPSCALE_OPTIONS:
        missing_upscale_nodes = (
            required_upscale_nodes(request.finishing.postprocess) - available
        )
        if missing_upscale_nodes:
            raise H3Error(
                f"{request.finishing.postprocess} requires current ComfyUI nodes: "
                + ", ".join(sorted(missing_upscale_nodes))
            )

    effective_sol, packed_tokens, sol_reason = services.policy.resolve_sol_policy(
        request.sampling.attention_mode,
        request.media.mode,
        resolved_width,
        resolved_height,
        policy_duration,
        request.media.first_image,
        request.media.last_image,
        use_turbo=use_turbo,
    )
    effective_sage = str(request.sampling.attention_mode).strip().lower() in {
        "sage",
        "sage 2",
        "sage2",
    }
    effective_sla = str(request.sampling.attention_mode).strip().lower() in {
        "sla",
        "sla attention",
        "sparse-linear",
    }
    if effective_sla:
        effective_sla_preset, effective_sla_inputs = resolve_sla_preset(
            request.sampling.sla_preset
        )
    else:
        effective_sla_preset, effective_sla_inputs = resolve_sla_preset(
            DEFAULT_SLA_PRESET
        )
    effective_cache_mode, cache_note = resolve_cache_policy(
        effective.cache_mode, use_turbo=use_turbo
    )
    if request.finishing.latent_upscale and effective_cache_mode.lower() != "off":
        cache_note = (
            f"{effective_cache_mode} was disabled because cache state is not "
            "safe across the low-resolution generation and high-resolution "
            "refinement samplers."
        )
        effective_cache_mode = "Off"
    missing = (
        required_nodes_for(
            request.media.mode,
            effective_sol,
            effective_cache_mode,
            use_sage=effective_sage,
            use_sla=effective_sla,
            use_turbo=use_turbo,
            turbo_variant=selected_turbo,
            model_filename=selected_model,
            turbo_lora_filename=turbo_lora_name or "",
            latent_upscale=bool(request.finishing.latent_upscale),
            latent_upscale_method=resolved_latent_upscale_method,
            result_format=request.output.result_format,
            image_vae=selected_image_vae,
            stage_model_offload=effective_stage_offload,
            smart_stage_offload=smart_stage_offload,
        )
        - available
    )
    if missing:
        raise H3Error("Missing ComfyUI nodes: " + ", ".join(sorted(missing)))

    refs_i = collect_reference_slots(
        request.media.ref_image_1,
        request.media.ref_image_2,
        request.media.ref_image_3,
        request.media.ref_image_4,
        request.media.ref_image_5,
        request.media.ref_image_6,
        request.media.ref_image_7,
        request.media.ref_image_8,
        request.media.ref_image_9,
    )
    refs_v = collect_reference_slots(
        request.media.ref_video_1, request.media.ref_video_2, request.media.ref_video_3
    )
    refs_a = collect_reference_slots(
        request.media.ref_audio_1, request.media.ref_audio_2, request.media.ref_audio_3
    )

    if request.media.mode == "Text to video":
        request.media.first_image = None
        request.media.last_image = None
    elif request.media.mode == "First / last frame":
        if not request.media.first_image and not request.media.last_image:
            raise H3Error("Provide a first frame, a last frame, or both.")
    elif request.media.mode == "Reference media":
        if not (refs_i or refs_v or refs_a):
            raise H3Error(
                "Reference mode requires at least one image, video, or audio file."
            )

    return PreparedH3(
        actual_seed=actual_seed,
        available=available,
        cache_note=cache_note,
        effective_cache_mode=effective_cache_mode,
        effective_sage=effective_sage,
        effective_scheduler=effective_scheduler,
        effective_sla=effective_sla,
        effective_sla_inputs=effective_sla_inputs,
        effective_sla_preset=effective_sla_preset,
        effective_sol=effective_sol,
        effective_stage_offload=effective_stage_offload,
        effective_steps=effective_steps,
        generation_frames=generation_frames,
        generation_note=generation_note,
        latent_source_height=latent_source_height,
        latent_source_width=latent_source_width,
        latent_split_config=latent_split_config,
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        models=models,
        packed_tokens=packed_tokens,
        plan=plan,
        refs_a=refs_a,
        refs_i=refs_i,
        refs_v=refs_v,
        requested_image_frames=requested_image_frames,
        resolved_height=resolved_height,
        resolved_latent_upscale_method=resolved_latent_upscale_method,
        resolved_width=resolved_width,
        selected_image_vae=selected_image_vae,
        selected_label=selected_label,
        selected_model=selected_model,
        selected_text_encoder=selected_text_encoder,
        selected_turbo=selected_turbo,
        smart_stage_offload=smart_stage_offload,
        sol_reason=sol_reason,
        turbo_lora_name=turbo_lora_name,
        turbo_strength=turbo_strength,
        voice_refs=voice_refs,
    )
