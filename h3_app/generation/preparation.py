"""Validate and provision the effective H3 request before graph construction."""

from __future__ import annotations

from h3_app.jobs import CURRENT_JOB, variant_seed
from h3_app.decoder_intent import VideoDecoder

import random
from dataclasses import dataclass
from typing import Any, Generator

from h3_app.catalog import (
    COMFY_UPSCALE_OPTIONS,
    DEFAULT_SLA_PRESET,
    GENERATION_POSTPROCESS_OPTIONS,
    H3_LATENT_UPSCALE_STANDARD,
    H3_REGENERATE_MODELS,
    H3_SEMANTIC_BRIDGE_NODE,
    LARRY_TURBO,
    REFINEMENT_LORA_SETTINGS,
    SAME_REFINEMENT_LORA,
    LTX25_UPSCALE_OPTIONS,
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
from h3_app.workflows.h3 import required_nodes_for, turbo_required_nodes
from h3_app.workflows.upscale import required_upscale_nodes
from h3_models import MODEL_SPECS

from .requests import H3Request
from .results import GenerationUpdate
from .services import GenerationServices


def _validate_sampling_steps(
    profile_key: str,
    use_turbo: bool,
    selected_turbo: str,
    effective_steps: int,
) -> None:
    if use_turbo:
        minimum = 4
        if effective_steps < minimum:
            raise H3Error(f"Turbo requires at least {minimum} steps.")
    elif effective_steps < 10:
        raise H3Error(
            "Normal H3 generation requires at least 10 steps. "
            "Use Generation=Turbo for lower-step generation."
        )


@dataclass(frozen=True)
class PreparedH3:
    refinement_lora_name: str | None
    refinement_variant: str | None
    actual_seed: int
    available: set[str]
    cache_note: str | None
    effective_cache_mode: str
    effective_sage: bool
    effective_scheduler: str
    effective_sla: bool
    effective_veda: bool
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


def _has_visual_conditioning(request: H3Request) -> bool:
    if request.media.mode == "First / last frame":
        return bool(request.media.first_image or request.media.last_image)
    if request.media.mode == "Reference media":
        return any(
            getattr(request.media, f"ref_{kind}_{i}")
            for kind, count in (("image", 9), ("video", 3))
            for i in range(1, count + 1)
        )
    return False


def _ensure_text_encoder(request, services, runtime, models, settings, *, started, progress):
    """Download the selected text encoder; returns its file name and whether it is BF16."""
    text_encoder_key, selected_text_encoder, bf16_text_encoder = settings
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
    return services.models.ensure_h3_text_encoder(models, request.sampling.text_encoder)


def _ensure_decoders(request, services, runtime, models, *, has_visual_conditioning, use_lightvae,
                     selected_image_vae, started, progress):
    """The selected video encoder/decoder and image decoder files."""
    if use_lightvae:
        progress(0, desc="Preparing LightVAE")
        services.models.ensure_lightvae(
            request.output.video_encoder if has_visual_conditioning else "Original",
            request.output.lightvae_decoder,
        )
    if request.output.use_int8_vae:
        progress(0, desc="Preparing INT8 video VAE")
        services.models.ensure_int8_video_vae(models)
    if request.output.use_lynnreal_vae:
        progress(0, desc="Preparing LynnReal Light INT8 video VAE")
        services.models.ensure_int8_video_vae(models, lynnreal=True)
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


def _ensure_profile_model(request, services, runtime, models, *, started, progress):
    """The selected base checkpoint for this task; returns (profile key, profile, file name)."""
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
    return profile_key, profile, selected_model


def _ensure_base_vaes(request, services, runtime, models, *, has_visual_conditioning, started, progress):
    # H3 conditioning requires a video VAE even for Audio output and the
    # single-frame image decoder. Provision the VAE used by add_model_stack,
    # independently of whether the result includes decoded video.
    if has_visual_conditioning or request.output.lightvae_decoder != "none" or not any(
        (
            request.output.use_int8_vae,
            request.output.use_trt_vae,
            request.output.use_lynnreal_vae,
        )
    ):
        video_vae_path = (
            runtime.comfy_dir
            / "models"
            / MODEL_SPECS["video_vae"].folder
            / models.video_vae
        )
        if not services.models.model_file_is_ready(video_vae_path):
            progress(0, desc="Downloading video VAE")
            yield GenerationUpdate(
                None,
                progress_status("Downloading video VAE on demand", started=started),
            )
            services.models.ensure_base_video_vae(models)

    if request.output.result_format != "Image":
        audio_vae_path = (
            runtime.comfy_dir
            / "models"
            / MODEL_SPECS["audio_vae"].folder
            / models.audio_vae
        )
        if not services.models.model_file_is_ready(audio_vae_path):
            progress(0, desc="Downloading audio VAE")
            yield GenerationUpdate(
                None,
                progress_status("Downloading audio VAE on demand", started=started),
            )
            services.models.ensure_audio_vae(models)


@dataclass(frozen=True)
class _TurboChoice:
    use_turbo: bool
    selected_turbo: str
    generation_note: str | None
    turbo_lora_name: str | None
    turbo_strength: float
    label: str


def _resolve_turbo(request, effective, services, models, profile, *, progress) -> _TurboChoice:
    requested_generation = str(effective.generation_mode).strip().lower()
    use_turbo = requested_generation == "turbo"
    selected_turbo = normalize_turbo_variant(effective.sampling.turbo_variant)
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
        label = f"{profile.label} · Turbo · {selected_turbo}"
        turbo_strength = turbo_strength_for(selected_turbo)
    else:
        label = f"{profile.label} · Normal"
        turbo_lora_name = None
        turbo_strength = 1.0
    return _TurboChoice(use_turbo, selected_turbo, generation_note, turbo_lora_name, turbo_strength, label)


def _run_label(request, turbo: _TurboChoice, *, has_visual_conditioning, effective_stage_offload,
               selected_image_vae) -> str:
    """What the status reports this run uses."""
    decoder_label = VideoDecoder.from_flags(
        request.output.use_int8_vae, request.output.use_lynnreal_vae,
        request.output.use_trt_vae, request.output.lightvae_decoder,
    ).value
    selected_label = turbo.label + f" · {decoder_label} VAE"
    if has_visual_conditioning:
        selected_label += f" · video encoder {request.output.video_encoder}"
    selected_label += (
        f" · text encoder {request.sampling.text_encoder} · stage offload "
        f"{'on' if effective_stage_offload else 'off'}"
    )
    if request.output.result_format == "Image":
        selected_label += f" · image decoder {selected_image_vae}"
    return selected_label


@dataclass(frozen=True)
class _LatentUpscale:
    refinement_lora_name: str | None = None
    refinement_variant: str | None = None
    model_name: str | None = None
    precision: str = "bf16"
    method: str = H3_LATENT_UPSCALE_STANDARD
    split_config: H3SplitUpscaleConfig | None = None
    # First-pass size, then the final size the UI asked for (on the 64-pixel grid).
    source_size: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)


def _prepare_latent_upscale(request, services, runtime, models, *, width, height, effective_steps, started,
                            progress):
    """Refinement LoRA, split config, sizes and upscaler for the optional native latent upscale."""
    if not request.finishing.latent_upscale:
        return _LatentUpscale(source_size=(width, height), size=(width, height))
    refinement_lora_name = None
    refinement_variant = None
    method = resolve_h3_latent_upscale_method(request.finishing.latent_upscale_method)
    refine_choice = request.finishing.latent_upscale_refine_lora
    if method in H3_REGENERATE_MODELS:
        refinement_variant = method
        key = H3_REGENERATE_MODELS[method]
        refinement_lora_name = MODEL_SPECS[key].local_name
        progress(0, desc=f"Preparing {method}")
        services.models.ensure_model_keys((key, "turbo_lora"))
        # This schedule is independent of the draft's sampler and step count.
        request.finishing.latent_upscale_refine_steps = 3
    elif refine_choice != SAME_REFINEMENT_LORA:
        if refine_choice not in REFINEMENT_LORA_SETTINGS:
            raise H3Error(f"Unknown refinement LoRA: {refine_choice}")
        refinement_variant = refine_choice
        refinement_lora_name = models.turbo_lora_for(
            request.media.mode, refine_choice
        )
        if not refinement_lora_name:
            raise H3Error(f"{refine_choice} refinement LoRA is not configured.")
        progress(0, desc=f"Preparing {refine_choice} refinement LoRA")
        services.models.ensure_turbo_lora(models, refine_choice, request.media.mode)
    split_config = resolve_h3_split_upscale_config(
        method,
        tile_width=request.finishing.latent_split_tile_width,
        tile_height=request.finishing.latent_split_tile_height,
        overlap_ratio=request.finishing.latent_split_overlap_ratio,
        fade_ratio=request.finishing.latent_split_fade_ratio,
        chunk_frames=request.finishing.latent_split_chunk_frames,
        temporal_overlap_frames=request.finishing.latent_split_temporal_overlap_frames,
        seam_denoise=request.finishing.latent_split_seam_denoise,
        seam_polish=request.finishing.latent_split_seam_polish,
    )
    source_width, source_height, width, height = h3_latent_upscale_dimensions(width, height)
    refine_steps = int(request.finishing.latent_upscale_refine_steps)
    if method not in H3_REGENERATE_MODELS and (refine_steps < 1 or refine_steps >= effective_steps):
        raise H3Error(
            "H3 latent upscale refinement steps must be at least 1 and smaller "
            f"than the total {effective_steps} sampling steps."
        )
    (
        _latent_upscale_key,
        model_name,
        precision,
    ) = h3_latent_upscaler_settings(request.finishing.latent_upscaler_model)
    destination = (
        runtime.comfy_dir
        / "models"
        / "latent_upscale_models"
        / model_name
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
    return _LatentUpscale(refinement_lora_name, refinement_variant, model_name, precision, method, split_config,
                          (source_width, source_height), (width, height))


def _check_backend_extras(request, services, available, *, use_lightvae, started, progress, models=None):
    """LightVAE, Semantic Bridge and finishing nodes the backend must offer."""
    if use_lightvae and "LightVAELoader" not in available:
        raise H3Error(
            "LightVAE is unavailable. Update setup_h3.py and restart ComfyUI; "
            "rebuild/redeploy Modal images."
        )
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
            required_upscale_nodes(request.finishing.postprocess, models) - available
        )
        if missing_upscale_nodes:
            raise H3Error(
                f"{request.finishing.postprocess} requires current ComfyUI nodes: "
                + ", ".join(sorted(missing_upscale_nodes))
            )


@dataclass(frozen=True)
class _Attention:
    sol: bool
    packed_tokens: int
    sol_reason: str
    sage: bool
    sla: bool
    veda: bool
    sla_preset: str
    sla_inputs: dict[str, Any]
    cache_mode: str
    cache_note: str | None


def _resolve_attention(request, effective, services, *, use_turbo, width, height, duration) -> _Attention:
    """Attention route (Sol, Sage, SLA, Veda) and cache mode for this run."""
    effective_sol, packed_tokens, sol_reason = services.policy.resolve_sol_policy(
        effective.sampling.attention_mode,
        request.media.mode,
        width,
        height,
        duration,
        request.media.first_image,
        request.media.last_image,
        use_turbo=use_turbo,
    )
    attention = str(effective.sampling.attention_mode).strip().lower()
    effective_sage = attention in {"sage", "sage 2", "sage2"}
    effective_sla = attention in {"sla", "sla attention", "sparse-linear"}
    effective_veda = attention == "veda"
    effective_sla_preset, effective_sla_inputs = resolve_sla_preset(
        request.sampling.sla_preset if effective_sla or effective_veda else DEFAULT_SLA_PRESET
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
    return _Attention(effective_sol, packed_tokens, sol_reason, effective_sage, effective_sla, effective_veda,
                      effective_sla_preset, effective_sla_inputs, effective_cache_mode, cache_note)


def _collect_references(request: H3Request) -> tuple[list[str], list[str], list[str]]:
    """The filled reference slots; checks each task has the media it needs."""
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
    return refs_i, refs_v, refs_a


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
    request.output.use_lynnreal_vae = effective.use_lynnreal_vae
    request.output.video_encoder = effective.video_encoder
    request.output.lightvae_decoder = effective.lightvae_decoder
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
    actual_seed = variant_seed(
        (CURRENT_JOB.get().variant if CURRENT_JOB.get() else 0),
        lambda: (
            random.randrange(0, 2**63 - 1)
            if int(request.output.seed) < 0
            else int(request.output.seed)
        ),
    )
    models = services.models.load_model_config()
    text_encoder_settings = services.models.h3_text_encoder_settings(
        models, request.sampling.text_encoder
    )
    bf16_text_encoder = text_encoder_settings[2]
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
    selected_text_encoder, bf16_text_encoder = yield from _ensure_text_encoder(
        request, services, runtime, models, text_encoder_settings, started=started, progress=progress
    )
    has_visual_conditioning = _has_visual_conditioning(request)
    use_lightvae = request.output.lightvae_decoder != "none" or (
        has_visual_conditioning and request.output.video_encoder == "LightVAE Lite"
    )
    yield from _ensure_decoders(
        request, services, runtime, models, has_visual_conditioning=has_visual_conditioning,
        use_lightvae=use_lightvae, selected_image_vae=selected_image_vae, started=started, progress=progress,
    )
    if request.finishing.postprocess == SEEDVR2_UPSCALE:
        seedvr2_upscale_model_names(models, request.finishing.seedvr2_model)
    elif request.finishing.postprocess in LTX25_UPSCALE_OPTIONS:
        ltx25_model_keys(request.finishing.ltx25_model)
    profile_key, profile, selected_model = yield from _ensure_profile_model(
        request, services, runtime, models, started=started, progress=progress
    )
    yield from _ensure_base_vaes(
        request, services, runtime, models, has_visual_conditioning=has_visual_conditioning,
        started=started, progress=progress,
    )

    turbo = _resolve_turbo(request, effective, services, models, profile, progress=progress)
    selected_label = _run_label(
        request, turbo, has_visual_conditioning=has_visual_conditioning,
        effective_stage_offload=effective_stage_offload, selected_image_vae=selected_image_vae,
    )

    # Variant defaults update outside the generation queue, while this
    # request deliberately honors any subsequent manual step adjustment.
    effective_steps = int(effective.sampling.steps)
    effective_scheduler = str(effective.sampling.scheduler)

    _validate_sampling_steps(
        profile_key,
        turbo.use_turbo,
        turbo.selected_turbo,
        effective_steps,
    )

    latent = yield from _prepare_latent_upscale(
        request, services, runtime, models, width=resolved_width, height=resolved_height,
        effective_steps=effective_steps, started=started, progress=progress,
    )
    latent_source_width, latent_source_height = latent.source_size
    resolved_width, resolved_height = latent.size

    info = services.execution.object_info()
    available = set(info)
    yield from _check_backend_extras(
        request, services, available, use_lightvae=use_lightvae, started=started, progress=progress, models=models
    )
    attention = _resolve_attention(
        request, effective, services, use_turbo=turbo.use_turbo,
        width=resolved_width, height=resolved_height, duration=policy_duration,
    )
    missing = (
        required_nodes_for(
            request.media.mode,
            attention.sol,
            attention.cache_mode,
            use_sage=attention.sage,
            use_sla=attention.sla,
            use_veda=attention.veda,
            use_turbo=turbo.use_turbo,
            turbo_variant=turbo.selected_turbo,
            model_filename=selected_model,
            turbo_lora_filename=turbo.turbo_lora_name or "",
            latent_upscale=bool(request.finishing.latent_upscale),
            latent_upscale_method=latent.method,
            result_format=request.output.result_format,
            image_vae=selected_image_vae,
            stage_model_offload=effective_stage_offload,
            smart_stage_offload=smart_stage_offload,
        )
        - available
    )
    if latent.refinement_variant:
        missing |= (
            turbo_required_nodes(latent.refinement_variant, latent.refinement_lora_name or "")
            - available
        )
    if missing:
        raise H3Error("Missing ComfyUI nodes: " + ", ".join(sorted(missing)))

    if attention.veda:
        progress(0, desc="Preparing Veda predictor")
        yield GenerationUpdate(
            None, progress_status("Preparing Veda predictor (download on first use)", started=started)
        )
        services.models.ensure_h3_veda_predictor(request.media.mode)

    refs_i, refs_v, refs_a = _collect_references(request)

    return PreparedH3(
        refinement_lora_name=latent.refinement_lora_name,
        refinement_variant=latent.refinement_variant,
        actual_seed=actual_seed,
        available=available,
        cache_note=attention.cache_note,
        effective_cache_mode=attention.cache_mode,
        effective_sage=attention.sage,
        effective_scheduler=effective_scheduler,
        effective_sla=attention.sla,
        effective_veda=attention.veda,
        effective_sla_inputs=attention.sla_inputs,
        effective_sla_preset=attention.sla_preset,
        effective_sol=attention.sol,
        effective_stage_offload=effective_stage_offload,
        effective_steps=effective_steps,
        generation_frames=generation_frames,
        generation_note=turbo.generation_note,
        latent_source_height=latent_source_height,
        latent_source_width=latent_source_width,
        latent_split_config=latent.split_config,
        latent_upscale_model_name=latent.model_name,
        latent_upscale_precision=latent.precision,
        models=models,
        packed_tokens=attention.packed_tokens,
        plan=plan,
        refs_a=refs_a,
        refs_i=refs_i,
        refs_v=refs_v,
        requested_image_frames=requested_image_frames,
        resolved_height=resolved_height,
        resolved_latent_upscale_method=latent.method,
        resolved_width=resolved_width,
        selected_image_vae=selected_image_vae,
        selected_label=selected_label,
        selected_model=selected_model,
        selected_text_encoder=selected_text_encoder,
        selected_turbo=turbo.selected_turbo,
        smart_stage_offload=smart_stage_offload,
        sol_reason=attention.sol_reason,
        turbo_lora_name=turbo.turbo_lora_name,
        turbo_strength=turbo.turbo_strength,
        voice_refs=voice_refs,
    )
