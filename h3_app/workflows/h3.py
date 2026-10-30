"""MiniMax H3 FL2VA and Ref2VA graph builders and their required nodes."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import hashlib
import inspect
import json
from typing import Any, Callable

from h3_app.catalog import (
    AUTO_SOL_TOKEN_THRESHOLD,
    CHUNK_FEED_FORWARD_NODE,
    CORE_SAMPLER_NODE,
    CORE_LORA_LOADER_NODE,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_IMAGE_VAE,
    DEFAULT_RESULT_FORMAT,
    DEFAULT_SLA_PRESET,
    DMAD_4STEP_LORA,
    FUSED_MODULATION_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_CONDITIONING_CACHE_NODE,
    H3_IMAGE_SLICES_NODE,
    H3_LATENT_UPSCALE_SPLIT,
    H3_LATENT_UPSCALE_STANDARD,
    H3_LATENT_UPSCALER_NODE,
    H3_LATENT_UPSCALER_LOADER_NODE,
    H3_NVENC_SAVE_NODE,
    H3_REFINEMENT_COMPILER_GUARD_NODE,
    H3_SEMANTIC_BRIDGE_NODE,
    H3_SEPARATE_AV_LATENT_NODE,
    H3_SIGMA_SHIFT_NODE,
    H3_SINGLE_FRAME_VAE_LOADER_NODE,
    H3_SPLIT_SPATIAL_PARAMS_NODE,
    H3_SPLIT_TEMPORAL_PARAMS_NODE,
    H3_SPLIT_UPSCALE_NODE,
    H3_STAGE_OFFLOAD_NODE,
    H3_STAGE_OFFLOAD_POLICY_NODE,
    LARRY_TURBO_LORA_NODE,
    LARRY_TURBO_SAMPLER_NODE,
    LIGHTX2V_4STEP_TURBO,
    LIGHTX2V_BYPASS_LORA_NODE,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    SAGE_ATTENTION_NODE,
    SINGLE_FRAME_IMAGE_VAE,
    SLA_ATTENTION_NODE,
    VEDA_ATTENTION_NODE,
    SOL_ATTENTION_NODE,
    SPECTRUM_DEFAULT_INPUTS,
    STANDARD_TURBO_LORA_SETTINGS,
    REFINEMENT_LORA_SETTINGS,
    H3_REGENERATE_MODELS,
    H3_REGENERATE_SIGMAS,
)
from h3_app.errors import H3Error
from h3_models import MODEL_SPECS
from h3_app.workflows.regenerate import add_regenerate_model
from h3_app.workflows.latent_upscale import add_h3_latent_upscale
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig, trt_vae_engine_name
from h3_app.policy import (
    H3SplitUpscaleConfig,
    frame_length,
    h3_latent_upscale_dimensions,
    image_sampling_length,
    normalize_image_vae,
    normalize_result_format,
    normalize_turbo_variant,
    resolve_h3_latent_upscale_method,
    resolve_sla_preset,
    selected_image_sampling_length,
    snap32,
    turbo_sampler_name,
    turbo_sigma_shifts,
    turbo_uses_custom_nodes,
    validate_image_frame_count,
)


def turbo_required_nodes(
    turbo_variant: str,
    lora_filename: str = "",
) -> set[str]:
    """Return the external node contract for one normalized Turbo variant."""
    if turbo_variant in H3_REGENERATE_MODELS:
        return {CORE_LORA_LOADER_NODE, CORE_SAMPLER_NODE, "ManualSigmas", "H3RegeneratePrepareLatent"}
    if turbo_variant in STANDARD_TURBO_LORA_SETTINGS:
        required = {CORE_LORA_LOADER_NODE, CORE_SAMPLER_NODE}
        if turbo_variant == DMAD_4STEP_LORA:
            required.add(H3_SIGMA_SHIFT_NODE)
        return required
    if turbo_uses_custom_nodes(turbo_variant):
        return {LARRY_TURBO_LORA_NODE, LARRY_TURBO_SAMPLER_NODE}
    required = {
        LIGHTX2V_BYPASS_LORA_NODE,
        CORE_SAMPLER_NODE,
        FUSED_MODULATION_NODE,
    }
    if turbo_sigma_shifts(turbo_variant, lora_filename) is not None:
        required.add(H3_SIGMA_SHIFT_NODE)
    return required


def add_turbo_model_patch(
    graph: Graph,
    model_ref: list[Any],
    *,
    lora_name: str,
    turbo_variant: str,
    strength: float,
    available_nodes: set[str],
) -> list[Any]:
    """Apply a Turbo LoRA and compatible model-level optimizations."""
    variant = (
        turbo_variant if turbo_variant in STANDARD_TURBO_LORA_SETTINGS
        else normalize_turbo_variant(turbo_variant)
    )
    required = turbo_required_nodes(variant, lora_name)
    missing = required - available_nodes
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise H3Error(
            f"{variant} Turbo requires unavailable nodes: {missing_names}. "
            "Re-run setup_h3.py and restart ComfyUI."
        )

    if variant in STANDARD_TURBO_LORA_SETTINGS:
        # The ComfyUI conversion embeds alpha tensors for the standard loader.
        turbo = graph.add(
            CORE_LORA_LOADER_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength_model=float(strength),
        )
        return Graph.out(turbo)

    if turbo_uses_custom_nodes(variant):
        turbo = graph.add(
            LARRY_TURBO_LORA_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength=float(strength),
            low_vram=False,
        )
        # Larry's documented sharp path is runtime bypass on every base. Its
        # node handles fused ConvRot INT8 fc2 projections through transient
        # weight patches because those kernels never call fc2.forward.
        return Graph.out(turbo)

    turbo = graph.add(
        LIGHTX2V_BYPASS_LORA_NODE,
        model=model_ref,
        lora_name=lora_name,
        strength=float(strength),
    )

    # The bundled node uses activation-space bypass for every base and the same
    # fused-INT8 fc2 exception as Larry. All bases retain the standard AdaLN
    # shape, so fused modulation remains compatible.
    fused_modulation = graph.add(
        FUSED_MODULATION_NODE,
        model=Graph.out(turbo),
        enabled=True,
    )
    return Graph.out(fused_modulation)


def add_model_stack(
    graph: Graph,
    model_name: str,
    models: ModelConfig,
    *,
    turbo_lora_name: str | None,
    turbo_variant: str,
    turbo_strength: float,
    use_sol: bool,
    sol_tau: float,
    sol_thresh_type: str,
    sol_exact_mode: str,
    sol_dense_steps: int,
    sol_step_off: float,
    sol_sink_tokens: int,
    cache_mode: str,
    fbcache_preset: str,
    fbcache_threshold: float,
    fbcache_start: float,
    fbcache_end: float,
    fbcache_max_hits: int,
    fbcache_temporal_guard: bool,
    easycache_threshold: float,
    easycache_start: float,
    easycache_end: float,
    easycache_verbose: bool,
    available_nodes: set[str],
    text_encoder_name: str | None = None,
    lightvae_decoder: str = "none",
    use_lynnreal_vae: bool = False,
    use_sol_vae: bool = False,
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    use_veda: bool = False,
    veda_predictor: str | None = None,
    sampling_steps: int = 4,
    sla_preset: str = DEFAULT_SLA_PRESET,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    if use_veda and (use_sla or use_sol or use_sage):
        raise H3Error("Veda must be selected as the only attention override.")
    unet = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model_ref = Graph.out(unet)

    if turbo_lora_name:
        model_ref = add_turbo_model_patch(
            graph,
            model_ref,
            lora_name=turbo_lora_name,
            turbo_variant=turbo_variant,
            strength=turbo_strength,
            available_nodes=available_nodes,
        )

    # Keep FirstBlockCache ahead of attention/object patches so its sampling and
    # diffusion wrappers own the outer execution context.
    cache_mode_normalized = str(cache_mode).strip().lower()
    if cache_mode_normalized == "firstblockcache":
        if "H3FirstBlockCache" not in available_nodes:
            raise H3Error(
                "FirstBlockCache was requested, but H3FirstBlockCache is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        cache = graph.add(
            "H3FirstBlockCache",
            model=model_ref,
            preset=str(fbcache_preset),
            residual_diff_threshold=float(fbcache_threshold),
            start_percent=float(fbcache_start),
            end_percent=float(fbcache_end),
            max_consecutive_cache_hits=max(1, int(fbcache_max_hits)),
            temporal_guard=bool(fbcache_temporal_guard),
        )
        model_ref = Graph.out(cache)

    if use_veda:
        if VEDA_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "Veda was requested, but VedaSparseAttention is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        preset, _ = resolve_sla_preset(sla_preset)
        dense_steps = "0"
        if preset == "Quality":
            dense_steps += f", {max(0, int(sampling_steps) - 1)}"
        veda = graph.add(
            VEDA_ATTENTION_NODE,
            model=model_ref,
            predictor=veda_predictor or MODEL_SPECS["veda_t2va"].local_name,
            # Ratios can retain too few tiles in the small first pass. Use
            # upstream's fixed budget for both stages and keep references dense.
            generated_sparsity="32",
            reference_sparsity="0%",
            full_attention_layers="",
            full_attention_steps=dense_steps,
            verbose=False,
        )
        model_ref = Graph.out(veda)

    if use_sla:
        if SLA_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "SLA was requested, but H3 SLA Attention is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        _sla_name, sla_inputs = resolve_sla_preset(sla_preset)
        sla = graph.add(
            SLA_ATTENTION_NODE,
            model=model_ref,
            **sla_inputs,
            # Preserve the audio-safe legacy sparse path and first-step anchor.
            # Kitchen's sparse engine cannot represent all audio/ref spans.
            engine="triton",
            use_int8_qk=False,
            tail_correction=False,
            dense_steps="0",
            enabled=True,
        )
        model_ref = Graph.out(sla)

    if use_sage:
        if SAGE_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "Sage 2 was requested, but Patch Sage Attention KJ is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        sage = graph.add(
            SAGE_ATTENTION_NODE,
            model=model_ref,
            sage_attention="auto",
            allow_compile=False,
        )
        model_ref = Graph.out(sage)

    if use_sol:
        if SOL_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "Sol-Attn was requested, but the H3 zero-copy Sol node is not loaded. "
                "Run deploy_h3.sh install and inspect the ComfyUI startup log."
            )
        exact_mode = (
            sol_exact_mode
            if sol_exact_mode in {"off", "exact_kv", "exact_kv_and_rows"}
            else "off"
        )
        dense_block_count = max(0, min(int(sol_dense_steps), 8))
        dense_blocks = ",".join(
            f"-{index}" for index in range(1, dense_block_count + 1)
        )

        # The H3-native path consumes strided q/k/v views directly, avoiding
        # the large contiguous copies required by generic attention hooks.
        sol = graph.add(
            SOL_ATTENTION_NODE,
            model=model_ref,
            enabled=True,
            tau=float(sol_tau),
            min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
            strict=False,
            thresh_type=(
                sol_thresh_type if sol_thresh_type in {"diag", "exact"} else "diag"
            ),
            int8_qk=False,
            int8_pv=False,
            sink_conditioning=exact_mode,
            dense_blocks=dense_blocks,
        )
        model_ref = Graph.out(sol)

    # The ConvRot quality checkpoints have the largest feed-forward activation
    # peak. Two-way token chunking preserves their row-wise quantization math
    # while substantially reducing peak VRAM.
    if "convrot" in model_name.lower():
        if CHUNK_FEED_FORWARD_NODE not in available_nodes:
            raise H3Error(
                "The quality model requires MiniMaxH3ChunkFeedForward, but the "
                "updated Sol-Attn plugin is not loaded. Re-run setup_h3.py."
            )
        chunked = graph.add(
            CHUNK_FEED_FORWARD_NODE,
            model=model_ref,
            enabled=True,
            chunks=2,
            min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
        )
        model_ref = Graph.out(chunked)

    # Keep Spectrum after LoRA, Sol-Attn, and feed-forward patches so actual
    # anchor evaluations observe the final H3 model path. The selected radio
    # mode prevents it from being stacked with either cache implementation.
    if cache_mode_normalized == "spectrum":
        if "SpectrumApplyMiniMaxH3" not in available_nodes:
            raise H3Error(
                "Spectrum was requested, but SpectrumApplyMiniMaxH3 is not "
                "loaded. Re-run setup_h3.py and restart ComfyUI."
            )
        spectrum = graph.add(
            "SpectrumApplyMiniMaxH3",
            model=model_ref,
            **SPECTRUM_DEFAULT_INPUTS,
        )
        model_ref = Graph.out(spectrum)

    if cache_mode_normalized == "easycache":
        if "EasyCache" not in available_nodes:
            raise H3Error(
                "EasyCache was requested, but the native ComfyUI EasyCache node "
                "is not loaded. Update ComfyUI and restart the service."
            )
        if not 0.0 <= float(easycache_threshold) <= 3.0:
            raise H3Error("EasyCache threshold must be between 0 and 3.")
        if not 0.0 <= float(easycache_start) < float(easycache_end) <= 1.0:
            raise H3Error("EasyCache requires 0 ≤ start percent < end percent ≤ 1.")
        cache = graph.add(
            "EasyCache",
            model=model_ref,
            reuse_threshold=float(easycache_threshold),
            start_percent=float(easycache_start),
            end_percent=float(easycache_end),
            verbose=bool(easycache_verbose),
        )
        model_ref = Graph.out(cache)

    sigma_shifts = turbo_sigma_shifts(turbo_variant, turbo_lora_name)
    if sigma_shifts is not None:
        if H3_SIGMA_SHIFT_NODE not in available_nodes:
            raise H3Error(
                f"{turbo_variant} Turbo requires MiniMaxH3SigmaShift. "
                "Update ComfyUI and restart the service."
            )
        shifted = graph.add(
            H3_SIGMA_SHIFT_NODE,
            model=model_ref,
            shift_video=sigma_shifts[0],
            shift_audio=sigma_shifts[1],
        )
        model_ref = Graph.out(shifted)

    clip = graph.add(
        "CLIPLoader",
        clip_name=text_encoder_name or models.text_encoder,
        type="minimax",
        device="default",
    )
    if lightvae_decoder not in {"none", "lite", "int8"}:
        raise H3Error("Unsupported LightVAE decoder.")
    if sum((use_sol_vae, use_int8_vae, use_trt_vae, use_lynnreal_vae, lightvae_decoder != "none")) > 1:
        raise H3Error("Select only one video decoder.")
    if use_trt_vae:
        if "MiniMaxH3TRTVAELoader" not in available_nodes:
            raise H3Error(
                "TensorRT VAE is unavailable. Run setup_h3.py and restart ComfyUI."
            )
        if not models.video_vae_trt_decoder:
            raise H3Error("TensorRT VAE is missing from the model catalog.")
        video_vae = graph.add(
            "MiniMaxH3TRTVAELoader",
            decoder=trt_vae_engine_name(models.video_vae_trt_decoder),
            encoder="None",
        )
    else:
        if use_sol_vae:
            video_vae_name = models.video_vae_sol_int8
        elif use_lynnreal_vae:
            if not models.video_vae_lynnreal_int8:
                raise H3Error("LynnReal Light INT8 VAE is not configured. Re-run setup_h3.py.")
            video_vae_name = models.video_vae_lynnreal_int8
        elif use_int8_vae:
            if not models.video_vae_int8:
                raise H3Error("INT8 video VAE is missing from the model catalog.")
            video_vae_name = models.video_vae_int8
        else:
            video_vae_name = models.video_vae
        video_vae = graph.add("VAELoader", vae_name=video_vae_name)
    if lightvae_decoder != "none":
        video_vae = add_lightvae(graph, Graph.out(video_vae), available_nodes,
                                 decoder=lightvae_decoder)
    audio_vae = graph.add("VAELoader", vae_name=models.audio_vae)
    return model_ref, Graph.out(clip), Graph.out(video_vae), Graph.out(audio_vae)


def add_lightvae(graph, base_vae, available_nodes, *, encoder="Original", decoder="none"):
    if "LightVAELoader" not in available_nodes:
        raise H3Error("LightVAE is unavailable. Update setup_h3.py and restart ComfyUI; rebuild/redeploy Modal images.")
    key = "lightvae_decoder_int8" if decoder == "int8" else "lightvae_decoder"
    return graph.add(
        "LightVAELoader", base_vae=base_vae,
        encoder=MODEL_SPECS["lightvae_encoder"].local_name if encoder == "LightVAE Lite" else "none",
        decoder=MODEL_SPECS[key].local_name if decoder != "none" else "none",
    )


def h3_conditioning_video_vae(
    graph: Graph,
    models: ModelConfig,
    decode_vae_ref: list[Any],
    *,
    use_trt_vae: bool,
    has_visual_conditioning: bool,
    video_encoder: str = "Original",
    available_nodes: set[str] | None = None,
) -> list[Any]:
    """Condition through the independently selected original or Lite encoder."""
    if video_encoder not in {"Original", "LightVAE Lite"}:
        raise H3Error("Unsupported video encoder.")
    if not has_visual_conditioning:
        return decode_vae_ref
    # Only the original full VAE can be reused for original encoding. Decoder
    # checkpoints (including quantized and distilled ones) never select encoding.
    node = graph.nodes.get(decode_vae_ref[0], {})
    if (video_encoder == "Original" and not use_trt_vae
            and node.get("class_type") == "VAELoader"
            and node.get("inputs", {}).get("vae_name") == models.video_vae):
        return decode_vae_ref
    encoder_vae = graph.add("VAELoader", vae_name=models.video_vae)
    if video_encoder == "LightVAE Lite":
        encoder_vae = add_lightvae(graph, Graph.out(encoder_vae), available_nodes or set(),
                                   encoder=video_encoder)
    return Graph.out(encoder_vae)


def h3_conditioning_cache_key(
    mode: str,
    prompt: str,
    text_encoder_name: str,
    media: list[tuple[str, str]],
    *,
    encoder_settings: dict[str, Any] | None = None,
) -> str:
    payload = [
        mode,
        prompt,
        text_encoder_name,
        media,
        encoder_settings or {},
    ]
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def add_h3_stage_offload(
    graph: Graph,
    conditioning_ref: list[Any],
    latent_ref: list[Any],
    additional_conditioning_ref: list[Any] | None = None,
    additional_latent_ref: list[Any] | None = None,
    enabled_ref: list[Any] | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Insert an execution barrier that unloads one H3 stage before the next."""
    inputs = {
        "conditioning": conditioning_ref,
        "latent": latent_ref,
        "additional_conditioning": additional_conditioning_ref or conditioning_ref,
        "additional_latent": additional_latent_ref or latent_ref,
    }
    if enabled_ref is not None:
        inputs["enabled"] = enabled_ref
    barrier = graph.add(H3_STAGE_OFFLOAD_NODE, **inputs)
    return tuple(Graph.out(barrier, slot) for slot in range(4))  # type: ignore[return-value]


def h3_refinement_attention_model(
    graph: Graph, model_ref: list[Any], *, refine_steps: int = 2,
    veda_dense_tail: bool = False,
) -> list[Any]:
    """Branch sparse attention so the first-step anchor is base-stage only.

    Refinement starts from the generated/upscaled latent at a late sigma, not
    fresh noise. Preserve the dense tail and later model patches, but give the
    refinement sampler independent attention/cache state. Rebuild from the
    attention node's unpatched input rather than stacking two sparse wrappers.
    """
    node = graph.nodes.get(model_ref[0])
    if node is None:
        return model_ref
    inputs = node["inputs"]
    if node["class_type"] == VEDA_ATTENTION_NODE:
        return Graph.out(
            graph.add(
                VEDA_ATTENTION_NODE,
                **{**inputs, "full_attention_steps": (
                    str(max(0, int(refine_steps) - 1)) if veda_dense_tail else ""
                )},
            ),
            model_ref[1],
        )
    if node["class_type"] == SLA_ATTENTION_NODE:
        return Graph.out(
            graph.add(
                SLA_ATTENTION_NODE,
                **{**inputs, "dense_steps": ""},
            ),
            model_ref[1],
        )
    upstream = inputs.get("model")
    if not isinstance(upstream, list) or len(upstream) != 2:
        return model_ref
    replacement = h3_refinement_attention_model(
        graph, upstream, refine_steps=refine_steps, veda_dense_tail=veda_dense_tail
    )
    if replacement == upstream:
        return model_ref
    return Graph.out(
        graph.add(
            node["class_type"],
            **{**inputs, "model": replacement},
        ),
        model_ref[1],
    )


def h3_refinement_lora_model(
    graph: Graph, model_ref: list[Any], *, lora_name: str,
    variant: str, available_nodes: set[str],
) -> list[Any]:
    """Rebuild the model branch with one refinement LoRA on the same base."""
    if variant not in REFINEMENT_LORA_SETTINGS and variant not in H3_REGENERATE_MODELS:
        raise H3Error(f"Unknown refinement LoRA: {variant}")
    node = graph.nodes[model_ref[0]]
    inputs = node["inputs"]
    if node["class_type"] == "UNETLoader":
        if variant in H3_REGENERATE_MODELS:
            return add_regenerate_model(graph, model_ref, variant)
        return add_turbo_model_patch(
            graph, model_ref, lora_name=lora_name, turbo_variant=variant,
            strength=REFINEMENT_LORA_SETTINGS[variant].strength,
            available_nodes=available_nodes,
        )
    upstream = inputs.get("model")
    if not isinstance(upstream, list) or len(upstream) != 2:
        raise H3Error("Cannot locate the base model for the refinement LoRA.")
    replacement = h3_refinement_lora_model(
        graph, upstream, lora_name=lora_name, variant=variant,
        available_nodes=available_nodes,
    )
    # These belong to the old LoRA. The replacement adds its own loader,
    # modulation patch, and (below) sigma shift as appropriate.
    if node["class_type"] in {
        CORE_LORA_LOADER_NODE, LARRY_TURBO_LORA_NODE,
        LIGHTX2V_BYPASS_LORA_NODE, FUSED_MODULATION_NODE, H3_SIGMA_SHIFT_NODE,
    }:
        return replacement
    return Graph.out(graph.add(node["class_type"], **{**inputs, "model": replacement}), model_ref[1])


def add_refinement_model(
    graph: Graph, model_ref: list[Any], *, lora_name: str | None,
    variant: str | None, available_nodes: set[str],
) -> list[Any] | None:
    if lora_name is None:
        return None
    if variant is None:
        raise H3Error("A refinement LoRA requires its variant.")
    model = h3_refinement_lora_model(
        graph, model_ref, lora_name=lora_name, variant=variant,
        available_nodes=available_nodes,
    )
    sigma_shifts = None if variant in H3_REGENERATE_MODELS else turbo_sigma_shifts(variant, lora_name)
    if sigma_shifts is not None:
        model = Graph.out(graph.add(
            H3_SIGMA_SHIFT_NODE, model=model,
            shift_video=sigma_shifts[0], shift_audio=sigma_shifts[1],
        ))
    return model


def finish_sampling(
    graph: Graph,
    *,
    model_ref: list[Any],
    conditioning_ref: list[Any],
    latent_ref: list[Any],
    video_vae_ref: list[Any],
    audio_vae_ref: list[Any],
    seed: int,
    steps: int,
    scheduler: str,
    turbo_variant: str | None,
    filename_prefix: str,
    sampler_name: str = "res_multistep",
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae_ref: list[Any] | None = None,
    single_frame_images: bool = False,
    initial_conditioning_ref: list[Any] | None = None,
    initial_latent_ref: list[Any] | None = None,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_width: int | None = None,
    latent_upscale_height: int | None = None,
    latent_upscale_refine_steps: int = 2,
    refinement_lora_name: str | None = None,
    refinement_variant: str | None = None,
    refinement_model_ref: list[Any] | None = None,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
    conditioning_cache_key: str | None = None,
    veda_dense_tail: bool = False,
) -> None:
    result_format = normalize_result_format(result_format)
    stage_offload_enabled_ref: list[Any] | None = None
    if stage_model_offload:
        if smart_stage_offload:
            if not conditioning_cache_key:
                raise H3Error("Smart H3 stage offload requires a conditioning key.")
            policy = graph.add(
                H3_STAGE_OFFLOAD_POLICY_NODE,
                conditioning=conditioning_ref,
                latent=latent_ref,
                additional_conditioning=initial_conditioning_ref or conditioning_ref,
                additional_latent=initial_latent_ref or latent_ref,
                cache_key=conditioning_cache_key,
            )
            conditioning_ref, latent_ref = Graph.out(policy, 0), Graph.out(policy, 1)
            initial_conditioning_ref = Graph.out(policy, 2)
            initial_latent_ref = Graph.out(policy, 3)
            stage_offload_enabled_ref = Graph.out(policy, 4)
        else:
            (
                conditioning_ref,
                latent_ref,
                initial_conditioning_ref,
                initial_latent_ref,
            ) = add_h3_stage_offload(
                graph,
                conditioning_ref,
                latent_ref,
                initial_conditioning_ref,
                initial_latent_ref,
            )
    if refinement_variant in H3_REGENERATE_MODELS:
        latent_upscale_refine_steps = 3
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    refinement_model_ref = (
        h3_refinement_attention_model(
            graph, refinement_model_ref or model_ref,
            refine_steps=latent_upscale_refine_steps, veda_dense_tail=veda_dense_tail,
        )
        if latent_upscale_model_name is not None
        else model_ref
    )
    if latent_upscale_model_name is not None and latent_split_config is None:
        refinement_guard = graph.add(
            H3_REFINEMENT_COMPILER_GUARD_NODE,
            model=refinement_model_ref,
            min_video_volume=1_500_000,
        )
        refinement_model_ref = Graph.out(refinement_guard)
    guider = graph.add(
        "BasicGuider", model=refinement_model_ref, conditioning=conditioning_ref
    )
    use_larry_sampler = turbo_variant is not None and turbo_uses_custom_nodes(
        turbo_variant
    )
    sampler = (
        graph.add(LARRY_TURBO_SAMPLER_NODE)
        if use_larry_sampler
        else graph.add(CORE_SAMPLER_NODE, sampler_name=sampler_name)
    )
    sigmas = graph.add(
        "BasicScheduler",
        model=model_ref,
        scheduler=scheduler,
        steps=int(steps),
        denoise=1.0,
    )
    if latent_upscale_model_name is not None:
        if initial_conditioning_ref is None or initial_latent_ref is None:
            raise H3Error("H3 latent upscaling requires a low-resolution H3 stage.")
        regenerate = refinement_variant in H3_REGENERATE_MODELS
        if regenerate:
            refine_sigmas = graph.add("ManualSigmas", sigmas=H3_REGENERATE_SIGMAS)
            refine_sigmas_ref = Graph.out(refine_sigmas)
        else:
            # Adapter replacements keep the draft schedule's low-noise tail.
            refine_sigmas = graph.add(
                "SplitSigmas", sigmas=Graph.out(sigmas),
                step=int(steps) - int(latent_upscale_refine_steps),
            )
            refine_sigmas_ref = Graph.out(refine_sigmas, 1)
        refinement_sampler = sampler
        if regenerate:
            refinement_sampler = graph.add(CORE_SAMPLER_NODE, sampler_name="euler")
        elif refinement_lora_name is not None:
            refinement_sampler = (
                graph.add(LARRY_TURBO_SAMPLER_NODE)
                if refinement_variant not in STANDARD_TURBO_LORA_SETTINGS
                and turbo_uses_custom_nodes(refinement_variant)
                else graph.add(
                    CORE_SAMPLER_NODE,
                    sampler_name=turbo_sampler_name(refinement_variant, refinement_lora_name),
                )
            )
        initial_guider = graph.add(
            "BasicGuider",
            model=model_ref,
            conditioning=initial_conditioning_ref,
        )
        initial_sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(initial_guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(sigmas),
            latent_image=initial_latent_ref,
        )
        initial_sampled_ref = Graph.out(initial_sampled)
        if stage_model_offload:
            (
                _unused_conditioning,
                initial_sampled_ref,
                _unused_additional_conditioning,
                _unused_additional_latent,
            ) = add_h3_stage_offload(
                graph,
                conditioning_ref,
                initial_sampled_ref,
                enabled_ref=stage_offload_enabled_ref,
            )
        separated = graph.add(
            H3_SEPARATE_AV_LATENT_NODE,
            latent=initial_sampled_ref,
        )
        upscaled_video = add_h3_latent_upscale(
            graph, Graph.out(separated, 0),
            model_name=latent_upscale_model_name,
            precision=latent_upscale_precision,
            width=latent_upscale_width, height=latent_upscale_height,
        )
        combined = graph.add(
            H3_COMBINE_AV_LATENT_NODE,
            video_latent=upscaled_video,
            audio_latent=Graph.out(separated, 1),
        )
        combined_ref = Graph.out(combined)
        if regenerate:
            combined_ref = Graph.out(graph.add(
                "H3RegeneratePrepareLatent", latent=combined_ref, audio_vae=audio_vae_ref,
            ))
        if stage_model_offload:
            (
                _unused_conditioning,
                combined_ref,
                _unused_additional_conditioning,
                _unused_additional_latent,
            ) = add_h3_stage_offload(
                graph,
                conditioning_ref,
                combined_ref,
                enabled_ref=stage_offload_enabled_ref,
            )
        if latent_split_config is not None:
            temporal_params = graph.add(
                H3_SPLIT_TEMPORAL_PARAMS_NODE,
                chunk_frames=latent_split_config.chunk_frames,
                temporal_overlap_frames=(latent_split_config.temporal_overlap_frames),
                anchor_strength=0.999,
                motion_anchor_frames="22",
                identity_anchor_frames=24,
            )
            spatial_params = graph.add(
                H3_SPLIT_SPATIAL_PARAMS_NODE,
                tile_width=latent_split_config.tile_width,
                tile_height=latent_split_config.tile_height,
                overlap_ratio=latent_split_config.overlap_ratio,
                fade_ratio=latent_split_config.fade_ratio,
                min_tile_size=256,
                seam_denoise=latent_split_config.seam_denoise,
            )
            sampled = graph.add(
                H3_SPLIT_UPSCALE_NODE,
                model=refinement_model_ref,
                conditioning=conditioning_ref,
                latent=combined_ref,
                noise=Graph.out(noise),
                sampler=Graph.out(refinement_sampler),
                sigmas=refine_sigmas_ref,
                cfg=1.0,
                temporal_split_param=Graph.out(temporal_params),
                spatial_split_param=Graph.out(spatial_params),
                seam_polish=latent_split_config.seam_polish,
                color_match=True,
            )
        else:
            sampled = graph.add(
                "SamplerCustomAdvanced",
                noise=Graph.out(noise),
                guider=Graph.out(guider),
                sampler=Graph.out(refinement_sampler),
                sigmas=refine_sigmas_ref,
                latent_image=combined_ref,
            )
        audio_samples = initial_sampled_ref
    else:
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(sigmas),
            latent_image=latent_ref,
        )
        audio_samples = Graph.out(sampled)
    sampled_ref = Graph.out(sampled)
    if stage_model_offload:
        (
            _unused_conditioning,
            sampled_ref,
            _unused_additional_conditioning,
            audio_samples,
        ) = add_h3_stage_offload(
            graph,
            conditioning_ref,
            sampled_ref,
            conditioning_ref,
            audio_samples,
            enabled_ref=stage_offload_enabled_ref,
        )
    if result_format == "Image":
        requested_frames = validate_image_frame_count(image_frames)
        decode_samples = sampled_ref
        if single_frame_images:
            slices = graph.add(
                H3_IMAGE_SLICES_NODE,
                latent=decode_samples,
                frames=requested_frames,
            )
            decode_samples = Graph.out(slices)
        images = graph.add(
            "VAEDecode",
            samples=decode_samples,
            vae=image_vae_ref or video_vae_ref,
        )
        image_ref = Graph.out(images)
        native_frames = image_sampling_length(requested_frames)
        if not single_frame_images and requested_frames != native_frames:
            selected = graph.add(
                "ImageFromBatch",
                image=image_ref,
                batch_index=0,
                length=requested_frames,
            )
            image_ref = Graph.out(selected)
        graph.add(
            "SaveImage",
            images=image_ref,
            filename_prefix=filename_prefix,
        )
        return

    if result_format == "Audio":
        audio = graph.add("VAEDecodeAudio", samples=audio_samples, vae=audio_vae_ref)
        graph.add(
            "SaveAudioMP3",
            audio=Graph.out(audio),
            filename_prefix=filename_prefix,
            quality="V0",
        )
        return

    images = graph.add("VAEDecode", samples=sampled_ref, vae=video_vae_ref)
    audio = graph.add("VAEDecodeAudio", samples=audio_samples, vae=audio_vae_ref)
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(audio),
        fps=24.0,
        bit_depth=8,
    )
    graph.add(
        H3_NVENC_SAVE_NODE,
        video=Graph.out(video),
        filename_prefix=filename_prefix,
        preset="p4",
        constant_quality=23,
    )


@dataclass(frozen=True)
class H3GraphOptions:
    """Everything an H3 graph needs besides its prompt, conditioning media and output naming.

    A new H3 setting becomes one field here; both builders accept it by name.
    """

    width: int
    height: int
    duration: float
    steps: int
    seed: int
    scheduler: str
    turbo_lora_name: str | None
    turbo_variant: str
    turbo_strength: float
    use_sol: bool
    sol_tau: float
    sol_thresh_type: str
    sol_exact_mode: str
    sol_dense_steps: int
    sol_step_off: float
    sol_sink_tokens: int
    cache_mode: str
    fbcache_preset: str
    fbcache_threshold: float
    fbcache_start: float
    fbcache_end: float
    fbcache_max_hits: int
    fbcache_temporal_guard: bool
    easycache_threshold: float
    easycache_start: float
    easycache_end: float
    easycache_verbose: bool
    model_name: str
    models: ModelConfig
    available_nodes: set[str]
    video_encoder: str = "Original"
    lightvae_decoder: str = "none"
    use_lynnreal_vae: bool = False
    use_sol_vae: bool = False
    use_int8_vae: bool = False
    use_trt_vae: bool = False
    use_sage: bool = False
    use_sla: bool = False
    use_veda: bool = False
    sla_preset: str = DEFAULT_SLA_PRESET
    latent_upscale_model_name: str | None = None
    latent_upscale_precision: str = "bf16"
    latent_upscale_refine_steps: int = 2
    refinement_lora_name: str | None = None
    refinement_variant: str | None = None
    latent_split_config: H3SplitUpscaleConfig | None = None
    result_format: str = DEFAULT_RESULT_FORMAT
    image_frames: int = DEFAULT_IMAGE_FRAMES
    image_vae: str = DEFAULT_IMAGE_VAE
    text_encoder_name: str | None = None
    encoder_small_input: bool = False
    reuse_unchanged_inputs: bool = True
    stage_model_offload: bool = False
    smart_stage_offload: bool = False
    # FL2VA only: Ref2VA never applies the bridge.
    semantic_bridge: bool = False
    semantic_bridge_alpha: float = 0.10


FL2VA_ONLY_OPTIONS = frozenset({"semantic_bridge", "semantic_bridge_alpha"})


def _options_signature(mode_parameters, *, exclude=frozenset()):
    """A builder's full keyword signature: its own media parameters, every option, output naming."""
    keyword = inspect.Parameter.KEYWORD_ONLY
    options = [
        inspect.Parameter(
            option.name, keyword, annotation=option.type,
            default=inspect.Parameter.empty if option.default is MISSING else option.default,
        )
        for option in fields(H3GraphOptions)
        if option.name not in exclude
    ]
    naming = [inspect.Parameter("output_stamp", keyword, default="0", annotation=str),
              inspect.Parameter("output_nonce", keyword, default="", annotation=str)]
    return inspect.Signature([*mode_parameters, *options, *naming], return_annotation=dict[str, Any])


def _add_options_model_stack(graph: Graph, options: H3GraphOptions, veda_key: str):
    o = options
    return add_model_stack(
        graph,
        o.model_name,
        o.models,
        turbo_lora_name=o.turbo_lora_name,
        turbo_variant=o.turbo_variant,
        turbo_strength=o.turbo_strength,
        use_sol=o.use_sol,
        sol_tau=o.sol_tau,
        sol_thresh_type=o.sol_thresh_type,
        sol_exact_mode=o.sol_exact_mode,
        sol_dense_steps=o.sol_dense_steps,
        sol_step_off=o.sol_step_off,
        sol_sink_tokens=o.sol_sink_tokens,
        cache_mode=o.cache_mode,
        fbcache_preset=o.fbcache_preset,
        fbcache_threshold=o.fbcache_threshold,
        fbcache_start=o.fbcache_start,
        fbcache_end=o.fbcache_end,
        fbcache_max_hits=o.fbcache_max_hits,
        fbcache_temporal_guard=o.fbcache_temporal_guard,
        easycache_threshold=o.easycache_threshold,
        easycache_start=o.easycache_start,
        easycache_end=o.easycache_end,
        easycache_verbose=o.easycache_verbose,
        available_nodes=o.available_nodes,
        text_encoder_name=o.text_encoder_name,
        use_lynnreal_vae=o.use_lynnreal_vae,
        use_sol_vae=o.use_sol_vae,
        lightvae_decoder=o.lightvae_decoder,
        use_int8_vae=o.use_int8_vae,
        use_trt_vae=o.use_trt_vae,
        use_sage=o.use_sage,
        use_sla=o.use_sla,
        use_veda=o.use_veda,
        sampling_steps=o.steps,
        veda_predictor=MODEL_SPECS[veda_key].local_name,
        sla_preset=o.sla_preset,
    )


def _build_h3_graph(
    task: str,
    prompt: str,
    options: H3GraphOptions,
    *,
    veda_key: str,
    has_visual_conditioning: bool,
    add_conditioning: Callable[[Graph, dict[str, Any], Any], tuple[str, list[tuple[str, str]]]],
    encoder_settings: dict[str, Any],
    semantic_bridge: bool,
    output_stamp: str,
    output_nonce: str,
) -> dict[str, Any]:
    """The graph both H3 tasks share around their conditioning node.

    ``add_conditioning(graph, inputs, audio_vae_ref)`` adds the task's media loaders,
    completes the node ``inputs`` in place, and returns the conditioning node type and
    the media identities for the conditioning cache key.
    """
    o = options
    graph = Graph()
    model_ref, clip_ref, video_vae_ref, audio_vae_ref = _add_options_model_stack(graph, o, veda_key)
    conditioning_vae_ref = h3_conditioning_video_vae(
        graph,
        o.models,
        video_vae_ref,
        use_trt_vae=o.use_trt_vae,
        video_encoder=o.video_encoder,
        available_nodes=o.available_nodes,
        has_visual_conditioning=has_visual_conditioning,
    )
    result_format = normalize_result_format(o.result_format)
    normalized_image_vae = normalize_image_vae(o.image_vae)
    single_frame_images = (
        result_format == "Image" and normalized_image_vae == SINGLE_FRAME_IMAGE_VAE
    )
    image_vae_ref = None
    if single_frame_images:
        if not o.models.image_vae_500k:
            raise H3Error("The 500K single-frame image VAE is not configured.")
        image_loader = graph.add(
            H3_SINGLE_FRAME_VAE_LOADER_NODE,
            base_vae_name=o.models.video_vae,
            decoder_name=o.models.image_vae_500k,
        )
        image_vae_ref = Graph.out(image_loader)

    target_width = snap32(o.width)
    target_height = snap32(o.height)
    source_width, source_height = target_width, target_height
    if o.latent_upscale_model_name is not None:
        source_width, source_height, target_width, target_height = (
            h3_latent_upscale_dimensions(target_width, target_height)
        )

    inputs: dict[str, Any] = {
        "vae": conditioning_vae_ref,
        "prompt": prompt,
        "length": (
            selected_image_sampling_length(o.image_frames, normalized_image_vae)
            if result_format == "Image"
            else frame_length(o.duration)
        ),
    }
    conditioning_node, conditioning_media = add_conditioning(graph, inputs, audio_vae_ref)

    # Tie the native H3 node's upstream CLIP identity to the actual
    # conditioning inputs. This forces changed prompts/media to execute while
    # unchanged conditioning can still reuse the encoded result.
    conditioning_cache_key = h3_conditioning_cache_key(
        task,
        prompt,
        o.text_encoder_name or o.models.text_encoder,
        conditioning_media,
        encoder_settings=encoder_settings,
    )
    cache_node = graph.add(
        H3_CONDITIONING_CACHE_NODE,
        clip=clip_ref,
        cache_key=conditioning_cache_key,
        encoder_small_input=o.encoder_small_input,
        reuse_conditioning=o.reuse_unchanged_inputs,
    )
    inputs["clip"] = Graph.out(cache_node)

    target_h3 = graph.add(
        conditioning_node,
        **inputs,
        width=target_width,
        height=target_height,
    )
    initial_h3 = target_h3
    if o.latent_upscale_model_name is not None:
        initial_h3 = graph.add(
            conditioning_node,
            **inputs,
            width=source_width,
            height=source_height,
        )
    target_conditioning = Graph.out(target_h3, 0)
    initial_conditioning = Graph.out(initial_h3, 0)
    if semantic_bridge and o.semantic_bridge and o.semantic_bridge_alpha != 0:
        if H3_SEMANTIC_BRIDGE_NODE not in o.available_nodes:
            raise H3Error(
                "Missing H3SemanticBridge node. Update provisioning and restart ComfyUI."
            )

        def bridge(conditioning):
            return Graph.out(
                graph.add(
                    H3_SEMANTIC_BRIDGE_NODE,
                    conditioning=conditioning,
                    alpha=o.semantic_bridge_alpha,
                )
            )

        target_conditioning = bridge(target_conditioning)
        initial_conditioning = (
            target_conditioning
            if initial_h3 == target_h3
            else bridge(initial_conditioning)
        )
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=target_conditioning,
        latent_ref=Graph.out(target_h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=o.seed,
        steps=o.steps,
        scheduler=o.scheduler,
        turbo_variant=o.turbo_variant if o.turbo_lora_name else None,
        sampler_name=turbo_sampler_name(o.turbo_variant, o.turbo_lora_name),
        filename_prefix=(
            f"h3/image_staging/{task}_{output_stamp}_{output_nonce}"
            if result_format == "Image"
            else f"audio/h3_{task}_{output_stamp}"
            if result_format == "Audio"
            else f"h3/{task}_{output_stamp}"
        ),
        result_format=o.result_format,
        image_frames=o.image_frames,
        image_vae_ref=image_vae_ref,
        single_frame_images=single_frame_images,
        initial_conditioning_ref=initial_conditioning,
        initial_latent_ref=Graph.out(initial_h3, 1),
        latent_upscale_model_name=o.latent_upscale_model_name,
        latent_upscale_precision=o.latent_upscale_precision,
        latent_upscale_width=target_width,
        latent_upscale_height=target_height,
        latent_upscale_refine_steps=o.latent_upscale_refine_steps,
        refinement_lora_name=o.refinement_lora_name,
        refinement_variant=o.refinement_variant,
        veda_dense_tail=o.use_veda and resolve_sla_preset(o.sla_preset)[0] == "Quality",
        refinement_model_ref=(
            add_refinement_model(
                graph, model_ref, lora_name=o.refinement_lora_name,
                variant=o.refinement_variant, available_nodes=o.available_nodes,
            ) if o.latent_upscale_model_name is not None else None
        ),
        latent_split_config=o.latent_split_config,
        stage_model_offload=o.stage_model_offload,
        smart_stage_offload=o.smart_stage_offload,
        conditioning_cache_key=conditioning_cache_key,
    )
    return graph.nodes


def build_fl2va_graph(
    *,
    prompt: str,
    first_image: str | None,
    last_image: str | None,
    voice_reference_audios: list[str] | None = None,
    output_stamp: str = "0",
    output_nonce: str = "",
    **settings: Any,
) -> dict[str, Any]:
    """Text, first/last-frame and FL2VA voice-reference generation. ``settings`` are :class:`H3GraphOptions`."""
    options = H3GraphOptions(**settings)

    def add_conditioning(graph, inputs, audio_vae_ref):
        conditioning_media: list[tuple[str, str]] = []
        if first_image:
            conditioning_media.append(("first_image", first_image))
            inputs["first_frame"] = Graph.out(graph.add("LoadImage", image=first_image))
        if last_image:
            conditioning_media.append(("last_image", last_image))
            inputs["last_frame"] = Graph.out(graph.add("LoadImage", image=last_image))
        if not voice_reference_audios:
            return "MiniMaxH3ImageToVideo", conditioning_media
        if not (first_image or last_image):
            raise H3Error("FL2VA voice references require a first or last frame.")
        if len(voice_reference_audios) > MAX_REFERENCE_AUDIOS:
            raise H3Error("FL2VA supports up to three voice references.")
        conditioning_node = "MiniMaxH3AudioConditioningT8"
        missing = {conditioning_node, "LoadAudio"} - options.available_nodes
        if missing:
            raise H3Error(
                "Missing FL2VA voice-reference nodes: "
                + ", ".join(sorted(missing))
                + ". Update provisioning and restart ComfyUI."
            )
        inputs["video_vae"] = inputs.pop("vae")
        inputs.update(
            audio_vae=audio_vae_ref,
            task_type="Hybrid",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=False,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
        )
        for index, path in enumerate(voice_reference_audios, 1):
            conditioning_media.append((f"fl2va_audio_{index}", path))
            inputs[f"ref_audios.ref_audio_{index}"] = Graph.out(graph.add("LoadAudio", audio=path))
        return conditioning_node, conditioning_media

    return _build_h3_graph(
        "fl2va", prompt, options,
        veda_key="veda_t2va",
        has_visual_conditioning=bool(first_image or last_image),
        add_conditioning=add_conditioning,
        encoder_settings={
            "encoder_small_input": options.encoder_small_input,
            **({"video_encoder": options.video_encoder} if options.video_encoder != "Original" else {}),
        },
        semantic_bridge=True,
        output_stamp=output_stamp,
        output_nonce=output_nonce,
    )


build_fl2va_graph.__signature__ = _options_signature([
    inspect.Parameter("prompt", inspect.Parameter.KEYWORD_ONLY, annotation=str),
    inspect.Parameter("first_image", inspect.Parameter.KEYWORD_ONLY, annotation=str | None),
    inspect.Parameter("last_image", inspect.Parameter.KEYWORD_ONLY, annotation=str | None),
    inspect.Parameter("voice_reference_audios", inspect.Parameter.KEYWORD_ONLY, default=None,
                      annotation=list[str] | None),
])


def build_ref2va_graph(
    *,
    prompt: str,
    reference_images: list[str],
    reference_videos: list[str],
    reference_audios: list[str],
    ref_image_size: str,
    output_stamp: str = "0",
    output_nonce: str = "",
    **settings: Any,
) -> dict[str, Any]:
    """Reference-media generation. ``settings`` are :class:`H3GraphOptions` except the FL2VA-only ones."""
    unexpected = FL2VA_ONLY_OPTIONS & set(settings)
    if unexpected:
        raise TypeError(f"build_ref2va_graph() got unexpected keyword arguments: {', '.join(sorted(unexpected))}")
    options = H3GraphOptions(**settings)

    def add_conditioning(graph, inputs, audio_vae_ref):
        inputs.update(audio_vae=audio_vae_ref, ref_image_size=ref_image_size)
        conditioning_media: list[tuple[str, str]] = []
        for index, path in enumerate(reference_images[:MAX_REFERENCE_IMAGES]):
            conditioning_media.append((f"reference_image_{index}", path))
            inputs[f"ref_images.ref_image_{index}"] = Graph.out(graph.add("LoadImage", image=path))
        for index, path in enumerate(reference_videos[:MAX_REFERENCE_VIDEOS]):
            conditioning_media.append((f"reference_video_{index}", path))
            loaded = graph.add("LoadVideo", file=path)
            components = graph.add("GetVideoComponents", video=Graph.out(loaded))
            inputs[f"ref_videos.ref_video_{index}"] = Graph.out(components, 0)
            inputs[f"ref_video_audios.ref_video_audio_{index}"] = Graph.out(components, 1)
        for index, path in enumerate(reference_audios[:MAX_REFERENCE_AUDIOS]):
            conditioning_media.append((f"reference_audio_{index}", path))
            inputs[f"ref_audios.ref_audio_{index}"] = Graph.out(graph.add("LoadAudio", audio=path))
        return "MiniMaxH3ReferenceToVideo", conditioning_media

    return _build_h3_graph(
        "ref2va", prompt, options,
        veda_key="veda_r2va",
        has_visual_conditioning=bool(reference_images or reference_videos),
        add_conditioning=add_conditioning,
        encoder_settings={
            **({"video_encoder": options.video_encoder} if options.video_encoder != "Original" else {}),
            "ref_image_size": ref_image_size,
            "encoder_small_input": options.encoder_small_input,
        },
        semantic_bridge=False,
        output_stamp=output_stamp,
        output_nonce=output_nonce,
    )


build_ref2va_graph.__signature__ = _options_signature([
    inspect.Parameter("prompt", inspect.Parameter.KEYWORD_ONLY, annotation=str),
    inspect.Parameter("reference_images", inspect.Parameter.KEYWORD_ONLY, annotation=list[str]),
    inspect.Parameter("reference_videos", inspect.Parameter.KEYWORD_ONLY, annotation=list[str]),
    inspect.Parameter("reference_audios", inspect.Parameter.KEYWORD_ONLY, annotation=list[str]),
    inspect.Parameter("ref_image_size", inspect.Parameter.KEYWORD_ONLY, annotation=str),
], exclude=FL2VA_ONLY_OPTIONS)


def required_nodes_for(
    mode: str,
    use_sol: bool,
    cache_mode: str,
    use_turbo: bool = False,
    turbo_variant: str = LIGHTX2V_4STEP_TURBO,
    model_filename: str = "",
    turbo_lora_filename: str = "",
    use_sage: bool = False,
    use_sla: bool = False,
    use_veda: bool = False,
    latent_upscale: bool = False,
    latent_upscale_method: str = H3_LATENT_UPSCALE_STANDARD,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_vae: str = DEFAULT_IMAGE_VAE,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
) -> set[str]:
    result_format = normalize_result_format(result_format)
    common = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "RandomNoise",
        "BasicGuider",
        "BasicScheduler",
        "SamplerCustomAdvanced",
    }
    if result_format == "Image":
        common |= {"VAEDecode", "SaveImage"}
        if normalize_image_vae(image_vae) == SINGLE_FRAME_IMAGE_VAE:
            common |= {
                H3_SINGLE_FRAME_VAE_LOADER_NODE,
                H3_IMAGE_SLICES_NODE,
            }
        else:
            common.add("ImageFromBatch")
    elif result_format == "Audio":
        common |= {"VAEDecodeAudio", "SaveAudioMP3"}
    else:
        common |= {
            "VAEDecode",
            "VAEDecodeAudio",
            "CreateVideo",
            H3_NVENC_SAVE_NODE,
        }
    if mode == "Reference media":
        common |= {
            "MiniMaxH3ReferenceToVideo",
            "LoadImage",
            "LoadVideo",
            "GetVideoComponents",
            "LoadAudio",
        }
    else:
        common |= {"MiniMaxH3ImageToVideo", "LoadImage"}
    if use_turbo:
        common |= turbo_required_nodes(turbo_variant, turbo_lora_filename)
    else:
        common.add(CORE_SAMPLER_NODE)
    if use_sol:
        common.add(SOL_ATTENTION_NODE)
    if use_sage:
        common.add(SAGE_ATTENTION_NODE)
    if use_sla:
        common.add(SLA_ATTENTION_NODE)
    if use_veda:
        common.add(VEDA_ATTENTION_NODE)
    if latent_upscale:
        common |= {
            "SplitSigmas",
            H3_LATENT_UPSCALER_NODE,
            H3_LATENT_UPSCALER_LOADER_NODE,
            H3_REFINEMENT_COMPILER_GUARD_NODE,
            H3_SEPARATE_AV_LATENT_NODE,
            H3_COMBINE_AV_LATENT_NODE,
        }
        if latent_upscale_method in H3_REGENERATE_MODELS:
            common |= turbo_required_nodes(latent_upscale_method)
        if (
            resolve_h3_latent_upscale_method(latent_upscale_method)
            == H3_LATENT_UPSCALE_SPLIT
        ):
            common |= {
                H3_SPLIT_TEMPORAL_PARAMS_NODE,
                H3_SPLIT_SPATIAL_PARAMS_NODE,
                H3_SPLIT_UPSCALE_NODE,
            }
    if stage_model_offload:
        common.add(H3_STAGE_OFFLOAD_NODE)
        if smart_stage_offload:
            common |= {
                H3_CONDITIONING_CACHE_NODE,
                H3_STAGE_OFFLOAD_POLICY_NODE,
            }
    if "convrot" in model_filename.lower():
        common.add(CHUNK_FEED_FORWARD_NODE)
    if str(cache_mode).strip().lower() == "firstblockcache":
        common.add("H3FirstBlockCache")
    elif str(cache_mode).strip().lower() == "spectrum":
        common.add("SpectrumApplyMiniMaxH3")
    elif str(cache_mode).strip().lower() == "easycache":
        common.add("EasyCache")
    return common
