"""Extracted h3 boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from h3_app.catalog import (
    AUTO_SOL_TOKEN_THRESHOLD,
    CHUNK_FEED_FORWARD_NODE,
    CORE_SAMPLER_NODE,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_IMAGE_VAE,
    DEFAULT_RESULT_FORMAT,
    DEFAULT_SLA_PRESET,
    FUSED_MODULATION_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_CONDITIONING_CACHE_NODE,
    H3_IMAGE_SLICES_NODE,
    H3_LATENT_UPSCALE_SCALE,
    H3_LATENT_UPSCALE_SPLIT,
    H3_LATENT_UPSCALE_STANDARD,
    H3_LATENT_UPSCALER_NODE,
    H3_NVENC_SAVE_NODE,
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
    SOL_ATTENTION_NODE,
    SPECTRUM_DEFAULT_INPUTS,
)
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig, trt_vae_engine_name
from h3_app.policy import (
    H3SplitUpscaleConfig,
    frame_length,
    h3_latent_upscale_dimensions,
    image_sampling_length,
    lightx2v_uses_768p_schedule,
    normalize_image_vae,
    normalize_result_format,
    normalize_turbo_variant,
    resolve_h3_latent_upscale_method,
    resolve_sla_preset,
    selected_image_sampling_length,
    snap32,
    turbo_sampler_name,
    turbo_uses_custom_nodes,
    validate_image_frame_count,
)


def turbo_required_nodes(
    turbo_variant: str,
    lora_filename: str = "",
) -> set[str]:
    """Return the external node contract for one normalized Turbo variant."""
    if turbo_uses_custom_nodes(turbo_variant):
        return {LARRY_TURBO_LORA_NODE, LARRY_TURBO_SAMPLER_NODE}
    required = {
        LIGHTX2V_BYPASS_LORA_NODE,
        CORE_SAMPLER_NODE,
        FUSED_MODULATION_NODE,
    }
    if lightx2v_uses_768p_schedule(turbo_variant, lora_filename):
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
    variant = normalize_turbo_variant(turbo_variant)
    required = turbo_required_nodes(variant, lora_name)
    missing = required - available_nodes
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise H3Error(
            f"{variant} Turbo requires unavailable nodes: {missing_names}. "
            "Re-run setup_h3.py and restart ComfyUI."
        )

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
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
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
            # Preserve the pre-1.4.8 sparse path and first-step anchor.
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

    if turbo_lora_name and lightx2v_uses_768p_schedule(turbo_variant, turbo_lora_name):
        if H3_SIGMA_SHIFT_NODE not in available_nodes:
            raise H3Error(
                "LightX2V 768p Turbo requires MiniMaxH3SigmaShift. "
                "Update ComfyUI and restart the service."
            )
        shifted = graph.add(
            H3_SIGMA_SHIFT_NODE,
            model=model_ref,
            shift_video=6.0,
            shift_audio=3.0,
        )
        model_ref = Graph.out(shifted)

    clip = graph.add(
        "CLIPLoader",
        clip_name=text_encoder_name or models.text_encoder,
        type="minimax",
        device="default",
    )
    if use_int8_vae and use_trt_vae:
        raise H3Error("Select either INT8 ConvRot VAE or TensorRT VAE, not both.")
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
        if use_int8_vae:
            if not models.video_vae_int8:
                raise H3Error("INT8 video VAE is missing from the model catalog.")
            video_vae_name = models.video_vae_int8
        else:
            video_vae_name = models.video_vae
        video_vae = graph.add("VAELoader", vae_name=video_vae_name)
    audio_vae = graph.add("VAELoader", vae_name=models.audio_vae)
    return model_ref, Graph.out(clip), Graph.out(video_vae), Graph.out(audio_vae)


def h3_conditioning_video_vae(
    graph: Graph,
    models: ModelConfig,
    decode_vae_ref: list[Any],
    *,
    use_trt_vae: bool,
    has_visual_conditioning: bool,
) -> list[Any]:
    """Use the reference VAE encoder for visual conditioning.

    Repeating one frame to satisfy the TensorRT encoder's fixed 17-frame
    profile changes its causal latent. TensorRT remains the final decoder.
    """
    if not (use_trt_vae and has_visual_conditioning):
        return decode_vae_ref
    encoder_vae = graph.add("VAELoader", vae_name=models.video_vae)
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


def h3_refinement_attention_model(graph: Graph, model_ref: list[Any]) -> list[Any]:
    """Branch the model patch chain so SLA's first-step anchor is base-stage only.

    Refinement starts from the generated/upscaled latent at a late sigma, not
    fresh noise. Preserve the selected SLA preset's dense tail and all later
    model patches, but give the refinement sampler independent SLA/cache state.
    Rebuild from SLA's unpatched input rather than stacking two SLA wrappers.
    """
    node = graph.nodes.get(model_ref[0])
    if node is None:
        return model_ref
    inputs = node["inputs"]
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
    replacement = h3_refinement_attention_model(graph, upstream)
    if replacement == upstream:
        return model_ref
    return Graph.out(
        graph.add(
            node["class_type"],
            **{**inputs, "model": replacement},
        ),
        model_ref[1],
    )


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
    latent_upscale_refine_steps: int = 2,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
    conditioning_cache_key: str | None = None,
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
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    refinement_model_ref = (
        h3_refinement_attention_model(graph, model_ref)
        if latent_upscale_model_name is not None
        else model_ref
    )
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
        refine_sigmas = graph.add(
            "SplitSigmas",
            sigmas=Graph.out(sigmas),
            step=int(steps) - int(latent_upscale_refine_steps),
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
        upscaled_video = graph.add(
            H3_LATENT_UPSCALER_NODE,
            latent=Graph.out(separated, 0),
            model_name=latent_upscale_model_name,
            mode="scale by multiplier",
            align=32,
            enable_temporal_chunking=True,
            force_unload=True,
            device="cuda",
            precision=latent_upscale_precision,
            # ComfyUI's prompt API flattens DynamicCombo children with dotted keys.
            **{"mode.scale": H3_LATENT_UPSCALE_SCALE},
        )
        combined = graph.add(
            H3_COMBINE_AV_LATENT_NODE,
            video_latent=Graph.out(upscaled_video),
            audio_latent=Graph.out(separated, 1),
        )
        combined_ref = Graph.out(combined)
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
                sampler=Graph.out(sampler),
                sigmas=Graph.out(refine_sigmas, 1),
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
                sampler=Graph.out(sampler),
                sigmas=Graph.out(refine_sigmas, 1),
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


def build_fl2va_graph(
    *,
    prompt: str,
    first_image: str | None,
    last_image: str | None,
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
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
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    reuse_unchanged_inputs: bool = True,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
    semantic_bridge: bool = False,
    semantic_bridge_alpha: float = 0.10,
    voice_reference_audios: list[str] | None = None,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    graph = Graph()
    model_ref, clip_ref, video_vae_ref, audio_vae_ref = add_model_stack(
        graph,
        model_name,
        models,
        turbo_lora_name=turbo_lora_name,
        turbo_variant=turbo_variant,
        turbo_strength=turbo_strength,
        use_sol=use_sol,
        sol_tau=sol_tau,
        sol_thresh_type=sol_thresh_type,
        sol_exact_mode=sol_exact_mode,
        sol_dense_steps=sol_dense_steps,
        sol_step_off=sol_step_off,
        sol_sink_tokens=sol_sink_tokens,
        cache_mode=cache_mode,
        fbcache_preset=fbcache_preset,
        fbcache_threshold=fbcache_threshold,
        fbcache_start=fbcache_start,
        fbcache_end=fbcache_end,
        fbcache_max_hits=fbcache_max_hits,
        fbcache_temporal_guard=fbcache_temporal_guard,
        easycache_threshold=easycache_threshold,
        easycache_start=easycache_start,
        easycache_end=easycache_end,
        easycache_verbose=easycache_verbose,
        available_nodes=available_nodes,
        text_encoder_name=text_encoder_name,
        use_int8_vae=use_int8_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
    )
    conditioning_vae_ref = h3_conditioning_video_vae(
        graph,
        models,
        video_vae_ref,
        use_trt_vae=use_trt_vae,
        has_visual_conditioning=bool(first_image or last_image),
    )
    normalized_image_vae = normalize_image_vae(image_vae)
    single_frame_images = (
        normalize_result_format(result_format) == "Image"
        and normalized_image_vae == SINGLE_FRAME_IMAGE_VAE
    )
    image_vae_ref = None
    if single_frame_images:
        if not models.image_vae_500k:
            raise H3Error("The 500K single-frame image VAE is not configured.")
        image_loader = graph.add(
            H3_SINGLE_FRAME_VAE_LOADER_NODE,
            base_vae_name=models.video_vae,
            decoder_name=models.image_vae_500k,
        )
        image_vae_ref = Graph.out(image_loader)

    target_width = snap32(width)
    target_height = snap32(height)
    source_width, source_height = target_width, target_height
    if latent_upscale_model_name is not None:
        source_width, source_height, target_width, target_height = (
            h3_latent_upscale_dimensions(target_width, target_height)
        )

    conditioning_media: list[tuple[str, str]] = []
    inputs: dict[str, Any] = {
        "vae": conditioning_vae_ref,
        "prompt": prompt,
        "length": (
            selected_image_sampling_length(image_frames, normalized_image_vae)
            if normalize_result_format(result_format) == "Image"
            else frame_length(duration)
        ),
    }
    if first_image:
        staged = first_image
        conditioning_media.append(("first_image", staged))
        loaded = graph.add(
            "LoadImage",
            image=staged,
        )
        inputs["first_frame"] = Graph.out(loaded)
    if last_image:
        staged = last_image
        conditioning_media.append(("last_image", staged))
        loaded = graph.add(
            "LoadImage",
            image=staged,
        )
        inputs["last_frame"] = Graph.out(loaded)

    conditioning_node = "MiniMaxH3ImageToVideo"
    if voice_reference_audios:
        if not (first_image or last_image):
            raise H3Error("FL2VA voice references require a first or last frame.")
        if len(voice_reference_audios) > MAX_REFERENCE_AUDIOS:
            raise H3Error("FL2VA supports up to three voice references.")
        conditioning_node = "MiniMaxH3AudioConditioningT8"
        missing = {conditioning_node, "LoadAudio"} - available_nodes
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
            staged = path
            conditioning_media.append((f"fl2va_audio_{index}", staged))
            loaded = graph.add("LoadAudio", audio=staged)
            inputs[f"ref_audios.ref_audio_{index}"] = Graph.out(loaded)

    # Tie the native H3 node's upstream CLIP identity to the actual
    # conditioning inputs. This forces changed prompts/media to execute while
    # unchanged conditioning can still reuse the encoded result.
    conditioning_cache_key = h3_conditioning_cache_key(
        "fl2va",
        prompt,
        text_encoder_name or models.text_encoder,
        conditioning_media,
    )
    cache_node = graph.add(
        H3_CONDITIONING_CACHE_NODE,
        clip=clip_ref,
        cache_key=conditioning_cache_key,
    )
    clip_ref = Graph.out(cache_node)
    inputs["clip"] = clip_ref

    target_h3 = graph.add(
        conditioning_node,
        **inputs,
        width=target_width,
        height=target_height,
    )
    initial_h3 = target_h3
    if latent_upscale_model_name is not None:
        initial_h3 = graph.add(
            conditioning_node,
            **inputs,
            width=source_width,
            height=source_height,
        )
    target_conditioning = Graph.out(target_h3, 0)
    initial_conditioning = Graph.out(initial_h3, 0)
    if semantic_bridge and semantic_bridge_alpha != 0:
        if H3_SEMANTIC_BRIDGE_NODE not in available_nodes:
            raise H3Error(
                "Missing H3SemanticBridge node. Update provisioning and restart ComfyUI."
            )

        def bridge(conditioning):
            return Graph.out(
                graph.add(
                    H3_SEMANTIC_BRIDGE_NODE,
                    conditioning=conditioning,
                    alpha=semantic_bridge_alpha,
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
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        sampler_name=turbo_sampler_name(turbo_variant, turbo_lora_name),
        filename_prefix=(
            f"h3/image_staging/fl2va_{output_stamp}_{output_nonce}"
            if normalize_result_format(result_format) == "Image"
            else f"audio/h3_fl2va_{output_stamp}"
            if normalize_result_format(result_format) == "Audio"
            else f"h3/fl2va_{output_stamp}"
        ),
        result_format=result_format,
        image_frames=image_frames,
        image_vae_ref=image_vae_ref,
        single_frame_images=single_frame_images,
        initial_conditioning_ref=initial_conditioning,
        initial_latent_ref=Graph.out(initial_h3, 1),
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_split_config=latent_split_config,
        stage_model_offload=stage_model_offload,
        smart_stage_offload=smart_stage_offload,
        conditioning_cache_key=conditioning_cache_key,
    )
    return graph.nodes


def build_ref2va_graph(
    *,
    prompt: str,
    reference_images: list[str],
    reference_videos: list[str],
    reference_audios: list[str],
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    ref_image_size: str,
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
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    smart_stage_offload: bool = False,
    reuse_unchanged_inputs: bool = True,
    stage_model_offload: bool = False,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    graph = Graph()
    model_ref, clip_ref, video_vae_ref, audio_vae_ref = add_model_stack(
        graph,
        model_name,
        models,
        turbo_lora_name=turbo_lora_name,
        turbo_variant=turbo_variant,
        turbo_strength=turbo_strength,
        use_sol=use_sol,
        sol_tau=sol_tau,
        sol_thresh_type=sol_thresh_type,
        sol_exact_mode=sol_exact_mode,
        sol_dense_steps=sol_dense_steps,
        sol_step_off=sol_step_off,
        sol_sink_tokens=sol_sink_tokens,
        cache_mode=cache_mode,
        fbcache_preset=fbcache_preset,
        fbcache_threshold=fbcache_threshold,
        fbcache_start=fbcache_start,
        fbcache_end=fbcache_end,
        fbcache_max_hits=fbcache_max_hits,
        fbcache_temporal_guard=fbcache_temporal_guard,
        easycache_threshold=easycache_threshold,
        easycache_start=easycache_start,
        easycache_end=easycache_end,
        easycache_verbose=easycache_verbose,
        available_nodes=available_nodes,
        text_encoder_name=text_encoder_name,
        use_int8_vae=use_int8_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
    )
    conditioning_vae_ref = h3_conditioning_video_vae(
        graph,
        models,
        video_vae_ref,
        use_trt_vae=use_trt_vae,
        has_visual_conditioning=bool(reference_images or reference_videos),
    )
    normalized_image_vae = normalize_image_vae(image_vae)
    single_frame_images = (
        normalize_result_format(result_format) == "Image"
        and normalized_image_vae == SINGLE_FRAME_IMAGE_VAE
    )
    image_vae_ref = None
    if single_frame_images:
        if not models.image_vae_500k:
            raise H3Error("The 500K single-frame image VAE is not configured.")
        image_loader = graph.add(
            H3_SINGLE_FRAME_VAE_LOADER_NODE,
            base_vae_name=models.video_vae,
            decoder_name=models.image_vae_500k,
        )
        image_vae_ref = Graph.out(image_loader)

    target_width = snap32(width)
    target_height = snap32(height)
    source_width, source_height = target_width, target_height
    if latent_upscale_model_name is not None:
        source_width, source_height, target_width, target_height = (
            h3_latent_upscale_dimensions(target_width, target_height)
        )

    conditioning_media: list[tuple[str, str]] = []
    inputs: dict[str, Any] = {
        "vae": conditioning_vae_ref,
        "audio_vae": audio_vae_ref,
        "prompt": prompt,
        "length": (
            selected_image_sampling_length(image_frames, normalized_image_vae)
            if normalize_result_format(result_format) == "Image"
            else frame_length(duration)
        ),
        "ref_image_size": ref_image_size,
    }

    for index, path in enumerate(reference_images[:MAX_REFERENCE_IMAGES]):
        staged = path
        conditioning_media.append((f"reference_image_{index}", staged))
        loaded = graph.add(
            "LoadImage",
            image=staged,
        )
        inputs[f"ref_images.ref_image_{index}"] = Graph.out(loaded)

    for index, path in enumerate(reference_videos[:MAX_REFERENCE_VIDEOS]):
        staged = path
        conditioning_media.append((f"reference_video_{index}", staged))
        loaded = graph.add("LoadVideo", file=staged)
        components = graph.add("GetVideoComponents", video=Graph.out(loaded))
        inputs[f"ref_videos.ref_video_{index}"] = Graph.out(components, 0)
        inputs[f"ref_video_audios.ref_video_audio_{index}"] = Graph.out(components, 1)

    for index, path in enumerate(reference_audios[:MAX_REFERENCE_AUDIOS]):
        staged = path
        conditioning_media.append((f"reference_audio_{index}", staged))
        loaded = graph.add(
            "LoadAudio",
            audio=staged,
        )
        inputs[f"ref_audios.ref_audio_{index}"] = Graph.out(loaded)

    conditioning_cache_key = h3_conditioning_cache_key(
        "ref2va",
        prompt,
        text_encoder_name or models.text_encoder,
        conditioning_media,
        encoder_settings={"ref_image_size": ref_image_size},
    )
    cache_node = graph.add(
        H3_CONDITIONING_CACHE_NODE,
        clip=clip_ref,
        cache_key=conditioning_cache_key,
    )
    clip_ref = Graph.out(cache_node)
    inputs["clip"] = clip_ref

    target_h3 = graph.add(
        "MiniMaxH3ReferenceToVideo",
        **inputs,
        width=target_width,
        height=target_height,
    )
    initial_h3 = target_h3
    if latent_upscale_model_name is not None:
        initial_h3 = graph.add(
            "MiniMaxH3ReferenceToVideo",
            **inputs,
            width=source_width,
            height=source_height,
        )
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=Graph.out(target_h3, 0),
        latent_ref=Graph.out(target_h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        sampler_name=turbo_sampler_name(turbo_variant, turbo_lora_name),
        filename_prefix=(
            f"h3/image_staging/ref2va_{output_stamp}_{output_nonce}"
            if normalize_result_format(result_format) == "Image"
            else f"audio/h3_ref2va_{output_stamp}"
            if normalize_result_format(result_format) == "Audio"
            else f"h3/ref2va_{output_stamp}"
        ),
        result_format=result_format,
        image_frames=image_frames,
        image_vae_ref=image_vae_ref,
        single_frame_images=single_frame_images,
        initial_conditioning_ref=Graph.out(initial_h3, 0),
        initial_latent_ref=Graph.out(initial_h3, 1),
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_split_config=latent_split_config,
        stage_model_offload=stage_model_offload,
        smart_stage_offload=smart_stage_offload,
        conditioning_cache_key=conditioning_cache_key,
    )
    return graph.nodes


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
    if latent_upscale:
        common |= {
            "SplitSigmas",
            H3_LATENT_UPSCALER_NODE,
            H3_SEPARATE_AV_LATENT_NODE,
            H3_COMBINE_AV_LATENT_NODE,
        }
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
