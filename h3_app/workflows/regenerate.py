"""Shared Open H3 Regenerate adapter stack and existing-video graph."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import (
    AUTO_SOL_TOKEN_THRESHOLD,
    CHUNK_FEED_FORWARD_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_LATENT_UPSCALER_NODE,
    H3_LATENT_UPSCALER_LOADER_NODE,
    H3_REFINEMENT_COMPILER_GUARD_NODE,
    H3_REFINEMENT, H3_CONDITIONING_CACHE_NODE, H3_SEMANTIC_BRIDGE_NODE, H3_NVENC_SAVE_NODE, H3_STAGE_OFFLOAD_POLICY_NODE,
    LARRY_TURBO_SAMPLER_NODE, PIKA_FLEX, PIKA_FLEX_SIGMAS,
    H3_REGENERATE_MODELS,
    H3_REGENERATE_SIGMAS,
    H3_SEPARATE_AV_LATENT_NODE,
)
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig
from h3_app.policy import snap32, turbo_sampler_name, turbo_uses_custom_nodes
from h3_app.refinement import resolve_refinement
from h3_models import MODEL_SPECS
from h3_app.workflows.latent_upscale import add_h3_latent_upscale


def add_regenerate_model(graph: Graph, model_ref: list[Any], option: str) -> list[Any]:
    """Kijai's files are refiner-only; use one explicit FL2VA Turbo companion.

    LMS Sharp is deliberately absent: this is an experimental refiner integration,
    not a reproduction of the upstream three-adapter merged bundle.
    """
    for key in ("turbo_lora", H3_REGENERATE_MODELS[option]):
        loaded = graph.add(
            "LoraLoaderModelOnly", model=model_ref,
            lora_name=MODEL_SPECS[key].local_name, strength_model=1.0,
        )
        model_ref = Graph.out(loaded)
    return model_ref


def required_h3_regenerate_nodes(models: ModelConfig | None = None, refinement=None) -> set[str]:
    if refinement is not None and models is not None:
        graph = build_h3_regenerate_graph(
            option=H3_REFINEMENT, source_video="required-nodes.mp4", seed=0, models=models,
            target_width=refinement.get("_target_width", 1920),
            target_height=refinement.get("_target_height", 1088), refinement=refinement,
        )
        return {node["class_type"] for node in graph.values()}
    required = {
        "LoadVideo", "GetVideoComponents", "ImageScale", "UNETLoader",
        "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
        "H3RegenerateEncodeVideo", "H3RegeneratePrepareLatent",
        H3_LATENT_UPSCALER_NODE, H3_LATENT_UPSCALER_LOADER_NODE,
        H3_SEPARATE_AV_LATENT_NODE, H3_COMBINE_AV_LATENT_NODE,
        "BasicGuider", "RandomNoise", "VAEDecodeTiled", "ImageFromBatch", "CreateVideo", "SaveVideo",
        H3_REFINEMENT_COMPILER_GUARD_NODE,
    }
    required |= {"LoraLoaderModelOnly", "KSamplerSelect", "ManualSigmas", "SamplerCustomAdvanced"}
    profile = models.default_profile if models else None
    if models and "convrot" in models.profile(profile).fl2va.lower():
        required.add(CHUNK_FEED_FORWARD_NODE)
    return required


def build_h3_regenerate_graph(
    *, option: str, source_video: str, seed: int, models: ModelConfig,
    target_width: int, target_height: int, prompt: str = "", fps: float = 24.0,
    output_stamp: str = "0", output_nonce: str = "", refinement=None,
) -> dict[str, Any]:
    """Existing video -> H3 encode -> learned 2x -> refiner; retain source timing/audio."""
    if option not in H3_REGENERATE_MODELS and option != H3_REFINEMENT:
        raise H3Error(f"Unknown H3 Regenerate option: {option}")
    config = resolve_refinement(refinement, models) if option == H3_REFINEMENT else None
    variant = config["variant"] if config else option
    base_width, base_height = snap32(target_width / 2), snap32(target_height / 2)
    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    resized = graph.add(
        "ImageScale", image=Graph.out(components), upscale_method="lanczos",
        width=base_width, height=base_height, crop="disabled",
    )
    if config:
        from .refinement import add_refinement_stack
        model, clip_ref, vae_ref, decoder_ref, audio_vae_ref, schedule_model = add_refinement_stack(
            graph, models, config, base_width * 2, base_height * 2,
        )
    else:
        vae_ref = Graph.out(graph.add("VAELoader", vae_name=models.video_vae))
        decoder_ref = vae_ref
        audio_vae_ref = Graph.out(graph.add("VAELoader", vae_name=models.audio_vae))
        clip_ref = Graph.out(graph.add("CLIPLoader", clip_name=models.text_encoder, type="minimax", device="default"))
    encoded = graph.add("H3RegenerateEncodeVideo", images=Graph.out(resized), vae=vae_ref)
    if config:
        from .h3 import h3_conditioning_cache_key
        cache_key = h3_conditioning_cache_key(
            "refinement", prompt, config["text_encoder"], [("source", source_video)],
            encoder_settings={"encoder_small_input": config["settings"]["encoder_small_input"],
                              "video_encoder": config["settings"]["video_encoder"]},
        )
        clip_ref = Graph.out(graph.add(H3_CONDITIONING_CACHE_NODE, clip=clip_ref, cache_key=cache_key,
            encoder_small_input=config["settings"]["encoder_small_input"],
            reuse_conditioning=config["settings"]["reuse_unchanged_inputs"]))
    conditioning_inputs = dict(clip=clip_ref, vae=vae_ref,
        prompt=prompt if config else (prompt.strip() or "Natural fine detail, consistent subjects, motion, lighting and color."),
        width=base_width * 2, height=base_height * 2, length=Graph.out(encoded, 1))
    reference = config and config["settings"]["mode"] == "Reference media"
    if reference:
        conditioning_inputs.update(audio_vae=audio_vae_ref, ref_image_size=config["settings"]["ref_image_size"])
    conditioning = graph.add("MiniMaxH3ReferenceToVideo" if reference else "MiniMaxH3ImageToVideo", **conditioning_inputs)
    conditioning_ref = Graph.out(conditioning)
    if config and config["settings"]["semantic_bridge"] and config["settings"]["semantic_bridge_alpha"]:
        conditioning_ref = Graph.out(graph.add(H3_SEMANTIC_BRIDGE_NODE, conditioning=conditioning_ref,
            alpha=config["settings"]["semantic_bridge_alpha"]))
    empty = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(conditioning, 1))
    encoded_ref = Graph.out(encoded)
    offload_enabled = None
    if config and config["settings"]["stage_model_offload"]:
        from .h3 import add_h3_stage_offload
        if config["settings"]["text_encoder"] == "BF16":
            policy = graph.add(H3_STAGE_OFFLOAD_POLICY_NODE, conditioning=conditioning_ref, latent=encoded_ref,
                additional_conditioning=conditioning_ref, additional_latent=encoded_ref, cache_key=cache_key)
            conditioning_ref, encoded_ref = Graph.out(policy), Graph.out(policy, 1)
            offload_enabled = Graph.out(policy, 4)
        conditioning_ref, encoded_ref, _, _ = add_h3_stage_offload(
            graph, conditioning_ref, encoded_ref, enabled_ref=offload_enabled)
    upscaled = add_h3_latent_upscale(
        graph, encoded_ref,
        model_name=config["model_name"] if config else MODEL_SPECS["h3_latent_upscaler_3d_bf16"].local_name,
        precision=config["precision"] if config else "bf16", width=base_width * 2, height=base_height * 2,
    )
    combined = graph.add(
        H3_COMBINE_AV_LATENT_NODE, video_latent=upscaled, audio_latent=Graph.out(empty, 1),
    )
    latent = Graph.out(combined)
    latent = Graph.out(graph.add("H3RegeneratePrepareLatent", latent=latent, audio_vae=audio_vae_ref))
    if not config:
        model_name = models.profile(models.default_profile).fl2va
        base = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
        model = add_regenerate_model(graph, Graph.out(base), variant)
        if "convrot" in model_name.lower():
            model = Graph.out(graph.add(CHUNK_FEED_FORWARD_NODE, model=model, enabled=True,
                chunks=2, min_tokens=AUTO_SOL_TOKEN_THRESHOLD))
    if config and config["settings"]["stage_model_offload"]:
        conditioning_ref, latent, _, _ = add_h3_stage_offload(graph, conditioning_ref, latent, enabled_ref=offload_enabled)
    model = Graph.out(graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000))
    guider = graph.add("BasicGuider", model=model, conditioning=conditioning_ref)
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    if not config:
        sampler = graph.add("KSamplerSelect", sampler_name="euler")
        sigmas = Graph.out(graph.add("ManualSigmas", sigmas=H3_REGENERATE_SIGMAS))
    else:
        from .h3 import add_refinement_schedule
        from h3_app.policy import normalize_turbo_variant
        from h3_app.catalog import SAME_REFINEMENT_LORA
        settings = config["settings"]
        generation_variant = normalize_turbo_variant(settings["turbo_variant"]) if settings["generation_mode"] == "Turbo" else None
        generation_lora = models.turbo_lora_for(settings["mode"], generation_variant) if generation_variant else None
        sampler = (graph.add(LARRY_TURBO_SAMPLER_NODE)
            if generation_lora and turbo_uses_custom_nodes(generation_variant)
            else graph.add("KSamplerSelect", sampler_name=turbo_sampler_name(generation_variant, generation_lora)))
        schedule = (graph.add("ManualSigmas", sigmas=PIKA_FLEX_SIGMAS)
            if generation_variant == PIKA_FLEX else graph.add("BasicScheduler", model=schedule_model,
                scheduler=settings["scheduler"], steps=int(settings["steps"]), denoise=1.0))
        override = variant in H3_REGENERATE_MODELS or settings["latent_upscale_refine_lora"] != SAME_REFINEMENT_LORA
        sigmas, sampler = add_refinement_schedule(
            graph, sigmas=schedule, steps=int(settings["steps"]), sampler=sampler,
            turbo_variant=generation_variant, refinement_lora_name=(config["lora"] or "regenerate") if override else None,
            refinement_variant=variant if override else None, refinement_model_ref=model,
            latent_upscale_refine_steps=config["steps"],
        )
    sampled = graph.add(
        "SamplerCustomAdvanced", noise=Graph.out(noise), guider=Graph.out(guider),
        sampler=Graph.out(sampler), sigmas=sigmas, latent_image=latent,
    )
    sampled_ref = Graph.out(sampled)
    if config and config["settings"]["stage_model_offload"]:
        _, sampled_ref, _, _ = add_h3_stage_offload(graph, conditioning_ref, sampled_ref, enabled_ref=offload_enabled)
    video_latent = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=sampled_ref)
    if config:
        decoded = graph.add("VAEDecode", samples=Graph.out(video_latent), vae=decoder_ref)
    else:
        decoded = graph.add("VAEDecodeTiled", samples=Graph.out(video_latent), vae=decoder_ref,
            tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    trimmed = graph.add("ImageFromBatch", image=Graph.out(decoded), batch_index=0, length=Graph.out(encoded, 2))
    final = graph.add(
        "ImageScale", image=Graph.out(trimmed), upscale_method="lanczos",
        width=target_width, height=target_height, crop="disabled",
    )
    video = graph.add("CreateVideo", images=Graph.out(final), audio=Graph.out(components, 1), fps=float(fps), bit_depth=8)
    prefix = f"h3/regenerate_{H3_REGENERATE_MODELS.get(variant, 'refinement')}_{output_stamp}_{output_nonce}"
    if config:
        graph.add(H3_NVENC_SAVE_NODE, video=Graph.out(video), filename_prefix=prefix, preset="p4", constant_quality=23)
    else:
        graph.add("SaveVideo", video=Graph.out(video), filename_prefix=prefix, format="auto", codec="auto")
    return graph.nodes
