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
    H3_REFINEMENT,
    H3_SPLIT_TEMPORAL_PARAMS_NODE, H3_SPLIT_SPATIAL_PARAMS_NODE, H3_SPLIT_UPSCALE_NODE,
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
    required = {
        "LoadVideo", "GetVideoComponents", "ImageScale", "UNETLoader",
        "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
        "H3RegenerateEncodeVideo", "H3RegeneratePrepareLatent",
        H3_LATENT_UPSCALER_NODE, H3_LATENT_UPSCALER_LOADER_NODE,
        H3_SEPARATE_AV_LATENT_NODE, H3_COMBINE_AV_LATENT_NODE,
        "BasicGuider", "RandomNoise", "VAEDecodeTiled", "ImageFromBatch", "CreateVideo", "SaveVideo",
        H3_REFINEMENT_COMPILER_GUARD_NODE,
    }
    profile = models.default_profile if models else None
    if refinement is not None and models is not None:
        from .h3 import turbo_required_nodes
        config = resolve_refinement(refinement, models)
        variant = config["variant"]
        profile = config["settings"]["model_profile"]
        if variant in H3_REGENERATE_MODELS:
            required |= {"LoraLoaderModelOnly", "KSamplerSelect", "ManualSigmas"}
        else:
            required |= turbo_required_nodes(variant, config["lora"]) if config["lora"] else {"KSamplerSelect"}
            required.add("ManualSigmas" if variant == PIKA_FLEX else "BasicScheduler")
            required.add("SplitSigmas")
        if config["split"]:
            required |= {H3_SPLIT_TEMPORAL_PARAMS_NODE, H3_SPLIT_SPATIAL_PARAMS_NODE, H3_SPLIT_UPSCALE_NODE}
        else:
            required.add("SamplerCustomAdvanced")
    else:
        required |= {"LoraLoaderModelOnly", "KSamplerSelect", "ManualSigmas", "SamplerCustomAdvanced"}
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
    vae = graph.add("VAELoader", vae_name=models.video_vae)
    audio_vae = graph.add("VAELoader", vae_name=models.audio_vae)
    encoded = graph.add("H3RegenerateEncodeVideo", images=Graph.out(resized), vae=Graph.out(vae))
    clip = graph.add("CLIPLoader", clip_name=models.text_encoder, type="minimax", device="default")
    conditioning = graph.add(
        "MiniMaxH3ImageToVideo", clip=Graph.out(clip), vae=Graph.out(vae),
        prompt=prompt.strip() or "Natural fine detail, consistent subjects, motion, lighting and color.",
        width=base_width * 2, height=base_height * 2, length=Graph.out(encoded, 1),
    )
    empty = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(conditioning, 1))
    upscaled = add_h3_latent_upscale(
        graph, Graph.out(encoded),
        model_name=config["model_name"] if config else MODEL_SPECS["h3_latent_upscaler_3d_bf16"].local_name,
        precision=config["precision"] if config else "bf16", width=base_width * 2, height=base_height * 2,
    )
    combined = graph.add(
        H3_COMBINE_AV_LATENT_NODE, video_latent=upscaled, audio_latent=Graph.out(empty, 1),
    )
    latent = Graph.out(combined)
    latent = Graph.out(graph.add("H3RegeneratePrepareLatent", latent=latent, audio_vae=Graph.out(audio_vae)))
    model_name = models.profile(config["settings"]["model_profile"] if config else models.default_profile).fl2va
    base = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model = Graph.out(base)
    if variant in H3_REGENERATE_MODELS:
        model = add_regenerate_model(graph, model, variant)
    elif config["lora"]:
        from .h3 import add_refinement_model, turbo_required_nodes
        model = add_refinement_model(
            graph, model, lora_name=config["lora"], variant=variant,
            available_nodes=turbo_required_nodes(variant, config["lora"]),
        )
    if "convrot" in model_name.lower():
        model = Graph.out(graph.add(
            CHUNK_FEED_FORWARD_NODE, model=model, enabled=True,
            chunks=2, min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
        ))
    model = Graph.out(graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000))
    guider = graph.add("BasicGuider", model=model, conditioning=Graph.out(conditioning))
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    if variant in H3_REGENERATE_MODELS:
        sampler = graph.add("KSamplerSelect", sampler_name="euler")
        sigmas = Graph.out(graph.add("ManualSigmas", sigmas=H3_REGENERATE_SIGMAS))
    else:
        sampler = (
            graph.add(LARRY_TURBO_SAMPLER_NODE)
            if config["lora"] and turbo_uses_custom_nodes(variant)
            else graph.add("KSamplerSelect", sampler_name=turbo_sampler_name(variant, config["lora"]))
        )
        schedule = (
            graph.add("ManualSigmas", sigmas=PIKA_FLEX_SIGMAS)
            if variant == PIKA_FLEX else graph.add(
                "BasicScheduler", model=model, scheduler=config["settings"]["scheduler"],
                steps=config["total"], denoise=1.0,
            )
        )
        sigmas = Graph.out(graph.add("SplitSigmas", sigmas=Graph.out(schedule), step=config["total"] - config["steps"]), 1)
    split = config["split"] if config else None
    if split:
        temporal = graph.add(H3_SPLIT_TEMPORAL_PARAMS_NODE, chunk_frames=split.chunk_frames,
            temporal_overlap_frames=split.temporal_overlap_frames, anchor_strength=0.999,
            motion_anchor_frames="22", identity_anchor_frames=24)
        spatial = graph.add(H3_SPLIT_SPATIAL_PARAMS_NODE, tile_width=split.tile_width,
            tile_height=split.tile_height, overlap_ratio=split.overlap_ratio,
            fade_ratio=split.fade_ratio, min_tile_size=256, seam_denoise=split.seam_denoise)
        sampled = graph.add(H3_SPLIT_UPSCALE_NODE, model=model, conditioning=Graph.out(conditioning),
            latent=latent, noise=Graph.out(noise), sampler=Graph.out(sampler), sigmas=sigmas,
            cfg=1.0, temporal_split_param=Graph.out(temporal), spatial_split_param=Graph.out(spatial),
            seam_polish=split.seam_polish, color_match=True)
    else:
        sampled = graph.add(
            "SamplerCustomAdvanced", noise=Graph.out(noise), guider=Graph.out(guider),
            sampler=Graph.out(sampler), sigmas=sigmas, latent_image=latent,
        )
    video_latent = graph.add(H3_SEPARATE_AV_LATENT_NODE, latent=Graph.out(sampled))
    decoded = graph.add(
        "VAEDecodeTiled", samples=Graph.out(video_latent), vae=Graph.out(vae),
        tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
    )
    trimmed = graph.add("ImageFromBatch", image=Graph.out(decoded), batch_index=0, length=Graph.out(encoded, 2))
    final = graph.add(
        "ImageScale", image=Graph.out(trimmed), upscale_method="lanczos",
        width=target_width, height=target_height, crop="disabled",
    )
    video = graph.add("CreateVideo", images=Graph.out(final), audio=Graph.out(components, 1), fps=float(fps), bit_depth=8)
    graph.add(
        "SaveVideo", video=Graph.out(video),
        filename_prefix=f"h3/regenerate_{H3_REGENERATE_MODELS.get(variant, 'refinement')}_{output_stamp}_{output_nonce}",
        format="auto", codec="auto",
    )
    return graph.nodes
