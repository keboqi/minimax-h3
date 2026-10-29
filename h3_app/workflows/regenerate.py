"""Shared Open H3 Regenerate adapter stack and existing-video graph."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import (
    AUTO_SOL_TOKEN_THRESHOLD,
    CHUNK_FEED_FORWARD_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_LATENT_UPSCALER_NODE,
    H3_REFINEMENT_COMPILER_GUARD_NODE,
    H3_REGENERATE_MODELS,
    H3_REGENERATE_SIGMAS,
    H3_SEPARATE_AV_LATENT_NODE,
)
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig
from h3_app.policy import snap32
from h3_models import MODEL_SPECS


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


def required_h3_regenerate_nodes(models: ModelConfig | None = None) -> set[str]:
    required = {
        "LoadVideo", "GetVideoComponents", "ImageScale", "UNETLoader",
        "LoraLoaderModelOnly", "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
        "H3RegenerateEncodeVideo", "H3RegeneratePrepareLatent",
        H3_LATENT_UPSCALER_NODE, H3_SEPARATE_AV_LATENT_NODE, H3_COMBINE_AV_LATENT_NODE,
        "BasicGuider", "RandomNoise", "KSamplerSelect", "ManualSigmas",
        "SamplerCustomAdvanced", "VAEDecodeTiled", "ImageFromBatch", "CreateVideo", "SaveVideo",
        H3_REFINEMENT_COMPILER_GUARD_NODE,
    }
    if models and "convrot" in models.profile(models.default_profile).fl2va.lower():
        required.add(CHUNK_FEED_FORWARD_NODE)
    return required


def build_h3_regenerate_graph(
    *, option: str, source_video: str, seed: int, models: ModelConfig,
    target_width: int, target_height: int, prompt: str = "", fps: float = 24.0,
    output_stamp: str = "0", output_nonce: str = "",
) -> dict[str, Any]:
    """Existing video -> H3 encode -> learned 2x -> refiner; retain source timing/audio."""
    if option not in H3_REGENERATE_MODELS:
        raise H3Error(f"Unknown H3 Regenerate option: {option}")
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
    upscaled = graph.add(
        H3_LATENT_UPSCALER_NODE, latent=Graph.out(encoded),
        model_name=MODEL_SPECS["h3_latent_upscaler_3d_bf16"].local_name,
        mode="scale by multiplier", align=32, enable_temporal_chunking=True,
        force_unload=True, device="cuda", precision="bf16", **{"mode.scale": 2.0},
    )
    combined = graph.add(
        H3_COMBINE_AV_LATENT_NODE, video_latent=Graph.out(upscaled), audio_latent=Graph.out(empty, 1),
    )
    prepared = graph.add("H3RegeneratePrepareLatent", latent=Graph.out(combined), audio_vae=Graph.out(audio_vae))
    model_name = models.profile(models.default_profile).fl2va
    base = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model = add_regenerate_model(graph, Graph.out(base), option)
    if "convrot" in model_name.lower():
        model = Graph.out(graph.add(
            CHUNK_FEED_FORWARD_NODE, model=model, enabled=True,
            chunks=2, min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
        ))
    model = Graph.out(graph.add(H3_REFINEMENT_COMPILER_GUARD_NODE, model=model, min_video_volume=1_500_000))
    guider = graph.add("BasicGuider", model=model, conditioning=Graph.out(conditioning))
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    sampler = graph.add("KSamplerSelect", sampler_name="euler")
    sigmas = graph.add("ManualSigmas", sigmas=H3_REGENERATE_SIGMAS)
    sampled = graph.add(
        "SamplerCustomAdvanced", noise=Graph.out(noise), guider=Graph.out(guider),
        sampler=Graph.out(sampler), sigmas=Graph.out(sigmas), latent_image=Graph.out(prepared),
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
        filename_prefix=f"h3/regenerate_{H3_REGENERATE_MODELS[option]}_{output_stamp}_{output_nonce}",
        format="auto", codec="auto",
    )
    return graph.nodes
