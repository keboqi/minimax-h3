"""Extracted upscale boundary."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import LTX25_SIGMAS, LTX25_UPSCALE, SEEDVR2_UPSCALE
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.model_types import (
    ModelConfig,
    ltx25_model_names,
    seedvr2_upscale_model_names,
)
from h3_app.policy import snap32
from h3_models import DEFAULT_LTX25_MODEL, DEFAULT_SEEDVR2_MODEL, MODEL_SPECS


def required_seedvr2_upscale_nodes() -> set[str]:
    return {
        "LoadVideo",
        "GetVideoComponents",
        "ImageScaleBy",
        "SeedVR2Preprocess",
        "VAELoader",
        "UNETLoader",
        "VAEEncodeTiled",
        "SeedVR2TemporalChunk",
        "SeedVR2Conditioning",
        "KSampler",
        "SeedVR2TemporalMerge",
        "VAEDecodeTiled",
        "SeedVR2PostProcessing",
        "CreateVideo",
        "SaveVideo",
    }


def required_seedvr2_image_upscale_nodes() -> set[str]:
    """Return the node contract for pre-generation still-image upscaling."""
    return {
        "LoadImage",
        "ImageScaleBy",
        "SeedVR2Preprocess",
        "VAELoader",
        "UNETLoader",
        "VAEEncodeTiled",
        "SeedVR2Conditioning",
        "KSampler",
        "VAEDecodeTiled",
        "SeedVR2PostProcessing",
        "SaveImage",
    }


def build_seedvr2_image_upscale_graph(
    *,
    source_images: list[tuple[str, str, float]],
    seed: int,
    models: ModelConfig,
    model_choice: str = DEFAULT_SEEDVR2_MODEL,
    output_token: str,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    """Build one native SeedVR2 graph for one or more independent stills."""
    if not source_images:
        raise H3Error("Select at least one input image to upscale.")
    assets = seedvr2_upscale_model_names(models, model_choice)
    graph = Graph()
    vae = graph.add("VAELoader", vae_name=assets["seedvr2_vae"])
    model = graph.add(
        "UNETLoader", unet_name=assets["seedvr2_dit"], weight_dtype="default"
    )
    model_ref = Graph.out(model)
    for image_index, (slot_key, source_image, scale_by) in enumerate(source_images):
        loaded = graph.add("LoadImage", image=source_image)
        resized = graph.add(
            "ImageScaleBy",
            image=Graph.out(loaded),
            upscale_method="lanczos",
            scale_by=float(scale_by),
        )
        prepared = graph.add("SeedVR2Preprocess", resized_images=Graph.out(resized))
        latent = graph.add(
            "VAEEncodeTiled",
            pixels=Graph.out(prepared),
            vae=Graph.out(vae),
            tile_size=1024,
            overlap=128,
            temporal_size=64,
            temporal_overlap=8,
        )
        conditioning = graph.add(
            "SeedVR2Conditioning",
            model=model_ref,
            vae_conditioning=Graph.out(latent),
        )
        sampled = graph.add(
            "KSampler",
            model=model_ref,
            seed=(int(seed) + image_index) % (2**63 - 1),
            steps=1,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=Graph.out(conditioning, 0),
            negative=Graph.out(conditioning, 1),
            latent_image=Graph.out(latent),
            denoise=1.0,
        )
        decoded = graph.add(
            "VAEDecodeTiled",
            samples=Graph.out(sampled),
            vae=Graph.out(vae),
            tile_size=1024,
            overlap=128,
            temporal_size=64,
            temporal_overlap=8,
        )
        restored = graph.add(
            "SeedVR2PostProcessing",
            images=Graph.out(decoded),
            original_resized_images=Graph.out(resized),
            color_correction_method="none",
        )
        graph.add(
            "SaveImage",
            images=Graph.out(restored),
            filename_prefix=f"h3/input_upscale/{output_token}_{slot_key}",
        )
    return graph.nodes


def build_seedvr2_upscale_graph(
    *,
    source_video: str,
    seed: int,
    models: ModelConfig,
    model_choice: str = DEFAULT_SEEDVR2_MODEL,
    target_width: int | None = None,
    source_width: int | None = None,
    fps: float = 24.0,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    """Build ComfyUI's native one-step SeedVR2 2x workflow."""
    assets = seedvr2_upscale_model_names(models, model_choice)
    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    resized = graph.add(
        "ImageScaleBy",
        image=Graph.out(components, 0),
        upscale_method="lanczos",
        scale_by=(
            float(target_width) / float(source_width)
            if target_width and source_width
            else 2.0
        ),
    )
    prepared = graph.add("SeedVR2Preprocess", resized_images=Graph.out(resized))
    vae = graph.add("VAELoader", vae_name=assets["seedvr2_vae"])
    model = graph.add(
        "UNETLoader", unet_name=assets["seedvr2_dit"], weight_dtype="default"
    )
    model_ref = Graph.out(model)
    latent = graph.add(
        "VAEEncodeTiled",
        pixels=Graph.out(prepared),
        vae=Graph.out(vae),
        tile_size=1024,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    chunks = graph.add(
        "SeedVR2TemporalChunk",
        latent=Graph.out(latent),
        temporal_overlap=1,
        # DynamicCombo selections are plain option strings in API prompts;
        # ComfyUI expands this to the mapping consumed by execute().
        chunking_mode="auto",
    )
    conditioning = graph.add(
        "SeedVR2Conditioning",
        model=model_ref,
        vae_conditioning=Graph.out(chunks, 0),
    )
    sampled = graph.add(
        "KSampler",
        model=model_ref,
        seed=int(seed),
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=Graph.out(conditioning, 0),
        negative=Graph.out(conditioning, 1),
        latent_image=Graph.out(chunks, 0),
        denoise=1.0,
    )
    merged = graph.add(
        "SeedVR2TemporalMerge",
        latents=Graph.out(sampled),
        temporal_overlap=Graph.out(chunks, 1),
    )
    decoded = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(merged),
        vae=Graph.out(vae),
        tile_size=1024,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    restored = graph.add(
        "SeedVR2PostProcessing",
        images=Graph.out(decoded),
        original_resized_images=Graph.out(resized),
        color_correction_method="none",
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(restored),
        audio=Graph.out(components, 1),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"seedvr2/upscale_{output_stamp}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_ltx25_upscale_nodes() -> set[str]:
    return {
        "LoadVideo",
        "GetVideoComponents",
        "ImageScale",
        "GetImageSize",
        "UNETLoader",
        "LTXICLoRALoaderModelOnly",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "LTXVConditioning",
        "EmptyLTXVLatentVideo",
        "LTXAddVideoICLoRAGuide",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "RandomNoise",
        "CFGGuider",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "LTXVCropGuides",
        "VAEDecodeTiled",
        "CreateVideo",
        "SaveVideo",
    }


def build_ltx25_upscale_graph(
    *,
    source_video: str,
    seed: int,
    model_choice: str = DEFAULT_LTX25_MODEL,
    prompt: str = "",
    width: int,
    height: int,
    target_width: int | None = None,
    target_height: int | None = None,
    fps: float = 24.0,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    """Build Lightricks' single-stage IC-LoRA generative 2x upscaler."""
    names = ltx25_model_names(model_choice)
    if target_width is None or target_height is None:
        base_width = snap32(width)
        base_height = snap32(height)
        target_width = base_width * 2
        target_height = base_height * 2
    else:
        base_width = snap32(int(target_width) // 2)
        base_height = snap32(int(target_height) // 2)
        target_width = base_width * 2
        target_height = base_height * 2

    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    guide = graph.add(
        "ImageScale",
        image=Graph.out(components, 0),
        upscale_method="lanczos",
        width=base_width,
        height=base_height,
        crop="disabled",
    )
    guide_size = graph.add("GetImageSize", image=Graph.out(guide))

    base_model = graph.add(
        "UNETLoader", unet_name=names["distilled"], weight_dtype="default"
    )
    upscaler = graph.add(
        "LTXICLoRALoaderModelOnly",
        model=Graph.out(base_model),
        lora_name=MODEL_SPECS["ltx25_pixel_upscaler_x2"].local_name,
        strength_model=1.0,
    )
    clip = graph.add(
        "CLIPLoader", clip_name=names["text_encoder"], type="ltxv", device="default"
    )
    video_vae = graph.add("VAELoader", vae_name=names["video_vae"])
    audio_vae = graph.add("VAELoader", vae_name=names["audio_vae"])
    positive = graph.add(
        "CLIPTextEncode",
        clip=Graph.out(clip),
        text=(prompt.strip() or "high quality, detailed video"),
    )
    negative = graph.add("CLIPTextEncode", clip=Graph.out(clip), text="")
    conditioned = graph.add(
        "LTXVConditioning",
        positive=Graph.out(positive),
        negative=Graph.out(negative),
        frame_rate=float(fps),
    )
    video_latent = graph.add(
        "EmptyLTXVLatentVideo",
        width=target_width,
        height=target_height,
        length=Graph.out(guide_size, 2),
        batch_size=1,
    )
    guided = graph.add(
        "LTXAddVideoICLoRAGuide",
        positive=Graph.out(conditioned, 0),
        negative=Graph.out(conditioned, 1),
        vae=Graph.out(video_vae),
        latent=Graph.out(video_latent),
        image=Graph.out(guide),
        frame_idx=0,
        strength=1.0,
        latent_downscale_factor=Graph.out(upscaler, 1),
        crop="disabled",
        use_tiled_encode=True,
        tile_size=512,
        tile_overlap=64,
    )
    audio_latent = graph.add(
        "LTXVEmptyLatentAudio",
        audio_vae=Graph.out(audio_vae),
        frames_number=Graph.out(guide_size, 2),
        frame_rate=int(round(float(fps))),
        batch_size=1,
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        video_latent=Graph.out(guided, 2),
        audio_latent=Graph.out(audio_latent),
    )
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add(
        "CFGGuider",
        model=Graph.out(upscaler),
        positive=Graph.out(guided, 0),
        negative=Graph.out(guided, 1),
        cfg=1.0,
    )
    sampler = graph.add("KSamplerSelect", sampler_name="euler_ancestral")
    sigmas = graph.add("ManualSigmas", sigmas=LTX25_SIGMAS)
    sampled = graph.add(
        "SamplerCustomAdvanced",
        noise=Graph.out(noise),
        guider=Graph.out(guider),
        sampler=Graph.out(sampler),
        sigmas=Graph.out(sigmas),
        latent_image=Graph.out(av_latent),
    )
    separated = graph.add("LTXVSeparateAVLatent", av_latent=Graph.out(sampled))
    cropped = graph.add(
        "LTXVCropGuides",
        positive=Graph.out(guided, 0),
        negative=Graph.out(guided, 1),
        latent=Graph.out(separated, 0),
    )
    images = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(cropped, 2),
        vae=Graph.out(video_vae),
        tile_size=512,
        overlap=64,
        temporal_size=128,
        temporal_overlap=32,
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(components, 1),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"ltx25/upscale_{output_stamp}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_upscale_nodes(option: str) -> set[str]:
    if option == SEEDVR2_UPSCALE:
        return required_seedvr2_upscale_nodes()
    if option == LTX25_UPSCALE:
        return required_ltx25_upscale_nodes()
    raise H3Error(f"Unknown AI post-processing method: {option}")


def build_upscale_graph(
    *,
    option: str,
    source_video: str,
    seed: int,
    models: ModelConfig,
    seedvr2_model: str = DEFAULT_SEEDVR2_MODEL,
    ltx25_model: str = DEFAULT_LTX25_MODEL,
    prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    fps: float = 24.0,
    output_stamp: str = "0",
    output_nonce: str = "",
) -> tuple[dict[str, Any], int]:
    """Build one selected AI-upscale workflow and return its sampler steps."""
    if option == SEEDVR2_UPSCALE:
        return (
            build_seedvr2_upscale_graph(
                source_video=source_video,
                output_stamp=output_stamp,
                output_nonce=output_nonce,
                seed=seed,
                models=models,
                model_choice=seedvr2_model,
                target_width=target_width,
                source_width=width,
                fps=fps,
            ),
            1,
        )
    if option == LTX25_UPSCALE:
        if width is None or height is None:
            raise H3Error("LTX-2.5 upscaling requires the source video dimensions.")
        return (
            build_ltx25_upscale_graph(
                source_video=source_video,
                output_stamp=output_stamp,
                output_nonce=output_nonce,
                seed=seed,
                model_choice=ltx25_model,
                prompt=prompt,
                width=width,
                height=height,
                target_width=target_width,
                target_height=target_height,
                fps=fps,
            ),
            8,
        )
    raise H3Error(f"Unknown AI post-processing method: {option}")
