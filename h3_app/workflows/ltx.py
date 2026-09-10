"""Extracted ltx boundary."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import LTX25_DEFAULTS, LTX25_SIGMAS
from h3_app.graph import Graph
from h3_app.model_types import ltx25_model_names
from h3_app.policy import ltx25_frame_length
from h3_models import DEFAULT_LTX25_MODEL


def required_ltx25_nodes(*, image_to_video: bool = False) -> set[str]:
    required = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "LTXVConditioning",
        "EmptyLTXVLatentVideo",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "RandomNoise",
        "CFGGuider",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "VAEDecodeTiled",
        "LTXVAudioVAEDecode",
        "CreateVideo",
        "SaveVideo",
    }
    if image_to_video:
        required |= {"LoadImage", "LTXVAddGuide"}
    return required


def build_ltx25_graph(
    *,
    model_choice: str = DEFAULT_LTX25_MODEL,
    prompt: str,
    negative_prompt: str,
    first_image: str | None,
    width: int,
    height: int,
    duration: float,
    fps: float,
    seed: int,
    cfg: float,
    sampler_name: str,
    image_strength: float,
    middle_image: str | None = None,
    middle_time: float = LTX25_DEFAULTS["middle_time"],
    middle_strength: float = LTX25_DEFAULTS["middle_strength"],
    end_image: str | None = None,
    end_strength: float = LTX25_DEFAULTS["end_strength"],
    output_stamp: str = "0",
    output_nonce: str = "",
) -> dict[str, Any]:
    """Build the official LTX-2.5 distilled T2V or multi-keyframe I2V graph."""
    names = ltx25_model_names(model_choice)
    graph = Graph()
    model = graph.add(
        "UNETLoader", unet_name=names["distilled"], weight_dtype="default"
    )
    clip = graph.add(
        "CLIPLoader", clip_name=names["text_encoder"], type="ltxv", device="default"
    )
    video_vae = graph.add("VAELoader", vae_name=names["video_vae"])
    audio_vae = graph.add("VAELoader", vae_name=names["audio_vae"])
    positive = graph.add("CLIPTextEncode", clip=Graph.out(clip), text=prompt)
    negative = graph.add("CLIPTextEncode", clip=Graph.out(clip), text=negative_prompt)
    conditioned = graph.add(
        "LTXVConditioning",
        positive=Graph.out(positive),
        negative=Graph.out(negative),
        frame_rate=float(fps),
    )

    frames = ltx25_frame_length(duration, fps)
    video_latent = graph.add(
        "EmptyLTXVLatentVideo",
        width=int(width),
        height=int(height),
        length=frames,
        batch_size=1,
    )
    video_latent_ref = Graph.out(video_latent)
    positive_ref = Graph.out(conditioned, 0)
    negative_ref = Graph.out(conditioned, 1)
    middle_frame_idx = min(
        frames - 2,
        max(1, int(round(float(middle_time) * float(fps)))),
    )
    keyframes = (
        (first_image, 0, image_strength),
        (middle_image, middle_frame_idx, middle_strength),
        (end_image, -1, end_strength),
    )
    for image, frame_idx, strength in keyframes:
        if not image:
            continue
        loaded = graph.add("LoadImage", image=image)
        guide = graph.add(
            "LTXVAddGuide",
            positive=positive_ref,
            negative=negative_ref,
            vae=Graph.out(video_vae),
            latent=video_latent_ref,
            image=Graph.out(loaded),
            frame_idx=int(frame_idx),
            strength=float(strength),
        )
        positive_ref = Graph.out(guide, 0)
        negative_ref = Graph.out(guide, 1)
        video_latent_ref = Graph.out(guide, 2)

    audio_latent = graph.add(
        "LTXVEmptyLatentAudio",
        audio_vae=Graph.out(audio_vae),
        frames_number=frames,
        frame_rate=int(round(float(fps))),
        batch_size=1,
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        video_latent=video_latent_ref,
        audio_latent=Graph.out(audio_latent),
    )
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add(
        "CFGGuider",
        model=Graph.out(model),
        positive=positive_ref,
        negative=negative_ref,
        cfg=float(cfg),
    )
    sampler = graph.add("KSamplerSelect", sampler_name=str(sampler_name))
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
    images = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(separated, 0),
        vae=Graph.out(video_vae),
        tile_size=512,
        overlap=64,
        temporal_size=128,
        temporal_overlap=32,
    )
    audio = graph.add(
        "LTXVAudioVAEDecode",
        samples=Graph.out(separated, 1),
        audio_vae=Graph.out(audio_vae),
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(audio),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"ltx25/generation_{output_stamp}",
        format="auto",
        codec="auto",
    )
    return graph.nodes
