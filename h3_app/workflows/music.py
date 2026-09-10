"""Extracted music boundary."""

from __future__ import annotations

from typing import Any

from h3_app.graph import Graph
from h3_models import MODEL_SPECS, MUSIC3_MODEL_CHOICES


def required_music3_nodes(tiled_decode: bool) -> set[str]:
    nodes = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxMusic3TextEncode",
        "ConditioningZeroOut",
        "EmptyMiniMaxMusic3LatentAudio",
        "KSampler",
        "SaveAudioMP3",
    }
    nodes.add("VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio")
    return nodes


def build_music3_graph(
    *,
    model_choice: str,
    caption: str,
    lyrics: str,
    max_duration: float,
    seed: int,
    steps: int,
    cfg: float,
    ar_cfg: float,
    top_k: int,
    tiled_decode: bool,
) -> dict[str, Any]:
    """Build the official native ComfyUI MiniMax Music 3 workflow."""
    graph = Graph()
    dit = MODEL_SPECS[MUSIC3_MODEL_CHOICES[str(model_choice)]].local_name
    text_encoder = MODEL_SPECS["music3_text_encoder"].local_name
    vae_name = MODEL_SPECS["music3_vae"].local_name
    model = graph.add("UNETLoader", unet_name=dit, weight_dtype="default")
    clip = graph.add(
        "CLIPLoader", clip_name=text_encoder, type="minimax", device="default"
    )
    vae = graph.add("VAELoader", vae_name=vae_name)
    conditioning = graph.add(
        "MiniMaxMusic3TextEncode",
        clip=Graph.out(clip),
        caption=str(caption),
        lyrics=str(lyrics),
        seed=int(seed),
        max_duration=float(max_duration),
        cfg_scale=float(ar_cfg),
        top_k=int(top_k),
    )
    negative = graph.add("ConditioningZeroOut", conditioning=Graph.out(conditioning))
    latent = graph.add(
        "EmptyMiniMaxMusic3LatentAudio",
        seconds=Graph.out(conditioning, 1),
        batch_size=1,
    )
    sampled = graph.add(
        "KSampler",
        model=Graph.out(model),
        positive=Graph.out(conditioning),
        negative=Graph.out(negative),
        latent_image=Graph.out(latent),
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name="euler",
        scheduler="simple",
        denoise=1.0,
    )
    decoder = "VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio"
    decode_inputs: dict[str, Any] = {
        "samples": Graph.out(sampled),
        "vae": Graph.out(vae),
    }
    if tiled_decode:
        decode_inputs.update(tile_size=1536, overlap=64)
    audio = graph.add(decoder, **decode_inputs)
    graph.add(
        # SaveAudioAdvanced's DynamicCombo currently validates through the API
        # but its normalized execution path drops ``format``. The dedicated
        # MP3 node has a stable flat schema and produces the same V0 output.
        "SaveAudioMP3",
        audio=Graph.out(audio),
        filename_prefix="audio/minimax_music3",
        quality="V0",
    )
    return graph.nodes
