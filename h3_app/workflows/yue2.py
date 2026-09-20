"""Native ComfyUI YuE2 workflow construction."""

from __future__ import annotations

from typing import Any

from h3_app.graph import Graph
from h3_models import MODEL_SPECS, YUE2_MODEL_CHOICES


def required_yue2_nodes(*, tiled_decode: bool, generate_abc: bool) -> set[str]:
    """Return the core nodes required for the selected YuE2 graph path."""
    nodes = {
        "CheckpointLoaderSimple",
        "YuE2GenerateMusic",
        "EmptyYuE2LatentAudio",
        "ConditioningZeroOut",
        "KSampler",
        "VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio",
        "SaveAudioMP3",
    }
    if generate_abc:
        nodes.add("YuE2GenerateABC")
    return nodes


def build_yue2_graph(
    *,
    model_choice: str,
    style: str,
    lyrics: str,
    abc: str,
    mode: str,
    max_duration: float,
    seed: int,
    steps: int,
    cfg: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_abc_tokens: int,
    abc_temperature: float,
    abc_top_p: float,
    abc_top_k: int,
    abc_repetition_penalty: float,
    abc_penalty_window: int,
    tiled_decode: bool,
) -> dict[str, Any]:
    """Build a native text-and-lyrics YuE2 song-generation graph."""
    choice = str(model_choice)
    if choice not in YUE2_MODEL_CHOICES:
        raise ValueError(f"Unknown YuE2 model: {model_choice}")
    graph = Graph()
    checkpoint = MODEL_SPECS[YUE2_MODEL_CHOICES[choice]].local_name
    loaded = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint)

    score = "" if str(mode) == "off" else str(abc or "").strip()
    generate_abc = not score and str(mode) in {"full", "melody"}
    if generate_abc:
        abc_node = graph.add(
            "YuE2GenerateABC",
            clip=Graph.out(loaded, 1),
            style=str(style),
            lyrics=str(lyrics),
            seed=int(seed),
            mode=str(mode),
            max_abc_tokens=int(max_abc_tokens),
            temperature=float(abc_temperature),
            top_p=float(abc_top_p),
            top_k=int(abc_top_k),
            repetition_penalty=float(abc_repetition_penalty),
            penalty_window=int(abc_penalty_window),
        )
        score_ref = Graph.out(abc_node)
    else:
        score_ref = score

    conditioning = graph.add(
        "YuE2GenerateMusic",
        clip=Graph.out(loaded, 1),
        style=str(style),
        lyrics=str(lyrics),
        abc=score_ref,
        seed=int(seed),
        mode=str(mode) if generate_abc or score else "full",
        max_duration=float(max_duration),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
    )
    negative = graph.add("ConditioningZeroOut", conditioning=Graph.out(conditioning))
    latent = graph.add(
        "EmptyYuE2LatentAudio", seconds=Graph.out(conditioning, 1), batch_size=1
    )
    sampled = graph.add(
        "KSampler",
        model=Graph.out(loaded, 0),
        positive=Graph.out(conditioning),
        negative=Graph.out(negative),
        latent_image=Graph.out(latent),
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name="dpm_2",
        scheduler="sgm_uniform",
        denoise=1.0,
    )
    decoder = "VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio"
    decode_inputs: dict[str, Any] = {
        "samples": Graph.out(sampled),
        "vae": Graph.out(loaded, 2),
    }
    if tiled_decode:
        decode_inputs.update(tile_size=1920, overlap=128)
    audio = graph.add(decoder, **decode_inputs)
    graph.add(
        "SaveAudioMP3", audio=Graph.out(audio), filename_prefix="audio/yue2", quality="V0"
    )
    return graph.nodes
