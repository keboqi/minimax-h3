"""Native ComfyUI workflow for Qwen Image 2.1 generation and editing."""

from __future__ import annotations

from typing import Any, Sequence

from h3_app.graph import Graph
from h3_models import (
    MODEL_SPECS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
)


def required_qwen_image21_nodes(
    *, editing: bool, use_attention_backend: bool = False
) -> set[str]:
    nodes = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImage21",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    if editing:
        nodes |= {"LoadImage", "QwenImage21Cache"}
    if use_attention_backend:
        nodes.add("ModelAttentionBackend")
    return nodes


def build_qwen_image21_graph(
    *,
    model_choice: str,
    text_encoder_choice: str,
    prompt: str,
    negative_prompt: str,
    reference_images: Sequence[str],
    width: int,
    height: int,
    reference_resolution: int,
    match_input_size: bool,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    cache_device: str,
    cache_dtype: str,
    attention_backend: str,
    output_stamp: str,
    output_nonce: str,
) -> dict[str, Any]:
    """Build the official native Qwen Image 2.1 graph.

    With references, image_1 is the edit target and subsequent images are
    prompt-addressable as <image2>, <image3>, and so on.
    """
    try:
        model_key = QWEN_IMAGE21_MODEL_CHOICES[str(model_choice)]
        encoder_key = QWEN_IMAGE21_TEXT_ENCODER_CHOICES[str(text_encoder_choice)]
    except KeyError as exc:
        raise ValueError(f"Unknown Qwen Image 2.1 model choice: {exc.args[0]}") from exc

    graph = Graph()
    model = graph.add(
        "UNETLoader",
        unet_name=MODEL_SPECS[model_key].local_name,
        weight_dtype="default",
    )
    clip = graph.add(
        "CLIPLoader",
        clip_name=MODEL_SPECS[encoder_key].local_name,
        type="qwen_image",
        device="default",
    )
    vae = graph.add("VAELoader", vae_name=MODEL_SPECS["qwen_image21_vae"].local_name)

    editing = bool(reference_images)
    conditioning_inputs: dict[str, Any] = {
        "clip": Graph.out(clip),
        "prompt": str(prompt),
        "negative_prompt": str(negative_prompt),
        "resolution": int(reference_resolution),
    }
    if editing:
        conditioning_inputs["vae"] = Graph.out(vae)
        for index, image in enumerate(reference_images, start=1):
            loaded = graph.add("LoadImage", image=str(image))
            conditioning_inputs[f"images.image_{index}"] = Graph.out(loaded)
    conditioning = graph.add("TextEncodeQwenImage21", **conditioning_inputs)

    if editing and bool(match_input_size):
        latent = Graph.out(conditioning, 2)
    else:
        empty = graph.add(
            "EmptyLatentImage",
            width=int(width),
            height=int(height),
            batch_size=1,
        )
        latent = Graph.out(empty)

    sampled_model = Graph.out(model)
    if str(attention_backend) != "pytorch attention":
        patched_model = graph.add(
            "ModelAttentionBackend",
            model=sampled_model,
            attention=str(attention_backend),
        )
        sampled_model = Graph.out(patched_model)
    if editing:
        cached = graph.add(
            "QwenImage21Cache",
            model=sampled_model,
            device=str(cache_device),
            dtype=str(cache_dtype),
        )
        sampled_model = Graph.out(cached)

    sampled = graph.add(
        "KSampler",
        model=sampled_model,
        positive=Graph.out(conditioning),
        negative=Graph.out(conditioning, 1),
        latent_image=latent,
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=str(sampler_name),
        scheduler=str(scheduler),
        denoise=1.0,
    )
    decoded = graph.add(
        "VAEDecode",
        samples=Graph.out(sampled),
        vae=Graph.out(vae),
    )
    graph.add(
        "SaveImage",
        images=Graph.out(decoded),
        filename_prefix=(
            f"h3/image_staging/qwen_image21_{output_stamp}_{output_nonce}"
        ),
    )
    return graph.nodes
