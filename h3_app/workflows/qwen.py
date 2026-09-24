"""Native ComfyUI workflow for Qwen Image 2.1 generation and editing."""

from __future__ import annotations

from typing import Any, Sequence

from h3_app.catalog import (
    QWEN_IMAGE21_SPECTRUM_EDIT_QUALITY_INPUTS,
    QWEN_IMAGE21_SPECTRUM_PREVIEW_INPUTS,
    QWEN_IMAGE21_SPECTRUM_QUALITY_INPUTS,
)
from h3_app.graph import Graph
from h3_models import (
    MODEL_SPECS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
)


def required_qwen_image21_nodes(
    *,
    editing: bool,
    use_spectrum: bool = False,
    turbo: bool = False,
    pdd: bool = False,
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
        "ModelAttentionBackend",
    }
    if editing:
        nodes |= {"LoadImage", "QwenImage21Cache"}
    if use_spectrum:
        nodes.add("QwenSpectrumModelPatcher")
    if turbo:
        nodes |= {
            "LoraLoaderModelOnly",
            "H3Qwen21TurboSigmas",
            "RandomNoise",
            "CFGGuider",
            "KSamplerSelect",
            "SamplerCustomAdvanced",
        }
    if pdd:
        nodes.discard("LoraLoaderModelOnly")
        nodes.discard("H3Qwen21TurboSigmas")
        nodes |= {"H3Qwen21PDDLoader", "H3Qwen21PDDSigmas"}
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
    accelerator: str,
    output_stamp: str,
    output_nonce: str,
    turbo_variant: str = "Off",
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
    viggle = turbo_variant == "Viggle Turbo v0.2"
    pdd = turbo_variant == "Alibaba PAI PDD 4-step"
    turbo = viggle or pdd
    if pdd and (
        int(steps) != 4 or float(cfg) != 1.0
        or str(sampler_name).lower() != "euler"
    ):
        raise ValueError("Alibaba PAI PDD requires 4 steps, CFG 1, and Euler.")
    if viggle:
        model = graph.add(
            "LoraLoaderModelOnly",
            model=Graph.out(model),
            lora_name=MODEL_SPECS["qwen_image21_viggle_v02_lora"].local_name,
            strength_model=1.0,
        )
    elif pdd:
        model = graph.add(
            "H3Qwen21PDDLoader",
            model=Graph.out(model),
            lora_name=MODEL_SPECS["qwen_image21_pdd_4step_lora"].local_name,
        )
    elif turbo_variant != "Off":
        raise ValueError(f"Unknown Qwen Image 2.1 Turbo variant: {turbo_variant}")
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

    # Always make the user's selection explicit. Newer ComfyUI checkpoints can
    # embed a preferred attention implementation per transformer block; an
    # explicit model override keeps "PyTorch attention" authoritative instead
    # of silently inheriting checkpoint metadata.
    patched_model = graph.add(
        "ModelAttentionBackend",
        model=Graph.out(model),
        attention=str(attention_backend),
    )
    sampled_model = Graph.out(patched_model)
    if editing:
        cached = graph.add(
            "QwenImage21Cache",
            model=sampled_model,
            device="off" if pdd else str(cache_device),
            dtype=str(cache_dtype),
        )
        sampled_model = Graph.out(cached)
    accelerator_key = str(accelerator).strip().lower()
    if accelerator_key != "off":
        if accelerator_key == "spectrum (preview)":
            spectrum_inputs = QWEN_IMAGE21_SPECTRUM_PREVIEW_INPUTS
        elif editing:
            spectrum_inputs = QWEN_IMAGE21_SPECTRUM_EDIT_QUALITY_INPUTS
        else:
            spectrum_inputs = QWEN_IMAGE21_SPECTRUM_QUALITY_INPUTS
        spectrum = graph.add(
            "QwenSpectrumModelPatcher",
            model=sampled_model,
            **spectrum_inputs,
        )
        sampled_model = Graph.out(spectrum)

    if turbo:
        noise = graph.add("RandomNoise", noise_seed=int(seed))
        guider = graph.add(
            "CFGGuider",
            model=sampled_model,
            positive=Graph.out(conditioning),
            negative=Graph.out(conditioning, 1),
            cfg=float(cfg),
        )
        sampler = graph.add("KSamplerSelect", sampler_name=str(sampler_name))
        if pdd:
            sigmas = graph.add("H3Qwen21PDDSigmas")
        else:
            sigmas = graph.add(
                "H3Qwen21TurboSigmas", latent_image=latent, steps=int(steps)
            )
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(sigmas),
            latent_image=latent,
        )
    else:
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
