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
    pruna: bool = False,
    viggle: bool = False,
    viggle_multistep: bool = False,
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
    if turbo and not viggle_multistep:
        nodes |= {
            "RandomNoise",
            "CFGGuider",
            "KSamplerSelect",
            "SamplerCustomAdvanced",
        }
        if viggle:
            nodes.add("H3Qwen21TurboSigmas")
    if viggle:
        nodes.add("H3Qwen21ViggleLora")
    if pdd:
        nodes.discard("LoraLoaderModelOnly")
        nodes.discard("H3Qwen21TurboSigmas")
        nodes |= {"H3Qwen21PDDLoader", "H3Qwen21PDDSigmas"}
    if pruna:
        nodes.add("LoraLoaderModelOnly")
        nodes.discard("H3Qwen21TurboSigmas")
        nodes.add("H3Qwen21PrunaSigmas")
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
    viggle_pass2_steps: int = 6,
    viggle_pass3_steps: int = 6,
    viggle_pass2_denoise: float = 0.6,
    viggle_pass3_denoise: float = 0.25,
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
    viggle_multistep = turbo_variant == "Viggle 3-pass (configurable)"
    viggle = turbo_variant in {
        "Viggle Turbo v0.2.1 (6-step)", "Viggle 3-pass (configurable)"
    }
    pdd = turbo_variant == "Alibaba PAI PDD 4-step"
    pruna_steps = {"Pruna 8-step": 8, "Pruna 5-step": 5}.get(turbo_variant)
    turbo = viggle or pdd or pruna_steps is not None
    if viggle_multistep:
        if not 1 <= int(viggle_pass2_steps) <= 12 or not 1 <= int(viggle_pass3_steps) <= 12:
            raise ValueError("Viggle refinement passes require 1–12 steps each.")
        if not 0.5 <= float(viggle_pass2_denoise) <= 0.7:
            raise ValueError("Viggle pass 2 denoise must be between 0.5 and 0.7.")
        if not 0.2 <= float(viggle_pass3_denoise) <= 0.3:
            raise ValueError("Viggle pass 3 denoise must be between 0.2 and 0.3.")
    elif viggle and (int(steps) != 6 or float(cfg) != 1.0):
        raise ValueError("Viggle Turbo v0.2.1 requires 6 steps and CFG 1.")
    if pdd and (
        int(steps) != 4 or float(cfg) != 1.0
        or str(sampler_name).lower() != "euler"
    ):
        raise ValueError("Alibaba PAI PDD requires 4 steps, CFG 1, and Euler.")
    if pruna_steps is not None and (
        int(steps) != pruna_steps or float(cfg) != 1.0
        or str(sampler_name).lower() != "euler"
    ):
        raise ValueError(
            f"{turbo_variant} requires {pruna_steps} steps, CFG 1, and Euler."
        )
    if pruna_steps is not None:
        model = graph.add(
            "LoraLoaderModelOnly",
            model=Graph.out(model),
            lora_name=MODEL_SPECS[
                f"qwen_image21_pruna_{pruna_steps}step_lora"
            ].local_name,
            strength_model=1.0,
        )
    elif viggle:
        # The Viggle adapter is attached after attention/cache patching below,
        # using its custom unmerged loader to preserve adapter precision.
        pass
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
    if viggle and not viggle_multistep:
        viggle_model = graph.add(
            "H3Qwen21ViggleLora",
            model=sampled_model,
            lora_name=MODEL_SPECS["qwen_image21_viggle_v02_lora"].local_name,
        )
        sampled_model = Graph.out(viggle_model)
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

    if viggle_multistep:
        composition = graph.add(
            "KSampler", model=sampled_model,
            positive=Graph.out(conditioning), negative=Graph.out(conditioning, 1),
            latent_image=latent, seed=int(seed), steps=int(steps), cfg=1.0,
            sampler_name="euler", scheduler="simple", denoise=1.0,
        )
        viggle_model = graph.add(
            "H3Qwen21ViggleLora", model=sampled_model,
            lora_name=MODEL_SPECS["qwen_image21_viggle_v02_lora"].local_name,
        )
        refine = graph.add(
            "KSampler", model=Graph.out(viggle_model),
            positive=Graph.out(conditioning), negative=Graph.out(conditioning, 1),
            latent_image=Graph.out(composition), seed=int(seed) + 1,
            steps=int(viggle_pass2_steps), cfg=1.0,
            sampler_name="euler", scheduler="simple",
            denoise=float(viggle_pass2_denoise),
        )
        sampled = graph.add(
            "KSampler", model=Graph.out(viggle_model),
            positive=Graph.out(conditioning), negative=Graph.out(conditioning, 1),
            latent_image=Graph.out(refine), seed=int(seed) + 2,
            steps=int(viggle_pass3_steps), cfg=1.0,
            sampler_name="euler", scheduler="simple",
            denoise=float(viggle_pass3_denoise),
        )
    elif turbo:
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
        elif pruna_steps is not None:
            sigmas = graph.add("H3Qwen21PrunaSigmas", steps=pruna_steps)
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
