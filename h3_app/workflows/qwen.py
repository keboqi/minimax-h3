"""Native ComfyUI workflow for Qwen Image 2.1 generation and editing."""

from __future__ import annotations

from typing import Any, Sequence

from h3_app.catalog import (
    QWEN_IMAGE21_DYNAMIC_SCHEDULER,
    QWEN_IMAGE21_SPECTRUM_EDIT_QUALITY_INPUTS,
    QWEN_IMAGE21_SPECTRUM_PREVIEW_INPUTS,
    QWEN_IMAGE21_SPECTRUM_QUALITY_INPUTS,
)
from h3_app.graph import Graph
from h3_models import (
    MODEL_SPECS,
    QWEN_IMAGE21_LIGHTNING,
    QWEN_IMAGE21_LIGHTNING_6STEP,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_OFFICIAL_TURBO,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
    QWEN_IMAGE21_TURBO_MODES,
    QWEN_IMAGE21_PROMPT_ENHANCERS,
)


# Qwen/Qwen-Image-2.1-Turbo model_index.json at d65dbc9a7e8f6b5479e33dee6030eaab2a906509.
# These are final sigmas: no resolution shift or terminal stretch is applied.
QWEN_IMAGE21_OFFICIAL_TURBO_SIGMAS = (
    1.0, 0.978453, 0.95418, 0.926626,
    0.89508, 0.845148, 0.704534, 0.414568, 0.0,
)
# Community six-step Lightning schedule from
# https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/5.
# Final sigmas, used with the Lightning LoRA at reduced strength.
QWEN_IMAGE21_LIGHTNING_6STEP_SIGMAS = (
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.55, 0.0,
)
# Full strength pushes subjects left in generations and edits; 0.50 is the
# reported fix (https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning/discussions/7).
# The UI exposes the strength so a better value can be found; it applies to both step counts.
QWEN_IMAGE21_LIGHTNING_STRENGTH = 0.5


def qwen_prompt_enhancer_key(choice, *, editing):
    if choice == "Off":
        return None
    try:
        return QWEN_IMAGE21_PROMPT_ENHANCERS[choice][int(editing)]
    except KeyError as exc:
        raise ValueError("Choose a valid native Qwen prompt enhancer.") from exc


def required_qwen_image21_nodes(
    *,
    editing: bool,
    use_spectrum: bool = False,
    turbo: bool = False,
    viggle: bool = False,
    nine_step: bool = False,
    official_turbo: bool = False,
    lightning: bool = False,
    lightning_6step: bool = False,
    scheduler: str = QWEN_IMAGE21_DYNAMIC_SCHEDULER,
    prompt_enhancer: str = "Off",
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
        nodes |= {"LoadImage", "JoinImageWithAlpha", "QwenImage21Cache"}
    if qwen_prompt_enhancer_key(prompt_enhancer, editing=editing):
        nodes.add("H3Qwen21PromptEnhance")
    if use_spectrum:
        nodes.add("QwenSpectrumModelPatcher")
    if turbo or scheduler == QWEN_IMAGE21_DYNAMIC_SCHEDULER:
        nodes |= {
            "RandomNoise",
            "CFGGuider",
            "KSamplerSelect",
            "SamplerCustomAdvanced",
        }
        if viggle:
            nodes.add("H3Qwen21TurboSigmas")
        if not turbo:
            nodes.add("H3Qwen21Sigmas")
    if viggle:
        nodes.add("H3Qwen21ViggleLora")
    if official_turbo:
        nodes |= {"LoraLoaderModelOnly", "ManualSigmas"}
    if lightning:
        nodes |= {
            "LoraLoaderModelOnly",
            "ManualSigmas" if lightning_6step else "H3Qwen21LightningSigmas",
        }
    if nine_step:
        nodes |= {"SplitSigmas", "DisableNoise"}
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
    prompt_enhancer: str = "Off",
    lightning_strength: float = QWEN_IMAGE21_LIGHTNING_STRENGTH,
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
    turbo = turbo_variant in QWEN_IMAGE21_TURBO_MODES
    official_turbo = turbo_variant == QWEN_IMAGE21_OFFICIAL_TURBO
    lightning_6step = turbo_variant == QWEN_IMAGE21_LIGHTNING_6STEP
    lightning = turbo_variant == QWEN_IMAGE21_LIGHTNING or lightning_6step
    viggle = turbo and not (official_turbo or lightning)
    nine_step = viggle and QWEN_IMAGE21_TURBO_MODES[turbo_variant][1] == 9
    if lightning and not 0.0 < float(lightning_strength) <= 1.5:
        raise ValueError("Lightning LoRA strength must be greater than 0 and at most 1.5.")
    if turbo:
        lora_key, expected_steps = QWEN_IMAGE21_TURBO_MODES[turbo_variant]
        if int(steps) != expected_steps or float(cfg) != 1.0:
            raise ValueError(f"{turbo_variant} requires {expected_steps} steps and CFG 1.")
        if nine_step and (sampler_name != "euler" or accelerator.lower() != "off"):
            raise ValueError("Viggle nine-step mode requires Euler and accelerator Off.")
        if official_turbo and (sampler_name != "euler" or accelerator.lower() != "off"):
            raise ValueError("Official Turbo requires Euler and accelerator Off.")
        if lightning and (sampler_name != "euler" or accelerator.lower() != "off"):
            raise ValueError("LightX2V Lightning requires Euler and accelerator Off.")
    if not turbo and turbo_variant != "Off":
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
    writer_images = {}
    if editing:
        conditioning_inputs["vae"] = Graph.out(vae)
        for index, image in enumerate(reference_images, start=1):
            loaded = graph.add("LoadImage", image=str(image))
            writer_images[f"image{index - 1}"] = Graph.out(loaded)
            # LoadImage separates RGB and inverse alpha. The native join node
            # restores straight RGBA for Qwen's alpha-aware reference VAE.
            rgba = graph.add(
                "JoinImageWithAlpha", image=Graph.out(loaded),
                alpha=Graph.out(loaded, 1),
            )
            conditioning_inputs[f"images.image_{index}"] = Graph.out(rgba)
    enhancer_key = qwen_prompt_enhancer_key(prompt_enhancer, editing=editing)
    if enhancer_key:
        from .qwen_prompt_instructions import T2I_SYSTEM, I2I_SYSTEM

        writer = graph.add("CLIPLoader", clip_name=MODEL_SPECS[enhancer_key].local_name,
                           type="qwen_image" if editing else "stable_diffusion", device="default")
        enhanced = graph.add("H3Qwen21PromptEnhance", clip=Graph.out(writer),
                             prompt=str(prompt) if editing else f"Requested aspect ratio: {width}:{height}.\n{prompt}",
                             system_prompt=I2I_SYSTEM if editing else T2I_SYSTEM,
                             max_length=4096, seed=0, **writer_images)
        conditioning_inputs["prompt"] = Graph.out(enhanced)
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
            device="off" if lightning else str(cache_device),
            dtype=str(cache_dtype),
        )
        sampled_model = Graph.out(cached)
    # Keep the base branch free of the runtime adapter. Separate sampling runs
    # reset Qwen's prefix KV cache through ComfyUI's pre_run/cleanup lifecycle.
    base_model = sampled_model
    if viggle:
        viggle_model = graph.add(
            "H3Qwen21ViggleLora",
            model=sampled_model,
            lora_name=MODEL_SPECS[lora_key].local_name,
        )
        sampled_model = Graph.out(viggle_model)
    elif official_turbo or lightning:
        lora_model = graph.add(
            "LoraLoaderModelOnly",
            model=sampled_model,
            lora_name=MODEL_SPECS[lora_key].local_name,
            strength_model=float(lightning_strength) if lightning else 1.0,
        )
        sampled_model = Graph.out(lora_model)
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

    if turbo or scheduler == QWEN_IMAGE21_DYNAMIC_SCHEDULER:
        noise = graph.add("RandomNoise", noise_seed=int(seed))
        guider = graph.add(
            "CFGGuider",
            model=sampled_model,
            positive=Graph.out(conditioning),
            negative=Graph.out(conditioning, 1),
            cfg=float(cfg),
        )
        sampler = graph.add("KSamplerSelect", sampler_name=str(sampler_name))
        if official_turbo:
            sigmas = graph.add(
                "ManualSigmas",
                sigmas=", ".join(map(str, QWEN_IMAGE21_OFFICIAL_TURBO_SIGMAS)),
            )
        elif lightning_6step:
            sigmas = graph.add(
                "ManualSigmas",
                sigmas=", ".join(map(str, QWEN_IMAGE21_LIGHTNING_6STEP_SIGMAS)),
            )
        elif lightning:
            sigmas = graph.add("H3Qwen21LightningSigmas", latent_image=latent)
        else:
            sigmas = graph.add(
                "H3Qwen21TurboSigmas" if viggle else "H3Qwen21Sigmas",
                latent_image=latent, steps=int(steps),
            )
        sampling_sigmas = Graph.out(sigmas)
        if nine_step:
            split = graph.add("SplitSigmas", sigmas=sampling_sigmas, step=7)
            sampling_sigmas = Graph.out(split)
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(guider),
            sampler=Graph.out(sampler),
            sigmas=sampling_sigmas,
            latent_image=latent,
        )
        if nine_step:
            base_guider = graph.add(
                "CFGGuider", model=base_model,
                positive=Graph.out(conditioning),
                negative=Graph.out(conditioning, 1), cfg=1.0,
            )
            no_noise = graph.add("DisableNoise")
            # Output 0 retains the noisy latent at the shared boundary sigma;
            # output 1 is a denoised prediction and cannot continue this path.
            sampled = graph.add(
                "SamplerCustomAdvanced", noise=Graph.out(no_noise),
                guider=Graph.out(base_guider), sampler=Graph.out(sampler),
                sigmas=Graph.out(split, 1), latent_image=Graph.out(sampled),
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
