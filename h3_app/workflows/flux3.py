"""FLUX 3 distilled graphs with explicit BFL sampling and upsampling settings."""

import math
import json
from pathlib import Path

from h3_app.graph import Graph
from h3_models import (FLUX3_MODEL_CHOICES, MODEL_SPECS, FLUX3_UPSAMPLER_CHOICES,
                       DEFAULT_FLUX3_UPSAMPLER, FLUX3_BASE_MODEL_CHOICES)
from .flux3_layout import parse_flux3_layout


# top_k=1 decodes greedily, so the seed cannot change the caption. A constant
# seed keeps the upsampler inputs identical across image-seed rerolls, letting
# ComfyUI reuse the cached expansion instead of reloading the 9B upsampler.
UPSAMPLER_SEED = 0
# BFL's official text-to-image upsampler instructions, shared with the remote writers.
UPSAMPLER_SYSTEM_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompt_flux3_t2i.txt"
UPSAMPLER_EDIT_SYSTEM_PATH = UPSAMPLER_SYSTEM_PATH.with_name("prompt_flux3_i2i.txt")
UPSAMPLER_MAX_LENGTH = 3584
# BFL reference 62e90b9278: alpha 6.93 -> 16, for distilled and base.
# Comfy's native Flux sampler takes ln(alpha), independent of canvas size.
FLUX3_SAMPLING_SHIFT = math.log(16.0)


def upsampler_system(path=UPSAMPLER_SYSTEM_PATH):
    return Path(path).read_text(encoding="utf-8").strip()


def flux3_model_keys(model_choice, expand_prompt, upsampler_choice=DEFAULT_FLUX3_UPSAMPLER):
    if model_choice not in FLUX3_MODEL_CHOICES:
        raise ValueError("Unknown FLUX 3 model choice.")
    if upsampler_choice not in FLUX3_UPSAMPLER_CHOICES:
        raise ValueError("Unknown FLUX 3 prompt upsampler.")
    return (FLUX3_MODEL_CHOICES[model_choice], "flux3_text_encoder", "flux3_vae",
            *((FLUX3_UPSAMPLER_CHOICES[upsampler_choice],) if expand_prompt else ()))


def flux3_sampling_defaults(model_choice):
    if model_choice not in FLUX3_MODEL_CHOICES:
        raise ValueError("Unknown FLUX 3 model choice.")
    return (50, 4.0) if model_choice in FLUX3_BASE_MODEL_CHOICES else (16, 1.0)


def build_flux3_graph(*, model_choice, prompt, reference_images=(), width=1024,
                      height=1024, seed=0, steps=None, expand_prompt=True,
                      output_stamp="0", output_nonce="flux3", system_prompt=None,
                      upsampler_choice=DEFAULT_FLUX3_UPSAMPLER, cfg=None, layout=""):
    flux3_model_keys(model_choice, expand_prompt, upsampler_choice)
    default_steps, default_cfg = flux3_sampling_defaults(model_choice)
    steps, cfg = default_steps if steps is None else steps, default_cfg if cfg is None else cfg
    if model_choice not in FLUX3_BASE_MODEL_CHOICES and cfg != 1.0:
        raise ValueError("Distilled FLUX 3 requires CFG 1.")
    elements = parse_flux3_layout(layout)
    if len(reference_images) > 10:
        raise ValueError("FLUX 3 supports at most ten reference images.")
    g = Graph()
    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))
    model = add("UNETLoader", unet_name=MODEL_SPECS[FLUX3_MODEL_CHOICES[model_choice]].local_name, weight_dtype="default")
    model = add("ModelSamplingFlux", model=model, base_shift=FLUX3_SAMPLING_SHIFT,
                max_shift=FLUX3_SAMPLING_SHIFT, width=width, height=height)
    clip = add("CLIPLoader", clip_name=MODEL_SPECS["flux3_text_encoder"].local_name, type="flux3", device="default")
    vae = add("VAELoader", vae_name=MODEL_SPECS["flux3_vae"].local_name)
    images = [add("LoadImage", image=path) for path in reference_images]
    if expand_prompt:
        writer = add("CLIPLoader", clip_name=MODEL_SPECS[FLUX3_UPSAMPLER_CHOICES[upsampler_choice]].local_name, type="flux3", device="default")
        if images:
            # Tokenize separate images rather than cropping/resizing them into
            # one tensor batch: the writer must see every reference in order.
            prompt = add("H3Flux3EditPrompt", clip=writer, prompt=prompt,
                         system_prompt=system_prompt or upsampler_system(UPSAMPLER_EDIT_SYSTEM_PATH),
                         max_length=UPSAMPLER_MAX_LENGTH, seed=UPSAMPLER_SEED,
                         **{f"image{i}": image for i, image in enumerate(images)})
        else:
            prompt = add("TextGenerate", clip=writer, prompt=prompt,
                         system_prompt=system_prompt or upsampler_system(),
                         max_length=UPSAMPLER_MAX_LENGTH, sampling_mode="on", thinking=False,
                         use_default_template=True, mtp="off", **{
                             "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1,
                             "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
                             "sampling_mode.repetition_penalty": 1.05,
                             "sampling_mode.presence_penalty": 1.5, "sampling_mode.seed": UPSAMPLER_SEED,
                         })
    if elements:
        array = add("H3Flux3LayoutElements", layout=json.dumps(elements, ensure_ascii=False))
        prompt = add("Flux3LayoutPrompt", prompt=prompt, element=array)
    positive = add("CLIPTextEncode", clip=clip, text=prompt)
    negative = (add("CLIPTextEncode", clip=clip, text="") if model_choice in FLUX3_BASE_MODEL_CHOICES
                else add("ConditioningZeroOut", conditioning=positive))
    latent = add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    if reference_images:
        refs = g.add("ReferenceMultiLatent", positive=positive, negative=negative, vae=vae,
                     resolution="resize", **{"resolution.megapixels": 1.0},
                     **{f"images.image{i}": image for i, image in enumerate(images)})
        positive, negative = Graph.out(refs, 0), Graph.out(refs, 1)
        # Keep the explicit output canvas; references condition, rather than set, its size.
    samples = add("KSampler", model=model, positive=positive, negative=negative,
                  latent_image=latent, seed=seed, steps=steps, cfg=cfg,
                  sampler_name="euler", scheduler="simple", denoise=1.0)
    image = add("VAEDecode", samples=samples, vae=vae)
    add("SaveImage", images=image, filename_prefix=f"h3/image_staging/flux3_{output_stamp}_{output_nonce}")
    return g.nodes
