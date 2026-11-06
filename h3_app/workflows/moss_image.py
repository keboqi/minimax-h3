"""MOSS-Image SFT and Base graphs on ComfyUI's native MOSS-Image support."""

import json

from h3_app.graph import Graph
from h3_models import (
    DEFAULT_MOSS_IMAGE_VAE,
    MODEL_SPECS,
    MOSS_IMAGE_MODEL_CHOICES,
    MOSS_IMAGE_TEXT_ENCODER_CHOICES,
    MOSS_IMAGE_TURBO_MODES,
    MOSS_IMAGE_VAE_CHOICES,
    moss_image_turbo_steps,
)
from .flux3_layout import parse_flux3_layout


# OpenMOSS reference sampling: CFG 3 with transformer layer 22 skipped in the
# negative pass only (``uncond_skip_layers=[22]``). Shift 5 is the model default.
MOSS_IMAGE_BASE_CFG = 3.0
MOSS_IMAGE_UNCOND_SKIP_LAYERS = "22"
MOSS_SPECTRUM_NODE = "SpectrumMOSSImageAdaptive"


def moss_spectrum_enabled(accelerate, turbo_variant, steps):
    # Upstream's eight warmup and two tail steps leave no forecasting window
    # for step-distilled LoRAs. Keep their trained schedule intact.
    return bool(accelerate) and turbo_variant == "Off" and steps > 10


def moss_image_model_keys(model_choice, text_encoder_choice, turbo_variant="Off", vae_choice=None):
    if model_choice not in MOSS_IMAGE_MODEL_CHOICES:
        raise ValueError("Unknown MOSS-Image model choice.")
    if text_encoder_choice not in MOSS_IMAGE_TEXT_ENCODER_CHOICES:
        raise ValueError("Unknown MOSS-Image text encoder.")
    vae_choice = DEFAULT_MOSS_IMAGE_VAE if vae_choice is None else vae_choice
    if vae_choice not in MOSS_IMAGE_VAE_CHOICES:
        raise ValueError("Unknown MOSS-Image VAE.")
    if turbo_variant != "Off" and turbo_variant not in MOSS_IMAGE_TURBO_MODES:
        raise ValueError(f"Unknown MOSS-Image Turbo LoRA: {turbo_variant}")
    if turbo_variant != "Off" and "_base_" in MOSS_IMAGE_MODEL_CHOICES[model_choice]:
        raise ValueError(f"{turbo_variant} is distilled from the SFT model; choose an SFT image model.")
    keys = (MOSS_IMAGE_MODEL_CHOICES[model_choice],
            MOSS_IMAGE_TEXT_ENCODER_CHOICES[text_encoder_choice], MOSS_IMAGE_VAE_CHOICES[vae_choice])
    if turbo_variant in MOSS_IMAGE_TURBO_MODES:
        keys += (MOSS_IMAGE_TURBO_MODES[turbo_variant][0],)
    return keys


def validate_moss_image_sampling(turbo_variant, steps, cfg, width=1024, height=1024):
    if turbo_variant == "Off":
        return
    if turbo_variant not in MOSS_IMAGE_TURBO_MODES:
        raise ValueError(f"Unknown MOSS-Image Turbo LoRA: {turbo_variant}")
    expected = moss_image_turbo_steps(turbo_variant, width, height)
    if int(steps) != expected:
        raise ValueError(f"{turbo_variant} requires exactly {expected} sampling steps at {width}x{height}.")
    if float(cfg) != 1.0:
        raise ValueError(f"{turbo_variant} requires CFG 1.")


def build_moss_image_graph(*, model_choice, text_encoder_choice, prompt, turbo_variant="Off",
                           vae_choice=DEFAULT_MOSS_IMAGE_VAE, width=1024, height=1024, seed=0, steps=50,
                           cfg=MOSS_IMAGE_BASE_CFG, layout="", accelerate=True,
                           output_stamp="0", output_nonce="moss"):
    moss_image_model_keys(model_choice, text_encoder_choice, turbo_variant, vae_choice)
    validate_moss_image_sampling(turbo_variant, steps, cfg, width, height)
    elements = parse_flux3_layout(layout)
    g = Graph()
    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))
    model = add("UNETLoader", unet_name=MODEL_SPECS[MOSS_IMAGE_MODEL_CHOICES[model_choice]].local_name,
                weight_dtype="default")
    if turbo_variant in MOSS_IMAGE_TURBO_MODES:
        model = add("LoraLoaderModelOnly", model=model, strength_model=1.0,
                    lora_name=MODEL_SPECS[MOSS_IMAGE_TURBO_MODES[turbo_variant][0]].local_name)
    guided = cfg > 1.0
    if guided:
        model = add("SkipLayerGuidanceDiTSimple", model=model, double_layers="",
                    single_layers=MOSS_IMAGE_UNCOND_SKIP_LAYERS, start_percent=0.0, end_percent=1.0)
    spectrum = moss_spectrum_enabled(accelerate, turbo_variant, steps)
    if spectrum:
        model = add(MOSS_SPECTRUM_NODE, model=model, warmup_steps=8,
                    detail_guard_steps=8, tail_actual_steps=2, post_guard_horizon_cap=6,
                    window_size=2.0, flex_window=0.5, cache_device="main_device",
                    debug=False, error_tolerance=1.0)
    clip = add("CLIPLoader", clip_name=MODEL_SPECS[MOSS_IMAGE_TEXT_ENCODER_CHOICES[text_encoder_choice]].local_name,
               type="moss_image", device="default")
    vae = add("VAELoader", vae_name=MODEL_SPECS[MOSS_IMAGE_VAE_CHOICES[vae_choice]].local_name)
    if elements:
        array = add("H3Flux3LayoutElements", layout=json.dumps(elements, ensure_ascii=False))
        prompt = add("MOSSImageLayoutPrompt", prompt=prompt, element=array, width=width, height=height,
                     character_boxes="")
    positive = add("CLIPTextEncode", clip=clip, text=prompt)
    # OpenMOSS uses an empty negative prompt; CFG 1 never evaluates it.
    negative = (add("CLIPTextEncode", clip=clip, text="") if guided
                else add("ConditioningZeroOut", conditioning=positive))
    latent = add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    samples = add("KSampler", model=model, positive=positive, negative=negative,
                  latent_image=latent, seed=seed, steps=steps, cfg=cfg,
                  sampler_name="euler", scheduler="sgm_uniform" if spectrum else "simple", denoise=1.0)
    image = add("VAEDecode", samples=samples, vae=vae)
    add("SaveImage", images=image, filename_prefix=f"h3/image_staging/moss_image_{output_stamp}_{output_nonce}")
    return g.nodes
