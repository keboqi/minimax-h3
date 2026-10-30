"""Resolve the shared native H3 controls for existing-video refinement."""

from h3_app.catalog import (
    UI_DEFAULTS, H3_REGENERATE_MODELS, SAME_REFINEMENT_LORA,
    REFINEMENT_LORA_SETTINGS, PIKA_FLEX,
)
from h3_app.errors import H3Error
from h3_app.contracts import GENERATION_FIELDS, GENERATION_COMPONENTS
from h3_app.policy import (
    h3_latent_upscaler_settings, resolve_h3_latent_upscale_method,
    normalize_turbo_variant, turbo_steps_for,
)


# The selected Media video supplies conditioning, timing and audio. Media supplies
# the target resolution. All applicable model/sampling controls come from H3.
_SOURCE_FIELDS = {"first_image", "last_image", "duration", "width", "height",
                  "result_format", "image_frames", "image_vae", "batch_count", "postprocess",
                  "upscale_resolution", "seedvr2_model", "ltx25_model",
                  "latent_upscale"}
H3_REFINEMENT_FIELDS = tuple(name for name in GENERATION_FIELDS
    if name not in _SOURCE_FIELDS and (name == "ref_image_size" or not name.startswith(("ref_image_", "ref_video_", "ref_audio_", "fl2va_audio_"))))
H3_REFINEMENT_INPUTS = tuple(dict(zip(GENERATION_FIELDS, GENERATION_COMPONENTS))[name]
                             for name in H3_REFINEMENT_FIELDS)


def refinement_settings(values):
    """Use H3's effective policy for an existing-video, native-refinement job."""
    from dataclasses import asdict
    from h3_app.settings import GenerationRequest, resolve_settings
    requested = {**UI_DEFAULTS, **(values or {}), "result_format": "Video", "latent_upscale": True}
    plan = resolve_settings(GenerationRequest.from_values(requested))
    if plan.issues:
        raise H3Error(" ".join(plan.issues))
    effective = plan.effective
    settings = {**requested, **asdict(effective.sampling), **asdict(effective.finishing),
                **{name: getattr(effective, name) for name in (
                    "cache_mode", "use_trt_vae", "use_int8_vae", "use_lynnreal_vae", "use_sol_vae",
                    "video_encoder", "lightvae_decoder", "semantic_bridge", "semantic_bridge_alpha")}}
    return settings


def resolve_refinement(values, models):
    settings = refinement_settings(values)
    from h3_app.model_types import h3_text_encoder_settings
    text_encoder = (h3_text_encoder_settings(models, values["text_encoder"])[1]
                    if values and "text_encoder" in values else models.text_encoder)
    method = resolve_h3_latent_upscale_method(settings["latent_upscale_method"])
    variant = settings["latent_upscale_refine_lora"]
    if method in H3_REGENERATE_MODELS:
        variant = method
    elif variant == SAME_REFINEMENT_LORA:
        variant = normalize_turbo_variant(settings["turbo_variant"]) if settings["generation_mode"] == "Turbo" else "Off"
    elif variant not in REFINEMENT_LORA_SETTINGS:
        raise H3Error(f"Unknown refinement LoRA: {variant}")
    regenerate = variant in H3_REGENERATE_MODELS
    lora = None if regenerate or variant == "Off" else models.turbo_lora_for(settings["mode"], variant)
    if variant != "Off" and not regenerate and not lora:
        raise H3Error(f"{variant} refinement LoRA is not configured.")
    key, model_name, precision = h3_latent_upscaler_settings(settings["latent_upscaler_model"])
    steps = 3 if regenerate else int(settings["latent_upscale_refine_steps"])
    generation_variant = (
        normalize_turbo_variant(settings["turbo_variant"])
        if settings["generation_mode"] == "Turbo" else None
    )
    total = 6 if variant == PIKA_FLEX else int(settings["steps"])
    if generation_variant == PIKA_FLEX and settings["latent_upscale_refine_lora"] != SAME_REFINEMENT_LORA and not regenerate:
        total = turbo_steps_for(variant)
    if not regenerate and not 1 <= steps < total:
        raise H3Error("Refinement steps must be smaller than the H3 sampling steps.")
    return dict(settings=settings, method=method, variant=variant, lora=lora, text_encoder=text_encoder,
                upscaler_key=key, model_name=model_name, precision=precision,
                steps=steps, total=total)
