"""Resolve the shared native H3 controls for existing-video refinement."""

from h3_app.catalog import (
    UI_DEFAULTS, H3_REGENERATE_MODELS, SAME_REFINEMENT_LORA,
    REFINEMENT_LORA_SETTINGS, PIKA_FLEX,
)
from h3_app.errors import H3Error
from h3_app.policy import (
    h3_latent_upscaler_settings, resolve_h3_latent_upscale_method,
    resolve_h3_split_upscale_config, normalize_turbo_variant, turbo_steps_for,
)


H3_REFINEMENT_INPUTS = (
    "latent_upscale_method", "latent_upscaler_model", "latent_upscale_refine_steps",
    "latent_upscale_refine_lora", "latent_split_tile_width", "latent_split_tile_height",
    "latent_split_overlap_ratio", "latent_split_fade_ratio", "latent_split_chunk_frames",
    "latent_split_temporal_overlap_frames", "latent_split_seam_denoise", "latent_split_seam_polish",
    "model_profile", "generation_mode", "turbo_variant", "steps", "scheduler",
)


def resolve_refinement(values, models):
    settings = {**UI_DEFAULTS, **(values or {})}
    method = resolve_h3_latent_upscale_method(settings["latent_upscale_method"])
    variant = settings["latent_upscale_refine_lora"]
    if method in H3_REGENERATE_MODELS:
        variant = method
    elif variant == SAME_REFINEMENT_LORA:
        variant = normalize_turbo_variant(settings["turbo_variant"]) if settings["generation_mode"] == "Turbo" else "Off"
    elif variant not in REFINEMENT_LORA_SETTINGS:
        raise H3Error(f"Unknown refinement LoRA: {variant}")
    regenerate = variant in H3_REGENERATE_MODELS
    lora = None if regenerate or variant == "Off" else models.turbo_lora_for("Text to video", variant)
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
    split = resolve_h3_split_upscale_config(method, **{
        name: settings[f"latent_split_{name}"] for name in (
            "tile_width", "tile_height", "overlap_ratio", "fade_ratio", "chunk_frames",
            "temporal_overlap_frames", "seam_denoise", "seam_polish",
        )
    })
    return dict(settings=settings, method=method, variant=variant, lora=lora,
                upscaler_key=key, model_name=model_name, precision=precision,
                steps=steps, total=total, split=split)
