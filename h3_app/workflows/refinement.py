"""Native H3 model, VAE and attention stack for existing-video refinement."""

import inspect
from h3_models import MODEL_SPECS

from h3_app.catalog import (
    H3_REGENERATE_MODELS, H3_CONDITIONING_CACHE_NODE, H3_SEMANTIC_BRIDGE_NODE,
    REFINEMENT_LORA_SETTINGS,
)
from h3_app.policy import resolve_sol_policy


def add_refinement_stack(graph, models, config, width, height):
    # Import at execution time: H3 generation also uses the Regenerate adapter helper.
    from .h3 import add_model_stack, h3_conditioning_video_vae, h3_refinement_attention_model, required_nodes_for

    settings = config["settings"]
    variant = config["variant"]
    profile = models.profile(settings["model_profile"])
    model_name = profile.ref2va if settings["mode"] == "Reference media" else profile.fl2va
    attention = str(settings["attention_mode"]).lower()
    sol, _, _ = resolve_sol_policy(
        settings["attention_mode"], settings["mode"], width, height,
        settings.get("_source_duration", 5), None, None,
        settings["generation_mode"] == "Turbo",
        attention_backend=settings.get("_attention_backend", "sol"),
    )
    flags = dict(use_sol=sol, use_sage=attention in {"sage", "sage 2", "sage2"},
                 use_sla=attention in {"sla", "sla attention", "sparse-linear"}, use_veda=attention == "veda")
    lora = config["lora"]
    available = required_nodes_for(
        settings["mode"], cache_mode=settings["cache_mode"], model_filename=model_name,
        use_turbo=bool(lora), turbo_variant=variant, turbo_lora_filename=lora or "", **flags,
    ) | {"LightVAELoader", "MiniMaxH3TRTVAELoader", H3_CONDITIONING_CACHE_NODE, H3_SEMANTIC_BRIDGE_NODE}
    kwargs = {name: settings[name] for name in inspect.signature(add_model_stack).parameters if name in settings}
    kwargs.update(
        turbo_lora_name=lora, turbo_variant=variant,
        turbo_strength=REFINEMENT_LORA_SETTINGS[variant].strength if lora else 1.0,
        text_encoder_name=config["text_encoder"], sampling_steps=config["steps"],
        available_nodes=available,
        veda_predictor=MODEL_SPECS["veda_r2va" if settings["mode"] == "Reference media" else "veda_t2va"].local_name,
        sol_step_off=settings.get("sol_step_off", 0),
        sol_sink_tokens=settings.get("sol_sink_tokens", 0), **flags,
    )
    model, clip, decoder, audio = add_model_stack(graph, model_name, models, **kwargs)
    if variant in H3_REGENERATE_MODELS:
        from .h3 import h3_refinement_lora_model
        # Preserve the same model patches while replacing only the adapter stack.
        model = h3_refinement_lora_model(graph, model, lora_name="regenerate", variant=variant,
                                        available_nodes=available)
    model = h3_refinement_attention_model(
        graph, model, refine_steps=config["steps"],
        veda_dense_tail=flags["use_veda"] and settings["sla_preset"] == "Quality",
    )
    encoder = h3_conditioning_video_vae(
        graph, models, decoder, use_trt_vae=settings["use_trt_vae"],
        has_visual_conditioning=True, video_encoder=settings["video_encoder"], available_nodes=available,
    )
    # The native refinement ladder is based on the generation model's sigma
    # shifts, even when a different adapter is selected for refinement.
    from h3_app.policy import normalize_turbo_variant, turbo_sigma_shifts
    from h3_app.graph import Graph
    from h3_app.catalog import H3_SIGMA_SHIFT_NODE
    base_ref = model
    while graph.nodes[base_ref[0]]["class_type"] != "UNETLoader":
        base_ref = graph.nodes[base_ref[0]]["inputs"]["model"]
    generation_variant = normalize_turbo_variant(settings["turbo_variant"]) if settings["generation_mode"] == "Turbo" else None
    generation_lora = models.turbo_lora_for(settings["mode"], generation_variant) if generation_variant else None
    shifts = turbo_sigma_shifts(generation_variant, generation_lora)
    schedule_model = base_ref if shifts is None else Graph.out(graph.add(
        H3_SIGMA_SHIFT_NODE, model=base_ref, shift_video=shifts[0], shift_audio=shifts[1]))
    return model, clip, encoder, decoder, audio, schedule_model
