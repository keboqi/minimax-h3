"""Build the performance section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class PerformanceSection:
    attention_mode: gr.components.Component
    cache_mode: gr.components.Component
    easycache_end: gr.components.Component
    easycache_settings: gr.components.Component
    easycache_start: gr.components.Component
    easycache_threshold: gr.components.Component
    easycache_verbose: gr.components.Component
    fbcache_end: gr.components.Component
    fbcache_max_hits: gr.components.Component
    fbcache_preset: gr.components.Component
    fbcache_settings: gr.components.Component
    fbcache_start: gr.components.Component
    fbcache_temporal_guard: gr.components.Component
    fbcache_threshold: gr.components.Component
    scheduler: gr.components.Component
    sla_preset: gr.components.Component
    sla_settings: gr.components.Component
    sol_dense_steps: gr.components.Component
    sol_exact_mode: gr.components.Component
    sol_quality_settings: gr.components.Component
    sol_settings: gr.components.Component
    sol_sink_tokens: gr.components.Component
    sol_step_off: gr.components.Component
    sol_tau: gr.components.Component
    sol_thresh_type: gr.components.Component
    turbo_variant: gr.components.Component


def build_performance_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    performance_section: gr.blocks.BlockContext,
) -> PerformanceSection:
    with performance_section:
        gr.Markdown(
            "Tune Turbo, attention, and caching. Defaults are recommended for most jobs."
        )
        turbo_variant = gr.Radio(
            list(services.TURBO_SETTINGS),
            value=defaults["turbo_variant"],
            label="Turbo implementation",
            info=(
                "Choose the Turbo adapter. Each variant supplies its trained step count; "
                "you can then adjust the number of steps."
            ),
        )
        scheduler = gr.Radio(
            ["simple", "beta", "normal"],
            value=defaults["scheduler"],
            label="Scheduler",
            info="Controls how sampling steps are distributed across denoising.",
        )
        attention_mode = gr.Radio(
            ["Sage 2", "Kitchen", "SLA", "Sol-Attn", "Auto"],
            value=defaults["attention_mode"],
            label="Attention",
            interactive=services.SERVER_ATTENTION_BACKEND == "sol",
            info=(
                f"Auto enables Sol-Attn for Reference mode or when estimated "
                f"packed target tokens reach {services.AUTO_SOL_TOKEN_THRESHOLD:,}; "
                "Sage 2 applies the pinned KJNodes model override. Kitchen "
                "selects the global ComfyUI backend. SLA is the default, uses "
                "the selected audio-safe block-sparse preset, and automatically "
                "keeps short sequences dense; it is intended for SLA-distilled "
                "H3 LoRAs. Auto uses Kitchen for "
                "smaller jobs. Sol "
                f"dense/fallback calls use {services.SERVER_DENSE_ATTENTION_BACKEND}."
            ),
        )
        with gr.Accordion(
            "SLA quality controls",
            open=False,
            visible=defaults["attention_mode"] == "SLA",
        ) as sla_settings:
            sla_preset = gr.Radio(
                list(services.SLA_PRESET_INPUTS),
                value=defaults["sla_preset"],
                label="SLA preset",
                info=(
                    "Fast uses validated 0.90 sparsity. Balanced uses the "
                    "LoRA-distilled 0.85 sparsity. Quality also runs the final "
                    "sampling step dense to recover fine detail; in a two-stage "
                    "latent-upscale workflow this applies to both stages. All "
                    "presets use 64-token blocks and protect audio."
                ),
            )

        with gr.Row(
            visible=defaults["attention_mode"] in {"Sol-Attn", "Auto"}
        ) as sol_settings:
            sol_tau = gr.Slider(
                0.5,
                1.5,
                value=defaults["sol_tau"],
                step=0.1,
                label="Sol-Attn tau",
            )
            sol_thresh_type = gr.Radio(
                ["diag", "exact"],
                value=defaults["sol_thresh_type"],
                label="Sol threshold",
                info="diag is faster; exact calculates a more precise routing threshold.",
            )
        with gr.Accordion(
            "Sol-Attn quality controls",
            open=False,
            visible=defaults["attention_mode"] in {"Sol-Attn", "Auto"},
        ) as sol_quality_settings:
            sol_exact_mode = gr.Radio(
                ["off", "exact_kv", "exact_kv_and_rows"],
                value=defaults["sol_exact_mode"],
                label="Exact H3 prefix mode",
                info=(
                    "exact_kv preserves text/condition/reference/audio KV "
                    "rows at low cost. exact_kv_and_rows also keeps prefix "
                    "query rows dense for maximum audio/conditioning fidelity."
                ),
            )
            with gr.Row():
                sol_dense_steps = gr.Slider(
                    0,
                    4,
                    value=defaults["sol_dense_steps"],
                    step=1,
                    label="Dense final transformer blocks",
                    info=(
                        "Keep the final N H3 transformer blocks dense. "
                        "The final block is the most approximation-sensitive."
                    ),
                )
            sol_step_off = gr.State(0.0)
            sol_sink_tokens = gr.State(0)
        with gr.Accordion("Sampling acceleration", open=False):
            cache_mode = gr.Radio(
                ["Spectrum", "FirstBlockCache", "EasyCache", "Off"],
                value=defaults["cache_mode"],
                label="Acceleration mode",
                info=(
                    "Spectrum is the normal H3 default based on broader community "
                    "speed testing. It forecasts selected transformer steps and uses "
                    "audio-isolated offline replay. FirstBlockCache is the lower-memory "
                    "fallback. Modes are mutually exclusive. Turbo defaults to Spectrum; "
                    "EasyCache and FirstBlockCache are opt-in experimental Turbo options."
                ),
            )
            with gr.Group(visible=False) as fbcache_settings:
                fbcache_preset = gr.Radio(
                    ["Safe", "Fast", "Aggressive", "Custom"],
                    value=defaults["fbcache_preset"],
                    label="FirstBlockCache preset",
                    info=(
                        "Fast is the recommended default. Named presets use "
                        "a protected 10–95% denoising window and at most two "
                        "consecutive cache hits."
                    ),
                )
                with gr.Row():
                    fbcache_threshold = gr.Slider(
                        0.0,
                        0.25,
                        value=defaults["fbcache_threshold"],
                        step=0.005,
                        label="FirstBlock threshold",
                        interactive=False,
                    )
                    fbcache_max_hits = gr.Slider(
                        1,
                        8,
                        value=defaults["fbcache_max_hits"],
                        step=1,
                        label="Max consecutive cache hits",
                        interactive=False,
                    )
                with gr.Row():
                    fbcache_start = gr.Slider(
                        0.0,
                        0.90,
                        value=defaults["fbcache_start"],
                        step=0.01,
                        label="Cache start percent",
                        interactive=False,
                    )
                    fbcache_end = gr.Slider(
                        0.10,
                        1.0,
                        value=defaults["fbcache_end"],
                        step=0.01,
                        label="Cache end percent",
                        interactive=False,
                    )
                fbcache_temporal_guard = gr.Checkbox(
                    value=defaults["fbcache_temporal_guard"],
                    label="Temporal frame guard",
                    info=(
                        "Checks the most-changed target-video latent frame "
                        "in addition to the global residual average."
                    ),
                )
            with gr.Group(visible=False) as easycache_settings:
                gr.Markdown("**EasyCache fallback settings**")
                easycache_threshold = gr.Slider(
                    0.0,
                    0.5,
                    value=defaults["easycache_threshold"],
                    step=0.01,
                    label="Reuse threshold",
                    info=(
                        "Higher skips more steps. Start at 0.10 for H3; "
                        "ComfyUI's generic default is 0.20."
                    ),
                )
                with gr.Row():
                    easycache_start = gr.Slider(
                        0.0,
                        0.9,
                        value=defaults["easycache_start"],
                        step=0.01,
                        label="Start percent",
                    )
                    easycache_end = gr.Slider(
                        0.1,
                        1.0,
                        value=defaults["easycache_end"],
                        step=0.01,
                        label="End percent",
                    )
                easycache_verbose = gr.Checkbox(
                    value=defaults["easycache_verbose"],
                    label="Log EasyCache decisions",
                    info="Logs skipped-step counts and estimated speedup in ComfyUI.",
                )

    return PerformanceSection(
        attention_mode=attention_mode,
        cache_mode=cache_mode,
        easycache_end=easycache_end,
        easycache_settings=easycache_settings,
        easycache_start=easycache_start,
        easycache_threshold=easycache_threshold,
        easycache_verbose=easycache_verbose,
        fbcache_end=fbcache_end,
        fbcache_max_hits=fbcache_max_hits,
        fbcache_preset=fbcache_preset,
        fbcache_settings=fbcache_settings,
        fbcache_start=fbcache_start,
        fbcache_temporal_guard=fbcache_temporal_guard,
        fbcache_threshold=fbcache_threshold,
        scheduler=scheduler,
        sla_preset=sla_preset,
        sla_settings=sla_settings,
        sol_dense_steps=sol_dense_steps,
        sol_exact_mode=sol_exact_mode,
        sol_quality_settings=sol_quality_settings,
        sol_settings=sol_settings,
        sol_sink_tokens=sol_sink_tokens,
        sol_step_off=sol_step_off,
        sol_tau=sol_tau,
        sol_thresh_type=sol_thresh_type,
        turbo_variant=turbo_variant,
    )
