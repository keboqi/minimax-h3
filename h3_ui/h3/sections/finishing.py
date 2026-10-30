"""Build the finishing section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr
from h3_app.catalog import REFINEMENT_LORA_CHOICES

if TYPE_CHECKING:
    from ..services import H3ViewServices


@dataclass(frozen=True)
class FinishingSection:
    ltx25_model: gr.components.Component
    generation_force_offload: gr.components.Component
    generation_ltx25_note: gr.components.Component
    generation_postprocess: gr.components.Component
    generation_postprocess_settings: gr.components.Component
    generation_seedvr2_model: gr.components.Component
    generation_upscale_resolution: gr.components.Component
    latent_upscale: gr.components.Component
    latent_upscale_method: gr.components.Component
    latent_upscale_refine_lora: gr.components.Component
    latent_upscale_refine_steps: gr.components.Component
    latent_upscale_settings: gr.components.Component
    latent_upscaler_model: gr.components.Component


def build_finishing_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    finishing_section: gr.blocks.BlockContext,
) -> FinishingSection:
    with finishing_section:
        gr.Markdown("**Native latent upscale**")
        latent_upscale = gr.Checkbox(
            value=defaults["latent_upscale"],
            label="Generate at half resolution, then latent upscale 2x",
            info=(
                "Runs inside H3 sampling, not after video generation. Width and "
                "height remain the H3 output resolution. Acceleration is disabled while this is enabled."
            ),
        )
        with gr.Group(visible=defaults["latent_upscale"]) as latent_upscale_settings:
            latent_upscaler_model = gr.Dropdown(
                choices=list(services.H3_LATENT_UPSCALER_MODEL_CHOICES),
                value=defaults["latent_upscaler_model"],
                label="Latent upscaler model",
                info=(
                    "Balanced uses BF16 and is the default. Fast uses FP16; "
                    "Quality uses FP32 and needs more memory. Downloaded on first use."
                ),
            )
            latent_upscale_refine_steps = gr.Slider(
                1,
                6,
                value=defaults["latent_upscale_refine_steps"],
                step=1,
                label="High-resolution refinement steps",
                info=(
                    "H3 first finishes all generation steps at half resolution. "
                    "The clean 2x latent is then lightly re-noised and refined. "
                    "Two expensive high-resolution steps is the default."
                ),
            )
            latent_upscale_refine_lora = gr.Dropdown(
                choices=list(REFINEMENT_LORA_CHOICES),
                value=defaults["latent_upscale_refine_lora"],
                label="High-resolution refinement LoRA",
                info=(
                    "Same as generation preserves the generation LoRA. Choose a different "
                    "LoRA for just the high-resolution pass; downloaded on first use. "
                    "LoRA step labels describe training, not the number of refinement steps. "
                    "Regenerate methods use their own adapter stack and three-step schedule. "
                    "PDMD and DMAD are experimental."
                ),
            )
            latent_upscale_method = gr.Dropdown(
                choices=list(services.H3_LATENT_UPSCALE_METHODS),
                value=defaults["latent_upscale_method"],
                label="High-resolution refinement method",
                info=(
                    "Full-frame is the normal path. "
                    "Regenerate rank 16/384 uses a dedicated refiner plus LightX2V "
                    "with three Euler steps, overriding refinement steps and LoRA above. "
                    "Rank 16 downloads 155 MB; rank 384 downloads 5.87 GB. "
                    "Both are experimental; LMS Sharp is not included."
                ),
            )
            gr.Markdown(
                "Final width and height must both be divisible by 64. For example, "
                "1024×1024 generates the first stage at 512×512 and finishes at "
                "1024×1024. Only the video latent is upscaled; H3 audio is preserved."
            )

        gr.Markdown("**After generation**")
        generation_postprocess = gr.Dropdown(
            choices=services.GENERATION_POSTPROCESS_OPTIONS,
            value=defaults["postprocess"],
            label="After generation",
            info=(
                "Optionally run SeedVR2, H3 Regenerate or LTX-2.5 upscaling immediately after the base H3 "
                "video finishes. The source video remains in the gallery."
            ),
        )
        with gr.Group(visible=False) as generation_postprocess_settings:
            generation_upscale_resolution = gr.Dropdown(
                choices=list(services.UPSCALE_RESOLUTION_PRESETS),
                value=services.DEFAULT_UPSCALE_RESOLUTION,
                label="Output resolution",
                info="Fits the source inside the selected square while preserving aspect ratio.",
            )
            generation_seedvr2_model = gr.Dropdown(
                choices=list(services.SEEDVR2_MODEL_CHOICES),
                value=defaults["seedvr2_model"],
                label="SeedVR2 model",
                info=(
                    "Downloaded on first use. 7B INT8 is the default quality/VRAM "
                    "balance; FP16 favors fidelity, 7B Sharp favors stronger detail, "
                    "and MXFP8/NVFP4 are experimental speed options."
                ),
            )
            ltx25_model = gr.Dropdown(
                choices=list(services.LTX25_MODEL_CHOICES),
                value=services.DEFAULT_LTX25_MODEL,
                label="Generation finishing LTX model",
                visible=False,
                info="Used for H3 finishing independently of the LTX generation workspace.",
            )
            generation_ltx25_note = gr.Markdown(
                "Uses the finishing model selected here and "
                "the H3 generation prompt. The gated 2x IC-LoRA downloads "
                "on first use.",
                visible=False,
            )
            generation_force_offload = gr.Checkbox(
                value=defaults["upscale_force_offload"],
                label="Unload H3 models before upscaling",
                info=(
                    "Reduces peak VRAM at the cost of reloading H3 for "
                    "the next generation."
                ),
            )
    return FinishingSection(
        ltx25_model=ltx25_model,
        generation_force_offload=generation_force_offload,
        generation_ltx25_note=generation_ltx25_note,
        generation_postprocess=generation_postprocess,
        generation_postprocess_settings=generation_postprocess_settings,
        generation_seedvr2_model=generation_seedvr2_model,
        generation_upscale_resolution=generation_upscale_resolution,
        latent_upscale=latent_upscale,
        latent_upscale_method=latent_upscale_method,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_upscale_refine_lora=latent_upscale_refine_lora,
        latent_upscale_settings=latent_upscale_settings,
        latent_upscaler_model=latent_upscaler_model,
    )
