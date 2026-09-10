"""Build the input upscale section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class InputUpscaleSection:
    input_upscale_downloads: gr.components.Component
    input_upscale_force_offload: gr.components.Component
    input_upscale_frame_height: gr.components.Component
    input_upscale_frame_preset: gr.components.Component
    input_upscale_frame_width: gr.components.Component
    input_upscale_model: gr.components.Component
    input_upscale_run: gr.components.Component
    input_upscale_seed: gr.components.Component
    input_upscale_slots: gr.components.Component
    input_upscale_status: gr.components.Component


def build_input_upscale_section(
    defaults: Mapping[str, Any], services: H3ViewServices
) -> InputUpscaleSection:
    with gr.Accordion(
        "Upscale input images with SeedVR2",
        open=False,
        elem_classes=["h3-advanced-block"],
    ):
        gr.Markdown(
            "Select uploaded start/end frames or reference pictures, then "
            "fit smaller images upward into a target frame before generation. "
            "Aspect ratio is preserved and images already at or beyond the "
            "frame are not downscaled."
        )
        input_upscale_slots = gr.CheckboxGroup(
            choices=list(services.INPUT_IMAGE_UPSCALE_SLOTS),
            value=[],
            label="Images to upscale",
        )
        input_upscale_frame_preset = gr.Dropdown(
            choices=list(services.INPUT_IMAGE_FRAME_PRESETS),
            value=services.DEFAULT_INPUT_IMAGE_FRAME_PRESET,
            label="Target frame preset",
        )
        with gr.Row():
            input_upscale_frame_width = gr.Number(
                value=1920,
                precision=0,
                label="Frame width",
            )
            input_upscale_frame_height = gr.Number(
                value=1920,
                precision=0,
                label="Frame height",
            )
        with gr.Row():
            input_upscale_model = gr.Dropdown(
                choices=list(services.SEEDVR2_MODEL_CHOICES),
                value=defaults["seedvr2_model"],
                label="SeedVR2 model",
            )
            input_upscale_seed = gr.Number(
                value=-1,
                precision=0,
                label="Seed (-1 random)",
            )
        input_upscale_force_offload = gr.Checkbox(
            value=False,
            label="Unload resident models before input upscaling",
            info="Useful when H3 or another large model is already resident in VRAM.",
        )
        input_upscale_run = gr.Button(
            "Upscale selected inputs to frame", variant="secondary"
        )
        input_upscale_downloads = gr.File(
            label="Selected input image files",
            file_count="multiple",
            interactive=False,
        )
        input_upscale_status = gr.Markdown()

    return InputUpscaleSection(
        input_upscale_downloads=input_upscale_downloads,
        input_upscale_force_offload=input_upscale_force_offload,
        input_upscale_frame_height=input_upscale_frame_height,
        input_upscale_frame_preset=input_upscale_frame_preset,
        input_upscale_frame_width=input_upscale_frame_width,
        input_upscale_model=input_upscale_model,
        input_upscale_run=input_upscale_run,
        input_upscale_seed=input_upscale_seed,
        input_upscale_slots=input_upscale_slots,
        input_upscale_status=input_upscale_status,
    )
