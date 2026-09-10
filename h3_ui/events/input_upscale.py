"""Bind input upscale actions."""

from __future__ import annotations

import gradio as gr

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_gpu_action, owned_generation


def bind_input_upscale(
    components: AppComponents, services: AppServices
) -> gr.events.Dependency:
    components.input_upscale_frame_preset.change(
        services.input_image_frame_preset_updates,
        inputs=[
            components.input_upscale_frame_preset,
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
        ],
        outputs=[
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    input_upscale_event = bind_gpu_action(
        components.input_upscale_run.click,
        owned_generation(services.upscale_selected_input_images, "h3-input"),
        inputs=[
            components.input_upscale_slots,
            components.input_upscale_model,
            components.input_upscale_seed,
            components.input_upscale_force_offload,
            components.input_upscale_frame_width,
            components.input_upscale_frame_height,
            components.first,
            components.last,
            components.ref_image_1,
            components.ref_image_2,
            components.ref_image_3,
            components.ref_image_4,
            components.ref_image_5,
            components.ref_image_6,
            components.ref_image_7,
            components.ref_image_8,
            components.ref_image_9,
        ],
        outputs=[
            components.first,
            components.last,
            components.ref_image_1,
            components.ref_image_2,
            components.ref_image_3,
            components.ref_image_4,
            components.ref_image_5,
            components.ref_image_6,
            components.ref_image_7,
            components.ref_image_8,
            components.ref_image_9,
            components.input_upscale_downloads,
            components.input_upscale_status,
        ],
        show_progress="minimal",
        api_name="upscale_h3_input_images",
    )

    return input_upscale_event
