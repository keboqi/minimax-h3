"""Bind generation actions."""

from __future__ import annotations

import gradio as gr

from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_gpu_action, owned_generation


def bind_generation(
    components: AppComponents, services: AppServices
) -> tuple[gr.events.Dependency, gr.events.Dependency]:
    generation_inputs = [
        components.batch_count,
        *(getattr(components, name) for name in GENERATION_COMPONENTS),
    ]
    generation_outputs = [
        components.output,
        components.output_2,
        components.output_3,
        components.output_4,
        components.image_output_group,
        components.image_output,
        components.image_selection,
        components.image_frame_paths,
        components.image_saved_files,
        components.audio_output,
        components.image_save_status,
        components.status,
    ]
    event = bind_gpu_action(
        components.run.click,
        owned_generation(
            services.generate_for_ui,
            "h3",
            ("batch_count", *GENERATION_FIELDS, "preset"),
            metadata_output=True,
        ),
        inputs=[*generation_inputs, components.preset],
        outputs=[*generation_outputs, components.settings_used],
        show_progress="minimal",
        api_name=False,
    )
    advanced_api_event = bind_gpu_action(
        gr.Button(visible=False).click,
        owned_generation(
            services.generate_for_ui, "h3", ("batch_count", *GENERATION_FIELDS)
        ),
        inputs=generation_inputs,
        outputs=generation_outputs,
        show_progress="minimal",
        api_name="generate_video_advanced",
    )

    return event, advanced_api_event
