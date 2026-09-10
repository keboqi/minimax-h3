"""Bind resolution actions."""

from __future__ import annotations

import gradio as gr

from ..contracts import AppComponents, AppServices
from ..settings_controller import SettingsController


def bind_resolution(
    components: AppComponents, services: AppServices, controller: SettingsController
) -> None:
    components.generation_postprocess.change(
        lambda value: (
            gr.update(visible=value in services.AI_POSTPROCESS_OPTIONS),
            gr.update(visible=value == services.SEEDVR2_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
            gr.update(visible=value == services.LTX25_UPSCALE),
        ),
        inputs=components.generation_postprocess,
        outputs=[
            components.generation_postprocess_settings,
            components.generation_seedvr2_model,
            components.generation_ltx25_note,
            components.generation_split_upscale,
            components.generation_split_seconds,
        ],
        queue=False,
        show_progress="hidden",
    )
    draft_resolution_event = components.draft_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "draft", latent_upscale, result_format
        ),
        inputs=[
            components.draft_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    fast_resolution_event = components.fast_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "fast", latent_upscale, result_format
        ),
        inputs=[
            components.fast_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    large_resolution_event = components.large_resolution.change(
        lambda name, latent_upscale, result_format: services.resolution_choice_updates(
            name, "large", latent_upscale, result_format
        ),
        inputs=[
            components.large_resolution,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    for resolution_event in (
        draft_resolution_event,
        fast_resolution_event,
        large_resolution_event,
    ):
        resolution_event.then(
            controller.refresh,
            inputs=[controller.memory, *controller.inputs],
            outputs=controller.outputs,
            queue=False,
            show_progress="hidden",
        )
    components.first.upload(
        fn=None,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        js=services.AUTO_RESOLUTION_JS,
        queue=False,
        show_progress="hidden",
    )
    first_change_event = components.first.change(
        fn=services.auto_resolution_from_start_frame,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        queue=False,
        show_progress="hidden",
    )
    first_change_event.then(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
    )
    auto_megapixels_change = components.auto_megapixels.input(
        fn=services.auto_resolution_from_start_frame,
        inputs=[
            components.first,
            components.width,
            components.height,
            components.result_format,
            components.latent_upscale,
            components.auto_megapixels,
        ],
        outputs=[
            components.width,
            components.height,
            components.resolution_info,
        ],
        queue=False,
        show_progress="hidden",
    )
    auto_megapixels_change.then(
        controller.refresh,
        inputs=[controller.memory, *controller.inputs],
        outputs=controller.outputs,
        queue=False,
        show_progress="hidden",
    )
    components.width.input(
        services.resolution_info_preview,
        inputs=[
            components.width,
            components.height,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[components.resolution_info],
        queue=False,
        show_progress="hidden",
    )
    components.height.input(
        services.resolution_info_preview,
        inputs=[
            components.width,
            components.height,
            components.latent_upscale,
            components.result_format,
        ],
        outputs=[components.resolution_info],
        queue=False,
        show_progress="hidden",
    )
    for res_event_trigger in (
        components.width.blur,
        components.height.blur,
        components.width.submit,
        components.height.submit,
    ):
        res_snap_event = res_event_trigger(
            services.resolution_control_updates,
            inputs=[
                components.width,
                components.height,
                components.latent_upscale,
                components.result_format,
            ],
            outputs=[
                components.width,
                components.height,
                components.resolution_info,
            ],
            queue=False,
            show_progress="hidden",
        )
        res_snap_event.then(
            controller.refresh,
            inputs=[controller.memory, *controller.inputs],
            outputs=controller.outputs,
            queue=False,
            show_progress="hidden",
        )
