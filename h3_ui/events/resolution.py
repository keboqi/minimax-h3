"""Bind resolution actions."""

from __future__ import annotations

import gradio as gr
from h3_app.catalog import AI_POSTPROCESS_OPTIONS, LTX25_TILED_UPSCALE, LTX25_UPSCALE, SEEDVR2_UPSCALE

from typing import TYPE_CHECKING

from ..contracts import AppComponents

if TYPE_CHECKING:
    from ..bootstrap import BootstrapServices
from ..settings_controller import SettingsController


def bind_resolution(
    components: AppComponents, services: BootstrapServices, controller: SettingsController
) -> None:
    components.generation_postprocess.change(
        lambda value: (
            gr.update(visible=value in AI_POSTPROCESS_OPTIONS),
            gr.update(visible=value == SEEDVR2_UPSCALE),
            gr.update(
                visible=value in {LTX25_UPSCALE, LTX25_TILED_UPSCALE},
                value=(
                    "Uses spatial tiles and temporal streaming. The detail-refinement adapter "
                    "downloads on first use. Uses a neutral detail prompt; segment splitting is an advanced fallback."
                    if value == LTX25_TILED_UPSCALE else
                    "Uses the finishing model selected here and the H3 generation prompt. "
                    "The gated 2x IC-LoRA downloads on first use."
                ),
            ),
            gr.update(visible=value in {LTX25_UPSCALE, LTX25_TILED_UPSCALE}),
            gr.update(visible=value in {LTX25_UPSCALE, LTX25_TILED_UPSCALE}),
            gr.update(visible=value in {LTX25_UPSCALE, LTX25_TILED_UPSCALE}),
        ),
        inputs=components.generation_postprocess,
        outputs=[
            components.generation_postprocess_settings,
            components.generation_seedvr2_model,
            components.generation_ltx25_note,
            components.generation_split_upscale,
            components.generation_split_seconds,
            components.ltx25_model,
        ],
        queue=False,
        show_progress="hidden",
    )
    # Each dimension edit resolves its alignment and the authoritative settings
    # in one response. A chained refresh would read browser controls again and
    # could observe the previous dimensions while the first update commits.
    def dimensions(width, height, info, **settings):
        return {"width": width, "height": height, **settings}, (info,)

    for tier, choice in (
        ("draft", components.draft_resolution),
        ("fast", components.fast_resolution),
        ("large", components.large_resolution),
    ):
        controller.bind_refresh(
            [choice.change],
            lambda name, latent_upscale, result_format, tier=tier: dimensions(
                *services.resolution_choice_updates(
                    name, tier, latent_upscale, result_format
                )
            ),
            inputs=[choice, components.latent_upscale, components.result_format],
            extra_outputs=[components.resolution_info],
        )

    def start_frame(first, width, height, result_format, latent_upscale, cap):
        return dimensions(
            *services.auto_resolution_from_start_frame(
                first, width, height, result_format, latent_upscale, cap
            )
        )

    start_frame_inputs = [
        components.first,
        components.width,
        components.height,
        components.result_format,
        components.latent_upscale,
        components.auto_megapixels,
    ]
    controller.bind_refresh(
        [components.first.change],
        start_frame,
        inputs=start_frame_inputs,
        extra_outputs=[components.resolution_info],
    )

    # The cap's .change also responds to preset updates. Writing the cap back
    # from this response would retrigger it indefinitely in Gradio. Without a
    # start frame the cap changes no dimension, and the preset or cap edit's
    # own settings response already describes it.
    def start_frame_cap(first, width, height, result_format, latent_upscale, cap):
        if not first:
            return None
        overrides, extras = start_frame(
            first, width, height, result_format, latent_upscale, cap
        )
        return {**overrides, "auto_megapixels": cap}, extras

    controller.bind_refresh(
        [components.auto_megapixels.change],
        start_frame_cap,
        inputs=start_frame_inputs,
        extra_outputs=[components.resolution_info],
        exclude=[components.auto_megapixels],
    )
    for dimension in (components.width, components.height):
        dimension.input(
            services.resolution_info_preview,
            inputs=[
                components.width,
                components.height,
                components.latent_upscale,
                components.result_format,
            ],
            outputs=[components.resolution_info],
            queue=False,
            trigger_mode="always_last",
            show_progress="hidden",
        )
    # Number input dispatches before its frontend binding commits; blur and
    # submit observe the edited value without programmatic echoes.
    controller.bind_refresh(
        [
            components.width.blur,
            components.height.blur,
            components.width.submit,
            components.height.submit,
        ],
        lambda width, height, latent_upscale, result_format: dimensions(
            *services.resolution_control_updates(
                width, height, latent_upscale, result_format
            )
        ),
        inputs=[
            components.width,
            components.height,
            components.latent_upscale,
            components.result_format,
        ],
        extra_outputs=[components.resolution_info],
    )
    controller.bind_refresh(
        [components.latent_upscale.input],
        lambda latent_upscale, width, height, result_format: dimensions(
            *services.latent_upscale_layout_updates(
                latent_upscale, width, height, result_format
            )[1:],
            latent_upscale=latent_upscale,
        ),
        inputs=[
            components.latent_upscale,
            components.width,
            components.height,
            components.result_format,
        ],
        extra_outputs=[components.resolution_info],
    )
