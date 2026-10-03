"""Bind prompt actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_prompt_action
from ..workspace_mode import workspace_enabled
from ..prompt_preview import bind_prompt_preview
import gradio as gr


def bind_prompt(components: AppComponents, services: AppServices) -> None:
    components.prompt_writer_backend.change(
        services.prompt_writer_backend_visibility,
        inputs=components.prompt_writer_backend,
        outputs=[
            components.local_prompt_writer_group,
            components.gemini_prompt_writer_group,
            components.lightning_prompt_writer_group,
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    inputs = [
        components.prompt,
        components.prompt_writer_backend,
        components.local_prompt_base_model,
        components.local_prompt_max_tokens,
        components.local_prompt_temperature,
        components.local_prompt_top_p,
        components.local_prompt_greedy,
        components.local_prompt_seed,
        components.gemini_prompt_model,
        components.gemini_api_key,
        components.lightning_api_key,
        components.mode,
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
        components.ref_video_1,
        components.ref_video_2,
        components.ref_video_3,
        components.ref_audio_1,
        components.ref_audio_2,
        components.ref_audio_3,
        components.duration,
        components.width,
        components.height,
        components.result_format,
        components.image_frames,
        components.fl2va_audio_1,
        components.fl2va_audio_2,
        components.fl2va_audio_3,
    ]
    api_trigger = (
        gr.Button(visible=False).click
        if workspace_enabled()
        else components.enhance_prompt_button.click
    )
    bind_prompt_action(
        api_trigger,
        services.enhance_h3_prompt,
        inputs=inputs,
        outputs=[components.prompt, components.enhance_prompt_status],
        show_progress="minimal",
        api_name="enhance_prompt",
    )

    if workspace_enabled():
        bind_prompt_preview(
            components,
            services.enhance_h3_prompt,
            [
                "prompt",
                "prompt_writer_backend",
                "local_prompt_base_model",
                "local_prompt_max_tokens",
                "local_prompt_temperature",
                "local_prompt_top_p",
                "local_prompt_greedy",
                "local_prompt_seed",
                "gemini_prompt_model",
                "gemini_api_key",
                "lightning_api_key",
                "mode",
                "first",
                "last",
                "ref_image_1",
                "ref_image_2",
                "ref_image_3",
                "ref_image_4",
                "ref_image_5",
                "ref_image_6",
                "ref_image_7",
                "ref_image_8",
                "ref_image_9",
                "ref_video_1",
                "ref_video_2",
                "ref_video_3",
                "ref_audio_1",
                "ref_audio_2",
                "ref_audio_3",
                "duration",
                "width",
                "height",
                "result_format",
                "image_frames",
                "fl2va_audio_1",
                "fl2va_audio_2",
                "fl2va_audio_3",
            ],
            inputs,
        )
