"""Bind prompt actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_gpu_action


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
    bind_gpu_action(
        components.enhance_prompt_button.click,
        services.enhance_h3_prompt,
        inputs=[
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
        ],
        outputs=[components.prompt, components.enhance_prompt_status],
        show_progress="minimal",
        api_name="enhance_prompt",
    )
