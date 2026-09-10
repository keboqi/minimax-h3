"""Bind layout actions."""

from __future__ import annotations

from ..contracts import AppComponents, AppServices
from ..job_bindings import bind_gpu_action


def bind_layout(components: AppComponents, services: AppServices) -> None:
    components.mode.change(
        services.mode_layout_updates,
        inputs=components.mode,
        outputs=[
            components.help_text,
            components.frame_group,
            components.reference_group,
            components.generation_mode,
            components.preset,
            components.steps,
            components.scheduler,
            components.cache_mode,
            components.attention_mode,
        ],
    )
    components.result_format.change(
        services.result_format_layout_updates,
        inputs=[
            components.result_format,
            components.width,
            components.height,
            components.first,
            components.latent_upscale,
        ],
        outputs=[
            components.duration,
            components.image_frames,
            components.image_vae,
            components.output,
            components.output_2,
            components.output_3,
            components.output_4,
            components.batch_count,
            components.image_output_group,
            components.audio_output,
            components.generation_postprocess,
            components.latent_upscale,
            components.width,
            components.height,
            components.resolution_info,
            components.run,
        ],
        queue=False,
        show_progress="hidden",
    )
    bind_gpu_action(
        components.trt_vae_compile.click,
        services.compile_trt_video_vae,
        outputs=components.status,
        show_progress="full",
    )
    components.image_vae.change(
        services.image_vae_frame_updates,
        inputs=components.image_vae,
        outputs=components.image_frames,
        queue=False,
        show_progress="hidden",
    )
    components.latent_upscale.change(
        services.latent_upscale_layout_updates,
        inputs=[
            components.latent_upscale,
            components.width,
            components.height,
            components.result_format,
        ],
        outputs=[
            components.latent_upscale_settings,
            components.width,
            components.height,
            components.resolution_info,
        ],
    )
    components.latent_upscale_method.change(
        services.latent_upscale_method_layout_update,
        inputs=components.latent_upscale_method,
        outputs=components.latent_split_settings,
        queue=False,
        show_progress="hidden",
    )
