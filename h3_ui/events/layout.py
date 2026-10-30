"""Bind layout actions."""

from __future__ import annotations

from typing import TYPE_CHECKING
import gradio as gr
from h3_app.catalog import H3_REGENERATE_MODELS

from ..contracts import AppComponents

if TYPE_CHECKING:
    from ..contracts import BootstrapServices
from ..job_bindings import bind_gpu_action


def bind_layout(components: AppComponents, services: BootstrapServices) -> None:
    # Mode, result-format and latent-upscale presentation belong to the
    # settings transition and navigation responses; a separate listener
    # would repeat the round trip and race those updates.
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
    components.latent_upscale_method.change(
        lambda method: (
            gr.update(interactive=method not in H3_REGENERATE_MODELS),
            gr.update(interactive=method not in H3_REGENERATE_MODELS),
        ),
        inputs=components.latent_upscale_method,
        outputs=[components.latent_upscale_refine_steps, components.latent_upscale_refine_lora],
        queue=False,
        show_progress="hidden",
    )
