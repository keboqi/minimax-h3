"""Bind layout actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        services.latent_upscale_method_layout_update,
        inputs=components.latent_upscale_method,
        outputs=components.latent_split_settings,
        queue=False,
        show_progress="hidden",
    )
