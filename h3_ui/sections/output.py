"""Build the output section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class OutputSection:
    auto_megapixels: gr.components.Component
    batch_count: gr.components.Component
    draft_resolution: gr.components.Component
    duration: gr.components.Component
    fast_resolution: gr.components.Component
    height: gr.components.Component
    image_frames: gr.components.Component
    large_resolution: gr.components.Component
    resolution_info: gr.components.Component
    seed: gr.components.Component
    steps: gr.components.Component
    width: gr.components.Component


def build_output_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    output_settings_section: gr.blocks.BlockContext,
) -> OutputSection:
    with output_settings_section:
        gr.Markdown("Choose the result length, quality target, canvas, and seed.")
        with gr.Row():
            duration = gr.Slider(
                2, 15, value=defaults["duration"], step=0.5, label="Seconds"
            )
            image_frames = gr.Slider(
                services.MIN_IMAGE_FRAMES,
                services.MAX_IMAGE_FRAMES,
                value=defaults["image_frames"],
                step=1,
                label="Image frames",
                visible=False,
                info=(
                    "The official VAE returns 1–20 decoded video frames. "
                    "Selecting the 500K decoder fixes this to one image."
                ),
            )
            steps = gr.Slider(
                4,
                30,
                value=defaults["steps"],
                step=1,
                label="Steps",
                info=(
                    "LightX2V 4-step is the default Turbo variant; Larry and the "
                    "8-step LightX2V variant keep their trained step counts. Increase Turbo "
                    "steps when a clip benefits from extra refinement; Normal H3 "
                    "presets normally use 15–20."
                ),
            )
        draft_resolution = gr.Dropdown(
            choices=list(services.DRAFT_RESOLUTIONS),
            value="16:9 · 1376×768",
            label="768p",
            info="768p sizes by aspect ratio.",
        )
        with gr.Accordion("More resolution presets", open=False):
            with gr.Row():
                fast_resolution = gr.Dropdown(
                    choices=list(services.FAST_RESOLUTIONS),
                    value=None,
                    label="1080p",
                    info="1080p sizes by aspect ratio, aligned to 32 pixels.",
                )
                large_resolution = gr.Dropdown(
                    choices=list(services.LARGE_RESOLUTIONS),
                    value=None,
                    label="2k",
                    info="1440p sizes by aspect ratio; needs more time and VRAM.",
                )
        with gr.Row():
            width = gr.Number(value=defaults["width"], precision=0, label="Width")
            height = gr.Number(value=defaults["height"], precision=0, label="Height")
            auto_megapixels = gr.Dropdown(
                choices=list(services.AUTO_RESOLUTION_MEGAPIXEL_PRESETS),
                value=services.DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
                label="Start-frame auto cap",
                info=(
                    "Maximum automatic resolution from the first frame; "
                    "manual sizes are unchanged."
                ),
            )
        resolution_info = gr.Markdown(
            services.resolution_summary(defaults["width"], defaults["height"])
        )
        with gr.Row():
            seed = gr.Number(
                value=defaults["seed"],
                precision=0,
                label="Seed",
                info=(
                    "Used for a single video. Batch videos always use "
                    "independent random seeds."
                ),
            )
            batch_count = gr.Slider(
                services.MIN_VIDEO_BATCH_COUNT,
                services.MAX_VIDEO_BATCH_COUNT,
                value=services.DEFAULT_VIDEO_BATCH_COUNT,
                step=1,
                label="Videos per batch",
                info="Generate up to four random-seed variants in one run.",
            )

    return OutputSection(
        auto_megapixels=auto_megapixels,
        batch_count=batch_count,
        draft_resolution=draft_resolution,
        duration=duration,
        fast_resolution=fast_resolution,
        height=height,
        image_frames=image_frames,
        large_resolution=large_resolution,
        resolution_info=resolution_info,
        seed=seed,
        steps=steps,
        width=width,
    )
