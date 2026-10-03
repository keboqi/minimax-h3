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
    aspect_ratio: gr.components.Component | None = None
    size_tier: gr.components.Component | None = None
    canvas_controls: tuple = ()
    canvas_preview: gr.Image | None = None
    canvas_status: gr.Markdown | None = None


def build_output_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    output_settings_section: gr.blocks.BlockContext,
) -> OutputSection:
    return build_workspace_output(defaults, services, output_settings_section)


def build_workspace_output(defaults, services, root):
    with root:
        gr.Markdown("### Output essentials")
        with gr.Row():
            duration = gr.Slider(
                2, 15, value=defaults["duration"], step=0.5, label="Seconds"
            )
            batch_count = gr.Slider(
                services.MIN_VIDEO_BATCH_COUNT,
                services.MAX_VIDEO_BATCH_COUNT,
                value=services.DEFAULT_VIDEO_BATCH_COUNT,
                step=1,
                label="Variations",
                info="Multiple videos use independent random seeds.",
            )
        with gr.Row():
            aspect_ratio = gr.Dropdown(
                ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"],
                value="16:9",
                label="Aspect ratio",
            )
            size_tier = gr.Dropdown(
                [
                    ("768p", "draft"),
                    ("1080p", "fast"),
                    ("2K", "large"),
                    ("Exact dimensions", "custom"),
                ],
                value="custom",
                label="Size",
            )
        resolution_info = gr.Markdown(
            services.resolution_summary(defaults["width"], defaults["height"])
        )
        with gr.Accordion("Exact dimensions, seed & image frames", open=False):
            with gr.Row():
                width = gr.Number(value=defaults["width"], precision=0, label="Width")
                height = gr.Number(
                    value=defaults["height"], precision=0, label="Height"
                )
            auto_megapixels = gr.Dropdown(
                list(services.AUTO_RESOLUTION_MEGAPIXEL_PRESETS),
                value=services.DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
                label="Start-frame auto cap",
                info="Caps automatic video size. Conditioned image output uses the first frame's aligned native dimensions.",
            )
            seed = gr.Number(
                value=defaults["seed"],
                precision=0,
                label="Seed",
                info="-1 chooses a random seed. Multiple videos always use independent seeds.",
            )
            image_frames = gr.Slider(
                services.MIN_IMAGE_FRAMES,
                services.MAX_IMAGE_FRAMES,
                value=defaults["image_frames"],
                step=1,
                label="Image frames",
                visible=False,
            )
        from h3_app.image_canvas import MODES

        with gr.Accordion("Conditioned image canvas (optional)", open=False):
            gr.Markdown(
                "Applies only to Image → First / last frame. Originals are preserved. "
                "Fit adds black padding; fill crops the center. Both frames use the same canvas."
            )
            canvas_mode = gr.Dropdown(
                MODES, value="Input-derived", label="Conditioned image canvas"
            )
            with gr.Row():
                canvas_width = gr.Number(
                    value=1024, precision=0, label="Image canvas width"
                )
                canvas_height = gr.Number(
                    value=1024, precision=0, label="Image canvas height"
                )
            canvas_preview = gr.Image(
                label="Working image canvas preview",
                interactive=False,
                visible=False,
                height=240,
            )
            canvas_status = gr.Markdown()
        with gr.Accordion("Sampling steps (advanced)", open=False):
            steps = gr.Slider(4, 30, value=defaults["steps"], step=1, label="Steps")
        draft_resolution = gr.Dropdown(
            list(services.DRAFT_RESOLUTIONS), value=None, label="768p", visible=False
        )
        fast_resolution = gr.Dropdown(
            list(services.FAST_RESOLUTIONS), value=None, label="1080p", visible=False
        )
        large_resolution = gr.Dropdown(
            list(services.LARGE_RESOLUTIONS), value=None, label="2k", visible=False
        )
    return OutputSection(
        auto_megapixels,
        batch_count,
        draft_resolution,
        duration,
        fast_resolution,
        height,
        image_frames,
        large_resolution,
        resolution_info,
        seed,
        steps,
        width,
        aspect_ratio,
        size_tier,
        (canvas_mode, canvas_width, canvas_height),
        canvas_preview,
        canvas_status,
    )
