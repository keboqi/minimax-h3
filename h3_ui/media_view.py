"""Typed builders for self-contained application views."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import gradio as gr


@dataclass(frozen=True)
class GalleryView:
    mode: gr.Radio
    refresh: gr.Button
    status: gr.Markdown
    paths: gr.State
    selected: gr.State
    upload_video: gr.File
    import_video: gr.Button
    grid: gr.Gallery
    shown: gr.State
    show_more: gr.Button
    manage: gr.Accordion
    confirm_delete: gr.Checkbox
    delete: gr.Button
    empty: gr.Button
    player: gr.Video
    image: gr.Image
    audio: gr.Audio
    download: gr.Markdown
    enhance: gr.Accordion
    postprocess: gr.Dropdown
    upscale_resolution: gr.Dropdown
    ai_settings: gr.Group
    seedvr2_model: gr.Dropdown
    ltx25_prompt: gr.Textbox
    post_seed: gr.Number
    force_offload: gr.Checkbox
    split_upscale: gr.Checkbox
    split_seconds: gr.Slider
    post_run: gr.Button
    post_stop: gr.Button
    post_status: gr.Markdown
    ltx25_model: gr.Dropdown
    deletion_confirmation: Any = None
    deletion_message: Any = None
    deletion_confirm: Any = None
    deletion_cancel: Any = None
    deletion_intent: Any = None


def build_gallery_view(
    root: gr.Group,
    *,
    postprocess_options: Sequence[str],
    resolution_choices: Sequence[str],
    default_resolution: str,
    seedvr2_choices: Sequence[str],
    default_seedvr2: str,
    ltx25_choices: Sequence[str] = (),
    default_ltx25: str | None = None,
) -> GalleryView:
    with root:
        gr.Markdown(
            "## Media gallery\nBrowse generated videos, images, or audio and import local media. Video and image outputs can also be enhanced.",
            elem_classes=["h3-gallery-heading"],
        )
        with gr.Row(equal_height=True, elem_classes=["h3-gallery-toolbar"]):
            mode = gr.Radio(
                choices=["Video", "Image", "Audio"],
                value="Video",
                label="Gallery type",
                scale=0,
                min_width=180,
            )
            refresh = gr.Button(
                "Refresh library", variant="secondary", scale=0, min_width=150
            )
            status = gr.Markdown(
                "Open this tab to scan generated videos.",
                elem_classes=["h3-gallery-status"],
            )
        paths = gr.State([])
        selected = gr.State(None)
        with gr.Accordion(
            "Import local media", open=False, elem_classes=["h3-gallery-card"]
        ):
            gr.Markdown(
                "Add an existing video, image, or audio file to the active library so it can be previewed alongside generated outputs."
            )
            with gr.Row(equal_height=True, elem_classes=["h3-gallery-import"]):
                upload_video = gr.File(
                    label="Choose a video, image, or audio file",
                    file_count="single",
                    file_types=["video", "image", "audio"],
                    type="filepath",
                    height=90,
                    scale=4,
                )
                import_video = gr.Button(
                    "Add to library", variant="primary", scale=0, min_width=170
                )
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-workspace"]):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown(
                    "### Library\nSelect an item to load the full video, image, or audio output.",
                    elem_classes=["h3-gallery-section-title"],
                )
                grid = gr.Gallery(
                    value=[],
                    label="Media library",
                    columns=3,
                    height=620,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="generated-video-gallery",
                    elem_classes=["h3-gallery-grid"],
                )
                shown = gr.State(48)
                show_more = gr.Button("Show more", interactive=False)
                with gr.Accordion(
                    "Manage library",
                    open=False,
                    elem_classes=["h3-gallery-card", "h3-gallery-danger"],
                ) as manage:
                    confirm_delete = gr.Checkbox(
                        value=False,
                        label="I understand deletion is permanent",
                        visible=False,
                        info="Required before deleting the selected item or emptying the generated library.",
                    )
                    with gr.Row(
                        equal_height=True, elem_classes=["h3-gallery-danger-actions"]
                    ):
                        delete = gr.Button("Delete selected", variant="stop")
                        empty = gr.Button("Empty generated library", variant="stop")
                    with gr.Group(visible=False) as deletion_confirmation:
                        deletion_message = gr.Markdown()
                        with gr.Row():
                            deletion_confirm = gr.Button(
                                "Confirm permanent deletion", variant="stop"
                            )
                            deletion_cancel = gr.Button("Keep media")
                        deletion_intent = gr.State(None)
            with gr.Column(scale=5, min_width=480):
                gr.Markdown(
                    "### Preview & enhance\nReview the selected item, download it, or create an enhanced copy.",
                    elem_classes=["h3-gallery-section-title"],
                )
                player = gr.Video(
                    label="Selected video",
                    height=420,
                    interactive=False,
                    elem_classes=["h3-gallery-player"],
                )
                image = gr.Image(
                    label="Selected image",
                    type="filepath",
                    height=420,
                    visible=False,
                    interactive=False,
                    elem_classes=["h3-gallery-player"],
                )
                audio = gr.Audio(
                    label="Selected audio",
                    type="filepath",
                    visible=False,
                    interactive=False,
                    elem_classes=["h3-gallery-player"],
                )
                download = gr.Markdown(elem_classes=["h3-gallery-download"])
                with gr.Accordion(
                    "Enhance selected media",
                    open=True,
                    elem_classes=["h3-gallery-card", "h3-gallery-enhance"],
                ) as enhance:
                    with gr.Row(equal_height=True):
                        postprocess = gr.Dropdown(
                            choices=list(postprocess_options),
                            value=postprocess_options[0],
                            label="Method",
                            info="SDR to HDR saves a 10-bit HLG/HEVC video; playback needs an HDR-capable player.",
                            scale=2,
                        )
                        upscale_resolution = gr.Dropdown(
                            choices=list(resolution_choices),
                            value=default_resolution,
                            label="Target resolution",
                            info="Fits the source inside this frame while preserving its aspect ratio.",
                            scale=2,
                        )
                    with gr.Group(
                        visible=True, elem_classes=["h3-gallery-ai-settings"]
                    ) as ai_settings:
                        seedvr2_model = gr.Dropdown(
                            choices=list(seedvr2_choices),
                            value=default_seedvr2,
                            label="SeedVR2 model",
                            visible=True,
                            info="Downloaded on first use. 7B INT8 is the default quality/VRAM balance; FP16 favors fidelity, 7B Sharp favors stronger detail, and MXFP8/NVFP4 are experimental speed options.",
                        )
                        ltx25_model = gr.Dropdown(
                            choices=list(ltx25_choices),
                            value=default_ltx25,
                            label="Finishing LTX model",
                            visible=False,
                            info="Used only for this finishing request. Downloaded on first use.",
                        )
                        ltx25_prompt = gr.Textbox(
                            label="LTX-2.5 scene prompt",
                            placeholder="Describe the source scene and desired fine detail",
                            lines=3,
                            visible=False,
                            info="Optional but recommended. Uses the finishing model selected here.",
                        )
                        with gr.Row():
                            post_seed = gr.Number(
                                value=-1, precision=0, label="Seed (-1 random)"
                            )
                            force_offload = gr.Checkbox(
                                value=False,
                                label="Unload resident models first",
                                info="Can lower peak VRAM before AI processing starts.",
                            )
                        split_upscale = gr.Checkbox(
                            value=False,
                            label="Split source into clips before LTX processing",
                            info="Opt in after an out-of-VRAM error. Processes clips independently, then concatenates them.",
                            visible=False,
                        )
                        split_seconds = gr.Slider(
                            1.0,
                            15.0,
                            value=5.0,
                            step=0.5,
                            label="Target clip length (seconds)",
                            info="The actual cut is adjusted to an LTX-valid frame count.",
                            visible=False,
                        )
                    with gr.Row(equal_height=True, elem_classes=["h3-gallery-actions"]):
                        post_run = gr.Button(
                            "Enhance selected media", variant="primary", scale=3
                        )
                        post_stop = gr.Button("Interrupt", scale=1)
                    post_status = gr.Markdown(elem_classes=["h3-gallery-post-status"])
    return GalleryView(
        mode,
        refresh,
        status,
        paths,
        selected,
        upload_video,
        import_video,
        grid,
        shown,
        show_more,
        manage,
        confirm_delete,
        delete,
        empty,
        player,
        image,
        audio,
        download,
        enhance,
        postprocess,
        upscale_resolution,
        ai_settings,
        seedvr2_model,
        ltx25_prompt,
        post_seed,
        force_offload,
        split_upscale,
        split_seconds,
        post_run,
        post_stop,
        post_status,
        ltx25_model,
        deletion_confirmation,
        deletion_message,
        deletion_confirm,
        deletion_cancel,
        deletion_intent,
    )
