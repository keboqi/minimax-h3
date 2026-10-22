"""Typed builders for self-contained application views."""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
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
    query: Any = None
    favorite_only: Any = None
    search: Any = None
    inspector: Any = None
    filters: Any = None
    scan_history: Any = None
    downloads: Any = None


def build_download_controls() -> SimpleNamespace:
    """Download a video from a page URL with yt-dlp, plus saved cookies for sign-in sites."""
    from h3_app.video_download import DEFAULT_QUALITY, QUALITIES

    with gr.Accordion("Download from a URL", open=False, elem_classes=["h3-gallery-card"]):
        gr.Markdown("Paste a video page link (YouTube, Bilibili, X and most video sites). The video joins the "
                    "library, ready for **Dub & Translate**.")
        with gr.Row(equal_height=True):
            url = gr.Textbox(label="Video URL", placeholder="https://…", scale=4)
            quality = gr.Dropdown(list(QUALITIES), value=DEFAULT_QUALITY, label="Quality", scale=1, min_width=140)
        with gr.Row():
            download = gr.Button("Download video", variant="primary", scale=2)
            info = gr.Button("Check link", scale=1)
            stop = gr.Button("Interrupt", scale=1)
        status = gr.Markdown()
        path = gr.Textbox(visible=False)
        with gr.Accordion("Cookies for sign-in downloads", open=False) as cookies:
            gr.Markdown("Age-restricted, members-only or sign-in videos need your browser's cookies for that "
                        "site. They are stored privately on this server and used only for that site's links.")
            sites = gr.Markdown()
            with gr.Tab("From cURL"):
                curl = gr.Textbox(label="Copy as cURL", lines=3, placeholder="curl 'https://www.youtube.com/…' -H 'cookie: …'",
                                  info="Signed in, open DevTools → Network, right-click a request on the site → "
                                       "Copy → Copy as cURL (bash).")
                with gr.Row(equal_height=True):
                    curl_domain = gr.Textbox(label="Site", placeholder="Detected from the command", scale=3)
                    import_curl = gr.Button("Save cookies", scale=1)
            with gr.Tab("From cookies.txt"):
                cookie_file = gr.File(label="Netscape cookies.txt", file_types=[".txt"], type="filepath", height=90)
                with gr.Row(equal_height=True):
                    file_domain = gr.Textbox(label="Site", placeholder="youtube.com", scale=3)
                    import_file = gr.Button("Save cookies", scale=1)
            with gr.Row(equal_height=True):
                saved_site = gr.Dropdown([], label="Saved site", scale=3)
                delete = gr.Button("Delete cookies", variant="stop", scale=1)
            cookie_status = gr.Markdown()
    return SimpleNamespace(url=url, quality=quality, download=download, info=info, stop=stop, status=status,
                           path=path, cookies=cookies, sites=sites, curl=curl, curl_domain=curl_domain,
                           import_curl=import_curl, cookie_file=cookie_file, file_domain=file_domain,
                           import_file=import_file, saved_site=saved_site, delete=delete,
                           cookie_status=cookie_status)


def build_gallery_view(
    root: gr.Column,
    *,
    postprocess_options: Sequence[str],
    resolution_choices: Sequence[str],
    default_resolution: str,
    seedvr2_choices: Sequence[str],
    default_seedvr2: str,
    ltx25_choices: Sequence[str] = (),
    default_ltx25: str | None = None,
    enhance: bool = True,
    downloads: bool = False,
) -> GalleryView:
    """Build the Media library; ``enhance=False`` omits GPU post-processing.

    ``downloads`` adds Download from a URL (yt-dlp) with its saved cookies.
    """
    with root:
        gr.Markdown(
            "## Media library\nYour outputs, in one place. Find a favorite, review the details"
            + (", and create an enhanced copy." if enhance else ", and keep the library tidy."),
            elem_classes=["h3-gallery-heading", "h3-view-heading"],
        )
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-toolbar"]):
            mode = gr.Radio(
                choices=["Video", "Image", "Audio"],
                value="Video",
                label="Media type",
                elem_id="h3-library-kind",
                scale=0,
                min_width=310,
            )
            refresh = gr.Button(
                "Refresh library",
                variant="secondary",
                scale=0,
                min_width=150,
                elem_classes=["h3-library-button"],
            )
            scan_history = gr.Button("Scan historical media", variant="secondary", scale=0, min_width=180)
            status = gr.Markdown(
                "Choose a media type to browse your library.",
                elem_classes=["h3-gallery-status"],
            )
        paths = gr.State([])
        selected = gr.State(None)
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-filters"]):
            query = gr.Textbox(
                label="Search media",
                placeholder="Search filenames, tags or settings…",
                scale=3,
            )
            favorite_only = gr.Checkbox(label="Favorites only", scale=1)
            search = gr.Button(
                "Search library",
                scale=0,
                min_width=150,
                elem_classes=["h3-library-button"],
            )
        filters = gr.State({"query": "", "favorite": False})
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-workspace"]):
            with gr.Column(scale=5, min_width=320):
                gr.Markdown(
                    "### Browse outputs\nSelect a thumbnail to preview it and see its details.",
                    elem_classes=["h3-gallery-section-title"],
                )
                grid = gr.Gallery(
                    value=[],
                    label="Media library",
                    columns=3,
                    height=420,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="generated-video-gallery",
                    elem_classes=["h3-gallery-grid"],
                )
                shown = gr.State(48)
                show_more = gr.Button(
                    "Show more", interactive=False, variant="secondary"
                )
                download_controls = build_download_controls() if downloads else None
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
                        delete = gr.Button(
                            "Delete selected", variant="stop", interactive=False
                        )
                        empty = gr.Button("Empty generated library", variant="stop")
                    with gr.Group(visible=False) as deletion_confirmation:
                        deletion_message = gr.Markdown()
                        with gr.Row():
                            deletion_confirm = gr.Button(
                                "Confirm permanent deletion", variant="stop"
                            )
                            deletion_cancel = gr.Button("Keep media")
                        deletion_intent = gr.State(None)
            with gr.Column(scale=4, min_width=320, elem_classes=["h3-preview-panel"]):
                gr.Markdown(
                    "### Selected media\nPreview and download your original."
                    + (" Enhancements are saved as a new copy." if enhance else ""),
                    elem_classes=["h3-gallery-section-title"],
                )
                player = gr.Video(
                    label="Selected video",
                    height=360,
                    interactive=False,
                    elem_classes=["h3-gallery-player"],
                )
                image = gr.Image(
                    label="Selected image",
                    type="filepath",
                    height=360,
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
                inspector = gr.Column(elem_classes=["h3-media-inspector"])
                with gr.Accordion(
                    "Enhance selected media",
                    open=False,
                    visible=enhance,
                    elem_classes=["h3-gallery-card", "h3-gallery-enhance"],
                ) as enhance_panel:
                    with gr.Row(equal_height=True):
                        postprocess = gr.Dropdown(
                            choices=list(postprocess_options),
                            value=postprocess_options[0],
                            label="Method",
                            info="Enhancements create a new copy.",
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
                            label="Advanced: split source into clips before LTX processing",
                            info="Normally leave off with Tiled Upscale. Opt in if memory is still insufficient. Processes clips independently, then concatenates them.",
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
                            "Enhance selected media",
                            variant="primary",
                            scale=3,
                            interactive=False,
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
        enhance_panel,
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
        query,
        favorite_only,
        search,
        inspector,
        filters,
        scan_history,
        download_controls,
    )
