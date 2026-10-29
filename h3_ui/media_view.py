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
    enhance: gr.Tab
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
    # The Details tab; other modules add the settings snapshot here.
    inspector: Any = None
    filters: Any = None
    scan_history: Any = None
    downloads: Any = None
    # The selected item's action row: comparison and Set as are added here.
    actions: Any = None
    # The Tags & lineage tab.
    annotations: Any = None
    import_panel: Any = None
    more_panel: Any = None
    engine: Any = None
    sort: Any = None
    favorite_toggle: Any = None
    counts: Any = None


SORT_CHOICES = (("Newest first", "newest"), ("Oldest first", "oldest"), ("Name", "name"))


def favorite_label(starred: bool) -> str:
    return "★ Favorite" if starred else "☆ Favorite"


def build_download_controls() -> SimpleNamespace:
    """Download a video from a page URL with yt-dlp, plus saved cookies for sign-in sites."""
    from h3_app.video_download import DEFAULT_QUALITY, QUALITIES

    with gr.Group(visible=False, elem_classes=["h3-gallery-card", "h3-gallery-panel"]) as panel:
        gr.Markdown("### Download from a URL\nPaste a video page link (YouTube, Bilibili, X and most video "
                    "sites). The video joins the library, ready for **Dub & Translate**.",
                    elem_classes=["h3-gallery-section-title"])
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
                           cookie_status=cookie_status, panel=panel)


def bind_panel_toggles(toggles: Sequence[tuple[gr.Button, Any]]) -> None:
    """Header buttons open one panel under the toolbar at a time; a second click closes it."""
    panels = [panel for _button, panel in toggles]
    shown = gr.State(None)
    for index, (button, _panel) in enumerate(toggles):

        def toggle(current, index=index):
            opened = None if current == index else index
            return (opened, *(gr.update(visible=opened == other) for other in range(len(panels))))

        button.click(toggle, inputs=shown, outputs=[shown, *panels], queue=False,
                     show_progress="hidden", api_name=False)


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
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-header"]):
            gr.Markdown(
                "## Media library\nYour outputs, in one place. Find a favorite, review the details"
                + (", and create an enhanced copy." if enhance else ", and keep the library tidy."),
                elem_classes=["h3-gallery-heading", "h3-view-heading"],
            )
            with gr.Row(equal_height=True, elem_classes=["h3-gallery-header-actions"]):
                refresh = gr.Button("Refresh", variant="secondary", size="md", scale=0, min_width=100)
                url_toggle = (
                    gr.Button("From a URL", variant="secondary", size="md", scale=0, min_width=120)
                    if downloads else None
                )
                import_toggle = gr.Button("Import file", variant="primary", size="md", scale=0, min_width=120)
                more_toggle = gr.Button("More", variant="secondary", size="md", scale=0, min_width=80)
        download_controls = build_download_controls() if downloads else None
        with gr.Group(visible=False, elem_classes=["h3-gallery-card", "h3-gallery-panel"]) as import_panel:
            gr.Markdown(
                "### Import local media\nAdd an existing video, image, or audio file to the active library "
                "so it can be previewed alongside generated outputs.",
                elem_classes=["h3-gallery-section-title"],
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
        with gr.Group(visible=False, elem_classes=["h3-gallery-card", "h3-gallery-panel"]) as more_panel:
            gr.Markdown(
                "### Library upkeep\nScan historical media to add older or externally created files "
                "and repair missing previews.",
                elem_classes=["h3-gallery-section-title"],
            )
            scan_history = gr.Button("Scan historical media", variant="secondary", scale=0, min_width=180)
        bind_panel_toggles(
            ([(url_toggle, download_controls.panel)] if downloads else [])
            + [(import_toggle, import_panel), (more_toggle, more_panel)]
        )
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-toolbar"]):
            mode = gr.Radio(
                choices=["Video", "Image", "Audio"],
                value="Video",
                label="Media type",
                elem_id="h3-library-kind",
                elem_classes=["h3-visually-hidden-label"],
                scale=0,
                min_width=260,
            )
            query = gr.Textbox(
                label="Search media",
                elem_classes=["h3-visually-hidden-label"],
                placeholder="Search filenames, tags or settings…",
                scale=3,
                min_width=220,
            )
            favorite_only = gr.Checkbox(label="Favorites only", scale=0, min_width=150)
            engine = gr.Dropdown(
                choices=[("All engines", "")], value="", label="Made with",
                scale=0, min_width=180, elem_classes=["h3-gallery-filter"],
            )
            sort = gr.Dropdown(
                choices=list(SORT_CHOICES), value="newest", label="Sort",
                scale=0, min_width=160, elem_classes=["h3-gallery-filter"],
            )
            search = gr.Button(
                "Search library",
                scale=0,
                min_width=150,
                elem_classes=["h3-library-button"],
            )
        paths = gr.State([])
        selected = gr.State(None)
        filters = gr.State({"query": "", "favorite": False, "engine": "", "sort": "newest"})
        with gr.Row(equal_height=False, elem_classes=["h3-gallery-workspace"]):
            with gr.Column(scale=5, min_width=320, elem_classes=["h3-gallery-browse"]):
                with gr.Row(equal_height=True, elem_classes=["h3-gallery-status-row"]):
                    status = gr.Markdown(
                        "Choose a media type to browse your library.",
                        elem_classes=["h3-gallery-status"],
                    )
                    counts = gr.Markdown(elem_classes=["h3-gallery-counts"])
                grid = gr.Gallery(
                    value=[],
                    label="Media library",
                    show_label=False,
                    columns=4,
                    height=560,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="generated-video-gallery",
                    elem_classes=["h3-gallery-grid"],
                )
                shown = gr.State(48)
                show_more = gr.Button(
                    "Show more", interactive=False, variant="secondary",
                    elem_classes=["h3-gallery-more"],
                )
            with gr.Column(scale=4, min_width=320, elem_classes=["h3-preview-panel", "h3-gallery-inspector"]):
                player = gr.Video(
                    label="Selected video",
                    height=320,
                    interactive=False,
                    elem_classes=["h3-gallery-player"],
                )
                image = gr.Image(
                    label="Selected image",
                    type="filepath",
                    height=320,
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
                with gr.Row(equal_height=True, elem_classes=["h3-gallery-title-row"]):
                    download = gr.Markdown(elem_classes=["h3-gallery-download"])
                    favorite_toggle = gr.Button(
                        favorite_label(False), interactive=False, scale=0, min_width=130,
                        elem_classes=["h3-gallery-star"],
                    )
                actions = gr.Column(elem_classes=["h3-gallery-actions-row"])
                with gr.Tabs(elem_classes=["h3-inspector-tabs"]):
                    with gr.Tab("Details", render_children=True):
                        inspector = gr.Column(elem_classes=["h3-media-inspector"])
                    with gr.Tab("Tags & lineage", render_children=True):
                        annotations = gr.Column(elem_classes=["h3-media-annotations"])
                    with gr.Tab(
                        "Enhance selected media",
                        visible=enhance,
                        render_children=True,
                        elem_classes=["h3-gallery-enhance"],
                    ) as enhance_panel:
                        gr.Markdown(
                            "Enhancing saves a new copy beside the original.",
                            elem_classes=["h3-gallery-note"],
                        )
                        with gr.Row(equal_height=True):
                            postprocess = gr.Dropdown(
                                choices=list(postprocess_options),
                                value=postprocess_options[0],
                                label="Method",
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
                                label="Scene prompt",
                                placeholder="Describe the source scene and desired fine detail",
                                lines=3,
                                visible=False,
                                info="Optional but recommended. Uses the finishing model selected here.",
                            )
                            with gr.Accordion("Seed and memory", open=False):
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
                with gr.Accordion(
                    "Manage library",
                    open=False,
                    elem_classes=["h3-gallery-danger"],
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
                        empty = gr.Button("Empty generated library", variant="secondary")
                    with gr.Group(visible=False, elem_classes=["h3-gallery-confirm"]) as deletion_confirmation:
                        deletion_message = gr.Markdown()
                        with gr.Row():
                            deletion_confirm = gr.Button(
                                "Confirm permanent deletion", variant="stop"
                            )
                            deletion_cancel = gr.Button("Keep media")
                        deletion_intent = gr.State(None)
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
        actions,
        annotations,
        import_panel,
        more_panel,
        engine,
        sort,
        favorite_toggle,
        counts,
    )
