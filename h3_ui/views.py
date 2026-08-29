"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


@dataclass(frozen=True)
class MusicView:
    caption: gr.Textbox
    lyrics: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    reference_images: tuple[gr.Image, gr.Image, gr.Image]
    enhance: gr.Button
    enhance_status: gr.Textbox
    output: gr.Audio
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    model: gr.Dropdown
    duration: gr.Slider
    seed: gr.Number
    tiled: gr.Checkbox
    steps: gr.Slider
    cfg: gr.Slider
    ar_cfg: gr.Slider
    top_k: gr.Slider


def build_music_view(
    root: gr.Group,
    *,
    prompt_models: Sequence[str],
    default_prompt_model: str,
    model_choices: Sequence[str],
    defaults: Mapping[str, Any],
) -> MusicView:
    with root:
        gr.Markdown(
            "## MiniMax Music 3\n"
            "Generate complete stereo songs with the native workflow on the shared "
            "ComfyUI backend. Write a detailed **caption** for style, vocals, and "
            "arrangement, then use section tags such as `[Intro]`, `[Verse]`, "
            "`[Chorus]`, `[Bridge]`, `[Instrumental]`, and `[Outro]` in the lyrics. "
            "[ComfyUI guide](https://docs.comfy.org/tutorials/audio/minimax/minimax-music-3) · "
            "[Official prompting skill](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter)"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                caption = gr.Textbox(
                    label="Music caption",
                    lines=12,
                    placeholder=(
                        "Global Metadata: genre, BPM, key, mood, production...\n\n"
                        "Vocal Details: singer, delivery, harmonies, effects...\n\n"
                        "Arrangement: instruments, groove, section-by-section evolution..."
                    ),
                )
                lyrics = gr.Textbox(
                    label="Lyrics and song structure",
                    lines=16,
                    placeholder=(
                        "[Intro]\n\n[Verse]\nWrite lyrics here...\n\n"
                        "[Chorus]\n...\n\n[Bridge]\n...\n\n[Outro]"
                    ),
                    info="For an instrumental, repeat [Instrumental] sections to guide length.",
                )
                with gr.Accordion("Gemini Music 3 prompt writer", open=False):
                    gr.Markdown(
                        "Create or enhance the caption from text, lyrics, and optional visual reference images."
                    )
                    with gr.Row():
                        prompt_model = gr.Dropdown(
                            choices=list(prompt_models),
                            value=default_prompt_model,
                            label="Gemini model",
                        )
                        api_key = gr.Textbox(
                            label="Temporary Gemini API key",
                            type="password",
                            placeholder="Uses GEMINI_API_KEY when blank",
                        )
                    with gr.Row():
                        references = tuple(
                            gr.Image(type="filepath", label=f"Reference image {index}")
                            for index in range(1, 4)
                        )
                    enhance = gr.Button("Generate / enhance Music 3 caption")
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
            with gr.Column(scale=2):
                output = gr.Audio(label="Generated song", type="filepath")
                with gr.Row():
                    run = gr.Button("Generate with Music 3", variant="primary")
                    stop = gr.Button("Interrupt")
                status = gr.Textbox(label="Status", lines=7)
                gr.Markdown("### Generation settings")
                model = gr.Dropdown(
                    choices=list(model_choices),
                    value=defaults["model"],
                    label="Diffusion model",
                    info="The selected DiT and shared encoder/decoder download on first use.",
                )
                with gr.Row():
                    duration = gr.Slider(
                        1,
                        300,
                        value=defaults["duration"],
                        step=1,
                        label="Maximum seconds",
                    )
                    seed = gr.Number(
                        value=defaults["seed"], precision=0, label="Seed (-1 random)"
                    )
                tiled = gr.Checkbox(
                    value=defaults["tiled_decode"],
                    label="Tiled audio decode",
                    info="Reduces peak VRAM for long songs; disable for fastest decode on high-VRAM GPUs.",
                )
                with gr.Accordion("Advanced sampling", open=False):
                    steps = gr.Slider(
                        1,
                        100,
                        value=defaults["steps"],
                        step=1,
                        label="Diffusion steps",
                    )
                    with gr.Row():
                        cfg = gr.Slider(
                            0,
                            10,
                            value=defaults["cfg"],
                            step=0.05,
                            label="Diffusion CFG",
                        )
                        ar_cfg = gr.Slider(
                            0,
                            10,
                            value=defaults["ar_cfg"],
                            step=0.05,
                            label="Autoregressive CFG",
                        )
                    top_k = gr.Slider(
                        1,
                        200,
                        value=defaults["top_k"],
                        step=1,
                        label="Autoregressive Top K",
                    )
                gr.Markdown(
                    "Output is saved as V0-quality MP3 under `ComfyUI/output/audio`. "
                    "Music 3 may end a song before the maximum duration."
                )
    return MusicView(
        caption,
        lyrics,
        prompt_model,
        api_key,
        references,
        enhance,
        enhance_status,
        output,
        run,
        stop,
        status,
        model,
        duration,
        seed,
        tiled,
        steps,
        cfg,
        ar_cfg,
        top_k,
    )


@dataclass(frozen=True)
class GalleryView:
    refresh: gr.Button
    status: gr.Markdown
    paths: gr.State
    selected: gr.State
    upload_video: gr.File
    import_video: gr.Button
    grid: gr.Gallery
    confirm_delete: gr.Checkbox
    delete: gr.Button
    empty: gr.Button
    player: gr.Video
    download: gr.Markdown
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


def build_gallery_view(
    root: gr.Group,
    *,
    postprocess_options: Sequence[str],
    resolution_choices: Sequence[str],
    default_resolution: str,
    seedvr2_choices: Sequence[str],
    default_seedvr2: str,
) -> GalleryView:
    with root:
        gr.Markdown(
            "## Video gallery\nBrowse generated work, import a local clip, and enhance the selected video.",
            elem_classes=["h3-gallery-heading"],
        )
        with gr.Row(equal_height=True, elem_classes=["h3-gallery-toolbar"]):
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
            "Import a local video", open=False, elem_classes=["h3-gallery-card"]
        ):
            gr.Markdown(
                "Add an existing video to this library so it can be previewed "
                "and post-processed alongside generated clips."
            )
            with gr.Row(equal_height=True, elem_classes=["h3-gallery-import"]):
                upload_video = gr.File(
                    label="Choose a video",
                    file_count="single",
                    file_types=["video"],
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
                    "### Library\nSelect a thumbnail to load the full video.",
                    elem_classes=["h3-gallery-section-title"],
                )
                grid = gr.Gallery(
                    value=[],
                    label="Video library",
                    columns=3,
                    height=620,
                    object_fit="cover",
                    allow_preview=False,
                    fit_columns=False,
                    elem_id="generated-video-gallery",
                    elem_classes=["h3-gallery-grid"],
                )
                with gr.Accordion(
                    "Manage library",
                    open=False,
                    elem_classes=["h3-gallery-card", "h3-gallery-danger"],
                ):
                    confirm_delete = gr.Checkbox(
                        value=False,
                        label="I understand deletion is permanent",
                        info=(
                            "Required before deleting the selected video "
                            "or emptying the generated library."
                        ),
                    )
                    with gr.Row(
                        equal_height=True,
                        elem_classes=["h3-gallery-danger-actions"],
                    ):
                        delete = gr.Button("Delete selected", variant="stop")
                        empty = gr.Button("Empty generated library", variant="stop")
            with gr.Column(scale=5, min_width=480):
                gr.Markdown(
                    "### Preview & enhance\nReview the selected clip, download it, or create an enhanced copy.",
                    elem_classes=["h3-gallery-section-title"],
                )
                player = gr.Video(
                    label="Selected video",
                    height=420,
                    elem_classes=["h3-gallery-player"],
                )
                download = gr.Markdown(elem_classes=["h3-gallery-download"])
                with gr.Accordion(
                    "Enhance selected video",
                    open=True,
                    elem_classes=["h3-gallery-card", "h3-gallery-enhance"],
                ):
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
                            info=(
                                "Downloaded on first use. 7B Sharp favors stronger detail; "
                                "NVFP4 variants are optimized for Blackwell GPUs."
                            ),
                        )
                        ltx25_prompt = gr.Textbox(
                            label="LTX-2.5 upscale prompt",
                            placeholder="Describe the source scene and desired fine detail",
                            lines=3,
                            visible=False,
                            info="Optional but recommended. Uses the transformer selected in the LTX 2.5 tab.",
                        )
                        with gr.Row():
                            post_seed = gr.Number(
                                value=-1, precision=0, label="Seed (-1 random)"
                            )
                            force_offload = gr.Checkbox(
                                value=False,
                                label="Unload resident models first",
                                info="Can lower peak VRAM before AI upscaling starts.",
                            )
                        split_upscale = gr.Checkbox(
                            value=False,
                            label="Split source into clips before LTX upscaling",
                            info=(
                                "Opt in after an out-of-VRAM error. Upscales clips "
                                "independently, then concatenates them."
                            ),
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
                            "Enhance selected video", variant="primary", scale=3
                        )
                        post_stop = gr.Button("Interrupt", scale=1)
                    post_status = gr.Markdown(elem_classes=["h3-gallery-post-status"])
    return GalleryView(
        refresh,
        status,
        paths,
        selected,
        upload_video,
        import_video,
        grid,
        confirm_delete,
        delete,
        empty,
        player,
        download,
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
    )


@dataclass(frozen=True)
class ApiView:
    prompt: gr.Textbox
    run: gr.Button
    stop: gr.Button
    download_url: gr.Textbox
    status: gr.Textbox


def build_api_view(root: gr.Group, guide: str) -> ApiView:
    with root:
        gr.Markdown(guide)
        with gr.Accordion("Try the default API request", open=False):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="Describe the video, camera motion, dialogue, and sound.",
            )
            with gr.Row():
                run = gr.Button("Generate with UI defaults", variant="primary")
                stop = gr.Button("Interrupt")
            download_url = gr.Textbox(label="Download URL", interactive=False)
            status = gr.Textbox(label="Status", lines=5)
    return ApiView(prompt, run, stop, download_url, status)
