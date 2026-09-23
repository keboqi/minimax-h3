"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr

from .prompt_writer_controls import build_remote_prompt_writer_controls


@dataclass(frozen=True)
class MusicView:
    caption: gr.Textbox
    lyrics: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
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


@dataclass(frozen=True)
class QwenImage21View:
    mode: gr.Dropdown
    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    reference_images: gr.File
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
    enhance: gr.Button
    enhance_status: gr.Textbox
    output: gr.Image
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    model: gr.Dropdown
    text_encoder: gr.Dropdown
    square_resolution: gr.Dropdown
    landscape_resolution: gr.Dropdown
    portrait_resolution: gr.Dropdown
    width: gr.Slider
    height: gr.Slider
    reference_resolution: gr.Dropdown
    match_input_size: gr.Checkbox
    seed: gr.Number
    steps: gr.Slider
    cfg: gr.Slider
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    cache_device: gr.Dropdown
    cache_dtype: gr.Dropdown
    attention_backend: gr.Dropdown
    accelerator: gr.Dropdown


def build_qwen_image21_view(
    root: gr.Group,
    *,
    model_choices: Sequence[str],
    text_encoder_choices: Sequence[str],
    prompt_models: Sequence[str],
    default_prompt_model: str,
    defaults: Mapping[str, Any],
) -> QwenImage21View:
    with root:
        gr.Markdown(
            "## Qwen Image 2.1\n"
            "Generate images or edit and combine reference images with the native "
            "ComfyUI workflow. In edit mode, the first image is the target. For "
            "multiple inputs, mention them as `<image1>`, `<image2>`, and so on; "
            "refer to a single input naturally without a tag. Models download on "
            "first use. [Model details](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                mode = gr.Dropdown(
                    choices=["Text to image", "Image edit"],
                    value=defaults["mode"],
                    label="Mode",
                )
                prompt = gr.Textbox(
                    label="Prompt / edit instruction",
                    lines=10,
                    placeholder=(
                        "Describe the image to create, or explain precisely what "
                        "to change while preserving the rest."
                    ),
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=3,
                    info="Used only when CFG is greater than 1.",
                )
                reference_images = gr.File(
                    label="Reference images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                gr.Markdown(
                    "Image edit supports up to 10 files. **image1** is the edit "
                    "target; later images are references. Numbered tags are required "
                    "only when multiple files are supplied."
                )
                with gr.Accordion("Prompt writer / enhancer", open=False):
                    gr.Markdown(
                        "Create or enhance the generation/edit prompt from text and the uploaded reference images."
                    )
                    writer = build_remote_prompt_writer_controls(
                        prompt_models, default_prompt_model
                    )
                    prompt_model = writer.model
                    api_key = writer.gemini_api_key
                    prompt_backend = writer.backend
                    lightning_api_key = writer.lightning_api_key
                    enhance = gr.Button("Generate / enhance Qwen prompt")
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
            with gr.Column(scale=2):
                output = gr.Image(
                    label="Generated image", type="filepath", interactive=False
                )
                with gr.Row():
                    run = gr.Button("Generate with Qwen Image 2.1", variant="primary")
                    stop = gr.Button("Interrupt")
                status = gr.Textbox(label="Status", lines=7)
                model = gr.Dropdown(
                    choices=list(model_choices),
                    value=defaults["model"],
                    label="Diffusion model",
                )
                text_encoder = gr.Dropdown(
                    choices=list(text_encoder_choices),
                    value=defaults["text_encoder"],
                    label="Qwen3-VL text encoder",
                )
                with gr.Row():
                    square_resolution = gr.Dropdown(
                        choices=[
                            "1K · 1:1 · 1024×1024",
                            "2K · 1:1 · 2048×2048",
                        ],
                        value=None,
                        label="Square",
                        info="Square presets at 1K and native 2K.",
                    )
                    landscape_resolution = gr.Dropdown(
                        choices=[
                            "1K · 4:3 · 1376×1024",
                            "1K · 3:2 · 1536×1024",
                            "1K · 16:9 · 1824×1024",
                            "2K · 4:3 · 2400×1792",
                            "2K · 3:2 · 2528×1696",
                            "2K · 16:9 · 2752×1536",
                        ],
                        value=None,
                        label="Landscape",
                        info="Landscape presets at 1K and native 2K.",
                    )
                    portrait_resolution = gr.Dropdown(
                        choices=[
                            "1K · 3:4 · 1024×1376",
                            "1K · 2:3 · 1024×1536",
                            "1K · 9:16 · 1024×1824",
                            "2K · 3:4 · 1792×2400",
                            "2K · 2:3 · 1696×2528",
                            "2K · 9:16 · 1536×2752",
                        ],
                        value=None,
                        label="Portrait",
                        info="Portrait presets at 1K and native 2K.",
                    )
                with gr.Row():
                    width = gr.Slider(
                        256, 2752, value=defaults["width"], step=32, label="Width"
                    )
                    height = gr.Slider(
                        256, 2752, value=defaults["height"], step=32, label="Height"
                    )
                match_input_size = gr.Checkbox(
                    value=defaults["match_input_size"],
                    label="Match first image size when editing",
                    info="Recommended. Disable to use the width and height above.",
                )
                reference_resolution = gr.Dropdown(
                    choices=[
                        ("Keep each source size", 0),
                        ("512 px pixel budget", 512),
                        ("768 px pixel budget", 768),
                        ("1024 px pixel budget", 1024),
                        ("1536 px pixel budget", 1536),
                        ("2048 px pixel budget", 2048),
                    ],
                    value=defaults["reference_resolution"],
                    label="Reference resolution",
                    info="0 keeps each source size (rounded to 32); 1024 normalizes pixel area.",
                )
                with gr.Row():
                    seed = gr.Number(
                        value=defaults["seed"], precision=0, label="Seed (-1 random)"
                    )
                    steps = gr.Slider(
                        1, 100, value=defaults["steps"], step=1, label="Steps"
                    )
                    cfg = gr.Slider(
                        0, 20, value=defaults["cfg"], step=0.1, label="CFG"
                    )
                with gr.Accordion("Advanced sampling and edit cache", open=False):
                    with gr.Row():
                        sampler = gr.Dropdown(
                            choices=[
                                "euler",
                                "euler_ancestral",
                                "er_sde",
                                "dpmpp_2m",
                                "dpmpp_sde",
                                "dpmpp_sde_gpu",
                            ],
                            value=defaults["sampler"],
                            label="Sampler",
                        )
                        scheduler = gr.Dropdown(
                            choices=["simple", "normal", "beta"],
                            value=defaults["scheduler"],
                            label="Scheduler",
                        )
                    with gr.Row():
                        cache_device = gr.Dropdown(
                            choices=["auto", "gpu", "cpu", "off"],
                            value=defaults["cache_device"],
                            label="Edit KV cache device",
                        )
                        cache_dtype = gr.Dropdown(
                            choices=["default", "int8", "int4"],
                            value=defaults["cache_dtype"],
                            label="Edit KV cache precision",
                        )
                    attention_backend = gr.Dropdown(
                        choices=[
                            ("PyTorch attention (official)", "pytorch attention"),
                            (
                                "Comfy Kitchen INT8 attention (experimental faster)",
                                "comfy kitchen attention",
                            ),
                        ],
                        value=defaults["attention_backend"],
                        label="Model attention backend",
                        info=(
                            "Kitchen attention can improve speed on supported NVIDIA/"
                            "AMD GPUs and falls back to PyTorch when unavailable."
                        ),
                    )
                    accelerator = gr.Dropdown(
                        choices=[
                            "Off",
                            "Spectrum (Quality)",
                            "Spectrum (Preview)",
                        ],
                        value=defaults["accelerator"],
                        label="Diffusion accelerator",
                        info=(
                            "Quality forecasts fewer middle steps and restores "
                            "more late detail; Preview is faster but can soften "
                            "outlines and fine texture. Spectrum remains experimental."
                        ),
                    )
                gr.Markdown(
                    "Official defaults are CFG 1 and 40 Euler/simple steps. Native "
                    "sizes include 2048×2048, 2400×1792, 1792×2400, 2528×1696, "
                    "1696×2528, 2752×1536, and 1536×2752. For transparent PNGs, "
                    "ask for an RGBA image with an alpha channel and transparent background."
                )
    return QwenImage21View(
        mode,
        prompt,
        negative_prompt,
        reference_images,
        prompt_model,
        api_key,
        prompt_backend,
        lightning_api_key,
        enhance,
        enhance_status,
        output,
        run,
        stop,
        status,
        model,
        text_encoder,
        square_resolution,
        landscape_resolution,
        portrait_resolution,
        width,
        height,
        reference_resolution,
        match_input_size,
        seed,
        steps,
        cfg,
        sampler,
        scheduler,
        cache_device,
        cache_dtype,
        attention_backend,
        accelerator,
    )

@dataclass(frozen=True)
class YuE2View:
    style: gr.Textbox
    lyrics: gr.Textbox
    abc: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
    enhance: gr.Button
    enhance_status: gr.Textbox
    mode: gr.Dropdown
    output: gr.Audio
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    model: gr.Dropdown
    duration: gr.Slider
    seed: gr.Number
    steps: gr.Slider
    cfg: gr.Slider
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    repetition_penalty: gr.Slider
    max_abc_tokens: gr.Slider
    abc_temperature: gr.Slider
    abc_top_p: gr.Slider
    abc_top_k: gr.Slider
    abc_repetition_penalty: gr.Slider
    abc_penalty_window: gr.Slider
    tiled: gr.Checkbox

def build_yue2_view(
    root: gr.Group,
    *,
    prompt_models: Sequence[str],
    default_prompt_model: str,
    model_choices: Sequence[str],
    defaults: Mapping[str, Any],
) -> YuE2View:
    with root:
        gr.Markdown("## YuE2\nGenerate full songs from a style prompt and sectioned lyrics on the shared ComfyUI backend. Full and Melody modes plan a score first; Direct skips planning. The INT8 checkpoint downloads on first use. Model license: CC-BY-NC-4.0. [Model details](https://huggingface.co/Comfy-Org/YuE2)")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                style = gr.Textbox(label="Style prompt", lines=6, placeholder="Indie pop, warm female vocal, clean guitar, restrained drums...")
                lyrics = gr.Textbox(label="Lyrics and song structure", lines=16, placeholder="[verse]\nLyrics here...\n\n[chorus]\n...")
                abc = gr.Textbox(label="ABC score override (optional)", lines=7, placeholder="Leave blank to let YuE2 create the score.")
                with gr.Accordion("Prompt writer / enhancer", open=False):
                    gr.Markdown(
                        "Create or enhance the style prompt and sectioned lyrics for YuE2."
                    )
                    writer = build_remote_prompt_writer_controls(
                        prompt_models, default_prompt_model
                    )
                    prompt_model = writer.model
                    api_key = writer.gemini_api_key
                    prompt_backend = writer.backend
                    lightning_api_key = writer.lightning_api_key
                    enhance = gr.Button("Generate / enhance YuE2 prompt")
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
            with gr.Column(scale=2):
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
                with gr.Row():
                    run = gr.Button("Generate with YuE2", variant="primary")
                    stop = gr.Button("Interrupt")
                status = gr.Textbox(label="Status", lines=7)
                model = gr.Dropdown(choices=list(model_choices), value=defaults["model"], label="Checkpoint", info="INT8 ConvRot uses less VRAM; downloaded on first use.")
                mode = gr.Dropdown(choices=[("Full score", "full"), ("Melody only", "melody"), ("Direct generation", "off")], value=defaults["mode"], label="Score mode")
                with gr.Row():
                    duration = gr.Slider(1, 900, value=defaults["duration"], step=1, label="Maximum seconds")
                    seed = gr.Number(value=defaults["seed"], precision=0, label="Seed (-1 random)")
                tiled = gr.Checkbox(value=defaults["tiled_decode"], label="Tiled audio decode", info="Recommended for long songs to reduce peak VRAM.")
                with gr.Accordion("Advanced sampling", open=False):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=defaults["steps"], step=1, label="Diffusion steps")
                        cfg = gr.Slider(0, 20, value=defaults["cfg"], step=0.05, label="Diffusion CFG")
                    with gr.Row():
                        temperature = gr.Slider(0, 5, value=defaults["temperature"], step=0.05, label="Acoustic temperature")
                        top_p = gr.Slider(0.01, 1, value=defaults["top_p"], step=0.01, label="Acoustic top-p")
                        top_k = gr.Slider(1, 32768, value=defaults["top_k"], step=1, label="Acoustic top-k")
                    repetition_penalty = gr.Slider(0.01, 10, value=defaults["repetition_penalty"], step=0.01, label="Acoustic repetition penalty")
                    with gr.Row():
                        max_abc_tokens = gr.Slider(1, 20000, value=defaults["max_abc_tokens"], step=1, label="Maximum ABC tokens")
                        abc_temperature = gr.Slider(0, 5, value=defaults["abc_temperature"], step=0.05, label="ABC temperature")
                        abc_top_p = gr.Slider(0.01, 1, value=defaults["abc_top_p"], step=0.01, label="ABC top-p")
                    with gr.Row():
                        abc_top_k = gr.Slider(1, 32768, value=defaults["abc_top_k"], step=1, label="ABC top-k")
                        abc_repetition_penalty = gr.Slider(0.01, 10, value=defaults["abc_repetition_penalty"], step=0.005, label="ABC repetition penalty")
                        abc_penalty_window = gr.Slider(1, 20000, value=defaults["abc_penalty_window"], step=1, label="ABC penalty window")
    return YuE2View(style, lyrics, abc, prompt_model, api_key, prompt_backend, lightning_api_key, enhance, enhance_status, mode, output, run, stop, status, model, duration, seed, steps, cfg, temperature, top_p, top_k, repetition_penalty, max_abc_tokens, abc_temperature, abc_top_p, abc_top_k, abc_repetition_penalty, abc_penalty_window, tiled)


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
                with gr.Accordion("Music 3 prompt writer", open=False):
                    gr.Markdown(
                        "Create or enhance the caption from text, lyrics, and optional visual reference images."
                    )
                    writer = build_remote_prompt_writer_controls(
                        prompt_models, default_prompt_model
                    )
                    prompt_model = writer.model
                    api_key = writer.gemini_api_key
                    prompt_backend = writer.backend
                    lightning_api_key = writer.lightning_api_key
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
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
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
        prompt_backend,
        lightning_api_key,
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
    mode: gr.Radio
    refresh: gr.Button
    status: gr.Markdown
    paths: gr.State
    selected: gr.State
    upload_video: gr.File
    import_video: gr.Button
    grid: gr.Gallery
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
                "Add an existing video, image, or audio file to the active library "
                "so it can be previewed alongside generated outputs."
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
                with gr.Accordion(
                    "Manage library",
                    open=False,
                    elem_classes=["h3-gallery-card", "h3-gallery-danger"],
                ) as manage:
                    confirm_delete = gr.Checkbox(
                        value=False,
                        label="I understand deletion is permanent",
                        info=(
                            "Required before deleting the selected item "
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
                                "Downloaded on first use. 7B INT8 is the default quality/VRAM "
                                "balance; FP16 favors fidelity, 7B Sharp favors stronger detail, "
                                "and MXFP8/NVFP4 are experimental speed options."
                            ),
                        )
                        ltx25_prompt = gr.Textbox(
                            label="LTX-2.5 scene prompt",
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
                                info="Can lower peak VRAM before AI processing starts.",
                            )
                        split_upscale = gr.Checkbox(
                            value=False,
                            label="Split source into clips before LTX processing",
                            info=(
                                "Opt in after an out-of-VRAM error. Processes clips "
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
