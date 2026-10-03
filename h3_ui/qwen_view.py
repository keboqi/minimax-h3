"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr
from h3_models import QWEN_IMAGE21_TURBO_MODES

from h3_app.catalog import (
    QWEN_EDIT_SIZE_MATCH,
    QWEN_EDIT_SIZE_MAX,
    QWEN_EDIT_SIZE_MANUAL,
    QWEN_IMAGE21_DYNAMIC_SCHEDULER,
)

from .prompt_review import build_prompt_review
from .prompt_writer_controls import build_remote_prompt_writer_controls


QWEN_1K_RESOLUTION_PRESETS: tuple[str, ...] = (
    "1K · 1:1 · 1024×1024",
    "1K · 4:3 · 1376×1024",
    "1K · 3:2 · 1536×1024",
    "1K · 16:9 · 1824×1024",
    "1K · 3:4 · 1024×1376",
    "1K · 2:3 · 1024×1536",
    "1K · 9:16 · 1024×1824",
)


QWEN_2K_RESOLUTION_PRESETS: tuple[str, ...] = (
    "2K · 1:1 · 2048×2048",
    "2K · 4:3 · 2400×1792",
    "2K · 3:2 · 2528×1696",
    "2K · 16:9 · 2752×1536",
    "2K · 3:4 · 1792×2400",
    "2K · 2:3 · 1696×2528",
    "2K · 9:16 · 1536×2752",
)


@dataclass(frozen=True)
class QwenImage21View:
    mode: gr.Dropdown
    prompt: gr.Textbox
    negative_prompt: gr.Textbox
    reference_images: gr.File
    batch_edit_inputs: gr.Checkbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
    enhance: gr.Button
    enhance_status: gr.Textbox
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    model: gr.Dropdown
    text_encoder: gr.Dropdown
    output_resolution_1k: gr.Dropdown
    output_resolution_2k: gr.Dropdown
    width: gr.Slider
    height: gr.Slider
    reference_resolution: gr.Dropdown
    edit_size: gr.Radio
    seed: gr.Number
    batch_count: gr.Slider
    steps: gr.Slider
    cfg: gr.Slider
    sampler: gr.Dropdown
    scheduler: gr.Dropdown
    cache_device: gr.Dropdown
    cache_dtype: gr.Dropdown
    attention_backend: gr.Dropdown
    accelerator: gr.Dropdown
    turbo_variant: gr.Dropdown
    preset: gr.Radio

    @property
    def output_resolution(self) -> gr.Dropdown:
        return self.output_resolution_1k


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
            "ComfyUI workflow. With separate editing off, the first image is the "
            "target. For multiple reference inputs, mention them as `<image1>`, "
            "`<image2>`, and so on; "
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
                    label="Input / reference images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                batch_edit_inputs = gr.Checkbox(
                    label="Edit each uploaded image separately",
                    value=False,
                    info=(
                        "Apply the same prompt and settings to each image on its own. "
                        "Outputs = uploaded images × Images per batch."
                    ),
                )
                gr.Markdown(
                    "With separate editing off, base editing supports up to 10 references "
                    "and Turbo supports up to 3. **image1** is the edit target; later "
                    "images are references. Numbered tags are required only when "
                    "multiple references are supplied."
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
                    enhance.h3_review = build_prompt_review()
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
            with gr.Column(scale=2):
                output = gr.Gallery(
                    label="Generated images", columns=2, height=500, interactive=False
                )
                with gr.Row():
                    run = gr.Button("Generate with Qwen Image 2.1", variant="primary")
                    stop = gr.Button("Interrupt")
                status = gr.Textbox(label="Status", lines=7)
                preset = gr.Radio(
                    choices=["Fast", "Normal", "Quality"],
                    value="Quality",
                    label="Qwen preset",
                    info="Sets the diffusion model, Turbo mode, steps, and accelerator. Controls remain editable.",
                )
                model = gr.Dropdown(
                    choices=list(model_choices),
                    value=defaults["model"],
                    label="Diffusion model",
                )
                turbo_variant = gr.Dropdown(
                    choices=["Off", *QWEN_IMAGE21_TURBO_MODES],
                    value="Off",
                    label="Turbo mode",
                    info=(
                        "v0.2.1 and v0.3 offer 6 turbo steps. v0.3 also offers "
                        "7 turbo steps + 2 base steps for finer detail. "
                        "Use Euler, CFG 1, and accelerator Off. "
                        "Research and evaluation use only."
                    ),
                )
                text_encoder = gr.Dropdown(
                    choices=list(text_encoder_choices),
                    value=defaults["text_encoder"],
                    label="Qwen3-VL text encoder",
                )
                with gr.Row():
                    output_resolution_1k = gr.Dropdown(
                        choices=list(QWEN_1K_RESOLUTION_PRESETS),
                        value=None,
                        label="1K resolution preset",
                        info="Native 1K sizes.",
                    )
                    output_resolution_2k = gr.Dropdown(
                        choices=list(QWEN_2K_RESOLUTION_PRESETS),
                        value=None,
                        label="2K resolution preset",
                        info="Native 2K sizes.",
                    )
                with gr.Row():
                    width = gr.Slider(
                        256, 2752, value=defaults["width"], step=32, label="Width"
                    )
                    height = gr.Slider(
                        256, 2752, value=defaults["height"], step=32, label="Height"
                    )
                edit_size = gr.Radio(
                    choices=[
                        QWEN_EDIT_SIZE_MATCH,
                        QWEN_EDIT_SIZE_MAX,
                        QWEN_EDIT_SIZE_MANUAL,
                    ],
                    value=defaults["edit_size"],
                    label="Edit output size",
                    info=(
                        "Match the edit target, scale its aspect ratio toward 4 MP "
                        "within 2752 × 2752, or use the width and height above. "
                        "Applies only to image editing. An output resolution preset takes priority."
                    ),
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
                    batch_count = gr.Slider(
                        1,
                        4,
                        value=1,
                        step=1,
                        label="Images per batch",
                        info="Uses the same inputs and a different seed for each image.",
                    )
                    steps = gr.Slider(
                        1, 100, value=defaults["steps"], step=1, label="Steps"
                    )
                    cfg = gr.Slider(0, 20, value=defaults["cfg"], step=0.1, label="CFG")
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
                            choices=[
                                (
                                    "Qwen 2.1 (resolution-aware)",
                                    QWEN_IMAGE21_DYNAMIC_SCHEDULER,
                                ),
                                "simple",
                                "normal",
                                "beta",
                            ],
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
                    "Base defaults are CFG 1 and 40 Euler steps with resolution-aware "
                    "scheduling. Native "
                    "sizes include 2048×2048, 2400×1792, 1792×2400, 2528×1696, "
                    "1696×2528, 2752×1536, and 1536×2752. For transparent PNGs, "
                    "ask for an RGBA image with an alpha channel and transparent background. "
                    "Reference images retain their alpha channels."
                )
    return QwenImage21View(
        mode,
        prompt,
        negative_prompt,
        reference_images,
        batch_edit_inputs,
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
        output_resolution_1k,
        output_resolution_2k,
        width,
        height,
        reference_resolution,
        edit_size,
        seed,
        batch_count,
        steps,
        cfg,
        sampler,
        scheduler,
        cache_device,
        cache_dtype,
        attention_backend,
        accelerator,
        turbo_variant,
        preset,
    )
