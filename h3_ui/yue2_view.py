"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


from .prompt_review import build_prompt_review
from .prompt_writer_controls import build_remote_prompt_writer_controls


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
        gr.Markdown(
            "## YuE2\nGenerate full songs from a style prompt and sectioned lyrics on the shared ComfyUI backend. Full and Melody modes plan a score first; Direct skips planning. The INT8 checkpoint downloads on first use. Model license: CC-BY-NC-4.0. [Model details](https://huggingface.co/Comfy-Org/YuE2)"
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                style = gr.Textbox(
                    label="Style prompt",
                    lines=6,
                    placeholder="Indie pop, warm female vocal, clean guitar, restrained drums...",
                )
                lyrics = gr.Textbox(
                    label="Lyrics and song structure",
                    lines=16,
                    placeholder="[verse]\nLyrics here...\n\n[chorus]\n...",
                )
                abc = gr.Textbox(
                    label="ABC score override (optional)",
                    lines=7,
                    placeholder="Leave blank to let YuE2 create the score.",
                )
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
                    enhance.h3_review = build_prompt_review()
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
                model = gr.Dropdown(
                    choices=list(model_choices),
                    value=defaults["model"],
                    label="Checkpoint",
                    info="INT8 ConvRot uses less VRAM; downloaded on first use.",
                )
                mode = gr.Dropdown(
                    choices=[
                        ("Full score", "full"),
                        ("Melody only", "melody"),
                        ("Direct generation", "off"),
                    ],
                    value=defaults["mode"],
                    label="Score mode",
                )
                with gr.Row():
                    duration = gr.Slider(
                        1,
                        900,
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
                    info="Recommended for long songs to reduce peak VRAM.",
                )
                with gr.Accordion("Advanced sampling", open=False):
                    with gr.Row():
                        steps = gr.Slider(
                            1,
                            100,
                            value=defaults["steps"],
                            step=1,
                            label="Diffusion steps",
                        )
                        cfg = gr.Slider(
                            0,
                            20,
                            value=defaults["cfg"],
                            step=0.05,
                            label="Diffusion CFG",
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            0,
                            5,
                            value=defaults["temperature"],
                            step=0.05,
                            label="Acoustic temperature",
                        )
                        top_p = gr.Slider(
                            0.01,
                            1,
                            value=defaults["top_p"],
                            step=0.01,
                            label="Acoustic top-p",
                        )
                        top_k = gr.Slider(
                            1,
                            32768,
                            value=defaults["top_k"],
                            step=1,
                            label="Acoustic top-k",
                        )
                    repetition_penalty = gr.Slider(
                        0.01,
                        10,
                        value=defaults["repetition_penalty"],
                        step=0.01,
                        label="Acoustic repetition penalty",
                    )
                    with gr.Row():
                        max_abc_tokens = gr.Slider(
                            1,
                            20000,
                            value=defaults["max_abc_tokens"],
                            step=1,
                            label="Maximum ABC tokens",
                        )
                        abc_temperature = gr.Slider(
                            0,
                            5,
                            value=defaults["abc_temperature"],
                            step=0.05,
                            label="ABC temperature",
                        )
                        abc_top_p = gr.Slider(
                            0.01,
                            1,
                            value=defaults["abc_top_p"],
                            step=0.01,
                            label="ABC top-p",
                        )
                    with gr.Row():
                        abc_top_k = gr.Slider(
                            1,
                            32768,
                            value=defaults["abc_top_k"],
                            step=1,
                            label="ABC top-k",
                        )
                        abc_repetition_penalty = gr.Slider(
                            0.01,
                            10,
                            value=defaults["abc_repetition_penalty"],
                            step=0.005,
                            label="ABC repetition penalty",
                        )
                        abc_penalty_window = gr.Slider(
                            1,
                            20000,
                            value=defaults["abc_penalty_window"],
                            step=1,
                            label="ABC penalty window",
                        )
    return YuE2View(
        style,
        lyrics,
        abc,
        prompt_model,
        api_key,
        prompt_backend,
        lightning_api_key,
        enhance,
        enhance_status,
        mode,
        output,
        run,
        stop,
        status,
        model,
        duration,
        seed,
        steps,
        cfg,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_abc_tokens,
        abc_temperature,
        abc_top_p,
        abc_top_k,
        abc_repetition_penalty,
        abc_penalty_window,
        tiled,
    )
