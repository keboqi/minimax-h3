"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr
from .image_library import build_image_input


from .prompt_review import build_prompt_review
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
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                gr.Markdown("### Compose")
                caption = gr.Textbox(
                    label="Music caption",
                    lines=4,
                    placeholder=(
                        "Global Metadata: genre, BPM, key, mood, production...\n\n"
                        "Vocal Details: singer, delivery, harmonies, effects...\n\n"
                        "Arrangement: instruments, groove, section-by-section evolution..."
                    ),
                )
                lyrics = gr.Textbox(
                    label="Lyrics and song structure",
                    lines=8,
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
                            build_image_input(type="filepath", label=f"Reference image {index}")
                            for index in range(1, 4)
                        )
                    enhance = gr.Button("Generate / enhance Music 3 caption")
                    enhance.h3_review = build_prompt_review()
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
                action_root = gr.Column(elem_classes=["h3-action-dock"])
                settings_root = gr.Column(elem_classes=["h3-essentials"])
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                gr.Markdown("### Results")
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
                output.h3_metadata_root = gr.Column()
                with action_root:
                    readiness = gr.HTML(
                        '<p class="h3-readiness" role="alert">Write a prompt to start.</p>'
                    )
                    with gr.Row():
                        run = gr.Button(
                            "Generate with Music 3",
                            variant="primary",
                            interactive=False,
                        )
                        stop = gr.Button("Interrupt", interactive=False)
                        run.h3_stop = stop
                        run.h3_readiness = readiness
                    status = gr.Textbox(
                        label="Generation progress", lines=2, interactive=False
                    )
                with settings_root:
                    gr.Markdown("### Generation settings")
                    with gr.Accordion("Model (advanced)", open=False):
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
                            value=defaults["seed"],
                            precision=0,
                            label="Seed (-1 random)",
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
