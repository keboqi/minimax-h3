"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr
from .image_library import build_image_input


from .prompt_review import build_prompt_review
from .prompt_writer_controls import build_remote_prompt_writer_controls
from .song_controls import (
    change_request, duration_presets, lyrics_policy, panel_heading, song_check_panel,
    song_choices, song_intro,
)
from .song_takes import SongTakes, build_take_history


@dataclass(frozen=True)
class MusicView:
    caption: gr.Textbox
    lyrics: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
    lightning_model: gr.Dropdown
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
    vocals: gr.Dropdown
    language: gr.Dropdown
    use: gr.Dropdown
    lyrics_policy: gr.Radio
    notes: gr.Textbox
    song_check: gr.HTML
    write_generate: gr.Button
    takes: SongTakes


def build_music_view(
    root: gr.Group,
    *,
    prompt_models: Sequence[str],
    default_prompt_model: str,
    model_choices: Sequence[str],
    defaults: Mapping[str, Any],
) -> MusicView:
    root.elem_classes = [*root.elem_classes, "h3-song"]
    with root:
        song_intro("MiniMax Music 3", "Your next song starts here.",
                   "An idea, a few lyrics, a feeling. Make it yours.", "SONG STUDIO")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                panel_heading("01", "The idea", "Describe the sound, story, and language you have in mind.")
                caption = gr.Textbox(
                    label="Song idea / music description",
                    lines=3,
                    placeholder=(
                        "A warm indie-pop song about taking the last train home, with English lyrics, "
                        "a gentle lead vocal, bright guitars, and a hopeful chorus."
                    ),
                    info="Start with an idea, or paste a finished music description to generate directly.",
                )
                with gr.Accordion("Try a starter idea", open=False):
                    gr.Examples(
                        examples=[
                            ["An upbeat pop song about starting over, with English lyrics, a warm lead vocal, "
                             "bright guitars, and a chorus that feels like sunrise."],
                            ["A gentle Mandarin ballad about missing home, with intimate vocals, piano, "
                             "soft strings, and a hopeful ending."],
                            ["An instrumental lo-fi track for a rainy evening, with mellow electric piano, "
                             "soft drums, warm bass, and no vocals."],
                        ],
                        inputs=caption, label="Choose an idea, then make it your own",
                        cache_examples=False,
                    )
                vocals, language, use = song_choices()
                notes = change_request()
                with gr.Row(elem_classes=["h3-song-write-action"]):
                    enhance = gr.Button("Write / improve song", variant="primary")
                    write_generate = gr.Button("Write & generate", variant="secondary")
                    gr.Markdown("**Write / improve** drafts for you to review first. **Write & generate** "
                                "applies the draft and starts a take; Undo still works.")
                enhance_status = gr.Textbox(
                    label="Prompt writer status", lines=1, interactive=False, show_label=False,
                    placeholder="Ready to help write your song.", elem_classes=["h3-song-writer-status"],
                )
                enhance.h3_review = build_prompt_review(preview_label="Suggested song draft", compact=True)
                panel_heading("02", "The draft", "Edit your lyrics here. Write keeps your words unless you choose Polish.",
                              classes="h3-song-draft-heading")
                with gr.Tabs(elem_classes=["h3-song-editor-tabs"]):
                    with gr.Tab("Lyrics"):
                        lyrics = gr.Textbox(
                            label="Lyrics and song structure", lines=9,
                            placeholder="Leave blank to write from your idea, or paste your lyrics here.",
                            info="Mark sections with [Verse], [Chorus], [Bridge]. For an instrumental, "
                                 "choose Who sings → Instrumental and leave this blank.",
                            elem_classes=["h3-song-lyrics"],
                        )
                        policy = lyrics_policy()
                    with gr.Tab("Writing tools"):
                        with gr.Accordion("Writer settings", open=True):
                            gr.Markdown("Choose a writing service and supply its API key if one isn't configured.")
                            writer = build_remote_prompt_writer_controls(prompt_models, default_prompt_model)
                        with gr.Accordion("Visual inspiration (optional)", open=False):
                            gr.Markdown("Use images to suggest a mood or story for the writer.")
                            with gr.Row():
                                references = tuple(
                                    build_image_input(type="filepath", label=f"Reference image {index}")
                                    for index in range(1, 4)
                                )
                song_check = song_check_panel("MiniMax Music 3", defaults["duration"])
                prompt_model = writer.model
                api_key = writer.gemini_api_key
                prompt_backend = writer.backend
                lightning_api_key = writer.lightning_api_key
                lightning_model = writer.lightning_model
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                panel_heading("03", "The track", "Turn your current draft into music.")
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
                output.h3_metadata_root = gr.Column()
                duration = gr.Slider(
                    1, 300, value=defaults["duration"], step=1, label="Maximum seconds",
                    info="A length limit, not an exact duration. The song may finish sooner.",
                )
                duration_presets(duration, ((30, "30 sec"), (60, "1 min"), (120, "2 min"), (180, "3 min")))
                with gr.Column(elem_classes=["h3-action-dock"]):
                    readiness = gr.HTML(
                        '<p class="h3-readiness" role="alert">Describe your song or add lyrics to start.</p>'
                    )
                    with gr.Row(elem_classes=["h3-generation-buttons"]):
                        run = gr.Button(
                            "Generate with Music 3",
                            variant="primary",
                            interactive=False,
                            elem_classes=["h3-primary-action"],
                            elem_id="h3-music3-generate",
                        )
                        stop = gr.Button("Interrupt", interactive=False)
                        run.h3_stop = stop
                        run.h3_readiness = readiness
                    status = gr.Textbox(
                        label="Generation progress", lines=2, interactive=False
                    )
                takes = build_take_history(with_score=False)
                with gr.Accordion("Generation settings (advanced)", open=False):
                    model = gr.Dropdown(
                        choices=list(model_choices), value=defaults["model"], label="Diffusion model",
                        info="Downloads on first use.",
                    )
                    seed = gr.Number(
                        value=defaults["seed"], precision=0, label="Seed (-1 random)",
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
                gr.Markdown("Listen here, download the MP3, or find your song in **Media**.")
    return MusicView(
        caption,
        lyrics,
        prompt_model,
        api_key,
        prompt_backend,
        lightning_api_key,
        lightning_model,
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
        vocals,
        language,
        use,
        policy,
        notes,
        song_check,
        write_generate,
        takes,
    )
