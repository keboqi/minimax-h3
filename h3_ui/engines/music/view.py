"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr

from ...image_library import build_image_input
from ...prompt_review import build_prompt_review
from ...prompt_writer_controls import build_remote_prompt_writer_controls
from ...song_controls import (
    DRAFT, IDEA, LISTEN, TAKE, card_heading, change_request, duration_presets, lyrics_policy,
    render_recap, song_check_panel, song_choices, song_intro, song_steps,
)
from ...song_takes import SongTakes, build_take_history


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
    recap: gr.HTML


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
                   "Describe it, shape the lyrics, record a take, then listen and refine.", "SONG STUDIO")
        steps = song_steps("h3-music3-steps")
        with steps.tabs:
            with steps.step(IDEA, "Idea"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("What would you like to make?",
                                     "Describe the sound, story, and language you have in mind. "
                                     "Everything below the idea is optional.")
                        caption = gr.Textbox(
                            label="Song idea / music description",
                            lines=4,
                            placeholder=(
                                "A warm indie-pop song about taking the last train home, with English lyrics, "
                                "a gentle lead vocal, bright guitars, and a hopeful chorus."
                            ),
                            info="Start with an idea, or paste a finished music description to generate directly.",
                        )
                        with gr.Accordion("Try a starter idea", open=False):
                            gr.Examples(
                                examples=[
                                    ["An upbeat pop song about starting over, with English lyrics, a warm lead "
                                     "vocal, bright guitars, and a chorus that feels like sunrise."],
                                    ["A gentle Mandarin ballad about missing home, with intimate vocals, piano, "
                                     "soft strings, and a hopeful ending."],
                                    ["An instrumental lo-fi track for a rainy evening, with mellow electric "
                                     "piano, soft drums, warm bass, and no vocals."],
                                ],
                                inputs=caption, label="Choose an idea, then make it your own",
                                cache_examples=False,
                            )
                        vocals, language, use = song_choices()
                        duration = gr.Slider(
                            1, 300, value=defaults["duration"], step=1, label="Maximum seconds",
                            info="A length limit, not an exact duration. Write plans a song that fits; "
                                 "the song may finish sooner.",
                        )
                        duration_presets(duration, ((30, "30 sec"), (60, "1 min"), (120, "2 min"), (180, "3 min")))
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Write the lyrics",
                                     "The writer drafts lyrics and a production description from your idea. "
                                     "You review every change on the next step.")
                        enhance = gr.Button("Write the song", variant="primary", elem_id="h3-music3-write",
                                            elem_classes=["h3-primary-action"])
                        write_generate = gr.Button("Write & generate a take", variant="secondary")
                        steps.button("I have my own lyrics →", DRAFT, classes=("h3-song-skip",))
                        gr.Markdown("**Write & generate** applies the draft without review and starts a "
                                    "take; Undo still works.", elem_classes=["h3-song-hint"])
                        with gr.Accordion("Writing service", open=False):
                            gr.Markdown("Choose a writing service and supply its API key if one isn't configured.")
                            writer = build_remote_prompt_writer_controls(prompt_models, default_prompt_model)
                        with gr.Accordion("Visual inspiration (optional)", open=False):
                            gr.Markdown("Use images to suggest a mood or story for the writer.")
                            references = tuple(
                                build_image_input(type="filepath", label=f"Reference image {index}")
                                for index in range(1, 4)
                            )
            with steps.step(DRAFT, "Lyrics"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("Shape the draft",
                                     "Review the writer's suggestion, then edit the lyrics. Write keeps "
                                     "your words unless you choose Polish.")
                        enhance_status = gr.Textbox(
                            label="Prompt writer status", lines=1, interactive=False, show_label=False,
                            placeholder="Press Write the song on the Idea step, or type your lyrics below.",
                            elem_classes=["h3-song-writer-status"],
                        )
                        enhance.h3_review = build_prompt_review(preview_label="Suggested song draft", compact=True)
                        lyrics = gr.Textbox(
                            label="Lyrics and song structure", lines=14,
                            placeholder="Leave blank to write from your idea, or paste your lyrics here.",
                            info="Mark sections with [Verse], [Chorus], [Bridge]. For an instrumental, "
                                 "choose Who sings → Instrumental and leave this blank.",
                            elem_classes=["h3-song-lyrics"],
                        )
                        policy = lyrics_policy()
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Song check", "How the draft fits the length, updated as you type.")
                        song_check = song_check_panel("MiniMax Music 3", defaults["duration"])
                        steps.button("Write again", DRAFT, press=enhance.elem_id)
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Idea", IDEA)
                            steps.button("Next: generate →", TAKE, variant="primary")
            with steps.step(TAKE, "Generate"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("Ready to record",
                                     "What the next take will use. Change it on the Idea or Lyrics step.")
                        recap = gr.HTML(render_recap(
                            engine="MiniMax Music 3", sound_label="Sound", description="", lyrics="",
                            max_seconds=defaults["duration"],
                        ))
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
                                diffusion_steps = gr.Slider(
                                    1, 100, value=defaults["steps"], step=1, label="Diffusion steps",
                                )
                                with gr.Row():
                                    cfg = gr.Slider(
                                        0, 10, value=defaults["cfg"], step=0.05, label="Diffusion CFG",
                                    )
                                    ar_cfg = gr.Slider(
                                        0, 10, value=defaults["ar_cfg"], step=0.05, label="Autoregressive CFG",
                                    )
                                top_k = gr.Slider(
                                    1, 200, value=defaults["top_k"], step=1, label="Autoregressive Top K",
                                )
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Record a take", "Each take is a new performance of the current draft.")
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            readiness = gr.HTML(
                                '<p class="h3-readiness" role="alert">Describe your song on the Idea step, or add lyrics, to start.</p>'
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
                        output = gr.Audio(
                            label="Generated song", type="filepath", interactive=False
                        )
                        output.h3_metadata_root = gr.Column()
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Lyrics", DRAFT)
                            steps.button("Listen & refine →", LISTEN, variant="primary")
            with steps.step(LISTEN, "Listen & refine"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        takes = build_take_history(with_score=False)
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Refine the song",
                                     "Say what you hear. The writer changes only that, and you review "
                                     "the new draft before the next take.")
                        notes = change_request()
                        steps.button("Revise the draft", DRAFT, variant="primary", press=enhance.elem_id)
                        steps.button("Generate another take", TAKE, press=run.elem_id)
                        gr.Markdown("Download any take from its player, or find every song in **Media**.",
                                    elem_classes=["h3-song-hint"])
        steps.follow(enhance, DRAFT)
        steps.follow(write_generate, DRAFT)
        steps.follow(run, TAKE)
        prompt_model = writer.model
        api_key = writer.gemini_api_key
        prompt_backend = writer.backend
        lightning_api_key = writer.lightning_api_key
        lightning_model = writer.lightning_model
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
        diffusion_steps,
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
        recap,
    )
