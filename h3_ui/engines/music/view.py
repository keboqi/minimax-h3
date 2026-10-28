"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr

from ..controls import generation_actions
from ...image_library import build_image_input
from ...prompt_review import build_prompt_review
from ...prompt_writer_controls import build_remote_prompt_writer_controls
from ...song_controls import (
    BRIEF, DRAFT, TAKES, change_request, duration_presets, lyrics_policy, press, reveal, section_inserts,
    song_check_panel, song_choices, studio_column,
)
from ...step_cards import card_heading, intro
from ...song_takes import SongTakes, build_take_history


@dataclass(frozen=True)
class MusicView:
    caption: gr.Textbox
    lyrics: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    openai_api_key: gr.Textbox
    openai_model: gr.Dropdown
    openai_base_url: gr.Textbox
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
        intro("MiniMax Music 3", "Your next song starts here.",
              "Describe it, shape the draft, record takes and compare them, all on one screen.", "SONG STUDIO")
        with gr.Row(equal_height=False, elem_classes=["h3-song-studio"]):
            with studio_column("music3", BRIEF, scale=2, min_width=280, title="Brief",
                               detail="Describe the sound, story, and language. Everything else is optional."):
                caption = gr.Textbox(
                    label="Song idea / music description",
                    lines=5,
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
                enhance = gr.Button("Write the song", variant="primary", elem_id="h3-music3-write",
                                    elem_classes=["h3-primary-action"])
                write_generate = gr.Button("Write & generate a take", variant="secondary")
                gr.Markdown("Write drafts lyrics and a production description for you to review. "
                            "**Write & generate** applies it without review and starts a take; Undo still "
                            "works.", elem_classes=["h3-song-hint"])
                with gr.Accordion("Writing service", open=False):
                    gr.Markdown("Choose a writing service and supply its API key if one isn't configured.")
                    writer = build_remote_prompt_writer_controls(prompt_models, default_prompt_model)
                with gr.Accordion("Visual inspiration (optional)", open=False):
                    gr.Markdown("Use images to suggest a mood or story for the writer.")
                    references = tuple(
                        build_image_input(type="filepath", label=f"Reference image {index}")
                        for index in range(1, 4)
                    )
            with studio_column("music3", DRAFT, scale=3, min_width=320, title="Draft",
                               detail="The song check updates as you type. Write keeps your words unless "
                                      "you choose Polish."):
                song_check = song_check_panel("MiniMax Music 3", defaults["duration"])
                enhance_status = gr.Textbox(
                    label="Prompt writer status", lines=1, interactive=False, show_label=False,
                    placeholder="Press Write the song, or type your lyrics below.",
                    elem_classes=["h3-song-writer-status"],
                )
                enhance.h3_review = build_prompt_review(preview_label="Suggested song draft", compact=True)
                lyrics = gr.Textbox(
                    label="Lyrics and song structure", lines=16,
                    placeholder="Leave blank to write from your idea, or paste your lyrics here.",
                    info="Mark sections with [Verse], [Chorus], [Bridge]. For an instrumental, "
                         "choose Who sings → Instrumental and leave this blank.",
                    elem_classes=["h3-song-lyrics"],
                )
                section_inserts(lyrics)
                policy = lyrics_policy()
            with studio_column("music3", TAKES, scale=2, min_width=300, title="Record",
                               detail="Each take is a new performance of the current draft."):
                with gr.Column(elem_classes=["h3-action-dock"]):
                    readiness = gr.HTML(
                        '<p class="h3-readiness" role="alert">Describe your song in the brief, or add lyrics, to start.</p>'
                    )
                    run, stop, status = generation_actions(
                        "Generate with Music 3", readiness=readiness, elem_id="h3-music3-generate"
                    )
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
                output.h3_metadata_root = gr.Column()
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
                takes = build_take_history(with_score=False)
                card_heading("What should change?",
                             "Say what you hear. The writer changes only that, and you review the new "
                             "draft before the next take.")
                notes = change_request()
                revise = press("Revise the draft", enhance.elem_id)
                gr.Markdown("Download any take from its player, or find every song in **Media**.",
                            elem_classes=["h3-song-hint"])
        # On a phone the draft sits below the brief and above the takes; show the review there.
        for trigger in (enhance, write_generate, revise):
            reveal(trigger, f"h3-music3-{DRAFT}")
        prompt_model = writer.model
        api_key = writer.gemini_api_key
        prompt_backend = writer.backend
        openai_api_key = writer.openai_api_key
        openai_model = writer.openai_model
        openai_base_url = writer.openai_base_url
    return MusicView(
        caption,
        lyrics,
        prompt_model,
        api_key,
        prompt_backend,
        openai_api_key,
        openai_model,
        openai_base_url,
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
    )
