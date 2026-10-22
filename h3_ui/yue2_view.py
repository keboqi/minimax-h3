"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


from .prompt_review import build_prompt_review
from .prompt_writer_controls import build_remote_prompt_writer_controls
from .song_controls import (
    DRAFT, IDEA, LISTEN, TAKE, card_heading, change_request, duration_presets, lyrics_policy,
    render_recap, song_check_panel, song_choices, song_intro, song_steps,
)
from .song_takes import SongTakes, build_take_history


@dataclass(frozen=True)
class YuE2View:
    style: gr.Textbox
    lyrics: gr.Textbox
    abc: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    lightning_api_key: gr.Textbox
    lightning_model: gr.Dropdown
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
    vocals: gr.Dropdown
    language: gr.Dropdown
    use: gr.Dropdown
    lyrics_policy: gr.Radio
    notes: gr.Textbox
    song_check: gr.HTML
    last_score: gr.Textbox
    reuse_score: gr.Button
    write_generate: gr.Button
    takes: SongTakes
    recap: gr.HTML


def build_yue2_view(
    root: gr.Group,
    *,
    prompt_models: Sequence[str],
    default_prompt_model: str,
    model_choices: Sequence[str],
    defaults: Mapping[str, Any],
) -> YuE2View:
    root.elem_classes = [*root.elem_classes, "h3-song"]
    with root:
        song_intro(
            "YuE2 · score-planned songs",
            "Plan the melody, then sing it.",
            "Sketch an idea and YuE2 writes a score before it sings, for songs up to 15 minutes. "
            'Model license: CC-BY-NC-4.0 · <a href="https://huggingface.co/Comfy-Org/YuE2" '
            'target="_blank" rel="noopener">model details</a>',
            "SCORE STUDIO",
        )
        steps = song_steps("h3-yue2-steps")
        with steps.tabs:
            with steps.step(IDEA, "Idea"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("What would you like to make?",
                                     "Describe the feeling, story, or genre in your own words. "
                                     "Everything below the idea is optional.")
                        style = gr.Textbox(
                            label="Song idea / style tags",
                            lines=4,
                            placeholder=(
                                "Indie pop with a warm female vocal, clean guitar, restrained drums, "
                                "and a hopeful, building chorus."
                            ),
                            info="Start with an idea, or paste a finished style prompt to generate directly.",
                        )
                        with gr.Accordion("Try a starter idea", open=False):
                            gr.Examples(
                                examples=[
                                    ["Anthemic indie rock about a night drive, with a raspy male vocal, "
                                     "driving drums, wide guitars, and a big final chorus."],
                                    ["A soulful R&B slow jam about a late-night phone call, with a smooth "
                                     "female vocal, Rhodes piano, and a deep, warm bass."],
                                    ["A bright Mandopop song about the first day of summer, with an airy "
                                     "female vocal, acoustic guitar, and handclaps."],
                                ],
                                inputs=style, label="Choose an idea, then make it your own",
                                cache_examples=False,
                            )
                        vocals, language, use = song_choices()
                        duration = gr.Slider(
                            1, 900, value=defaults["duration"], step=1, label="Maximum seconds",
                            info="A length limit, not an exact duration. Write plans a song that fits; "
                                 "the song may finish sooner.",
                        )
                        duration_presets(duration, ((60, "1 min"), (120, "2 min"), (180, "3 min"), (300, "5 min")))
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Write the lyrics",
                                     "The writer drafts lyrics and style tags from your idea. You review "
                                     "every change on the next step.")
                        enhance = gr.Button("Write the song", variant="primary", elem_id="h3-yue2-write",
                                            elem_classes=["h3-primary-action"])
                        write_generate = gr.Button("Write & generate a take", variant="secondary")
                        steps.button("I have my own lyrics →", DRAFT, classes=("h3-song-skip",))
                        gr.Markdown("**Write & generate** applies the draft without review and starts a "
                                    "take; Undo still works.", elem_classes=["h3-song-hint"])
                        with gr.Accordion("Writing service", open=False):
                            gr.Markdown("Choose a writing service and supply its API key if one isn't configured.")
                            writer = build_remote_prompt_writer_controls(prompt_models, default_prompt_model)
            with steps.step(DRAFT, "Lyrics & score"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("Shape the draft",
                                     "YuE2 sings these lyrics. Write keeps your words unless you choose Polish.")
                        enhance_status = gr.Textbox(
                            label="Prompt writer status", lines=1, interactive=False, show_label=False,
                            placeholder="Press Write the song on the Idea step, or type your lyrics below.",
                            elem_classes=["h3-song-writer-status"],
                        )
                        enhance.h3_review = build_prompt_review(preview_label="Suggested song draft", compact=True)
                        with gr.Tabs(elem_classes=["h3-song-editor-tabs"]):
                            with gr.Tab("Lyrics"):
                                lyrics = gr.Textbox(
                                    label="Lyrics and song structure", lines=14,
                                    placeholder="Leave blank to write from your idea, or paste your lyrics here.",
                                    info="YuE2 needs lyrics or section tags. For an instrumental, choose "
                                         "Who sings → Instrumental and Write fills in the sections.",
                                    elem_classes=["h3-song-lyrics"],
                                )
                                policy = lyrics_policy()
                            with gr.Tab("Score", render_children=True):
                                gr.Markdown(
                                    "YuE2 writes a score (melody and chords) before it sings. Liked a melody? "
                                    "Reuse its score and change the style or sound around it. The score guides "
                                    "the melody; the voice and performance still vary from take to take."
                                )
                                last_score = gr.Textbox(
                                    label="Score from the latest take", lines=5, interactive=False,
                                    placeholder="Appears after a take that planned its own score.",
                                )
                                reuse_score = gr.Button("Reuse this score for the next take", size="sm")
                                abc = gr.Textbox(
                                    label="Score to use (optional)",
                                    lines=7,
                                    placeholder="Leave blank to let YuE2 compose a new score.",
                                    info="Ignored by No score (fastest).",
                                )
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Song check", "How the draft fits the length, updated as you type.")
                        song_check = song_check_panel("YuE2", defaults["duration"])
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
                            engine="YuE2", sound_label="Style", description="", lyrics="",
                            max_seconds=defaults["duration"],
                        ))
                        # A Dropdown keeps the published /generate_yue2 schema unchanged.
                        mode = gr.Dropdown(
                            choices=[
                                ("Plan melody + chords (best)", "full"),
                                ("Plan melody only", "melody"),
                                ("No score (fastest)", "off"),
                            ],
                            value=defaults["mode"],
                            label="Composition",
                            info="Planning a score first gives a stronger melody and clearer song form.",
                        )
                        with gr.Accordion("Generation settings (advanced)", open=False):
                            model = gr.Dropdown(
                                choices=list(model_choices),
                                value=defaults["model"],
                                label="Checkpoint",
                                info="INT8 ConvRot uses less VRAM; downloads on first use.",
                            )
                            seed = gr.Number(
                                value=defaults["seed"],
                                precision=0,
                                label="Seed (-1 random)",
                            )
                            tiled = gr.Checkbox(
                                value=defaults["tiled_decode"],
                                label="Tiled audio decode",
                                info="Recommended for long songs to reduce peak VRAM.",
                            )
                            with gr.Accordion("Sampling", open=False):
                                with gr.Row():
                                    diffusion_steps = gr.Slider(
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
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Record a take", "Each take is a new performance of the current draft.")
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            readiness = gr.HTML(
                                '<p class="h3-readiness" role="alert">Describe your song on the Idea step and add lyrics to start.</p>'
                            )
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                run = gr.Button(
                                    "Generate with YuE2",
                                    variant="primary",
                                    interactive=False,
                                    elem_classes=["h3-primary-action"],
                                    elem_id="h3-yue2-generate",
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
                        takes = build_take_history(with_score=True)
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Refine the song",
                                     "Say what you hear. The writer changes only that, and you review "
                                     "the new draft before the next take. To keep a melody, reuse its "
                                     "score first.")
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
    return YuE2View(
        style,
        lyrics,
        abc,
        prompt_model,
        api_key,
        prompt_backend,
        lightning_api_key,
        lightning_model,
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
        diffusion_steps,
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
        vocals,
        language,
        use,
        policy,
        notes,
        song_check,
        last_score,
        reuse_score,
        write_generate,
        takes,
        recap,
    )
