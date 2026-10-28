"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


from ..controls import generation_actions
from ...prompt_review import build_prompt_review
from ...prompt_writer_controls import build_remote_prompt_writer_controls
from ...song_controls import (
    BRIEF, DRAFT, TAKES, change_request, duration_presets, lyrics_policy, press, reveal, section_inserts,
    song_check_panel, song_choices, studio_column,
)
from ...step_cards import card_heading, intro
from ...song_takes import SongTakes, build_take_history


@dataclass(frozen=True)
class YuE2View:
    style: gr.Textbox
    lyrics: gr.Textbox
    abc: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    prompt_backend: gr.Radio
    openai_api_key: gr.Textbox
    openai_model: gr.Dropdown
    openai_base_url: gr.Textbox
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
        intro(
            "YuE2 · score-planned songs",
            "Plan the melody, then sing it.",
            "Sketch an idea and YuE2 writes a score before it sings, for songs up to 15 minutes. "
            'Model license: CC-BY-NC-4.0 · <a href="https://huggingface.co/Comfy-Org/YuE2" '
            'target="_blank" rel="noopener">model details</a>',
            "SCORE STUDIO",
        )
        with gr.Row(equal_height=False, elem_classes=["h3-song-studio"]):
            with studio_column("yue2", BRIEF, scale=2, min_width=280, title="Brief",
                               detail="Describe the feeling, story, or genre in your own words. "
                                      "Everything else is optional."):
                style = gr.Textbox(
                    label="Song idea / style tags",
                    lines=5,
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
                enhance = gr.Button("Write the song", variant="primary", elem_id="h3-yue2-write",
                                    elem_classes=["h3-primary-action"])
                write_generate = gr.Button("Write & generate a take", variant="secondary")
                gr.Markdown("Write drafts lyrics and style tags for you to review. **Write & generate** "
                            "applies them without review and starts a take; Undo still works.",
                            elem_classes=["h3-song-hint"])
                with gr.Accordion("Writing service", open=False):
                    gr.Markdown("Choose a writing service and supply its API key if one isn't configured.")
                    writer = build_remote_prompt_writer_controls(prompt_models, default_prompt_model)
            with studio_column("yue2", DRAFT, scale=3, min_width=320, title="Draft",
                               detail="YuE2 sings these lyrics. Write keeps your words unless you choose Polish."):
                song_check = song_check_panel("YuE2", defaults["duration"])
                enhance_status = gr.Textbox(
                    label="Prompt writer status", lines=1, interactive=False, show_label=False,
                    placeholder="Press Write the song, or type your lyrics below.",
                    elem_classes=["h3-song-writer-status"],
                )
                enhance.h3_review = build_prompt_review(preview_label="Suggested song draft", compact=True)
                with gr.Tabs(elem_classes=["h3-song-editor-tabs"]):
                    with gr.Tab("Lyrics"):
                        lyrics = gr.Textbox(
                            label="Lyrics and song structure", lines=16,
                            placeholder="Leave blank to write from your idea, or paste your lyrics here.",
                            info="YuE2 needs lyrics or section tags. For an instrumental, choose "
                                 "Who sings → Instrumental and Write fills in the sections.",
                            elem_classes=["h3-song-lyrics"],
                        )
                        section_inserts(lyrics)
                        policy = lyrics_policy()
                    with gr.Tab("Score", render_children=True):
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
            with studio_column("yue2", TAKES, scale=2, min_width=300, title="Record",
                               detail="Each take is a new performance of the current draft."):
                with gr.Column(elem_classes=["h3-action-dock"]):
                    readiness = gr.HTML(
                        '<p class="h3-readiness" role="alert">Describe your song in the brief and add lyrics to start.</p>'
                    )
                    run, stop, status = generation_actions(
                        "Generate with YuE2", readiness=readiness, elem_id="h3-yue2-generate"
                    )
                output = gr.Audio(
                    label="Generated song", type="filepath", interactive=False
                )
                output.h3_metadata_root = gr.Column()
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
                takes = build_take_history(with_score=True)
                card_heading("What should change?",
                             "Say what you hear. The writer changes only that, and you review the new "
                             "draft before the next take. To keep a melody, reuse its score first.")
                notes = change_request()
                revise = press("Revise the draft", enhance.elem_id)
                gr.Markdown("Download any take from its player, or find every song in **Media**.",
                            elem_classes=["h3-song-hint"])
        # On a phone the draft sits below the brief and above the takes; show the review there.
        for trigger in (enhance, write_generate, revise):
            reveal(trigger, f"h3-yue2-{DRAFT}")
        prompt_model = writer.model
        api_key = writer.gemini_api_key
        prompt_backend = writer.backend
        openai_api_key = writer.openai_api_key
        openai_model = writer.openai_model
        openai_base_url = writer.openai_base_url
    return YuE2View(
        style,
        lyrics,
        abc,
        prompt_model,
        api_key,
        prompt_backend,
        openai_api_key,
        openai_model,
        openai_base_url,
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
    )
