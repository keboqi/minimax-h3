"""Dub & Translate as three steps: Source, Review lines and Dub.

The steps share Song's stepper: each pairs a working card with an action card
that holds the step's next move. A finished analysis, or a reopened session,
opens Review lines.

- Source: the file (uploaded, from Media, or an earlier session), its languages
  and how to find the lines; the action card lists the analysis steps and marks
  the one running.
- Review lines: a board of every line (filters, a timeline per speaker, picks and
  Dub switches) beside an inspector that edits the focused line. The plain table
  stays available for bulk edits.
- Dub: a card per speaker, voices, timing and mix; the action card says what the
  dub will do and plays the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from h3_app.dubbing.pipeline import (
    PIPELINE_GEMINI, PIPELINE_LOCAL, PIPELINE_MOSS, SEPARATOR_MODELS, SOURCE_LANGUAGES, TARGET_LANGUAGES,
    TIMING_MATCH, TIMING_NATURAL, TIMING_SMART, TRANSLATOR_GEMINI, TRANSLATOR_HYMT, TRANSLATORS, VOICE_LINE,
    VOICE_MODES, VOICE_PRESETS,
)
from h3_app.dubbing.segments import TABLE_HEADERS, TABLE_TYPES

from ..step_cards import StepStrip, card_heading, intro, step_strip
from . import presenters, widgets
from .player import ELEM_CLASS as PLAYER_CLASS, PLAYER_JS

SOURCE, LINES, DUB = range(3)
STEPS_ID = "h3-dub-steps"
DEFAULT_SEPARATOR = "Balanced (BS-Roformer)"
TRANSCRIPTIONS = (
    (PIPELINE_LOCAL, "Qwen3-ASR on this server",
     "Private and free, with word timing. Every line is one speaker, voiced from its own audio."),
    (PIPELINE_MOSS, "MOSS on this server",
     "Tells speakers apart, so each keeps their own voice. Needs Linux; up to 90 minutes."),
    (PIPELINE_GEMINI, "Gemini (cloud)",
     "Tells speakers apart and describes them. Uploads the vocals; needs an API key."),
)
TIMING_CHOICES = (
    (TIMING_SMART, "Smart timing",
     "Keeps each take's own pace, using the silence around it, shorter pauses and a gentle speed-up before "
     "anything faster. Lines still too long are marked."),
    (TIMING_MATCH, "Match the original",
     "Every take starts and ends where its line does. Best for lip-sync and on-screen cues."),
    (TIMING_NATURAL, "Natural pace",
     "Keeps each take's own pace and speeds up long ones only up to a limit, so takes may overlap."),
)


@dataclass(frozen=True)
class DubbingView:
    steps: StepStrip
    session: gr.Textbox
    # Source
    source: gr.File
    library: gr.Dropdown
    refresh_library: gr.Button
    media_tab: gr.Tab
    sessions_tab: gr.Tab
    recent: gr.HTML
    source_card: gr.HTML
    source_language: gr.Dropdown
    target_language: gr.Dropdown
    notes: gr.Textbox
    pipeline: gr.HTML
    translate: gr.Checkbox
    translator: gr.Dropdown
    gemini: gr.Group
    gemini_model: gr.Dropdown
    api_key: gr.Textbox
    options: gr.Accordion
    separate: gr.Checkbox
    separator_model: gr.Dropdown
    reuse: gr.Checkbox
    local: gr.Group
    aligner: gr.Checkbox
    merge_gap: gr.Slider
    refine_timestamps: gr.Checkbox
    plan: gr.HTML
    analyze: gr.Button
    stop_analyze: gr.Button
    analyze_status: gr.Markdown
    # Review lines
    board: gr.HTML
    lines: gr.Dataframe
    # The line the inspector shows (see DubbingHandlers.focus_line).
    inspected: gr.State
    inspector: gr.HTML
    previous_line: gr.Button
    next_line: gr.Button
    start: gr.Textbox
    end: gr.Textbox
    speaker: gr.Dropdown
    original: gr.Textbox
    translation: gr.Textbox
    line_dub: gr.Checkbox
    fit: gr.Markdown
    add_line: gr.Button
    delete_line: gr.Button
    line_numbers: gr.Textbox
    preview: gr.Button
    listen_original: gr.Button
    new_take: gr.Button
    retranslate: gr.Button
    translation_context: gr.Checkbox
    stop_preview: gr.Button
    line_status: gr.Markdown
    preview_audio: gr.Audio
    original_audio: gr.Audio
    # Dub
    speakers_panel: gr.Group
    speakers: gr.HTML
    speakers_status: gr.Markdown
    voice_mode: gr.Radio
    preset: gr.Radio
    timing: gr.HTML
    max_speed: gr.Slider
    speech_volume: gr.Slider
    background_volume: gr.Slider
    keep_original: gr.Checkbox
    advanced: gr.Accordion
    regenerate: gr.Textbox
    parallel: gr.Slider
    readiness: gr.HTML
    run: gr.Button
    stop: gr.Button
    status: gr.Markdown
    player: gr.HTML
    video: gr.Video
    audio: gr.Audio
    files: gr.File
    export_source: gr.State
    export_subtitles: gr.Button
    exported_video: gr.File
    export_status: gr.Markdown
    word_timing: gr.Checkbox
    align_words: gr.Button
    stop_align_words: gr.Button


def opened_session_js(steps: StepStrip) -> str:
    """Browser code that opens Review lines when a different session arrives.

    Analysis reports errors and interruptions as status lines, so its event always
    ends cleanly; only a new session id means there are new lines to review. A click
    that lands while the step strip is still updating is lost, so it is repeated until
    Review lines is open (for at most two seconds).
    """
    return (
        "(id) => { "
        f"const root = document.getElementById({steps.tabs.elem_id!r}); "
        "if (!id || !root || root.dataset.session === id) return; "
        "root.dataset.session = id; "
        f"({steps.go(LINES)})(); "
        "let tries = 0; "
        "const check = () => { "
        f"const tab = root.querySelectorAll(':scope > .tab-wrapper [role=\"tab\"]')[{LINES}]; "
        "if (!tab || tab.getAttribute('aria-selected') === 'true' || ++tries > 20) return; "
        "tab.click(); setTimeout(check, 100); }; "
        "setTimeout(check, 100); }"
    )


def default_plan() -> str:
    return presenters.plan_html(presenters.plan_steps(
        separate=True, separator_model=DEFAULT_SEPARATOR, reuse=True, transcription=PIPELINE_LOCAL, translate=True,
        translator=TRANSLATOR_HYMT, target_language="English"))


def build_dubbing_view(root: gr.Column, *, gemini_models: Sequence[str], default_gemini_model: str) -> DubbingView:
    root.elem_classes = [*root.elem_classes, "h3-song", "h3-dub"]
    with root:
        intro("Dub & Translate", "Hear it in another language.",
              "Bring a video or recording, check every translated line, then dub it in each speaker's "
              "own voice.", "DUB STUDIO")
        session = gr.Textbox(visible=False)
        steps = step_strip(STEPS_ID)
        with steps.tabs:
            with steps.step(SOURCE, "Source"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer", "h3-dub-source"]):
                        card_heading("What should be dubbed?")
                        with gr.Tabs(elem_classes=["h3-dub-source-tabs"]):
                            with gr.Tab("Upload"):
                                source = gr.File(label="Video or audio", file_count="single",
                                                 file_types=["video", "audio"], type="filepath", height=120)
                            with gr.Tab("From Media") as media_tab:
                                with gr.Row(equal_height=True, elem_classes=["h3-dub-library"]):
                                    library = gr.Dropdown([], value=None, label="Video or audio in Media", scale=4)
                                    refresh_library = gr.Button("Refresh", size="sm", scale=0, min_width=90)
                                gr.Markdown("Online videos: download them first in **Media → Download from a "
                                            "URL**; they appear here when done.", elem_classes=["h3-song-hint"])
                            with gr.Tab("Earlier session") as sessions_tab:
                                gr.Markdown("Analyzed sessions are kept for 14 days. Open one to review and dub "
                                            "its lines again.", elem_classes=["h3-song-hint"])
                                recent = widgets.recent_sessions()
                        source_card = gr.HTML(presenters.source_card(None))
                        card_heading("Languages")
                        with gr.Row(equal_height=True, elem_classes=["h3-dub-languages"]):
                            source_language = gr.Dropdown(list(SOURCE_LANGUAGES), value="Auto-detect",
                                                          label="Spoken language", min_width=160)
                            gr.HTML('<span class="h3-dub-arrow" aria-hidden="true">→</span>', min_width=24,
                                    elem_classes=["h3-dub-arrow-cell"])
                            target_language = gr.Dropdown(list(TARGET_LANGUAGES), value="English",
                                                          label="Dub into", allow_custom_value=True, min_width=160,
                                                          info="Type any language not listed.")
                        notes = gr.Textbox(label="Translation notes (optional)", lines=2,
                                           placeholder="Names, terms or tone to keep, e.g. 'Keep the name Mei; "
                                                       "casual tone.'")
                        card_heading("How to find the lines",
                                     "Finds who speaks when, transcribes every line and translates it.")
                        pipeline = widgets.choice_cards(TRANSCRIPTIONS, PIPELINE_LOCAL, label="Transcription",
                                                        name="h3-dub-transcription")
                        with gr.Row(equal_height=True):
                            translator = gr.Dropdown(list(TRANSLATORS), value=TRANSLATOR_HYMT, label="Translator")
                            translate = gr.Checkbox(True, label="Translate now",
                                                    info="Off: transcribe only; translate while reviewing.")
                        with gr.Group(visible=False) as gemini:
                            gemini_model = gr.Dropdown(list(gemini_models), value=default_gemini_model,
                                                       label="Gemini model")
                            api_key = gr.Textbox(label="Gemini API key for this session", type="password",
                                                 placeholder="Uses GEMINI_API_KEY when empty")
                        with gr.Accordion(presenters.options_label(
                                separate=True, separator_model=DEFAULT_SEPARATOR, reuse=True,
                                transcription=PIPELINE_LOCAL, aligner=True, merge_gap=0.35),
                                open=False, elem_classes=["h3-dub-options"]) as options:
                            separate = gr.Checkbox(True, label="Separate vocals from the background",
                                                   info="Keeps music and effects under the dub and gives clean "
                                                        "voice references. Off: the original is ducked under the "
                                                        "dub.")
                            separator_model = gr.Dropdown(list(SEPARATOR_MODELS), value=DEFAULT_SEPARATOR,
                                                          label="Separation model")
                            reuse = gr.Checkbox(True, label="Reuse earlier results for this file",
                                                info="An earlier analysis of the same file with the same settings "
                                                     "lends its separated vocals and transcript. Off: redo both.")
                            with gr.Group() as local:
                                aligner = gr.Checkbox(True, label="Word timing with Qwen3-ForcedAligner",
                                                      info="Splits long speech into subtitle-sized lines at "
                                                           "sentence ends.")
                                merge_gap = gr.Slider(0.0, 1.5, value=0.35, step=0.05,
                                                      label="Join pauses shorter than (s)",
                                                      info="Speech separated by a shorter pause is transcribed "
                                                           "together.")
                            refine_timestamps = gr.Checkbox(True, label="Refine speech boundaries with Align (experimental)",
                                info="Uses [Desert Ant Labs Align](https://desertant.com/models/align/). "
                                     "Refines Qwen's word timing on speech chunks. Keeps original timing on "
                                     "failure. Requires Linux and Node.js 22+.")
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Analyze", "Review lines opens when it is done.")
                        plan = gr.HTML(default_plan())
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                analyze = gr.Button("Analyze source", variant="primary", scale=3,
                                                    elem_classes=["h3-primary-action"])
                                stop_analyze = gr.Button("Interrupt", scale=1, interactive=False)
                            analyze_status = gr.Markdown(elem_classes=["h3-dub-status"])
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("Review lines →", LINES)
            with steps.step(LINES, "Review lines"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer", "h3-dub-review"]):
                        board = widgets.line_board()
                        with gr.Accordion("Edit as a table", open=False, elem_classes=["h3-dub-table"]):
                            gr.Markdown("Every line at once: edit times, speakers, the original or the "
                                        "translation; changes save as you type. A new row is saved once it has "
                                        "a start, an end and text.", elem_classes=["h3-song-hint"])
                            lines = gr.Dataframe(headers=list(TABLE_HEADERS), datatype=list(TABLE_TYPES), value=[],
                                                 type="array", interactive=True, wrap=True, max_height=560,
                                                 label="Lines", show_label=False, static_columns=[0],
                                                 column_widths=["5%", "9%", "9%", "9%", "30%", "32%", "6%"],
                                                 elem_classes=["h3-dub-lines"])
                    with gr.Column(scale=2, min_width=300,
                                   elem_classes=["h3-preview-panel", "h3-song-stage", "h3-dub-inspector"]):
                        with gr.Row(equal_height=True, elem_classes=["h3-dub-inspector-top"]):
                            inspector = gr.HTML(presenters.inspector_heading(None, None), scale=4)
                            previous_line = gr.Button("‹", size="sm", scale=0, min_width=44,
                                                      elem_classes=["h3-dub-step-line"])
                            next_line = gr.Button("›", size="sm", scale=0, min_width=44,
                                                  elem_classes=["h3-dub-step-line"])
                        with gr.Row(equal_height=True):
                            start = gr.Textbox(label="Start", min_width=90)
                            end = gr.Textbox(label="End", min_width=90)
                            speaker = gr.Dropdown([], value=None, label="Speaker", allow_custom_value=True,
                                                  min_width=120)
                        original = gr.Textbox(label="Original", lines=2)
                        translation = gr.Textbox(label="Translation", lines=3, placeholder="Not translated yet")
                        line_dub = gr.Checkbox(True, label="Dub this line",
                                               info="Off: the line keeps its original voice.")
                        fit = gr.Markdown(elem_classes=["h3-dub-fit"])
                        line_numbers = gr.Textbox(visible=False)
                        inspected = gr.State(None)
                        translation_context = gr.Checkbox(
                            True, label="Include context (Hy-MT)",
                            info="Translate again uses the previous two lines and translation notes. "
                                 "Off: translate each line on its own.")
                        with gr.Row(elem_classes=["h3-dub-line-actions"]):
                            listen_original = gr.Button("Listen to original")
                            preview = gr.Button("Preview", variant="primary")
                            new_take = gr.Button("New take")
                            retranslate = gr.Button("Translate again")
                            stop_preview = gr.Button("Interrupt", interactive=False)
                        line_status = gr.Markdown(elem_classes=["h3-dub-status"])
                        original_audio = gr.Audio(label="Original segment", type="numpy", interactive=False)
                        preview_audio = gr.Audio(label="Dub preview", type="numpy",
                                                 interactive=False)
                        gr.Markdown("Listen to original plays the source segment at its current Start and End. "
                                    "Preview hears the dub fitted to its timing; New take records it again, and "
                                    "the dub keeps the newest take. With lines picked on the board, these use "
                                    "them; Translate again with none fills the untranslated lines.",
                                    elem_classes=["h3-song-hint"])
                        with gr.Row(elem_classes=["h3-dub-line-edits"]):
                            add_line = gr.Button("Add a line after", size="sm")
                            delete_line = gr.Button("Delete line", size="sm", variant="stop")
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Source", SOURCE)
                            steps.button("Next: dub →", DUB, variant="primary")
            with steps.step(DUB, "Dub"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer", "h3-dub-settings"]):
                        with gr.Group(visible=False, elem_classes=["h3-dub-speakers-group"]) as speakers_panel:
                            card_heading("Speakers", "Cloned from the original audio with Drama 3. Give a speaker "
                                                     "a saved voice instead, or make it louder or quieter. "
                                                     "Changes save at once.")
                            speakers = widgets.speaker_cards()
                            speakers_status = gr.Markdown(elem_classes=["h3-dub-status"])
                        card_heading("Voices")
                        with gr.Row(equal_height=True):
                            voice_mode = gr.Radio(list(VOICE_MODES), value=VOICE_LINE, label="Clone from",
                                                  elem_classes=["h3-dub-segmented"], min_width=260,
                                                  info="Each line's own voice keeps its delivery; one voice per "
                                                       "speaker is steadier. Lines under 2 s use their speaker's "
                                                       "voice.")
                            preset = gr.Radio(list(VOICE_PRESETS), value="Balance", label="Delivery",
                                              elem_classes=["h3-dub-segmented"], min_width=220)
                        card_heading("Timing")
                        timing = widgets.choice_cards(TIMING_CHOICES, TIMING_SMART, label="Timing",
                                                      name="h3-dub-timing")
                        max_speed = gr.Slider(1.0, 2.0, value=1.35, step=0.05, label="Maximum speed-up",
                                              info="Takes still too long at this speed overlap the next line.")
                        card_heading("Mix and output")
                        with gr.Row():
                            speech_volume = gr.Slider(0, 200, value=100, step=5, label="Dub volume (%)")
                            background_volume = gr.Slider(0, 200, value=100, step=5,
                                                          label="Background music and effects (%)")
                        keep_original = gr.Checkbox(True, label="Keep the original audio as a second track",
                                                    info="Players can switch between the dub and the original.")
                        word_timing = gr.Checkbox(
                            False, label="Word-by-word subtitles",
                            info="Times every word of both subtitle tracks with Qwen3's aligner. To add it to a "
                                 "finished dub, turn it on and dub again: unchanged lines keep their takes.")
                        with gr.Accordion("Advanced: 16 parallel requests, no forced fresh takes", open=False,
                                          elem_classes=["h3-dub-options"]) as advanced:
                            regenerate = gr.Textbox(label="Fresh takes for lines", placeholder="e.g. 4, 9-10",
                                                    info="These lines get new takes even when unchanged.")
                            parallel = gr.Slider(1, 32, value=16, step=1, label="Parallel requests")
                    with gr.Column(scale=2, min_width=300, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Dub it")
                        readiness = gr.HTML(presenters.readiness_html(None))
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                run = gr.Button("Generate dub", variant="primary", scale=3,
                                                elem_classes=["h3-primary-action"])
                                stop = gr.Button("Interrupt", scale=1, interactive=False)
                            status = gr.Markdown(elem_classes=["h3-dub-status"])
                        # The dub with either voice, the music on or off and synced subtitles.
                        player = gr.HTML("", visible=False, js_on_load=PLAYER_JS, elem_classes=[PLAYER_CLASS])
                        with gr.Row():
                            align_words = gr.Button("Generate / repair word alignment", interactive=False)
                            stop_align_words = gr.Button("Interrupt alignment", interactive=False)
                        gr.Markdown("Align the latest dub without generating voices again. Successful word "
                                    "timings are reused; missing timings are retried.", elem_classes=["h3-song-hint"])
                        with gr.Accordion("Dubbed files", open=False):
                            video = gr.Video(label="Dubbed video", interactive=False)
                            audio = gr.Audio(label="Dubbed audio", type="filepath", interactive=False)
                            files = gr.File(label="Subtitles", file_count="multiple", interactive=False, height=96)
                        export_source = gr.State(None)
                        export_subtitles = gr.Button("Export video with burned-in translated subtitles",
                                                     interactive=False)
                        export_status = gr.Markdown()
                        exported_video = gr.File(label="Video with burned-in subtitles", interactive=False)
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Fix lines", LINES)
                        gr.Markdown("Every dub is also saved to **Media**.", elem_classes=["h3-song-hint"])
    run.h3_stop = stop
    align_words.h3_stop = stop_align_words
    analyze.h3_stop = stop_analyze
    preview.h3_stop = new_take.h3_stop = retranslate.h3_stop = stop_preview
    return DubbingView(
        steps, session, source, library, refresh_library, media_tab, sessions_tab, recent, source_card,
        source_language, target_language, notes, pipeline, translate, translator, gemini, gemini_model, api_key,
        options, separate, separator_model, reuse, local, aligner, merge_gap, refine_timestamps, plan, analyze, stop_analyze,
        analyze_status, board, lines, inspected, inspector, previous_line, next_line, start, end, speaker, original,
        translation, line_dub, fit, add_line, delete_line, line_numbers, preview, listen_original, new_take, retranslate,
        translation_context,
        stop_preview, line_status, preview_audio, original_audio, speakers_panel, speakers, speakers_status, voice_mode, preset,
        timing, max_speed, speech_volume, background_volume, keep_original, advanced, regenerate,
        parallel, readiness, run, stop, status, player, video, audio, files,
        export_source, export_subtitles, exported_video, export_status, word_timing, align_words, stop_align_words,
    )


def gemini_visibility(pipeline: str, translator: str):
    """The Gemini model and key matter when Gemini transcribes or translates."""
    return gr.update(visible=pipeline == PIPELINE_GEMINI or translator == TRANSLATOR_GEMINI)


def local_visibility(pipeline: str):
    return gr.update(visible=pipeline == PIPELINE_LOCAL)


def speed_visibility(timing: str):
    return gr.update(visible=timing in (TIMING_SMART, TIMING_NATURAL))


def advanced_label(parallel, regenerate):
    """The Advanced accordion's title: its settings at a glance."""
    fresh = str(regenerate or "").strip()
    return gr.update(label=f"Advanced: {int(parallel or 16)} parallel requests, "
                           + (f"fresh takes for {fresh}" if fresh else "no forced fresh takes"))
