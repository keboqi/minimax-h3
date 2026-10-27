"""Dub & Translate as three steps: Source, Review lines and Dub.

The steps share Song's stepper: each pairs a working card with an action card
that holds the step's next move. A finished analysis, or a reopened session,
opens Review lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from h3_app.dubbing.pipeline import (
    PIPELINE_GEMINI, PIPELINE_LOCAL, PIPELINES, SEPARATOR_MODELS, SOURCE_LANGUAGES, TARGET_LANGUAGES,
    TIMING_MATCH, TIMING_NATURAL, TIMINGS, TRANSLATOR_GEMINI, TRANSLATOR_HYMT, TRANSLATORS, VOICE_LINE,
    VOICE_MODES, VOICE_PRESETS,
)
from h3_app.dubbing.segments import TABLE_HEADERS, TABLE_TYPES

from ..step_cards import StepStrip, card_heading, intro, step_strip
from .player import ELEM_CLASS as PLAYER_CLASS, PLAYER_JS

SOURCE, LINES, DUB = range(3)
STEPS_ID = "h3-dub-steps"


@dataclass(frozen=True)
class DubbingView:
    steps: StepStrip
    session: gr.Textbox
    source: gr.File
    library: gr.Dropdown
    refresh_library: gr.Button
    sessions: gr.Dropdown
    reopen: gr.Accordion
    separate: gr.Checkbox
    separator_model: gr.Dropdown
    pipeline: gr.Radio
    source_language: gr.Dropdown
    target_language: gr.Dropdown
    translate: gr.Checkbox
    translator: gr.Dropdown
    gemini: gr.Group
    gemini_model: gr.Dropdown
    api_key: gr.Textbox
    local: gr.Accordion
    aligner: gr.Checkbox
    merge_gap: gr.Slider
    notes: gr.Textbox
    reuse: gr.Checkbox
    analyze: gr.Button
    stop_analyze: gr.Button
    analyze_status: gr.Markdown
    summary: gr.Markdown
    lines: gr.Dataframe
    line_numbers: gr.Textbox
    retranslate: gr.Button
    preview: gr.Button
    new_take: gr.Button
    stop_preview: gr.Button
    line_status: gr.Markdown
    preview_audio: gr.Audio
    voice_mode: gr.Radio
    preset: gr.Radio
    speakers_panel: gr.Accordion
    speaker: gr.Dropdown
    speaker_voice: gr.Dropdown
    speaker_volume: gr.Slider
    speakers_overview: gr.Markdown
    timing: gr.Radio
    max_speed: gr.Slider
    parallel: gr.Slider
    speech_volume: gr.Slider
    background_volume: gr.Slider
    keep_original: gr.Checkbox
    burn_subtitles: gr.Checkbox
    regenerate: gr.Textbox
    dub_summary: gr.Markdown
    run: gr.Button
    stop: gr.Button
    status: gr.Markdown
    player: gr.HTML
    video: gr.Video
    audio: gr.Audio
    files: gr.File


def opened_session_js(steps: StepStrip) -> str:
    """Browser code that opens Review lines when a different session arrives.

    Analysis reports errors and interruptions as status lines, so its event always
    ends cleanly; only a new session id means there are new lines to review.
    """
    return (
        "(id) => { "
        f"const root = document.getElementById({steps.tabs.elem_id!r}); "
        "if (!id || !root || root.dataset.session === id) return; "
        "root.dataset.session = id; "
        f"({steps.go(LINES)})(); }}"
    )


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
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("What should be dubbed?",
                                     "Upload a video or recording, or pick one from Media. Downloads from "
                                     "Media → Download from a URL appear there.")
                        source = gr.File(label="Video or audio", file_count="single", file_types=["video", "audio"],
                                         type="filepath", height=120)
                        with gr.Row(equal_height=True, elem_classes=["h3-dub-library"]):
                            library = gr.Dropdown([], value=None, label="Or pick from Media", scale=4)
                            refresh_library = gr.Button("Refresh", size="sm", scale=0, min_width=90)
                        with gr.Row():
                            source_language = gr.Dropdown(list(SOURCE_LANGUAGES), value="Auto-detect",
                                                          label="Spoken language")
                            target_language = gr.Dropdown(list(TARGET_LANGUAGES), value="English",
                                                          label="Dub into", allow_custom_value=True)
                        notes = gr.Textbox(label="Translation notes", lines=2,
                                           placeholder="Names, terms or tone to keep, e.g. 'Keep the name Mei; "
                                                       "casual tone.'")
                        separate = gr.Checkbox(True, label="Separate vocals from the background",
                                               info="Keeps music and effects under the dub and gives clean voice "
                                                    "references. Off: the original is ducked under the dub.")
                        separator_model = gr.Dropdown(list(SEPARATOR_MODELS), value="Balanced (BS-Roformer)",
                                                      label="Separation model")
                        with gr.Accordion("Continue an earlier session", open=False) as reopen:
                            gr.Markdown("Analyzed sessions are kept for 14 days. Pick one to review and dub "
                                        "its lines again.", elem_classes=["h3-song-hint"])
                            sessions = gr.Dropdown([], value=None, label="Earlier sessions", show_label=False)
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Find and translate the lines",
                                     "Finds who speaks when, transcribes every line and translates it. "
                                     "You review each line next.")
                        pipeline = gr.Radio(list(PIPELINES), value=PIPELINE_LOCAL, label="Transcription",
                                            info="Local runs on this server; MOSS needs Linux. MOSS and Gemini "
                                                 "also tell speakers apart, so each keeps their own voice.")
                        with gr.Group(visible=False) as gemini:
                            gemini_model = gr.Dropdown(list(gemini_models), value=default_gemini_model,
                                                       label="Gemini model")
                            api_key = gr.Textbox(label="Gemini API key for this session", type="password",
                                                 placeholder="Uses GEMINI_API_KEY when empty")
                        with gr.Row(equal_height=True):
                            translate = gr.Checkbox(True, label="Translate now",
                                                    info="Off: transcribe only; translate on the next step.")
                            translator = gr.Dropdown(list(TRANSLATORS), value=TRANSLATOR_HYMT, label="Translator")
                        reuse = gr.Checkbox(True, label="Reuse earlier results for this file",
                                            info="An earlier analysis of the same file with the same settings "
                                                 "lends its separated vocals and transcript. Off: redo both.")
                        with gr.Accordion("Local transcription options", open=False) as local:
                            aligner = gr.Checkbox(True, label="Word timing with Qwen3-ForcedAligner",
                                                  info="Splits long speech into subtitle-sized lines at "
                                                       "sentence ends.")
                            merge_gap = gr.Slider(0.0, 1.5, value=0.35, step=0.05,
                                                  label="Join pauses shorter than (s)",
                                                  info="Speech separated by a shorter pause is transcribed "
                                                       "together.")
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                analyze = gr.Button("Analyze source", variant="primary", scale=3,
                                                    elem_classes=["h3-primary-action"])
                                stop_analyze = gr.Button("Interrupt", scale=1, interactive=False)
                            analyze_status = gr.Markdown(elem_classes=["h3-dub-status"])
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("Review lines →", LINES)
            with steps.step(LINES, "Review lines"):
                card_heading("Check every line",
                             "Click a line to pick it for the actions below. Edit times, speakers, the original "
                             "or the translation; changes save as you type. Untick Dub to keep a line's "
                             "original voice.")
                summary = gr.Markdown("Analyze a source to see its lines here.", elem_classes=["h3-dub-summary"])
                lines = gr.Dataframe(headers=list(TABLE_HEADERS), datatype=list(TABLE_TYPES), value=[], type="array",
                                     interactive=True, wrap=True, max_height=560, label="Lines", show_label=False,
                                     static_columns=[0], column_widths=["5%", "9%", "9%", "9%", "30%", "32%", "6%"],
                                     elem_classes=["h3-dub-lines"])
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("Work on lines",
                                     "Preview hears a line fitted to its timing; New take records it again, and "
                                     "the dub keeps the newest take.")
                        line_numbers = gr.Textbox(label="Lines", placeholder="Click a line, or type 3, 5-7",
                                                  info="Translate again with no lines picked fills the "
                                                       "untranslated ones, or every line after you change "
                                                       "Dub into.")
                        with gr.Row(elem_classes=["h3-dub-line-actions"]):
                            preview = gr.Button("Preview", variant="primary")
                            new_take = gr.Button("New take")
                            retranslate = gr.Button("Translate again")
                            stop_preview = gr.Button("Interrupt", interactive=False)
                        line_status = gr.Markdown(elem_classes=["h3-dub-status"])
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Line preview", "The picked line as it will play in the dub.")
                        preview_audio = gr.Audio(label="Line preview", show_label=False, type="numpy",
                                                 interactive=False)
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Source", SOURCE)
                            steps.button("Next: dub →", DUB, variant="primary")
            with steps.step(DUB, "Dub"):
                with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
                    with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                        card_heading("Voices and timing", "Voices are cloned from the original audio with "
                                                          "Drama 3, and each take is fitted to its line.")
                        voice_mode = gr.Radio(list(VOICE_MODES), value=VOICE_LINE, label="Voice",
                                              info="Each line keeps its own delivery; one voice per speaker is "
                                                   "steadier. Lines under 2 s always use their speaker's voice.")
                        preset = gr.Radio(list(VOICE_PRESETS), value="Balance", label="Delivery")
                        with gr.Accordion("Speakers", open=False, visible=False) as speakers_panel:
                            gr.Markdown("MOSS or Gemini told these speakers apart. Give one a saved Drama 3 voice "
                                        "instead of its cloned original, or make it louder or quieter in the "
                                        "dub. Changes save at once.", elem_classes=["h3-song-hint"])
                            speaker = gr.Dropdown([], value=None, label="Speaker")
                            with gr.Row():
                                speaker_voice = gr.Dropdown([], value=None, label="Voice", scale=3)
                                speaker_volume = gr.Slider(0, 200, value=100, step=5, label="Volume (%)", scale=2)
                            speakers_overview = gr.Markdown(elem_classes=["h3-dub-summary"])
                        timing = gr.Radio(list(TIMINGS), value=TIMING_MATCH, label="Timing",
                                          info="Match starts and ends every take exactly where the original line "
                                               "does. Natural keeps each take's pace and only speeds up long ones.")
                        max_speed = gr.Slider(1.0, 2.0, value=1.35, step=0.05, label="Maximum speed-up",
                                              visible=False,
                                              info="Takes still too long overlap the next line.")
                        card_heading("Mix and output")
                        with gr.Row():
                            speech_volume = gr.Slider(0, 200, value=100, step=5, label="Dub volume (%)")
                            background_volume = gr.Slider(0, 200, value=100, step=5, label="Background volume (%)")
                        keep_original = gr.Checkbox(True, label="Keep the original audio as a second track")
                        burn_subtitles = gr.Checkbox(False, label="Burn translated subtitles into the video",
                                                     info="Re-encodes the video. Selectable subtitle tracks are "
                                                          "always added.")
                        with gr.Accordion("Advanced", open=False):
                            regenerate = gr.Textbox(label="Fresh takes for lines", placeholder="e.g. 4, 9-10",
                                                    info="These lines get new takes even when unchanged.")
                            parallel = gr.Slider(1, 32, value=16, step=1, label="Parallel requests")
                    with gr.Column(scale=2, min_width=280, elem_classes=["h3-preview-panel", "h3-song-stage"]):
                        card_heading("Dub it", "Unchanged lines reuse their saved takes, so dubbing again after "
                                               "a few fixes is quick.")
                        dub_summary = gr.Markdown("Analyze a source first.", elem_classes=["h3-dub-summary"])
                        with gr.Column(elem_classes=["h3-action-dock"]):
                            with gr.Row(elem_classes=["h3-generation-buttons"]):
                                run = gr.Button("Generate dub", variant="primary", scale=3,
                                                elem_classes=["h3-primary-action"])
                                stop = gr.Button("Interrupt", scale=1, interactive=False)
                            status = gr.Markdown(elem_classes=["h3-dub-status"])
                        # The dub with either voice, the music on or off and synced subtitles.
                        player = gr.HTML("", visible=False, js_on_load=PLAYER_JS, elem_classes=[PLAYER_CLASS])
                        with gr.Accordion("Dubbed files", open=False):
                            video = gr.Video(label="Dubbed video", interactive=False)
                            audio = gr.Audio(label="Dubbed audio", type="filepath", interactive=False)
                            files = gr.File(label="Subtitles", file_count="multiple", interactive=False, height=96)
                        with gr.Row(elem_classes=["h3-song-step-nav"]):
                            steps.button("← Fix lines", LINES)
                        gr.Markdown("Every dub is also saved to **Media**.", elem_classes=["h3-song-hint"])
    run.h3_stop = stop
    analyze.h3_stop = stop_analyze
    preview.h3_stop = new_take.h3_stop = retranslate.h3_stop = stop_preview
    return DubbingView(
        steps, session, source, library, refresh_library, sessions, reopen, separate, separator_model, pipeline,
        source_language, target_language, translate, translator, gemini, gemini_model, api_key, local, aligner,
        merge_gap, notes, reuse, analyze, stop_analyze, analyze_status, summary, lines, line_numbers, retranslate,
        preview, new_take, stop_preview, line_status, preview_audio, voice_mode, preset, speakers_panel, speaker,
        speaker_voice, speaker_volume, speakers_overview, timing, max_speed, parallel,
        speech_volume, background_volume, keep_original, burn_subtitles, regenerate, dub_summary, run, stop, status,
        player, video, audio, files,
    )


def gemini_visibility(pipeline: str, translator: str):
    """The Gemini model and key matter when Gemini transcribes or translates."""
    return gr.update(visible=pipeline == PIPELINE_GEMINI or translator == TRANSLATOR_GEMINI)


def local_visibility(pipeline: str):
    return gr.update(visible=pipeline == PIPELINE_LOCAL)


def speed_visibility(timing: str):
    return gr.update(visible=timing == TIMING_NATURAL)
