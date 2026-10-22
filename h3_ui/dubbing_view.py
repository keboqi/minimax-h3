"""Dub & Translate: source, transcription, line review and dubbing in four steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from h3_app.dubbing.pipeline import (
    PIPELINE_GEMINI, PIPELINE_LOCAL, PIPELINES, SEPARATOR_MODELS, SOURCE_LANGUAGES, TARGET_LANGUAGES,
    TRANSLATOR_GEMINI, TRANSLATOR_HYMT, TRANSLATORS, VOICE_LINE, VOICE_MODES, VOICE_PRESETS,
)
from h3_app.dubbing.segments import TABLE_HEADERS, TABLE_TYPES


@dataclass(frozen=True)
class DubbingView:
    session: gr.Textbox
    source: gr.File
    library: gr.Dropdown
    refresh_library: gr.Button
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
    analyze: gr.Button
    stop_analyze: gr.Button
    analyze_status: gr.Markdown
    lines: gr.Dataframe
    table_status: gr.Markdown
    line_numbers: gr.Textbox
    retranslate: gr.Button
    preview: gr.Button
    stop_preview: gr.Button
    preview_audio: gr.Audio
    voice_mode: gr.Radio
    preset: gr.Dropdown
    max_speed: gr.Slider
    parallel: gr.Slider
    speech_volume: gr.Slider
    background_volume: gr.Slider
    keep_original: gr.Checkbox
    burn_subtitles: gr.Checkbox
    regenerate: gr.Textbox
    run: gr.Button
    stop: gr.Button
    status: gr.Markdown
    video: gr.Video
    audio: gr.Audio
    files: gr.File


def _step(number: int, title: str, text: str) -> None:
    gr.Markdown(f"### {number} · {title}\n{text}", elem_classes=["h3-dub-step"])


def build_dubbing_view(root: gr.Column, *, gemini_models: Sequence[str], default_gemini_model: str) -> DubbingView:
    with root:
        gr.Markdown(
            "## Dub & Translate\nTranslate the speech in a video or recording and dub it with each speaker's "
            "own cloned voice. Analyze the source, review and edit every line, then dub. Unchanged lines "
            "reuse their saved takes, so you can fix a few lines and dub again quickly.",
            elem_classes=["h3-view-heading"],
        )
        session = gr.Textbox(visible=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                _step(1, "Source", "Upload a video or audio file, or pick one from Media "
                                   "(downloads from **Media → Download from a URL** appear there).")
                source = gr.File(label="Video or audio", file_count="single", file_types=["video", "audio"],
                                 type="filepath", height=110)
                with gr.Row(equal_height=True):
                    library = gr.Dropdown([], value=None, label="Or pick from Media", scale=4,
                                          info="Used when nothing is uploaded above.")
                    refresh_library = gr.Button("Refresh", size="sm", scale=0, min_width=90)
                with gr.Row(equal_height=True):
                    separate = gr.Checkbox(True, label="Separate vocals from the background",
                                           info="Keeps the music and effects under the dub and gives clean "
                                                "voice references. Off: the original is ducked under the dub.")
                    separator_model = gr.Dropdown(list(SEPARATOR_MODELS), value="Balanced (BS-Roformer)",
                                                  label="Separation model")
            with gr.Column(scale=1, min_width=320):
                _step(2, "Transcribe and translate", "Local runs OmniVAD, Qwen3-ASR with ForcedAligner timing "
                                                     "and Hy-MT on this server; Gemini also labels speakers.")
                pipeline = gr.Radio(list(PIPELINES), value=PIPELINE_LOCAL, label="Transcription")
                with gr.Row():
                    source_language = gr.Dropdown(list(SOURCE_LANGUAGES), value="Auto-detect",
                                                  label="Spoken language")
                    target_language = gr.Dropdown(list(TARGET_LANGUAGES), value="English", label="Dub into",
                                                  allow_custom_value=True)
                with gr.Row(equal_height=True):
                    translate = gr.Checkbox(True, label="Translate while transcribing",
                                            info="Off: lines keep only the original text; translate later.")
                    translator = gr.Dropdown(list(TRANSLATORS), value=TRANSLATOR_HYMT, label="Translator")
                with gr.Group(visible=False) as gemini:
                    with gr.Row():
                        gemini_model = gr.Dropdown(list(gemini_models), value=default_gemini_model,
                                                   label="Gemini transcription model")
                        api_key = gr.Textbox(label="Gemini API key for this session", type="password",
                                             placeholder="Uses GEMINI_API_KEY when empty")
                with gr.Accordion("Local transcription options", open=False) as local:
                    aligner = gr.Checkbox(True, label="Word timing with Qwen3-ForcedAligner",
                                          info="Splits long speech into subtitle-sized lines at sentence ends.")
                    merge_gap = gr.Slider(0.0, 1.5, value=0.35, step=0.05, label="Join pauses shorter than (s)",
                                          info="Speech separated by a shorter pause is transcribed together.")
                notes = gr.Textbox(label="Translation notes", lines=2,
                                   placeholder="Names, terms or tone to keep, e.g. 'Keep the name Mei; casual tone.'")
                with gr.Row():
                    analyze = gr.Button("Analyze source", variant="primary", scale=3)
                    stop_analyze = gr.Button("Interrupt", scale=1, interactive=False)
                analyze_status = gr.Markdown()
        _step(3, "Review lines", "Edit times, speakers, the original text or the translation; changes save as "
                                 "you type. Untick **Dub** to keep a line's original voice.")
        lines = gr.Dataframe(headers=list(TABLE_HEADERS), datatype=list(TABLE_TYPES), value=[], type="array",
                             interactive=True, wrap=True, max_height=560, label="Lines",
                             column_widths=["5%", "9%", "9%", "9%", "30%", "32%", "6%"])
        table_status = gr.Markdown()
        with gr.Row(equal_height=True):
            line_numbers = gr.Textbox(label="Line numbers", placeholder="e.g. 3, 5-7", scale=2,
                                      info="For Translate again and Preview. Empty translates only untranslated lines.")
            retranslate = gr.Button("Translate again", scale=1)
            preview = gr.Button("Preview line", scale=1)
            stop_preview = gr.Button("Interrupt", scale=0, min_width=110, interactive=False)
        preview_audio = gr.Audio(label="Line preview (fitted to its timing)", type="numpy", interactive=False)
        _step(4, "Dub", "Voices are cloned from the original audio with Drama 3. Lines too long for their slot "
                        "are sped up, up to the limit below.")
        with gr.Row():
            voice_mode = gr.Radio(list(VOICE_MODES), value=VOICE_LINE, label="Voice",
                                  info="Each line keeps its own delivery; one voice per speaker is steadier. "
                                       "Lines shorter than 2 s always use their speaker's voice.")
            preset = gr.Dropdown(list(VOICE_PRESETS), value="Balance", label="Speech preset")
        with gr.Row():
            max_speed = gr.Slider(1.0, 2.0, value=1.35, step=0.05, label="Maximum speed-up")
            parallel = gr.Slider(1, 32, value=8, step=1, label="Parallel requests")
        with gr.Row():
            speech_volume = gr.Slider(0, 200, value=100, step=5, label="Dub volume (%)")
            background_volume = gr.Slider(0, 200, value=100, step=5, label="Background volume (%)")
        with gr.Row():
            keep_original = gr.Checkbox(True, label="Keep the original audio as a second track")
            burn_subtitles = gr.Checkbox(False, label="Burn translated subtitles into the video",
                                         info="Re-encodes the video. Selectable subtitle tracks are always added.")
        regenerate = gr.Textbox(label="Fresh takes for lines", placeholder="e.g. 4, 9-10",
                                info="These lines get new takes even when unchanged.")
        with gr.Row():
            run = gr.Button("Generate dub", variant="primary", scale=3, elem_classes=["h3-primary-action"])
            stop = gr.Button("Interrupt", scale=1, interactive=False)
        status = gr.Markdown()
        with gr.Row(equal_height=False):
            video = gr.Video(label="Dubbed video", interactive=False)
            with gr.Column():
                audio = gr.Audio(label="Dubbed audio", type="filepath", interactive=False)
                files = gr.File(label="Subtitles", file_count="multiple", interactive=False)
    run.h3_stop = stop
    analyze.h3_stop = stop_analyze
    preview.h3_stop = stop_preview
    return DubbingView(
        session, source, library, refresh_library, separate, separator_model, pipeline, source_language,
        target_language, translate, translator, gemini, gemini_model, api_key, local, aligner, merge_gap, notes,
        analyze, stop_analyze, analyze_status, lines, table_status, line_numbers, retranslate, preview,
        stop_preview, preview_audio, voice_mode, preset, max_speed, parallel, speech_volume, background_volume,
        keep_original, burn_subtitles, regenerate, run, stop, status, video, audio, files,
    )


def gemini_visibility(pipeline: str, translator: str):
    """The Gemini model and key matter when Gemini transcribes or translates."""
    return gr.update(visible=pipeline == PIPELINE_GEMINI or translator == TRANSLATOR_GEMINI)


def local_visibility(pipeline: str):
    return gr.update(visible=pipeline == PIPELINE_LOCAL)
