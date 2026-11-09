"""The Dubbing tab: the page in a frame, and the API-only endpoints beside it.

Everything people do happens on the page at ``/dubbing/`` (``page.py``), outside Gradio, as with
Soundstage; the tab only shows it. The hidden endpoints keep Dubbing's published API
(``/dub_analyze_source`` and ``/dub_generate``) for scripted clients.
"""

from __future__ import annotations

import gradio as gr

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from h3_app.dubbing import pipeline
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB

from ..job_bindings import bind_gpu_action, owned_generation

FRAME = ('<iframe class="h3-dubbing-frame" src="/dubbing/" title="Dubbing" '
         'allow="fullscreen; clipboard-write"></iframe>')


def build_dubbing_tab(root, actions, *, gpu=True):
    """Fill ``root`` with the page; with a GPU, also publish the API endpoints."""
    with root:
        gr.HTML(FRAME)
        bind_dubbing_api(actions, gpu=gpu)


def bind_dubbing_api(actions, *, gpu=True):
    """The analysis and dub endpoints, with the parameter names the old tab published.

    Without a GPU there are none: the Workshop prepares projects on its page only.
    """
    if not gpu:
        return
    with gr.Group(visible=False):
        analyze_inputs = [
            gr.File(label="Video or audio", type="filepath"),
            gr.Dropdown([], value=None, label="Video or audio in Media", allow_custom_value=True),
            gr.Checkbox(True, label="Separate vocals from the background"),
            gr.Dropdown(list(pipeline.SEPARATOR_MODELS), value="Balanced (BS-Roformer)", label="Separation model"),
            gr.Textbox(pipeline.PIPELINE_LOCAL, label="Transcription"),
            gr.Dropdown(list(pipeline.SOURCE_LANGUAGES), value="Auto-detect", label="Spoken language"),
            gr.Dropdown([pipeline.SAME_LANGUAGE, *pipeline.TARGET_LANGUAGES], value="English", label="Dub into",
                        allow_custom_value=True),
            gr.Checkbox(True, label="Translate now"),
            gr.Dropdown(list(pipeline.TRANSLATORS), value=pipeline.TRANSLATOR_HYMT, label="Translator"),
            gr.Checkbox(True, label="Word timing with Qwen3-ForcedAligner"),
            gr.Slider(0.0, 1.5, value=0.35, step=0.05, label="Join pauses shorter than (s)"),
            gr.Dropdown(list(GEMINI_PROMPT_MODELS), value=DEFAULT_GEMINI_PROMPT_MODEL, label="Gemini model"),
            gr.Textbox("", label="Gemini API key for this session", type="password"),
            gr.Textbox("", label="Translation notes (optional)"),
            gr.Checkbox(True, label="Reuse earlier results for this file"),
            gr.Checkbox(True, label="Refine speech boundaries with Align (experimental)"),
            gr.Checkbox(False, label="Include previous lines (Hy-MT)"),
            gr.State(None),
            gr.Textbox("", label="Reply"),
            gr.Dropdown(list(pipeline.TRANSLATION_STYLES), value=pipeline.DEFAULT_STYLE, label="Translation style",
                        allow_custom_value=True),
            gr.Checkbox(False, label="Rewrite in the same language"),
        ]
        analyze_outputs = [gr.Textbox(label="Session"), gr.JSON(label="Board"), gr.Dataframe(label="Lines"),
                           gr.Markdown(), gr.HTML()]
        dub_inputs = [
            gr.Textbox(label="Session"),
            gr.Radio(list(pipeline.VOICE_MODES), value=pipeline.VOICE_LINE, label="Clone from"),
            gr.Radio(list(pipeline.VOICE_PRESETS), value="Balance", label="Delivery"),
            gr.Textbox(pipeline.TIMING_SMART, label="Timing"),
            gr.Slider(1.0, 2.0, value=1.35, step=0.05, label="Maximum speed-up"),
            gr.Slider(1, 32, value=16, step=1, label="Parallel requests"),
            gr.Slider(0, 200, value=100, step=5, label="Dub volume (%)"),
            gr.Slider(0, 200, value=100, step=5, label="Background music and effects (%)"),
            gr.Checkbox(True, label="Keep the original audio as a second track"),
            gr.Textbox("", label="Fresh takes for lines"),
            gr.Checkbox(False, label="Word-by-word subtitles"),
        ]
        dub_outputs = [gr.HTML(), gr.Video(label="Dubbed video"), gr.Audio(label="Dubbed audio", type="filepath"),
                       gr.File(label="Subtitles", file_count="multiple"), gr.Markdown(), gr.State(None),
                       gr.Button(visible=False), gr.File(label="Video with burned-in subtitles"), gr.Markdown()]
    bind_gpu_action(gr.Button(visible=False).click, owned_generation(actions.analyze_source, FAMILY_ANALYZE),
                    inputs=analyze_inputs, outputs=analyze_outputs, api_name="dub_analyze_source",
                    show_progress="hidden")
    bind_gpu_action(gr.Button(visible=False).click, owned_generation(actions.generate_dub, FAMILY_DUB),
                    inputs=dub_inputs, outputs=dub_outputs, api_name="dub_generate", show_progress="hidden")
