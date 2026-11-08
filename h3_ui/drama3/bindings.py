"""Soundstage's actions, and its API-only endpoints for scripted clients.

The Soundstage page outside Gradio is the only place people work; these endpoints keep the
published API (``/generate_drama3``, ``/enhance_drama3_text``, the library and the batch
endpoints...) on hidden components, so none of them shows or runs from the studio itself.
"""

from dataclasses import dataclass, fields
import inspect
from typing import Callable

import gradio as gr

from h3_app.drama3.voice_design import ENGINE_QWEN
from h3_app.jobs import JobCancelled

from ..job_bindings import GPU_QUEUE, bind_gpu_action, bind_prompt_action, owned_generation
from .controls import DRAMA3_VOICE_SLOTS


@dataclass(frozen=True)
class Drama3Actions:
    """Every Soundstage handler, by name: the page and the API endpoints call these."""

    generate_drama3: Callable
    enhance_drama3_text: Callable
    transcribe_drama3_voice: Callable
    interrupt: Callable
    generate_drama3_voice: Callable
    generate_drama3_voices: Callable
    copy_drama3_writer_prompt: Callable
    apply_drama3_writer_reply: Callable
    generate_drama3_voice_takes: Callable | None = None
    generate_drama3_sound: Callable | None = None
    generate_drama3_sounds: Callable | None = None
    sync_drama3_script_sounds: Callable | None = None
    regenerate_drama3_line: Callable | None = None
    remix_drama3: Callable | None = None
    repair_drama3_words: Callable | None = None
    prepare_drama3_cast: Callable | None = None
    preload_drama3_models: Callable | None = None
    drama3_library_groups: Callable | None = None
    drama3_library_speakers: Callable | None = None
    drama3_library_save_speaker: Callable | None = None
    drama3_library_speaker_clip: Callable | None = None
    drama3_library_sounds: Callable | None = None
    drama3_library_save_sound: Callable | None = None
    drama3_library_sound_clip: Callable | None = None
    drama3_result_details: Callable | None = None
    drama3_result_audio: Callable | None = None
    generate_drama3_voices_batch: Callable | None = None
    generate_drama3_sounds_batch: Callable | None = None
    generate_drama3_batch: Callable | None = None
    load_drama3_example: Callable | None = None
    load_drama3_example_zh: Callable | None = None
    load_drama3_example_zh_latest: Callable | None = None

    @classmethod
    def of(cls, handlers, *, interrupt: Callable):
        """Every action ``handlers`` defines under its field name."""
        names = {field.name for field in fields(cls)} - {"interrupt"}
        return cls(interrupt=interrupt, **{name: getattr(handlers, name) for name in names if hasattr(handlers, name)})



def _bind_library_api(actions):
    """API-only endpoints for scripted clients: saved speakers, saved sounds, result details and audio.

    Each endpoint has its own hidden trigger, so none of them runs from the page.
    """
    if actions.drama3_library_speakers is None:
        return
    with gr.Group(visible=False):
        group = gr.Textbox(label="Speaker group", value="Global")
        name = gr.Textbox(label="Name")
        clip = gr.File(label="Clip", type="filepath")
        transcript = gr.Textbox(label="Exact transcript")
        description = gr.Textbox(label="Description")
        duration = gr.Number(label="Duration", value=5, precision=0)
        result = gr.Textbox(label="Result file name")
        listing = gr.JSON(label="Listing")
        entry = gr.JSON(label="Saved entry")
        details = gr.JSON(label="Result details")
        download = gr.File(label="Saved clip")
    quiet = {"queue": False, "show_progress": "hidden"}
    for fn, inputs, outputs, api_name in (
        (actions.drama3_library_groups, [], listing, "drama3_library_groups"),
        (actions.drama3_library_speakers, [group], listing, "drama3_library_speakers"),
        (actions.drama3_library_save_speaker, [group, name, clip, transcript, description], entry,
         "drama3_library_save_speaker"),
        (actions.drama3_library_speaker_clip, [group, name], [download, entry], "drama3_library_speaker_clip"),
        (actions.drama3_library_sounds, [], listing, "drama3_library_sounds"),
        (actions.drama3_library_save_sound, [name, clip, description, duration], entry,
         "drama3_library_save_sound"),
        (actions.drama3_library_sound_clip, [name], [download, entry], "drama3_library_sound_clip"),
        (actions.drama3_result_details, [result], details, "drama3_result_details"),
        (actions.drama3_result_audio, [result], download, "drama3_result_audio"),
    ):
        if fn is not None:
            gr.Button(visible=False).click(fn, inputs=inputs, outputs=outputs, api_name=api_name, **quiet)



def _preload_callback(actions, input_names):
    """The model preload as a GPU-queue job; an interrupt ends it quietly, since a preload is optional."""
    run = owned_generation(actions.preload_drama3_models, "drama3_preload", input_names)

    def preload(*args):
        try:
            yield from run(*args)
        except JobCancelled:
            return

    preload.__signature__, preload.__annotations__ = run.__signature__, run.__annotations__
    return preload



def _bind_batch_api(actions):
    """API-only GPU endpoints for scripted clients: batch voices, sounds and renders, and model preload.

    Each is one GPU-queue job, so a script pays one model load and one queue wait for a whole
    cast, soundscape or book. Hidden triggers keep them off the page.
    """
    if actions.generate_drama3_voices_batch is None:
        return
    with gr.Group(visible=False):
        briefs = gr.Textbox(label="Voice briefs (JSON list)")
        language = gr.Textbox(label="Default preview language", value="Auto")
        sound_items = gr.Textbox(label="Sounds (JSON list)")
        engine = gr.Textbox(label="Voice engine", value=ENGINE_QWEN)
        mode = gr.Textbox(label="Script mode", value="Advanced (sound enhanced)")
        clips = gr.File(label="Clips", file_count="multiple", type="filepath")
        listing = gr.JSON(label="Listing")
        preload_status = gr.Textbox(label="Preload status")
        jobs = gr.Textbox(label="Render jobs (JSON list)")
        fix_rounds = gr.Number(label="Failed-take rounds", value=1, precision=0)
        render_status = gr.Textbox(label="Batch status")
    for fn, family, inputs, outputs, api_name in (
        (actions.generate_drama3_voices_batch, "drama3_voice", [briefs, language], [clips, listing],
         "generate_drama3_voices_batch"),
        (actions.generate_drama3_sounds_batch, "drama3_sound", [sound_items], [clips, listing],
         "generate_drama3_sounds_batch"),
        (actions.generate_drama3_batch, "drama3", [jobs, fix_rounds], [listing, render_status],
         "generate_drama3_batch"),
    ):
        if fn is not None:
            bind_gpu_action(gr.Button(visible=False).click, owned_generation(fn, family), inputs=inputs,
                            outputs=outputs, show_progress="hidden", api_name=api_name)
    if actions.preload_drama3_models is not None:
        gr.Button(visible=False).click(_preload_callback(actions, ("voice_engine", "script_mode")),
                                       inputs=[engine, mode], outputs=preload_status, show_progress="hidden",
                                       api_name="preload_drama3_models", **GPU_QUEUE)


# How the API parameters of the speech, writer, voice and sound endpoints are typed.
_NUMBERS = {"temperature", "top_p", "top_k", "max_frames", "parallel", "retries", "max_batch_lines", "shown_slots",
            "duration"}
_FLAGS = {"word_timing", "strict_source"}


def _api_input(name: str, default):
    """A hidden component for one named API parameter, typed as the old controls were."""
    value = None if default is inspect.Parameter.empty else default
    if name == "audio" or name.endswith("_audio"):
        return gr.Audio(label=name, type="filepath", value=value)
    if name in _FLAGS:
        return gr.Checkbox(label=name, value=bool(value))
    if name in _NUMBERS or name.endswith("_duration"):
        return gr.Number(label=name, value=value)
    return gr.Textbox(label=name, value=value)


def _api_inputs(callback, skip=()):
    """Hidden inputs for ``callback``'s published parameters, in order."""
    parameters = [parameter for parameter in inspect.signature(callback).parameters.values()
                  if parameter.name not in {"self", "progress", "request", *skip}
                  and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
                                             inspect.Parameter.KEYWORD_ONLY)]
    return [_api_input(parameter.name, parameter.default) for parameter in parameters]


def _bind_generation_api(actions):
    """The speech, writer, voice, transcript and sound endpoints, with their published parameter names.

    Speech takes no project name here: a scripted result belongs to no saved project.
    """
    from h3_ui.engines.controls import owned_view_generation

    writer_outputs = 2 + 3 * DRAMA3_VOICE_SLOTS + (DRAMA3_VOICE_SLOTS + 3) + DRAMA3_VOICE_SLOTS + 1
    with gr.Group(visible=False):
        speech = _api_inputs(actions.generate_drama3, skip=("project_name",))
        line = gr.Textbox(label="line")
        speech_out = [gr.Audio(label="Generated speech", type="filepath", interactive=False), gr.Textbox(label="Generation progress")]
        writer = _api_inputs(actions.enhance_drama3_text)
        writer_out = [gr.Textbox(label=f"output_{index}") for index in range(writer_outputs)]
        voice = _api_inputs(actions.generate_drama3_voice)
        voice_out = [gr.Audio(label="Reference clip", type="filepath", interactive=False), gr.Textbox(label="Exact transcript"),
                     gr.Textbox(label="Voice status")]
        transcribe = _api_inputs(actions.transcribe_drama3_voice)
        transcribe_out = [gr.Textbox(label="Exact transcript"), gr.Textbox(label="Transcript status")]
        sound = _api_inputs(actions.generate_drama3_sound) if actions.generate_drama3_sound else []
        sound_out = [gr.Audio(label="Sound clip", type="filepath", interactive=False), gr.Textbox(label="Sound status")]
    for callback, inputs, api_name in (
        (actions.generate_drama3, speech, "generate_drama3"),
        (actions.remix_drama3, speech, "remix_drama3"),
        (actions.regenerate_drama3_line, [line, *speech], "regenerate_drama3_line"),
    ):
        if callback is not None:
            bind_gpu_action(gr.Button(visible=False).click, owned_view_generation(callback, "drama3", inputs),
                            inputs=inputs, outputs=speech_out, api_name=api_name, show_progress="hidden")
    bind_prompt_action(gr.Button(visible=False).click, actions.enhance_drama3_text, inputs=writer,
                       outputs=writer_out, api_name="enhance_drama3_text", show_progress="hidden")
    bind_gpu_action(gr.Button(visible=False).click, owned_generation(actions.generate_drama3_voice, "drama3_voice"),
                    inputs=voice, outputs=voice_out, api_name="generate_drama3_voice", show_progress="hidden")
    bind_gpu_action(gr.Button(visible=False).click, actions.transcribe_drama3_voice, inputs=transcribe,
                    outputs=transcribe_out, api_name="transcribe_drama3_voice", show_progress="hidden")
    if sound:
        bind_gpu_action(gr.Button(visible=False).click, owned_generation(actions.generate_drama3_sound, "drama3_sound"),
                        inputs=sound, outputs=sound_out, api_name="generate_drama3_sound", show_progress="hidden")


def bind_soundstage_api(actions, *, gpu=True):
    """Every API-only Soundstage endpoint; without a GPU, none (the Workshop only prepares)."""
    if not gpu:
        return
    _bind_generation_api(actions)
    _bind_library_api(actions)
    _bind_batch_api(actions)
