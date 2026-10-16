"""FLUX 3 image and Drama 3 speech controls and owned job bindings."""

from dataclasses import dataclass
from typing import Callable

import gradio as gr
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL,
    GEMINI_PROMPT_MODELS,
    QWEN_EDIT_SIZE_MANUAL,
    QWEN_EDIT_SIZE_MATCH,
    QWEN_EDIT_SIZE_MAX,
)
from h3_app.prompt_service import DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_MODES
from h3_app.generation.optional_engines import DRAMA3_BACKENDS
from h3_models import FLUX3_MODEL_CHOICES
from .image_library import build_image_file_input
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from .prompt_writer_controls import RemotePromptWriterControls, build_remote_prompt_writer_controls

# Speakers 0..3 get reference-voice slots; the script may still cast up to 16 speakers.
# Drama 3 voices speakers 0..15; the first few slots show, more on demand.
DRAMA3_VOICE_SLOTS = 16
DRAMA3_VISIBLE_SLOTS = 4
DRAMA3_SLOT_INPUT_NAMES = tuple(f"speaker_{speaker}_{field}" for speaker in range(DRAMA3_VOICE_SLOTS)
                                for field in ("voice", "audio", "transcript"))
UPLOAD_VOICE = "Upload a clip"
DRAMA3_SAMPLING_PRESETS = {
    "Strict": (0.7, 0.7, 30),
    "Balance": (1.0, 0.9, 0),
    "Expressive": (2.0, 1.0, 0),
}


@dataclass(frozen=True)
class OptionalEngineActions:
    generate_flux3: Callable
    generate_drama3: Callable
    switch_drama3_backend: Callable
    enhance_flux3_prompt: Callable
    enhance_drama3_text: Callable
    transcribe_drama3_voice: Callable
    drama3_speaker_names: Callable[[], list[str]]
    select_drama3_speaker: Callable
    save_drama3_speaker: Callable
    delete_drama3_speaker: Callable
    interrupt: Callable


@dataclass(frozen=True)
class Flux3View:
    prompt: gr.Textbox
    model: gr.Dropdown
    reference_images: gr.File
    width: gr.Slider
    height: gr.Slider
    edit_size: gr.Radio
    seed: gr.Number
    steps: gr.Slider
    expand_prompt: gr.Checkbox
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox


@dataclass(frozen=True)
class Drama3VoiceSlot:
    speaker: int
    group: gr.Group
    preset: gr.Dropdown
    upload: gr.Group
    audio: gr.Audio
    transcribe: gr.Button
    transcript: gr.Textbox
    save_row: gr.Column
    save_name: gr.Textbox
    save: gr.Button
    delete: gr.Button


@dataclass(frozen=True)
class Drama3View:
    text: gr.Textbox
    speaker_mode: gr.Radio
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    voices: tuple[Drama3VoiceSlot, ...]
    shown_slots: gr.State
    add_speaker: gr.Button
    remove_speaker: gr.Button
    voice_status: gr.Markdown
    seed: gr.Number
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    max_frames: gr.Slider
    split_long_text: gr.Checkbox
    experts: gr.Dropdown
    expert_cache_gb: gr.Slider
    backend: gr.Radio
    parallel: gr.Slider
    retries: gr.Slider
    output: gr.Audio
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox


def _actions(label):
    with gr.Row(elem_classes=["h3-generation-buttons"]):
        run = gr.Button(label, variant="primary", elem_classes=["h3-primary-action"])
        stop = gr.Button("Interrupt", interactive=False)
        run.h3_stop = stop
    status = gr.Textbox(label="Generation progress", lines=3, interactive=False)
    return run, stop, status


def _writer(button_label):
    writer = build_remote_prompt_writer_controls(GEMINI_PROMPT_MODELS, DEFAULT_GEMINI_PROMPT_MODEL)
    enhance = gr.Button(button_label)
    status = gr.Textbox(label="Writer status", lines=2, interactive=False)
    return writer, enhance, status


def build_flux3_view(root):
    with root:
        gr.Markdown("## FLUX 3 Image Klein\nGenerate images or edit with up to ten references. "
                    "Weights download on first use. Accept access conditions for "
                    "[the model repository](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) "
                    "using your Hugging Face account and configure its token on the server.")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                prompt = gr.Textbox(label="Image prompt / complete edit caption", lines=7,
                    placeholder="Describe the scene, composition, subjects and lighting.")
                refs = build_image_file_input(label="Reference images (optional, up to 10)", file_count="multiple",
                                              file_types=["image"], type="filepath")
                expand = gr.Checkbox(value=True, label="Expand the prompt locally",
                    info="Runs BFL's built-in upsampler on the GPU before text-to-image. Turned off for edits "
                         "and after a remote rewrite, which already returns a full caption.")
                with gr.Accordion("Rewrite with Lightning AI or Gemini", open=False):
                    gr.Markdown("Uses BFL's official upsampler instructions. Edits send the reference "
                                "images and return a full edit caption.")
                    writer, enhance, enhance_status = _writer("Rewrite prompt")
                with gr.Row():
                    width = gr.Slider(256, 4096, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 4096, value=1024, step=16, label="Height")
                edit_size = gr.Radio([QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL],
                    value=QWEN_EDIT_SIZE_MATCH, label="Edit output size", visible=False,
                    info="Keep the first reference's aspect ratio at its own size (capped at 4 MP), "
                         "scale it toward 4 MP, or use the width and height above. Applies only to edits.")
                refs.change(lambda paths: (gr.update(value=not bool(paths), interactive=not bool(paths)),
                                           gr.update(visible=bool(paths))),
                            inputs=refs, outputs=[expand, edit_size], queue=False, api_name=False)
                with gr.Accordion("Model and sampling", open=False):
                    model = gr.Dropdown(list(FLUX3_MODEL_CHOICES), value="NVFP4 (Blackwell)", label="Distilled model")
                    steps = gr.Slider(1, 100, value=16, step=1, label="Steps")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Generated image", columns=1, interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate image")
    return Flux3View(prompt, model, refs, width, height, edit_size, seed, steps, expand,
                     writer, enhance, enhance_status, output, run, stop, status)


def _voice_slot(speaker, names):
    with gr.Group(visible=speaker < DRAMA3_VISIBLE_SLOTS) as group:
        preset = gr.Dropdown([UPLOAD_VOICE, *names], value=UPLOAD_VOICE,
                             label=f"Speaker {speaker} voice · <|speaker:{speaker}|>")
        with gr.Group() as upload:
            audio = gr.Audio(label=f"Speaker {speaker} reference clip", type="filepath")
            transcribe = gr.Button("Auto transcribe", size="sm")
        transcript = gr.Textbox(label=f"Speaker {speaker} exact transcript", lines=2,
                                placeholder="Every word spoken in the clip.")
        with gr.Column() as save_row:
            save_name = gr.Textbox(label=f"Save speaker {speaker} as", placeholder="Name")
            save = gr.Button("Save speaker", size="sm")
        delete = gr.Button("Delete saved speaker", size="sm", visible=False)
    return Drama3VoiceSlot(speaker, group, preset, upload, audio, transcribe, transcript,
                           save_row, save_name, save, delete)


def build_drama3_view(root, speaker_names=()):
    with root:
        gr.Markdown("## Drama 3 TTS (experimental)\nBuilt with Fish Audio. "
                    "Start each line with `<|speaker:N|>` for multi-speaker scenes, and place cues such as "
                    "`[sighs]`, `[laughs]` or `<whisper>…</whisper>` in the text. "
                    "The FP8 model and codec download on first use (~40 GB). With experts in RAM, allow "
                    "approximately 48 GB system RAM. [Model and license](https://huggingface.co/fishaudio/drama-3)")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                sampling_preset = gr.Radio(
                    list(DRAMA3_SAMPLING_PRESETS), value="Balance", label="Speech preset",
                    info="Temperature / top-p / top-k: Strict 0.7 / 0.7 / 30 · "
                         "Balance 1.0 / 0.9 / 0 · Expressive 2.0 / 1.0 / 0 (uncapped). "
                         "Custom slider values clear the selection.",
                )
                text = gr.Textbox(label="Speech text", lines=8,
                    placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                    gr.Markdown("Detects the speakers, adds `<|speaker:N|>` tags and expression cues, and "
                                "keeps your lines. Speakers with a reference voice keep their numbers.")
                    speaker_mode = gr.Radio(
                        list(DRAMA3_SPEAKER_MODES), value=DRAMA3_SPEAKER_AUTO, label="Speakers",
                        info="Auto decides from the text. Adapt to voices uses exactly one speaker per "
                             "reference voice or saved speaker below. Single keeps one voice throughout.",
                    )
                    writer, enhance, enhance_status = _writer("Rewrite script")
                with gr.Accordion("Speaker voices (optional)", open=False):
                    gr.Markdown("Clone a voice for any speaker, up to 16. Speakers without a reference "
                                "get a voice from the seed and the script's style cues. Saved speakers are "
                                "reusable by name.")
                    voices = tuple(_voice_slot(speaker, list(speaker_names))
                                   for speaker in range(DRAMA3_VOICE_SLOTS))
                    shown_slots = gr.State(DRAMA3_VISIBLE_SLOTS)
                    with gr.Row():
                        add_speaker = gr.Button("Add speaker", size="sm")
                        remove_speaker = gr.Button("Remove last speaker", size="sm")
                    voice_status = gr.Markdown("")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Speech and memory settings", open=False):
                    backend = gr.Radio(list(DRAMA3_BACKENDS), value=DRAMA3_BACKENDS[0], label="Backend",
                        info="ComfyUI generates the whole text in one take and can invent voices. "
                             "SGLang-Omni (Linux) is for long scripts where every speaker has a "
                             "voice below: each speaker's lines go out with that speaker's clip, in parallel, and "
                             "near-silent takes can be retried when retries are enabled. First use installs the backend and downloads a "
                             "44 GB checkpoint; the server then stays loaded between Drama 3 jobs and stops when "
                             "another engine needs the GPU. SGLang-Omni does not support seed control.")
                    with gr.Group(visible=False) as served_settings:
                        parallel = gr.Slider(1, 32, value=8, step=1, label="Parallel requests")
                        retries = gr.Slider(0, 4, value=0, step=1, label="Retries per chunk",
                            info="Default 0 keeps the first take. Enable to retry near-silent audio; duration warnings do not trigger retries.")
                    temperature = gr.Slider(0, 2, value=1, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.05, 1, value=0.9, step=0.05, label="Top-p")
                    top_k = gr.Slider(0, 4096, value=0, step=1, label="Top-k",
                        info="0 keeps all candidates. Small caps can make speech stall at pauses.")
                    max_frames = gr.Slider(1, 8192, value=8192, step=1, label="Maximum audio frames per take",
                        info="21.5 frames per second; 8192 is about 6.3 minutes. A take stops when the "
                             "speech ends, so this only limits a take that runs on.")
                    with gr.Group() as comfy_settings:
                        split = gr.Checkbox(value=False, label="Split long text into connected chunks",
                            info="Off generates the whole text in one take, avoiding silent gaps at chunk joins.")
                        experts = gr.Dropdown(["in RAM", "in VRAM"], value="in VRAM", label="Expert weights",
                            info="In VRAM requires room for the whole model plus working memory.")
                        cache = gr.Slider(0, 256, value=0, step=0.5, label="Expert VRAM cache (GB, 0 automatic)")
                    backend.change(
                        lambda choice: (gr.update(visible=choice != "ComfyUI"),
                                        gr.update(visible=choice == "ComfyUI")),
                        inputs=backend, outputs=[served_settings, comfy_settings],
                        queue=False, show_progress="hidden", api_name=False,
                    )
                sampling_controls = [temperature, top_p, top_k]
                sampling_preset.input(
                    lambda name: DRAMA3_SAMPLING_PRESETS[name],
                    inputs=sampling_preset, outputs=sampling_controls,
                    queue=False, show_progress="hidden", api_name=False,
                )
                # Derive the label from the sliders, including restored browser settings.
                # Only user input on the preset applies values, avoiding a feedback loop.
                def matching_preset(temperature, top_p, top_k):
                    values = (temperature, top_p, top_k)
                    return next((name for name, preset in DRAMA3_SAMPLING_PRESETS.items()
                                 if values == preset), None)

                for control in sampling_controls:
                    control.change(
                        matching_preset, inputs=sampling_controls, outputs=sampling_preset,
                        queue=False, show_progress="hidden", api_name=False,
                    )
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate speech")
    return Drama3View(text, speaker_mode, writer, enhance, enhance_status, voices, shown_slots, add_speaker,
                      remove_speaker, voice_status, seed, temperature,
                      top_p, top_k, max_frames, split, experts, cache, backend, parallel, retries,
                      output, run, stop, status)


def settings_components(view):
    """Persistable components by setting name; voice slots and nested groups excluded."""
    components = {name: value for name, value in vars(view).items()
                  if isinstance(value, gr.components.Component)}
    components.update(prompt_backend=view.writer.backend, prompt_model=view.writer.model,
                      lightning_model=view.writer.lightning_model)
    return components


def _writer_inputs(writer):
    return [writer.model, writer.gemini_api_key, writer.backend, writer.lightning_api_key, writer.lightning_model]


def _bind_generation(view, family, generate, interrupt, inputs):
    event = bind_gpu_action(view.run.click, owned_generation(generate, family), inputs=inputs,
                            outputs=[view.output, view.status], api_name=f"generate_{family}",
                            show_progress="minimal")
    stopped = view.stop.click(owned_interrupt(interrupt, family), outputs=view.status, queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return event


def bind_flux3(view, actions):
    bind_prompt_action(
        view.enhance.click, actions.enhance_flux3_prompt,
        inputs=[view.prompt, view.reference_images, view.width, view.height, *_writer_inputs(view.writer)],
        outputs=[view.prompt, view.enhance_status, view.expand_prompt],
        show_progress="minimal", api_name="enhance_flux3_prompt",
    )
    return _bind_generation(view, "flux3", actions.generate_flux3, actions.interrupt, [
        view.prompt, view.model, view.reference_images, view.width, view.height,
        view.edit_size, view.seed, view.steps, view.expand_prompt,
    ])


def drama3_voice_inputs(view):
    return [component for slot in view.voices for component in (slot.preset, slot.audio, slot.transcript)]


def refresh_drama3_presets_on_load(demo, view, actions):
    """Show speakers saved from other sessions; saves and deletes refresh in place."""
    presets = [slot.preset for slot in view.voices]
    demo.load(
        lambda: [gr.update(choices=[UPLOAD_VOICE, *actions.drama3_speaker_names()])] * len(presets),
        outputs=presets, queue=False, show_progress="hidden", api_name=False,
    )


def _slot_visibility(count):
    """Show ``count`` slots; disable Add at the limit and Remove at one slot."""
    count = max(1, min(DRAMA3_VOICE_SLOTS, int(count)))
    return (count, *[gr.update(visible=index < count) for index in range(DRAMA3_VOICE_SLOTS)],
            gr.update(interactive=count < DRAMA3_VOICE_SLOTS), gr.update(interactive=count > 1))


def bind_drama3(view, actions):
    bind_gpu_action(
        view.backend.input, actions.switch_drama3_backend,
        inputs=view.backend, outputs=view.status,
        show_progress="minimal", api_name=False,
    )
    presets = [slot.preset for slot in view.voices]
    groups = [slot.group for slot in view.voices]
    view.add_speaker.click(lambda count: _slot_visibility(count + 1), inputs=view.shown_slots,
                           outputs=[view.shown_slots, *groups, view.add_speaker, view.remove_speaker],
                           queue=False, show_progress="hidden", api_name=False)
    def remove(count):
        # Removing a slot clears it, so a hidden slot never sends a voice.
        count = max(1, min(DRAMA3_VOICE_SLOTS, int(count)))
        cleared = [gr.update()] * (3 * DRAMA3_VOICE_SLOTS)
        if count > 1:
            cleared[3 * (count - 1):3 * count] = [gr.update(value=UPLOAD_VOICE), gr.update(value=None),
                                                  gr.update(value="")]
        return (*_slot_visibility(count - 1), *cleared)

    view.remove_speaker.click(
        remove, inputs=view.shown_slots,
        outputs=[view.shown_slots, *groups, view.add_speaker, view.remove_speaker,
                 *drama3_voice_inputs(view)],
        queue=False, show_progress="hidden", api_name=False,
    )
    bind_prompt_action(
        view.enhance.click, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *_writer_inputs(view.writer), *drama3_voice_inputs(view)],
        outputs=[view.text, view.enhance_status],
        show_progress="minimal", api_name="enhance_drama3_text",
    )
    for slot in view.voices:
        bind_gpu_action(
            slot.transcribe.click, actions.transcribe_drama3_voice, inputs=[slot.audio],
            outputs=[slot.transcript, view.voice_status], show_progress="minimal",
            api_name="transcribe_drama3_voice" if slot.speaker == 0 else False,
        )
        slot.preset.change(
            actions.select_drama3_speaker, inputs=[slot.preset],
            outputs=[slot.upload, slot.transcript, slot.save_row, slot.delete, view.voice_status],
            queue=False, show_progress="hidden", api_name=False,
        )
        slot.save.click(
            lambda name, audio, transcript, speaker=slot.speaker: actions.save_drama3_speaker(
                speaker, name, audio, transcript, len(presets)),
            inputs=[slot.save_name, slot.audio, slot.transcript],
            outputs=[view.voice_status, *presets], queue=False, api_name=False,
        )
        slot.delete.click(
            lambda name, speaker=slot.speaker: actions.delete_drama3_speaker(speaker, name, len(presets)),
            inputs=[slot.preset], outputs=[view.voice_status, *presets], queue=False, api_name=False,
        )
    return _bind_generation(view, "drama3", actions.generate_drama3, actions.interrupt, [
        view.text, *drama3_voice_inputs(view), view.seed, view.temperature, view.top_p, view.top_k,
        view.max_frames, view.split_long_text, view.experts, view.expert_cache_gb,
        view.backend, view.parallel, view.retries,
    ])
