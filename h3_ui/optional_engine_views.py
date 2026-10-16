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
from h3_models import FLUX3_MODEL_CHOICES
from .image_library import build_image_file_input
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from .prompt_writer_controls import RemotePromptWriterControls, build_remote_prompt_writer_controls

# Speakers 0..3 get reference-voice slots; the script may still cast up to 16 speakers.
DRAMA3_VOICE_SLOTS = 4
UPLOAD_VOICE = "Upload a clip"


@dataclass(frozen=True)
class OptionalEngineActions:
    generate_flux3: Callable
    generate_drama3: Callable
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
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    voices: tuple[Drama3VoiceSlot, ...]
    voice_status: gr.Markdown
    seed: gr.Number
    temperature: gr.Slider
    top_p: gr.Slider
    top_k: gr.Slider
    max_frames: gr.Slider
    split_long_text: gr.Checkbox
    experts: gr.Dropdown
    expert_cache_gb: gr.Slider
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
    with gr.Group():
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
    return Drama3VoiceSlot(speaker, preset, upload, audio, transcribe, transcript,
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
                text = gr.Textbox(label="Speech text", lines=8,
                    placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                    gr.Markdown("Detects the speakers, adds `<|speaker:N|>` tags and expression cues, and "
                                "keeps your lines. Speakers with a reference voice keep their numbers.")
                    writer, enhance, enhance_status = _writer("Rewrite script")
                with gr.Accordion("Speaker voices (optional)", open=False):
                    gr.Markdown("Clone a voice for any of the first four speakers. Speakers without a "
                                "reference get a voice from the seed and the script's style cues. Saved "
                                "speakers are reusable by name.")
                    voices = tuple(_voice_slot(speaker, list(speaker_names))
                                   for speaker in range(DRAMA3_VOICE_SLOTS))
                    voice_status = gr.Markdown("")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Speech and memory settings", open=False):
                    temperature = gr.Slider(0, 2, value=1, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.05, 1, value=0.9, step=0.05, label="Top-p")
                    top_k = gr.Slider(1, 4096, value=30, step=1, label="Top-k")
                    max_frames = gr.Slider(1, 8192, value=600, step=1, label="Maximum audio frames per chunk",
                        info="21.5 frames per second; 600 caps each chunk at about 28 seconds.")
                    split = gr.Checkbox(value=True, label="Split long text into connected chunks")
                    experts = gr.Dropdown(["in RAM", "in VRAM"], value="in RAM", label="Expert weights",
                        info="In VRAM requires room for the whole model plus working memory.")
                    cache = gr.Slider(0, 256, value=0, step=0.5, label="Expert VRAM cache (GB, 0 automatic)")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate speech")
    return Drama3View(text, writer, enhance, enhance_status, voices, voice_status, seed, temperature,
                      top_p, top_k, max_frames, split, experts, cache, output, run, stop, status)


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


def bind_drama3(view, actions):
    presets = [slot.preset for slot in view.voices]
    bind_prompt_action(
        view.enhance.click, actions.enhance_drama3_text,
        inputs=[view.text, *_writer_inputs(view.writer), *drama3_voice_inputs(view)],
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
    ])
