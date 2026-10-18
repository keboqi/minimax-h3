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
from h3_app.workflows.flux3 import flux3_sampling_defaults
from h3_app.voice_design import LANGUAGES as VOICE_DESIGN_LANGUAGES
from h3_app.drama3_voice_design import ENGINE_DRAMA3, ENGINE_QWEN, VOICE_DESIGN_ENGINES
from h3_models import (FLUX3_MODEL_CHOICES, FLUX3_UPSAMPLER_CHOICES, DEFAULT_FLUX3_UPSAMPLER,
                       DRAMA3_MODEL_CHOICES, DEFAULT_DRAMA3_MODEL, DRAMA3_ASR_CHOICES, DEFAULT_DRAMA3_ASR)
from .image_library import build_image_file_input
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from .prompt_writer_controls import RemotePromptWriterControls, build_remote_prompt_writer_controls

# Drama 3 voices speakers 0..15; the first few slots show, more on demand.
DRAMA3_VOICE_SLOTS = 16
DRAMA3_VISIBLE_SLOTS = 4
DRAMA3_SLOT_INPUT_NAMES = tuple(f"speaker_{speaker}_{field}" for speaker in range(DRAMA3_VOICE_SLOTS)
                                for field in ("voice", "audio", "transcript"))
DRAMA3_CAST_INPUT_NAMES = tuple(f"speaker_{speaker}_{field}" for speaker in range(DRAMA3_VOICE_SLOTS)
                               for field in ("name", "description", "preview_text"))
UPLOAD_VOICE = "Upload a clip"
DRAMA3_SAMPLING_PRESETS = {
    "Strict": (0.7, 0.7, 30),
    "Balance": (1.0, 0.9, 0),
    "Expressive": (2.0, 1.0, 30),
}


def drama3_voice_choices(names):
    # Keep the existing API value while showing both ways to supply a reference.
    return [("Upload or generate a voice", UPLOAD_VOICE), *names]


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
    generate_drama3_voice: Callable
    generate_drama3_voices: Callable
    copy_drama3_writer_prompt: Callable
    apply_drama3_writer_reply: Callable


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
    upsampler: gr.Dropdown
    cfg: gr.Slider
    layout: gr.Textbox


@dataclass(frozen=True)
class Drama3VoiceSlot:
    speaker: int
    group: gr.Accordion
    preset: gr.Dropdown
    upload: gr.Group
    audio: gr.Audio
    transcribe: gr.Button
    transcript: gr.Textbox
    save_row: gr.Column
    save_name: gr.Textbox
    save: gr.Button
    delete: gr.Button
    description: gr.Textbox
    preview_text: gr.Textbox
    language: gr.Dropdown
    generate: gr.Button
    stop: gr.Button
    status: gr.Markdown
    has_clip: gr.Checkbox


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
    model: gr.Dropdown
    asr_model: gr.Dropdown
    voice_engine: gr.Radio
    generate_voices: gr.Button
    stop_voices: gr.Button
    skip_existing: gr.Checkbox
    voices_panel: gr.Accordion
    copy_prompt: gr.Button
    writer_prompt: gr.Textbox
    pasted_reply: gr.Textbox
    apply_reply: gr.Button


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
                prompt = gr.Textbox(label="Image prompt / edit request", lines=7,
                    placeholder="Describe a scene or the changes to make to your references.")
                refs = build_image_file_input(label="Reference images (optional, up to 10)", file_count="multiple",
                                              file_types=["image"], type="filepath")
                expand = gr.Checkbox(value=True, label="Expand the prompt locally",
                    info="Runs BFL's upsampler on the GPU for generation and edits, using all reference images. "
                         "Turned off after a remote rewrite, which already returns an expanded prompt.")
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
                refs.change(lambda paths: (gr.update(value=True, interactive=True),
                                           gr.update(visible=bool(paths))),
                            inputs=refs, outputs=[expand, edit_size], queue=False, api_name=False)
                with gr.Accordion("Model and sampling", open=False):
                    model = gr.Dropdown(list(FLUX3_MODEL_CHOICES), value="NVFP4 (Blackwell)", label="Image model",
                                        info="Distilled uses 16 steps and CFG 1. Base uses 50 steps and CFG 4.")
                    upsampler = gr.Dropdown(list(FLUX3_UPSAMPLER_CHOICES), value=DEFAULT_FLUX3_UPSAMPLER,
                                             label="Prompt upsampler",
                                             info="W4A8 uses less VRAM. Used only when local expansion is enabled.")
                    steps = gr.Slider(1, 100, value=16, step=1, label="Steps")
                    cfg = gr.Slider(1, 20, value=1, step=0.1, label="Base CFG", visible=False)
                    model.change(lambda choice: (flux3_sampling_defaults(choice)[0],
                        gr.update(value=flux3_sampling_defaults(choice)[1], visible=choice.startswith("Base "))),
                        inputs=model, outputs=[steps, cfg], queue=False, api_name=False)
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Layout boxes (optional)", open=False):
                    layout = gr.Textbox(label="Layout boxes", lines=4,
                        placeholder='[{"bbox": [100, 100, 700, 450], "desc": "a red bicycle"}]',
                        info="JSON list. bbox is [top, left, bottom, right] on a 0–1000 grid. "
                             "Boxes guide placement; they are not pixel masks. Use type text and text for lettering. "
                             "Descriptions can name <ref_image_0>, <ref_image_1>, etc. for reference-guided layouts.")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Generated image", columns=1, interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate image")
    return Flux3View(prompt, model, refs, width, height, edit_size, seed, steps, expand,
                     writer, enhance, enhance_status, output, run, stop, status, upsampler, cfg, layout)


def drama3_slot_title(speaker, name="", preset=UPLOAD_VOICE, has_clip=False):
    """Panel title: index, character name, and whether a voice is ready."""
    title = f"Speaker {speaker}"
    if str(name or "").strip():
        title += f" · {str(name).strip()}"
    if preset and preset != UPLOAD_VOICE:
        title += " · saved voice"
    elif has_clip:
        title += " · voice ready"
    return title


def drama3_voices_title(shown=DRAMA3_VISIBLE_SLOTS, presets=(), has_clips=()):
    """Speaker voices panel title with the shown speaker count and how many have a voice."""
    shown = max(1, min(DRAMA3_VOICE_SLOTS, int(shown or 1)))
    ready = sum(1 for speaker in range(shown)
                if (speaker < len(presets) and presets[speaker] and presets[speaker] != UPLOAD_VOICE)
                or (speaker < len(has_clips) and has_clips[speaker]))
    return f"Speaker voices (optional) \u00b7 {shown} speaker{'s' if shown != 1 else ''} \u00b7 {ready} with a voice"


def _voice_slot(speaker, names):
    with gr.Accordion(drama3_slot_title(speaker), open=speaker == 0,
                      visible=speaker < DRAMA3_VISIBLE_SLOTS) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(drama3_voice_choices(names), value=UPLOAD_VOICE,
                                 label=f"Speaker {speaker} voice · <|speaker:{speaker}|>")
            save_name = gr.Textbox(label=f"Speaker {speaker} name", placeholder="Character or role")
        description = gr.Textbox(label=f"Speaker {speaker} voice description", lines=3,
                                  placeholder="Age range, pitch, timbre, accent, articulation and pace.")
        with gr.Group() as upload:
            preview_text = gr.Textbox(label=f"Speaker {speaker} preview speech", lines=2,
                                     placeholder="Words a generated reference clip will speak.")
            language = gr.Dropdown(list(VOICE_DESIGN_LANGUAGES), value="Auto", label="Preview language",
                                   visible=False)
            with gr.Row():
                generate = gr.Button("Generate voice", size="sm")
                stop = gr.Button("Interrupt voice generation", size="sm", interactive=False)
                generate.h3_stop = stop
            audio = gr.Audio(label=f"Speaker {speaker} reference clip", type="filepath")
            transcribe = gr.Button("Auto transcribe", size="sm")
        transcript = gr.Textbox(label=f"Speaker {speaker} exact transcript", lines=2,
                                placeholder="Every word spoken in the clip.")
        with gr.Row() as save_row:
            save = gr.Button("Save speaker", size="sm")
        delete = gr.Button("Delete saved speaker", size="sm", visible=False)
        status = gr.Markdown("")
        has_clip = gr.Checkbox(False, visible=False)
    return Drama3VoiceSlot(speaker, group, preset, upload, audio, transcribe, transcript,
                           save_row, save_name, save, delete, description, preview_text, language, generate, stop,
                           status, has_clip)


def build_drama3_view(root, speaker_names=()):
    with root:
        gr.Markdown("## Drama 3 TTS (experimental)\nBuilt with Fish Audio. "
                    "Start each line with `<|speaker:N|>` for multi-speaker scenes, and place cues such as "
                    "`[sighs]`, `[laughs]` or `<whisper>…</whisper>` in the text. "
                    "The selected model and codec download on first use (~40 GB for FP8/INT8). With experts in RAM, allow "
                    "approximately 48 GB system RAM for FP8/INT8 or 40 GB for W6A8. "
                    "[Model and license](https://huggingface.co/fishaudio/drama-3)")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                sampling_preset = gr.Radio(
                    list(DRAMA3_SAMPLING_PRESETS), value="Balance", label="Speech preset",
                    info="Temperature / top-p / top-k: Strict 0.7 / 0.7 / 30 · "
                         "Balance 1.0 / 0.9 / 0 (uncapped) · Expressive 2.0 / 1.0 / 30. "
                         "Custom slider values clear the selection.",
                )
                text = gr.Textbox(label="Speech text", lines=8,
                    placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                    gr.Markdown("Separates narration and dialogue, adds `<|speaker:N|>` tags and expression cues, "
                                "and fills matching voice slots with speaker names, descriptions and preview speech. "
                                "Speakers keep their assigned numbers and existing reference clips.")
                    speaker_mode = gr.Radio(
                        list(DRAMA3_SPEAKER_MODES), value=DRAMA3_SPEAKER_AUTO, label="Speakers",
                        info="Auto decides from the text. Adapt to voices uses exactly one speaker per "
                             "reference voice or saved speaker below. Single keeps one voice throughout.",
                    )
                    writer, enhance, enhance_status = _writer("Rewrite script")
                    with gr.Accordion("Use another chat model", open=False):
                        gr.Markdown("Copy the full writer prompt (system prompt, speaker slots and speech text) "
                                    "into any chat model. Paste its JSON reply here; it is checked like the "
                                    "built-in writer before the script and speaker slots change.")
                        copy_prompt = gr.Button("Show full writer prompt", size="sm")
                        writer_prompt = gr.Textbox(label="Full writer prompt", lines=8, max_lines=20,
                                                   interactive=False, visible=False, buttons=["copy"])
                        pasted_reply = gr.Textbox(label="Pasted writer reply", lines=6, max_lines=20,
                                                  placeholder='{"script": "<|speaker:0|> ...", "speakers": [...]}')
                        apply_reply = gr.Button("Apply pasted reply", size="sm")
                with gr.Accordion(drama3_voices_title(), open=False) as voices_panel:
                    gr.Markdown("Each speaker has its own panel. Upload a reference clip or generate one from "
                                "the voice description and preview speech; a generated clip uses the preview speech "
                                "as its transcript. Save either kind as a reusable speaker preset. Speakers without "
                                "clips use Drama 3's invented voices.")
                    voice_engine = gr.Radio(
                        list(VOICE_DESIGN_ENGINES), value=ENGINE_DRAMA3, label="Voice generator",
                        info="Drama 3 casts the voice from the description as a delivery cue, on the selected "
                             "backend with the model and sampling settings below. "
                             "[Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) "
                             "runs in its own isolated runtime, downloaded on first use.",
                    )
                    with gr.Row(equal_height=True):
                        skip_existing = gr.Checkbox(
                            True, label="Skip speakers with a voice",
                            info="Uncheck to regenerate clips, for example after changing the backend.")
                        asr_model = gr.Dropdown(list(DRAMA3_ASR_CHOICES), value=DEFAULT_DRAMA3_ASR,
                                                label="Voice transcript model",
                                                info="For Auto transcribe. 0.6B uses less memory. "
                                                     "Check names and every spoken word.")
                    with gr.Row():
                        generate_voices = gr.Button("Generate all speaker voices", size="sm")
                        stop_voices = gr.Button("Interrupt", size="sm", interactive=False)
                        generate_voices.h3_stop = stop_voices
                    voice_status = gr.Markdown("")
                    voices = tuple(_voice_slot(speaker, list(speaker_names))
                                   for speaker in range(DRAMA3_VOICE_SLOTS))
                    shown_slots = gr.State(DRAMA3_VISIBLE_SLOTS)
                    with gr.Row():
                        add_speaker = gr.Button("Add speaker", size="sm")
                        remove_speaker = gr.Button("Remove last speaker", size="sm")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Speech and memory settings", open=False):
                    backend = gr.Radio(list(DRAMA3_BACKENDS), value=DRAMA3_BACKENDS[0], label="Backend",
                        info="ComfyUI generates the whole text in one take and can invent voices. "
                             "SGLang-Omni (Linux) is for long scripts where every speaker has a "
                             "voice below: each speaker's lines go out with that speaker's clip, in parallel, and "
                             "near-silent takes can be retried when retries are enabled. First use installs the backend and downloads a "
                             "44 GB checkpoint; the server then stays loaded between Drama 3 jobs and sleeps or stops when "
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
                        model = gr.Dropdown(list(DRAMA3_MODEL_CHOICES), value=DEFAULT_DRAMA3_MODEL,
                                            label="Drama model",
                                            info="FP8 preserves the original weights. INT8 is useful with RAM experts; "
                                                 "W6A8 reduces model RAM to about 40 GB including working memory.")
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
                      output, run, stop, status, model, asr_model, voice_engine,
                      generate_voices, stop_voices, skip_existing, voices_panel, copy_prompt, writer_prompt,
                      pasted_reply, apply_reply)


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
        view.edit_size, view.seed, view.steps, view.expand_prompt, view.upsampler, view.cfg, view.layout,
    ])


def drama3_voice_inputs(view):
    return [component for slot in view.voices for component in (slot.preset, slot.audio, slot.transcript)]


def drama3_cast_inputs(view):
    return [component for slot in view.voices
            for component in (slot.save_name, slot.description, slot.preview_text)]


def refresh_drama3_presets_on_load(demo, view, actions):
    """Show speakers saved from other sessions; saves and deletes refresh in place."""
    presets = [slot.preset for slot in view.voices]
    demo.load(
        lambda: [gr.update(choices=drama3_voice_choices(actions.drama3_speaker_names()))] * len(presets),
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
        metadata = [gr.update()] * (3 * DRAMA3_VOICE_SLOTS)
        if count > 1:
            cleared[3 * (count - 1):3 * count] = [gr.update(value=UPLOAD_VOICE), gr.update(value=None),
                                                  gr.update(value="")]
            metadata[3 * (count - 1):3 * count] = [gr.update(value="")] * 3
        return (*_slot_visibility(count - 1), *cleared, *metadata)

    view.remove_speaker.click(
        remove, inputs=view.shown_slots,
        outputs=[view.shown_slots, *groups, view.add_speaker, view.remove_speaker,
                 *drama3_voice_inputs(view), *drama3_cast_inputs(view)],
        queue=False, show_progress="hidden", api_name=False,
    )
    writer_outputs = [view.text, view.enhance_status, *drama3_cast_inputs(view),
                      view.shown_slots, *groups, view.add_speaker, view.remove_speaker]
    writer_slots = [*drama3_voice_inputs(view), *drama3_cast_inputs(view), view.shown_slots]
    view.copy_prompt.click(actions.copy_drama3_writer_prompt,
                           inputs=[view.text, view.speaker_mode, *writer_slots],
                           outputs=[view.writer_prompt, view.enhance_status],
                           queue=False, show_progress="hidden", api_name=False)
    view.apply_reply.click(actions.apply_drama3_writer_reply,
                           inputs=[view.pasted_reply, view.text, view.speaker_mode, *writer_slots],
                           outputs=writer_outputs, queue=False, show_progress="hidden", api_name=False)
    counted = [view.shown_slots, *presets, *[slot.has_clip for slot in view.voices]]
    for trigger in (view.shown_slots.change, *[slot.preset.change for slot in view.voices],
                    *[slot.has_clip.change for slot in view.voices]):
        trigger(lambda shown, *values: gr.update(label=drama3_voices_title(
                    shown, values[:DRAMA3_VOICE_SLOTS], values[DRAMA3_VOICE_SLOTS:])),
                inputs=counted, outputs=view.voices_panel,
                queue=False, show_progress="hidden", api_name=False)
    bind_prompt_action(
        view.enhance.click, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *_writer_inputs(view.writer), *drama3_voice_inputs(view),
                *drama3_cast_inputs(view), view.shown_slots],
        outputs=[view.text, view.enhance_status, *drama3_cast_inputs(view),
                 view.shown_slots, *groups, view.add_speaker, view.remove_speaker],
        show_progress="minimal", api_name="enhance_drama3_text",
    )
    voice_settings = [view.voice_engine, view.backend, view.seed, view.temperature, view.top_p, view.top_k,
                      view.max_frames, view.experts, view.expert_cache_gb, view.model]
    generated_all = bind_gpu_action(
        view.generate_voices.click, owned_generation(actions.generate_drama3_voices, "drama3_voice"),
        inputs=[*voice_settings, view.parallel, view.shown_slots, view.skip_existing,
                *[component for slot in view.voices
                  for component in (slot.preset, slot.has_clip, slot.description, slot.preview_text, slot.language)]],
        outputs=[*[component for slot in view.voices for component in (slot.audio, slot.transcript, slot.status)],
                 view.voice_status],
        show_progress="minimal", api_name=False,
    )
    stopped_all = view.stop_voices.click(owned_interrupt(actions.interrupt, "drama3_voice"),
                                         outputs=view.voice_status, queue=False, api_name=False)
    stopped_all.then(fn=None, cancels=[generated_all], queue=False, api_name=False)
    languages = [slot.language for slot in view.voices]
    view.voice_engine.change(lambda engine: [gr.update(visible=engine == ENGINE_QWEN)] * len(languages),
                             inputs=view.voice_engine, outputs=languages,
                             queue=False, show_progress="hidden", api_name=False)
    for slot in view.voices:
        generated = bind_gpu_action(
            slot.generate.click, owned_generation(actions.generate_drama3_voice, "drama3_voice"),
            inputs=[slot.description, slot.preview_text, slot.language, *voice_settings],
            outputs=[slot.audio, slot.transcript, slot.status], show_progress="minimal",
            api_name="generate_drama3_voice" if slot.speaker == 0 else False,
        )
        stopped = slot.stop.click(owned_interrupt(actions.interrupt, "drama3_voice"),
                                  outputs=slot.status, queue=False, api_name=False)
        stopped.then(fn=None, cancels=[generated], queue=False, api_name=False)
        slot.audio.change(lambda audio: bool(audio), inputs=slot.audio, outputs=slot.has_clip,
                          queue=False, show_progress="hidden", api_name=False)
        for trigger in (slot.save_name.change, slot.preset.change, slot.has_clip.change):
            trigger(lambda name, preset, has_clip, speaker=slot.speaker: gr.update(
                        label=drama3_slot_title(speaker, name, preset, has_clip)),
                    inputs=[slot.save_name, slot.preset, slot.has_clip], outputs=slot.group,
                    queue=False, show_progress="hidden", api_name=False)
        slot.audio.upload(lambda: gr.update(value="", interactive=True), outputs=slot.transcript,
                          queue=False, show_progress="hidden", api_name=False)
        bind_gpu_action(
            slot.transcribe.click, actions.transcribe_drama3_voice, inputs=[slot.audio, view.asr_model],
            outputs=[slot.transcript, slot.status], show_progress="minimal",
            api_name="transcribe_drama3_voice" if slot.speaker == 0 else False,
        )
        slot.preset.change(
            actions.select_drama3_speaker, inputs=[slot.preset],
            outputs=[slot.upload, slot.transcript, slot.save_row, slot.delete, slot.status,
                     slot.save_name, slot.description, slot.preview_text, slot.audio],
            queue=False, show_progress="hidden", api_name=False,
        )
        slot.save.click(
            lambda name, audio, transcript, description, speaker=slot.speaker: actions.save_drama3_speaker(
                speaker, name, audio, transcript, len(presets), description),
            inputs=[slot.save_name, slot.audio, slot.transcript, slot.description],
            outputs=[slot.status, *presets], queue=False, api_name=False,
        )
        slot.delete.click(
            lambda name, speaker=slot.speaker: actions.delete_drama3_speaker(speaker, name, len(presets)),
            inputs=[slot.preset], outputs=[slot.status, *presets], queue=False, api_name=False,
        )
    return _bind_generation(view, "drama3", actions.generate_drama3, actions.interrupt, [
        view.text, *drama3_voice_inputs(view), view.seed, view.temperature, view.top_p, view.top_k,
        view.max_frames, view.split_long_text, view.experts, view.expert_cache_gb,
        view.backend, view.parallel, view.retries, view.model,
    ])
