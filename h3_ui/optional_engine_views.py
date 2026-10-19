"""FLUX 3 and MOSS-Image image and Drama 3 speech controls and owned job bindings."""

from dataclasses import dataclass
import json
from typing import Callable

import gradio as gr
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL,
    DEFAULT_MOSS_IMAGE_PRESET,
    GEMINI_PROMPT_MODELS,
    MOSS_IMAGE_PRESETS,
    QWEN_EDIT_SIZE_MANUAL,
    QWEN_EDIT_SIZE_MATCH,
    QWEN_EDIT_SIZE_MAX,
)
from h3_app.drama3_voice_design import ENGINE_DRAMA3, ENGINE_QWEN, VOICE_DESIGN_ENGINES
from h3_app.generation.optional_engines import DRAMA3_BACKENDS
from h3_app.prompt_service import (
    DRAMA3_PROMPT_FAILED,
    DRAMA3_SCRIPT_ADVANCED,
    DRAMA3_SCRIPT_BASIC,
    DRAMA3_SCRIPT_MODES,
    DRAMA3_SPEAKER_AUTO,
    DRAMA3_SPEAKER_MODES,
)
from h3_app.voice_design import LANGUAGES as VOICE_DESIGN_LANGUAGES
from h3_app.workflows.flux3 import flux3_sampling_defaults
from h3_app.workflows.moss_image_prompt import MOSS_IMAGE_REWRITE_MODES
from h3_models import (
    DEFAULT_DRAMA3_ASR,
    DEFAULT_DRAMA3_MODEL,
    DEFAULT_FLUX3_UPSAMPLER,
    DRAMA3_ASR_CHOICES,
    DRAMA3_MODEL_CHOICES,
    FLUX3_MODEL_CHOICES,
    FLUX3_UPSAMPLER_CHOICES,
    DEFAULT_MOSS_IMAGE_TEXT_ENCODER,
    MOSS_IMAGE_MODEL_CHOICES,
    MOSS_IMAGE_TEXT_ENCODER_CHOICES,
    MOSS_IMAGE_TURBO_MODES,
    MOSS_IMAGE_VAE_CHOICES,
)
from .image_library import build_image_file_input
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from .prompt_writer_controls import RemotePromptWriterControls, build_remote_prompt_writer_controls

# Drama 3 voices speakers 0..15; the first few slots show, more on demand.
DRAMA3_VOICE_SLOTS = 16
DRAMA3_VISIBLE_SLOTS = 4
DRAMA3_SLOT_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("voice", "audio", "transcript")
)
DRAMA3_CAST_INPUT_NAMES = tuple(
    f"speaker_{speaker}_{field}"
    for speaker in range(DRAMA3_VOICE_SLOTS)
    for field in ("name", "description", "preview_text")
)
UPLOAD_VOICE = "Upload a clip"

# Drama 3 sound effects & background music slots (Stable Audio 3 Medium, SGLang-Omni only).
DRAMA3_SOUND_SLOTS = 16
DRAMA3_VISIBLE_SOUND_SLOTS = 4
UPLOAD_SOUND = "Upload a clip"
DRAMA3_SOUND_SLOT_INPUT_NAMES = tuple(
    f"sound_{index}_{field}"
    for index in range(DRAMA3_SOUND_SLOTS)
    for field in ("preset", "audio", "name", "description", "duration")
)
DRAMA3_SOUND_META_INPUT_NAMES = tuple(
    f"sound_{index}_{field}"
    for index in range(DRAMA3_SOUND_SLOTS)
    for field in ("name", "description", "duration")
)

DRAMA3_SAMPLING_PRESETS = {
    "Strict": (0.7, 0.7, 30),
    "Balance": (1.0, 0.9, 0),
    "Expressive": (2.0, 1.0, 30),
}

# Unqueued, unlisted presentation events.
_QUIET = {"queue": False, "show_progress": "hidden", "api_name": False}
# On every server event, Gradio's frontend refreshes the loading status of each
# component any earlier server event used, with a search of the whole layout per
# component. Labels, flags and visibility derived from values the browser
# already holds are therefore computed there (fn=None), never in a round trip.
_SKIP_JS = '{"__type__": "update"}'


# Native-grid size presets, labelled like Qwen's so qwen_resolution_preset_values parses them.
FLUX3_1K_RESOLUTION_PRESETS = (
    "1K · 1:1 · 1024×1024", "1K · 4:3 · 1152×864", "1K · 3:2 · 1248×832", "1K · 16:9 · 1360×768",
    "1K · 3:4 · 864×1152", "1K · 2:3 · 832×1248", "1K · 9:16 · 768×1360",
)
# FLUX 3's 4 MP cap.
FLUX3_2K_RESOLUTION_PRESETS = (
    "2K · 1:1 · 2048×2048", "2K · 4:3 · 2304×1728", "2K · 3:2 · 2448×1632", "2K · 16:9 · 2720×1536",
    "2K · 3:4 · 1728×2304", "2K · 2:3 · 1632×2448", "2K · 9:16 · 1536×2720",
)
# OpenMOSS recommends 1M to 2M pixels.
MOSS_IMAGE_1M_RESOLUTION_PRESETS = tuple(name.replace("1K", "1M", 1) for name in FLUX3_1K_RESOLUTION_PRESETS)
MOSS_IMAGE_2M_RESOLUTION_PRESETS = (
    "2M · 1:1 · 1456×1456", "2M · 4:3 · 1664×1248", "2M · 3:2 · 1776×1184", "2M · 16:9 · 1920×1088",
    "2M · 3:4 · 1248×1664", "2M · 2:3 · 1184×1776", "2M · 9:16 · 1088×1920",
)


def _resolution_presets(small, large, small_label, large_label, small_info, large_info):
    with gr.Row():
        first = gr.Dropdown(list(small), value=None, label=small_label, info=small_info)
        second = gr.Dropdown(list(large), value=None, label=large_label, info=large_info)
    return first, second


def bind_resolution_presets(view, *, extra_outputs=(), extra_values=()):
    """Mirror Qwen: a preset sets the size and clears the other; a manual size clears both."""
    from .qwen_bindings import qwen_resolution_preset_values

    pairs = ((view.resolution_small, view.resolution_large), (view.resolution_large, view.resolution_small))
    for chosen, other in pairs:
        chosen.input(
            lambda name: ((*qwen_resolution_preset_values(name), None, *extra_values) if name
                          else tuple(gr.update() for _ in range(3 + len(extra_outputs)))),
            inputs=chosen, outputs=[view.width, view.height, other, *extra_outputs], **_QUIET,
        )
    for dimension in (view.width, view.height):
        dimension.input(lambda: (None, None), outputs=[view.resolution_small, view.resolution_large], **_QUIET)


def drama3_voice_choices(names):
    # Keep the existing API value while showing both ways to supply a reference.
    return [("Upload or generate a voice", UPLOAD_VOICE), *names]


def drama3_sound_choices(names):
    return [("Upload or generate a sound", UPLOAD_SOUND), *names]


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
    drama3_sound_names: Callable[[], list[str]] = lambda: []
    select_drama3_sound: Callable | None = None
    save_drama3_sound: Callable | None = None
    delete_drama3_sound: Callable | None = None
    generate_drama3_sound: Callable | None = None
    generate_drama3_sounds: Callable | None = None
    sync_drama3_script_sounds: Callable | None = None
    switch_drama3_script_mode: Callable | None = None
    generate_moss_image: Callable | None = None
    enhance_moss_image_prompt: Callable | None = None


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
    resolution_small: gr.Dropdown
    resolution_large: gr.Dropdown


@dataclass(frozen=True)
class MossImageView:
    prompt: gr.Textbox
    preset: gr.Radio
    turbo_variant: gr.Dropdown
    model: gr.Dropdown
    text_encoder: gr.Dropdown
    vae: gr.Dropdown
    width: gr.Slider
    height: gr.Slider
    seed: gr.Number
    steps: gr.Slider
    cfg: gr.Slider
    layout: gr.Textbox
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    rewrite_mode: gr.Radio
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    resolution_small: gr.Dropdown
    resolution_large: gr.Dropdown


def moss_image_preset_values(preset):
    """Apply a MOSS-Image preset (Turbo LoRA, steps, CFG, model, VAE) without locking the controls."""
    return MOSS_IMAGE_PRESETS[preset]


def moss_image_turbo_defaults(variant, preset=DEFAULT_MOSS_IMAGE_PRESET):
    """Turbo LoRAs fix their steps at CFG 1; Off restores the selected base preset."""
    if variant in MOSS_IMAGE_TURBO_MODES:
        return MOSS_IMAGE_TURBO_MODES[variant][1], 1.0
    base = preset if MOSS_IMAGE_PRESETS.get(preset, ("",))[0] == "Off" else "Balanced"
    return MOSS_IMAGE_PRESETS[base][1:3]


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
class Drama3SoundSlot:
    index: int
    group: gr.Accordion
    preset: gr.Dropdown
    upload: gr.Group
    audio: gr.Audio
    save_row: gr.Row
    save_name: gr.Textbox
    save: gr.Button
    delete: gr.Button
    description: gr.Textbox
    duration: gr.Slider
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
    shown_slots: gr.Number
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
    script_mode: gr.Radio | None = None
    sounds_panel: gr.Accordion | None = None
    sounds: tuple[Drama3SoundSlot, ...] = ()
    shown_sound_slots: gr.Number | None = None
    add_sound: gr.Button | None = None
    remove_sound: gr.Button | None = None
    generate_sounds: gr.Button | None = None
    stop_sounds: gr.Button | None = None
    skip_existing_sounds: gr.Checkbox | None = None
    sound_status: gr.Markdown | None = None
    auto: gr.Button | None = None
    stop_auto: gr.Button | None = None
    auto_status: gr.Markdown | None = None
    auto_sound_slots: gr.Number | None = None


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
                resolution_small, resolution_large = _resolution_presets(
                    FLUX3_1K_RESOLUTION_PRESETS, FLUX3_2K_RESOLUTION_PRESETS,
                    "1K resolution preset", "2K resolution preset",
                    "Native 1K sizes.", "Native 2K sizes, up to 4 MP. Edits use these instead of the first image size.")
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
                     writer, enhance, enhance_status, output, run, stop, status, upsampler, cfg, layout,
                     resolution_small, resolution_large)


def build_moss_image_view(root):
    turbo, steps_default, cfg_default, model_default, vae_default = MOSS_IMAGE_PRESETS[DEFAULT_MOSS_IMAGE_PRESET]
    with root:
        gr.Markdown("## MOSS-Image\nOpenMOSS MOSS-Image SFT and Base text-to-image with strong text rendering and "
                    "optional layout boxes. Weights download on first use from "
                    "[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image).")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                prompt = gr.Textbox(label="Image prompt", lines=7,
                    placeholder="Describe the scene, including any text to render in quotes.")
                with gr.Accordion("Rewrite with Lightning AI or Gemini", open=False):
                    gr.Markdown("Follows OpenMOSS's official prompt-enhancer skill. The writer returns a detailed "
                                "caption or a Layout with boxes for the width and height below; keep that size "
                                "when generating a Layout.")
                    rewrite_mode = gr.Radio(list(MOSS_IMAGE_REWRITE_MODES), value="Auto", label="Rewrite as",
                        info="Auto picks a Layout for posters, panels or placed text, otherwise a caption.")
                    writer, enhance, enhance_status = _writer("Rewrite prompt")
                preset = gr.Radio(list(MOSS_IMAGE_PRESETS), value=DEFAULT_MOSS_IMAGE_PRESET, label="MOSS-Image preset",
                    info="Fast uses INT8 SFT, the small VAE and the 8-step DMD Turbo LoRA at CFG 1. "
                         "Balanced and Quality run BF16 SFT with the full VAE for 25 or 50 steps at CFG 3.")
                resolution_small, resolution_large = _resolution_presets(
                    MOSS_IMAGE_1M_RESOLUTION_PRESETS, MOSS_IMAGE_2M_RESOLUTION_PRESETS,
                    "1M resolution preset", "2M resolution preset",
                    "About one megapixel, the default.", "About two megapixels, the top of OpenMOSS's recommended range.")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                with gr.Accordion("Model and sampling", open=False):
                    turbo_variant = gr.Dropdown(["Off", *MOSS_IMAGE_TURBO_MODES], value=turbo, label="Turbo LoRA",
                        info="A step-distilled LoRA sets its own step count and CFG 1.")
                    model = gr.Dropdown(list(MOSS_IMAGE_MODEL_CHOICES), value=model_default,
                                        label="Image model",
                                        info="SFT is the tuned release; Base is the pretrained checkpoint. "
                                             "The Turbo LoRA needs SFT.")
                    text_encoder = gr.Dropdown(list(MOSS_IMAGE_TEXT_ENCODER_CHOICES),
                                               value=DEFAULT_MOSS_IMAGE_TEXT_ENCODER, label="Qwen3.5 9B text encoder")
                    vae = gr.Dropdown(list(MOSS_IMAGE_VAE_CHOICES), value=vae_default, label="VAE",
                                      info="The full VAE is shared with FLUX 3; the small one decodes faster.")
                    steps = gr.Slider(1, 100, value=steps_default, step=1, label="Steps")
                    cfg = gr.Slider(1, 10, value=cfg_default, step=0.1, label="CFG",
                        info="Above 1, the empty negative pass skips layer 22 as in OpenMOSS's reference sampler.")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Layout boxes (optional)", open=False):
                    layout = gr.Textbox(label="Layout boxes", lines=4,
                        placeholder='[{"bbox": [100, 100, 700, 450], "desc": "a red bicycle"}]',
                        info="JSON list. bbox is [top, left, bottom, right] on a 0–1000 grid. "
                             "Use type text and text for lettering. Leave empty when the prompt is a "
                             "rewritten Layout, which carries its own boxes.")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Generated image", columns=1, interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate image")
    return MossImageView(prompt, preset, turbo_variant, model, text_encoder, vae, width, height, seed, steps, cfg,
                         layout, output, run, stop, status, rewrite_mode, writer, enhance, enhance_status,
                         resolution_small, resolution_large)


@dataclass(frozen=True)
class _SlotKind:
    """Wording and limits shared by a slot list's titles and their browser twins."""
    slot: str
    upload: str
    saved: str
    ready: str
    panel: str
    noun: str
    asset: str
    total: int


_VOICE_KIND = _SlotKind("Speaker", UPLOAD_VOICE, "saved voice", "voice ready",
                        "Speaker voices (optional)", "speaker", "voice", DRAMA3_VOICE_SLOTS)
_SOUND_KIND = _SlotKind("Sound", UPLOAD_SOUND, "saved sound", "clip ready",
                        "Sound effects and music (SGLang-Omni)", "sound", "clip", DRAMA3_SOUND_SLOTS)


def _clamp_count(count, total):
    try:
        return max(1, min(total, int(count or 1)))
    except (TypeError, ValueError):
        return 1


def _slot_title(kind, index, name, preset, has_clip):
    title = f"{kind.slot} {index}"
    if str(name or "").strip():
        title += f" \u00b7 {str(name).strip()}"
    if preset and preset != kind.upload:
        title += f" \u00b7 {kind.saved}"
    elif has_clip:
        title += f" \u00b7 {kind.ready}"
    return title


def _panel_title(kind, shown, presets, has_clips):
    shown = _clamp_count(shown, kind.total)
    ready = sum(1 for index in range(shown)
                if (index < len(presets) and presets[index] and presets[index] != kind.upload)
                or (index < len(has_clips) and has_clips[index]))
    return f"{kind.panel} \u00b7 {shown} {kind.noun}{'s' if shown != 1 else ''} \u00b7 {ready} with a {kind.asset}"


def _count_js(total):
    return f"Math.max(1, Math.min({total}, Math.trunc(Number(shown)) || 1))"


def _slot_title_js(kind, index):
    """Browser twin of ``_slot_title``; also derives the slot's has_clip flag from its audio."""
    return f"""(name, preset, audio) => {{
  const ready = Boolean(audio);
  const trimmed = String(name ?? "").trim();
  let title = {json.dumps(f"{kind.slot} {index}")};
  if (trimmed) title += " \u00b7 " + trimmed;
  if (preset && preset !== {json.dumps(kind.upload)}) title += {json.dumps(f" · {kind.saved}")};
  else if (ready) title += {json.dumps(f" · {kind.ready}")};
  return [{{__type__: "update", label: title}}, ready];
}}"""


def _panel_title_js(kind):
    """Browser twin of ``_panel_title``; inputs are the count, each preset, then each has_clip flag."""
    return f"""(shown, ...values) => {{
  const count = {_count_js(kind.total)};
  let ready = 0;
  for (let index = 0; index < count; index += 1) {{
    const preset = values[index];
    if ((preset && preset !== {json.dumps(kind.upload)}) || values[{kind.total} + index]) ready += 1;
  }}
  const label = {json.dumps(f"{kind.panel} · ")} + count + " " + {json.dumps(kind.noun)}
    + (count === 1 ? "" : "s") + " \u00b7 " + ready + {json.dumps(f" with a {kind.asset}")};
  return {{__type__: "update", label}};
}}"""


def _slot_count_js(total, step, cleared=()):
    """Browser twin of ``_slot_visibility`` for Add (+1) and Remove (-1).

    Outputs: the count, each slot group, Add, Remove, then one output per slot
    for each field group in ``cleared``. A removed slot is reset to those values
    so a hidden slot never sends anything.
    """
    return f"""(shown) => {{
  const skip = {_SKIP_JS};
  const before = {_count_js(total)};
  const count = Math.max(1, Math.min({total}, before + ({step})));
  const outputs = [count];
  for (let index = 0; index < {total}; index += 1) {{
    outputs.push((index < count) === (index < before) ? skip : {{__type__: "update", visible: index < count}});
  }}
  outputs.push({{__type__: "update", interactive: count < {total}}}, {{__type__: "update", interactive: count > 1}});
  for (const fields of {json.dumps([list(group) for group in cleared])}) {{
    for (let index = 0; index < {total}; index += 1) {{
      for (const value of fields) {{
        outputs.push(count < before && index === before - 1 ? {{__type__: "update", value}} : skip);
      }}
    }}
  }}
  return outputs;
}}"""


def drama3_slot_title(speaker, name="", preset=UPLOAD_VOICE, has_clip=False):
    """Panel title: index, character name, and whether a voice is ready."""
    return _slot_title(_VOICE_KIND, speaker, name, preset, has_clip)


def drama3_voices_title(shown=DRAMA3_VISIBLE_SLOTS, presets=(), has_clips=()):
    """Speaker voices panel title with the shown speaker count and how many have a voice."""
    return _panel_title(_VOICE_KIND, shown, presets, has_clips)


def drama3_sound_slot_title(index, name="", preset=UPLOAD_SOUND, has_clip=False):
    """Sound slot title: index, sound name, and whether a clip is ready."""
    return _slot_title(_SOUND_KIND, index, name, preset, has_clip)


def drama3_sounds_title(shown=DRAMA3_VISIBLE_SOUND_SLOTS, presets=(), has_clips=()):
    """Sound effects and music panel title with shown count and ready count."""
    return _panel_title(_SOUND_KIND, shown, presets, has_clips)


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


def _sound_slot(index, names):
    with gr.Accordion(
        drama3_sound_slot_title(index),
        open=index == 0,
        visible=index < DRAMA3_VISIBLE_SOUND_SLOTS,
    ) as group:
        with gr.Row(equal_height=True):
            preset = gr.Dropdown(
                drama3_sound_choices(names),
                value=UPLOAD_SOUND,
                label=f"Sound {index} preset",
            )
            save_name = gr.Textbox(
                label=f"Sound {index} name · <sound name=\"...\">",
                placeholder="e.g. rain, thunder, noir_piano",
            )
        description = gr.Textbox(
            label=f"Sound {index} prompt (Stable Audio 3 Medium)",
            lines=2,
            placeholder="Detailed English prompt for sound effect or instrumental background music.",
        )
        duration = gr.Slider(
            1,
            47,
            value=5,
            step=1,
            label=f"Sound {index} duration (seconds)",
            info="Use ~3–6s for one-shot SFX and ~10–15s for looping ambience or background music.",
        )
        with gr.Group() as upload:
            with gr.Row():
                generate = gr.Button("Generate sound", size="sm")
                stop = gr.Button("Interrupt sound generation", size="sm", interactive=False)
                generate.h3_stop = stop
            audio = gr.Audio(label=f"Sound {index} clip", type="filepath")
        with gr.Row() as save_row:
            save = gr.Button("Save sound", size="sm")
        delete = gr.Button("Delete saved sound", size="sm", visible=False)
        status = gr.Markdown("")
        has_clip = gr.Checkbox(False, visible=False)
    return Drama3SoundSlot(
        index,
        group,
        preset,
        upload,
        audio,
        save_row,
        save_name,
        save,
        delete,
        description,
        duration,
        generate,
        stop,
        status,
        has_clip,
    )


def build_drama3_view(root, speaker_names=(), sound_names=()):
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
                script_mode = gr.Radio(
                    list(DRAMA3_SCRIPT_MODES),
                    value=DRAMA3_SCRIPT_BASIC,
                    label="Script mode",
                    info="Basic keeps pure dialogue and vocal expression cues (automatically strips sound tags). "
                         "Advanced adds nested <parallel>/<sequential> sound effects, background music "
                         "(generated with Stable Audio 3 Medium), and speaker vocal effects (SGLang-Omni only).",
                )
                text = gr.Textbox(label="Speech text", lines=8,
                    placeholder="<|speaker:0|> [with warm excitement] Welcome back!\n<|speaker:1|> [laughs] Took you long enough.")
                with gr.Row():
                    auto = gr.Button("Auto: rewrite, voice, sound and speak", variant="primary", size="sm")
                    stop_auto = gr.Button("Interrupt auto", size="sm", interactive=False)
                auto_status = gr.Markdown(
                    "Paste the content and pick Basic or Advanced. Auto rewrites the script, generates every "
                    "speaker voice, generates every sound effect (Advanced only), then generates the speech, "
                    "using the writer and settings below.")
                auto_sound_slots = gr.Number(0, precision=0, visible=False)
                with gr.Accordion("Rewrite as a Drama 3 script", open=False):
                    gr.Markdown("Separates narration and dialogue, adds `<|speaker:N|>` tags and expression cues, "
                                "and fills matching voice slots with speaker names, descriptions and preview speech. "
                                "In Advanced mode, also stages `<parallel>`/`<sequential>` sound effects, atmosphere music, "
                                "and speaker vocal effects (`<phone_call>`, `<robotic_voice>`, etc.).")
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
                    shown_slots = gr.Number(DRAMA3_VISIBLE_SLOTS, precision=0, visible=False)
                    with gr.Row():
                        add_speaker = gr.Button("Add speaker", size="sm")
                        remove_speaker = gr.Button("Remove last speaker", size="sm")
                with gr.Accordion(drama3_sounds_title(), open=False, visible=False) as sounds_panel:
                    gr.Markdown(
                        "Manage and generate `<sound name=\"...\">` clips for Advanced mode "
                        "using [Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium). "
                        "Rewriting a script in Advanced mode populates these slots automatically. "
                        "Any `<sound>` without a pre-generated clip is generated automatically during speech synthesis."
                    )
                    skip_existing_sounds = gr.Checkbox(
                        True,
                        label="Skip sounds with a clip",
                        info="Uncheck to regenerate existing sound clips.",
                    )
                    with gr.Row():
                        generate_sounds = gr.Button("Generate all sounds", size="sm")
                        stop_sounds = gr.Button("Interrupt", size="sm", interactive=False)
                        generate_sounds.h3_stop = stop_sounds
                    sound_status = gr.Markdown("")
                    sounds = tuple(_sound_slot(i, list(sound_names)) for i in range(DRAMA3_SOUND_SLOTS))
                    shown_sound_slots = gr.Number(DRAMA3_VISIBLE_SOUND_SLOTS, precision=0, visible=False)
                    with gr.Row():
                        add_sound = gr.Button("Add sound", size="sm")
                        remove_sound = gr.Button("Remove last sound", size="sm")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                with gr.Accordion("Speech and memory settings", open=False):
                    backend = gr.Radio(list(DRAMA3_BACKENDS), value=DRAMA3_BACKENDS[0], label="Backend",
                        info="ComfyUI generates the whole text in one take and can invent voices. "
                             "SGLang-Omni (Linux) is for long scripts where every speaker has a "
                             "voice below: each speaker's lines go out with that speaker's clip, in parallel, and "
                             "near-silent takes can be retried when retries are enabled. Advanced sound-enhanced mode "
                             "works exclusively with SGLang-Omni.")
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
                        fn=None,
                        js="(choice) => [{__type__: 'update', visible: choice !== 'ComfyUI'}, "
                           "{__type__: 'update', visible: choice === 'ComfyUI'}]",
                        inputs=backend, outputs=[served_settings, comfy_settings], **_QUIET,
                    )
                sampling_controls = [temperature, top_p, top_k]
                presets_js = json.dumps(DRAMA3_SAMPLING_PRESETS)
                sampling_preset.input(fn=None, js=f"(name) => ({presets_js})[name]",
                                      inputs=sampling_preset, outputs=sampling_controls, **_QUIET)
                gr.on(
                    [control.change for control in sampling_controls], fn=None,
                    js=f"""(...values) => Object.entries({presets_js}).find(
                        ([, preset]) => preset.every((value, index) => value === values[index]))?.[0] ?? null""",
                    inputs=sampling_controls, outputs=sampling_preset, **_QUIET,
                )
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Audio(label="Generated speech", type="filepath", interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate speech")
    return Drama3View(
        text, speaker_mode, writer, enhance, enhance_status, voices, shown_slots, add_speaker,
        remove_speaker, voice_status, seed, temperature,
        top_p, top_k, max_frames, split, experts, cache, backend, parallel, retries,
        output, run, stop, status, model, asr_model, voice_engine,
        generate_voices, stop_voices, skip_existing, voices_panel, copy_prompt, writer_prompt,
        pasted_reply, apply_reply, script_mode, sounds_panel, sounds, shown_sound_slots,
        add_sound, remove_sound, generate_sounds, stop_sounds, skip_existing_sounds, sound_status,
        auto, stop_auto, auto_status, auto_sound_slots,
    )


def settings_components(view):
    """Persistable components by setting name; voice slots and nested groups excluded."""
    components = {name: value for name, value in vars(view).items()
                  if isinstance(value, gr.components.Component)}
    writer = getattr(view, "writer", None)
    if writer is not None:
        components.update(prompt_backend=writer.backend, prompt_model=writer.model,
                          lightning_model=writer.lightning_model)
    return components


def _writer_inputs(writer):
    return [writer.model, writer.gemini_api_key, writer.backend, writer.lightning_api_key, writer.lightning_model]


def _owned_view_generation(generate, family, inputs):
    import inspect

    sig_names = tuple(
        name
        for name, param in inspect.signature(generate).parameters.items()
        if name not in {"request", "progress"}
        and param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )
    input_names = sig_names[: len(inputs)] if len(sig_names) >= len(inputs) and sig_names else None
    return owned_generation(generate, family, input_names=input_names)


def _bind_generation(view, family, generate, interrupt, inputs):
    event = bind_gpu_action(view.run.click, _owned_view_generation(generate, family, inputs), inputs=inputs,
                            outputs=[view.output, view.status], api_name=f"generate_{family}",
                            show_progress="minimal")
    stopped = view.stop.click(owned_interrupt(interrupt, family), outputs=view.status, queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return event


def bind_flux3(view, actions):
    # As in Qwen, a chosen preset takes priority over matching the first reference's size.
    bind_resolution_presets(view, extra_outputs=(view.edit_size,), extra_values=(QWEN_EDIT_SIZE_MANUAL,))
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


def bind_moss_image(view, actions):
    bind_resolution_presets(view)
    bind_prompt_action(
        view.enhance.click, actions.enhance_moss_image_prompt,
        inputs=[view.prompt, view.width, view.height, view.rewrite_mode, *_writer_inputs(view.writer)],
        outputs=[view.prompt, view.enhance_status, view.layout],
        show_progress="minimal", api_name="enhance_moss_image_prompt",
    )
    view.preset.change(moss_image_preset_values, inputs=view.preset,
                       outputs=[view.turbo_variant, view.steps, view.cfg, view.model, view.vae], **_QUIET)
    view.turbo_variant.input(moss_image_turbo_defaults, inputs=[view.turbo_variant, view.preset],
                             outputs=[view.steps, view.cfg], **_QUIET)
    return _bind_generation(view, "moss_image", actions.generate_moss_image, actions.interrupt, [
        view.prompt, view.preset, view.turbo_variant, view.model, view.text_encoder, view.vae, view.width,
        view.height,
        view.seed, view.steps, view.cfg, view.layout,
    ])


def drama3_voice_inputs(view):
    return [component for slot in view.voices for component in (slot.preset, slot.audio, slot.transcript)]


def drama3_cast_inputs(view):
    return [component for slot in view.voices
            for component in (slot.save_name, slot.description, slot.preview_text)]


def drama3_sound_inputs(view):
    return [
        component
        for slot in view.sounds
        for component in (slot.preset, slot.audio, slot.save_name, slot.description, slot.duration)
    ]


def drama3_sound_meta_inputs(view):
    return [
        component
        for slot in view.sounds
        for component in (slot.save_name, slot.description, slot.duration)
    ]


def _refresh_choices_on_open(panel, presets, choices, names):
    """Show presets saved from other sessions when the panel opens; saves and deletes refresh in place."""
    def refresh():
        current = choices(names())
        # The page was served with the build-time choices; resend only when the store differs.
        values = [choice[1] if isinstance(choice, tuple) else choice for choice in current]
        if values == [choice[1] for choice in presets[0].choices]:
            return [gr.skip()] * len(presets)
        return [gr.update(choices=current)] * len(presets)

    panel.expand(refresh, outputs=presets, **_QUIET)


def _slot_visibility(count, previous=None, total=DRAMA3_VOICE_SLOTS):
    """Show ``count`` slots; disable Add at the limit and Remove at one slot.

    With the ``previous`` count only slots whose visibility changes are sent:
    each visibility update of an unrendered slot makes Gradio re-walk its layout.
    """
    count = _clamp_count(count, total)
    if previous is not None:
        previous = _clamp_count(previous, total)
        if previous == count:
            return (gr.skip(),) * (total + 3)
    groups = [gr.skip() if previous is not None and (index < count) == (index < previous)
              else gr.update(visible=index < count) for index in range(total)]
    return (count, *groups, gr.update(interactive=count < total), gr.update(interactive=count > 1))


def _sound_slot_visibility(count, previous=None):
    """Show ``count`` sound slots; disable Add at the limit and Remove at one slot."""
    return _slot_visibility(count, previous, DRAMA3_SOUND_SLOTS)


def _bind_slot_presentation(kind, slots, shown, panel, add, remove, cleared):
    """Slot titles, has_clip flags, the panel title and Add/Remove, all in the browser.

    ``cleared`` pairs a slot's field getter with the values a removed slot is reset to.
    """
    for index, slot in enumerate(slots):
        gr.on([slot.save_name.change, slot.preset.change, slot.audio.change], fn=None,
              js=_slot_title_js(kind, index), inputs=[slot.save_name, slot.preset, slot.audio],
              outputs=[slot.group, slot.has_clip], **_QUIET)
    presets = [slot.preset for slot in slots]
    flags = [slot.has_clip for slot in slots]
    gr.on([shown.change, *[preset.change for preset in presets], *[flag.change for flag in flags]],
          fn=None, js=_panel_title_js(kind), inputs=[shown, *presets, *flags], outputs=panel, **_QUIET)
    groups = [slot.group for slot in slots]
    add.click(fn=None, js=_slot_count_js(kind.total, 1), inputs=shown,
              outputs=[shown, *groups, add, remove], **_QUIET)
    fields = [component for fields_of, _values in cleared for slot in slots for component in fields_of(slot)]
    remove.click(fn=None, js=_slot_count_js(kind.total, -1, [values for _fields_of, values in cleared]),
                 inputs=shown, outputs=[shown, *groups, add, remove, *fields], **_QUIET)


def _bind_script_mode(view, actions):
    bind_gpu_action(
        view.backend.input, actions.switch_drama3_backend,
        inputs=view.backend, outputs=view.status,
        show_progress="minimal", api_name=False,
    )
    if view.script_mode is None or actions.switch_drama3_script_mode is None:
        return
    view.script_mode.change(
        actions.switch_drama3_script_mode,
        inputs=[view.script_mode, view.text, view.backend],
        outputs=[view.text, view.backend, view.sounds_panel, view.status], **_QUIET,
    )
    # Advanced needs SGLang-Omni; choosing ComfyUI falls back to Basic, whose
    # change event hides the sounds panel.
    view.backend.input(
        fn=None,
        js=f"(choice, mode) => choice === 'ComfyUI' && mode === {json.dumps(DRAMA3_SCRIPT_ADVANCED)} "
           f"? {json.dumps(DRAMA3_SCRIPT_BASIC)} : {_SKIP_JS}",
        inputs=[view.backend, view.script_mode], outputs=view.script_mode, **_QUIET,
    )


def _writer_slot_inputs(view):
    mode_inputs = [view.script_mode] if view.script_mode is not None else []
    return [*drama3_voice_inputs(view), *drama3_cast_inputs(view), view.shown_slots, *mode_inputs]


def _writer_outputs(view):
    return [view.text, view.enhance_status, *drama3_cast_inputs(view),
            view.shown_slots, *[slot.group for slot in view.voices],
            view.add_speaker, view.remove_speaker]


def _bind_writer(view, actions):
    """Rewrite, copy and paste; returns the events that may change the script."""
    writer_slots = _writer_slot_inputs(view)
    writer_outputs = _writer_outputs(view)
    view.copy_prompt.click(actions.copy_drama3_writer_prompt,
                           inputs=[view.text, view.speaker_mode, *writer_slots],
                           outputs=[view.writer_prompt, view.enhance_status], **_QUIET)
    applied = view.apply_reply.click(
        actions.apply_drama3_writer_reply,
        inputs=[view.pasted_reply, view.text, view.speaker_mode, *writer_slots],
        outputs=writer_outputs, **_QUIET,
    )
    enhanced = bind_prompt_action(
        view.enhance.click, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *_writer_inputs(view.writer), *writer_slots],
        outputs=writer_outputs, show_progress="minimal", api_name="enhance_drama3_text",
    )
    return enhanced, applied


def _voice_settings(view):
    return [view.voice_engine, view.backend, view.seed, view.temperature, view.top_p, view.top_k,
            view.max_frames, view.experts, view.expert_cache_gb, view.model]


def _bind_voice_batch(trigger, view, actions):
    return bind_gpu_action(
        trigger, owned_generation(actions.generate_drama3_voices, "drama3_voice"),
        inputs=[*_voice_settings(view), view.parallel, view.shown_slots, view.skip_existing,
                *[component for slot in view.voices
                  for component in (slot.preset, slot.has_clip, slot.description, slot.preview_text, slot.language)]],
        outputs=[*[component for slot in view.voices for component in (slot.audio, slot.transcript, slot.status)],
                 view.voice_status],
        show_progress="minimal", api_name=False,
    )


def _bind_voice_slots(view, actions):
    presets = [slot.preset for slot in view.voices]
    _bind_slot_presentation(
        _VOICE_KIND, view.voices, view.shown_slots, view.voices_panel, view.add_speaker, view.remove_speaker,
        cleared=[(lambda slot: (slot.preset, slot.audio, slot.transcript), (UPLOAD_VOICE, None, "")),
                 (lambda slot: (slot.save_name, slot.description, slot.preview_text), ("", "", ""))],
    )
    _refresh_choices_on_open(view.voices_panel, presets, drama3_voice_choices, actions.drama3_speaker_names)
    voice_settings = _voice_settings(view)
    generated_all = _bind_voice_batch(view.generate_voices.click, view, actions)
    stopped_all = view.stop_voices.click(owned_interrupt(actions.interrupt, "drama3_voice"),
                                         outputs=view.voice_status, queue=False, api_name=False)
    stopped_all.then(fn=None, cancels=[generated_all], queue=False, api_name=False)
    view.voice_engine.change(
        fn=None,
        js=f"(engine) => Array.from({{length: {len(view.voices)}}}, "
           f"() => ({{__type__: 'update', visible: engine === {json.dumps(ENGINE_QWEN)}}}))",
        inputs=view.voice_engine, outputs=[slot.language for slot in view.voices], **_QUIET,
    )
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
        # A new upload needs its own transcript.
        slot.audio.upload(fn=None, js="() => ({__type__: 'update', value: '', interactive: true})",
                          outputs=slot.transcript, **_QUIET)
        bind_gpu_action(
            slot.transcribe.click, actions.transcribe_drama3_voice, inputs=[slot.audio, view.asr_model],
            outputs=[slot.transcript, slot.status], show_progress="minimal",
            api_name="transcribe_drama3_voice" if slot.speaker == 0 else False,
        )
        slot.preset.change(
            actions.select_drama3_speaker, inputs=[slot.preset],
            outputs=[slot.upload, slot.transcript, slot.save_row, slot.delete, slot.status,
                     slot.save_name, slot.description, slot.preview_text, slot.audio], **_QUIET,
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


def _sync_script_sounds(trigger, view, actions):
    return trigger(
        actions.sync_drama3_script_sounds,
        inputs=[view.text, view.script_mode, view.shown_sound_slots, *drama3_sound_meta_inputs(view)],
        outputs=[*drama3_sound_meta_inputs(view), view.shown_sound_slots,
                 *[slot.group for slot in view.sounds], view.add_sound, view.remove_sound],
        **_QUIET,
    )


def _bind_sound_batch(trigger, view, actions, shown):
    return bind_gpu_action(
        trigger, owned_generation(actions.generate_drama3_sounds, "drama3_sound"),
        inputs=[shown, view.skip_existing_sounds,
                *[component for slot in view.sounds
                  for component in (slot.preset, slot.has_clip, slot.save_name, slot.description,
                                    slot.duration)]],
        outputs=[*[component for slot in view.sounds for component in (slot.audio, slot.status)],
                 view.sound_status],
        show_progress="minimal", api_name=False,
    )


def _bind_sound_slots(view, actions, script_events):
    presets = [slot.preset for slot in view.sounds]
    _bind_slot_presentation(
        _SOUND_KIND, view.sounds, view.shown_sound_slots, view.sounds_panel, view.add_sound, view.remove_sound,
        cleared=[(lambda slot: (slot.preset, slot.audio, slot.save_name, slot.description, slot.duration),
                  (UPLOAD_SOUND, None, "", "", 5))],
    )
    _refresh_choices_on_open(view.sounds_panel, presets, drama3_sound_choices, actions.drama3_sound_names)
    if actions.sync_drama3_script_sounds is not None:
        for event in script_events:
            _sync_script_sounds(event.then, view, actions)
    if actions.generate_drama3_sounds is not None:
        generated_all = _bind_sound_batch(view.generate_sounds.click, view, actions, view.shown_sound_slots)
        stopped_all = view.stop_sounds.click(owned_interrupt(actions.interrupt, "drama3_sound"),
                                             outputs=view.sound_status, queue=False, api_name=False)
        stopped_all.then(fn=None, cancels=[generated_all], queue=False, api_name=False)
    for slot in view.sounds:
        if actions.generate_drama3_sound is not None:
            generated = bind_gpu_action(
                slot.generate.click, owned_generation(actions.generate_drama3_sound, "drama3_sound"),
                inputs=[slot.description, slot.duration], outputs=[slot.audio, slot.status],
                show_progress="minimal", api_name="generate_drama3_sound" if slot.index == 0 else False,
            )
            stopped = slot.stop.click(owned_interrupt(actions.interrupt, "drama3_sound"),
                                      outputs=slot.status, queue=False, api_name=False)
            stopped.then(fn=None, cancels=[generated], queue=False, api_name=False)
        if actions.select_drama3_sound is not None:
            slot.preset.change(
                actions.select_drama3_sound, inputs=[slot.preset],
                outputs=[slot.upload, slot.save_row, slot.delete, slot.status, slot.save_name,
                         slot.description, slot.duration, slot.audio], **_QUIET,
            )
        if actions.save_drama3_sound is not None:
            slot.save.click(
                lambda name, audio, description, duration, index=slot.index: actions.save_drama3_sound(
                    index, name, audio, description, duration, len(presets)),
                inputs=[slot.save_name, slot.audio, slot.description, slot.duration],
                outputs=[slot.status, *presets], queue=False, api_name=False,
            )
        if actions.delete_drama3_sound is not None:
            slot.delete.click(
                lambda name, index=slot.index: actions.delete_drama3_sound(index, name, len(presets)),
                inputs=[slot.preset], outputs=[slot.status, *presets], queue=False, api_name=False,
            )


def bind_drama3(view, actions):
    _bind_script_mode(view, actions)
    script_events = _bind_writer(view, actions)
    _bind_voice_slots(view, actions)
    if view.sounds:
        _bind_sound_slots(view, actions, script_events)
    generation_inputs = [
        view.text, *drama3_voice_inputs(view), view.seed, view.temperature, view.top_p, view.top_k,
        view.max_frames, view.split_long_text, view.experts, view.expert_cache_gb,
        view.backend, view.parallel, view.retries, view.model,
    ]
    if view.script_mode is not None:
        generation_inputs.append(view.script_mode)
    if view.sounds:
        generation_inputs.extend(drama3_sound_inputs(view))
    event = _bind_generation(view, "drama3", actions.generate_drama3, actions.interrupt, generation_inputs)
    _bind_auto(view, actions, generation_inputs)
    return event


def drama3_auto_rewritten(status):
    """Stop Auto when the rewrite failed, keeping the writer's reason."""
    status = str(status or "")
    if status.startswith(DRAMA3_PROMPT_FAILED):
        raise gr.Error(f"Auto stopped: {status}")
    return "Auto: script rewritten. Generating speaker voices\u2026"


def drama3_auto_sound_slots(script_mode, shown_sound_slots):
    """Advanced generates the shown sound slots; Basic has no sounds, so none are generated."""
    if script_mode != DRAMA3_SCRIPT_ADVANCED:
        return 0, "Auto: voices done. Generating speech\u2026"
    return (_clamp_count(shown_sound_slots, DRAMA3_SOUND_SLOTS),
            "Auto: voices done. Generating sound effects\u2026")


def _bind_auto(view, actions, generation_inputs):
    """Auto: rewrite the script, generate every speaker voice, every sound, then the speech.

    Each step reuses its button's callback and job family and starts only after
    the previous step succeeds; Interrupt auto stops the running job and the chain.
    """
    if view.auto is None:
        return
    # Auto status, Interrupt auto, then the Auto button itself, which reads "running" until the chain ends.
    controls = [view.auto_status, view.stop_auto, view.auto]
    idle = (gr.update(interactive=False), gr.update(value=view.auto.value, interactive=True))
    started = view.auto.click(
        lambda: ("Auto: rewriting the script\u2026", gr.update(interactive=True),
                 gr.update(value="Auto: running\u2026", interactive=False)),
        outputs=controls, **_QUIET,
    )
    rewritten = bind_prompt_action(
        started.then, actions.enhance_drama3_text,
        inputs=[view.text, view.speaker_mode, *_writer_inputs(view.writer), *_writer_slot_inputs(view)],
        outputs=_writer_outputs(view), show_progress="minimal", api_name=False,
    )
    checked = rewritten.success(drama3_auto_rewritten, inputs=view.enhance_status, outputs=view.auto_status,
                                **_QUIET)
    steps = [rewritten, checked]
    if view.sounds and actions.sync_drama3_script_sounds is not None:
        checked = _sync_script_sounds(checked.success, view, actions)
        steps.append(checked)
    voiced = _bind_voice_batch(checked.success, view, actions)
    steps.append(voiced)
    if view.sounds and actions.generate_drama3_sounds is not None:
        counted = voiced.success(drama3_auto_sound_slots, inputs=[view.script_mode, view.shown_sound_slots],
                                 outputs=[view.auto_sound_slots, view.auto_status], **_QUIET)
        sounded = _bind_sound_batch(counted.success, view, actions, view.auto_sound_slots)
        ready = sounded.success(lambda: "Auto: generating speech\u2026", outputs=view.auto_status, **_QUIET)
        steps += [counted, sounded, ready]
    else:
        ready = voiced.success(lambda: "Auto: voices done. Generating speech\u2026",
                               outputs=view.auto_status, **_QUIET)
        steps.append(ready)
    spoken = bind_gpu_action(
        ready.success, _owned_view_generation(actions.generate_drama3, "drama3", generation_inputs),
        inputs=generation_inputs, outputs=[view.output, view.status], show_progress="minimal",
        api_name=False,
    )
    steps.append(spoken)
    spoken.success(lambda: ("Auto: done.", *idle), outputs=controls, **_QUIET)
    for step in steps:
        step.failure(lambda: ("Auto stopped. See the step's status for the reason.", *idle),
                     outputs=controls, **_QUIET)

    def stop_auto(request: gr.Request):
        for family in ("drama3_voice", "drama3_sound", "drama3"):
            actions.interrupt(request, family=family)
        return "Auto interrupted.", *idle

    stopped = view.stop_auto.click(stop_auto, outputs=controls, queue=False, api_name=False)
    stopped.then(fn=None, cancels=steps, queue=False, api_name=False)
