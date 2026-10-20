"""FLUX 3 and MOSS-Image views, bindings and optional-engine action contract."""

from dataclasses import dataclass
from typing import Callable
import gradio as gr
from h3_app.catalog import DEFAULT_MOSS_IMAGE_PRESET, MOSS_IMAGE_PRESETS, QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.workflows.flux3 import flux3_sampling_defaults
from h3_app.workflows.moss_image_prompt import MOSS_IMAGE_REWRITE_MODES
from h3_models import DEFAULT_FLUX3_UPSAMPLER, FLUX3_MODEL_CHOICES, FLUX3_UPSAMPLER_CHOICES, DEFAULT_MOSS_IMAGE_TEXT_ENCODER, MOSS_IMAGE_MODEL_CHOICES, MOSS_IMAGE_TEXT_ENCODER_CHOICES, MOSS_IMAGE_TURBO_MODES, MOSS_IMAGE_VAE_CHOICES, moss_image_turbo_steps
from .image_library import build_image_file_input
from .job_bindings import bind_prompt_action
from .prompt_writer_controls import RemotePromptWriterControls
from h3_ui.optional_engine_controls import (
    _QUIET,
    _actions,
    _bind_generation,
    _writer,
    _writer_inputs,
)


FLUX3_1K_RESOLUTION_PRESETS = (
    "1K · 1:1 · 1024×1024", "1K · 4:3 · 1152×864", "1K · 3:2 · 1248×832", "1K · 16:9 · 1360×768",
    "1K · 3:4 · 864×1152", "1K · 2:3 · 832×1248", "1K · 9:16 · 768×1360",
)


FLUX3_2K_RESOLUTION_PRESETS = (
    "2K · 1:1 · 2048×2048", "2K · 4:3 · 2304×1728", "2K · 3:2 · 2448×1632", "2K · 16:9 · 2720×1536",
    "2K · 3:4 · 1728×2304", "2K · 2:3 · 1632×2448", "2K · 9:16 · 1536×2720",
)


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
    generate_drama3_voice: Callable
    generate_drama3_voices: Callable
    copy_drama3_writer_prompt: Callable
    apply_drama3_writer_reply: Callable
    speaker_library_actions: object | None = None
    generate_drama3_voice_takes: Callable | None = None
    drama3_speaker_groups: Callable = lambda: ["Global"]
    drama3_speaker_group_size: Callable | None = None
    create_drama3_speaker_group: Callable | None = None
    match_drama3_speaker: Callable | None = None
    match_drama3_speakers: Callable | None = None
    save_all_drama3_speakers: Callable | None = None
    drama3_sound_names: Callable[[], list[str]] = lambda: []
    select_drama3_sound: Callable | None = None
    save_drama3_sound: Callable | None = None
    delete_drama3_sound: Callable | None = None
    generate_drama3_sound: Callable | None = None
    generate_drama3_sounds: Callable | None = None
    sync_drama3_script_sounds: Callable | None = None
    switch_drama3_script_mode: Callable | None = None
    regenerate_drama3_line: Callable | None = None
    remix_drama3: Callable | None = None
    drama3_takes: Callable | None = None
    drama3_take_selected: Callable | None = None
    drama3_take_audio: Callable | None = None
    drama3_subtitles: Callable | None = None
    generate_moss_image: Callable | None = None
    enhance_moss_image_prompt: Callable | None = None
    generate_fibo2: Callable | None = None


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
    accelerate: gr.Checkbox


def moss_image_preset_values(preset):
    """Apply a MOSS-Image preset (Turbo LoRA, steps, CFG, model, VAE) without locking the controls."""
    return MOSS_IMAGE_PRESETS[preset]


def moss_image_turbo_defaults(variant, preset=DEFAULT_MOSS_IMAGE_PRESET, width=1024, height=1024):
    """Turbo LoRAs fix their steps for the output size at CFG 1; Off restores the selected base preset."""
    if variant in MOSS_IMAGE_TURBO_MODES:
        return moss_image_turbo_steps(variant, width, height), 1.0
    base = preset if MOSS_IMAGE_PRESETS.get(preset, ("",))[0] == "Off" else "Balanced"
    return MOSS_IMAGE_PRESETS[base][1:3]


def moss_image_turbo_size_steps(variant, steps, width, height):
    """Resizing keeps a Turbo LoRA on the step count it was distilled for at that size."""
    return moss_image_turbo_steps(variant, width, height) if variant in MOSS_IMAGE_TURBO_MODES else steps


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
                accelerate = gr.Checkbox(value=True, label="Accelerate with Spectrum",
                    info="Adaptive hidden-state prediction. Uses Euler + sgm_uniform; may change fine details. "
                         "Automatically bypassed for Turbo LoRAs and runs of 10 steps or fewer.")
                resolution_small, resolution_large = _resolution_presets(
                    MOSS_IMAGE_1M_RESOLUTION_PRESETS, MOSS_IMAGE_2M_RESOLUTION_PRESETS,
                    "1M resolution preset", "2M resolution preset",
                    "About one megapixel, the default.", "About two megapixels, the top of OpenMOSS's recommended range.")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                with gr.Accordion("Model and sampling", open=False):
                    turbo_variant = gr.Dropdown(["Off", *MOSS_IMAGE_TURBO_MODES], value=turbo, label="Turbo LoRA",
                        info="A step-distilled LoRA sets its own step count and CFG 1. "
                             "The 4-step LoRA uses 6 steps up to 1536x1024 and 8 above.")
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
                         resolution_small, resolution_large, accelerate)


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
    view.turbo_variant.input(moss_image_turbo_defaults,
                             inputs=[view.turbo_variant, view.preset, view.width, view.height],
                             outputs=[view.steps, view.cfg], **_QUIET)
    for size in (view.width, view.height):
        size.change(moss_image_turbo_size_steps, inputs=[view.turbo_variant, view.steps, view.width, view.height],
                    outputs=view.steps, **_QUIET)
    return _bind_generation(view, "moss_image", actions.generate_moss_image, actions.interrupt, [
        view.prompt, view.preset, view.turbo_variant, view.model, view.text_encoder, view.vae, view.width,
        view.height,
        view.seed, view.steps, view.cfg, view.layout, view.accelerate,
    ])
