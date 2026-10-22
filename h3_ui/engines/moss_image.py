"""MOSS-Image in Create: view, presets, bindings and handlers."""

from dataclasses import dataclass

import gradio as gr

from h3_app import prompt_service
from h3_app.catalog import DEFAULT_LIGHTNING_PROMPT_MODEL, DEFAULT_MOSS_IMAGE_PRESET, MOSS_IMAGE_PRESETS
from h3_app.generation.moss_image import generate_moss_image
from h3_app.generation.requests import MossImageRequest
from h3_app.workflows.moss_image_prompt import MOSS_IMAGE_REWRITE_MODES, is_layout_prompt
from h3_models import (
    DEFAULT_MOSS_IMAGE_TEXT_ENCODER, MOSS_IMAGE_MODEL_CHOICES, MOSS_IMAGE_TEXT_ENCODER_CHOICES,
    MOSS_IMAGE_TURBO_MODES, MOSS_IMAGE_VAE_CHOICES, moss_image_turbo_steps,
)

from ..job_bindings import bind_prompt_action
from ..prompt_writer_controls import RemotePromptWriterControls
from .controls import (
    QUIET, bind_generation, bind_resolution_presets, generation_actions, prompt_writer, resolution_presets,
    writer_inputs,
)
from .feature import EngineDeps, EngineFeature
from .flux3 import FLUX3_1K_RESOLUTION_PRESETS

MOSS_IMAGE_1M_RESOLUTION_PRESETS = tuple(name.replace("1K", "1M", 1) for name in FLUX3_1K_RESOLUTION_PRESETS)


MOSS_IMAGE_2M_RESOLUTION_PRESETS = (
    "2M · 1:1 · 1456×1456", "2M · 4:3 · 1664×1248", "2M · 3:2 · 1776×1184", "2M · 16:9 · 1920×1088",
    "2M · 3:4 · 1248×1664", "2M · 2:3 · 1184×1776", "2M · 9:16 · 1088×1920",
)


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


class MossImageHandlers:
    """Generation and rewriting; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_moss_image(self, prompt, preset, turbo_variant, model_choice, text_encoder_choice, vae_choice,
                            width, height, seed, steps, cfg, layout="", accelerate=True,
                            progress=gr.Progress(track_tqdm=False)):
        request = MossImageRequest(
            prompt, preset, turbo_variant, model_choice, text_encoder_choice, vae_choice, width, height, seed,
            steps, cfg, layout or "", accelerate,
        )
        yield from generate_moss_image(request, self.deps.generation_services(), self.deps.runtime_config(),
                                       progress=progress)

    def enhance_moss_image_prompt(self, prompt, width, height, rewrite_mode, model, temporary_api_key,
                                  backend="Gemini", lightning_api_key="",
                                  lightning_model=DEFAULT_LIGHTNING_PROMPT_MODEL):
        """Rewrite with OpenMOSS's prompt-enhancer recipe; a Layout clears the manual boxes."""
        enhanced, status = prompt_service.enhance_moss_image_prompt(
            prompt, model, temporary_api_key, width, height, rewrite_mode,
            backend=backend, lightning_api_key=lightning_api_key,
            lightning_model=lightning_model, runtime=self.deps.runtime_config(),
        )
        failed = status.startswith("Prompt enhancement failed:")
        return enhanced, status, "" if not failed and is_layout_prompt(enhanced) else gr.skip()


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
                    writer, enhance, enhance_status = prompt_writer("Rewrite prompt")
                preset = gr.Radio(list(MOSS_IMAGE_PRESETS), value=DEFAULT_MOSS_IMAGE_PRESET, label="MOSS-Image preset",
                    info="Fast uses INT8 SFT, the small VAE and the 8-step DMD Turbo LoRA at CFG 1. "
                         "Balanced and Quality run BF16 SFT with the full VAE for 25 or 50 steps at CFG 3.")
                accelerate = gr.Checkbox(value=True, label="Accelerate with Spectrum",
                    info="Adaptive hidden-state prediction. Uses Euler + sgm_uniform; may change fine details. "
                         "Automatically bypassed for Turbo LoRAs and runs of 10 steps or fewer.")
                resolution_small, resolution_large = resolution_presets(
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
                run, stop, status = generation_actions("Generate image")
    return MossImageView(prompt, preset, turbo_variant, model, text_encoder, vae, width, height, seed, steps, cfg,
                         layout, output, run, stop, status, rewrite_mode, writer, enhance, enhance_status,
                         resolution_small, resolution_large, accelerate)


def bind_moss_image(view, deps: EngineDeps) -> MossImageHandlers:
    handlers = MossImageHandlers(deps)
    bind_resolution_presets(view)
    bind_prompt_action(
        view.enhance.click, handlers.enhance_moss_image_prompt,
        inputs=[view.prompt, view.width, view.height, view.rewrite_mode, *writer_inputs(view.writer)],
        outputs=[view.prompt, view.enhance_status, view.layout],
        show_progress="minimal", api_name="enhance_moss_image_prompt",
    )
    view.preset.change(moss_image_preset_values, inputs=view.preset,
                       outputs=[view.turbo_variant, view.steps, view.cfg, view.model, view.vae], **QUIET)
    view.turbo_variant.input(moss_image_turbo_defaults,
                             inputs=[view.turbo_variant, view.preset, view.width, view.height],
                             outputs=[view.steps, view.cfg], **QUIET)
    for size in (view.width, view.height):
        size.change(moss_image_turbo_size_steps, inputs=[view.turbo_variant, view.steps, view.width, view.height],
                    outputs=view.steps, **QUIET)
    bind_generation(view, "moss_image", handlers.generate_moss_image, deps.interrupt, [
        view.prompt, view.preset, view.turbo_variant, view.model, view.text_encoder, view.vae, view.width,
        view.height,
        view.seed, view.steps, view.cfg, view.layout, view.accelerate,
    ])
    return handlers


FEATURE = EngineFeature("moss_image", lambda root, deps: build_moss_image_view(root), bind_moss_image)
