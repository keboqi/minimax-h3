"""FLUX 3 Image Klein in Create: view, bindings and handlers."""

from dataclasses import dataclass, replace
from pathlib import Path

import gradio as gr

from h3_app.prompt_writers import flux3 as flux3_writer
from h3_app.catalog import (
    DEFAULT_LIGHTNING_PROMPT_MODEL, QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX,
)
from h3_app.generation.flux3 import generate_flux3, validate_flux3
from h3_app.generation.requests import Flux3Request
from h3_app.workflows.flux3 import flux3_sampling_defaults
from h3_models import DEFAULT_FLUX3_UPSAMPLER, FLUX3_MODEL_CHOICES, FLUX3_UPSAMPLER_CHOICES

from ..image_library import build_image_file_input
from ..job_bindings import bind_prompt_action
from ..prompt_writer_controls import RemotePromptWriterControls
from .controls import (
    bind_generation, bind_resolution_presets, generation_actions, prompt_writer, resolution_presets, writer_inputs,
)
from .feature import EngineDeps, EngineFeature

FLUX3_1K_RESOLUTION_PRESETS = (
    "1K · 1:1 · 1024×1024", "1K · 4:3 · 1152×864", "1K · 3:2 · 1248×832", "1K · 16:9 · 1360×768",
    "1K · 3:4 · 864×1152", "1K · 2:3 · 832×1248", "1K · 9:16 · 768×1360",
)


FLUX3_2K_RESOLUTION_PRESETS = (
    "2K · 1:1 · 2048×2048", "2K · 4:3 · 2304×1728", "2K · 3:2 · 2448×1632", "2K · 16:9 · 2720×1536",
    "2K · 3:4 · 1728×2304", "2K · 2:3 · 1632×2448", "2K · 9:16 · 1536×2720",
)


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


class Flux3Handlers:
    """Generation and rewriting; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_flux3(self, prompt, model_choice, reference_images, width, height, edit_size, seed, steps,
                       expand_prompt, upsampler_choice="INT8 ConvRot", cfg=None, layout="",
                       progress=gr.Progress(track_tqdm=False)):
        paths = reference_images or []
        if isinstance(paths, (str, Path)):
            paths = [paths]
        request = Flux3Request(
            prompt, model_choice, tuple(str(path) for path in paths), width, height, edit_size, seed, steps,
            expand_prompt, upsampler_choice, cfg, layout,
        )
        validate_flux3(request)
        request = replace(request, reference_images=tuple(
            self.deps.stage_file(path, "flux3_references", reuse=True) for path in request.reference_images
        ))
        yield from generate_flux3(request, self.deps.generation_services(), self.deps.runtime_config(),
                                  progress=progress)

    def enhance_flux3_prompt(self, prompt, reference_images, width, height, model, temporary_api_key,
                             backend="Gemini", lightning_api_key="",
                             lightning_model=DEFAULT_LIGHTNING_PROMPT_MODEL):
        """Rewrite with the official upsampler prompt; turn local expansion off on success."""
        enhanced, status = flux3_writer.enhance_flux3_prompt(
            prompt, model, temporary_api_key, reference_images, width, height,
            backend=backend, lightning_api_key=lightning_api_key,
            lightning_model=lightning_model, runtime=self.deps.runtime_config(),
        )
        failed = status.startswith("Prompt enhancement failed:")
        return enhanced, status, gr.skip() if failed else False


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
                    writer, enhance, enhance_status = prompt_writer("Rewrite prompt")
                resolution_small, resolution_large = resolution_presets(
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
                run, stop, status = generation_actions("Generate image")
    return Flux3View(prompt, model, refs, width, height, edit_size, seed, steps, expand,
                     writer, enhance, enhance_status, output, run, stop, status, upsampler, cfg, layout,
                     resolution_small, resolution_large)


def bind_flux3(view, deps: EngineDeps) -> Flux3Handlers:
    handlers = Flux3Handlers(deps)
    # As in Qwen, a chosen preset takes priority over matching the first reference's size.
    bind_resolution_presets(view, extra_outputs=(view.edit_size,), extra_values=(QWEN_EDIT_SIZE_MANUAL,))
    bind_prompt_action(
        view.enhance.click, handlers.enhance_flux3_prompt,
        inputs=[view.prompt, view.reference_images, view.width, view.height, *writer_inputs(view.writer)],
        outputs=[view.prompt, view.enhance_status, view.expand_prompt],
        show_progress="minimal", api_name="enhance_flux3_prompt",
    )
    bind_generation(view, "flux3", handlers.generate_flux3, deps.interrupt, [
        view.prompt, view.model, view.reference_images, view.width, view.height,
        view.edit_size, view.seed, view.steps, view.expand_prompt, view.upsampler, view.cfg, view.layout,
    ])
    return handlers


FEATURE = EngineFeature("flux3", lambda root, deps: build_flux3_view(root), bind_flux3, library_label="FLUX 3",
                        library_inputs=lambda view: (view.reference_images,))
