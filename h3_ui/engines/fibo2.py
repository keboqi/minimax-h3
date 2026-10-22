"""Fibo 2.0 in Create: view, bindings and handlers."""

from dataclasses import dataclass, replace
from pathlib import Path

import gradio as gr

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
from h3_app.generation.fibo2 import generate_fibo2, validate_fibo2
from h3_app.generation.requests import Fibo2Request
from h3_app.workflows.fibo2 import (
    FIBO2_1M_RESOLUTION_PRESETS,
    FIBO2_2M_RESOLUTION_PRESETS,
    FIBO2_EDIT_SIZE_TRAINED,
    FIBO2_EDIT_SIZES,
    FIBO2_MODELS,
)
from h3_models import DEFAULT_FIBO2_PRECISION, FIBO2_MODEL_CHOICES

from ..image_library import build_image_file_input
from .controls import QUIET, bind_generation, bind_resolution_presets, generation_actions, resolution_presets
from .feature import EngineDeps, EngineFeature


@dataclass(frozen=True)
class Fibo2View:
    prompt: gr.Textbox
    model: gr.Dropdown
    precision: gr.Dropdown
    reference_images: gr.File
    width: gr.Slider
    height: gr.Slider
    edit_size: gr.Radio
    seed: gr.Number
    steps: gr.Slider
    cfg: gr.Slider
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    resolution_small: gr.Dropdown
    resolution_large: gr.Dropdown


def fibo2_sampling_defaults(model):
    turbo = model == "Edit Turbo"
    return gr.update(value=4 if turbo else 30, interactive=not turbo), gr.update(value=1 if turbo else 5, interactive=not turbo)


class Fibo2Handlers:
    """Generation; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps

    def generate_fibo2(self, prompt, model_choice, precision, reference_images, width, height, edit_size, seed,
                       steps, cfg, progress=gr.Progress(track_tqdm=False)):
        paths = reference_images or []
        if isinstance(paths, (str, Path)):
            paths = [paths]
        request = Fibo2Request(
            prompt, model_choice, precision, tuple(str(path) for path in paths), width, height, edit_size,
            seed, steps, cfg,
        )
        validate_fibo2(request)
        request = replace(request, reference_images=tuple(
            self.deps.stage_file(path, "fibo2_references", reuse=True) for path in request.reference_images
        ))
        yield from generate_fibo2(request, self.deps.generation_services(), self.deps.runtime_config(),
                                  progress=progress)


def build_fibo2_view(root):
    with root:
        gr.Markdown("## Fibo 2.0\nDev generates and edits images; Edit Turbo edits in 4 steps. "
                    "Weights download from [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0) "
                    "on first use, with no access request. The weights are for noncommercial use.")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                with gr.Row():
                    model = gr.Dropdown(list(FIBO2_MODELS), value="Dev", label="Fibo model")
                    precision = gr.Dropdown(list(FIBO2_MODEL_CHOICES["Dev"]), value=DEFAULT_FIBO2_PRECISION,
                                            label="Precision")
                prompt = gr.Textbox(label="Image prompt / edit instruction / VGL JSON", lines=8,
                    placeholder="Describe a scene, paste a structured caption, or describe an edit.",
                    info="Structured VGL JSON gives detailed control. Edits also accept plain instructions. "
                         "Refer to images in order as <image_1>, <image_2>, etc.")
                refs = build_image_file_input(label="Reference images (optional for Dev, up to 4)",
                                               file_count="multiple", file_types=["image"], type="filepath")
                resolution_small, resolution_large = resolution_presets(
                    FIBO2_1M_RESOLUTION_PRESETS, FIBO2_2M_RESOLUTION_PRESETS,
                    "1M resolution preset", "2M resolution preset",
                    "Bria's trained ~1 MP sizes, recommended.",
                    "About two megapixels, above the trained size; quality may vary. "
                    "Edits use these instead of the edit output size.")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                edit_size = gr.Radio(list(FIBO2_EDIT_SIZES), value=FIBO2_EDIT_SIZE_TRAINED,
                    label="Edit output size", visible=False,
                    info="Keep the first reference's aspect ratio at Bria's trained ~1 MP size, at its own size "
                         "(capped at 2 MP), or scaled up to 2 MP; or use the width and height above. "
                         "A single reference is center-cropped to the output size. Applies only to edits.")
                refs.change(lambda paths: gr.update(visible=bool(paths)), inputs=refs, outputs=edit_size,
                            queue=False, api_name=False)
                with gr.Accordion("Sampling", open=False):
                    steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                    cfg = gr.Slider(1, 20, value=5, step=0.1, label="CFG")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Generated image", columns=1, interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = generation_actions("Generate / edit image")
    return Fibo2View(prompt, model, precision, refs, width, height, edit_size, seed, steps, cfg,
                     output, run, stop, status, resolution_small, resolution_large)


def bind_fibo2(view, deps: EngineDeps) -> Fibo2Handlers:
    handlers = Fibo2Handlers(deps)
    # As in FLUX 3, a chosen preset takes priority over the edit output size.
    bind_resolution_presets(view, extra_outputs=(view.edit_size,), extra_values=(QWEN_EDIT_SIZE_MANUAL,))
    view.model.change(fibo2_sampling_defaults, inputs=view.model, outputs=[view.steps, view.cfg], **QUIET)
    bind_generation(view, "fibo2", handlers.generate_fibo2, deps.interrupt, [
        view.prompt, view.model, view.precision, view.reference_images, view.width, view.height,
        view.edit_size, view.seed, view.steps, view.cfg,
    ])
    return handlers


FEATURE = EngineFeature("fibo2", build_fibo2_view, bind_fibo2, library_label="Fibo 2.0",
                        library_inputs=lambda view: (view.reference_images,))
