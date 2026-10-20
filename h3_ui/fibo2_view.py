"""Fibo generation and edit controls."""

from dataclasses import dataclass

import gradio as gr

from h3_app.workflows.fibo2 import FIBO2_MODELS

from .image_library import build_image_file_input
from .optional_engine_controls import _QUIET, _actions, _bind_generation


@dataclass(frozen=True)
class Fibo2View:
    prompt: gr.Textbox
    model: gr.Dropdown
    reference_images: gr.File
    mask_image: gr.File
    width: gr.Slider
    height: gr.Slider
    seed: gr.Number
    steps: gr.Slider
    cfg: gr.Slider
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox


def fibo2_sampling_defaults(model):
    turbo = model == "Edit Turbo"
    return gr.update(value=4 if turbo else 50, interactive=not turbo), gr.update(value=1 if turbo else 5, interactive=not turbo)


def build_fibo2_view(root):
    with root:
        gr.Markdown("## Fibo 2.0\nDev generates and edits images; Edit Turbo edits in 4 steps. "
                    "Weights download on first use. Accept the access conditions for "
                    "[Dev](https://huggingface.co/briaai/Fibo-2.0-dev) and "
                    "[Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo), "
                    "then configure your Hugging Face token on the server. The Dev weights are for noncommercial use.")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                model = gr.Dropdown(list(FIBO2_MODELS), value="Dev", label="Fibo model")
                prompt = gr.Textbox(label="Image prompt / edit instruction / VGL JSON", lines=8,
                    placeholder="Describe a scene, paste a structured caption, or describe an edit.",
                    info="Structured VGL JSON gives detailed control. Edits also accept plain instructions. "
                         "Refer to images in order as <image_1>, <image_2>, etc.")
                refs = build_image_file_input(label="Reference images (optional for Dev, up to 4)",
                                               file_count="multiple", file_types=["image"], type="filepath")
                mask = build_image_file_input(label="Edit mask (optional; white changes, black preserves)",
                                               file_types=["image"], type="filepath")
                gr.Markdown("Masks need exactly one reference and the same pixel dimensions. "
                            "The model uses the mask as edit guidance; pixels outside it may still change.")
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                with gr.Accordion("Sampling", open=False):
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                    cfg = gr.Slider(1, 20, value=5, step=0.1, label="CFG")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Generated image", columns=1, interactive=False)
                output.h3_metadata_root = gr.Column()
                run, stop, status = _actions("Generate / edit image")
    return Fibo2View(prompt, model, refs, mask, width, height, seed, steps, cfg, output, run, stop, status)


def bind_fibo2(view, actions):
    view.model.change(fibo2_sampling_defaults, inputs=view.model, outputs=[view.steps, view.cfg], **_QUIET)
    return _bind_generation(view, "fibo2", actions.generate_fibo2, actions.interrupt, [
        view.prompt, view.model, view.reference_images, view.mask_image, view.width, view.height,
        view.seed, view.steps, view.cfg,
    ])
