"""Build the common Image view; generation and event bindings have separate owners."""

from dataclasses import dataclass
import gradio as gr
from h3_app.capabilities import engines_for_task
from h3_app.catalog import (
    H3_EDIT_SIZE_MATCH, H3_IMAGE_4M_RESOLUTION_PRESETS, H3_IMAGE_8M_RESOLUTION_PRESETS, H3_IMAGE_EDIT_SIZE_MAX,
    QWEN_EDIT_SIZE_MANUAL,
)
from .engines.controls import QUIET, generation_actions, prompt_writer, resolution_presets
from .image_controller import IMAGE_MODELS, reference_tag
from .image_library import build_image_file_input
from .prompt_review import build_prompt_review
from .prompt_writer_controls import RemotePromptWriterControls

SIZE_FIELDS = ("output_resolution_1k", "output_resolution_2k", "resolution_small", "resolution_large",
               "width", "height", "edit_size", "seed")
OPTION_FIELDS = {
    "qwen": ("preset", "model", "turbo_variant", "lightning_strength", "text_encoder", "steps", "cfg",
             "negative_prompt", "sampler", "scheduler", "reference_resolution", "batch_count", "batch_edit_inputs",
             "attention_backend", "accelerator", "cache_device", "cache_dtype", "prompt_enhancer"),
    "flux3": ("model", "expand_prompt", "upsampler", "steps", "cfg", "layout"),
    "moss_image": ("preset", "turbo_variant", "model", "text_encoder", "vae", "steps", "cfg", "accelerate",
                   "rewrite_mode", "layout"),
    "fibo2": ("model", "precision", "steps", "cfg"),
}


@dataclass(frozen=True)
class ImageWorkspace:
    prompt: gr.Textbox
    mode: gr.Radio
    reference_images: gr.File
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    h3_width: gr.Slider
    h3_height: gr.Slider
    h3_seed: gr.Number
    h3_edit_size: gr.Radio
    h3_resolution_small: gr.Dropdown
    h3_resolution_large: gr.Dropdown

    reference_panel: gr.Column
    guidance: gr.Markdown
    readiness: gr.HTML
    size_groups: dict[str, gr.Group]
    option_groups: dict[str, gr.Group]
    base_summary: gr.Markdown
    writer: RemotePromptWriterControls
    enhance: gr.Button
    enhance_status: gr.Textbox
    selected_result: gr.State
    use_result: gr.Button
    result_edit_actions: gr.Group
    reference_source: gr.Number
    reference_destination: gr.Number
    move_reference: gr.Button

    def preference_components(self):
        return {
            **{name: getattr(self, name) for name in ("mode", "h3_width", "h3_height", "h3_edit_size", "h3_seed")},
            "prompt_backend": self.writer.backend, "prompt_model": self.writer.model,
            "openai_model": self.writer.openai_model,
        }


def _move(component):
    component.unrender()
    component.render()


def _build_size_settings(engine_views):
    size_groups = {}
    with gr.Accordion("Output size and seed", open=True):
        for engine in IMAGE_MODELS:
            with gr.Group(visible=engine == "qwen") as size_groups[engine]:
                if engine == "h3":
                    h3_small, h3_large = resolution_presets(
                        H3_IMAGE_4M_RESOLUTION_PRESETS, H3_IMAGE_8M_RESOLUTION_PRESETS,
                        "4M resolution preset", "8M resolution preset",
                        "A recommended size for edits.",
                        "Pruna's recommended size for generation, also good for edits. "
                        "Edits use these instead of the first image size.")
                    with gr.Row():
                        h3_width = gr.Slider(256, 8192, value=1024, step=32, label="Width")
                        h3_height = gr.Slider(256, 8192, value=1024, step=32, label="Height")
                    h3_edit_size = gr.Radio(
                        [H3_EDIT_SIZE_MATCH, H3_IMAGE_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL],
                        value=H3_EDIT_SIZE_MATCH, label="Edit output size", visible=False,
                        info="Keep the first image's aspect ratio at its own size (capped at 8 MP), "
                             "scale it toward 8 MP, or use the width and height above. Applies only to edits.")
                    h3_seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                else:
                    for name in SIZE_FIELDS:
                        component = getattr(engine_views[engine], name, None)
                        if component is not None:
                            _move(component)
    return size_groups, h3_width, h3_height, h3_seed, h3_edit_size, h3_small, h3_large


def _build_model_settings(engine_views, h3, labels):
    option_groups = {}
    with gr.Accordion("Model settings", open=False):
        for engine in IMAGE_MODELS:
            with gr.Group(visible=engine == "qwen") as option_groups[engine]:
                gr.Markdown(f"### {labels[engine]}")
                if engine == "h3":
                    base = gr.Markdown(f"Uses MiniMax H3 settings: **{h3.model_profile.value}**.\n\n"
                                       "P-H3-Image uses 8 steps, Euler and CFG 1.")
                    settings = gr.Button("Open Video → MiniMax H3 settings")
                    from .engines.remake import open_engine_js
                    settings.click(None, js=open_engine_js("Video", "MiniMax H3"), **QUIET)
                else:
                    for name in OPTION_FIELDS[engine]:
                        _move(getattr(engine_views[engine], name))
    return option_groups, base


def build_image_view(root, engine, engine_views, h3):
    labels = {engine.id: engine.label for engine in engines_for_task("Image")}
    with root:
        gr.Markdown("## Image\nCreate an image or edit your references. Switch models without losing your draft.")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                mode = gr.Radio([("Generate", "Text to image"), ("Edit", "Image edit")],
                                value="Text to image", label="Image task")
                prompt = gr.Textbox(label="Image prompt", lines=6, elem_id="h3-image-prompt",
                                    placeholder="Describe the image you want to create.")
                with gr.Column(visible=False) as reference_panel:
                    refs = build_image_file_input(
                        label="Reference images, in order", thumbnails=True, file_count="multiple",
                        file_types=["image"], type="filepath",
                        thumbnail_tags=((engine, engine_views["qwen"].batch_edit_inputs),
                                        lambda index, count, model, separate:
                                        reference_tag(model, index, count, separate)))
                    with gr.Accordion("Reorder references", open=False):
                        gr.Markdown("The first image is the edit target. Move later references to change their order.")
                        with gr.Row():
                            source = gr.Number(value=1, minimum=1, precision=0, label="From position")
                            destination = gr.Number(value=1, minimum=1, precision=0, label="To position")
                            move = gr.Button("Move image", size="sm")
                guidance = gr.Markdown()
                readiness = gr.HTML('<p class="h3-readiness" role="status">Write an image prompt to start.</p>')
                run, stop, status = generation_actions("Generate image", readiness=readiness)
                run.h3_dynamic_label = True
            with gr.Column(scale=2, min_width=320, elem_classes=["h3-preview-panel"]):
                output = gr.Gallery(label="Image results", columns=2, height=500, interactive=False, elem_id="h3-image-results")
                output.h3_metadata_root = gr.Column()
                selected = gr.State(None)
                with gr.Group() as result_edit_actions:
                    use = gr.Button("Use selected result as edit input")
                    gr.Markdown("Select a result, then use it as the starting image for your next edit.")
        # Results precede optional settings in the mobile stack.
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                size_groups, h3_width, h3_height, h3_seed, h3_edit_size, h3_small, h3_large = (
                    _build_size_settings(engine_views))
            with gr.Column(scale=2, min_width=320):
                option_groups, base = _build_model_settings(engine_views, h3, labels)
                with gr.Accordion("Prompt assistance", open=False):
                    writer, enhance, enhance_status = prompt_writer("Suggest a prompt")
                    enhance.h3_review = build_prompt_review(compact=True)
    return ImageWorkspace(
        prompt=prompt, mode=mode, reference_images=refs, output=output, run=run, stop=stop, status=status,
        h3_width=h3_width, h3_height=h3_height, h3_seed=h3_seed, h3_edit_size=h3_edit_size,
        h3_resolution_small=h3_small, h3_resolution_large=h3_large, reference_panel=reference_panel,
        guidance=guidance, readiness=readiness, size_groups=size_groups, option_groups=option_groups,
        base_summary=base, writer=writer, enhance=enhance, enhance_status=enhance_status,
        selected_result=selected, use_result=use, result_edit_actions=result_edit_actions,
        reference_source=source, reference_destination=destination,
        move_reference=move,
    )



def build_image_workspace(root, app_views, engine_views, handlers, h3, deps, demo):
    from .image_bindings import bind_image_workspace

    view = build_image_view(root, app_views.engine, engine_views, h3)
    controller = bind_image_workspace(view, app_views, engine_views, handlers, h3, deps, demo)
    return view, controller
