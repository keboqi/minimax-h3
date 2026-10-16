"""FLUX 3 image and Drama 3 speech controls and owned job bindings."""

from dataclasses import dataclass
import gradio as gr
from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_models import FLUX3_MODEL_CHOICES
from .image_library import build_image_file_input
from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt


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
    output: gr.Gallery
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox


@dataclass(frozen=True)
class Drama3View:
    text: gr.Textbox
    reference_audio: gr.Audio
    transcript: gr.Textbox
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
                    info="Uses the official prompt upsampler for text-to-image. For edits, write the full desired scene with your changes included; expansion is disabled.")
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
                run, stop, status = _actions("Generate with FLUX 3")
    return Flux3View(prompt, model, refs, width, height, edit_size, seed, steps, expand, output, run, stop, status)


def build_drama3_view(root):
    with root:
        gr.Markdown("## Drama 3 TTS (experimental)\nBuilt with Fish Audio. "
                    "Generate expressive speech; tags such as `[sighs]` and `[laughs]` belong in the text. "
                    "The FP8 model and codec download on first use (~40 GB). With experts in RAM, allow "
                    "approximately 48 GB system RAM. [Model and license](https://huggingface.co/fishaudio/drama-3)")
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=3, min_width=320, elem_classes=["h3-composer"]):
                text = gr.Textbox(label="Speech text", lines=8, placeholder="[with warm excitement] Welcome! Let's get started.")
                with gr.Accordion("Voice reference (optional)", open=False):
                    audio = gr.Audio(label="Reference voice", type="filepath")
                    transcript = gr.Textbox(label="Exact reference transcript", lines=3,
                        info="Include every spoken word in the clip. Leave both fields empty to generate a voice from the seed.")
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
                run, stop, status = _actions("Generate speech with Drama 3")
    return Drama3View(text, audio, transcript, seed, temperature, top_p, top_k,
                      max_frames, split, experts, cache, output, run, stop, status)


def bind_optional_engine(view, family, generate, interrupt):
    inputs = ([view.prompt, view.model, view.reference_images, view.width, view.height,
               view.edit_size, view.seed, view.steps, view.expand_prompt] if family == "flux3" else
              [view.text, view.reference_audio, view.transcript, view.seed, view.temperature,
               view.top_p, view.top_k, view.max_frames, view.split_long_text, view.experts, view.expert_cache_gb])
    event = bind_gpu_action(view.run.click, owned_generation(generate, family), inputs=inputs,
                           outputs=[view.output, view.status], api_name=f"generate_{family}", show_progress="minimal")
    stopped = view.stop.click(owned_interrupt(interrupt, family), outputs=view.status, queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return event
