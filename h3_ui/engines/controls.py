"""Building blocks every engine view shares: Generate/Interrupt/progress, the remote prompt
writer, resolution presets and the controls saved as browser settings."""

import inspect

import gradio as gr
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS

from ..job_bindings import bind_gpu_action, owned_generation, owned_interrupt
from ..prompt_writer_controls import build_remote_prompt_writer_controls

# Small, unqueued UI updates that need no progress or API endpoint.
QUIET = {"queue": False, "show_progress": "hidden", "api_name": False}

# A js callback result that leaves its output unchanged.
SKIP_JS = '{"__type__": "update"}'


def generation_actions(label):
    """The primary action, its Interrupt and the progress box, in the shared placement."""
    with gr.Row(elem_classes=["h3-generation-buttons"]):
        run = gr.Button(label, variant="primary", elem_classes=["h3-primary-action"])
        stop = gr.Button("Interrupt", interactive=False)
        run.h3_stop = stop
    status = gr.Textbox(label="Generation progress", lines=3, interactive=False)
    return run, stop, status


def prompt_writer(button_label):
    """Gemini/Lightning writer controls, its button and its status box."""
    writer = build_remote_prompt_writer_controls(GEMINI_PROMPT_MODELS, DEFAULT_GEMINI_PROMPT_MODEL)
    enhance = gr.Button(button_label)
    status = gr.Textbox(label="Writer status", lines=2, interactive=False)
    return writer, enhance, status


def writer_inputs(writer):
    return [writer.model, writer.gemini_api_key, writer.backend, writer.lightning_api_key, writer.lightning_model]


def settings_components(view):
    """Persistable components by setting name; voice slots and nested groups excluded."""
    components = {name: value for name, value in vars(view).items()
                  if isinstance(value, gr.components.Component)}
    writer = getattr(view, "writer", None)
    if writer is not None:
        components.update(prompt_backend=writer.backend, prompt_model=writer.model,
                          lightning_model=writer.lightning_model)
    return components


def owned_view_generation(generate, family, inputs):
    sig_names = tuple(
        name
        for name, param in inspect.signature(generate).parameters.items()
        if name not in {"request", "progress"}
        and param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    )
    input_names = sig_names[: len(inputs)] if len(sig_names) >= len(inputs) and sig_names else None
    return owned_generation(generate, family, input_names=input_names)


def bind_generation(view, family, generate, interrupt, inputs):
    """Run ``generate`` on the shared GPU queue as ``/generate_<family>``; Interrupt cancels it."""
    event = bind_gpu_action(view.run.click, owned_view_generation(generate, family, inputs), inputs=inputs,
                            outputs=[view.output, view.status], api_name=f"generate_{family}",
                            show_progress="minimal")
    stopped = view.stop.click(owned_interrupt(interrupt, family), outputs=view.status, queue=False, api_name=False)
    stopped.then(fn=None, cancels=[event], queue=False, api_name=False)
    return event


def resolution_presets(small, large, small_label, large_label, small_info, large_info):
    with gr.Row():
        first = gr.Dropdown(list(small), value=None, label=small_label, info=small_info)
        second = gr.Dropdown(list(large), value=None, label=large_label, info=large_info)
    return first, second


def bind_resolution_presets(view, *, extra_outputs=(), extra_values=()):
    """Mirror Qwen: a preset sets the size and clears the other; a manual size clears both."""
    from ..qwen_bindings import qwen_resolution_preset_values

    pairs = ((view.resolution_small, view.resolution_large), (view.resolution_large, view.resolution_small))
    for chosen, other in pairs:
        chosen.input(
            lambda name: ((*qwen_resolution_preset_values(name), None, *extra_values) if name
                          else tuple(gr.update() for _ in range(3 + len(extra_outputs)))),
            inputs=chosen, outputs=[view.width, view.height, other, *extra_outputs], **QUIET,
        )
    for dimension in (view.width, view.height):
        dimension.input(lambda: (None, None), outputs=[view.resolution_small, view.resolution_large], **QUIET)
