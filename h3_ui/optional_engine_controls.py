"""Shared generation, prompt-writer and persistence controls for optional engines."""

import gradio as gr
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt
from .prompt_writer_controls import build_remote_prompt_writer_controls


_QUIET = {"queue": False, "show_progress": "hidden", "api_name": False}


_SKIP_JS = '{"__type__": "update"}'


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
