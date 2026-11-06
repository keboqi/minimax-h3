"""Bind Image navigation, draft actions, generation and prompt review."""

from html import escape
from math import isfinite
import gradio as gr
from h3_app.capabilities import engines_for_task
from h3_app.errors import H3Error
from .engines.controls import QUIET
from .image_controller import (
    IMAGE_MODELS, ImageGenerationController, draft_issue, image_mode, reference_limit, suggest_prompt,
)
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt


def move_reference(paths, source, destination):
    paths = list(paths or [])
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not isfinite(n)
           or int(n) != n or not 1 <= n <= len(paths) for n in (source, destination)):
        raise gr.Error("Choose reference positions within the uploaded images.")
    paths.insert(int(destination) - 1, paths.pop(int(source) - 1))
    return paths


def bind_image_workspace(view, app_views, engine_views, handlers, h3, deps, demo):
    labels = {engine.id: engine.label for engine in engines_for_task("Image")}
    prompt, mode, refs = view.prompt, view.mode, view.reference_images
    output, selected, use = view.output, view.selected_result, view.use_result
    run, stop, status = view.run, view.stop, view.status
    reference_panel, guidance, readiness = view.reference_panel, view.guidance, view.readiness
    size_groups, option_groups = view.size_groups, view.option_groups
    enhance, enhance_status = view.enhance, view.enhance_status
    view.move_reference.click(move_reference,
                              inputs=[refs, view.reference_source, view.reference_destination], outputs=refs, **QUIET)

    def choose_result(items, event: gr.SelectData):
        index = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
        return items[index][0] if items and 0 <= index < len(items) else None

    output.select(choose_result, inputs=output, outputs=selected, **QUIET)
    output.change(lambda: None, outputs=selected, **QUIET)
    refs.change(lambda paths: "Image edit" if paths else gr.skip(), inputs=refs, outputs=mode, **QUIET)

    def use_result(engine, path):
        if engine == "moss_image":
            raise gr.Error("MOSS-Image supports generation only. Choose an edit-capable model first.")
        if not path:
            raise gr.Error("Select an image result first.")
        return [path], "Image edit"

    use.click(use_result, inputs=[app_views.engine, selected], outputs=[refs, mode], **QUIET)
    h3.model_profile.change(
        lambda profile: f"Uses MiniMax H3 settings: **{profile}**.\n\nP-H3-Image uses 8 steps, Euler and CFG 1.",
        inputs=h3.model_profile, outputs=view.base_summary, **QUIET)

    state_inputs = [app_views.engine, mode, prompt, refs, engine_views["qwen"].turbo_variant,
                    engine_views["fibo2"].model, engine_views["qwen"].batch_edit_inputs]

    edit_controls = [engine_views[engine].edit_size for engine in ("qwen", "flux3", "fibo2")]
    edit_controls.extend([engine_views["qwen"].batch_edit_inputs, engine_views["qwen"].reference_resolution])
    cache_controls = [engine_views["qwen"].cache_device, engine_views["qwen"].cache_dtype]
    def edit_actions(engine):
        if engine not in IMAGE_MODELS:
            return gr.skip(), gr.skip()
        return gr.update(visible=engine != "moss_image"), gr.update(visible=engine != "moss_image")

    gr.on(triggers=[app_views.engine.change, demo.load], fn=edit_actions, inputs=app_views.engine,
          outputs=[mode, view.result_edit_actions], trigger_mode="always_last", **QUIET)

    layout_outputs = [reference_panel, prompt, *size_groups.values(), *option_groups.values(),
                      *edit_controls, *cache_controls]

    def layout(engine, intent):
        if engine not in IMAGE_MODELS:
            return (gr.skip(),) * len(layout_outputs)
        editing = image_mode(engine, intent) == "Image edit"
        return (
            gr.update(visible=editing),
            gr.update(label="Edit instruction" if editing else "Image prompt",
                      placeholder="Describe what to change and what to preserve." if editing else
                      "Describe the image you want to create."),
            *(gr.update(visible=key == engine) for key in IMAGE_MODELS),
            *(gr.update(visible=key == engine) for key in IMAGE_MODELS),
            *(gr.update(visible=editing) for _ in edit_controls),
            *(gr.update(interactive=editing) for _ in cache_controls),
        )

    gr.on(triggers=[app_views.engine.change, mode.change, demo.load], fn=layout,
          inputs=[app_views.engine, mode], outputs=layout_outputs, trigger_mode="always_last", **QUIET)

    def readiness_state(engine, intent, text, paths, variant, fibo_model, separate):
        if engine not in IMAGE_MODELS:
            return (gr.skip(),) * 4
        intent = image_mode(engine, intent)
        editing = intent == "Image edit"
        issue = draft_issue(engine, intent, text, paths, variant, fibo_model, separate)
        limit = reference_limit(engine, variant)
        note = ("MOSS-Image generates from text only." if engine == "moss_image" else
                f"Up to {limit} references per edit. Upload order is preserved.")
        if editing and engine == "qwen":
            note += " First image is the target; use `<image1>`, `<image2>`, … for multiple references."
        if engine == "fibo2":
            note += " Plain text or VGL JSON supported; refer to images as `<image_1>`, `<image_2>`, …"
        if editing and engine == "flux3":
            note += " Layout descriptions can refer to `<ref_image_0>`, `<ref_image_1>`, …"
        if not editing and paths:
            note += f" {len(paths)} saved reference(s) are inactive in Generate mode."
        ready = issue or f"Ready to {'edit' if editing else 'generate'} with {labels[engine]}."
        return (
            note, f'<p class="h3-readiness" role="status">{escape(ready)}</p>',
            gr.update(value="Edit image" if editing else "Generate image", interactive=not issue),
            gr.update(interactive=engine in {"qwen", "flux3", "moss_image"} and not (editing and engine == "moss_image")),
        )

    gr.on(triggers=[component.change for component in state_inputs] + [demo.load], fn=readiness_state,
          inputs=state_inputs, outputs=[guidance, readiness, run, enhance],
          trigger_mode="always_last", **QUIET)

    controller = ImageGenerationController(engine_views, handlers, h3, view, deps)

    def generate(engine, intent, text, paths, *values):
        captured = dict(zip(controller.fields, values, strict=True))
        try:
            updates = controller.generate(engine, image_mode(engine, intent), text, paths, captured)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        for result, message in updates:
            if isinstance(result, str):
                result = [result]
            yield (result if result is not None else gr.skip()), f"{labels[engine]} · {message}"

    inputs = [app_views.engine, mode, prompt, refs, *controller.fields.values()]
    event = bind_gpu_action(run.click, owned_generation(generate, "image", input_names=(
        "engine", "mode", "prompt", "reference_images", *controller.fields)),
        inputs=inputs, outputs=[output, status], api_name=False, show_progress="minimal")
    stopped = stop.click(owned_interrupt(deps.interrupt, "image"), outputs=status, **QUIET)
    stopped.then(None, cancels=[event], **QUIET)

    writer = view.writer
    prompt_fields = {
        "moss_rewrite_mode": engine_views["moss_image"].rewrite_mode,
        "qwen_edit_size": engine_views["qwen"].edit_size,
        **{f"{engine}_{name}": getattr(engine_views[engine], name)
           for engine in ("qwen", "flux3", "moss_image") for name in ("width", "height")},
        "writer_model": writer.model, "writer_key": writer.gemini_api_key, "writer_backend": writer.backend,
        "writer_openai_key": writer.openai_api_key, "writer_openai_model": writer.openai_model,
        "writer_base_url": writer.openai_base_url,
    }

    def suggest_selected(engine, intent, text, paths, *values):
        try:
            return suggest_prompt(handlers, engine, image_mode(engine, intent), text, paths,
                                  dict(zip(prompt_fields, values, strict=True)))
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    suggestion = bind_prompt_action(enhance.click, suggest_selected,
                                    inputs=[app_views.engine, mode, prompt, refs, *prompt_fields.values()],
                                    outputs=[prompt, enhance_status], api_name=False, show_progress="minimal")

    def accepted_suggestion(engine, text, message):
        if message != "Suggested prompt accepted.":
            return gr.skip(), gr.skip()
        from h3_app.workflows.moss_image_prompt import is_layout_prompt

        return (False if engine == "flux3" else gr.skip(),
                "" if engine == "moss_image" and is_layout_prompt(text) else gr.skip())

    suggestion.acceptance.success(accepted_suggestion, inputs=[app_views.engine, prompt, enhance_status],
                                  outputs=[engine_views["flux3"].expand_prompt, engine_views["moss_image"].layout],
                                  **QUIET)
