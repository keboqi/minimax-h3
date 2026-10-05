"""Top-level Gradio navigation containers.

Views are declared as real tab children up front, then populated by the main
application. This keeps tab ownership explicit without coupling the shell to
the large set of generation callbacks.
"""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable
import gradio as gr
from h3_app.capabilities import engines_for_task, TASKS


@dataclass(frozen=True)
class AppViews:
    tabs: gr.Tabs
    generation: gr.Row
    qwen_image21: gr.Group
    ltx25: gr.Group
    music3: gr.Group
    yue2: gr.Group
    gallery: gr.Column
    api: gr.Group
    gallery_tab: gr.Tab
    system: gr.Group | None = None
    jobs: gr.Column | None = None
    task: gr.Radio | None = None
    engine: gr.Dropdown | None = None
    engine_tabs: gr.Tabs | None = None
    engine_help: gr.Markdown | None = None
    jobs_count: gr.HTML | None = None
    generation_settings: gr.Accordion | None = None


@dataclass(frozen=True)
class NavigationBindings:
    restore: Callable
    inputs: tuple
    outputs: tuple
    engine_value: gr.Textbox


def create_app_views(*, summary_root=None) -> AppViews:
    return create_workspace_views(summary_root=summary_root)


def create_workspace_views(*, summary_root=None) -> AppViews:
    with summary_root if summary_root is not None else nullcontext():
        jobs_count = gr.HTML(
            '<p class="h3-system-status" role="status">No active jobs in this session.</p>',
            elem_classes=["h3-jobs-summary"],
        )
    tabs = gr.Tabs(elem_id="h3-main-tabs", selected="create", overflow_behavior="wrap")
    with tabs:
        with gr.Tab("Create", id="create"):
            with gr.Row(elem_classes=["h3-task-picker"]):
                task = gr.Radio(
                    TASKS, value="Video", label="Task", scale=2, interactive=False
                )
                engine = gr.Dropdown(
                    [(e.label, e.id) for e in engines_for_task("Video")],
                    value="h3",
                    label="Engine",
                    scale=1,
                    interactive=False,
                )
            engine_help = gr.Markdown(engines_for_task("Video")[0].description)
            with gr.Tabs(selected="h3", elem_id="h3-engine-tabs") as engine_tabs:
                with gr.Tab("MiniMax H3", id="h3"):
                    gr.HTML(
                        '<nav class="h3-mobile-nav" aria-label="Workspace sections"><a href="#h3-composer">Compose</a><a href="#h3-generate-actions">Generate</a><a href="#h3-preview">Preview</a></nav>',
                        elem_classes=["h3-mobile-nav-container"],
                    )
                    generation = gr.Row(elem_classes=["h3-generator-shell"])
                    generation_settings = gr.Accordion(
                        "Advanced settings", open=False, elem_id="h3-advanced-settings"
                    )
                with gr.Tab("LTX 2.5", id="ltx"):
                    ltx25 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Qwen Image 2.1", id="qwen"):
                    qwen = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("MiniMax Music 3", id="music"):
                    music = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("YuE2", id="yue2"):
                    yue2 = gr.Group(elem_classes=["h3-engine-view"])
        with gr.Tab("Media", id="media") as gallery_tab:
            gallery = gr.Column(elem_classes=["h3-gallery-shell"])
        with gr.Tab("Jobs", id="jobs"):
            jobs = gr.Column(elem_classes=["h3-jobs-shell"])
        with gr.Tab("System", id="system"):
            system = gr.Group()
        with gr.Tab("API & workflows", id="api"):
            api = gr.Group()
    return AppViews(
        tabs,
        generation,
        qwen,
        ltx25,
        music,
        yue2,
        gallery,
        api,
        gallery_tab,
        system,
        jobs,
        task,
        engine,
        engine_tabs,
        engine_help,
        jobs_count,
        generation_settings,
    )


def bind_workspace_navigation(
    views: AppViews, result_format, controller=None
) -> NavigationBindings | None:
    if views.task is None:
        return
    memory = gr.State({})
    engine_value = gr.Textbox(value="h3", visible=False)
    restoring = gr.Checkbox(value=True, visible=False)

    def output_intent(task, settings_memory, values):
        values = list(values)
        desired = {"Video": "Video", "Image": "Image", "Audio / Music": "Audio"}[task]
        values[controller.names.index("result_format")] = desired
        updates = list(
            controller.update(settings_memory, *values, action="result_format")
        )
        index = controller.outputs.index(result_format)
        updates[index] = {**updates[index], "value": desired}
        return updates

    def select_task(
        task, remembered, pending_restore, settings_memory, *settings_values
    ):
        if pending_restore:
            return (gr.skip(),) * (4 + len(controller.outputs))
        choices = engines_for_task(task)
        available = {e.id for e in choices}
        selected = (remembered or {}).get(task)
        if selected not in available:
            selected = choices[0].id
        detail = next((e.description for e in choices if e.id == selected))
        return (
            gr.update(choices=[(e.label, e.id) for e in choices], value=selected),
            gr.update(selected=selected),
            detail,
            selected,
            *output_intent(task, settings_memory, settings_values),
        )

    views.task.change(
        select_task,
        inputs=[
            views.task,
            memory,
            restoring,
            controller.memory,
            *controller.inputs,
        ],
        outputs=[
            views.engine,
            views.engine_tabs,
            views.engine_help,
            engine_value,
            *controller.outputs,
        ],
        queue=True,
        concurrency_id="h3-settings",
        concurrency_limit=1,
        api_name=False,
        show_progress="hidden",
    )

    def select_engine(task, engine, remembered, pending_restore):
        if pending_restore:
            return (gr.skip(),) * 4
        choices = engines_for_task(task)
        selected = next((e for e in choices if e.id == engine), choices[0])
        remembered = {**(remembered or {}), task: selected.id}
        return (
            gr.update(selected=selected.id),
            selected.description,
            remembered,
            selected.id,
        )

    views.engine.input(
        select_engine,
        inputs=[views.task, views.engine, memory, restoring],
        outputs=[views.engine_tabs, views.engine_help, memory, engine_value],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )

    def finish_restore(task, engine, settings_memory, *settings_values):
        choices = engines_for_task(task)
        selected = next((item for item in choices if item.id == engine), choices[0])
        return (
            gr.update(
                choices=[(item.label, item.id) for item in choices],
                value=selected.id,
                interactive=True,
            ),
            gr.update(selected=selected.id),
            selected.description,
            *output_intent(task, settings_memory, settings_values),
            False,
            gr.update(interactive=True),
            {task: selected.id},
        )

    return NavigationBindings(
        finish_restore,
        (views.task, engine_value, controller.memory, *controller.inputs),
        (
            views.engine,
            views.engine_tabs,
            views.engine_help,
            *controller.outputs,
            restoring,
            views.task,
            memory,
        ),
        engine_value,
    )
