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
    engine: gr.Radio | None = None
    engine_tabs: gr.Tabs | None = None
    engine_help: gr.Markdown | None = None
    jobs_count: gr.HTML | None = None
    # Fine-tune: the H3 settings tabs under Compose and Preview.
    generation_settings: gr.Column | None = None
    flux3: gr.Group | None = None
    drama3: gr.Group | None = None
    moss_image: gr.Group | None = None
    fibo2: gr.Group | None = None
    dubbing: gr.Column | None = None
    create_tab: gr.Tab | None = None
    projects: gr.Column | None = None
    jobs_tab: gr.Tab | None = None
    # Every top-level tab, in order; Jobs polls only while it is the open one.
    main_tabs: tuple = ()


@dataclass(frozen=True)
class NavigationBindings:
    # restore(task, engine, settings_memory, settings_values) returns
    # (component, update) pairs for ``outputs`` during preference restoration.
    restore: Callable
    outputs: tuple
    engine_value: gr.Textbox


# Output components owned by the result format but not by settings policy,
# with their positions in result_format_layout_updates.
FORMAT_LAYOUT_OUTPUTS = (
    ("output", 3),
    ("output_2", 4),
    ("output_3", 5),
    ("output_4", 6),
    ("image_output_group", 8),
    ("audio_output", 9),
    ("resolution_info", 14),
)
# Dub & Translate has no H3 output; its task keeps the H3 settings on Video.
FORMAT_FOR_TASK = {"Video": "Video", "Image": "Image", "Audio / Music": "Audio", "TTS": "Audio", "Dub": "Video"}
TASK_LABELS = {"TTS": "Soundstage", "Audio / Music": "Song", "Dub": "Dub & Translate"}


def create_app_views(*, summary_root=None, task_root=None) -> AppViews:
    return create_workspace_views(summary_root=summary_root, task_root=task_root)


def create_workspace_views(*, summary_root=None, task_root=None) -> AppViews:
    """``task_root`` places the task picker in the top bar; without one it sits atop Create."""
    with summary_root if summary_root is not None else nullcontext():
        jobs_count = gr.HTML(
            '<p class="h3-system-status" role="status">No active jobs in this session.</p>',
            elem_classes=["h3-jobs-summary"],
        )

    def task_picker():
        # Keep persisted task values stable while updating the visible names.
        return gr.Radio(
            [(TASK_LABELS.get(task, task), task) for task in TASKS],
            value="Video", label="Task", show_label=False, container=False, interactive=False,
        )

    if task_root is not None:
        with task_root:
            task = task_picker()
    tabs = gr.Tabs(elem_id="h3-main-tabs", selected="create", overflow_behavior="wrap")
    with tabs:
        with gr.Tab("Create", id="create") as create_tab:
            if task_root is None:
                with gr.Row(elem_classes=["h3-task-picker"]):
                    task = task_picker()
            # One line: the task's engines as pills, then the chosen engine's description.
            with gr.Row(elem_classes=["h3-engine-bar"]):
                engine = gr.Radio(
                    [(e.label, e.id) for e in engines_for_task("Video")],
                    value="h3",
                    label="Engine",
                    container=False,
                    scale=0,
                    min_width=0,
                    interactive=False,
                )
                engine_help = gr.Markdown(
                    engines_for_task("Video")[0].description,
                    elem_classes=["h3-engine-help"],
                )
            with gr.Tabs(selected="h3", elem_id="h3-engine-tabs") as engine_tabs:
                with gr.Tab("MiniMax H3", id="h3"):
                    gr.HTML(
                        '<nav class="h3-mobile-nav" aria-label="Composer sections"><a href="#h3-composer">Compose</a><a href="#h3-preview">Preview</a><a href="#h3-generate-actions">Generate</a><a href="#h3-fine-tune">Fine-tune</a></nav>',
                        elem_classes=["h3-mobile-nav-container"],
                    )
                    generation = gr.Row(elem_classes=["h3-generator-shell"])
                    generation_settings = gr.Column(elem_id="h3-fine-tune")
                with gr.Tab("LTX 2.5", id="ltx"):
                    ltx25 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Qwen Image 2.1", id="qwen"):
                    qwen = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("MiniMax Music 3", id="music"):
                    music = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("YuE2", id="yue2"):
                    yue2 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("FLUX 3 Image Klein", id="flux3"):
                    flux3 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("MOSS-Image", id="moss_image"):
                    moss_image = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Fibo 2.0", id="fibo2"):
                    fibo2 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Soundstage", id="drama3"):
                    drama3 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Dub & Translate", id="dub"):
                    dubbing = gr.Column(elem_classes=["h3-engine-view", "h3-dub-shell"])
        with gr.Tab("Projects", id="projects") as projects_tab:
            projects = gr.Column(elem_classes=["h3-projects-shell"])
        with gr.Tab("Media", id="media") as gallery_tab:
            gallery = gr.Column(elem_classes=["h3-gallery-shell"])
        with gr.Tab("Jobs", id="jobs") as jobs_tab:
            jobs = gr.Column(elem_classes=["h3-jobs-shell"])
        with gr.Tab("System", id="system") as system_tab:
            system = gr.Group()
        with gr.Tab("API & workflows", id="api") as api_tab:
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
        flux3,
        drama3,
        moss_image,
        fibo2,
        dubbing,
        create_tab,
        projects,
        jobs_tab,
        (create_tab, projects_tab, gallery_tab, jobs_tab, system_tab, api_tab),
    )


def bind_workspace_navigation(
    views: AppViews,
    result_format,
    controller=None,
    *,
    format_layout: Callable | None = None,
    preferences=None,
) -> NavigationBindings | None:
    """Bind task and engine selection; each choice resolves in one response."""
    if views.task is None:
        return
    memory = gr.State({})
    engine_value = gr.Textbox(value="h3", visible=False)
    restoring = gr.Checkbox(value=True, visible=False)
    components = controller.components
    layout_outputs = tuple(components[name] for name, _ in FORMAT_LAYOUT_OUTPUTS)
    no_preferences = (gr.skip(),) if preferences is not None else ()

    def output_intent(task, settings_memory, values, snap=True):
        """Settings and result-format presentation for the task's output.

        Restoration keeps the saved dimensions; only a task switch realigns
        them for the new result format.
        """
        values = list(values)
        session = values[-1]
        desired = FORMAT_FOR_TASK[task]
        settled = (session.memory or settings_memory or {}).get("values", {})

        def browser(name):
            return values[controller.inputs.index(components[name])]

        def value(name):
            return settled.get(name, browser(name))

        overrides = {"result_format": desired}
        layout = None
        if format_layout is not None:
            layout = list(format_layout(
                desired,
                value("width"),
                value("height"),
                browser("first"),
                value("latent_upscale"),
            ))
            if snap:
                overrides.update(width=layout[12]["value"], height=layout[13]["value"])
            else:
                layout[14] = gr.skip()
        updates = list(
            controller.update(
                settings_memory, *values, action="result_format", overrides=overrides
            )
        )
        for name, override in overrides.items():
            # Rewriting an unchanged value would still fire change listeners.
            if override != browser(name):
                index = controller.outputs.index(components[name])
                updates[index] = {**updates[index], "value": override}
        extras = (
            [layout[index] for _, index in FORMAT_LAYOUT_OUTPUTS]
            if layout is not None
            else [gr.skip() for _ in layout_outputs]
        )
        return updates + extras

    def stage(session, task, engine):
        if preferences is not None:
            session.stage({"workspace.task": task, "workspace.engine": engine})

    def select_task(
        task, remembered, pending_restore, settings_memory, *settings_values
    ):
        if pending_restore:
            return (gr.skip(),) * (4 + len(controller.outputs) + len(layout_outputs))
        choices = engines_for_task(task)
        available = {e.id for e in choices}
        selected = (remembered or {}).get(task)
        if selected not in available:
            selected = choices[0].id
        detail = next((e.description for e in choices if e.id == selected))
        stage(settings_values[-1], task, selected)
        return (
            gr.update(choices=[(e.label, e.id) for e in choices], value=selected),
            gr.update(selected=selected),
            detail,
            selected,
            *output_intent(task, settings_memory, settings_values),
        )

    # Input, not change: restoration writes the task without re-running this.
    views.task.input(
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
            *layout_outputs,
        ],
        queue=False,
        trigger_mode="always_last",
        api_name=False,
        show_progress="hidden",
    )

    def select_engine(task, engine, remembered, pending_restore, session):
        if pending_restore:
            return (gr.skip(),) * (4 + len(no_preferences))
        choices = engines_for_task(task)
        selected = next((e for e in choices if e.id == engine), choices[0])
        remembered = {**(remembered or {}), task: selected.id}
        saved = ()
        if preferences is not None:
            saved = (
                session.remember({"workspace.task": task, "workspace.engine": selected.id}),
            )
        return (
            gr.update(selected=selected.id),
            selected.description,
            remembered,
            selected.id,
            *saved,
        )

    views.engine.input(
        select_engine,
        inputs=[views.task, views.engine, memory, restoring, controller.session],
        outputs=[
            views.engine_tabs,
            views.engine_help,
            memory,
            engine_value,
            *([preferences] if preferences is not None else []),
        ],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )

    outputs = (
        views.engine,
        views.engine_tabs,
        views.engine_help,
        *controller.outputs,
        *layout_outputs,
        restoring,
        views.task,
        memory,
    )

    def finish_restore(task, engine, settings_memory, settings_values):
        choices = engines_for_task(task)
        selected = next((item for item in choices if item.id == engine), choices[0])
        stage(settings_values[-1], task, selected.id)
        updates = (
            gr.update(
                choices=[(item.label, item.id) for item in choices],
                value=selected.id,
                interactive=True,
            ),
            gr.update(selected=selected.id),
            selected.description,
            *output_intent(task, settings_memory, settings_values, snap=False),
            False,
            gr.update(interactive=True),
            {task: selected.id},
        )
        return list(zip(outputs, updates, strict=True))

    return NavigationBindings(finish_restore, outputs, engine_value)
