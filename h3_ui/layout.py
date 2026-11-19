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


H3_FRAME = ('<iframe class="h3-video-frame" src="/h3/" title="MiniMax H3" '
            'allow="fullscreen; clipboard-write" scrolling="no"></iframe>')
IMAGE_FRAME = ('<iframe class="h3-video-frame" src="/image/" title="Image" '
               'allow="fullscreen; clipboard-write" scrolling="no"></iframe>')
LTX_FRAME = ('<iframe class="h3-video-frame" src="/ltx/" title="LTX 2.5" '
             'allow="fullscreen; clipboard-write" scrolling="no"></iframe>')
SONG_FRAME = ('<iframe class="h3-video-frame" src="/song/" title="Song" '
              'allow="fullscreen; clipboard-write" scrolling="no"></iframe>')
MEDIA_FRAME = ('<iframe class="h3-video-frame" src="/media/" title="Media" '
               'allow="fullscreen; clipboard-write" scrolling="no"></iframe>')
# The Image and Song pages show their own model pills, so the studio's stay away for those tasks.
# The engine tab that holds each such task's own page.
PAGE_TABS = {"Image": "image", "Audio / Music": "song"}


def show_engine_picker(task: str, choices) -> bool:
    """Only multi-engine tasks without their own page show the top-bar engine switcher."""
    return task not in PAGE_TABS and len(choices) > 1


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
    jobs_count: gr.HTML | None = None
    # Fine-tune: the H3 settings tabs under Compose and Preview.
    generation_settings: gr.Column | None = None
    flux3: gr.Group | None = None
    drama3: gr.Group | None = None
    moss_image: gr.Group | None = None
    fibo2: gr.Group | None = None
    dubbing: gr.Column | None = None
    create_tab: gr.Tab | None = None
    jobs_tab: gr.Tab | None = None
    # Every top-level tab, in order; Jobs polls only while it is the open one.
    main_tabs: tuple = ()
    storyboard: gr.Column | None = None
    remake: gr.Column | None = None
    image: gr.Column | None = None
    models: gr.Column | None = None
    models_tab: gr.Tab | None = None
    engine_bar: gr.Column | gr.Row | None = None


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
# Dubbing has no H3 output; its task keeps the H3 settings on Video.
FORMAT_FOR_TASK = {"Video": "Video", "Image": "Image", "Audio / Music": "Audio", "Remake": "Video", "Storyboard": "Video",
                   "TTS": "Audio", "Dub": "Video"}
TASK_LABELS = {"TTS": "Soundstage", "Audio / Music": "Song", "Dub": "Dubbing"}


def create_app_views(*, summary_root=None, task_root=None, engine_root=None, include_models: bool = False) -> AppViews:
    return create_workspace_views(
        summary_root=summary_root,
        task_root=task_root,
        engine_root=engine_root,
        include_models=include_models,
    )


def create_workspace_views(*, summary_root=None, task_root=None, engine_root=None, include_models: bool = False) -> AppViews:
    """``task_root`` and ``engine_root`` place controls in the top bar; without them they sit atop Create."""
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

    def engine_picker():
        return gr.Radio(
            [(e.label, e.id) for e in engines_for_task("Video")],
            value="h3",
            label="Engine",
            show_label=False,
            container=False,
            scale=0,
            min_width=0,
            interactive=False,
        )

    if task_root is not None:
        with task_root:
            task = task_picker()
    if engine_root is not None:
        with engine_root:
            engine = engine_picker()
        engine_bar = engine_root
    tabs = gr.Tabs(elem_id="h3-main-tabs", selected="create", overflow_behavior="wrap")
    with tabs:
        with gr.Tab("Create", id="create") as create_tab:
            if task_root is None:
                with gr.Row(elem_classes=["h3-task-picker"]):
                    task = task_picker()
            if engine_root is None:
                with gr.Row(elem_classes=["h3-engine-bar"]) as engine_bar:
                    engine = engine_picker()
            with gr.Tabs(selected="h3", elem_id="h3-engine-tabs") as engine_tabs:
                with gr.Tab("MiniMax H3", id="h3"):
                    # Everything people do happens on the page at /h3/ (h3/page.py), outside Gradio.
                    gr.HTML(H3_FRAME)
                    # The Gradio view stays built but unseen: it keeps the published API endpoints
                    # and the components the Image workspace and navigation still read.
                    generation = gr.Row(elem_classes=["h3-generator-shell"], visible=False)
                    generation_settings = gr.Column(elem_id="h3-fine-tune", visible=False)
                with gr.Tab("Image workspace", id="image"):
                    # Everything people do happens on the page at /image/ (image/page.py), outside Gradio.
                    gr.HTML(IMAGE_FRAME)
                    # The Gradio view stays built but unseen: its job callback runs the page's generations.
                    image = gr.Column(elem_classes=["h3-engine-view"], visible=False)
                with gr.Tab("LTX 2.5", id="ltx"):
                    # Everything people do happens on the page at /ltx/ (ltx/page.py), outside Gradio.
                    gr.HTML(LTX_FRAME)
                    # The Gradio view stays built but unseen: its job callback runs the page's videos, and it
                    # keeps the published API endpoints.
                    ltx25 = gr.Group(elem_classes=["h3-engine-view"], visible=False)
                with gr.Tab("Qwen Image 2.1", id="qwen"):
                    qwen = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Song", id="song"):
                    # Everything people do happens on the page at /song/ (song/page.py), outside Gradio.
                    gr.HTML(SONG_FRAME)
                with gr.Tab("MiniMax Music 3", id="music"):
                    # The Gradio views stay built but unseen: their job callbacks run the page's takes, and
                    # they keep the published API endpoints.
                    music = gr.Group(elem_classes=["h3-engine-view"], visible=False)
                with gr.Tab("YuE2", id="yue2"):
                    yue2 = gr.Group(elem_classes=["h3-engine-view"], visible=False)
                with gr.Tab("FLUX 3 Image Klein", id="flux3"):
                    flux3 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("MOSS-Image", id="moss_image"):
                    moss_image = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Fibo 2.0", id="fibo2"):
                    fibo2 = gr.Group(elem_classes=["h3-engine-view"])
                with gr.Tab("Remake", id="remake"):
                    remake = gr.Column(elem_classes=["h3-engine-view"])
                with gr.Tab("Storyboard", id="storyboard"):
                    storyboard = gr.Column(elem_classes=["h3-engine-view", "h3-storyboard-shell"])
                with gr.Tab("Soundstage", id="drama3"):
                    drama3 = gr.Column(elem_classes=["h3-engine-view", "h3-soundstage-shell"])
                with gr.Tab("Dubbing", id="dub"):
                    dubbing = gr.Column(elem_classes=["h3-engine-view", "h3-dub-shell"])
        with gr.Tab("Media", id="media") as gallery_tab:
            # Everything people do happens on the page at /media/ (media/page.py), outside Gradio.
            gr.HTML(MEDIA_FRAME)
            # The Gradio library stays built but unseen: it keeps the published API, the enhancement job
            # callback and the components other views still read.
            gallery = gr.Column(elem_classes=["h3-gallery-shell"], visible=False)
        models = None
        models_tab = None
        if include_models:
            with gr.Tab("Models", id="models") as models_tab:
                models = gr.Column(elem_classes=["h3-models-shell"])
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
        jobs_count,
        generation_settings,
        flux3,
        drama3,
        moss_image,
        fibo2,
        dubbing,
        create_tab,
        jobs_tab,
        (create_tab, gallery_tab, models_tab, jobs_tab, system_tab, api_tab)
        if models_tab is not None
        else (create_tab, gallery_tab, jobs_tab, system_tab, api_tab),
        storyboard=storyboard,
        remake=remake,
        image=image,
        models=models,
        models_tab=models_tab,
        engine_bar=engine_bar,
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
        if task in PAGE_TABS:
            # Image and Song have their own composers; visiting them must not rewrite Video settings.
            return [gr.skip() for _ in (*controller.outputs, *layout_outputs)]
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
                value("auto_megapixels"),
                value("image_edit_size"),
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

    engine_bar_outputs = (views.engine_bar,) if views.engine_bar is not None else ()

    def select_task(
        task, remembered, pending_restore, settings_memory, *settings_values
    ):
        if pending_restore:
            return (gr.skip(),) * (3 + len(engine_bar_outputs) + len(controller.outputs) + len(layout_outputs))
        choices = engines_for_task(task)
        available = {e.id for e in choices}
        selected = (remembered or {}).get(task)
        if selected not in available:
            selected = choices[0].id
        visible = show_engine_picker(task, choices)
        stage(settings_values[-1], task, selected)
        return (
            gr.update(choices=[(e.label, e.id) for e in choices], value=selected, visible=visible),
            *((gr.update(visible=visible),) if views.engine_bar is not None else ()),
            gr.update(selected=PAGE_TABS.get(task, selected)),
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
            *engine_bar_outputs,
            views.engine_tabs,
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
            return (gr.skip(),) * (3 + len(no_preferences))
        choices = engines_for_task(task)
        selected = next((e for e in choices if e.id == engine), choices[0])
        remembered = {**(remembered or {}), task: selected.id}
        saved = ()
        if preferences is not None:
            saved = (
                session.remember({"workspace.task": task, "workspace.engine": selected.id}),
            )
        return (
            gr.update(selected=PAGE_TABS.get(task, selected.id)),
            remembered,
            selected.id,
            *saved,
        )

    views.engine.input(
        select_engine,
        inputs=[views.task, views.engine, memory, restoring, controller.session],
        outputs=[
            views.engine_tabs,
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
        *engine_bar_outputs,
        views.engine_tabs,
        *controller.outputs,
        *layout_outputs,
        restoring,
        views.task,
        memory,
    )

    def finish_restore(task, engine, settings_memory, settings_values):
        choices = engines_for_task(task)
        selected = next((item for item in choices if item.id == engine), choices[0])
        visible = show_engine_picker(task, choices)
        stage(settings_values[-1], task, selected.id)
        updates = (
            gr.update(
                choices=[(item.label, item.id) for item in choices],
                value=selected.id,
                interactive=True,
                visible=visible,
            ),
            *((gr.update(visible=visible),) if views.engine_bar is not None else ()),
            gr.update(selected=PAGE_TABS.get(task, selected.id)),
            *output_intent(task, settings_memory, settings_values, snap=False),
            False,
            gr.update(interactive=True),
            {task: selected.id},
        )
        return list(zip(outputs, updates, strict=True))

    return NavigationBindings(finish_restore, outputs, engine_value)
