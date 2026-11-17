"""Soundstage in the CPU Workshop: the page in a frame, for preparing projects without a GPU.

Projects live in the shared workspace database, so a project written and cast here opens on
the GPU endpoint, where it is spoken.
"""
from functools import partial

import gradio as gr

from .bindings import Drama3Actions
from .handlers import Drama3Handlers, Drama3Services
from .tab import build_soundstage_tab


def build_soundstage_workshop(tab, ui, runtime, store, *, images=None):
    from h3_app.drama3.serving.takes import takes_root

    from .page import install_soundstage_page
    from .project_visuals import planning_export
    from ..api_images import interrupt_api_image

    export = planning_export(runtime)
    if images is not None:
        from .visuals import video_export
        from ..image.page import Live
        export = video_export(images.controller, images.deps)
        export.engines = [("GPT Image", "gpt_image")]
        export.settings = Live(images.pages.job_options)

    handlers = Drama3Handlers(Drama3Services(lambda: runtime, None, None, None))
    actions = Drama3Actions.of(handlers, interrupt=None)
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    build_soundstage_tab(root, actions, gpu=False)
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(
        install_soundstage_page, actions=actions, store=store, output_dir=runtime.output_dir,
        takes_dir=lambda: takes_root(runtime), runtime_config=lambda: runtime, gpu=False,
        interrupt=interrupt_api_image, video_export=export))
    return actions
