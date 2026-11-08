"""Soundstage in the CPU Workshop: the page in a frame, for preparing projects without a GPU.

Projects live in the shared workspace database, so a project written and cast here opens on
the GPU endpoint, where it is spoken.
"""
from functools import partial

import gradio as gr

from .bindings import Drama3Actions
from .handlers import Drama3Handlers, Drama3Services
from .tab import build_soundstage_tab


def build_soundstage_workshop(tab, ui, runtime, store):
    from h3_app.drama3.serving.takes import takes_root

    from .page import install_soundstage_page

    handlers = Drama3Handlers(Drama3Services(lambda: runtime, None, None, None))
    actions = Drama3Actions.of(handlers, interrupt=None)
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    build_soundstage_tab(root, actions, gpu=False)
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(
        install_soundstage_page, actions=actions, store=store, output_dir=runtime.output_dir,
        takes_dir=lambda: takes_root(runtime), runtime_config=lambda: runtime, gpu=False))
    return actions
