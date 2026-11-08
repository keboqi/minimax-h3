"""CPU-only Sound Stage project editing in Workshop, in the tab and on the Soundstage page."""
from functools import partial

import gradio as gr

from .bindings import Drama3Actions, bind_drama3
from .handlers import Drama3Handlers, Drama3Services
from .projects import bind_projects
from .view import DRAMA3_CAST_JS, DRAMA3_SCRIPT_HIGHLIGHT_JS, build_drama3_view


def build_soundstage_workshop(tab, ui, runtime, store):
    handlers = Drama3Handlers(Drama3Services(lambda: runtime, None, None, None))
    actions = Drama3Actions.of(handlers, interrupt=None)
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    view = build_drama3_view(root, handlers.drama3_speaker_names(), handlers.drama3_sound_names(),
                             handlers.drama3_speaker_groups(), handlers.drama3_speaker_group_size("Global"), gpu=False)
    bind_drama3(view, actions, gpu=False)
    bind_projects(view, store, runtime.output_dir, ui.load, gpu=False)
    ui.load(None, js=DRAMA3_CAST_JS, api_name=False)
    ui.load(None, js=DRAMA3_SCRIPT_HIGHLIGHT_JS, api_name=False)
    # The page without a GPU: write, cast and manage voices here, then speak the project in Studio.
    from h3_app.drama3.serving.takes import takes_root
    from .page import install_soundstage_page

    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(
        install_soundstage_page, actions=actions, store=store, output_dir=runtime.output_dir,
        takes_dir=lambda: takes_root(runtime), runtime_config=lambda: runtime, gpu=False))
    return view
