"""CPU-only project preparation, line editing and result review."""
from functools import partial

import gradio as gr

from h3_app.catalog import GEMINI_PROMPT_MODELS, DEFAULT_GEMINI_PROMPT_MODEL
from h3_ui.media_console import media_controller
from .bindings import bind_dubbing
from .handlers import DubbingHandlers, DubbingServices
from .projects import bind_projects
from .share import install_share_routes
from .view import build_dubbing_view


def build_dubbing_workshop(tab, ui, runtime):
    controller = media_controller(runtime)
    actions = DubbingHandlers(DubbingServices(lambda: runtime, None, controller.list_media_paths, gpu=False))
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    view = build_dubbing_view(root, gemini_models=GEMINI_PROMPT_MODELS,
                              default_gemini_model=DEFAULT_GEMINI_PROMPT_MODEL, gpu=False)
    bind_dubbing(view, create_tab=tab, actions=actions, gpu=False)
    bind_projects(view, actions, ui.load)
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(install_share_routes, runtime=runtime))
    return view
