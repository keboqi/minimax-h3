"""Dubbing in the CPU Workshop: the page in a frame, for preparing projects without a GPU.

Projects and sessions live in the shared workspace, so a project analyzed with Gemini and edited here
opens in Studio, which voices it. Workshop also serves the dubs' share links.
"""
from functools import partial

import gradio as gr

from h3_ui.media_console import media_controller
from .handlers import DubbingHandlers, DubbingServices
from .page import install_dubbing_page
from .share import install_share_routes
from .tab import build_dubbing_tab


def build_dubbing_workshop(tab, ui, runtime):
    controller = media_controller(runtime)
    actions = DubbingHandlers(DubbingServices(lambda: runtime, None, controller.list_media_paths, gpu=False))
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    build_dubbing_tab(root, actions, gpu=False)
    ui.h3_routes = (*getattr(ui, "h3_routes", ()), partial(install_share_routes, runtime=runtime),
                    partial(install_dubbing_page, handlers=actions, gpu=False))
    return actions
