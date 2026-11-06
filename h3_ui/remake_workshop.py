"""CPU-only Remake preparation using the same project files as Studio."""
from functools import partial

import gradio as gr

from .engines.feature import EngineDeps
from .engines.remake import FRAME, RemakeHandlers
from .remake_page import install_remake_page


def build_remake_workshop(tab, ui, runtime, *, library=None):
    handlers = RemakeHandlers(EngineDeps(
        runtime_config=lambda: runtime, generation_services=lambda: None,
        stage_file=None, interrupt=None,
    ))
    if library is not None:
        from .media_console import media_controller
        from .storyboard.handlers import MediaLibrary

        controller = media_controller(runtime)
        handlers.library = MediaLibrary(library.search.h3_asset_index, controller.refresh_media_page,
                                        controller.gallery_media_download_path)
    with tab, gr.Column(elem_classes=["h3-workspace"]):
        gr.HTML(FRAME)
    ui.h3_routes = (*getattr(ui, "h3_routes", ()),
                    partial(install_remake_page, handlers=handlers, gpu=False))
    return handlers
