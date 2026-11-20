"""CPU-only Remake preparation using the same project files as Studio."""
from functools import partial

import gradio as gr

from .api_images import interrupt_api_image
from .engines.feature import EngineDeps
from .engines.remake import FRAME, RemakeHandlers
from .media_console import add_workshop_routes, workshop_column, workshop_media_library
from .remake_page import install_remake_page


def build_remake_workshop(tab, ui, runtime, *, library=None, images=None):
    handlers = RemakeHandlers(EngineDeps(
        runtime_config=lambda: runtime, generation_services=lambda: None,
        stage_file=None, interrupt=None,
    ))
    handlers.library = workshop_media_library(runtime, library)
    with workshop_column(tab):
        gr.HTML(FRAME)
    add_workshop_routes(
        ui,
        partial(
            install_remake_page,
            handlers=handlers,
            gpu=False,
            image_pages=images.pages if images else None,
            interrupt=interrupt_api_image,
        ),
    )
    return handlers
