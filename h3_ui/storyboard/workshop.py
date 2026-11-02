"""Storyboard in the Prism Workshop: everything except generating, on CPU.

Storyboards live in the shared workspace database, so a storyboard prepared here
opens on the GPU endpoint, where **Generate all remaining scenes** makes its takes.
"""

from __future__ import annotations

from functools import partial

import gradio as gr

from h3_app import storyboard as sb
from h3_app.config import RuntimeConfig
from h3_app.provenance import write_snapshot

from ..image_library import bind_image_library
from .bindings import bind_storyboard
from .handlers import StoryboardHandlers, StoryboardServices
from .view import StoryboardView, build_storyboard_view


def workshop_handlers(runtime: RuntimeConfig, store) -> StoryboardHandlers:
    from h3_app.prompt_writers.h3 import enhance_h3_prompt

    service = sb.StoryboardService(store, lambda: runtime)
    return StoryboardHandlers(StoryboardServices(
        service=lambda: service,
        generate_take=None,
        request_defaults=sb.request_defaults,
        publish=write_snapshot,
        enhance_prompt=partial(enhance_h3_prompt, runtime=runtime),
    ))


def build_storyboard_workshop(tab: gr.Tab, ui: gr.Blocks, runtime: RuntimeConfig, store, *,
                              library=None) -> StoryboardView:
    """Build and bind the GPU-free Storyboard in ``tab``; ``library`` lends its media picker index."""
    with tab, gr.Column(elem_classes=["h3-workspace"]) as root:
        pass
    view = build_storyboard_view(root, gpu=False)
    bind_storyboard(view, workshop_handlers(runtime, store), gpu=False, load_trigger=ui.load)
    if library is not None:
        from h3_ui.media_console import media_controller

        controller = media_controller(runtime)
        for component in (view.cast_image, view.first_image, view.last_image, view.ref_images):
            bind_image_library(
                component,
                index=library.search.h3_asset_index,
                refresh_page=controller.refresh_media_page,
                validate_path=controller.gallery_media_download_path,
            )
    return view
