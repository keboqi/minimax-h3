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

from ..media_console import workshop_column, workshop_media_library
from .handlers import MediaLibrary, StoryboardHandlers, StoryboardServices
from .tab import build_storyboard_tab


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
                              library=None, remake=None) -> StoryboardHandlers:
    """The GPU-free Storyboard in ``tab``; ``library`` lends its media picker index."""
    handlers = workshop_handlers(runtime, store)
    build_storyboard_tab(workshop_column(tab), ui, handlers, gpu=False, remake=remake)
    handlers.library = workshop_media_library(runtime, library)
    return handlers
