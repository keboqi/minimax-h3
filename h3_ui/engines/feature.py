"""The record each Create engine registers, and the dependencies it is built with."""

from dataclasses import dataclass, field
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class EngineDeps:
    """Runtime services an engine's handlers need, passed in when the UI is built.

    Callables are late-bound so tests and the application can swap the runtime.
    """

    runtime_config: Callable[[], Any]
    generation_services: Callable[[], Any]
    stage_file: Callable[..., str]
    interrupt: Callable


@dataclass(frozen=True)
class EngineFeature:
    """How the workspace builds, binds and saves one engine.

    ``id`` is the engine's id in :data:`h3_app.capabilities.ENGINES`, its Create tab id
    and its browser-settings prefix. Navigation stays in ``capabilities`` because it
    must not import Gradio.
    """

    id: str
    build: Callable[[gr.Group], Any]  # root -> view
    bind: Callable[[Any, EngineDeps], Any]  # (view, deps) -> handlers
    # Media "Use in" destinations: label prefix and the view's image inputs.
    library_label: str = ""
    library_inputs: Callable[[Any], tuple] = field(default=lambda view: ())
