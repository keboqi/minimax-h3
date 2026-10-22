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
    # The model controller: downloads and model inventory.
    models: Callable[[], Any] = lambda: None
    # The managed output behind a path Gradio may have copied into its cache.
    resolve_result: Callable[[Any], Any] = lambda path: path


def _picker_destinations(label, inputs):
    """One Media "Use in" destination per library-picker input of the engine."""
    return {
        f"{label} · {component.label}": component
        for component in inputs
        if hasattr(component, "h3_library_picker") and not hasattr(component, "h3_reference_slot")
    }


@dataclass(frozen=True)
class EngineFeature:
    """How the workspace builds, binds and saves one engine.

    ``id`` is the engine's id in :data:`h3_app.capabilities.ENGINES` and its Create
    tab id. ``settings_key`` names its view in ``AppViews`` and prefixes its browser
    settings; it differs from ``id`` only for engines whose saved settings predate
    the engine ids. Navigation stays in ``capabilities`` because it must not import
    Gradio.
    """

    id: str
    build: Callable[[gr.Group, "EngineDeps"], Any]  # (root, deps) -> view
    bind: Callable[[Any, "EngineDeps"], Any]  # (view, deps) -> handlers
    settings_key: str = ""
    # Components saved as browser settings, by name (filtered by the persistence allowlist).
    settings: Callable[[Any], dict[str, Any]] | None = None
    # Media "Use in" destinations: label prefix and the view's image inputs.
    library_label: str = ""
    library_inputs: Callable[[Any], tuple] = field(default=lambda view: ())
    library_destinations: Callable[[Any], dict[str, Any]] | None = None

    @property
    def key(self) -> str:
        return self.settings_key or self.id

    def settings_of(self, view) -> dict[str, Any]:
        if self.settings is not None:
            return self.settings(view)
        from .controls import settings_components

        return settings_components(view)

    def destinations_of(self, view) -> dict[str, Any]:
        if self.library_destinations is not None:
            return self.library_destinations(view)
        if not self.library_label:
            return {}
        return _picker_destinations(self.library_label, self.library_inputs(view))
