"""Bind the H3 API tab's generation."""

from __future__ import annotations

import gradio as gr

from typing import TYPE_CHECKING

from ..api_bindings import bind_api_view
from ..contracts import AppComponents

if TYPE_CHECKING:
    from ..contracts import BootstrapServices


def bind_other_generation(components: AppComponents, services: BootstrapServices) -> gr.events.Dependency:
    return bind_api_view(components.api_components, generate=services.generate_with_ui_defaults)
