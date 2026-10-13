"""Bind ltx actions."""

from __future__ import annotations

import gradio as gr

from typing import TYPE_CHECKING

from ..contracts import AppComponents
from ..ltx_bindings import bind_ltx_view

if TYPE_CHECKING:
    from ..bootstrap import BootstrapServices


def bind_ltx(components: AppComponents, services: BootstrapServices) -> gr.events.Dependency:
    ltx25_event = bind_ltx_view(
        components.ltx25_components,
        render_workflow=services.render_ltx25_workflow_details,
        prepare_workflow=services.prepare_ltx25_official_workflow,
        prepare_all_models=services.prepare_all_ltx25_official_models,
        render_inventory=services.render_ltx25_official_model_inventory,
        enhance_prompt=services.enhance_ltx25_prompt,
        generate=services.generate_ltx25,
    )

    return ltx25_event
