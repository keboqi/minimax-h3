"""Bind other generation actions."""

from __future__ import annotations

import gradio as gr

from typing import TYPE_CHECKING

from ..api_bindings import bind_api_view
from ..contracts import AppComponents
from ..music_bindings import bind_music_view
from ..qwen_bindings import bind_qwen_image21_view
from ..yue2_bindings import bind_yue2_view

if TYPE_CHECKING:
    from ..bootstrap import BootstrapServices


def bind_other_generation(components: AppComponents, services: BootstrapServices) -> tuple[
    gr.events.Dependency,
    gr.events.Dependency,
    gr.events.Dependency,
    gr.events.Dependency,
]:
    music3_event = bind_music_view(
        components.music3_components,
        enhance_prompt=services.enhance_music3_prompt,
        generate=services.generate_music3,
    )
    api_event = bind_api_view(
        components.api_components, generate=services.generate_with_ui_defaults
    )
    yue2_event = bind_yue2_view(
        components.yue2_components,
        enhance_prompt=services.enhance_yue2_prompt,
        generate=services.generate_yue2,
    )
    qwen_event = bind_qwen_image21_view(
        components.qwen_image21_components,
        enhance_prompt=services.enhance_qwen_image21_prompt,
        generate=services.generate_qwen_image21,
    )

    return music3_event, api_event, yue2_event, qwen_event
