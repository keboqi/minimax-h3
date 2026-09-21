"""Bind other generation actions."""

from __future__ import annotations

import gradio as gr

from ..contracts import AppComponents, AppServices


def bind_other_generation(
    components: AppComponents, services: AppServices
) -> tuple[
    gr.events.Dependency,
    gr.events.Dependency,
    gr.events.Dependency,
    gr.events.Dependency,
]:
    music3_event = services.bind_music_view(
        components.music3_components,
        enhance_prompt=services.enhance_music3_prompt,
        generate=services.generate_music3,
    )
    api_event = services.bind_api_view(
        components.api_components, generate=services.generate_with_ui_defaults
    )
    yue2_event = services.bind_yue2_view(
        components.yue2_components,
        enhance_prompt=services.enhance_yue2_prompt,
        generate=services.generate_yue2,
    )
    qwen_event = services.bind_qwen_image21_view(
        components.qwen_image21_components,
        enhance_prompt=services.enhance_qwen_image21_prompt,
        generate=services.generate_qwen_image21,
    )

    return music3_event, api_event, yue2_event, qwen_event
