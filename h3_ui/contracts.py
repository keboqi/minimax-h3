"""Explicit application binding contracts."""

from dataclasses import dataclass, fields
from typing import Any
import gradio as gr
from .h3_view import H3View


@dataclass(frozen=True)
class AppComponents(H3View):
    ltx25_components: Any
    qwen_image21_components: Any
    music3_components: Any
    yue2_components: Any
    api_components: Any
    ltx25_stop: gr.components.Component
    ltx25_status: gr.components.Component
    qwen_image21_stop: gr.components.Component
    qwen_image21_status: gr.components.Component
    music3_stop: gr.components.Component
    music3_status: gr.components.Component
    yue2_stop: gr.components.Component
    yue2_status: gr.components.Component
    api_stop: gr.components.Component
    api_status: gr.components.Component
    system_summary: gr.components.Component
    health: gr.components.Component
    unload_models: gr.components.Component
    memory_status: gr.components.Component
    gallery_components: Any
    gallery_tab: gr.components.Component

    @classmethod
    def from_mapping(cls, values):
        return cls(**{item.name: values[item.name] for item in fields(cls)})

    def as_mapping(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

