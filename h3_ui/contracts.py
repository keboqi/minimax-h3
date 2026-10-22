"""Explicit application binding contracts."""

from dataclasses import dataclass, fields
from typing import Any
import gradio as gr
from .h3_view import H3View


@dataclass(frozen=True)
class AppComponents(H3View):
    api_components: Any
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

