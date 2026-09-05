"""Explicit application binding contracts."""

from dataclasses import dataclass, fields
from typing import Any, Callable
import gradio as gr
from .h3_view import H3View


@dataclass(frozen=True)
class AppComponents(H3View):
    ltx25_model: gr.components.Component
    ltx25_components: Any
    music3_components: Any
    api_components: Any
    ltx25_stop: gr.components.Component
    ltx25_status: gr.components.Component
    music3_stop: gr.components.Component
    music3_status: gr.components.Component
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


@dataclass(frozen=True)
class AppServices:
    resolve_request_settings: Callable[..., Any]
    describe_settings: Callable[..., Any]
    mode_layout_updates: Callable[..., Any]
    result_format_layout_updates: Callable[..., Any]
    compile_trt_video_vae: Callable[..., Any]
    image_vae_frame_updates: Callable[..., Any]
    latent_upscale_layout_updates: Callable[..., Any]
    latent_upscale_method_layout_update: Callable[..., Any]
    bind_ltx_view: Callable[..., Any]
    render_ltx25_workflow_details: Callable[..., Any]
    prepare_ltx25_official_workflow: Callable[..., Any]
    prepare_all_ltx25_official_models: Callable[..., Any]
    render_ltx25_official_model_inventory: Callable[..., Any]
    enhance_ltx25_prompt: Callable[..., Any]
    generate_ltx25: Callable[..., Any]
    AI_POSTPROCESS_OPTIONS: Any
    SEEDVR2_UPSCALE: Any
    LTX25_UPSCALE: Any
    AUTO_RESOLUTION_JS: Any
    auto_resolution_from_start_frame: Callable[..., Any]
    fbcache_preset_defaults: Callable[..., Any]
    resolution_info_preview: Callable[..., Any]
    resolution_control_updates: Callable[..., Any]
    input_image_frame_preset_updates: Callable[..., Any]
    upscale_selected_input_images: Callable[..., Any]
    generate_for_ui: Callable[..., Any]
    select_all_image_frames: Callable[..., Any]
    save_selected_image_frames: Callable[..., Any]
    prompt_writer_backend_visibility: Callable[..., Any]
    enhance_h3_prompt: Callable[..., Any]
    bind_music_view: Callable[..., Any]
    enhance_music3_prompt: Callable[..., Any]
    generate_music3: Callable[..., Any]
    bind_api_view: Callable[..., Any]
    generate_with_ui_defaults: Callable[..., Any]
    interrupt: Callable[..., Any]
    refresh_backend_views: Callable[..., Any]
    unload_all_models: Callable[..., Any]
    bind_gallery_view: Callable[..., Any]
    refresh_gallery: Callable[..., Any]
    select_gallery_video: Callable[..., Any]
    import_gallery_video: Callable[..., Any]
    postprocess_selected_gallery_video: Callable[..., Any]
    delete_selected_gallery_video: Callable[..., Any]
    empty_generated_gallery: Callable[..., Any]

    @classmethod
    def from_mapping(cls, values):
        return cls(**{item.name: values[item.name] for item in fields(cls)})

    def as_mapping(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}
