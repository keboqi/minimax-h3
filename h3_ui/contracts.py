"""Explicit application binding contracts."""

from dataclasses import dataclass, fields
from typing import Any, Callable
import gradio as gr
from .engines.feature import EngineDeps
from .h3.view import H3View


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


@dataclass(frozen=True)
class BootstrapCatalog:
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS: Any
    AUTO_SOL_TOKEN_THRESHOLD: Any
    COMFY_DIR: Any
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS: Any
    DEFAULT_GEMINI_PROMPT_MODEL: Any
    DEFAULT_INPUT_IMAGE_FRAME_PRESET: Any
    DEFAULT_LOCAL_PROMPT_BASE_MODEL: Any
    DEFAULT_LTX25_MODEL: Any
    DEFAULT_PROMPT_WRITER_BACKEND: Any
    DEFAULT_UPSCALE_RESOLUTION: Any
    DEFAULT_VIDEO_BATCH_COUNT: Any
    DRAFT_RESOLUTIONS: Any
    FAST_RESOLUTIONS: Any
    GEMINI_PROMPT_MODELS: Any
    GENERATION_POSTPROCESS_OPTIONS: Any
    H3_LATENT_UPSCALER_MODEL_CHOICES: Any
    H3_LATENT_UPSCALE_METHODS: Any
    H3_LATENT_UPSCALE_SPLIT: Any
    H3_TEXT_ENCODER_CHOICES: Any
    IMAGE_VAE_CHOICES: Any
    INPUT_IMAGE_FRAME_PRESETS: Any
    INPUT_IMAGE_UPSCALE_SLOTS: Any
    LARGE_RESOLUTIONS: Any
    OPENAI_PROMPT_MODEL: Any
    LOCAL_PROMPT_BASE_MODELS: Any
    LTX25_DEFAULTS: Any
    LTX25_MODEL_CHOICES: Any
    LTX25_WORKFLOWS: Any
    MAX_IMAGE_FRAMES: Any
    MAX_VIDEO_BATCH_COUNT: Any
    MIN_IMAGE_FRAMES: Any
    MIN_VIDEO_BATCH_COUNT: Any
    MODEL_PROFILE_CHOICES: Any
    MUSIC3_DEFAULTS: Any
    MUSIC3_MODEL_CHOICES: Any
    POSTPROCESS_OPTIONS: Any
    PROMPT_WRITER_BACKENDS: Any
    QWEN_IMAGE21_DEFAULTS: Any
    QWEN_IMAGE21_MODEL_CHOICES: Any
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES: Any
    RESULT_FORMATS: Any
    SEEDVR2_MODEL_CHOICES: Any
    SERVER_ATTENTION_BACKEND: Any
    SERVER_DENSE_ATTENTION_BACKEND: Any
    SLA_PRESET_INPUTS: Any
    TURBO_SETTINGS: Any
    UI_DEFAULTS: Any
    UPSCALE_RESOLUTION_PRESETS: Any
    YUE2_DEFAULTS: Any
    YUE2_MODEL_CHOICES: Any


@dataclass(frozen=True)
class BootstrapServices:
    api_get: Callable
    api_guide: Callable
    api_post: Callable
    auto_resolution_from_start_frame: Callable
    backend_status: Callable
    compact_backend_status: Callable
    compact_settings_summary: Callable
    compile_trt_video_vae: Callable
    delete_selected_gallery_media: Callable
    describe_settings: Callable
    empty_generated_media_gallery: Callable
    enhance_h3_prompt: Callable
    fbcache_preset_defaults: Callable
    generate_for_ui: Callable
    generate_with_ui_defaults: Callable
    drama3_actions: Any
    dubbing_actions: Any
    engine_deps: EngineDeps
    generation_readiness_state: Callable
    gallery_media_download_path: Callable
    image_vae_frame_updates: Callable
    import_gallery_media: Callable
    input_image_frame_preset_updates: Callable
    interrupt: Callable
    latent_upscale_layout_updates: Callable
    latent_upscale_method_layout_update: Callable
    list_media_paths: Callable
    mode_help: Callable
    postprocess_selected_gallery_media: Callable
    prompt_writer_backend_visibility: Callable
    reference_prompt_help: Callable
    refresh_backend_views: Callable
    refresh_media_page: Callable
    render_snapshot: Callable
    resolution_choice_updates: Callable
    resolution_control_updates: Callable
    resolution_info_preview: Callable
    resolution_summary: Callable
    resolve_request_settings: Callable
    result_format_layout_updates: Callable
    result_settings_for_media: Callable
    save_selected_image_frames: Callable
    select_all_image_frames: Callable
    select_gallery_media: Callable
    unload_all_models: Callable
    upscale_selected_input_images: Callable
