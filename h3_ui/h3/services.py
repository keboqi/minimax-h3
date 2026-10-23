"""Catalogs and callbacks the H3 view and its sections are built with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class H3ViewServices:
    LTX25_MODEL_CHOICES: Mapping[str, Any]
    RESULT_FORMATS: Sequence[str]
    MODEL_PROFILE_CHOICES: Sequence[str]
    H3_TEXT_ENCODER_CHOICES: Mapping[str, Any]
    IMAGE_VAE_CHOICES: Sequence[str]
    mode_help: Callable[[str], str]
    PROMPT_WRITER_BACKENDS: Sequence[str]
    DEFAULT_PROMPT_WRITER_BACKEND: str
    LOCAL_PROMPT_BASE_MODELS: Mapping[str, Any]
    DEFAULT_LOCAL_PROMPT_BASE_MODEL: str
    GEMINI_PROMPT_MODELS: Sequence[str]
    DEFAULT_GEMINI_PROMPT_MODEL: str
    OPENAI_PROMPT_MODEL: str
    reference_prompt_help: Callable[[], str]
    INPUT_IMAGE_UPSCALE_SLOTS: Sequence[str]
    INPUT_IMAGE_FRAME_PRESETS: Mapping[str, Any]
    DEFAULT_INPUT_IMAGE_FRAME_PRESET: str
    SEEDVR2_MODEL_CHOICES: Mapping[str, Any]
    compact_settings_summary: Callable[..., Any]
    DEFAULT_LTX25_MODEL: str
    generation_readiness_state: Callable[..., Any]
    MIN_IMAGE_FRAMES: int
    MAX_IMAGE_FRAMES: int
    FAST_RESOLUTIONS: Mapping[str, Any]
    DRAFT_RESOLUTIONS: Mapping[str, Any]
    LARGE_RESOLUTIONS: Mapping[str, Any]
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS: Mapping[str, Any]
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS: str
    resolution_summary: Callable[..., Any]
    MIN_VIDEO_BATCH_COUNT: int
    MAX_VIDEO_BATCH_COUNT: int
    DEFAULT_VIDEO_BATCH_COUNT: int
    TURBO_SETTINGS: Mapping[str, Any]
    SERVER_ATTENTION_BACKEND: str
    AUTO_SOL_TOKEN_THRESHOLD: int
    SERVER_DENSE_ATTENTION_BACKEND: str
    SLA_PRESET_INPUTS: Mapping[str, Any]
    H3_LATENT_UPSCALER_MODEL_CHOICES: Mapping[str, Any]
    H3_LATENT_UPSCALE_METHODS: Sequence[str]
    H3_LATENT_UPSCALE_SPLIT: str
    GENERATION_POSTPROCESS_OPTIONS: Sequence[str]
    UPSCALE_RESOLUTION_PRESETS: Mapping[str, Any]
    DEFAULT_UPSCALE_RESOLUTION: str
