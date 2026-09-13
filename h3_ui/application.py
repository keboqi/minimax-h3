"""Application composition and temporary public compatibility adapters."""

# ruff: noqa: F401, E402
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import html
import inspect
import json
import math
import mimetypes
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest.mock
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable
from urllib.parse import quote, urlsplit, urlunsplit

import aiohttp
import gradio as gr
from gradio import networking as gradio_networking
import httpx
import requests
import uvicorn
import websocket
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from h3_models import (
    DEFAULT_H3_LATENT_UPSCALER_MODEL,
    DEFAULT_MUSIC3_MODEL,
    DEFAULT_LTX25_MODEL,
    DEFAULT_SEEDVR2_MODEL,
    H3_LATENT_UPSCALER_MODEL_CHOICES,
    H3_TEXT_ENCODER_CHOICES,
    LTX25_MODEL_CHOICES,
    LTX25_ICLORA_MODEL_KEYS,
    LTX25_SHARED_MODEL_KEYS,
    MIN_VALID_MODEL_BYTES,
    MODEL_SPECS,
    MUSIC3_MODEL_CHOICES,
    MUSIC3_SHARED_MODEL_KEYS,
    PROFILE_LABELS,
    SEEDVR2_MODEL_CHOICES,
    TRT_VAE_ENGINE_BUILD_ID,
    TRT_VAE_ENGINE_MARKER,
    TRT_VAE_RUNTIME_MODEL_KEYS,
    resolve_hf_token,
    stale_model_keys,
    sync_models,
)
from h3_requirements import LTX25_WORKFLOW_FILENAMES, SWIFTVR_HF_REPO
from h3_prompt_rewriter import (
    BASE_MODEL_CHOICES as LOCAL_PROMPT_BASE_MODELS,
    DEFAULT_BASE_MODEL_LABEL as DEFAULT_LOCAL_PROMPT_BASE_MODEL,
    resolution_for_size as local_prompt_resolution,
    rewrite_prompt as rewrite_local_h3_prompt,
    task_for_inputs as local_prompt_task,
    unload_prompt_rewriter,
)
from h3_ui.presentation import (
    backend_status_html,
    generation_readiness as generation_readiness_state,
    mode_presentation,
    result_format_presentation,
)
from h3_ui.bindings import (
    bind_api_view,
    bind_gallery_view,
    bind_interrupts,
    bind_ltx_view,
    bind_music_view,
    bind_preflight,
    bind_summary,
)
from h3_ui.app_bindings import bind_app
from h3_ui.contracts import AppComponents, AppServices
from h3_ui.h3_view import build_h3_view, H3ViewServices
from h3_ui.layout import create_app_views
from h3_ui.persistence import bind_browser_settings
from h3_ui.ltx_view import build_ltx_view
from h3_ui.styles import H3_SETUP_CSS, H3_UI_CSS
from h3_ui.views import build_api_view, build_gallery_view, build_music_view

from dataclasses import asdict
from h3_app.settings import (
    GenerationRequest,
    ResolutionContext,
    resolve_settings,
    preset_settings,
    PRESETS,
)
from h3_app.contracts import GenerationArguments, GENERATION_FIELDS
from h3_app import media as media_store
from h3_app import server as server_routes
from h3_app.server import (
    _proxy_headers,
    _rewrite_comfy_text,
    _comfy_upstream_path,
    _append_set_cookies,
)
from h3_app.graph import Graph
from h3_app.comfy import ComfyClient
from h3_app.jobs import (
    gpu_maintenance,
    JOBS,
    CURRENT_JOB,
    check_cancelled,
    scoped_graph,
)
from h3_app.processes import run_process
from h3_app.provenance import (
    RUN_CONTEXT,
    write_snapshot,
    read_snapshot,
    render_snapshot,
    snapshot_path,
)
from h3_ui.settings_presentation import render_settings

from h3_app.errors import (
    H3Error,
)
from h3_app.catalog import (
    AI_POSTPROCESS_OPTIONS,
    AUDIO_EXTENSIONS,
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
    AUTO_RESOLUTION_PIXEL_CAP,
    AUTO_SOL_TOKEN_THRESHOLD,
    CHUNK_FEED_FORWARD_NODE,
    COMFY_PROXY_PATH,
    COMFY_UPSCALE_OPTIONS,
    COMFY_POSTPROCESS_OPTIONS,
    LTX25_DECOMPRESSION,
    LTX25_DEBLUR,
    LTX25_POSTPROCESS_MODELS,
    LTX25_RESTORATION_OPTIONS,
    CORE_LORA_LOADER_NODE,
    CORE_SAMPLER_NODE,
    DEFAULT_ACCELERATOR,
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
    DEFAULT_FBCACHE_END,
    DEFAULT_FBCACHE_MAX_HITS,
    DEFAULT_FBCACHE_PRESET,
    DEFAULT_FBCACHE_START,
    DEFAULT_FBCACHE_TEMPORAL_GUARD,
    DEFAULT_FBCACHE_THRESHOLD,
    DEFAULT_GEMINI_PROMPT_MODEL,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_IMAGE_VAE,
    DEFAULT_INPUT_IMAGE_FRAME_PRESET,
    DEFAULT_PROMPT_WRITER_BACKEND,
    DEFAULT_RESULT_FORMAT,
    DEFAULT_SLA_PRESET,
    DEFAULT_TURBO,
    DEFAULT_UPSCALE_RESOLUTION,
    DEFAULT_VIDEO_BATCH_COUNT,
    DRAFT_RESOLUTIONS,
    FAST_RESOLUTIONS,
    FUSED_MODULATION_NODE,
    GEMINI_API_ROOT,
    GEMINI_PROMPT_MODELS,
    GENERATION_POSTPROCESS_OPTIONS,
    H3_COMBINE_AV_LATENT_NODE,
    H3_CONDITIONING_CACHE_NODE,
    H3_IMAGE_SLICES_NODE,
    H3_LATENT_UPSCALER_NODE,
    H3_LATENT_UPSCALE_METHODS,
    H3_LATENT_UPSCALE_SCALE,
    H3_LATENT_UPSCALE_SPLIT,
    H3_LATENT_UPSCALE_STANDARD,
    H3_NVENC_SAVE_NODE,
    H3_SEMANTIC_BRIDGE_NODE,
    H3_SEPARATE_AV_LATENT_NODE,
    H3_SIGMA_SHIFT_NODE,
    H3_SINGLE_FRAME_VAE_LOADER_NODE,
    H3_SPLIT_SPATIAL_PARAMS_NODE,
    H3_SPLIT_TEMPORAL_PARAMS_NODE,
    H3_SPLIT_UPSCALE_NODE,
    H3_STAGE_OFFLOAD_NODE,
    H3_STAGE_OFFLOAD_POLICY_NODE,
    IMAGE_EXTENSIONS,
    IMAGE_VAE_CHOICES,
    INPUT_IMAGE_FRAME_PRESETS,
    INPUT_IMAGE_UPSCALE_SLOTS,
    LARGE_RESOLUTIONS,
    LARRY_TURBO,
    LARRY_TURBO_LORA_NODE,
    LARRY_TURBO_SAMPLER_NODE,
    LIGHTNING_API_ROOT,
    LIGHTNING_PROMPT_MODEL,
    LIGHTX2V_4STEP_TURBO,
    LIGHTX2V_8STEP_TURBO,
    LIGHTX2V_BYPASS_LORA_NODE,
    LTX25_DEFAULTS,
    LTX25_SIGMAS,
    LTX25_UPSCALE,
    LTX25_WORKFLOWS,
    LTX25_WORKFLOW_COMMON_MODEL_KEYS,
    MAX_IMAGE_FRAMES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MAX_VIDEO_BATCH_COUNT,
    MIN_IMAGE_FRAMES,
    MIN_VIDEO_BATCH_COUNT,
    MODEL_PROFILE_CHOICES,
    MUSIC3_DEFAULTS,
    NATIVE_PIXEL_CAP,
    OFFICIAL_IMAGE_VAE,
    POSTPROCESS_OPTIONS,
    PROMPT_WRITER_BACKENDS,
    REFERENCE_VIDEO_TRANSCODE_CACHE_VERSION,
    RESOLUTION_TIERS,
    RESULT_FORMATS,
    SAGE_ATTENTION_NODE,
    SAMPLING_PRESETS,
    SAMPLING_PRESET_TEXT_ENCODERS,
    SEEDVR2_UPSCALE,
    SINGLE_FRAME_IMAGE_VAE,
    SLA_ATTENTION_NODE,
    SLA_PRESET_INPUTS,
    SOL_ATTENTION_NODE,
    SPECTRUM_DEFAULT_INPUTS,
    STAGED_INPUT_HASH_CHUNK_BYTES,
    SWIFTVR_UPSCALE,
    TURBO_SETTINGS,
    TurboSpec,
    UI_DEFAULTS,
    UPSCALE_RESOLUTION_PRESETS,
    UVICORN_WEBSOCKET_OPTIONS,
    VIDEO_EXTENSIONS,
)
from h3_app.policy import (
    H3SplitUpscaleConfig,
    active_fl2va_voice_references,
    auto_resolution_pixel_cap,
    collect_reference_slots,
    estimate_packed_tokens,
    frame_length,
    h3_latent_upscale_dimensions,
    h3_latent_upscaler_settings,
    image_sampling_length,
    is_lightx2v_turbo_lora,
    lightx2v_uses_768p_schedule,
    ltx25_frame_length,
    normalize_image_vae,
    normalize_paths,
    normalize_result_format,
    normalize_turbo_variant,
    resolution_choice_values,
    resolution_for_aspect_ratio,
    resolution_summary,
    resolve_cache_policy,
    resolve_h3_latent_upscale_method,
    resolve_h3_split_upscale_config,
    resolve_sla_preset,
    selected_image_sampling_length,
    single_frame_image_sampling_length,
    snap32,
    snap64,
    snap_to_grid,
    turbo_sampler_name,
    turbo_steps_for,
    turbo_strength_for,
    turbo_uses_custom_nodes,
    upscale_target_dimensions,
    validate_image_frame_count,
    validate_resolution,
    video_latent_t,
)
from h3_app.model_types import (
    ModelConfig,
    ModelProfile,
    h3_text_encoder_settings,
    ltx25_model_keys,
    ltx25_model_names,
    ltx25_official_inventory_keys,
    ltx25_workflow_entry,
    ltx25_workflow_model_keys,
    model_file_is_ready,
    music3_model_keys,
    seedvr2_upscale_model_names,
    trt_vae_engine_name,
)
from h3_app import model_service
import h3_app.workflows.ltx as ltx_workflow
import h3_app.workflows.h3 as h3_workflow
import h3_app.workflows.upscale as upscale_workflow
from h3_app.execution import ExecutionRunner, PromptId, Submission
from h3_app import prompt_service
import h3_app.staging as staging
from h3_app import media_tools
import h3_app.swiftvr as swiftvr
from h3_app import gallery_store
from h3_app import outputs
from h3_app.outputs import OutputContext
from h3_app.config import RuntimeConfig
from h3_app.provenance import copy_media
from dataclasses import replace

from h3_app.workflows.music import (
    build_music3_graph,
    required_music3_nodes,
)
from h3_app.status import (
    StageTimings,
    graph_class_types,
    node_stage,
    progress_status,
)
from h3_app.media_types import (
    UpscaleClipBatch,
    VideoMetadata,
)
from h3_app.generation import (
    requests as generation_requests,
    services as generation_services,
)
from h3_app.generation import (
    h3 as h3_generation,
    ltx as ltx_generation,
    music as music_generation,
)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
RUNTIME = RuntimeConfig.from_environment(SCRIPT_DIR)


COMFY_URL = RUNTIME.comfy_url

# Modal supports RFC 6455 WebSockets but deliberately does not support the
# RFC 7692 permessage-deflate extension. Uvicorn enables that extension by
# default, which makes Modal close the connection as soon as ComfyUI sends its
# initial status frame. Keep the transport deterministic on Modal and local
# standalone servers by using the dependency we install and disabling RFC 7692.

COMFY_DIR = RUNTIME.comfy_dir
INPUT_DIR = RUNTIME.input_dir
OUTPUT_DIR = RUNTIME.output_dir
MODELS_CONFIG = RUNTIME.models_config
SERVER_ATTENTION_BACKEND = RUNTIME.attention_backend
SERVER_DENSE_ATTENTION_BACKEND = RUNTIME.dense_attention_backend
SERVER_MEMORY_PROFILE = RUNTIME.memory_profile


PROMPT_ENHANCER_SYSTEM_PATH = RUNTIME.prompt_system_path
PROMPT_ENHANCER_SYSTEMS = RUNTIME.prompt_systems


LTX25_WORKFLOW_TEMPLATE_DIR = RUNTIME.workflow_dir


# The official H3 workflow uses a 768×1344 pixel-area native canvas. Larger
# entries are still available for workflows that can handle them.


# Preset dimensions use the model-required 32-pixel grid (1080p aligns to 1088).


REQUEST_TIMEOUT = RUNTIME.request_timeout
GENERATION_TIMEOUT = RUNTIME.generation_timeout
POLL_SECONDS = RUNTIME.poll_seconds
OUTPUTS_DIR = RUNTIME.outputs_dir
GALLERY_THUMBNAILS_DIR = RUNTIME.gallery_thumbnails_dir
GALLERY_LIMIT = RUNTIME.gallery_limit


GALLERY_METADATA_CACHE_LIMIT = RUNTIME.gallery_metadata_cache_limit

HTTP = requests.Session()


def _runtime_config() -> RuntimeConfig:
    """Build an explicit service context from the legacy application aliases."""
    return replace(
        RUNTIME,
        input_root=INPUT_DIR,
        output_root=OUTPUT_DIR,
        thumbnail_root=GALLERY_THUMBNAILS_DIR,
        comfy_url=COMFY_URL,
        comfy_dir=COMFY_DIR,
        models_config=MODELS_CONFIG,
        attention_backend=SERVER_ATTENTION_BACKEND,
        dense_attention_backend=SERVER_DENSE_ATTENTION_BACKEND,
        memory_profile=SERVER_MEMORY_PROFILE,
        request_timeout=REQUEST_TIMEOUT,
        generation_timeout=GENERATION_TIMEOUT,
        poll_seconds=POLL_SECONDS,
        outputs_dir=OUTPUTS_DIR,
        gallery_limit=GALLERY_LIMIT,
        gallery_metadata_cache_limit=GALLERY_METADATA_CACHE_LIMIT,
    )


def build_server(demo: gr.Blocks, allowed_paths: list[str]) -> FastAPI:
    return server_routes.build_server(
        demo,
        allowed_paths,
        server_routes.ServerConfig(
            COMFY_URL,
            OUTPUT_DIR,
            OUTPUTS_DIR,
            LTX25_WORKFLOWS,
            LTX25_WORKFLOW_TEMPLATE_DIR,
            VIDEO_EXTENSIONS,
            H3_SETUP_CSS,
        ),
    )


def _gemini_api_key(temporary_key: str | None) -> str:
    return prompt_service._gemini_api_key(temporary_key)


def _lightning_api_key(temporary_key: str | None) -> str:
    return prompt_service._lightning_api_key(temporary_key)


def _uploaded_media_path(value: Any) -> Path | None:
    return prompt_service._uploaded_media_path(value)


def _gemini_mime_type(path: Path) -> str:
    return prompt_service._gemini_mime_type(path)


def _gemini_error(response: requests.Response, action: str) -> H3Error:
    return prompt_service._gemini_error(response, action)


def _upload_gemini_file(
    session: requests.Session,
    path: Path,
    api_key: str,
) -> dict[str, Any]:
    return prompt_service._upload_gemini_file(session, path, api_key)


def _wait_for_gemini_file(
    session: requests.Session,
    file_info: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    return prompt_service._wait_for_gemini_file(session, file_info, api_key)


def _active_prompt_media(
    mode: str,
    first_image: Any,
    last_image: Any,
    reference_images: Iterable[Any],
    reference_videos: Iterable[Any],
    reference_audios: Iterable[Any],
) -> list[tuple[str, Path]]:
    return prompt_service._active_prompt_media(
        mode,
        first_image,
        last_image,
        reference_images,
        reference_videos,
        reference_audios,
    )


def _enhance_prompt_from_media(
    *,
    prompt: str,
    model: str,
    temporary_api_key: str,
    target: str,
    system_path: Path,
    media_values: Iterable[tuple[str, Any]],
    context: str,
) -> tuple[str, str]:
    return prompt_service._enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target=target,
        system_path=system_path,
        media_values=media_values,
        context=context,
    )


def enhance_music3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
) -> tuple[str, str, str]:
    return prompt_service.enhance_music3_prompt(
        prompt,
        model,
        temporary_api_key,
        lyrics,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        runtime=_runtime_config(),
    )


def enhance_ltx25_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    start_image: Any,
    middle_image: Any,
    end_image: Any,
    duration: float,
    width: int,
    height: int,
) -> tuple[str, str]:
    return prompt_service.enhance_ltx25_prompt(
        prompt,
        model,
        temporary_api_key,
        mode,
        start_image,
        middle_image,
        end_image,
        duration,
        width,
        height,
        runtime=_runtime_config(),
    )


def _enhance_h3_prompt_with_gemini(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
) -> tuple[str, str]:
    return prompt_service._enhance_h3_prompt_with_gemini(
        prompt,
        model,
        temporary_api_key,
        mode,
        first_image,
        last_image,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        ref_image_4,
        ref_image_5,
        ref_image_6,
        ref_image_7,
        ref_image_8,
        ref_image_9,
        ref_video_1,
        ref_video_2,
        ref_video_3,
        ref_audio_1,
        ref_audio_2,
        ref_audio_3,
        duration,
        width,
        height,
        result_format,
        image_frames,
        runtime=_runtime_config(),
    )


def _enhance_h3_prompt_with_lightning(
    prompt: str,
    temporary_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
) -> tuple[str, str]:
    return prompt_service._enhance_h3_prompt_with_lightning(
        prompt,
        temporary_api_key,
        mode,
        first_image,
        last_image,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        ref_image_4,
        ref_image_5,
        ref_image_6,
        ref_image_7,
        ref_image_8,
        ref_image_9,
        ref_video_1,
        ref_video_2,
        ref_video_3,
        ref_audio_1,
        ref_audio_2,
        ref_audio_3,
        duration,
        width,
        height,
        result_format,
        image_frames,
        runtime=_runtime_config(),
    )


def fl2va_prompt_voice_context(prompt: str, mode: str, *slots: Any):
    return prompt_service.fl2va_prompt_voice_context(prompt, mode, *slots)


@gpu_maintenance("prompt-enhance")
def enhance_h3_prompt(
    prompt: str,
    backend: str,
    local_base_model: str,
    local_max_new_tokens: int,
    local_temperature: float,
    local_top_p: float,
    local_greedy: bool,
    local_seed: int,
    gemini_model: str,
    gemini_api_key: str,
    lightning_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    fl2va_audio_1: Any = None,
    fl2va_audio_2: Any = None,
    fl2va_audio_3: Any = None,
) -> tuple[str, str]:
    return prompt_service.enhance_h3_prompt(
        prompt,
        backend,
        local_base_model,
        local_max_new_tokens,
        local_temperature,
        local_top_p,
        local_greedy,
        local_seed,
        gemini_model,
        gemini_api_key,
        lightning_api_key,
        mode,
        first_image,
        last_image,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        ref_image_4,
        ref_image_5,
        ref_image_6,
        ref_image_7,
        ref_image_8,
        ref_image_9,
        ref_video_1,
        ref_video_2,
        ref_video_3,
        ref_audio_1,
        ref_audio_2,
        ref_audio_3,
        duration,
        width,
        height,
        result_format,
        image_frames,
        fl2va_audio_1,
        fl2va_audio_2,
        fl2va_audio_3,
        runtime=_runtime_config(),
    )


def prompt_writer_backend_visibility(backend: str) -> tuple[Any, Any, Any]:
    """Show only controls belonging to the selected prompt-writer backend."""
    local_selected = backend == "Local MiniMax-H3 8B"
    return (
        gr.update(visible=local_selected),
        gr.update(visible=backend == "Gemini"),
        gr.update(visible=backend == "Lightning AI"),
    )


def load_model_config() -> ModelConfig:
    return model_service.load_model_config(runtime=_runtime_config())


def trt_vae_decoder_paths(models: ModelConfig) -> tuple[Path, Path, Path]:
    return model_service.trt_vae_decoder_paths(models, runtime=_runtime_config())


def _load_trt_vae_compiler(node_path: Path) -> Any:
    return model_service._load_trt_vae_compiler(node_path, runtime=_runtime_config())


def trt_vae_runtime_fingerprint(models: ModelConfig | None = None) -> str:
    return model_service.trt_vae_runtime_fingerprint(models, runtime=_runtime_config())


def is_trt_engine_loadable(engine_path: Path) -> bool:
    return model_service.is_trt_engine_loadable(engine_path, runtime=_runtime_config())


def trt_vae_engine_is_current(models: ModelConfig) -> bool:
    return model_service.trt_vae_engine_is_current(models, runtime=_runtime_config())


def ensure_h3_text_encoder(models: ModelConfig, model_choice: str) -> tuple[str, bool]:
    return model_service.ensure_h3_text_encoder(
        models, model_choice, runtime=_runtime_config()
    )


def text_encoder_offload_update(model_choice: str):
    """Keep the UI memory option aligned with the selected encoder tier."""
    bf16 = H3_TEXT_ENCODER_CHOICES.get(str(model_choice)) == "text_encoder_bf16"
    return gr.update(
        value=bf16,
        interactive=not bf16,
        info=(
            "Required for the 51.5 GB BF16 encoder; models are unloaded between "
            "text encoding, diffusion, latent upscaling, and VAE decoding."
            if bf16
            else "Unload resident models at each H3 stage boundary to reduce peak VRAM."
        ),
    )


def ensure_h3_semantic_bridge() -> None:
    return model_service.ensure_h3_semantic_bridge(runtime=_runtime_config())


def ensure_h3_latent_upscaler_model(model_choice: str) -> bool:
    return model_service.ensure_h3_latent_upscaler_model(
        model_choice, runtime=_runtime_config()
    )


def ensure_profile_model(
    profile_key: str,
    profile: ModelProfile,
    mode: str,
) -> bool:
    return model_service.ensure_profile_model(
        profile_key, profile, mode, runtime=_runtime_config()
    )


def ensure_turbo_lora(models: ModelConfig, turbo_variant: str, mode: str) -> bool:
    return model_service.ensure_turbo_lora(
        models, turbo_variant, mode, runtime=_runtime_config()
    )


def ensure_int8_video_vae(models: ModelConfig) -> bool:
    return model_service.ensure_int8_video_vae(models, runtime=_runtime_config())


def ensure_trt_video_vae(
    models: ModelConfig,
    *,
    require_engine: bool = True,
) -> bool:
    return model_service.ensure_trt_video_vae(
        models, require_engine=require_engine, runtime=_runtime_config()
    )


def _build_trt_video_vae_engine(
    models: ModelConfig,
    progress: Any,
) -> None:
    return model_service._build_trt_video_vae_engine(
        models, progress, release_backend=unload_comfy_models, runtime=_runtime_config()
    )


def ensure_trt_video_vae_engine(
    models: ModelConfig,
    *,
    force: bool = False,
    progress=gr.Progress(track_tqdm=False),
) -> bool:
    return model_service.ensure_trt_video_vae_engine(
        models,
        force=force,
        progress=progress,
        release_backend=unload_comfy_models,
        runtime=_runtime_config(),
    )


def compile_trt_video_vae(
    progress=gr.Progress(track_tqdm=False),
) -> str:
    """Build the local TensorRT decoder engine from the manual UI action."""
    try:
        ensure_trt_video_vae_engine(
            load_model_config(),
            force=True,
            progress=progress,
        )
        return "TensorRT VAE decoder compiled and ready to use."
    except Exception as exc:
        return f"TensorRT VAE compilation failed: {exc}"


def ensure_single_frame_image_vae(models: ModelConfig) -> bool:
    return model_service.ensure_single_frame_image_vae(
        models, runtime=_runtime_config()
    )


def missing_ltx25_model_names(model_choice: str = DEFAULT_LTX25_MODEL) -> list[str]:
    return model_service.missing_ltx25_model_names(
        model_choice, runtime=_runtime_config()
    )


def ensure_ltx25_models(model_choice: str = DEFAULT_LTX25_MODEL) -> bool:
    return model_service.ensure_ltx25_models(model_choice, runtime=_runtime_config())


def render_ltx25_official_model_inventory() -> str:
    keys = ltx25_official_inventory_keys()
    installed = 0
    rows = []
    for key in keys:
        spec = MODEL_SPECS[key]
        path = COMFY_DIR / "models" / spec.folder / spec.local_name
        ready = model_file_is_ready(path)
        installed += int(ready)
        status = "✅ Installed" if ready else "⬇️ Available"
        source = f"[{spec.repo_id}](https://huggingface.co/{spec.repo_id})"
        rows.append(f"| `{spec.local_name}` | `{spec.folder}` | {status} | {source} |")
    return (
        f"**Installed: {installed}/{len(keys)}**\n\n"
        "| Model | ComfyUI folder | Status | Source / license |\n"
        "|---|---|---|---|\n" + "\n".join(rows)
    )


def render_ltx25_workflow_details(workflow_label: str) -> str:
    entry = ltx25_workflow_entry(workflow_label)
    extra_names = [MODEL_SPECS[key].local_name for key in entry["extra_models"]]
    extras = (
        ", ".join(f"`{name}`" for name in extra_names)
        if extra_names
        else "No use-case-specific checkpoint."
    )
    return (
        f"### {workflow_label}\n\n"
        f"{entry['description']}\n\n"
        f"**Inputs:** {entry['inputs']}\n\n"
        f"**Additional checkpoint(s):** {extras}\n\n"
        "Model preparation also installs the official BF16 transformer, text "
        "encoder, prompt enhancer, audio VAE, and (for video workflows) the "
        "full diffusion-decoder video VAE. These are large gated downloads.\n\n"
        f"[Download official workflow JSON](/ltx25-workflows/{entry['id']}.json) · "
        "[Open ComfyUI](/comfyui/)\n\n"
        "The template is also installed under **Workflows → Browse → LTX 2.5**. "
        "Download its models here first, then reload ComfyUI so its model "
        "dropdowns rescan the shared model folders."
    )


def _prepare_ltx25_model_set(required_keys: Iterable[str], label: str):
    """Lazily fetch a named set and refresh the visible model inventory."""
    try:
        required_keys = tuple(dict.fromkeys(required_keys))
        stale = stale_model_keys(
            root=COMFY_DIR / "models",
            manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
            model_keys=required_keys,
        )
        if not stale:
            yield (
                f"Ready: all models for **{label}** are installed.",
                render_ltx25_official_model_inventory(),
            )
            return
        names = ", ".join(MODEL_SPECS[key].local_name for key in stale)
        yield (
            f"Downloading models for **{label}**: {names}",
            render_ltx25_official_model_inventory(),
        )
        sync_models(
            root=COMFY_DIR / "models",
            manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
            token=resolve_hf_token(),
            log_prefix="[ltx25-workflow-on-demand]",
            model_keys=stale,
            download_workers=min(len(stale), 4),
        )
        missing = [
            MODEL_SPECS[key].local_name
            for key in required_keys
            if not model_file_is_ready(
                COMFY_DIR
                / "models"
                / MODEL_SPECS[key].folder
                / MODEL_SPECS[key].local_name
            )
        ]
        if missing:
            raise H3Error("Downloads did not produce: " + ", ".join(missing))
        yield (
            f"Ready: installed all models for **{label}**. Open ComfyUI and "
            "refresh model definitions or reload the page.",
            render_ltx25_official_model_inventory(),
        )
    except Exception as exc:
        yield (
            "Error downloading official workflow models. Open the Source / "
            "license links below, accept any gated terms, and authenticate "
            "with `hf auth login` or HF_TOKEN. "
            f"Details: {exc}",
            render_ltx25_official_model_inventory(),
        )


def prepare_ltx25_official_workflow(workflow_label: str):
    """Lazily fetch every checkpoint referenced by one official template."""
    yield from _prepare_ltx25_model_set(
        ltx25_workflow_model_keys(workflow_label),
        workflow_label,
    )


def prepare_all_ltx25_official_models():
    """Download every missing model displayed in the official inventory."""
    yield from _prepare_ltx25_model_set(
        ltx25_official_inventory_keys(),
        "all official workflow models",
    )


def ensure_seedvr2_upscale_models(
    models: ModelConfig,
    model_choice: str,
) -> bool:
    return model_service.ensure_seedvr2_upscale_models(
        models, model_choice, runtime=_runtime_config()
    )


def ensure_ltx25_upscale_models(
    model_choice: str = DEFAULT_LTX25_MODEL, *, option: str = LTX25_UPSCALE,
) -> bool:
    return model_service.ensure_ltx25_upscale_models(
        model_choice, runtime=_runtime_config(), option=option,
    )


def missing_music3_model_names(model_choice: str) -> list[str]:
    return model_service.missing_music3_model_names(
        model_choice, runtime=_runtime_config()
    )


def ensure_music3_models(model_choice: str) -> bool:
    return model_service.ensure_music3_models(model_choice, runtime=_runtime_config())


def api_get(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).get(path, **kwargs)


def api_post(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).post(path, **kwargs)


def object_info() -> dict[str, Any]:
    return api_get("/object_info").json()


# Gradio runs this in the browser when the file is selected. It deliberately
# uses the local File/preview object, so reading dimensions does not add a
# second server upload or require a Python callback.
AUTO_RESOLUTION_JS = r"""async (_value, currentWidth, currentHeight, resultFormat, latentUpscale, autoMegapixels) => {
    const root = document.getElementById("first-frame-image");
    const input = root?.querySelector('input[type="file"]')
        || document.querySelector('#first-frame-image input[type="file"]');
    const file = input?.files?.[0];
    const preview = root?.querySelector("img")
        || document.querySelector('#first-frame-image img');
    let imageWidth = 0;
    let imageHeight = 0;

    if (file) {
        try {
            const bitmap = await createImageBitmap(file);
            imageWidth = bitmap.width;
            imageHeight = bitmap.height;
            bitmap.close();
        } catch (_) {
            const objectUrl = URL.createObjectURL(file);
            try {
                const image = new Image();
                image.src = objectUrl;
                await image.decode();
                imageWidth = image.naturalWidth;
                imageHeight = image.naturalHeight;
            } finally {
                URL.revokeObjectURL(objectUrl);
            }
        }
    }

    // The preview is a local-only fallback for browsers without bitmap APIs
    // or when Gradio has already replaced the file input after upload.
    if (!imageWidth || !imageHeight) {
        imageWidth = preview?.naturalWidth || 0;
        imageHeight = preview?.naturalHeight || 0;
    }

    // Gradio may clear the native input as soon as its upload completes. Do
    // not overwrite valid controls with null in that race; the next change
    // event can still apply the automatic size when a local preview exists.
    if (!imageWidth || !imageHeight) {
        return [
            Number(currentWidth) || 864,
            Number(currentHeight) || 480,
            "Resolution unchanged: image dimensions were unavailable locally.",
        ];
    }

    if (String(resultFormat).toLowerCase() === "audio") {
        return [
            Number(currentWidth) || 864,
            Number(currentHeight) || 480,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        ];
    }

    const grid = latentUpscale ? 64 : 32;
    const snap = (value) => Math.max(grid, Math.round(value / grid) * grid);
    if (String(resultFormat).toLowerCase() === "image") {
        const width = snap(imageWidth);
        const height = snap(imageHeight);
        const megapixels = (width * height / 1000000).toFixed(2);
        const displayRatio = (width / height).toFixed(2);
        return [
            width,
            height,
            `**Image mode: start frame resolution** · **${width}×${height}** · ${megapixels} MP · ${displayRatio}:1 · ${grid}-pixel aligned`,
        ];
    }

    const selectedMegapixels = Number.parseInt(String(autoMegapixels), 10) || 4;
    const pixelLimit = selectedMegapixels * 1000000;
    if (imageWidth * imageHeight < pixelLimit) {
        const width = snap(imageWidth);
        const height = snap(imageHeight);
        const megapixels = (width * height / 1000000).toFixed(2);
        const displayRatio = (width / height).toFixed(2);
        return [
            width,
            height,
            `**Start frame native resolution** · **${width}×${height}** · ${megapixels} MP · ${displayRatio}:1 · ${grid}-pixel aligned`,
        ];
    }

    const maxPixels = pixelLimit - 1;
    const ratio = imageWidth / imageHeight;
    let width;
    let height;
    if (ratio >= 1) {
        width = Math.max(grid, Math.floor(Math.sqrt(maxPixels * ratio) / grid) * grid);
        height = Math.max(grid, Math.floor(width / ratio / grid) * grid);
    } else {
        height = Math.max(grid, Math.floor(Math.sqrt(maxPixels / ratio) / grid) * grid);
        width = Math.max(grid, Math.floor(height * ratio / grid) * grid);
    }
    while (width * height >= pixelLimit) {
        if (ratio >= 1) {
            width = Math.max(grid, width - grid);
            height = Math.max(grid, Math.floor(width / ratio / grid) * grid);
        } else {
            height = Math.max(grid, height - grid);
            width = Math.max(grid, Math.floor(height * ratio / grid) * grid);
        }
    }
    const megapixels = (width * height / 1000000).toFixed(2);
    const displayRatio = (width / height).toFixed(2);
    return [
        width,
        height,
        `**Auto from start frame** · **${width}×${height}** · ${megapixels} MP · ${displayRatio}:1 · ${grid}-pixel aligned`,
    ];
}"""


def resolution_control_updates(
    width: int | float | None,
    height: int | float | None,
    latent_upscale: bool,
    result_format: str,
) -> tuple[int | float, int | float, str]:
    if normalize_result_format(result_format) == "Audio":
        return (
            width if width is not None else 32,
            height if height is not None else 32,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        )
    alignment = 64 if latent_upscale else 32
    try:
        resolved_width = snap_to_grid(width, alignment) if width is not None else 864
        resolved_height = snap_to_grid(height, alignment) if height is not None else 480
    except Exception:
        resolved_width, resolved_height = 864, 480
    prefix = "**Latent upscale 64-pixel alignment** · " if latent_upscale else ""
    return (
        resolved_width,
        resolved_height,
        prefix + resolution_summary(resolved_width, resolved_height),
    )


def resolution_info_preview(
    width: int | float | None,
    height: int | float | None,
    latent_upscale: bool,
    result_format: str,
) -> str:
    """Return only the resolution info text without snapping the input values.

    Called on ``.input()`` so the user sees live feedback while typing, without
    the width/height fields being overwritten mid-keystroke.  The actual snap
    happens on ``.blur()`` or ``.submit()`` via ``resolution_control_updates``.
    """
    if normalize_result_format(result_format) == "Audio":
        return (
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32."
        )
    if width is None or height is None:
        return ""
    try:
        alignment = 64 if latent_upscale else 32
        resolved_width = snap_to_grid(width, alignment)
        resolved_height = snap_to_grid(height, alignment)
    except Exception as exc:
        return f"⚠️ {exc}"
    prefix = "**Latent upscale 64-pixel alignment** · " if latent_upscale else ""
    return prefix + resolution_summary(resolved_width, resolved_height)


def auto_resolution_from_start_frame(
    first_image: Any,
    current_width: int | float | None,
    current_height: int | float | None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    latent_upscale: bool = False,
    auto_megapixels: str = DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
) -> tuple[int | float, int | float, str]:
    """Apply the automatic ratio resolution after Gradio stages the image."""
    fallback_width = current_width or UI_DEFAULTS["width"]
    fallback_height = current_height or UI_DEFAULTS["height"]
    normalized_result = normalize_result_format(result_format)
    if normalized_result == "Audio":
        return (
            fallback_width,
            fallback_height,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        )
    paths = normalize_paths(first_image)
    if not paths:
        return (
            fallback_width,
            fallback_height,
            resolution_summary(fallback_width, fallback_height),
        )

    try:
        from PIL import Image

        with Image.open(paths[0]) as image:
            width, height = resolution_for_aspect_ratio(
                *image.size,
                preserve_native=normalized_result == "Image",
                alignment=64 if latent_upscale else 32,
                pixel_cap=auto_resolution_pixel_cap(auto_megapixels),
            )
    except Exception as exc:
        return (
            fallback_width,
            fallback_height,
            f"⚠️ Unable to read start frame dimensions: {exc}",
        )
    return width, height, resolution_summary(width, height)


def resolution_choice_updates(
    name: str,
    tier: str,
    latent_upscale: bool,
    result_format: str,
) -> tuple[int | float, int | float, str]:
    """Resolve and align a preset before updating its controls.

    Keeping preset lookup and alignment in one callback prevents the UI from
    briefly writing the raw preset dimensions before latent upscale snaps them
    to its required 64-pixel grid.
    """
    width, height, _summary = resolution_choice_values(name, tier)
    return resolution_control_updates(width, height, latent_upscale, result_format)


def start_frame_generation_resolution(
    first_image: Any,
    *,
    alignment: int,
) -> tuple[int, int] | None:
    paths = normalize_paths(first_image)
    if not paths:
        return None
    try:
        from PIL import Image

        with Image.open(paths[0]) as image:
            return resolution_for_aspect_ratio(
                *image.size,
                preserve_native=True,
                alignment=alignment,
            )
    except Exception as exc:
        raise H3Error(f"Unable to read start frame dimensions: {exc}") from exc


def generation_resolution(
    width: int | float,
    height: int | float,
    *,
    result_format: str,
    latent_upscale: bool,
    mode: str,
    first_image: Any,
) -> tuple[int, int]:
    normalized_result = normalize_result_format(result_format)
    if normalized_result == "Audio":
        return 32, 32
    alignment = 64 if latent_upscale else 32
    if normalized_result == "Image" and mode == "First / last frame" and first_image:
        start_resolution = start_frame_generation_resolution(
            first_image, alignment=alignment
        )
        if start_resolution is not None:
            return start_resolution
    return snap_to_grid(width, alignment), snap_to_grid(height, alignment)


def resolve_sol_policy(
    attention_mode: str,
    mode: str,
    width: int,
    height: int,
    duration: float,
    first_image: str | None,
    last_image: str | None,
    use_turbo: bool = False,
) -> tuple[bool, int, str]:
    requested = str(attention_mode).strip().lower()
    tokens = estimate_packed_tokens(
        mode, width, height, duration, first_image, last_image
    )
    if requested in {"dense", "kitchen", "comfy-kitchen"}:
        return False, tokens, "forced Comfy Kitchen"
    if requested in {"sage", "sage 2", "sage2"}:
        return False, tokens, "forced Sage 2"
    if requested in {"sla", "sla attention", "sparse-linear"}:
        return False, tokens, "forced SLA"
    if SERVER_ATTENTION_BACKEND != "sol":
        return False, tokens, "Sol backend unavailable"
    if requested in {"sol-attn", "sol", "sparse"}:
        return True, tokens, "forced Sol-Attn"
    # Reference conditioning is not available to the estimator before ComfyUI
    # encodes the uploaded media. Treat it as a large job in both generation
    # modes instead of making a decision from the target tokens alone.
    if mode == "Reference media":
        prefix = "Auto Turbo" if use_turbo else "Auto"
        return True, tokens, f"{prefix}: reference mode"
    if use_turbo:
        enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
        return (
            enabled,
            tokens,
            (
                f"Auto Turbo: {tokens:,} target tokens "
                f"{'≥' if enabled else '<'} {AUTO_SOL_TOKEN_THRESHOLD:,}"
            ),
        )
    enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
    return (
        enabled,
        tokens,
        (
            f"Auto: {tokens:,} target tokens "
            f"{'≥' if enabled else '<'} {AUTO_SOL_TOKEN_THRESHOLD:,}"
        ),
    )


def mode_layout_updates(mode: str):
    """Update task-specific inputs without changing the acceleration choice."""
    presentation = mode_presentation(mode)
    return (
        mode_help(mode),
        gr.update(visible=presentation.show_frames),
        gr.update(visible=presentation.show_references),
        gr.update(interactive=True),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def result_format_layout_updates(
    result_format: str,
    current_width: int | float,
    current_height: int | float,
    first_image: Any,
    latent_upscale: bool,
):
    presentation = result_format_presentation(normalize_result_format(result_format))
    result_format = presentation.format
    is_image = presentation.is_image
    is_audio = presentation.is_audio
    display_width, display_height = current_width, current_height
    if not is_audio:
        alignment = 64 if latent_upscale else 32
        if first_image:
            try:
                display_width, display_height, _ = auto_resolution_from_start_frame(
                    first_image,
                    current_width,
                    current_height,
                    result_format,
                    latent_upscale,
                )
            except Exception:
                display_width = snap_to_grid(current_width, alignment)
                display_height = snap_to_grid(current_height, alignment)
        else:
            display_width = snap_to_grid(current_width, alignment)
            display_height = snap_to_grid(current_height, alignment)
    return (
        gr.update(visible=not is_image),
        gr.update(visible=is_image),
        gr.update(visible=is_image),
        gr.update(visible=presentation.is_video),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        (
            gr.update(visible=True)
            if presentation.is_video
            else gr.update(visible=False)
        ),
        gr.update(visible=is_image),
        gr.update(visible=is_audio),
        (
            gr.update(interactive=True)
            if presentation.is_video
            else gr.update(interactive=False)
        ),
        (gr.update(interactive=False) if is_audio else gr.update(interactive=True)),
        gr.update(value=display_width, interactive=not is_audio),
        gr.update(value=display_height, interactive=not is_audio),
        (
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32."
            if is_audio
            else resolution_summary(display_width, display_height)
        ),
        gr.update(value=presentation.action_label),
    )


def image_vae_frame_updates(image_vae: Any):
    single = normalize_image_vae(image_vae) != DEFAULT_IMAGE_VAE
    return gr.update(
        interactive=not single,
        info="Effective output: one frame with the 500K decoder."
        if single
        else "Choose 1–20 decoded frames.",
    )


def latent_upscale_layout_updates(
    enabled: bool,
    width: int | float,
    height: int | float,
    result_format: str,
):
    if normalize_result_format(result_format) == "Audio":
        return (
            gr.update(visible=False),
            width,
            height,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        )
    if enabled:
        resolved_width, resolved_height = snap64(width), snap64(height)
        note = "**Latent upscale 64-pixel alignment** · "
    else:
        resolved_width, resolved_height = validate_resolution(width, height)
        note = ""
    return (
        gr.update(visible=bool(enabled)),
        resolved_width,
        resolved_height,
        note + resolution_summary(resolved_width, resolved_height),
    )


def latent_upscale_method_layout_update(method: str):
    return gr.update(
        visible=(resolve_h3_latent_upscale_method(method) == H3_LATENT_UPSCALE_SPLIT)
    )


def generation_mode_defaults(name: str, turbo_variant: str = DEFAULT_TURBO):
    """Normal/Turbo is independent from the selected base-model profile.

    Important: when entering Turbo, do not change preset.value. Changing it
    would fire preset.change() and race with the Turbo Steps update.
    """
    if str(name).strip().lower() == "turbo":
        return (
            gr.update(interactive=True),
            gr.update(
                value=turbo_steps_for(turbo_variant),
                interactive=True,
            ),
            "simple",
            DEFAULT_ACCELERATOR,
            "SLA",
        )

    return (
        gr.update(value="Balanced", interactive=True),
        gr.update(value=18, interactive=True),
        "simple",
        DEFAULT_ACCELERATOR,
        "SLA",
    )


def turbo_variant_defaults(turbo_variant: str, generation_mode: str):
    """Apply variant sampling defaults only while Turbo is selected."""
    if str(generation_mode).strip().lower() != "turbo":
        return gr.update(), gr.update()
    return gr.update(
        value=turbo_steps_for(turbo_variant),
        interactive=True,
    ), "simple"


def fbcache_preset_defaults(name: str):
    key = str(name).strip().lower()
    if key == "safe":
        values = (0.08, 0.10, 0.95, 2)
        interactive = False
    elif key == "aggressive":
        values = (0.12, 0.10, 0.95, 2)
        interactive = False
    elif key == "custom":
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    else:
        values = (0.10, 0.10, 0.95, 2)
        interactive = False
    return tuple(gr.update(value=value, interactive=interactive) for value in values)


from h3_app.processes import run_media_process


def file_content_sha256(path: Path) -> str:
    return staging.file_content_sha256(path)


def staged_input_is_ready(path: Path) -> bool:
    return staging.staged_input_is_ready(path)


def materialize_staged_input(
    source: Path,
    destination: Path,
    *,
    transcode_video: bool,
) -> None:
    return staging.materialize_staged_input(
        source, destination, transcode_video=transcode_video
    )


def stage_file(
    path: str,
    category: str,
    transcode_video: bool = False,
    reuse: bool = False,
) -> str:
    return staging.stage_file(
        path, category, transcode_video, reuse, runtime=_runtime_config()
    )


turbo_required_nodes = h3_workflow.turbo_required_nodes


add_turbo_model_patch = h3_workflow.add_turbo_model_patch


add_model_stack = h3_workflow.add_model_stack


h3_conditioning_video_vae = h3_workflow.h3_conditioning_video_vae


h3_conditioning_cache_key = h3_workflow.h3_conditioning_cache_key


add_h3_stage_offload = h3_workflow.add_h3_stage_offload


h3_refinement_attention_model = h3_workflow.h3_refinement_attention_model


finish_sampling = h3_workflow.finish_sampling


def build_fl2va_graph(
    *,
    prompt: str,
    first_image: str | None,
    last_image: str | None,
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    turbo_lora_name: str | None,
    turbo_variant: str,
    turbo_strength: float,
    use_sol: bool,
    sol_tau: float,
    sol_thresh_type: str,
    sol_exact_mode: str,
    sol_dense_steps: int,
    sol_step_off: float,
    sol_sink_tokens: int,
    cache_mode: str,
    fbcache_preset: str,
    fbcache_threshold: float,
    fbcache_start: float,
    fbcache_end: float,
    fbcache_max_hits: int,
    fbcache_temporal_guard: bool,
    easycache_threshold: float,
    easycache_start: float,
    easycache_end: float,
    easycache_verbose: bool,
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    reuse_unchanged_inputs: bool = True,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
    semantic_bridge: bool = False,
    semantic_bridge_alpha: float = 0.10,
    voice_reference_audios: list[str] | None = None,
) -> dict[str, Any]:
    first_image = (
        stage_file(first_image, "keyframes", reuse=reuse_unchanged_inputs)
        if first_image
        else None
    )
    last_image = (
        stage_file(last_image, "keyframes", reuse=reuse_unchanged_inputs)
        if last_image
        else None
    )
    voice_reference_audios = [
        stage_file(path, "fl2va_voice_audios", reuse=reuse_unchanged_inputs)
        for path in (voice_reference_audios or [])
    ]
    return h3_workflow.build_fl2va_graph(
        prompt=prompt,
        first_image=first_image,
        last_image=last_image,
        width=width,
        height=height,
        duration=duration,
        steps=steps,
        seed=seed,
        scheduler=scheduler,
        turbo_lora_name=turbo_lora_name,
        turbo_variant=turbo_variant,
        turbo_strength=turbo_strength,
        use_sol=use_sol,
        sol_tau=sol_tau,
        sol_thresh_type=sol_thresh_type,
        sol_exact_mode=sol_exact_mode,
        sol_dense_steps=sol_dense_steps,
        sol_step_off=sol_step_off,
        sol_sink_tokens=sol_sink_tokens,
        cache_mode=cache_mode,
        fbcache_preset=fbcache_preset,
        fbcache_threshold=fbcache_threshold,
        fbcache_start=fbcache_start,
        fbcache_end=fbcache_end,
        fbcache_max_hits=fbcache_max_hits,
        fbcache_temporal_guard=fbcache_temporal_guard,
        easycache_threshold=easycache_threshold,
        easycache_start=easycache_start,
        easycache_end=easycache_end,
        easycache_verbose=easycache_verbose,
        model_name=model_name,
        models=models,
        available_nodes=available_nodes,
        use_int8_vae=use_int8_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_split_config=latent_split_config,
        result_format=result_format,
        image_frames=image_frames,
        image_vae=image_vae,
        text_encoder_name=text_encoder_name,
        reuse_unchanged_inputs=reuse_unchanged_inputs,
        stage_model_offload=stage_model_offload,
        smart_stage_offload=smart_stage_offload,
        semantic_bridge=semantic_bridge,
        semantic_bridge_alpha=semantic_bridge_alpha,
        voice_reference_audios=voice_reference_audios,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


def build_ref2va_graph(
    *,
    prompt: str,
    reference_images: list[str],
    reference_videos: list[str],
    reference_audios: list[str],
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    scheduler: str,
    ref_image_size: str,
    turbo_lora_name: str | None,
    turbo_variant: str,
    turbo_strength: float,
    use_sol: bool,
    sol_tau: float,
    sol_thresh_type: str,
    sol_exact_mode: str,
    sol_dense_steps: int,
    sol_step_off: float,
    sol_sink_tokens: int,
    cache_mode: str,
    fbcache_preset: str,
    fbcache_threshold: float,
    fbcache_start: float,
    fbcache_end: float,
    fbcache_max_hits: int,
    fbcache_temporal_guard: bool,
    easycache_threshold: float,
    easycache_start: float,
    easycache_end: float,
    easycache_verbose: bool,
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    smart_stage_offload: bool = False,
    reuse_unchanged_inputs: bool = True,
    stage_model_offload: bool = False,
) -> dict[str, Any]:
    reference_images = [
        stage_file(path, "reference_images", reuse=reuse_unchanged_inputs)
        for path in reference_images
    ]
    reference_videos = [
        stage_file(
            path, "reference_videos", transcode_video=True, reuse=reuse_unchanged_inputs
        )
        for path in reference_videos
    ]
    reference_audios = [
        stage_file(path, "reference_audios", reuse=reuse_unchanged_inputs)
        for path in reference_audios
    ]
    return h3_workflow.build_ref2va_graph(
        prompt=prompt,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_audios=reference_audios,
        width=width,
        height=height,
        duration=duration,
        steps=steps,
        seed=seed,
        scheduler=scheduler,
        ref_image_size=ref_image_size,
        turbo_lora_name=turbo_lora_name,
        turbo_variant=turbo_variant,
        turbo_strength=turbo_strength,
        use_sol=use_sol,
        sol_tau=sol_tau,
        sol_thresh_type=sol_thresh_type,
        sol_exact_mode=sol_exact_mode,
        sol_dense_steps=sol_dense_steps,
        sol_step_off=sol_step_off,
        sol_sink_tokens=sol_sink_tokens,
        cache_mode=cache_mode,
        fbcache_preset=fbcache_preset,
        fbcache_threshold=fbcache_threshold,
        fbcache_start=fbcache_start,
        fbcache_end=fbcache_end,
        fbcache_max_hits=fbcache_max_hits,
        fbcache_temporal_guard=fbcache_temporal_guard,
        easycache_threshold=easycache_threshold,
        easycache_start=easycache_start,
        easycache_end=easycache_end,
        easycache_verbose=easycache_verbose,
        model_name=model_name,
        models=models,
        available_nodes=available_nodes,
        use_int8_vae=use_int8_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        latent_split_config=latent_split_config,
        result_format=result_format,
        image_frames=image_frames,
        image_vae=image_vae,
        text_encoder_name=text_encoder_name,
        smart_stage_offload=smart_stage_offload,
        reuse_unchanged_inputs=reuse_unchanged_inputs,
        stage_model_offload=stage_model_offload,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


required_ltx25_nodes = ltx_workflow.required_ltx25_nodes


def build_ltx25_graph(
    *,
    model_choice: str = DEFAULT_LTX25_MODEL,
    prompt: str,
    negative_prompt: str,
    first_image: str | None,
    width: int,
    height: int,
    duration: float,
    fps: float,
    seed: int,
    cfg: float,
    sampler_name: str,
    image_strength: float,
    middle_image: str | None = None,
    middle_time: float = LTX25_DEFAULTS["middle_time"],
    middle_strength: float = LTX25_DEFAULTS["middle_strength"],
    end_image: str | None = None,
    end_strength: float = LTX25_DEFAULTS["end_strength"],
) -> dict[str, Any]:
    first_image = stage_file(first_image, "ltx25_keyframes") if first_image else None
    middle_image = stage_file(middle_image, "ltx25_keyframes") if middle_image else None
    end_image = stage_file(end_image, "ltx25_keyframes") if end_image else None
    return ltx_workflow.build_ltx25_graph(
        model_choice=model_choice,
        prompt=prompt,
        negative_prompt=negative_prompt,
        first_image=first_image,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        seed=seed,
        cfg=cfg,
        sampler_name=sampler_name,
        image_strength=image_strength,
        middle_image=middle_image,
        middle_time=middle_time,
        middle_strength=middle_strength,
        end_image=end_image,
        end_strength=end_strength,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


required_seedvr2_upscale_nodes = upscale_workflow.required_seedvr2_upscale_nodes


required_seedvr2_image_upscale_nodes = (
    upscale_workflow.required_seedvr2_image_upscale_nodes
)


def build_seedvr2_image_upscale_graph(
    *,
    source_images: list[tuple[str, str, float]],
    seed: int,
    models: ModelConfig,
    model_choice: str = DEFAULT_SEEDVR2_MODEL,
    output_token: str,
) -> dict[str, Any]:
    return upscale_workflow.build_seedvr2_image_upscale_graph(
        source_images=source_images,
        seed=seed,
        models=models,
        model_choice=model_choice,
        output_token=output_token,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


def build_seedvr2_upscale_graph(
    *,
    source_video: str,
    seed: int,
    models: ModelConfig,
    model_choice: str = DEFAULT_SEEDVR2_MODEL,
    target_width: int | None = None,
    source_width: int | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    return upscale_workflow.build_seedvr2_upscale_graph(
        source_video=source_video,
        seed=seed,
        models=models,
        model_choice=model_choice,
        target_width=target_width,
        source_width=source_width,
        fps=fps,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


required_ltx25_upscale_nodes = upscale_workflow.required_ltx25_upscale_nodes


def build_ltx25_upscale_graph(
    *,
    source_video: str,
    seed: int,
    model_choice: str = DEFAULT_LTX25_MODEL,
    prompt: str = "",
    width: int,
    height: int,
    target_width: int | None = None,
    target_height: int | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    return upscale_workflow.build_ltx25_upscale_graph(
        source_video=source_video,
        seed=seed,
        model_choice=model_choice,
        prompt=prompt,
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
        fps=fps,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


required_upscale_nodes = upscale_workflow.required_upscale_nodes


def build_upscale_graph(
    *,
    option: str,
    source_video: str,
    seed: int,
    models: ModelConfig,
    seedvr2_model: str = DEFAULT_SEEDVR2_MODEL,
    ltx25_model: str = DEFAULT_LTX25_MODEL,
    prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    fps: float = 24.0,
) -> tuple[dict[str, Any], int]:
    return upscale_workflow.build_upscale_graph(
        option=option,
        source_video=source_video,
        seed=seed,
        models=models,
        seedvr2_model=seedvr2_model,
        ltx25_model=ltx25_model,
        prompt=prompt,
        width=width,
        height=height,
        target_width=target_width,
        target_height=target_height,
        fps=fps,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


required_nodes_for = h3_workflow.required_nodes_for


def output_context() -> OutputContext:
    job = CURRENT_JOB.get()
    return OutputContext(_runtime_config(), job.output_token if job else None)


def _submission(prompt_id, graph=None):
    if isinstance(prompt_id, PromptId):
        return prompt_id.submission
    job = CURRENT_JOB.get()
    return Submission(
        str(prompt_id),
        graph or {},
        ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP),
        time.monotonic() + GENERATION_TIMEOUT,
        GENERATION_TIMEOUT,
        POLL_SECONDS,
        job=job,
        check_cancelled=job.check if job else lambda: None,
    )


def submit_prompt(graph: dict[str, Any], client_id: str) -> str:
    return ExecutionRunner(
        _runtime_config(),
        ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP),
        JOBS,
        connect=websocket.create_connection,
    ).submit(graph, client_id, CURRENT_JOB.get())


def websocket_url(client_id: str) -> str:
    parsed = urlsplit(COMFY_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/ws"
    return urlunsplit((scheme, parsed.netloc, path, f"clientId={quote(client_id)}", ""))


def queue_position(prompt_id: str) -> tuple[str, int | None]:
    return _submission(prompt_id).queue_position(str(prompt_id))


def stream_comfy_progress(ws, prompt_id, graph, started):
    submission = _submission(prompt_id, graph)
    if submission.socket is None:
        submission.socket = ws
    yield from submission.progress()


def poll_comfy_progress(prompt_id, graph):
    yield from _submission(prompt_id, graph).progress()


def wait_for_history(prompt_id):
    return _submission(prompt_id).history()


def walk_saved_refs(value):
    yield from outputs.walk_saved_refs(value)


def _history_output_candidates(history, extensions, *, directory=None):
    return outputs._history_output_candidates(
        history, extensions, directory=directory, context=output_context()
    )


def _recent_output_candidates(directory, extensions, queued_at):
    return outputs._recent_output_candidates(
        directory, extensions, queued_at, context=output_context()
    )


def resolve_output(history: dict[str, Any], queued_at: float) -> Path:
    return outputs.resolve_output(history, queued_at, context=output_context())


def resolve_audio_output(history: dict[str, Any], queued_at: float) -> Path:
    return outputs.resolve_audio_output(history, queued_at, context=output_context())


def resolve_image_outputs(
    history: dict[str, Any], queued_at: float, expected_count: int
) -> list[Path]:
    return outputs.resolve_image_outputs(
        history, queued_at, expected_count, context=output_context()
    )


def resolve_seedvr2_input_upscale_outputs(
    history: dict[str, Any],
    queued_at: float,
    output_token: str,
    slot_keys: Iterable[str],
) -> dict[str, Path]:
    return outputs.resolve_seedvr2_input_upscale_outputs(
        history, queued_at, output_token, slot_keys, context=output_context()
    )


def input_image_frame_preset_updates(
    preset: str,
    current_width: int | float,
    current_height: int | float,
) -> tuple[Any, Any]:
    dimensions = INPUT_IMAGE_FRAME_PRESETS.get(str(preset))
    if dimensions is None:
        return gr.update(value=current_width), gr.update(value=current_height)
    return dimensions


def input_image_upscale_dimensions(
    image_path: str,
    frame_width: int | float,
    frame_height: int | float,
) -> tuple[int, int, int, int, float]:
    """Fit an image upward into a bounding frame without ever downscaling it."""
    try:
        target_width = int(round(float(frame_width)))
        target_height = int(round(float(frame_height)))
    except (TypeError, ValueError) as exc:
        raise H3Error("Input upscale frame width and height must be numbers.") from exc
    if target_width < 1 or target_height < 1:
        raise H3Error("Input upscale frame width and height must be positive.")

    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as image:
            source_width, source_height = ImageOps.exif_transpose(image).size
    except Exception as exc:
        raise H3Error(f"Could not read input image dimensions: {exc}") from exc
    if source_width < 1 or source_height < 1:
        raise H3Error("Input image has invalid dimensions.")

    scale_by = min(
        target_width / source_width,
        target_height / source_height,
    )
    if scale_by <= 1.0:
        return source_width, source_height, source_width, source_height, 1.0
    destination_width = min(target_width, round(source_width * scale_by))
    destination_height = min(target_height, round(source_height * scale_by))
    return (
        source_width,
        source_height,
        destination_width,
        destination_height,
        scale_by,
    )


def upscale_selected_input_images(
    selected_slots: Iterable[str],
    model_choice: str,
    seed: int,
    force_offload: bool,
    frame_width: int,
    frame_height: int,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    progress=gr.Progress(track_tqdm=False),
):
    """Upscale selected H3 stills to fit a frame and replace their UI values."""
    slot_labels = list(INPUT_IMAGE_UPSCALE_SLOTS)
    slot_keys = [
        "first",
        "last",
        *(f"picture_{index}" for index in range(1, MAX_REFERENCE_IMAGES + 1)),
    ]
    image_values = (
        first_image,
        last_image,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        ref_image_4,
        ref_image_5,
        ref_image_6,
        ref_image_7,
        ref_image_8,
        ref_image_9,
    )
    selected = list(dict.fromkeys(str(value) for value in (selected_slots or [])))
    if not selected:
        raise gr.Error("Select at least one start, end, or reference image to upscale.")
    unknown = sorted(set(selected) - set(slot_labels))
    if unknown:
        raise gr.Error("Unknown input image selection: " + ", ".join(unknown))

    values_by_label = dict(zip(slot_labels, image_values))
    keys_by_label = dict(zip(slot_labels, slot_keys))
    staged: list[tuple[str, str, float]] = []
    unchanged_results: dict[str, Path] = {}
    dimension_notes: list[str] = []
    for label in selected:
        paths = normalize_paths(values_by_label[label])
        if not paths:
            raise gr.Error(f"Upload {label} before selecting it for upscaling.")
        source_path = Path(paths[0]).resolve()
        try:
            source_width, source_height, dest_width, dest_height, scale_by = (
                input_image_upscale_dimensions(
                    str(source_path), frame_width, frame_height
                )
            )
        except Exception as exc:
            raise gr.Error(f"Could not prepare {label}: {exc}") from exc
        slot_key = keys_by_label[label]
        if scale_by > 1.0:
            staged.append(
                (
                    slot_key,
                    stage_file(str(source_path), "input_image_upscale", reuse=True),
                    scale_by,
                )
            )
            dimension_notes.append(
                f"{label}: {source_width}×{source_height} → {dest_width}×{dest_height}"
            )
        else:
            unchanged_results[slot_key] = source_path
            dimension_notes.append(f"{label}: {source_width}×{source_height} unchanged")

    actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
    generated_keys = {slot_key for slot_key, _path, _scale in staged}
    results = dict(unchanged_results)
    try:
        if staged:
            progress(0, desc="Checking SeedVR2 input upscaler")
            available = set(object_info())
            missing = required_seedvr2_image_upscale_nodes() - available
            if missing:
                raise H3Error(
                    "SeedVR2 input upscaling requires current ComfyUI nodes: "
                    + ", ".join(sorted(missing))
                )
            models = load_model_config()
            ensure_seedvr2_upscale_models(models, model_choice)
            if force_offload:
                progress(0, desc="Unloading resident models")
                unload_comfy_models()

            output_token = uuid.uuid4().hex
            graph = build_seedvr2_image_upscale_graph(
                source_images=staged,
                seed=actual_seed,
                models=models,
                model_choice=model_choice,
                output_token=output_token,
            )
            client_id = str(uuid.uuid4())
            queued_at = time.time()
            prompt_id = submit_prompt(graph, client_id)
            for stage, completed, total, step, step_total in poll_comfy_progress(
                prompt_id, graph
            ):
                if step is not None and step_total:
                    progress((step, step_total), desc=f"Upscaling inputs: {stage}")
                elif total:
                    progress((completed, total), desc=f"Upscaling inputs: {stage}")
            history = wait_for_history(prompt_id)
            results.update(
                resolve_seedvr2_input_upscale_outputs(
                    history,
                    queued_at,
                    output_token,
                    generated_keys,
                )
            )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Input image upscaling failed: {exc}") from exc

    component_updates: list[Any] = []
    downloads: list[str] = []
    for slot_key in slot_keys:
        if slot_key in generated_keys:
            path = str(results[slot_key])
            component_updates.append(gr.update(value=path))
        else:
            component_updates.append(gr.update())
        if slot_key in results:
            downloads.append(str(results[slot_key]))
    progress(1, desc="Input images ready")
    summary = (
        f"Fit selected images into a {int(frame_width)}×{int(frame_height)} frame: "
        f"{len(staged)} upscaled with SeedVR2 {model_choice}, "
        f"{len(unchanged_results)} already large enough. Aspect ratios were preserved "
        "and no image was downscaled.\n\n" + "  \n".join(dimension_notes)
    )
    return (*component_updates, downloads, summary)


def image_frame_labels(frame_paths: Iterable[Any]) -> list[str]:
    return [f"Frame {index + 1}" for index, _ in enumerate(frame_paths)]


def select_all_image_frames(frame_paths: Iterable[Any]) -> list[str]:
    return image_frame_labels(frame_paths)


def save_selected_image_frames(
    frame_paths: Iterable[Any], selected_labels: Iterable[str]
) -> tuple[list[str], str]:
    paths = [Path(path).resolve() for path in normalize_paths(frame_paths)]
    labels = list(selected_labels or [])
    if not paths:
        raise gr.Error("Generate image frames before saving a selection.")
    if not labels:
        raise gr.Error("Select at least one image frame to save.")

    selected_indices: list[int] = []
    for label in labels:
        match = re.fullmatch(r"Frame\s+(\d+)", str(label).strip())
        if not match:
            raise gr.Error(f"Invalid frame selection: {label}")
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(paths):
            raise gr.Error(f"Frame selection is out of range: {label}")
        if index not in selected_indices:
            selected_indices.append(index)

    staging_root = (OUTPUT_DIR / "h3" / "image_staging").resolve()
    for index in selected_indices:
        source = paths[index]
        if not source.is_relative_to(staging_root) or not source.is_file():
            raise gr.Error(
                "A selected frame is outside the H3 image staging directory."
            )

    destination = (
        OUTPUT_DIR
        / "h3"
        / "images"
        / f"selection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    saved: list[str] = []
    for index in selected_indices:
        target = destination / f"frame_{index + 1:03d}.png"
        copy_media(paths[index], target)
        saved.append(str(target))
    return saved, f"Saved {len(saved)} selected frame(s) to `{destination}`."


def has_encoder(name: str) -> bool:
    return media_tools.has_encoder(name)


def ensure_swiftvr_checkpoint() -> tuple[Path, bool]:
    return swiftvr.ensure_swiftvr_checkpoint(runtime=_runtime_config())


def import_swiftvr_pipeline() -> Any:
    return swiftvr.import_swiftvr_pipeline(runtime=_runtime_config())


def postprocess_swiftvr_video(
    source: Path,
    *,
    fps: float,
    target_width: int,
    target_height: int,
) -> Path:
    return swiftvr.postprocess_swiftvr_video(
        source,
        fps=fps,
        target_width=target_width,
        target_height=target_height,
        runtime=_runtime_config(),
    )


def postprocess_video(source: Path, option: str) -> Path:
    return media_tools.postprocess_video(source, option, runtime=_runtime_config())


def probe_video_metadata(source: Path) -> VideoMetadata:
    return media_tools.probe_video_metadata(source)


def prepare_upscale_clip_batch(
    source: Path,
    *,
    category: str,
    split_enabled: bool,
    split_seconds: float,
    metadata: VideoMetadata,
) -> UpscaleClipBatch:
    return media_tools.prepare_upscale_clip_batch(
        source,
        category=category,
        split_enabled=split_enabled,
        split_seconds=split_seconds,
        metadata=metadata,
        runtime=_runtime_config(),
    )


def concat_upscaled_clips(
    source: Path,
    clips: list[Path],
    *,
    option: str,
    duration: float,
    frame_count: int,
) -> Path:
    return media_tools.concat_upscaled_clips(
        source,
        clips,
        option=option,
        duration=duration,
        frame_count=frame_count,
        runtime=_runtime_config(),
    )


def cleanup_upscale_clip_batch(
    batch: UpscaleClipBatch | None,
    outputs: Iterable[Path] = (),
) -> None:
    return media_tools.cleanup_upscale_clip_batch(batch, outputs)


def unload_comfy_models() -> None:
    """Explicitly clear model residency only when the user opts into it."""
    api_post("/free", json={"unload_models": True, "free_memory": True})


@gpu_maintenance("unload")
def unload_all_models() -> tuple[str, str]:
    """Unload every resident ComfyUI model and refresh the backend summary."""
    local_was_loaded = unload_prompt_rewriter()
    try:
        unload_comfy_models()
        local_note = " Local 8B prompt writer unloaded." if local_was_loaded else ""
        return (
            f"All models unloaded and cached VRAM released.{local_note}",
            backend_status(),
        )
    except Exception as exc:
        local_note = " Local 8B prompt writer was unloaded." if local_was_loaded else ""
        return f"ComfyUI VRAM release failed: {exc}.{local_note}", backend_status()


def video_download_path(video: str | Path) -> str:
    return gallery_store.video_download_path(video, runtime=_runtime_config())


def managed_video_path(
    video: str | Path,
    *,
    require_file: bool = True,
) -> Path:
    return gallery_store.managed_video_path(
        video, require_file=require_file, runtime=_runtime_config()
    )


def absolute_video_url(
    video: str | Path,
    request: gr.Request,
    *,
    download: bool = False,
) -> str:
    relative_url = video_download_path(video)
    base_url = str(request.request.base_url).rstrip("/")
    url = f"{base_url}{relative_url}"
    return f"{url}?download=1" if download else url


def absolute_video_download_url(video: str | Path, request: gr.Request) -> str:
    return absolute_video_url(video, request, download=True)


def gallery_video_paths(*, limit: int | None = GALLERY_LIMIT) -> list[Path]:
    return gallery_store.gallery_video_paths(limit=limit, runtime=_runtime_config())


def gallery_thumbnail(video: Path) -> Path | None:
    return gallery_store.gallery_thumbnail(video, runtime=_runtime_config())


def gallery_thumbnail_path(video: str | Path) -> Path:
    return gallery_store.gallery_thumbnail_path(video, runtime=_runtime_config())


def gallery_video_resolution(video: Path) -> tuple[int, int] | None:
    return gallery_store.gallery_video_resolution(video, runtime=_runtime_config())


def gallery_resolution_text(video: Path) -> str:
    return gallery_store.gallery_resolution_text(video, runtime=_runtime_config())


def generated_video_family(video: str | Path) -> str:
    return gallery_store.generated_video_family(video, runtime=_runtime_config())


def forget_gallery_metadata(video: str | Path | None = None) -> None:
    return gallery_store.forget_gallery_metadata(video)


def import_gallery_video(uploaded_video: str | None) -> GalleryMutationResult:
    if not uploaded_video:
        return gallery_mutation_result(
            "Choose a local video first.", clear_selection=False
        )
    source = Path(uploaded_video).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
        return gallery_mutation_result(
            "The selected file is not a supported video.", clear_selection=False
        )
    destination_dir = OUTPUTS_DIR / "imports"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        destination_dir
        / f"import_{int(time.time())}_{uuid.uuid4().hex[:8]}{source.suffix.lower()}"
    )
    copy_media(source, destination)
    return gallery_mutation_result(
        f"Imported `{source.name}`",
        selected_video=str(destination),
        clear_selection=False,
    )


def refresh_gallery() -> tuple[list[tuple[str, str]], list[str], str]:
    videos = gallery_video_paths()
    items: list[tuple[str, str]] = []
    selectable_paths: list[str] = []
    failed = 0
    for video in videos:
        thumbnail = gallery_thumbnail(video)
        if thumbnail is None:
            failed += 1
        try:
            stat = video.stat()
        except OSError:
            continue
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        size_mb = stat.st_size / (1024 * 1024)
        resolution = gallery_resolution_text(video)
        caption = (
            f"{generated_video_family(video)} · {video.name} · {resolution} · "
            f"{timestamp} · {size_mb:.1f} MB"
        )
        # Gradio Gallery accepts videos as well as images. Falling back to the
        # video itself keeps the item selectable when FFmpeg cannot make a poster.
        items.append((str(thumbnail or video), caption))
        selectable_paths.append(str(video))
    detail = f"{len(items)} generated video{'s' if len(items) != 1 else ''}"
    if failed:
        detail += f" · {failed} thumbnail{'s' if failed != 1 else ''} unavailable"
    return items, selectable_paths, detail


def select_gallery_video(
    paths: list[str],
    request: gr.Request,
    evt: gr.SelectData,
) -> tuple[str | None, str, str | None]:
    index = evt.index
    if isinstance(index, (tuple, list)):
        index = index[0]
    try:
        video = paths[int(index)]
    except (IndexError, TypeError, ValueError):
        return None, "", None
    download_url = absolute_video_download_url(video, request)
    resolution = gallery_resolution_text(managed_video_path(video))
    # Return the local path to gr.Video. Gradio treats arbitrary HTTP URLs as
    # remote fetches and can reject its own public hostname during validation.
    return (
        video,
        f"**Resolution:** {resolution} · [Download video]({download_url})",
        video,
    )


GalleryMutationResult = tuple[
    list[tuple[str, str]],
    list[str],
    str,
    Any,
    Any,
    str | None,
    bool,
]

GalleryPostprocessResult = tuple[
    list[tuple[str, str]],
    list[str],
    str,
    Any,
    Any,
    str | None,
    bool,
    str,
]


def gallery_mutation_result(
    message: str,
    *,
    selected_video: str | None = None,
    clear_selection: bool,
) -> GalleryMutationResult:
    items, paths, detail = refresh_gallery()
    player = None if clear_selection else gr.skip()
    download = "" if clear_selection else gr.skip()
    selected = None if clear_selection else selected_video
    return (
        items,
        paths,
        f"{message} · {detail}",
        player,
        download,
        selected,
        False,
    )


def gallery_progress_result(message: str) -> GalleryPostprocessResult:
    return (
        gr.skip(),
        gr.skip(),
        message,
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        message,
    )


def gallery_processed_result(
    result: Path,
    option: str,
    elapsed: float,
    request: gr.Request,
) -> GalleryPostprocessResult:
    items, paths, detail = refresh_gallery()
    download_url = absolute_video_download_url(result, request)
    resolution = gallery_resolution_text(result)
    return (
        items,
        paths,
        f"Completed {option} in {elapsed:.1f}s · {detail}",
        str(result),
        f"**Resolution:** {resolution} · [Download processed video]({download_url})",
        str(result),
        False,
        f"Completed {option} in {elapsed:.1f}s",
    )


def postprocess_selected_gallery_video(
    selected_video: str | None,
    option: str,
    seed: int,
    seedvr2_model: str,
    ltx25_model: str,
    ltx25_prompt: str,
    force_offload: bool,
    split_upscale: bool,
    split_seconds: float,
    upscale_resolution: str,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=False),
):
    """Create a new post-processed output from one selected gallery video."""
    started = time.monotonic()
    ws: websocket.WebSocket | None = None
    clip_batch: UpscaleClipBatch | None = None
    clip_outputs: list[Path] = []
    try:
        if not selected_video:
            raise H3Error("Select a gallery video first.")
        if option not in POSTPROCESS_OPTIONS:
            raise H3Error("Choose a post-processing method.")
        source = managed_video_path(selected_video)
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        progress(0, desc=f"Preparing {option}")
        yield gallery_progress_result(f"Preparing `{source.name}` for {option}")

        if option == SWIFTVR_UPSCALE:
            progress(0, desc="Checking SwiftVR runtime and checkpoint")
            yield gallery_progress_result(
                "Checking SwiftVR runtime and downloading its checkpoint on first use."
            )
            metadata = probe_video_metadata(source)
            target_width, target_height = upscale_target_dimensions(
                metadata.width, metadata.height, upscale_resolution
            )
            result = postprocess_swiftvr_video(
                source,
                fps=metadata.fps,
                target_width=target_width,
                target_height=target_height,
            )
            progress(1, desc="Complete")
            yield gallery_processed_result(
                result, option, time.monotonic() - started, request
            )
            return

        if option not in COMFY_POSTPROCESS_OPTIONS:
            result = postprocess_video(source, option)
            progress(1, desc="Complete")
            yield gallery_processed_result(
                result, option, time.monotonic() - started, request
            )
            return

        models = load_model_config()
        available = set(object_info())
        missing = required_upscale_nodes(option) - available
        if missing:
            raise H3Error(
                f"{option} is unavailable. Missing ComfyUI nodes: "
                + ", ".join(sorted(missing))
            )

        if option == SEEDVR2_UPSCALE:
            model_status = f"SeedVR2 {seedvr2_model}"
            yield gallery_progress_result(f"Checking {model_status} models")
            downloaded = ensure_seedvr2_upscale_models(models, seedvr2_model)
            stage_bucket = "seedvr2_upscale"
        else:
            model_status = f"{option} with {ltx25_model}"
            yield gallery_progress_result(f"Checking {model_status} models")
            downloaded = ensure_ltx25_upscale_models(ltx25_model, option=option)
            stage_bucket = (
                LTX25_POSTPROCESS_MODELS[option]
                if option in LTX25_RESTORATION_OPTIONS else "ltx25_upscale"
            )

        if downloaded:
            yield gallery_progress_result(f"{option} models downloaded")
        metadata = probe_video_metadata(source)
        target_width, target_height = (
            (metadata.width, metadata.height)
            if option in LTX25_RESTORATION_OPTIONS
            else upscale_target_dimensions(
                metadata.width, metadata.height, upscale_resolution
            )
        )
        use_split = option in LTX25_POSTPROCESS_MODELS and bool(split_upscale)
        clip_batch = prepare_upscale_clip_batch(
            source,
            category=stage_bucket,
            split_enabled=use_split,
            split_seconds=split_seconds,
            metadata=metadata,
        )
        clip_count = len(clip_batch.sources)
        if use_split:
            yield gallery_progress_result(
                f"Split `{source.name}` into {clip_count} LTX-safe clips "
                f"(target {float(split_seconds):g}s each)"
            )
        if force_offload:
            yield gallery_progress_result(f"Unloading resident models before {option}")
            unload_comfy_models()

        if not use_split:
            graph, configured_steps = build_upscale_graph(
                option=option,
                source_video=clip_batch.sources[0],
                seed=actual_seed,
                models=models,
                seedvr2_model=seedvr2_model,
                ltx25_model=ltx25_model,
                prompt=ltx25_prompt,
                width=metadata.width,
                height=metadata.height,
                target_width=target_width,
                target_height=target_height,
                fps=metadata.fps,
            )

        client_id = str(uuid.uuid4())
        ws = None  # Progress connections belong to the execution runner.
        if use_split:
            for clip_index, staged_source in enumerate(clip_batch.sources):
                clip_seed = (actual_seed + clip_index) % (2**63 - 1)
                graph, configured_steps = build_upscale_graph(
                    option=option,
                    source_video=staged_source,
                    seed=clip_seed,
                    models=models,
                    seedvr2_model=seedvr2_model,
                    ltx25_model=ltx25_model,
                    prompt=ltx25_prompt,
                    width=metadata.width,
                    height=metadata.height,
                    target_width=target_width,
                    target_height=target_height,
                    fps=metadata.fps,
                )
                queued_at = time.time()
                prompt_id = submit_prompt(graph, client_id)
                clip_label = f"Clip {clip_index + 1}/{clip_count}"
                yield gallery_progress_result(
                    progress_status(
                        f"{option} queued",
                        started=started,
                        detail=f"{clip_label} 路 job `{prompt_id}` 路 seed {clip_seed}",
                    )
                )
                updates = (
                    stream_comfy_progress(ws, prompt_id, graph, started)
                    if ws is not None
                    else poll_comfy_progress(prompt_id, graph)
                )
                for stage, completed_nodes, total_nodes, step, step_total in updates:
                    if stage == "Generating video and audio":
                        stage = f"Processing with {option}"
                    if step is not None and step_total:
                        progress(
                            (clip_index * step_total + step, clip_count * step_total),
                            desc=f"{clip_label}: {stage}",
                        )
                    elif total_nodes:
                        progress(
                            (
                                clip_index * total_nodes + completed_nodes,
                                clip_count * total_nodes,
                            ),
                            desc=f"{clip_label}: {stage}",
                        )
                    yield gallery_progress_result(
                        progress_status(
                            f"{clip_label}: {stage}",
                            started=started,
                            completed_nodes=completed_nodes,
                            total_nodes=total_nodes,
                            step=step,
                            step_total=step_total,
                            configured_steps=(
                                configured_steps if step is not None else None
                            ),
                            detail=f"Post-process job `{prompt_id}`",
                        )
                    )
                clip_outputs.append(
                    resolve_output(wait_for_history(prompt_id), queued_at)
                )
            yield gallery_progress_result(
                f"Concatenating {clip_count} processed clips and restoring source audio"
            )
            result = concat_upscaled_clips(
                source,
                clip_outputs,
                option=option,
                duration=metadata.duration,
                frame_count=metadata.frame_count,
            )
            progress(1, desc="Complete")
            yield gallery_processed_result(
                result, option, time.monotonic() - started, request
            )
            return

        queued_at = time.time()
        prompt_id = submit_prompt(graph, client_id)
        yield gallery_progress_result(
            progress_status(
                f"{option} queued",
                started=started,
                detail=f"Job `{prompt_id}` · seed {actual_seed}",
            )
        )
        updates = (
            stream_comfy_progress(ws, prompt_id, graph, started)
            if ws is not None
            else poll_comfy_progress(prompt_id, graph)
        )
        for stage, completed_nodes, total_nodes, step, step_total in updates:
            if stage == "Generating video and audio":
                stage = f"Processing with {option}"
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total_nodes:
                progress((completed_nodes, total_nodes), desc=stage)
            yield gallery_progress_result(
                progress_status(
                    stage,
                    started=started,
                    completed_nodes=completed_nodes,
                    total_nodes=total_nodes,
                    step=step,
                    step_total=step_total,
                    configured_steps=configured_steps if step is not None else None,
                    detail=f"Post-process job `{prompt_id}`",
                )
            )

        result = resolve_output(wait_for_history(prompt_id), queued_at)
        progress(1, desc="Complete")
        yield gallery_processed_result(
            result, option, time.monotonic() - started, request
        )
    except Exception as exc:
        yield gallery_progress_result(f"Post-processing failed: {exc}")
    finally:
        cleanup_upscale_clip_batch(
            clip_batch,
            clip_outputs if clip_batch and clip_batch.temporary_inputs else (),
        )
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def delete_selected_gallery_video(
    selected_video: str | None,
    confirmed: bool,
) -> GalleryMutationResult:
    if not confirmed:
        return gallery_mutation_result(
            "Confirm permanent deletion first.",
            selected_video=selected_video,
            clear_selection=False,
        )
    if not selected_video:
        return gallery_mutation_result(
            "Select a video to delete.",
            clear_selection=True,
        )
    try:
        video = managed_video_path(selected_video)
        thumbnail = gallery_thumbnail_path(video)
        name = video.name
        video.unlink()
        snapshot_path(video).unlink(missing_ok=True)
        thumbnail.unlink(missing_ok=True)
        forget_gallery_metadata(video)
        return gallery_mutation_result(
            f"Deleted `{name}`.",
            clear_selection=True,
        )
    except (H3Error, OSError) as exc:
        return gallery_mutation_result(
            f"Delete failed: {exc}",
            clear_selection=True,
        )


def empty_generated_gallery(
    selected_video: str | None,
    confirmed: bool,
) -> GalleryMutationResult:
    if not confirmed:
        return gallery_mutation_result(
            "Confirm permanent deletion first.",
            selected_video=selected_video,
            clear_selection=False,
        )
    deleted = 0
    failed = 0
    for candidate in gallery_video_paths(limit=None):
        try:
            video = managed_video_path(candidate)
            gallery_thumbnail_path(video).unlink(missing_ok=True)
            video.unlink()
            snapshot_path(video).unlink(missing_ok=True)
            deleted += 1
        except (H3Error, OSError):
            failed += 1
    if GALLERY_THUMBNAILS_DIR.is_dir():
        for thumbnail in GALLERY_THUMBNAILS_DIR.iterdir():
            if thumbnail.is_file() and thumbnail.suffix.lower() in {".jpg", ".tmp"}:
                try:
                    thumbnail.unlink()
                except OSError:
                    failed += 1
    forget_gallery_metadata()
    result = f"Deleted {deleted} generated video{'s' if deleted != 1 else ''}."
    if failed:
        result += f" {failed} file{'s' if failed != 1 else ''} could not be deleted."
    return gallery_mutation_result(result, clear_selection=True)


def backend_status() -> str:
    try:
        stats = api_get("/system_stats").json()
        live_nodes = set(object_info())
        devices = stats.get("devices", [])
        device = devices[0] if devices else {}
        gpu = device.get("name", "unknown GPU")
        vram_total = device.get("vram_total")
        vram_free = device.get("vram_free")
        if isinstance(vram_total, (int, float)) and isinstance(vram_free, (int, float)):
            vram_text = (
                f" · {vram_free / 2**30:.1f}/{vram_total / 2**30:.1f} GiB VRAM free"
            )
        elif isinstance(vram_total, (int, float)):
            vram_text = f" · {vram_total / 2**30:.1f} GiB VRAM"
        else:
            vram_text = ""
        models = load_model_config()
        easycache_status = "available" if "EasyCache" in live_nodes else "unavailable"
        fbcache_status = (
            "available" if "H3FirstBlockCache" in live_nodes else "unavailable"
        )
        spectrum_status = (
            "available" if "SpectrumApplyMiniMaxH3" in live_nodes else "unavailable"
        )
        profile_lines = [f"**Spectrum accelerator**: {spectrum_status}"]
        for profile in models.profiles.values():
            profile_lines.append(
                f"**{profile.label}** · FL2VA `{profile.fl2va}` · "
                f"Ref2VA `{profile.ref2va}`"
            )
        if models.larry_turbo_lora:
            profile_lines.append(
                f"**Larry Turbo v4-600 EMA** | LoRA `{models.larry_turbo_lora}` | "
                "6-step default | strength 1.0 | custom loader/sampler"
            )
        if models.turbo_lora:
            profile_lines.append(
                f"**LightX2V Turbo / 4-step** · FL2VA v1.2 `{models.turbo_lora}` · "
                f"Ref2VA v0.1 544p `{models.turbo_ref_lora}` · strength 1.0"
            )
        if models.turbo_8step_lora:
            profile_lines.append(
                f"**LightX2V Turbo v1.0 / 8-step 768p** · FL2VA `{models.turbo_8step_lora}` · "
                f"Ref2VA `{models.turbo_8step_ref_lora}` · "
                "8-step default · strength 1.0 · FL2VA and Ref2VA"
            )
        return (
            f"Connected · {gpu}{vram_text} · sparse: {SERVER_ATTENTION_BACKEND} · "
            f"dense: {SERVER_DENSE_ATTENTION_BACKEND} · "
            f"memory: {SERVER_MEMORY_PROFILE} · FirstBlockCache: {fbcache_status} · "
            f"EasyCache: {easycache_status}  \n" + "  \n".join(profile_lines)
        )
    except Exception as exc:
        return f"Backend unavailable: {exc}"


def _generation_services() -> generation_services.GenerationServices:
    return generation_services.GenerationServices(
        workflows=generation_services.WorkflowsServices(
            build_fl2va_graph=build_fl2va_graph,
            build_ltx25_graph=build_ltx25_graph,
            build_ref2va_graph=build_ref2va_graph,
            build_upscale_graph=build_upscale_graph,
        ),
        media=generation_services.MediaServices(
            cleanup_upscale_clip_batch=cleanup_upscale_clip_batch,
            concat_upscaled_clips=concat_upscaled_clips,
            postprocess_swiftvr_video=postprocess_swiftvr_video,
            postprocess_video=postprocess_video,
            prepare_upscale_clip_batch=prepare_upscale_clip_batch,
            probe_video_metadata=probe_video_metadata,
            resolve_audio_output=resolve_audio_output,
            resolve_image_outputs=resolve_image_outputs,
            resolve_output=resolve_output,
        ),
        models=generation_services.ModelsServices(
            ensure_h3_latent_upscaler_model=ensure_h3_latent_upscaler_model,
            ensure_h3_semantic_bridge=ensure_h3_semantic_bridge,
            ensure_h3_text_encoder=ensure_h3_text_encoder,
            ensure_int8_video_vae=ensure_int8_video_vae,
            ensure_ltx25_models=ensure_ltx25_models,
            ensure_ltx25_upscale_models=ensure_ltx25_upscale_models,
            ensure_music3_models=ensure_music3_models,
            ensure_profile_model=ensure_profile_model,
            ensure_seedvr2_upscale_models=ensure_seedvr2_upscale_models,
            ensure_single_frame_image_vae=ensure_single_frame_image_vae,
            ensure_trt_video_vae_engine=ensure_trt_video_vae_engine,
            ensure_turbo_lora=ensure_turbo_lora,
            load_model_config=load_model_config,
            missing_ltx25_model_names=missing_ltx25_model_names,
            missing_music3_model_names=missing_music3_model_names,
            trt_vae_decoder_paths=trt_vae_decoder_paths,
            unload_comfy_models=unload_comfy_models,
            h3_text_encoder_settings=h3_text_encoder_settings,
            model_file_is_ready=model_file_is_ready,
            unload_prompt_rewriter=unload_prompt_rewriter,
        ),
        execution=generation_services.ExecutionServices(
            object_info=object_info,
            poll_comfy_progress=poll_comfy_progress,
            stream_comfy_progress=stream_comfy_progress,
            submit_prompt=submit_prompt,
            wait_for_history=wait_for_history,
        ),
        policy=generation_services.PolicyServices(
            resolve_request_settings=resolve_request_settings,
            resolve_sol_policy=resolve_sol_policy,
        ),
    )


def generate(
    mode: str,
    model_profile: str,
    text_encoder: str,
    stage_model_offload: bool,
    generation_mode: str,
    turbo_variant: str,
    prompt: str,
    first_image: str | None,
    last_image: str | None,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    steps: int,
    scheduler: str,
    seed: int,
    attention_mode: str,
    sla_preset: str,
    sol_tau: float,
    sol_thresh_type: str,
    sol_exact_mode: str,
    sol_dense_steps: int,
    sol_step_off: float,
    sol_sink_tokens: int,
    cache_mode: str,
    fbcache_preset: str,
    fbcache_threshold: float,
    fbcache_start: float,
    fbcache_end: float,
    fbcache_max_hits: int,
    fbcache_temporal_guard: bool,
    easycache_threshold: float,
    easycache_start: float,
    easycache_end: float,
    easycache_verbose: bool,
    ref_image_size: str,
    postprocess: str,
    reuse_unchanged_inputs: bool = True,
    latent_upscale: bool = False,
    latent_upscaler_model: str = DEFAULT_H3_LATENT_UPSCALER_MODEL,
    latent_upscale_refine_steps: int = 2,
    latent_upscale_method: str = H3_LATENT_UPSCALE_STANDARD,
    latent_split_tile_width: int = 512,
    latent_split_tile_height: int = 512,
    latent_split_overlap_ratio: float = 0.25,
    latent_split_fade_ratio: float = 0.50,
    latent_split_chunk_frames: int = 73,
    latent_split_temporal_overlap_frames: int = 22,
    latent_split_seam_denoise: float = 0.75,
    latent_split_seam_polish: str = "off",
    upscale_force_offload: bool = False,
    upscale_split_enabled: bool = False,
    upscale_split_seconds: float = 5.0,
    upscale_resolution: str = DEFAULT_UPSCALE_RESOLUTION,
    seedvr2_model: str = DEFAULT_SEEDVR2_MODEL,
    ltx25_model: str = DEFAULT_LTX25_MODEL,
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    image_vae: str = DEFAULT_IMAGE_VAE,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    semantic_bridge: bool = True,
    semantic_bridge_alpha: float = 0.10,
    fl2va_audio_1: Any = None,
    fl2va_audio_2: Any = None,
    fl2va_audio_3: Any = None,
    progress=gr.Progress(track_tqdm=False),
):
    request = generation_requests.H3Request.from_values(
        {
            "mode": mode,
            "model_profile": model_profile,
            "text_encoder": text_encoder,
            "stage_model_offload": stage_model_offload,
            "generation_mode": generation_mode,
            "turbo_variant": turbo_variant,
            "prompt": prompt,
            "first_image": first_image,
            "last_image": last_image,
            "ref_image_1": ref_image_1,
            "ref_image_2": ref_image_2,
            "ref_image_3": ref_image_3,
            "ref_image_4": ref_image_4,
            "ref_image_5": ref_image_5,
            "ref_image_6": ref_image_6,
            "ref_image_7": ref_image_7,
            "ref_image_8": ref_image_8,
            "ref_image_9": ref_image_9,
            "ref_video_1": ref_video_1,
            "ref_video_2": ref_video_2,
            "ref_video_3": ref_video_3,
            "ref_audio_1": ref_audio_1,
            "ref_audio_2": ref_audio_2,
            "ref_audio_3": ref_audio_3,
            "duration": duration,
            "width": width,
            "height": height,
            "steps": steps,
            "scheduler": scheduler,
            "seed": seed,
            "attention_mode": attention_mode,
            "sla_preset": sla_preset,
            "sol_tau": sol_tau,
            "sol_thresh_type": sol_thresh_type,
            "sol_exact_mode": sol_exact_mode,
            "sol_dense_steps": sol_dense_steps,
            "sol_step_off": sol_step_off,
            "sol_sink_tokens": sol_sink_tokens,
            "cache_mode": cache_mode,
            "fbcache_preset": fbcache_preset,
            "fbcache_threshold": fbcache_threshold,
            "fbcache_start": fbcache_start,
            "fbcache_end": fbcache_end,
            "fbcache_max_hits": fbcache_max_hits,
            "fbcache_temporal_guard": fbcache_temporal_guard,
            "easycache_threshold": easycache_threshold,
            "easycache_start": easycache_start,
            "easycache_end": easycache_end,
            "easycache_verbose": easycache_verbose,
            "ref_image_size": ref_image_size,
            "postprocess": postprocess,
            "reuse_unchanged_inputs": reuse_unchanged_inputs,
            "latent_upscale": latent_upscale,
            "latent_upscaler_model": latent_upscaler_model,
            "latent_upscale_refine_steps": latent_upscale_refine_steps,
            "latent_upscale_method": latent_upscale_method,
            "latent_split_tile_width": latent_split_tile_width,
            "latent_split_tile_height": latent_split_tile_height,
            "latent_split_overlap_ratio": latent_split_overlap_ratio,
            "latent_split_fade_ratio": latent_split_fade_ratio,
            "latent_split_chunk_frames": latent_split_chunk_frames,
            "latent_split_temporal_overlap_frames": latent_split_temporal_overlap_frames,
            "latent_split_seam_denoise": latent_split_seam_denoise,
            "latent_split_seam_polish": latent_split_seam_polish,
            "upscale_force_offload": upscale_force_offload,
            "upscale_split_enabled": upscale_split_enabled,
            "upscale_split_seconds": upscale_split_seconds,
            "upscale_resolution": upscale_resolution,
            "seedvr2_model": seedvr2_model,
            "ltx25_model": ltx25_model,
            "use_int8_vae": use_int8_vae,
            "use_trt_vae": use_trt_vae,
            "image_vae": image_vae,
            "result_format": result_format,
            "image_frames": image_frames,
            "semantic_bridge": semantic_bridge,
            "semantic_bridge_alpha": semantic_bridge_alpha,
            "fl2va_audio_1": fl2va_audio_1,
            "fl2va_audio_2": fl2va_audio_2,
            "fl2va_audio_3": fl2va_audio_3,
        }
    )
    yield from h3_generation.generate(
        request,
        _generation_services(),
        _runtime_config(),
        run_context=RUN_CONTEXT.get(),
        progress=progress,
    )


def generate_ltx25(
    mode: str,
    model_choice: str,
    prompt: str,
    negative_prompt: str,
    first_image: str | None,
    duration: float,
    fps: float,
    width: int,
    height: int,
    seed: int,
    cfg: float,
    sampler_name: str,
    image_strength: float,
    middle_image: str | None = None,
    middle_time: float = LTX25_DEFAULTS["middle_time"],
    middle_strength: float = LTX25_DEFAULTS["middle_strength"],
    end_image: str | None = None,
    end_strength: float = LTX25_DEFAULTS["end_strength"],
    progress=gr.Progress(track_tqdm=False),
):
    request = generation_requests.LtxRequest(
        **{
            "mode": mode,
            "model_choice": model_choice,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "first_image": first_image,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": seed,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "image_strength": image_strength,
            "middle_image": middle_image,
            "middle_time": middle_time,
            "middle_strength": middle_strength,
            "end_image": end_image,
            "end_strength": end_strength,
        }
    )
    yield from ltx_generation.generate_ltx25(
        request, _generation_services(), _runtime_config(), progress=progress
    )


def generate_music3(
    model_choice: str,
    caption: str,
    lyrics: str,
    max_duration: float,
    seed: int,
    steps: int,
    cfg: float,
    ar_cfg: float,
    top_k: int,
    tiled_decode: bool,
    progress=gr.Progress(track_tqdm=False),
):
    request = generation_requests.MusicRequest(
        **{
            "model_choice": model_choice,
            "caption": caption,
            "lyrics": lyrics,
            "max_duration": max_duration,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "ar_cfg": ar_cfg,
            "top_k": top_k,
            "tiled_decode": tiled_decode,
        }
    )
    yield from music_generation.generate_music3(
        request, _generation_services(), _runtime_config(), progress=progress
    )


def interrupt(request: gr.Request, family: str = "h3") -> str:
    try:
        return (
            JOBS.cancel(request.session_hash, family, api_get, api_post)
            if request.session_hash
            else "No session-owned job to cancel."
        )
    except Exception as exc:
        return f"Interrupt failed: {exc}"


def preset_values(name: str, generation_mode: str = "Normal"):
    values = asdict(preset_settings(name, generation_mode))
    return (
        *list(values.values())[:-1],
        text_encoder_offload_update(values["text_encoder"]),
    )


def compact_settings_summary(
    mode: str,
    model_profile: str,
    text_encoder: str,
    stage_model_offload: bool,
    reuse_unchanged_inputs: bool,
    use_int8_vae: bool,
    generation_mode: str,
    turbo_variant: str,
    duration: float,
    width: int,
    height: int,
    steps: int,
    scheduler: str,
    attention_mode: str,
    sla_preset: str,
    cache_mode: str,
    latent_upscale: bool,
    latent_upscaler_model: str,
    latent_upscale_refine_steps: int,
    postprocess: str,
    seedvr2_model: str,
    ltx25_model: str,
    force_offload: bool,
    split_upscale: bool,
    split_seconds: float,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    latent_upscale_method: str = H3_LATENT_UPSCALE_STANDARD,
    latent_split_tile_width: int = 512,
    latent_split_tile_height: int = 512,
    latent_split_overlap_ratio: float = 0.25,
    latent_split_fade_ratio: float = 0.50,
    latent_split_chunk_frames: int = 73,
    latent_split_temporal_overlap_frames: int = 22,
    latent_split_seam_denoise: float = 0.75,
    latent_split_seam_polish: str = "off",
    use_trt_vae: bool = False,
) -> str:
    return describe_settings(locals())


def result_settings_for_media(path):
    if isinstance(path, (tuple, list)):
        path = path[0] if path else None
    if isinstance(path, dict):
        path = path.get("path")
    if path and not read_snapshot(path):
        name = Path(path).name
        if re.search(r"[0-9a-f]{32}", name):
            candidates = [
                candidate
                for root in (OUTPUT_DIR, OUTPUTS_DIR)
                for candidate in root.rglob(name)
                if candidate.resolve().is_relative_to(root.resolve())
                and read_snapshot(candidate)
            ]
            if len(candidates) == 1:
                path = candidates[0]
    return render_snapshot(path)


def resolve_request_settings(values: dict):
    request = GenerationRequest.from_values(values)
    dimensions = None
    first = values.get("first_image", values.get("first"))
    if (
        request.output.result_format == "Image"
        and request.mode == "First / last frame"
        and first
    ):
        dimensions = start_frame_generation_resolution(
            first, alignment=64 if request.finishing.latent_upscale else 32
        )
    return resolve_settings(
        request, ResolutionContext(dimensions, SERVER_MEMORY_PROFILE)
    )


def describe_settings(values: dict) -> str:
    try:
        return render_settings(resolve_request_settings(values), values)
    except (ValueError, TypeError, H3Error) as exc:
        return (
            '<div role="alert">Check generation settings: '
            + html.escape(str(exc))
            + "</div>"
        )


def reference_prompt_help() -> str:
    return (
        f"Use `<Picture 1>` through `<Picture {MAX_REFERENCE_IMAGES}>`, "
        f"`<Video 1>` through `<Video {MAX_REFERENCE_VIDEOS}>`, and "
        f"`<Audio 1>` through `<Audio {MAX_REFERENCE_AUDIOS}>` in the prompt."
    )


def mode_help(mode: str) -> str:
    if mode == "Reference media":
        return (
            "Reference tags are ordered as images, then videos, then standalone audio. "
            + reference_prompt_help()
        )
    if mode == "First / last frame":
        return "Upload a first frame, a last frame, or both. This uses the FL2VA model."
    return "Prompt-only generation using FL2VA with native stereo audio."


def generate_with_ui_defaults(
    prompt: str,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate with UI defaults and return a reusable public download URL."""
    defaults = UI_DEFAULTS
    updates = generate(
        mode=defaults["mode"],
        model_profile=defaults["model_profile"],
        text_encoder=defaults["text_encoder"],
        stage_model_offload=defaults["stage_model_offload"],
        use_int8_vae=defaults["use_int8_vae"],
        use_trt_vae=defaults["use_trt_vae"],
        image_vae=defaults["image_vae"],
        result_format=defaults["result_format"],
        image_frames=defaults["image_frames"],
        generation_mode=defaults["generation_mode"],
        turbo_variant=defaults["turbo_variant"],
        prompt=prompt,
        first_image=None,
        last_image=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        ref_image_5=None,
        ref_image_6=None,
        ref_image_7=None,
        ref_image_8=None,
        ref_image_9=None,
        ref_video_1=None,
        ref_video_2=None,
        ref_video_3=None,
        ref_audio_1=None,
        ref_audio_2=None,
        ref_audio_3=None,
        duration=defaults["duration"],
        width=defaults["width"],
        height=defaults["height"],
        steps=defaults["steps"],
        scheduler=defaults["scheduler"],
        seed=defaults["seed"],
        attention_mode=defaults["attention_mode"],
        sol_tau=defaults["sol_tau"],
        sla_preset=defaults["sla_preset"],
        sol_thresh_type=defaults["sol_thresh_type"],
        sol_exact_mode=defaults["sol_exact_mode"],
        sol_dense_steps=defaults["sol_dense_steps"],
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode=defaults["cache_mode"],
        fbcache_preset=defaults["fbcache_preset"],
        fbcache_threshold=defaults["fbcache_threshold"],
        fbcache_start=defaults["fbcache_start"],
        fbcache_end=defaults["fbcache_end"],
        fbcache_max_hits=defaults["fbcache_max_hits"],
        fbcache_temporal_guard=defaults["fbcache_temporal_guard"],
        easycache_threshold=defaults["easycache_threshold"],
        easycache_start=defaults["easycache_start"],
        easycache_end=defaults["easycache_end"],
        easycache_verbose=defaults["easycache_verbose"],
        ref_image_size=defaults["ref_image_size"],
        postprocess=defaults["postprocess"],
        semantic_bridge=defaults["semantic_bridge"],
        semantic_bridge_alpha=defaults["semantic_bridge_alpha"],
        reuse_unchanged_inputs=defaults["reuse_unchanged_inputs"],
        latent_upscale=defaults["latent_upscale"],
        latent_upscaler_model=defaults["latent_upscaler_model"],
        latent_upscale_refine_steps=defaults["latent_upscale_refine_steps"],
        latent_upscale_method=defaults["latent_upscale_method"],
        latent_split_tile_width=defaults["latent_split_tile_width"],
        latent_split_tile_height=defaults["latent_split_tile_height"],
        latent_split_overlap_ratio=defaults["latent_split_overlap_ratio"],
        latent_split_fade_ratio=defaults["latent_split_fade_ratio"],
        latent_split_chunk_frames=defaults["latent_split_chunk_frames"],
        latent_split_temporal_overlap_frames=defaults[
            "latent_split_temporal_overlap_frames"
        ],
        latent_split_seam_denoise=defaults["latent_split_seam_denoise"],
        latent_split_seam_polish=defaults["latent_split_seam_polish"],
        seedvr2_model=defaults["seedvr2_model"],
        ltx25_model=DEFAULT_LTX25_MODEL,
        upscale_force_offload=defaults["upscale_force_offload"],
        upscale_split_enabled=defaults["upscale_split_enabled"],
        upscale_split_seconds=defaults["upscale_split_seconds"],
        progress=progress,
    )
    for video, status in updates:
        download_url = (
            absolute_video_download_url(video, request) if video is not None else None
        )
        yield download_url, status


def video_batch_seeds(seed: int, batch_count: int) -> list[int]:
    """Resolve distinct seeds while preserving the single-run random-seed behavior."""
    count = int(batch_count)
    if not MIN_VIDEO_BATCH_COUNT <= count <= MAX_VIDEO_BATCH_COUNT:
        raise H3Error(
            f"Video batch count must be between {MIN_VIDEO_BATCH_COUNT} and "
            f"{MAX_VIDEO_BATCH_COUNT}."
        )
    base_seed = int(seed)
    if count == 1:
        return [base_seed]
    return random.sample(range(0, 2**63 - 1), count)


def generate_for_ui(batch_count: int, *args: Any):
    """Adapt shared generation to UI outputs, including 1-4 video variants."""
    arguments = GenerationArguments.from_positional(args)
    result_format = normalize_result_format(arguments.values["result_format"])
    count = int(batch_count) if result_format == "Video" else 1
    seeds = (
        video_batch_seeds(int(arguments.values["seed"]), count)
        if result_format == "Video"
        else [int(arguments.values["seed"])]
    )
    first_update = True
    for batch_index, batch_seed in enumerate(seeds):
        batch_arguments = arguments.with_seed(batch_seed)
        prefix = f"Video {batch_index + 1}/{count} · " if count > 1 else ""
        for result, status in generate(**batch_arguments.values):
            status = prefix + status
            video_updates = [gr.update() for _ in range(MAX_VIDEO_BATCH_COUNT)]
            if first_update:
                first_update = False
                video_updates = [
                    gr.update(
                        value=None,
                        visible=(result_format == "Video" and index < count),
                    )
                    for index in range(MAX_VIDEO_BATCH_COUNT)
                ]
                yield (
                    *video_updates,
                    gr.update(visible=result_format == "Image"),
                    gr.update(value=[]),
                    gr.update(choices=[], value=[]),
                    [],
                    gr.update(value=None),
                    gr.update(value=None, visible=result_format == "Audio"),
                    gr.update(value=""),
                    status,
                )
                if result is None:
                    continue

            if result is None:
                yield (
                    *video_updates,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    status,
                )
                continue

            if result_format == "Image":
                paths = normalize_paths(result)
                labels = image_frame_labels(paths)
                gallery = [(path, label) for path, label in zip(paths, labels)]
                yield (
                    *video_updates,
                    gr.update(visible=True),
                    gr.update(value=gallery),
                    gr.update(choices=labels, value=[]),
                    paths,
                    gr.update(value=None),
                    gr.update(),
                    gr.update(),
                    status,
                )
            elif result_format == "Audio":
                yield (
                    *video_updates,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(value=result, visible=True),
                    gr.update(),
                    status,
                )
            else:
                video_updates[batch_index] = gr.update(value=result, visible=True)
                yield (
                    *video_updates,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    status,
                )


def api_guide() -> str:
    defaults = UI_DEFAULTS
    return f"""## Generate through the API

The `/generate_video` endpoint accepts a prompt and preserves the existing **Video** defaults from the **MiniMax H3** tab, including default-on reuse of unchanged prompt/media conditioning:

`{defaults["mode"]}` · `{defaults["model_profile"]}` · `{defaults["generation_mode"]} / {defaults["turbo_variant"]}` · `{defaults["duration"]}s` · `{defaults["width"]}×{defaults["height"]}` · `{defaults["steps"]} steps` · `{defaults["scheduler"]}` scheduler · random seed

Install the client and submit a job:

```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
download_url, status = client.predict(
    "A cinematic tracking shot through a rain-soaked neon city",
    api_name="/generate_video",
)
print(download_url)
print(status)
```

`download_url` is an HTTP URL served by this app, so it can be opened in a browser or downloaded with `curl -L -O` while the app is running.

For every control exposed by the MiniMax H3 tab, including **Image** and **Audio** result formats, use `/generate_video_advanced` and inspect the app's [OpenAPI schema](/gradio_api/openapi.json) for its current parameter list. Image selections can be persisted through `/save_h3_image_frames`. API requests share the same single-job queue as the UI.
"""


def compact_backend_status(detail: str) -> str:
    """Render a calm, glanceable status while retaining diagnostics separately."""
    return backend_status_html(detail)


def refresh_backend_views() -> tuple[str, str]:
    detail = backend_status()
    return compact_backend_status(detail), detail


def generation_preflight(
    mode: str,
    prompt: str,
    first_image: Any,
    last_image: Any,
    *reference_media: Any,
) -> tuple[str, Any]:
    """Keep invalid jobs out of the expensive backend queue."""
    readiness = generation_readiness_state(
        mode, prompt, first_image, last_image, reference_media
    )
    return readiness.html, gr.update(interactive=readiness.ready)


def build_ui() -> gr.Blocks:
    defaults = UI_DEFAULTS
    initial_backend = backend_status()
    with gr.Blocks(title="MiniMax H3 Local") as demo:
        gr.HTML(
            '<section class="h3-hero"><h1>MiniMax H3 Local</h1>'
            "<p>Create video, images, audio, and music on the shared ComfyUI backend · "
            '<a href="/comfyui/" target="_blank" rel="noopener noreferrer">'
            "Open ComfyUI ↗</a></p></section>"
        )
        system_summary = gr.HTML(compact_backend_status(initial_backend))
        with gr.Accordion("System details and VRAM", open=False):
            with gr.Row(equal_height=True):
                health = gr.Markdown(initial_backend)
                unload_models = gr.Button(
                    "Unload all models / free VRAM",
                    scale=0,
                )
            memory_status = gr.Markdown()
        app_views = create_app_views()
        generation_view = app_views.generation
        ltx25_view = app_views.ltx25
        music3_view = app_views.music3
        gallery_view = app_views.gallery
        api_view = app_views.api
        gallery_tab = app_views.gallery_tab
        h3_components = build_h3_view(
            generation_view,
            defaults,
            H3ViewServices(
                AUTO_RESOLUTION_MEGAPIXEL_PRESETS=AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
                AUTO_SOL_TOKEN_THRESHOLD=AUTO_SOL_TOKEN_THRESHOLD,
                DEFAULT_AUTO_RESOLUTION_MEGAPIXELS=DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
                DEFAULT_GEMINI_PROMPT_MODEL=DEFAULT_GEMINI_PROMPT_MODEL,
                DEFAULT_INPUT_IMAGE_FRAME_PRESET=DEFAULT_INPUT_IMAGE_FRAME_PRESET,
                DEFAULT_LOCAL_PROMPT_BASE_MODEL=DEFAULT_LOCAL_PROMPT_BASE_MODEL,
                DEFAULT_LTX25_MODEL=DEFAULT_LTX25_MODEL,
                DEFAULT_PROMPT_WRITER_BACKEND=DEFAULT_PROMPT_WRITER_BACKEND,
                DEFAULT_UPSCALE_RESOLUTION=DEFAULT_UPSCALE_RESOLUTION,
                DEFAULT_VIDEO_BATCH_COUNT=DEFAULT_VIDEO_BATCH_COUNT,
                DRAFT_RESOLUTIONS=DRAFT_RESOLUTIONS,
                FAST_RESOLUTIONS=FAST_RESOLUTIONS,
                GEMINI_PROMPT_MODELS=GEMINI_PROMPT_MODELS,
                GENERATION_POSTPROCESS_OPTIONS=GENERATION_POSTPROCESS_OPTIONS,
                H3_LATENT_UPSCALE_METHODS=H3_LATENT_UPSCALE_METHODS,
                H3_LATENT_UPSCALE_SPLIT=H3_LATENT_UPSCALE_SPLIT,
                H3_LATENT_UPSCALER_MODEL_CHOICES=H3_LATENT_UPSCALER_MODEL_CHOICES,
                H3_TEXT_ENCODER_CHOICES=H3_TEXT_ENCODER_CHOICES,
                IMAGE_VAE_CHOICES=IMAGE_VAE_CHOICES,
                INPUT_IMAGE_FRAME_PRESETS=INPUT_IMAGE_FRAME_PRESETS,
                INPUT_IMAGE_UPSCALE_SLOTS=INPUT_IMAGE_UPSCALE_SLOTS,
                LARGE_RESOLUTIONS=LARGE_RESOLUTIONS,
                LIGHTNING_PROMPT_MODEL=LIGHTNING_PROMPT_MODEL,
                LOCAL_PROMPT_BASE_MODELS=LOCAL_PROMPT_BASE_MODELS,
                MAX_IMAGE_FRAMES=MAX_IMAGE_FRAMES,
                MAX_VIDEO_BATCH_COUNT=MAX_VIDEO_BATCH_COUNT,
                MIN_IMAGE_FRAMES=MIN_IMAGE_FRAMES,
                MIN_VIDEO_BATCH_COUNT=MIN_VIDEO_BATCH_COUNT,
                MODEL_PROFILE_CHOICES=MODEL_PROFILE_CHOICES,
                PROMPT_WRITER_BACKENDS=PROMPT_WRITER_BACKENDS,
                RESULT_FORMATS=RESULT_FORMATS,
                SEEDVR2_MODEL_CHOICES=SEEDVR2_MODEL_CHOICES,
                SERVER_ATTENTION_BACKEND=SERVER_ATTENTION_BACKEND,
                SERVER_DENSE_ATTENTION_BACKEND=SERVER_DENSE_ATTENTION_BACKEND,
                SLA_PRESET_INPUTS=SLA_PRESET_INPUTS,
                TURBO_SETTINGS=TURBO_SETTINGS,
                UPSCALE_RESOLUTION_PRESETS=UPSCALE_RESOLUTION_PRESETS,
                compact_settings_summary=compact_settings_summary,
                generation_readiness_state=generation_readiness_state,
                mode_help=mode_help,
                reference_prompt_help=reference_prompt_help,
                resolution_summary=resolution_summary,
            ),
        )

        ltx25_components = build_ltx_view(
            ltx25_view,
            model_choices=LTX25_MODEL_CHOICES,
            defaults=LTX25_DEFAULTS,
            prompt_models=GEMINI_PROMPT_MODELS,
            default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
            workflows=tuple(LTX25_WORKFLOWS),
            initial_workflow_details=render_ltx25_workflow_details(
                next(iter(LTX25_WORKFLOWS))
            ),
            model_inventory_text=render_ltx25_official_model_inventory(),
        )
        music3_components = build_music_view(
            music3_view,
            prompt_models=GEMINI_PROMPT_MODELS,
            default_prompt_model=DEFAULT_GEMINI_PROMPT_MODEL,
            model_choices=MUSIC3_MODEL_CHOICES,
            defaults=MUSIC3_DEFAULTS,
        )
        gallery_components = build_gallery_view(
            gallery_view,
            postprocess_options=POSTPROCESS_OPTIONS,
            resolution_choices=tuple(UPSCALE_RESOLUTION_PRESETS),
            default_resolution=DEFAULT_UPSCALE_RESOLUTION,
            seedvr2_choices=SEEDVR2_MODEL_CHOICES,
            default_seedvr2=defaults["seedvr2_model"],
        )

        with gallery_view:
            gallery_settings_used = gr.HTML("Select an output to inspect its settings.")
        gallery_components.selected.change(
            render_snapshot,
            inputs=gallery_components.selected,
            outputs=gallery_settings_used,
            queue=False,
            api_name=False,
        )
        for root, view in (
            (ltx25_view, ltx25_components),
            (music3_view, music3_components),
        ):
            with root:
                metadata_view = gr.HTML(
                    "Settings used will appear with the generated result."
                )
            view.output.change(
                result_settings_for_media,
                inputs=view.output,
                outputs=metadata_view,
                queue=False,
                api_name=False,
                show_progress="hidden",
            )
        api_components = build_api_view(api_view, api_guide())
        app_components = dict(h3_components.values)
        app_components.update(
            {
                "api_components": api_components,
                "api_status": api_components.status,
                "api_stop": api_components.stop,
                "gallery_components": gallery_components,
                "gallery_tab": gallery_tab,
                "health": health,
                "ltx25_components": ltx25_components,
                "ltx25_model": ltx25_components.model,
                "ltx25_status": ltx25_components.status,
                "ltx25_stop": ltx25_components.stop,
                "memory_status": memory_status,
                "music3_components": music3_components,
                "music3_status": music3_components.status,
                "music3_stop": music3_components.stop,
                "system_summary": system_summary,
                "unload_models": unload_models,
            }
        )
        settings_controller = bind_app(
            AppComponents.from_mapping(app_components),
            AppServices(
                resolve_request_settings=resolve_request_settings,
                describe_settings=describe_settings,
                AI_POSTPROCESS_OPTIONS=AI_POSTPROCESS_OPTIONS,
                AUTO_RESOLUTION_JS=AUTO_RESOLUTION_JS,
                LTX25_UPSCALE=LTX25_UPSCALE,
                SEEDVR2_UPSCALE=SEEDVR2_UPSCALE,
                auto_resolution_from_start_frame=auto_resolution_from_start_frame,
                bind_api_view=bind_api_view,
                bind_gallery_view=bind_gallery_view,
                bind_ltx_view=bind_ltx_view,
                bind_music_view=bind_music_view,
                compile_trt_video_vae=compile_trt_video_vae,
                delete_selected_gallery_video=delete_selected_gallery_video,
                empty_generated_gallery=empty_generated_gallery,
                enhance_h3_prompt=enhance_h3_prompt,
                enhance_ltx25_prompt=enhance_ltx25_prompt,
                enhance_music3_prompt=enhance_music3_prompt,
                fbcache_preset_defaults=fbcache_preset_defaults,
                generate_for_ui=generate_for_ui,
                generate_ltx25=generate_ltx25,
                generate_music3=generate_music3,
                generate_with_ui_defaults=generate_with_ui_defaults,
                image_vae_frame_updates=image_vae_frame_updates,
                import_gallery_video=import_gallery_video,
                input_image_frame_preset_updates=input_image_frame_preset_updates,
                interrupt=interrupt,
                latent_upscale_layout_updates=latent_upscale_layout_updates,
                latent_upscale_method_layout_update=latent_upscale_method_layout_update,
                mode_layout_updates=mode_layout_updates,
                postprocess_selected_gallery_video=postprocess_selected_gallery_video,
                prepare_all_ltx25_official_models=prepare_all_ltx25_official_models,
                prepare_ltx25_official_workflow=prepare_ltx25_official_workflow,
                prompt_writer_backend_visibility=prompt_writer_backend_visibility,
                refresh_backend_views=refresh_backend_views,
                refresh_gallery=refresh_gallery,
                render_ltx25_official_model_inventory=render_ltx25_official_model_inventory,
                render_ltx25_workflow_details=render_ltx25_workflow_details,
                resolution_control_updates=resolution_control_updates,
                resolution_choice_updates=resolution_choice_updates,
                resolution_info_preview=resolution_info_preview,
                result_format_layout_updates=result_format_layout_updates,
                save_selected_image_frames=save_selected_image_frames,
                select_all_image_frames=select_all_image_frames,
                select_gallery_video=select_gallery_video,
                unload_all_models=unload_all_models,
                upscale_selected_input_images=upscale_selected_input_images,
            ),
        )
        browser_settings = {
            **{
                f"h3.{name}": component
                for name, component in h3_components.values.items()
            },
            **{
                f"ltx25.{name}": component
                for name, component in vars(ltx25_components).items()
            },
            **{
                f"music3.{name}": component
                for name, component in vars(music3_components).items()
            },
            **{
                f"gallery.{name}": component
                for name, component in vars(gallery_components).items()
            },
        }
        bind_browser_settings(
            demo,
            browser_settings,
            controller=settings_controller,
        )
    return demo


def selftest() -> None:
    # Alias __main__ so mocks target this exact application instance.
    sys.modules.setdefault("gradio_app", sys.modules[__name__])
    from tests.legacy_selftest import selftest as run_contracts

    run_contracts()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    allowed_paths = [
        str(OUTPUT_DIR.resolve()),
        str(OUTPUTS_DIR.resolve()),
    ]
    print(
        "[h3-ui] Runtime configuration:",
        {
            "comfy_url": COMFY_URL,
            "comfy_dir": str(COMFY_DIR),
            "models_config": str(MODELS_CONFIG),
            "comfy_output": str(OUTPUT_DIR),
            "gradio_output": str(OUTPUTS_DIR),
            "attention_backend": SERVER_ATTENTION_BACKEND,
            "dense_attention_backend": SERVER_DENSE_ATTENTION_BACKEND,
            "allowed_paths": allowed_paths,
        },
        flush=True,
    )
    demo = build_ui().queue(default_concurrency_limit=1, max_size=8)
    app = build_server(demo, allowed_paths)
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share_enabled = os.getenv("GRADIO_SHARE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    shutdown_requested = threading.Event()

    def handle_shutdown(signum, frame):
        shutdown_requested.set()
        server.should_exit = True
        server.force_exit = True

    try:
        signal.signal(signal.SIGINT, handle_shutdown)
    except (ValueError, AttributeError):
        pass
    try:
        signal.signal(signal.SIGTERM, handle_shutdown)
    except (ValueError, AttributeError):
        pass
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, handle_shutdown)
        except (ValueError, AttributeError):
            pass

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            **UVICORN_WEBSOCKET_OPTIONS,
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if share_enabled:
        try:
            # The app is mounted into a custom FastAPI server, so demo.launch()
            # cannot create the tunnel without starting a second server. Use
            # the same Gradio tunnel implementation against this server.
            share_url = gradio_networking.setup_tunnel(
                local_host="127.0.0.1",
                local_port=port,
                share_token=demo.share_token,
                share_server_address=getattr(demo, "share_server_address", None),
                share_server_tls_certificate=getattr(
                    demo, "share_server_tls_certificate", None
                ),
            )
            print(f"[h3-ui] Public Gradio URL: {share_url}", flush=True)
        except Exception as exc:
            print(
                f"[h3-ui] Could not create Gradio share link: {exc}",
                flush=True,
            )

    try:
        while server_thread.is_alive() and not shutdown_requested.is_set():
            server_thread.join(timeout=0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.should_exit = True
        server.force_exit = True
        server_thread.join(timeout=3)
        try:
            demo.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
