"""Runtime configuration and stable Gradio API adapters."""

from __future__ import annotations
from fastapi import Response
from h3_models import (
    DEFAULT_MUSIC3_MODEL,
    LTX25_ICLORA_MODEL_KEYS,
)
from h3_requirements import LTX25_WORKFLOW_FILENAMES
from h3_prompt_rewriter import (
    rewrite_prompt as rewrite_local_h3_prompt,
)
from h3_app.contracts import GENERATION_FIELDS
from h3_app.server import (
    _proxy_headers,
    _rewrite_comfy_text,
    _comfy_upstream_path,
    _append_set_cookies,
)
from h3_app.graph import Graph
from h3_app.catalog import (
    CHUNK_FEED_FORWARD_NODE,
    COMFY_UPSCALE_OPTIONS,
    LTX25_CQ_ENHANCER,
    LTX25_REFINE_DETAILS,
    LTX25_RESTORE,
    LTX25_DECOMPRESSION,
    LTX25_DEBLUR,
    LTX25_SDR_TO_HDR,
    CORE_LORA_LOADER_NODE,
    CORE_SAMPLER_NODE,
    FUSED_MODULATION_NODE,
    H3_COMBINE_AV_LATENT_NODE,
    H3_CONDITIONING_CACHE_NODE,
    H3_IMAGE_SLICES_NODE,
    H3_LATENT_UPSCALER_NODE,
    H3_NVENC_SAVE_NODE,
    H3_REFINEMENT_COMPILER_GUARD_NODE,
    H3_SEMANTIC_BRIDGE_NODE,
    H3_SEPARATE_AV_LATENT_NODE,
    H3_SIGMA_SHIFT_NODE,
    H3_SINGLE_FRAME_VAE_LOADER_NODE,
    H3_SPLIT_SPATIAL_PARAMS_NODE,
    H3_SPLIT_TEMPORAL_PARAMS_NODE,
    H3_SPLIT_UPSCALE_NODE,
    H3_STAGE_OFFLOAD_NODE,
    H3_STAGE_OFFLOAD_POLICY_NODE,
    LARRY_TURBO,
    LARRY_TURBO_LORA_NODE,
    LARRY_TURBO_SAMPLER_NODE,
    LIGHTNING_API_ROOT,
    LIGHTX2V_4STEP_TURBO,
    LIGHTX2V_8STEP_TURBO,
    LIGHTX2V_BYPASS_LORA_NODE,
    LTX25_SIGMAS,
    OFFICIAL_IMAGE_VAE,
    RESOLUTION_TIERS,
    SAGE_ATTENTION_NODE,
    SAMPLING_PRESETS,
    SAMPLING_PRESET_TEXT_ENCODERS,
    SINGLE_FRAME_IMAGE_VAE,
    SLA_ATTENTION_NODE,
    SOL_ATTENTION_NODE,
)
from h3_app.policy import (
    active_fl2va_voice_references,
    collect_reference_slots,
    frame_length,
    h3_latent_upscale_dimensions,
    image_sampling_length,
    ltx25_frame_length,
    resolve_cache_policy,
    resolve_h3_split_upscale_config,
    resolve_sla_preset,
    selected_image_sampling_length,
    single_frame_image_sampling_length,
    turbo_sampler_name,
    turbo_strength_for,
    turbo_uses_custom_nodes,
)
from h3_app.workflows.music import build_music3_graph, required_music3_nodes
from h3_app.status import StageTimings, graph_class_types, node_stage
from h3_app.processes import run_media_process

import inspect
import json
import shutil
import httpx
from .prompt_controller import PromptController, PromptServices
from .model_controller import ModelController, ModelServices
from .media_controller import (
    MediaController,
    MediaServices,
    GalleryMutationResult,
    GalleryPostprocessResult,
    GalleryMediaMutationResult,
    GalleryMediaPostprocessResult,
)

import argparse
import html
import math
import os
import random
import re
import signal
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlsplit, urlunsplit
import gradio as gr
from gradio import networking as gradio_networking
import requests
import uvicorn
import websocket
from fastapi import FastAPI
from h3_models import (
    DEFAULT_H3_LATENT_UPSCALER_MODEL,
    DEFAULT_LTX25_MODEL,
    DEFAULT_SEEDVR2_MODEL,
    H3_LATENT_UPSCALER_MODEL_CHOICES,
    H3_TEXT_ENCODER_CHOICES,
    LTX25_MODEL_CHOICES,
    MODEL_SPECS,
    MUSIC3_MODEL_CHOICES,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
    YUE2_MODEL_CHOICES,
    SEEDVR2_MODEL_CHOICES,
    resolve_hf_token,
    stale_model_keys,
    sync_models,
)
from h3_prompt_rewriter import (
    BASE_MODEL_CHOICES as LOCAL_PROMPT_BASE_MODELS,
    DEFAULT_BASE_MODEL_LABEL as DEFAULT_LOCAL_PROMPT_BASE_MODEL,
    unload_prompt_rewriter,
)
from h3_ui.presentation import (
    backend_status_html,
    generation_readiness as generation_readiness_state,
    mode_presentation,
    result_format_presentation,
)
from h3_ui.qwen_bindings import qwen_resolution_preset_values
from h3_ui.styles import H3_SETUP_CSS
from dataclasses import asdict
from h3_app.settings import (
    GenerationRequest,
    ResolutionContext,
    resolve_settings,
    preset_settings,
)
from h3_app.contracts import GenerationArguments
from h3_app import server as server_routes
from h3_app.comfy import ComfyClient
from h3_app.jobs import gpu_maintenance, JOBS, CURRENT_JOB
from h3_app.provenance import (
    RUN_CONTEXT,
    write_snapshot,
    read_snapshot,
    render_snapshot,
    snapshot_path,
)
from h3_ui.settings_presentation import render_settings
from h3_app.errors import H3Error
from h3_app.catalog import (
    AI_POSTPROCESS_OPTIONS,
    AUDIO_EXTENSIONS,
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
    AUTO_SOL_TOKEN_THRESHOLD,
    COMFY_POSTPROCESS_OPTIONS,
    LTX25_POSTPROCESS_MODELS,
    LTX25_SAME_RESOLUTION_OPTIONS,
    DEFAULT_ACCELERATOR,
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
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
    GEMINI_PROMPT_MODELS,
    GENERATION_POSTPROCESS_OPTIONS,
    H3_LATENT_UPSCALE_METHODS,
    H3_LATENT_UPSCALE_SPLIT,
    H3_LATENT_UPSCALE_STANDARD,
    IMAGE_EXTENSIONS,
    IMAGE_VAE_CHOICES,
    INPUT_IMAGE_FRAME_PRESETS,
    INPUT_IMAGE_UPSCALE_SLOTS,
    LARGE_RESOLUTIONS,
    LIGHTNING_PROMPT_MODEL,
    LTX25_DEFAULTS,
    LTX25_UPSCALE,
    LTX25_WORKFLOWS,
    MAX_IMAGE_FRAMES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MAX_VIDEO_BATCH_COUNT,
    MIN_IMAGE_FRAMES,
    MIN_VIDEO_BATCH_COUNT,
    MODEL_PROFILE_CHOICES,
    MUSIC3_DEFAULTS,
    QWEN_IMAGE21_DEFAULTS,
    QWEN_EDIT_SIZE_MATCH,
    QWEN_EDIT_SIZE_MAX,
    QWEN_EDIT_SIZE_MANUAL,
    YUE2_DEFAULTS,
    POSTPROCESS_OPTIONS,
    PROMPT_WRITER_BACKENDS,
    RESULT_FORMATS,
    SEEDVR2_UPSCALE,
    SLA_PRESET_INPUTS,
    SWIFTVR_UPSCALE,
    TURBO_SETTINGS,
    UI_DEFAULTS,
    UPSCALE_RESOLUTION_PRESETS,
    UVICORN_WEBSOCKET_OPTIONS,
    VIDEO_EXTENSIONS,
)
from h3_app.policy import (
    H3SplitUpscaleConfig,
    auto_resolution_pixel_cap,
    estimate_packed_tokens,
    normalize_image_vae,
    normalize_paths,
    normalize_result_format,
    resolution_choice_values,
    resolution_for_aspect_ratio,
    resolution_summary,
    resolve_h3_latent_upscale_method,
    snap64,
    snap_to_grid,
    turbo_steps_for,
    upscale_target_dimensions,
    validate_resolution,
)
from h3_app.model_types import (
    ModelConfig,
    ModelProfile,
    h3_text_encoder_settings,
    ltx25_official_inventory_keys,
    ltx25_workflow_entry,
    ltx25_workflow_model_keys,
    model_file_is_ready,
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
from h3_app.status import progress_status
from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.generation import (
    requests as generation_requests,
    services as generation_services,
)
from h3_app.generation import (
    h3 as h3_generation,
    ltx as ltx_generation,
    music as music_generation,
    qwen as qwen_generation,
    yue2 as yue2_generation,
)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
RUNTIME = RuntimeConfig.from_environment(SCRIPT_DIR)
COMFY_URL = RUNTIME.comfy_url
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
REQUEST_TIMEOUT = RUNTIME.request_timeout
GENERATION_TIMEOUT = RUNTIME.generation_timeout
POLL_SECONDS = RUNTIME.poll_seconds
OUTPUTS_DIR = RUNTIME.outputs_dir
GALLERY_THUMBNAILS_DIR = RUNTIME.gallery_thumbnails_dir
GALLERY_LIMIT = RUNTIME.gallery_limit
GALLERY_METADATA_CACHE_LIMIT = RUNTIME.gallery_metadata_cache_limit
HTTP = requests.Session()


def _promptcontroller():
    return PromptController(
        PromptServices(
            H3Error=H3Error,
            JOBS=JOBS,
            Path=Path,
            QWEN_EDIT_SIZE_MANUAL=QWEN_EDIT_SIZE_MANUAL,
            QWEN_EDIT_SIZE_MATCH=QWEN_EDIT_SIZE_MATCH,
            QWEN_EDIT_SIZE_MAX=QWEN_EDIT_SIZE_MAX,
            _runtime_config=_runtime_config,
            gr=gr,
            nullcontext=nullcontext,
            prompt_service=prompt_service,
            qwen_generation=qwen_generation,
            qwen_edit_size_flags=qwen_edit_size_flags,
        )
    )


def _modelcontroller():
    return ModelController(
        ModelServices(
            COMFY_DIR=COMFY_DIR,
            H3Error=H3Error,
            H3_TEXT_ENCODER_CHOICES=H3_TEXT_ENCODER_CHOICES,
            MODELS_CONFIG=MODELS_CONFIG,
            MODEL_SPECS=MODEL_SPECS,
            _runtime_config=_runtime_config,
            gr=gr,
            ltx25_official_inventory_keys=ltx25_official_inventory_keys,
            ltx25_workflow_entry=ltx25_workflow_entry,
            ltx25_workflow_model_keys=ltx25_workflow_model_keys,
            model_file_is_ready=model_file_is_ready,
            model_service=model_service,
            resolve_hf_token=resolve_hf_token,
            stale_model_keys=stale_model_keys,
            sync_models=sync_models,
            unload_comfy_models=unload_comfy_models,
            load_model_config=load_model_config,
            render_ltx25_official_model_inventory=render_ltx25_official_model_inventory,
            _prepare_ltx25_model_set=_prepare_ltx25_model_set,
            ensure_trt_video_vae_engine=ensure_trt_video_vae_engine,
        )
    )


def _mediacontroller():
    return MediaController(
        MediaServices(
            AUDIO_EXTENSIONS=AUDIO_EXTENSIONS,
            COMFY_POSTPROCESS_OPTIONS=COMFY_POSTPROCESS_OPTIONS,
            GALLERY_THUMBNAILS_DIR=GALLERY_THUMBNAILS_DIR,
            H3Error=H3Error,
            IMAGE_EXTENSIONS=IMAGE_EXTENSIONS,
            LTX25_POSTPROCESS_MODELS=LTX25_POSTPROCESS_MODELS,
            LTX25_SAME_RESOLUTION_OPTIONS=LTX25_SAME_RESOLUTION_OPTIONS,
            OUTPUTS_DIR=OUTPUTS_DIR,
            POSTPROCESS_OPTIONS=POSTPROCESS_OPTIONS,
            Path=Path,
            SEEDVR2_UPSCALE=SEEDVR2_UPSCALE,
            SWIFTVR_UPSCALE=SWIFTVR_UPSCALE,
            UPSCALE_RESOLUTION_PRESETS=UPSCALE_RESOLUTION_PRESETS,
            VIDEO_EXTENSIONS=VIDEO_EXTENSIONS,
            _runtime_config=_runtime_config,
            build_seedvr2_image_upscale_graph=build_seedvr2_image_upscale_graph,
            build_upscale_graph=build_upscale_graph,
            cleanup_upscale_clip_batch=cleanup_upscale_clip_batch,
            concat_upscaled_clips=concat_upscaled_clips,
            copy_media=copy_media,
            ensure_ltx25_upscale_models=ensure_ltx25_upscale_models,
            ensure_seedvr2_upscale_models=ensure_seedvr2_upscale_models,
            gallery_store=gallery_store,
            gr=gr,
            input_image_upscale_dimensions=input_image_upscale_dimensions,
            load_model_config=load_model_config,
            object_info=object_info,
            poll_comfy_progress=poll_comfy_progress,
            postprocess_swiftvr_video=postprocess_swiftvr_video,
            postprocess_video=postprocess_video,
            prepare_upscale_clip_batch=prepare_upscale_clip_batch,
            probe_video_metadata=probe_video_metadata,
            progress_status=progress_status,
            random=random,
            required_seedvr2_image_upscale_nodes=required_seedvr2_image_upscale_nodes,
            required_upscale_nodes=required_upscale_nodes,
            resolve_output=resolve_output,
            resolve_seedvr2_input_upscale_outputs=resolve_seedvr2_input_upscale_outputs,
            snapshot_path=snapshot_path,
            stage_file=stage_file,
            stream_comfy_progress=stream_comfy_progress,
            submit_prompt=submit_prompt,
            time=time,
            unload_comfy_models=unload_comfy_models,
            upscale_target_dimensions=upscale_target_dimensions,
            uuid=uuid,
            wait_for_history=wait_for_history,
            write_snapshot=write_snapshot,
            gallery_audio_paths=gallery_audio_paths,
            empty_generated_gallery=empty_generated_gallery,
            gallery_processed_result=gallery_processed_result,
            generated_video_family=generated_video_family,
            refresh_gallery=refresh_gallery,
            gallery_media_processed_result=gallery_media_processed_result,
            gallery_video_paths=gallery_video_paths,
            absolute_gallery_media_download_url=absolute_gallery_media_download_url,
            managed_gallery_audio_path=managed_gallery_audio_path,
            managed_video_path=managed_video_path,
            gallery_image_paths=gallery_image_paths,
            postprocess_selected_gallery_image=postprocess_selected_gallery_image,
            managed_gallery_image_path=managed_gallery_image_path,
            gallery_media_download_path=gallery_media_download_path,
            refresh_gallery_page=refresh_gallery_page,
            gallery_mutation_result=gallery_mutation_result,
            gallery_thumbnail=gallery_thumbnail,
            refresh_media_gallery=refresh_media_gallery,
            gallery_progress_result=gallery_progress_result,
            delete_selected_gallery_video=delete_selected_gallery_video,
            gallery_preview_updates=gallery_preview_updates,
            absolute_video_download_url=absolute_video_download_url,
            gallery_media_progress_result=gallery_media_progress_result,
            gallery_thumbnail_path=gallery_thumbnail_path,
            forget_gallery_metadata=forget_gallery_metadata,
            gallery_media_mode=gallery_media_mode,
            select_gallery_video=select_gallery_video,
            video_download_path=video_download_path,
            postprocess_selected_gallery_video=postprocess_selected_gallery_video,
            gallery_image_resolution_text=gallery_image_resolution_text,
            gallery_media_mutation_result=gallery_media_mutation_result,
            absolute_video_url=absolute_video_url,
            gallery_resolution_text=gallery_resolution_text,
            refresh_media_page=refresh_media_page,
        )
    )


def _runtime_config() -> RuntimeConfig:
    """Build an explicit service context from startup configuration."""
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
    css = getattr(demo, "h3_css", None)
    if not isinstance(css, str):
        css = H3_SETUP_CSS
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
            IMAGE_EXTENSIONS,
            AUDIO_EXTENSIONS,
            css,
        ),
    )


def _gemini_api_key(temporary_key: str | None) -> str:
    return _promptcontroller()._gemini_api_key(temporary_key)


def _lightning_api_key(temporary_key: str | None) -> str:
    return _promptcontroller()._lightning_api_key(temporary_key)


def _uploaded_media_path(value: Any) -> Path | None:
    return _promptcontroller()._uploaded_media_path(value)


def _gemini_mime_type(path: Path) -> str:
    return _promptcontroller()._gemini_mime_type(path)


def _gemini_error(response: requests.Response, action: str) -> H3Error:
    return _promptcontroller()._gemini_error(response, action)


def _upload_gemini_file(
    session: requests.Session, path: Path, api_key: str
) -> dict[str, Any]:
    return _promptcontroller()._upload_gemini_file(session, path, api_key)


def _wait_for_gemini_file(
    session: requests.Session, file_info: dict[str, Any], api_key: str
) -> dict[str, Any]:
    return _promptcontroller()._wait_for_gemini_file(session, file_info, api_key)


def _active_prompt_media(
    mode: str,
    first_image: Any,
    last_image: Any,
    reference_images: Iterable[Any],
    reference_videos: Iterable[Any],
    reference_audios: Iterable[Any],
) -> list[tuple[str, Path]]:
    return _promptcontroller()._active_prompt_media(
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
    return _promptcontroller()._enhance_prompt_from_media(
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
    backend: str = "Lightning AI",
    lightning_api_key: str = "",
) -> tuple[str, str, str]:
    return _promptcontroller().enhance_music3_prompt(
        prompt,
        model,
        temporary_api_key,
        lyrics,
        ref_image_1,
        ref_image_2,
        ref_image_3,
        backend,
        lightning_api_key,
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
    backend: str = "Lightning AI",
    lightning_api_key: str = "",
) -> tuple[str, str]:
    return _promptcontroller().enhance_ltx25_prompt(
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
        backend,
        lightning_api_key,
    )


def qwen_edit_size_flags(edit_size: str) -> tuple[bool, bool]:
    return _promptcontroller().qwen_edit_size_flags(edit_size)


def enhance_qwen_image21_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    reference_images: Any,
    width: int,
    height: int,
    backend: str = "Lightning AI",
    lightning_api_key: str = "",
    edit_size: str = QWEN_EDIT_SIZE_MATCH,
) -> tuple[str, str]:
    return _promptcontroller().enhance_qwen_image21_prompt(
        prompt,
        model,
        temporary_api_key,
        mode,
        reference_images,
        width,
        height,
        backend,
        lightning_api_key,
        edit_size,
    )


def enhance_yue2_prompt(
    style: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    mode: str,
    duration: float,
    backend: str = "Lightning AI",
    lightning_api_key: str = "",
) -> tuple[str, str, str]:
    return _promptcontroller().enhance_yue2_prompt(
        style,
        model,
        temporary_api_key,
        lyrics,
        mode,
        duration,
        backend,
        lightning_api_key,
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
    return _promptcontroller()._enhance_h3_prompt_with_gemini(
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
    return _promptcontroller()._enhance_h3_prompt_with_lightning(
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
    )


def fl2va_prompt_voice_context(prompt: str, mode: str, *slots: Any):
    return _promptcontroller().fl2va_prompt_voice_context(prompt, mode, *slots)


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
    return _promptcontroller().enhance_h3_prompt(
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
    )


def prompt_writer_backend_visibility(backend: str) -> tuple[Any, Any, Any]:
    return _promptcontroller().prompt_writer_backend_visibility(backend)


def load_model_config() -> ModelConfig:
    return _modelcontroller().load_model_config()


def trt_vae_decoder_paths(models: ModelConfig) -> tuple[Path, Path, Path]:
    return _modelcontroller().trt_vae_decoder_paths(models)


def _load_trt_vae_compiler(node_path: Path) -> Any:
    return _modelcontroller()._load_trt_vae_compiler(node_path)


def trt_vae_runtime_fingerprint(models: ModelConfig | None = None) -> str:
    return _modelcontroller().trt_vae_runtime_fingerprint(models)


def is_trt_engine_loadable(engine_path: Path) -> bool:
    return _modelcontroller().is_trt_engine_loadable(engine_path)


def trt_vae_engine_is_current(models: ModelConfig) -> bool:
    return _modelcontroller().trt_vae_engine_is_current(models)


def ensure_h3_text_encoder(models: ModelConfig, model_choice: str) -> tuple[str, bool]:
    return _modelcontroller().ensure_h3_text_encoder(models, model_choice)


def text_encoder_offload_update(model_choice: str):
    return _modelcontroller().text_encoder_offload_update(model_choice)


def ensure_h3_semantic_bridge() -> None:
    return _modelcontroller().ensure_h3_semantic_bridge()


def ensure_h3_latent_upscaler_model(model_choice: str) -> bool:
    return _modelcontroller().ensure_h3_latent_upscaler_model(model_choice)


def ensure_profile_model(profile_key: str, profile: ModelProfile, mode: str) -> bool:
    return _modelcontroller().ensure_profile_model(profile_key, profile, mode)


def ensure_turbo_lora(models: ModelConfig, turbo_variant: str, mode: str) -> bool:
    return _modelcontroller().ensure_turbo_lora(models, turbo_variant, mode)


def ensure_base_video_vae(models: ModelConfig) -> bool:
    return _modelcontroller().ensure_base_video_vae(models)


def ensure_audio_vae(models: ModelConfig) -> bool:
    return _modelcontroller().ensure_audio_vae(models)


def ensure_int8_video_vae(models: ModelConfig, *, lynnreal: bool = False) -> bool:
    return _modelcontroller().ensure_int8_video_vae(models, lynnreal=lynnreal)


def ensure_trt_video_vae(models: ModelConfig, *, require_engine: bool = True) -> bool:
    return _modelcontroller().ensure_trt_video_vae(
        models, require_engine=require_engine
    )


def _build_trt_video_vae_engine(models: ModelConfig, progress: Any) -> None:
    return _modelcontroller()._build_trt_video_vae_engine(models, progress)


def ensure_trt_video_vae_engine(
    models: ModelConfig, *, force: bool = False, progress=gr.Progress(track_tqdm=False)
) -> bool:
    return _modelcontroller().ensure_trt_video_vae_engine(
        models, force=force, progress=progress
    )


def compile_trt_video_vae(progress=gr.Progress(track_tqdm=False)) -> str:
    return _modelcontroller().compile_trt_video_vae(progress)


def ensure_single_frame_image_vae(models: ModelConfig) -> bool:
    return _modelcontroller().ensure_single_frame_image_vae(models)


def missing_ltx25_model_names(model_choice: str = DEFAULT_LTX25_MODEL) -> list[str]:
    return _modelcontroller().missing_ltx25_model_names(model_choice)


def ensure_ltx25_models(model_choice: str = DEFAULT_LTX25_MODEL) -> bool:
    return _modelcontroller().ensure_ltx25_models(model_choice)


def ensure_ltx25_ingredients_model() -> bool:
    return _modelcontroller().ensure_ltx25_ingredients_model()


def render_ltx25_official_model_inventory() -> str:
    return _modelcontroller().render_ltx25_official_model_inventory()


def render_ltx25_workflow_details(workflow_label: str) -> str:
    return _modelcontroller().render_ltx25_workflow_details(workflow_label)


def _prepare_ltx25_model_set(required_keys: Iterable[str], label: str):
    return _modelcontroller()._prepare_ltx25_model_set(required_keys, label)


def prepare_ltx25_official_workflow(workflow_label: str):
    return _modelcontroller().prepare_ltx25_official_workflow(workflow_label)


def prepare_all_ltx25_official_models():
    return _modelcontroller().prepare_all_ltx25_official_models()


def ensure_seedvr2_upscale_models(models: ModelConfig, model_choice: str) -> bool:
    return _modelcontroller().ensure_seedvr2_upscale_models(models, model_choice)


def ensure_ltx25_upscale_models(
    model_choice: str = DEFAULT_LTX25_MODEL, *, option: str = LTX25_UPSCALE
) -> bool:
    return _modelcontroller().ensure_ltx25_upscale_models(model_choice, option=option)


def missing_music3_model_names(model_choice: str) -> list[str]:
    return _modelcontroller().missing_music3_model_names(model_choice)


def ensure_music3_models(model_choice: str) -> bool:
    return _modelcontroller().ensure_music3_models(model_choice)


def missing_qwen_image21_model_names(
    model_choice: str, text_encoder_choice: str, turbo_variant: str = "Off"
) -> list[str]:
    return _modelcontroller().missing_qwen_image21_model_names(
        model_choice, text_encoder_choice, turbo_variant
    )


def ensure_qwen_image21_models(
    model_choice: str, text_encoder_choice: str, turbo_variant: str = "Off"
) -> bool:
    return _modelcontroller().ensure_qwen_image21_models(
        model_choice, text_encoder_choice, turbo_variant
    )


def missing_yue2_model_names(model_choice: str) -> list[str]:
    return _modelcontroller().missing_yue2_model_names(model_choice)


def ensure_yue2_models(model_choice: str) -> bool:
    return _modelcontroller().ensure_yue2_models(model_choice)


def api_get(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).get(path, **kwargs)


def api_post(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).post(path, **kwargs)


def object_info() -> dict[str, Any]:
    return api_get("/object_info").json()


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
        resolved_width, resolved_height = (864, 480)
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
    return (width, height, resolution_summary(width, height))


def resolution_choice_updates(
    name: str, tier: str, latent_upscale: bool, result_format: str
) -> tuple[int | float, int | float, str]:
    """Resolve and align a preset before updating its controls.

    Keeping preset lookup and alignment in one callback prevents the UI from
    briefly writing the raw preset dimensions before latent upscale snaps them
    to its required 64-pixel grid.
    """
    width, height, _summary = resolution_choice_values(name, tier)
    return resolution_control_updates(width, height, latent_upscale, result_format)


def start_frame_generation_resolution(
    first_image: Any, *, alignment: int
) -> tuple[int, int] | None:
    paths = normalize_paths(first_image)
    if not paths:
        return None
    try:
        from PIL import Image

        with Image.open(paths[0]) as image:
            return resolution_for_aspect_ratio(
                *image.size, preserve_native=True, alignment=alignment
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
        return (32, 32)
    alignment = 64 if latent_upscale else 32
    if normalized_result == "Image" and mode == "First / last frame" and first_image:
        start_resolution = start_frame_generation_resolution(
            first_image, alignment=alignment
        )
        if start_resolution is not None:
            return start_resolution
    return (snap_to_grid(width, alignment), snap_to_grid(height, alignment))


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
        return (False, tokens, "forced Comfy Kitchen")
    if requested in {"sage", "sage 2", "sage2"}:
        return (False, tokens, "forced Sage 2")
    if requested in {"sla", "sla attention", "sparse-linear"}:
        return (False, tokens, "forced SLA")
    if SERVER_ATTENTION_BACKEND != "sol":
        return (False, tokens, "Sol backend unavailable")
    if requested in {"sol-attn", "sol", "sparse"}:
        return (True, tokens, "forced Sol-Attn")
    if mode == "Reference media":
        prefix = "Auto Turbo" if use_turbo else "Auto"
        return (True, tokens, f"{prefix}: reference mode")
    if use_turbo:
        enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
        return (
            enabled,
            tokens,
            f"Auto Turbo: {tokens:,} target tokens {('≥' if enabled else '<')} {AUTO_SOL_TOKEN_THRESHOLD:,}",
        )
    enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
    return (
        enabled,
        tokens,
        f"Auto: {tokens:,} target tokens {('≥' if enabled else '<')} {AUTO_SOL_TOKEN_THRESHOLD:,}",
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
    display_width, display_height = (current_width, current_height)
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
        gr.update(visible=True) if presentation.is_video else gr.update(visible=False),
        gr.update(visible=is_image),
        gr.update(visible=is_audio),
        (
            gr.update(interactive=True)
            if presentation.is_video
            else gr.update(interactive=False)
        ),
        gr.update(interactive=False) if is_audio else gr.update(interactive=True),
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
        info=(
            "Effective output: one frame with the 500K decoder."
            if single
            else "Choose 1–20 decoded frames."
        ),
    )


def latent_upscale_layout_updates(
    enabled: bool, width: int | float, height: int | float, result_format: str
):
    if normalize_result_format(result_format) == "Audio":
        return (
            gr.update(visible=False),
            width,
            height,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        )
    if enabled:
        resolved_width, resolved_height = (snap64(width), snap64(height))
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
        visible=resolve_h3_latent_upscale_method(method) == H3_LATENT_UPSCALE_SPLIT
    )


def generation_mode_defaults(name: str, turbo_variant: str = DEFAULT_TURBO):
    """Normal/Turbo is independent from the selected base-model profile.

    Important: when entering Turbo, do not change preset.value. Changing it
    would fire preset.change() and race with the Turbo Steps update.
    """
    if str(name).strip().lower() == "turbo":
        return (
            gr.update(interactive=True),
            gr.update(value=turbo_steps_for(turbo_variant), interactive=True),
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
        return (gr.update(), gr.update())
    return (gr.update(value=turbo_steps_for(turbo_variant), interactive=True), "simple")


def fbcache_preset_defaults(name: str):
    key = str(name).strip().lower()
    if key == "safe":
        values = (0.08, 0.1, 0.95, 2)
        interactive = False
    elif key == "aggressive":
        values = (0.12, 0.1, 0.95, 2)
        interactive = False
    elif key == "custom":
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    else:
        values = (0.1, 0.1, 0.95, 2)
        interactive = False
    return tuple((gr.update(value=value, interactive=interactive) for value in values))


def file_content_sha256(path: Path) -> str:
    return staging.file_content_sha256(path)


def staged_input_is_ready(path: Path) -> bool:
    return staging.staged_input_is_ready(path)


def materialize_staged_input(
    source: Path, destination: Path, *, transcode_video: bool
) -> None:
    return staging.materialize_staged_input(
        source, destination, transcode_video=transcode_video
    )


def stage_file(
    path: str, category: str, transcode_video: bool = False, reuse: bool = False
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
    use_lynnreal_vae: bool = False,
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    refinement_lora_name: str | None = None,
    refinement_variant: str | None = None,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    encoder_small_input: bool = False,
    reuse_unchanged_inputs: bool = True,
    stage_model_offload: bool = False,
    smart_stage_offload: bool = False,
    semantic_bridge: bool = False,
    semantic_bridge_alpha: float = 0.1,
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
        for path in voice_reference_audios or []
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
        use_lynnreal_vae=use_lynnreal_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        refinement_lora_name=refinement_lora_name,
        refinement_variant=refinement_variant,
        latent_split_config=latent_split_config,
        result_format=result_format,
        image_frames=image_frames,
        image_vae=image_vae,
        text_encoder_name=text_encoder_name,
        encoder_small_input=encoder_small_input,
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
    use_lynnreal_vae: bool = False,
    use_int8_vae: bool = False,
    use_trt_vae: bool = False,
    use_sage: bool = False,
    use_sla: bool = False,
    sla_preset: str = DEFAULT_SLA_PRESET,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    refinement_lora_name: str | None = None,
    refinement_variant: str | None = None,
    latent_split_config: H3SplitUpscaleConfig | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    image_vae: str = DEFAULT_IMAGE_VAE,
    text_encoder_name: str | None = None,
    encoder_small_input: bool = False,
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
        use_lynnreal_vae=use_lynnreal_vae,
        use_trt_vae=use_trt_vae,
        use_sage=use_sage,
        use_sla=use_sla,
        sla_preset=sla_preset,
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
        refinement_lora_name=refinement_lora_name,
        refinement_variant=refinement_variant,
        latent_split_config=latent_split_config,
        result_format=result_format,
        image_frames=image_frames,
        image_vae=image_vae,
        text_encoder_name=text_encoder_name,
        encoder_small_input=encoder_small_input,
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
    reference_images: tuple[str, ...] = (),
) -> dict[str, Any]:
    first_image = stage_file(first_image, "ltx25_keyframes") if first_image else None
    middle_image = stage_file(middle_image, "ltx25_keyframes") if middle_image else None
    end_image = stage_file(end_image, "ltx25_keyframes") if end_image else None
    reference_sheet = None
    if reference_images:
        from PIL import Image, ImageOps

        count = len(reference_images)
        columns = min(3, count)
        rows = math.ceil(count / columns)
        panel_width, panel_height = (width // columns, height // rows)
        with tempfile.TemporaryDirectory() as temporary:
            sheet = Image.new("RGB", (width, height), "black")
            for index, path in enumerate(reference_images):
                with Image.open(path) as source:
                    panel = ImageOps.contain(
                        ImageOps.exif_transpose(source).convert("RGB"),
                        (panel_width - 8, panel_height - 8),
                    )
                x = index % columns * panel_width + (panel_width - panel.width) // 2
                y = index // columns * panel_height + (panel_height - panel.height) // 2
                sheet.paste(panel, (x, y))
            sheet_path = Path(temporary) / "reference_sheet.png"
            sheet.save(sheet_path)
            reference_sheet = stage_file(str(sheet_path), "ltx25_references")
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
        reference_sheet=reference_sheet,
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
    preset: str, current_width: int | float, current_height: int | float
) -> tuple[Any, Any]:
    dimensions = INPUT_IMAGE_FRAME_PRESETS.get(str(preset))
    if dimensions is None:
        return (gr.update(value=current_width), gr.update(value=current_height))
    return dimensions


def input_image_upscale_dimensions(
    image_path: str, frame_width: int | float, frame_height: int | float
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
    scale_by = min(target_width / source_width, target_height / source_height)
    if scale_by <= 1.0:
        return (source_width, source_height, source_width, source_height, 1.0)
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
    selected = list(dict.fromkeys((str(value) for value in selected_slots or [])))
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
                    history, queued_at, output_token, generated_keys
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
        f"Fit selected images into a {int(frame_width)}×{int(frame_height)} frame: {len(staged)} upscaled with SeedVR2 {model_choice}, {len(unchanged_results)} already large enough. Aspect ratios were preserved and no image was downscaled.\n\n"
        + "  \n".join(dimension_notes)
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
        match = re.fullmatch("Frame\\s+(\\d+)", str(label).strip())
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
    return (saved, f"Saved {len(saved)} selected frame(s) to `{destination}`.")


def has_encoder(name: str) -> bool:
    return media_tools.has_encoder(name)


def ensure_swiftvr_checkpoint() -> tuple[Path, bool]:
    return swiftvr.ensure_swiftvr_checkpoint(runtime=_runtime_config())


def import_swiftvr_pipeline() -> Any:
    return swiftvr.import_swiftvr_pipeline(runtime=_runtime_config())


def postprocess_swiftvr_video(
    source: Path, *, fps: float, target_width: int, target_height: int
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
    source: Path, clips: list[Path], *, option: str, duration: float, frame_count: int
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
    batch: UpscaleClipBatch | None, outputs: Iterable[Path] = ()
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
        return (f"ComfyUI VRAM release failed: {exc}.{local_note}", backend_status())


def video_download_path(video: str | Path) -> str:
    return _mediacontroller().video_download_path(video)


def managed_video_path(video: str | Path, *, require_file: bool = True) -> Path:
    return _mediacontroller().managed_video_path(video, require_file=require_file)


def absolute_video_url(
    video: str | Path, request: gr.Request, *, download: bool = False
) -> str:
    return _mediacontroller().absolute_video_url(video, request, download=download)


def absolute_video_download_url(video: str | Path, request: gr.Request) -> str:
    return _mediacontroller().absolute_video_download_url(video, request)


def gallery_video_paths(*, limit: int | None = GALLERY_LIMIT) -> list[Path]:
    return _mediacontroller().gallery_video_paths(limit=limit)


def gallery_thumbnail(video: Path) -> Path | None:
    return _mediacontroller().gallery_thumbnail(video)


def gallery_thumbnail_path(video: str | Path) -> Path:
    return _mediacontroller().gallery_thumbnail_path(video)


def gallery_video_resolution(video: Path) -> tuple[int, int] | None:
    return _mediacontroller().gallery_video_resolution(video)


def gallery_resolution_text(video: Path) -> str:
    return _mediacontroller().gallery_resolution_text(video)


def generated_video_family(video: str | Path) -> str:
    return _mediacontroller().generated_video_family(video)


def forget_gallery_metadata(video: str | Path | None = None) -> None:
    return _mediacontroller().forget_gallery_metadata(video)


def managed_gallery_image_path(image: str | Path, *, require_file: bool = True) -> Path:
    return _mediacontroller().managed_gallery_image_path(
        image, require_file=require_file
    )


def gallery_image_paths(*, limit: int | None = GALLERY_LIMIT) -> list[Path]:
    return _mediacontroller().gallery_image_paths(limit=limit)


def gallery_image_resolution_text(image: Path) -> str:
    return _mediacontroller().gallery_image_resolution_text(image)


def managed_gallery_audio_path(audio: str | Path, *, require_file: bool = True) -> Path:
    return _mediacontroller().managed_gallery_audio_path(
        audio, require_file=require_file
    )


def gallery_audio_paths(*, limit: int | None = GALLERY_LIMIT) -> list[Path]:
    return _mediacontroller().gallery_audio_paths(limit=limit)


def gallery_media_mode(mode: str) -> str:
    return _mediacontroller().gallery_media_mode(mode)


def gallery_media_download_path(media: str | Path, mode: str) -> str:
    return _mediacontroller().gallery_media_download_path(media, mode)


def absolute_gallery_media_download_url(
    media: str | Path, mode: str, request: gr.Request
) -> str:
    return _mediacontroller().absolute_gallery_media_download_url(media, mode, request)


def import_gallery_media(mode: str, uploaded_media: str | None):
    return _mediacontroller().import_gallery_media(mode, uploaded_media)


def import_gallery_video(uploaded_video: str | None) -> GalleryMutationResult:
    return _mediacontroller().import_gallery_video(uploaded_video)


GALLERY_PAGE_SIZE = 48


def refresh_gallery_page(limit: int = GALLERY_PAGE_SIZE) -> gallery_store.AssetPage:
    return _mediacontroller().refresh_gallery_page(limit)


def refresh_gallery(limit: int = GALLERY_PAGE_SIZE):
    return _mediacontroller().refresh_gallery(limit)


def select_gallery_video(
    paths: list[str], request: gr.Request, evt: gr.SelectData
) -> tuple[str | None, str, str | None]:
    return _mediacontroller().select_gallery_video(paths, request, evt)


def list_media_paths(mode):
    return _mediacontroller().list_media_paths(mode)


def refresh_media_page(
    mode: str = "Video", limit: int = GALLERY_PAGE_SIZE
) -> gallery_store.AssetPage:
    return _mediacontroller().refresh_media_page(mode, limit)


def refresh_media_gallery(mode: str = "Video", limit: int = GALLERY_PAGE_SIZE):
    return _mediacontroller().refresh_media_gallery(mode, limit)


def gallery_preview_updates(
    mode: str,
    *,
    video: str | None = None,
    image: str | None = None,
    audio: str | None = None,
) -> tuple[Any, Any, Any]:
    return _mediacontroller().gallery_preview_updates(
        mode, video=video, image=image, audio=audio
    )


def select_gallery_media(
    mode: str, paths: list[str], request: gr.Request, evt: gr.SelectData
) -> tuple[Any, Any, Any, str, str | None]:
    return _mediacontroller().select_gallery_media(mode, paths, request, evt)


def gallery_media_mutation_result(
    mode: str, message: str, *, selected_media: str | None = None, clear_selection: bool
) -> GalleryMediaMutationResult:
    return _mediacontroller().gallery_media_mutation_result(
        mode, message, selected_media=selected_media, clear_selection=clear_selection
    )


def gallery_media_progress_result(message: str) -> GalleryMediaPostprocessResult:
    return _mediacontroller().gallery_media_progress_result(message)


def gallery_media_processed_result(
    mode: str, result: Path, option: str, elapsed: float, request: gr.Request
) -> GalleryMediaPostprocessResult:
    return _mediacontroller().gallery_media_processed_result(
        mode, result, option, elapsed, request
    )


def gallery_mutation_result(
    message: str, *, selected_video: str | None = None, clear_selection: bool
) -> GalleryMutationResult:
    return _mediacontroller().gallery_mutation_result(
        message, selected_video=selected_video, clear_selection=clear_selection
    )


def gallery_progress_result(message: str) -> GalleryPostprocessResult:
    return _mediacontroller().gallery_progress_result(message)


def gallery_processed_result(
    result: Path, option: str, elapsed: float, request: gr.Request
) -> GalleryPostprocessResult:
    return _mediacontroller().gallery_processed_result(result, option, elapsed, request)


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
    return _mediacontroller().postprocess_selected_gallery_video(
        selected_video,
        option,
        seed,
        seedvr2_model,
        ltx25_model,
        ltx25_prompt,
        force_offload,
        split_upscale,
        split_seconds,
        upscale_resolution,
        request,
        progress,
    )


def postprocess_selected_gallery_image(
    selected_image: str | None,
    option: str,
    seed: int,
    seedvr2_model: str,
    force_offload: bool,
    upscale_resolution: str,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=False),
):
    return _mediacontroller().postprocess_selected_gallery_image(
        selected_image,
        option,
        seed,
        seedvr2_model,
        force_offload,
        upscale_resolution,
        request,
        progress,
    )


def postprocess_selected_gallery_media(
    mode: str,
    selected_media: str | None,
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
    return _mediacontroller().postprocess_selected_gallery_media(
        mode,
        selected_media,
        option,
        seed,
        seedvr2_model,
        ltx25_model,
        ltx25_prompt,
        force_offload,
        split_upscale,
        split_seconds,
        upscale_resolution,
        request,
        progress,
    )


def delete_selected_gallery_video(
    selected_video: str | None, confirmed: bool
) -> GalleryMutationResult:
    return _mediacontroller().delete_selected_gallery_video(selected_video, confirmed)


def delete_selected_gallery_media(
    mode: str, selected_media: str | None, confirmed: bool
) -> GalleryMediaMutationResult:
    return _mediacontroller().delete_selected_gallery_media(
        mode, selected_media, confirmed
    )


def empty_generated_gallery(
    selected_video: str | None, confirmed: bool
) -> GalleryMutationResult:
    return _mediacontroller().empty_generated_gallery(selected_video, confirmed)


def empty_generated_media_gallery(
    mode: str, selected_media: str | None, confirmed: bool
) -> GalleryMediaMutationResult:
    return _mediacontroller().empty_generated_media_gallery(
        mode, selected_media, confirmed
    )


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
                f" · {vram_free / 2 ** 30:.1f}/{vram_total / 2 ** 30:.1f} GiB VRAM free"
            )
        elif isinstance(vram_total, (int, float)):
            vram_text = f" · {vram_total / 2 ** 30:.1f} GiB VRAM"
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
                f"**{profile.label}** · FL2VA `{profile.fl2va}` · Ref2VA `{profile.ref2va}`"
            )
        for label, filename in (
            ("PDMD / 2-step", models.pdmd_2step_lora),
            ("PDMD / 4-step", models.pdmd_4step_lora),
        ):
            if filename:
                profile_lines.append(
                    f"**{label} (experimental)** | LoRA `{filename}` | FL2VA / Ref2VA | Euler/simple | strength 1.0 | downloads on first use"
                )
        if models.taomate_turbo_lora:
            profile_lines.append(
                f"**TaoMate-H3 / 3-step** | LoRA `{models.taomate_turbo_lora}` | FL2VA / Ref2VA | Euler/simple | strength 0.7 | downloads on first use"
            )
        if models.larry_turbo_lora:
            profile_lines.append(
                f"**Larry Turbo v4-600 EMA** | LoRA `{models.larry_turbo_lora}` | 6-step default | strength 1.0 | custom loader/sampler"
            )
        if models.turbo_lora:
            profile_lines.append(
                f"**LightX2V Turbo / 4-step** · FL2VA v1.2 `{models.turbo_lora}` · Ref2VA v0.1 544p `{models.turbo_ref_lora}` · strength 1.0"
            )
        if models.turbo_8step_lora:
            profile_lines.append(
                f"**LightX2V Turbo v1.0 / 8-step 768p** · FL2VA `{models.turbo_8step_lora}` · Ref2VA `{models.turbo_8step_ref_lora}` · 8-step default · strength 1.0 · FL2VA and Ref2VA"
            )
        return (
            f"Connected · {gpu}{vram_text} · sparse: {SERVER_ATTENTION_BACKEND} · dense: {SERVER_DENSE_ATTENTION_BACKEND} · memory: {SERVER_MEMORY_PROFILE} · FirstBlockCache: {fbcache_status} · EasyCache: {easycache_status}  \n"
            + "  \n".join(profile_lines)
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
            ensure_audio_vae=ensure_audio_vae,
            ensure_base_video_vae=ensure_base_video_vae,
            ensure_h3_latent_upscaler_model=ensure_h3_latent_upscaler_model,
            ensure_h3_semantic_bridge=ensure_h3_semantic_bridge,
            ensure_h3_text_encoder=ensure_h3_text_encoder,
            ensure_int8_video_vae=ensure_int8_video_vae,
            ensure_ltx25_models=ensure_ltx25_models,
            ensure_ltx25_ingredients_model=ensure_ltx25_ingredients_model,
            ensure_ltx25_upscale_models=ensure_ltx25_upscale_models,
            ensure_music3_models=ensure_music3_models,
            ensure_qwen_image21_models=ensure_qwen_image21_models,
            ensure_yue2_models=ensure_yue2_models,
            ensure_profile_model=ensure_profile_model,
            ensure_seedvr2_upscale_models=ensure_seedvr2_upscale_models,
            ensure_single_frame_image_vae=ensure_single_frame_image_vae,
            ensure_trt_video_vae_engine=ensure_trt_video_vae_engine,
            ensure_turbo_lora=ensure_turbo_lora,
            load_model_config=load_model_config,
            missing_ltx25_model_names=missing_ltx25_model_names,
            missing_music3_model_names=missing_music3_model_names,
            missing_qwen_image21_model_names=missing_qwen_image21_model_names,
            missing_yue2_model_names=missing_yue2_model_names,
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
    latent_split_fade_ratio: float = 0.5,
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
    semantic_bridge_alpha: float = 0.1,
    fl2va_audio_1: Any = None,
    fl2va_audio_2: Any = None,
    fl2va_audio_3: Any = None,
    encoder_small_input: bool = False,
    use_lynnreal_vae: bool = False,
    latent_upscale_refine_lora: str = "Same as generation",
    progress=gr.Progress(track_tqdm=False),
):
    request = generation_requests.H3Request.from_values(
        {
            "mode": mode,
            "model_profile": model_profile,
            "text_encoder": text_encoder,
            "encoder_small_input": encoder_small_input,
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
            "latent_upscale_refine_lora": latent_upscale_refine_lora,
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
            "use_lynnreal_vae": use_lynnreal_vae,
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
    reference_images: list[str] | None = None,
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
            "reference_images": tuple(reference_images or ()),
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


def generate_qwen_image21(
    mode: str,
    model_choice: str,
    text_encoder_choice: str,
    prompt: str,
    negative_prompt: str,
    reference_images: Any,
    width: int,
    height: int,
    reference_resolution: int,
    edit_size: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    cache_device: str,
    cache_dtype: str,
    attention_backend: str = "pytorch attention",
    accelerator: str = "Off",
    turbo_variant: str = "Off",
    output_resolution_1k: str | None = None,
    output_resolution_2k: str | None = None,
    batch_count: int = 1,
    batch_edit_inputs: bool = False,
    *,
    output_resolution: str | None = None,
    progress=gr.Progress(track_tqdm=False),
):
    selected_preset = output_resolution_2k or output_resolution_1k or output_resolution
    match_input_size, max_resolution = qwen_edit_size_flags(edit_size)
    if selected_preset:
        width, height = qwen_resolution_preset_values(selected_preset)
        match_input_size = False
        max_resolution = False
    uploaded = reference_images or []
    if isinstance(uploaded, (str, Path)):
        uploaded = [uploaded]
    staged = tuple(
        (
            stage_file(str(path), "qwen_image21_references", reuse=True)
            for path in uploaded
        )
    )
    request = generation_requests.QwenImage21Request(
        mode=mode,
        model_choice=model_choice,
        text_encoder_choice=text_encoder_choice,
        prompt=prompt,
        negative_prompt=negative_prompt,
        reference_images=staged,
        width=width,
        height=height,
        reference_resolution=reference_resolution,
        match_input_size=match_input_size,
        seed=seed,
        batch_count=batch_count,
        batch_edit_inputs=bool(batch_edit_inputs),
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        cache_device=cache_device,
        cache_dtype=cache_dtype,
        attention_backend=attention_backend,
        accelerator=accelerator,
        turbo_variant=turbo_variant,
        max_resolution=bool(max_resolution),
    )
    yield from qwen_generation.generate_qwen_image21(
        request, _generation_services(), _runtime_config(), progress=progress
    )


def generate_yue2(
    model_choice: str,
    style: str,
    lyrics: str,
    abc: str,
    mode: str,
    max_duration: float,
    seed: int,
    steps: int,
    cfg: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_abc_tokens: int,
    abc_temperature: float,
    abc_top_p: float,
    abc_top_k: int,
    abc_repetition_penalty: float,
    abc_penalty_window: int,
    tiled_decode: bool,
    progress=gr.Progress(track_tqdm=False),
):
    request = generation_requests.YuE2Request(
        model_choice=model_choice,
        style=style,
        lyrics=lyrics,
        abc=abc,
        mode=mode,
        max_duration=max_duration,
        seed=seed,
        steps=steps,
        cfg=cfg,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_abc_tokens=max_abc_tokens,
        abc_temperature=abc_temperature,
        abc_top_p=abc_top_p,
        abc_top_k=abc_top_k,
        abc_repetition_penalty=abc_repetition_penalty,
        abc_penalty_window=abc_penalty_window,
        tiled_decode=tiled_decode,
    )
    yield from yue2_generation.generate_yue2(
        request, _generation_services(), _runtime_config(), progress=progress
    )


def interrupt(request: gr.Request, family: str = "h3") -> str:
    try:
        from .job_admission import require_owner

        return JOBS.cancel(require_owner(request), family, api_get, api_post)
    except Exception as exc:
        return f"Interrupt failed: {exc}"


def preset_values(name: str, generation_mode: str = "Normal"):
    values = asdict(preset_settings(name, generation_mode))
    return (
        *(
            (
                text_encoder_offload_update(values["text_encoder"])
                if key == "stage_model_offload"
                else value
            )
            for key, value in values.items()
        ),
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
    latent_split_fade_ratio: float = 0.5,
    latent_split_chunk_frames: int = 73,
    latent_split_temporal_overlap_frames: int = 22,
    latent_split_seam_denoise: float = 0.75,
    latent_split_seam_polish: str = "off",
    use_trt_vae: bool = False,
    use_lynnreal_vae: bool = False,
) -> str:
    return describe_settings(locals())


def result_settings_for_media(value):

    def media_path(item):
        if isinstance(item, dict):
            return media_path(item.get("path") or item.get("name") or item.get("image"))
        if isinstance(item, (tuple, list)):
            return media_path(item[0]) if item else None
        return item if isinstance(item, (str, Path)) else None

    paths = []
    items = value if isinstance(value, list) else [value]
    for item in items:
        path = media_path(item)
        if not path:
            continue
        if not read_snapshot(path):
            name = Path(path).name
            if re.search("[0-9a-f]{32}", name):
                candidates = [
                    candidate
                    for root in (OUTPUT_DIR, OUTPUTS_DIR)
                    for candidate in root.rglob(name)
                    if candidate.resolve().is_relative_to(root.resolve())
                    and read_snapshot(candidate)
                ]
                if len(candidates) == 1:
                    path = candidates[0]
        paths.append(path)
    return render_snapshot(paths)


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
    return f"Use `<Picture 1>` through `<Picture {MAX_REFERENCE_IMAGES}>`, `<Video 1>` through `<Video {MAX_REFERENCE_VIDEOS}>`, and `<Audio 1>` through `<Audio {MAX_REFERENCE_AUDIOS}>` in the prompt."


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
    prompt: str, request: gr.Request, progress=gr.Progress(track_tqdm=False)
):
    """Generate with UI defaults and return a reusable public download URL."""
    defaults = UI_DEFAULTS
    updates = generate(
        mode=defaults["mode"],
        model_profile=defaults["model_profile"],
        text_encoder=defaults["text_encoder"],
        encoder_small_input=defaults["encoder_small_input"],
        stage_model_offload=defaults["stage_model_offload"],
        use_int8_vae=defaults["use_int8_vae"],
        use_lynnreal_vae=defaults["use_lynnreal_vae"],
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
        latent_upscale_refine_lora=defaults["latent_upscale_refine_lora"],
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
        yield (download_url, status)


def video_batch_seeds(seed: int, batch_count: int) -> list[int]:
    """Resolve distinct seeds while preserving the single-run random-seed behavior."""
    count = int(batch_count)
    if not MIN_VIDEO_BATCH_COUNT <= count <= MAX_VIDEO_BATCH_COUNT:
        raise H3Error(
            f"Video batch count must be between {MIN_VIDEO_BATCH_COUNT} and {MAX_VIDEO_BATCH_COUNT}."
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
    job = CURRENT_JOB.get()
    seeds = (
        list(job.replay_seeds)
        if job is not None and job.replay_seeds
        else (
            video_batch_seeds(int(arguments.values["seed"]), count)
            if result_format == "Video"
            else [int(arguments.values["seed"])]
        )
    )
    indices = list(range(count))
    if job is not None:
        if job.replay_seeds:
            seeds = list(job.replay_seeds)
            indices = list(job.replay_indices)
            count = len(seeds)
        else:
            seeds = [
                random.randrange(0, 2**63 - 1) if seed < 0 else seed for seed in seeds
            ]
            job.variant_seeds.update(enumerate(seeds))
    first_update = True
    for batch_index, batch_seed in enumerate(seeds):
        if job is not None:
            job.variant = indices[batch_index]
            job.check()
        batch_arguments = arguments.with_seed(batch_seed)
        prefix = f"Video {batch_index + 1}/{count} · " if count > 1 else ""
        for result, status in generate(**batch_arguments.values):
            status = prefix + status
            video_updates = [gr.update() for _ in range(MAX_VIDEO_BATCH_COUNT)]
            if first_update:
                first_update = False
                video_updates = [
                    gr.update(
                        value=None, visible=result_format == "Video" and index < count
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
    return f"""## Generate through the API\n\nThe `/generate_video` endpoint accepts a prompt and preserves the existing **Video** defaults from the **MiniMax H3** tab, including default-on reuse of unchanged prompt/media conditioning:\n\n`{defaults['mode']}` · `{defaults['model_profile']}` · `{defaults['generation_mode']} / {defaults['turbo_variant']}` · `{defaults['duration']}s` · `{defaults['width']}×{defaults['height']}` · `{defaults['steps']} steps` · `{defaults['scheduler']}` scheduler · random seed\n\nInstall the client and submit a job:\n\n```bash\npip install gradio_client\n```\n\n```python\nfrom gradio_client import Client\n\nclient = Client("http://127.0.0.1:7860")\ndownload_url, status = client.predict(\n    "A cinematic tracking shot through a rain-soaked neon city",\n    api_name="/generate_video",\n)\nprint(download_url)\nprint(status)\n```\n\n`download_url` is an HTTP URL served by this app, so it can be opened in a browser or downloaded with `curl -L -O` while the app is running.\n\nFor every control exposed by the MiniMax H3 tab, including **Image** and **Audio** result formats, use `/generate_video_advanced` and inspect the app's [OpenAPI schema](/gradio_api/openapi.json) for its current parameter list. Image selections can be persisted through `/save_h3_image_frames`. API requests share the same single-job queue as the UI.\n"""


def compact_backend_status(detail: str) -> str:
    """Render a calm, glanceable status while retaining diagnostics separately."""
    if detail.startswith("Connected"):
        detail = "Connected · ComfyUI"
    return backend_status_html(detail)


def refresh_backend_views() -> tuple[str, str]:
    detail = backend_status()
    return (compact_backend_status(detail), detail)


def generation_preflight(
    mode: str, prompt: str, first_image: Any, last_image: Any, *reference_media: Any
) -> tuple[str, Any]:
    """Keep invalid jobs out of the expensive backend queue."""
    readiness = generation_readiness_state(
        mode, prompt, first_image, last_image, reference_media
    )
    return (readiness.html, gr.update(interactive=readiness.ready))


def build_ui() -> gr.Blocks:
    """Compose the workspace using explicit catalogs and callbacks."""
    from .bootstrap import BootstrapCatalog, BootstrapServices, build_ui as compose_ui
    from h3_app.jobs import JOBS
    from h3_app.workspace_store import default_store

    JOBS.configure(default_store(OUTPUTS_DIR))
    from h3_app.media import history_output_candidates

    JOBS.history_outputs = lambda history, entry: [
        str(path)
        for path in history_output_candidates(
            _runtime_config().output_dir,
            history,
            VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS,
        )
    ]
    from h3_app.generation.finish_video import restore_finishing

    JOBS.finishing_factory = lambda job: restore_finishing(job, _generation_services())

    return compose_ui(
        BootstrapCatalog(
            AI_POSTPROCESS_OPTIONS=AI_POSTPROCESS_OPTIONS,
            AUTO_RESOLUTION_MEGAPIXEL_PRESETS=AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
            AUTO_SOL_TOKEN_THRESHOLD=AUTO_SOL_TOKEN_THRESHOLD,
            COMFY_DIR=COMFY_DIR,
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
            H3_LATENT_UPSCALER_MODEL_CHOICES=H3_LATENT_UPSCALER_MODEL_CHOICES,
            H3_LATENT_UPSCALE_METHODS=H3_LATENT_UPSCALE_METHODS,
            H3_LATENT_UPSCALE_SPLIT=H3_LATENT_UPSCALE_SPLIT,
            H3_TEXT_ENCODER_CHOICES=H3_TEXT_ENCODER_CHOICES,
            IMAGE_VAE_CHOICES=IMAGE_VAE_CHOICES,
            INPUT_IMAGE_FRAME_PRESETS=INPUT_IMAGE_FRAME_PRESETS,
            INPUT_IMAGE_UPSCALE_SLOTS=INPUT_IMAGE_UPSCALE_SLOTS,
            LARGE_RESOLUTIONS=LARGE_RESOLUTIONS,
            LIGHTNING_PROMPT_MODEL=LIGHTNING_PROMPT_MODEL,
            LOCAL_PROMPT_BASE_MODELS=LOCAL_PROMPT_BASE_MODELS,
            LTX25_DEFAULTS=LTX25_DEFAULTS,
            LTX25_MODEL_CHOICES=LTX25_MODEL_CHOICES,
            LTX25_UPSCALE=LTX25_UPSCALE,
            LTX25_WORKFLOWS=LTX25_WORKFLOWS,
            MAX_IMAGE_FRAMES=MAX_IMAGE_FRAMES,
            MAX_VIDEO_BATCH_COUNT=MAX_VIDEO_BATCH_COUNT,
            MIN_IMAGE_FRAMES=MIN_IMAGE_FRAMES,
            MIN_VIDEO_BATCH_COUNT=MIN_VIDEO_BATCH_COUNT,
            MODEL_PROFILE_CHOICES=MODEL_PROFILE_CHOICES,
            MUSIC3_DEFAULTS=MUSIC3_DEFAULTS,
            MUSIC3_MODEL_CHOICES=MUSIC3_MODEL_CHOICES,
            POSTPROCESS_OPTIONS=POSTPROCESS_OPTIONS,
            PROMPT_WRITER_BACKENDS=PROMPT_WRITER_BACKENDS,
            QWEN_IMAGE21_DEFAULTS=QWEN_IMAGE21_DEFAULTS,
            QWEN_IMAGE21_MODEL_CHOICES=QWEN_IMAGE21_MODEL_CHOICES,
            QWEN_IMAGE21_TEXT_ENCODER_CHOICES=QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
            RESULT_FORMATS=RESULT_FORMATS,
            SEEDVR2_MODEL_CHOICES=SEEDVR2_MODEL_CHOICES,
            SEEDVR2_UPSCALE=SEEDVR2_UPSCALE,
            SERVER_ATTENTION_BACKEND=SERVER_ATTENTION_BACKEND,
            SERVER_DENSE_ATTENTION_BACKEND=SERVER_DENSE_ATTENTION_BACKEND,
            SLA_PRESET_INPUTS=SLA_PRESET_INPUTS,
            TURBO_SETTINGS=TURBO_SETTINGS,
            UI_DEFAULTS=UI_DEFAULTS,
            UPSCALE_RESOLUTION_PRESETS=UPSCALE_RESOLUTION_PRESETS,
            YUE2_DEFAULTS=YUE2_DEFAULTS,
            YUE2_MODEL_CHOICES=YUE2_MODEL_CHOICES,
        ),
        BootstrapServices(
            api_get=api_get,
            api_guide=api_guide,
            api_post=api_post,
            auto_resolution_from_start_frame=auto_resolution_from_start_frame,
            backend_status=backend_status,
            compact_backend_status=compact_backend_status,
            compact_settings_summary=compact_settings_summary,
            compile_trt_video_vae=compile_trt_video_vae,
            delete_selected_gallery_media=delete_selected_gallery_media,
            describe_settings=describe_settings,
            empty_generated_media_gallery=empty_generated_media_gallery,
            enhance_h3_prompt=enhance_h3_prompt,
            enhance_ltx25_prompt=enhance_ltx25_prompt,
            enhance_music3_prompt=enhance_music3_prompt,
            enhance_qwen_image21_prompt=enhance_qwen_image21_prompt,
            enhance_yue2_prompt=enhance_yue2_prompt,
            fbcache_preset_defaults=fbcache_preset_defaults,
            generate_for_ui=generate_for_ui,
            generate_ltx25=generate_ltx25,
            generate_music3=generate_music3,
            generate_qwen_image21=generate_qwen_image21,
            generate_with_ui_defaults=generate_with_ui_defaults,
            generate_yue2=generate_yue2,
            generation_readiness_state=generation_readiness_state,
            image_vae_frame_updates=image_vae_frame_updates,
            import_gallery_media=import_gallery_media,
            input_image_frame_preset_updates=input_image_frame_preset_updates,
            interrupt=interrupt,
            latent_upscale_layout_updates=latent_upscale_layout_updates,
            latent_upscale_method_layout_update=latent_upscale_method_layout_update,
            list_media_paths=list_media_paths,
            mode_help=mode_help,
            mode_layout_updates=mode_layout_updates,
            postprocess_selected_gallery_media=postprocess_selected_gallery_media,
            prepare_all_ltx25_official_models=prepare_all_ltx25_official_models,
            prepare_ltx25_official_workflow=prepare_ltx25_official_workflow,
            prompt_writer_backend_visibility=prompt_writer_backend_visibility,
            reference_prompt_help=reference_prompt_help,
            refresh_backend_views=refresh_backend_views,
            refresh_media_gallery=refresh_media_gallery,
            refresh_media_page=refresh_media_page,
            render_ltx25_official_model_inventory=render_ltx25_official_model_inventory,
            render_ltx25_workflow_details=render_ltx25_workflow_details,
            render_snapshot=render_snapshot,
            resolution_choice_updates=resolution_choice_updates,
            resolution_control_updates=resolution_control_updates,
            resolution_info_preview=resolution_info_preview,
            resolution_summary=resolution_summary,
            resolve_request_settings=resolve_request_settings,
            result_format_layout_updates=result_format_layout_updates,
            result_settings_for_media=result_settings_for_media,
            save_selected_image_frames=save_selected_image_frames,
            select_all_image_frames=select_all_image_frames,
            select_gallery_media=select_gallery_media,
            unload_all_models=unload_all_models,
            upscale_selected_input_images=upscale_selected_input_images,
        ),
    )


def selftest() -> None:
    from tests.service_selftest import selftest as run_contracts

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
    allowed_paths = [str(OUTPUT_DIR.resolve()), str(OUTPUTS_DIR.resolve())]
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
            proxy_headers=True,
            forwarded_allow_ips="*",
            **UVICORN_WEBSOCKET_OPTIONS,
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    if share_enabled:
        try:
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
            print(f"[h3-ui] Could not create Gradio share link: {exc}", flush=True)
    try:
        while server_thread.is_alive() and (not shutdown_requested.is_set()):
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

__all__ = [
    "AUTO_SOL_TOKEN_THRESHOLD",
    "Any",
    "CHUNK_FEED_FORWARD_NODE",
    "COMFY_UPSCALE_OPTIONS",
    "CORE_LORA_LOADER_NODE",
    "CORE_SAMPLER_NODE",
    "DEFAULT_AUTO_RESOLUTION_MEGAPIXELS",
    "DEFAULT_GEMINI_PROMPT_MODEL",
    "DEFAULT_LTX25_MODEL",
    "DEFAULT_MUSIC3_MODEL",
    "DEFAULT_SEEDVR2_MODEL",
    "DEFAULT_TURBO",
    "FUSED_MODULATION_NODE",
    "GEMINI_PROMPT_MODELS",
    "GENERATION_FIELDS",
    "GENERATION_POSTPROCESS_OPTIONS",
    "Graph",
    "H3Error",
    "H3SplitUpscaleConfig",
    "H3_COMBINE_AV_LATENT_NODE",
    "H3_CONDITIONING_CACHE_NODE",
    "H3_IMAGE_SLICES_NODE",
    "H3_LATENT_UPSCALER_NODE",
    "H3_LATENT_UPSCALE_SPLIT",
    "H3_LATENT_UPSCALE_STANDARD",
    "H3_NVENC_SAVE_NODE",
    "H3_REFINEMENT_COMPILER_GUARD_NODE",
    "H3_SEMANTIC_BRIDGE_NODE",
    "H3_SEPARATE_AV_LATENT_NODE",
    "H3_SIGMA_SHIFT_NODE",
    "H3_SINGLE_FRAME_VAE_LOADER_NODE",
    "H3_SPLIT_SPATIAL_PARAMS_NODE",
    "H3_SPLIT_TEMPORAL_PARAMS_NODE",
    "H3_SPLIT_UPSCALE_NODE",
    "H3_STAGE_OFFLOAD_NODE",
    "H3_STAGE_OFFLOAD_POLICY_NODE",
    "IMAGE_EXTENSIONS",
    "INPUT_IMAGE_FRAME_PRESETS",
    "INPUT_IMAGE_UPSCALE_SLOTS",
    "JOBS",
    "LARRY_TURBO",
    "LARRY_TURBO_LORA_NODE",
    "LARRY_TURBO_SAMPLER_NODE",
    "LIGHTNING_API_ROOT",
    "LIGHTNING_PROMPT_MODEL",
    "LIGHTX2V_4STEP_TURBO",
    "LIGHTX2V_8STEP_TURBO",
    "LIGHTX2V_BYPASS_LORA_NODE",
    "LTX25_CQ_ENHANCER",
    "LTX25_DEBLUR",
    "LTX25_DECOMPRESSION",
    "LTX25_ICLORA_MODEL_KEYS",
    "LTX25_REFINE_DETAILS",
    "LTX25_RESTORE",
    "LTX25_SDR_TO_HDR",
    "LTX25_SIGMAS",
    "LTX25_UPSCALE",
    "LTX25_WORKFLOWS",
    "LTX25_WORKFLOW_FILENAMES",
    "MODEL_PROFILE_CHOICES",
    "MODEL_SPECS",
    "MUSIC3_DEFAULTS",
    "ModelConfig",
    "ModelProfile",
    "OFFICIAL_IMAGE_VAE",
    "POSTPROCESS_OPTIONS",
    "PROMPT_WRITER_BACKENDS",
    "Path",
    "RESOLUTION_TIERS",
    "Response",
    "SAGE_ATTENTION_NODE",
    "SAMPLING_PRESETS",
    "SAMPLING_PRESET_TEXT_ENCODERS",
    "SEEDVR2_UPSCALE",
    "SINGLE_FRAME_IMAGE_VAE",
    "SLA_ATTENTION_NODE",
    "SLA_PRESET_INPUTS",
    "SOL_ATTENTION_NODE",
    "SWIFTVR_UPSCALE",
    "StageTimings",
    "UI_DEFAULTS",
    "UVICORN_WEBSOCKET_OPTIONS",
    "VIDEO_EXTENSIONS",
    "_append_set_cookies",
    "_comfy_upstream_path",
    "_proxy_headers",
    "_rewrite_comfy_text",
    "active_fl2va_voice_references",
    "auto_resolution_pixel_cap",
    "backend_status_html",
    "build_music3_graph",
    "collect_reference_slots",
    "estimate_packed_tokens",
    "frame_length",
    "gallery_store",
    "generation_readiness_state",
    "gr",
    "graph_class_types",
    "h3_latent_upscale_dimensions",
    "h3_text_encoder_settings",
    "h3_workflow",
    "httpx",
    "image_sampling_length",
    "inspect",
    "json",
    "ltx25_frame_length",
    "ltx25_official_inventory_keys",
    "ltx25_workflow_model_keys",
    "math",
    "model_file_is_ready",
    "model_service",
    "node_stage",
    "normalize_result_format",
    "os",
    "outputs",
    "progress_status",
    "prompt_service",
    "random",
    "replace",
    "required_music3_nodes",
    "resolution_choice_values",
    "resolution_for_aspect_ratio",
    "resolve_cache_policy",
    "resolve_h3_split_upscale_config",
    "resolve_hf_token",
    "resolve_sla_preset",
    "rewrite_local_h3_prompt",
    "run_media_process",
    "selected_image_sampling_length",
    "shutil",
    "single_frame_image_sampling_length",
    "staging",
    "stale_model_keys",
    "swiftvr",
    "sync_models",
    "tempfile",
    "threading",
    "time",
    "turbo_sampler_name",
    "turbo_steps_for",
    "turbo_strength_for",
    "turbo_uses_custom_nodes",
    "unload_prompt_rewriter",
    "upscale_target_dimensions",
    "validate_resolution",
    "write_snapshot",
]
