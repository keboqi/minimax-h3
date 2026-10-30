"""Runtime configuration and stable Gradio API adapters."""

from __future__ import annotations
from h3_app.public_url import launch_cloudflare, public_url_settings, start_gradio_share, stop_cloudflare

from .prompt_controller import PromptController, PromptServices
from .model_controller import ModelController, ModelServices
from .dubbing.handlers import LIBRARY_LIMIT as DUBBING_LIBRARY_LIMIT, DubbingHandlers, DubbingServices
from .drama3.handlers import Drama3Handlers, Drama3Services
from .media_controller import MediaController, MediaServices

from .h3.layout import (
    resolution_control_updates,
    resolution_info_preview,
    auto_resolution_from_start_frame,
    resolution_choice_updates,
    result_format_layout_updates,
    image_vae_frame_updates,
    latent_upscale_layout_updates,
    fbcache_preset_defaults,
    input_image_frame_preset_updates,
    select_all_image_frames,
)
from .h3.handlers import H3Handlers, H3Services, api_guide, mode_help, reference_prompt_help
from h3_app.policy import input_image_upscale_dimensions
import argparse
import math
import os
import signal
import tempfile
import threading
import inspect
import time
import uuid
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
    MUSIC3_MODEL_CHOICES,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
    YUE2_MODEL_CHOICES,
    SEEDVR2_MODEL_CHOICES,
)
from h3_prompt_rewriter import (
    BASE_MODEL_CHOICES as LOCAL_PROMPT_BASE_MODELS,
    DEFAULT_BASE_MODEL_LABEL as DEFAULT_LOCAL_PROMPT_BASE_MODEL,
    unload_prompt_rewriter,
)
from h3_ui.presentation import (
    backend_status_html,
    generation_readiness as generation_readiness_state,
)
from h3_ui.styles import H3_SETUP_CSS
from h3_app import server as server_routes
from h3_app.comfy import ComfyClient
from h3_app.jobs import gpu_maintenance, JOBS, CURRENT_JOB
from h3_app.provenance import RUN_CONTEXT, write_snapshot, render_snapshot
from h3_app.errors import H3Error
from h3_app.catalog import (
    AUDIO_EXTENSIONS,
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
    AUTO_SOL_TOKEN_THRESHOLD,
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
    DEFAULT_GEMINI_PROMPT_MODEL,
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_IMAGE_VAE,
    DEFAULT_INPUT_IMAGE_FRAME_PRESET,
    DEFAULT_PROMPT_WRITER_BACKEND,
    DEFAULT_RESULT_FORMAT,
    DEFAULT_UPSCALE_RESOLUTION,
    DEFAULT_VIDEO_BATCH_COUNT,
    DRAFT_RESOLUTIONS,
    FAST_RESOLUTIONS,
    GEMINI_PROMPT_MODELS,
    GENERATION_POSTPROCESS_OPTIONS,
    H3_REFINEMENT,
    H3_LATENT_UPSCALE_METHODS,
    H3_LATENT_UPSCALE_STANDARD,
    IMAGE_EXTENSIONS,
    IMAGE_VAE_CHOICES,
    INPUT_IMAGE_FRAME_PRESETS,
    INPUT_IMAGE_UPSCALE_SLOTS,
    LARGE_RESOLUTIONS,
    OPENAI_PROMPT_MODEL,
    LTX25_DEFAULTS,
    LTX25_WORKFLOWS,
    MAX_IMAGE_FRAMES,
    MAX_VIDEO_BATCH_COUNT,
    MIN_IMAGE_FRAMES,
    MIN_VIDEO_BATCH_COUNT,
    MODEL_PROFILE_CHOICES,
    MUSIC3_DEFAULTS,
    QWEN_IMAGE21_DEFAULTS,
    YUE2_DEFAULTS,
    POSTPROCESS_OPTIONS,
    PROMPT_WRITER_BACKENDS,
    RESULT_FORMATS,
    SLA_PRESET_INPUTS,
    TURBO_SETTINGS,
    UI_DEFAULTS,
    UPSCALE_RESOLUTION_PRESETS,
    UVICORN_WEBSOCKET_OPTIONS,
    VIDEO_EXTENSIONS,
)
from h3_app.policy import estimate_packed_tokens, resolution_summary, upscale_target_dimensions
from h3_app.model_types import ModelConfig, h3_text_encoder_settings, model_file_is_ready
from h3_app import model_service
import h3_app.workflows.ltx as ltx_workflow
import h3_app.workflows.h3 as h3_workflow
import h3_app.workflows.upscale as upscale_workflow
from h3_app.execution import ExecutionRunner, PromptId, Submission
import h3_app.staging as staging
from h3_app import media_tools
from h3_app import outputs
from h3_app.outputs import OutputContext
from h3_app.config import RuntimeConfig
from dataclasses import replace
from h3_app.media_types import VideoMetadata
from h3_app.generation import (
    requests as generation_requests,
    services as generation_services,
)
from h3_app.generation import (
    h3 as h3_generation,
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
LTX25_WORKFLOW_TEMPLATE_DIR = RUNTIME.workflow_dir
REQUEST_TIMEOUT = RUNTIME.request_timeout
GENERATION_TIMEOUT = RUNTIME.generation_timeout
POLL_SECONDS = RUNTIME.poll_seconds
OUTPUTS_DIR = RUNTIME.outputs_dir
GALLERY_THUMBNAILS_DIR = RUNTIME.gallery_thumbnails_dir
GALLERY_LIMIT = RUNTIME.gallery_limit
GALLERY_METADATA_CACHE_LIMIT = RUNTIME.gallery_metadata_cache_limit
HTTP = requests.Session()


# One controller of each kind. Their records hold late-bound effects, so patching
# this module's ComfyUI, graph and file functions reaches them.
MODELS = ModelController(ModelServices(
    _runtime_config=lambda: _runtime_config(),
    unload_comfy_models=lambda: unload_comfy_models(),
))


MEDIA = MediaController(MediaServices(
    _runtime_config=lambda: _runtime_config(),
    catalog_inventory=lambda mode, limit: _catalog_inventory(mode, limit),
    load_model_config=lambda: MODELS.load_model_config(),
    ensure_ltx25_upscale_models=lambda *args, **kwargs: MODELS.ensure_ltx25_upscale_models(*args, **kwargs),
    ensure_seedvr2_upscale_models=lambda *args, **kwargs: MODELS.ensure_seedvr2_upscale_models(*args, **kwargs),
    ensure_cq_image_enhance_models=lambda: MODELS.ensure_cq_image_enhance_models(),
    ensure_h3_regenerate_models=lambda *args, **kwargs: MODELS.ensure_h3_regenerate_models(*args, **kwargs),
    build_seedvr2_image_upscale_graph=lambda *args, **kwargs: build_seedvr2_image_upscale_graph(*args, **kwargs),
    build_cq_image_enhance_graph=lambda *args, **kwargs: build_cq_image_enhance_graph(*args, **kwargs),
    build_upscale_graph=lambda *args, **kwargs: build_upscale_graph(*args, **kwargs),
    object_info=lambda: object_info(),
    submit_prompt=lambda *args, **kwargs: submit_prompt(*args, **kwargs),
    poll_comfy_progress=lambda *args, **kwargs: poll_comfy_progress(*args, **kwargs),
    stream_comfy_progress=lambda *args, **kwargs: stream_comfy_progress(*args, **kwargs),
    wait_for_history=lambda *args, **kwargs: wait_for_history(*args, **kwargs),
    resolve_output=lambda *args, **kwargs: resolve_output(*args, **kwargs),
    resolve_seedvr2_input_upscale_outputs=lambda *args, **kwargs: resolve_seedvr2_input_upscale_outputs(*args, **kwargs),
    unload_comfy_models=lambda: unload_comfy_models(),
    stage_file=lambda *args, **kwargs: stage_file(*args, **kwargs),
    input_image_upscale_dimensions=lambda *args, **kwargs: input_image_upscale_dimensions(*args, **kwargs),
    upscale_target_dimensions=lambda *args, **kwargs: upscale_target_dimensions(*args, **kwargs),
    postprocess_video=lambda *args, **kwargs: postprocess_video(*args, **kwargs),
    probe_video_metadata=lambda *args, **kwargs: probe_video_metadata(*args, **kwargs),
    write_snapshot=lambda *args, **kwargs: write_snapshot(*args, **kwargs),
))


H3 = H3Handlers(H3Services(
    # Late-bound so tests that replace this module's generator reach the callbacks.
    runtime_config=lambda: _runtime_config(),
    generate=lambda *args, **kwargs: generate(*args, **kwargs),
    download_url=lambda video, request: MEDIA.absolute_video_download_url(video, request),
))


PROMPTS = PromptController(PromptServices(_runtime_config=lambda: _runtime_config()))


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
            GALLERY_THUMBNAILS_DIR,
        ),
    )


def ensure_lightvae(video_encoder: str, decoder: str) -> bool:
    return model_service.ensure_lightvae(video_encoder, decoder, runtime=_runtime_config())


def api_get(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).get(path, **kwargs)


def api_post(path: str, **kwargs: Any) -> requests.Response:
    return ComfyClient(COMFY_URL, REQUEST_TIMEOUT, HTTP).post(path, **kwargs)


def object_info() -> dict[str, Any]:
    return api_get("/object_info").json()


def resolve_sol_policy(attention_mode, mode, width, height, duration, first_image, last_image, use_turbo=False):
    from h3_app.policy import resolve_sol_policy as shared_sol_policy
    return shared_sol_policy(attention_mode, mode, width, height, duration, first_image, last_image,
                             use_turbo, attention_backend=SERVER_ATTENTION_BACKEND)


def stage_file(
    path: str, category: str, transcode_video: bool = False, reuse: bool = False
) -> str:
    return staging.stage_file(
        path, category, transcode_video, reuse, runtime=_runtime_config()
    )


def _submission_signature(builder):
    """``builder``'s signature without the output naming each submission supplies."""
    signature = inspect.signature(builder)
    return signature.replace(parameters=[
        parameter for parameter in signature.parameters.values()
        if parameter.name not in {"output_stamp", "output_nonce"}
    ])


def _output_naming() -> dict[str, str]:
    return {"output_stamp": str(int(time.time())), "output_nonce": uuid.uuid4().hex[:8]}


def build_fl2va_graph(*, first_image, last_image, voice_reference_audios=None,
                      reuse_unchanged_inputs: bool = True, **settings) -> dict[str, Any]:
    """Stage the frames and voice samples, then build with a fresh output name."""
    def staged(path, category):
        return stage_file(path, category, reuse=reuse_unchanged_inputs) if path else None

    return h3_workflow.build_fl2va_graph(
        first_image=staged(first_image, "keyframes"),
        last_image=staged(last_image, "keyframes"),
        voice_reference_audios=[staged(path, "fl2va_voice_audios") for path in voice_reference_audios or []],
        reuse_unchanged_inputs=reuse_unchanged_inputs,
        **settings,
        **_output_naming(),
    )


build_fl2va_graph.__signature__ = _submission_signature(h3_workflow.build_fl2va_graph)


def build_ref2va_graph(*, reference_images, reference_videos, reference_audios,
                       reuse_unchanged_inputs: bool = True, **settings) -> dict[str, Any]:
    """Stage the reference media, then build with a fresh output name."""
    return h3_workflow.build_ref2va_graph(
        reference_images=[stage_file(path, "reference_images", reuse=reuse_unchanged_inputs)
                          for path in reference_images],
        reference_videos=[stage_file(path, "reference_videos", transcode_video=True,
                                     reuse=reuse_unchanged_inputs) for path in reference_videos],
        reference_audios=[stage_file(path, "reference_audios", reuse=reuse_unchanged_inputs)
                          for path in reference_audios],
        reuse_unchanged_inputs=reuse_unchanged_inputs,
        **settings,
        **_output_naming(),
    )


build_ref2va_graph.__signature__ = _submission_signature(h3_workflow.build_ref2va_graph)


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


required_seedvr2_image_upscale_nodes = (
    upscale_workflow.required_seedvr2_image_upscale_nodes
)
required_cq_image_enhance_nodes = upscale_workflow.required_cq_image_enhance_nodes
build_cq_image_enhance_graph = upscale_workflow.build_cq_image_enhance_graph


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
    refinement=None,
) -> tuple[dict[str, Any], int]:
    if option == H3_REFINEMENT:
        refinement = {**(refinement or {}), "_attention_backend": SERVER_ATTENTION_BACKEND}
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
        refinement=refinement,
        output_stamp=str(int(time.time())),
        output_nonce=uuid.uuid4().hex[:8],
    )


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


def stream_comfy_progress(ws, prompt_id, graph, started):
    submission = _submission(prompt_id, graph)
    if submission.socket is None:
        submission.socket = ws
    yield from submission.progress()


def poll_comfy_progress(prompt_id, graph):
    yield from _submission(prompt_id, graph).progress()


def wait_for_history(prompt_id):
    return _submission(prompt_id).history()


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


def postprocess_video(source: Path, option: str) -> Path:
    return media_tools.postprocess_video(source, option, runtime=_runtime_config())


def probe_video_metadata(source: Path) -> VideoMetadata:
    return media_tools.probe_video_metadata(source)


def unload_comfy_models() -> None:
    """Explicitly clear model residency only when the user opts into it."""
    api_post("/free", json={"unload_models": True, "free_memory": True})


def free_comfy_vram(timeout: float = 60.0) -> None:
    """Unload ComfyUI models before another process takes the GPU, and wait for it.

    /free only queues flags. Unchanged global free VRAM is not acknowledgement:
    wait for ComfyUI's own reported Torch allocation to clear instead.
    """
    unload_prompt_rewriter()
    unload_comfy_models()

    deadline, cleared = time.monotonic() + timeout, 0
    detail = "GPU allocation statistics unavailable"
    while time.monotonic() < deadline:
        job = CURRENT_JOB.get()
        if job is not None:
            job.check()
        try:
            devices = api_get("/system_stats", timeout=min(5.0, max(0.1, deadline - time.monotonic()))).json().get("devices")
            if not devices:
                raise ValueError("no device statistics returned")
            gpu_devices = [device for device in devices if device.get("type") != "cpu"]
            reserved = [int(device["torch_vram_total"]) for device in gpu_devices]
            if any(value < 0 for value in reserved):
                raise ValueError("invalid GPU allocation statistics")
            detail = f"{sum(reserved) / 2**30:.2f} GiB still reserved by ComfyUI's Torch allocator"
            cleared = cleared + 1 if all(value <= 256 * 2**20 for value in reserved) else 0
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
            cleared = 0
            detail = f"GPU allocation statistics unavailable: {exc}"
        if cleared >= 2:
            return
        time.sleep(1.0)
    raise H3Error(f"ComfyUI VRAM release did not complete within {timeout:g}s: {detail}. "
                  "SGLang startup was not allowed to continue.")


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


GALLERY_PAGE_SIZE = 48


def _catalog_inventory(mode, limit):
    from h3_app.media_catalog import catalog_inventory

    return catalog_inventory(mode, limit)


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
        models = MODELS.load_model_config()
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
            ("PDMD / 4-step", models.pdmd_4step_lora),
            ("DMAD / 4-step · shifts 12/2", models.dmad_4step_lora),
        ):
            if filename:
                profile_lines.append(
                    f"**{label} (experimental)** | LoRA `{filename}` | FL2VA / Ref2VA | Euler/simple | strength 1.0 | downloads on first use"
                )
        if models.larry_turbo_lora:
            profile_lines.append(
                f"**Larry Turbo v4-600 EMA** | LoRA `{models.larry_turbo_lora}` | 6-step default | strength 1.0 | custom loader/sampler"
            )
        if models.turbo_lora:
            profile_lines.append(
                f"**LightX2V Turbo / 4-step 768p** · FL2VA v1.3 `{models.turbo_lora}` · Ref2VA v1.0 `{models.turbo_ref_lora}` · strength 1.0"
            )
        if models.turbo_8step_lora:
            profile_lines.append(
                f"**LightX2V Turbo v1.1 / 8-step 768p** · FL2VA `{models.turbo_8step_lora}` · Ref2VA `{models.turbo_8step_ref_lora}` · 8-step default · strength 1.0 · FL2VA and Ref2VA"
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
            postprocess_video=postprocess_video,
            probe_video_metadata=probe_video_metadata,
            resolve_audio_output=resolve_audio_output,
            resolve_image_outputs=resolve_image_outputs,
            resolve_output=resolve_output,
            stage_file=stage_file,
        ),
        models=generation_services.ModelsServices(
            ensure_audio_vae=MODELS.ensure_audio_vae,
            ensure_base_video_vae=MODELS.ensure_base_video_vae,
            ensure_h3_latent_upscaler_model=MODELS.ensure_h3_latent_upscaler_model,
            ensure_h3_semantic_bridge=MODELS.ensure_h3_semantic_bridge,
            ensure_h3_veda_predictor=MODELS.ensure_h3_veda_predictor,
            ensure_h3_text_encoder=MODELS.ensure_h3_text_encoder,
            ensure_int8_video_vae=MODELS.ensure_int8_video_vae,
            ensure_lightvae=ensure_lightvae,
            ensure_ltx25_models=MODELS.ensure_ltx25_models,
            ensure_ltx25_ingredients_model=MODELS.ensure_ltx25_ingredients_model,
            ensure_ltx25_upscale_models=MODELS.ensure_ltx25_upscale_models,
            ensure_h3_regenerate_models=MODELS.ensure_h3_regenerate_models,
            ensure_music3_models=MODELS.ensure_music3_models,
            ensure_qwen_image21_models=MODELS.ensure_qwen_image21_models,
            ensure_yue2_models=MODELS.ensure_yue2_models,
            ensure_model_keys=lambda keys: MODELS.ensure_model_keys(keys),
            ensure_profile_model=MODELS.ensure_profile_model,
            ensure_seedvr2_upscale_models=MODELS.ensure_seedvr2_upscale_models,
            ensure_single_frame_image_vae=MODELS.ensure_single_frame_image_vae,
            ensure_trt_video_vae_engine=MODELS.ensure_trt_video_vae_engine,
            ensure_turbo_lora=MODELS.ensure_turbo_lora,
            load_model_config=MODELS.load_model_config,
            missing_ltx25_model_names=MODELS.missing_ltx25_model_names,
            missing_music3_model_names=MODELS.missing_music3_model_names,
            missing_qwen_image21_model_names=MODELS.missing_qwen_image21_model_names,
            missing_yue2_model_names=MODELS.missing_yue2_model_names,
            trt_vae_decoder_paths=MODELS.trt_vae_decoder_paths,
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
            resolve_request_settings=H3.resolve_request_settings,
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
    upscale_force_offload: bool = False,
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
    video_encoder: str = "Original",
    lightvae_decoder: str = "none",
    use_sol_vae: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    values = dict(locals())
    del values["progress"]
    request = generation_requests.H3Request.from_values(values)
    yield from h3_generation.generate(
        request,
        _generation_services(),
        _runtime_config(),
        run_context=RUN_CONTEXT.get(),
        progress=progress,
    )


drama3_handlers = Drama3Handlers(Drama3Services(
    # Late-bound so the application's runtime hooks stay patchable.
    runtime_config=lambda: _runtime_config(),
    stage_file=lambda *args: stage_file(*args),
    generation_services=lambda: _generation_services(),
    free_gpu=lambda: free_comfy_vram(),
    resolve_result=lambda path: H3.resolve_result_path(path),
))


def _dubbing_media(mode):
    """Newest managed media of one kind for Dub & Translate's source picker."""
    inventory = _catalog_inventory(mode, DUBBING_LIBRARY_LIMIT)
    return list(inventory.paths) if inventory is not None else MEDIA.list_media_paths(mode)


dubbing_handlers = DubbingHandlers(DubbingServices(
    # Late-bound so the application's runtime hooks stay patchable.
    runtime_config=lambda: _runtime_config(),
    free_gpu=lambda: free_comfy_vram(),
    list_media=lambda mode: _dubbing_media(mode),
))


def _engine_deps():
    from h3_ui.engines import EngineDeps

    # Late-bound so the application's runtime hooks stay patchable.
    return EngineDeps(
        runtime_config=lambda: _runtime_config(),
        generation_services=lambda: _generation_services(),
        stage_file=lambda *args, **kwargs: stage_file(*args, **kwargs),
        interrupt=interrupt,
        models=lambda: MODELS,
        resolve_result=lambda path: H3.resolve_result_path(path),
    )


def _drama3_actions():
    from h3_ui.drama3.speaker_library import SpeakerLibraryActions
    from h3_ui.drama3.bindings import Drama3Actions
    return Drama3Actions.of(
        drama3_handlers, interrupt=interrupt, speaker_library_actions=SpeakerLibraryActions(_runtime_config),
    )


def interrupt(request: gr.Request, family: str = "h3") -> str:
    try:
        from .job_admission import require_owner

        return JOBS.cancel(require_owner(request), family, api_get, api_post)
    except Exception as exc:
        return f"Interrupt failed: {exc}"


def compact_backend_status(detail: str) -> str:
    """Render a calm, glanceable status while retaining diagnostics separately."""
    if detail.startswith("Connected"):
        detail = "Connected · ComfyUI"
    return backend_status_html(detail)


def refresh_backend_views() -> tuple[str, str]:
    detail = backend_status()
    return (compact_backend_status(detail), detail)


def build_ui() -> gr.Blocks:
    """Compose the workspace using explicit catalogs and callbacks."""
    from .bootstrap import build_ui as compose_ui
    from .contracts import BootstrapCatalog, BootstrapServices
    from h3_app.jobs import JOBS
    from h3_app.workspace_store import default_store

    JOBS.configure(default_store(OUTPUTS_DIR))
    from h3_app.media_catalog import configure_media_catalog

    configure_media_catalog(JOBS.store, _runtime_config)
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
            H3_TEXT_ENCODER_CHOICES=H3_TEXT_ENCODER_CHOICES,
            IMAGE_VAE_CHOICES=IMAGE_VAE_CHOICES,
            INPUT_IMAGE_FRAME_PRESETS=INPUT_IMAGE_FRAME_PRESETS,
            INPUT_IMAGE_UPSCALE_SLOTS=INPUT_IMAGE_UPSCALE_SLOTS,
            LARGE_RESOLUTIONS=LARGE_RESOLUTIONS,
            OPENAI_PROMPT_MODEL=OPENAI_PROMPT_MODEL,
            LOCAL_PROMPT_BASE_MODELS=LOCAL_PROMPT_BASE_MODELS,
            LTX25_DEFAULTS=LTX25_DEFAULTS,
            LTX25_MODEL_CHOICES=LTX25_MODEL_CHOICES,
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
            compact_settings_summary=H3.compact_settings_summary,
            compile_trt_video_vae=MODELS.compile_trt_video_vae,
            delete_selected_gallery_media=MEDIA.delete_selected_gallery_media,
            describe_settings=H3.describe_settings,
            empty_generated_media_gallery=MEDIA.empty_generated_media_gallery,
            enhance_h3_prompt=PROMPTS.enhance_h3_prompt,
            fbcache_preset_defaults=fbcache_preset_defaults,
            generate_for_ui=H3.generate_for_ui,
            generate_with_ui_defaults=H3.generate_with_ui_defaults,
            drama3_actions=_drama3_actions(),
            dubbing_actions=dubbing_handlers,
            engine_deps=_engine_deps(),
            generation_readiness_state=generation_readiness_state,
            image_vae_frame_updates=image_vae_frame_updates,
            import_gallery_media=MEDIA.import_gallery_media,
            input_image_frame_preset_updates=input_image_frame_preset_updates,
            interrupt=interrupt,
            latent_upscale_layout_updates=latent_upscale_layout_updates,
            list_media_paths=MEDIA.list_media_paths,
            gallery_media_download_path=MEDIA.gallery_media_download_path,
            mode_help=mode_help,
            postprocess_selected_gallery_media=MEDIA.postprocess_selected_gallery_media,
            prompt_writer_backend_visibility=PROMPTS.prompt_writer_backend_visibility,
            reference_prompt_help=reference_prompt_help,
            refresh_backend_views=refresh_backend_views,
            refresh_media_page=MEDIA.refresh_media_page,
            render_snapshot=render_snapshot,
            resolution_choice_updates=resolution_choice_updates,
            resolution_control_updates=resolution_control_updates,
            resolution_info_preview=resolution_info_preview,
            resolution_summary=resolution_summary,
            resolve_request_settings=H3.resolve_request_settings,
            result_format_layout_updates=result_format_layout_updates,
            result_settings_for_media=H3.result_settings_for_media,
            save_selected_image_frames=MEDIA.save_selected_image_frames,
            select_all_image_frames=select_all_image_frames,
            select_gallery_media=MEDIA.select_gallery_media,
            unload_all_models=unload_all_models,
            upscale_selected_input_images=MEDIA.upscale_selected_input_images,
        ),
    )


def selftest() -> None:
    from tests.__main__ import main as run_tests

    result = run_tests([])
    if result:
        raise SystemExit(result)


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
    from h3_ui.snapshot_preload import run_before_serving

    # Modal's preloaded endpoints load models here, before the workspace opens.
    run_before_serving()
    demo = build_ui().queue(default_concurrency_limit=1, max_size=8)
    app = build_server(demo, allowed_paths)
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    cloudflare_enabled, share_enabled = public_url_settings()
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
    cloudflare_process = None
    server_thread.start()
    try:
        while server_thread.is_alive() and not server.started:
            if shutdown_requested.wait(timeout=0.1):
                break
        if (
            server_thread.is_alive()
            and server.started
            and not shutdown_requested.is_set()
        ):
            if cloudflare_enabled:
                cloudflare_process = launch_cloudflare(
                    port, cache_dir=SCRIPT_DIR / ".cache" / "cloudflared"
                )
            if share_enabled:
                share_url = start_gradio_share(
                    lambda: gradio_networking.setup_tunnel(
                        local_host="127.0.0.1",
                        local_port=port,
                        share_token=demo.share_token,
                        share_server_address=getattr(
                            demo, "share_server_address", None
                        ),
                        share_server_tls_certificate=getattr(
                            demo, "share_server_tls_certificate", None
                        ),
                    ),
                    stop=shutdown_requested,
                )
                # The workspace carries queue updates over WebSockets, so a
                # Cloudflare quick tunnel serves it fully when Share is down.
                if (
                    share_url is None
                    and cloudflare_process is None
                    and not shutdown_requested.is_set()
                ):
                    print(
                        "[h3-ui] Gradio Share is unavailable; opening a Cloudflare URL instead",
                        flush=True,
                    )
                    cloudflare_process = launch_cloudflare(
                        port, cache_dir=SCRIPT_DIR / ".cache" / "cloudflared"
                    )
        while server_thread.is_alive() and (not shutdown_requested.is_set()):
            server_thread.join(timeout=0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.should_exit = True
        server.force_exit = True
        try:
            stop_cloudflare(cloudflare_process)
        finally:
            server_thread.join(timeout=3)
            try:
                demo.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
