#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
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
    LTX25_MODEL_CHOICES,
    LTX25_ICLORA_MODEL_KEYS,
    LTX25_SHARED_MODEL_KEYS,
    MIN_VALID_MODEL_BYTES,
    MODEL_SPECS,
    MUSIC3_MODEL_CHOICES,
    MUSIC3_SHARED_MODEL_KEYS,
    PROFILE_LABELS,
    SEEDVR2_MODEL_CHOICES,
    resolve_hf_token,
    stale_model_keys,
    sync_models,
)
from h3_prompt_rewriter import (
    BASE_MODEL_CHOICES as LOCAL_PROMPT_BASE_MODELS,
    DEFAULT_BASE_MODEL_LABEL as DEFAULT_LOCAL_PROMPT_BASE_MODEL,
    resolution_for_size as local_prompt_resolution,
    rewrite_prompt as rewrite_local_h3_prompt,
    task_for_inputs as local_prompt_task,
    unload_prompt_rewriter,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _detect_comfy_dir() -> Path:
    configured = os.getenv("COMFY_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        SCRIPT_DIR / "h3" / "ComfyUI",
        SCRIPT_DIR / "ComfyUI",
        Path.cwd() / "h3" / "ComfyUI",
        Path.cwd() / "ComfyUI",
    ]
    for candidate in candidates:
        if (candidate / "main.py").is_file():
            return candidate.resolve()

    return (SCRIPT_DIR / "h3" / "ComfyUI").resolve()


COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_PROXY_PATH = "/comfyui"
# Modal supports RFC 6455 WebSockets but deliberately does not support the
# RFC 7692 permessage-deflate extension. Uvicorn enables that extension by
# default, which makes Modal close the connection as soon as ComfyUI sends its
# initial status frame. Keep the transport deterministic on Modal and local
# standalone servers by using the dependency we install and disabling RFC 7692.
UVICORN_WEBSOCKET_OPTIONS = {
    "ws": "wsproto",
    "ws_per_message_deflate": False,
}
COMFY_DIR = _detect_comfy_dir()
INPUT_DIR = COMFY_DIR / "input"
OUTPUT_DIR = COMFY_DIR / "output"
MODELS_CONFIG = Path(
    os.environ.get("MODELS_CONFIG", str(COMFY_DIR.parent / "h3_models.json"))
).expanduser().resolve()
SERVER_ATTENTION_BACKEND = os.getenv("SERVER_ATTENTION_BACKEND", "sol").lower()
SERVER_DENSE_ATTENTION_BACKEND = os.getenv(
    "SERVER_DENSE_ATTENTION_BACKEND", "comfy-kitchen"
).lower()
SERVER_MEMORY_PROFILE = os.getenv("SERVER_MEMORY_PROFILE", "unknown").lower()
AUTO_SOL_TOKEN_THRESHOLD = 8_192
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
GEMINI_PROMPT_MODELS = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)
DEFAULT_GEMINI_PROMPT_MODEL = GEMINI_PROMPT_MODELS[0]
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com"
PROMPT_WRITER_BACKENDS = ("Local MiniMax-H3 8B", "Gemini")
DEFAULT_PROMPT_WRITER_BACKEND = PROMPT_WRITER_BACKENDS[0]
PROMPT_ENHANCER_SYSTEM_PATH = SCRIPT_DIR / "prompt.txt"
PROMPT_ENHANCER_SYSTEMS = {
    "MiniMax H3": PROMPT_ENHANCER_SYSTEM_PATH,
    "MiniMax Music 3": SCRIPT_DIR / "prompt_music3.txt",
    "LTX-2.5": SCRIPT_DIR / "prompt_ltx25.txt",
}
DEFAULT_FBCACHE_PRESET = "Fast"
DEFAULT_FBCACHE_THRESHOLD = 0.10
DEFAULT_FBCACHE_START = 0.10
DEFAULT_FBCACHE_END = 0.95
DEFAULT_FBCACHE_MAX_HITS = 2
DEFAULT_FBCACHE_TEMPORAL_GUARD = True
DEFAULT_ACCELERATOR = "Spectrum"
LIGHTX2V_4STEP_TURBO = "LightX2V / 4-step (FL2V 768p · Ref2V 544p)"
LIGHTX2V_8STEP_TURBO = "LightX2V v1.0 / 8-step 544p"
LARRY_TURBO = "Larry v4-600 EMA"
DEFAULT_TURBO = LIGHTX2V_4STEP_TURBO
RESULT_FORMATS = ("Video", "Image", "Audio")
DEFAULT_RESULT_FORMAT = RESULT_FORMATS[0]
DEFAULT_IMAGE_FRAMES = 5
MIN_IMAGE_FRAMES = 1
MAX_IMAGE_FRAMES = 20
LTX25_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX25_DEFAULTS = {
    "model": DEFAULT_LTX25_MODEL,
    "mode": "Text to video",
    "duration": 5,
    "fps": 24,
    "width": 960,
    "height": 544,
    "seed": -1,
    "cfg": 1.0,
    "sampler": "euler_ancestral",
    "image_strength": 0.7,
    "middle_time": 2.5,
    "middle_strength": 0.7,
    "end_strength": 0.7,
}
MUSIC3_DEFAULTS = {
    "model": DEFAULT_MUSIC3_MODEL,
    "duration": 120,
    "seed": -1,
    "steps": 30,
    "cfg": 1.7,
    "ar_cfg": 1.7,
    "top_k": 50,
    "tiled_decode": True,
}
LTX25_WORKFLOW_TEMPLATE_DIR = (
    COMFY_DIR / "user" / "default" / "workflows" / "LTX 2.5"
)
LTX25_WORKFLOWS = {
    "Text / image to video — single stage": {
        "id": "t2v-i2v-single-stage",
        "filename": "LTX-2.5_T2V_I2V_Single_Stage_Distilled.json",
        "description": "The official 8-step text-to-video and start-image workflow.",
        "inputs": "Prompt and optional start image.",
        "extra_models": (),
    },
    "Text / image to video — two stage": {
        "id": "t2v-i2v-two-stage",
        "filename": "LTX-2.5_T2V_I2V_Two_Stage_Distilled.json",
        "description": "Generates low resolution, then performs a 2x latent refinement pass.",
        "inputs": "Prompt and optional start image.",
        "extra_models": ("ltx25_spatial_upscaler",),
    },
    "Text to audio": {
        "id": "text-to-audio",
        "filename": "LTX-2.5_T2A_Single_Stage_Distilled.json",
        "description": "Generates standalone audio from a text description.",
        "inputs": "Prompt, duration, and frame rate; no image or video.",
        "extra_models": (),
        "audio_only": True,
    },
    "Ingredients / reference sheet — IC-LoRA": {
        "id": "iclora-ingredients",
        "filename": "LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json",
        "description": "Uses a reference sheet of characters, props, wardrobe, or locations.",
        "inputs": "Prompt and one reference-sheet image.",
        "extra_models": ("ltx25_iclora_ingredients",),
    },
    "Video to video / instant shave — IC-LoRA": {
        "id": "iclora-video-to-video",
        "filename": "LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled.json",
        "description": "Applies the official instant-shave edit while preserving timing and audio.",
        "inputs": "Source video, prompt, and optional start image.",
        "extra_models": ("ltx25_iclora_instant_shave",),
    },
    "Motion tracking — IC-LoRA": {
        "id": "iclora-motion-track",
        "filename": "LTX-2.5_ICLoRA_Motion_Track_Distilled.json",
        "description": "Draws sparse point tracks that control motion through the clip.",
        "inputs": "Prompt, start image, and tracks drawn in ComfyUI.",
        "extra_models": ("ltx25_iclora_motion_track",),
    },
    "Video inpaint — two stage IC-LoRA": {
        "id": "iclora-inpaint",
        "filename": "LTX-2.5_ICLoRA_Inpaint_Two_Stage_Distilled.json",
        "description": "Regenerates masked areas and blends them into a source video.",
        "inputs": "Source video, matching black/white mask video, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_in_outpaint", "ltx25_spatial_upscaler",
        ),
    },
    "Video outpaint — two stage IC-LoRA": {
        "id": "iclora-outpaint",
        "filename": "LTX-2.5_ICLoRA_Outpaint_Two_Stage_Distilled.json",
        "description": "Extends a source video beyond its original frame and refines at 2x.",
        "inputs": "Source video, target canvas/padding, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_in_outpaint", "ltx25_spatial_upscaler",
        ),
    },
    "Pose / depth / canny control — IC-LoRA": {
        "id": "iclora-union-control",
        "filename": "LTX-2.5_ICLoRA_Union_Control_Distilled.json",
        "description": "Controls generation from pose, depth, or canny guidance extracted from video.",
        "inputs": "Reference video, control type, prompt, and optional start image.",
        "extra_models": (
            "ltx25_iclora_union_control", "ltx25_spatial_upscaler",
        ),
    },
}
LTX25_WORKFLOW_COMMON_MODEL_KEYS = (
    "ltx25_distilled",
    "ltx25_text_encoder",
    "ltx25_video_vae",
    "ltx25_video_vae_full",
    "ltx25_audio_vae",
    "ltx25_text_enhancer",
)
MODEL_PROFILE_CHOICES = list(PROFILE_LABELS.values())
CORE_LORA_LOADER_NODE = "LoraLoaderModelOnly"
CORE_SAMPLER_NODE = "KSamplerSelect"
LARRY_TURBO_LORA_NODE = "MiniMaxH3TurboLoRA"
LARRY_TURBO_SAMPLER_NODE = "MiniMaxH3TurboSampler"
LIGHTX2V_BYPASS_LORA_NODE = "H3LightX2VBypassLoRA"
H3_LATENT_UPSCALER_NODE = "MinimaxH3LatentUpscalerNode3D"
H3_SEPARATE_AV_LATENT_NODE = "H3SeparateAVLatent"
H3_COMBINE_AV_LATENT_NODE = "H3CombineAVLatent"
H3_LATENT_UPSCALE_SCALE = 2.0
SOL_ATTENTION_NODE = "MiniMaxH3MemoryEfficientSolAttentionPatch"
SAGE_ATTENTION_NODE = "PathchSageAttentionKJ"
FUSED_MODULATION_NODE = "MiniMaxH3FusedModulation"
CHUNK_FEED_FORWARD_NODE = "MiniMaxH3ChunkFeedForward"
SEEDVR2_UPSCALE = "SeedVR2 2x"
LTX25_UPSCALE = "LTX-2.5 IC-LoRA 2x"
COMFY_UPSCALE_OPTIONS = {SEEDVR2_UPSCALE, LTX25_UPSCALE}
POSTPROCESS_OPTIONS = [
    SEEDVR2_UPSCALE,
    LTX25_UPSCALE,
    "48 fps interpolation",
]
GENERATION_POSTPROCESS_OPTIONS = ["None", SEEDVR2_UPSCALE, LTX25_UPSCALE]


@dataclass(frozen=True)
class TurboSpec:
    steps: int
    strength: float
    lora_attr: str
    ref_lora_attr: str
    custom_nodes: bool = False


TURBO_SETTINGS = {
    LARRY_TURBO: TurboSpec(
        steps=6,
        strength=1.0,
        lora_attr="larry_turbo_lora",
        ref_lora_attr="larry_turbo_ref_lora",
        custom_nodes=True,
    ),
    LIGHTX2V_4STEP_TURBO: TurboSpec(
        steps=4,
        strength=1.0,
        lora_attr="turbo_lora",
        ref_lora_attr="turbo_ref_lora",
    ),
    LIGHTX2V_8STEP_TURBO: TurboSpec(
        steps=8,
        strength=1.0,
        lora_attr="turbo_8step_lora",
        ref_lora_attr="turbo_8step_ref_lora",
    ),
}
UI_DEFAULTS = {
    "mode": "Text to video",
    "result_format": DEFAULT_RESULT_FORMAT,
    "image_frames": DEFAULT_IMAGE_FRAMES,
    "model_profile": "Quality",
    "use_int8_vae": False,
    "generation_mode": "Turbo",
    "turbo_variant": DEFAULT_TURBO,
    "duration": 5,
    "width": 864,
    "height": 480,
    "steps": 4,
    "scheduler": "simple",
    "seed": -1,
    "attention_mode": "Sage 2",
    "sol_tau": 1.0,
    "sol_thresh_type": "diag",
    "sol_exact_mode": "exact_kv",
    "sol_dense_steps": 1,
    "cache_mode": DEFAULT_ACCELERATOR,
    "fbcache_preset": DEFAULT_FBCACHE_PRESET,
    "fbcache_threshold": DEFAULT_FBCACHE_THRESHOLD,
    "fbcache_start": DEFAULT_FBCACHE_START,
    "fbcache_end": DEFAULT_FBCACHE_END,
    "fbcache_max_hits": DEFAULT_FBCACHE_MAX_HITS,
    "fbcache_temporal_guard": DEFAULT_FBCACHE_TEMPORAL_GUARD,
    "easycache_threshold": 0.10,
    "easycache_start": 0.15,
    "easycache_end": 0.85,
    "easycache_verbose": False,
    "ref_image_size": "match",
    "latent_upscale": False,
    "latent_upscaler_model": DEFAULT_H3_LATENT_UPSCALER_MODEL,
    "latent_upscale_refine_steps": 2,
    "postprocess": "None",
    "seedvr2_model": DEFAULT_SEEDVR2_MODEL,
    "upscale_force_offload": False,
    "upscale_split_enabled": False,
    "upscale_split_seconds": 5.0,
}
SPECTRUM_DEFAULT_INPUTS = {
    "enabled": True,
    "blend_weight": 0.50,
    "degree": 1,
    "ridge_lambda": 0.10,
    "window_size": 2.0,
    "flex_window": 0.75,
    "warmup_steps": 1,
    "tail_actual_steps": 1,
    "max_history": 8,
    "debug": False,
    "history_storage": "system_ram",
    "bootstrap_first_forecast": True,
    "anchor_residual_feedback": False,
    "selective_rollback_correction": False,
    "offline_smoothing_replay": True,
    "audio_blend_weight": 0.0,
    # Spectrum separates the unbounded replay archive from the capped
    # causal history. Keep all replay anchors in host RAM unless a workflow
    # explicitly opts into the higher-VRAM path.
    "offline_archive_storage": "system_ram",
    # v0.2.7 keeps legacy behavior when model-aware forecasting is off. Make
    # that compatibility policy explicit instead of relying on the node default.
    "model_aware_mode": "off",
    "model_aware_risk_threshold": 0.65,
}

# The official H3 workflow uses a 768×1344 pixel-area native canvas. Larger
# entries are still available for workflows that can handle them.
NATIVE_PIXEL_CAP = 768 * 1344
AUTO_RESOLUTION_PIXEL_CAP = 2_000_000 - 1
DRAFT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 512×512": (512, 512),
    "16:9 · 608×352": (608, 352),
    "9:16 · 352×608": (352, 608),
    "4:3 · 640×480": (640, 480),
    "3:4 · 480×640": (480, 640),
    "4:5 · 512×640": (512, 640),
    "5:4 · 640×512": (640, 512),
}

FAST_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 768×768": (768, 768),
    "16:9 · 864×480": (864, 480),
    "9:16 · 480×864": (480, 864),
    "4:3 · 1024×768": (1024, 768),
    "3:4 · 768×1024": (768, 1024),
    "4:5 · 768×960": (768, 960),
    "5:4 · 960×768": (960, 768),
    "21:9 · 1344×576": (1344, 576),
    "9:21 · 576×1344": (576, 1344),
}

LARGE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1 · 1024×1024": (1024, 1024),
    "16:9 · 1344×768": (1344, 768),
    "9:16 · 768×1344": (768, 1344),
    "4:3 · 1536×1152": (1536, 1152),
    "3:4 · 1152×1536": (1152, 1536),
    "4:5 · 1024×1280": (1024, 1280),
    "5:4 · 1280×1024": (1280, 1024),
    "21:9 · 1536×672": (1536, 672),
    "9:21 · 672×1536": (672, 1536),
}

RESOLUTION_TIERS: dict[str, dict[str, tuple[int, int]]] = {
    "draft": DRAFT_RESOLUTIONS,
    "fast": FAST_RESOLUTIONS,
    "large": LARGE_RESOLUTIONS,
}

SAMPLING_PRESETS: dict[str, tuple[int, float, str, str, str, int]] = {
    "Quality": (20, 0.8, "exact", "beta", "exact_kv_and_rows", 1),
    "Balanced": (18, 1.0, "diag", "simple", "exact_kv", 1),
    "Fast": (15, 1.2, "diag", "simple", "off", 1),
}
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
GENERATION_TIMEOUT = float(os.getenv("GENERATION_TIMEOUT", "10800"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1"))
OUTPUTS_DIR = Path(os.getenv("GRADIO_OUTPUT_DIR", COMFY_DIR.parent / "gradio_outputs"))
GALLERY_THUMBNAILS_DIR = OUTPUTS_DIR / ".gallery_thumbnails"
GALLERY_LIMIT = max(1, int(os.getenv("GRADIO_GALLERY_LIMIT", "200")))
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".gif"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
GALLERY_METADATA_CACHE_LIMIT = max(
    GALLERY_LIMIT,
    int(os.getenv("GRADIO_GALLERY_METADATA_CACHE_LIMIT", "512")),
)

HTTP = requests.Session()
_GALLERY_RESOLUTION_CACHE: dict[tuple[str, int, int], tuple[int, int] | None] = {}

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_COMFY_REWRITE_TYPES = (
    "text/html",
    "text/css",
)


def _proxy_headers(headers: Any) -> dict[str, str]:
    connection = next(
        (
            value
            for key, value in headers.items()
            if key.lower() == "connection"
        ),
        "",
    )
    connection_tokens = {
        token.strip().lower()
        for token in connection.split(",")
        if token.strip()
    }
    blocked = _HOP_BY_HOP_HEADERS | connection_tokens | {"host", "content-length"}
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in blocked
    }


def _rewrite_comfy_text(content: str, content_type: str) -> str:
    """Keep ComfyUI's static browser assets inside the proxy prefix."""
    prefix = COMFY_PROXY_PATH
    if content_type.startswith("text/html"):
        content = re.sub(
            r"(?i)(\b(?:href|src|action)\s*=\s*[\"']?)/(?!/)",
            rf"\1{prefix}/",
            content,
        )
        if not re.search(r"(?i)<base(?:\s|>)", content):
            content = re.sub(
                r"(?i)(<head(?:\s[^>]*)?>)",
                rf'\1<base href="{prefix}/">',
                content,
                count=1,
            )
    elif content_type.startswith("text/css"):
        content = re.sub(
            r"(?i)(url\(\s*[\"']?)/(?!/)",
            rf"\1{prefix}/",
            content,
        )
    return content


def _comfy_upstream_path(path: str, raw_path: bytes | None) -> str:
    """Preserve encoded userdata separators consumed by ASGI route matching."""
    prefix = f"{COMFY_PROXY_PATH}/".encode("ascii")
    if isinstance(raw_path, bytes) and raw_path.startswith(prefix):
        try:
            return raw_path[len(prefix):].decode("ascii")
        except UnicodeDecodeError:
            pass
    return quote(path, safe="/:@")


async def _close_websocket(socket: WebSocket, code: int = 1000) -> None:
    try:
        await socket.close(code=code)
    except (RuntimeError, WebSocketDisconnect):
        # The browser may already have completed its side of the close handshake.
        pass


async def _relay_comfy_websocket(
    socket: WebSocket,
    upstream: aiohttp.ClientWebSocketResponse,
) -> None:
    async def browser_to_comfy() -> None:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("text")
            if payload is not None:
                await upstream.send_str(payload)
            else:
                await upstream.send_bytes(message.get("bytes", b""))

    async def comfy_to_browser() -> None:
        async for message in upstream:
            if message.type == aiohttp.WSMsgType.TEXT:
                await socket.send_text(message.data)
            elif message.type == aiohttp.WSMsgType.BINARY:
                await socket.send_bytes(message.data)
            elif message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(
                    f"ComfyUI websocket failed: {upstream.exception()}"
                )
            elif message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            }:
                return

    tasks = {
        asyncio.create_task(browser_to_comfy()),
        asyncio.create_task(comfy_to_browser()),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(
            result,
            asyncio.CancelledError,
        ):
            raise result


def _append_set_cookies(response: Response, headers: httpx.Headers) -> Response:
    for cookie in headers.get_list("set-cookie"):
        response.headers.append("set-cookie", cookie)
    return response


def build_server(demo: gr.Blocks, allowed_paths: list[str]) -> FastAPI:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=30),
        follow_redirects=False,
        trust_env=False,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(lifespan=lifespan)
    @app.get(
        "/ltx25-workflows/{workflow_id}.json",
        name="download_ltx25_workflow",
        include_in_schema=False,
    )
    async def download_ltx25_workflow(workflow_id: str) -> FileResponse:
        entry = next(
            (
                candidate for candidate in LTX25_WORKFLOWS.values()
                if candidate["id"] == workflow_id
            ),
            None,
        )
        if entry is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        root = LTX25_WORKFLOW_TEMPLATE_DIR.resolve()
        candidate = (root / entry["filename"]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(
                status_code=404,
                detail="Workflow template is not installed; re-run setup_h3.py",
            )
        return FileResponse(
            candidate,
            filename=entry["filename"],
            media_type="application/json",
        )

    @app.get(COMFY_PROXY_PATH, include_in_schema=False)
    async def comfy_slash_redirect() -> RedirectResponse:
        return RedirectResponse(f"{COMFY_PROXY_PATH}/", status_code=307)

    @app.get(
        "/downloads/{bucket}/{file_path:path}",
        name="download_generated_video",
        include_in_schema=False,
    )
    async def download_generated_video(
        bucket: str,
        file_path: str,
        download: bool = False,
    ) -> FileResponse:
        roots = {
            "comfy": OUTPUT_DIR.resolve(),
            "gradio": OUTPUTS_DIR.resolve(),
        }
        root = roots.get(bucket)
        if root is None:
            raise HTTPException(status_code=404, detail="Video not found")
        candidate = (root / file_path).resolve()
        if (
            not candidate.is_relative_to(root)
            or candidate.suffix.lower() not in VIDEO_EXTENSIONS
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(
            candidate,
            filename=candidate.name if download else None,
        )

    @app.api_route(
        f"{COMFY_PROXY_PATH}/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def comfy_http_proxy(path: str, request: Request) -> Response:
        upstream_path = _comfy_upstream_path(
            path,
            request.scope.get("raw_path"),
        )
        target = f"{COMFY_URL}/{upstream_path}"
        if request.url.query:
            target += f"?{request.url.query}"
        upstream_request = client.build_request(
            request.method,
            target,
            headers=_proxy_headers(request.headers),
            content=request.stream(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            print(f"[h3-ui] ComfyUI HTTP proxy error: {exc}", flush=True)
            return PlainTextResponse(
                "ComfyUI backend is unavailable",
                status_code=502,
            )
        headers = _proxy_headers(upstream.headers)
        headers.pop("set-cookie", None)
        location = headers.get("location")
        if location and location.startswith("/") and not location.startswith("//"):
            headers["location"] = f"{COMFY_PROXY_PATH}{location}"

        content_type = upstream.headers.get("content-type", "")
        if any(content_type.startswith(kind) for kind in _COMFY_REWRITE_TYPES):
            body = await upstream.aread()
            await upstream.aclose()
            encoding = upstream.encoding or "utf-8"
            rewritten = _rewrite_comfy_text(
                body.decode(encoding, errors="replace"),
                content_type,
            )
            for name in ("content-encoding", "content-length", "etag"):
                headers.pop(name, None)
            response = Response(
                content=rewritten.encode(encoding),
                status_code=upstream.status_code,
                headers=headers,
                media_type=None,
            )
            return _append_set_cookies(response, upstream.headers)

        async def stream_response() -> Any:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(
            stream_response(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=None,
        )
        return _append_set_cookies(response, upstream.headers)

    @app.websocket(f"{COMFY_PROXY_PATH}/{{path:path}}")
    async def comfy_websocket_proxy(socket: WebSocket, path: str) -> None:
        query = f"?{socket.url.query}" if socket.url.query else ""
        upstream_url = re.sub(r"^http", "ws", COMFY_URL, count=1)
        upstream_path = _comfy_upstream_path(
            path,
            socket.scope.get("raw_path"),
        )
        upstream_url += f"/{upstream_path}{query}"
        protocols = [
            value.strip()
            for value in socket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
            async with aiohttp.ClientSession(
                timeout=timeout,
                trust_env=False,
            ) as session:
                async with session.ws_connect(
                    upstream_url,
                    protocols=protocols,
                    max_msg_size=0,
                    autoping=True,
                ) as upstream:
                    await socket.accept(subprotocol=upstream.protocol or None)
                    await _relay_comfy_websocket(socket, upstream)
                    await _close_websocket(socket, upstream.close_code or 1000)
        except WebSocketDisconnect:
            # The browser or an outer deployment proxy completed the close.
            return
        except Exception as exc:
            print(
                "[h3-ui] ComfyUI websocket proxy error: "
                f"{type(exc).__name__}: {exc!r}",
                flush=True,
            )
            await _close_websocket(socket, code=1011)

    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        allowed_paths=allowed_paths,
        show_error=True,
    )


class H3Error(RuntimeError):
    pass


def _gemini_api_key(temporary_key: str | None) -> str:
    key = str(temporary_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set GEMINI_API_KEY in the server environment or enter a temporary "
            "Gemini API key in Prompt enhancer."
        )
    return key


def _uploaded_media_path(value: Any) -> Path | None:
    """Resolve filepath values returned by Gradio media components."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
    elif isinstance(value, dict):
        raw = value.get("path") or value.get("name")
        candidate = Path(raw) if raw else None
    elif isinstance(value, (tuple, list)) and value:
        # Older Gradio Video versions may return (video_path, subtitles_path).
        return _uploaded_media_path(value[0])
    else:
        raw = getattr(value, "path", None) or getattr(value, "name", None)
        candidate = Path(raw) if raw else None
    if candidate is None or not candidate.is_file():
        raise H3Error(f"Prompt-enhancer media file does not exist: {candidate}")
    return candidate


def _gemini_mime_type(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type:
        return mime_type
    suffix_defaults = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    return suffix_defaults.get(path.suffix.lower(), "application/octet-stream")


def _gemini_error(response: requests.Response, action: str) -> H3Error:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or response.text
    except (ValueError, AttributeError):
        detail = response.text
    detail = re.sub(r"\s+", " ", str(detail)).strip()[:800]
    return H3Error(f"Gemini {action} failed (HTTP {response.status_code}): {detail}")


def _upload_gemini_file(
    session: requests.Session,
    path: Path,
    api_key: str,
) -> dict[str, Any]:
    mime_type = _gemini_mime_type(path)
    size = path.stat().st_size
    headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    started = session.post(
        f"{GEMINI_API_ROOT}/upload/v1beta/files",
        headers=headers,
        json={"file": {"displayName": path.name[:512]}},
        timeout=60,
    )
    if not started.ok:
        raise _gemini_error(started, f"upload initialization for {path.name}")
    upload_url = started.headers.get("x-goog-upload-url")
    if not upload_url:
        raise H3Error(f"Gemini did not return an upload URL for {path.name}.")
    with path.open("rb") as source:
        uploaded = session.post(
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": mime_type,
            },
            data=source,
            timeout=600,
        )
    if not uploaded.ok:
        raise _gemini_error(uploaded, f"upload for {path.name}")
    file_info = uploaded.json().get("file", {})
    if not file_info.get("name") or not file_info.get("uri"):
        raise H3Error(f"Gemini returned incomplete file metadata for {path.name}.")
    return file_info


def _wait_for_gemini_file(
    session: requests.Session,
    file_info: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    name = str(file_info["name"])
    deadline = time.monotonic() + 600
    while str(file_info.get("state", "ACTIVE")).upper() == "PROCESSING":
        if time.monotonic() >= deadline:
            raise H3Error(f"Gemini timed out while processing {name}.")
        time.sleep(2)
        response = session.get(
            f"{GEMINI_API_ROOT}/v1beta/{name}",
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        if not response.ok:
            raise _gemini_error(response, f"file status check for {name}")
        file_info = response.json()
    state = str(file_info.get("state", "ACTIVE")).upper()
    if state == "FAILED":
        detail = file_info.get("error", {}).get("message", "unknown processing error")
        raise H3Error(f"Gemini could not process {name}: {detail}")
    return file_info


def _active_prompt_media(
    mode: str,
    first_image: Any,
    last_image: Any,
    reference_images: Iterable[Any],
    reference_videos: Iterable[Any],
    reference_audios: Iterable[Any],
) -> list[tuple[str, Path]]:
    media: list[tuple[str, Path]] = []
    if mode == "First / last frame":
        frame_values = []
        if first_image is not None:
            frame_values.append(("<Picture 1> (first frame)", first_image))
        if last_image is not None:
            picture_number = 2 if first_image is not None else 1
            frame_values.append(
                (f"<Picture {picture_number}> (last frame)", last_image)
            )
        for label, value in frame_values:
            path = _uploaded_media_path(value) if value is not None else None
            if path is not None:
                media.append((label, path))
        return media
    if mode != "Reference media":
        return media
    groups = (
        ("Picture", reference_images),
        ("Video", reference_videos),
        ("Audio", reference_audios),
    )
    for prefix, values in groups:
        for index, value in enumerate(values, 1):
            path = _uploaded_media_path(value) if value is not None else None
            if path is not None:
                media.append((f"<{prefix} {index}>", path))
    return media


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
    """Generate a model-specific prompt from text and optional image inputs."""
    try:
        if model not in GEMINI_PROMPT_MODELS:
            raise H3Error(f"Unsupported Gemini prompt model: {model}")
        key = _gemini_api_key(temporary_api_key)
        if not system_path.is_file():
            raise H3Error(f"Missing {target} system prompt: {system_path}")
        system_prompt = system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error(f"{target} system prompt is empty.")
        media: list[tuple[str, Path]] = []
        for label, value in media_values:
            if value is not None:
                path = _uploaded_media_path(value)
                if path is not None:
                    media.append((label, path))
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload an image before enhancing.")
        parts: list[dict[str, Any]] = [{
            "text": (
                f"Create the final {target} prompt from the following user input.\n"
                f"{context}\nUser text:\n"
                f"{rough_prompt or '(No text supplied; infer only from the images.)'}"
            )
        }]
        uploaded_names: list[str] = []
        with requests.Session() as session:
            try:
                for label, path in media:
                    parts.append({"text": f"The next uploaded image is {label}."})
                    file_info = _upload_gemini_file(session, path, key)
                    uploaded_names.append(str(file_info["name"]))
                    file_info = _wait_for_gemini_file(session, file_info, key)
                    parts.append({"fileData": {
                        "mimeType": file_info.get("mimeType") or _gemini_mime_type(path),
                        "fileUri": file_info["uri"],
                    }})
                response = session.post(
                    f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 16384},
                    },
                    timeout=600,
                )
                if not response.ok:
                    raise _gemini_error(response, "prompt generation")
                payload = response.json()
                candidates = payload.get("candidates") or []
                text_parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                enhanced = "".join(str(part.get("text", "")) for part in text_parts).strip()
                if enhanced.startswith("```") and enhanced.endswith("```"):
                    enhanced = re.sub(r"^```[^\n]*\n?", "", enhanced)
                    enhanced = re.sub(r"\n?```$", "", enhanced).strip()
                if not enhanced:
                    reason = candidates[0].get("finishReason") if candidates else payload.get("promptFeedback", {}).get("blockReason", "no candidate")
                    raise H3Error(f"Gemini returned no enhanced prompt ({reason}).")
                return enhanced, f"Enhanced {target} prompt with {model} using {len(media)} image(s)."
            finally:
                for name in uploaded_names:
                    try:
                        session.delete(f"{GEMINI_API_ROOT}/v1beta/{name}", headers={"x-goog-api-key": key}, timeout=30)
                    except requests.RequestException:
                        pass
    except (H3Error, requests.RequestException, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"


def enhance_music3_prompt(
    prompt: str, model: str, temporary_api_key: str,
    lyrics: str, ref_image_1: Any, ref_image_2: Any, ref_image_3: Any,
) -> tuple[str, str, str]:
    caption, status = _enhance_prompt_from_media(
        prompt=prompt, model=model, temporary_api_key=temporary_api_key,
        target="MiniMax Music 3", system_path=PROMPT_ENHANCER_SYSTEMS["MiniMax Music 3"],
        media_values=(("Music reference image 1", ref_image_1), ("Music reference image 2", ref_image_2), ("Music reference image 3", ref_image_3)),
        context=f"Existing lyrics (preserve them unless formatting only):\n{lyrics or '(none)'}",
    )
    # Music 3's writer returns both fields using explicit markers. Keep a
    # graceful fallback for older/custom prompt responses that return caption
    # text only.
    generated_lyrics = str(lyrics or "").strip()
    marker_match = re.search(
        r"(?is)^\s*CAPTION:\s*(.*?)\s*LYRICS:\s*(.*)\s*$", caption
    )
    if marker_match:
        caption = marker_match.group(1).strip()
        candidate_lyrics = marker_match.group(2).strip()
        if not generated_lyrics and candidate_lyrics.upper() != "N/A":
            generated_lyrics = candidate_lyrics
    return caption, generated_lyrics, status


def enhance_ltx25_prompt(
    prompt: str, model: str, temporary_api_key: str, mode: str,
    start_image: Any, middle_image: Any, end_image: Any,
    duration: float, width: int, height: int,
) -> tuple[str, str]:
    return _enhance_prompt_from_media(
        prompt=prompt, model=model, temporary_api_key=temporary_api_key,
        target="LTX-2.5", system_path=PROMPT_ENHANCER_SYSTEMS["LTX-2.5"],
        media_values=(("Start keyframe", start_image), ("Middle keyframe", middle_image), ("End keyframe", end_image)),
        context=f"Mode: {mode}\nDuration: {float(duration):.2f} seconds\nOutput: {int(width)}x{int(height)}",
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
    """Generate or enhance an H3 prompt from the active text and media inputs."""
    try:
        if model not in GEMINI_PROMPT_MODELS:
            raise H3Error(f"Unsupported Gemini prompt model: {model}")
        key = _gemini_api_key(temporary_api_key)
        if not PROMPT_ENHANCER_SYSTEM_PATH.is_file():
            raise H3Error(
                f"Missing prompt enhancer system prompt: {PROMPT_ENHANCER_SYSTEM_PATH}"
            )
        system_prompt = PROMPT_ENHANCER_SYSTEM_PATH.read_text(
            encoding="utf-8"
        ).strip()
        if not system_prompt:
            raise H3Error("prompt.txt is empty.")
        media = _active_prompt_media(
            mode,
            first_image,
            last_image,
            (
                ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
                ref_image_6, ref_image_7, ref_image_8, ref_image_9,
            ),
            (ref_video_1, ref_video_2, ref_video_3),
            (ref_audio_1, ref_audio_2, ref_audio_3),
        )
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload media before enhancing.")

        normalized_result = normalize_result_format(result_format)
        timing_request = (
            f"Requested image frames: {validate_image_frame_count(image_frames)}"
            if normalized_result == "Image"
            else f"Requested duration: {float(duration):.2f} seconds"
        )
        requested_width = 32 if normalized_result == "Audio" else int(width)
        requested_height = 32 if normalized_result == "Audio" else int(height)
        parts: list[dict[str, Any]] = [{
            "text": (
                "Create the final MiniMax H3 prompt from the following user request.\n"
                f"UI mode: {mode}\nResult format: {normalized_result}\n"
                f"{timing_request}\n"
                f"Requested output: {requested_width}x{requested_height}\n"
                f"User text:\n{rough_prompt or '(No text supplied; infer only from the media.)'}"
            )
        }]
        uploaded_names: list[str] = []
        with requests.Session() as session:
            try:
                for label, path in media:
                    parts.append({"text": f"The next uploaded file is {label}."})
                    file_info = _upload_gemini_file(session, path, key)
                    uploaded_names.append(str(file_info["name"]))
                    file_info = _wait_for_gemini_file(session, file_info, key)
                    parts.append({
                        "fileData": {
                            "mimeType": file_info.get("mimeType")
                            or _gemini_mime_type(path),
                            "fileUri": file_info["uri"],
                        }
                    })
                response = session.post(
                    f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
                    headers={
                        "x-goog-api-key": key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {
                            "temperature": 0.6,
                            "maxOutputTokens": 16384,
                        },
                    },
                    timeout=600,
                )
                if not response.ok:
                    raise _gemini_error(response, "prompt generation")
                payload = response.json()
                candidates = payload.get("candidates") or []
                text_parts = (
                    candidates[0].get("content", {}).get("parts", [])
                    if candidates else []
                )
                enhanced = "".join(
                    str(part.get("text", "")) for part in text_parts
                ).strip()
                if enhanced.startswith("```") and enhanced.endswith("```"):
                    enhanced = re.sub(r"^```[^\n]*\n?", "", enhanced)
                    enhanced = re.sub(r"\n?```$", "", enhanced).strip()
                if not enhanced:
                    reason = (
                        candidates[0].get("finishReason") if candidates else
                        payload.get("promptFeedback", {}).get(
                            "blockReason", "no candidate"
                        )
                    )
                    raise H3Error(f"Gemini returned no enhanced prompt ({reason}).")
                return (
                    enhanced,
                    f"Enhanced with {model} using {len(media)} media file(s).",
                )
            finally:
                for name in uploaded_names:
                    try:
                        session.delete(
                            f"{GEMINI_API_ROOT}/v1beta/{name}",
                            headers={"x-goog-api-key": key},
                            timeout=30,
                        )
                    except requests.RequestException:
                        pass
    except (H3Error, requests.RequestException, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"


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
    """Dispatch H3 prompt enhancement to the selected local or Gemini backend."""
    if backend == "Gemini":
        return _enhance_h3_prompt_with_gemini(
            prompt,
            gemini_model,
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
    if backend != "Local MiniMax-H3 8B":
        return str(prompt or ""), f"Prompt enhancement failed: unsupported backend {backend!r}."
    try:
        if normalize_result_format(result_format) != "Video":
            raise H3Error("The local 8B writer supports H3 audio-video prompts only.")
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt:
            raise H3Error("Enter a text prompt before enhancing locally.")
        numeric_duration = float(duration)
        if not numeric_duration.is_integer():
            raise H3Error(
                "The local 8B writer was trained for whole-second durations; "
                "choose an integer duration from 4 to 15 seconds."
            )
        task = local_prompt_task(mode, first_image, last_image)
        resolution = local_prompt_resolution(width, height, task)
        return rewrite_local_h3_prompt(
            prompt=rough_prompt,
            task=task,
            resolution=resolution,
            duration=int(numeric_duration),
            first_frame=first_image,
            last_frame=last_image,
            base_model=local_base_model,
            max_new_tokens=int(local_max_new_tokens),
            temperature=float(local_temperature),
            top_p=float(local_top_p),
            greedy=bool(local_greedy),
            seed=int(local_seed),
        )
    except (H3Error, OSError, RuntimeError, ValueError) as exc:
        return str(prompt or ""), f"Local prompt enhancement failed: {exc}"


def prompt_writer_backend_visibility(backend: str) -> tuple[Any, Any]:
    """Show only controls belonging to the selected prompt-writer backend."""
    local_selected = backend == "Local MiniMax-H3 8B"
    return gr.update(visible=local_selected), gr.update(visible=not local_selected)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    duration: float = 0.0
    frame_count: int = 0
    has_audio: bool = False


@dataclass(frozen=True)
class UpscaleClipBatch:
    sources: tuple[str, ...]
    temporary_inputs: tuple[Path, ...] = ()
    temporary_directory: Path | None = None


def normalize_turbo_variant(value: str) -> str:
    return value if value in TURBO_SETTINGS else DEFAULT_TURBO


def turbo_steps_for(value: str) -> int:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].steps


def turbo_strength_for(value: str) -> float:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].strength


def turbo_uses_custom_nodes(value: str) -> bool:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].custom_nodes


def is_original_bf16_model(model_filename: str) -> bool:
    originals = {
        MODEL_SPECS["original_fl2va"].local_name,
        MODEL_SPECS["original_ref2va"].local_name,
    }
    return Path(str(model_filename)).name in originals


@dataclass
class ModelProfile:
    label: str
    fl2va: str
    ref2va: str
    fl2va_source: str = "unknown"
    ref2va_source: str = "unknown"


@dataclass
class ModelConfig:
    profiles: dict[str, ModelProfile]
    default_profile: str
    text_encoder: str
    video_vae: str
    audio_vae: str
    video_vae_int8: str | None = None
    video_vae_int8_source: str = "unknown"
    turbo_lora: str | None = None
    turbo_source: str = "unknown"
    turbo_ref_lora: str | None = None
    turbo_ref_source: str = "unknown"
    turbo_8step_lora: str | None = None
    turbo_8step_source: str = "unknown"
    turbo_8step_ref_lora: str | None = None
    turbo_8step_ref_source: str = "unknown"
    larry_turbo_lora: str | None = None
    larry_turbo_source: str = "unknown"
    larry_turbo_ref_lora: str | None = None
    larry_turbo_ref_source: str = "unknown"
    seedvr2_dit: str | None = None
    seedvr2_dit_source: str = "unknown"
    seedvr2_models: dict[str, str] | None = None
    seedvr2_vae: str | None = None
    seedvr2_vae_source: str = "unknown"

    def profile_key(self, name: str) -> str:
        key = str(name).strip().lower()
        if key not in self.profiles:
            key = self.default_profile
        if key not in self.profiles:
            raise H3Error(f"Unknown model profile: {name}")
        return key

    def profile(self, name: str) -> ModelProfile:
        return self.profiles[self.profile_key(name)]

    def turbo_lora_for(self, mode: str, turbo_variant: str) -> str | None:
        reference = str(mode).strip().lower() == "reference media"
        spec = TURBO_SETTINGS[normalize_turbo_variant(turbo_variant)]
        return getattr(self, spec.ref_lora_attr if reference else spec.lora_attr)


def load_model_config() -> ModelConfig:
    if not MODELS_CONFIG.exists():
        raise H3Error(
            f"Missing model configuration: {MODELS_CONFIG}. Run setup_h3.py first."
        )
    data = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))

    if "profiles" not in data:
        legacy = ModelProfile(
            label="Speed",
            fl2va=data["fl2va"],
            ref2va=data["ref2va"],
            fl2va_source=data.get("fl2va_source", "legacy"),
            ref2va_source=data.get("ref2va_source", "legacy"),
        )
        profiles = {"speed": legacy}
        default_profile = "speed"
    else:
        profiles = {
            key.lower(): ModelProfile(**value)
            for key, value in data["profiles"].items()
        }
        default_profile = str(data.get("default_profile", "speed")).lower()

    return ModelConfig(
        profiles=profiles,
        default_profile=default_profile,
        text_encoder=data["text_encoder"],
        video_vae=data["video_vae"],
        audio_vae=data["audio_vae"],
        video_vae_int8=data.get("video_vae_int8"),
        video_vae_int8_source=data.get("video_vae_int8_source", "unknown"),
        turbo_lora=data.get("turbo_lora"),
        turbo_source=data.get("turbo_source", "unknown"),
        turbo_ref_lora=data.get("turbo_ref_lora", data.get("turbo_lora")),
        turbo_ref_source=data.get(
            "turbo_ref_source", data.get("turbo_source", "unknown")
        ),
        turbo_8step_lora=data.get("turbo_8step_lora"),
        turbo_8step_source=data.get("turbo_8step_source", "unknown"),
        turbo_8step_ref_lora=data.get(
            "turbo_8step_ref_lora", data.get("turbo_8step_lora")
        ),
        turbo_8step_ref_source=data.get(
            "turbo_8step_ref_source", data.get("turbo_8step_source", "unknown")
        ),
        larry_turbo_lora=data.get("larry_turbo_lora"),
        larry_turbo_source=data.get("larry_turbo_source", "unknown"),
        larry_turbo_ref_lora=data.get(
            "larry_turbo_ref_lora", data.get("larry_turbo_lora")
        ),
        larry_turbo_ref_source=data.get(
            "larry_turbo_ref_source", data.get("larry_turbo_source", "unknown")
        ),
        seedvr2_dit=data.get("seedvr2_dit"),
        seedvr2_dit_source=data.get("seedvr2_dit_source", "unknown"),
        seedvr2_models=data.get("seedvr2_models"),
        seedvr2_vae=data.get("seedvr2_vae"),
        seedvr2_vae_source=data.get("seedvr2_vae_source", "unknown"),
    )


def model_file_is_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > MIN_VALID_MODEL_BYTES


def h3_latent_upscaler_settings(model_choice: str) -> tuple[str, str, str]:
    key = H3_LATENT_UPSCALER_MODEL_CHOICES.get(str(model_choice))
    if key is None:
        raise H3Error(f"Unknown H3 latent upscaler model: {model_choice}")
    precision = {
        "h3_latent_upscaler_3d_bf16": "bf16",
        "h3_latent_upscaler_3d_fp16": "fp16",
        "h3_latent_upscaler_3d_fp32": "fp32",
    }[key]
    return key, MODEL_SPECS[key].local_name, precision


def ensure_h3_latent_upscaler_model(model_choice: str) -> bool:
    """Download the selected native H3 latent upscaler on first use."""
    model_key, filename, _precision = h3_latent_upscaler_settings(model_choice)
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-latent-upscaler-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    destination = COMFY_DIR / "models" / MODEL_SPECS[model_key].folder / filename
    if not model_file_is_ready(destination):
        raise H3Error(
            f"On-demand H3 latent upscaler download did not produce {filename}."
        )
    return True


def ensure_profile_model(
    profile_key: str,
    profile: ModelProfile,
    mode: str,
) -> bool:
    """Download a lazy profile checkpoint before submitting its workflow."""
    reference = str(mode).strip().lower() == "reference media"
    filename = profile.ref2va if reference else profile.fl2va
    model_key = f"{profile_key}_{'ref2va' if reference else 'fl2va'}"
    destination = (
        COMFY_DIR / "models" / MODEL_SPECS[model_key].folder / filename
    )
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand model download did not produce {filename}.")
    return True


def ensure_turbo_lora(models: ModelConfig, turbo_variant: str, mode: str) -> bool:
    """Download a non-default Turbo LoRA only when its variant is selected."""
    variant = normalize_turbo_variant(turbo_variant)
    reference = str(mode).strip().lower() == "reference media"
    if variant == LARRY_TURBO:
        model_key = "larry_turbo_lora"
        filename = models.larry_turbo_ref_lora if reference else models.larry_turbo_lora
    elif variant == LIGHTX2V_8STEP_TURBO:
        model_key = "turbo_8step_lora"
        filename = models.turbo_8step_ref_lora if reference else models.turbo_8step_lora
    else:
        return False
    if not filename:
        raise H3Error(f"{variant} Turbo LoRA is not configured.")
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=(model_key,),
    ):
        return False
    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-turbo-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    destination = COMFY_DIR / "models" / MODEL_SPECS[model_key].folder / filename
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand Turbo LoRA download did not produce {filename}.")
    return True


def ensure_int8_video_vae(models: ModelConfig) -> bool:
    """Download the optional INT8 ConvRot video VAE on first use."""
    filename = models.video_vae_int8
    if not filename:
        raise H3Error(
            "The model configuration predates INT8 video VAE support. "
            "Re-run setup_h3.py before enabling it."
        )
    destination = (
        COMFY_DIR / "models" / MODEL_SPECS["video_vae_int8"].folder / filename
    )
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=("video_vae_int8",),
    ):
        return False

    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[h3-int8-vae-on-demand]",
        model_keys=("video_vae_int8",),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand INT8 VAE download did not produce {filename}.")
    return True


def ltx25_model_keys(model_choice: str = DEFAULT_LTX25_MODEL) -> dict[str, str]:
    selected_key = LTX25_MODEL_CHOICES.get(str(model_choice))
    if selected_key is None:
        raise H3Error(f"Unknown LTX-2.5 model: {model_choice}")
    return {
        "distilled": selected_key,
        **{
            key.removeprefix("ltx25_"): key
            for key in LTX25_SHARED_MODEL_KEYS
        },
    }


def ltx25_model_names(model_choice: str = DEFAULT_LTX25_MODEL) -> dict[str, str]:
    return {
        role: MODEL_SPECS[key].local_name
        for role, key in ltx25_model_keys(model_choice).items()
    }


def missing_ltx25_model_names(model_choice: str = DEFAULT_LTX25_MODEL) -> list[str]:
    stale = stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
        model_keys=ltx25_model_keys(model_choice).values(),
    )
    return [MODEL_SPECS[key].local_name for key in stale]


def ensure_ltx25_models(model_choice: str = DEFAULT_LTX25_MODEL) -> bool:
    """Download the gated LTX-2.5 model set only when its tab is used."""
    required_keys = tuple(ltx25_model_keys(model_choice).values())
    if not missing_ltx25_model_names(model_choice):
        return False
    token = resolve_hf_token()
    if token is None:
        raise H3Error(
            "No Hugging Face credential was found. On standalone, run "
            "`hf auth login` as the same user that launches run_h3.sh, or set "
            "HF_TOKEN. On Modal, attach a Secret containing HF_TOKEN to the "
            "serve function and redeploy."
        )
    try:
        sync_models(
            root=COMFY_DIR / "models",
            manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
            token=token,
            log_prefix="[ltx25-on-demand]",
            model_keys=required_keys,
            download_workers=len(required_keys),
        )
    except Exception as exc:
        raise H3Error(
            "LTX-2.5 model download failed. Accept the Lightricks/LTX-2.5 "
            "Hugging Face license and authenticate with `hf auth login` or "
            "HF_TOKEN, then retry. "
            f"Details: {exc}"
        ) from exc
    for key in required_keys:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(COMFY_DIR / "models" / spec.folder / spec.local_name):
            raise H3Error(f"On-demand LTX-2.5 download did not produce {spec.local_name}.")
    return True


def ltx25_workflow_entry(workflow_label: str) -> dict[str, Any]:
    entry = LTX25_WORKFLOWS.get(str(workflow_label))
    if entry is None:
        raise H3Error(f"Unknown official LTX-2.5 workflow: {workflow_label}")
    return entry


def ltx25_workflow_model_keys(workflow_label: str) -> tuple[str, ...]:
    entry = ltx25_workflow_entry(workflow_label)
    common = list(LTX25_WORKFLOW_COMMON_MODEL_KEYS)
    if entry.get("audio_only"):
        common.remove("ltx25_video_vae")
        common.remove("ltx25_video_vae_full")
    return tuple(dict.fromkeys((*common, *entry["extra_models"])))


def ltx25_official_inventory_keys() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *LTX25_WORKFLOW_COMMON_MODEL_KEYS,
                *LTX25_ICLORA_MODEL_KEYS,
                "ltx25_spatial_upscaler",
            )
        )
    )


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
        rows.append(
            f"| `{spec.local_name}` | `{spec.folder}` | {status} | {source} |"
        )
    return (
        f"**Installed: {installed}/{len(keys)}**\n\n"
        "| Model | ComfyUI folder | Status | Source / license |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
    )


def render_ltx25_workflow_details(workflow_label: str) -> str:
    entry = ltx25_workflow_entry(workflow_label)
    extra_names = [
        MODEL_SPECS[key].local_name for key in entry["extra_models"]
    ]
    extras = (
        ", ".join(f"`{name}`" for name in extra_names)
        if extra_names else "No use-case-specific checkpoint."
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
                COMFY_DIR / "models" / MODEL_SPECS[key].folder
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


def seedvr2_upscale_model_names(
    models: ModelConfig,
    model_choice: str,
) -> dict[str, str]:
    choice = str(model_choice)
    if choice not in SEEDVR2_MODEL_CHOICES:
        raise H3Error(f"Unknown SeedVR2 model: {model_choice}")
    configured_models = models.seedvr2_models or {}
    selected = configured_models.get(choice)
    if selected is None and choice == DEFAULT_SEEDVR2_MODEL:
        selected = models.seedvr2_dit
    configured = {"seedvr2_dit": selected, "seedvr2_vae": models.seedvr2_vae}
    missing_config = [key for key, value in configured.items() if not value]
    if missing_config:
        raise H3Error(
            "The model configuration predates SeedVR2 support. Re-run "
            "setup_h3.py before selecting SeedVR2. Missing keys: "
            + ", ".join(missing_config)
        )
    return {key: str(value) for key, value in configured.items()}


def ensure_seedvr2_upscale_models(
    models: ModelConfig,
    model_choice: str,
) -> bool:
    configured = seedvr2_upscale_model_names(models, model_choice)
    selected_key = SEEDVR2_MODEL_CHOICES[str(model_choice)]
    assets = (
        (selected_key, configured["seedvr2_dit"]),
        ("seedvr2_vae", configured["seedvr2_vae"]),
    )
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    missing_files = stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=(key for key, _filename in assets),
    )
    if not missing_files:
        return False

    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[seedvr2-on-demand]",
        model_keys=tuple(key for key, _filename in assets),
        download_workers=len(assets),
    )
    for key, filename in assets:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(
            COMFY_DIR / "models" / spec.folder / filename
        ):
            raise H3Error(
                f"On-demand SeedVR2 download did not produce {filename}."
            )
    return True


def ensure_ltx25_upscale_models(
    model_choice: str = DEFAULT_LTX25_MODEL,
) -> bool:
    """Lazily install the selected LTX base set and the 2x upscaler IC-LoRA."""
    base_downloaded = ensure_ltx25_models(model_choice)
    upscaler_key = "ltx25_pixel_upscaler_x2"
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    if not stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=(upscaler_key,),
    ):
        return base_downloaded

    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[ltx25-upscale-on-demand]",
        model_keys=(upscaler_key,),
        download_workers=1,
    )
    spec = MODEL_SPECS[upscaler_key]
    if not model_file_is_ready(COMFY_DIR / "models" / spec.folder / spec.local_name):
        raise H3Error(
            f"On-demand LTX-2.5 upscaler download did not produce {spec.local_name}."
        )
    return True


def music3_model_keys(model_choice: str) -> tuple[str, ...]:
    choice = str(model_choice)
    if choice not in MUSIC3_MODEL_CHOICES:
        raise H3Error(f"Unknown MiniMax Music 3 model: {model_choice}")
    return (MUSIC3_MODEL_CHOICES[choice], *MUSIC3_SHARED_MODEL_KEYS)


def missing_music3_model_names(model_choice: str) -> list[str]:
    missing = stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
        model_keys=music3_model_keys(model_choice),
    )
    return [MODEL_SPECS[key].local_name for key in missing]


def ensure_music3_models(model_choice: str) -> bool:
    """Lazily install the selected Music 3 DiT and its shared model files."""
    keys = music3_model_keys(model_choice)
    manifest_path = MODELS_CONFIG.parent / "h3_model_manifest.json"
    missing = stale_model_keys(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        model_keys=keys,
    )
    if not missing:
        return False
    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=manifest_path,
        token=resolve_hf_token(),
        log_prefix="[music3-on-demand]",
        model_keys=keys,
        download_workers=len(keys),
    )
    return True


def api_get(path: str, **kwargs: Any) -> requests.Response:
    response = HTTP.get(f"{COMFY_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def api_post(path: str, **kwargs: Any) -> requests.Response:
    response = HTTP.post(f"{COMFY_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def object_info() -> dict[str, Any]:
    return api_get("/object_info").json()


def frame_length(duration: float) -> int:
    frames = max(5, round(float(duration) * 24))
    return frames + (5 - (frames % 17)) % 17


def normalize_result_format(value: Any) -> str:
    requested = str(value or DEFAULT_RESULT_FORMAT).strip().lower()
    for result_format in RESULT_FORMATS:
        if requested == result_format.lower():
            return result_format
    raise H3Error(
        f"Unsupported result format: {value!r}. Choose {', '.join(RESULT_FORMATS)}."
    )


def validate_image_frame_count(value: Any) -> int:
    try:
        frames = int(value)
    except (TypeError, ValueError) as exc:
        raise H3Error("Image frame count must be an integer.") from exc
    if not MIN_IMAGE_FRAMES <= frames <= MAX_IMAGE_FRAMES:
        raise H3Error(
            f"Image frame count must be between {MIN_IMAGE_FRAMES} and "
            f"{MAX_IMAGE_FRAMES}."
        )
    return frames


def image_sampling_length(frame_count: int) -> int:
    """Return the smallest native H3 temporal packet covering the request."""
    return 5 if validate_image_frame_count(frame_count) <= 5 else 22


def snap_to_grid(value: int | float, grid: int = 32) -> int:
    grid = max(1, int(grid))
    return max(grid, round(int(value) / grid) * grid)


def snap32(value: int | float) -> int:
    return snap_to_grid(value, 32)


def snap64(value: int | float) -> int:
    return snap_to_grid(value, 64)


def validate_resolution(width: int | float, height: int | float) -> tuple[int, int]:
    resolved_width = snap32(width)
    resolved_height = snap32(height)
    return resolved_width, resolved_height


def h3_latent_upscale_dimensions(
    width: int | float,
    height: int | float,
) -> tuple[int, int, int, int]:
    """Return auto-aligned 2x source/target canvases for native H3 upscale."""
    target_width, target_height = snap64(width), snap64(height)
    source_width = target_width // 2
    source_height = target_height // 2
    return source_width, source_height, target_width, target_height


def resolution_for_aspect_ratio(
    source_width: int | float,
    source_height: int | float,
    *,
    preserve_native: bool = False,
    alignment: int = 32,
) -> tuple[int, int]:
    """Return an aligned native or sub-2 MP canvas matching an image ratio."""
    width = float(source_width)
    height = float(source_height)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise H3Error("The start frame has no usable image dimensions.")

    if preserve_native:
        return snap_to_grid(width, alignment), snap_to_grid(height, alignment)

    # Do not upscale a smaller source image. Preserve its native dimensions;
    # only oversized inputs need aspect-ratio-based downscaling.
    if width * height < 2_000_000:
        return int(width), int(height)

    ratio = width / height
    if ratio >= 1:
        resolved_width = max(
            alignment,
            math.floor(math.sqrt(AUTO_RESOLUTION_PIXEL_CAP * ratio) / alignment)
            * alignment,
        )
        resolved_height = max(
            alignment,
            math.floor(resolved_width / ratio / alignment) * alignment,
        )
    else:
        resolved_height = max(
            alignment,
            math.floor(math.sqrt(AUTO_RESOLUTION_PIXEL_CAP / ratio) / alignment)
            * alignment,
        )
        resolved_width = max(
            alignment,
            math.floor(resolved_height * ratio / alignment) * alignment,
        )

    # Rounding to the model's 32-pixel grid can cross the strict 2 MP boundary.
    while resolved_width * resolved_height >= 2_000_000:
        if ratio >= 1:
            resolved_width = max(alignment, resolved_width - alignment)
            resolved_height = max(
                alignment,
                math.floor(resolved_width / ratio / alignment) * alignment,
            )
        else:
            resolved_height = max(alignment, resolved_height - alignment)
            resolved_width = max(
                alignment,
                math.floor(resolved_height * ratio / alignment) * alignment,
            )
    return resolved_width, resolved_height


# Gradio runs this in the browser when the file is selected. It deliberately
# uses the local File/preview object, so reading dimensions does not add a
# second server upload or require a Python callback.
AUTO_RESOLUTION_JS = r"""async (_value, currentWidth, currentHeight, resultFormat, latentUpscale) => {
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

    if (imageWidth * imageHeight < 2000000) {
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

    const maxPixels = 2000000 - 1;
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
    while (width * height >= 2000000) {
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


def resolution_summary(width: int | float, height: int | float) -> str:
    try:
        resolved_width, resolved_height = validate_resolution(width, height)
    except Exception as exc:
        return f"⚠️ {exc}"
    pixels = resolved_width * resolved_height
    megapixels = pixels / 1_000_000
    baseline = 864 * 480
    relative = pixels / baseline
    canvas = "native-sized" if pixels <= NATIVE_PIXEL_CAP else "extended/experimental"
    ratio = resolved_width / resolved_height
    return (
        f"**{resolved_width}×{resolved_height}** · {megapixels:.2f} MP · "
        f"{ratio:.2f}:1 · {canvas} · approximately **{relative:.1f}×** "
        "the pixel workload of 864×480."
    )


def resolution_control_updates(
    width: int | float,
    height: int | float,
    latent_upscale: bool,
    result_format: str,
) -> tuple[int | float, int | float, str]:
    if normalize_result_format(result_format) == "Audio":
        return (
            width,
            height,
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32.",
        )
    alignment = 64 if latent_upscale else 32
    resolved_width = snap_to_grid(width, alignment)
    resolved_height = snap_to_grid(height, alignment)
    prefix = "**Latent upscale 64-pixel alignment** · " if latent_upscale else ""
    return (
        resolved_width,
        resolved_height,
        prefix + resolution_summary(resolved_width, resolved_height),
    )


def auto_resolution_from_start_frame(
    first_image: Any,
    current_width: int | float | None,
    current_height: int | float | None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    latent_upscale: bool = False,
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
        return fallback_width, fallback_height, resolution_summary(
            fallback_width, fallback_height
        )

    try:
        from PIL import Image

        with Image.open(paths[0]) as image:
            width, height = resolution_for_aspect_ratio(
                *image.size,
                preserve_native=normalized_result == "Image",
                alignment=64 if latent_upscale else 32,
            )
    except Exception as exc:
        return fallback_width, fallback_height, f"⚠️ Unable to read start frame dimensions: {exc}"
    return width, height, resolution_summary(width, height)


def resolution_choice_values(name: str, tier: str) -> tuple[int, int, str]:
    key = str(tier).strip().lower()
    if key not in RESOLUTION_TIERS:
        raise H3Error(f"Unknown resolution tier: {tier}")

    table = RESOLUTION_TIERS[key]
    if name not in table:
        name = "16:9 · 864×480" if key == "fast" else next(iter(table))

    width, height = table[name]
    return (
        width,
        height,
        f"**{key.title()}** · " + resolution_summary(width, height),
    )


def normalize_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Path):
            result.append(str(item))
        elif hasattr(item, "name"):
            result.append(str(item.name))
        else:
            result.append(str(item))
    return result


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


def collect_reference_slots(*groups: Any) -> list[str]:
    collected: list[str] = []
    for group in groups:
        collected.extend(normalize_paths(group))
    return [path for path in collected if path]


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def estimate_packed_tokens(
    mode: str,
    width: int,
    height: int,
    duration: float,
    first_image: str | None = None,
    last_image: str | None = None,
) -> int:
    frames = frame_length(duration)
    latent_t = video_latent_t(frames)
    spatial_rows = max(1, width // 32) * max(1, height // 32)
    video_tokens = latent_t * spatial_rows
    audio_t = round((frames / 24.0) * 40.0)
    audio_tokens = 2 * audio_t
    keyframe_tokens = 0
    if mode == "First / last frame":
        keyframe_tokens = spatial_rows * int(bool(first_image)) + spatial_rows * int(bool(last_image))
    return video_tokens + audio_tokens + keyframe_tokens


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
        return enabled, tokens, (
            f"Auto Turbo: {tokens:,} target tokens "
            f"{'≥' if enabled else '<'} {AUTO_SOL_TOKEN_THRESHOLD:,}"
        )
    enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
    return enabled, tokens, (
        f"Auto: {tokens:,} target tokens "
        f"{'≥' if enabled else '<'} {AUTO_SOL_TOKEN_THRESHOLD:,}"
    )


def mode_layout_updates(mode: str):
    """Update task-specific inputs without changing the acceleration choice."""
    show_frames = mode == "First / last frame"
    show_refs = mode == "Reference media"

    if show_refs:
        return (
            mode_help(mode),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(interactive=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    return (
        mode_help(mode),
        gr.update(visible=show_frames),
        gr.update(visible=False),
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
    result_format = normalize_result_format(result_format)
    is_image = result_format == "Image"
    is_audio = result_format == "Audio"
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
        gr.update(visible=result_format == "Video", value=None),
        gr.update(visible=is_image),
        gr.update(visible=is_audio, value=None),
        (
            gr.update(interactive=True)
            if result_format == "Video"
            else gr.update(value="None", interactive=False)
        ),
        (
            gr.update(value=False, interactive=False)
            if is_audio
            else gr.update(interactive=True)
        ),
        gr.update(value=display_width, interactive=not is_audio),
        gr.update(value=display_height, interactive=not is_audio),
        (
            "**Audio result** · resolution controls are ignored; H3 samples at 32×32."
            if is_audio
            else resolution_summary(display_width, display_height)
        ),
        gr.update(value=f"Generate {result_format.lower()}"),
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


def generation_mode_defaults(name: str, turbo_variant: str = DEFAULT_TURBO):
    """Normal/Turbo is independent from the selected base-model profile.

    Important: when entering Turbo, do not change preset.value. Changing it
    would fire preset.change() and race with the Turbo Steps update.
    """
    if str(name).strip().lower() == "turbo":
        return (
            gr.update(interactive=False),
            gr.update(
                value=turbo_steps_for(turbo_variant),
                interactive=True,
            ),
            "simple",
            DEFAULT_ACCELERATOR,
            "Sage 2",
        )

    return (
        gr.update(value="Balanced", interactive=True),
        gr.update(value=18, interactive=True),
        "simple",
        DEFAULT_ACCELERATOR,
        "Sage 2",
    )


def turbo_variant_defaults(turbo_variant: str, generation_mode: str):
    """Apply variant sampling defaults only while Turbo is selected."""
    if str(generation_mode).strip().lower() != "turbo":
        return gr.update(), gr.update()
    return gr.update(
        value=turbo_steps_for(turbo_variant),
        interactive=True,
    ), "simple"


def resolve_cache_policy(
    cache_mode: str, *, use_turbo: bool
) -> tuple[str, str | None]:
    """Allow opt-in Turbo accelerators with quality warnings."""
    requested = str(cache_mode).strip()
    normalized = requested.lower()
    if not use_turbo or normalized == "off":
        return requested, None
    if normalized == "spectrum":
        return requested, (
            "Spectrum forecasting with Turbo is experimental. Compare the same "
            "prompt and seed with acceleration Off for quality-critical output."
        )
    if normalized == "easycache":
        return requested, (
            "EasyCache with Turbo is experimental and may amplify low-step "
            "approximation error. Compare the same prompt and seed with Off."
        )
    if normalized == "firstblockcache":
        return requested, (
            "FirstBlockCache with Turbo is experimental and may amplify low-step "
            "approximation error. Compare the same prompt and seed with Off."
        )
    return "Off", (
        f"{requested or 'Selected block cache'} was disabled automatically: "
        "the selected acceleration mode is not supported with Turbo."
    )


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
    return tuple(
        gr.update(value=value, interactive=interactive)
        for value in values
    )


def stage_file(path: str, category: str, transcode_video: bool = False) -> str:
    src = Path(path)
    if not src.is_file():
        raise H3Error(f"Input file does not exist: {src}")
    target_dir = INPUT_DIR / "h3_gradio" / category
    target_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex

    if transcode_video:
        dst = target_dir / f"{token}.mp4"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-t", "15", "-vf", "fps=24",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise H3Error(f"Reference-video conversion failed: {proc.stderr.strip()}")
    else:
        suffix = src.suffix.lower() or ".bin"
        dst = target_dir / f"{token}{suffix}"
        shutil.copy2(src, dst)

    return dst.relative_to(INPUT_DIR).as_posix()


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self._next = 1

    def add(self, class_type: str, **inputs: Any) -> str:
        node_id = str(self._next)
        self._next += 1
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    @staticmethod
    def out(node_id: str, slot: int = 0) -> list[Any]:
        return [node_id, slot]


def turbo_required_nodes(
    turbo_variant: str, model_filename: str = ""
) -> set[str]:
    """Return the external node contract for one normalized Turbo variant."""
    if turbo_uses_custom_nodes(turbo_variant):
        return {LARRY_TURBO_LORA_NODE, LARRY_TURBO_SAMPLER_NODE}
    lora_node = (
        LIGHTX2V_BYPASS_LORA_NODE
        if is_original_bf16_model(model_filename)
        else CORE_LORA_LOADER_NODE
    )
    return {lora_node, CORE_SAMPLER_NODE, FUSED_MODULATION_NODE}


def add_turbo_model_patch(
    graph: Graph,
    model_ref: list[Any],
    *,
    lora_name: str,
    model_filename: str,
    turbo_variant: str,
    strength: float,
    available_nodes: set[str],
) -> list[Any]:
    """Apply a Turbo LoRA and compatible model-level optimizations."""
    variant = normalize_turbo_variant(turbo_variant)
    runtime_bypass = is_original_bf16_model(model_filename)
    required = turbo_required_nodes(variant, model_filename)
    missing = required - available_nodes
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise H3Error(
            f"{variant} Turbo requires unavailable nodes: {missing_names}. "
            "Re-run setup_h3.py and restart ComfyUI."
        )

    if turbo_uses_custom_nodes(variant):
        turbo = graph.add(
            LARRY_TURBO_LORA_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength=float(strength),
            low_vram=not runtime_bypass,
        )
        # Larry bypasses projections on Original BF16 and merges on compact
        # Speed/Quality bases. Its pinned node receives a provisioning-time
        # modality-row fix; keep fused modulation disabled for both paths.
        return Graph.out(turbo)

    if runtime_bypass:
        turbo = graph.add(
            LIGHTX2V_BYPASS_LORA_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength=float(strength),
        )
    else:
        turbo = graph.add(
            CORE_LORA_LOADER_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength_model=float(strength),
        )

    # LightX2V bypasses Original BF16 to avoid retaining a second copy of every
    # patched backbone weight. Compact Speed/Quality bases keep the faster core
    # merge path. Both retain the standard AdaLN shape, so fusion is compatible.
    fused_modulation = graph.add(
        FUSED_MODULATION_NODE,
        model=Graph.out(turbo),
        enabled=True,
    )
    return Graph.out(fused_modulation)


def add_model_stack(
    graph: Graph,
    model_name: str,
    models: ModelConfig,
    *,
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
    available_nodes: set[str],
    use_int8_vae: bool = False,
    use_sage: bool = False,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    unet = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model_ref = Graph.out(unet)

    if turbo_lora_name:
        model_ref = add_turbo_model_patch(
            graph,
            model_ref,
            lora_name=turbo_lora_name,
            model_filename=model_name,
            turbo_variant=turbo_variant,
            strength=turbo_strength,
            available_nodes=available_nodes,
        )

    # Keep FirstBlockCache ahead of attention/object patches so its sampling and
    # diffusion wrappers own the outer execution context.
    cache_mode_normalized = str(cache_mode).strip().lower()
    if cache_mode_normalized == "firstblockcache":
        if "H3FirstBlockCache" not in available_nodes:
            raise H3Error(
                "FirstBlockCache was requested, but H3FirstBlockCache is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        cache = graph.add(
            "H3FirstBlockCache",
            model=model_ref,
            preset=str(fbcache_preset),
            residual_diff_threshold=float(fbcache_threshold),
            start_percent=float(fbcache_start),
            end_percent=float(fbcache_end),
            max_consecutive_cache_hits=max(1, int(fbcache_max_hits)),
            temporal_guard=bool(fbcache_temporal_guard),
        )
        model_ref = Graph.out(cache)


    if use_sage:
        if SAGE_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "Sage 2 was requested, but Patch Sage Attention KJ is not loaded. "
                "Re-run setup_h3.py and restart ComfyUI."
            )
        sage = graph.add(
            SAGE_ATTENTION_NODE,
            model=model_ref,
            sage_attention="auto",
            allow_compile=False,
        )
        model_ref = Graph.out(sage)

    if use_sol:
        if SOL_ATTENTION_NODE not in available_nodes:
            raise H3Error(
                "Sol-Attn was requested, but the H3 zero-copy Sol node is not loaded. "
                "Run deploy_h3.sh install and inspect the ComfyUI startup log."
            )
        exact_mode = (
            sol_exact_mode
            if sol_exact_mode in {"off", "exact_kv", "exact_kv_and_rows"}
            else "off"
        )
        dense_block_count = max(0, min(int(sol_dense_steps), 8))
        dense_blocks = ",".join(
            f"-{index}" for index in range(1, dense_block_count + 1)
        )

        # The H3-native path consumes strided q/k/v views directly, avoiding
        # the large contiguous copies required by generic attention hooks.
        sol = graph.add(
            SOL_ATTENTION_NODE,
            model=model_ref,
            enabled=True,
            tau=float(sol_tau),
            min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
            strict=False,
            thresh_type=(
                sol_thresh_type if sol_thresh_type in {"diag", "exact"} else "diag"
            ),
            int8_qk=False,
            int8_pv=False,
            sink_conditioning=exact_mode,
            dense_blocks=dense_blocks,
        )
        model_ref = Graph.out(sol)

    # The ConvRot quality checkpoints have the largest feed-forward activation
    # peak. Two-way token chunking preserves their row-wise quantization math
    # while substantially reducing peak VRAM.
    if "convrot" in model_name.lower():
        if CHUNK_FEED_FORWARD_NODE not in available_nodes:
            raise H3Error(
                "The quality model requires MiniMaxH3ChunkFeedForward, but the "
                "updated Sol-Attn plugin is not loaded. Re-run setup_h3.py."
            )
        chunked = graph.add(
            CHUNK_FEED_FORWARD_NODE,
            model=model_ref,
            enabled=True,
            chunks=2,
            min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
        )
        model_ref = Graph.out(chunked)

    # Keep Spectrum after LoRA, Sol-Attn, and feed-forward patches so actual
    # anchor evaluations observe the final H3 model path. The selected radio
    # mode prevents it from being stacked with either cache implementation.
    if cache_mode_normalized == "spectrum":
        if "SpectrumApplyMiniMaxH3" not in available_nodes:
            raise H3Error(
                "Spectrum was requested, but SpectrumApplyMiniMaxH3 is not "
                "loaded. Re-run setup_h3.py and restart ComfyUI."
            )
        spectrum = graph.add(
            "SpectrumApplyMiniMaxH3",
            model=model_ref,
            **SPECTRUM_DEFAULT_INPUTS,
        )
        model_ref = Graph.out(spectrum)

    if cache_mode_normalized == "easycache":
        if "EasyCache" not in available_nodes:
            raise H3Error(
                "EasyCache was requested, but the native ComfyUI EasyCache node "
                "is not loaded. Update ComfyUI and restart the service."
            )
        if not 0.0 <= float(easycache_threshold) <= 3.0:
            raise H3Error("EasyCache threshold must be between 0 and 3.")
        if not 0.0 <= float(easycache_start) < float(easycache_end) <= 1.0:
            raise H3Error(
                "EasyCache requires 0 ≤ start percent < end percent ≤ 1."
            )
        cache = graph.add(
            "EasyCache",
            model=model_ref,
            reuse_threshold=float(easycache_threshold),
            start_percent=float(easycache_start),
            end_percent=float(easycache_end),
            verbose=bool(easycache_verbose),
        )
        model_ref = Graph.out(cache)

    clip = graph.add(
        "CLIPLoader",
        clip_name=models.text_encoder,
        type="minimax",
        device="default",
    )
    if use_int8_vae:
        if not models.video_vae_int8:
            raise H3Error(
                "INT8 video VAE was requested but is missing from the model catalog. "
                "Re-run setup_h3.py."
            )
        video_vae_name = models.video_vae_int8
    else:
        video_vae_name = models.video_vae
    video_vae = graph.add("VAELoader", vae_name=video_vae_name)
    audio_vae = graph.add("VAELoader", vae_name=models.audio_vae)
    return model_ref, Graph.out(clip), Graph.out(video_vae), Graph.out(audio_vae)


def finish_sampling(
    graph: Graph,
    *,
    model_ref: list[Any],
    conditioning_ref: list[Any],
    latent_ref: list[Any],
    video_vae_ref: list[Any],
    audio_vae_ref: list[Any],
    seed: int,
    steps: int,
    scheduler: str,
    turbo_variant: str | None,
    filename_prefix: str,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    initial_conditioning_ref: list[Any] | None = None,
    initial_latent_ref: list[Any] | None = None,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
) -> None:
    result_format = normalize_result_format(result_format)
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add("BasicGuider", model=model_ref, conditioning=conditioning_ref)
    use_larry_sampler = (
        turbo_variant is not None
        and turbo_uses_custom_nodes(turbo_variant)
    )
    sampler = (
        graph.add(LARRY_TURBO_SAMPLER_NODE)
        if use_larry_sampler
        else graph.add(CORE_SAMPLER_NODE, sampler_name="res_multistep")
    )
    sigmas = graph.add(
        "BasicScheduler",
        model=model_ref,
        scheduler=scheduler,
        steps=int(steps),
        denoise=1.0,
    )
    if latent_upscale_model_name is not None:
        if initial_conditioning_ref is None or initial_latent_ref is None:
            raise H3Error("H3 latent upscaling requires a low-resolution H3 stage.")
        refine_sigmas = graph.add(
            "SplitSigmas",
            sigmas=Graph.out(sigmas),
            step=int(steps) - int(latent_upscale_refine_steps),
        )
        initial_guider = graph.add(
            "BasicGuider",
            model=model_ref,
            conditioning=initial_conditioning_ref,
        )
        initial_sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(initial_guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(sigmas),
            latent_image=initial_latent_ref,
        )
        separated = graph.add(
            H3_SEPARATE_AV_LATENT_NODE,
            latent=Graph.out(initial_sampled),
        )
        upscaled_video = graph.add(
            H3_LATENT_UPSCALER_NODE,
            latent=Graph.out(separated, 0),
            model_name=latent_upscale_model_name,
            scale=H3_LATENT_UPSCALE_SCALE,
            device="cuda",
            precision=latent_upscale_precision,
        )
        combined = graph.add(
            H3_COMBINE_AV_LATENT_NODE,
            video_latent=Graph.out(upscaled_video),
            audio_latent=Graph.out(separated, 1),
        )
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(refine_sigmas, 1),
            latent_image=Graph.out(combined),
        )
        audio_samples = Graph.out(initial_sampled)
    else:
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=Graph.out(noise),
            guider=Graph.out(guider),
            sampler=Graph.out(sampler),
            sigmas=Graph.out(sigmas),
            latent_image=latent_ref,
        )
        audio_samples = Graph.out(sampled)
    if result_format == "Image":
        requested_frames = validate_image_frame_count(image_frames)
        images = graph.add(
            "VAEDecode", samples=Graph.out(sampled), vae=video_vae_ref
        )
        image_ref = Graph.out(images)
        native_frames = image_sampling_length(requested_frames)
        if requested_frames != native_frames:
            selected = graph.add(
                "ImageFromBatch",
                image=image_ref,
                batch_index=0,
                length=requested_frames,
            )
            image_ref = Graph.out(selected)
        graph.add(
            "SaveImage",
            images=image_ref,
            filename_prefix=filename_prefix,
        )
        return

    if result_format == "Audio":
        audio = graph.add(
            "VAEDecodeAudio", samples=audio_samples, vae=audio_vae_ref
        )
        graph.add(
            "SaveAudioMP3",
            audio=Graph.out(audio),
            filename_prefix=filename_prefix,
            quality="V0",
        )
        return

    images = graph.add("VAEDecode", samples=Graph.out(sampled), vae=video_vae_ref)
    audio = graph.add("VAEDecodeAudio", samples=audio_samples, vae=audio_vae_ref)
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(audio),
        fps=24.0,
        bit_depth=8,
    )
    # SaveVideo.codec is a ComfyUI DynamicCombo. API prompts must send the
    # selected option key as a plain string. Sending {"codec": "auto"} causes
    # dynamic schema expansion to discard the input, and execute() then raises
    # "missing 1 required positional argument: codec".
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=filename_prefix,
        format="auto",
        codec="auto",
    )


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
    use_sage: bool = False,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
) -> dict[str, Any]:
    graph = Graph()
    model_ref, clip_ref, video_vae_ref, audio_vae_ref = add_model_stack(
        graph,
        model_name,
        models,
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
        available_nodes=available_nodes,
        use_int8_vae=use_int8_vae,
        use_sage=use_sage,
    )

    target_width = snap32(width)
    target_height = snap32(height)
    source_width, source_height = target_width, target_height
    if latent_upscale_model_name is not None:
        source_width, source_height, target_width, target_height = (
            h3_latent_upscale_dimensions(target_width, target_height)
        )

    inputs: dict[str, Any] = {
        "clip": clip_ref,
        "vae": video_vae_ref,
        "prompt": prompt,
        "length": (
            image_sampling_length(image_frames)
            if normalize_result_format(result_format) == "Image"
            else frame_length(duration)
        ),
    }
    if first_image:
        loaded = graph.add("LoadImage", image=stage_file(first_image, "keyframes"))
        inputs["first_frame"] = Graph.out(loaded)
    if last_image:
        loaded = graph.add("LoadImage", image=stage_file(last_image, "keyframes"))
        inputs["last_frame"] = Graph.out(loaded)

    target_h3 = graph.add(
        "MiniMaxH3ImageToVideo",
        **inputs,
        width=target_width,
        height=target_height,
    )
    initial_h3 = target_h3
    if latent_upscale_model_name is not None:
        initial_h3 = graph.add(
            "MiniMaxH3ImageToVideo",
            **inputs,
            width=source_width,
            height=source_height,
        )
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=Graph.out(target_h3, 0),
        latent_ref=Graph.out(target_h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        filename_prefix=(
            f"h3/image_staging/fl2va_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if normalize_result_format(result_format) == "Image"
            else f"audio/h3_fl2va_{int(time.time())}"
            if normalize_result_format(result_format) == "Audio"
            else f"h3/fl2va_{int(time.time())}"
        ),
        result_format=result_format,
        image_frames=image_frames,
        initial_conditioning_ref=Graph.out(initial_h3, 0),
        initial_latent_ref=Graph.out(initial_h3, 1),
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
    )
    return graph.nodes


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
    use_sage: bool = False,
    latent_upscale_model_name: str | None = None,
    latent_upscale_precision: str = "bf16",
    latent_upscale_refine_steps: int = 2,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
) -> dict[str, Any]:
    graph = Graph()
    model_ref, clip_ref, video_vae_ref, audio_vae_ref = add_model_stack(
        graph,
        model_name,
        models,
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
        available_nodes=available_nodes,
        use_int8_vae=use_int8_vae,
        use_sage=use_sage,
    )

    target_width = snap32(width)
    target_height = snap32(height)
    source_width, source_height = target_width, target_height
    if latent_upscale_model_name is not None:
        source_width, source_height, target_width, target_height = (
            h3_latent_upscale_dimensions(target_width, target_height)
        )

    inputs: dict[str, Any] = {
        "clip": clip_ref,
        "vae": video_vae_ref,
        "audio_vae": audio_vae_ref,
        "prompt": prompt,
        "length": (
            image_sampling_length(image_frames)
            if normalize_result_format(result_format) == "Image"
            else frame_length(duration)
        ),
        "ref_image_size": ref_image_size,
    }

    for index, path in enumerate(reference_images[:MAX_REFERENCE_IMAGES]):
        loaded = graph.add("LoadImage", image=stage_file(path, "reference_images"))
        inputs[f"ref_images.ref_image_{index}"] = Graph.out(loaded)

    for index, path in enumerate(reference_videos[:MAX_REFERENCE_VIDEOS]):
        staged = stage_file(path, "reference_videos", transcode_video=True)
        loaded = graph.add("LoadVideo", file=staged)
        components = graph.add("GetVideoComponents", video=Graph.out(loaded))
        inputs[f"ref_videos.ref_video_{index}"] = Graph.out(components, 0)
        inputs[f"ref_video_audios.ref_video_audio_{index}"] = Graph.out(components, 1)

    for index, path in enumerate(reference_audios[:MAX_REFERENCE_AUDIOS]):
        loaded = graph.add("LoadAudio", audio=stage_file(path, "reference_audios"))
        inputs[f"ref_audios.ref_audio_{index}"] = Graph.out(loaded)

    target_h3 = graph.add(
        "MiniMaxH3ReferenceToVideo",
        **inputs,
        width=target_width,
        height=target_height,
    )
    initial_h3 = target_h3
    if latent_upscale_model_name is not None:
        initial_h3 = graph.add(
            "MiniMaxH3ReferenceToVideo",
            **inputs,
            width=source_width,
            height=source_height,
        )
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=Graph.out(target_h3, 0),
        latent_ref=Graph.out(target_h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        filename_prefix=(
            f"h3/image_staging/ref2va_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            if normalize_result_format(result_format) == "Image"
            else f"audio/h3_ref2va_{int(time.time())}"
            if normalize_result_format(result_format) == "Audio"
            else f"h3/ref2va_{int(time.time())}"
        ),
        result_format=result_format,
        image_frames=image_frames,
        initial_conditioning_ref=Graph.out(initial_h3, 0),
        initial_latent_ref=Graph.out(initial_h3, 1),
        latent_upscale_model_name=latent_upscale_model_name,
        latent_upscale_precision=latent_upscale_precision,
        latent_upscale_refine_steps=latent_upscale_refine_steps,
    )
    return graph.nodes


def ltx25_frame_length(duration: float, fps: float) -> int:
    """LTX-2.5 requires a frame count of 8n+1."""
    return 1 + max(1, int(float(duration) * float(fps)) // 8) * 8


def required_ltx25_nodes(*, image_to_video: bool = False) -> set[str]:
    required = {
        "UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
        "LTXVConditioning", "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent", "RandomNoise", "CFGGuider", "KSamplerSelect",
        "ManualSigmas", "SamplerCustomAdvanced", "LTXVSeparateAVLatent",
        "VAEDecodeTiled", "LTXVAudioVAEDecode", "CreateVideo", "SaveVideo",
    }
    if image_to_video:
        required |= {"LoadImage", "LTXVAddGuide"}
    return required


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
    """Build the official LTX-2.5 distilled T2V or multi-keyframe I2V graph."""
    names = ltx25_model_names(model_choice)
    graph = Graph()
    model = graph.add(
        "UNETLoader", unet_name=names["distilled"], weight_dtype="default"
    )
    clip = graph.add(
        "CLIPLoader", clip_name=names["text_encoder"], type="ltxv", device="default"
    )
    video_vae = graph.add("VAELoader", vae_name=names["video_vae"])
    audio_vae = graph.add("VAELoader", vae_name=names["audio_vae"])
    positive = graph.add("CLIPTextEncode", clip=Graph.out(clip), text=prompt)
    negative = graph.add(
        "CLIPTextEncode", clip=Graph.out(clip), text=negative_prompt
    )
    conditioned = graph.add(
        "LTXVConditioning",
        positive=Graph.out(positive),
        negative=Graph.out(negative),
        frame_rate=float(fps),
    )

    frames = ltx25_frame_length(duration, fps)
    video_latent = graph.add(
        "EmptyLTXVLatentVideo",
        width=int(width),
        height=int(height),
        length=frames,
        batch_size=1,
    )
    video_latent_ref = Graph.out(video_latent)
    positive_ref = Graph.out(conditioned, 0)
    negative_ref = Graph.out(conditioned, 1)
    middle_frame_idx = min(
        frames - 2,
        max(1, int(round(float(middle_time) * float(fps)))),
    )
    keyframes = (
        (first_image, 0, image_strength),
        (middle_image, middle_frame_idx, middle_strength),
        (end_image, -1, end_strength),
    )
    for image, frame_idx, strength in keyframes:
        if not image:
            continue
        loaded = graph.add(
            "LoadImage", image=stage_file(image, "ltx25_keyframes")
        )
        guide = graph.add(
            "LTXVAddGuide",
            positive=positive_ref,
            negative=negative_ref,
            vae=Graph.out(video_vae),
            latent=video_latent_ref,
            image=Graph.out(loaded),
            frame_idx=int(frame_idx),
            strength=float(strength),
        )
        positive_ref = Graph.out(guide, 0)
        negative_ref = Graph.out(guide, 1)
        video_latent_ref = Graph.out(guide, 2)

    audio_latent = graph.add(
        "LTXVEmptyLatentAudio",
        audio_vae=Graph.out(audio_vae),
        frames_number=frames,
        frame_rate=int(round(float(fps))),
        batch_size=1,
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        video_latent=video_latent_ref,
        audio_latent=Graph.out(audio_latent),
    )
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add(
        "CFGGuider",
        model=Graph.out(model),
        positive=positive_ref,
        negative=negative_ref,
        cfg=float(cfg),
    )
    sampler = graph.add("KSamplerSelect", sampler_name=str(sampler_name))
    sigmas = graph.add("ManualSigmas", sigmas=LTX25_SIGMAS)
    sampled = graph.add(
        "SamplerCustomAdvanced",
        noise=Graph.out(noise),
        guider=Graph.out(guider),
        sampler=Graph.out(sampler),
        sigmas=Graph.out(sigmas),
        latent_image=Graph.out(av_latent),
    )
    separated = graph.add("LTXVSeparateAVLatent", av_latent=Graph.out(sampled))
    images = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(separated, 0),
        vae=Graph.out(video_vae),
        tile_size=512,
        overlap=64,
        temporal_size=128,
        temporal_overlap=32,
    )
    audio = graph.add(
        "LTXVAudioVAEDecode",
        samples=Graph.out(separated, 1),
        audio_vae=Graph.out(audio_vae),
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(audio),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"ltx25/generation_{int(time.time())}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_seedvr2_upscale_nodes() -> set[str]:
    return {
        "LoadVideo",
        "GetVideoComponents",
        "ImageScaleBy",
        "SeedVR2Preprocess",
        "VAELoader",
        "UNETLoader",
        "VAEEncodeTiled",
        "SeedVR2TemporalChunk",
        "SeedVR2Conditioning",
        "KSampler",
        "SeedVR2TemporalMerge",
        "VAEDecodeTiled",
        "SeedVR2PostProcessing",
        "CreateVideo",
        "SaveVideo",
    }


def build_seedvr2_upscale_graph(
    *,
    source_video: str,
    seed: int,
    models: ModelConfig,
    model_choice: str = DEFAULT_SEEDVR2_MODEL,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Build ComfyUI's native one-step SeedVR2 2x workflow."""
    assets = seedvr2_upscale_model_names(models, model_choice)
    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    resized = graph.add(
        "ImageScaleBy",
        image=Graph.out(components, 0),
        upscale_method="lanczos",
        scale_by=2.0,
    )
    prepared = graph.add("SeedVR2Preprocess", resized_images=Graph.out(resized))
    vae = graph.add("VAELoader", vae_name=assets["seedvr2_vae"])
    model = graph.add(
        "UNETLoader", unet_name=assets["seedvr2_dit"], weight_dtype="default"
    )
    model_ref = Graph.out(model)
    latent = graph.add(
        "VAEEncodeTiled",
        pixels=Graph.out(prepared),
        vae=Graph.out(vae),
        tile_size=1024,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    chunks = graph.add(
        "SeedVR2TemporalChunk",
        latent=Graph.out(latent),
        temporal_overlap=1,
        # DynamicCombo selections are plain option strings in API prompts;
        # ComfyUI expands this to the mapping consumed by execute().
        chunking_mode="auto",
    )
    conditioning = graph.add(
        "SeedVR2Conditioning",
        model=model_ref,
        vae_conditioning=Graph.out(chunks, 0),
    )
    sampled = graph.add(
        "KSampler",
        model=model_ref,
        seed=int(seed),
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=Graph.out(conditioning, 0),
        negative=Graph.out(conditioning, 1),
        latent_image=Graph.out(chunks, 0),
        denoise=1.0,
    )
    merged = graph.add(
        "SeedVR2TemporalMerge",
        latents=Graph.out(sampled),
        temporal_overlap=Graph.out(chunks, 1),
    )
    decoded = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(merged),
        vae=Graph.out(vae),
        tile_size=1024,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    restored = graph.add(
        "SeedVR2PostProcessing",
        images=Graph.out(decoded),
        original_resized_images=Graph.out(resized),
        color_correction_method="none",
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(restored),
        audio=Graph.out(components, 1),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"seedvr2/upscale_{int(time.time())}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_ltx25_upscale_nodes() -> set[str]:
    return {
        "LoadVideo",
        "GetVideoComponents",
        "ImageScale",
        "GetImageSize",
        "UNETLoader",
        "LTXICLoRALoaderModelOnly",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "LTXVConditioning",
        "EmptyLTXVLatentVideo",
        "LTXAddVideoICLoRAGuide",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "RandomNoise",
        "CFGGuider",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "LTXVCropGuides",
        "VAEDecodeTiled",
        "CreateVideo",
        "SaveVideo",
    }


def build_ltx25_upscale_graph(
    *,
    source_video: str,
    seed: int,
    model_choice: str = DEFAULT_LTX25_MODEL,
    prompt: str = "",
    width: int,
    height: int,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Build Lightricks' single-stage IC-LoRA generative 2x upscaler."""
    names = ltx25_model_names(model_choice)
    base_width = snap32(width)
    base_height = snap32(height)
    target_width = base_width * 2
    target_height = base_height * 2

    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    guide = graph.add(
        "ImageScale",
        image=Graph.out(components, 0),
        upscale_method="lanczos",
        width=base_width,
        height=base_height,
        crop="disabled",
    )
    guide_size = graph.add("GetImageSize", image=Graph.out(guide))

    base_model = graph.add(
        "UNETLoader", unet_name=names["distilled"], weight_dtype="default"
    )
    upscaler = graph.add(
        "LTXICLoRALoaderModelOnly",
        model=Graph.out(base_model),
        lora_name=MODEL_SPECS["ltx25_pixel_upscaler_x2"].local_name,
        strength_model=1.0,
    )
    clip = graph.add(
        "CLIPLoader", clip_name=names["text_encoder"], type="ltxv", device="default"
    )
    video_vae = graph.add("VAELoader", vae_name=names["video_vae"])
    audio_vae = graph.add("VAELoader", vae_name=names["audio_vae"])
    positive = graph.add(
        "CLIPTextEncode",
        clip=Graph.out(clip),
        text=(prompt.strip() or "high quality, detailed video"),
    )
    negative = graph.add("CLIPTextEncode", clip=Graph.out(clip), text="")
    conditioned = graph.add(
        "LTXVConditioning",
        positive=Graph.out(positive),
        negative=Graph.out(negative),
        frame_rate=float(fps),
    )
    video_latent = graph.add(
        "EmptyLTXVLatentVideo",
        width=target_width,
        height=target_height,
        length=Graph.out(guide_size, 2),
        batch_size=1,
    )
    guided = graph.add(
        "LTXAddVideoICLoRAGuide",
        positive=Graph.out(conditioned, 0),
        negative=Graph.out(conditioned, 1),
        vae=Graph.out(video_vae),
        latent=Graph.out(video_latent),
        image=Graph.out(guide),
        frame_idx=0,
        strength=1.0,
        latent_downscale_factor=Graph.out(upscaler, 1),
        crop="disabled",
        use_tiled_encode=True,
        tile_size=512,
        tile_overlap=64,
    )
    audio_latent = graph.add(
        "LTXVEmptyLatentAudio",
        audio_vae=Graph.out(audio_vae),
        frames_number=Graph.out(guide_size, 2),
        frame_rate=int(round(float(fps))),
        batch_size=1,
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        video_latent=Graph.out(guided, 2),
        audio_latent=Graph.out(audio_latent),
    )
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add(
        "CFGGuider",
        model=Graph.out(upscaler),
        positive=Graph.out(guided, 0),
        negative=Graph.out(guided, 1),
        cfg=1.0,
    )
    sampler = graph.add("KSamplerSelect", sampler_name="euler_ancestral")
    sigmas = graph.add("ManualSigmas", sigmas=LTX25_SIGMAS)
    sampled = graph.add(
        "SamplerCustomAdvanced",
        noise=Graph.out(noise),
        guider=Graph.out(guider),
        sampler=Graph.out(sampler),
        sigmas=Graph.out(sigmas),
        latent_image=Graph.out(av_latent),
    )
    separated = graph.add("LTXVSeparateAVLatent", av_latent=Graph.out(sampled))
    cropped = graph.add(
        "LTXVCropGuides",
        positive=Graph.out(guided, 0),
        negative=Graph.out(guided, 1),
        latent=Graph.out(separated, 0),
    )
    images = graph.add(
        "VAEDecodeTiled",
        samples=Graph.out(cropped, 2),
        vae=Graph.out(video_vae),
        tile_size=512,
        overlap=64,
        temporal_size=128,
        temporal_overlap=32,
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(components, 1),
        fps=float(fps),
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"ltx25/upscale_{int(time.time())}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_upscale_nodes(option: str) -> set[str]:
    if option == SEEDVR2_UPSCALE:
        return required_seedvr2_upscale_nodes()
    if option == LTX25_UPSCALE:
        return required_ltx25_upscale_nodes()
    raise H3Error(f"Unknown AI post-processing method: {option}")


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
    fps: float = 24.0,
) -> tuple[dict[str, Any], int]:
    """Build one selected AI-upscale workflow and return its sampler steps."""
    if option == SEEDVR2_UPSCALE:
        return (
            build_seedvr2_upscale_graph(
                source_video=source_video,
                seed=seed,
                models=models,
                model_choice=seedvr2_model,
                fps=fps,
            ),
            1,
        )
    if option == LTX25_UPSCALE:
        if width is None or height is None:
            raise H3Error("LTX-2.5 upscaling requires the source video dimensions.")
        return (
            build_ltx25_upscale_graph(
                source_video=source_video,
                seed=seed,
                model_choice=ltx25_model,
                prompt=prompt,
                width=width,
                height=height,
                fps=fps,
            ),
            8,
        )
    raise H3Error(f"Unknown AI post-processing method: {option}")


def required_music3_nodes(tiled_decode: bool) -> set[str]:
    nodes = {
        "UNETLoader", "CLIPLoader", "VAELoader",
        "MiniMaxMusic3TextEncode", "ConditioningZeroOut",
        "EmptyMiniMaxMusic3LatentAudio", "KSampler",
        "SaveAudioMP3",
    }
    nodes.add("VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio")
    return nodes


def build_music3_graph(
    *,
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
) -> dict[str, Any]:
    """Build the official native ComfyUI MiniMax Music 3 workflow."""
    graph = Graph()
    dit = MODEL_SPECS[MUSIC3_MODEL_CHOICES[str(model_choice)]].local_name
    text_encoder = MODEL_SPECS["music3_text_encoder"].local_name
    vae_name = MODEL_SPECS["music3_vae"].local_name
    model = graph.add("UNETLoader", unet_name=dit, weight_dtype="default")
    clip = graph.add(
        "CLIPLoader", clip_name=text_encoder, type="minimax", device="default"
    )
    vae = graph.add("VAELoader", vae_name=vae_name)
    conditioning = graph.add(
        "MiniMaxMusic3TextEncode",
        clip=Graph.out(clip), caption=str(caption), lyrics=str(lyrics),
        seed=int(seed), max_duration=float(max_duration),
        cfg_scale=float(ar_cfg), top_k=int(top_k),
    )
    negative = graph.add(
        "ConditioningZeroOut", conditioning=Graph.out(conditioning)
    )
    latent = graph.add(
        "EmptyMiniMaxMusic3LatentAudio",
        seconds=Graph.out(conditioning, 1), batch_size=1,
    )
    sampled = graph.add(
        "KSampler",
        model=Graph.out(model), positive=Graph.out(conditioning),
        negative=Graph.out(negative), latent_image=Graph.out(latent),
        seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name="euler", scheduler="simple", denoise=1.0,
    )
    decoder = "VAEDecodeAudioTiled" if tiled_decode else "VAEDecodeAudio"
    decode_inputs: dict[str, Any] = {
        "samples": Graph.out(sampled), "vae": Graph.out(vae),
    }
    if tiled_decode:
        decode_inputs.update(tile_size=1536, overlap=64)
    audio = graph.add(decoder, **decode_inputs)
    graph.add(
        # SaveAudioAdvanced's DynamicCombo currently validates through the API
        # but its normalized execution path drops ``format``. The dedicated
        # MP3 node has a stable flat schema and produces the same V0 output.
        "SaveAudioMP3", audio=Graph.out(audio),
        filename_prefix="audio/minimax_music3",
        quality="V0",
    )
    return graph.nodes


def required_nodes_for(
    mode: str,
    use_sol: bool,
    cache_mode: str,
    use_turbo: bool = False,
    turbo_variant: str = LIGHTX2V_4STEP_TURBO,
    model_filename: str = "",
    use_sage: bool = False,
    latent_upscale: bool = False,
    result_format: str = DEFAULT_RESULT_FORMAT,
) -> set[str]:
    result_format = normalize_result_format(result_format)
    common = {
        "UNETLoader", "CLIPLoader", "VAELoader", "RandomNoise",
        "BasicGuider", "BasicScheduler",
        "SamplerCustomAdvanced",
    }
    if result_format == "Image":
        common |= {"VAEDecode", "ImageFromBatch", "SaveImage"}
    elif result_format == "Audio":
        common |= {"VAEDecodeAudio", "SaveAudioMP3"}
    else:
        common |= {
            "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
        }
    if mode == "Reference media":
        common |= {"MiniMaxH3ReferenceToVideo", "LoadImage", "LoadVideo", "GetVideoComponents", "LoadAudio"}
    else:
        common |= {"MiniMaxH3ImageToVideo", "LoadImage"}
    if use_turbo:
        common |= turbo_required_nodes(turbo_variant, model_filename)
    else:
        common.add(CORE_SAMPLER_NODE)
    if use_sol:
        common.add(SOL_ATTENTION_NODE)
    if use_sage:
        common.add(SAGE_ATTENTION_NODE)
    if latent_upscale:
        common |= {
            "SplitSigmas",
            H3_LATENT_UPSCALER_NODE,
            H3_SEPARATE_AV_LATENT_NODE,
            H3_COMBINE_AV_LATENT_NODE,
        }
    if "convrot" in model_filename.lower():
        common.add(CHUNK_FEED_FORWARD_NODE)
    if str(cache_mode).strip().lower() == "firstblockcache":
        common.add("H3FirstBlockCache")
    elif str(cache_mode).strip().lower() == "spectrum":
        common.add("SpectrumApplyMiniMaxH3")
    elif str(cache_mode).strip().lower() == "easycache":
        common.add("EasyCache")
    return common


def submit_prompt(graph: dict[str, Any], client_id: str) -> str:
    response = api_post("/prompt", json={"prompt": graph, "client_id": client_id})
    payload = response.json()
    if "prompt_id" not in payload:
        raise H3Error(json.dumps(payload, indent=2))
    return str(payload["prompt_id"])


def websocket_url(client_id: str) -> str:
    parsed = urlsplit(COMFY_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/ws"
    return urlunsplit((scheme, parsed.netloc, path, f"clientId={quote(client_id)}", ""))


def graph_class_types(graph: dict[str, Any]) -> set[str]:
    return {
        str(node.get("class_type", ""))
        for node in graph.values()
    }


def node_stage(class_type: str, workflow_classes: set[str] | None = None) -> str:
    """Turn ComfyUI implementation node names into useful user-facing stages."""
    name = str(class_type)
    workflow_classes = workflow_classes or set()
    if name in {
        "UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple",
        "LatentUpscaleModelLoader", CORE_LORA_LOADER_NODE,
        "LTXICLoRALoaderModelOnly", LARRY_TURBO_LORA_NODE,
        LIGHTX2V_BYPASS_LORA_NODE,
    }:
        return "Loading models"
    if name.startswith("Load") or name == "GetVideoComponents":
        return "Preparing reference media"
    if name == "LTXVAddGuide":
        return "Applying LTX-2.5 keyframes"
    if name == "LTXAddVideoICLoRAGuide":
        return "Encoding source video for LTX-2.5 2x upscaling"
    if name == "LTXVCropGuides":
        return "Removing LTX-2.5 reference tokens"
    if name in {"CLIPTextEncode", "LTXVConditioning"}:
        return "Encoding LTX-2.5 prompt"
    if name == "MiniMaxMusic3TextEncode":
        return "Composing song structure and acoustic conditioning"
    if name == "EmptyMiniMaxMusic3LatentAudio":
        return "Preparing Music 3 audio latents"
    if name in {
        "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
    }:
        return "Preparing LTX-2.5 audio-video latents"
    if name == "VAEEncodeTiled":
        if "SeedVR2Preprocess" in workflow_classes:
            return "Encoding H3 video for SeedVR2"
        return "Encoding video"
    if name == "SeedVR2Preprocess":
        return "Preparing SeedVR2 input"
    if name == "SeedVR2TemporalChunk":
        return "Splitting SeedVR2 video into VRAM-safe chunks"
    if name == "SeedVR2Conditioning":
        return "Preparing SeedVR2 conditioning"
    if name == "SeedVR2TemporalMerge":
        return "Merging SeedVR2 chunks"
    if name == "SeedVR2PostProcessing":
        return "Restoring SeedVR2 output"
    if "ImageToVideo" in name or "ReferenceToVideo" in name:
        return "Encoding prompt and conditioning"
    if name in {
        SOL_ATTENTION_NODE, SAGE_ATTENTION_NODE, FUSED_MODULATION_NODE,
        CHUNK_FEED_FORWARD_NODE, "SpectrumApplyMiniMaxH3",
        "H3FirstBlockCache", "EasyCache",
    }:
        return "Configuring generation model"
    if name in {
        "RandomNoise", "BasicGuider", "CFGGuider", CORE_SAMPLER_NODE,
        "BasicScheduler", "ManualSigmas", "SplitSigmas",
    }:
        return "Preparing sampler"
    if name == H3_SEPARATE_AV_LATENT_NODE:
        return "Separating H3 video and audio latents"
    if name == H3_LATENT_UPSCALER_NODE:
        return "Upscaling H3 video latent 2x"
    if name == H3_COMBINE_AV_LATENT_NODE:
        return "Recombining H3 video and audio latents"
    if name == "KSampler" and "MiniMaxMusic3TextEncode" in workflow_classes:
        return "Generating music"
    if name == "SamplerCustomAdvanced" or "Sampler" in name:
        return "Generating video and audio"
    if name in {
        "VAEDecode", "VAEDecodeAudio", "VAEDecodeTiled",
        "LTXVSeparateAVLatent", "LTXVAudioVAEDecode",
    }:
        return "Decoding output"
    if name == "CreateVideo":
        return "Assembling video"
    if name.startswith("Save"):
        return "Saving audio" if "Audio" in name else "Saving video"
    return name


def queue_position(prompt_id: str) -> tuple[str, int | None]:
    """Return ComfyUI's actual queue state and one-based waiting position."""
    try:
        payload = api_get("/queue").json()
        for item in payload.get("queue_running", []):
            if len(item) > 1 and str(item[1]) == prompt_id:
                return "running", None
        for position, item in enumerate(payload.get("queue_pending", []), start=1):
            if len(item) > 1 and str(item[1]) == prompt_id:
                return "queued", position
    except Exception:
        pass
    return "unknown", None


def progress_status(
    stage: str,
    *,
    started: float,
    completed_nodes: int = 0,
    total_nodes: int = 0,
    step: int | None = None,
    step_total: int | None = None,
    configured_steps: int | None = None,
    detail: str | None = None,
) -> str:
    elapsed = time.monotonic() - started
    lines = [f"{stage} · elapsed {elapsed:.1f}s"]
    if step is not None and step_total:
        percent = 100 * step / step_total
        if configured_steps and step_total != configured_steps:
            lines.append(
                f"Overall generation progress {step}/{step_total} ({percent:.0f}%)"
            )
            lines.append(f"Sampling schedule {configured_steps} steps (UI setting)")
        elif configured_steps:
            lines.append(
                f"Sampler step {step}/{configured_steps} ({percent:.0f}%)"
            )
        else:
            lines.append(f"Stage progress {step}/{step_total} ({percent:.0f}%)")
    if total_nodes:
        lines.append(f"Workflow nodes {completed_nodes}/{total_nodes}")
    if detail:
        lines.append(detail)
    return "\n".join(lines)


def stream_comfy_progress(
    ws: websocket.WebSocket,
    prompt_id: str,
    graph: dict[str, Any],
    started: float,
) -> Iterable[tuple[str, int, int, int | None, int | None]]:
    """Yield live node and sampler progress from ComfyUI's websocket."""
    total_nodes = len(graph)
    workflow_classes = graph_class_types(graph)
    completed: set[str] = set()
    current_node: str | None = None
    deadline = time.monotonic() + GENERATION_TIMEOUT
    ws.settimeout(max(0.25, min(POLL_SECONDS, 1.0)))

    while time.monotonic() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            history = api_get(f"/history/{prompt_id}").json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise H3Error(f"ComfyUI execution failed: {status.get('messages', [])}")
                if status.get("completed") or history.get("outputs"):
                    return
            state, position = queue_position(prompt_id)
            if state == "queued":
                yield f"Waiting in queue (position {position})", len(completed), total_nodes, None, None
            elif state == "running" and current_node is None:
                yield "Starting workflow", len(completed), total_nodes, None, None
            continue
        except websocket.WebSocketException:
            yield from poll_comfy_progress(prompt_id, graph)
            return

        # Binary messages are latent previews; progress metadata arrives as JSON.
        if not isinstance(raw, str):
            continue
        if not raw:
            yield from poll_comfy_progress(prompt_id, graph)
            return
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = str(message.get("type", ""))
        data = message.get("data", {})
        event_prompt_id = data.get("prompt_id")
        if event_prompt_id is not None and str(event_prompt_id) != prompt_id:
            continue

        if event_type == "execution_error":
            node = str(data.get("node_type") or data.get("node_id") or "workflow")
            error = data.get("exception_message") or data.get("exception_type") or "unknown error"
            raise H3Error(f"ComfyUI failed in {node}: {error}")
        if event_type == "execution_interrupted":
            raise H3Error("Generation interrupted.")
        if event_type in {"execution_success", "execution_complete"}:
            return
        if event_type == "execution_cached":
            completed.update(str(node) for node in data.get("nodes", []))
            yield "Restoring cached workflow results", len(completed), total_nodes, None, None
            continue
        if event_type == "executed":
            node_id = str(data.get("node", ""))
            if node_id:
                completed.add(node_id)
            continue
        if event_type == "executing":
            node = data.get("node")
            if node is None:
                return
            if current_node and current_node != str(node):
                completed.add(current_node)
            current_node = str(node)
            class_type = graph.get(current_node, {}).get("class_type", "Processing")
            yield node_stage(
                class_type, workflow_classes
            ), len(completed), total_nodes, None, None
            continue
        if event_type == "progress":
            node_id = str(data.get("node") or current_node or "")
            class_type = graph.get(node_id, {}).get("class_type", "Processing")
            value = int(data.get("value", 0))
            maximum = int(data.get("max", 0))
            yield node_stage(
                class_type, workflow_classes
            ), len(completed), total_nodes, value, maximum
    raise H3Error(f"Generation timed out after {GENERATION_TIMEOUT:.0f} seconds")


def poll_comfy_progress(
    prompt_id: str,
    graph: dict[str, Any],
) -> Iterable[tuple[str, int, int, int | None, int | None]]:
    """Compatibility fallback for ComfyUI deployments without `/ws`."""
    deadline = time.monotonic() + GENERATION_TIMEOUT
    while time.monotonic() < deadline:
        payload = api_get(f"/history/{prompt_id}").json()
        item = payload.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise H3Error(f"ComfyUI execution failed: {status.get('messages', [])}")
            if status.get("completed") or item.get("outputs"):
                return
        state, position = queue_position(prompt_id)
        if state == "queued":
            stage = f"Waiting in queue (position {position})"
        elif state == "running":
            stage = "Running workflow (live step events unavailable)"
        else:
            stage = "Waiting for ComfyUI"
        yield stage, 0, len(graph), None, None
        time.sleep(POLL_SECONDS)
    raise H3Error(f"Generation timed out after {GENERATION_TIMEOUT:.0f} seconds")


def wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + GENERATION_TIMEOUT
    while time.monotonic() < deadline:
        payload = api_get(f"/history/{prompt_id}").json()
        item = payload.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise H3Error(f"ComfyUI execution failed: {status.get('messages', [])}")
            if status.get("completed") or item.get("outputs"):
                return item
        time.sleep(POLL_SECONDS)
    raise H3Error(f"Generation timed out after {GENERATION_TIMEOUT:.0f} seconds")


def walk_saved_refs(value: Any) -> Iterable[dict[str, str]]:
    if isinstance(value, dict):
        if "filename" in value:
            yield {
                "filename": str(value["filename"]),
                "subfolder": str(value.get("subfolder", "")),
                "type": str(value.get("type", "output")),
            }
        for child in value.values():
            yield from walk_saved_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_saved_refs(child)


def resolve_output(history: dict[str, Any], queued_at: float) -> Path:
    candidates: list[Path] = []
    output_root = OUTPUT_DIR.resolve()
    for ref in walk_saved_refs(history.get("outputs", {})):
        if ref["type"] != "output":
            continue
        path = (OUTPUT_DIR / ref["subfolder"] / ref["filename"]).resolve()
        if path.is_relative_to(output_root) and path.is_file():
            candidates.append(path)
    videos = [p for p in candidates if p.suffix.lower() in VIDEO_EXTENSIONS]
    if videos:
        return max(videos, key=lambda p: p.stat().st_mtime)

    # Newer SaveVideo UI payloads can be omitted from the public history shape.
    # Fall back only to files created after this job was queued.
    recent = [
        p for p in OUTPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.stat().st_mtime >= queued_at - 2
    ]
    if recent:
        return max(recent, key=lambda p: p.stat().st_mtime)
    raise H3Error("Generation completed, but no saved video could be located.")


def resolve_audio_output(history: dict[str, Any], queued_at: float) -> Path:
    candidates: list[Path] = []
    output_root = OUTPUT_DIR.resolve()
    for ref in walk_saved_refs(history.get("outputs", {})):
        if ref["type"] != "output":
            continue
        path = (OUTPUT_DIR / ref["subfolder"] / ref["filename"]).resolve()
        if (
            path.is_relative_to(output_root)
            and path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
        ):
            candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    recent = [
        path for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.stat().st_mtime >= queued_at - 2
    ]
    if recent:
        return max(recent, key=lambda path: path.stat().st_mtime)
    raise H3Error("Generation completed, but no saved audio could be located.")


def resolve_image_outputs(
    history: dict[str, Any], queued_at: float, expected_count: int
) -> list[Path]:
    expected_count = validate_image_frame_count(expected_count)
    output_root = OUTPUT_DIR.resolve()
    staging_root = (OUTPUT_DIR / "h3" / "image_staging").resolve()
    candidates: list[Path] = []
    for ref in walk_saved_refs(history.get("outputs", {})):
        if ref["type"] != "output":
            continue
        path = (OUTPUT_DIR / ref["subfolder"] / ref["filename"]).resolve()
        if (
            path.is_relative_to(output_root)
            and path.is_relative_to(staging_root)
            and path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            candidates.append(path)
    unique = sorted(set(candidates), key=lambda path: path.name)
    if len(unique) >= expected_count:
        return unique[:expected_count]

    recent = sorted(
        (
            path.resolve()
            for path in staging_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stat().st_mtime >= queued_at - 2
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if len(recent) >= expected_count:
        return recent[-expected_count:]
    raise H3Error(
        "Generation completed, but the decoded image frame batch could not be located."
    )


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
            raise gr.Error("A selected frame is outside the H3 image staging directory.")

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
        shutil.copy2(paths[index], target)
        saved.append(str(target))
    return saved, f"Saved {len(saved)} selected frame(s) to `{destination}`."


def has_encoder(name: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return name in proc.stdout


def postprocess_video(source: Path, option: str) -> Path:
    if option == "None":
        return source
    if option in COMFY_UPSCALE_OPTIONS:
        raise H3Error(f"{option} must run through the ComfyUI workflow.")
    if option != "48 fps interpolation":
        raise H3Error(f"Unsupported post-processing method: {option}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUTS_DIR / f"{source.stem}_{uuid.uuid4().hex[:8]}.mp4"
    filters = ["minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir"]
    encoder = "h264_nvenc" if has_encoder("h264_nvenc") else "libx264"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", encoder, "-preset", "p5", "-cq", "19"]
    else:
        cmd += ["-c:v", encoder, "-preset", "medium", "-crf", "18"]
    cmd += ["-c:a", "aac", "-b:a", "192k", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise H3Error(f"Post-processing failed: {proc.stderr.strip()}")
    return target


def probe_video_metadata(source: Path) -> VideoMetadata:
    """Return dimensions and frame rate needed by AI upscale workflows."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(source),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise H3Error(f"Could not inspect selected video: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
        streams = payload["streams"]
        stream = next(
            item for item in streams if item.get("codec_type", "video") == "video"
        )
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        numerator, denominator = str(rate).split("/", 1)
        fps = float(numerator) / float(denominator)
        if fps <= 0:
            raise ValueError("non-positive frame rate")
        width = int(stream["width"])
        height = int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("non-positive dimensions")
        raw_duration = (
            payload.get("format", {}).get("duration")
            or stream.get("duration")
        )
        duration = float(raw_duration)
        if duration <= 0:
            raise ValueError("non-positive duration")
        raw_frames = stream.get("nb_frames")
        frame_count = int(raw_frames) if str(raw_frames).isdigit() else round(duration * fps)
        if frame_count <= 0:
            raise ValueError("non-positive frame count")
        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            duration=duration,
            frame_count=frame_count,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )
    except (
        KeyError, IndexError, StopIteration, TypeError, ValueError,
        ZeroDivisionError,
    ) as exc:
        raise H3Error("Could not determine the selected video's metadata.") from exc


def prepare_upscale_clip_batch(
    source: Path,
    *,
    category: str,
    split_enabled: bool,
    split_seconds: float,
    metadata: VideoMetadata,
) -> UpscaleClipBatch:
    """Stage one source, or exact-frame LTX-safe source clips, for ComfyUI."""
    if not split_enabled:
        return UpscaleClipBatch((stage_file(str(source), category),))
    seconds = float(split_seconds)
    if not 1.0 <= seconds <= 15.0:
        raise H3Error("Upscale clip length must be between 1 and 15 seconds.")
    if metadata.duration <= 0 or metadata.frame_count <= 0:
        raise H3Error("Could not determine the source duration for split upscaling.")

    # LTX video lengths are 8n+1. Every clip is padded to this exact size;
    # concat_upscaled_clips trims only the final padded tail back off.
    clip_frames = ltx25_frame_length(seconds, metadata.fps)
    clip_duration = clip_frames / metadata.fps
    clip_count = max(1, math.ceil(metadata.frame_count / clip_frames))
    directory = INPUT_DIR / "h3_gradio" / category / f"clips_{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    try:
        for index in range(clip_count):
            path = directory / f"clip_{index + 1:04d}.mp4"
            start = index * clip_frames / metadata.fps
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source), "-ss", f"{start:.9f}",
                "-map", "0:v:0",
            ]
            if metadata.has_audio:
                cmd += ["-map", "0:a:0?", "-af", "apad"]
            cmd += [
                "-vf", (
                    f"fps={metadata.fps:.9f},"
                    f"tpad=stop_mode=clone:stop_duration={clip_duration:.9f}"
                ),
                "-frames:v", str(clip_frames),
                "-t", f"{clip_duration:.9f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
            ]
            if metadata.has_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            cmd += ["-avoid_negative_ts", "make_zero", str(path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0 or not path.is_file():
                raise H3Error(
                    f"Could not create upscale clip {index + 1}/{clip_count}: "
                    f"{proc.stderr.strip()}"
                )
            paths.append(path)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return UpscaleClipBatch(
        tuple(path.relative_to(INPUT_DIR).as_posix() for path in paths),
        tuple(paths),
        directory,
    )


def concat_upscaled_clips(
    source: Path,
    clips: list[Path],
    *,
    option: str,
    duration: float,
    frame_count: int,
) -> Path:
    """Losslessly concatenate upscale video streams and restore source audio."""
    if not clips:
        raise H3Error("No upscaled clips were produced.")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    manifest = OUTPUTS_DIR / f".upscale_concat_{token}.txt"
    target = OUTPUTS_DIR / f"{source.stem}_{token}.mp4"

    def manifest_path(path: Path) -> str:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        return f"file '{escaped}'"

    manifest.write_text(
        "\n".join(manifest_path(path) for path in clips) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-t", f"{float(duration):.9f}",
        "-frames:v", str(int(frame_count)),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not target.is_file():
            target.unlink(missing_ok=True)
            raise H3Error(f"Could not concatenate upscaled clips: {proc.stderr.strip()}")
        return target
    finally:
        manifest.unlink(missing_ok=True)


def cleanup_upscale_clip_batch(
    batch: UpscaleClipBatch | None,
    outputs: Iterable[Path] = (),
) -> None:
    """Remove only the explicitly tracked inputs and intermediate outputs."""
    for output in outputs:
        output.unlink(missing_ok=True)
    if batch is None:
        return
    for path in batch.temporary_inputs:
        path.unlink(missing_ok=True)
    if batch.temporary_directory is not None:
        try:
            batch.temporary_directory.rmdir()
        except OSError:
            pass


def unload_comfy_models() -> None:
    """Explicitly clear model residency only when the user opts into it."""
    api_post("/free", json={"unload_models": True, "free_memory": True})


def unload_all_models() -> tuple[str, str]:
    """Unload every resident ComfyUI model and refresh the backend summary."""
    local_was_loaded = unload_prompt_rewriter()
    try:
        unload_comfy_models()
        local_note = " Local 8B prompt writer unloaded." if local_was_loaded else ""
        return f"All models unloaded and cached VRAM released.{local_note}", backend_status()
    except Exception as exc:
        local_note = " Local 8B prompt writer was unloaded." if local_was_loaded else ""
        return f"ComfyUI VRAM release failed: {exc}.{local_note}", backend_status()


def video_download_path(video: str | Path) -> str:
    """Return a safe, public route for a generated video on this server."""
    resolved = managed_video_path(video, require_file=False)
    for bucket, root in (
        ("comfy", OUTPUT_DIR.resolve()),
        ("gradio", OUTPUTS_DIR.resolve()),
    ):
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            return f"/downloads/{bucket}/{quote(relative, safe='/')}"
    raise H3Error("Generated video is outside the configured output directories.")


def managed_video_path(
    video: str | Path,
    *,
    require_file: bool = True,
) -> Path:
    """Resolve a video only when it belongs to a managed output directory."""
    resolved = Path(video).resolve()
    roots = (OUTPUT_DIR.resolve(), OUTPUTS_DIR.resolve())
    if (
        resolved.suffix.lower() not in VIDEO_EXTENSIONS
        or not any(resolved.is_relative_to(root) for root in roots)
        or (require_file and not resolved.is_file())
    ):
        raise H3Error("Video is not a managed generated output.")
    return resolved


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
    """Return generated videos, optionally limited to the newest entries."""
    videos: dict[Path, Path] = {}
    for root in (OUTPUT_DIR, OUTPUTS_DIR):
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        for candidate in root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                resolved = candidate.resolve()
                if resolved.is_relative_to(resolved_root):
                    videos[resolved] = candidate
            except OSError:
                continue

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    ordered = sorted(videos.values(), key=modified, reverse=True)
    return ordered if limit is None else ordered[:max(0, int(limit))]


def gallery_thumbnail(video: Path) -> Path | None:
    """Create a small cached poster image for a video."""
    temporary: Path | None = None
    try:
        video_mtime = video.stat().st_mtime
        thumbnail = gallery_thumbnail_path(video)
        cache_key = thumbnail.stem
        if thumbnail.is_file() and thumbnail.stat().st_mtime >= video_mtime:
            return thumbnail

        GALLERY_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        temporary = GALLERY_THUMBNAILS_DIR / f"{cache_key}.tmp.jpg"
        temporary.unlink(missing_ok=True)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", "0.1", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=480:-2:force_original_aspect_ratio=decrease",
            str(temporary),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(thumbnail)
        return thumbnail
    except (OSError, subprocess.TimeoutExpired):
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return None


def gallery_thumbnail_path(video: str | Path) -> Path:
    cache_key = hashlib.sha256(
        str(Path(video).resolve()).encode("utf-8")
    ).hexdigest()[:24]
    return GALLERY_THUMBNAILS_DIR / f"{cache_key}.jpg"


def gallery_video_resolution(video: Path) -> tuple[int, int] | None:
    """Return cached source dimensions without decoding the full video."""
    try:
        resolved = video.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    if cache_key in _GALLERY_RESOLUTION_CACHE:
        return _GALLERY_RESOLUTION_CACHE[cache_key]

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(resolved),
    ]
    resolution: tuple[int, int] | None = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            stream = json.loads(proc.stdout)["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            if width > 0 and height > 0:
                resolution = (width, height)
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ):
        pass

    if len(_GALLERY_RESOLUTION_CACHE) >= GALLERY_METADATA_CACHE_LIMIT:
        _GALLERY_RESOLUTION_CACHE.clear()
    _GALLERY_RESOLUTION_CACHE[cache_key] = resolution
    return resolution


def gallery_resolution_text(video: Path) -> str:
    resolution = gallery_video_resolution(video)
    return f"{resolution[0]}×{resolution[1]}" if resolution else "resolution unavailable"


def generated_video_family(video: str | Path) -> str:
    """Identify direct ComfyUI outputs while keeping all managed videos eligible."""
    resolved = Path(video).resolve()
    output_root = OUTPUT_DIR.resolve()
    if resolved.is_relative_to(output_root):
        relative = resolved.relative_to(output_root)
        top_level = relative.parts[0].lower() if relative.parts else ""
        if top_level == "h3":
            return "MiniMax H3"
        if top_level == "ltx25":
            return "LTX-2.5"
    return "Post-processed"


def forget_gallery_metadata(video: str | Path | None = None) -> None:
    if video is None:
        _GALLERY_RESOLUTION_CACHE.clear()
        return
    resolved = str(Path(video).resolve())
    for key in [key for key in _GALLERY_RESOLUTION_CACHE if key[0] == resolved]:
        _GALLERY_RESOLUTION_CACHE.pop(key, None)


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
    return video, f"**Resolution:** {resolution} · [Download video]({download_url})", video


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

        if option not in COMFY_UPSCALE_OPTIONS:
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
            model_status = f"LTX-2.5 {ltx25_model} and 2x IC-LoRA"
            yield gallery_progress_result(f"Checking {model_status} models")
            downloaded = ensure_ltx25_upscale_models(ltx25_model)
            stage_bucket = "ltx25_upscale"

        if downloaded:
            yield gallery_progress_result(f"{option} models downloaded")
        metadata = probe_video_metadata(source)
        use_split = option == LTX25_UPSCALE and bool(split_upscale)
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
                fps=metadata.fps,
            )

        client_id = str(uuid.uuid4())
        try:
            ws = websocket.create_connection(
                websocket_url(client_id),
                timeout=max(1.0, min(REQUEST_TIMEOUT, 10.0)),
            )
        except Exception:
            ws = None
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
                        stage = f"Upscaling with {option}"
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
                f"Concatenating {clip_count} upscaled clips and restoring source audio"
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
                stage = f"Upscaling with {option}"
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
        if isinstance(vram_total, (int, float)) and isinstance(
            vram_free, (int, float)
        ):
            vram_text = (
                f" · {vram_free / 2**30:.1f}/{vram_total / 2**30:.1f} GiB VRAM free"
            )
        elif isinstance(vram_total, (int, float)):
            vram_text = f" · {vram_total / 2**30:.1f} GiB VRAM"
        else:
            vram_text = ""
        models = load_model_config()
        easycache_status = "available" if "EasyCache" in live_nodes else "unavailable"
        fbcache_status = "available" if "H3FirstBlockCache" in live_nodes else "unavailable"
        spectrum_status = "available" if "SpectrumApplyMiniMaxH3" in live_nodes else "unavailable"
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
                f"**LightX2V Turbo / 4-step** · FL2VA v1.0 `{models.turbo_lora}` · "
                f"Ref2VA v0.1 544p `{models.turbo_ref_lora}` · strength 1.0"
            )
        if models.turbo_8step_lora:
            profile_lines.append(
                f"**LightX2V Turbo v1.0 / 8-step 544p** · LoRA `{models.turbo_8step_lora}` · "
                "8-step default · strength 1.0 · FL2VA and experimental Ref2VA"
            )
        return (
            f"Connected · {gpu}{vram_text} · sparse: {SERVER_ATTENTION_BACKEND} · "
            f"dense: {SERVER_DENSE_ATTENTION_BACKEND} · "
            f"memory: {SERVER_MEMORY_PROFILE} · FirstBlockCache: {fbcache_status} · "
            f"EasyCache: {easycache_status}  \n"
            + "  \n".join(profile_lines)
        )
    except Exception as exc:
        return f"Backend unavailable: {exc}"


def generate(
    mode: str,
    model_profile: str,
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
    latent_upscale: bool = False,
    latent_upscaler_model: str = DEFAULT_H3_LATENT_UPSCALER_MODEL,
    latent_upscale_refine_steps: int = 2,
    upscale_force_offload: bool = False,
    upscale_split_enabled: bool = False,
    upscale_split_seconds: float = 5.0,
    seedvr2_model: str = DEFAULT_SEEDVR2_MODEL,
    ltx25_model: str = DEFAULT_LTX25_MODEL,
    use_int8_vae: bool = False,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    progress=gr.Progress(track_tqdm=False),
):
    started = time.monotonic()
    queued_at = time.time()
    ws: websocket.WebSocket | None = None
    fallback_video: Path | None = None
    clip_batch: UpscaleClipBatch | None = None
    clip_outputs: list[Path] = []
    try:
        unload_prompt_rewriter()
        progress(0, desc="Validating request")
        yield None, progress_status("Validating request", started=started)
        result_format = normalize_result_format(result_format)
        requested_image_frames = validate_image_frame_count(image_frames)
        if postprocess not in GENERATION_POSTPROCESS_OPTIONS:
            raise H3Error("Unsupported post-processing method.")
        if not prompt.strip():
            raise H3Error("Prompt is required.")
        if result_format != "Image" and not 2 <= float(duration) <= 15:
            raise H3Error("Duration must be between 2 and 15 seconds.")
        if result_format != "Video":
            postprocess = "None"
        if result_format == "Audio":
            latent_upscale = False
        generation_frames = (
            image_sampling_length(requested_image_frames)
            if result_format == "Image"
            else frame_length(duration)
        )
        policy_duration = generation_frames / 24.0
        resolved_width, resolved_height = generation_resolution(
            width,
            height,
            result_format=result_format,
            latent_upscale=bool(latent_upscale),
            mode=mode,
            first_image=first_image,
        )
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        models = load_model_config()
        if use_int8_vae:
            progress(0, desc="Preparing INT8 video VAE")
            ensure_int8_video_vae(models)
        if postprocess == SEEDVR2_UPSCALE:
            seedvr2_upscale_model_names(models, seedvr2_model)
        elif postprocess == LTX25_UPSCALE:
            ltx25_model_keys(ltx25_model)
        profile_key = models.profile_key(model_profile)
        profile = models.profiles[profile_key]

        selected_model = (
            profile.ref2va if mode == "Reference media" else profile.fl2va
        )
        selected_path = COMFY_DIR / "models" / "diffusion_models" / selected_model
        if not model_file_is_ready(selected_path):
            progress(0, desc=f"Downloading {profile.label} model")
            yield None, progress_status(
                f"Downloading {profile.label} model on demand",
                started=started,
            )
        ensure_profile_model(profile_key, profile, mode)

        requested_generation = str(generation_mode).strip().lower()
        use_turbo = requested_generation == "turbo"
        selected_turbo = normalize_turbo_variant(turbo_variant)
        generation_note = (
            f"Reference Turbo is experimental and currently uses the "
            f"FL2VA-trained {selected_turbo} LoRA."
            if mode == "Reference media" and use_turbo
            else None
        )

        if use_turbo:
            turbo_lora_name = models.turbo_lora_for(mode, selected_turbo)
            if not turbo_lora_name:
                raise H3Error(
                    f"{selected_turbo} Turbo LoRA is not provisioned. "
                    "Re-run setup/provisioning."
                )
            if ensure_turbo_lora(models, selected_turbo, mode):
                progress(0, desc=f"Downloaded {selected_turbo} Turbo LoRA")
            selected_label = f"{profile.label} · Turbo · {selected_turbo}"
            turbo_strength = turbo_strength_for(selected_turbo)
        else:
            selected_label = f"{profile.label} · Normal"
            turbo_lora_name = None
            turbo_strength = 1.0
        selected_label += (
            " · INT8 ConvRot VAE" if use_int8_vae else " · FP16 VAE"
        )

        # Variant defaults update outside the generation queue, while this
        # request deliberately honors any subsequent manual step adjustment.
        effective_steps = int(steps)
        effective_scheduler = str(scheduler)

        if use_turbo:
            if effective_steps < 4:
                raise H3Error("Turbo requires at least 4 steps.")
        elif effective_steps < 10:
            raise H3Error(
                "Normal H3 generation requires at least 10 steps. "
                "Use Generation=Turbo for lower-step generation."
            )

        latent_upscale_model_name: str | None = None
        latent_upscale_precision = "bf16"
        latent_source_width, latent_source_height = resolved_width, resolved_height
        if latent_upscale:
            (
                latent_source_width,
                latent_source_height,
                resolved_width,
                resolved_height,
            ) = h3_latent_upscale_dimensions(resolved_width, resolved_height)
            refine_steps = int(latent_upscale_refine_steps)
            if refine_steps < 1 or refine_steps >= effective_steps:
                raise H3Error(
                    "H3 latent upscale refinement steps must be at least 1 and smaller "
                    f"than the total {effective_steps} sampling steps."
                )
            (
                _latent_upscale_key,
                latent_upscale_model_name,
                latent_upscale_precision,
            ) = h3_latent_upscaler_settings(latent_upscaler_model)
            destination = (
                COMFY_DIR
                / "models"
                / "latent_upscale_models"
                / latent_upscale_model_name
            )
            if not model_file_is_ready(destination):
                progress(0, desc="Downloading H3 latent upscaler")
                yield None, progress_status(
                    f"Downloading {latent_upscaler_model} H3 latent upscaler",
                    started=started,
                )
            ensure_h3_latent_upscaler_model(latent_upscaler_model)

        info = object_info()
        available = set(info)
        if postprocess in COMFY_UPSCALE_OPTIONS:
            missing_upscale_nodes = required_upscale_nodes(postprocess) - available
            if missing_upscale_nodes:
                raise H3Error(
                    f"{postprocess} requires current ComfyUI nodes: "
                    + ", ".join(sorted(missing_upscale_nodes))
                )

        effective_sol, packed_tokens, sol_reason = resolve_sol_policy(
            attention_mode, mode, resolved_width, resolved_height, policy_duration,
            first_image, last_image, use_turbo=use_turbo,
        )
        effective_sage = str(attention_mode).strip().lower() in {
            "sage", "sage 2", "sage2"
        }
        effective_cache_mode, cache_note = resolve_cache_policy(
            cache_mode, use_turbo=use_turbo
        )
        if latent_upscale and effective_cache_mode.lower() != "off":
            cache_note = (
                f"{effective_cache_mode} was disabled because cache state is not "
                "safe across the low-resolution generation and high-resolution "
                "refinement samplers."
            )
            effective_cache_mode = "Off"
        missing = required_nodes_for(
            mode,
            effective_sol,
            effective_cache_mode,
            use_sage=effective_sage,
            use_turbo=use_turbo,
            turbo_variant=selected_turbo,
            model_filename=selected_model,
            latent_upscale=bool(latent_upscale),
            result_format=result_format,
        ) - available
        if missing:
            raise H3Error("Missing ComfyUI nodes: " + ", ".join(sorted(missing)))

        refs_i = collect_reference_slots(
            ref_image_1, ref_image_2, ref_image_3,
            ref_image_4, ref_image_5, ref_image_6,
            ref_image_7, ref_image_8, ref_image_9,
        )
        refs_v = collect_reference_slots(ref_video_1, ref_video_2, ref_video_3)
        refs_a = collect_reference_slots(ref_audio_1, ref_audio_2, ref_audio_3)

        if mode == "Text to video":
            first_image = None
            last_image = None
        elif mode == "First / last frame":
            if not first_image and not last_image:
                raise H3Error("Provide a first frame, a last frame, or both.")
        elif mode == "Reference media":
            if not (refs_i or refs_v or refs_a):
                raise H3Error("Reference mode requires at least one image, video, or audio file.")

        progress(0, desc="Building ComfyUI workflow")
        yield None, progress_status(
            "Preparing inputs and building workflow", started=started
        )
        if mode == "Reference media":
            graph = build_ref2va_graph(
                prompt=prompt,
                reference_images=refs_i,
                reference_videos=refs_v,
                reference_audios=refs_a,
                width=resolved_width, height=resolved_height, duration=float(duration),
                result_format=result_format, image_frames=requested_image_frames,
                steps=effective_steps, seed=actual_seed, scheduler=effective_scheduler,
                ref_image_size=ref_image_size,
                turbo_lora_name=turbo_lora_name, turbo_variant=selected_turbo,
                turbo_strength=turbo_strength,
                use_sol=effective_sol, sol_tau=float(sol_tau),
                use_sage=effective_sage,
                sol_thresh_type=sol_thresh_type,
                sol_exact_mode=sol_exact_mode,
                sol_dense_steps=int(sol_dense_steps),
                sol_step_off=float(sol_step_off),
                sol_sink_tokens=int(sol_sink_tokens),
                cache_mode=effective_cache_mode,
                fbcache_preset=str(fbcache_preset),
                fbcache_threshold=float(fbcache_threshold),
                fbcache_start=float(fbcache_start),
                fbcache_end=float(fbcache_end),
                fbcache_max_hits=int(fbcache_max_hits),
                fbcache_temporal_guard=bool(fbcache_temporal_guard),
                easycache_threshold=float(easycache_threshold),
                easycache_start=float(easycache_start),
                easycache_end=float(easycache_end),
                easycache_verbose=bool(easycache_verbose),
                model_name=selected_model,
                models=models, available_nodes=available,
                use_int8_vae=bool(use_int8_vae),
                latent_upscale_model_name=latent_upscale_model_name,
                latent_upscale_precision=latent_upscale_precision,
                latent_upscale_refine_steps=int(latent_upscale_refine_steps),
            )
        else:
            graph = build_fl2va_graph(
                prompt=prompt, first_image=first_image, last_image=last_image,
                width=resolved_width, height=resolved_height, duration=float(duration),
                result_format=result_format, image_frames=requested_image_frames,
                steps=effective_steps, seed=actual_seed, scheduler=effective_scheduler,
                turbo_lora_name=turbo_lora_name, turbo_variant=selected_turbo,
                turbo_strength=turbo_strength,
                use_sol=effective_sol, sol_tau=float(sol_tau),
                use_sage=effective_sage,
                sol_thresh_type=sol_thresh_type,
                sol_exact_mode=sol_exact_mode,
                sol_dense_steps=int(sol_dense_steps),
                sol_step_off=float(sol_step_off),
                sol_sink_tokens=int(sol_sink_tokens),
                cache_mode=effective_cache_mode,
                fbcache_preset=str(fbcache_preset),
                fbcache_threshold=float(fbcache_threshold),
                fbcache_start=float(fbcache_start),
                fbcache_end=float(fbcache_end),
                fbcache_max_hits=int(fbcache_max_hits),
                fbcache_temporal_guard=bool(fbcache_temporal_guard),
                easycache_threshold=float(easycache_threshold),
                easycache_start=float(easycache_start),
                easycache_end=float(easycache_end),
                easycache_verbose=bool(easycache_verbose),
                model_name=selected_model,
                models=models, available_nodes=available,
                use_int8_vae=bool(use_int8_vae),
                latent_upscale_model_name=latent_upscale_model_name,
                latent_upscale_precision=latent_upscale_precision,
                latent_upscale_refine_steps=int(latent_upscale_refine_steps),
            )

        client_id = str(uuid.uuid4())
        websocket_note = None
        try:
            ws = websocket.create_connection(
                websocket_url(client_id),
                timeout=max(1.0, min(REQUEST_TIMEOUT, 10.0)),
            )
        except Exception as exc:
            websocket_note = (
                "Live node/step events unavailable; using queue polling "
                f"({type(exc).__name__})."
            )
        prompt_id = submit_prompt(graph, client_id)
        sol_status = (
            f"zero-copy on ({sol_thresh_type}, τ={float(sol_tau):.1f}, "
            f"{sol_exact_mode}, dense-tail-blocks={int(sol_dense_steps)})"
            if effective_sol else "off"
        )
        if effective_cache_mode.lower() == "firstblockcache":
            cache_status = (
                f"FirstBlockCache {fbcache_preset} "
                f"(threshold={float(fbcache_threshold):.3f}, "
                f"window={float(fbcache_start):.2f}–{float(fbcache_end):.2f}, "
                f"max-hits={int(fbcache_max_hits)}, "
                f"temporal-guard={'on' if fbcache_temporal_guard else 'off'})"
            )
        elif effective_cache_mode.lower() == "spectrum":
            cache_status = (
                "Spectrum (degree=1, audio-isolated offline replay, "
                "audio-blend=0, history/archive=system RAM)"
            )
        elif effective_cache_mode.lower() == "easycache":
            cache_status = (
                f"EasyCache (threshold={float(easycache_threshold):.2f}, "
                f"{float(easycache_start):.2f}–{float(easycache_end):.2f})"
            )
        else:
            cache_status = "off"
        queued_status = (
            f"Queued `{prompt_id}` · seed {actual_seed} · "
            f"{resolved_width}×{resolved_height} · {generation_frames} sampled frames · "
            f"result {result_format.lower()}"
            + (
                f" ({requested_image_frames} decoded frame(s)) · "
                if result_format == "Image"
                else " · "
            )
            + f"model {selected_label} · {effective_steps} steps/{effective_scheduler} · "
            f"attention {sol_status} ({sol_reason}; ~{packed_tokens:,} target tokens) · "
            f"dense-backend {SERVER_DENSE_ATTENTION_BACKEND} · "
            f"cache {cache_status}"
        )
        if latent_upscale:
            queued_status += (
                f"\n\nH3 latent upscale: {latent_upscaler_model} · "
                f"full {effective_steps}-step generation at "
                f"{latent_source_width}×{latent_source_height} → "
                f"{resolved_width}×{resolved_height} · "
                f"{int(latent_upscale_refine_steps)} low-denoise refinement steps."
            )
        if cache_note:
            queued_status += f"\n\nAcceleration notice: {cache_note}"
        if generation_note:
            queued_status += f"\n\nGeneration notice: {generation_note}"
        if websocket_note:
            queued_status += f"\n\nProgress notice: {websocket_note}"
        progress(0, desc="Queued in ComfyUI")
        yield None, queued_status

        updates = (
            stream_comfy_progress(ws, prompt_id, graph, started)
            if ws is not None
            else poll_comfy_progress(prompt_id, graph)
        )
        for stage, completed_nodes, total_nodes, step, step_total in updates:
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total_nodes:
                progress((completed_nodes, total_nodes), desc=stage)
            yield None, progress_status(
                stage,
                started=started,
                completed_nodes=completed_nodes,
                total_nodes=total_nodes,
                step=step,
                step_total=step_total,
                configured_steps=(
                    effective_steps
                    if stage == "Generating video and audio"
                    else None
                ),
                detail=f"Job `{prompt_id}`",
            )

        progress(1, desc="Resolving generated output")
        yield None, progress_status(
            "Generation complete; locating output",
            started=started,
            completed_nodes=len(graph),
            total_nodes=len(graph),
        )
        history = wait_for_history(prompt_id)
        if result_format == "Image":
            results = resolve_image_outputs(
                history, queued_at, requested_image_frames
            )
            elapsed = time.monotonic() - started
            progress(1, desc="Complete")
            yield [str(path) for path in results], (
                f"Completed in {elapsed:.1f}s · {len(results)} image frame(s) ready "
                f"for selection · seed {actual_seed}"
            )
            return
        if result_format == "Audio":
            result = resolve_audio_output(history, queued_at)
            elapsed = time.monotonic() - started
            progress(1, desc="Complete")
            yield str(result), (
                f"Completed in {elapsed:.1f}s · audio {result.name} · "
                f"seed {actual_seed} · "
                f"{elapsed / float(duration):.1f}s compute per output second"
            )
            return
        source = resolve_output(history, queued_at)
        fallback_video = source
        if postprocess != "None":
            progress(0, desc="Post-processing video")
            yield None, progress_status(
                f"Post-processing: {postprocess}", started=started
            )
        if postprocess in COMFY_UPSCALE_OPTIONS:
            if postprocess == SEEDVR2_UPSCALE:
                model_status = f"SeedVR2 {seedvr2_model}"
                stage_bucket = "seedvr2_upscale"
                yield None, progress_status(
                    f"Checking {model_status} models", started=started
                )
                downloaded = ensure_seedvr2_upscale_models(models, seedvr2_model)
            else:
                model_status = f"LTX-2.5 {ltx25_model} and 2x IC-LoRA"
                stage_bucket = "ltx25_upscale"
                yield None, progress_status(
                    f"Checking {model_status} models", started=started
                )
                downloaded = ensure_ltx25_upscale_models(ltx25_model)

            if downloaded:
                yield None, progress_status(f"{postprocess} models downloaded", started=started)
            metadata = probe_video_metadata(source)
            use_split = postprocess == LTX25_UPSCALE and bool(upscale_split_enabled)
            clip_batch = prepare_upscale_clip_batch(
                source,
                category=stage_bucket,
                split_enabled=use_split,
                split_seconds=upscale_split_seconds,
                metadata=metadata,
            )
            clip_count = len(clip_batch.sources)
            if use_split:
                yield None, progress_status(
                    f"Split source into {clip_count} LTX-safe clips",
                    started=started,
                    detail=f"Target clip length {float(upscale_split_seconds):g}s",
                )
            staged_source = clip_batch.sources[0]
            unload_h3_before_upscale = bool(upscale_force_offload)
            if unload_h3_before_upscale:
                progress(0, desc="Unloading H3 models")
                yield None, progress_status(
                    f"Unloading H3 models before {postprocess}", started=started
                )
                unload_comfy_models()

            upscale_graph, configured_upscale_steps = build_upscale_graph(
                option=postprocess,
                source_video=staged_source,
                seed=actual_seed,
                models=models,
                seedvr2_model=seedvr2_model,
                ltx25_model=ltx25_model,
                prompt=prompt,
                width=resolved_width,
                height=resolved_height,
                fps=metadata.fps,
            )

            upscale_queued_at = time.time()
            upscale_prompt_id = submit_prompt(upscale_graph, client_id)
            unload_note = (
                "H3 unload enabled"
                if unload_h3_before_upscale
                else "automatic residency"
            )
            yield None, progress_status(
                f"{postprocess} queued", started=started,
                detail=(f"Job `{upscale_prompt_id}` · {resolved_width * 2}×"
                        f"{resolved_height * 2} · {unload_note}"),
            )
            upscale_updates = (
                stream_comfy_progress(ws, upscale_prompt_id, upscale_graph, started)
                if ws is not None else poll_comfy_progress(upscale_prompt_id, upscale_graph)
            )
            for stage, completed_nodes, total_nodes, step, step_total in upscale_updates:
                if stage == "Generating video and audio":
                    stage = f"Upscaling with {postprocess}"
                if step is not None and step_total:
                    progress((step, step_total), desc=stage)
                elif total_nodes:
                    progress((completed_nodes, total_nodes), desc=stage)
                yield None, progress_status(
                    stage, started=started, completed_nodes=completed_nodes,
                    total_nodes=total_nodes, step=step, step_total=step_total,
                    configured_steps=configured_upscale_steps if step is not None else None,
                    detail=f"Upscale job `{upscale_prompt_id}`",
                )
            upscale_history = wait_for_history(upscale_prompt_id)
            result = resolve_output(upscale_history, upscale_queued_at)
            clip_outputs.append(result)
            for clip_index, staged_source in enumerate(
                clip_batch.sources[1:], start=1
            ):
                clip_seed = (actual_seed + clip_index) % (2**63 - 1)
                upscale_graph, configured_upscale_steps = build_upscale_graph(
                    option=postprocess,
                    source_video=staged_source,
                    seed=clip_seed,
                    models=models,
                    seedvr2_model=seedvr2_model,
                    ltx25_model=ltx25_model,
                    prompt=prompt,
                    width=resolved_width,
                    height=resolved_height,
                    fps=metadata.fps,
                )
                upscale_queued_at = time.time()
                upscale_prompt_id = submit_prompt(upscale_graph, client_id)
                clip_label = f"Clip {clip_index + 1}/{clip_count}"
                yield None, progress_status(
                    f"{postprocess} queued",
                    started=started,
                    detail=(
                        f"{clip_label} 路 job `{upscale_prompt_id}` 路 "
                        f"seed {clip_seed} 路 {unload_note}"
                    ),
                )
                upscale_updates = (
                    stream_comfy_progress(
                        ws, upscale_prompt_id, upscale_graph, started
                    )
                    if ws is not None
                    else poll_comfy_progress(upscale_prompt_id, upscale_graph)
                )
                for stage, completed_nodes, total_nodes, step, step_total in upscale_updates:
                    if stage == "Generating video and audio":
                        stage = f"Upscaling with {postprocess}"
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
                    yield None, progress_status(
                        f"{clip_label}: {stage}",
                        started=started,
                        completed_nodes=completed_nodes,
                        total_nodes=total_nodes,
                        step=step,
                        step_total=step_total,
                        configured_steps=(
                            configured_upscale_steps if step is not None else None
                        ),
                        detail=f"Upscale job `{upscale_prompt_id}`",
                    )
                clip_outputs.append(
                    resolve_output(
                        wait_for_history(upscale_prompt_id), upscale_queued_at
                    )
                )
            if use_split:
                yield None, progress_status(
                    "Concatenating upscaled clips and restoring source audio",
                    started=started,
                    detail=f"{clip_count} clips",
                )
                result = concat_upscaled_clips(
                    source,
                    clip_outputs,
                    option=postprocess,
                    duration=metadata.duration,
                    frame_count=metadata.frame_count,
                )
        else:
            result = postprocess_video(source, postprocess)
        elapsed = time.monotonic() - started
        progress(1, desc="Complete")
        yield str(result), (
            f"Completed in {elapsed:.1f}s · output {result.name} · seed {actual_seed} · "
            f"{elapsed / float(duration):.1f}s compute per output second"
        )
    except Exception as exc:
        fallback = str(fallback_video) if fallback_video is not None else None
        suffix = " The completed H3 video is still available." if fallback else ""
        yield fallback, f"Error: {exc}{suffix}"
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
    """Run an LTX-2.5 job through the same ComfyUI queue as H3."""
    started = time.monotonic()
    queued_at = time.time()
    ws: websocket.WebSocket | None = None
    try:
        unload_prompt_rewriter()
        progress(0, desc="Validating LTX-2.5 request")
        yield None, progress_status("Validating LTX-2.5 request", started=started)
        if not str(prompt).strip():
            raise H3Error("Prompt is required.")
        if not 1 <= float(duration) <= 20:
            raise H3Error("LTX-2.5 duration must be between 1 and 20 seconds.")
        if not 1 <= float(fps) <= 60:
            raise H3Error("Frame rate must be between 1 and 60 fps.")
        resolved_width, resolved_height = validate_resolution(width, height)
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        image_to_video = str(mode).strip().lower() == "image to video"
        if image_to_video and not first_image:
            raise H3Error("Image-to-video mode requires a start frame.")
        keyframe_strengths = {
            "Start image": image_strength,
            "Middle image": middle_strength,
            "End image": end_strength,
        }
        for label, strength in keyframe_strengths.items():
            if not 0 <= float(strength) <= 1:
                raise H3Error(f"{label} strength must be between 0 and 1.")
        if (
            image_to_video
            and middle_image
            and not 0 < float(middle_time) < float(duration)
        ):
            raise H3Error("Middle keyframe time must be inside the video duration.")

        missing_files = missing_ltx25_model_names(model_choice)
        if missing_files:
            progress(0, desc="Downloading LTX-2.5 models")
            yield None, progress_status(
                "Downloading gated LTX-2.5 models on demand", started=started,
                detail=(
                    "Accept the Hugging Face model license and authenticate "
                    "with `hf auth login` or HF_TOKEN if required."
                ),
            )
        ensure_ltx25_models(model_choice)

        available = set(object_info())
        missing_nodes = required_ltx25_nodes(image_to_video=image_to_video) - available
        if missing_nodes:
            raise H3Error(
                "LTX-2.5 requires a current ComfyUI with LTXVideo nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        progress(0, desc="Building LTX-2.5 workflow")
        graph = build_ltx25_graph(
            model_choice=model_choice,
            prompt=str(prompt).strip(),
            negative_prompt=str(negative_prompt or ""),
            first_image=first_image if image_to_video else None,
            width=resolved_width,
            height=resolved_height,
            duration=float(duration),
            fps=float(fps),
            seed=actual_seed,
            cfg=float(cfg),
            sampler_name=str(sampler_name),
            image_strength=float(image_strength),
            middle_image=middle_image if image_to_video else None,
            middle_time=float(middle_time),
            middle_strength=float(middle_strength),
            end_image=end_image if image_to_video else None,
            end_strength=float(end_strength),
        )

        client_id = str(uuid.uuid4())
        try:
            ws = websocket.create_connection(
                websocket_url(client_id),
                timeout=max(1.0, min(REQUEST_TIMEOUT, 10.0)),
            )
        except Exception:
            ws = None
        prompt_id = submit_prompt(graph, client_id)
        frames = ltx25_frame_length(duration, fps)
        yield None, (
            f"Queued LTX-2.5 job `{prompt_id}` 路 seed {actual_seed} 路 "
            f"{resolved_width}×{resolved_height} 路 {frames} frames at {float(fps):g} fps 路 "
            f"{model_choice} distilled 8-step model"
        )

        updates = (
            stream_comfy_progress(ws, prompt_id, graph, started)
            if ws is not None else poll_comfy_progress(prompt_id, graph)
        )
        for stage, completed_nodes, total_nodes, step, step_total in updates:
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total_nodes:
                progress((completed_nodes, total_nodes), desc=stage)
            yield None, progress_status(
                stage,
                started=started,
                completed_nodes=completed_nodes,
                total_nodes=total_nodes,
                step=step,
                step_total=step_total,
                configured_steps=8 if stage == "Generating video and audio" else None,
                detail=f"LTX-2.5 job `{prompt_id}`",
            )

        history = wait_for_history(prompt_id)
        result = resolve_output(history, queued_at)
        elapsed = time.monotonic() - started
        progress(1, desc="Complete")
        yield str(result), (
            f"LTX-2.5 completed in {elapsed:.1f}s 路 output {result.name} 路 "
            f"seed {actual_seed}"
        )
    except Exception as exc:
        yield None, f"Error: {exc}"
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


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
    """Run MiniMax Music 3 through the same ComfyUI queue as video jobs."""
    started = time.monotonic()
    queued_at = time.time()
    ws: websocket.WebSocket | None = None
    try:
        unload_prompt_rewriter()
        yield None, progress_status("Validating Music 3 request", started=started)
        if not str(caption).strip():
            raise H3Error("A music caption is required.")
        if not 1 <= float(max_duration) <= 300:
            raise H3Error("Maximum duration must be between 1 and 300 seconds.")
        if not 1 <= int(steps) <= 100:
            raise H3Error("Sampling steps must be between 1 and 100.")
        if not 0 <= float(cfg) <= 100 or not 0 <= float(ar_cfg) <= 100:
            raise H3Error("CFG values must be between 0 and 100.")
        if not 1 <= int(top_k) <= 8192:
            raise H3Error("Top K must be between 1 and 8192.")
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)

        missing_files = missing_music3_model_names(model_choice)
        if missing_files:
            yield None, progress_status(
                "Downloading MiniMax Music 3 models on demand",
                started=started,
                detail=", ".join(missing_files),
            )
        ensure_music3_models(model_choice)
        available = set(object_info())
        missing_nodes = required_music3_nodes(bool(tiled_decode)) - available
        if missing_nodes:
            raise H3Error(
                "MiniMax Music 3 requires ComfyUI 0.33.0 or newer; missing nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        graph = build_music3_graph(
            model_choice=model_choice,
            caption=str(caption).strip(),
            lyrics=str(lyrics or "").strip(),
            max_duration=float(max_duration),
            seed=actual_seed,
            steps=int(steps),
            cfg=float(cfg),
            ar_cfg=float(ar_cfg),
            top_k=int(top_k),
            tiled_decode=bool(tiled_decode),
        )
        client_id = str(uuid.uuid4())
        try:
            ws = websocket.create_connection(
                websocket_url(client_id),
                timeout=max(1.0, min(REQUEST_TIMEOUT, 10.0)),
            )
        except Exception:
            ws = None
        prompt_id = submit_prompt(graph, client_id)
        yield None, (
            f"Queued Music 3 job `{prompt_id}` · seed {actual_seed} · "
            f"up to {float(max_duration):g}s · {model_choice}"
        )
        updates = (
            stream_comfy_progress(ws, prompt_id, graph, started)
            if ws is not None else poll_comfy_progress(prompt_id, graph)
        )
        for stage, completed_nodes, total_nodes, step, step_total in updates:
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total_nodes:
                progress((completed_nodes, total_nodes), desc=stage)
            yield None, progress_status(
                stage,
                started=started,
                completed_nodes=completed_nodes,
                total_nodes=total_nodes,
                step=step,
                step_total=step_total,
                configured_steps=int(steps) if stage == "Generating music" else None,
                detail=f"Music 3 job `{prompt_id}`",
            )
        result = resolve_audio_output(wait_for_history(prompt_id), queued_at)
        elapsed = time.monotonic() - started
        progress(1, desc="Complete")
        yield str(result), (
            f"Music 3 completed in {elapsed:.1f}s · output {result.name} · "
            f"seed {actual_seed}"
        )
    except Exception as exc:
        yield None, f"Error: {exc}"
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def interrupt() -> str:
    try:
        api_post("/interrupt", json={})
        return "Interrupt requested."
    except Exception as exc:
        return f"Interrupt failed: {exc}"


def preset_values(name: str):
    return SAMPLING_PRESETS.get(str(name), SAMPLING_PRESETS["Balanced"])


def compact_settings_summary(
    mode: str,
    model_profile: str,
    use_int8_vae: bool,
    generation_mode: str,
    turbo_variant: str,
    duration: float,
    width: int,
    height: int,
    steps: int,
    scheduler: str,
    attention_mode: str,
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
) -> str:
    model_profile = (
        f"{model_profile} / VAE: "
        f"{'INT8 ConvRot' if use_int8_vae else 'FP16'}"
    )
    if generation_mode == "Turbo":
        generation_mode = f"Turbo / {turbo_variant}"
    result_format = normalize_result_format(result_format)
    if result_format == "Image":
        try:
            timing = f"{validate_image_frame_count(image_frames)} image frames"
        except H3Error:
            timing = "— image frames"
    else:
        try:
            timing = f"{float(duration):g}s"
        except (TypeError, ValueError):
            timing = "—s"
    if result_format == "Audio":
        resolution = "32×32 automatic canvas"
    else:
        try:
            resolution = f"{int(width)}×{int(height)}"
        except (TypeError, ValueError):
            resolution = "—×—"
    try:
        step_count = f"{int(steps)} steps"
    except (TypeError, ValueError):
        step_count = "— steps"
    postprocess_note = postprocess
    if postprocess == SEEDVR2_UPSCALE:
        postprocess_note += (
            f" / {seedvr2_model} / unload first: "
            f"{'on' if force_offload else 'off'}"
        )
    elif postprocess == LTX25_UPSCALE:
        postprocess_note += (
            f" / {ltx25_model} / unload first: "
            f"{'on' if force_offload else 'off'}"
        )
        if split_upscale:
            postprocess_note += f" / split: {float(split_seconds):g}s clips"
    latent_note = "Off"
    if latent_upscale:
        latent_note = (
            f"2x / {latent_upscaler_model} / "
            f"{int(latent_upscale_refine_steps)} refinement steps"
        )
    return (
        "**Current setup**  \n"
        f"{mode} · Result: {result_format} · {model_profile} / {generation_mode} · "
        f"{timing} · {resolution} · "
        f"{step_count} / {scheduler} · Attention: {attention_mode} · "
        f"Cache: {cache_mode} · Native latent: {latent_note} · "
        f"Post: {postprocess_note}"
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
        use_int8_vae=defaults["use_int8_vae"],
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
        latent_upscale=defaults["latent_upscale"],
        latent_upscaler_model=defaults["latent_upscaler_model"],
        latent_upscale_refine_steps=defaults["latent_upscale_refine_steps"],
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


def generate_for_ui(*args: Any):
    """Adapt the shared result path to the three result components in the H3 tab."""
    if len(args) < 2:
        raise H3Error("Missing result-format inputs.")
    result_format = normalize_result_format(args[-2])
    first_update = True
    for result, status in generate(*args):
        if first_update:
            first_update = False
            yield (
                gr.update(value=None, visible=result_format == "Video"),
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
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), status,
            )
            continue

        if result_format == "Image":
            paths = normalize_paths(result)
            labels = image_frame_labels(paths)
            gallery = [(path, label) for path, label in zip(paths, labels)]
            yield (
                gr.update(),
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
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(value=result, visible=True), gr.update(), status,
            )
        else:
            yield (
                gr.update(value=result, visible=True),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), status,
            )


def api_guide() -> str:
    defaults = UI_DEFAULTS
    return f"""## Generate through the API

The `/generate_video` endpoint accepts a prompt and preserves the existing **Video** defaults from the **MiniMax H3** tab:

`{defaults['mode']}` · `{defaults['model_profile']}` · `{defaults['generation_mode']} / {defaults['turbo_variant']}` · `{defaults['duration']}s` · `{defaults['width']}×{defaults['height']}` · `{defaults['steps']} steps` · `{defaults['scheduler']}` scheduler · random seed

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


def build_ui() -> gr.Blocks:
    defaults = UI_DEFAULTS
    with gr.Blocks(title="MiniMax H3 Local") as demo:
        gr.Markdown("# MiniMax H3 Local\nNative ComfyUI graphs for T2V, first/last-frame video, and reference media.")
        gr.HTML(
            '<a href="/comfyui/" target="_blank" rel="noopener noreferrer">'
            "Open ComfyUI ↗</a>"
        )
        with gr.Row(equal_height=True):
            health = gr.Markdown(backend_status())
            unload_models = gr.Button(
                "Unload all models / free VRAM",
                scale=0,
            )
        memory_status = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("MiniMax H3") as generate_tab:
                gr.HTML("")
            with gr.Tab("LTX 2.5") as ltx25_tab:
                gr.HTML("")
            with gr.Tab("MiniMax Music 3") as music3_tab:
                gr.HTML("")
            with gr.Tab("Gallery") as gallery_tab:
                gr.HTML("")
            with gr.Tab("API") as api_tab:
                gr.HTML("")
        with gr.Row() as generation_view:
            with gr.Column(scale=3):
                with gr.Row():
                    mode = gr.Radio(
                        ["Text to video", "First / last frame", "Reference media"],
                        value=defaults["mode"], label="Conditioning mode",
                    )
                    result_format = gr.Radio(
                        RESULT_FORMATS,
                        value=defaults["result_format"],
                        label="Result format",
                        info="H3 always samples vision and audio; this selects what is decoded and shown.",
                    )
                with gr.Row():
                    model_profile = gr.Radio(
                        MODEL_PROFILE_CHOICES,
                        value=defaults["model_profile"],
                        label="Base model",
                        info=(
                            "Speed uses the rebuilt single-pass NVFP4 files. "
                            "Quality uses the mixed NVFP4/FP8/INT8 ConvRot files. "
                            "Original uses the official BF16 files. Speed and Original "
                            "download when first selected."
                        ),
                    )
                    generation_mode = gr.Radio(
                        ["Normal", "Turbo"],
                        value=defaults["generation_mode"],
                        label="Generation",
                        info=(
                            "Turbo uses the implementation selected in Generation settings. "
                            "Reference media temporarily uses the corresponding FL2VA-trained "
                            "LoRA and is experimental."
                        ),
                    )
                use_int8_vae = gr.Checkbox(
                    value=defaults["use_int8_vae"],
                    label="Experimental INT8 ConvRot video VAE",
                    info=(
                        "Default off. Downloads on first use and accelerates H3 video "
                        "encode/decode; switch off for the reviewed FP16 path."
                    ),
                )
                help_text = gr.Markdown(mode_help("Text to video"))
                prompt = gr.Textbox(
                    label="Prompt", lines=12,
                    placeholder="Describe shots, camera motion, dialogue, sound effects, ambience, music, and any tagged references.",
                )
                with gr.Accordion("Prompt writer / enhancer", open=False):
                    gr.Markdown(
                        "The local MiniMax-H3 8B writer supports T2VA, I2VA, "
                        "L2VA, and FL2VA. Gemini remains available for Reference "
                        "media and other multimodal enhancement."
                    )
                    prompt_writer_backend = gr.Radio(
                        PROMPT_WRITER_BACKENDS,
                        value=DEFAULT_PROMPT_WRITER_BACKEND,
                        label="Prompt writer",
                    )
                    with gr.Group(visible=True) as local_prompt_writer_group:
                        local_prompt_base_model = gr.Dropdown(
                            choices=list(LOCAL_PROMPT_BASE_MODELS),
                            value=DEFAULT_LOCAL_PROMPT_BASE_MODEL,
                            label="Local base model",
                            info=(
                                "FP8 is the default lower-memory checkpoint. BF16 is "
                                "available as the full-precision alternative."
                            ),
                        )
                        with gr.Accordion("Local decoding settings", open=False):
                            local_prompt_greedy = gr.Checkbox(
                                value=True, label="Greedy decoding"
                            )
                            local_prompt_max_tokens = gr.Slider(
                                256, 8192, value=4096, step=256,
                                label="Max new tokens",
                            )
                            with gr.Row():
                                local_prompt_temperature = gr.Slider(
                                    0.1, 2.0, value=0.7, step=0.1,
                                    label="Temperature (sampling)",
                                )
                                local_prompt_top_p = gr.Slider(
                                    0.05, 1.0, value=0.8, step=0.05,
                                    label="Top-p (sampling)",
                                )
                            local_prompt_seed = gr.Number(
                                value=42, precision=0, label="Seed"
                            )
                    with gr.Group(visible=False) as gemini_prompt_writer_group:
                        gr.Markdown(
                            "Uses the active inputs with `prompt.txt`. Set "
                            "`GEMINI_API_KEY` on the server or enter a temporary key; "
                            "the server does not store UI keys."
                        )
                        with gr.Row():
                            gemini_prompt_model = gr.Dropdown(
                                choices=list(GEMINI_PROMPT_MODELS),
                                value=DEFAULT_GEMINI_PROMPT_MODEL,
                                label="Gemini model",
                            )
                            gemini_api_key = gr.Textbox(
                                label="Temporary Gemini API key",
                                type="password",
                                placeholder="Uses GEMINI_API_KEY when blank",
                            )
                    enhance_prompt_button = gr.Button("Generate / enhance prompt")
                    enhance_prompt_status = gr.Textbox(
                        label="Prompt enhancer status", lines=2, interactive=False
                    )
                with gr.Group(visible=False) as frame_group:
                    gr.Markdown("### First / last frame inputs")
                    with gr.Row():
                        first = gr.Image(
                            type="filepath",
                            label="First frame (auto resolution)",
                            elem_id="first-frame-image",
                        )
                        last = gr.Image(type="filepath", label="Last frame")
                with gr.Group(visible=False) as reference_group:
                    gr.Markdown("### Reference media")
                    gr.Markdown(reference_prompt_help())
                    with gr.Row():
                        ref_image_1 = gr.Image(type="filepath", label="Picture 1")
                        ref_image_2 = gr.Image(type="filepath", label="Picture 2")
                        ref_image_3 = gr.Image(type="filepath", label="Picture 3")
                    with gr.Row():
                        ref_image_4 = gr.Image(type="filepath", label="Picture 4")
                        ref_image_5 = gr.Image(type="filepath", label="Picture 5")
                        ref_image_6 = gr.Image(type="filepath", label="Picture 6")
                    with gr.Row():
                        ref_image_7 = gr.Image(type="filepath", label="Picture 7")
                        ref_image_8 = gr.Image(type="filepath", label="Picture 8")
                        ref_image_9 = gr.Image(type="filepath", label="Picture 9")
                    with gr.Row():
                        ref_video_1 = gr.Video(label="Video 1")
                        ref_video_2 = gr.Video(label="Video 2")
                        ref_video_3 = gr.Video(label="Video 3")
                    with gr.Row():
                        ref_audio_1 = gr.Audio(type="filepath", label="Audio 1")
                        ref_audio_2 = gr.Audio(type="filepath", label="Audio 2")
                        ref_audio_3 = gr.Audio(type="filepath", label="Audio 3")
                    ref_size = gr.Radio(
                        ["match", "max"],
                        value=defaults["ref_image_size"],
                        label="Reference image size",
                    )
            with gr.Column(scale=2):
                settings_overview = gr.Markdown(
                    compact_settings_summary(
                        defaults["mode"], defaults["model_profile"],
                        defaults["use_int8_vae"],
                        defaults["generation_mode"], defaults["turbo_variant"],
                        defaults["duration"], defaults["width"], defaults["height"],
                        defaults["steps"], defaults["scheduler"],
                        defaults["attention_mode"], defaults["cache_mode"],
                        defaults["latent_upscale"],
                        defaults["latent_upscaler_model"],
                        defaults["latent_upscale_refine_steps"],
                        defaults["postprocess"], defaults["seedvr2_model"],
                        DEFAULT_LTX25_MODEL,
                        defaults["upscale_force_offload"],
                        defaults["upscale_split_enabled"],
                        defaults["upscale_split_seconds"],
                        defaults["result_format"], defaults["image_frames"],
                    )
                )
                output = gr.Video(label="Generated video")
                with gr.Group(visible=False) as image_output_group:
                    image_output = gr.Gallery(
                        value=[],
                        label="Generated image frames",
                        columns=4,
                        object_fit="contain",
                        allow_preview=True,
                        height=520,
                    )
                    image_frame_paths = gr.State([])
                    image_selection = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Frames to save",
                    )
                    with gr.Row():
                        image_select_all = gr.Button("Select all")
                        image_clear_selection = gr.Button("Clear selection")
                        image_save_selected = gr.Button(
                            "Save selected frames", variant="primary"
                        )
                    image_saved_files = gr.File(
                        label="Saved image files",
                        file_count="multiple",
                        interactive=False,
                    )
                    image_save_status = gr.Markdown()
                audio_output = gr.Audio(
                    label="Generated audio", type="filepath", visible=False
                )
                with gr.Row():
                    run = gr.Button("Generate video", variant="primary", scale=2)
                    stop = gr.Button("Interrupt", scale=1)
                    refresh = gr.Button("Refresh status", scale=1)
                status = gr.Textbox(label="Status", lines=5)
                gr.Markdown("### Generation settings")
                turbo_variant = gr.Radio(
                    list(TURBO_SETTINGS),
                    value=defaults["turbo_variant"],
                    label="Turbo implementation",
                    info=(
                        "Original BF16 uses memory-safe runtime LoRA bypass. Speed and "
                        "Quality use faster merged weights. Larry also uses its adaptive "
                        "sampler; all variants run at strength 1.0."
                    ),
                )
                preset = gr.Radio(
                    ["Quality", "Balanced", "Fast"],
                    value="Balanced",
                    label="Sampling preset",
                    interactive=False,
                    info=(
                        "Normal-generation preset. Disabled in Turbo so it cannot overwrite "
                        "the Turbo Steps/Scheduler controls."
                    ),
                )
                with gr.Row():
                    duration = gr.Slider(
                        2, 15, value=defaults["duration"], step=0.5, label="Seconds"
                    )
                    image_frames = gr.Slider(
                        MIN_IMAGE_FRAMES,
                        MAX_IMAGE_FRAMES,
                        value=defaults["image_frames"],
                        step=1,
                        label="Image frames",
                        visible=False,
                        info=(
                            "H3 samples a native 5- or 22-frame packet, then returns "
                            "exactly this many decoded frames."
                        ),
                    )
                    steps = gr.Slider(
                        4, 30, value=defaults["steps"], step=1, label="Steps",
                        info=(
                            "LightX2V 4-step is the default Turbo variant; Larry and the "
                            "8-step LightX2V variant keep their trained step counts. Increase Turbo "
                            "steps when a clip benefits from extra refinement; Normal H3 "
                            "presets normally use 15–20."
                        ),
                    )
                with gr.Row():
                    draft_resolution = gr.Dropdown(
                        choices=list(DRAFT_RESOLUTIONS),
                        value=None,
                        label="Draft",
                        info="Quick test sizes by aspect ratio.",
                    )
                    fast_resolution = gr.Dropdown(
                        choices=list(FAST_RESOLUTIONS),
                        value="16:9 · 864×480",
                        label="Fast",
                        info="Recommended working sizes by aspect ratio.",
                    )
                    large_resolution = gr.Dropdown(
                        choices=list(LARGE_RESOLUTIONS),
                        value=None,
                        label="Large",
                        info="Higher-resolution sizes by aspect ratio.",
                    )
                with gr.Row():
                    width = gr.Number(value=defaults["width"], precision=0, label="Width")
                    height = gr.Number(value=defaults["height"], precision=0, label="Height")
                resolution_info = gr.Markdown(
                    resolution_summary(defaults["width"], defaults["height"])
                )
                with gr.Row():
                    scheduler = gr.Radio(
                        ["simple", "beta", "normal"],
                        value=defaults["scheduler"], label="Scheduler",
                    )
                    seed = gr.Number(
                        value=defaults["seed"], precision=0, label="Seed (-1 random)"
                    )
                attention_mode = gr.Radio(
                    ["Sage 2", "Kitchen", "Sol-Attn", "Auto"],
                    value=defaults["attention_mode"],
                    label="Attention",
                    interactive=SERVER_ATTENTION_BACKEND == "sol",
                    info=(
                        f"Auto enables Sol-Attn for Reference mode or when estimated "
                        f"packed target tokens reach {AUTO_SOL_TOKEN_THRESHOLD:,}; "
                        "Sage 2 is the measured-fastest default and applies the pinned "
                        "KJNodes model override. Kitchen selects the global ComfyUI "
                        "backend. Auto uses Kitchen for smaller jobs. Sol "
                        f"dense/fallback calls use {SERVER_DENSE_ATTENTION_BACKEND}."
                    ),
                )
                with gr.Row():
                    sol_tau = gr.Slider(
                        0.5, 1.5, value=defaults["sol_tau"], step=0.1,
                        label="Sol-Attn tau"
                    )
                    sol_thresh_type = gr.Radio(
                        ["diag", "exact"],
                        value=defaults["sol_thresh_type"],
                        label="Sol threshold",
                        info="diag is faster; exact calculates a more precise routing threshold.",
                    )
                with gr.Accordion("Zero-copy Sol-Attn quality controls", open=False):
                    sol_exact_mode = gr.Radio(
                        ["off", "exact_kv", "exact_kv_and_rows"],
                        value=defaults["sol_exact_mode"],
                        label="Exact H3 prefix mode",
                        info=(
                            "exact_kv preserves text/condition/reference/audio KV "
                            "rows at low cost. exact_kv_and_rows also keeps prefix "
                            "query rows dense for maximum audio/conditioning fidelity."
                        ),
                    )
                    with gr.Row():
                        sol_dense_steps = gr.Slider(
                            0, 4, value=defaults["sol_dense_steps"], step=1,
                            label="Dense final transformer blocks",
                            info=(
                                "Keep the final N H3 transformer blocks dense. "
                                "The final block is the most approximation-sensitive."
                            ),
                        )
                    sol_step_off = gr.State(0.0)
                    sol_sink_tokens = gr.State(0)
                with gr.Accordion("Sampling acceleration", open=False):
                    cache_mode = gr.Radio(
                        ["Spectrum", "FirstBlockCache", "EasyCache", "Off"],
                        value=defaults["cache_mode"],
                        label="Acceleration mode",
                        info=(
                            "Spectrum is the normal H3 default based on broader community "
                            "speed testing. It forecasts selected transformer steps and uses "
                            "audio-isolated offline replay. FirstBlockCache is the lower-memory "
                            "fallback. Modes are mutually exclusive. Turbo defaults to Spectrum; "
                            "EasyCache and FirstBlockCache are opt-in experimental Turbo options."
                        ),
                    )
                    fbcache_preset = gr.Radio(
                        ["Safe", "Fast", "Aggressive", "Custom"],
                        value=defaults["fbcache_preset"],
                        label="FirstBlockCache preset",
                        info=(
                            "Fast is the recommended default. Named presets use "
                            "a protected 10–95% denoising window and at most two "
                            "consecutive cache hits."
                        ),
                    )
                    with gr.Row():
                        fbcache_threshold = gr.Slider(
                            0.0, 0.25,
                            value=defaults["fbcache_threshold"],
                            step=0.005,
                            label="FirstBlock threshold",
                            interactive=False,
                        )
                        fbcache_max_hits = gr.Slider(
                            1, 8,
                            value=defaults["fbcache_max_hits"],
                            step=1,
                            label="Max consecutive cache hits",
                            interactive=False,
                        )
                    with gr.Row():
                        fbcache_start = gr.Slider(
                            0.0, 0.90,
                            value=defaults["fbcache_start"],
                            step=0.01,
                            label="Cache start percent",
                            interactive=False,
                        )
                        fbcache_end = gr.Slider(
                            0.10, 1.0,
                            value=defaults["fbcache_end"],
                            step=0.01,
                            label="Cache end percent",
                            interactive=False,
                        )
                    fbcache_temporal_guard = gr.Checkbox(
                        value=defaults["fbcache_temporal_guard"],
                        label="Temporal frame guard",
                        info=(
                            "Checks the most-changed target-video latent frame "
                            "in addition to the global residual average."
                        ),
                    )
                    gr.Markdown("**EasyCache fallback settings**")
                    easycache_threshold = gr.Slider(
                        0.0, 0.5, value=defaults["easycache_threshold"], step=0.01,
                        label="Reuse threshold",
                        info=(
                            "Higher skips more steps. Start at 0.10 for H3; "
                            "ComfyUI's generic default is 0.20."
                        ),
                    )
                    with gr.Row():
                        easycache_start = gr.Slider(
                            0.0, 0.9, value=defaults["easycache_start"], step=0.01,
                            label="Start percent",
                        )
                        easycache_end = gr.Slider(
                            0.1, 1.0, value=defaults["easycache_end"], step=0.01,
                            label="End percent",
                        )
                    easycache_verbose = gr.Checkbox(
                        value=defaults["easycache_verbose"],
                        label="Log EasyCache decisions",
                        info="Logs skipped-step counts and estimated speedup in ComfyUI.",
                    )

                gr.Markdown("### Native H3 latent upscale")
                latent_upscale = gr.Checkbox(
                    value=defaults["latent_upscale"],
                    label="Generate at half resolution, then latent upscale 2x",
                    info=(
                        "Runs inside H3 sampling, not after video generation. Width and "
                        "height remain the final output resolution. Disabled by default."
                    ),
                )
                with gr.Group(visible=defaults["latent_upscale"]) as latent_upscale_settings:
                    latent_upscaler_model = gr.Dropdown(
                        choices=list(H3_LATENT_UPSCALER_MODEL_CHOICES),
                        value=defaults["latent_upscaler_model"],
                        label="Latent upscaler model",
                        info=(
                            "Balanced uses BF16 and is the default. Fast uses FP16; "
                            "Quality uses FP32 and needs more memory. Downloaded on first use."
                        ),
                    )
                    latent_upscale_refine_steps = gr.Slider(
                        1, 6,
                        value=defaults["latent_upscale_refine_steps"],
                        step=1,
                        label="High-resolution refinement steps",
                        info=(
                            "H3 first finishes all generation steps at half resolution. "
                            "The clean 2x latent is then lightly re-noised and refined. "
                            "Two expensive high-resolution steps is the default."
                        ),
                    )
                    gr.Markdown(
                        "Final width and height must both be divisible by 64. For example, "
                        "1024×1024 generates the first stage at 512×512 and finishes at "
                        "1024×1024. Only the video latent is upscaled; H3 audio is preserved."
                    )

                gr.Markdown("### Generation post-processing")
                generation_postprocess = gr.Dropdown(
                    choices=GENERATION_POSTPROCESS_OPTIONS,
                    value=defaults["postprocess"],
                    label="After generation",
                    info=(
                        "Optionally run SeedVR2 or LTX-2.5 2x immediately after the base H3 "
                        "video finishes. The source video remains in the gallery."
                    ),
                )
                with gr.Group(visible=False) as generation_postprocess_settings:
                    generation_seedvr2_model = gr.Dropdown(
                        choices=list(SEEDVR2_MODEL_CHOICES),
                        value=defaults["seedvr2_model"],
                        label="SeedVR2 model",
                        info=(
                            "Downloaded on first use. 7B Sharp favors stronger "
                            "detail; NVFP4 variants are optimized for Blackwell GPUs."
                        ),
                    )
                    generation_ltx25_note = gr.Markdown(
                        "Uses the transformer selected in the **LTX 2.5** tab and "
                        "the H3 generation prompt. The gated 2x IC-LoRA downloads "
                        "on first use.",
                        visible=False,
                    )
                    generation_force_offload = gr.Checkbox(
                        value=defaults["upscale_force_offload"],
                        label="Unload H3 models before upscaling",
                        info=(
                            "Reduces peak VRAM at the cost of reloading H3 for "
                            "the next generation."
                        ),
                    )
                    generation_split_upscale = gr.Checkbox(
                        value=defaults["upscale_split_enabled"],
                        label="Split source into clips before LTX upscaling",
                        info=(
                            "Opt in after an out-of-VRAM error. Each clip is "
                            "upscaled independently and concatenated afterward."
                        ),
                        visible=False,
                    )
                    generation_split_seconds = gr.Slider(
                        1.0,
                        15.0,
                        value=defaults["upscale_split_seconds"],
                        step=0.5,
                        label="Target clip length (seconds)",
                        info=(
                            "5 seconds is the recommended starting point. The "
                            "actual cut is adjusted to an LTX-valid frame count."
                        ),
                        visible=False,
                    )

        with gr.Group(visible=False) as ltx25_view:
            gr.Markdown(
                "## LTX-2.5 audio-video generation\n"
                "Official single-stage distilled workflow on the shared ComfyUI backend. "
                "The gated model assets download on first use; accept the "
                "[LTX-2.5 model license](https://huggingface.co/Lightricks/LTX-2.5) "
                "before generating."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    ltx25_model = gr.Dropdown(
                        choices=list(LTX25_MODEL_CHOICES),
                        value=LTX25_DEFAULTS["model"],
                        label="Transformer model",
                        info=(
                            "INT8 ConvRot is the default lower-memory option. "
                            "Only the selected transformer downloads."
                        ),
                    )
                    ltx25_mode = gr.Radio(
                        ["Text to video", "Image to video"],
                        value=LTX25_DEFAULTS["mode"],
                        label="Mode",
                    )
                    ltx25_prompt = gr.Textbox(
                        label="Positive prompt",
                        lines=10,
                        placeholder=(
                            "Describe the action chronologically, then the setting, "
                            "camera movement, lighting, dialogue, sound effects, and music."
                        ),
                    )
                    with gr.Accordion("Gemini LTX-2.5 prompt writer", open=False):
                        gr.Markdown("Create or enhance the prompt from text and optional start, middle, or end images.")
                        with gr.Row():
                            ltx25_prompt_model = gr.Dropdown(
                                choices=list(GEMINI_PROMPT_MODELS), value=DEFAULT_GEMINI_PROMPT_MODEL,
                                label="Gemini model",
                            )
                            ltx25_api_key = gr.Textbox(
                                label="Temporary Gemini API key", type="password",
                                placeholder="Uses GEMINI_API_KEY when blank",
                            )
                        ltx25_enhance = gr.Button("Generate / enhance LTX-2.5 prompt")
                        ltx25_enhance_status = gr.Textbox(label="Prompt writer status", lines=2, interactive=False)
                    ltx25_negative = gr.Textbox(
                        label="Negative prompt",
                        lines=3,
                        placeholder="Optional artifacts or qualities to avoid",
                    )
                    with gr.Group(visible=False) as ltx25_image_group:
                        gr.Markdown(
                            "Choose a required start frame and optional middle/end "
                            "frames. Each image is applied at its clip position."
                        )
                        with gr.Row():
                            ltx25_image = gr.Image(
                                type="filepath", label="Start keyframe (required)"
                            )
                            ltx25_image_strength = gr.Slider(
                                0.0, 1.0,
                                value=LTX25_DEFAULTS["image_strength"],
                                step=0.05,
                                label="Start strength",
                            )
                        with gr.Accordion(
                            "Optional middle and end keyframes", open=False
                        ):
                            with gr.Row():
                                ltx25_middle_image = gr.Image(
                                    type="filepath", label="Middle keyframe"
                                )
                                with gr.Column():
                                    ltx25_middle_time = gr.Slider(
                                        0.1, 19.9,
                                        value=LTX25_DEFAULTS["middle_time"],
                                        step=0.1,
                                        label="Middle position (seconds)",
                                    )
                                    ltx25_middle_strength = gr.Slider(
                                        0.0, 1.0,
                                        value=LTX25_DEFAULTS["middle_strength"],
                                        step=0.05,
                                        label="Middle strength",
                                    )
                            with gr.Row():
                                ltx25_end_image = gr.Image(
                                    type="filepath", label="End keyframe"
                                )
                                ltx25_end_strength = gr.Slider(
                                    0.0, 1.0,
                                    value=LTX25_DEFAULTS["end_strength"],
                                    step=0.05,
                                    label="End strength",
                                )
                with gr.Column(scale=2):
                    ltx25_output = gr.Video(label="Generated LTX-2.5 video")
                    with gr.Row():
                        ltx25_run = gr.Button("Generate with LTX-2.5", variant="primary")
                        ltx25_stop = gr.Button("Interrupt")
                    ltx25_status = gr.Textbox(label="Status", lines=6)
                    gr.Markdown("### Generation settings")
                    with gr.Row():
                        ltx25_duration = gr.Slider(
                            1, 20, value=LTX25_DEFAULTS["duration"], step=0.5,
                            label="Seconds",
                        )
                        ltx25_fps = gr.Slider(
                            1, 60, value=LTX25_DEFAULTS["fps"], step=1,
                            label="FPS",
                        )
                    with gr.Row():
                        ltx25_width = gr.Number(
                            value=LTX25_DEFAULTS["width"], precision=0, label="Width"
                        )
                        ltx25_height = gr.Number(
                            value=LTX25_DEFAULTS["height"], precision=0, label="Height"
                        )
                    gr.Markdown(
                        "Width and height are snapped to multiples of 32. Frame count is "
                        "automatically snapped to `8n + 1`."
                    )
                    with gr.Row():
                        ltx25_seed = gr.Number(
                            value=LTX25_DEFAULTS["seed"], precision=0,
                            label="Seed (-1 random)",
                        )
                        ltx25_cfg = gr.Slider(
                            0.0, 3.0, value=LTX25_DEFAULTS["cfg"], step=0.05,
                            label="CFG",
                        )
                    ltx25_sampler = gr.Dropdown(
                        ["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_sde"],
                        value=LTX25_DEFAULTS["sampler"],
                        label="Sampler",
                    )
                    gr.Markdown(
                        "Uses the official distilled 8-step sigma schedule and the "
                        "lower-memory convolutional LTX-2.5 video VAE. INT8 is an "
                        "embedded-quantized ComfyUI checkpoint."
                    )

            with gr.Accordion(
                "Official workflows and model downloads",
                open=True,
            ):
                gr.Markdown(
                    "The complete official Lightricks LTX-2.5 workflow set is "
                    "installed in the bundled ComfyUI editor. These visual "
                    "workflows cover audio-only generation, two-stage refinement, "
                    "video editing, reference sheets, motion tracks, in/outpainting, "
                    "and pose/depth/canny control. **ComfyUI does not download "
                    "missing dropdown models automatically.** Use the buttons "
                    "below first. The official LTX 2.5 templates reuse LTX 2.3 "
                    "IC-LoRAs, so those filenames are expected. Each linked "
                    "Hugging Face repository can require separate license "
                    "acceptance; access to the main LTX-2.5 repository does "
                    "not grant access to every IC-LoRA."
                )
                with gr.Row():
                    ltx25_workflow = gr.Dropdown(
                        choices=list(LTX25_WORKFLOWS),
                        value=next(iter(LTX25_WORKFLOWS)),
                        label="Official workflow",
                        scale=3,
                    )
                    ltx25_prepare_workflow = gr.Button(
                        "Download selected workflow models",
                        variant="primary",
                        scale=1,
                    )
                with gr.Row():
                    ltx25_prepare_all_models = gr.Button(
                        "Download all missing models",
                        variant="secondary",
                    )
                    ltx25_refresh_models = gr.Button(
                        "Refresh model availability",
                        variant="secondary",
                    )
                ltx25_workflow_details = gr.Markdown(
                    render_ltx25_workflow_details(next(iter(LTX25_WORKFLOWS)))
                )
                ltx25_workflow_status = gr.Markdown()
                ltx25_model_inventory = gr.Markdown(
                    render_ltx25_official_model_inventory()
                )

        with gr.Group(visible=False) as music3_view:
            gr.Markdown(
                "## MiniMax Music 3\n"
                "Generate complete stereo songs with the native workflow on the shared "
                "ComfyUI backend. Write a detailed **caption** for style, vocals, and "
                "arrangement, then use section tags such as `[Intro]`, `[Verse]`, "
                "`[Chorus]`, `[Bridge]`, `[Instrumental]`, and `[Outro]` in the lyrics. "
                "[ComfyUI guide](https://docs.comfy.org/tutorials/audio/minimax/minimax-music-3) · "
                "[Official prompting skill](https://github.com/MiniMax-AI/MiniMax-Music3/tree/main/skills/music-caption-rewriter)"
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    music3_caption = gr.Textbox(
                        label="Music caption",
                        lines=12,
                        placeholder=(
                            "Global Metadata: genre, BPM, key, mood, production...\n\n"
                            "Vocal Details: singer, delivery, harmonies, effects...\n\n"
                            "Arrangement: instruments, groove, section-by-section evolution..."
                        ),
                    )
                    music3_lyrics = gr.Textbox(
                        label="Lyrics and song structure",
                        lines=16,
                        placeholder=(
                            "[Intro]\n\n[Verse]\nWrite lyrics here...\n\n"
                            "[Chorus]\n...\n\n[Bridge]\n...\n\n[Outro]"
                        ),
                        info="For an instrumental, repeat [Instrumental] sections to guide length.",
                    )
                    with gr.Accordion("Gemini Music 3 prompt writer", open=False):
                        gr.Markdown("Create or enhance the caption from text, lyrics, and optional visual reference images.")
                        with gr.Row():
                            music3_prompt_model = gr.Dropdown(
                                choices=list(GEMINI_PROMPT_MODELS), value=DEFAULT_GEMINI_PROMPT_MODEL,
                                label="Gemini model",
                            )
                            music3_api_key = gr.Textbox(
                                label="Temporary Gemini API key", type="password",
                                placeholder="Uses GEMINI_API_KEY when blank",
                            )
                        with gr.Row():
                            music3_ref_image_1 = gr.Image(type="filepath", label="Reference image 1")
                            music3_ref_image_2 = gr.Image(type="filepath", label="Reference image 2")
                            music3_ref_image_3 = gr.Image(type="filepath", label="Reference image 3")
                        music3_enhance = gr.Button("Generate / enhance Music 3 caption")
                        music3_enhance_status = gr.Textbox(label="Prompt writer status", lines=2, interactive=False)
                with gr.Column(scale=2):
                    music3_output = gr.Audio(
                        label="Generated song", type="filepath"
                    )
                    with gr.Row():
                        music3_run = gr.Button(
                            "Generate with Music 3", variant="primary"
                        )
                        music3_stop = gr.Button("Interrupt")
                    music3_status = gr.Textbox(label="Status", lines=7)
                    gr.Markdown("### Generation settings")
                    music3_model = gr.Dropdown(
                        choices=list(MUSIC3_MODEL_CHOICES),
                        value=MUSIC3_DEFAULTS["model"],
                        label="Diffusion model",
                        info="The selected DiT and shared encoder/decoder download on first use.",
                    )
                    with gr.Row():
                        music3_duration = gr.Slider(
                            1, 300, value=MUSIC3_DEFAULTS["duration"], step=1,
                            label="Maximum seconds",
                        )
                        music3_seed = gr.Number(
                            value=MUSIC3_DEFAULTS["seed"], precision=0,
                            label="Seed (-1 random)",
                        )
                    music3_tiled = gr.Checkbox(
                        value=MUSIC3_DEFAULTS["tiled_decode"],
                        label="Tiled audio decode",
                        info="Reduces peak VRAM for long songs; disable for fastest decode on high-VRAM GPUs.",
                    )
                    with gr.Accordion("Advanced sampling", open=False):
                        music3_steps = gr.Slider(
                            1, 100, value=MUSIC3_DEFAULTS["steps"], step=1,
                            label="Diffusion steps",
                        )
                        with gr.Row():
                            music3_cfg = gr.Slider(
                                0, 10, value=MUSIC3_DEFAULTS["cfg"], step=0.05,
                                label="Diffusion CFG",
                            )
                            music3_ar_cfg = gr.Slider(
                                0, 10, value=MUSIC3_DEFAULTS["ar_cfg"], step=0.05,
                                label="Autoregressive CFG",
                            )
                        music3_top_k = gr.Slider(
                            1, 200, value=MUSIC3_DEFAULTS["top_k"], step=1,
                            label="Autoregressive Top K",
                        )
                    gr.Markdown(
                        "Output is saved as V0-quality MP3 under `ComfyUI/output/audio`. "
                        "Music 3 may end a song before the maximum duration."
                    )

        with gr.Group(visible=False) as gallery_view:
            with gr.Row():
                gr.Markdown(
                    "## Generated videos\n"
                    "Only poster thumbnails are loaded here. Select one to load and play it."
                )
                gallery_refresh = gr.Button("Refresh gallery", scale=0)
                gallery_delete = gr.Button(
                    "Delete selected",
                    variant="stop",
                    scale=0,
                )
                gallery_empty = gr.Button(
                    "Empty all generated",
                    variant="stop",
                    scale=0,
                )
            gallery_status = gr.Markdown("Open this tab to scan generated videos.")
            gallery_paths = gr.State([])
            gallery_selected = gr.State(None)
            gallery_confirm_delete = gr.Checkbox(
                value=False,
                label="Confirm permanent deletion",
                info="Required for Delete selected and Empty all generated.",
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=280):
                    gallery_grid = gr.Gallery(
                        value=[],
                        label="Generated video thumbnails",
                        columns=2,
                        height=540,
                        object_fit="cover",
                        allow_preview=False,
                        fit_columns=False,
                        elem_id="generated-video-gallery",
                    )
                with gr.Column(scale=5, min_width=480):
                    gallery_player = gr.Video(
                        label="Selected video",
                        height=540,
                    )
                    gallery_download = gr.Markdown()
                    with gr.Accordion("Post-process selected video", open=True):
                        gallery_postprocess = gr.Dropdown(
                            choices=POSTPROCESS_OPTIONS,
                            value=POSTPROCESS_OPTIONS[0],
                            label="Method",
                        )
                        with gr.Group(visible=True) as gallery_ai_settings:
                            gallery_seedvr2_model = gr.Dropdown(
                                choices=list(SEEDVR2_MODEL_CHOICES),
                                value=defaults["seedvr2_model"],
                                label="SeedVR2 model",
                                visible=True,
                                info=(
                                    "Downloaded on first use. 7B Sharp favors stronger "
                                    "detail; NVFP4 variants are optimized for Blackwell GPUs."
                                ),
                            )
                            gallery_ltx25_prompt = gr.Textbox(
                                label="LTX-2.5 upscale prompt",
                                placeholder="Describe the source scene and desired fine detail",
                                lines=3,
                                visible=False,
                                info=(
                                    "Optional but recommended. Uses the transformer selected "
                                    "in the LTX 2.5 tab."
                                ),
                            )
                            with gr.Row():
                                gallery_post_seed = gr.Number(
                                    value=-1,
                                    precision=0,
                                    label="Seed (-1 random)",
                                )
                                gallery_force_offload = gr.Checkbox(
                                    value=False,
                                    label="Unload resident models first",
                                    info="Can lower peak VRAM before AI upscaling starts.",
                                )
                            gallery_split_upscale = gr.Checkbox(
                                value=False,
                                label="Split source into clips before LTX upscaling",
                                info=(
                                    "Opt in after an out-of-VRAM error. Upscales "
                                    "clips independently, then concatenates them."
                                ),
                                visible=False,
                            )
                            gallery_split_seconds = gr.Slider(
                                1.0,
                                15.0,
                                value=5.0,
                                step=0.5,
                                label="Target clip length (seconds)",
                                info=(
                                    "The actual cut is adjusted to an LTX-valid "
                                    "frame count."
                                ),
                                visible=False,
                            )
                        with gr.Row():
                            gallery_post_run = gr.Button(
                                "Post-process selected",
                                variant="primary",
                            )
                            gallery_post_stop = gr.Button("Interrupt")
                        gallery_post_status = gr.Markdown()

        with gr.Group(visible=False) as api_view:
            gr.Markdown(api_guide())
            with gr.Accordion("Try the default API request", open=False):
                api_prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="Describe the video, camera motion, dialogue, and sound.",
                )
                with gr.Row():
                    api_run = gr.Button("Generate with UI defaults", variant="primary")
                    api_stop = gr.Button("Interrupt")
                api_download_url = gr.Textbox(
                    label="Download URL",
                    interactive=False,
                )
                api_status = gr.Textbox(label="Status", lines=5)

        settings_inputs = [
            mode, model_profile, use_int8_vae, generation_mode, turbo_variant,
            duration, width, height,
            steps, scheduler, attention_mode, cache_mode,
            latent_upscale,
            latent_upscaler_model,
            latent_upscale_refine_steps,
            generation_postprocess,
            generation_seedvr2_model,
            ltx25_model,
            generation_force_offload,
            generation_split_upscale,
            generation_split_seconds,
            result_format,
            image_frames,
        ]
        for settings_control in settings_inputs:
            # Width and height have dedicated input callbacks below so their
            # 32/64-pixel alignment is applied before the summary is refreshed.
            if settings_control in (width, height):
                continue
            settings_control.change(
                compact_settings_summary,
                inputs=settings_inputs,
                outputs=settings_overview,
            )

        mode.change(
            mode_layout_updates,
            inputs=mode,
            outputs=[
                help_text, frame_group, reference_group, generation_mode,
                preset, steps, scheduler, cache_mode, attention_mode,
            ],
        )
        result_format.change(
            result_format_layout_updates,
            inputs=[result_format, width, height, first, latent_upscale],
            outputs=[
                duration,
                image_frames,
                output,
                image_output_group,
                audio_output,
                generation_postprocess,
                latent_upscale,
                width,
                height,
                resolution_info,
                run,
            ],
            queue=False,
            show_progress="hidden",
        )
        latent_upscale.change(
            latent_upscale_layout_updates,
            inputs=[latent_upscale, width, height, result_format],
            outputs=[
                latent_upscale_settings,
                width,
                height,
                resolution_info,
            ],
        )
        ltx25_mode.change(
            lambda value: gr.update(visible=value == "Image to video"),
            inputs=ltx25_mode,
            outputs=ltx25_image_group,
            queue=False,
            show_progress="hidden",
        )
        ltx25_workflow.change(
            render_ltx25_workflow_details,
            inputs=ltx25_workflow,
            outputs=ltx25_workflow_details,
            queue=False,
            show_progress="hidden",
        )
        ltx25_prepare_workflow.click(
            prepare_ltx25_official_workflow,
            inputs=ltx25_workflow,
            outputs=[ltx25_workflow_status, ltx25_model_inventory],
            show_progress="minimal",
        )
        ltx25_prepare_all_models.click(
            prepare_all_ltx25_official_models,
            outputs=[ltx25_workflow_status, ltx25_model_inventory],
            show_progress="minimal",
        )
        ltx25_refresh_models.click(
            render_ltx25_official_model_inventory,
            outputs=ltx25_model_inventory,
            queue=False,
            show_progress="hidden",
        )
        gallery_postprocess.change(
            lambda value: (
                gr.update(visible=value in COMFY_UPSCALE_OPTIONS),
                gr.update(visible=value == SEEDVR2_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
            ),
            inputs=gallery_postprocess,
            outputs=[
                gallery_ai_settings,
                gallery_seedvr2_model,
                gallery_ltx25_prompt,
                gallery_split_upscale,
                gallery_split_seconds,
            ],
            queue=False,
            show_progress="hidden",
        )
        generation_postprocess.change(
            lambda value: (
                gr.update(visible=value in COMFY_UPSCALE_OPTIONS),
                gr.update(visible=value == SEEDVR2_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
            ),
            inputs=generation_postprocess,
            outputs=[
                generation_postprocess_settings,
                generation_seedvr2_model,
                generation_ltx25_note,
                generation_split_upscale,
                generation_split_seconds,
            ],
            queue=False,
            show_progress="hidden",
        )
        generation_mode.change(
            generation_mode_defaults,
            inputs=[generation_mode, turbo_variant],
            outputs=[preset, steps, scheduler, cache_mode, attention_mode],
            queue=False,
            show_progress="hidden",
        )
        turbo_variant.change(
            turbo_variant_defaults,
            inputs=[turbo_variant, generation_mode],
            outputs=[steps, scheduler],
            queue=False,
            show_progress="hidden",
        )
        preset.change(
            preset_values,
            inputs=preset,
            outputs=[
                steps, sol_tau, sol_thresh_type, scheduler,
                sol_exact_mode, sol_dense_steps,
            ],
        )
        draft_resolution_event = draft_resolution.change(
            lambda name: resolution_choice_values(name, "draft"),
            inputs=draft_resolution,
            outputs=[width, height, resolution_info],
        )
        fast_resolution_event = fast_resolution.change(
            lambda name: resolution_choice_values(name, "fast"),
            inputs=fast_resolution,
            outputs=[width, height, resolution_info],
        )
        large_resolution_event = large_resolution.change(
            lambda name: resolution_choice_values(name, "large"),
            inputs=large_resolution,
            outputs=[width, height, resolution_info],
        )
        for resolution_event in (
            draft_resolution_event,
            fast_resolution_event,
            large_resolution_event,
        ):
            aligned_resolution_event = resolution_event.then(
                resolution_control_updates,
                inputs=[width, height, latent_upscale, result_format],
                outputs=[width, height, resolution_info],
                queue=False,
                show_progress="hidden",
            )
            aligned_resolution_event.then(
                compact_settings_summary,
                inputs=settings_inputs,
                outputs=settings_overview,
                queue=False,
                show_progress="hidden",
            )
        # Run immediately from the browser while the native File object is
        # still available. The staged-file callback below is a hosted-Gradio
        # fallback for environments that clear the input before JS runs.
        first.upload(
            fn=None,
            inputs=[first, width, height, result_format, latent_upscale],
            outputs=[width, height, resolution_info],
            js=AUTO_RESOLUTION_JS,
            queue=False,
            show_progress="hidden",
        )
        first_change_event = first.change(
            fn=auto_resolution_from_start_frame,
            inputs=[first, width, height, result_format, latent_upscale],
            outputs=[width, height, resolution_info],
            queue=False,
            show_progress="hidden",
        )
        first_change_event.then(
            compact_settings_summary,
            inputs=settings_inputs,
            outputs=settings_overview,
            queue=False,
            show_progress="hidden",
        )
        fbcache_preset.change(
            fbcache_preset_defaults,
            inputs=fbcache_preset,
            outputs=[
                fbcache_threshold,
                fbcache_start,
                fbcache_end,
                fbcache_max_hits,
            ],
        )
        width_input_event = width.input(
            resolution_control_updates,
            inputs=[width, height, latent_upscale, result_format],
            outputs=[width, height, resolution_info],
            queue=False,
            show_progress="hidden",
        )
        height_input_event = height.input(
            resolution_control_updates,
            inputs=[width, height, latent_upscale, result_format],
            outputs=[width, height, resolution_info],
            queue=False,
            show_progress="hidden",
        )
        for resolution_input_event in (width_input_event, height_input_event):
            resolution_input_event.then(
                compact_settings_summary,
                inputs=settings_inputs,
                outputs=settings_overview,
                queue=False,
                show_progress="hidden",
            )
        event = run.click(
            generate_for_ui,
            inputs=[
                mode, model_profile, generation_mode, turbo_variant,
                prompt, first, last,
                ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6,
                ref_image_7, ref_image_8, ref_image_9,
                ref_video_1, ref_video_2, ref_video_3,
                ref_audio_1, ref_audio_2, ref_audio_3,
                duration, width, height, steps, scheduler, seed, attention_mode, sol_tau,
                sol_thresh_type, sol_exact_mode, sol_dense_steps,
                sol_step_off, sol_sink_tokens,
                cache_mode, fbcache_preset, fbcache_threshold, fbcache_start,
                fbcache_end, fbcache_max_hits, fbcache_temporal_guard,
                easycache_threshold, easycache_start, easycache_end, easycache_verbose,
                ref_size, generation_postprocess,
                latent_upscale, latent_upscaler_model, latent_upscale_refine_steps,
                generation_force_offload,
                generation_split_upscale, generation_split_seconds,
                generation_seedvr2_model,
                ltx25_model,
                use_int8_vae,
                result_format,
                image_frames,
            ],
            outputs=[
                output,
                image_output_group,
                image_output,
                image_selection,
                image_frame_paths,
                image_saved_files,
                audio_output,
                image_save_status,
                status,
            ],
            show_progress="minimal",
            api_name="generate_video_advanced",
        )
        image_select_all.click(
            select_all_image_frames,
            inputs=image_frame_paths,
            outputs=image_selection,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        image_clear_selection.click(
            lambda: [],
            outputs=image_selection,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        image_save_selected.click(
            save_selected_image_frames,
            inputs=[image_frame_paths, image_selection],
            outputs=[image_saved_files, image_save_status],
            show_progress="minimal",
            api_name="save_h3_image_frames",
        )
        prompt_writer_backend.change(
            prompt_writer_backend_visibility,
            inputs=prompt_writer_backend,
            outputs=[local_prompt_writer_group, gemini_prompt_writer_group],
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        enhance_prompt_button.click(
            enhance_h3_prompt,
            inputs=[
                prompt,
                prompt_writer_backend,
                local_prompt_base_model,
                local_prompt_max_tokens,
                local_prompt_temperature,
                local_prompt_top_p,
                local_prompt_greedy,
                local_prompt_seed,
                gemini_prompt_model,
                gemini_api_key,
                mode, first, last,
                ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5,
                ref_image_6, ref_image_7, ref_image_8, ref_image_9,
                ref_video_1, ref_video_2, ref_video_3,
                ref_audio_1, ref_audio_2, ref_audio_3,
                duration, width, height,
                result_format, image_frames,
            ],
            outputs=[prompt, enhance_prompt_status],
            show_progress="minimal",
            api_name="enhance_prompt",
        )
        ltx25_enhance.click(
            enhance_ltx25_prompt,
            inputs=[
                ltx25_prompt, ltx25_prompt_model, ltx25_api_key, ltx25_mode,
                ltx25_image, ltx25_middle_image, ltx25_end_image,
                ltx25_duration, ltx25_width, ltx25_height,
            ],
            outputs=[ltx25_prompt, ltx25_enhance_status],
            show_progress="minimal",
            api_name="enhance_ltx25_prompt",
        )
        music3_enhance.click(
            enhance_music3_prompt,
            inputs=[
                music3_caption, music3_prompt_model, music3_api_key, music3_lyrics,
                music3_ref_image_1, music3_ref_image_2, music3_ref_image_3,
            ],
            outputs=[music3_caption, music3_lyrics, music3_enhance_status],
            show_progress="minimal",
            api_name="enhance_music3_prompt",
        )
        ltx25_event = ltx25_run.click(
            generate_ltx25,
            inputs=[
                ltx25_mode,
                ltx25_model,
                ltx25_prompt,
                ltx25_negative,
                ltx25_image,
                ltx25_duration,
                ltx25_fps,
                ltx25_width,
                ltx25_height,
                ltx25_seed,
                ltx25_cfg,
                ltx25_sampler,
                ltx25_image_strength,
                ltx25_middle_image,
                ltx25_middle_time,
                ltx25_middle_strength,
                ltx25_end_image,
                ltx25_end_strength,
            ],
            outputs=[ltx25_output, ltx25_status],
            show_progress="minimal",
            api_name="generate_ltx25_video",
        )
        music3_event = music3_run.click(
            generate_music3,
            inputs=[
                music3_model,
                music3_caption,
                music3_lyrics,
                music3_duration,
                music3_seed,
                music3_steps,
                music3_cfg,
                music3_ar_cfg,
                music3_top_k,
                music3_tiled,
            ],
            outputs=[music3_output, music3_status],
            show_progress="minimal",
            api_name="generate_music3",
        )
        api_event = api_run.click(
            generate_with_ui_defaults,
            inputs=api_prompt,
            outputs=[api_download_url, api_status],
            show_progress="minimal",
            api_name="generate_video",
        )
        stop.click(interrupt, outputs=status, cancels=[event])
        ltx25_stop.click(interrupt, outputs=ltx25_status, cancels=[ltx25_event])
        music3_stop.click(interrupt, outputs=music3_status, cancels=[music3_event])
        api_stop.click(interrupt, outputs=api_status, cancels=[api_event])
        refresh.click(backend_status, outputs=health)
        unload_models.click(
            unload_all_models,
            outputs=[memory_status, health],
            show_progress="minimal",
            api_name=False,
        )
        generate_tab.select(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[
                generation_view, ltx25_view, music3_view, gallery_view, api_view
            ],
        )
        ltx25_tab.select(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[
                generation_view, ltx25_view, music3_view, gallery_view, api_view
            ],
        )
        music3_tab.select(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[
                generation_view, ltx25_view, music3_view, gallery_view, api_view
            ],
        )
        gallery_event = gallery_tab.select(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "",
                None,
                False,
            ),
            outputs=[
                generation_view,
                ltx25_view,
                music3_view,
                gallery_view,
                api_view,
                gallery_player,
                gallery_download,
                gallery_selected,
                gallery_confirm_delete,
            ],
        )
        gallery_event.then(
            refresh_gallery,
            outputs=[gallery_grid, gallery_paths, gallery_status],
            show_progress="hidden",
        )
        gallery_refresh.click(
            refresh_gallery,
            outputs=[gallery_grid, gallery_paths, gallery_status],
            show_progress="minimal",
        )
        gallery_grid.select(
            select_gallery_video,
            inputs=gallery_paths,
            outputs=[gallery_player, gallery_download, gallery_selected],
            show_progress="minimal",
        )
        gallery_mutation_outputs = [
            gallery_grid,
            gallery_paths,
            gallery_status,
            gallery_player,
            gallery_download,
            gallery_selected,
            gallery_confirm_delete,
        ]
        gallery_post_event = gallery_post_run.click(
            postprocess_selected_gallery_video,
            inputs=[
                gallery_selected,
                gallery_postprocess,
                gallery_post_seed,
                gallery_seedvr2_model,
                ltx25_model,
                gallery_ltx25_prompt,
                gallery_force_offload,
                gallery_split_upscale,
                gallery_split_seconds,
            ],
            outputs=gallery_mutation_outputs + [gallery_post_status],
            show_progress="minimal",
            api_name=False,
        )
        gallery_post_stop.click(
            interrupt,
            outputs=gallery_post_status,
            cancels=[gallery_post_event],
            api_name=False,
        )
        gallery_delete.click(
            delete_selected_gallery_video,
            inputs=[gallery_selected, gallery_confirm_delete],
            outputs=gallery_mutation_outputs,
            show_progress="minimal",
            api_name=False,
        )
        gallery_empty.click(
            empty_generated_gallery,
            inputs=[gallery_selected, gallery_confirm_delete],
            outputs=gallery_mutation_outputs,
            show_progress="minimal",
            api_name=False,
        )
        api_tab.select(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            ),
            outputs=[
                generation_view, ltx25_view, music3_view, gallery_view, api_view
            ],
        )
    return demo


def selftest() -> None:
    assert MODEL_PROFILE_CHOICES == ["Speed", "Quality", "Original"]
    assert GEMINI_PROMPT_MODELS == (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    )
    with tempfile.TemporaryDirectory() as enhancer_temp:
        enhancer_root = Path(enhancer_temp)
        first_path = enhancer_root / "first.png"
        last_path = enhancer_root / "last.png"
        video_path = enhancer_root / "reference.mp4"
        for path in (first_path, last_path, video_path):
            path.write_bytes(b"test")
        last_only = _active_prompt_media(
            "First / last frame", None, str(last_path), (), (), ()
        )
        assert last_only == [("<Picture 1> (last frame)", last_path)]
        both_frames = _active_prompt_media(
            "First / last frame", str(first_path), str(last_path), (), (), ()
        )
        assert [label for label, _ in both_frames] == [
            "<Picture 1> (first frame)",
            "<Picture 2> (last frame)",
        ]
        references = _active_prompt_media(
            "Reference media",
            None,
            None,
            (None, str(first_path)),
            (str(video_path),),
            (),
        )
        assert [label for label, _ in references] == ["<Picture 2>", "<Video 1>"]
    music_graph = build_music3_graph(
        model_choice=DEFAULT_MUSIC3_MODEL,
        caption="Global Metadata: test song",
        lyrics="[Instrumental]",
        max_duration=30,
        seed=7,
        steps=30,
        cfg=1.7,
        ar_cfg=1.7,
        top_k=50,
        tiled_decode=True,
    )
    music_classes = graph_class_types(music_graph)
    assert required_music3_nodes(True) == music_classes
    music_encode = next(
        node for node in music_graph.values()
        if node["class_type"] == "MiniMaxMusic3TextEncode"
    )
    assert music_encode["inputs"]["max_duration"] == 30.0
    assert music_encode["inputs"]["top_k"] == 50
    music_save = next(
        node for node in music_graph.values()
        if node["class_type"] == "SaveAudioMP3"
    )
    assert music_save["inputs"]["quality"] == "V0"
    assert "format" not in music_save["inputs"]
    rewritten_html = _rewrite_comfy_text(
        (
            '<html><head></head><body><script src="/assets/app.js">'
            "</script></body></html>"
        ),
        "text/html; charset=utf-8",
    )
    assert '<base href="/comfyui/">' in rewritten_html
    assert 'src="/comfyui/assets/app.js"' in rewritten_html
    rewritten_css = _rewrite_comfy_text(
        'src:url("/assets/font.woff2")',
        "text/css",
    )
    assert 'url("/comfyui/assets/font.woff2")' in rewritten_css
    assert _rewrite_comfy_text("const route = '/api';", "application/javascript") == (
        "const route = '/api';"
    )
    assert _comfy_upstream_path(
        "api/userdata/workflows/LTX 2.5/example.json",
        (
            b"/comfyui/api/userdata/"
            b"workflows%2FLTX%202.5%2Fexample.json"
        ),
    ) == "api/userdata/workflows%2FLTX%202.5%2Fexample.json"
    assert _comfy_upstream_path(
        "assets/app.js",
        b"/comfyui/assets/app.js",
    ) == "assets/app.js"
    existing_base = _rewrite_comfy_text(
        '<html><head><base href="/custom/"></head></html>',
        "text/html",
    )
    assert existing_base.count("<base ") == 1
    filtered_headers = _proxy_headers(
        {
            "Connection": "keep-alive, x-private",
            "X-Private": "drop",
            "X-Test": "ok",
        }
    )
    assert filtered_headers == {"X-Test": "ok"}
    cookie_response = _append_set_cookies(
        Response(),
        httpx.Headers(
            [
                ("set-cookie", "one=1; Path=/"),
                ("set-cookie", "two=2; Path=/"),
            ]
        ),
    )
    assert cookie_response.headers.getlist("set-cookie") == [
        "one=1; Path=/",
        "two=2; Path=/",
    ]
    fake = ModelConfig(
        profiles={
            "speed": ModelProfile(
                label="Speed",
                fl2va="fl2va_speed.safetensors",
                ref2va="ref2va_speed.safetensors",
            ),
            "quality": ModelProfile(
                label="Quality",
                fl2va="fl2va_quality_convrot.safetensors",
                ref2va="ref2va_quality_convrot.safetensors",
            ),
            "original": ModelProfile(
                label="Original",
                fl2va="minimax_h3_fl2va_pruned_bf16.safetensors",
                ref2va="minimax_h3_ref2va_pruned_bf16.safetensors",
            ),
        },
        default_profile="speed",
        text_encoder="text.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
        video_vae_int8="video_vae_int8_convrot.safetensors",
        video_vae_int8_source="test",
        turbo_lora="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_source="test",
        turbo_ref_lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        turbo_ref_source="ref2v-test",
        turbo_8step_lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        turbo_8step_source="test",
        turbo_8step_ref_lora="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        turbo_8step_ref_source="shared-fl2va-test",
        larry_turbo_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_source="test",
        larry_turbo_ref_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_ref_source="shared-fl2va-test",
        seedvr2_dit="seedvr2_7b_nvfp4.safetensors",
        seedvr2_dit_source="test",
        seedvr2_models={
            "3B NVFP4": "seedvr2_3b_nvfp4.safetensors",
            "3B INT8": "seedvr2_3b_int8_convrot.safetensors",
            "7B NVFP4": "seedvr2_7b_nvfp4.safetensors",
            "7B Sharp NVFP4": "seedvr2_7b_sharp_nvfp4.safetensors",
        },
        seedvr2_vae="seedvr2_ema_vae_fp16.safetensors",
        seedvr2_vae_source="test",
    )
    available = required_nodes_for(
        "Text to video",
        True,
        "FirstBlockCache",
        use_turbo=True,
    ) | required_nodes_for("Reference media", True, "EasyCache", True)
    available |= required_nodes_for(
        "Text to video",
        False,
        "Off",
        latent_upscale=True,
    )
    available.add("SpectrumApplyMiniMaxH3")
    available.add(CHUNK_FEED_FORWARD_NODE)
    available.add(SAGE_ATTENTION_NODE)
    available |= {
        LARRY_TURBO_LORA_NODE,
        LARRY_TURBO_SAMPLER_NODE,
        LIGHTX2V_BYPASS_LORA_NODE,
    }
    assert fake.turbo_lora_for("Text to video", LIGHTX2V_4STEP_TURBO) == fake.turbo_lora
    assert fake.turbo_lora_for("Reference media", LIGHTX2V_4STEP_TURBO) == fake.turbo_ref_lora
    assert fake.turbo_lora_for("Text to video", LIGHTX2V_8STEP_TURBO) == fake.turbo_8step_lora
    assert fake.turbo_lora_for("Reference media", LIGHTX2V_8STEP_TURBO) == fake.turbo_8step_ref_lora
    assert fake.turbo_lora_for("Text to video", LARRY_TURBO) == fake.larry_turbo_lora
    reference_updates = mode_layout_updates("Reference media")
    assert reference_updates[3].get("interactive") is True
    assert "value" not in reference_updates[3]
    # Avoid staging files in selftest; build prompt-only T2V and check graph wiring.
    graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=18, seed=1,
        scheduler="simple", turbo_lora_name="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_variant=LIGHTX2V_4STEP_TURBO, turbo_strength=1.0,
        use_sol=True, sol_tau=1.0,
        sol_thresh_type="exact",
        sol_exact_mode="exact_kv_and_rows",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="FirstBlockCache",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("speed").fl2va,
        models=fake, available_nodes=available,
    )
    classes = {node["class_type"] for node in graph.values()}
    expected = {
        "MiniMaxH3ImageToVideo",
        "SamplerCustomAdvanced",
        "SaveVideo",
        SOL_ATTENTION_NODE,
        FUSED_MODULATION_NODE,
        "H3FirstBlockCache",
        CORE_LORA_LOADER_NODE,
    }
    missing = expected - classes
    if missing:
        raise SystemExit(f"Selftest failed; missing nodes: {missing}")

    image_graph = build_fl2va_graph(
        prompt="image result test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=4, seed=2,
        scheduler="simple", turbo_lora_name=None,
        turbo_variant=LIGHTX2V_4STEP_TURBO, turbo_strength=1.0,
        use_sol=False, sol_tau=1.0, sol_thresh_type="diag",
        sol_exact_mode="off", sol_dense_steps=0, sol_step_off=0.0,
        sol_sink_tokens=0, cache_mode="Off", fbcache_preset="Fast",
        fbcache_threshold=0.10, fbcache_start=0.10, fbcache_end=0.95,
        fbcache_max_hits=2, fbcache_temporal_guard=True,
        easycache_threshold=0.10, easycache_start=0.15,
        easycache_end=0.85, easycache_verbose=False,
        model_name=fake.profile("speed").fl2va, models=fake,
        available_nodes=available, result_format="Image", image_frames=20,
    )
    image_conditioning = next(
        node for node in image_graph.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    assert image_conditioning["inputs"]["length"] == 22
    assert {"ImageFromBatch", "SaveImage"} <= {
        node["class_type"] for node in image_graph.values()
    }

    assert normalize_result_format("image") == "Image"
    assert image_sampling_length(1) == 5
    assert image_sampling_length(5) == 5
    assert image_sampling_length(6) == 22
    assert image_sampling_length(20) == 22
    assert {"VAEDecode", "ImageFromBatch", "SaveImage"} <= required_nodes_for(
        "Text to video", False, "Off", result_format="Image"
    )
    assert {"VAEDecodeAudio", "SaveAudioMP3"} <= required_nodes_for(
        "Text to video", False, "Off", result_format="Audio"
    )

    image_result_graph = Graph()
    finish_sampling(
        image_result_graph,
        model_ref=["model", 0], conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0], video_vae_ref=["video_vae", 0],
        audio_vae_ref=["audio_vae", 0], seed=1, steps=4,
        scheduler="simple", turbo_variant=None,
        filename_prefix="h3/image_staging/selftest",
        result_format="Image", image_frames=3,
    )
    image_result_classes = {
        node["class_type"] for node in image_result_graph.nodes.values()
    }
    assert {"VAEDecode", "ImageFromBatch", "SaveImage"} <= image_result_classes
    assert not image_result_classes & {"VAEDecodeAudio", "CreateVideo", "SaveVideo"}
    image_slice = next(
        node for node in image_result_graph.nodes.values()
        if node["class_type"] == "ImageFromBatch"
    )
    assert image_slice["inputs"]["length"] == 3

    audio_result_graph = Graph()
    finish_sampling(
        audio_result_graph,
        model_ref=["model", 0], conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0], video_vae_ref=["video_vae", 0],
        audio_vae_ref=["audio_vae", 0], seed=1, steps=4,
        scheduler="simple", turbo_variant=None,
        filename_prefix="audio/h3_selftest", result_format="Audio",
    )
    audio_result_classes = {
        node["class_type"] for node in audio_result_graph.nodes.values()
    }
    assert {"VAEDecodeAudio", "SaveAudioMP3"} <= audio_result_classes
    assert not audio_result_classes & {"VAEDecode", "CreateVideo", "SaveVideo"}

    original_image_output_dir = globals()["OUTPUT_DIR"]
    with tempfile.TemporaryDirectory() as image_temp:
        image_root = Path(image_temp)
        staging = image_root / "h3" / "image_staging"
        staging.mkdir(parents=True)
        staged_paths = []
        image_refs = []
        for index in range(3):
            staged = staging / f"packet_{index:05d}.png"
            staged.write_bytes(f"frame-{index}".encode())
            staged_paths.append(staged)
            image_refs.append(
                {
                    "filename": staged.name,
                    "subfolder": "h3/image_staging",
                    "type": "output",
                }
            )
        globals()["OUTPUT_DIR"] = image_root
        try:
            resolved_frames = resolve_image_outputs(
                {"outputs": {"save": {"images": image_refs}}}, time.time(), 3
            )
            assert resolved_frames == staged_paths
            saved_frames, _ = save_selected_image_frames(
                resolved_frames, ["Frame 1", "Frame 3"]
            )
            assert [Path(path).name for path in saved_frames] == [
                "frame_001.png", "frame_003.png"
            ]
            assert all(Path(path).is_file() for path in saved_frames)
        finally:
            globals()["OUTPUT_DIR"] = original_image_output_dir

    original_ui_generate = globals()["generate"]

    def fake_image_generate(*_args: Any):
        yield None, "working"
        yield ["frame-a.png", "frame-b.png"], "complete"

    globals()["generate"] = fake_image_generate
    try:
        ui_updates = list(generate_for_ui("Image", 2))
    finally:
        globals()["generate"] = original_ui_generate
    assert len(ui_updates) == 2
    assert all(len(update) == 9 for update in ui_updates)
    assert ui_updates[-1][4] == ["frame-a.png", "frame-b.png"]

    latent_graph = build_fl2va_graph(
        prompt="latent upscale test", first_image=None, last_image=None,
        width=1024, height=1024, duration=5, steps=8, seed=2,
        scheduler="beta", turbo_lora_name=None,
        turbo_variant=LIGHTX2V_8STEP_TURBO, turbo_strength=1.0,
        use_sol=False, sol_tau=1.0, sol_thresh_type="diag",
        sol_exact_mode="off", sol_dense_steps=0, sol_step_off=0.0,
        sol_sink_tokens=0, cache_mode="Off", fbcache_preset="Fast",
        fbcache_threshold=0.10, fbcache_start=0.10, fbcache_end=0.95,
        fbcache_max_hits=2, fbcache_temporal_guard=True,
        easycache_threshold=0.10, easycache_start=0.15,
        easycache_end=0.85, easycache_verbose=False,
        model_name=fake.profile("speed").fl2va, models=fake,
        available_nodes=available,
        latent_upscale_model_name="minimax_h3_latent_upscaler_3d_bf16.safetensors",
        latent_upscale_precision="bf16", latent_upscale_refine_steps=2,
    )
    latent_conditioning = [
        node for node in latent_graph.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    ]
    assert {(node["inputs"]["width"], node["inputs"]["height"])
            for node in latent_conditioning} == {(512, 512), (1024, 1024)}
    latent_samplers = [
        (node_id, node) for node_id, node in latent_graph.items()
        if node["class_type"] == "SamplerCustomAdvanced"
    ]
    assert len(latent_samplers) == 2
    split_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == "SplitSigmas"
    )
    scheduler_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == "BasicScheduler"
    )
    noise_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == "RandomNoise"
    )
    assert latent_graph[split_id]["inputs"]["step"] == 6
    assert latent_samplers[0][1]["inputs"]["sigmas"] == Graph.out(scheduler_id)
    assert latent_samplers[1][1]["inputs"]["sigmas"] == Graph.out(split_id, 1)
    assert latent_samplers[0][1]["inputs"]["noise"] == Graph.out(noise_id)
    assert latent_samplers[1][1]["inputs"]["noise"] == Graph.out(noise_id)
    separate_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == H3_SEPARATE_AV_LATENT_NODE
    )
    upscaler_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == H3_LATENT_UPSCALER_NODE
    )
    combine_id = next(
        node_id for node_id, node in latent_graph.items()
        if node["class_type"] == H3_COMBINE_AV_LATENT_NODE
    )
    assert latent_graph[upscaler_id]["inputs"] == {
        "latent": Graph.out(separate_id, 0),
        "model_name": "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "scale": 2.0,
        "device": "cuda",
        "precision": "bf16",
    }
    assert latent_graph[combine_id]["inputs"]["video_latent"] == Graph.out(
        upscaler_id
    )
    assert latent_graph[combine_id]["inputs"]["audio_latent"] == Graph.out(
        separate_id, 1
    )
    initial_sampler_id = latent_samplers[0][0]
    audio_decode = next(
        node for node in latent_graph.values()
        if node["class_type"] == "VAEDecodeAudio"
    )
    assert audio_decode["inputs"]["samples"] == Graph.out(initial_sampler_id)
    assert h3_latent_upscale_dimensions(1024, 1024) == (512, 512, 1024, 1024)
    assert h3_latent_upscale_dimensions(864, 480) == (448, 256, 896, 512)

    sol_nodes = [
        node for node in graph.values()
        if node["class_type"] == SOL_ATTENTION_NODE
    ]
    assert len(sol_nodes) == 1
    assert sol_nodes[0]["inputs"]["thresh_type"] == "exact"
    assert sol_nodes[0]["inputs"]["sink_conditioning"] == "exact_kv_and_rows"
    assert sol_nodes[0]["inputs"]["dense_blocks"] == "-1"
    assert sol_nodes[0]["inputs"]["min_tokens"] == AUTO_SOL_TOKEN_THRESHOLD
    assert sol_nodes[0]["inputs"]["int8_qk"] is False

    sage_graph = Graph()
    add_model_stack(
        sage_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Off",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
        use_sage=True,
    )
    sage_nodes = [
        node for node in sage_graph.nodes.values()
        if node["class_type"] == SAGE_ATTENTION_NODE
    ]
    assert len(sage_nodes) == 1
    assert sage_nodes[0]["inputs"]["sage_attention"] == "auto"
    assert sage_nodes[0]["inputs"]["allow_compile"] is False
    assert not any(
        node["class_type"] == SOL_ATTENTION_NODE
        for node in sage_graph.nodes.values()
    )

    cache_nodes = [
        node for node in graph.values()
        if node["class_type"] == "H3FirstBlockCache"
    ]
    assert len(cache_nodes) == 1
    assert cache_nodes[0]["inputs"]["preset"] == "Fast"
    assert cache_nodes[0]["inputs"]["residual_diff_threshold"] == 0.10
    assert cache_nodes[0]["inputs"]["start_percent"] == 0.10
    assert cache_nodes[0]["inputs"]["end_percent"] == 0.95
    assert cache_nodes[0]["inputs"]["max_consecutive_cache_hits"] == 2
    assert cache_nodes[0]["inputs"]["temporal_guard"] is True

    # Turbo LoRA -> fused modulation -> FirstBlockCache -> Sol preserves every
    # wrapper's execution boundary. The inverse cache/Sol ordering reproduces
    # the runtime failure where Sol's executor.original() bypasses the cache.
    turbo_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == CORE_LORA_LOADER_NODE
    )
    fused_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == FUSED_MODULATION_NODE
    )
    cache_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == "H3FirstBlockCache"
    )
    sol_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == SOL_ATTENTION_NODE
    )
    assert graph[fused_id]["inputs"]["model"] == [turbo_id, 0]
    assert graph[fused_id]["inputs"]["enabled"] is True
    assert graph[cache_id]["inputs"]["model"] == [fused_id, 0]
    assert graph[sol_id]["inputs"]["model"] == [cache_id, 0]
    assert graph[cache_id]["inputs"]["model"] != [sol_id, 0]

    spectrum_graph = Graph()
    add_model_stack(
        spectrum_graph,
        fake.profile("quality").fl2va,
        fake,
        turbo_lora_name=None,
        turbo_variant=LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=True,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Spectrum",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
    )
    spectrum_id = next(
        node_id for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    spectrum_sol_id = next(
        node_id for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == SOL_ATTENTION_NODE
    )
    spectrum_chunk_id = next(
        node_id for node_id, node in spectrum_graph.nodes.items()
        if node["class_type"] == CHUNK_FEED_FORWARD_NODE
    )
    spectrum_inputs = spectrum_graph.nodes[spectrum_id]["inputs"]
    assert spectrum_inputs["model"] == [spectrum_chunk_id, 0]
    assert spectrum_graph.nodes[spectrum_chunk_id]["inputs"]["model"] == [
        spectrum_sol_id, 0
    ]
    assert spectrum_inputs["offline_smoothing_replay"] is True
    assert spectrum_inputs["audio_blend_weight"] == 0.0
    assert spectrum_inputs["offline_archive_storage"] == "system_ram"
    assert spectrum_inputs["model_aware_mode"] == "off"
    assert spectrum_inputs["model_aware_risk_threshold"] == 0.65
    assert not any(
        node["class_type"] == "H3FirstBlockCache"
        for node in spectrum_graph.nodes.values()
    )


    save_nodes = [node for node in graph.values() if node["class_type"] == "SaveVideo"]
    assert len(save_nodes) == 1
    assert save_nodes[0]["inputs"]["codec"] == "auto"
    assert isinstance(save_nodes[0]["inputs"]["codec"], str)

    ltx25_graph = build_ltx25_graph(
        prompt="a test shot",
        negative_prompt="artifacts",
        first_image=None,
        width=960,
        height=544,
        duration=5,
        fps=24,
        seed=9,
        cfg=1.0,
        sampler_name="euler_ancestral",
        image_strength=0.7,
    )
    ltx25_nodes = list(ltx25_graph.values())
    ltx25_classes = {node["class_type"] for node in ltx25_nodes}
    assert required_ltx25_nodes() <= ltx25_classes
    assert not ({"LoadImage", "LTXVAddGuide"} & ltx25_classes)
    ltx25_unet = next(
        node for node in ltx25_nodes if node["class_type"] == "UNETLoader"
    )
    assert ltx25_unet["inputs"]["unet_name"] == (
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    )
    ltx25_video_latent = next(
        node for node in ltx25_nodes
        if node["class_type"] == "EmptyLTXVLatentVideo"
    )
    assert ltx25_video_latent["inputs"]["length"] == 121
    ltx25_sigmas = next(
        node for node in ltx25_nodes if node["class_type"] == "ManualSigmas"
    )
    assert ltx25_sigmas["inputs"]["sigmas"] == LTX25_SIGMAS
    ltx25_save = next(
        node for node in ltx25_nodes if node["class_type"] == "SaveVideo"
    )
    assert ltx25_save["inputs"]["filename_prefix"].startswith("ltx25/")
    assert ltx25_frame_length(5, 24) == 121
    assert len(LTX25_WORKFLOWS) == 9
    assert len({entry["id"] for entry in LTX25_WORKFLOWS.values()}) == 9
    assert all(
        entry["filename"].startswith("LTX-2.5_")
        and entry["filename"].endswith(".json")
        for entry in LTX25_WORKFLOWS.values()
    )
    for label in LTX25_WORKFLOWS:
        assert set(ltx25_workflow_model_keys(label)) <= set(MODEL_SPECS)
        assert "/ltx25-workflows/" in render_ltx25_workflow_details(label)
    audio_label = next(
        label for label, entry in LTX25_WORKFLOWS.items()
        if entry.get("audio_only")
    )
    assert not {
        "ltx25_video_vae", "ltx25_video_vae_full",
    } & set(ltx25_workflow_model_keys(audio_label))
    video_label = next(
        label for label, entry in LTX25_WORKFLOWS.items()
        if not entry.get("audio_only")
    )
    assert {
        "ltx25_video_vae", "ltx25_video_vae_full",
    } <= set(ltx25_workflow_model_keys(video_label))
    inventory = render_ltx25_official_model_inventory()
    assert all(MODEL_SPECS[key].repo_id in inventory for key in LTX25_ICLORA_MODEL_KEYS)
    assert set(LTX25_ICLORA_MODEL_KEYS) <= set(ltx25_official_inventory_keys())
    original_stage_file = globals()["stage_file"]
    try:
        globals()["stage_file"] = (
            lambda path, category: f"{category}/{Path(path).name}"
        )
        ltx25_keyframe_graph = build_ltx25_graph(
            prompt="a guided test shot",
            negative_prompt="artifacts",
            first_image="start.png",
            middle_image="middle.png",
            middle_time=2.5,
            end_image="end.png",
            width=960,
            height=544,
            duration=5,
            fps=24,
            seed=10,
            cfg=1.0,
            sampler_name="euler_ancestral",
            image_strength=0.8,
            middle_strength=0.65,
            end_strength=0.9,
        )
    finally:
        globals()["stage_file"] = original_stage_file
    keyframe_nodes = ltx25_keyframe_graph.values()
    guides = [
        node for node in keyframe_nodes if node["class_type"] == "LTXVAddGuide"
    ]
    assert required_ltx25_nodes(image_to_video=True) <= {
        node["class_type"] for node in keyframe_nodes
    }
    assert [node["inputs"]["frame_idx"] for node in guides] == [0, 60, -1]
    assert [node["inputs"]["strength"] for node in guides] == [0.8, 0.65, 0.9]
    guide_ids = [
        node_id for node_id, node in ltx25_keyframe_graph.items()
        if node["class_type"] == "LTXVAddGuide"
    ]
    assert guides[1]["inputs"]["positive"] == Graph.out(guide_ids[0], 0)
    assert guides[1]["inputs"]["negative"] == Graph.out(guide_ids[0], 1)
    assert guides[1]["inputs"]["latent"] == Graph.out(guide_ids[0], 2)
    assert guides[2]["inputs"]["positive"] == Graph.out(guide_ids[1], 0)
    assert guides[2]["inputs"]["negative"] == Graph.out(guide_ids[1], 1)
    assert guides[2]["inputs"]["latent"] == Graph.out(guide_ids[1], 2)
    keyframe_guider = next(
        node for node in keyframe_nodes if node["class_type"] == "CFGGuider"
    )
    assert keyframe_guider["inputs"]["positive"] == Graph.out(guide_ids[2], 0)
    assert keyframe_guider["inputs"]["negative"] == Graph.out(guide_ids[2], 1)
    keyframe_av_latent = next(
        node for node in keyframe_nodes
        if node["class_type"] == "LTXVConcatAVLatent"
    )
    assert keyframe_av_latent["inputs"]["video_latent"] == Graph.out(
        guide_ids[2], 2
    )

    seedvr2_graph = build_seedvr2_upscale_graph(
        source_video="h3_gradio/seedvr2_upscale/source.mp4",
        seed=7,
        models=fake,
        fps=48.0,
    )
    seedvr2_nodes = list(seedvr2_graph.values())
    assert required_seedvr2_upscale_nodes() <= {
        node["class_type"] for node in seedvr2_nodes
    }
    seedvr2_scale = next(
        node for node in seedvr2_nodes if node["class_type"] == "ImageScaleBy"
    )
    assert seedvr2_scale["inputs"]["scale_by"] == 2.0
    seedvr2_vae_nodes = [
        node
        for node in seedvr2_nodes
        if node["class_type"] in {"VAEEncodeTiled", "VAEDecodeTiled"}
    ]
    assert len(seedvr2_vae_nodes) == 2
    assert all(node["inputs"]["tile_size"] == 1024 for node in seedvr2_vae_nodes)
    seedvr2_sampler = next(
        node for node in seedvr2_nodes if node["class_type"] == "KSampler"
    )
    seedvr2_chunks = next(
        node
        for node in seedvr2_nodes
        if node["class_type"] == "SeedVR2TemporalChunk"
    )
    assert seedvr2_chunks["inputs"]["chunking_mode"] == "auto"
    assert node_stage("VAEEncodeTiled", graph_class_types(seedvr2_graph)) == (
        "Encoding H3 video for SeedVR2"
    )
    assert node_stage(
        "SeedVR2TemporalChunk", graph_class_types(seedvr2_graph)
    ) == (
        "Splitting SeedVR2 video into VRAM-safe chunks"
    )
    assert seedvr2_sampler["inputs"]["steps"] == 1
    assert seedvr2_sampler["inputs"]["denoise"] == 1.0
    seedvr2_video = next(
        node for node in seedvr2_nodes if node["class_type"] == "CreateVideo"
    )
    assert seedvr2_video["inputs"]["fps"] == 48.0
    for seedvr2_choice, expected_name in fake.seedvr2_models.items():
        choice_graph = build_seedvr2_upscale_graph(
            source_video="source.mp4",
            seed=7,
            models=fake,
            model_choice=seedvr2_choice,
        )
        choice_loader = next(
            node
            for node in choice_graph.values()
            if node["class_type"] == "UNETLoader"
        )
        assert choice_loader["inputs"]["unet_name"] == expected_name

    ltx25_upscale_graph = build_ltx25_upscale_graph(
        source_video="h3_gradio/ltx25_upscale/source.mp4",
        seed=11,
        model_choice="INT8 ConvRot",
        prompt="a detailed test scene",
        width=864,
        height=480,
        fps=24.0,
    )
    ltx25_upscale_nodes = list(ltx25_upscale_graph.values())
    assert required_ltx25_upscale_nodes() <= {
        node["class_type"] for node in ltx25_upscale_nodes
    }
    ltx25_upscale_loader = next(
        node for node in ltx25_upscale_nodes
        if node["class_type"] == "LTXICLoRALoaderModelOnly"
    )
    assert ltx25_upscale_loader["inputs"]["lora_name"] == (
        "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    )
    ltx25_upscale_unet = next(
        node for node in ltx25_upscale_nodes if node["class_type"] == "UNETLoader"
    )
    assert ltx25_upscale_unet["inputs"]["unet_name"] == (
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    )
    ltx25_upscale_latent = next(
        node for node in ltx25_upscale_nodes
        if node["class_type"] == "EmptyLTXVLatentVideo"
    )
    assert ltx25_upscale_latent["inputs"]["width"] == 1728
    assert ltx25_upscale_latent["inputs"]["height"] == 960
    assert any(
        node["class_type"] == "LTXVCropGuides" for node in ltx25_upscale_nodes
    )
    assert COMFY_UPSCALE_OPTIONS == {SEEDVR2_UPSCALE, LTX25_UPSCALE}
    assert UVICORN_WEBSOCKET_OPTIONS == {
        "ws": "wsproto",
        "ws_per_message_deflate": False,
    }
    assert POSTPROCESS_OPTIONS == [
        SEEDVR2_UPSCALE, LTX25_UPSCALE, "48 fps interpolation"
    ]
    assert GENERATION_POSTPROCESS_OPTIONS == [
        "None", SEEDVR2_UPSCALE, LTX25_UPSCALE
    ]

    assert resolution_choice_values("9:16 · 768×1344", "large")[:2] == (768, 1344)
    assert resolution_choice_values("1:1 · 1024×1024", "large")[:2] == (1024, 1024)
    assert set(RESOLUTION_TIERS) == {"draft", "fast", "large"}
    assert preset_values("Quality")[0] == 20
    assert preset_values("Balanced")[0] == 18
    assert preset_values("Fast")[0] == 15
    assert all(len(values) == 6 for values in SAMPLING_PRESETS.values())
    assert preset_values("unknown") == preset_values("Balanced")
    assert UI_DEFAULTS["steps"] == turbo_steps_for(UI_DEFAULTS["turbo_variant"])
    assert UI_DEFAULTS["width"] == 864 and UI_DEFAULTS["height"] == 480
    assert 'api_name="/generate_video"' in api_guide()
    captured_free_call: dict[str, Any] = {}
    original_api_post = globals()["api_post"]
    original_backend_status = globals()["backend_status"]

    def fake_api_post(path: str, **kwargs: Any) -> None:
        captured_free_call["path"] = path
        captured_free_call["kwargs"] = kwargs

    globals()["api_post"] = fake_api_post
    globals()["backend_status"] = lambda: "refreshed backend"
    try:
        unload_comfy_models()
        unload_message, unload_status = unload_all_models()
    finally:
        globals()["api_post"] = original_api_post
        globals()["backend_status"] = original_backend_status
    assert captured_free_call == {
        "path": "/free",
        "kwargs": {"json": {"unload_models": True, "free_memory": True}},
    }
    assert unload_message == "All models unloaded and cached VRAM released."
    assert unload_status == "refreshed backend"
    captured_api_call: dict[str, Any] = {}
    original_generate = globals()["generate"]
    original_download_url = globals()["absolute_video_download_url"]

    def fake_generate(*args: Any, **kwargs: Any):
        captured_api_call["args"] = args
        captured_api_call["kwargs"] = kwargs
        yield "video.mp4", "complete"

    def fake_download_url(video: str, _request: Any) -> str:
        return f"https://example.test/downloads/{video}"

    globals()["generate"] = fake_generate
    globals()["absolute_video_download_url"] = fake_download_url
    try:
        assert list(generate_with_ui_defaults("API prompt", object())) == [
            ("https://example.test/downloads/video.mp4", "complete")
        ]
    finally:
        globals()["generate"] = original_generate
        globals()["absolute_video_download_url"] = original_download_url
    assert captured_api_call["args"] == ()
    api_kwargs = captured_api_call["kwargs"]
    assert api_kwargs["prompt"] == "API prompt"
    assert api_kwargs["mode"] == "Text to video"
    assert api_kwargs["model_profile"] == "Quality"
    assert api_kwargs["turbo_variant"] == DEFAULT_TURBO
    for key, expected in UI_DEFAULTS.items():
        assert api_kwargs[key] == expected
    assert all(api_kwargs[f"ref_image_{index}"] is None for index in range(1, 10))
    assert all(api_kwargs[f"ref_video_{index}"] is None for index in range(1, 4))
    assert all(api_kwargs[f"ref_audio_{index}"] is None for index in range(1, 4))
    assert video_download_path(OUTPUT_DIR / "h3" / "result video.mp4") == (
        "/downloads/comfy/h3/result%20video.mp4"
    )
    original_output_dir = globals()["OUTPUT_DIR"]
    original_outputs_dir = globals()["OUTPUTS_DIR"]
    original_thumbnails_dir = globals()["GALLERY_THUMBNAILS_DIR"]
    original_gallery_thumbnail = globals()["gallery_thumbnail"]
    original_gallery_video_resolution = globals()["gallery_video_resolution"]
    with tempfile.TemporaryDirectory() as gallery_temp:
        gallery_root = Path(gallery_temp)
        comfy_test_output = gallery_root / "comfy"
        gradio_test_output = gallery_root / "gradio"
        comfy_test_output.mkdir()
        gradio_test_output.mkdir()
        fallback_video = comfy_test_output / "fallback.mp4"
        fallback_video.write_bytes(b"test")
        globals()["OUTPUT_DIR"] = comfy_test_output
        globals()["OUTPUTS_DIR"] = gradio_test_output
        globals()["GALLERY_THUMBNAILS_DIR"] = gradio_test_output / ".thumbs"
        globals()["gallery_thumbnail"] = lambda _video: None
        globals()["gallery_video_resolution"] = lambda _video: (864, 480)
        try:
            h3_video = comfy_test_output / "h3" / "minimax.mp4"
            ltx25_video = comfy_test_output / "ltx25" / "ltx.mp4"
            h3_video.parent.mkdir()
            ltx25_video.parent.mkdir()
            h3_video.write_bytes(b"h3")
            ltx25_video.write_bytes(b"ltx25")
            assert len(gallery_video_paths(limit=1)) == 1
            assert len(gallery_video_paths(limit=None)) == 3
            discovered_families = {
                generated_video_family(video) for video in gallery_video_paths()
            }
            assert {"MiniMax H3", "LTX-2.5", "Post-processed"} <= discovered_families
            h3_video.unlink()
            ltx25_video.unlink()
            gallery_items, gallery_paths, gallery_detail = refresh_gallery()

            class FakeGalleryRequest:
                class request:
                    base_url = "https://example.test/"

            class FakeSelectEvent:
                index = 0

            gallery_play_url, gallery_download_link, selected_video = select_gallery_video(
                gallery_paths,
                FakeGalleryRequest(),  # type: ignore[arg-type]
                FakeSelectEvent(),  # type: ignore[arg-type]
            )
            unconfirmed_delete = delete_selected_gallery_video(
                selected_video, False
            )
            fallback_exists_after_unconfirmed = fallback_video.exists()
            confirmed_delete = delete_selected_gallery_video(
                selected_video, True
            )
            fallback_exists_after_delete = fallback_video.exists()

            empty_video_1 = comfy_test_output / "empty-1.mp4"
            empty_video_2 = gradio_test_output / "empty-2.mp4"
            empty_video_1.write_bytes(b"test")
            empty_video_2.write_bytes(b"test")
            unconfirmed_empty = empty_generated_gallery(None, False)
            empty_exists_after_unconfirmed = (
                empty_video_1.exists() and empty_video_2.exists()
            )
            confirmed_empty = empty_generated_gallery(None, True)
            empty_exists_after_delete = (
                empty_video_1.exists() or empty_video_2.exists()
            )
            try:
                managed_video_path(gallery_root / "outside.mp4", require_file=False)
                raise AssertionError("Unmanaged gallery path was accepted")
            except H3Error:
                pass
        finally:
            globals()["OUTPUT_DIR"] = original_output_dir
            globals()["OUTPUTS_DIR"] = original_outputs_dir
            globals()["GALLERY_THUMBNAILS_DIR"] = original_thumbnails_dir
            globals()["gallery_thumbnail"] = original_gallery_thumbnail
            globals()["gallery_video_resolution"] = original_gallery_video_resolution
        assert len(gallery_items) == 1
        assert "864×480" in gallery_items[0][1]
        assert gallery_paths == [str(fallback_video)]
        assert gallery_play_url == str(fallback_video)
        assert selected_video == str(fallback_video)
        assert gallery_download_link.endswith(
            "/downloads/comfy/fallback.mp4?download=1)"
        )
        assert "**Resolution:** 864×480" in gallery_download_link
        assert "1 generated video" in gallery_detail
        assert "1 thumbnail" in gallery_detail
        assert fallback_exists_after_unconfirmed is True
        assert "Confirm permanent deletion" in unconfirmed_delete[2]
        assert fallback_exists_after_delete is False
        assert "Deleted `fallback.mp4`" in confirmed_delete[2]
        assert empty_exists_after_unconfirmed is True
        assert "Confirm permanent deletion" in unconfirmed_empty[2]
        assert empty_exists_after_delete is False
        assert "Deleted 2 generated videos" in confirmed_empty[2]
    assert estimate_packed_tokens("Text to video", 1344, 768, 5) >= AUTO_SOL_TOKEN_THRESHOLD
    kitchen_policy = resolve_sol_policy(
        "Kitchen", "Text to video", 608, 352, 2, None, None
    )
    assert kitchen_policy[0] is False and kitchen_policy[2] == "forced Comfy Kitchen"
    sage_policy = resolve_sol_policy(
        "Sage 2", "Text to video", 608, 352, 2, None, None
    )
    assert sage_policy[0] is False and sage_policy[2] == "forced Sage 2"
    assert resolve_sol_policy("Auto", "Text to video", 608, 352, 2, None, None)[0] is False
    assert resolve_sol_policy(
        "Auto", "Text to video", 608, 352, 2, None, None, use_turbo=True
    )[0] is False
    reference_turbo_sol = resolve_sol_policy(
        "Auto", "Reference media", 608, 352, 2, None, None, use_turbo=True
    )
    assert reference_turbo_sol[0] is True
    assert reference_turbo_sol[2] == "Auto Turbo: reference mode"
    turbo_sol_enabled, _, turbo_sol_reason = resolve_sol_policy(
        "Auto", "Text to video", 1344, 768, 5, None, None, use_turbo=True
    )
    assert turbo_sol_enabled is True
    assert turbo_sol_reason.startswith("Auto Turbo:")
    assert validate_resolution(865, 481) == (864, 480)
    assert validate_resolution(2048, 2048) == (2048, 2048)
    assert generation_resolution(
        1344, 768, result_format="Audio", latent_upscale=False,
        mode="Text to video", first_image=None,
    ) == (32, 32)
    assert generation_resolution(
        865, 481, result_format="Video", latent_upscale=True,
        mode="Text to video", first_image=None,
    ) == (896, 512)
    auto_landscape = resolution_for_aspect_ratio(4096, 2304)
    auto_portrait = resolution_for_aspect_ratio(2304, 4096)
    assert resolution_for_aspect_ratio(1024, 1024) == (1024, 1024)
    assert auto_landscape[0] % 32 == 0 and auto_landscape[1] % 32 == 0
    assert auto_portrait[0] % 32 == 0 and auto_portrait[1] % 32 == 0
    assert auto_landscape[0] * auto_landscape[1] < 2_000_000
    assert auto_portrait[0] * auto_portrait[1] < 2_000_000
    assert abs(auto_landscape[0] / auto_landscape[1] - 16 / 9) < 0.1
    assert abs(auto_portrait[0] / auto_portrait[1] - 9 / 16) < 0.1
    assert resolution_for_aspect_ratio(
        4096, 2304, preserve_native=True
    ) == (4096, 2304)
    assert resolution_for_aspect_ratio(
        4010, 2250, preserve_native=True, alignment=64
    ) == (4032, 2240)
    assert resolution_control_updates(865, 481, True, "Video")[0:2] == (
        896, 512
    )
    assert "32×32" in resolution_control_updates(
        1344, 768, False, "Audio"
    )[2]
    with tempfile.TemporaryDirectory() as resolution_temp:
        from PIL import Image

        native_start = Path(resolution_temp) / "native-start.png"
        Image.new("RGB", (2048, 1152)).save(native_start)
        assert generation_resolution(
            864, 480, result_format="Image", latent_upscale=False,
            mode="First / last frame", first_image=str(native_start),
        ) == (2048, 1152)
        assert auto_resolution_from_start_frame(
            str(native_start), 864, 480, "Image", False
        )[:2] == (2048, 1152)
    unchanged = auto_resolution_from_start_frame(None, 640, 480)
    assert unchanged[:2] == (640, 480)
    assert frame_length(5) == 124
    assert frame_length(15) == 362
    assert websocket_url("client id").startswith("ws://")
    assert "clientId=client%20id" in websocket_url("client id")
    assert node_stage("SamplerCustomAdvanced") == "Generating video and audio"
    assert node_stage("VAEDecode") == "Decoding output"
    rendered_progress = progress_status(
        "Generating video and audio", started=time.monotonic(),
        completed_nodes=7, total_nodes=12, step=2, step_total=4,
        configured_steps=4,
    )
    assert "Sampler step 2/4 (50%)" in rendered_progress
    assert "Workflow nodes 7/12" in rendered_progress
    expanded_progress = progress_status(
        "Generating video and audio", started=time.monotonic(),
        step=3, step_total=12, configured_steps=6,
    )
    assert "Overall generation progress 3/12 (25%)" in expanded_progress
    assert "Sampling schedule 6 steps (UI setting)" in expanded_progress
    assert "Sampler step 3/12" not in expanded_progress

    class FakeProgressSocket:
        def __init__(self) -> None:
            self.messages = iter([
                json.dumps({
                    "type": "executing",
                    "data": {"prompt_id": "test-job", "node": "1"},
                }),
                json.dumps({
                    "type": "progress",
                    "data": {
                        "prompt_id": "test-job", "node": "1",
                        "value": 3, "max": 4,
                    },
                }),
                json.dumps({
                    "type": "executing",
                    "data": {"prompt_id": "test-job", "node": None},
                }),
            ])

        def settimeout(self, _timeout: float) -> None:
            pass

        def recv(self) -> str:
            return next(self.messages)

    live_updates = list(stream_comfy_progress(
        FakeProgressSocket(),  # type: ignore[arg-type]
        "test-job",
        {"1": {"class_type": "SamplerCustomAdvanced", "inputs": {}}},
        time.monotonic(),
    ))
    assert live_updates[0][0] == "Generating video and audio"
    assert live_updates[1][3:] == (3, 4)
    turbo_defaults = generation_mode_defaults("Turbo")
    assert DEFAULT_TURBO == LIGHTX2V_4STEP_TURBO
    assert turbo_defaults[1]["value"] == 4
    assert turbo_defaults[1]["interactive"] is True
    assert turbo_defaults[2:] == ("simple", "Spectrum", "Sage 2")
    larry_defaults = generation_mode_defaults("Turbo", LARRY_TURBO)
    assert larry_defaults[1]["value"] == 6
    assert larry_defaults[1]["interactive"] is True
    assert larry_defaults[2:] == ("simple", "Spectrum", "Sage 2")
    lightx_defaults = generation_mode_defaults("Turbo", LIGHTX2V_4STEP_TURBO)
    assert lightx_defaults[1]["value"] == 4
    assert lightx_defaults[1]["interactive"] is True
    assert lightx_defaults[2:] == ("simple", "Spectrum", "Sage 2")
    lightx_8step_defaults = generation_mode_defaults(
        "Turbo", LIGHTX2V_8STEP_TURBO
    )
    assert lightx_8step_defaults[1]["value"] == 8
    assert lightx_8step_defaults[1]["interactive"] is True
    assert lightx_8step_defaults[2:] == ("simple", "Spectrum", "Sage 2")
    normal_defaults = generation_mode_defaults("Normal")
    assert normal_defaults[1]["value"] == 18
    assert normal_defaults[1]["interactive"] is True
    assert normal_defaults[2:] == ("simple", "Spectrum", "Sage 2")
    assert resolve_cache_policy("Off", use_turbo=True) == ("Off", None)
    turbo_spectrum, turbo_spectrum_note = resolve_cache_policy(
        "Spectrum", use_turbo=True
    )
    assert turbo_spectrum == "Spectrum" and turbo_spectrum_note
    turbo_easycache, turbo_easycache_note = resolve_cache_policy(
        "EasyCache", use_turbo=True
    )
    assert turbo_easycache == "EasyCache" and turbo_easycache_note
    turbo_firstblock, turbo_firstblock_note = resolve_cache_policy(
        "FirstBlockCache", use_turbo=True
    )
    assert turbo_firstblock == "FirstBlockCache" and turbo_firstblock_note
    assert SERVER_DENSE_ATTENTION_BACKEND == "comfy-kitchen"

    quality_turbo_graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=8, seed=2,
        scheduler="simple",
        turbo_lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        turbo_variant=LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False, sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="Spectrum",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        model_name=fake.profile("quality").fl2va,
        models=fake, available_nodes=available,
        use_int8_vae=True,
    )
    quality_unets = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "UNETLoader"
    ]
    assert len(quality_unets) == 1
    assert quality_unets[0]["inputs"]["unet_name"] == fake.profile("quality").fl2va
    quality_video_vae = next(
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "VAELoader"
        and node["inputs"]["vae_name"] == fake.video_vae_int8
    )
    assert quality_video_vae["inputs"]["vae_name"] == fake.video_vae_int8
    turbo_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == CORE_LORA_LOADER_NODE
    ]
    assert len(turbo_nodes) == 1
    assert turbo_nodes[0]["inputs"]["strength_model"] == 1.0
    assert turbo_nodes[0]["inputs"]["lora_name"].endswith(
        "v1.0_768p_comfyui_bf16.safetensors"
    )
    quality_fused_id = next(
        node_id for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == FUSED_MODULATION_NODE
    )
    quality_turbo_id = next(
        node_id for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == CORE_LORA_LOADER_NODE
    )
    assert quality_turbo_graph[quality_fused_id]["inputs"]["model"] == [
        quality_turbo_id, 0
    ]
    chunk_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == CHUNK_FEED_FORWARD_NODE
    ]
    assert len(chunk_nodes) == 1
    assert chunk_nodes[0]["inputs"]["chunks"] == 2
    assert chunk_nodes[0]["inputs"]["min_tokens"] == AUTO_SOL_TOKEN_THRESHOLD
    lightx_spectrum_id = next(
        node_id for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    lightx_chunk_id = next(
        node_id for node_id, node in quality_turbo_graph.items()
        if node["class_type"] == CHUNK_FEED_FORWARD_NODE
    )
    assert quality_turbo_graph[lightx_spectrum_id]["inputs"]["model"] == [
        lightx_chunk_id, 0
    ]
    assert quality_turbo_graph[lightx_spectrum_id]["inputs"][
        "offline_archive_storage"
    ] == "system_ram"
    assert not any(
        node["class_type"] == LARRY_TURBO_SAMPLER_NODE
        for node in quality_turbo_graph.values()
    )

    larry_graph = Graph()
    larry_model, _, larry_video_vae, larry_audio_vae = add_model_stack(
        larry_graph,
        fake.profile("speed").fl2va,
        fake,
        turbo_lora_name=fake.larry_turbo_lora,
        turbo_variant=LARRY_TURBO,
        turbo_strength=1.0,
        use_sol=False, sol_tau=1.0, sol_thresh_type="diag",
        sol_exact_mode="off", sol_dense_steps=1, sol_step_off=0.0,
        sol_sink_tokens=0, cache_mode="Spectrum", fbcache_preset="Fast",
        fbcache_threshold=0.10, fbcache_start=0.10, fbcache_end=0.95,
        fbcache_max_hits=2, fbcache_temporal_guard=True,
        easycache_threshold=0.10, easycache_start=0.15,
        easycache_end=0.85, easycache_verbose=False,
        available_nodes=available,
    )
    finish_sampling(
        larry_graph,
        model_ref=larry_model,
        conditioning_ref=["conditioning", 0],
        latent_ref=["latent", 0],
        video_vae_ref=larry_video_vae,
        audio_vae_ref=larry_audio_vae,
        seed=3, steps=6, scheduler="simple", turbo_variant=LARRY_TURBO,
        filename_prefix="h3/larry_test",
    )
    larry_loader = next(
        node for node in larry_graph.nodes.values()
        if node["class_type"] == LARRY_TURBO_LORA_NODE
    )
    assert larry_loader["inputs"]["strength"] == 1.0
    assert larry_loader["inputs"]["low_vram"] is True
    larry_loader_id = next(
        node_id for node_id, node in larry_graph.nodes.items()
        if node["class_type"] == LARRY_TURBO_LORA_NODE
    )
    larry_spectrum_id = next(
        node_id for node_id, node in larry_graph.nodes.items()
        if node["class_type"] == "SpectrumApplyMiniMaxH3"
    )
    assert larry_graph.nodes[larry_spectrum_id]["inputs"]["model"] == [
        larry_loader_id, 0
    ]
    assert larry_model == [larry_spectrum_id, 0]
    assert larry_graph.nodes[larry_spectrum_id]["inputs"][
        "offline_archive_storage"
    ] == "system_ram"
    assert FUSED_MODULATION_NODE not in {
        node["class_type"] for node in larry_graph.nodes.values()
    }
    assert FUSED_MODULATION_NODE not in turbo_required_nodes(LARRY_TURBO)
    assert FUSED_MODULATION_NODE in turbo_required_nodes(LIGHTX2V_4STEP_TURBO)
    assert FUSED_MODULATION_NODE in turbo_required_nodes(LIGHTX2V_8STEP_TURBO)
    assert turbo_uses_custom_nodes(LARRY_TURBO) is True
    assert turbo_uses_custom_nodes(LIGHTX2V_4STEP_TURBO) is False
    assert turbo_uses_custom_nodes(LIGHTX2V_8STEP_TURBO) is False
    assert any(
        node["class_type"] == LARRY_TURBO_SAMPLER_NODE
        for node in larry_graph.nodes.values()
    )
    assert not any(
        node["class_type"] == CORE_SAMPLER_NODE
        for node in larry_graph.nodes.values()
    )

    def turbo_route_graph(profile_name: str, variant: str) -> Graph:
        route_graph = Graph()
        profile = fake.profile(profile_name)
        add_model_stack(
            route_graph,
            profile.fl2va,
            fake,
            turbo_lora_name=fake.turbo_lora_for("Text to video", variant),
            turbo_variant=variant,
            turbo_strength=turbo_strength_for(variant),
            use_sol=False,
            sol_tau=1.0,
            sol_thresh_type="diag",
            sol_exact_mode="off",
            sol_dense_steps=1,
            sol_step_off=0.0,
            sol_sink_tokens=0,
            cache_mode="Off",
            fbcache_preset="Fast",
            fbcache_threshold=0.10,
            fbcache_start=0.10,
            fbcache_end=0.95,
            fbcache_max_hits=2,
            fbcache_temporal_guard=True,
            easycache_threshold=0.10,
            easycache_start=0.15,
            easycache_end=0.85,
            easycache_verbose=False,
            available_nodes=available,
        )
        return route_graph

    for profile_name in ("speed", "quality", "original"):
        original = profile_name == "original"
        larry_route = turbo_route_graph(profile_name, LARRY_TURBO)
        larry_route_loader = next(
            node for node in larry_route.nodes.values()
            if node["class_type"] == LARRY_TURBO_LORA_NODE
        )
        assert larry_route_loader["inputs"]["low_vram"] is not original

        for lightx_variant in (
            LIGHTX2V_4STEP_TURBO,
            LIGHTX2V_8STEP_TURBO,
        ):
            lightx_route = turbo_route_graph(profile_name, lightx_variant)
            route_classes = {
                node["class_type"] for node in lightx_route.nodes.values()
            }
            expected_loader = (
                LIGHTX2V_BYPASS_LORA_NODE
                if original else CORE_LORA_LOADER_NODE
            )
            rejected_loader = (
                CORE_LORA_LOADER_NODE
                if original else LIGHTX2V_BYPASS_LORA_NODE
            )
            assert expected_loader in route_classes
            assert rejected_loader not in route_classes
            assert FUSED_MODULATION_NODE in route_classes

    original_name = fake.profile("original").fl2va
    assert is_original_bf16_model(original_name) is True
    assert LIGHTX2V_BYPASS_LORA_NODE in turbo_required_nodes(
        LIGHTX2V_4STEP_TURBO, original_name
    )
    assert CORE_LORA_LOADER_NODE not in turbo_required_nodes(
        LIGHTX2V_4STEP_TURBO, original_name
    )

    ref_turbo_graph = Graph()
    add_model_stack(
        ref_turbo_graph,
        fake.profile("quality").ref2va,
        fake,
        turbo_lora_name=fake.turbo_ref_lora,
        turbo_variant=LIGHTX2V_4STEP_TURBO,
        turbo_strength=1.0,
        use_sol=False,
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_exact_mode="off",
        sol_dense_steps=1,
        sol_step_off=0.0,
        sol_sink_tokens=0,
        cache_mode="EasyCache",
        fbcache_preset="Fast",
        fbcache_threshold=0.10,
        fbcache_start=0.10,
        fbcache_end=0.95,
        fbcache_max_hits=2,
        fbcache_temporal_guard=True,
        easycache_threshold=0.10,
        easycache_start=0.15,
        easycache_end=0.85,
        easycache_verbose=False,
        available_nodes=available,
    )
    ref_unet = next(
        node for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == "UNETLoader"
    )
    ref_lora = next(
        node for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == CORE_LORA_LOADER_NODE
    )
    assert ref_unet["inputs"]["unet_name"] == fake.profile("quality").ref2va
    assert ref_lora["inputs"]["lora_name"] == fake.turbo_ref_lora
    assert ref_lora["inputs"]["strength_model"] == 1.0
    ref_easycache = next(
        node for node in ref_turbo_graph.nodes.values()
        if node["class_type"] == "EasyCache"
    )
    assert ref_easycache["inputs"]["reuse_threshold"] == 0.10

    sched_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "BasicScheduler"
    ]
    assert len(sched_nodes) == 1
    assert sched_nodes[0]["inputs"]["steps"] == 8
    print(
        f"Selftest OK: {len(graph)} nodes, 5s=124 frames, "
        f"15s=362 frames, tiered resolution presets valid, Sol exact valid, "
        f"Sol Auto/Turbo policy valid, Spectrum default + Sol/ConvRot order valid, "
        f"zero-copy Sol + FirstBlockCache composition valid, "
        f"LightX fused modulation + Larry compatibility + ConvRot FFN chunking valid, "
        f"Spectrum v0.2.14 legacy Turbo composition + block-cache guard valid, "
        f"selectable Larry/LightX2V Turbo on "
        f"FL2VA/Ref2VA + synchronized editable Turbo steps valid, "
        f"video/image/audio result branches + image selection saving valid, "
        f"SaveVideo codec API valid, prompt API download URL valid, "
        f"gallery resolution/fallback/deletion guards + VRAM unload valid, "
        f"9 official LTX-2.5 workflow mappings valid, MiniMax Music 3 graph valid, "
        f"/comfyui proxy rewrites valid"
    )


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
        server_thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
