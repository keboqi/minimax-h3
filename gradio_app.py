#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable
from urllib.parse import quote, urlsplit, urlunsplit

import aiohttp
import gradio as gr
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
    DEFAULT_LTX25_MODEL,
    DEFAULT_SEEDVR2_MODEL,
    LTX25_MODEL_CHOICES,
    LTX25_SHARED_MODEL_KEYS,
    MIN_VALID_MODEL_BYTES,
    MODEL_SPECS,
    PROFILE_LABELS,
    SEEDVR2_MODEL_CHOICES,
    resolve_hf_token,
    stale_model_keys,
    sync_models,
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
    "SERVER_DENSE_ATTENTION_BACKEND", "pytorch"
).lower()
SERVER_MEMORY_PROFILE = os.getenv("SERVER_MEMORY_PROFILE", "unknown").lower()
AUTO_SOL_TOKEN_THRESHOLD = 8_192
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
DEFAULT_FBCACHE_PRESET = "Fast"
DEFAULT_FBCACHE_THRESHOLD = 0.10
DEFAULT_FBCACHE_START = 0.10
DEFAULT_FBCACHE_END = 0.95
DEFAULT_FBCACHE_MAX_HITS = 2
DEFAULT_FBCACHE_TEMPORAL_GUARD = True
DEFAULT_ACCELERATOR = "Spectrum"
LIGHTX2V_4STEP_TURBO = "LightX2V v1.0 / 4-step 768p"
LIGHTX2V_8STEP_TURBO = "LightX2V v1.0 / 8-step 544p"
LARRY_TURBO = "Larry v4-600 EMA"
DEFAULT_TURBO = LARRY_TURBO
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
SOL_ATTENTION_NODE = "MiniMaxH3MemoryEfficientSolAttentionPatch"
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
    "model_profile": "Quality",
    "use_int8_vae": False,
    "generation_mode": "Turbo",
    "turbo_variant": DEFAULT_TURBO,
    "duration": 5,
    "width": 864,
    "height": 480,
    "steps": 6,
    "scheduler": "simple",
    "seed": -1,
    "attention_mode": "Auto",
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
    "postprocess": "None",
    "seedvr2_model": DEFAULT_SEEDVR2_MODEL,
    "upscale_force_offload": False,
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
# entries are kept explicitly marked as extended/experimental.
NATIVE_PIXEL_CAP = 768 * 1344
EXTENDED_PIXEL_CAP = 1920 * 1088
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


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int


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


def validate_ltx25_nvfp4_header(model_choice: str) -> None:
    selected_key = LTX25_MODEL_CHOICES.get(str(model_choice))
    if selected_key != "ltx25_distilled_nvfp4":
        return

    from safetensors import safe_open

    spec = MODEL_SPECS[selected_key]
    path = COMFY_DIR / "models" / spec.folder / spec.local_name
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        has_quant_markers = any(
            key.endswith(".comfy_quant") for key in checkpoint.keys()
        )
    if not has_quant_markers:
        raise H3Error(
            "The LTX-2.5 NVFP4 checkpoint lacks ComfyUI quantization markers. "
            "Remove the stale file and retry so the Comfy-ready build is downloaded."
        )


def ensure_ltx25_models(model_choice: str = DEFAULT_LTX25_MODEL) -> bool:
    """Download the gated LTX-2.5 model set only when its tab is used."""
    required_keys = tuple(ltx25_model_keys(model_choice).values())
    if not missing_ltx25_model_names(model_choice):
        validate_ltx25_nvfp4_header(model_choice)
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
    validate_ltx25_nvfp4_header(model_choice)
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
        common.remove("ltx25_video_vae_full")
    return tuple(dict.fromkeys((*common, *entry["extra_models"])))


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
        "Prepare its models here first, then configure its visual nodes in ComfyUI."
    )


def prepare_ltx25_official_workflow(workflow_label: str):
    """Lazily fetch every checkpoint referenced by one official template."""
    try:
        required_keys = ltx25_workflow_model_keys(workflow_label)
        stale = stale_model_keys(
            root=COMFY_DIR / "models",
            manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
            model_keys=required_keys,
        )
        if not stale:
            yield f"Ready: all models for **{workflow_label}** are installed."
            return
        names = ", ".join(MODEL_SPECS[key].local_name for key in stale)
        yield f"Downloading models for **{workflow_label}**: {names}"
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
            f"Ready: installed all models for **{workflow_label}**. "
            "Open ComfyUI and load the template from **LTX 2.5**."
        )
    except Exception as exc:
        yield (
            "Error preparing official workflow models. Accept the linked "
            "Hugging Face licenses and authenticate with `hf auth login` or "
            "HF_TOKEN, then retry. "
            f"Details: {exc}"
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


def snap32(value: int | float) -> int:
    return max(32, round(int(value) / 32) * 32)


def validate_resolution(width: int | float, height: int | float) -> tuple[int, int]:
    resolved_width = snap32(width)
    resolved_height = snap32(height)
    pixels = resolved_width * resolved_height
    if pixels > EXTENDED_PIXEL_CAP:
        raise H3Error(
            f"Resolution {resolved_width}×{resolved_height} is above the UI safety "
            f"limit of 1920×1088 ({EXTENDED_PIXEL_CAP / 1_000_000:.2f} MP)."
        )
    return resolved_width, resolved_height


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
    if SERVER_ATTENTION_BACKEND != "sol":
        return False, tokens, "Sol backend unavailable"
    if requested == "dense":
        return False, tokens, "forced dense"
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
            "Auto",
        )

    return (
        gr.update(value="Balanced", interactive=True),
        gr.update(value=18, interactive=True),
        "simple",
        DEFAULT_ACCELERATOR,
        "Auto",
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
) -> None:
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
    sampled = graph.add(
        "SamplerCustomAdvanced",
        noise=Graph.out(noise),
        guider=Graph.out(guider),
        sampler=Graph.out(sampler),
        sigmas=Graph.out(sigmas),
        latent_image=latent_ref,
    )
    images = graph.add("VAEDecode", samples=Graph.out(sampled), vae=video_vae_ref)
    audio = graph.add("VAEDecodeAudio", samples=Graph.out(sampled), vae=audio_vae_ref)
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
    )

    inputs: dict[str, Any] = {
        "clip": clip_ref,
        "vae": video_vae_ref,
        "prompt": prompt,
        "width": snap32(width),
        "height": snap32(height),
        "length": frame_length(duration),
    }
    if first_image:
        loaded = graph.add("LoadImage", image=stage_file(first_image, "keyframes"))
        inputs["first_frame"] = Graph.out(loaded)
    if last_image:
        loaded = graph.add("LoadImage", image=stage_file(last_image, "keyframes"))
        inputs["last_frame"] = Graph.out(loaded)

    h3 = graph.add("MiniMaxH3ImageToVideo", **inputs)
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=Graph.out(h3, 0),
        latent_ref=Graph.out(h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        filename_prefix=f"h3/fl2va_{int(time.time())}",
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
    )

    inputs: dict[str, Any] = {
        "clip": clip_ref,
        "vae": video_vae_ref,
        "audio_vae": audio_vae_ref,
        "prompt": prompt,
        "width": snap32(width),
        "height": snap32(height),
        "length": frame_length(duration),
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

    h3 = graph.add("MiniMaxH3ReferenceToVideo", **inputs)
    finish_sampling(
        graph,
        model_ref=model_ref,
        conditioning_ref=Graph.out(h3, 0),
        latent_ref=Graph.out(h3, 1),
        video_vae_ref=video_vae_ref,
        audio_vae_ref=audio_vae_ref,
        seed=seed,
        steps=steps,
        scheduler=scheduler,
        turbo_variant=turbo_variant if turbo_lora_name else None,
        filename_prefix=f"h3/ref2va_{int(time.time())}",
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


def required_nodes_for(
    mode: str,
    use_sol: bool,
    cache_mode: str,
    use_turbo: bool = False,
    turbo_variant: str = LIGHTX2V_4STEP_TURBO,
    model_filename: str = "",
) -> set[str]:
    common = {
        "UNETLoader", "CLIPLoader", "VAELoader", "RandomNoise",
        "BasicGuider", "BasicScheduler",
        "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
        "CreateVideo", "SaveVideo",
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
        SOL_ATTENTION_NODE, FUSED_MODULATION_NODE,
        CHUNK_FEED_FORWARD_NODE, "SpectrumApplyMiniMaxH3",
        "H3FirstBlockCache", "EasyCache",
    }:
        return "Configuring generation model"
    if name in {
        "RandomNoise", "BasicGuider", "CFGGuider", CORE_SAMPLER_NODE,
        "BasicScheduler", "ManualSigmas",
    }:
        return "Preparing sampler"
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
        return "Saving video"
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
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", str(source),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise H3Error(f"Could not inspect selected video: {proc.stderr.strip()}")
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        numerator, denominator = str(rate).split("/", 1)
        fps = float(numerator) / float(denominator)
        if fps <= 0:
            raise ValueError("non-positive frame rate")
        width = int(stream["width"])
        height = int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("non-positive dimensions")
        return VideoMetadata(fps=fps, width=width, height=height)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise H3Error("Could not determine the selected video's metadata.") from exc


def unload_comfy_models() -> None:
    """Explicitly clear model residency only when the user opts into it."""
    api_post("/free", json={"unload_models": True, "free_memory": True})


def unload_all_models() -> tuple[str, str]:
    """Unload every resident ComfyUI model and refresh the backend summary."""
    try:
        unload_comfy_models()
        return "All models unloaded and cached VRAM released.", backend_status()
    except Exception as exc:
        return f"VRAM release failed: {exc}", backend_status()


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
    request: gr.Request,
    progress=gr.Progress(track_tqdm=False),
):
    """Create a new post-processed output from one selected gallery video."""
    started = time.monotonic()
    ws: websocket.WebSocket | None = None
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
        staged_source = stage_file(str(source), stage_bucket)
        if force_offload:
            yield gallery_progress_result(f"Unloading resident models before {option}")
            unload_comfy_models()

        metadata = probe_video_metadata(source)
        graph, configured_steps = build_upscale_graph(
            option=option,
            source_video=staged_source,
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
                f"**LightX2V Turbo v1.0 / 4-step 768p** · LoRA `{models.turbo_lora}` · "
                "4-step default · strength 1.0 · FL2VA and experimental Ref2VA"
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
    upscale_force_offload: bool = False,
    seedvr2_model: str = DEFAULT_SEEDVR2_MODEL,
    ltx25_model: str = DEFAULT_LTX25_MODEL,
    use_int8_vae: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    started = time.monotonic()
    queued_at = time.time()
    ws: websocket.WebSocket | None = None
    fallback_video: Path | None = None
    try:
        progress(0, desc="Validating request")
        yield None, progress_status("Validating request", started=started)
        if postprocess not in GENERATION_POSTPROCESS_OPTIONS:
            raise H3Error("Unsupported post-processing method.")
        if not prompt.strip():
            raise H3Error("Prompt is required.")
        if not 2 <= float(duration) <= 15:
            raise H3Error("Duration must be between 2 and 15 seconds.")
        resolved_width, resolved_height = validate_resolution(width, height)
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
            attention_mode, mode, resolved_width, resolved_height, float(duration),
            first_image, last_image, use_turbo=use_turbo,
        )
        effective_cache_mode, cache_note = resolve_cache_policy(
            cache_mode, use_turbo=use_turbo
        )
        missing = required_nodes_for(
            mode,
            effective_sol,
            effective_cache_mode,
            use_turbo=use_turbo,
            turbo_variant=selected_turbo,
            model_filename=selected_model,
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
                steps=effective_steps, seed=actual_seed, scheduler=effective_scheduler,
                ref_image_size=ref_image_size,
                turbo_lora_name=turbo_lora_name, turbo_variant=selected_turbo,
                turbo_strength=turbo_strength,
                use_sol=effective_sol, sol_tau=float(sol_tau),
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
            )
        else:
            graph = build_fl2va_graph(
                prompt=prompt, first_image=first_image, last_image=last_image,
                width=resolved_width, height=resolved_height, duration=float(duration),
                steps=effective_steps, seed=actual_seed, scheduler=effective_scheduler,
                turbo_lora_name=turbo_lora_name, turbo_variant=selected_turbo,
                turbo_strength=turbo_strength,
                use_sol=effective_sol, sol_tau=float(sol_tau),
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
            f"{resolved_width}×{resolved_height} · {frame_length(duration)} frames · "
            f"model {selected_label} · {effective_steps} steps/{effective_scheduler} · "
            f"attention {sol_status} ({sol_reason}; ~{packed_tokens:,} target tokens) · "
            f"dense-backend {SERVER_DENSE_ATTENTION_BACKEND} · "
            f"cache {cache_status}"
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
            staged_source = stage_file(str(source), stage_bucket)
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
            raise H3Error("Image-to-video mode requires a first frame.")
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
    postprocess: str,
    seedvr2_model: str,
    ltx25_model: str,
    force_offload: bool,
) -> str:
    model_profile = (
        f"{model_profile} / VAE: "
        f"{'INT8 ConvRot' if use_int8_vae else 'FP16'}"
    )
    if generation_mode == "Turbo":
        generation_mode = f"Turbo / {turbo_variant}"
    try:
        seconds = f"{float(duration):g}s"
    except (TypeError, ValueError):
        seconds = "—s"
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
    return (
        "**Current setup**  \n"
        f"{mode} · {model_profile} / {generation_mode} · {seconds} · {resolution} · "
        f"{step_count} / {scheduler} · Attention: {attention_mode} · "
        f"Cache: {cache_mode} · Post: {postprocess_note}"
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
        seedvr2_model=defaults["seedvr2_model"],
        ltx25_model=DEFAULT_LTX25_MODEL,
        upscale_force_offload=defaults["upscale_force_offload"],
        progress=progress,
    )
    for video, status in updates:
        download_url = (
            absolute_video_download_url(video, request) if video is not None else None
        )
        yield download_url, status


def api_guide() -> str:
    defaults = UI_DEFAULTS
    return f"""## Generate through the API

The `/generate_video` endpoint accepts a prompt and uses the same defaults as the **MiniMax H3** tab:

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

For every control exposed by the MiniMax H3 tab, use `/generate_video_advanced` and inspect the app's [OpenAPI schema](/gradio_api/openapi.json) for its current parameter list. API requests share the same single-job queue as the UI.
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
            with gr.Tab("Gallery") as gallery_tab:
                gr.HTML("")
            with gr.Tab("API") as api_tab:
                gr.HTML("")
        with gr.Row() as generation_view:
            with gr.Column(scale=3):
                mode = gr.Radio(
                    ["Text to video", "First / last frame", "Reference media"],
                    value=defaults["mode"], label="Mode",
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
                with gr.Group(visible=False) as frame_group:
                    gr.Markdown("### First / last frame inputs")
                    with gr.Row():
                        first = gr.Image(type="filepath", label="First frame")
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
                        defaults["postprocess"], defaults["seedvr2_model"],
                        DEFAULT_LTX25_MODEL,
                        defaults["upscale_force_offload"],
                    )
                )
                output = gr.Video(label="Generated video")
                with gr.Row():
                    run = gr.Button("Generate", variant="primary", scale=2)
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
                    steps = gr.Slider(
                        4, 30, value=defaults["steps"], step=1, label="Steps",
                        info=(
                            "Larry defaults to 6 steps; LightX2V variants default to their "
                            "trained 4 or 8 steps. Increase Turbo "
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
                    ["Auto", "Sol-Attn", "Dense"],
                    value=defaults["attention_mode"],
                    label="Attention",
                    interactive=SERVER_ATTENTION_BACKEND == "sol",
                    info=(
                        f"Auto enables Sol-Attn for Reference mode or when estimated "
                        f"packed target tokens reach {AUTO_SOL_TOKEN_THRESHOLD:,}; "
                        "smaller jobs stay dense. Sol forced-dense/fallback calls use "
                        f"the native {SERVER_DENSE_ATTENTION_BACKEND} backend."
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
                            "NVFP4 is the default for RTX PRO 6000/Blackwell. "
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
                    ltx25_negative = gr.Textbox(
                        label="Negative prompt",
                        lines=3,
                        placeholder="Optional artifacts or qualities to avoid",
                    )
                    with gr.Group(visible=False) as ltx25_image_group:
                        gr.Markdown(
                            "Add a required start keyframe and optional middle/end "
                            "keyframes. Each image is applied at its clip position."
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
                        "lower-memory convolutional LTX-2.5 video VAE. NVFP4 and "
                        "INT8 are embedded-quantized ComfyUI checkpoints."
                    )

            with gr.Accordion("Official advanced workflows", open=False):
                gr.Markdown(
                    "The complete official Lightricks LTX-2.5 workflow set is "
                    "installed in the bundled ComfyUI editor. These visual "
                    "workflows cover audio-only generation, two-stage refinement, "
                    "video editing, reference sheets, motion tracks, in/outpainting, "
                    "and pose/depth/canny control."
                )
                with gr.Row():
                    ltx25_workflow = gr.Dropdown(
                        choices=list(LTX25_WORKFLOWS),
                        value=next(iter(LTX25_WORKFLOWS)),
                        label="Official workflow",
                        scale=3,
                    )
                    ltx25_prepare_workflow = gr.Button(
                        "Prepare required models",
                        variant="secondary",
                        scale=1,
                    )
                ltx25_workflow_details = gr.Markdown(
                    render_ltx25_workflow_details(next(iter(LTX25_WORKFLOWS)))
                )
                ltx25_workflow_status = gr.Markdown()

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
            generation_postprocess,
            generation_seedvr2_model,
            ltx25_model,
            generation_force_offload,
        ]
        for settings_control in settings_inputs:
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
            outputs=ltx25_workflow_status,
            show_progress="minimal",
        )
        gallery_postprocess.change(
            lambda value: (
                gr.update(visible=value in COMFY_UPSCALE_OPTIONS),
                gr.update(visible=value == SEEDVR2_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
            ),
            inputs=gallery_postprocess,
            outputs=[
                gallery_ai_settings,
                gallery_seedvr2_model,
                gallery_ltx25_prompt,
            ],
            queue=False,
            show_progress="hidden",
        )
        generation_postprocess.change(
            lambda value: (
                gr.update(visible=value in COMFY_UPSCALE_OPTIONS),
                gr.update(visible=value == SEEDVR2_UPSCALE),
                gr.update(visible=value == LTX25_UPSCALE),
            ),
            inputs=generation_postprocess,
            outputs=[
                generation_postprocess_settings,
                generation_seedvr2_model,
                generation_ltx25_note,
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
        draft_resolution.change(
            lambda name: resolution_choice_values(name, "draft"),
            inputs=draft_resolution,
            outputs=[width, height, resolution_info],
        )
        fast_resolution.change(
            lambda name: resolution_choice_values(name, "fast"),
            inputs=fast_resolution,
            outputs=[width, height, resolution_info],
        )
        large_resolution.change(
            lambda name: resolution_choice_values(name, "large"),
            inputs=large_resolution,
            outputs=[width, height, resolution_info],
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
        width.change(resolution_summary, inputs=[width, height], outputs=resolution_info)
        height.change(resolution_summary, inputs=[width, height], outputs=resolution_info)
        event = run.click(
            generate,
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
                generation_force_offload, generation_seedvr2_model,
                ltx25_model,
                use_int8_vae,
            ],
            outputs=[output, status],
            show_progress="minimal",
            api_name="generate_video_advanced",
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
        api_event = api_run.click(
            generate_with_ui_defaults,
            inputs=api_prompt,
            outputs=[api_download_url, api_status],
            show_progress="minimal",
            api_name="generate_video",
        )
        stop.click(interrupt, outputs=status, cancels=[event])
        ltx25_stop.click(interrupt, outputs=ltx25_status, cancels=[ltx25_event])
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
            ),
            outputs=[generation_view, ltx25_view, gallery_view, api_view],
        )
        ltx25_tab.select(
            lambda: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[generation_view, ltx25_view, gallery_view, api_view],
        )
        gallery_event = gallery_tab.select(
            lambda: (
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
                gr.update(visible=True),
            ),
            outputs=[generation_view, ltx25_view, gallery_view, api_view],
        )
    return demo


def selftest() -> None:
    assert MODEL_PROFILE_CHOICES == ["Speed", "Quality", "Original"]
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
        turbo_ref_lora="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_ref_source="shared-fl2va-test",
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
    available.add("SpectrumApplyMiniMaxH3")
    available.add(CHUNK_FEED_FORWARD_NODE)
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
        "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
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
    turbo_defaults = generation_mode_defaults("Turbo", LARRY_TURBO)
    assert turbo_defaults[1]["value"] == 6
    assert turbo_defaults[1]["interactive"] is True
    assert turbo_defaults[2:] == ("simple", "Spectrum", "Auto")
    lightx_defaults = generation_mode_defaults("Turbo", LIGHTX2V_4STEP_TURBO)
    assert lightx_defaults[1]["value"] == 4
    assert lightx_defaults[1]["interactive"] is True
    assert lightx_defaults[2:] == ("simple", "Spectrum", "Auto")
    lightx_8step_defaults = generation_mode_defaults(
        "Turbo", LIGHTX2V_8STEP_TURBO
    )
    assert lightx_8step_defaults[1]["value"] == 8
    assert lightx_8step_defaults[1]["interactive"] is True
    assert lightx_8step_defaults[2:] == ("simple", "Spectrum", "Auto")
    normal_defaults = generation_mode_defaults("Normal")
    assert normal_defaults[1]["value"] == 18
    assert normal_defaults[1]["interactive"] is True
    assert normal_defaults[2:] == ("simple", "Spectrum", "Auto")
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
    assert SERVER_DENSE_ATTENTION_BACKEND in {"pytorch", "sage"}

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
        f"Spectrum v0.2.7 legacy Turbo composition + block-cache guard valid, "
        f"selectable Larry/LightX2V Turbo on "
        f"FL2VA/Ref2VA + synchronized editable Turbo steps valid, "
        f"SaveVideo codec API valid, prompt API download URL valid, "
        f"gallery resolution/fallback/deletion guards + VRAM unload valid, "
        f"9 official LTX-2.5 workflow mappings valid, /comfyui proxy rewrites valid"
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
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        log_level="info",
        **UVICORN_WEBSOCKET_OPTIONS,
    )


if __name__ == "__main__":
    main()
