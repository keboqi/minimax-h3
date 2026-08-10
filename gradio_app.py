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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlsplit, urlunsplit

import aiohttp
import gradio as gr
import httpx
import requests
import uvicorn
import websocket
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from h3_models import (
    LTX_UPSCALE_MODEL_KEYS,
    MIN_VALID_MODEL_BYTES,
    MODEL_SPECS,
    PROFILE_LABELS,
    SEEDVR2_UPSCALE_MODEL_KEYS,
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
ALLOW_UNSAFE_H3_COMPILE = os.getenv("ALLOW_UNSAFE_H3_COMPILE", "0") == "1"
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
LIGHTX2V_TURBO = "LightX2V v0.1"
LARRY_TURBO = "Larry v4-600 EMA"
DEFAULT_TURBO = LARRY_TURBO
MODEL_PROFILE_CHOICES = list(PROFILE_LABELS.values())
CORE_LORA_LOADER_NODE = "LoraLoaderModelOnly"
CORE_SAMPLER_NODE = "KSamplerSelect"
LARRY_TURBO_LORA_NODE = "MiniMaxH3TurboLoRA"
LARRY_TURBO_SAMPLER_NODE = "MiniMaxH3TurboSampler"
SOL_ATTENTION_NODE = "MiniMaxH3MemoryEfficientSolAttentionPatch"
FUSED_MODULATION_NODE = "MiniMaxH3FusedModulation"
CHUNK_FEED_FORWARD_NODE = "MiniMaxH3ChunkFeedForward"
LTX_UPSCALE = "LTX 2.3 generative 2x"
SEEDVR2_UPSCALE = "SeedVR2 2x"
FLASHVSR_UPSCALE = "FlashVSR 2x"
COMFY_UPSCALE_OPTIONS = {LTX_UPSCALE, SEEDVR2_UPSCALE, FLASHVSR_UPSCALE}
LTX_REFINEMENT_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
LTX_NEGATIVE_PROMPT = (
    "pc game, console game, video game, cartoon, childish, ugly, slow, "
    "slo-mo, blurry, low quality, still frame, watermark, overlay, titles, "
    "has blurbox, has subtitles"
)
TURBO_SETTINGS = {
    LARRY_TURBO: {"steps": 6, "strength": 1.0},
    LIGHTX2V_TURBO: {"steps": 4, "strength": 0.75},
}
UI_DEFAULTS = {
    "mode": "Text to video",
    "model_profile": "Quality",
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
    "compile_model": False,
    "ref_image_size": "match",
    "postprocess": "None",
    "ltx_force_offload": False,
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
    # Spectrum v0.2.5 separates the unbounded replay archive from the capped
    # causal history. Keep all replay anchors in host RAM unless a workflow
    # explicitly opts into the higher-VRAM path.
    "offline_archive_storage": "system_ram",
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

SAMPLING_PRESETS: dict[str, tuple[int, bool, float, str, str, str, int]] = {
    "Quality": (20, False, 0.8, "exact", "beta", "exact_kv_and_rows", 1),
    "Balanced": (18, False, 1.0, "diag", "simple", "exact_kv", 1),
    "Fast": (15, False, 1.2, "diag", "simple", "off", 1),
}
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
GENERATION_TIMEOUT = float(os.getenv("GENERATION_TIMEOUT", "10800"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1"))
OUTPUTS_DIR = Path(os.getenv("GRADIO_OUTPUT_DIR", COMFY_DIR.parent / "gradio_outputs"))
GALLERY_THUMBNAILS_DIR = OUTPUTS_DIR / ".gallery_thumbnails"
GALLERY_LIMIT = max(1, int(os.getenv("GRADIO_GALLERY_LIMIT", "200")))
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".gif"})

HTTP = requests.Session()

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


async def _close_websocket(socket: WebSocket, code: int = 1000) -> None:
    try:
        await socket.close(code=code)
    except RuntimeError:
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
    app = FastAPI()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=30),
        follow_redirects=False,
        trust_env=False,
    )

    @app.on_event("shutdown")
    async def close_proxy_client() -> None:
        await client.aclose()

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
        target = f"{COMFY_URL}/{quote(path, safe='/:@')}"
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
        upstream_url += f"/{quote(path, safe='/:@')}{query}"
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
        except Exception as exc:
            print(f"[h3-ui] ComfyUI websocket proxy error: {exc}", flush=True)
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


def normalize_turbo_variant(value: str) -> str:
    return value if value in TURBO_SETTINGS else DEFAULT_TURBO


def turbo_steps_for(value: str) -> int:
    return int(TURBO_SETTINGS[normalize_turbo_variant(value)]["steps"])


def turbo_strength_for(value: str) -> float:
    return float(TURBO_SETTINGS[normalize_turbo_variant(value)]["strength"])


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
    turbo_lora: str | None = None
    turbo_source: str = "unknown"
    turbo_ref_lora: str | None = None
    turbo_ref_source: str = "unknown"
    larry_turbo_lora: str | None = None
    larry_turbo_source: str = "unknown"
    larry_turbo_ref_lora: str | None = None
    larry_turbo_ref_source: str = "unknown"
    ltx_checkpoint: str | None = None
    ltx_checkpoint_source: str = "unknown"
    ltx_distilled_lora: str | None = None
    ltx_distilled_lora_source: str = "unknown"
    ltx_text_encoder: str | None = None
    ltx_text_encoder_source: str = "unknown"
    ltx_spatial_upscaler: str | None = None
    ltx_spatial_upscaler_source: str = "unknown"
    seedvr2_dit: str | None = None
    seedvr2_dit_source: str = "unknown"
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
        if normalize_turbo_variant(turbo_variant) == LARRY_TURBO:
            return self.larry_turbo_ref_lora if reference else self.larry_turbo_lora
        return self.turbo_ref_lora if reference else self.turbo_lora


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
        turbo_lora=data.get("turbo_lora"),
        turbo_source=data.get("turbo_source", "unknown"),
        turbo_ref_lora=data.get("turbo_ref_lora", data.get("turbo_lora")),
        turbo_ref_source=data.get(
            "turbo_ref_source", data.get("turbo_source", "unknown")
        ),
        larry_turbo_lora=data.get("larry_turbo_lora"),
        larry_turbo_source=data.get("larry_turbo_source", "unknown"),
        larry_turbo_ref_lora=data.get(
            "larry_turbo_ref_lora", data.get("larry_turbo_lora")
        ),
        larry_turbo_ref_source=data.get(
            "larry_turbo_ref_source", data.get("larry_turbo_source", "unknown")
        ),
        ltx_checkpoint=data.get("ltx_checkpoint"),
        ltx_checkpoint_source=data.get("ltx_checkpoint_source", "unknown"),
        ltx_distilled_lora=data.get("ltx_distilled_lora"),
        ltx_distilled_lora_source=data.get("ltx_distilled_lora_source", "unknown"),
        ltx_text_encoder=data.get("ltx_text_encoder"),
        ltx_text_encoder_source=data.get("ltx_text_encoder_source", "unknown"),
        ltx_spatial_upscaler=data.get("ltx_spatial_upscaler"),
        ltx_spatial_upscaler_source=data.get("ltx_spatial_upscaler_source", "unknown"),
        seedvr2_dit=data.get("seedvr2_dit"),
        seedvr2_dit_source=data.get("seedvr2_dit_source", "unknown"),
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
    destination = COMFY_DIR / "models" / "diffusion_models" / filename
    if model_file_is_ready(destination):
        return False

    model_key = f"{profile_key}_{'ref2va' if reference else 'fl2va'}"
    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
        token=os.getenv("HF_TOKEN") or None,
        log_prefix="[h3-on-demand]",
        model_keys=(model_key,),
        download_workers=1,
    )
    if not model_file_is_ready(destination):
        raise H3Error(f"On-demand model download did not produce {filename}.")
    return True


def ltx_upscale_model_names(models: ModelConfig) -> dict[str, str]:
    """Return a validated LTX asset map from the generated model config."""
    configured = {
        "ltx_checkpoint": models.ltx_checkpoint,
        "ltx_distilled_lora": models.ltx_distilled_lora,
        "ltx_text_encoder": models.ltx_text_encoder,
        "ltx_spatial_upscaler": models.ltx_spatial_upscaler,
    }
    missing_config = [key for key, value in configured.items() if not value]
    if missing_config:
        raise H3Error(
            "The model configuration predates LTX upscale support. Re-run "
            "setup_h3.py before selecting LTX upscale. Missing keys: "
            + ", ".join(missing_config)
        )
    return {key: str(value) for key, value in configured.items()}


def ensure_ltx_upscale_models(models: ModelConfig) -> bool:
    """Download the complete LTX upscale stack only when it is selected."""
    configured = ltx_upscale_model_names(models)

    missing_files = []
    for key in LTX_UPSCALE_MODEL_KEYS:
        spec = MODEL_SPECS[key]
        path = COMFY_DIR / "models" / spec.folder / configured[key]
        if not model_file_is_ready(path):
            missing_files.append(key)
    if not missing_files:
        return False

    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
        token=os.getenv("HF_TOKEN") or None,
        log_prefix="[ltx-on-demand]",
        model_keys=LTX_UPSCALE_MODEL_KEYS,
        download_workers=len(LTX_UPSCALE_MODEL_KEYS),
    )
    for key in LTX_UPSCALE_MODEL_KEYS:
        spec = MODEL_SPECS[key]
        path = COMFY_DIR / "models" / spec.folder / configured[key]
        if not model_file_is_ready(path):
            raise H3Error(f"On-demand LTX download did not produce {configured[key]}.")
    return True


def seedvr2_upscale_model_names(models: ModelConfig) -> dict[str, str]:
    configured = {
        "seedvr2_dit": models.seedvr2_dit,
        "seedvr2_vae": models.seedvr2_vae,
    }
    missing_config = [key for key, value in configured.items() if not value]
    if missing_config:
        raise H3Error(
            "The model configuration predates SeedVR2 support. Re-run "
            "setup_h3.py before selecting SeedVR2. Missing keys: "
            + ", ".join(missing_config)
        )
    return {key: str(value) for key, value in configured.items()}


def ensure_seedvr2_upscale_models(models: ModelConfig) -> bool:
    configured = seedvr2_upscale_model_names(models)
    missing_files = []
    for key in SEEDVR2_UPSCALE_MODEL_KEYS:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(COMFY_DIR / "models" / spec.folder / configured[key]):
            missing_files.append(key)
    if not missing_files:
        return False

    sync_models(
        root=COMFY_DIR / "models",
        manifest_path=MODELS_CONFIG.parent / "h3_model_manifest.json",
        token=os.getenv("HF_TOKEN") or None,
        log_prefix="[seedvr2-on-demand]",
        model_keys=SEEDVR2_UPSCALE_MODEL_KEYS,
        download_workers=len(SEEDVR2_UPSCALE_MODEL_KEYS),
    )
    for key in SEEDVR2_UPSCALE_MODEL_KEYS:
        spec = MODEL_SPECS[key]
        if not model_file_is_ready(COMFY_DIR / "models" / spec.folder / configured[key]):
            raise H3Error(
                f"On-demand SeedVR2 download did not produce {configured[key]}."
            )
    return True


def is_quantized_h3_model(filename: str) -> bool:
    name = filename.lower()
    quantized_markers = (
        "nvfp4", "int8", "convrot", "fp8", "awq", "int4", "w4a",
    )
    return any(marker in name for marker in quantized_markers)


def resolve_compile_request(
    requested: bool,
    model_filename: str,
) -> tuple[bool, str | None]:
    if not requested:
        return False, None
    if is_quantized_h3_model(model_filename) and not ALLOW_UNSAFE_H3_COMPILE:
        return (
            False,
            "torch.compile was automatically disabled because the selected "
            f"H3 model is quantized ({model_filename}). Full-model Dynamo "
            "capture currently fails on ComfyUI's quantized tensor wrappers. "
            "Sol-Attn remains enabled independently.",
        )
    return True, None


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
    """Allow Spectrum for Turbo while rejecting unvalidated block caches."""
    requested = str(cache_mode).strip()
    normalized = requested.lower()
    if not use_turbo or normalized == "off":
        return requested, None
    if normalized == "spectrum":
        return requested, (
            "Spectrum forecasting with Turbo is experimental. Compare the same "
            "prompt and seed with acceleration Off for quality-critical output."
        )
    return "Off", (
        f"{requested or 'Selected block cache'} was disabled automatically: "
        "Turbo currently permits only the validated Spectrum v0.2.5 sampler path."
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


def turbo_required_nodes(turbo_variant: str) -> set[str]:
    """Return the external node contract for one normalized Turbo variant."""
    if normalize_turbo_variant(turbo_variant) == LARRY_TURBO:
        return {LARRY_TURBO_LORA_NODE, LARRY_TURBO_SAMPLER_NODE}
    return {CORE_LORA_LOADER_NODE, CORE_SAMPLER_NODE, FUSED_MODULATION_NODE}


def add_turbo_model_patch(
    graph: Graph,
    model_ref: list[Any],
    *,
    lora_name: str,
    turbo_variant: str,
    strength: float,
    available_nodes: set[str],
) -> list[Any]:
    """Apply a Turbo LoRA and compatible model-level optimizations."""
    variant = normalize_turbo_variant(turbo_variant)
    required = turbo_required_nodes(variant)
    missing = required - available_nodes
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise H3Error(
            f"{variant} Turbo requires unavailable nodes: {missing_names}. "
            "Re-run setup_h3.py and restart ComfyUI."
        )

    if variant == LARRY_TURBO:
        turbo = graph.add(
            LARRY_TURBO_LORA_NODE,
            model=model_ref,
            lora_name=lora_name,
            strength=float(strength),
            low_vram=False,
        )
        # Larry replaces AdaLN projection forwards at runtime. Its pinned node
        # receives a provisioning-time modality-row fix; keep this additional
        # fusion layer disabled until that composition has GPU validation.
        # Sol attention and FFN chunking do not replace the AdaLN projection.
        return Graph.out(turbo)

    turbo = graph.add(
        CORE_LORA_LOADER_NODE,
        model=model_ref,
        lora_name=lora_name,
        strength_model=float(strength),
    )

    # LightX2V uses the core LoRA loader and retains the standard H3 AdaLN shape,
    # so Sol's bit-exact fused modulation remains compatible.
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
    compile_model: bool,
    available_nodes: set[str],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    unet = graph.add("UNETLoader", unet_name=model_name, weight_dtype="default")
    model_ref = Graph.out(unet)

    if turbo_lora_name:
        model_ref = add_turbo_model_patch(
            graph,
            model_ref,
            lora_name=turbo_lora_name,
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

    if compile_model:
        if "TorchCompileModel" not in available_nodes:
            raise H3Error("TorchCompileModel is unavailable; update ComfyUI.")
        compiled = graph.add("TorchCompileModel", model=model_ref, backend="inductor")
        model_ref = Graph.out(compiled)

    clip = graph.add(
        "CLIPLoader",
        clip_name=models.text_encoder,
        type="minimax",
        device="default",
    )
    video_vae = graph.add("VAELoader", vae_name=models.video_vae)
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
        and normalize_turbo_variant(turbo_variant) == LARRY_TURBO
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
    compile_model: bool,
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
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
        compile_model=compile_model,
        available_nodes=available_nodes,
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
    compile_model: bool,
    model_name: str,
    models: ModelConfig,
    available_nodes: set[str],
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
        compile_model=compile_model,
        available_nodes=available_nodes,
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


def required_ltx_upscale_nodes() -> set[str]:
    return {
        "LoadVideo",
        "GetVideoComponents",
        "ImageFromBatch",
        "CheckpointLoaderSimple",
        CORE_LORA_LOADER_NODE,
        "LTXAVTextEncoderLoader",
        "LTXVAudioVAELoader",
        "CLIPTextEncode",
        "LTXVConditioning",
        "VAEEncodeTiled",
        "LatentUpscaleModelLoader",
        "LTXVLatentUpsampler",
        "LTXVImgToVideoInplace",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "RandomNoise",
        "CFGGuider",
        CORE_SAMPLER_NODE,
        "ManualSigmas",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "VAEDecodeTiled",
        "CreateVideo",
        "SaveVideo",
    }


def build_ltx_upscale_graph(
    *,
    source_video: str,
    prompt: str,
    frame_count: int,
    seed: int,
    models: ModelConfig,
) -> dict[str, Any]:
    """Build the native LTX 2.3 decode/encode/x2/refine post-process graph."""
    assets = ltx_upscale_model_names(models)

    graph = Graph()
    loaded_video = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded_video))
    ltx_frame_count = max(1, ((int(frame_count) - 1) // 8) * 8 + 1)
    source_frames = graph.add(
        "ImageFromBatch",
        image=Graph.out(components, 0),
        batch_index=0,
        length=ltx_frame_count,
    )
    source_images = Graph.out(source_frames)
    first_frame = graph.add(
        "ImageFromBatch", image=source_images, batch_index=0, length=1
    )

    checkpoint = graph.add(
        "CheckpointLoaderSimple", ckpt_name=assets["ltx_checkpoint"]
    )
    ltx_model = graph.add(
        CORE_LORA_LOADER_NODE,
        model=Graph.out(checkpoint, 0),
        lora_name=assets["ltx_distilled_lora"],
        strength_model=0.5,
    )
    text_encoder = graph.add(
        "LTXAVTextEncoderLoader",
        text_encoder=assets["ltx_text_encoder"],
        ckpt_name=assets["ltx_checkpoint"],
        device="default",
    )
    positive = graph.add(
        "CLIPTextEncode", clip=Graph.out(text_encoder), text=prompt
    )
    negative = graph.add(
        "CLIPTextEncode",
        clip=Graph.out(text_encoder),
        text=LTX_NEGATIVE_PROMPT,
    )
    conditioning = graph.add(
        "LTXVConditioning",
        positive=Graph.out(positive),
        negative=Graph.out(negative),
        frame_rate=24.0,
    )

    video_latent = graph.add(
        "VAEEncodeTiled",
        pixels=source_images,
        vae=Graph.out(checkpoint, 2),
        tile_size=768,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )
    upscale_model = graph.add(
        "LatentUpscaleModelLoader", model_name=assets["ltx_spatial_upscaler"]
    )
    upscaled = graph.add(
        "LTXVLatentUpsampler",
        samples=Graph.out(video_latent),
        upscale_model=Graph.out(upscale_model),
        vae=Graph.out(checkpoint, 2),
    )
    guided_video = graph.add(
        "LTXVImgToVideoInplace",
        vae=Graph.out(checkpoint, 2),
        image=Graph.out(first_frame),
        latent=Graph.out(upscaled),
        strength=1.0,
        bypass=False,
    )

    audio_vae = graph.add(
        "LTXVAudioVAELoader", ckpt_name=assets["ltx_checkpoint"]
    )
    empty_audio = graph.add(
        "LTXVEmptyLatentAudio",
        audio_vae=Graph.out(audio_vae),
        frames_number=ltx_frame_count,
        frame_rate=24.0,
        batch_size=1,
    )
    av_latent = graph.add(
        "LTXVConcatAVLatent",
        video_latent=Graph.out(guided_video),
        audio_latent=Graph.out(empty_audio),
    )
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add(
        "CFGGuider",
        model=Graph.out(ltx_model),
        positive=Graph.out(conditioning, 0),
        negative=Graph.out(conditioning, 1),
        cfg=1.0,
    )
    sampler = graph.add(CORE_SAMPLER_NODE, sampler_name="euler_cfg_pp")
    sigmas = graph.add("ManualSigmas", sigmas=LTX_REFINEMENT_SIGMAS)
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
        vae=Graph.out(checkpoint, 2),
        tile_size=768,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(images),
        audio=Graph.out(components, 1),
        fps=24.0,
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"ltx/upscale_{int(time.time())}",
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
    *, source_video: str, seed: int, models: ModelConfig
) -> dict[str, Any]:
    """Build ComfyUI's native one-step SeedVR2 3B INT8 2x workflow."""
    assets = seedvr2_upscale_model_names(models)
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
    latent = graph.add(
        "VAEEncodeTiled",
        pixels=Graph.out(prepared),
        vae=Graph.out(vae),
        tile_size=512,
        overlap=128,
        temporal_size=64,
        temporal_overlap=8,
    )
    chunks = graph.add(
        "SeedVR2TemporalChunk",
        latent=Graph.out(latent),
        temporal_overlap=1,
        chunking_mode={"chunking_mode": "auto"},
    )
    conditioning = graph.add(
        "SeedVR2Conditioning",
        model=Graph.out(model),
        vae_conditioning=Graph.out(chunks, 0),
    )
    sampled = graph.add(
        "KSampler",
        model=Graph.out(model),
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
        tile_size=512,
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
        fps=24.0,
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


def required_flashvsr_upscale_nodes() -> set[str]:
    return {"LoadVideo", "GetVideoComponents", "AILab_FlashVSR", "CreateVideo", "SaveVideo"}


def build_flashvsr_upscale_graph(*, source_video: str, seed: int) -> dict[str, Any]:
    """Build the balanced FlashVSR v1.1 2x tiled workflow."""
    graph = Graph()
    loaded = graph.add("LoadVideo", file=source_video)
    components = graph.add("GetVideoComponents", video=Graph.out(loaded))
    upscaled = graph.add(
        "AILab_FlashVSR",
        frames=Graph.out(components, 0),
        audio=Graph.out(components, 1),
        preset="Balanced (2x Quality)",
        scale=2,
        unload_model=True,
        seed=max(1, int(seed)),
    )
    video = graph.add(
        "CreateVideo",
        images=Graph.out(upscaled, 0),
        audio=Graph.out(upscaled, 1),
        fps=24.0,
        bit_depth=8,
    )
    graph.add(
        "SaveVideo",
        video=Graph.out(video),
        filename_prefix=f"flashvsr/upscale_{int(time.time())}",
        format="auto",
        codec="auto",
    )
    return graph.nodes


def required_nodes_for(
    mode: str,
    use_sol: bool,
    cache_mode: str,
    compile_model: bool,
    use_turbo: bool = False,
    turbo_variant: str = LIGHTX2V_TURBO,
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
        common |= turbo_required_nodes(turbo_variant)
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
    if compile_model:
        common.add("TorchCompileModel")
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


def node_stage(class_type: str) -> str:
    """Turn ComfyUI implementation node names into useful user-facing stages."""
    name = str(class_type)
    if name in {
        "UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple",
        "LTXAVTextEncoderLoader", "LTXVAudioVAELoader",
        "LatentUpscaleModelLoader", CORE_LORA_LOADER_NODE,
        LARRY_TURBO_LORA_NODE,
    }:
        return "Loading models"
    if name.startswith("Load") or name == "GetVideoComponents":
        return "Preparing reference media"
    if name == "VAEEncodeTiled":
        return "Encoding H3 video for LTX"
    if name == "LTXVLatentUpsampler":
        return "Upscaling LTX video latent"
    if "ImageToVideo" in name or "ReferenceToVideo" in name:
        return "Encoding prompt and conditioning"
    if name in {
        "TorchCompileModel", SOL_ATTENTION_NODE, FUSED_MODULATION_NODE,
        CHUNK_FEED_FORWARD_NODE, "SpectrumApplyMiniMaxH3",
        "H3FirstBlockCache", "EasyCache",
    }:
        return "Configuring generation model"
    if name in {
        "RandomNoise", "BasicGuider", CORE_SAMPLER_NODE, "BasicScheduler",
    }:
        return "Preparing sampler"
    if name == "SamplerCustomAdvanced" or "Sampler" in name:
        return "Generating video and audio"
    if name in {"VAEDecode", "VAEDecodeAudio", "VAEDecodeTiled"}:
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
            yield node_stage(class_type), len(completed), total_nodes, None, None
            continue
        if event_type == "progress":
            node_id = str(data.get("node") or current_node or "")
            class_type = graph.get(node_id, {}).get("class_type", "Processing")
            value = int(data.get("value", 0))
            maximum = int(data.get("max", 0))
            yield node_stage(class_type), len(completed), total_nodes, value, maximum
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
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUTS_DIR / f"{source.stem}_{uuid.uuid4().hex[:8]}.mp4"
    filters: list[str] = []
    if "2×" in option:
        filters.append("scale=iw*2:ih*2:flags=lanczos")
    if "48 fps" in option:
        filters.append("minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
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


def unload_comfy_models() -> None:
    """Explicitly clear model residency only when the user opts into it."""
    api_post("/free", json={"unload_models": True, "free_memory": True})


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


def gallery_video_paths() -> list[Path]:
    """Return the newest generated videos without loading their media data."""
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

    return sorted(videos.values(), key=modified, reverse=True)[:GALLERY_LIMIT]


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
        caption = f"{video.name} · {timestamp} · {size_mb:.1f} MB"
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
    # Return the local path to gr.Video. Gradio treats arbitrary HTTP URLs as
    # remote fetches and can reject its own public hostname during validation.
    return video, f"[Download video]({download_url})", video


GalleryMutationResult = tuple[
    list[tuple[str, str]],
    list[str],
    str,
    Any,
    Any,
    str | None,
    bool,
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
    for candidate in gallery_video_paths():
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
        vram = device.get("vram_total")
        vram_text = f" · {vram / 2**30:.1f} GiB" if isinstance(vram, (int, float)) else ""
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
                f"**LightX2V Turbo** · LoRA `{models.turbo_lora}` · "
                "4-step default · strength 0.75 · FL2VA and experimental Ref2VA"
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
    compile_model: bool,
    ref_image_size: str,
    postprocess: str,
    ltx_force_offload: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    started = time.monotonic()
    queued_at = time.time()
    ws: websocket.WebSocket | None = None
    fallback_video: Path | None = None
    try:
        progress(0, desc="Validating request")
        yield None, progress_status("Validating request", started=started)
        if not prompt.strip():
            raise H3Error("Prompt is required.")
        if not 2 <= float(duration) <= 15:
            raise H3Error("Duration must be between 2 and 15 seconds.")
        resolved_width, resolved_height = validate_resolution(width, height)
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        models = load_model_config()
        if postprocess == LTX_UPSCALE:
            ltx_upscale_model_names(models)
        elif postprocess == SEEDVR2_UPSCALE:
            seedvr2_upscale_model_names(models)
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

        effective_compile, compile_note = resolve_compile_request(
            bool(compile_model), selected_model
        )

        info = object_info()
        available = set(info)
        if postprocess == LTX_UPSCALE:
            missing_ltx_nodes = required_ltx_upscale_nodes() - available
            if missing_ltx_nodes:
                raise H3Error(
                    "LTX upscale requires newer native ComfyUI nodes: "
                    + ", ".join(sorted(missing_ltx_nodes))
                )
        elif postprocess == SEEDVR2_UPSCALE:
            missing_seedvr2_nodes = required_seedvr2_upscale_nodes() - available
            if missing_seedvr2_nodes:
                raise H3Error(
                    "SeedVR2 requires current native ComfyUI nodes: "
                    + ", ".join(sorted(missing_seedvr2_nodes))
                )
        elif postprocess == FLASHVSR_UPSCALE:
            missing_flashvsr_nodes = required_flashvsr_upscale_nodes() - available
            if missing_flashvsr_nodes:
                raise H3Error(
                    "FlashVSR is not installed or failed to load. Re-run setup_h3.py. "
                    "Missing nodes: " + ", ".join(sorted(missing_flashvsr_nodes))
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
            effective_compile,
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
                compile_model=effective_compile, model_name=selected_model,
                models=models, available_nodes=available,
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
                compile_model=effective_compile, model_name=selected_model,
                models=models, available_nodes=available,
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
            f"cache {cache_status} · "
            f"compile {'on' if effective_compile else 'off'}"
        )
        if compile_note:
            queued_status += f"\n\nCompatibility notice: {compile_note}"
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
            if postprocess == LTX_UPSCALE:
                yield None, progress_status("Checking LTX 2.3 upscale models", started=started)
                downloaded = ensure_ltx_upscale_models(models)
                stage_bucket = "ltx_upscale"
            elif postprocess == SEEDVR2_UPSCALE:
                yield None, progress_status("Checking SeedVR2 upscale models", started=started)
                downloaded = ensure_seedvr2_upscale_models(models)
                stage_bucket = "seedvr2_upscale"
            else:
                downloaded = False
                stage_bucket = "flashvsr_upscale"

            if downloaded:
                yield None, progress_status(f"{postprocess} models downloaded", started=started)
            staged_source = stage_file(str(source), stage_bucket)
            unload_h3_before_upscale = bool(ltx_force_offload)
            if unload_h3_before_upscale:
                progress(0, desc="Unloading H3 models")
                yield None, progress_status(
                    f"Unloading H3 models before {postprocess}", started=started
                )
                unload_comfy_models()

            if postprocess == LTX_UPSCALE:
                upscale_graph = build_ltx_upscale_graph(
                    source_video=staged_source, prompt=prompt,
                    frame_count=frame_length(duration), seed=actual_seed, models=models,
                )
                configured_upscale_steps = 3
            elif postprocess == SEEDVR2_UPSCALE:
                upscale_graph = build_seedvr2_upscale_graph(
                    source_video=staged_source, seed=actual_seed, models=models
                )
                configured_upscale_steps = 1
            else:
                upscale_graph = build_flashvsr_upscale_graph(
                    source_video=staged_source, seed=actual_seed
                )
                configured_upscale_steps = 1

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
    ltx_force_offload: bool,
) -> str:
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
    offload_note = (
        f" · H3 unload: {'on' if ltx_force_offload else 'off'}"
        if postprocess in COMFY_UPSCALE_OPTIONS
        else ""
    )
    return (
        "**Current setup**  \n"
        f"{mode} · {model_profile} / {generation_mode} · {seconds} · {resolution} · "
        f"{step_count} / {scheduler} · Attention: {attention_mode} · "
        f"Cache: {cache_mode} · Post: {postprocess}{offload_note}"
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
        compile_model=defaults["compile_model"],
        ref_image_size=defaults["ref_image_size"],
        postprocess=defaults["postprocess"],
        ltx_force_offload=defaults["ltx_force_offload"],
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

The `/generate_video` endpoint accepts a prompt and uses the same defaults as the **Generate** tab:

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

For every control exposed by the Generate tab, use `/generate_video_advanced` and inspect the app's [OpenAPI schema](/gradio_api/openapi.json) for its current parameter list. API requests share the same single-job queue as the UI.
"""


def build_ui() -> gr.Blocks:
    sol_default = SERVER_ATTENTION_BACKEND == "sol"
    defaults = UI_DEFAULTS
    with gr.Blocks(title="MiniMax H3 Local") as demo:
        gr.Markdown("# MiniMax H3 Local\nNative ComfyUI graphs for T2V, first/last-frame video, and reference media.")
        gr.HTML(
            '<a href="/comfyui/" target="_blank" rel="noopener noreferrer">'
            "Open ComfyUI ↗</a>"
        )
        health = gr.Markdown(backend_status())
        with gr.Tabs():
            with gr.Tab("Generate") as generate_tab:
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
                        defaults["generation_mode"], defaults["turbo_variant"],
                        defaults["duration"], defaults["width"], defaults["height"],
                        defaults["steps"], defaults["scheduler"],
                        defaults["attention_mode"], defaults["cache_mode"],
                        defaults["postprocess"], defaults["ltx_force_offload"],
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
                        "Larry uses its quantization-aware loader and adaptive sampler at "
                        "strength 1.0; LightX2V uses the native LoRA loader at strength 0.75."
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
                            "Larry defaults to 6 steps and LightX2V to 4. Increase Turbo "
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
                            "other block caches are disabled in Turbo."
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

                compile_model = gr.Checkbox(
                    value=defaults["compile_model"],
                    label=(
                        "torch.compile (unsafe for quantized H3)"
                        if not ALLOW_UNSAFE_H3_COMPILE
                        else "torch.compile (experimental override enabled)"
                    ),
                    interactive=ALLOW_UNSAFE_H3_COMPILE,
                    info=(
                        "Disabled because NVFP4/INT8 H3 weights use tensor wrappers "
                        "that currently fail full-model Dynamo tracing. Sol-Attn does "
                        "not require this checkbox."
                    ),
                )
                postprocess = gr.Dropdown(
                    [
                        "None", "2× Lanczos", "48 fps interpolation",
                        "2× Lanczos + 48 fps", LTX_UPSCALE,
                        SEEDVR2_UPSCALE, FLASHVSR_UPSCALE,
                    ],
                    value=defaults["postprocess"], label="Post-processing",
                )
                ltx_force_offload = gr.Checkbox(
                    value=defaults["ltx_force_offload"],
                    label="Unload H3 models before AI upscale",
                    info=(
                        "Off by default: ComfyUI manages model residency. Enable to "
                        "unload H3 before LTX, FlashVSR, or SeedVR2 and reduce peak "
                        "VRAM at the cost of reloading H3 next time."
                    ),
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
            mode, model_profile, generation_mode, turbo_variant,
            duration, width, height,
            steps, scheduler, attention_mode, cache_mode, postprocess,
            ltx_force_offload,
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
                steps, compile_model, sol_tau, sol_thresh_type, scheduler,
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
                compile_model, ref_size, postprocess, ltx_force_offload,
            ],
            outputs=[output, status],
            show_progress="minimal",
            api_name="generate_video_advanced",
        )
        api_event = api_run.click(
            generate_with_ui_defaults,
            inputs=api_prompt,
            outputs=[api_download_url, api_status],
            show_progress="minimal",
            api_name="generate_video",
        )
        stop.click(interrupt, outputs=status, cancels=[event])
        api_stop.click(interrupt, outputs=api_status, cancels=[api_event])
        refresh.click(backend_status, outputs=health)
        generate_tab.select(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[generation_view, gallery_view, api_view],
        )
        gallery_event = gallery_tab.select(
            lambda: (
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
                gr.update(visible=True),
            ),
            outputs=[generation_view, gallery_view, api_view],
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
        turbo_lora="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_source="test",
        turbo_ref_lora="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_ref_source="shared-fl2va-test",
        larry_turbo_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_source="test",
        larry_turbo_ref_lora="minimax_h3_turbo_v4_step600_ema.safetensors",
        larry_turbo_ref_source="shared-fl2va-test",
        ltx_checkpoint="ltx-2.3-22b-dev-fp8.safetensors",
        ltx_checkpoint_source="test",
        ltx_distilled_lora="ltx-distilled.safetensors",
        ltx_distilled_lora_source="test",
        ltx_text_encoder="gemma-3-fp4.safetensors",
        ltx_text_encoder_source="test",
        ltx_spatial_upscaler="ltx-spatial-upscaler-x2.safetensors",
        ltx_spatial_upscaler_source="test",
        seedvr2_dit="seedvr2_3b_int8_convrot.safetensors",
        seedvr2_dit_source="test",
        seedvr2_vae="seedvr2_ema_vae_fp16.safetensors",
        seedvr2_vae_source="test",
    )
    available = required_nodes_for("Text to video", True, "FirstBlockCache", True, use_turbo=True) | required_nodes_for("Reference media", True, "EasyCache", True)
    available.add("SpectrumApplyMiniMaxH3")
    available.add(CHUNK_FEED_FORWARD_NODE)
    available |= {LARRY_TURBO_LORA_NODE, LARRY_TURBO_SAMPLER_NODE}
    assert fake.turbo_lora_for("Text to video", LIGHTX2V_TURBO) == fake.turbo_lora
    assert fake.turbo_lora_for("Reference media", LIGHTX2V_TURBO) == fake.turbo_ref_lora
    assert fake.turbo_lora_for("Text to video", LARRY_TURBO) == fake.larry_turbo_lora
    reference_updates = mode_layout_updates("Reference media")
    assert reference_updates[3].get("interactive") is True
    assert "value" not in reference_updates[3]
    # Avoid staging files in selftest; build prompt-only T2V and check graph wiring.
    graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=18, seed=1,
        scheduler="simple", turbo_lora_name="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_variant=LIGHTX2V_TURBO, turbo_strength=1.0,
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
        compile_model=True,
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
        "TorchCompileModel",
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
        turbo_variant=LIGHTX2V_TURBO,
        turbo_strength=0.75,
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
        compile_model=False,
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
    assert not any(
        node["class_type"] == "H3FirstBlockCache"
        for node in spectrum_graph.nodes.values()
    )


    save_nodes = [node for node in graph.values() if node["class_type"] == "SaveVideo"]
    assert len(save_nodes) == 1
    assert save_nodes[0]["inputs"]["codec"] == "auto"
    assert isinstance(save_nodes[0]["inputs"]["codec"], str)

    ltx_graph = build_ltx_upscale_graph(
        source_video="h3_gradio/ltx_upscale/source.mp4",
        prompt="preserve fine facial and textile details",
        frame_count=124,
        seed=7,
        models=fake,
    )
    ltx_nodes = list(ltx_graph.values())
    ltx_classes = {node["class_type"] for node in ltx_nodes}
    assert required_ltx_upscale_nodes() <= ltx_classes
    ltx_lora = next(node for node in ltx_nodes if node["class_type"] == CORE_LORA_LOADER_NODE)
    assert ltx_lora["inputs"]["strength_model"] == 0.5
    ltx_sigmas = next(node for node in ltx_nodes if node["class_type"] == "ManualSigmas")
    assert ltx_sigmas["inputs"]["sigmas"] == LTX_REFINEMENT_SIGMAS
    ltx_audio = next(node for node in ltx_nodes if node["class_type"] == "LTXVEmptyLatentAudio")
    assert ltx_audio["inputs"]["frames_number"] == 121
    assert ltx_audio["inputs"]["batch_size"] == 1
    ltx_frame_nodes = [node for node in ltx_nodes if node["class_type"] == "ImageFromBatch"]
    assert sorted(node["inputs"]["length"] for node in ltx_frame_nodes) == [1, 121]
    ltx_video = next(node for node in ltx_nodes if node["class_type"] == "CreateVideo")
    ltx_components_id = next(
        node_id
        for node_id, node in ltx_graph.items()
        if node["class_type"] == "GetVideoComponents"
    )
    assert ltx_video["inputs"]["audio"] == [ltx_components_id, 1]
    assert UI_DEFAULTS["ltx_force_offload"] is False

    seedvr2_graph = build_seedvr2_upscale_graph(
        source_video="h3_gradio/seedvr2_upscale/source.mp4", seed=7, models=fake
    )
    seedvr2_nodes = list(seedvr2_graph.values())
    assert required_seedvr2_upscale_nodes() <= {
        node["class_type"] for node in seedvr2_nodes
    }
    seedvr2_scale = next(
        node for node in seedvr2_nodes if node["class_type"] == "ImageScaleBy"
    )
    assert seedvr2_scale["inputs"]["scale_by"] == 2.0
    seedvr2_sampler = next(
        node for node in seedvr2_nodes if node["class_type"] == "KSampler"
    )
    assert seedvr2_sampler["inputs"]["steps"] == 1
    assert seedvr2_sampler["inputs"]["denoise"] == 1.0

    flashvsr_graph = build_flashvsr_upscale_graph(
        source_video="h3_gradio/flashvsr_upscale/source.mp4", seed=7
    )
    flashvsr_nodes = list(flashvsr_graph.values())
    assert required_flashvsr_upscale_nodes() <= {
        node["class_type"] for node in flashvsr_nodes
    }
    flashvsr = next(
        node for node in flashvsr_nodes if node["class_type"] == "AILab_FlashVSR"
    )
    assert flashvsr["inputs"]["scale"] == 2
    assert flashvsr["inputs"]["preset"] == "Balanced (2x Quality)"
    assert flashvsr["inputs"]["unload_model"] is True

    assert resolution_choice_values("9:16 · 768×1344", "large")[:2] == (768, 1344)
    assert resolution_choice_values("1:1 · 1024×1024", "large")[:2] == (1024, 1024)
    assert set(RESOLUTION_TIERS) == {"draft", "fast", "large"}
    assert preset_values("Quality")[0] == 20
    assert preset_values("Balanced")[0] == 18
    assert preset_values("Fast")[0] == 15
    assert preset_values("unknown") == preset_values("Balanced")
    assert UI_DEFAULTS["steps"] == turbo_steps_for(UI_DEFAULTS["turbo_variant"])
    assert UI_DEFAULTS["width"] == 864 and UI_DEFAULTS["height"] == 480
    assert 'api_name="/generate_video"' in api_guide()
    captured_free_call: dict[str, Any] = {}
    original_api_post = globals()["api_post"]

    def fake_api_post(path: str, **kwargs: Any) -> None:
        captured_free_call["path"] = path
        captured_free_call["kwargs"] = kwargs

    globals()["api_post"] = fake_api_post
    try:
        unload_comfy_models()
    finally:
        globals()["api_post"] = original_api_post
    assert captured_free_call == {
        "path": "/free",
        "kwargs": {"json": {"unload_models": True, "free_memory": True}},
    }
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
        try:
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
        assert len(gallery_items) == 1
        assert gallery_paths == [str(fallback_video)]
        assert gallery_play_url == str(fallback_video)
        assert selected_video == str(fallback_video)
        assert gallery_download_link.endswith(
            "/downloads/comfy/fallback.mp4?download=1)"
        )
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
    lightx_defaults = generation_mode_defaults("Turbo", LIGHTX2V_TURBO)
    assert lightx_defaults[1]["value"] == 4
    assert lightx_defaults[1]["interactive"] is True
    assert lightx_defaults[2:] == ("simple", "Spectrum", "Auto")
    normal_defaults = generation_mode_defaults("Normal")
    assert normal_defaults[1]["value"] == 18
    assert normal_defaults[1]["interactive"] is True
    assert normal_defaults[2:] == ("simple", "Spectrum", "Auto")
    assert resolve_cache_policy("Off", use_turbo=True) == ("Off", None)
    turbo_spectrum, turbo_spectrum_note = resolve_cache_policy(
        "Spectrum", use_turbo=True
    )
    assert turbo_spectrum == "Spectrum" and turbo_spectrum_note
    for block_cache in ("FirstBlockCache", "EasyCache"):
        resolved_cache, resolved_note = resolve_cache_policy(
            block_cache, use_turbo=True
        )
        assert resolved_cache == "Off" and block_cache in str(resolved_note)
    assert SERVER_DENSE_ATTENTION_BACKEND in {"pytorch", "sage"}

    quality_turbo_graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=8, seed=2,
        scheduler="simple",
        turbo_lora_name="minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_v0.1_comfy.safetensors",
        turbo_variant=LIGHTX2V_TURBO,
        turbo_strength=0.75,
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
        compile_model=False,
        model_name=fake.profile("quality").fl2va,
        models=fake, available_nodes=available,
    )
    quality_unets = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "UNETLoader"
    ]
    assert len(quality_unets) == 1
    assert quality_unets[0]["inputs"]["unet_name"] == fake.profile("quality").fl2va
    turbo_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == CORE_LORA_LOADER_NODE
    ]
    assert len(turbo_nodes) == 1
    assert turbo_nodes[0]["inputs"]["strength_model"] == 0.75
    assert turbo_nodes[0]["inputs"]["lora_name"].endswith(
        "v0.1_comfy.safetensors"
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
        easycache_end=0.85, easycache_verbose=False, compile_model=False,
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
    assert larry_loader["inputs"]["low_vram"] is False
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
    assert FUSED_MODULATION_NODE in turbo_required_nodes(LIGHTX2V_TURBO)
    assert any(
        node["class_type"] == LARRY_TURBO_SAMPLER_NODE
        for node in larry_graph.nodes.values()
    )
    assert not any(
        node["class_type"] == CORE_SAMPLER_NODE
        for node in larry_graph.nodes.values()
    )

    ref_turbo_graph = Graph()
    add_model_stack(
        ref_turbo_graph,
        fake.profile("quality").ref2va,
        fake,
        turbo_lora_name=fake.turbo_ref_lora,
        turbo_variant=LIGHTX2V_TURBO,
        turbo_strength=0.75,
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
        compile_model=False,
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
    assert ref_lora["inputs"]["strength_model"] == 0.75

    sched_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "BasicScheduler"
    ]
    assert len(sched_nodes) == 1
    assert sched_nodes[0]["inputs"]["steps"] == 8
    effective, reason = resolve_compile_request(True, "minimax_h3_fl2va_pruned_nvfp4.safetensors")
    if not ALLOW_UNSAFE_H3_COMPILE:
        assert effective is False and reason
    print(
        f"Selftest OK: {len(graph)} nodes, 5s=124 frames, "
        f"15s=362 frames, tiered resolution presets valid, Sol exact valid, "
        f"Sol Auto/Turbo policy valid, Spectrum default + Sol/ConvRot order valid, "
        f"zero-copy Sol + FirstBlockCache composition valid, "
        f"LightX fused modulation + Larry compatibility + ConvRot FFN chunking valid, "
        f"Spectrum v0.2.5 Turbo composition + block-cache guard valid, "
        f"selectable Larry/LightX2V Turbo on "
        f"FL2VA/Ref2VA + synchronized editable Turbo steps valid, compile guard active, "
        f"SaveVideo codec API valid, prompt API download URL valid, "
        f"gallery fallback/deletion guards valid, "
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
    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
