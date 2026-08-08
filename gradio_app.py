#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import requests

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
DEFAULT_FBCACHE_PRESET = "Fast"
DEFAULT_FBCACHE_THRESHOLD = 0.10
DEFAULT_FBCACHE_START = 0.10
DEFAULT_FBCACHE_END = 0.95
DEFAULT_FBCACHE_MAX_HITS = 2
DEFAULT_FBCACHE_TEMPORAL_GUARD = True

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

HTTP = requests.Session()


class H3Error(RuntimeError):
    pass


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

    def profile(self, name: str) -> ModelProfile:
        key = str(name).strip().lower()
        if key not in self.profiles:
            key = self.default_profile
        if key not in self.profiles:
            raise H3Error(f"Unknown model profile: {name}")
        return self.profiles[key]


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
    )


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
    if use_turbo:
        return False, tokens, "Auto: Turbo stays dense pending Sol+Turbo validation"
    # References add presentation and conditioning rows which are expensive and
    # difficult to know exactly before ComfyUI encodes the media, so Auto uses
    # Sol for Ref2VA. FL2VA uses a conservative target-token lower bound.
    if mode == "Reference media":
        return True, tokens, "Auto: reference mode"
    enabled = tokens >= AUTO_SOL_TOKEN_THRESHOLD
    return enabled, tokens, (
        f"Auto: {tokens:,} target tokens "
        f"{'≥' if enabled else '<'} {AUTO_SOL_TOKEN_THRESHOLD:,}"
    )


def mode_layout_updates(mode: str):
    """Update task-specific inputs and keep Reference media Normal-only."""
    show_frames = mode == "First / last frame"
    show_refs = mode == "Reference media"

    if show_refs:
        return (
            mode_help(mode),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value="Normal", interactive=False),
            gr.update(value="Balanced", interactive=True),
            18,
            "simple",
            "FirstBlockCache",
            "Auto",
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


def generation_mode_defaults(name: str):
    """Normal/Turbo is independent from the selected Speed/Quality base.

    Important: when entering Turbo, do not change preset.value. Changing it
    would fire preset.change() and race with the Turbo Steps update.
    """
    if str(name).strip().lower() == "turbo":
        return (
            gr.update(interactive=False),
            4,
            "simple",
            "Off",
            "Dense",
        )

    return (
        gr.update(value="Balanced", interactive=True),
        18,
        "simple",
        "FirstBlockCache",
        "Auto",
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


def add_model_stack(
    graph: Graph,
    model_name: str,
    models: ModelConfig,
    *,
    turbo_lora_name: str | None,
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
        if "LoraLoaderModelOnly" not in available_nodes:
            raise H3Error(
                "Turbo was requested, but core LoraLoaderModelOnly is unavailable. "
                "Update ComfyUI and restart the service."
            )
        turbo = graph.add(
            "LoraLoaderModelOnly",
            model=model_ref,
            lora_name=turbo_lora_name,
            strength_model=float(turbo_strength),
        )
        model_ref = Graph.out(turbo)

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
        if "MiniMaxH3MemoryEfficientSolAttentionPatch" not in available_nodes:
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
            "MiniMaxH3MemoryEfficientSolAttentionPatch",
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
        if "MiniMaxH3ChunkFeedForward" not in available_nodes:
            raise H3Error(
                "The quality model requires MiniMaxH3ChunkFeedForward, but the "
                "updated Sol-Attn plugin is not loaded. Re-run setup_h3.py."
            )
        chunked = graph.add(
            "MiniMaxH3ChunkFeedForward",
            model=model_ref,
            enabled=True,
            chunks=2,
            min_tokens=AUTO_SOL_TOKEN_THRESHOLD,
        )
        model_ref = Graph.out(chunked)

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
    filename_prefix: str,
) -> None:
    noise = graph.add("RandomNoise", noise_seed=int(seed))
    guider = graph.add("BasicGuider", model=model_ref, conditioning=conditioning_ref)
    sampler = graph.add("KSamplerSelect", sampler_name="res_multistep")
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

    for index, path in enumerate(reference_images[:9]):
        loaded = graph.add("LoadImage", image=stage_file(path, "reference_images"))
        inputs[f"ref_images.ref_image_{index}"] = Graph.out(loaded)

    for index, path in enumerate(reference_videos[:3]):
        staged = stage_file(path, "reference_videos", transcode_video=True)
        loaded = graph.add("LoadVideo", file=staged)
        components = graph.add("GetVideoComponents", video=Graph.out(loaded))
        inputs[f"ref_videos.ref_video_{index}"] = Graph.out(components, 0)
        inputs[f"ref_video_audios.ref_video_audio_{index}"] = Graph.out(components, 1)

    for index, path in enumerate(reference_audios[:3]):
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
        filename_prefix=f"h3/ref2va_{int(time.time())}",
    )
    return graph.nodes


def required_nodes_for(
    mode: str,
    use_sol: bool,
    cache_mode: str,
    compile_model: bool,
    use_turbo: bool = False,
    model_filename: str = "",
) -> set[str]:
    common = {
        "UNETLoader", "CLIPLoader", "VAELoader", "RandomNoise",
        "BasicGuider", "KSamplerSelect", "BasicScheduler",
        "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
        "CreateVideo", "SaveVideo",
    }
    if mode == "Reference media":
        common |= {"MiniMaxH3ReferenceToVideo", "LoadImage", "LoadVideo", "GetVideoComponents", "LoadAudio"}
    else:
        common |= {"MiniMaxH3ImageToVideo", "LoadImage"}
    if use_turbo:
        common.add("LoraLoaderModelOnly")
    if use_sol:
        common.add("MiniMaxH3MemoryEfficientSolAttentionPatch")
    if "convrot" in model_filename.lower():
        common.add("MiniMaxH3ChunkFeedForward")
    if str(cache_mode).strip().lower() == "firstblockcache":
        common.add("H3FirstBlockCache")
    elif str(cache_mode).strip().lower() == "easycache":
        common.add("EasyCache")
    if compile_model:
        common.add("TorchCompileModel")
    return common


def submit_prompt(graph: dict[str, Any]) -> str:
    response = api_post("/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())})
    payload = response.json()
    if "prompt_id" not in payload:
        raise H3Error(json.dumps(payload, indent=2))
    return str(payload["prompt_id"])


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
    video_exts = {".mp4", ".webm", ".mov", ".mkv", ".gif"}
    videos = [p for p in candidates if p.suffix.lower() in video_exts]
    if videos:
        return max(videos, key=lambda p: p.stat().st_mtime)

    # Newer SaveVideo UI payloads can be omitted from the public history shape.
    # Fall back only to files created after this job was queued.
    recent = [
        p for p in OUTPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in video_exts and p.stat().st_mtime >= queued_at - 2
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
        profile_lines = []
        for key in ("speed", "quality"):
            if key not in models.profiles:
                continue
            profile = models.profiles[key]
            profile_lines.append(
                f"**{profile.label}** · FL2VA `{profile.fl2va}` · "
                f"Ref2VA `{profile.ref2va}`"
            )
        if models.turbo_lora:
            profile_lines.append(
                f"**LightX2V Turbo** · LoRA `{models.turbo_lora}` · "
                "4-step default · strength 0.75 · Speed or Quality FL2VA"
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
    prompt: str,
    first_image: str | None,
    last_image: str | None,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
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
):
    started = time.monotonic()
    queued_at = time.time()
    try:
        if not prompt.strip():
            raise H3Error("Prompt is required.")
        if not 2 <= float(duration) <= 15:
            raise H3Error("Duration must be between 2 and 15 seconds.")
        resolved_width, resolved_height = validate_resolution(width, height)
        actual_seed = random.randrange(0, 2**63 - 1) if int(seed) < 0 else int(seed)
        models = load_model_config()
        profile = models.profile(model_profile)

        requested_generation = str(generation_mode).strip().lower()
        generation_note = None
        if mode == "Reference media" and requested_generation == "turbo":
            use_turbo = False
            generation_note = "Turbo is FL2VA-only; Reference media used Normal generation."
        else:
            use_turbo = requested_generation == "turbo"

        selected_model = profile.ref2va if mode == "Reference media" else profile.fl2va
        if use_turbo:
            if not models.turbo_lora:
                raise H3Error("Turbo LoRA is not provisioned. Re-run setup/provisioning.")
            selected_label = f"{profile.label} · Turbo"
            turbo_lora_name = models.turbo_lora
            turbo_strength = 0.75
        else:
            selected_label = f"{profile.label} · Normal"
            turbo_lora_name = None
            turbo_strength = 1.0

        # Turbo defaults to 4/simple in the UI, but the user may deliberately
        # change Steps or Scheduler after selecting Turbo. Honor those visible
        # controls instead of silently overriding them at graph-build time.
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

        effective_sol, packed_tokens, sol_reason = resolve_sol_policy(
            attention_mode, mode, resolved_width, resolved_height, float(duration),
            first_image, last_image, use_turbo=use_turbo,
        )
        effective_cache_mode = str(cache_mode).strip()
        cache_note = None
        if use_turbo and effective_cache_mode.lower() != "off":
            effective_cache_mode = "Off"
            cache_note = "FirstBlock/EasyCache disabled automatically in Turbo mode pending validation."
        missing = required_nodes_for(
            mode,
            effective_sol,
            effective_cache_mode,
            effective_compile,
            use_turbo=use_turbo,
            model_filename=selected_model,
        ) - available
        if missing:
            raise H3Error("Missing ComfyUI nodes: " + ", ".join(sorted(missing)))

        refs_i = collect_reference_slots(
            ref_image_1, ref_image_2, ref_image_3,
            ref_image_4, ref_image_5, ref_image_6,
        )
        refs_v = collect_reference_slots(ref_video_1, ref_video_2)
        refs_a = collect_reference_slots(ref_audio_1, ref_audio_2)

        if mode == "Text to video":
            first_image = None
            last_image = None
        elif mode == "First / last frame":
            if not first_image and not last_image:
                raise H3Error("Provide a first frame, a last frame, or both.")
        elif mode == "Reference media":
            if not (refs_i or refs_v or refs_a):
                raise H3Error("Reference mode requires at least one image, video, or audio file.")

        if mode == "Reference media":
            graph = build_ref2va_graph(
                prompt=prompt,
                reference_images=refs_i,
                reference_videos=refs_v,
                reference_audios=refs_a,
                width=resolved_width, height=resolved_height, duration=float(duration),
                steps=effective_steps, seed=actual_seed, scheduler=effective_scheduler,
                ref_image_size=ref_image_size,
                turbo_lora_name=turbo_lora_name, turbo_strength=turbo_strength,
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
                turbo_lora_name=turbo_lora_name, turbo_strength=turbo_strength,
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

        prompt_id = submit_prompt(graph)
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
        yield None, queued_status
        history = wait_for_history(prompt_id)
        source = resolve_output(history, queued_at)
        result = postprocess_video(source, postprocess)
        elapsed = time.monotonic() - started
        yield str(result), (
            f"Completed in {elapsed:.1f}s · output {result.name} · seed {actual_seed} · "
            f"{elapsed / float(duration):.1f}s compute per output second"
        )
    except Exception as exc:
        yield None, f"Error: {exc}"


def interrupt() -> str:
    try:
        api_post("/interrupt", json={})
        return "Interrupt requested."
    except Exception as exc:
        return f"Interrupt failed: {exc}"


def preset_values(name: str):
    return SAMPLING_PRESETS.get(str(name), SAMPLING_PRESETS["Balanced"])


def mode_help(mode: str) -> str:
    if mode == "Reference media":
        return (
            "Reference tags are ordered as images, then videos, then standalone audio. "
            "Use placeholders like `<Picture 1>`, `<Picture 2>`, `<Video 1>`, and `<Audio 1>` in the prompt."
        )
    if mode == "First / last frame":
        return "Upload a first frame, a last frame, or both. This uses the FL2VA model."
    return "Prompt-only generation using FL2VA with native stereo audio."


def build_ui() -> gr.Blocks:
    sol_default = SERVER_ATTENTION_BACKEND == "sol"
    with gr.Blocks(title="MiniMax H3 Local") as demo:
        gr.Markdown("# MiniMax H3 Local\nNative ComfyUI graphs for T2V, first/last-frame video, and reference media.")
        health = gr.Markdown(backend_status())
        with gr.Row():
            with gr.Column(scale=3):
                mode = gr.Radio(
                    ["Text to video", "First / last frame", "Reference media"],
                    value="Text to video", label="Mode",
                )
                with gr.Row():
                    model_profile = gr.Radio(
                        ["Speed", "Quality"],
                        value="Quality",
                        label="Base model",
                        info=(
                            "Speed uses the rebuilt single-pass NVFP4 files. "
                            "Quality uses the larger mixed NVFP4/FP8/INT8 ConvRot files."
                        ),
                    )
                    generation_mode = gr.Radio(
                        ["Normal", "Turbo"],
                        value="Turbo",
                        label="Generation",
                        info=(
                            "Turbo defaults to 4 steps and uses the H3 Turbo LoRA plus "
                            "dual-clock sampler on whichever FL2VA base model is selected. "
                            "Reference media is Normal-only."
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
                    gr.Markdown(
                        "Use `<Picture 1>`–`<Picture 6>`, `<Video 1>`–`<Video 2>`, "
                        "and `<Audio 1>`–`<Audio 2>` in the prompt."
                    )
                    with gr.Row():
                        ref_image_1 = gr.Image(type="filepath", label="Picture 1")
                        ref_image_2 = gr.Image(type="filepath", label="Picture 2")
                        ref_image_3 = gr.Image(type="filepath", label="Picture 3")
                    with gr.Row():
                        ref_image_4 = gr.Image(type="filepath", label="Picture 4")
                        ref_image_5 = gr.Image(type="filepath", label="Picture 5")
                        ref_image_6 = gr.Image(type="filepath", label="Picture 6")
                    with gr.Row():
                        ref_video_1 = gr.Video(label="Video 1")
                        ref_video_2 = gr.Video(label="Video 2")
                    with gr.Row():
                        ref_audio_1 = gr.Audio(type="filepath", label="Audio 1")
                        ref_audio_2 = gr.Audio(type="filepath", label="Audio 2")
                    ref_size = gr.Radio(["match", "max"], value="match", label="Reference image size")
            with gr.Column(scale=2):
                output = gr.Video(label="Generated video")
                status = gr.Textbox(label="Status", lines=5)
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
                    duration = gr.Slider(2, 15, value=5, step=0.5, label="Seconds")
                    steps = gr.Slider(
                        4, 30, value=4, step=1, label="Steps",
                        info=(
                            "Turbo defaults to 4 steps, but this remains editable. "
                            "Normal H3 presets normally use 15–20."
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
                    width = gr.Number(value=864, precision=0, label="Width")
                    height = gr.Number(value=480, precision=0, label="Height")
                resolution_info = gr.Markdown(resolution_summary(864, 480))
                with gr.Row():
                    scheduler = gr.Radio(["simple", "beta", "normal"], value="simple", label="Scheduler")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                attention_mode = gr.Radio(
                    ["Auto", "Sol-Attn", "Dense"],
                    value="Auto",
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
                        0.5, 1.5, value=1.0, step=0.1, label="Sol-Attn tau"
                    )
                    sol_thresh_type = gr.Radio(
                        ["diag", "exact"],
                        value="diag",
                        label="Sol threshold",
                        info="diag is faster; exact calculates a more precise routing threshold.",
                    )
                with gr.Accordion("Zero-copy Sol-Attn quality controls", open=True):
                    sol_exact_mode = gr.Radio(
                        ["off", "exact_kv", "exact_kv_and_rows"],
                        value="exact_kv",
                        label="Exact H3 prefix mode",
                        info=(
                            "exact_kv preserves text/condition/reference/audio KV "
                            "rows at low cost. exact_kv_and_rows also keeps prefix "
                            "query rows dense for maximum audio/conditioning fidelity."
                        ),
                    )
                    with gr.Row():
                        sol_dense_steps = gr.Slider(
                            0, 4, value=1, step=1,
                            label="Dense final transformer blocks",
                            info=(
                                "Keep the final N H3 transformer blocks dense. "
                                "The final block is the most approximation-sensitive."
                            ),
                        )
                    sol_step_off = gr.State(0.0)
                    sol_sink_tokens = gr.State(0)
                with gr.Accordion("Cache acceleration", open=True):
                    cache_mode = gr.Radio(
                        ["FirstBlockCache", "EasyCache", "Off"],
                        value="Off",
                        label="Cache mode",
                        info=(
                            "FirstBlockCache is the recommended H3 default. It keeps every "
                            "scheduler step but can reuse blocks 1–49 when block-0 residuals "
                            "are sufficiently similar. Generation=Turbo automatically forces cache Off."
                        ),
                    )
                    fbcache_preset = gr.Radio(
                        ["Safe", "Fast", "Aggressive", "Custom"],
                        value=DEFAULT_FBCACHE_PRESET,
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
                            value=DEFAULT_FBCACHE_THRESHOLD,
                            step=0.005,
                            label="FirstBlock threshold",
                            interactive=False,
                        )
                        fbcache_max_hits = gr.Slider(
                            1, 8,
                            value=DEFAULT_FBCACHE_MAX_HITS,
                            step=1,
                            label="Max consecutive cache hits",
                            interactive=False,
                        )
                    with gr.Row():
                        fbcache_start = gr.Slider(
                            0.0, 0.90,
                            value=DEFAULT_FBCACHE_START,
                            step=0.01,
                            label="Cache start percent",
                            interactive=False,
                        )
                        fbcache_end = gr.Slider(
                            0.10, 1.0,
                            value=DEFAULT_FBCACHE_END,
                            step=0.01,
                            label="Cache end percent",
                            interactive=False,
                        )
                    fbcache_temporal_guard = gr.Checkbox(
                        value=DEFAULT_FBCACHE_TEMPORAL_GUARD,
                        label="Temporal frame guard",
                        info=(
                            "Checks the most-changed target-video latent frame "
                            "in addition to the global residual average."
                        ),
                    )
                    gr.Markdown("**EasyCache fallback settings**")
                    easycache_threshold = gr.Slider(
                        0.0, 0.5, value=0.10, step=0.01,
                        label="Reuse threshold",
                        info=(
                            "Higher skips more steps. Start at 0.10 for H3; "
                            "ComfyUI's generic default is 0.20."
                        ),
                    )
                    with gr.Row():
                        easycache_start = gr.Slider(
                            0.0, 0.9, value=0.15, step=0.01,
                            label="Start percent",
                        )
                        easycache_end = gr.Slider(
                            0.1, 1.0, value=0.85, step=0.01,
                            label="End percent",
                        )
                    easycache_verbose = gr.Checkbox(
                        value=False,
                        label="Log EasyCache decisions",
                        info="Logs skipped-step counts and estimated speedup in ComfyUI.",
                    )

                compile_model = gr.Checkbox(
                    value=False,
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
                    ["None", "2× Lanczos", "48 fps interpolation", "2× Lanczos + 48 fps"],
                    value="None", label="Post-processing",
                )
                with gr.Row():
                    run = gr.Button("Generate", variant="primary")
                    stop = gr.Button("Interrupt")
                    refresh = gr.Button("Refresh status")

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
            inputs=generation_mode,
            outputs=[preset, steps, scheduler, cache_mode, attention_mode],
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
                mode, model_profile, generation_mode, prompt, first, last,
                ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6,
                ref_video_1, ref_video_2, ref_audio_1, ref_audio_2,
                duration, width, height, steps, scheduler, seed, attention_mode, sol_tau,
                sol_thresh_type, sol_exact_mode, sol_dense_steps,
                sol_step_off, sol_sink_tokens,
                cache_mode, fbcache_preset, fbcache_threshold, fbcache_start,
                fbcache_end, fbcache_max_hits, fbcache_temporal_guard,
                easycache_threshold, easycache_start, easycache_end, easycache_verbose,
                compile_model, ref_size, postprocess,
            ],
            outputs=[output, status],
        )
        stop.click(interrupt, outputs=status, cancels=[event])
        refresh.click(backend_status, outputs=health)
    return demo


def selftest() -> None:
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
        },
        default_profile="speed",
        text_encoder="text.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
        turbo_lora="minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        turbo_source="test",
    )
    available = required_nodes_for("Text to video", True, "FirstBlockCache", True, use_turbo=True) | required_nodes_for("Reference media", True, "EasyCache", True)
    available.add("MiniMaxH3ChunkFeedForward")
    # Avoid staging files in selftest; build prompt-only T2V and check graph wiring.
    graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=18, seed=1,
        scheduler="simple", turbo_lora_name="minimax_h3_turbo_4step_ema_ckpt850.safetensors", turbo_strength=1.0,
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
        "MiniMaxH3MemoryEfficientSolAttentionPatch",
        "H3FirstBlockCache",
        "LoraLoaderModelOnly",
        "TorchCompileModel",
    }
    missing = expected - classes
    if missing:
        raise SystemExit(f"Selftest failed; missing nodes: {missing}")

    sol_nodes = [
        node for node in graph.values()
        if node["class_type"] == "MiniMaxH3MemoryEfficientSolAttentionPatch"
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

    # Sol must consume the FirstBlockCache model output. The inverse ordering
    # reproduces the runtime failure where Sol's executor.original() bypasses
    # the cache diffusion wrapper.
    cache_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == "H3FirstBlockCache"
    )
    sol_id = next(
        node_id for node_id, node in graph.items()
        if node["class_type"] == "MiniMaxH3MemoryEfficientSolAttentionPatch"
    )
    assert graph[sol_id]["inputs"]["model"] == [cache_id, 0]
    assert graph[cache_id]["inputs"]["model"] != [sol_id, 0]


    save_nodes = [node for node in graph.values() if node["class_type"] == "SaveVideo"]
    assert len(save_nodes) == 1
    assert save_nodes[0]["inputs"]["codec"] == "auto"
    assert isinstance(save_nodes[0]["inputs"]["codec"], str)

    assert resolution_choice_values("9:16 · 768×1344", "large")[:2] == (768, 1344)
    assert resolution_choice_values("1:1 · 1024×1024", "large")[:2] == (1024, 1024)
    assert set(RESOLUTION_TIERS) == {"draft", "fast", "large"}
    assert preset_values("Quality")[0] == 20
    assert preset_values("Balanced")[0] == 18
    assert preset_values("Fast")[0] == 15
    assert preset_values("unknown") == preset_values("Balanced")
    assert estimate_packed_tokens("Text to video", 1344, 768, 5) >= AUTO_SOL_TOKEN_THRESHOLD
    assert resolve_sol_policy("Auto", "Text to video", 608, 352, 2, None, None)[0] is False
    assert validate_resolution(865, 481) == (864, 480)
    assert frame_length(5) == 124
    assert frame_length(15) == 362
    turbo_defaults = generation_mode_defaults("Turbo")
    assert turbo_defaults[1:] == (4, "simple", "Off", "Dense")
    normal_defaults = generation_mode_defaults("Normal")
    assert normal_defaults[1:] == (18, "simple", "FirstBlockCache", "Auto")
    assert SERVER_DENSE_ATTENTION_BACKEND in {"pytorch", "sage"}

    quality_turbo_graph = build_fl2va_graph(
        prompt="test", first_image=None, last_image=None,
        width=864, height=480, duration=5, steps=8, seed=2,
        scheduler="simple",
        turbo_lora_name="minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_v0.1_comfy.safetensors",
        turbo_strength=0.75,
        use_sol=False, sol_tau=1.0,
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
        if node["class_type"] == "LoraLoaderModelOnly"
    ]
    assert len(turbo_nodes) == 1
    assert turbo_nodes[0]["inputs"]["strength_model"] == 0.75
    assert turbo_nodes[0]["inputs"]["lora_name"].endswith(
        "v0.1_comfy.safetensors"
    )
    chunk_nodes = [
        node for node in quality_turbo_graph.values()
        if node["class_type"] == "MiniMaxH3ChunkFeedForward"
    ]
    assert len(chunk_nodes) == 1
    assert chunk_nodes[0]["inputs"]["chunks"] == 2
    assert chunk_nodes[0]["inputs"]["min_tokens"] == AUTO_SOL_TOKEN_THRESHOLD
    assert not any(
        node["class_type"] == "MiniMaxH3TurboSampler"
        for node in quality_turbo_graph.values()
    )
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
        f"Sol Auto policy valid, zero-copy Sol + FirstBlockCache composition valid, "
        f"ConvRot FFN chunking valid, Kijai LightX2V Turbo + editable steps valid, compile guard active, "
        f"SaveVideo codec API valid"
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
    build_ui().queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        allowed_paths=allowed_paths,
        show_error=True,
    )


if __name__ == "__main__":
    main()
