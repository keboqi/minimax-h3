"""Extracted policy boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from h3_app.catalog import (
    AUTO_RESOLUTION_MEGAPIXEL_PRESETS,
    AUTO_RESOLUTION_PIXEL_CAP,
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
    DEFAULT_IMAGE_VAE,
    DEFAULT_RESULT_FORMAT,
    DEFAULT_TURBO,
    H3_LATENT_UPSCALE_SPLIT,
    H3_LATENT_UPSCALE_STANDARD,
    IMAGE_VAE_CHOICES,
    LIGHTX2V_4STEP_TURBO,
    LIGHTX2V_8STEP_TURBO,
    MAX_IMAGE_FRAMES,
    MIN_IMAGE_FRAMES,
    NATIVE_PIXEL_CAP,
    RESOLUTION_TIERS,
    RESULT_FORMATS,
    SINGLE_FRAME_IMAGE_VAE,
    SLA_PRESET_INPUTS,
    TURBO_SETTINGS,
    UPSCALE_RESOLUTION_PRESETS,
)
from h3_app.errors import H3Error
from h3_models import H3_LATENT_UPSCALER_MODEL_CHOICES, MODEL_SPECS


def upscale_target_dimensions(
    source_width: int, source_height: int, preset: str
) -> tuple[int, int]:
    try:
        max_width, max_height = UPSCALE_RESOLUTION_PRESETS[str(preset)]
        source_width, source_height = int(source_width), int(source_height)
    except (KeyError, TypeError, ValueError) as exc:
        raise H3Error(f"Unknown upscale resolution preset: {preset}") from exc
    scale = min(max_width / source_width, max_height / source_height)
    # SeedVR2 and SwiftVR can preserve the fitted source ratio exactly.
    # LTX applies its required latent alignment inside its graph.
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return min(width, max_width), min(height, max_height)


def normalize_turbo_variant(value: str) -> str:
    return value if value in TURBO_SETTINGS else DEFAULT_TURBO


def turbo_steps_for(value: str) -> int:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].steps


def turbo_strength_for(value: str) -> float:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].strength


def turbo_uses_custom_nodes(value: str) -> bool:
    return TURBO_SETTINGS[normalize_turbo_variant(value)].custom_nodes


def is_lightx2v_turbo_lora(turbo_variant: str, lora_filename: str | None) -> bool:
    """Return whether a configured LightX2V Turbo LoRA is active."""
    return bool(lora_filename) and normalize_turbo_variant(turbo_variant) in {
        LIGHTX2V_4STEP_TURBO,
        LIGHTX2V_8STEP_TURBO,
    }


def lightx2v_uses_768p_schedule(turbo_variant: str, lora_filename: str | None) -> bool:
    """Official 768p FL2VA Turbo checkpoints use video/audio shifts 6/3."""
    return is_lightx2v_turbo_lora(turbo_variant, lora_filename) and (
        "768p" in Path(str(lora_filename)).name.lower()
    )


def turbo_sampler_name(turbo_variant: str, lora_filename: str | None) -> str:
    # ModelTC's official ComfyUI guide defaults every LightX2V Turbo LoRA to
    # Euler. A missing LoRA means normal generation and keeps res_multistep.
    return (
        "euler"
        if is_lightx2v_turbo_lora(turbo_variant, lora_filename)
        else "res_multistep"
    )


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


@dataclass(frozen=True)
class H3SplitUpscaleConfig:
    tile_width: int
    tile_height: int
    overlap_ratio: float
    fade_ratio: float
    chunk_frames: int
    temporal_overlap_frames: int
    seam_denoise: float
    seam_polish: str


def resolve_h3_latent_upscale_method(value: str | None) -> str:
    normalized = str(value or H3_LATENT_UPSCALE_STANDARD).strip().lower()
    aliases = {
        "full-frame": H3_LATENT_UPSCALE_STANDARD,
        "full frame": H3_LATENT_UPSCALE_STANDARD,
        "standard": H3_LATENT_UPSCALE_STANDARD,
        H3_LATENT_UPSCALE_STANDARD.lower(): H3_LATENT_UPSCALE_STANDARD,
        "split": H3_LATENT_UPSCALE_SPLIT,
        "mmh3 split upscale": H3_LATENT_UPSCALE_SPLIT,
        H3_LATENT_UPSCALE_SPLIT.lower(): H3_LATENT_UPSCALE_SPLIT,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise H3Error(
            "Unknown H3 latent upscale method. Choose full-frame refinement or "
            "MMH3 Split Upscale."
        ) from exc


def resolve_h3_split_upscale_config(
    method: str | None,
    *,
    tile_width: int | float,
    tile_height: int | float,
    overlap_ratio: float,
    fade_ratio: float,
    chunk_frames: int | float,
    temporal_overlap_frames: int | float,
    seam_denoise: float,
    seam_polish: str,
) -> H3SplitUpscaleConfig | None:
    if resolve_h3_latent_upscale_method(method) == H3_LATENT_UPSCALE_STANDARD:
        return None

    resolved_tile_width = int(tile_width)
    resolved_tile_height = int(tile_height)
    resolved_chunk_frames = int(chunk_frames)
    resolved_temporal_overlap = int(temporal_overlap_frames)
    resolved_overlap_ratio = float(overlap_ratio)
    resolved_fade_ratio = float(fade_ratio)
    resolved_seam_denoise = float(seam_denoise)
    resolved_seam_polish = str(seam_polish).strip().lower()
    if not 64 <= resolved_tile_width <= 16384 or resolved_tile_width % 32:
        raise H3Error("MMH3 tile width must be a multiple of 32 from 64 to 16384.")
    if not 64 <= resolved_tile_height <= 16384 or resolved_tile_height % 32:
        raise H3Error("MMH3 tile height must be a multiple of 32 from 64 to 16384.")
    if not 0.0 <= resolved_overlap_ratio <= 0.90:
        raise H3Error("MMH3 spatial overlap must be between 0.0 and 0.90.")
    if not 0.0 <= resolved_fade_ratio <= 1.0:
        raise H3Error("MMH3 fade ratio must be between 0.0 and 1.0.")
    if not 5 <= resolved_chunk_frames <= 100000:
        raise H3Error("MMH3 temporal chunk length must be from 5 to 100000 frames.")
    if not 0 <= resolved_temporal_overlap <= 100000:
        raise H3Error("MMH3 temporal overlap must be from 0 to 100000 frames.")
    if resolved_temporal_overlap >= resolved_chunk_frames:
        raise H3Error(
            "MMH3 temporal overlap must be non-negative and smaller than the chunk."
        )
    if not 0.1 <= resolved_seam_denoise <= 1.0:
        raise H3Error("MMH3 seam denoise must be between 0.1 and 1.0.")
    if resolved_seam_polish not in {"off", "auto", "all"}:
        raise H3Error("MMH3 seam polish must be off, auto, or all.")
    return H3SplitUpscaleConfig(
        tile_width=resolved_tile_width,
        tile_height=resolved_tile_height,
        overlap_ratio=resolved_overlap_ratio,
        fade_ratio=resolved_fade_ratio,
        chunk_frames=resolved_chunk_frames,
        temporal_overlap_frames=resolved_temporal_overlap,
        seam_denoise=resolved_seam_denoise,
        seam_polish=resolved_seam_polish,
    )


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


def normalize_image_vae(value: Any) -> str:
    requested = str(value or DEFAULT_IMAGE_VAE).strip().lower()
    for choice in IMAGE_VAE_CHOICES:
        if requested == choice.lower():
            return choice
    raise H3Error(
        f"Unsupported image VAE: {value!r}. Choose {', '.join(IMAGE_VAE_CHOICES)}."
    )


def image_sampling_length(frame_count: int) -> int:
    """Return the smallest native H3 temporal packet covering the request."""
    return 5 if validate_image_frame_count(frame_count) <= 5 else 22


def single_frame_image_sampling_length(frame_count: int) -> int:
    """Use the publisher-supported first slice of H3's shortest packet."""
    frames = validate_image_frame_count(frame_count)
    if frames != 1:
        raise H3Error(
            "Single-frame 500K supports exactly one output image. "
            "Select the official video VAE for multi-frame image results."
        )
    return 5


def selected_image_sampling_length(frame_count: int, image_vae: Any) -> int:
    return (
        single_frame_image_sampling_length(frame_count)
        if normalize_image_vae(image_vae) == SINGLE_FRAME_IMAGE_VAE
        else image_sampling_length(frame_count)
    )


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
    pixel_cap: int = AUTO_RESOLUTION_PIXEL_CAP,
) -> tuple[int, int]:
    """Return an aligned native or capped canvas matching an image ratio."""
    width = float(source_width)
    height = float(source_height)
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
    ):
        raise H3Error("The start frame has no usable image dimensions.")

    if preserve_native:
        return snap_to_grid(width, alignment), snap_to_grid(height, alignment)

    # Do not upscale a smaller source image. Preserve its native dimensions;
    # only oversized inputs need aspect-ratio-based downscaling.
    resolved_pixel_cap = int(pixel_cap)
    if resolved_pixel_cap < 1:
        raise H3Error("The automatic resolution pixel cap must be positive.")
    if width * height <= resolved_pixel_cap:
        return int(width), int(height)

    ratio = width / height
    if ratio >= 1:
        resolved_width = max(
            alignment,
            math.floor(math.sqrt(resolved_pixel_cap * ratio) / alignment) * alignment,
        )
        resolved_height = max(
            alignment,
            math.floor(resolved_width / ratio / alignment) * alignment,
        )
    else:
        resolved_height = max(
            alignment,
            math.floor(math.sqrt(resolved_pixel_cap / ratio) / alignment) * alignment,
        )
        resolved_width = max(
            alignment,
            math.floor(resolved_height * ratio / alignment) * alignment,
        )

    # Rounding to the model grid can cross the selected pixel boundary.
    while resolved_width * resolved_height > resolved_pixel_cap:
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


def auto_resolution_pixel_cap(value: str | None) -> int:
    preset = str(value or DEFAULT_AUTO_RESOLUTION_MEGAPIXELS)
    try:
        return AUTO_RESOLUTION_MEGAPIXEL_PRESETS[preset]
    except KeyError as exc:
        raise H3Error(f"Unknown automatic resolution cap: {value}") from exc


def resolution_choice_values(name: str, tier: str) -> tuple[int, int, str]:
    key = str(tier).strip().lower()
    if key not in RESOLUTION_TIERS:
        raise H3Error(f"Unknown resolution tier: {tier}")

    table = RESOLUTION_TIERS[key]
    if name not in table:
        name = next(choice for choice in table if choice.startswith("16:9 ·"))

    width, height = table[name]
    return (
        width,
        height,
        f"**{dict(draft='768p', fast='1080p', large='2k')[key]}** · "
        + resolution_summary(width, height),
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
        keyframe_tokens = spatial_rows * int(bool(first_image)) + spatial_rows * int(
            bool(last_image)
        )
    return video_tokens + audio_tokens + keyframe_tokens


def resolve_sla_preset(value: str) -> tuple[str, dict[str, Any]]:
    requested = str(value).strip().lower()
    if requested == "balance":
        requested = "balanced"
    for name, inputs in SLA_PRESET_INPUTS.items():
        if requested == name.lower():
            return name, dict(inputs)
    raise H3Error("Unknown SLA preset. Choose Fast, Balanced, or Quality.")


def resolve_cache_policy(cache_mode: str, *, use_turbo: bool) -> tuple[str, str | None]:
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


def active_fl2va_voice_references(mode: str, *slots: Any) -> list[str]:
    if mode != "First / last frame":
        return []
    seen_empty = False
    for value in slots:
        if not value:
            seen_empty = True
        elif seen_empty:
            raise H3Error("Fill FL2VA voice slots in order, starting with voice 1.")
    return collect_reference_slots(*slots)


def ltx25_frame_length(duration: float, fps: float) -> int:
    """LTX-2.5 requires a frame count of 8n+1."""
    return 1 + max(1, int(float(duration) * float(fps)) // 8) * 8
