"""Resolution, result-format and frame controls of the H3 composer: Gradio updates computed from the current values."""

from __future__ import annotations

from typing import Any, Iterable
import gradio as gr
from h3_ui.presentation import result_format_presentation
from h3_app.errors import H3Error
from h3_app.catalog import (
    DEFAULT_AUTO_RESOLUTION_MEGAPIXELS,
    DEFAULT_IMAGE_VAE,
    DEFAULT_RESULT_FORMAT,
    H3_LATENT_UPSCALE_SPLIT,
    INPUT_IMAGE_FRAME_PRESETS,
    UI_DEFAULTS,
)
from h3_app.policy import (
    auto_resolution_pixel_cap,
    normalize_image_vae,
    normalize_paths,
    normalize_result_format,
    resolution_choice_values,
    resolution_for_aspect_ratio,
    resolution_summary,
    resolve_h3_latent_upscale_method,
    snap64,
    snap_to_grid,
    validate_resolution,
)


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


def input_image_frame_preset_updates(
    preset: str, current_width: int | float, current_height: int | float
) -> tuple[Any, Any]:
    dimensions = INPUT_IMAGE_FRAME_PRESETS.get(str(preset))
    if dimensions is None:
        return (gr.update(value=current_width), gr.update(value=current_height))
    return dimensions


def image_frame_labels(frame_paths: Iterable[Any]) -> list[str]:
    return [f"Frame {index + 1}" for index, _ in enumerate(frame_paths)]


def select_all_image_frames(frame_paths: Iterable[Any]) -> list[str]:
    return image_frame_labels(frame_paths)
