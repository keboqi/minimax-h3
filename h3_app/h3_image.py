"""Shared H3 Image limits, prompt recipes and canvas geometry."""

import math
from pathlib import Path

from .errors import H3Error

H3_IMAGE_OFFICIAL = "MiniMax H3 Image (Official)"
H3_IMAGE_PRUNA = "Pruna P-H3-Image"
H3_IMAGE_BACKENDS = (H3_IMAGE_OFFICIAL, H3_IMAGE_PRUNA)
H3_IMAGE_MAX_REFERENCES = 16
H3_IMAGE_MAX_PIXELS = 20 * 1024 * 1024
H3_IMAGE_EDIT_PIXELS = 4096 * 4096
H3_IMAGE_EDIT_SIZE_MAX = "Max resolution (up to 16 MP)"
H3_IMAGE_16M_PRESETS = (
    "16M · 1:1 · 4096×4096", "16M · 4:3 · 4736×3552", "16M · 3:2 · 5024×3360",
    "16M · 16:9 · 5472×3072", "16M · 21:9 · 6272×2688",
    "16M · 3:4 · 3552×4736", "16M · 2:3 · 3360×5024", "16M · 9:16 · 3072×5472",
)


def h3_image_prompt_path(editing=False, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return root / "prompts" / ("prompt_h3_image_edit.txt" if editing else "prompt_h3_image_t2i.txt")


def validate_h3_image_canvas(width, height):
    """Reject malformed dimensions without rounding or silently truncating them."""
    for name, value in (("Width", width), ("Height", height)):
        try:
            valid = (not isinstance(value, bool) and math.isfinite(value)
                     and int(value) == value and 256 <= value <= 8192 and value % 32 == 0)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise H3Error(f"{name} must be 256–8192, in increments of 32.")
    if width * height > H3_IMAGE_MAX_PIXELS:
        raise H3Error("MiniMax H3 Image supports canvases up to 20 MP.")


def h3_image_canvas_description(width, height):
    divisor = math.gcd(int(width), int(height))
    return (f"Selected target canvas: {width}x{height}; "
            f"aspect ratio {int(width) // divisor}:{int(height) // divisor}.")


def official_edit_dimensions(width, height, *, maximize):
    """Scale both axes together, then align to the model grid within the pixel cap."""
    if width <= 0 or height <= 0:
        raise H3Error("The first reference image has invalid dimensions.")
    minimum_scale = 256 / min(width, height)
    maximum_scale = min(8192 / max(width, height), math.sqrt(H3_IMAGE_EDIT_PIXELS / (width * height)))
    if minimum_scale > maximum_scale:
        raise H3Error("The first reference image's aspect ratio is outside MiniMax H3 Image's supported canvas.")
    scale = maximum_scale if maximize else min(max(1.0, minimum_scale), maximum_scale)
    scaled = [max(256, min(8192, round(side * scale / 32) * 32)) for side in (width, height)]
    # Rounding up both sides can cross the area ceiling. Remove one grid cell
    # at a time from the axis whose adjustment best preserves the source ratio.
    ratio = width / height
    while scaled[0] * scaled[1] > H3_IMAGE_EDIT_PIXELS:
        candidates = [(scaled[0] - 32, scaled[1]), (scaled[0], scaled[1] - 32)]
        scaled = list(min((size for size in candidates if min(size) >= 256),
                          key=lambda size: abs(math.log(size[0] / size[1] / ratio))))
    return tuple(scaled)
