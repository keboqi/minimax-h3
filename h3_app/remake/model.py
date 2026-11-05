"""What a Viggle Remake run must satisfy, taken from its sampler and ComfyUI nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from h3_app.errors import H3Error

FPS = 24
MIN_FRAMES, MAX_FRAMES = 90, 345
# Viggle recommends staying at or below this length.
RECOMMENDED_MAX_FRAMES = 294
SHORT_EDGES = tuple(range(480, 737, 32))
# Follows the source: the finished video matches its shape and, up to a limit, its size.
AUTO_SHORT_EDGE = 0
DEFAULT_SHORT_EDGE = AUTO_SHORT_EDGE
GRID = 32
# Auto stops here: a 1080p finish, because a doubled 736 is far heavier than a regular H3 job.
AUTO_MAX_OUTPUT_EDGE = 1088
# No side reaches past what Viggle's largest canvas holds, whatever the source's shape.
MAX_CANVAS_PIXELS = 1280 * 736
MAX_PICTURES = 4
SWAP, EDIT = "Swap", "Edit"
SIGMAS = {
    SWAP: "1.0, 0.990476, 0.965243, 0.923077, 0.8, 0.631579, 0.0",
    EDIT: "1.0, 0.983607, 0.96, 0.923077, 0.857143, 0.705882, 0.0",
}
STEPS = 6
# Normal generation refuses fewer steps. The request carries this to pass that check;
# Remake's own schedule replaces it once the run is prepared.
REQUEST_STEPS = 10
LORA_KEYS = {SWAP: "remake_swap_lora", EDIT: "remake_edit_lora"}
# Pinned keyframes hold for the steps at or above this sigma: the first four of six.
KEYFRAME_UNTIL = 0.9
SECTIONS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
)
REMAKE_NODES = frozenset({
    "RemakeConditioning", "RemakeKeyframeUntil", "RemakeNoise", "RemakeHoldAudio", "RemakeHoldFrames",
    "LoraLoaderBypassModelOnly", "ManualSigmas", "LoadVideo", "GetVideoComponents",
})


@dataclass(frozen=True)
class RemakeGraph:
    """The parts of a run the H3 request does not carry. Paths are relative to ComfyUI's input."""

    mode: str
    source_video: str
    pictures: tuple[str, ...]
    pins: tuple[int | None, ...]
    frames: int
    output_prefix: str | None = None
    held_frames: str | None = None

    @property
    def lora_key(self) -> str:
        return LORA_KEYS[self.mode]

    @property
    def sigmas(self) -> str:
        return SIGMAS[self.mode]


def frames_for_seconds(seconds: float) -> int:
    """The longest allowed length (17n+5 frames) that fits inside ``seconds``."""
    available = int(float(seconds) * FPS + 1e-6)
    if available < MIN_FRAMES:
        raise H3Error(f"Select at least {MIN_FRAMES / FPS:.2f} seconds of the source.")
    return min(MAX_FRAMES, (available - 5) // 17 * 17 + 5)


def canvas_size(short_edge: int, source_width: int, source_height: int) -> tuple[int, int]:
    """The canvas in the source's shape, as ``(width, height)``, with both sides on the grid.

    Rounding down keeps Viggle's sizes for 16:9 sources, and anything left over is center-cropped.
    """
    short_edge = int(short_edge)
    if short_edge not in SHORT_EDGES:
        raise H3Error(f"Remake's short edge must be one of {SHORT_EDGES[0]}–{SHORT_EDGES[-1]} in steps of 32.")
    shorter, longer = sorted((int(source_width), int(source_height)))
    long_edge = min(short_edge * longer // shorter, MAX_CANVAS_PIXELS // short_edge) // GRID * GRID
    return (long_edge, short_edge) if source_width >= source_height else (short_edge, long_edge)


def output_size(width: int, height: int, latent_upscale: bool) -> tuple[int, int]:
    """The finished video's size: latent upscale doubles the sampled canvas."""
    return (width * 2, height * 2) if latent_upscale else (width, height)


@dataclass(frozen=True)
class RenderSize:
    """What a run samples, and whether latent upscale doubles it; ``None`` leaves that to H3."""

    width: int
    height: int
    latent_upscale: bool | None = None

    def upscale(self, h3_setting: Any) -> bool:
        return bool(h3_setting) if self.latent_upscale is None else self.latent_upscale


def render_size(short_edge: int, source_width: int, source_height: int) -> RenderSize:
    """The size of a run. A chosen short edge is sampled as is; Auto reaches the source's own size.

    Auto samples directly while the canvas can reach the source, and past that samples half the
    size and doubles it, so the finish doesn't depend on the H3 latent upscale setting.
    """
    if int(short_edge) != AUTO_SHORT_EDGE:
        return RenderSize(*canvas_size(short_edge, source_width, source_height))
    target = min(source_width, source_height, AUTO_MAX_OUTPUT_EDGE)
    upscale = target > SHORT_EDGES[-1]
    wanted = (target / 2 if upscale else target) + GRID / 2
    sampled = max(SHORT_EDGES[0], min(SHORT_EDGES[-1], int(wanted) // GRID * GRID))
    return RenderSize(*canvas_size(sampled, source_width, source_height), upscale)


def refine_tail_steps(sigmas: str, requested: int) -> int:
    """Refinement steps below the keyframe hold: its conditioning stays at the sampled size."""
    values = [float(value) for value in sigmas.split(",")]
    return max(1, min(int(requested), sum(0 < value < KEYFRAME_UNTIL for value in values)))


def check_pins(pins: tuple[int | None, ...], pictures: int, frames: int) -> None:
    """One entry per picture, or none at all; ``None`` leaves a picture unpinned."""
    if not pins:
        return
    if len(pins) != pictures:
        raise H3Error("Pin every picture to a frame, or turn pinning off.")
    if any(pin is not None and not 0 <= pin < frames for pin in pins):
        raise H3Error(f"Pins must be frames from 0 to {frames - 1} of the selection.")


def missing_sections(prompt: str) -> list[str]:
    """The six-section headers absent from ``prompt``, each searched after the one before it."""
    position, missing = 0, []
    for name in SECTIONS:
        found = re.compile(rf"^\s*{name}\s*:", re.MULTILINE).search(str(prompt or ""), position)
        if found is None:
            missing.append(name)
        else:
            position = found.end()
    return missing


def request_overrides(*, prompt: str, first_picture: str, width: int, height: int, frames: int,
                      latent_upscale: bool) -> dict[str, Any]:
    """The H3 request fields Remake owns; everything else stays the MiniMax H3 selection.

    First / last frame selects the FL2VA checkpoint and marks the run as visually conditioned.
    Remake's own graph consumes the pictures, and a video's size never derives from them.
    """
    final_width, final_height = output_size(width, height, latent_upscale)
    return {
        "mode": "First / last frame",
        "result_format": "Video",
        "prompt": prompt,
        "first_image": first_picture,
        "last_image": None,
        "duration": frames / FPS,
        "width": final_width,
        "height": final_height,
        "generation_mode": "Normal",
        "steps": REQUEST_STEPS,
        "scheduler": "simple",
        "semantic_bridge": False,
        "semantic_bridge_alpha": 0.0,
        "latent_upscale": bool(latent_upscale),
        "latent_upscale_refine_lora": "Same as generation",
        "fl2va_audio_1": None,
        "fl2va_audio_2": None,
        "fl2va_audio_3": None,
    }
