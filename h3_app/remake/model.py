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
DEFAULT_SHORT_EDGE = 480
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
    "RemakeConditioning", "RemakeKeyframeUntil", "RemakeNoise", "RemakeHoldAudio",
    "LoraLoaderBypassModelOnly", "ManualSigmas", "LoadVideo", "GetVideoComponents", "ImageScale",
})


@dataclass(frozen=True)
class RemakeGraph:
    """The parts of a run the H3 request does not carry. Paths are relative to ComfyUI's input."""

    mode: str
    source_video: str
    pictures: tuple[str, ...]
    pins: tuple[int, ...]
    frames: int

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
    """Remake's 16:9-class canvas, in the source's orientation, as ``(width, height)``."""
    short_edge = int(short_edge)
    if short_edge not in SHORT_EDGES:
        raise H3Error(f"Remake's short edge must be one of {SHORT_EDGES[0]}–{SHORT_EDGES[-1]} in steps of 32.")
    long_edge = short_edge * 16 // 9 // 32 * 32
    return (long_edge, short_edge) if source_width >= source_height else (short_edge, long_edge)


def output_size(width: int, height: int, latent_upscale: bool) -> tuple[int, int]:
    """The finished video's size: latent upscale doubles the sampled canvas."""
    return (width * 2, height * 2) if latent_upscale else (width, height)


def check_pins(pins: tuple[int, ...], pictures: int, frames: int) -> None:
    """Remake takes one pin per picture, or none at all."""
    if not pins:
        return
    if len(pins) != pictures:
        raise H3Error("Pin every picture to a frame, or turn pinning off.")
    if any(not 0 <= pin < frames for pin in pins):
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
        "latent_upscale_refine_lora": "Same as generation",
        "fl2va_audio_1": None,
        "fl2va_audio_2": None,
        "fl2va_audio_3": None,
    }
