"""FLUX 3 prompt writer using BFL's official upsampler instructions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from . import common


COMMON_CANVAS_RATIOS = ((1, 1), (5, 4), (4, 3), (3, 2), (16, 9), (2, 1), (21, 9))


def _canvas_ratio(width: int, height: int) -> str:
    """Name the canvas as a common W:H when within 2%, else the exact reduced ratio."""
    width, height = int(width), int(height)
    ratio = width / height
    candidates = [*COMMON_CANVAS_RATIOS, *((h, w) for w, h in COMMON_CANVAS_RATIOS)]
    w, h = min(candidates, key=lambda pair: abs(math.log(ratio * pair[1] / pair[0])))
    if abs(math.log(ratio * h / w)) <= 0.02:
        return f"{w}:{h}"
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"


FLUX3_REFERENCE_ORDINALS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)


def enhance_flux3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    reference_images: Any,
    width: int,
    height: int,
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Expand a FLUX 3 request with BFL's official upsampler instructions."""
    references = reference_images or []
    if isinstance(references, (str, Path, dict)):
        references = [references]
    if len(references) > len(FLUX3_REFERENCE_ORDINALS):
        return str(prompt or ""), "Prompt enhancement failed: FLUX 3 supports at most ten reference images."
    request = str(prompt or "").strip()
    if references:
        # The official edit upsampler sees the references in order.
        target = "FLUX 3 Image Klein edit"
        request_text = request or "Make the most plausible improvement to the supplied images."
    else:
        if not request:
            return str(prompt or ""), "Prompt enhancement failed: write an image request first."
        target = "FLUX 3 Image Klein"
        request_text = f"TARGET CANVAS: {_canvas_ratio(width, height)}\n{request}"
    return common.write_prompt(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target=target,
        system_path=runtime.prompt_systems[target],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=tuple(
            (f"the {FLUX3_REFERENCE_ORDINALS[index]} image", image)
            for index, image in enumerate(references)
        ),
        context="",
        request_text=request_text,
    )
