"""Viggle Remake's six-section prompt from an edit request, the source and the painted frames."""

from __future__ import annotations

from typing import Any, Sequence

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from h3_app.remake import FPS, SECTIONS, missing_sections

from . import common


def remake_writer_context(mode: str, frames: int, width: int, height: int, pictures: int,
                          pins: Sequence[int]) -> str:
    lines = [
        f"Remake type: {mode}",
        f"Output duration: {frames / FPS:.3f} seconds ({frames} frames at {FPS} fps)",
        f"Output canvas: {width}x{height}",
        f"Painted frames: {pictures}",
        "Sections, in order: " + ", ".join(SECTIONS),
    ]
    lines += [
        f"<Picture {index}> is pinned at {pin / FPS:.3f} seconds (frame {pin})."
        for index, pin in enumerate(pins, 1)
    ]
    return "\n".join(lines)


def enhance_remake_prompt(
    request: str,
    mode: str,
    source_clip: Any,
    pictures: Sequence[Any],
    pins: Sequence[int],
    frames: int,
    width: int,
    height: int,
    model: str,
    temporary_api_key: str,
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """The writer's reply and its status; only Gemini reads the source clip itself."""
    media = [(f"<Picture {index}>, a painted frame", path) for index, path in enumerate(pictures, 1)]
    if backend == "Gemini":
        media.insert(0, ("<Video 1>, the source video", source_clip))
    prompt, status = common.write_prompt(
        prompt=request,
        model=model,
        temporary_api_key=temporary_api_key,
        target="Viggle Remake",
        system_path=runtime.prompt_systems["Viggle Remake"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=media,
        context=remake_writer_context(mode, frames, width, height, len(pictures), pins),
    )
    if "failed" in status:
        return prompt, status
    if backend != "Gemini":
        status += " This service can't read video, so it only saw your painted frames."
    missing = missing_sections(prompt)
    if missing:
        status += " Check this prompt: it is missing " + ", ".join(missing) + "."
    return prompt, status
