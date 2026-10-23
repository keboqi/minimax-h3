"""LTX 2.5 prompt writer from text and keyframes."""

from __future__ import annotations

from typing import Any

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from . import common


def enhance_ltx25_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    start_image: Any,
    middle_image: Any,
    end_image: Any,
    duration: float,
    width: int,
    height: int,
    backend: str = "Gemini",
    openai_api_key: str = "",
    openai_model: str = DEFAULT_OPENAI_PROMPT_MODEL,
    openai_base_url: str | None = None,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    return common.write_prompt(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="LTX-2.5",
        system_path=runtime.prompt_systems["LTX-2.5"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=(
            ("Start keyframe", start_image),
            ("Middle keyframe", middle_image),
            ("End keyframe", end_image),
        ),
        context=f"Mode: {mode}\nDuration: {float(duration):.2f} seconds\nOutput: {int(width)}x{int(height)}",
    )
