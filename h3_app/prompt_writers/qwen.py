"""Qwen Image 2.1 prompt writer for generation and edits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL
from h3_app.config import RuntimeConfig
from . import common


def enhance_qwen_image21_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
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
    references = reference_images or []
    if isinstance(references, (str, Path, dict)):
        references = [references]
    reference_count = len(references)
    if reference_count == 1:
        media_values = (("Input image (edit target)", references[0]),)
        edit_context = (
            "In Image edit mode, the uploaded image is the edit target. "
            "Refer to it naturally and do not use an <image1> tag."
        )
    else:
        media_values = tuple(
            (f"<image{index}>", image)
            for index, image in enumerate(references, 1)
        )
        edit_context = (
            "In Image edit mode with multiple inputs, <image1> is the edit "
            "target and later images are references. Use the numbered image "
            "tags verbatim."
        )
    return common.write_prompt(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="Qwen Image 2.1",
        system_path=runtime.prompt_systems["Qwen Image 2.1"],
        backend=backend,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        media_values=media_values,
        context=(
            f"Mode: {mode}\nOutput: {int(width)}x{int(height)}\n"
            f"{edit_context}"
        ),
    )
