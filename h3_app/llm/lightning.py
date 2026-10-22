"""Lightning AI client over its OpenAI-compatible chat API: keys, models and image parts."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from h3_app.catalog import DEFAULT_LIGHTNING_PROMPT_MODEL, LIGHTNING_API_ROOT, LIGHTNING_PROMPT_MODELS
from h3_app.errors import H3Error

from . import media_type


def api_key(temporary_key: str | None) -> str:
    """A key entered in the UI, else LIGHTNING_API_KEY; never stored."""
    key = str(temporary_key or "").strip() or os.getenv("LIGHTNING_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set LIGHTNING_API_KEY in the server environment or enter a temporary "
            "Lightning API key in Prompt enhancer."
        )
    return key


def model(name: str | None) -> str:
    selected = str(name or "").strip() or DEFAULT_LIGHTNING_PROMPT_MODEL
    if selected not in LIGHTNING_PROMPT_MODELS:
        raise H3Error(f"Unsupported Lightning AI prompt model: {selected}")
    return selected


def is_image(path: Path) -> bool:
    return media_type(path).startswith("image/")


def image_part(path: Path) -> dict[str, Any]:
    """An inline high-detail image for a chat message."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:{media_type(path)};base64,{encoded}", "detail": "high"}}


def complete(key: str, model_name: str, system_prompt: str, content: list[dict[str, Any]]) -> str:
    """The assistant's reply text. Client errors become H3Error with the same message."""
    from openai import OpenAI, OpenAIError

    try:
        client = OpenAI(base_url=LIGHTNING_API_ROOT, api_key=key, timeout=600.0)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return str(completion.choices[0].message.content or "")
    except (OpenAIError, IndexError) as exc:
        raise H3Error(str(exc)) from exc
