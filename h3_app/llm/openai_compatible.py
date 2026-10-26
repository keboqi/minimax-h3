"""OpenAI-compatible chat client: keys, endpoint, models and image parts."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL, OPENAI_API_ROOT, OPENAI_BASE_URL_ENV
from h3_app.errors import H3Error

from . import media_type


def api_key(temporary_key: str | None, temporary_url: str | None = None) -> str:
    """A key entered in the UI, else OPENAI_API_KEY; never stored.

    The server's key goes only to the server's endpoint, so an endpoint entered
    in the UI needs a key entered with it.
    """
    key = str(temporary_key or "").strip()
    if key:
        return key
    if str(temporary_url or "").strip():
        raise H3Error(
            "Enter a temporary API key with the temporary endpoint; the server's "
            "OPENAI_API_KEY is sent only to OPENAI_BASE_URL or the official OpenAI API."
        )
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set OPENAI_API_KEY in the server environment or enter a temporary "
            "OpenAI API key in Prompt enhancer."
        )
    return key


_OPERATION_SUFFIXES = ("/chat/completions", "/responses", "/completions")


def endpoint(temporary_url: str | None = None) -> str:
    """An endpoint entered in the UI, else OPENAI_BASE_URL, else the official OpenAI API; never stored.

    Portals such as Azure AI Foundry show the full operation URL
    (``.../openai/v1/responses``); the client appends its own path, so a pasted
    operation suffix is dropped to leave the base URL.
    """
    url = (str(temporary_url or "").strip() or os.getenv(OPENAI_BASE_URL_ENV, "").strip()
           or OPENAI_API_ROOT).rstrip("/")
    for suffix in _OPERATION_SUFFIXES:
        if url.lower().endswith(suffix):
            return url[: -len(suffix)]
    return url


def model(name: str | None) -> str:
    """Any model the endpoint serves; the catalog list only offers suggestions."""
    return str(name or "").strip() or DEFAULT_OPENAI_PROMPT_MODEL


def is_image(path: Path) -> bool:
    return media_type(path).startswith("image/")


def image_part(path: Path) -> dict[str, Any]:
    """An inline high-detail image for a chat message."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:{media_type(path)};base64,{encoded}", "detail": "high"}}


def complete(key: str, model_name: str, system_prompt: str, content: list[dict[str, Any]],
             temporary_url: str | None = None) -> str:
    """The assistant's reply text. Client errors become H3Error with the same message."""
    from openai import NotFoundError, OpenAI, OpenAIError

    base_url = endpoint(temporary_url)
    azure = (urlparse(base_url).hostname or "").endswith(".azure.com")
    if base_url == OPENAI_API_ROOT or azure:
        # Router names such as openai/gpt-6.1-sol are bare model IDs on OpenAI's own API,
        # and Azure deployment names cannot contain a slash.
        model_name = model_name.removeprefix("openai/")
    try:
        client = OpenAI(base_url=base_url, api_key=key, timeout=600.0)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return str(completion.choices[0].message.content or "")
    except NotFoundError as exc:
        if azure and "DeploymentNotFound" in str(exc):
            raise H3Error(
                f"Azure has no deployment named {model_name!r}. Type the deployment's Name "
                "from its Details page into OpenAI-compatible model."
            ) from exc
        raise H3Error(str(exc)) from exc
    except (OpenAIError, IndexError) as exc:
        raise H3Error(str(exc)) from exc
