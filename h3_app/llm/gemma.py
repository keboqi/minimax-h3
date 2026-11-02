"""Free-tier Gemma helpers for small checks, always with a code fallback.

Gemma runs on the Gemini API with the same key, but each model allows only 30
requests and 16K tokens a minute (14.4K requests a day). Every call here is a
best effort: no key, no budget left, a timeout or an unreadable reply all
return None, and the caller keeps its fixed rules. Set H3_GEMMA=0 to turn it off.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import datetime
import functools
import inspect
import os
import threading
import time
from typing import Any, Callable, Iterator

import requests

from h3_app.catalog import GEMINI_API_ROOT

from . import repair_json

MODELS = ("gemma-4-31b-it", "gemma-4-26b-a4b-it")
REQUESTS_PER_MINUTE = 30
TOKENS_PER_MINUTE = 16_000
REQUESTS_PER_DAY = 14_400
# A busy model (429 or 5xx) rests a minute; one the key cannot use (404, e.g. a retired ID) an hour.
BUSY_REST_SECONDS = 60.0
MISSING_REST_SECONDS = 3600.0
DEFAULT_TIMEOUT = 15.0

_key: ContextVar[str] = ContextVar("gemma_key", default="")


@contextmanager
def credentials(temporary_key: str | None) -> Iterator[None]:
    """Use a key entered in the UI for Gemma calls made inside this block."""
    token = _key.set(str(temporary_key or "").strip())
    try:
        yield
    finally:
        _key.reset(token)


def uses_writer_key(writer: Callable) -> Callable:
    """Run a prompt writer with its own Gemini key for the Gemma checks it makes.

    The writer takes ``temporary_api_key`` and ``backend``; an OpenAI-compatible
    writer has no Gemini key to share, so Gemma falls back to GEMINI_API_KEY.
    """
    signature = inspect.signature(writer)

    @functools.wraps(writer)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        backend = bound.get("backend", signature.parameters["backend"].default)
        with credentials(bound.get("temporary_api_key") if backend == "Gemini" else ""):
            return writer(*args, **kwargs)

    return wrapped


def enabled() -> bool:
    return os.getenv("H3_GEMMA", "1").strip().lower() not in ("0", "off", "false", "no")


def _api_key() -> str:
    return _key.get() or os.getenv("GEMINI_API_KEY", "").strip()


def estimate_tokens(text: str) -> int:
    """A cautious count: about four ASCII characters a token, one token per other character."""
    text = str(text or "")
    ascii_chars = sum(char.isascii() for char in text)
    return ascii_chars // 4 + (len(text) - ascii_chars) + 1


class _Budget:
    """Sliding one-minute windows per model, plus a daily request count."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.minute: dict[str, deque[list]] = {model: deque() for model in MODELS}
        self.day: dict[str, tuple[datetime.date, int]] = {}
        self.resting: dict[str, float] = {}

    def reserve(self, tokens: int) -> tuple[str, list] | None:
        """A model with room for ``tokens`` and its ledger entry, or None when all are spent."""
        now = time.monotonic()
        today = datetime.date.today()
        with self.lock:
            for model in MODELS:
                if self.resting.get(model, 0.0) > now:
                    continue
                window = self.minute[model]
                while window and now - window[0][0] >= 60.0:
                    window.popleft()
                day, count = self.day.get(model, (today, 0))
                count = count if day == today else 0
                if (len(window) >= REQUESTS_PER_MINUTE or count >= REQUESTS_PER_DAY
                        or sum(entry[1] for entry in window) + tokens > TOKENS_PER_MINUTE):
                    continue
                entry = [now, tokens]
                window.append(entry)
                self.day[model] = (today, count + 1)
                return model, entry
        return None

    def settle(self, entry: list, tokens: int) -> None:
        with self.lock:
            entry[1] = tokens

    def rest(self, model: str, seconds: float) -> None:
        with self.lock:
            self.resting[model] = time.monotonic() + seconds


_BUDGET = _Budget()


def _reply_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    return "".join(str(part.get("text", "")) for part in parts if not part.get("thought")).strip()


def ask(prompt: str, *, max_output_tokens: int = 512, timeout: float = DEFAULT_TIMEOUT) -> str | None:
    """Gemma's reply to one self-contained prompt, or None so the caller falls back.

    The instructions travel in the user turn: Gemma on the Gemini API has no system role.
    """
    key = _api_key() if enabled() else ""
    if not key or not str(prompt or "").strip():
        return None
    tokens = estimate_tokens(prompt) + max_output_tokens
    body = {"contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_output_tokens}}
    deadline = time.monotonic() + timeout
    for _attempt in MODELS:
        remaining = deadline - time.monotonic()
        reserved = _BUDGET.reserve(tokens) if remaining >= 1.0 else None
        if reserved is None:
            return None
        model, entry = reserved
        try:
            response = requests.post(
                f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body, timeout=remaining,
            )
        except requests.RequestException:
            return None
        if response.status_code in (404, 429) or response.status_code >= 500:
            _BUDGET.rest(model, MISSING_REST_SECONDS if response.status_code == 404 else BUSY_REST_SECONDS)
            continue  # the other model has its own quota and deployment
        if not response.ok:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        used = (payload.get("usageMetadata") or {}).get("totalTokenCount")
        if isinstance(used, int) and used > 0:
            _BUDGET.settle(entry, used)
        return _reply_text(payload) or None
    return None


def ask_json(prompt: str, **kwargs) -> dict | None:
    """Gemma's reply as one JSON object, repaired if needed; None when there is none."""
    reply = ask(prompt, **kwargs)
    value = repair_json(reply) if reply else None
    return value if isinstance(value, dict) else None
