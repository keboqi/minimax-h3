"""Gemini Decision: typed yes/no, choice and score questions answered with probabilities.

The Decisions API answers up to 1,000 independent questions about one shared input in a
single request and returns calibrated probabilities instead of text. Every question sees
the whole input but none sees the others, so a batch names in each question which part of
the input it is about.

``decide`` splits a large question set into requests that stay inside the limits, and
retries a failed chunk in halves so one blocked item does not lose the whole batch.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import requests

from h3_app.catalog import GEMINI_API_ROOT
from h3_app.errors import H3Error

from . import media_type
from .gemini import error

MODEL = "gemini-decision-1"
ENDPOINT = f"{GEMINI_API_ROOT}/v1beta/decisions"
MAX_QUESTIONS = 1000
# Requests are kept well under the 1,000-question ceiling: a smaller batch loses less to one
# failure, and questions about many items in one input are answered more reliably.
DEFAULT_CHUNK = 200
# Inline media is limited to 100 MB per request; base64 grows it by a third.
MAX_INLINE_BYTES = 60 * 1024 * 1024
RETRY_STATUS = (429, 500, 502, 503, 504)
# Refused (HTTP 400) requests one ``decide`` call may retry in halves before it gives up.
MAX_REFUSED = 12


class DecisionError(H3Error):
    """A failed Decision request; ``status`` is the HTTP status, 0 when none was received."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    """Set H3_DECISION=0 to turn every Decision check off."""
    return os.getenv("H3_DECISION", "1").strip().lower() not in ("0", "off", "false", "no")


def key_or_none(temporary_key: str | None = None) -> str | None:
    """A key entered in the UI, else GEMINI_API_KEY; None when there is none or checks are off.

    Decision checks are always optional: without a key the caller keeps its other rules.
    """
    if not enabled():
        return None
    key = str(temporary_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


# Questions ---------------------------------------------------------------------
def yes_no(instructions: str, *, yes: str = "", no: str = "") -> dict[str, Any]:
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if yes or no:
        question["criteria"] = {"true_description": yes, "false_description": no}
    return question


def choice(instructions: str, options: Mapping[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: Iterable[str]) -> dict[str, Any]:
    """An ordered rubric, lowest level first."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# Input blocks ------------------------------------------------------------------
def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": str(value)}


def media(path: str | Path, *, resolution: str = "low") -> dict[str, Any]:
    """An image, video or audio file sent inline."""
    path = Path(path)
    mime = media_type(path)
    kind = mime.split("/", 1)[0]
    if kind not in ("image", "video", "audio"):
        raise H3Error(f"{path.name} is not an image, video or audio file.")
    if path.stat().st_size > MAX_INLINE_BYTES:
        raise H3Error(f"{path.name} is too large to check ({path.stat().st_size // 2**20} MB).")
    block = {"type": kind, "data": base64.b64encode(path.read_bytes()).decode("ascii"), "mime_type": mime}
    if kind != "audio":
        block["resolution"] = resolution
    return block


# Answers -----------------------------------------------------------------------
def yes_probability(answer: Mapping[str, Any] | None) -> float | None:
    """The probability a yes/no answer is yes, or None when there is no such answer."""
    if not isinstance(answer, Mapping) or answer.get("type") != "noul":
        return None
    try:
        return float(answer["noul"])
    except (KeyError, TypeError, ValueError):
        return None


def score_fraction(answer: Mapping[str, Any] | None) -> float | None:
    """A score answer on a 0–1 scale (its expected level over the top level), or None."""
    if not isinstance(answer, Mapping) or answer.get("type") != "score":
        return None
    try:
        levels = len(answer.get("legend") or answer.get("probabilities") or {})
        value = float(answer["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value / (levels - 1))) if levels > 1 else None


# Requests ----------------------------------------------------------------------
def _post(http: requests.Session, key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    for attempt in range(3):
        response = http.post(ENDPOINT, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                             json=body, timeout=max(1.0, deadline - time.monotonic()))
        backoff = 2.0 * (attempt + 1)
        if response.ok or response.status_code not in RETRY_STATUS or attempt == 2 \
                or deadline - time.monotonic() < backoff + 5:
            break
        time.sleep(backoff)
    if not response.ok:
        raise DecisionError(str(error(response, "Decision check")), response.status_code)
    try:
        answers = response.json().get("answers")
    except (ValueError, AttributeError):
        answers = None
    if not isinstance(answers, dict):
        raise DecisionError("Gemini Decision returned no answers.")
    return answers


def decide(questions: Mapping[str, dict[str, Any]], input: Any, *, key: str,
           chunk: int = DEFAULT_CHUNK, timeout: float = 120,
           http: requests.Session | None = None) -> dict[str, dict[str, Any]]:
    """Every question's answer, keyed like ``questions``; questions it could not answer are left out.

    ``input`` is shared by every request: a string, one block or a list of blocks. Questions
    go out ``chunk`` at a time. A chunk the API refuses as a bad request (a blocked item or a
    limit) is retried in halves until single questions fail, which are then skipped, up to
    ``MAX_REFUSED`` refusals; a lone question refused before any answer means the input itself
    is blocked, and stops it. Any other failure (a bad key, rate limits, no connection) stops
    at once. If nothing at all could be answered, the first error is raised.
    """
    items = list(questions.items())
    if not items:
        return {}
    chunk = max(1, min(int(chunk), MAX_QUESTIONS))
    own = http is None
    http = http or requests.Session()
    found: dict[str, dict[str, Any]] = {}
    first_error: DecisionError | None = None
    refused = 0

    def ask(part: list[tuple[str, dict[str, Any]]]) -> None:
        nonlocal first_error, refused
        try:
            found.update(_post(http, key, {"model": MODEL, "input": input, "questions": dict(part)}, timeout))
        except requests.RequestException as exc:
            raise DecisionError(f"Gemini Decision could not be reached: {exc}") from exc
        except DecisionError as exc:
            first_error = first_error or exc
            refused += 1
            # A single question refused before anything was answered means the shared input itself
            # is blocked; splitting further would only repeat the refusal for every question.
            if exc.status != 400 or refused > MAX_REFUSED or (len(part) == 1 and not found):
                raise
            if len(part) > 1:
                middle = len(part) // 2
                ask(part[:middle])
                ask(part[middle:])

    try:
        for start in range(0, len(items), chunk):
            ask(items[start:start + chunk])
    except DecisionError:
        if not found:
            raise
    finally:
        if own:
            http.close()
    if not found and first_error is not None:
        raise first_error
    return found
