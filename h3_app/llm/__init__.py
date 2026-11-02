"""Clients for the hosted language models behind the prompt writers and Dub & Translate."""

from __future__ import annotations

import mimetypes
from pathlib import Path
import re

_SUFFIX_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


def media_type(path: Path) -> str:
    """The MIME type sent for an uploaded file, with fallbacks for common media."""
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    return _SUFFIX_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _closes(text: str) -> bool:
    """Whether the first JSON object or array in ``text`` reaches its closing bracket."""
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            escaped, quoted = (False, True) if escaped else (char == "\\", char != '"')
        elif char == '"':
            quoted = depth > 0
        elif char in "{[":
            depth += 1
        elif char in "}]" and depth:
            depth -= 1
            if not depth:
                return True
    return False


def repair_json(text: str):
    """The JSON value json-repair recovers from a malformed but finished reply, or None.

    It drops trailing commas and fixes quoting and stray prose around the JSON. A reply
    that never closes its outer bracket was cut off, and closing it would silently drop
    the rest, so that is never repaired.
    """
    text = str(text or "")
    if not _closes(text):
        return None
    try:
        import json_repair
    except ImportError:
        return None
    try:
        value = json_repair.loads(text)
    except Exception:  # noqa: BLE001 - a repair failure is just "not JSON"
        return None
    # json-repair returns "" when it finds no JSON at all.
    return value if isinstance(value, (dict, list)) else None


def strip_code_fence(text: str) -> str:
    """Remove one Markdown code fence that wraps the whole reply."""
    text = str(text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text
