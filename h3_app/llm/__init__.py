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


def strip_code_fence(text: str) -> str:
    """Remove one Markdown code fence that wraps the whole reply."""
    text = str(text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text
