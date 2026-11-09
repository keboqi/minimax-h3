"""What a Soundstage result records for its rich view, shared by the mix, Speak and Media.

Imports no audio code, so the GPU-free Library Console can read results too.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

from h3_app.provenance import read_snapshot

# The settings snapshot's family for Soundstage results.
DRAMA3_FAMILY = "Drama 3"
# The narrator stands in the middle; characters take these places in order of first line.
NARRATOR_NAMES = frozenset({"narrator", "narration", "旁白", "storyteller", "story teller"})
CHARACTER_POSITIONS = (-0.3, 0.3, -0.15, 0.15, -0.45, 0.45)


def script_fingerprint(text: str) -> str:
    """Identify the source script without retaining another copy of its content."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def is_narrator(speaker: str) -> bool:
    return " ".join(str(speaker).split()).casefold() in NARRATOR_NAMES


def result_settings(path: str | Path | None) -> dict:
    """A result's settings snapshot, or {}."""
    return (read_snapshot(path) or {}).get("settings") or {}
