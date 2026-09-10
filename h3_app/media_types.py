"""Extracted media types boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    duration: float = 0.0
    frame_count: int = 0
    has_audio: bool = False


@dataclass(frozen=True)
class UpscaleClipBatch:
    sources: tuple[str, ...]
    temporary_inputs: tuple[Path, ...] = ()
    temporary_directory: Path | None = None
