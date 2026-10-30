"""Plain records for probed video metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    duration: float = 0.0
    frame_count: int = 0
    has_audio: bool = False
