"""Explicit task/engine navigation capabilities; no model initialization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCapability:
    id: str
    label: str
    tasks: tuple[str, ...]
    description: str


ENGINES = (
    EngineCapability(
        "h3",
        "MiniMax H3",
        ("Video", "Image", "Audio / Music"),
        "Joint video/audio generation. Image and audio tasks decode those parts of the H3 result.",
    ),
    EngineCapability(
        "ltx",
        "LTX 2.5",
        ("Video",),
        "Video generation with start, middle, end and ingredient conditioning.",
    ),
    EngineCapability(
        "qwen",
        "Qwen Image 2.1",
        ("Image",),
        "Native image generation and image editing with multiple references.",
    ),
    EngineCapability(
        "music",
        "MiniMax Music 3",
        ("Audio / Music",),
        "Song generation from a caption and structured lyrics.",
    ),
    EngineCapability(
        "yue2",
        "YuE2",
        ("Audio / Music",),
        "Music and song generation with lyrics and score-specific controls.",
    ),
)
TASKS = ("Video", "Image", "Audio / Music")


def engines_for_task(task: str) -> tuple[EngineCapability, ...]:
    if task not in TASKS:
        raise ValueError("Unsupported creative task.")
    return tuple(engine for engine in ENGINES if task in engine.tasks)
