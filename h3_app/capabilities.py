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
ENGINES += (
    EngineCapability("flux3", "FLUX 3 Image Klein", ("Image",),
                     "Distilled image generation and editing with up to ten references."),
    EngineCapability("moss_image", "MOSS-Image", ("Image",),
                     "OpenMOSS 5B image generation and reference editing, with layout boxes and Turbo LoRAs."),
    EngineCapability("fibo2", "Fibo 2.0", ("Image",),
                     "Dev image generation and editing, with 4-step Edit Turbo and ordered reference images."),
    EngineCapability("drama3", "Soundstage (Drama 3)", ("TTS",),
                     "Audio production with scripts, cast voices, sound effects, music, takes and mixing. "
                     "Speech powered by Fish Audio Drama 3."),
    EngineCapability("dub", "Dub & Translate", ("Dub",),
                     "Translate a video's speech and dub it with each speaker's cloned voice: transcribe, "
                     "review every line, then dub with Drama 3."),
)
TASKS = ("Video", "Image", "Audio / Music", "TTS", "Dub")
# Navigation and preference restoration use the first engine as the default.
DEFAULT_ENGINES = {"Image": "qwen", "Audio / Music": "music"}


def engines_for_task(task: str) -> tuple[EngineCapability, ...]:
    if task not in TASKS:
        raise ValueError("Unsupported creative task.")
    engines = tuple(engine for engine in ENGINES if task in engine.tasks)
    default_engine = DEFAULT_ENGINES.get(task)
    return tuple(sorted(engines, key=lambda engine: engine.id != default_engine))
