"""Named updates rendered by UI and API adapters."""

from typing import Any, NamedTuple


class GenerationUpdate(NamedTuple):
    output: str | list[str] | dict[str, Any] | None
    status: str


def fresh_audio(path: str) -> dict[str, Any]:
    """A finished audio result that starts playback from the beginning.

    Gradio's player keeps its playback position across value changes and seeks a
    new file to it; this is the dict ``gr.update(value=..., playback_position=0)``
    produces, built here so generation code needs no Gradio import.
    """
    return {"__type__": "update", "value": str(path), "playback_position": 0}
