"""Named updates rendered by UI and API adapters."""

from typing import NamedTuple


class GenerationUpdate(NamedTuple):
    output: str | list[str] | None
    status: str
