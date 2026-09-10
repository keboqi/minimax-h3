"""Named execution updates and the UI-independent progress callback contract."""

from typing import Any, NamedTuple, Protocol


class ProgressCallback(Protocol):
    def __call__(self, value: Any, *, desc: str = "") -> None: ...


def no_progress(*args: Any, **kwargs: Any) -> None:
    pass


class ProgressUpdate(NamedTuple):
    stage: str
    completed_nodes: int
    total_nodes: int
    step: int | None = None
    step_total: int | None = None
