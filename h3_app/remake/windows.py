"""Splitting a long selection into windows Remake can render."""

from __future__ import annotations

from typing import Any, Iterable

from h3_app.errors import H3Error

from .model import FPS, MAX_FRAMES, MIN_FRAMES

LENGTHS = tuple(range(MIN_FRAMES, MAX_FRAMES + 1, 17))
DEFAULT_WINDOW_FRAMES = 226
# Viggle's hop: a continuing window starts on this many frames of the render before it.
HOLD_FRAMES = 17
# Viggle: the model drops and reorders short shots in a window with many cuts.
MAX_CUTS_PER_WINDOW = 4


def selection_frames(start: Any, end: Any) -> int:
    return int((float(end or 0) - float(start or 0)) * FPS + 1e-6)


def needs_windows(start: Any, end: Any, max_window: int) -> bool:
    """Whether the selection is too long for one window; a shorter excess just rounds down."""
    return selection_frames(start, end) >= int(max_window) + 17


def plan_windows(total: int, cuts: Iterable[int], *, max_frames: int) -> list[dict[str, Any]]:
    """Windows over ``total`` frames, ending on cuts where possible.

    ``render`` is the frames a window samples, always an allowed length. ``own`` is the part of
    it the final video keeps. A window that doesn't start on a cut ``continues`` the one before:
    its first ``hold`` frames are the last frames that window kept, so the two meet in a plain cut.
    """
    lengths = [length for length in LENGTHS if length <= int(max_frames)]
    if not lengths or total < lengths[0]:
        raise H3Error(f"Select at least {MIN_FRAMES / FPS:.2f} seconds of the source.")
    longest = lengths[-1]

    def fit(frames: int) -> int | None:
        return next((length for length in lengths if length >= frames), None)

    cuts = sorted({int(cut) for cut in cuts if 0 < int(cut) < total})
    windows: list[dict[str, Any]] = []
    own_start, render_start, hold = 0, 0, 0
    while total - render_start > longest:
        candidates = [
            cut for cut in cuts
            if render_start + lengths[0] <= cut <= render_start + longest and cut > own_start
            and render_start + fit(cut - render_start) <= total
        ]
        calm = [cut for cut in candidates
                if sum(render_start < c < render_start + fit(cut - render_start) for c in cuts) <= MAX_CUTS_PER_WINDOW]
        end = max(calm) if calm else min(candidates) if candidates else None
        length = fit(end - render_start) if end is not None else longest
        own_end = end if end is not None else render_start + longest
        windows.append({"render": [render_start, render_start + length], "own": [own_start, own_end],
                        "continues": bool(hold), "hold": hold})
        if end is not None:
            own_start, render_start, hold = end, end, 0
        else:
            own_start, render_start, hold = own_end, own_end - HOLD_FRAMES, HOLD_FRAMES
    length = fit(max(total - render_start, lengths[0]))
    # The last window ends with the selection, reaching back into the previous one if it must.
    render_start = min(render_start, total - length)
    if hold:  # Its held frames stay the last the previous window keeps.
        own_start = windows[-1]["own"][1] = render_start + hold
    windows.append({"render": [render_start, total], "own": [own_start, total], "continues": bool(hold), "hold": hold})
    return windows


def carried_frame(previous: dict[str, Any], window: dict[str, Any]) -> tuple[int, int | None]:
    """The last kept frame of ``previous``: its index there, and its pin in ``window``.

    After a cut that frame shows another shot, and held frames already fix it in place, so either
    way it passes on the look without a pin. Only projects from before held frames pin it.
    """
    last = previous["own"][1] - 1
    pinned = window["continues"] and not window.get("hold")
    return last - previous["render"][0], max(0, last - window["render"][0]) if pinned else None
