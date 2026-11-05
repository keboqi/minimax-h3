"""Splitting a long selection into windows Remake can render."""

from __future__ import annotations

from typing import Any, Iterable

from h3_app.errors import H3Error

from .model import FPS, MAX_FRAMES, MIN_FRAMES

LENGTHS = tuple(range(MIN_FRAMES, MAX_FRAMES + 1, 17))
DEFAULT_WINDOW_FRAMES = 226
DEFAULT_MIX_FRAMES = 12
# Viggle: the model drops and reorders short shots in a window with many cuts.
MAX_CUTS_PER_WINDOW = 4


def selection_frames(start: Any, end: Any) -> int:
    return int((float(end or 0) - float(start or 0)) * FPS + 1e-6)


def needs_windows(start: Any, end: Any, max_window: int) -> bool:
    """Whether the selection is too long for one window; a shorter excess just rounds down."""
    return selection_frames(start, end) >= int(max_window) + 17


def plan_windows(total: int, cuts: Iterable[int], *, max_frames: int, mix: int) -> list[dict[str, Any]]:
    """Windows over ``total`` frames, ending on cuts where possible.

    ``render`` is the frames a window samples, always an allowed length. ``own`` is the part of
    it the final video keeps. A window that doesn't start on a cut ``continues`` the one before:
    it starts ``mix`` frames before that window ends, and that overlap crossfades when joined.
    """
    lengths = [length for length in LENGTHS if length <= int(max_frames)]
    if not lengths or total < lengths[0]:
        raise H3Error(f"Select at least {MIN_FRAMES / FPS:.2f} seconds of the source.")
    if not 0 <= int(mix) < lengths[0]:
        raise H3Error(f"Mix frames must be below {lengths[0]}.")
    longest = lengths[-1]

    def fit(frames: int) -> int | None:
        return next((length for length in lengths if length >= frames), None)

    cuts = sorted({int(cut) for cut in cuts if 0 < int(cut) < total})
    windows: list[dict[str, Any]] = []
    own_start, render_start, continues = 0, 0, False
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
                        "continues": continues})
        if end is not None:
            own_start, render_start, continues = end, end, False
        else:
            own_start = render_start = own_end - int(mix)
            continues = True
    length = fit(max(total - render_start, lengths[0]))
    # The last window ends with the selection, reaching back into the previous one if it must.
    render_start = min(render_start, total - length)
    windows.append({"render": [render_start, total], "own": [own_start, total], "continues": continues})
    return windows


def carried_frame(previous: dict[str, Any], window: dict[str, Any]) -> tuple[int, int | None]:
    """The last kept frame of ``previous``: its index there, and its pin in ``window``.

    After a cut that frame shows another shot, so it passes on the look without a pin.
    """
    last = previous["own"][1] - 1
    # With no mix frames the next window starts just after it, on its first frame.
    pin = max(0, last - window["render"][0]) if window["continues"] else None
    return last - previous["render"][0], pin
