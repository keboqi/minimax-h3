"""Dubbing timing layout, Smart timing placement solver, and FFmpeg atempo filter construction."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

import numpy as np

TIMING_SMART = "Smart timing"
TIMING_MATCH = "Match the original timing"
TIMING_NATURAL = "Natural pace"
TIMINGS = (TIMING_SMART, TIMING_MATCH, TIMING_NATURAL)

SMART_GAP = 0.1
SMART_LEAD = 0.3
SMART_DELAY = 0.5
SMART_SPEED = 1.25
SMART_PAUSE = (0.25, 0.15)
TEMPO_WARNING = (0.75, 2.0)
LINE_GAP = 0.05


def slot_end(lines: list[dict], index: int, duration: float) -> float:
    """Where a line's take must end: just before the next line starts, never before its own end.

    Lines that overlap it in the source (another speaker at the same time) do not cut it short.
    """
    segment = lines[index]
    later = [s["start"] for s in lines[index + 1:] if s["start"] >= segment["end"] - 1e-6]
    limit = (min(later) - LINE_GAP) if later else (duration or segment["end"] + 2.0)
    return max(segment["end"], limit)


def atempo_chain(tempo: float) -> str:
    """An atempo filter for any tempo: each stage stays within 0.5–2x, as in index-tts-vllm."""
    tempo = max(0.01, float(tempo))
    factors = []
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    factors.append(tempo)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


@dataclass(frozen=True)
class SmartPlacement:
    start: float  # where the take starts on the timeline
    seconds: float  # how long it plays once fitted
    tighten: bool  # its inner pauses are shortened
    tempo: float
    room: float  # the time it could use without overlapping, borrowing included
    squeezed: bool  # still too long after the gentle speed-up: sped up to the limit, and marked


def smart_layout(lines: list[dict], natural: dict[int, float], tightened: Callable[[int], float],
                 max_speed: float) -> dict[int, SmartPlacement]:
    """Place each voiced line (``natural``: its take's own length) on the timeline, trying in turn:

    1. its own length at its own start, into the silence before the next line (less SMART_GAP);
    2. starting up to SMART_LEAD earlier, into silence the previous dubbed line left;
    3. its inner pauses shortened (``tightened`` gives that length);
    4. speeding up to SMART_SPEED;
    5. pushing the next line up to SMART_DELAY later, out of the time that line will not use,
       then speeding up to SMART_SPEED again;
    6. speeding up to ``max_speed`` and marking it; what still does not fit overlaps the next line.

    Lines not in ``natural`` keep their original voice at their original times and are not moved.
    Lines that overlap in the source (two speakers at once) do not hold each other back. A line is
    never started more than SMART_DELAY late, so an overlap cannot push every later line along.
    """
    gentle = min(SMART_SPEED, max(1.0, float(max_speed)))
    next_line = [next((later for later in range(index + 1, len(lines))
                       if lines[later]["start"] >= line["end"] - 1e-6), None)
                 for index, line in enumerate(lines)]

    def slack(index: int) -> float:
        """How far a dubbed line can be pushed and still play at its own length before its own next line."""
        if index not in natural:
            return 0.0
        if next_line[index] is None:
            return SMART_DELAY
        room = lines[next_line[index]]["start"] - SMART_GAP - lines[index]["start"]
        return min(SMART_DELAY, max(0.0, room - natural[index]))

    def fits(start: float, seconds: float, end_limit: float) -> bool:
        return start + seconds <= end_limit + 1e-6

    def sped(start: float, seconds: float, end_limit: float, cap: float) -> float | None:
        """The tempo that fits ``seconds`` between ``start`` and ``end_limit``, if it is at most ``cap``."""
        room = end_limit - start
        return seconds / room if room > 0 and seconds <= room * cap else None

    placements: dict[int, SmartPlacement] = {}
    finished: list[tuple[float, float]] = []  # (original end, placed end) of the lines before
    previous_end = None
    for index, line in enumerate(lines):
        while finished and finished[0][0] <= line["start"] + 1e-6:
            placed_end = heapq.heappop(finished)[1]
            previous_end = placed_end if previous_end is None else max(previous_end, placed_end)
        if index not in natural:
            heapq.heappush(finished, (line["end"], line["end"]))
            continue
        after = line["start"] if previous_end is None else min(previous_end + SMART_GAP,
                                                              line["start"] + SMART_DELAY)
        normal, earliest = max(line["start"], after), max(0.0, line["start"] - SMART_LEAD, after)
        following = next_line[index]
        limit = math.inf if following is None else lines[following]["start"] - SMART_GAP
        borrowed = limit if following is None else limit + slack(following)
        seconds, tighten, tempo, squeezed = natural[index], False, 1.0, False
        if not fits(earliest, seconds, limit):
            short = tightened(index)
            if short < seconds - 0.01:
                seconds, tighten = short, True
        if fits(earliest, seconds, limit):
            end_limit = limit
        elif (speed := sped(earliest, seconds, limit, gentle)) is not None:
            tempo, end_limit = speed, limit
        elif fits(earliest, seconds, borrowed):
            end_limit = borrowed
        elif (speed := sped(earliest, seconds, borrowed, gentle)) is not None:
            tempo, end_limit = speed, borrowed
        else:
            room = borrowed - earliest
            tempo = max(1.0, min(seconds / room if room > 0 else math.inf, float(max_speed)))
            end_limit, squeezed = borrowed, True
        length = seconds / tempo
        start = earliest if tempo > 1.0 else max(earliest, min(normal, end_limit - length))
        placements[index] = SmartPlacement(round(start, 3), length, tighten, tempo, end_limit - earliest, squeezed)
        heapq.heappush(finished, (line["end"], start + length))
    return placements


def smart_note(placement: SmartPlacement) -> str:
    """Why a squeezed line is marked: how fast it plays and whether it still overlaps."""
    over = placement.seconds - placement.room
    if over > 0.05:
        return f"too long even at {placement.tempo:.2f}x: overlaps the next line by {over:.1f}s"
    return f"too long for its slot: sped up {placement.tempo:.2f}x"


@dataclass(frozen=True)
class Timed:
    samples: np.ndarray
    start: float
    speed: float
    note: str
    available: float
    squeezed: bool = False
