"""Script anchors for plans written before speech exists."""

from bisect import bisect_right
from copy import deepcopy
from math import ceil, isfinite
import re

from h3_app.errors import H3Error
from .serving.planner import spoken_text
from .serving.tree import parse_drama3_tree, iter_speech_nodes
from .visuals import FPS


def script_cues(text):
    return [dict(speaker=turn.speaker, text=turn.text)
            for node in iter_speech_nodes(parse_drama3_tree(text)) for turn in node.turns
            if _plain(turn.text)]


def preview_settings(text, title="", voices=None):
    cues = script_cues(text)
    if not cues:
        raise H3Error("Write a script before planning shots.")
    return dict(title=title, voices=voices or {}, script=text, script_cues=cues,
                duration=max(400, len(cues) * 10), untimed=True)


def bind_plan(plan, settings):
    """Resolve zero-based script cue + fractional offset against actual spoken lines.

    Normalized character positions handle speech chunk splitting without assuming its pace.
    The original anchors remain in the saved project for subsequent speech generations.
    """
    result = deepcopy(plan)
    rows = result.get("shots") or []
    if not any("cue" in row for row in rows):
        return result
    cues = settings.get("script_cues") or plan.get("script_cues") or []
    lengths = [len(_plain(cue["text"])) for cue in cues]
    starts, total = [], 0
    for length in lengths:
        starts.append(total)
        total += length
    timed, positions, cursor = [], [], 0
    for chunk in settings.get("chunks") or []:
        for line in chunk.get("lines") or [chunk]:
            length = len(_plain(line.get("text", "")))
            if length and line.get("start") is not None and line.get("end") is not None:
                positions.append(cursor)
                timed.append(line)
                cursor += length
    previous, previous_frame = -1, -1
    frames = ceil(settings["duration"] * FPS)
    if len(rows) > frames:
        raise H3Error("The generated speech is too short to hold every planned shot.")
    if not settings.get("untimed") and not timed:
        raise H3Error("Generate a mix with timed dialogue to adapt the shot plan.")
    if timed and _plain("".join(c["text"] for c in cues)) != _plain("".join(l["text"] for l in timed)):
        raise H3Error("The spoken script differs from the shot plan. Plan shots again for this script.")
    for index, row in enumerate(rows):
        cue, offset = row.get("cue"), row.get("offset", 0)
        if type(cue) is not int or not 0 <= cue < len(cues):
            raise H3Error("Every untimed shot needs a valid script cue index.")
        if type(offset) not in (int, float) or not isfinite(offset) or not 0 <= offset < 1:
            raise H3Error("A shot's cue offset must be between zero and one (exclusive).")
        position = starts[cue] + offset * lengths[cue]
        if (index == 0 and position != 0) or position <= previous:
            raise H3Error("Shot anchors must start at cue 0, offset 0, and advance through the script.")
        previous = position
        if timed:
            at = max(0, bisect_right(positions, position) - 1)
            line = timed[at]
            fraction = (position - positions[at]) / len(_plain(line["text"]))
            row["start"] = _line_time(line, fraction)
        else:
            row["start"] = position / max(1, total) * settings["duration"]
        if index == 0:
            row["start"] = 0
        frame = min(frames - len(rows) + index, max(previous_frame + 1, round(row["start"] * FPS)))
        row["start"] = frame / FPS
        previous_frame = frame
    return result


def _plain(text):
    return re.sub(r"[\W_]+", "", spoken_text(text)).casefold()


def _line_time(line, fraction):
    """Use word alignment when available, otherwise interpolate within this spoken line."""
    words = line.get("words") or []
    if words and _plain("".join(word.get("text", "") for word in words)) == _plain(line["text"]):
        target = fraction * len(_plain(line["text"]))
        cursor = 0
        for word in words:
            length = len(_plain(word.get("text", "")))
            if length and cursor + length > target and word.get("start") is not None and word.get("end") is not None:
                return word["start"] + (target - cursor) / length * (word["end"] - word["start"])
            cursor += length
    return line["start"] + fraction * (line["end"] - line["start"])
