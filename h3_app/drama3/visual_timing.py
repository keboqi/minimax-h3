"""Persistent dialogue anchors, wording adaptation and binding to each mix's clock."""

from bisect import bisect_right
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import accumulate
from math import ceil, isfinite, nextafter
import re
from uuid import uuid4

from h3_app.errors import H3Error
from .serving.planner import spoken_text
from .serving.tree import parse_drama3_tree, iter_speech_nodes
from .visuals import FPS


class _TextPositions:
    """The same character clock for source cues, split speech chunks and word timing."""

    def __init__(self, cues):
        self.lengths = [len(_plain(cue["text"])) for cue in cues]
        if not self.lengths or not all(self.lengths):
            raise H3Error("Dialogue cues must contain spoken words.")
        boundaries = list(accumulate(self.lengths, initial=0))
        self.starts, self.total = boundaries[:-1], boundaries[-1]

    def locate(self, position):
        index = max(0, bisect_right(self.starts, position) - 1)
        return index, (position - self.starts[index]) / self.lengths[index]


def script_cues(text, *, default_speaker="Speaker 0", resolve=None):
    return [dict(speaker=turn.speaker, text=turn.text)
            for node in iter_speech_nodes(parse_drama3_tree(text, default_speaker=default_speaker, resolve=resolve))
            for turn in node.turns
            if _plain(turn.text)]


def preview_settings(text, title="", voices=None, *, cues=None):
    cues = script_cues(text) if cues is None else cues
    if not cues:
        raise H3Error("Write a script before planning shots.")
    return dict(title=title, voices=voices or {}, script=text, script_cues=cues,
                duration=max(400, len(cues) * 10), untimed=True)


def mix_cues(settings):
    """Dialogue in script order; sound timestamps are context, never dialogue indexes."""
    lines = _timed_lines(settings)
    if settings.get("script_cues"):
        cues = deepcopy(settings["script_cues"])
        timed = [line for line in lines if line.get("start") is not None and line.get("end") is not None]
        if timed and cue_text(cues) == cue_text(timed):
            clock = _TextPositions(timed)
            cursor = 0
            for cue in cues:
                length = len(_plain(cue["text"]))
                for key, position in (("start", cursor), ("end", cursor + length - 1e-9)):
                    at, fraction = clock.locate(position)
                    cue[key] = _line_time(timed[at], fraction)
                cursor += length
        return cues
    return [{key: value for key, value in line.items() if key in ("speaker", "text", "start", "end")}
            for line in lines]


def shot_speech_cues(settings):
    """Spoken lines on the mix clock, or the shot plan's preview clock before recording."""
    cues = mix_cues(settings)
    if settings.get("untimed") and cues:
        clock = _TextPositions(cues)
        duration = settings["duration"]
        # Match the frame-rounded cue boundaries used by bind_plan and exposed by plan_rows.
        boundaries = [round(round(position / clock.total * duration * FPS) / FPS, 3)
                      for position in (*clock.starts, clock.total)]
        cues = [dict(cue, start=start, end=end)
                for cue, start, end in zip(cues, boundaries, boundaries[1:])]
    return sorted((cue for cue in cues if cue.get("start") is not None and cue.get("end") is not None),
                  key=lambda cue: cue["start"])


def _timed_lines(settings):
    return [dict(speaker=chunk.get("name") or chunk.get("speaker") or "", **line)
            for chunk in settings.get("chunks") or []
            for line in chunk.get("lines") or [{key: chunk[key] for key in ("text", "start", "end", "words")
                                                if key in chunk}]
            if _plain(line.get("text", ""))]


def cue_text(cues):
    return _plain("".join(cue["text"] for cue in cues))


def same_dialogue(left, right):
    """Ignore chunk boundaries, while retaining who actually spoke each stretch of words."""
    def identity(cues):
        result = []
        for cue in cues:
            speaker = str(cue.get("speaker", "")).casefold()
            text = _plain(cue["text"])
            if result and result[-1][0] == speaker:
                result[-1] = (speaker, result[-1][1] + text)
            else:
                result.append((speaker, text))
        return result

    return identity(left) == identity(right)


def anchor_plan(plan, settings):
    """Migrate timestamp replies and older saved plans to persistent cue anchors."""
    result = deepcopy(plan)
    rows = result.get("shots")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise H3Error("The visual planner must return a nonempty list of shots.")
    anchored = ["cue" in row for row in rows]
    if any(anchored) and not all(anchored):
        raise H3Error("Use cue anchors for every shot, rather than mixing anchors and timestamps.")
    cues = result.get("script_cues") or mix_cues(settings)
    if not cues:
        raise H3Error("A shot plan needs dialogue cues.")
    # Keep only source identity in the project; actual timings belong to the mix.
    result["script_cues"] = [dict(speaker=cue.get("speaker", ""), text=cue["text"]) for cue in cues]
    if not any(anchored):
        lines = _timed_lines(settings)
        if not lines or any(line.get("start") is None or line.get("end") is None for line in lines):
            raise H3Error("Plans made before speech need cue and offset anchors instead of start times.")
        cue_clock, line_clock = _TextPositions(cues), _TextPositions(lines)
        if cue_text(lines) != cue_text(cues):
            raise H3Error("The spoken script differs from the shot plan. Plan shots again for this script.")
        previous = -1
        for index, row in enumerate(rows):
            time = row.get("start")
            if type(time) not in (int, float) or not isfinite(time) or not 0 <= time < settings["duration"]:
                raise H3Error("Every timestamp must be within the mix.")
            if (index == 0 and time != 0) or time <= previous:
                raise H3Error("Shot boundaries must start at zero and advance through the mix.")
            previous = time
            at = max((i for i, line in enumerate(lines) if line["start"] <= time), default=0)
            line = lines[at]
            # Invert the same word-aware interpolation used during binding.
            low, high = 0.0, 1.0
            for _ in range(40):
                mid = (low + high) / 2
                if _line_time(line, mid) < time:
                    low = mid
                else:
                    high = mid
            position = line_clock.starts[at] + (low + high) / 2 * line_clock.lengths[at]
            cue, offset = cue_clock.locate(position)
            row.update(cue=cue, offset=min(1 - 1e-9, offset))
            # A picture may start in a sound-only tail or pause after the last word.
            # Keep that relationship to the cue instead of snapping it to the word's end.
            if time >= line["end"]:
                until = lines[at + 1]["start"] if at + 1 < len(lines) else settings["duration"]
                if until > line["end"]:
                    row.update(phase="after", offset=min(1 - 1e-9, (time - line["end"]) / (until - line["end"])))
            if index == 0:
                row.update(cue=0, offset=0)
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            row["id"] = uuid4().hex
        seen.add(row["id"])
        row.pop("start", None)
    return result


def adapt_plan(plan, cues):
    """Keep the storyboard through wording corrections, without guessing at structural edits.

    A same-speaker line retains its identity. Character alignment moves an anchor with its
    surrounding words rather than shifting a beat merely because an earlier word got longer.
    """
    result = deepcopy(plan)
    old = result.get("script_cues") or []
    cues = [dict(speaker=cue.get("speaker", ""), text=cue["text"]) for cue in cues]
    error = "The script changed beyond a wording correction. Review or plan shots again for this script."
    if not old or len(old) != len(cues):
        raise H3Error(error)
    matches = []
    for before, after in zip(old, cues):
        a, b = _plain(before["text"]), _plain(after["text"])
        if str(before.get("speaker", "")).casefold() != str(after.get("speaker", "")).casefold() or not a or not b:
            raise H3Error(error)
        match = SequenceMatcher(None, a, b, autojunk=False)
        short_typo = len(a) == len(b) <= 4 and sum(x != y for x, y in zip(a, b)) <= 1
        if a != b and match.ratio() < 0.6 and not short_typo:
            raise H3Error(error)
        matches.append(match)
    previous = None
    for row in result.get("shots") or []:
        cue = row.get("cue")
        if type(cue) is not int or not 0 <= cue < len(cues):
            raise H3Error("Every shot needs a valid script cue index.")
        match = matches[cue]
        offset = row.get("offset", 0)
        if type(offset) not in (int, float) or not isfinite(offset) or not 0 <= offset < 1:
            raise H3Error("A shot's cue offset must be between zero and one (exclusive).")
        if match.a != match.b and offset and row.get("phase", "speech") == "speech":
            position = offset * len(match.a)
            for _tag, a, end_a, b, end_b in match.get_opcodes():
                if a <= position < end_a:
                    moved = b + (position - a) / (end_a - a) * (end_b - b)
                    row["offset"] = min(1 - 1e-9, moved / len(match.b))
                    break
        phase = row.get("phase", "speech")
        order = (cue, phase == "after", row.get("offset", 0))
        # Removing words can collapse several original anchors onto one character boundary.
        # Preserve their shot order; binding will still give each at least one video frame.
        if previous is not None and order[:2] == previous[:2] and order[2] <= previous[2]:
            row["offset"] = nextafter(previous[2], 1.0)
            if row["offset"] >= 1:
                raise H3Error(error)
            order = (*order[:2], row["offset"])
        previous = order
        row.pop("start", None)
    result["script_cues"] = cues
    return result


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
    if settings.get("script_cues") and cues != plan.get("script_cues"):
        result = adapt_plan(result, cues)
        rows = result.get("shots") or []
    cue_clock = _TextPositions(cues)
    timed = [line for line in _timed_lines(settings) if line.get("start") is not None and line.get("end") is not None]
    line_clock = _TextPositions(timed) if timed else None
    previous, previous_frame = None, -1
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
        phase = row.get("phase", "speech")
        if phase not in ("speech", "after"):
            raise H3Error("A shot's phase must be speech or after.")
        position = cue_clock.starts[cue] + offset * cue_clock.lengths[cue]
        order = (cue, phase == "after", offset)
        if (index == 0 and order != (0, False, 0)) or (previous is not None and order <= previous):
            raise H3Error("Shot anchors must start at cue 0, offset 0, and advance through the script.")
        previous = order
        if timed:
            if phase == "after":
                position = cue_clock.starts[cue] + cue_clock.lengths[cue] - 1e-9
            at, fraction = line_clock.locate(position)
            line = timed[at]
            if phase == "after":
                until = timed[at + 1]["start"] if at + 1 < len(timed) else settings["duration"]
                row["start"] = line["end"] + offset * max(0, until - line["end"])
            else:
                row["start"] = _line_time(line, fraction)
        else:
            row["start"] = position / cue_clock.total * settings["duration"]
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
