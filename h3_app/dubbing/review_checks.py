"""Review lines helpers: Gemini Decision translation checks and lines worth combining.

Translation checks ask Gemini Decision two yes/no questions per line in batched requests
(does the translation keep the original's meaning, and does it sound like natural speech).
Lines that fail are marked under the line as checked (its start, speaker and both texts),
so editing the original or the translation clears the mark.

Combine suggestions are fixed rules over times, speakers and punctuation; they only
pick lines for the viewer to review, and nothing is combined until the viewer says so.
"""

from __future__ import annotations

import re
from typing import Any

from h3_app.errors import H3Error
from h3_app.llm import decisions

from . import pipeline, session as sessions

# Session key for lines a translation check flagged, filed like failed takes.
TRANSLATION_CHECKS = "translation_checks"
# A line is flagged below these probabilities of "yes".
MEANING_THRESHOLD = 0.5
NATURAL_THRESHOLD = 0.35
# Two questions per line: a request carries this many lines.
LINES_PER_REQUEST = 100


def _check_mark(line: dict) -> tuple[float, str, str, str]:
    """What a translation-check mark is filed under: the line as checked, not its position, so
    adding or deleting other lines keeps it and editing this line's start or either text drops it."""
    return (round(float(line["start"]), 3), str(line.get("speaker") or ""), str(line["source"]),
            str(line["translation"]))


def _stored_checks(session: sessions.Session) -> dict[tuple[float, str, str, str], str]:
    marks = {}
    stored = session.data.get(TRANSLATION_CHECKS)
    for entry in stored if isinstance(stored, list) else ():
        try:
            marks[_check_mark(entry)] = str(entry["reason"])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return marks


def _save_checks(latest: sessions.Session, marks: dict[tuple[float, str, str, str], str]) -> None:
    latest.data[TRANSLATION_CHECKS] = [
        {"start": start, "speaker": speaker, "source": source, "translation": translation, "reason": reason}
        for (start, speaker, source, translation), reason in marks.items()]


def translation_checks(session: sessions.Session) -> dict[int, str]:
    """The voiced lines a translation check flagged, with why; editing a line's text clears it."""
    marks = _stored_checks(session)
    return {index: marks[_check_mark(line)] for index, line in enumerate(session.segments)
            if line["translation"] and line.get("dub", True) and _check_mark(line) in marks}


def _questions(index: int, line: dict, target: str, same: bool) -> dict[str, dict[str, Any]]:
    original, new = line["source"].strip(), line["translation"].strip()
    if same:
        meaning = (f"Line {index + 1} was rewritten in the same language for a dub. Does the new text keep "
                   f"the original's meaning, names, numbers and negation?\noriginal: {original}\nnew: {new}")
        natural = f"Does the new text of line {index + 1} sound like natural spoken dialogue?\nnew: {new}"
    else:
        meaning = (f"Line {index + 1} was translated into {target} for a dub. Does the translation keep the "
                   f"original's meaning, names, numbers and negation, without adding or dropping anything "
                   f"important?\noriginal: {original}\ntranslation: {new}")
        natural = (f"Does the translation of line {index + 1} sound like natural spoken {target} dialogue, "
                   f"not a literal word-for-word rendering or an explanation?\ntranslation: {new}")
    return {f"meaning_{index}": decisions.yes_no(meaning), f"natural_{index}": decisions.yes_no(natural)}


def _reason(meaning: float | None, natural: float | None) -> str:
    problems = []
    if meaning is not None and meaning < MEANING_THRESHOLD:
        problems.append(f"the meaning may differ from the original ({round(meaning * 100)}% likely to match)")
    if natural is not None and natural < NATURAL_THRESHOLD:
        problems.append("it may not sound natural when spoken")
    return ("Translation check: " + "; ".join(problems) + ".") if problems else ""


def check_translations(session_id: str, indices: list[int] | None, *, runtime,
                       api_key: str = "") -> tuple[sessions.Session, str]:
    """Check the chosen lines (all voiced, translated lines without a choice) with Gemini Decision.

    Returns the session as saved and a status line. Marks replace earlier ones for the
    checked lines only.
    """
    key = decisions.key_or_none(api_key)
    if key is None:
        raise H3Error("Translation checks use Gemini Decision: set GEMINI_API_KEY on the server "
                      "or enter a temporary Gemini API key in Analyze.")
    session = sessions.load(runtime, session_id)
    lines = session.segments
    same = pipeline.same_language(session)
    target = session.data.get("target_language") or "the target language"
    chosen = range(len(lines)) if not indices else sorted({i for i in indices if 0 <= i < len(lines)})
    checked = [i for i in chosen if lines[i]["translation"].strip() and lines[i]["source"].strip()
               and lines[i].get("dub", True) and not (same and pipeline.unchanged(lines[i]))]
    if not checked:
        return session, "No translated lines to check."
    context = decisions.text(
        f"Lines of a video dub from {session.data.get('source_language') or 'the original language'}"
        f"{'' if same else f' into {target}'}. Each question shows one line.")
    verdicts: dict[int, str] = {}
    answered = set()
    for first in range(0, len(checked), LINES_PER_REQUEST):
        batch = checked[first:first + LINES_PER_REQUEST]
        questions = {}
        for index in batch:
            questions.update(_questions(index, lines[index], target, same))
        answers = decisions.decide(questions, context, key=key, timeout=90)
        for index in batch:
            meaning = decisions.yes_probability(answers.get(f"meaning_{index}"))
            natural = decisions.yes_probability(answers.get(f"natural_{index}"))
            if meaning is None and natural is None:
                continue
            answered.add(index)
            verdicts[index] = _reason(meaning, natural)
    marked = {_check_mark(lines[i]): reason for i, reason in verdicts.items() if reason}
    cleared = {_check_mark(lines[i]) for i in answered}

    def change(latest: sessions.Session) -> None:
        current = {_check_mark(line) for line in latest.segments}
        marks = {mark: reason for mark, reason in _stored_checks(latest).items()
                 if mark in current and mark not in cleared}
        marks.update({mark: reason for mark, reason in marked.items() if mark in current})
        _save_checks(latest, marks)

    sessions.modify(session, change)
    session = sessions.load(runtime, session_id)
    flagged = sorted(i + 1 for i, reason in verdicts.items() if reason)
    skipped = len(checked) - len(answered)
    status = (f"Checked {len(answered)} line{'s' if len(answered) != 1 else ''}: "
              + (f"{len(flagged)} to review (line {', '.join(map(str, flagged[:20]))}"
                 f"{', …' if len(flagged) > 20 else ''})." if flagged else "none look wrong."))
    if skipped:
        status += f" {skipped} could not be checked."
    return session, status


# Lines worth combining ---------------------------------------------------------------
# Lines this close together, of one speaker, may be one sentence cut in pieces.
COMBINE_GAP = 0.6
# A combined line stays a comfortable subtitle and dub take.
COMBINE_MAX_SECONDS = 8.0
COMBINE_MAX_CHARS = 120
# A line this short is a fragment even when it ends a sentence ("Yes." "Okay.").
SHORT_SECONDS = 1.2
_SENTENCE_END = re.compile(r"[.!?。！？…]['\"”’」』)）]*\s*$")


# By context, Gemini Decision confirms a neighbouring pair at or above this probability.
CONTEXT_THRESHOLD = 0.6
_CONTEXT_QUESTION = (
    "Should line {first} and line {second} be one subtitle line? Answer yes only if they are one sentence or "
    "one thought said by the same person, cut in two by the transcription, judging by the words and the lines "
    "around them. Speaker labels may be missing or wrong, so do not rely on them.\n{context}")


def _text(line: dict) -> str:
    return (line.get("translation") or line.get("source") or "").strip()


def _near(left: dict, right: dict) -> bool:
    """Close enough in time to be one line, both dubbed or both not, and short enough together."""
    return (right["start"] - left["end"] <= COMBINE_GAP
            and bool(left.get("dub", True)) == bool(right.get("dub", True))
            and right["end"] - left["start"] <= COMBINE_MAX_SECONDS
            and len(_text(left)) + len(_text(right)) + 1 <= COMBINE_MAX_CHARS)


def _reads_as_one(left: dict, right: dict) -> bool:
    """The fixed rule: one speaker, and one sentence split across lines or a fragment next to its neighbour."""
    if left["speaker"] != right["speaker"]:
        return False
    short = (left["end"] - left["start"] < SHORT_SECONDS) or (right["end"] - right["start"] < SHORT_SECONDS)
    return short or not _SENTENCE_END.search(left.get("source") or _text(left))


def _chain(lines: list[dict], joined: set[int]) -> list[list[int]]:
    """Groups from joined neighbour pairs (``i`` joins line i and i + 1), cut where a group would grow too long."""
    groups: list[list[int]] = []
    current: list[int] = []
    for index in range(len(lines)):
        if current and index - 1 in joined:
            start, chars = lines[current[0]]["start"], sum(len(_text(lines[i])) + 1 for i in current)
            if (lines[index]["end"] - start <= COMBINE_MAX_SECONDS
                    and chars + len(_text(lines[index])) <= COMBINE_MAX_CHARS):
                current.append(index)
                continue
        if len(current) > 1:
            groups.append(current)
        current = [index]
    if len(current) > 1:
        groups.append(current)
    return groups


def combine_suggestions(lines: list[dict]) -> list[list[int]]:
    """Runs of neighbouring lines that read as one by fixed rules: one speaker, a short gap, a short result.

    Each run has two or more line numbers (0-based), in order; runs never overlap.
    Lines that are not dubbed stay apart from dubbed ones.
    """
    return _chain(lines, {i for i in range(len(lines) - 1)
                          if _near(lines[i], lines[i + 1]) and _reads_as_one(lines[i], lines[i + 1])})


def _context_lines(lines: list[dict], index: int) -> str:
    shown = []
    for number in range(max(0, index - 1), min(len(lines), index + 3)):
        line = lines[number]
        speaker = line.get("speaker") or "unknown"
        shown.append(f"line {number + 1} ({speaker}, {line['start']:.1f}-{line['end']:.1f}s): "
                     f"{(line.get('source') or '').strip()}")
    return "\n".join(shown)


def context_suggestions(lines: list[dict], *, key: str) -> tuple[list[list[int]], int]:
    """Groups confirmed by context: every neighbouring pair close in time, whatever its speaker labels,
    is asked of Gemini Decision in one batch, with the lines around it. Returns the groups and how
    many pairs were asked. Raises ``H3Error`` when the check fails.
    """
    candidates = [i for i in range(len(lines) - 1) if _near(lines[i], lines[i + 1])]
    if not candidates:
        return [], 0
    questions = {f"join_{i}": decisions.yes_no(_CONTEXT_QUESTION.format(
        first=i + 1, second=i + 2, context=_context_lines(lines, i))) for i in candidates}
    answers = decisions.decide(questions, decisions.text(
        "Transcribed lines of a video, in order, with times. Speaker labels come from automatic separation "
        "and may be missing or wrong."), key=key, timeout=90)
    joined = set()
    for i in candidates:
        probability = decisions.yes_probability(answers.get(f"join_{i}"))
        if probability is not None and probability >= CONTEXT_THRESHOLD:
            joined.add(i)
    return _chain(lines, joined), len(candidates)


def detect_combine(session_id: str, *, runtime, context: bool = True,
                   api_key: str = "") -> tuple[list[list[int]], bool, str]:
    """Lines worth combining: by context with Gemini Decision when asked and a key is set, else by the
    fixed rules. Returns the groups, whether context decided them, and a status line."""
    lines = sessions.load(runtime, session_id).segments
    key = decisions.key_or_none(api_key) if context else None
    note = ""
    if key is not None:
        try:
            groups, asked = context_suggestions(lines, key=key)
            found = sum(len(group) for group in groups)
            return groups, True, (f"Checked {asked} neighbouring pair{'s' if asked != 1 else ''} by context: "
                                  + (f"{found} lines in {len(groups)} group{'s' if len(groups) != 1 else ''}."
                                     if groups else "none read as one line."))
        except H3Error as exc:
            note = f" The context check failed ({exc}), so fixed rules were used."
    elif context:
        note = " No Gemini key, so fixed rules were used (they trust the speaker labels)."
    groups = combine_suggestions(lines)
    found = sum(len(group) for group in groups)
    return groups, False, ((f"Picked {found} lines in {len(groups)} group{'s' if len(groups) != 1 else ''}."
                            if groups else "No lines look like they should be combined.") + note)
