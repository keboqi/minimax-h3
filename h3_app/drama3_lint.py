"""Repair and review a writer's Drama 3 script after it passes validation.

The writer prompt asks for these rules, but models slip; fixing them in code is
more reliable than more prompt wording. Repairs are limited to placements Drama 3
cannot perform well (they risk silent gaps); everything else is reported as a note.
"""

from __future__ import annotations

import re

from h3_app.drama3_serving.speaker_effects import SPEAKER_EFFECTS
from h3_app.workflows.drama3 import SPEAKER_TAG

SCOPED_TAGS = ("whisper", "soft", "emphasis", "stress", "forceful", "fast", "slow")
# Audible events may legitimately close a line: "I did it! [laughs]".
TRAILING_EVENTS = frozenset({"laughs", "laughing", "sighs", "gasps", "clears throat"})
MAX_CUE_WORDS = 5

CUE_RE = re.compile(r"\[([^\[\]]*)\]")
_TRAILING_CUE_RE = re.compile(r"\s*\[([^\[\]]*)\]\s*$")
_OPENING_SCOPED_RE = re.compile(rf"^<({'|'.join(SCOPED_TAGS)})>", re.IGNORECASE)
_LEADING_EFFECTS_RE = re.compile(rf"^((?:\s*<(?:{'|'.join(SPEAKER_EFFECTS)})>)*)\s*", re.IGNORECASE)
_MARKUP_RE = re.compile(r"<\|[^|>]*\|>|\[[^\[\]]*\]|<[^<>]*>")
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
_CLAUSE_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:…“”\"‘’'（）()《》\[\]—~\-\s]+")


def _repair_line(body: str, notes: dict[str, int]) -> str:
    effects_match = _LEADING_EFFECTS_RE.match(body)
    effects = effects_match.group(1).strip() if effects_match else ""
    spoken = body[effects_match.end():] if effects_match else body

    opening = _OPENING_SCOPED_RE.match(spoken)
    if opening:  # a scoped tag may not open a line: unwrap it
        tag = opening.group(1)
        spoken = spoken[opening.end():]
        spoken = re.sub(rf"</{tag}>", "", spoken, count=1, flags=re.IGNORECASE)
        notes["unwrapped"] += 1

    trailing = _TRAILING_CUE_RE.search(spoken)
    if trailing and trailing.group(1).strip().lower() not in TRAILING_EVENTS:
        rest = spoken[:trailing.start()]
        if _MARKUP_RE.sub("", rest).strip():  # a cue shapes only the words after it
            spoken = rest.rstrip()
            notes["trailing"] += 1

    for cue in CUE_RE.findall(spoken):
        if len(re.findall(r"[A-Za-z']+", cue)) > MAX_CUE_WORDS:
            notes["long"] += 1
    return f"{effects} {spoken.strip()}".strip() if effects else spoken.strip()


def lint_drama3_script(script: str) -> tuple[str, list[str]]:
    """The script with risky cue placements repaired, plus notes for the status line."""
    notes = {"unwrapped": 0, "trailing": 0, "long": 0}
    lines = []
    for line in str(script or "").split("\n"):
        body = line.lstrip(" \t")
        match = SPEAKER_TAG.match(body)
        if match:
            indent = line[:len(line) - len(body)]
            line = f"{indent}{match.group(0)} {_repair_line(body[match.end():].strip(), notes)}"
        lines.append(line)
    messages = []
    if notes["unwrapped"]:
        messages.append(f"unwrapped {notes['unwrapped']} line-opening scoped tag(s)")
    if notes["trailing"]:
        messages.append(f"removed {notes['trailing']} cue(s) at line ends, which shape no words")
    if notes["long"]:
        messages.append(f"{notes['long']} cue(s) run over {MAX_CUE_WORDS} words; shorter cues are more reliable")
    return "\n".join(lines), messages


def _units(text: str) -> list[str]:
    """Comparable units: characters for CJK, lowercase words otherwise.

    Latin letters are dropped from CJK text so censored spellings such as
    "jing察" still match the writer's normalized "警察".
    """
    if _CJK_RE.search(text):
        return [char for char in text if not char.isspace() and not ("a" <= char.lower() <= "z")]
    return re.findall(r"\w+", text.lower())


def _bigrams(units: list[str]) -> set[tuple[str, str]]:
    return set(zip(units, units[1:]))


def missing_source_phrases(source: str, script: str, *, min_units: int = 4) -> list[str] | None:
    """Source phrases the script does not speak, or ``None`` when the source looks like a brief.

    A phrase counts as spoken when most of its unit bigrams appear in the script,
    so punctuation changes, split quotes and normalized spellings still match.
    """
    spoken = _MARKUP_RE.sub(" ", str(script or ""))
    spoken_bigrams = _bigrams(_units(spoken)) | _bigrams(_units(_CLAUSE_SPLIT_RE.sub("", spoken)))
    phrases = [phrase.strip() for phrase in _CLAUSE_SPLIT_RE.split(_MARKUP_RE.sub(" ", str(source or "")))]
    checked, missing = 0, []
    for phrase in phrases:
        units = _units(phrase)
        if len(units) < min_units:
            continue
        checked += 1
        grams = _bigrams(units)
        if sum(gram in spoken_bigrams for gram in grams) < 0.75 * len(grams):
            missing.append(phrase)
    if not checked or len(missing) > checked / 2:
        return None  # a brief or a translation request, not lines to preserve
    return missing


def coverage_note(source: str, script: str, *, examples: int = 3) -> str:
    missing = missing_source_phrases(source, script)
    if not missing:
        return ""
    shown = "; ".join(f"“{phrase[:24]}”" for phrase in missing[:examples])
    return (f"{len(missing)} source phrase(s) are missing or reworded in the script, e.g. {shown}. "
            "Review before generating.")
