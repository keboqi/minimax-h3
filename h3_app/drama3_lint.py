"""Repair and review a writer's Drama 3 script after it passes validation.

The writer prompt asks for these rules, but models slip; fixing them in code is
more reliable than more prompt wording, and keeps the prompt free to ask for rich
direction. Repairs are limited to placements Drama 3 cannot perform well (they risk
silent gaps or skipped words); everything else is reported as a note.
"""

from __future__ import annotations

import re

from h3_app.drama3_serving.planner import WIDE_CHAR, insert_cue
from h3_app.drama3_serving.speaker_effects import SPEAKER_EFFECTS
from h3_app.workflows.drama3 import SPEAKER_TAG

SCOPED_TAGS = ("whisper", "soft", "emphasis", "stress", "forceful", "fast", "slow")
# Audible events may legitimately close a line: "I did it! [laughs]".
TRAILING_EVENTS = frozenset({"laughs", "laughing", "sighs", "gasps", "clears throat"})
MAX_CUE_WORDS = 5
MAX_CUES_PER_LINE = 2
# A script with fewer directed lines than this share gets a note (and a direction pass).
MIN_DIRECTED_SHARE = 0.6

CUE_RE = re.compile(r"\[([^\[\]]*)\]")
_TRAILING_CUE_RE = re.compile(r"\s*\[([^\[\]]*)\]\s*$")
_OPENING_SCOPED_RE = re.compile(rf"^<({'|'.join(SCOPED_TAGS)})>", re.IGNORECASE)
_OPENING_CUE_RE = re.compile(r"^\[([^\[\]]*)\]\s*")
_SCOPED_RE = re.compile(rf"</?(?:{'|'.join(SCOPED_TAGS)})>", re.IGNORECASE)
_LEADING_EFFECTS_RE = re.compile(rf"^((?:\s*<(?:{'|'.join(SPEAKER_EFFECTS)})>)*)\s*", re.IGNORECASE)
_MARKUP_RE = re.compile(r"<\|[^|>]*\|>|\[[^\[\]]*\]|<[^<>]*>")
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
_CLAUSE_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:…“”\"‘’'（）()《》\[\]—~\-\s]+")
# Phrases end at punctuation, not at spaces: an English phrase is several words.
_PHRASE_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:…“”\"‘’（）()《》\[\]—~]+|\s-+\s|\n+")
# Cues that ask for silence, which Drama 3 may hold far too long or never end.
_PAUSE_CUE_RE = re.compile(r"^(?:(?:a|long|short|brief|small|dramatic)\s+)?(?:pause|pauses|beat|silence|"
                           r"hesitates|hesitation|breathing|breath|swallows)$", re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"\.{3,}|…+|。{3,}")
_SENTENCE_END = "。！？.!?"
_PUNCTUATION = ",，、;；:：.!?。！？"


def _repair_ellipses(spoken: str, notes: dict[str, int]) -> str:
    """An ellipsis becomes a comma, or a period where the line ends; next to punctuation it goes."""
    def replace(match):
        notes["ellipses"] += 1
        before, after = spoken[:match.start()].rstrip(), spoken[match.end():]
        if (before and before[-1] in _PUNCTUATION) or (after.lstrip()[:1] and after.lstrip()[0] in _PUNCTUATION):
            return ""
        wide = bool(WIDE_CHAR.match(before[-1:]))
        if not _MARKUP_RE.sub("", after).strip():
            return "。" if wide else "."
        return "，" if wide else ","

    return _ELLIPSIS_RE.sub(replace, spoken)


def _drop_cues(spoken: str, keep) -> str:
    """``spoken`` without the cues ``keep(match)`` rejects, with the spaces they leave tidied."""
    kept = CUE_RE.sub(lambda match: match.group(0) if keep(match) else "", spoken)
    return re.sub(r"[ \t]{2,}", " ", kept).strip()


def _cap_cues(spoken: str, notes: dict[str, int]) -> str:
    """At most one cue per sentence and ``MAX_CUES_PER_LINE`` per line; the first ones stay."""
    kept, sentences = 0, set()

    def keep(match):
        nonlocal kept
        before = _MARKUP_RE.sub("", spoken[:match.start()]).rstrip()
        # Sentences ended before the cue; a cue right after a sentence's end opens the next one.
        sentence = sum(ch in _SENTENCE_END for ch in before)
        if sentence in sentences or kept >= MAX_CUES_PER_LINE:
            notes["capped"] += 1
            return False
        sentences.add(sentence)
        kept += 1
        return True

    return _drop_cues(spoken, keep)


def _repair_line(body: str, notes: dict[str, int]) -> str:
    effects_match = _LEADING_EFFECTS_RE.match(body)
    effects = effects_match.group(1).strip() if effects_match else ""
    spoken = body[effects_match.end():] if effects_match else body

    spoken = _repair_ellipses(spoken, notes)

    def not_a_pause(match):
        if _PAUSE_CUE_RE.match(match.group(1).strip()):
            notes["pauses"] += 1
            return False
        return True

    spoken = _drop_cues(spoken, not_a_pause)

    opening = _OPENING_SCOPED_RE.match(spoken)
    if opening:  # a scoped tag may not open a line: unwrap it
        tag = opening.group(1)
        spoken = spoken[opening.end():]
        spoken = re.sub(rf"</{tag}>", "", spoken, count=1, flags=re.IGNORECASE)
        notes["unwrapped"] += 1

    cue = _OPENING_CUE_RE.match(spoken)
    if cue and _MARKUP_RE.sub("", spoken[cue.end():]).strip():  # a cue may not open a line: move it in
        rest = spoken[cue.end():]
        while (extra := _OPENING_CUE_RE.match(rest)) is not None:  # stacked opening cues: keep the first
            rest = rest[extra.end():]
            notes["capped"] += 1
        moved = insert_cue(rest, cue.group(0))
        spoken = moved if moved is not None else rest
        notes["moved" if moved is not None else "dropped"] += 1

    trailing = _TRAILING_CUE_RE.search(spoken)
    if trailing and trailing.group(1).strip().lower() not in TRAILING_EVENTS:
        rest = spoken[:trailing.start()]
        if _MARKUP_RE.sub("", rest).strip():  # a cue shapes only the words after it
            spoken = rest.rstrip()
            notes["trailing"] += 1

    spoken = _cap_cues(spoken, notes)

    for cue_text in CUE_RE.findall(spoken):
        if len(re.findall(r"[A-Za-z']+", cue_text)) > MAX_CUE_WORDS:
            notes["long"] += 1
    return f"{effects} {spoken.strip()}".strip() if effects else spoken.strip()


def speaker_lines(script: str) -> list[str]:
    """The bodies of a tagged script's speaker lines, in order."""
    bodies = []
    for line in str(script or "").split("\n"):
        body = line.lstrip(" \t")
        match = SPEAKER_TAG.match(body)
        if match:
            bodies.append(body[match.end():].strip())
    return bodies


def is_directed(body: str) -> bool:
    """Whether a speaker line carries delivery direction: a bracketed cue or a scoped tag."""
    return bool(CUE_RE.search(body) or _SCOPED_RE.search(body))


def direction_share(script: str) -> tuple[int, int]:
    """(directed, total) speaker lines of a tagged script."""
    bodies = speaker_lines(script)
    return sum(map(is_directed, bodies)), len(bodies)


def under_directed(script: str) -> bool:
    """Whether a script of several lines leaves too many of them without direction."""
    directed, total = direction_share(script)
    return total >= 3 and directed < MIN_DIRECTED_SHARE * total


def lint_drama3_script(script: str) -> tuple[str, list[str]]:
    """The script with risky cue placements repaired, plus notes for the status line."""
    notes = dict.fromkeys(("unwrapped", "moved", "dropped", "ellipses", "pauses", "capped", "trailing", "long"), 0)
    lines = []
    for line in str(script or "").split("\n"):
        body = line.lstrip(" \t")
        match = SPEAKER_TAG.match(body)
        if match:
            indent = line[:len(line) - len(body)]
            line = f"{indent}{match.group(0)} {_repair_line(body[match.end():].strip(), notes)}"
        lines.append(line)
    repaired = "\n".join(lines)
    messages = []
    if notes["unwrapped"]:
        messages.append(f"unwrapped {notes['unwrapped']} line-opening scoped tag(s)")
    if notes["moved"]:
        messages.append(f"moved {notes['moved']} line-opening cue(s) after the opening words")
    if notes["dropped"]:
        messages.append(f"removed {notes['dropped']} line-opening cue(s) with no room after the opening words")
    if notes["ellipses"]:
        messages.append(f"replaced {notes['ellipses']} ellipsis(es) with commas or periods")
    if notes["pauses"]:
        messages.append(f"removed {notes['pauses']} pause or silence cue(s); punctuation paces the line")
    if notes["capped"]:
        messages.append(f"removed {notes['capped']} cue(s) beyond one per sentence and {MAX_CUES_PER_LINE} per line")
    if notes["trailing"]:
        messages.append(f"removed {notes['trailing']} cue(s) at line ends, which shape no words")
    if notes["long"]:
        messages.append(f"{notes['long']} cue(s) run over {MAX_CUE_WORDS} words; shorter cues are more reliable")
    if under_directed(repaired):
        directed, total = direction_share(repaired)
        messages.append(f"only {directed} of {total} lines carry delivery direction")
    cues = [CUE_RE.findall(body) for body in speaker_lines(repaired)]
    repeats = sum(1 for previous, current in zip(cues, cues[1:])
                  if previous and current and previous[0].strip().lower() == current[0].strip().lower())
    if repeats:
        messages.append(f"{repeats} line(s) open with the same cue as the line before")
    return repaired, messages


_LOOSE_SPACE_RE = re.compile(r"\s+(?=[,.!?;:，。！？；：、”’」』）)])|(?<=[“‘「『（(])\s+")


def _words_only(body: str) -> str:
    """A speaker line's words and punctuation without cues or scoped tags: what direction may not change.

    Spaces between words count, so "now here" never becomes "nowhere"; a space an inserted
    cue leaves beside CJK text or before punctuation does not.
    """
    text = " ".join(_SCOPED_RE.sub("", CUE_RE.sub(" ", body)).split())
    text = _LOOSE_SPACE_RE.sub("", text)
    return re.sub(rf"(?<={WIDE_CHAR.pattern}) | (?={WIDE_CHAR.pattern})", "", text)


def merge_direction(script: str, directed: str) -> tuple[str, int, int]:
    """(script, lines taking new direction, lines refused) with the director's lines merged in.

    The director may only insert cues and scoped tags. A speaker line is taken from
    ``directed`` when it keeps its speaker, its vocal effects and every word; otherwise
    the line stays as it was. Unless both scripts have the same lines with identical
    staging, sound and pause lines, nothing is merged.
    """
    original = [line for line in str(script or "").split("\n") if line.strip()]
    proposed = [line.strip() for line in str(directed or "").split("\n") if line.strip()]
    if len(original) != len(proposed):
        return script, 0, len(speaker_lines(script))
    merged, taken, refused = [], 0, 0
    for line, new in zip(original, proposed):
        body = line.lstrip(" \t")
        indent = line[:len(line) - len(body)]
        old_tag, new_tag = SPEAKER_TAG.match(body), SPEAKER_TAG.match(new)
        if old_tag is None or new_tag is None:
            if old_tag is not None or new_tag is not None or " ".join(body.split()) != " ".join(new.split()):
                return script, 0, len(speaker_lines(script))
            merged.append(line)
            continue
        old_body, new_body = body[old_tag.end():].strip(), new[new_tag.end():].strip()
        old_effects, new_effects = _LEADING_EFFECTS_RE.match(old_body), _LEADING_EFFECTS_RE.match(new_body)
        same = (old_tag.group(0) == new_tag.group(0)
                and "".join(old_effects.group(1).split()) == "".join(new_effects.group(1).split())
                and _words_only(old_body) == _words_only(new_body))
        if same and new_body != old_body:
            merged.append(f"{indent}{old_tag.group(0)} {new_body}")
            taken += 1
        else:
            merged.append(line)
            refused += not same
    return "\n".join(merged), taken, refused


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


def missing_source_phrases(source: str, script: str, *, min_units: int = 4,
                           source_mode: str = "") -> list[str] | None:
    """Source phrases the script does not speak, or ``None`` when the source is a brief.

    A phrase counts as spoken when most of its unit bigrams appear in the script,
    so punctuation changes, split quotes and normalized spellings still match.
    ``source_mode`` is the writer's own reading of the source: "preserve" checks it
    however much is missing, "brief" skips the check, and otherwise a source whose
    phrases are mostly missing is taken for a brief.
    """
    if source_mode == "brief":
        return None
    spoken = _MARKUP_RE.sub(" ", str(script or ""))
    spoken_bigrams = _bigrams(_units(spoken)) | _bigrams(_units(_CLAUSE_SPLIT_RE.sub("", spoken)))
    phrases = [phrase.strip() for phrase in _PHRASE_SPLIT_RE.split(_MARKUP_RE.sub(" ", str(source or "")))]
    checked, missing = 0, []
    for phrase in phrases:
        units = _units(phrase)
        if len(units) < min_units:
            continue
        checked += 1
        grams = _bigrams(units)
        if sum(gram in spoken_bigrams for gram in grams) < 0.75 * len(grams):
            missing.append(phrase)
    if not checked or (source_mode != "preserve" and len(missing) > checked / 2):
        return None  # a brief or a translation request, not lines to preserve
    return missing


def coverage_note(source: str, script: str, *, examples: int = 3, source_mode: str = "") -> str:
    missing = missing_source_phrases(source, script, source_mode=source_mode)
    if not missing:
        return ""
    shown = "; ".join(f"“{phrase[:24]}”" for phrase in missing[:examples])
    return (f"{len(missing)} source phrase(s) are missing or reworded in the script, e.g. {shown}. "
            "Review before generating.")
