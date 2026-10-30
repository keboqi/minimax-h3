"""Repair and review a writer's Drama 3 script after it passes validation.

The writer prompt asks for these rules, but models slip; fixing them in code is
more reliable than more prompt wording, and keeps the prompt free to ask for rich
direction. Repairs are limited to placements Drama 3 cannot perform well (they risk
silent gaps or skipped words); everything else is reported as a note.
"""

from __future__ import annotations

import re

from h3_app.drama3.phonemes import note_problems
from h3_app.drama3.serving.planner import AUDIBLE_EVENTS, WIDE_CHAR, insert_cue
from h3_app.drama3.serving.rooms import scene_room_name
from h3_app.drama3.serving.speaker_effects import ROOM_EFFECT_ROOMS, SPEAKER_TAGS
from h3_app.drama3.serving.tree import staging_problems, tag_attrs
from h3_app.workflows.drama3 import SPEAKER_TAG

SCOPED_TAGS = ("whisper", "soft", "emphasis", "stress", "forceful", "fast", "slow")
# Audible events may legitimately close a line: "I did it! [laughs]".
TRAILING_EVENTS = AUDIBLE_EVENTS
MAX_CUE_WORDS = 5
MAX_CUES_PER_LINE = 2
# A script with fewer directed lines than this share gets a note (and a direction pass).
MIN_DIRECTED_SHARE = 0.6

CUE_RE = re.compile(r"\[([^\[\]]*)\]")
_TRAILING_CUE_RE = re.compile(r"\s*\[([^\[\]]*)\]\s*$")
_OPENING_SCOPED_RE = re.compile(rf"^<({'|'.join(SCOPED_TAGS)})>", re.IGNORECASE)
_OPENING_CUE_RE = re.compile(r"^\[([^\[\]]*)\]\s*")
_SCOPED_RE = re.compile(rf"</?(?:{'|'.join(SCOPED_TAGS)})>", re.IGNORECASE)
_LEADING_EFFECTS_RE = re.compile(rf"^((?:\s*<(?:{'|'.join(SPEAKER_TAGS)})>)*)\s*", re.IGNORECASE)
_SINGLE_EFFECT_RE = re.compile(rf"<({'|'.join(SPEAKER_TAGS)})>", re.IGNORECASE)
_NARRATOR_LINE_RE = re.compile(r"^(?P<prefix>(?:narrator|narration|旁白|story\s*teller)[ \t]*[:：])[ \t]*(?P<body>.*)$",
                               re.IGNORECASE)
_NAMED_SPEAKER_LINE_RE = re.compile(r"^(?P<prefix>[^<>:\n\r]{1,40}[:：])[ \t]*(?P<body>.*)$")
_PARALLEL_TAG_RE = re.compile(r"<parallel\b[^<>]*>", re.IGNORECASE)
_PARALLEL_CLOSE_RE = re.compile(r"</parallel\s*>", re.IGNORECASE)
_MARKUP_RE = re.compile(r"<\|[^|>]*\|>|\[[^\[\]]*\]|<[^<>]*>|\{[^{}\n]*\}")
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


def _strip_redundant_room_effects(effects: str, active_room: str, notes: dict[str, int]) -> str:
    if not effects or not active_room:
        return effects
    kept: list[str] = []
    for match in _SINGLE_EFFECT_RE.finditer(effects):
        name = match.group(1).lower()
        if ROOM_EFFECT_ROOMS.get(name) == active_room or name == active_room:
            notes["redundant_room"] += 1
        else:
            kept.append(f"<{name}>")
    return "".join(kept)


def _repair_line(body: str, notes: dict[str, int], *, active_room: str = "") -> str:
    effects_match = _LEADING_EFFECTS_RE.match(body)
    effects = effects_match.group(1).strip() if effects_match else ""
    spoken = body[effects_match.end():] if effects_match else body
    effects = _strip_redundant_room_effects(effects, active_room, notes)

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


# Shallow direction: most cues are one bare word, one speaker keeps repeating a cue, or a
# long stretch of lines goes undirected, though enough lines carry something.
MIN_SHALLOW_CUES = 4
MAX_BARE_SHARE = 0.6
MAX_REPEATS = 3
MAX_FLAT_RUN = 5


def shallow_direction(script: str) -> str:
    """Why a script's direction is too shallow to perform well, or "" when it is not."""
    lines = [(match.group(1), body[match.end():]) for body in (line.lstrip(" \t") for line in str(script or "").split("\n"))
             if (match := SPEAKER_TAG.match(body))]
    cues = [(speaker, cue.strip().lower()) for speaker, body in lines for cue in CUE_RE.findall(body)
            if cue.strip().lower() not in AUDIBLE_EVENTS]
    if len(cues) >= MIN_SHALLOW_CUES:
        bare = sum(len(re.findall(r"[\w']+", cue)) == 1 for _speaker, cue in cues)
        if bare > MAX_BARE_SHARE * len(cues):
            return f"{bare} of {len(cues)} cues are a single bare word"
        counts: dict[tuple[str, str], int] = {}
        for key in cues:
            counts[key] = counts.get(key, 0) + 1
        (speaker, cue), most = max(counts.items(), key=lambda item: item[1])
        if most >= MAX_REPEATS and most * 3 >= sum(n for (who, _), n in counts.items() if who == speaker):
            return f"one speaker repeats [{cue}] {most} times"
    run = longest = 0
    for _speaker, body in lines:
        run = 0 if is_directed(body) else run + 1
        longest = max(longest, run)
    if longest >= MAX_FLAT_RUN:
        return f"{longest} lines in a row carry no direction"
    return ""


def needs_direction(script: str) -> bool:
    """Whether a script should get a direction pass: too few directed lines, or shallow direction."""
    return under_directed(script) or bool(shallow_direction(script))


def lint_drama3_script(script: str) -> tuple[str, list[str]]:
    """The script with risky cue placements repaired, plus notes for the status line."""
    notes = dict.fromkeys(
        ("unwrapped", "moved", "dropped", "ellipses", "pauses", "capped", "trailing", "long",
         "redundant_room", "narrator_effects"),
        0,
    )
    lines = []
    room_stack: list[str] = []
    for line in str(script or "").split("\n"):
        body = line.lstrip(" \t")
        indent = line[:len(line) - len(body)]
        for open_tag in _PARALLEL_TAG_RE.findall(body):
            room_stack.append(scene_room_name(tag_attrs(open_tag).get("room", "")))
        active_room = next((room for room in reversed(room_stack) if room), "")
        match = SPEAKER_TAG.match(body)
        if match:
            line = f"{indent}{match.group(0)} {_repair_line(body[match.end():].strip(), notes, active_room=active_room)}"
        elif narrator_match := _NARRATOR_LINE_RE.match(body):
            narrator_body = narrator_match.group("body")
            effects_match = _LEADING_EFFECTS_RE.match(narrator_body)
            if effects_match and effects_match.group(1).strip():
                notes["narrator_effects"] += len(_SINGLE_EFFECT_RE.findall(effects_match.group(1)))
                narrator_body = narrator_body[effects_match.end():].strip()
            line = f"{indent}{narrator_match.group('prefix')} {narrator_body}".rstrip()
        elif named_match := _NAMED_SPEAKER_LINE_RE.match(body):
            named_body = named_match.group("body")
            effects_match = _LEADING_EFFECTS_RE.match(named_body)
            if effects_match and effects_match.group(1).strip() and active_room:
                effects = _strip_redundant_room_effects(effects_match.group(1).strip(), active_room, notes)
                rest = named_body[effects_match.end():].strip()
                named_body = f"{effects} {rest}".strip() if effects else rest
                line = f"{indent}{named_match.group('prefix')} {named_body}".rstrip()
        for _close_tag in _PARALLEL_CLOSE_RE.findall(body):
            if room_stack:
                room_stack.pop()
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
    if notes["redundant_room"]:
        messages.append(f"removed {notes['redundant_room']} redundant room effect tag(s) already set by the scene")
    if notes["narrator_effects"]:
        messages.append(f"removed {notes['narrator_effects']} vocal effect or distance tag(s) from narration")
    if notes["long"]:
        messages.append(f"{notes['long']} cue(s) run over {MAX_CUE_WORDS} words; shorter cues are more reliable")
    if under_directed(repaired):
        directed, total = direction_share(repaired)
        messages.append(f"only {directed} of {total} lines carry delivery direction")
    elif shallow := shallow_direction(repaired):
        messages.append(f"direction is shallow: {shallow}")
    cues = [CUE_RE.findall(body) for body in speaker_lines(repaired)]
    repeats = sum(1 for previous, current in zip(cues, cues[1:])
                  if previous and current and previous[0].strip().lower() == current[0].strip().lower())
    if repeats:
        messages.append(f"{repeats} line(s) open with the same cue as the line before")
    # A scene with an unreadable room or stage is still performed, without them.
    problems = [problem for tag in _PARALLEL_TAG_RE.findall(repaired) for problem in staging_problems(tag)]
    messages.extend(f"{problem}; the scene plays without it" for problem in dict.fromkeys(problems))
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
                and _words_only(old_body) == _words_only(new_body)
                # A cue or space between a word and its pronunciation note would detach the note.
                and not note_problems(new_body))
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


def _skip_bigrams(units: list[str]) -> set[tuple[str, str]]:
    return set(zip(units, units[2:]))


def _fixed_slip_bigrams(units: list[str], spoken_bigrams: set, spoken_skips: set) -> int:
    """Missing bigrams one corrected unit explains: the writer fixes obvious typos.

    A replaced unit drops the two bigrams around it while its neighbours stay two
    apart in the script; a removed duplicate leaves them adjacent. A slip at either
    end of the phrase drops one bigram and leaves no such trace.
    """
    pairs = list(zip(units, units[1:]))
    hits = [pair in spoken_bigrams for pair in pairs]
    best = int(not hits[0]) or int(not hits[-1])
    for k in range(1, len(units) - 1):
        around = (units[k - 1], units[k + 1])
        if not hits[k - 1] and not hits[k] and (around in spoken_skips or around in spoken_bigrams):
            return 2
    return best


def missing_source_phrases(source: str, script: str, *, min_units: int = 4,
                           source_mode: str = "") -> list[str] | None:
    """Source phrases the script does not speak, or ``None`` when the source is a brief.

    A phrase counts as spoken when most of its unit bigrams appear in the script,
    so punctuation changes, split quotes and normalized spellings still match, and
    a phrase whose only change is one corrected word or character also counts.
    ``source_mode`` is the writer's own reading of the source: "preserve" checks it
    however much is missing, "brief" skips the check, and otherwise a source whose
    phrases are mostly missing is taken for a brief.
    """
    if source_mode == "brief":
        return None
    spoken = _MARKUP_RE.sub(" ", str(script or ""))
    spoken_units = (_units(spoken), _units(_CLAUSE_SPLIT_RE.sub("", spoken)))
    spoken_bigrams = set().union(*map(_bigrams, spoken_units))
    spoken_skips = set().union(*map(_skip_bigrams, spoken_units))
    phrases = [phrase.strip() for phrase in _PHRASE_SPLIT_RE.split(_MARKUP_RE.sub(" ", str(source or "")))]
    checked, missing = 0, []
    for phrase in phrases:
        units = _units(phrase)
        if len(units) < min_units:
            continue
        checked += 1
        grams = _bigrams(units)
        missed = sum(gram not in spoken_bigrams for gram in grams)
        # Spoken when mostly matched, or when one corrected slip explains every miss.
        if missed > 0.25 * len(grams) and missed > _fixed_slip_bigrams(units, spoken_bigrams, spoken_skips):
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
