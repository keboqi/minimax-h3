"""Split a Drama 3 script into speaker-based chunks, each sent with its speaker's voice."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from h3_app.drama3.phonemes import NOTE_RE
from h3_app.workflows.drama3 import SPEAKER_TAG

from .speaker_effects import extract_speaker_effects
from .tree import SpeechTurn

# About one or two sentences of speech; Fish Audio's own chunker uses 300 bytes.
CHUNK_BYTES = 400
CONTROL = re.compile(r"<\|[^|>]*\|>|\[[^\[\]]*\]|</?[A-Za-z][\w-]*>|\{[^{}\n]*\}")
WIDE_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
# A sentence ends at terminal punctuation outside a cue; cues never contain line breaks.
# Full-width punctuation needs no following space (Chinese/Japanese) and keeps a closing quote.
CJK_CLOSE = "”’」』）)"
SENTENCE_END = re.compile(rf"(?:(?<=[.!?;])\s+|(?<=[。！？；])(?![{CJK_CLOSE}])\s*|(?<=[。！？；][{CJK_CLOSE}])\s*)(?![^\[]*\])")
# Fallback cut points for a single sentence longer than the chunk limit.
CLAUSE_END = re.compile(r"(?:(?<=[,:])\s+|(?<=[，、：])\s*)(?![^\[]*\])")
CJK_JOIN = re.compile(rf"[。！？；，、：{CJK_CLOSE}]$")


@dataclass(frozen=True)
class Chunk:
    index: int
    speaker: str  # the speaker's label: its Cast name, or "Speaker N"
    text: str  # speech and cues for one speaker, without the speaker's name
    speaker_effects: tuple[str, ...] = ()
    # ``text`` cut where one script line ends and the next begins, for per-line subtitles.
    lines: tuple[str, ...] = ()


def parse_turns(script: str) -> list[tuple[str, str]]:
    """Ordered (speaker, text) turns of a tagged script; text before the first tag is Speaker 0."""
    parts = SPEAKER_TAG.split(str(script or ""))
    turns = [("Speaker 0", parts[0])] + [(f"Speaker {int(speaker)}", body)
                                         for speaker, body in zip(parts[1::2], parts[2::2])]
    return [(speaker, " ".join(body.split())) for speaker, body in turns if body.strip()]


def parse_turns_with_effects(script: str) -> list[SpeechTurn]:
    """Ordered SpeechTurn items with speaker_effects extracted."""
    turns: list[SpeechTurn] = []
    for speaker, raw_body in parse_turns(script):
        cleaned, effects = extract_speaker_effects(raw_body)
        if cleaned:
            turns.append(SpeechTurn(speaker=speaker, text=cleaned, speaker_effects=effects, raw_text=raw_body))
    return turns


# Audible events sound once where they stand; every other bracketed cue sets the delivery after it.
AUDIBLE_EVENTS = frozenset({"laughs", "laughing", "small laugh", "sighs", "gasps", "clears throat", "chuckles",
                            "sniffs", "scoffs", "groans", "shaky breath", "exhales", "inhales", "coughs"})
_CUE = re.compile(r"\[([^\[\]]*)\]")
_PHRASE_BREAK = ",，、;；:：—.!?。！？"


def delivery_cues(text: str) -> list[str]:
    """The bracketed delivery cues in ``text``, in order, without audible events such as [laughs]."""
    return [match.group(0) for match in _CUE.finditer(str(text or ""))
            if match.group(1).strip().lower() not in AUDIBLE_EVENTS]


def _outside_markup(text: str) -> Iterable[tuple[int, str]]:
    """(index, character) of ``text`` outside [cues], <tags> and {pronunciation notes}."""
    square = angle = brace = 0
    for index, ch in enumerate(text):
        if ch == "[":
            square += 1
        elif ch == "]":
            square = max(0, square - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif not square and not angle and not brace:
            yield index, ch


def _has_words(text: str) -> bool:
    return any(ch.isalnum() for _index, ch in _outside_markup(text))


def opening_cut(text: str) -> int | None:
    """Where a cue can follow the opening words of ``text`` with spoken words still after it.

    After the first phrase (its comma or sentence end, with any closing quote); a single
    phrase is cut after its first word, or its first two CJK characters.
    """
    seen = 0
    for index, ch in _outside_markup(text):
        if ch in _PHRASE_BREAK and seen:
            end = index + 1
            while end < len(text) and text[end] in CJK_CLOSE + "\"'":
                end += 1
            if _has_words(text[end:]):
                return end
        elif ch.isalnum():
            seen += 1
    seen = 0
    for index, ch in _outside_markup(text):
        if WIDE_CHAR.match(ch):
            seen += 1
            if seen == 2 and _has_words(text[index + 1:]):
                return index + 1
        elif ch.isspace() and seen and _has_words(text[index:]):
            return index
        elif ch.isalnum():
            seen = max(seen, 1)
    return None


def insert_cue(text: str, cue: str) -> str | None:
    """``text`` with ``cue`` placed after its opening words, or None when there is no such place."""
    cut = opening_cut(text)
    if cut is None:
        return None
    # A word keeps its pronunciation note: neither 银行{yin2 hang2} nor 尉迟恭{yu4 chi2 gong1}
    # is ever split by a cue, which moves past the note instead.
    for note in NOTE_RE.finditer(text):
        if note.start() - len(note.group(1).split()) < cut < note.end():
            cut = note.end()
    if not _has_words(text[cut:]):
        return None
    left, right = text[:cut].rstrip(), text[cut:].lstrip()
    # CJK punctuation needs no space before the cue, as in "当星辰排列成十字，[ominous] 诸神".
    separator = "" if CJK_JOIN.search(left) else " "
    return f"{left}{separator}{cue} {right}"


def _sentences(text: str, limit: int) -> list[str]:
    pieces = []
    for sentence in filter(None, SENTENCE_END.split(text)):
        too_long = len(sentence.encode("utf-8")) > limit
        pieces.extend(filter(None, CLAUSE_END.split(sentence)) if too_long else [sentence])
    return pieces


def _pack(sentences: list[str], limit: int) -> list[str]:
    chunks, current = [], ""
    for sentence in sentences:
        # CJK text has no spaces between sentences; don't add any when rejoining.
        separator = "" if CJK_JOIN.search(current) else " "
        candidate = f"{current}{separator}{sentence}".strip()
        if current and len(candidate.encode("utf-8")) > limit:
            chunks.append(current)
            candidate = sentence
        current = candidate
    return [*chunks, current] if current else chunks


def _visible(text: str) -> int:
    return sum(not ch.isspace() for ch in text)


def _cut(text: str, cuts: list[int]) -> tuple[str, ...]:
    """``text`` cut before each of its non-space characters numbered in ``cuts``."""
    pieces, last, seen, pending = [], 0, 0, sorted(cuts)
    for position, ch in enumerate(text):
        if ch.isspace():
            continue
        while pending and pending[0] == seen:
            pieces.append(text[last:position])
            last = position
            pending.pop(0)
        seen += 1
    pieces.append(text[last:])
    return tuple(piece.strip() for piece in pieces if piece.strip())


def plan_turns_chunks(
    turns: Iterable[SpeechTurn],
    *,
    start_index: int = 0,
    limit: int = CHUNK_BYTES,
) -> list[Chunk]:
    """Merge consecutive turns of one speaker with identical speaker_effects, then cut near ``limit``.

    Splitting and rejoining only ever drops or adds whitespace, so counting non-space
    characters finds where each merged turn's text lands in the chunks.
    """
    merged: list[tuple[str, str, tuple[str, ...], list[int]]] = []
    for turn in turns:
        text = " ".join((turn.text or "").split())
        if not text:
            continue
        if merged and merged[-1][0] == turn.speaker and merged[-1][2] == turn.speaker_effects:
            speaker, joined, effects, ends = merged[-1]
            merged[-1] = (speaker, f"{joined} {text}", effects, [*ends, ends[-1] + _visible(text)])
        else:
            merged.append((turn.speaker, text, turn.speaker_effects, [_visible(text)]))
    pieces = []
    for speaker, text, effects, ends in merged:
        position, cue = 0, None
        for piece in _pack(_sentences(text, limit), limit):
            size = _visible(piece)
            lines = _cut(piece, [end - position for end in ends[:-1] if position < end < position + size])
            position += size
            # Each request starts fresh, so a turn split across requests repeats its last delivery cue
            # in its first sentence, unless that sentence sets its own.
            opening = _sentences(piece, limit)[0] if piece else ""
            if cue and not delivery_cues(opening) and lines and piece.startswith(lines[0]):
                carried = insert_cue(lines[0], cue)
                if carried is not None:
                    piece, lines = carried + piece[len(lines[0]):], (carried, *lines[1:])
            cue = (delivery_cues(piece) or [cue])[-1]
            pieces.append((speaker, piece, effects, lines))
    return [
        Chunk(start_index + offset, speaker, text, effects, lines)
        for offset, (speaker, text, effects, lines) in enumerate(pieces)
    ]


def plan_chunks(script: str, limit: int = CHUNK_BYTES) -> list[Chunk]:
    """Merge consecutive lines of one speaker, then cut at sentence ends near ``limit``."""
    return plan_turns_chunks(parse_turns_with_effects(script), start_index=0, limit=limit)


def spoken_text(text: str) -> str:
    """The words a listener hears: cues and tags removed."""
    return " ".join(CONTROL.sub(" ", str(text or "")).split())


def expected_seconds(text: str) -> float:
    """Rough spoken length: ~2.5 words/s, ~4.5 CJK characters/s, ~0.5 s per cue."""
    cues = len(re.findall(r"\[[^\[\]]*\]", str(text or "")))
    spoken = spoken_text(text)
    wide = len(WIDE_CHAR.findall(spoken))
    words = len(WIDE_CHAR.sub(" ", spoken).split())
    return words / 2.5 + wide / 4.5 + cues * 0.5
