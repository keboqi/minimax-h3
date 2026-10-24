"""Pinyin pronunciation notes for Chinese characters Drama 3 might misread.

A script marks a word with tone-number pinyin in braces right after it, one syllable
per character: ``银行{yin2 hang2}``, ``还{huan2}``. The script, subtitles, takes and
checks keep the characters; only the request sent to Drama 3 replaces each noted
character with Fish Audio's phoneme tags, ``<|phoneme_start|>hang2<|phoneme_end|>``,
which the Drama 3 tokenizer reserves for this.
"""

from __future__ import annotations

import re

from h3_app.errors import H3Error

PHONEME_START = "<|phoneme_start|>"
PHONEME_END = "<|phoneme_end|>"
# Lowercase tone3 pinyin: letters, then the tone (5 is neutral); ü is written v, as pypinyin does.
SYLLABLE_RE = re.compile(r"[a-z]+[1-5]")
NOTE_RE = re.compile(r"\{([^{}\n]*)\}")
HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
MAX_NOTE_SYLLABLES = 8


def _syllables(note: str) -> list[str]:
    return note.replace("ü", "v").replace("u:", "v").split()


def note_problems(text: str) -> list[str]:
    """Why each malformed pronunciation note in ``text`` cannot be used; empty when all are fine."""
    text = str(text or "")
    problems = []
    for match in NOTE_RE.finditer(text):
        note = match.group(0)
        syllables = _syllables(match.group(1))
        if not syllables or len(syllables) > MAX_NOTE_SYLLABLES:
            problems.append(f"{note} needs 1-{MAX_NOTE_SYLLABLES} pinyin syllables")
            continue
        bad = [syllable for syllable in syllables if not SYLLABLE_RE.fullmatch(syllable)]
        if bad:
            problems.append(f"{note}: write lowercase pinyin with a tone number 1-5, e.g. hang2 ({', '.join(bad)})")
            continue
        before = text[max(0, match.start() - len(syllables)):match.start()]
        if len(before) < len(syllables) or not all(HAN_RE.fullmatch(char) for char in before):
            problems.append(f"{note} must follow exactly {len(syllables)} Chinese character(s) it reads, "
                            "with nothing in between")
    stray = NOTE_RE.sub("", text)
    if "{" in stray or "}" in stray:
        problems.append("A brace without its pair; pronunciation notes look like 银行{yin2 hang2}.")
    return problems


def strip_notes(text: str) -> str:
    """``text`` as it reads: the characters without their pronunciation notes."""
    return NOTE_RE.sub("", str(text or ""))


def to_phoneme_tags(text: str) -> str:
    """Replace every noted character with its Fish Audio phoneme tag, for the Drama 3 request."""
    text = str(text or "")
    problems = note_problems(text)
    if problems:
        raise H3Error(problems[0])
    pieces, cursor = [], 0
    for match in NOTE_RE.finditer(text):
        syllables = _syllables(match.group(1))
        start = match.start() - len(syllables)
        pieces.append(text[cursor:start])
        pieces.extend(f"{PHONEME_START}{syllable}{PHONEME_END}" for syllable in syllables)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)
