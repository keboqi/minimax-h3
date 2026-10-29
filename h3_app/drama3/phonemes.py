"""Pronunciation notes for words Drama 3 might misread: pinyin for Chinese, Arpabet for English.

A script marks a word with its reading in braces right after it. Chinese takes
tone-number pinyin, one syllable per character: ``银行{yin2 hang2}``, ``还{huan2}``.
An English word takes CMU Arpabet with a stress digit on each vowel:
``Siobhan{SH IH0 V AO1 N}``. The script, subtitles, takes and checks keep the words;
only the request sent to Drama 3 replaces each noted character, or each noted English
word, with Fish Audio's phoneme tags, ``<|phoneme_start|>hang2<|phoneme_end|>``, which
the Drama 3 tokenizer reserves for this.
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
ARPABET_VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
ARPABET_CONSONANTS = frozenset("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
# An English note follows the word it reads, with nothing in between.
ENGLISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*$")
MAX_NOTE_PHONEMES = 24


_UPPERCASE_PHONEME_RE = re.compile(r"[A-Z]+[0-9]?")


def is_arpabet(note: str) -> bool:
    """Whether a note's content reads as Arpabet (all-uppercase phonemes) rather than pinyin."""
    return bool(note.split()) and all(_UPPERCASE_PHONEME_RE.fullmatch(phoneme) for phoneme in note.split())


def _arpabet_problem(note: str) -> str:
    phonemes = note.split()
    if len(phonemes) > MAX_NOTE_PHONEMES:
        return f"{{{note}}} needs 1-{MAX_NOTE_PHONEMES} Arpabet phonemes"
    bad = [phoneme for phoneme in phonemes if not (
        phoneme in ARPABET_CONSONANTS or (phoneme[:-1] in ARPABET_VOWELS and phoneme[-1:] in "012"))]
    if bad:
        return (f"{{{note}}}: write CMU Arpabet with a stress digit 0-2 on each vowel, "
                f"e.g. SH IH0 V AO1 N ({', '.join(bad)})")
    if not any(phoneme[:-1] in ARPABET_VOWELS for phoneme in phonemes):
        return f"{{{note}}} has no vowel"
    return ""


def note_start(text: str, match: re.Match) -> int:
    """Where the word a valid note reads begins: its English word, or its characters."""
    if is_arpabet(match.group(1)):
        word = ENGLISH_WORD_RE.search(text[:match.start()])
        return word.start() if word else match.start()
    return match.start() - len(_syllables(match.group(1)))


def _syllables(note: str) -> list[str]:
    return note.replace("ü", "v").replace("u:", "v").split()


def note_problems(text: str) -> list[str]:
    """Why each malformed pronunciation note in ``text`` cannot be used; empty when all are fine."""
    text = str(text or "")
    problems = []
    for match in NOTE_RE.finditer(text):
        note = match.group(0)
        if is_arpabet(match.group(1)):
            problem = _arpabet_problem(match.group(1))
            if not problem and not ENGLISH_WORD_RE.search(text[:match.start()]):
                problem = f"{note} must follow the English word it reads, with nothing in between"
            if problem:
                problems.append(problem)
            continue
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
        problems.append("A brace without its pair; pronunciation notes look like 银行{yin2 hang2} or "
                        "Siobhan{SH IH0 V AO1 N}.")
    return problems


def strip_notes(text: str) -> str:
    """``text`` as it reads: the characters without their pronunciation notes."""
    return NOTE_RE.sub("", str(text or ""))


def to_phoneme_tags(text: str) -> str:
    """Replace every noted character or English word with Fish Audio phoneme tags, for the Drama 3 request.

    A Chinese note becomes one tag per syllable, an English note one tag for the word.
    """
    text = str(text or "")
    problems = note_problems(text)
    if problems:
        raise H3Error(problems[0])
    pieces, cursor = [], 0
    for match in NOTE_RE.finditer(text):
        pieces.append(text[cursor:note_start(text, match)])
        if is_arpabet(match.group(1)):
            pieces.append(f"{PHONEME_START}{' '.join(match.group(1).split())}{PHONEME_END}")
        else:
            pieces.extend(f"{PHONEME_START}{syllable}{PHONEME_END}" for syllable in _syllables(match.group(1)))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)
