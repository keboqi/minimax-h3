"""Small Gemma checks for the Drama 3 writers; code keeps the final say.

Each check asks Gemma about a few short items, verifies the answer with the same
rules that rejected the original, and returns nothing it cannot verify, so a
missing key or a spent free quota leaves the fixed rules exactly as they were.
"""

from __future__ import annotations

import json

from h3_app.drama3.phonemes import HAN_RE, NOTE_RE, is_arpabet, note_problems, strip_notes
from h3_app.llm import gemma

MAX_READINGS = 20
# Enough text before a note for its longest reading (8 characters or one English word).
_READING_CONTEXT = 40

_READING_REQUEST = """Fix pronunciation notes for a Chinese/English speech synthesizer.
A note follows the word it reads, in braces:
- Chinese: lowercase tone-number pinyin, one syllable per character, tone 1-5 (5 is neutral), ü written v:
  银行{{yin2 hang2}}, 还{{huan2}}, 女{{nv3}}.
- English: CMU Arpabet in capitals, a stress digit 0-2 on every vowel: Siobhan{{SH IH0 V AO1 N}}.
Each item gives the text just before a broken note, the note, and what is wrong.
Return the corrected reading of the same word, without braces. Use null if unsure.
Reply with only JSON: {{"readings": ["<reading or null>", ...]}} in item order.
Items:
{items}"""


def _reading_problems(before: str, reading: str) -> list[str]:
    return note_problems(f"{strip_notes(before)[-_READING_CONTEXT:]}{{{reading}}}")


def _same_span(old: str, new: str, word: str) -> bool:
    """Whether a pinyin fix still reads the same characters: one syllable each, never fewer or more."""
    if is_arpabet(new):
        return True
    if word and all(HAN_RE.fullmatch(char) for char in word):
        return len(new.split()) == len(word)
    return len(new.split()) == len(old.split())


def fix_readings(items: list[tuple[str, str]], *, words: bool = False) -> list[str | None]:
    """A checked replacement for each ``(text before the note, broken reading)``; None where there is none.

    With ``words`` the first element is exactly the word the note reads, not the text before it.
    """
    items = items[:MAX_READINGS]
    if not items:
        return []
    listing = "\n".join(
        json.dumps({"before": strip_notes(before)[-_READING_CONTEXT:], "note": reading,
                    "problem": "; ".join(_reading_problems(before, reading))}, ensure_ascii=False)
        for before, reading in items)
    answer = gemma.ask_json(_READING_REQUEST.format(items=listing), max_output_tokens=48 * len(items) + 32)
    readings = answer.get("readings") if answer else None
    if not isinstance(readings, list) or len(readings) != len(items):
        return [None] * len(items)
    fixed = []
    for (before, reading), new in zip(items, readings):
        new = " ".join(str(new).split()) if isinstance(new, str) else ""
        valid = new and not _reading_problems(before, new) and _same_span(reading, new, before if words else "")
        fixed.append(new if valid else None)
    return fixed


def repair_reading_notes(text: str) -> str:
    """``text`` with each malformed pronunciation note fixed where Gemma's reading checks out.

    Notes that stay malformed are left for the validator to report as before.
    """
    text = str(text or "")
    if "{" not in text or not note_problems(text):
        return text
    broken = [match for match in NOTE_RE.finditer(text)
              if _reading_problems(text[:match.start()], match.group(1))]
    readings = fix_readings([(text[:match.start()], match.group(1)) for match in broken])
    for match, reading in sorted(zip(broken, readings), key=lambda pair: -pair[0].start()):
        if reading is not None:
            text = f"{text[:match.start()]}{{{reading}}}{text[match.end():]}"
    return text
