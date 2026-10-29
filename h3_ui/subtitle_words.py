"""Subtitle text with each aligned word in its own timed span, for word-by-word highlighting.

Aligned words carry only letters and digits (CJK one character each), so they are matched
back onto the shown text the way the aligner checks a script: punctuation and spaces are
skipped, and each ride along with the word before them. Text the words do not cover
exactly is shown whole, without word timing.
"""

from __future__ import annotations

from html import escape

from h3_app.dubbing.tools_worker import normalized_text

WORD_CLASS = "h3-word"


def word_spans_html(text: str, words) -> str:
    """``text`` escaped, as one ``<span data-start data-end>`` per aligned word where they fit it."""
    text = str(text or "")
    timed = []
    for word in words or ():
        try:
            unit, start, end = normalized_text(word["text"]), float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            return escape(text)
        if unit:
            timed.append((unit, start, end))
    if not timed:
        return escape(text)
    # (start, end) character ranges of ``text`` for each word: its own letters, then what follows.
    pieces, index, cursor = [], 0, 0
    for unit, start, end in timed:
        first, matched = None, ""
        while index < len(text) and len(matched) < len(unit):
            folded = normalized_text(text[index])
            if folded and first is None:
                first = index
            matched += folded
            index += 1
        if matched != unit or first is None:
            return escape(text)
        pieces.append((first, start, end))
        cursor = index
    if normalized_text(text[cursor:]):
        return escape(text)
    out = escape(text[:pieces[0][0]])
    for number, (first, start, end) in enumerate(pieces):
        stop = pieces[number + 1][0] if number + 1 < len(pieces) else len(text)
        out += (f'<span class="{WORD_CLASS}" data-start="{start:.3f}" data-end="{end:.3f}">'
                f"{escape(text[first:stop])}</span>")
    return out
