"""Subtitle text with each aligned word in its own timed span, for word-by-word highlighting.

Words are matched onto the text by ``h3_app.burned_subtitles.word_pieces``, which the burned-in
video subtitles share; text the words do not cover exactly is shown whole, without word timing.
"""

from __future__ import annotations

from html import escape

from h3_app.burned_subtitles import word_pieces

WORD_CLASS = "h3-word"


def word_spans_html(text: str, words) -> str:
    """``text`` escaped, as one ``<span data-start data-end>`` per aligned word where they fit it."""
    text = str(text or "")
    pieces = word_pieces(text, words)
    if pieces is None:
        return escape(text)
    lead, timed = pieces
    return escape(lead) + "".join(f'<span class="{WORD_CLASS}" data-start="{start:.3f}" data-end="{end:.3f}">'
                                  f"{escape(piece)}</span>" for piece, start, end in timed)
