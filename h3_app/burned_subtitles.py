"""Burned-in video subtitles as an ASS script, drawn by FFmpeg's libass filter.

Each line shows in a dark box near the bottom, and with aligned words each word turns from
white to teal as it is spoken. A second, smaller row can sit under it (a dub's original
words under its translation). Lines that overlap in time stack; lines without word timing
show whole. Soundstage's illustrated videos and Dub & Translate's exports both use it.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

from h3_app.dubbing.tools_worker import normalized_text

LINGER_SECONDS = 0.6  # a Soundstage line stays this long after its last word, unless the next one starts
# ASS colours are &HAABBGGRR, with alpha 00 opaque and FF invisible.
SPOKEN = "&H00CFE68E"  # #8ee6cf, the spoken-word teal of the page previews
WHITE = "&H00FFFFFF"
SOFT = "&H00DEE3D7"  # #d7e3de, a dub's original words under the translation
BOX = "&H57000000"  # black at about 66% opacity, as the previews' caption box
HIDDEN = "&HFF000000"
PADDING = 8  # the box's padding around the words, in video pixels


@dataclass(frozen=True)
class Line:
    speaker: str | None  # shown in bold before the words; None or "" for no name
    text: str
    start: float
    end: float
    # (text before the first word, [(word with what follows it, start, end), ...]), or None.
    words: tuple[str, list[tuple[str, float, float]]] | None = None


@dataclass(frozen=True)
class Row:
    """How one row of captions looks: its size as a share of the video's width, and its colour."""

    share: float = 0.022
    colour: str = WHITE


SOUNDSTAGE = Row()
DUB_TRANSLATION = Row(0.024)
DUB_SOURCE = Row(0.018, SOFT)


def word_pieces(text: str, words) -> tuple[str, list[tuple[str, float, float]]] | None:
    """``text`` split as the aligned ``words`` cover it: the text before the first word, then each
    word's text with what follows it, and its (start, end). None when the words do not fit ``text``.

    Aligned words carry only letters and digits (CJK one character each), so they are matched
    back onto the shown text the way the aligner checks a script: punctuation and spaces are
    skipped, and each ride along with the word before them.
    """
    text = str(text or "")
    timed = []
    for word in words or ():
        try:
            unit, start, end = normalized_text(word["text"]), float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if unit:
            timed.append((unit, start, end))
    if not timed:
        return None
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
            return None
        pieces.append((first, start, end))
        cursor = index
    if normalized_text(text[cursor:]):
        return None
    timed = [(text[first:pieces[number + 1][0] if number + 1 < len(pieces) else len(text)], start, end)
             for number, (first, start, end) in enumerate(pieces)]
    return text[:pieces[0][0]], timed


def subtitle_font(platform: str = sys.platform) -> str:
    """A font with Latin and CJK letters on this system; libass falls back per glyph for others."""
    if platform.startswith("win"):
        return "Microsoft YaHei"
    if platform == "darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


def font_size(scale_percent: float, width: int = 1280, share: float = SOUNDSTAGE.share) -> int:
    """``share`` of the video's width, times the chosen size in percent: at least 12 px and
    never more than 5% of the width."""
    return max(12, round(min(width * share * scale_percent / 100.0, width * 0.05)))


def _plain(text: str) -> str:
    """Text as ASS shows it literally: braces would start override tags, a backslash an escape."""
    return (str(text).replace("\\", "\\​").replace("{", "(").replace("}", ")")
            .replace("\r", " ").replace("\n", " "))


def _clock(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    secs, centis = divmod(centis, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _dialogue(line: Line, shown_until: float, style: str) -> str:
    name = f"{{\\b1}}{_plain(line.speaker)}:{{\\b0}} " if line.speaker else ""
    words = _plain(line.text)
    if line.words is not None and line.words[1]:
        lead, timed = line.words
        # Karaoke: a word shows in the secondary colour (the row's own) until its syllable starts,
        # then in the primary one (teal). Animated \t tags would also do it, but libass then stops
        # moving an overlapping line above the one under it. Syllable lengths are in centiseconds,
        # taken from rounded times so their sum never drifts from the words.
        at = [max(0, round((start - line.start) * 100)) for _piece, start, _end in timed]
        last = max(at[-1] + 1, round((timed[-1][2] - line.start) * 100))
        words = f"{_plain(lead)}{{\\1c{SPOKEN}&\\k{at[0]}}}" + "".join(
            f"{{\\k{max(0, following - this)}}}{_plain(piece)}"
            for (piece, _start, _end), this, following in zip(timed, at, [*at[1:], last]))
    return f"Dialogue: 0,{_clock(line.start)},{_clock(shown_until)},{style},,0,0,0,,{name}{words}"


def _events(lines: list[Line], style: str, linger: float) -> list[str]:
    lines = sorted((line for line in lines if line.text.strip() and line.end >= line.start),
                   key=lambda line: (line.start, line.end))
    out = []
    for index, line in enumerate(lines):
        # Lingers after its words until the next line starts; a line that overlaps it shows above it.
        until = line.end + linger
        if linger:
            following = next((later.start for later in lines[index + 1:] if later.start >= line.end), None)
            if following is not None:
                until = max(line.end, min(until, following))
        out.append(_dialogue(line, until, style))
    return out


def ass_subtitles(lines: list[Line], *, width: int = 1280, height: int = 720, size: float = 100,
                  row: Row = SOUNDSTAGE, below: list[Line] | None = None, below_row: Row = DUB_SOURCE,
                  linger: float = LINGER_SECONDS, font: str | None = None) -> str:
    """An ASS script showing ``lines`` over a ``width``×``height`` video at ``size`` percent.

    ``below`` lines (a dub's original words) show in ``below_row``'s smaller, softer style at
    the bottom, and ``lines`` then sit just above them.
    """
    font = font or subtitle_font()
    margin_x, margin_v = round(width * 0.04), round(height * 0.05)
    points = font_size(size, width, row.share)
    styles = []

    def style(name, row_points, colour, raised):
        # BorderStyle 4 (libass) draws one box in the back colour behind the whole line, padded
        # by Outline; per-run boxes (style 3) overlap into darker stripes. Letters get no outline.
        styles.append(f"Style: {name},{font},{row_points},{colour},{colour},{HIDDEN},{BOX},0,0,0,0,100,100,0,0,"
                      f"4,{PADDING},0,2,{margin_x},{margin_x},{margin_v + raised},1")

    raised = 0
    events = []
    if below:
        under = font_size(size, width, below_row.share)
        style("Below", under, below_row.colour, 0)
        events += _events(below, "Below", linger)
        # One row of the smaller text and its box, then a small gap: a wrapped row underneath
        # pushes the line up further, as libass keeps boxes from overlapping.
        raised = round(under * 1.3) + 2 * PADDING + 4
    style("Caption", points, row.colour, raised)
    events += _events(lines, "Caption", linger)
    return "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
    ]) + "\n"
