"""Burned-in subtitles for the illustrated video: the page's captions as an ASS file.

Each line shows as "Speaker: words" in a dark box near the bottom, one line at a time unless
speech overlaps, and with aligned words each word turns from white to teal as it is spoken,
as the page's video preview shows it. Lines without word timing show whole, in white.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

WIDTH, HEIGHT = 1280, 720
LINGER_SECONDS = 0.6  # a line stays this long after its last word, unless the next one starts
# ASS colours are &HAABBGGRR, with alpha 00 opaque and FF invisible.
SPOKEN = "&H00CFE68E"  # #8ee6cf, the preview's spoken-word teal
WHITE = "&H00FFFFFF"
BOX = "&H57000000"  # black at about 66% opacity, as the preview's caption box
HIDDEN = "&HFF000000"


@dataclass(frozen=True)
class Line:
    speaker: str
    text: str
    start: float
    end: float
    # (text before the first word, [(word with what follows it, start, end), ...]), or None.
    words: tuple[str, list[tuple[str, float, float]]] | None = None


def subtitle_font(platform: str = sys.platform) -> str:
    """A font with Latin and CJK letters on this system; libass falls back per glyph for others."""
    if platform.startswith("win"):
        return "Microsoft YaHei"
    if platform == "darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


def font_size(scale_percent: float, width: int = WIDTH) -> int:
    """The preview's caption size: 2.2% of the frame's width (12-40 px), times the chosen size,
    never more than 5% of the width."""
    base = min(max(width * 0.022, 12.0), 40.0)
    return round(min(base * scale_percent / 100.0, width * 0.05))


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


def _dialogue(line: Line, shown_until: float) -> str:
    name = f"{{\\b1}}{_plain(line.speaker)}:{{\\b0}} "
    words = _plain(line.text)
    if line.words is not None and line.words[1]:
        lead, timed = line.words
        # Karaoke: a word shows in the secondary colour (white) until its syllable starts, then in
        # the primary one (teal). Animated \t tags would also do it, but libass then stops moving
        # an overlapping line above the one under it. Syllable lengths are in centiseconds, taken
        # from rounded times so their sum never drifts from the words.
        at = [max(0, round((start - line.start) * 100)) for _piece, start, _end in timed]
        last = max(at[-1] + 1, round((timed[-1][2] - line.start) * 100))
        words = f"{_plain(lead)}{{\\1c{SPOKEN}&\\k{at[0]}}}" + "".join(
            f"{{\\k{max(0, following - this)}}}{_plain(piece)}"
            for (piece, _start, _end), this, following in zip(timed, at, [*at[1:], last]))
    return f"Dialogue: 0,{_clock(line.start)},{_clock(shown_until)},Caption,,0,0,0,,{name}{words}"


def ass_subtitles(lines: list[Line], *, size: float = 100, font: str | None = None,
                  width: int = WIDTH, height: int = HEIGHT) -> str:
    """An ASS script showing ``lines`` over a ``width``×``height`` video at ``size`` percent."""
    lines = sorted((line for line in lines if line.text.strip() and line.end >= line.start),
                   key=lambda line: (line.start, line.end))
    margin_x, margin_v = round(width * 0.04), round(height * 0.05)
    font, points = font or subtitle_font(), font_size(size, width)
    margins = f"2,{margin_x},{margin_x},{margin_v},1"
    out = [
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
        # BorderStyle 4 (libass) draws one box in the back colour behind the whole line, padded by
        # Outline; per-run boxes (style 3) overlap into darker stripes. The letters get no outline.
        f"Style: Caption,{font},{points},{WHITE},{WHITE},{HIDDEN},{BOX},0,0,0,0,100,100,0,0,4,8,0,{margins}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for index, line in enumerate(lines):
        # Lingers after its words until the next line starts; a line that overlaps it shows above it.
        following = next((later.start for later in lines[index + 1:] if later.start >= line.end), None)
        until = line.end + LINGER_SECONDS
        if following is not None:
            until = max(line.end, min(until, following))
        out.append(_dialogue(line, until))
    return "\n".join(out) + "\n"
