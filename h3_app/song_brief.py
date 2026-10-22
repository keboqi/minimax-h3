"""Plain-language song choices compiled into a producer's plan.

A user who knows no music theory picks who sings, in what language, what the
track is for, and how long it may run. This module turns those choices into
what a producer and topliner would decide before writing: a song form sized to
the length, a lyric-line budget per section, and the engine brief. It also
checks a draft the way a producer would before booking studio time: does the
material fit the length, is there a hook, are lines singable.

Pure functions only; estimates are rules of thumb, never promises of exact
playback length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape

AUTO = "Auto"
INSTRUMENTAL = "Instrumental"
VOCAL_CHOICES = (AUTO, "Female lead", "Male lead", "Duet", "Group / choir", INSTRUMENTAL)
LANGUAGE_CHOICES = (
    AUTO, "English", "Mandarin Chinese", "Cantonese", "Japanese", "Korean",
    "Spanish", "French", "German", "Portuguese",
)
FULL_SONG, HOOK, BACKGROUND = "Full song", "Short hook / jingle", "Background music"
USE_CHOICES = (AUTO, FULL_SONG, HOOK, BACKGROUND)
KEEP_LYRICS, POLISH_LYRICS = "Keep my lyrics", "Polish my lyrics"
LYRICS_POLICIES = (KEEP_LYRICS, POLISH_LYRICS)

DEFAULT_BPM = 100
LINE_BARS = 2  # A pop lyric line usually occupies two bars of 4/4.
INSTRUMENTAL_REQUEST = re.compile(
    r"\b(instrumental|no vocals?|without vocals?|no singing|no lyrics)\b|纯音乐|无人声|インスト",
    re.IGNORECASE,
)
_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")
_BPM_RANGE = re.compile(r"(\d{2,3})\s*(?:-|–|—|to)\s*(\d{2,3})\s*bpm", re.IGNORECASE)
_BPM = re.compile(r"(\d{2,3})\s*bpm|bpm\s*[:=]?\s*(\d{2,3})", re.IGNORECASE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_VOWEL_GROUPS = re.compile(r"[aeiouyàáâäãåèéêëìíîïòóôöõùúûüæœ]+", re.IGNORECASE)


@dataclass(frozen=True)
class PlannedSection:
    tag: str
    bars: int
    lines: int
    role: str


@dataclass(frozen=True)
class SongPlan:
    use: str
    instrumental: bool
    bpm: int
    bpm_assumed: bool
    sections: tuple[PlannedSection, ...]
    genre: Genre | None = None

    @property
    def seconds(self) -> float:
        return sum(section.bars for section in self.sections) * bar_seconds(self.bpm)


# (tag, bars, lyric lines, producer's role for the section)
_HOOK = (
    ("Intro", 2, 0, "a one-bar pickup into the hook; no waiting"),
    ("Chorus", 8, 4, "the hook, stated immediately and complete"),
    ("Outro", 2, 0, "a clean button ending on the tonic"),
)
_SHORT = (
    ("Intro", 4, 0, "establish groove and key centre"),
    ("Verse", 8, 4, "set the scene, lower energy, conversational melody"),
    ("Chorus", 8, 4, "the hook: highest energy, title line, wider register"),
    ("Outro", 4, 0, "resolve, ring out or end on the downbeat"),
)
_MEDIUM = (
    ("Intro", 4, 0, "establish groove and a signature motif"),
    ("Verse", 8, 4, "story and detail, sparse arrangement"),
    ("Pre-Chorus", 4, 2, "lift: rising melody, building drums, tension"),
    ("Chorus", 8, 4, "the hook: full band, title line"),
    ("Verse", 8, 4, "advance the story; add one new layer"),
    ("Chorus", 8, 4, "hook again, bigger than the first time"),
    ("Outro", 4, 0, "resolve on the hook motif"),
)
_SHORT_TWO_CHORUS = (
    ("Intro", 2, 0, "a quick groove-in"),
    ("Verse", 8, 4, "set the scene, lower energy, conversational melody"),
    ("Chorus", 8, 4, "the hook: highest energy, title line"),
    ("Verse", 8, 4, "advance the story; add one new layer"),
    ("Chorus", 8, 4, "hook again, bigger"),
    ("Outro", 2, 0, "button ending"),
)
_MEDIUM_BRIDGE = (
    ("Intro", 4, 0, "establish groove and a signature motif"),
    ("Verse", 8, 4, "story and detail, sparse arrangement"),
    ("Pre-Chorus", 4, 2, "lift: rising melody, building drums, tension"),
    ("Chorus", 8, 4, "the hook: full band, title line"),
    ("Verse", 8, 4, "advance the story; add one new layer"),
    ("Pre-Chorus", 4, 2, "lift again"),
    ("Chorus", 8, 4, "hook, bigger than the first time"),
    ("Bridge", 8, 4, "contrast: new chords or perspective, strip back"),
    ("Chorus", 8, 4, "final chorus: biggest moment, harmonies, ad-libs"),
    ("Outro", 4, 0, "resolve on the hook motif"),
)
_RADIO = (
    ("Intro", 4, 0, "establish groove and a signature motif"),
    ("Verse", 16, 8, "story and detail, sparse arrangement"),
    ("Pre-Chorus", 4, 2, "lift: rising melody and tension"),
    ("Chorus", 8, 4, "the hook: full band, title line"),
    ("Verse", 8, 4, "advance the story; add one new layer"),
    ("Pre-Chorus", 4, 2, "lift again"),
    ("Chorus", 8, 4, "hook, bigger than the first time"),
    ("Bridge", 8, 4, "contrast: new chords or perspective, strip back"),
    ("Chorus", 8, 4, "final chorus: biggest moment, harmonies, ad-libs"),
    ("Outro", 4, 0, "resolve on the hook motif"),
)
_VOCAL_FORMS = (_HOOK, _SHORT, _SHORT_TWO_CHORUS, _MEDIUM, _MEDIUM_BRIDGE, _RADIO)
# Longer limits grow the radio form the way a producer would: a solo or
# breakdown, a doubled last chorus, a third verse, then a longer ride-out.
_SOLO = ("Instrumental", 8, 0, "solo or breakdown before the last chorus")
_LAST_CHORUS = ("Chorus", 8, 4, "repeat the hook to close")
_THIRD_VERSE = (("Verse", 8, 4, "third verse: a new angle on the story"),
                ("Chorus", 8, 4, "hook again"))
_INSTRUMENTAL_FORMS = (
    (
        ("Intro", 2, 0, "motif in one instrument"),
        ("Instrumental", 8, 0, "main theme, full texture"),
        ("Outro", 2, 0, "button ending"),
    ),
    (
        ("Intro", 4, 0, "establish groove and motif"),
        ("Instrumental", 8, 0, "theme A"),
        ("Instrumental", 8, 0, "theme A with added layer"),
        ("Outro", 4, 0, "resolve"),
    ),
    (
        ("Intro", 4, 0, "establish groove and motif"),
        ("Instrumental", 8, 0, "theme A, sparse"),
        ("Instrumental", 8, 0, "theme A, fuller"),
        ("Instrumental", 8, 0, "theme B: contrast, new chords"),
        ("Instrumental", 8, 0, "theme A, fullest"),
        ("Outro", 4, 0, "resolve"),
    ),
)
_BACKGROUND_ROLES = {
    "Instrumental": "steady texture that stays under dialogue; no busy lead",
}
# Rap is hook-first, with one-bar verse lines.
_RAP_FORMS = (
    _HOOK,
    (
        ("Intro", 2, 0, "beat in, a tag or ad-lib"),
        ("Chorus", 8, 4, "hook first: short, rhythmic, chantable"),
        ("Verse", 16, 16, "dense one-bar lines, internal rhyme, a clear story or boast"),
        ("Chorus", 8, 4, "hook again"),
        ("Outro", 2, 0, "beat rides out"),
    ),
    (
        ("Intro", 4, 0, "beat in, a tag or ad-lib"),
        ("Chorus", 8, 4, "hook first: short, rhythmic, chantable"),
        ("Verse", 16, 16, "dense one-bar lines, internal rhyme, a clear story or boast"),
        ("Chorus", 8, 4, "hook again"),
        ("Verse", 16, 16, "switch the flow or cadence; raise the stakes"),
        ("Chorus", 8, 4, "hook, with ad-libs"),
        ("Outro", 4, 0, "beat rides out or cuts clean"),
    ),
    (
        ("Intro", 4, 0, "beat in, a tag or ad-lib"),
        ("Chorus", 8, 4, "hook first: short, rhythmic, chantable"),
        ("Verse", 16, 16, "dense one-bar lines, internal rhyme, a clear story or boast"),
        ("Chorus", 8, 4, "hook again"),
        ("Verse", 16, 16, "switch the flow or cadence; raise the stakes"),
        ("Chorus", 8, 4, "hook again"),
        ("Verse", 16, 16, "beat drop-outs for emphasis; land the punchline"),
        ("Chorus", 8, 4, "hook, with ad-libs"),
        ("Outro", 4, 0, "beat rides out or cuts clean"),
    ),
)
# Longer limits grow the largest rap form verse by verse, each followed by the hook.
_RAP_MORE = (
    ("Verse", 16, 16, "a new flow or a guest-style switch; new angle on the story"),
    ("Verse", 16, 16, "closing verse: bring the story home"),
)
_RAP_HOOK = ("Chorus", 8, 4, "hook again")


@dataclass(frozen=True)
class Genre:
    """Genre-native defaults: conventions are conditional, not universal."""

    name: str
    bpm: int
    note: str
    verse_line_bars: int = LINE_BARS
    long_line: int = 14  # syllables before a sung line sounds rushed
    roles: tuple[tuple[str, str], ...] = ()


POP = Genre("verse-chorus", DEFAULT_BPM, "a clear hook, verses lower than choruses, contrast before the last chorus")
_GENRES = (
    (
        re.compile(r"\b(rap|rapper|hip[- ]?hop|trap|drill|grime|boom[- ]?bap)\b|说唱|嘻哈|ラップ", re.IGNORECASE),
        Genre("rap", 90, "hook-first; dense one-bar verse lines with internal rhyme; a short chantable hook",
              verse_line_bars=1, long_line=22),
    ),
    (
        re.compile(r"\b(edm|house|techno|trance|club|electro|dubstep|dnb|drum (?:and|&|n) bass|dance)\b",
                   re.IGNORECASE),
        Genre("dance", 124, "builds and drops; the drop carries the hook", roles=(
            ("Intro", "groove intro, kick and bass, easy to mix in"),
            ("Pre-Chorus", "build: risers, snare roll, filter opening"),
            ("Chorus", "drop: the hook lands with full kick and bass"),
            ("Bridge", "breakdown: pads and vocal, then a second build"),
            ("Outro", "groove outro that can be mixed out"),
        )),
    ),
    (
        re.compile(r"\b(ballad|folk|acoustic|lullaby|singer[- ]songwriter|country)\b|抒情|民谣", re.IGNORECASE),
        Genre("ballad", 76, "restraint; growth through harmony, a second instrument, or delivery rather than loudness",
              roles=(
                  ("Verse", "intimate: voice with one or two instruments"),
                  ("Pre-Chorus", "lift through harmony rather than drums"),
                  ("Chorus", "the hook, opening up with harmony or a second instrument"),
                  ("Bridge", "the emotional turn; strip right back"),
              )),
    ),
    (
        re.compile(r"\b(ambient|cinematic|soundtrack|meditation|lo[- ]?fi|drone|new age)\b|氛围", re.IGNORECASE),
        Genre("ambient", 80, "gradual evolution around a recurring motif; no sudden drops", roles=(
            ("Instrumental", "texture evolves slowly around the motif"),
            ("Chorus", "the motif returns, wider and warmer"),
        )),
    ),
)


def detect_genre(*texts: str) -> Genre:
    text = " ".join(str(text or "") for text in texts)
    return next((genre for pattern, genre in _GENRES if pattern.search(text)), POP)


def bar_seconds(bpm: float) -> float:
    return 240.0 / float(bpm)


def detect_bpm(*texts: str) -> int | None:
    for text in texts:
        text = str(text or "")
        if match := _BPM_RANGE.search(text):
            low, high = sorted(int(value) for value in match.groups())
            if 40 <= low <= high <= 220:
                return round((low + high) / 2)
        for match in _BPM.finditer(text):
            value = int(match.group(1) or match.group(2))
            if 40 <= value <= 220:
                return value
    return None


def wants_instrumental(vocals: str, use: str, idea: str) -> bool:
    if vocals == INSTRUMENTAL:
        return True
    if vocals not in (AUTO, "", None):
        return False
    return use == BACKGROUND or bool(INSTRUMENTAL_REQUEST.search(str(idea or "")))


def resolve_use(use: str, max_seconds: float) -> str:
    if use in (FULL_SONG, HOOK, BACKGROUND):
        return use
    return HOOK if float(max_seconds) <= 40 else FULL_SONG


def plan_song(
    max_seconds: float,
    *,
    vocals: str = AUTO,
    use: str = AUTO,
    idea: str = "",
    bpm: int | None = None,
) -> SongPlan:
    """Size a conventional form so it ends naturally inside the length limit."""
    use = resolve_use(use, max_seconds)
    instrumental = wants_instrumental(vocals, use, idea)
    genre = detect_genre(idea)
    assumed = bpm is None
    bpm = int(bpm or genre.bpm)
    # Leave a little headroom so the ending is composed, not truncated.
    budget = max(1, int(float(max_seconds) * 0.97 / bar_seconds(bpm)))
    vocal_forms = _RAP_FORMS if genre.verse_line_bars == 1 else _VOCAL_FORMS
    if use == HOOK:
        forms = _INSTRUMENTAL_FORMS[:1] if instrumental else vocal_forms[:1]
    else:
        forms = _INSTRUMENTAL_FORMS if instrumental else vocal_forms
    sections = list(forms[0])
    for form in forms:
        if sum(bars for _, bars, _, _ in form) <= budget:
            sections = list(form)
    if use != HOOK:
        sections = _extend(sections, budget, instrumental, largest=forms[-1])
    elif sum(bars for _, bars, _, _ in sections) > budget:
        # A slow jingle: a one-bar pickup and button, and a shorter hook,
        # rather than overrun the limit.
        (intro, _, _, intro_role), (tag, _, lines, role), (outro, _, _, outro_role) = sections
        bars = max(4, (budget - 2) // 2 * 2)
        sections = [
            (intro, 1, 0, intro_role),
            (tag, bars, (bars // LINE_BARS) if lines else 0, role),
            (outro, 1, 0, outro_role),
        ]
    roles = dict(genre.roles)
    if use == BACKGROUND:
        roles.update(_BACKGROUND_ROLES)
    sections = [(tag, bars, lines, roles.get(tag, role)) for tag, bars, lines, role in sections]
    if genre.name == "ballad":
        last = max((i for i, section in enumerate(sections) if section[0] == "Chorus"), default=None)
        if last is not None and sum(section[0] == "Chorus" for section in sections) > 1:
            tag, bars, lines, _role = sections[last]
            sections[last] = (tag, bars, lines,
                              "last chorus: biggest, or deliberately stripped back if that lands harder")
    return SongPlan(
        use, instrumental, bpm, assumed,
        tuple(PlannedSection(*section) for section in sections), genre,
    )


def _extend(sections, budget, instrumental, *, largest):
    def total():
        return sum(bars for _, bars, _, _ in sections)

    if instrumental:
        variation = 0
        bars = 16 if budget >= 96 else 8  # long cues develop in longer phrases
        while total() + bars <= budget and variation < 8:
            variation += 1
            role = "development: vary the theme" if variation % 2 else "theme returns, fuller"
            sections.insert(len(sections) - 1, ("Instrumental", bars, 0, role))
        return sections
    # Only the family's largest form grows; a smaller one still had room to spare.
    if len(sections) < len(largest):
        return sections
    if largest is _RAP_FORMS[-1]:
        # Rap grows by another verse and hook rather than a solo or bridge.
        for verse in _RAP_MORE:
            if total() + verse[1] + _RAP_HOOK[1] > budget:
                break
            sections[-1:-1] = (verse, _RAP_HOOK)
    else:
        if total() + _SOLO[1] <= budget:
            sections.insert(len(sections) - 2, _SOLO)
        if total() + _LAST_CHORUS[1] <= budget:
            sections.insert(len(sections) - 1, _LAST_CHORUS)
        if total() + sum(bars for _, bars, _, _ in _THIRD_VERSE) <= budget:
            bridge = next(i for i, section in enumerate(sections) if section[0] == "Bridge")
            sections[bridge:bridge] = _THIRD_VERSE
    tag, bars, lines, _role = sections[-1]
    extra = min(12, (budget - total()) // 4 * 4)
    if extra > 0:
        sections[-1] = (tag, bars + extra, lines, "extended ride-out on the hook motif")
    return sections


def format_plan(plan: SongPlan) -> str:
    """The plan as a compact brief the writer can follow."""
    genre = plan.genre or POP
    tempo = f"~{plan.bpm} BPM" + (f" (assumed for {genre.name})" if plan.bpm_assumed else "")
    rows = []
    for section in plan.sections:
        material = f"{section.lines} lines" if section.lines else "no vocals"
        rows.append(f"  [{section.tag}] {section.bars} bars, {material}: {section.role}")
    density = (
        "verse lines are about 1 bar, hook lines about 2"
        if genre.verse_line_bars == 1 else f"one lyric line is about {LINE_BARS} bars"
    )
    return (
        f"Style profile: {genre.name}: {genre.note}. Treat it as a default, not a rule.\n"
        f"Suggested form at {tempo}, about {_clock(plan.seconds)} ({density}):\n"
        + "\n".join(rows)
    )


def build_brief(
    *,
    engine: str,
    max_seconds: float,
    vocals: str = AUTO,
    language: str = AUTO,
    use: str = AUTO,
    lyrics_policy: str = KEEP_LYRICS,
    notes: str = "",
    idea: str = "",
    lyrics: str = "",
    mode: str | None = None,
) -> str:
    """Writer context: the user's plain choices plus the producer's plan."""
    plan = plan_song(
        max_seconds, vocals=vocals, use=use, idea=idea,
        bpm=detect_bpm(idea),
    )
    has_lyrics = bool(str(lyrics or "").strip())
    lines = [
        "Song brief (from the app's simple controls; explicit words in the user's "
        "text win on conflict):",
        f"- Made for: {plan.use}",
        f"- Vocals: {INSTRUMENTAL if plan.instrumental else (vocals or AUTO)}"
        + (" (choose what suits the idea)" if vocals in (AUTO, "", None) and not plan.instrumental else ""),
        f"- Lyric language: {language if language not in (AUTO, '', None) else 'same as the idea text'}",
        f"- Maximum length: {float(max_seconds):.0f} seconds; plan a composed ending "
        "just inside it so the song is not cut off",
    ]
    if mode:
        lines.append(f"- Score mode: {mode}")
    if has_lyrics:
        lines.append(
            "- Existing lyrics: "
            + (
                "polish for singability (fix tags, split overlong lines, tighten "
                "rhyme and stress) while keeping meaning, language, and hook"
                if lyrics_policy == POLISH_LYRICS
                else "keep them exactly; build the production around their sections"
            )
        )
        lines.append(f"- Style profile: {plan.genre.name}: {plan.genre.note} (a default, not a rule)")
        lines.append(lyric_summary(lyrics, max_seconds, bpm=plan.bpm, genre=plan.genre))
    else:
        lines.append(format_plan(plan))
        lines.append(
            "Line counts assume the tempo above. If you choose a faster tempo, add "
            "lines; slower, remove lines, so the sung material still fits the length."
        )
    if str(notes or "").strip():
        lines.append(
            "Revision notes (apply these and keep everything else):\n" + str(notes).strip()
        )
    lines.append(f"Engine: {engine}")
    return "\n".join(lines)


@dataclass(frozen=True)
class LyricSection:
    tag: str
    lines: tuple[str, ...]


def parse_lyrics(lyrics: str) -> list[LyricSection]:
    sections: list[LyricSection] = []
    tag, lines = "", []
    for raw in str(lyrics or "").splitlines():
        match = _TAG.match(raw)
        if match:
            if tag or lines:
                sections.append(LyricSection(tag, tuple(lines)))
            tag, lines = match.group(1).strip(), []
            rest = match.group(2).strip()
            if rest:
                lines.append(rest)
        elif raw.strip():
            lines.append(raw.strip())
    if tag or lines:
        sections.append(LyricSection(tag, tuple(lines)))
    return sections


def syllables(line: str) -> int:
    text = str(line)
    count = len(_CJK.findall(text))
    for word in _WORD.findall(_CJK.sub(" ", text)):
        groups = len(_VOWEL_GROUPS.findall(word))
        if groups > 1 and word.lower().endswith("e") and not word.lower().endswith(("le", "ee")):
            groups -= 1
        count += max(1, groups)
    return count


def _is_cjk(line: str) -> bool:
    return len(_CJK.findall(line)) > len(_WORD.findall(_CJK.sub(" ", line)))


def _kind(tag: str) -> str:
    name = tag.lower()
    for kind in ("pre-chorus", "prechorus", "chorus", "hook", "refrain", "verse", "bridge",
                 "intro", "outro", "instrumental", "solo", "break", "interlude", "drop"):
        if name.startswith(kind) or kind in name:
            return {
                "prechorus": "pre-chorus", "hook": "chorus", "refrain": "chorus",
                "solo": "instrumental", "break": "instrumental",
                "interlude": "instrumental", "drop": "instrumental",
            }.get(kind, kind)
    return "other"


def _rapped(section: LyricSection, genre: Genre) -> bool:
    return genre.verse_line_bars == 1 and _kind(section.tag) == "verse"


def _section_seconds(section: LyricSection, bpm: int, genre: Genre = POP) -> float:
    bar = bar_seconds(bpm)
    if not section.lines:
        return (4 if _kind(section.tag) in {"intro", "outro"} else 8) * bar
    # Sung lines move in eighth notes over about two bars; rap verses in
    # sixteenths over about one.
    line_bars, note = (1, 15.0) if _rapped(section, genre) else (LINE_BARS, 30.0)
    total = 0.0
    for line in section.lines:
        sung = syllables(line) * note / bpm + 60.0 / bpm  # plus a breath
        weight = 0.5 if line.startswith("(") and line.endswith(")") else 1.0
        total += max(line_bars * bar, sung) * weight
    return total


def estimate_seconds(lyrics: str, *, bpm: int = DEFAULT_BPM, genre: Genre = POP) -> float:
    return sum(_section_seconds(section, bpm, genre) for section in parse_lyrics(lyrics))


def lyric_summary(lyrics: str, max_seconds: float, *, bpm: int | None = None,
                  genre: Genre = POP) -> str:
    tempo = int(bpm or genre.bpm)
    sections = parse_lyrics(lyrics)
    estimate = estimate_seconds(lyrics, bpm=tempo, genre=genre)
    form = " · ".join(f"[{s.tag}] {len(s.lines)}" if s.tag else f"(untagged) {len(s.lines)}"
                      for s in sections)
    return (
        f"Existing lyric form (lines per section): {form}\n"
        f"Estimated sung length at ~{tempo} BPM: {_clock(estimate)} of the "
        f"{_clock(float(max_seconds))} limit"
    )


@dataclass(frozen=True)
class Finding:
    level: str  # "warn" blocks nothing but needs attention; "tip" is advice
    text: str


def check_song(
    *,
    engine: str,
    description: str,
    lyrics: str,
    max_seconds: float,
    vocals: str = AUTO,
    use: str = AUTO,
) -> tuple[SongPlan, list[LyricSection], float, list[Finding]]:
    """Producer's pre-flight check of the current draft. No network, no GPU."""
    bpm = detect_bpm(description)
    plan = plan_song(max_seconds, vocals=vocals, use=use, idea=description, bpm=bpm)
    tempo, genre = plan.bpm, plan.genre
    sections = parse_lyrics(lyrics)
    estimate = estimate_seconds(lyrics, bpm=tempo, genre=genre)
    sung = [line for section in sections for line in section.lines]
    findings: list[Finding] = []
    words = len(str(description or "").split())

    if not str(description or "").strip():
        findings.append(Finding(
            "warn",
            "Add a description of the sound, or press Write to draft one"
            + (" from your lyrics." if sung else "."),
        ))
    if engine == "YuE2":
        if not str(lyrics or "").strip():
            findings.append(Finding("warn", "YuE2 needs lyrics or section tags; Write can draft them."))
        if words > 45 or str(description or "").count(".") > 1:
            findings.append(Finding(
                "tip",
                "YuE2 follows short comma-separated tags best (genre, mood, tempo, voice, "
                "instruments). Write turns a long description into tags.",
            ))
    if plan.instrumental and sung:
        findings.append(Finding(
            "warn", "You chose an instrumental, but the lyrics contain words that will be sung.",
        ))
    if sung and not any(section.tag for section in sections):
        findings.append(Finding(
            "warn",
            "No section tags. Add [Verse], [Chorus] and so on so the model knows where the hook goes.",
        ))
    if sections:
        limit = float(max_seconds)
        if estimate > limit * 1.05:
            findings.append(Finding(
                "warn",
                f"The draft needs about {_clock(estimate)} but the limit is {_clock(limit)}, "
                f"so the ending will likely be cut. Set the length to about "
                f"{_clock(_round_up(estimate * 1.1))} or trim a section.",
            ))
        elif sung and limit >= 60 and estimate < limit * 0.4:
            findings.append(Finding(
                "tip",
                f"The lyrics fill about {_clock(estimate)}; the song will likely end early. "
                "That is fine, or add a verse and another chorus.",
            ))
    kinds = [_kind(section.tag) for section in sections if section.lines]
    if sung and not plan.instrumental:
        if "chorus" not in kinds and estimate > 45:
            findings.append(Finding(
                "tip", "No [Chorus]. A repeated hook is what makes a song memorable.",
            ))
        elif kinds.count("chorus") == 1 and estimate > 100:
            findings.append(Finding(
                "tip", "The chorus appears once. A hook usually needs to come back two or three times.",
            ))
        long_lines = [
            line for section in sections for line in section.lines
            if syllables(line) > (
                genre.long_line if _rapped(section, genre) else (16 if _is_cjk(line) else 14)
            )
        ]
        if long_lines:
            findings.append(Finding(
                "tip",
                f"{len(long_lines)} line(s) are long enough to sound rushed, such as "
                f"“{_clip(long_lines[0])}”. Split them or use Polish my lyrics.",
            ))
        # Rap flows vary line length on purpose.
        uneven = "" if genre.verse_line_bars == 1 else _uneven_sections(sections)
        if uneven:
            findings.append(Finding(
                "tip",
                f"Line lengths jump around in [{uneven}]. Matching syllable counts in "
                "parallel lines makes the melody repeat naturally.",
            ))
    return plan, sections, estimate, findings


def _uneven_sections(sections: list[LyricSection]) -> str:
    for section in sections:
        counts = [syllables(line) for line in section.lines if not line.startswith("(")]
        if len(counts) >= 4 and max(counts) - min(counts) > max(6, min(counts)):
            return section.tag or "untagged"
    return ""


_KIND_CLASS = {
    "intro": "edge", "outro": "edge", "verse": "verse", "pre-chorus": "lift",
    "chorus": "hook", "bridge": "contrast", "instrumental": "inst",
}


def render_check(
    *,
    engine: str,
    description: str,
    lyrics: str,
    max_seconds: float,
    vocals: str = AUTO,
    use: str = AUTO,
) -> str:
    plan, sections, estimate, findings = check_song(
        engine=engine, description=description, lyrics=lyrics,
        max_seconds=max_seconds, vocals=vocals, use=use,
    )
    limit = float(max_seconds)
    if sections:
        blocks = [
            (section.tag or "Lyrics", _section_seconds(section, plan.bpm, plan.genre), len(section.lines))
            for section in sections
        ]
        caption = (
            f"Your draft: about {_clock(estimate)} of {_clock(limit)} · "
            f"{sum(count for *_, count in blocks)} sung lines"
        )
    else:
        blocks = [(s.tag, s.bars * bar_seconds(plan.bpm), s.lines) for s in plan.sections]
        caption = (
            f"Suggested shape for {_clock(limit)}: about {_clock(plan.seconds)}"
            + (" · instrumental" if plan.instrumental else "")
        )
    scale = max(limit, estimate, plan.seconds, 1.0)
    chips = "".join(
        f'<span class="h3-song-chip h3-song-chip-{_KIND_CLASS.get(_kind(tag), "other")}" '
        f'style="flex-grow:{max(seconds / scale, 0.02):.3f}" '
        f'title="{escape(tag)} · ~{seconds:.0f}s{f" · {count} lines" if count else ""}">'
        f"{escape(tag)}</span>"
        for tag, seconds, count in blocks
    )
    spare = max(0.0, limit - sum(seconds for _, seconds, _ in blocks))
    if spare > 0:
        chips += f'<span class="h3-song-chip h3-song-chip-spare" style="flex-grow:{spare / scale:.3f}"></span>'
    style = "" if plan.genre is POP else f"{plan.genre.name} · "
    tempo = style + f"~{plan.bpm} BPM" + (" assumed" if plan.bpm_assumed else " from your description")
    items = "".join(
        f'<li class="h3-song-check-{finding.level}">{escape(finding.text)}</li>'
        for finding in findings
    ) or '<li class="h3-song-check-ok">Looks ready. Generate, then try another take if the first is not right.</li>'
    return (
        '<div class="h3-song-check" role="status">'
        f'<p class="h3-song-check-caption">{escape(caption)} <span>· {escape(tempo)}</span></p>'
        f'<div class="h3-song-timeline" aria-hidden="true">{chips}</div>'
        f"<ul>{items}</ul></div>"
    )


def take_note(actual_seconds: float | None, max_seconds: float) -> str:
    """After a take: a song that ran to the limit was probably cut off."""
    if not actual_seconds:
        return ""
    limit = float(max_seconds)
    if actual_seconds >= limit - max(1.5, limit * 0.02):
        return (
            f"Length {_clock(actual_seconds)} reached the {_clock(limit)} limit, so the ending may be "
            f"cut. For a composed ending, raise the length to about {_clock(_round_up(limit * 1.25))} "
            "or trim a section."
        )
    if limit >= 60 and actual_seconds < limit * 0.3:
        return (
            f"Length {_clock(actual_seconds)} is much shorter than the {_clock(limit)} limit. "
            "Listen for missing sections or lyrics; try another take if it stopped early."
        )
    return f"Length {_clock(actual_seconds)} of {_clock(limit)}; it ended on its own."


def instrumental_tags(plan: SongPlan) -> str:
    """YuE2 needs section tags even without words; follow the planned form."""
    return "\n\n".join(f"[{section.tag}]" for section in plan.sections)


def _round_up(seconds: float) -> int:
    return int(-(-seconds // 15) * 15)


def _clock(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _clip(text: str, limit: int = 48) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
