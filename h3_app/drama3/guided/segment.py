"""Stage 0: split the user's text into numbered source units, so no model has to copy it.

Code, not the model, decides where one unit ends and the next begins: paragraphs, then
quotation boundaries, then sentence ends (at most two sentences a unit). The content stage
only says who speaks each unit, and ``attribute_units`` turns that into locked utterances
whose words are exactly the source's.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from h3_app.drama3.serving.speaker_effects import SPEAKER_TAGS
from h3_app.errors import H3Error

from .model import HEADING, LINE, NARRATOR, Utterance

SCRIPT, PROSE, BRIEF = "script", "prose", "brief"
QUOTE, NARRATION, LABELLED = "quote", "narration", "line"

NARRATOR_ALIASES = frozenset({"narrator", "narration", "旁白", "story teller", "storyteller"})
MAX_SENTENCES = 2

_QUOTE_RE = re.compile(r"“([^“”]*)”|\"([^\"]*)\"|「([^「」]*)」|『([^『』]*)』")
_SPEAKER_TAG_LINE_RE = re.compile(r"^(\s*)<\|speaker:(\d+)\|>\s*(.*)$")
# A script label: a short name, an optional (cue), then a colon. Prose such as 她问：“…” is told
# apart by the label's length and punctuation.
_LABEL_RE = re.compile(r"^\s*(?P<name>[^\s:：(（<>\[\]|#][^:：(（<>\[\]|\n]{0,63}?)\s*"
                       r"(?:[(（](?P<hint>[^)）]{0,40})[)）])?\s*[:：]\s*(?P<body>.*)$")
_LABEL_PUNCTUATION_RE = re.compile(r"[，,。.!?！？“”\"「」;；、]")
_EFFECT_TAGS_RE = re.compile(r"^\s*((?:<(?:%s)>\s*)+)" % "|".join(map(re.escape, SPEAKER_TAGS)), re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"(?:[。！？]+|…+|\.{3,}|[.!?]+(?=\s|$))[”\"'」』)）]*")
_SPOKEN_RE = re.compile(r"\w", re.UNICODE)
_CJK_PIPE_RE = re.compile(r"(?<=[\u3400-\u9fff])\|(?=[\u3400-\u9fff])")
# Explicit brief instructions ("Write a scene where...", "... 30 seconds", "请写一段…") stay briefs
# even when they quote a sample line or span several sentences.
_BRIEF_RE = re.compile(
    r"^\s*(?:please\s+)?(?:write|create|generate|make|draft|compose|imagine)\b"
    r"|\b(?:write|create|generate|make|draft|compose)\s+(?:a|an|the|me\s+a)?\s*"
    r"(?:short\s+|brief\s+|tense\s+|funny\s+|dramatic\s+)*"
    r"(?:scene|script|dialogue|audio\s+drama|drama|conversation|exchange|monologue|skit|piece)\b"
    r"|^\s*(?:a\s+)?(?:short\s+|brief\s+)?(?:scene|dialogue|audio\s+drama|drama|conversation|script)\s+"
    r"(?:where|about|between|in\s+which|with)\b"
    r"|^\s*(?:two|three|four)\s+[^.!?]*\b(?:arguing|talking)\b"
    r"|^\s*(?:a|an)\s+(?:short|brief)\s+[^.!?]*\b(?:story|scene|dialogue|drama)\s*[.!]?$"
    r"|^\s*(?:请|帮我)?(?:写|创作|生成|编)(?:一段|一个|一篇|一出)"
    r"|(?:关于.+的(?:广播剧|对话|场景|短剧))",
    re.IGNORECASE,
)


def replace_speaker_tag_line(line: str, names: dict[int, str] | None = None) -> str:
    """Turn a leading ``<|speaker:N|>`` tag on ``line`` into a ``Name: ...`` script label."""
    match = _SPEAKER_TAG_LINE_RE.match(line)
    if match is None:
        return line
    indent, raw_id, rest = match.groups()
    name = (names or {}).get(int(raw_id)) or f"Speaker {raw_id}"
    if re.match(r"^(?:[(（][^)）]{0,40}[)）]\s*)?[:：]", rest):
        return f"{indent}{name}{rest}"
    return f"{indent}{name}: {rest}"


@dataclass(frozen=True)
class SourceUnit:
    id: str
    kind: str  # quote, narration, a labelled script line, or a heading
    text: str  # the words to speak, quotation marks removed
    paragraph: int
    start: int  # the unit's span in its paragraph, quotation marks included
    end: int
    label: str = ""  # a script line's label, as written
    hint: str = ""  # a script line's parenthesized cue, such as "surprised"
    effects: tuple[str, ...] = ()  # a script line's leading speaker effect tags


@dataclass(frozen=True)
class Source:
    kind: str
    paragraphs: tuple[str, ...]
    units: tuple[SourceUnit, ...]


def _label(line: str) -> re.Match | None:
    line = replace_speaker_tag_line(line)
    match = _LABEL_RE.match(line)
    if match is None or not match["body"].strip():
        return None
    name = match["name"].strip()
    # Names are short: 守护精灵, Old Captain Reyes. 父亲慢慢点头：“…” is an attribution.
    if _LABEL_PUNCTUATION_RE.search(name):
        return None
    # A long label is a capitalized name ("Commander Alex Morgan Reyes"), not a clause before a colon.
    if len(name.split()) > 3 and not all(word[:1].isupper() for word in name.split()):
        return None
    if len(re.findall(r"[^\x00-\x7f]", name)) > 10:
        return None
    # A prose clause before a quote is spoken narration, even with several names/words.
    if re.search(r'\s+(?:said|says|asked|asks|replied|answered|shouted|whispered|murmured|added|cried)\b', name):
        return None
    if re.search(r"(?:说|问|答|喊|道|点头)$", name):
        return None
    return match


def input_kind(text: str) -> str:
    """A guess at what the user gave: a labelled script, prose to perform, or a brief to write from."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    spoken = [line for line in lines if not is_control_line(line)]
    if spoken and sum(_label(line) is not None for line in spoken) >= 0.6 * len(spoken):
        return SCRIPT
    joined = " ".join(lines)
    if len(joined) <= 500 and _BRIEF_RE.search(joined):
        return BRIEF
    if _QUOTE_RE.search(joined):
        return PROSE
    return PROSE


def source_request(text: str) -> tuple[str, str]:
    """Separate an explicit expansion instruction on its own line from supplied speech."""
    lines = text.splitlines()
    if len([line for line in lines if line.strip()]) < 2:
        return text, ''
    instruction = re.compile(
        r'\s*(?:please\s+)?(?:expand\s+(?:with\s+(?:more\s+)?(?:dialogue|reactions)|'
        r'(?:the\s+)?(?:following|supplied|provided)\s+(?:text|script|story)(?:\s+with\s+more\s+dialogue)?)'
        r'|add\s+(?:more\s+)?(?:dialogue|a new scene|reactions)\s+to\s+(?:the\s+)?'
        r'(?:following|supplied|provided)\s+(?:text|script|story))\s*[.!]?\s*'
        r'|\s*(?:请|帮我)?(?:扩写(?:以下|下面|提供的)(?:文本|剧本|故事)|'
        r'(?:为|给)(?:以下|下面的)(?:文本|剧本|故事)(?:增加|添加)对话)\s*[。！]?\s*', re.I)
    # Only a complete directive at an edge of the request may be removed. Ordinary
    # imperatives inside narration ("Expand your chest...") remain performable text.
    occupied = [index for index, line in enumerate(lines) if line.strip()]
    edges = {occupied[0], occupied[-1]}
    selected = {index for index in edges if instruction.fullmatch(lines[index])}
    directives = [line for index, line in enumerate(lines) if index in selected]
    source = [line for index, line in enumerate(lines) if index not in selected]
    return '\n'.join(source), '\n'.join(directives)


def is_control_line(line: str) -> bool:
    """Supported standalone staging, never a piece of quoted story prose."""
    from h3_app.drama3.serving.tree import CONTAINER_TAG_RE, PAUSE_TAG_RE, parse_sound_tag

    line = line.strip()
    return bool(CONTAINER_TAG_RE.fullmatch(line) or PAUSE_TAG_RE.fullmatch(line)
                or (line.casefold().startswith('<sound') and parse_sound_tag(line) is not None))


def restore_structure(source: Source, utterances: list[Utterance], directed: str) -> str:
    """Put directed spoken units back between the original standalone controls."""
    lines = directed.splitlines()
    if len(lines) != len(utterances):
        raise H3Error("The directed script changed the number of source units.")
    units = {unit.id: unit for unit in source.units}
    by_paragraph: dict[int, list[str]] = {}
    for utterance, line in zip(utterances, lines):
        if not utterance.sources:
            raise H3Error("An inserted line cannot be placed inside existing staging without a source anchor.")
        index = units[utterance.sources[0]].paragraph
        by_paragraph.setdefault(index, []).append(line)
    result = []
    for index, paragraph in enumerate(source.paragraphs):
        result.extend([paragraph] if is_control_line(paragraph) else by_paragraph.get(index, []))
    return "\n".join(result)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in ``text``, whitespace excluded."""
    spans, start = [], 0
    for match in _SENTENCE_END_RE.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    trimmed = []
    for begin, end in spans:
        piece = text[begin:end]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            trimmed.append((begin + lead, begin + len(piece.rstrip())))
    return trimmed


def _chunks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """``text[start:end]`` cut at sentence ends into pieces of at most two sentences."""
    spans = [(start + a, start + b) for a, b in _sentence_spans(text[start:end])]
    return [(group[0][0], group[-1][1]) for group in
            (spans[i:i + MAX_SENTENCES] for i in range(0, len(spans), MAX_SENTENCES))]


def _is_heading(paragraph: str) -> bool:
    text = paragraph.strip()
    return (len(text) <= 40 and not _QUOTE_RE.search(text) and _label(text) is None
            and not re.search(r"[.!?。！？…，,;；:：]$", text))


def performance_controls(text: str) -> str:
    """Convert only recognized performance annotations; prose asides remain spoken."""
    from h3_app.drama3.serving.planner import AUDIBLE_EVENTS

    events = {event: event for event in AUDIBLE_EVENTS}
    events.update({'笑': 'laughs', '叹气': 'sighs', 'whispering': 'whispering',
                   '低声': 'whispering', '耳语': 'whispering'})

    def annotation(match):
        content = next(group for group in match.groups() if group is not None).strip().casefold()
        return '[' + events[content] + ']' if content in events else match.group()

    return re.sub(r'\(([^()]*)\)|（([^（）]*)）|\*([^*\n]+)\*', annotation, text)


def segment_source(text: str, kind: str | None = None) -> Source:
    """The user's script or prose as numbered units ``s1…sN``."""
    paragraphs = tuple(
        replace_speaker_tag_line(_CJK_PIPE_RE.sub("", line.strip()))
        for line in str(text or "").replace("\r\n", "\n").split("\n") if line.strip()
    )
    kind = kind or input_kind(text)
    units: list[SourceUnit] = []

    def add(unit_kind, words, index, start, end, **extra):
        if _SPOKEN_RE.search(words):
            words = performance_controls(words)
            units.append(SourceUnit(f"s{len(units) + 1}", unit_kind, words.strip(), index, start, end, **extra))

    first = 0
    if len(paragraphs) >= 2:
        while first < len(paragraphs) - 1 and not is_control_line(paragraphs[first]) and _is_heading(paragraphs[first]):
            first += 1
    if first:
        # A multi-level heading ("第一卷 …" then "第一章 …") is read once, as one title.
        add(HEADING, " ".join(paragraphs[:first]), 0, 0, len(paragraphs[0]))
    for index in range(first, len(paragraphs)):
        paragraph = paragraphs[index]
        if is_control_line(paragraph):
            continue
        if kind == SCRIPT and (match := _label(paragraph)) is not None:
            body_start = match.start("body")
            body = paragraph[body_start:]
            effects_match = _EFFECT_TAGS_RE.match(body)
            effects = tuple(re.findall(r"<([^<>]+)>", effects_match.group(1).lower())) if effects_match else ()
            if effects_match:
                body_start += effects_match.end()
            text_span = paragraph[body_start:].strip()
            if (quoted := _QUOTE_RE.fullmatch(text_span)) is not None:
                inner = next(group for group in quoted.groups() if group is not None)
                span_start = paragraph.find(inner, body_start)
                span_end = span_start + len(inner)
            else:
                span_start, span_end = body_start, len(paragraph)
            label, hint = match["name"].strip(), (match["hint"] or "").strip()
            for start, end in _chunks(paragraph, span_start, span_end):
                add(LABELLED, paragraph[start:end], index, start, end, label=label, hint=hint, effects=effects)
                hint = ""  # the cue applies to the line's first piece only
            continue
        position = 0
        for quote in _QUOTE_RE.finditer(paragraph):
            for start, end in _chunks(paragraph, position, quote.start()):
                add(NARRATION, paragraph[start:end], index, start, end)
            words = next(group for group in quote.groups() if group is not None)
            add(QUOTE, words, index, quote.start(), quote.end())
            position = quote.end()
        for start, end in _chunks(paragraph, position, len(paragraph)):
            add(NARRATION, paragraph[start:end], index, start, end)
    return Source(kind, paragraphs, tuple(units))


def attribute_units(source: Source, speakers: dict[str, str | tuple[str, ...] | list[str]],
                    *, break_after: set[str] | None = None) -> list[Utterance]:
    """Locked utterances from the source units and the content stage's ``speakers`` (unit ID → name(s)).

    Narration and headings default to the narrator and a script line to its label; every quote
    needs a speaker (or a list/tuple of speakers when spoken in unison). A quote the narrator
    reads (a quoted word inside narration) is read in place: it joins the narration around it,
    quotation marks included.
    """
    names: list[tuple[str, ...]] = []
    missing = []
    for unit in source.units:
        raw = speakers.get(unit.id) or {QUOTE: "", LABELLED: unit.label}.get(unit.kind, NARRATOR)
        items = raw if isinstance(raw, (list, tuple)) else (raw,)
        unit_names = tuple(
            NARRATOR if str(item).strip().casefold() in NARRATOR_ALIASES else str(item).strip()
            for item in items if str(item or "").strip()
        )
        if not unit_names:
            missing.append(unit.id)
        names.append(unit_names)
    if missing:
        raise H3Error(f"No speaker for quoted unit(s) {', '.join(missing)}.")

    runs: list[list[int]] = []
    for position, unit in enumerate(source.units):
        previous = source.units[position - 1] if runs else None
        joins = (previous is not None and previous.paragraph == unit.paragraph and names[position - 1] == names[position]
                 and names[position] == (NARRATOR,) and QUOTE in (previous.kind, unit.kind)
                 and previous.id not in (break_after or set()))
        if joins:
            runs[-1].append(position)
        else:
            runs.append([position])
    utterances = []
    for run in runs:
        units = [source.units[position] for position in run]
        paragraph = source.paragraphs[units[0].paragraph]
        following = source.units[run[-1] + 1] if run[-1] + 1 < len(source.units) else None
        # Read in place up to the next unit, so punctuation after a closing quote ("“老地方”。") stays.
        end = following.start if following is not None and following.paragraph == units[0].paragraph else len(paragraph)
        text = units[0].text if len(units) == 1 else paragraph[units[0].start:end]
        text = performance_controls(text)
        kind = HEADING if units[0].kind == HEADING else LINE
        for speaker_name in names[run[0]]:
            utterances.append(Utterance(f"u{len(utterances) + 1}", speaker_name, text.strip(), kind,
                                        units[0].effects, tuple(unit.id for unit in units), units[0].hint))
    return utterances
