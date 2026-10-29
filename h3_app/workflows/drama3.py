"""Drama 3 script validation and the Qwen3-ASR transcription graph (ComfyUI-Drama3 revision 67a733a)."""

import re

from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_models import MODEL_SPECS, DRAMA3_ASR_CHOICES, DEFAULT_DRAMA3_ASR

MAX_SPEAKER = 15
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
TOKEN = re.compile(r"<\|[^|>]*\|>")


def validate_drama3_text(text, location="Speech text"):
    """Mirror the node pack's check_text: only <|speaker:0..15|> structural tokens."""
    text = str(text or "").strip()
    if not text:
        raise H3Error(f"{location} is empty.")
    for match in TOKEN.finditer(text):
        tag = SPEAKER_TAG.fullmatch(match.group(0))
        if tag is None and match.group(0).startswith("<|phoneme_"):
            raise H3Error(f"{location} contains {match.group(0)}; write a pronunciation as pinyin after the "
                          "word instead, e.g. 银行{yin2 hang2}.")
        if tag is None:
            raise H3Error(f"{location} contains {match.group(0)}; Drama 3 only accepts speaker tags there.")
        digits = tag.group(1)
        if str(int(digits)) != digits or int(digits) > MAX_SPEAKER:
            raise H3Error(f"Drama 3 speaker tags are <|speaker:0|> to <|speaker:{MAX_SPEAKER}|>.")
    if len(re.findall(r"<\|", text)) != len(TOKEN.findall(text)):
        raise H3Error(f"{location} has an unterminated <| token.")
    from h3_app.drama3.phonemes import note_problems

    problems = note_problems(text)
    if problems:
        raise H3Error(f"{location}: {problems[0]}")
    return text


def text_speakers(text):
    """Speaker numbers the text uses; untagged text is speaker 0."""
    from h3_app.drama3.serving.tree import strip_drama3_sound_tags

    raw = str(text or "")
    speakers = {int(digits) for digits in SPEAKER_TAG.findall(raw)}
    cleaned = strip_drama3_sound_tags(raw)
    leading = SPEAKER_TAG.split(cleaned)[0]
    return speakers | ({0} if leading.strip() or not speakers else set())


# Scripts are edited and written as "Name: line"; Drama 3 itself reads "<|speaker:N|> line".
# A name is a Cast slot's name or "Speaker N"; a full-width colon works too.
NAME_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<name>[^\s:：<>\[\]|#][^:：<>\[\]|\n]{0,63}?)[ \t]*[:：][ \t]*(?P<body>.*)$")
_GENERIC_NAME = re.compile(r"speaker\s*(\d+)")
_TAG_LINE = re.compile(r"^(?P<indent>[ \t]*)<\|speaker:(?P<id>[0-9]+)\|>[ \t]*", re.MULTILINE)


def _name_key(name):
    return " ".join(str(name or "").split()).casefold()


def speaker_label(speaker, names):
    """The name a script line shows for ``speaker``: its Cast name, else "Speaker N"."""
    name = " ".join(str((names or {}).get(speaker) or "").split())
    # A name the line syntax cannot carry back (a colon, or markup) keeps the generic label.
    parsed = NAME_LINE.fullmatch(f"{name}: x") if name else None
    return name if parsed and parsed["name"] == name else f"Speaker {speaker}"


def _speaker_lookup(names):
    lookup = {}
    for speaker, name in sorted((names or {}).items()):
        key = _name_key(name)
        if not key:
            continue
        speaker = int(speaker)
        if key in lookup:
            raise H3Error(f'Speaker {lookup[key]} and Speaker {speaker} share the name "{name}". '
                          "Give each Cast slot a unique name.")
        generic = _GENERIC_NAME.fullmatch(key)
        if generic and int(generic[1]) <= MAX_SPEAKER and int(generic[1]) != speaker:
            raise H3Error(f'The name "{name}" is reserved for Speaker {int(generic[1])}. '
                          f"Give Speaker {speaker} a different Cast name.")
        lookup[key] = speaker
    return lookup


def _named_speaker(match, lookup):
    key = _name_key(match["name"])
    if key in lookup:
        return lookup[key]
    generic = _GENERIC_NAME.fullmatch(key)
    return int(generic[1]) if generic and int(generic[1]) <= MAX_SPEAKER else None


def resolve_speaker(name, names):
    """The speaker label a line's ``name`` refers to, or None when it names no Cast slot.

    ``names`` maps Cast slots to names; "Speaker N" also refers to slot N.
    """
    match = NAME_LINE.fullmatch(f"{' '.join(str(name or '').split())}: x")
    speaker = _named_speaker(match, _speaker_lookup(names)) if match else None
    return None if speaker is None else speaker_label(speaker, names)


def request_cast(request):
    """A request's Cast: names by slot, the default speaker's label, and the name resolver.

    Text before any speaker line belongs to the lowest slot with a voice.
    """
    names = {int(speaker): name for speaker, name in getattr(request, "voice_names", ()) or ()}
    _speaker_lookup(names)  # reject ambiguous labels before building name-keyed references
    voiced = sorted(int(speaker) for speaker, _audio, _transcript in request.voices)
    return names, speaker_label(voiced[0] if voiced else 0, names), lambda name: resolve_speaker(name, names)


def names_to_tags(script, names):
    """``Name: line`` becomes ``<|speaker:N|> line`` for Cast names and "Speaker N".

    Other lines, including ones already tagged, are left as they are.
    """
    lookup = _speaker_lookup(names)
    lines = []
    for line in str(script or "").replace("\r\n", "\n").split("\n"):
        match = NAME_LINE.match(line)
        speaker = _named_speaker(match, lookup) if match else None
        lines.append(line if speaker is None else f"{match['indent']}<|speaker:{speaker}|> {match['body']}".rstrip())
    return "\n".join(lines)


def tags_to_names(script, names):
    """``<|speaker:N|> line`` at a line start becomes ``Name: line``, for reading and editing."""
    return _TAG_LINE.sub(lambda match: f"{match['indent']}{speaker_label(int(match['id']), names)}: ",
                         str(script or ""))


_BLOCK = re.compile(r"(<sound\b[^>]*>.*?</sound\s*>|</?(?:parallel|sequential)\b[^<>]*>)",
                    re.IGNORECASE | re.DOTALL)
_OPEN_CONTAINER = re.compile(r"<(?:parallel|sequential)\b[^<>]*>", re.IGNORECASE)
_CLOSE_CONTAINER = re.compile(r"</(?:parallel|sequential)\s*>", re.IGNORECASE)


def format_script(script, indent="  "):
    """One turn, sound or staging tag per line, indented by its <parallel>/<sequential> depth.

    The parser already reads sounds and staging tags as their own lines and ignores
    indentation and blank lines, so the formatted script performs the same.
    """
    lines, depth = [], 0
    for index, piece in enumerate(_BLOCK.split(str(script or "").replace("\r\n", "\n"))):
        if index % 2:  # a captured sound or staging tag
            if _CLOSE_CONTAINER.fullmatch(piece):
                depth = max(depth - 1, 0)
            lines.append(indent * depth + " ".join(piece.split()))
            if _OPEN_CONTAINER.fullmatch(piece):
                depth += 1
            continue
        lines.extend(indent * depth + line.strip() for line in piece.split("\n") if line.strip())
    return "\n".join(lines)


def unknown_speaker_names(script, names):
    """Speaker-like ``Name:`` prefixes matching no Cast name, in a script that names its speakers.

    Prose with no named lines is not checked, so a heading such as "Chapter 1: Rain" in
    plain text is still spoken.
    """
    lookup = _speaker_lookup(names)
    matches = [match for match in map(NAME_LINE.match, str(script or "").split("\n")) if match]
    if not any(_named_speaker(match, lookup) is not None for match in matches):
        return []
    unknown = (" ".join(match["name"].split()) for match in matches if _named_speaker(match, lookup) is None)
    return list(dict.fromkeys(name for name in unknown if len(name.split()) <= 3))


# About 6.3 minutes of codec frames. A take stops at its end-of-speech token,
# so the cap only bounds a take that never ends.
MAX_FRAMES = 8192


def drama3_asr_model_keys(model_choice=DEFAULT_DRAMA3_ASR):
    try:
        return (DRAMA3_ASR_CHOICES[model_choice],)
    except KeyError as exc:
        raise H3Error("Choose a valid Qwen3-ASR model.") from exc


def build_drama3_asr_graph(*, audio, language="auto", model_choice=DEFAULT_DRAMA3_ASR):
    """Transcribe one staged clip; PreviewAny returns the text in the job history."""
    g = Graph()
    model = g.add("Qwen3ASRLoader", model_name=MODEL_SPECS[drama3_asr_model_keys(model_choice)[0]].local_name)
    clip = g.add("LoadAudio", audio=audio)
    transcript = g.add("Qwen3ASRTranscribe", model=Graph.out(model), audio=Graph.out(clip),
                       language=language, context="", max_new_tokens=4096)
    preview = g.add("PreviewAny", source=Graph.out(transcript))
    return g.nodes, preview
