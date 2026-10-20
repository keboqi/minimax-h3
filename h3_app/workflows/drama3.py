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
        if tag is None:
            raise H3Error(f"{location} contains {match.group(0)}; Drama 3 only accepts speaker tags there.")
        digits = tag.group(1)
        if str(int(digits)) != digits or int(digits) > MAX_SPEAKER:
            raise H3Error(f"Drama 3 speaker tags are <|speaker:0|> to <|speaker:{MAX_SPEAKER}|>.")
    if len(re.findall(r"<\|", text)) != len(TOKEN.findall(text)):
        raise H3Error(f"{location} has an unterminated <| token.")
    return text


def text_speakers(text):
    """Speaker numbers the text uses; untagged text is speaker 0."""
    from h3_app.drama3_serving.tree import strip_drama3_sound_tags

    raw = str(text or "")
    speakers = {int(digits) for digits in SPEAKER_TAG.findall(raw)}
    cleaned = strip_drama3_sound_tags(raw)
    leading = SPEAKER_TAG.split(cleaned)[0]
    return speakers | ({0} if leading.strip() or not speakers else set())


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
