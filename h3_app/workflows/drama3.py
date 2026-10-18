"""Graph contract for ComfyUI-Drama3 revision 67a733a."""

import re

from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_models import (MODEL_SPECS, DRAMA3_MODEL_CHOICES, DEFAULT_DRAMA3_MODEL,
                       DRAMA3_ASR_CHOICES, DEFAULT_DRAMA3_ASR)
from h3_requirements import DRAMA3_REF

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



# The node's cap, about 6.3 minutes. A take stops at its end-of-speech token,
# so the cap only bounds a take that never ends.
MAX_FRAMES = 8192


def drama3_model_keys(model_choice=DEFAULT_DRAMA3_MODEL):
    try:
        return (DRAMA3_MODEL_CHOICES[model_choice], "drama3_codec")
    except KeyError as exc:
        raise H3Error("Choose a valid Drama 3 model.") from exc


def drama3_asr_model_keys(model_choice=DEFAULT_DRAMA3_ASR):
    try:
        return (DRAMA3_ASR_CHOICES[model_choice],)
    except KeyError as exc:
        raise H3Error("Choose a valid Qwen3-ASR model.") from exc


def speaker_transcript(speaker, transcript):
    """Lead a reference transcript with its slot's speaker tag, replacing any typed one."""
    body = SPEAKER_TAG.sub("", str(transcript), count=1) if SPEAKER_TAG.match(str(transcript).strip()) else str(transcript)
    return f"<|speaker:{int(speaker)}|> {body.strip()}"


def build_drama3_graph(*, text, voices=(), seed=0, temperature=1.0, top_p=0.9,
                       top_k=0, max_frames=MAX_FRAMES, split_long_text=False,
                       experts="in VRAM", expert_cache_gb=0.0, model_choice=DEFAULT_DRAMA3_MODEL):
    """``voices`` holds (speaker, staged audio, transcript); each clones <|speaker:N|>."""
    g = Graph()
    def add(kind, **inputs):
        return Graph.out(g.add(kind, **inputs))
    model = add("Drama3Loader", model_name=MODEL_SPECS[drama3_model_keys(model_choice)[0]].local_name,
                experts=experts, expert_cache_gb=expert_cache_gb, second_gpu="none", split_layer=0)
    codec = add("Drama3CodecLoader", codec_name=MODEL_SPECS["drama3_codec"].local_name)
    references = {}
    for index, (speaker, audio, transcript) in enumerate(voices):
        clip = add("LoadAudio", audio=audio)
        references[f"voices.voice{index}"] = add(
            "H3Drama3CachedVoice", codec=codec, audio=clip,
            transcript=speaker_transcript(speaker, transcript),
            codec_revision=MODEL_SPECS["drama3_codec"].expected_sha256,
            encoder_revision=DRAMA3_REF,
        )
    codes = add("Drama3Generate", model=model, text=text, seed=seed,
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames, split_long_text=split_long_text,
                speakers="together", **references)
    # Keep authored pauses and joint dialogue instead of inheriting upstream's
    # new per-line generation and silence-shortening defaults.
    audio = add("Drama3Decode", codec=codec, codes=codes, shorten_long_pauses=False)
    add("SaveAudioMP3", audio=audio, filename_prefix="drama3/Drama3", quality="V0")
    return g.nodes


def build_drama3_asr_graph(*, audio, language="auto", model_choice=DEFAULT_DRAMA3_ASR):
    """Transcribe one staged clip; PreviewAny returns the text in the job history."""
    g = Graph()
    model = g.add("Qwen3ASRLoader", model_name=MODEL_SPECS[drama3_asr_model_keys(model_choice)[0]].local_name)
    clip = g.add("LoadAudio", audio=audio)
    transcript = g.add("Qwen3ASRTranscribe", model=Graph.out(model), audio=Graph.out(clip),
                       language=language, context="", max_new_tokens=4096)
    preview = g.add("PreviewAny", source=Graph.out(transcript))
    return g.nodes, preview
