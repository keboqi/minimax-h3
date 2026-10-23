"""Gemini transcription, speaker labelling and translation for Dub & Translate."""

from __future__ import annotations

import json
from pathlib import Path

from h3_app.catalog import GEMINI_PROMPT_MODELS
from h3_app.errors import H3Error
from h3_app.llm import gemini

from . import segments as segment_tools

TRANSCRIBE_PROMPT = (
    "You are a professional interpreter and diarization analyst. Transcribe every line of speech in this "
    "audio{translate_clause} {non_speech}"
    "Conversations can switch speakers without a pause: detect each change by voice (pitch, timbre, accent, "
    "cadence) and never put two speakers in one segment. "
    "Return JSON only, with two keys. \"speakers\": an array in order of first appearance, each with \"id\" "
    "(speaker1, speaker2, ...) and \"description\" (at most 12 words: gender, approximate age, tone). "
    "\"segments\": an array of utterances, each with \"start\" and \"end\" (mm:ss.mmm with milliseconds), "
    "\"speaker\" (an id from speakers), \"source_text\" (the original-language words) and \"translated_text\""
    "{translated_clause}. "
    "Timestamps cut the audio for voice cloning, so they must be exact: start at the first syllable, end at "
    "the last, with no padding and no rounding to whole seconds. Keep each segment under about 12 seconds, "
    "splitting long speeches at sentence ends."
)
NON_SPEECH = ("Ignore laughter, shouts, singing, crowd noise and background voices that are not clear "
              "dialogue. ")
TRANSLATE_PROMPT = (
    "Translate each line of this {source}video dialogue into natural, spoken {target} that a voice actor can "
    "dub. {timing} Keep names, numbers and tone.{notes} Return only a JSON array with exactly {count} strings, "
    "in order.\n\nLines:\n{lines}"
)
SAME_LENGTH = "Keep each translation about as long to say as the original so it fits the same timing."
TIMED = ("Each line gives its text and the seconds it lasts on screen. The translation is dubbed into exactly "
         "that time, so it must be comfortable to say in it (about 2.5 words or 4.5 CJK characters a second); "
         "prefer shorter, natural wording over a literal translation when a line is short on time.")


def _model(model: str) -> str:
    if model not in GEMINI_PROMPT_MODELS:
        raise H3Error(f"Unsupported Gemini model: {model}")
    return model


def _generate(session: gemini.Session, model: str, parts: list[dict], *, json_reply: bool) -> str:
    config = {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 65536}
    if json_reply:
        config["responseMimeType"] = "application/json"
    payload = session.generate(model, parts, generation_config=config, timeout=1800, action="transcription")
    text = gemini.reply_text(payload)
    if not text:
        raise H3Error(f"Gemini returned no text ({gemini.empty_reply_reason(payload)}).")
    return text


def transcribe(audio: Path, *, model: str, api_key: str, target_language: str, translate: bool,
               ignore_non_speech: bool, custom_prompt: str = "") -> tuple[list[dict], dict[str, str], str]:
    """Segments, speaker descriptions and the raw reply for one audio file."""
    key = gemini.api_key(api_key)
    prompt = str(custom_prompt or "").strip() or TRANSCRIBE_PROMPT.format(
        translate_clause=f" and translate it into {target_language} with natural, conversational wording that "
                         "can be said in the same time as the original line, since it is dubbed over it."
        if translate else " in its original language without translating it.",
        non_speech=NON_SPEECH if ignore_non_speech else "",
        translated_clause=f" (the {target_language} translation)" if translate else " (an empty string)",
    )
    with gemini.Session(key) as session:
        raw = _generate(session, _model(model), [{"text": prompt}, session.file_part(Path(audio))],
                        json_reply=True)
    try:
        found, speakers = segment_tools.parse_transcript_reply(raw)
    except ValueError as exc:
        raise H3Error(f"Gemini's transcript could not be read: {exc}") from exc
    if not found:
        raise H3Error("Gemini found no speech in this audio.")
    return found, speakers, raw


def translate(texts: list[str], *, model: str, api_key: str, target_language: str,
              source_language: str = "", notes: str = "", seconds: list[float] | None = None) -> list[str]:
    """Translate lines in order as one request, so names and tone stay consistent.

    ``seconds``, one per line, is how long each line lasts; the translations are worded to fit it.
    """
    key = gemini.api_key(api_key)
    timed = seconds is not None and len(seconds) == len(texts)
    lines = ([{"text": text, "seconds": round(float(length), 1)} for text, length in zip(texts, seconds)]
             if timed else texts)
    prompt = TRANSLATE_PROMPT.format(
        source=f"{source_language} " if source_language else "", target=target_language,
        timing=TIMED if timed else SAME_LENGTH,
        notes=f" Context: {notes.strip()}" if notes.strip() else "", count=len(texts),
        lines=json.dumps(lines, ensure_ascii=False, indent=0),
    )
    with gemini.Session(key) as session:
        raw = _generate(session, _model(model), [{"text": prompt}], json_reply=True)
    try:
        return segment_tools.parse_translation_reply(raw, len(texts))
    except ValueError as exc:
        raise H3Error(f"Gemini's translation could not be read: {exc}") from exc
