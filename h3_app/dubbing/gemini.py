"""Gemini transcription, speaker labelling and translation for Dub & Translate."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from h3_app import prompt_service
from h3_app.catalog import GEMINI_API_ROOT, GEMINI_PROMPT_MODELS
from h3_app.errors import H3Error

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
    "dub. Keep each translation about as long to say as the original so it fits the same timing, and keep "
    "names, numbers and tone.{notes} Return only a JSON array with exactly {count} strings, in order.\n\n"
    "Lines:\n{lines}"
)


def _model(model: str) -> str:
    if model not in GEMINI_PROMPT_MODELS:
        raise H3Error(f"Unsupported Gemini model: {model}")
    return model


def _generate(session: requests.Session, key: str, model: str, parts: list[dict], *, json_reply: bool) -> str:
    config = {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 65536}
    if json_reply:
        config["responseMimeType"] = "application/json"
    response = session.post(
        f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"role": "user", "parts": parts}], "generationConfig": config},
        timeout=1800,
    )
    if not response.ok:
        raise prompt_service._gemini_error(response, "transcription")
    payload = response.json()
    candidates = payload.get("candidates") or []
    text = "".join(str(part.get("text", "")) for part in
                   (candidates[0].get("content", {}).get("parts", []) if candidates else [])).strip()
    if not text:
        reason = (candidates[0].get("finishReason") if candidates
                  else payload.get("promptFeedback", {}).get("blockReason", "no candidate"))
        raise H3Error(f"Gemini returned no text ({reason}).")
    return text


def transcribe(audio: Path, *, model: str, api_key: str, target_language: str, translate: bool,
               ignore_non_speech: bool, custom_prompt: str = "") -> tuple[list[dict], dict[str, str], str]:
    """Segments, speaker descriptions and the raw reply for one audio file."""
    key = prompt_service._gemini_api_key(api_key)
    prompt = str(custom_prompt or "").strip() or TRANSCRIBE_PROMPT.format(
        translate_clause=f" and translate it into {target_language} with natural, conversational wording."
        if translate else " in its original language without translating it.",
        non_speech=NON_SPEECH if ignore_non_speech else "",
        translated_clause=f" (the {target_language} translation)" if translate else " (an empty string)",
    )
    with requests.Session() as session:
        info = prompt_service._upload_gemini_file(session, Path(audio), key)
        try:
            info = prompt_service._wait_for_gemini_file(session, info, key)
            raw = _generate(session, key, _model(model), [
                {"text": prompt},
                {"fileData": {"mimeType": info.get("mimeType") or "audio/mpeg", "fileUri": info["uri"]}},
            ], json_reply=True)
        finally:
            try:
                session.delete(f"{GEMINI_API_ROOT}/v1beta/{info['name']}", headers={"x-goog-api-key": key},
                               timeout=30)
            except requests.RequestException:
                pass
    try:
        found, speakers = segment_tools.parse_transcript_reply(raw)
    except ValueError as exc:
        raise H3Error(f"Gemini's transcript could not be read: {exc}") from exc
    if not found:
        raise H3Error("Gemini found no speech in this audio.")
    return found, speakers, raw


def translate(texts: list[str], *, model: str, api_key: str, target_language: str,
              source_language: str = "", notes: str = "") -> list[str]:
    """Translate lines in order as one request, so names and tone stay consistent."""
    key = prompt_service._gemini_api_key(api_key)
    prompt = TRANSLATE_PROMPT.format(
        source=f"{source_language} " if source_language else "", target=target_language,
        notes=f" Context: {notes.strip()}" if notes.strip() else "", count=len(texts),
        lines=json.dumps(texts, ensure_ascii=False, indent=0),
    )
    with requests.Session() as session:
        raw = _generate(session, key, _model(model), [{"text": prompt}], json_reply=True)
    try:
        return segment_tools.parse_translation_reply(raw, len(texts))
    except ValueError as exc:
        raise H3Error(f"Gemini's translation could not be read: {exc}") from exc
