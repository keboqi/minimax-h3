"""Dubbing speaker reference selection, clean speech detection, and per-speaker voice/volume settings."""

from __future__ import annotations

from typing import Any

from h3_app.errors import H3Error

from . import nemotron, segments as segment_tools, session as sessions

PIPELINE_LOCAL = "Local: Qwen3-ASR + aligner + Hy-MT"
PIPELINE_MOSS = "Local: MOSS Transcribe + Diarize + Hy-MT"
PIPELINE_NEMOTRON = "Local: Nemotron Transcribe + Hy-MT"
PIPELINE_GEMINI = "Gemini (cloud)"
PIPELINE_MANUAL = "Chat model (paste its reply)"
PIPELINES = (PIPELINE_LOCAL, PIPELINE_MOSS, PIPELINE_NEMOTRON, PIPELINE_GEMINI, PIPELINE_MANUAL)
SPEAKER_PIPELINES = frozenset({PIPELINE_MOSS, PIPELINE_NEMOTRON, PIPELINE_GEMINI, PIPELINE_MANUAL})
OVERLAP_PIPELINES = frozenset({PIPELINE_NEMOTRON, PIPELINE_GEMINI, PIPELINE_MANUAL})

VOICE_LINE = "Each line's own voice"
VOICE_NEAREST = "Nearest clean voice"
VOICE_SPEAKER = "One voice per speaker"
VOICE_OVERLAP = "Separate overlapping voices"
VOICE_MODES = (VOICE_LINE, VOICE_NEAREST, VOICE_SPEAKER, VOICE_OVERLAP)

MIN_REFERENCE_SECONDS = 2.0
REFERENCE_RANGE = (3.0, 15.0)
REFERENCE_PAD = 0.08
ORIGINAL_VOICE = ""
SPEAKER_VOLUME = (0.0, 200.0)


def speaker_reference(lines: list[dict], speaker: str) -> int | None:
    """The index of the speaker's best reference line: the longest within 3–15 s, else the closest to it."""
    own = [(i, s) for i, s in enumerate(lines) if s["speaker"] == speaker and s["source"]]
    if not own:
        return None
    low, high = REFERENCE_RANGE
    fitting = [(i, s) for i, s in own if low <= s["end"] - s["start"] <= high]
    if fitting:
        return max(fitting, key=lambda item: item[1]["end"] - item[1]["start"])[0]
    return min(own, key=lambda item: min(abs(item[1]["end"] - item[1]["start"] - low),
                                         abs(item[1]["end"] - item[1]["start"] - high)))[0]


def default_voice_mode(pipeline: str | None) -> str:
    """Nemotron's speaker labels cover overlaps, so its sessions clone from clean speech by default;
    other transcribers can mislabel a line, which would clone the wrong person."""
    return VOICE_NEAREST if pipeline == PIPELINE_NEMOTRON else VOICE_LINE


def speaker_activity(session: sessions.Session) -> dict[str, list[tuple[float, float]]]:
    """When each speaker talks, for telling clean speech from cross-talk.

    Nemotron's words and diarization as transcribed, when the session has them; also every
    line as transcribed and as edited, so a moved or deleted line never hides speech that is
    still in the recording, and a line given to another speaker counts for both.
    """
    activity: dict[str, list[tuple[float, float]]] = {}
    transcript = session.data.get("transcript")
    transcript = transcript if isinstance(transcript, dict) else {}
    stored = transcript.get("activity")
    for speaker, spans in (stored.items() if isinstance(stored, dict) else ()):
        for span in spans if isinstance(spans, list) else ():
            try:
                activity.setdefault(str(speaker), []).append((float(span[0]), float(span[1])))
            except (IndexError, TypeError, ValueError):
                continue
    for hint in transcript.get("overlaps") or ():
        if not isinstance(hint, dict):
            continue
        try:
            start, end = float(hint["start"]), float(hint["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            for speaker in hint.get("speakers") or ():
                activity.setdefault(segment_tools.speaker_id(speaker), []).append((start, end))
    for line in [*(transcript.get("segments") or []), *session.segments]:
        try:
            activity.setdefault(segment_tools.speaker_id(line.get("speaker")), []).append(
                (float(line["start"]), float(line["end"])))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return activity


def overlap_references_available(session: sessions.Session) -> bool:
    transcript = session.data.get("transcript")
    return (session.data.get("pipeline") in OVERLAP_PIPELINES
            and isinstance(transcript, dict) and bool(transcript.get("segments")))


def _clean(activity: dict[str, list[tuple[float, float]]], speaker: str, start: float, end: float) -> bool:
    """No one else speaks during ``start``–``end``, padding included."""
    low, high = start - REFERENCE_PAD, end + REFERENCE_PAD
    return not any(other != speaker and span_start < high and span_end > low
                   for other, spans in activity.items() for span_start, span_end in spans)


def _native_words(session: sessions.Session, segment: dict) -> list[dict]:
    """The transcriber's own words for an unedited line (Nemotron), else nothing."""
    transcript = session.data.get("transcript")
    words = transcript.get("words") if isinstance(transcript, dict) else None
    if not isinstance(words, list):
        return []
    found = [word for word in words if isinstance(word, dict) and word.get("speaker") == segment["speaker"]
             and float(word["start"]) >= segment["start"] - 0.01 and float(word["end"]) <= segment["end"] + 0.01]
    if not found or nemotron.join_words([word["text"] for word in found]) != segment["source"].strip():
        return []
    return found


def _trimmed(session: sessions.Session, segment: dict) -> tuple[dict, str]:
    """A long clean clip cut to at most REFERENCE_RANGE's top at a word, ideally a sentence end;
    without native words the whole line is used."""
    if segment["end"] - segment["start"] <= REFERENCE_RANGE[1]:
        return segment, segment["source"]
    words = _native_words(session, segment)
    kept = [word for word in words if word["end"] - segment["start"] <= REFERENCE_RANGE[1]]
    if not kept:
        return segment, segment["source"]
    ends = [index for index, word in enumerate(kept)
            if nemotron.SENTENCE_END.search(word["text"]) and word["end"] - segment["start"] >= MIN_REFERENCE_SECONDS]
    kept = kept[:ends[-1] + 1] if ends else kept
    return ({"start": segment["start"], "end": kept[-1]["end"]},
            nemotron.join_words([word["text"] for word in kept]))


def clean_lines(lines: list[dict], activity: dict[str, list[tuple[float, float]]]) -> set[int]:
    """The lines usable as a clean reference: transcribed, at least MIN_REFERENCE_SECONDS, nobody else talking."""
    return {index for index, line in enumerate(lines)
            if line["source"] and line["end"] - line["start"] >= MIN_REFERENCE_SECONDS
            and _clean(activity, line["speaker"], line["start"], line["end"])}


def nearest_clean(lines: list[dict], index: int, clean: set[int]) -> int | None:
    """The line to clone line ``index`` from: itself when clean, else the same speaker's closest
    clean line, before or after, anywhere in the recording (ties: the earlier one); None when the
    speaker never speaks alone."""
    if index in clean:
        return index
    target = lines[index]

    def distance(line: dict) -> float:
        return max(0.0, target["start"] - line["end"], line["start"] - target["end"])

    found = [other for other in clean if lines[other]["speaker"] == target["speaker"]]
    return min(found, key=lambda other: (distance(lines[other]), lines[other]["start"])) if found else None


def speaker_settings_available(session: sessions.Session) -> bool:
    """Per-speaker settings need real speakers, which only MOSS, Nemotron and Gemini tell apart."""
    return session.data.get("pipeline") in SPEAKER_PIPELINES


def _speaker_volume(value: Any) -> float:
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return 100.0
    return min(SPEAKER_VOLUME[1], max(SPEAKER_VOLUME[0], volume)) if volume == volume else 100.0


def speaker_settings(session: sessions.Session) -> dict[str, dict]:
    """``{speaker: {"voice": key or ORIGINAL_VOICE, "volume": percent}}``; empty without real speakers."""
    stored = session.data.get("speaker_settings")
    if not speaker_settings_available(session) or not isinstance(stored, dict):
        return {}
    return {str(speaker): {"voice": str(entry.get("voice") or ORIGINAL_VOICE),
                           "volume": _speaker_volume(entry.get("volume"))}
            for speaker, entry in stored.items() if isinstance(entry, dict)}


def session_speakers(session: sessions.Session) -> list[str]:
    """The speakers of the session's lines, in order of first appearance."""
    return list(dict.fromkeys(segment["speaker"] for segment in session.segments))


def set_speaker(runtime, session_id: str, speaker: str, *, voice: str, volume: float) -> sessions.Session:
    """Save one speaker's voice and volume; the default voice at 100% clears its entry."""
    from h3_app.drama3.speakers import load_speaker

    session = sessions.load(runtime, session_id)
    if not speaker_settings_available(session):
        raise H3Error("Per-speaker voices need a session transcribed with MOSS, Nemotron or Gemini, which tell "
                      "speakers apart.")
    speaker, voice = str(speaker or "").strip(), str(voice or ORIGINAL_VOICE)
    if speaker not in session_speakers(session):
        raise H3Error("Choose one of this session's speakers.")
    if voice != ORIGINAL_VOICE:
        load_speaker(voice, runtime)  # still saved
    entry = {"voice": voice, "volume": _speaker_volume(volume)}

    def change(latest: sessions.Session) -> None:
        settings = latest.data.setdefault("speaker_settings", {})
        if entry == {"voice": ORIGINAL_VOICE, "volume": 100.0}:
            settings.pop(speaker, None)
        else:
            settings[speaker] = entry

    return sessions.modify(session, change)
