"""Translation styles, language rules, problem marks, and subtitle timing for Dub & Translate."""

from __future__ import annotations

from pathlib import Path
import re
import time
import uuid

from h3_app.burned_subtitles import DUB_SOURCE, DUB_TRANSLATION, Line, ass_subtitles, word_pieces
from h3_app.errors import H3Error

from . import segments as segment_tools, session as sessions, tools_worker
from .results import Player
from .timing import LINE_GAP, TEMPO_WARNING

# Translation styles for Hy-MT and Gemini; anything else typed is used as written.
TRANSLATION_STYLES = {
    "Precise": "precise and faithful to the original meaning, without embellishment",
    "Natural": "natural, everyday spoken language",
    "Formal": "formal and polite",
    "Casual": "casual and friendly",
    "Dramatic": "expressive and cinematic, like a film dub",
    "Children": "simple words and short sentences for children aged 6 to 12",
    "Teens": "lively, modern language for teenagers",
    "Older adults": "clear, respectful language for older adults",
}
DEFAULT_STYLE = "Precise"
# "Dub into" this to keep the spoken language: each line's new text starts as a copy of what was said
# (to fix a few lines) or as a rewrite for an audience. Only lines whose new text differs are voiced.
SAME_LANGUAGE = segment_tools.SAME_LANGUAGE
# Styles offered for a rewrite; anything else typed is used as written.
REWRITE_STYLES = {
    "Simpler words": "plain, simple words anyone can follow, with jargon replaced by everyday words",
    "Children": TRANSLATION_STYLES["Children"],
    "Teens": TRANSLATION_STYLES["Teens"],
    "Older adults": TRANSLATION_STYLES["Older adults"],
    "Formal": TRANSLATION_STYLES["Formal"],
}
DEFAULT_REWRITE_STYLE = "Simpler words"
# Session keys for lines shown in Review lines: their take failed its check or could not be
# generated, or Smart timing found them still too long in the last dub.
TAKE_PROBLEMS = "take_problems"
TIMING_PROBLEMS = "timing_problems"
# What a subtitled export can show, by the result player's Subtitles switch names, and how
# its progress names them.
SUBTITLE_MODES = {"translation": "translated", "source": "original", "both": "translated and original"}


def style_text(style: str) -> str:
    """The instruction for a preset or typed style; empty means the default."""
    style = str(style or "").strip() or DEFAULT_STYLE
    return TRANSLATION_STYLES.get(style, style)


def rewrite_style_text(style: str) -> str:
    """The instruction for a rewrite preset or typed style; empty means the default."""
    style = str(style or "").strip() or DEFAULT_REWRITE_STYLE
    return REWRITE_STYLES.get(style, TRANSLATION_STYLES.get(style, style))


def same_language(session: sessions.Session) -> bool:
    """Whether the session keeps its spoken language: lines are fixed or rewritten, not translated."""
    return session.data.get("target_language") == SAME_LANGUAGE


def spoken_language(session: sessions.Session) -> str:
    """The source's language: as chosen, else as transcribed; empty when unknown."""
    transcript = session.data.get("transcript") or {}
    return str(session.data.get("source_language") or transcript.get("language") or "")


def dub_language(session: sessions.Session) -> str:
    """The language the dubbed lines are in."""
    return spoken_language(session) if same_language(session) else str(session.data.get("target_language") or "")


def dub_label(session: sessions.Session) -> str:
    """How the dub is named: its language, or for the same language what was done to it."""
    if not same_language(session):
        return str(session.data.get("target_language") or "")
    spoken = spoken_language(session) or "Original"
    style = str(session.data.get("rewrite_style") or "").strip()
    return f"{spoken} · for {style}" if session.data.get("rewrite") and style else f"{spoken} · edited"


def unchanged(line: dict) -> bool:
    return line["translation"].strip() == line["source"].strip()


def apply_same_language_rule(session: sessions.Session, lines: list[dict]) -> list[dict]:
    """In a same-language session, a line whose new text changed is voiced only when it differs from
    what was said; a line whose text is unchanged keeps the Dub switch the viewer set.

    ``lines`` are the lines about to be saved; each is matched to its saved version by start, speaker
    and original text.
    """
    if not same_language(session):
        return lines
    saved = {(round(float(line["start"]), 3), line["speaker"], line["source"]): line for line in session.segments}
    slots = {(round(float(line["start"]), 3), line["speaker"]): line for line in session.segments}
    for line in lines:
        start = round(float(line["start"]), 3)
        before = saved.get((start, line["speaker"], line["source"]))
        if before is None:
            # What was said itself was corrected: a line still holding its untouched copy follows it.
            slot = slots.get((start, line["speaker"]))
            if slot is not None and unchanged(slot) and line["translation"] == slot["translation"]:
                line["translation"], line["dub"] = line["source"], False
                continue
        if before is None or before["translation"] != line["translation"]:
            line["dub"] = bool(line["translation"].strip()) and not unchanged(line)
    return lines


def manual_transcript(reply: str) -> dict:
    """The transcript in a chat model's pasted reply, read before any slow step can start."""
    if not str(reply or "").strip():
        raise H3Error("Paste the chat model's JSON reply first.")
    try:
        found, speakers = segment_tools.parse_transcript_reply(reply)
    except ValueError as exc:
        raise H3Error(f"The pasted reply could not be read: {exc}") from exc
    if not found:
        raise H3Error("The pasted reply has no timed lines.")
    # Keep observations with the source transcript so editing the dialogue cannot
    # accidentally move a face hint to a different speaker or line.
    visual_speakers = []
    for segment in found:
        hint = segment.pop("visual_speaker", None)
        if hint is not None:
            visual_speakers.append({"speaker": segment["speaker"], **hint})
    transcript = {"segments": found, "speakers": speakers}
    if visual_speakers:
        transcript["visual_speakers"] = visual_speakers
    return transcript


def moss_lines(found: list[dict]) -> list[dict]:
    """MOSS's speaker turns as subtitle-sized lines: long turns are split at sentence ends."""
    lines = []
    for turn in found:
        start, end, text = float(turn["start"]), float(turn["end"]), str(turn.get("text") or "")
        parts = ([{"start": start, "end": end, "text": text}]
                 if end - start <= tools_worker.MAX_LINE_SECONDS and len(text) <= tools_worker.MAX_LINE_CHARS
                 else tools_worker.proportional_lines(start, end, text))
        lines += [{"start": part["start"], "end": part["end"], "speaker": turn.get("speaker"), "source": part["text"]}
                  for part in parts]
    return segment_tools.normalize(lines)


def translation_issues(lines: list[dict]) -> list[int]:
    """Recheck saved text so edits clear warnings and older sessions get them too."""
    return [index for index, line in enumerate(lines)
            if line.get("dub", True) and line["translation"]
            and tools_worker.invalid_translation(line["translation"], line["source"])]


def translation_advice(lines: list[dict]) -> str:
    found = translation_issues(lines)
    if not found:
        return ""
    numbers = ", ".join(str(index + 1) for index in found[:20]) + (", …" if len(found) > 20 else "")
    return (f" Review {len(found)} translation(s) that may contain explanations: {numbers}. "
            "Edit these lines or translate them again before dubbing.")


def long_lines(lines: list[dict]) -> list[int]:
    """Lines to dub whose translation would need more than ``TEMPO_WARNING``'s speed-up to fit the line.

    Uses Drama 3's estimate of spoken length, so it can be checked before anything is voiced.
    """
    from h3_app.drama3.serving.planner import expected_seconds

    return [index for index, line in enumerate(lines)
            if line["translation"] and line.get("dub", True)
            and expected_seconds(line["translation"]) > (line["end"] - line["start"]) * TEMPO_WARNING[1]]


def length_advice(lines: list[dict]) -> str:
    """A sentence naming the translations that look too long to say in time; empty when none do."""
    found = long_lines(lines)
    if not found:
        return ""
    numbers = ", ".join(str(index + 1) for index in found[:20]) + (", …" if len(found) > 20 else "")
    return (f" {len(found)} translation(s) look too long to say in their line's time: {numbers}. Shorten them, "
            "or translate them again with Gemini, which is told how long each line lasts.")


def _take_mark(segment: dict) -> tuple[float, str, str]:
    """What a failed-take mark is filed under: the line as it was voiced, not its position,
    so adding or deleting other lines keeps it, and editing this line's start or text drops it.
    The speaker tells apart two people saying the same words at once (marks saved before it
    was added match any speaker's line)."""
    return round(float(segment["start"]), 3), str(segment.get("speaker") or ""), str(segment["translation"])


def _stored_marks(session: sessions.Session, key: str) -> dict[tuple[float, str, str], str]:
    stored = session.data.get(key)
    marks = {}
    for entry in stored if isinstance(stored, list) else ():
        try:
            marks[_take_mark(entry)] = str(entry.get("reason") or "failed")
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return marks


def _save_marks(latest: sessions.Session, key: str, marks: dict[tuple[float, str, str], str]) -> None:
    latest.data[key] = [{"start": start, "speaker": speaker, "translation": translation, "reason": reason}
                        for (start, speaker, translation), reason in marks.items()]


def _line_marks(session: sessions.Session, key: str) -> dict[int, str]:
    marks = _stored_marks(session, key)
    found = {}
    for index, line in enumerate(session.segments):
        start, _speaker, translation = mark = _take_mark(line)
        reason = marks.get(mark) or marks.get((start, "", translation))
        if line["translation"] and line.get("dub", True) and reason:
            found[index] = reason
    return found


def take_problems(session: sessions.Session) -> dict[int, str]:
    """The lines the next dub voices whose current take failed its check or could not be generated, with why."""
    return _line_marks(session, TAKE_PROBLEMS)


def timing_problems(session: sessions.Session) -> dict[int, str]:
    """The lines Smart timing found still too long in the last dub, with how fast they played."""
    return _line_marks(session, TIMING_PROBLEMS)


def kept_original(lines: list[dict], voiced: set[int]) -> list[tuple[float, float, set[str]]]:
    """The original-voice stretches (start, end, speakers) of the lines not voiced, overlaps merged,
    so speech two such lines share is restored once instead of twice."""
    merged: list[list] = []
    for line in sorted((line for index, line in enumerate(lines) if index not in voiced),
                       key=lambda line: line["start"]):
        if merged and line["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], line["end"])
            merged[-1][2].add(line["speaker"])
        else:
            merged.append([line["start"], line["end"], {line["speaker"]}])
    return [(start, end, speakers) for start, end, speakers in merged]


def subtitle_lines(lines: list[dict], spoken: dict[int, tuple[float, float]]) -> list[dict]:
    """Translated subtitles timed to the dub: each starts with its dubbed line and stays up while
    it is still being spoken, but never past the start of the next line after it, except the last.
    """
    starts = [spoken[index][0] if index in spoken else line["start"] for index, line in enumerate(lines)]
    shown = []
    for index, line in enumerate(lines):
        start, spoken_end = spoken.get(index, (line["start"], line["end"]))
        end = max(line["end"], spoken_end)
        # Another speaker talking over this line does not take its subtitle down.
        later = [starts[other] for other in range(index + 1, len(lines))
                 if lines[other]["start"] >= line["end"] - 1e-6]
        if later:
            end = min(end, max(min(line["end"], spoken_end), min(later) - LINE_GAP))
        shown.append({**line, "start": round(start, 3), "end": round(max(end, start + 0.05), 3)})
    return shown


def _filter_path(path: Path) -> str:
    """A path for an ffmpeg filter argument (drive colons and quotes escaped)."""
    return "'" + path.resolve().as_posix().replace("\\", "/").replace(":", r"\:").replace("'", r"\'") + "'"


def _language_code(language: str) -> str:
    codes = {"english": "eng", "chinese": "zho", "cantonese": "yue", "japanese": "jpn", "korean": "kor",
             "spanish": "spa", "french": "fra", "german": "deu", "portuguese": "por", "italian": "ita",
             "russian": "rus", "arabic": "ara", "indonesian": "ind", "vietnamese": "vie", "thai": "tha",
             "hindi": "hin", "turkish": "tur", "malay": "msa", "dutch": "nld"}
    return codes.get(str(language or "").strip().lower(), "und")


def _output_stem(session: sessions.Session, language: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", session.data.get("name") or "dub", flags=re.UNICODE).strip("_")[:60] or "dub"
    tag = re.sub(r"[^\w]+", "", language or "")[:16] or "dub"
    return f"{name}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"


def dub_subtitles(player: Player, mode: str, width: int, height: int) -> str | None:
    """A dub's subtitles as an ASS script for burning in, as its result player shows them: the
    translation timed to the dub, the original words timed to the source, or both, the original
    smaller under the translation. Aligned words light up as they are spoken. None with no lines."""
    def rows(field: str) -> list[Line]:
        shown = []
        for cue in player.lines:
            text = str(cue.get(field) or "")
            if not text.strip():
                continue
            # The translation follows the dub, which Smart timing may start a little earlier or later.
            start, end = ((cue.get("shown_start", cue["start"]), cue.get("shown_end", cue["end"]))
                          if field == "translation" else (cue["start"], cue["end"]))
            shown.append(Line(None, text, float(start), float(end), word_pieces(text, cue.get(f"{field}_words"))))
        return shown

    main = rows("source" if mode == "source" else "translation")
    below = rows("source") if mode == "both" else None
    if not main and not below:
        return None
    return ass_subtitles(main, width=width, height=height, row=DUB_TRANSLATION, below=below,
                         below_row=DUB_SOURCE, linger=0.0)
