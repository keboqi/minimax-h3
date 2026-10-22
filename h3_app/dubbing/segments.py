"""Pure segment helpers: timestamps, model replies, subtitles and the editor table.

A segment is ``{"start", "end", "speaker", "source", "translation", "dub"}`` with
times in seconds. ``dub`` False keeps the original voice for that line.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

TABLE_HEADERS = ["#", "Start", "End", "Speaker", "Original", "Translation", "Dub"]
TABLE_TYPES = ["number", "str", "str", "str", "str", "str", "bool"]
DEFAULT_SPEAKER = "speaker1"
MIN_SEGMENT_SECONDS = 0.05


def parse_time(value: Any) -> float | None:
    """Seconds from ``73.5``, ``"01:13.500"``, ``"0:01:13,5"`` or ``"73.5s"``; None when unreadable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip().lower().removesuffix("s").replace(",", ".")
    if not text:
        return None
    try:
        total = 0.0
        for part in text.split(":"):
            total = total * 60 + float(part.strip() or 0)
        return max(0.0, total)
    except ValueError:
        return None


def format_time(seconds: float, *, hours: bool = False) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    h, rest = divmod(milliseconds, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    if hours or h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    h, rest = divmod(milliseconds, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def speaker_id(value: Any, fallback: int = 1) -> str:
    """``"Speaker 2"``, ``2`` or ``"spk_2"`` → ``speaker2``; free names are kept, trimmed."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1:
        return f"speaker{int(value)}"
    text = str(value or "").strip()
    if not text:
        return f"speaker{max(1, fallback)}"
    match = re.fullmatch(r"(?:speaker|spk|s)[\s_-]*(\d+)", text, re.IGNORECASE)
    return f"speaker{int(match.group(1))}" if match else text[:40]


def normalize(segments: Iterable[dict]) -> list[dict]:
    """Sorted, valid segments with consistent keys; unreadable or empty rows are dropped."""
    cleaned = []
    for raw in segments:
        if not isinstance(raw, dict):
            continue
        start = parse_time(raw.get("start", raw.get("start_time")))
        end = parse_time(raw.get("end", raw.get("end_time")))
        source = str(raw.get("source", raw.get("source_text", raw.get("text", ""))) or "").strip()
        translation = str(raw.get("translation", raw.get("translated_text", "")) or "").strip()
        if start is None or end is None or not (source or translation):
            continue
        cleaned.append({
            "start": round(start, 3),
            "end": round(max(end, start + MIN_SEGMENT_SECONDS), 3),
            "speaker": speaker_id(raw.get("speaker")),
            "source": source,
            "translation": translation,
            "dub": bool(raw.get("dub", True)),
        })
    cleaned.sort(key=lambda segment: (segment["start"], segment["end"]))
    return cleaned


# ---------------------------------------------------------------------------
# Model replies
# ---------------------------------------------------------------------------


def _json_payload(text: str) -> Any:
    text = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # A reply that wraps the JSON in prose: take the outermost object or array.
    pairs = [(text.find(opener), opener, closer) for opener, closer in (("{", "}"), ("[", "]"))]
    for first, _opener, closer in sorted(pair for pair in pairs if pair[0] != -1):
        last = text.rfind(closer)
        if last > first:
            try:
                return json.loads(text[first:last + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("The reply is not valid JSON.")


def parse_transcript_reply(text: str) -> tuple[list[dict], dict[str, str]]:
    """Segments and ``{speaker: description}`` from a transcription reply.

    Accepts ``{"speakers": [...], "segments": [...]}`` or a bare segment array.
    """
    data = _json_payload(text)
    speakers: dict[str, str] = {}
    if isinstance(data, dict):
        for index, entry in enumerate(data.get("speakers") or [], 1):
            entry = entry if isinstance(entry, dict) else {"description": str(entry)}
            key = speaker_id(entry.get("id") or entry.get("speaker") or entry.get("name"), index)
            speakers[key] = str(entry.get("description") or "").strip()
        data = data.get("segments")
    if not isinstance(data, list):
        raise ValueError("The reply has no segments array.")
    return normalize(data), speakers


def parse_translation_reply(text: str, count: int) -> list[str]:
    """``count`` translations from a JSON array reply (strings, or objects with a text field)."""
    data = _json_payload(text)
    if isinstance(data, dict):
        data = data.get("translations") or data.get("lines") or data.get("items")
    if not isinstance(data, list):
        raise ValueError("The reply is not a JSON array.")
    texts = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("translation") or item.get("text") or item.get("translated_text") or ""
        texts.append(str(item or "").strip())
    if len(texts) != count:
        raise ValueError(f"Expected {count} translations, got {len(texts)}.")
    return texts


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def to_srt(segments: Iterable[dict], field: str) -> str:
    blocks = []
    for segment in segments:
        text = str(segment.get(field) or "").strip()
        if text:
            blocks.append(f"{len(blocks) + 1}\n{srt_time(segment['start'])} --> {srt_time(segment['end'])}\n{text}\n")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Editor table
# ---------------------------------------------------------------------------


def to_rows(segments: Iterable[dict]) -> list[list]:
    return [[index, format_time(s["start"]), format_time(s["end"]), s["speaker"], s["source"],
             s["translation"], bool(s.get("dub", True))] for index, s in enumerate(segments, 1)]


def _cell(row, index, default=""):
    try:
        value = row[index]
    except (IndexError, KeyError, TypeError):
        return default
    return default if value is None or value != value else value  # NaN from pandas → default


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def from_rows(rows: Any) -> list[dict]:
    """Segments from the editor's rows (lists, or a pandas DataFrame), keeping the user's edits."""
    if hasattr(rows, "values") and hasattr(rows, "columns"):
        rows = rows.values.tolist()
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    segments = []
    for row in rows or []:
        start, end = parse_time(_cell(row, 1)), parse_time(_cell(row, 2))
        source, translation = str(_cell(row, 4)).strip(), str(_cell(row, 5)).strip()
        if start is None or end is None or not (source or translation):
            continue
        segments.append({"start": start, "end": end, "speaker": _cell(row, 3) or DEFAULT_SPEAKER,
                         "source": source, "translation": translation, "dub": _truthy(_cell(row, 6, True))})
    return normalize(segments)


def parse_line_numbers(text: str, count: int) -> list[int]:
    """Zero-based indices from ``"3, 5-7"`` (one-based, as the table shows them); empty means none."""
    indices: set[int] = set()
    for part in re.split(r"[,\s]+", str(text or "").strip()):
        if not part:
            continue
        first, _, last = part.partition("-")
        try:
            low, high = int(first), int(last or first)
        except ValueError as exc:
            raise ValueError(f"{part!r} is not a line number or range.") from exc
        indices.update(range(min(low, high) - 1, max(low, high)))
    bad = sorted(index + 1 for index in indices if not 0 <= index < count)
    if bad:
        raise ValueError(f"No line {bad[0]}; the table has {count} line(s).")
    return sorted(indices)
