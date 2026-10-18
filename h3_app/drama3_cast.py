"""Validate writer output before applying a script, its indexed cast, and optional sound briefs."""

from __future__ import annotations

import json
import re

from h3_app.drama3_serving.speaker_effects import strip_speaker_effects
from h3_app.drama3_serving.tree import (
    CONTAINER_TAG_RE,
    PAUSE_TAG_RE,
    SOUND_BLOCK_RE,
    _parse_sound_tag,
    extract_script_sounds,
)
from h3_app.drama3_sounds import normalize_sound_name
from h3_app.drama3_speakers import normalize_name
from h3_app.errors import H3Error
from h3_app.workflows.drama3 import MAX_SPEAKER, SPEAKER_TAG, text_speakers, validate_drama3_text


def _normalize_and_validate_script_lines(raw_script: str) -> str:
    script = validate_drama3_text(raw_script, "The writer's script")
    text = script.replace("\r\n", "\n")
    text = SOUND_BLOCK_RE.sub(r"\n\1\n", text)
    text = CONTAINER_TAG_RE.sub(r"\n\1\n", text)
    lines: list[str] = []
    has_speaker_line = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if CONTAINER_TAG_RE.fullmatch(line):
            lines.append(line)
            continue
        if PAUSE_TAG_RE.fullmatch(line):
            lines.append(line)
            continue
        if line.lower().startswith("<sound") and line.lower().endswith("</sound>"):
            if _parse_sound_tag(line) is None:
                raise H3Error("Each <sound> tag must include a non-empty sound description.")
            lines.append(line)
            continue
        match = SPEAKER_TAG.match(line)
        if not match or SPEAKER_TAG.search(line[match.end() :]):
            raise H3Error("Each script line must start with exactly one speaker tag and contain speech.")
        spoken_body = PAUSE_TAG_RE.sub("", strip_speaker_effects(line[match.end() :])).strip()
        if not spoken_body:
            raise H3Error("Each script line must start with exactly one speaker tag and contain speech.")
        has_speaker_line = True
        lines.append(line)
    if not has_speaker_line:
        raise H3Error("Each script line must start with exactly one speaker tag and contain speech.")
    return "\n".join(lines)


def parse_script_cast_and_sounds(response: str) -> tuple[str, list[dict], list[dict]]:
    """Parse and validate script, cast entries, and sound briefs from writer JSON."""
    try:
        data = json.loads(response)
    except (ValueError, TypeError) as exc:
        raise H3Error("The writer must return JSON with script and speakers.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("script"), str):
        raise H3Error("The writer returned no script string.")
    script = _normalize_and_validate_script_lines(data["script"])
    raw = data.get("speakers")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_SPEAKER + 1:
        raise H3Error("The writer must return 1–16 speakers.")
    cast, ids, names = [], set(), set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise H3Error("Each speaker must be an object.")
        speaker = entry.get("speaker_id")
        if type(speaker) is not int or not 0 <= speaker <= MAX_SPEAKER or speaker in ids:
            raise H3Error("Speaker IDs must be unique integers from 0 to 15.")
        fields = {}
        for field, limit in (("name", 64), ("description", 2000), ("preview_text", 1000)):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
                raise H3Error(f"Speaker {speaker} needs a {field} of 1–{limit} characters.")
            fields[field] = value.strip()
        fields["name"] = normalize_name(fields["name"])
        if fields["name"].casefold() in names:
            raise H3Error("A character name must map to only one speaker ID.")
        names.add(fields["name"].casefold())
        # Qwen's preview is spoken text, never Drama control syntax or stage directions.
        if any(char in fields["preview_text"] for char in "<>[]"):
            raise H3Error(f"Speaker {speaker}'s preview must contain only spoken words, without controls.")
        ids.add(speaker)
        cast.append({"speaker_id": speaker, **fields})
    if ids != text_speakers(script):
        raise H3Error("The cast's speaker IDs do not match the script's speaker tags.")

    sounds = extract_script_sounds(script)
    by_name = {item["name"].casefold(): item for item in sounds}
    raw_sounds = data.get("sounds")
    if isinstance(raw_sounds, list):
        for entry in raw_sounds:
            if not isinstance(entry, dict):
                continue
            raw_name = str(entry.get("name") or "").strip()
            raw_desc = str(entry.get("description") or "").strip()
            if not raw_name and not raw_desc:
                continue
            try:
                clean_name = normalize_sound_name(raw_name or re.sub(r"\s+", " ", raw_desc)[:32])
            except H3Error:
                continue
            duration = max(1, min(int(entry.get("duration") or (10 if entry.get("loop") else 5)), 47))
            key = clean_name.casefold()
            if key in by_name:
                if raw_desc:
                    by_name[key]["description"] = raw_desc
                if "duration" in entry:
                    by_name[key]["duration"] = duration
            elif raw_desc:
                item = {
                    "name": clean_name,
                    "description": raw_desc,
                    "loop": bool(entry.get("loop", False)),
                    "volume": float(entry.get("volume", 0.2 if entry.get("loop") else 1.0)),
                    "duration": duration,
                }
                by_name[key] = item
                sounds.append(item)
    return script, sorted(cast, key=lambda entry: entry["speaker_id"]), sounds


def parse_script_cast(response: str) -> tuple[str, list[dict]]:
    """Require one cast entry per used ID; never silently renumber an authored script."""
    script, cast, _sounds = parse_script_cast_and_sounds(response)
    return script, cast
