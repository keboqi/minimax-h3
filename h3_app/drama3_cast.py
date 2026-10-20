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
    sound_placement_attrs,
)
from h3_app.drama3_sounds import normalize_sound_name
from h3_app.drama3_speakers import normalize_name
from h3_app.errors import H3Error
from h3_app.workflows.drama3 import MAX_SPEAKER, SPEAKER_TAG, names_to_tags, text_speakers, validate_drama3_text

SPEAKER_LINE_ERROR = ("Each script line must start with one speaker's name from the speakers list, "
                      'as in "Mia: Hello.", and contain speech.')


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
            raise H3Error(SPEAKER_LINE_ERROR)
        spoken_body = PAUSE_TAG_RE.sub("", strip_speaker_effects(line[match.end() :])).strip()
        if not spoken_body:
            raise H3Error(SPEAKER_LINE_ERROR)
        has_speaker_line = True
        lines.append(line)
    if not has_speaker_line:
        raise H3Error(SPEAKER_LINE_ERROR)
    return "\n".join(lines)


def parse_script_cast_and_sounds(response: str) -> tuple[str, list[dict], list[dict]]:
    """Parse and validate script, cast entries, and sound briefs from writer JSON.

    The script names its speakers ("Mia: Hello."), as the speakers list does; the
    returned script uses Drama 3 speaker tags. Tagged lines are accepted too.
    """
    try:
        data = json.loads(response)
    except (ValueError, TypeError) as exc:
        raise H3Error("The writer must return JSON with script and speakers.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("script"), str):
        raise H3Error("The writer returned no script string.")
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
    script = _normalize_and_validate_script_lines(
        names_to_tags(data["script"], {entry["speaker_id"]: entry["name"] for entry in cast}))
    if ids != text_speakers(script):
        raise H3Error("The speakers list does not match the speakers the script names.")

    sounds = extract_script_sounds(script)
    by_name = {item["name"].casefold(): item for item in sounds}
    briefs: dict[str, dict] = {}
    raw_sounds = data.get("sounds")
    if isinstance(raw_sounds, list):
        for entry in raw_sounds:
            if not isinstance(entry, dict):
                continue
            raw_name = str(entry.get("name") or "").strip()
            # Angle brackets would end the rewritten <sound> tag early.
            raw_desc = " ".join(re.sub(r"[<>]", " ", str(entry.get("description") or "")).split())
            if not raw_name and not raw_desc:
                continue
            try:
                clean_name = normalize_sound_name(raw_name or raw_desc[:32])
            except H3Error:
                continue
            loop = entry.get("loop") is True or str(entry.get("loop", "")).lower() in ("true", "1", "yes")
            duration = _clamped_number(entry.get("duration"), 10 if loop else 5, 1, 47, integer=True)
            key = clean_name.casefold()
            if key in by_name:
                if raw_desc:
                    by_name[key]["description"] = raw_desc
                    briefs.setdefault(key, {})["description"] = raw_desc
                if _clamped_number(entry.get("duration"), None, 1, 47) is not None:
                    by_name[key]["duration"] = duration
                    briefs.setdefault(key, {})["duration"] = duration
            elif raw_desc:
                item = {
                    "name": clean_name,
                    "description": raw_desc,
                    "loop": loop,
                    "volume": _clamped_number(entry.get("volume"), 0.2 if loop else 1.0, 0.0, 2.0),
                    "duration": duration,
                }
                by_name[key] = item
                sounds.append(item)
    if briefs:
        # The script is what the sound slots and the renderer read, so carry the briefs into its tags.
        script = SOUND_BLOCK_RE.sub(lambda match: _apply_sound_brief(match.group(1), briefs), script)
    return script, sorted(cast, key=lambda entry: entry["speaker_id"]), sounds


def _clamped_number(value, default, low, high, *, integer=False):
    """A writer-supplied number within [low, high]; unparsable or missing values use ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if number != number:  # NaN
        if default is None:
            return None
        number = float(default)
    number = max(low, min(number, high))
    return int(round(number)) if integer else number


def _apply_sound_brief(tag: str, briefs: dict[str, dict]) -> str:
    node = _parse_sound_tag(tag)
    brief = briefs.get(node.name.casefold()) if node is not None else None
    if not brief:
        return tag
    attrs = [f'name="{node.name.replace(chr(34), chr(39))}"']
    if node.loop:
        attrs.append('loop="true"')
    if node.volume != 1.0:
        attrs.append(f'volume="{node.volume:g}"')
    attrs.append(f'duration="{brief.get("duration", node.duration)}"')
    attrs.extend(sound_placement_attrs(node))
    return f"<sound {' '.join(attrs)}>{brief.get('description') or node.description}</sound>"


MAX_TITLE_LENGTH = 100


def clean_title(value) -> str:
    """A one-line plain title without control characters or tag syntax; "" when absent."""
    if not isinstance(value, str):
        return ""
    title = " ".join(re.sub(r"[\x00-\x1f<>\[\]{}|]", " ", value).split())
    return title[:MAX_TITLE_LENGTH].rstrip()


def parse_writer_title(response: str) -> str:
    """The optional writer title; a reply without one (or not JSON) has none."""
    try:
        data = json.loads(response)
    except (ValueError, TypeError):
        return ""
    return clean_title(data.get("title")) if isinstance(data, dict) else ""


def parse_writer_source_mode(response: str) -> str:
    """The writer's reading of the source, "preserve" or "brief"; "" when it gave neither."""
    try:
        data = json.loads(response)
    except (ValueError, TypeError):
        return ""
    mode = data.get("source_mode") if isinstance(data, dict) else None
    mode = mode.strip().lower() if isinstance(mode, str) else ""
    return mode if mode in ("preserve", "brief") else ""


def parse_script_cast(response: str) -> tuple[str, list[dict]]:
    """Require one cast entry per used ID; never silently renumber an authored script."""
    script, cast, _sounds = parse_script_cast_and_sounds(response)
    return script, cast
