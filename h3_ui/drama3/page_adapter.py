"""Compatibility boundary for positional Gradio inputs, updates and status strings.

The published scripted API keeps its existing shapes; page orchestration consumes
named results here. Component visibility offsets belong only in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from h3_app.prompt_writers import drama3 as writer
from .controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_VOICE, UPLOAD_SOUND
from .page_common import SKIP, value, speech_value
VOICE_META = ("save_name", "description", "preview_text")
SOUND_META = ("save_name", "description", "duration")

@dataclass(frozen=True)
class WriterAnswer:
    outputs: list
    status: str
    failed: bool

    @classmethod
    def read(cls, outputs):
        outputs = list(outputs)
        status = str(outputs[1])
        return cls(outputs, status, status.startswith(writer.DRAMA3_PROMPT_FAILED))


@dataclass(frozen=True)
class SpeechProgress:
    output: Any
    status: str
    error: str | None

    @classmethod
    def read(cls, update, *, alignment=False):
        status = str(update[1] or "") if len(update) > 1 else ""
        prefix = "Word alignment unavailable:" if alignment else "Error:"
        error = status.removeprefix(prefix).strip().split("\n\n")[0] if status.startswith(prefix) else None
        return cls(update[0] if update else None, status, error)


@dataclass(frozen=True)
class SlotProgress:
    kind: str
    slot: int
    audio: Any = SKIP
    transcript: Any = SKIP
    status: Any = SKIP


@dataclass(frozen=True)
class CastProgress:
    slots: tuple[SlotProgress, ...]
    status: str | None = None

    @classmethod
    def read(cls, target, slot, update):
        items = [value(item) for item in update]
        if target == "voice":
            audio, transcript, status = items[:3]
            return cls((SlotProgress("voice", slot, audio, transcript, status),))
        if target == "sound":
            audio, status = items[:2]
            return cls((SlotProgress("sound", slot, audio=audio, status=status),))
        if target == "transcribe":
            transcript, status = items[:2]
            return cls((SlotProgress("voice", slot, transcript=transcript, status=status),))
        voices = items[:3 * DRAMA3_VOICE_SLOTS]
        sounds = items[3 * DRAMA3_VOICE_SLOTS + 1:3 * DRAMA3_VOICE_SLOTS + 1 + 2 * DRAMA3_SOUND_SLOTS]
        slots = [SlotProgress("voice", index, *voices[3 * index:3 * index + 3])
                 for index in range(DRAMA3_VOICE_SLOTS)]
        slots += [SlotProgress("sound", index, audio=sounds[2 * index], status=sounds[2 * index + 1])
                  for index in range(DRAMA3_SOUND_SLOTS)]
        overall = items[3 * DRAMA3_VOICE_SLOTS + 1 + 2 * DRAMA3_SOUND_SLOTS + 1:]
        return cls(tuple(slots), str(overall[0]) if overall and overall[0] is not SKIP else None)

    @property
    def error(self):
        return next((str(item.status) for item in self.slots
                     if item.status is not SKIP and "failed" in str(item.status).lower()), None)


def _clip(clips: Path, current):
    if not current:
        return None
    path = clips / Path(str(current)).name
    return str(path) if path.is_file() else None


def voice_inputs(data: dict, clip: Callable[[Any], str | None]) -> list[Any]:
    """Each voice slot's preset, clip and transcript, as the tab's speech and writer handlers take them."""
    values: list[Any] = []
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_preset") or UPLOAD_VOICE, clip(data.get(f"voice_{slot}_audio")),
                   data.get(f"voice_{slot}_transcript") or ""]
    return values


def writer_slots(data: dict, clip: Callable[[Any], str | None], setting) -> list[Any]:
    """The writer's slot inputs from a project: voices, cast details, shown slots, script mode and group."""
    values = voice_inputs(data, clip)
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_{field}") or "" for field in VOICE_META]
    return [*values, int(data.get("shown_slots") or 1), data.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC,
            setting(data, "speaker_group")]


def _apply_cast(data: dict, outputs: list[Any]) -> list[Any]:
    """Write the script, cast details, shown slots, presets and title from a writer answer into ``data``.

    ``outputs`` is the tab's writer outputs: script, status, three details per slot, the shown slot
    count with each slot's visibility and Add/Remove, a preset per slot, then the title. Returns
    what follows them.
    """
    script, rest = outputs[0], outputs[2:]
    if value(script) is not SKIP:
        data["text"] = value(script)
    details, rest = rest[:3 * DRAMA3_VOICE_SLOTS], rest[3 * DRAMA3_VOICE_SLOTS:]
    for slot in range(DRAMA3_VOICE_SLOTS):
        for index, field in enumerate(VOICE_META):
            changed = value(details[3 * slot + index])
            if changed is not SKIP:
                data[f"voice_{slot}_{field}"] = changed
    shown, rest = value(rest[0]), rest[DRAMA3_VOICE_SLOTS + 3:]
    if shown is not SKIP:
        data["shown_slots"] = int(shown)
    presets, rest = rest[:DRAMA3_VOICE_SLOTS], rest[DRAMA3_VOICE_SLOTS:]
    for slot, preset in enumerate(presets):
        if value(preset) is not SKIP:
            data[f"voice_{slot}_preset"] = value(preset)
    title, rest = value(rest[0]), rest[1:]
    if title is not SKIP:
        data["title"] = title
    return rest


def _apply_sounds(data: dict, meta: list[Any], visibility: list[Any]) -> None:
    for slot in range(DRAMA3_SOUND_SLOTS):
        for index, field in enumerate(SOUND_META):
            changed = value(meta[3 * slot + index])
            if changed is not SKIP:
                data[f"sound_{slot}_{field}"] = changed
    shown = value(visibility[0])
    if shown is not SKIP:
        data["shown_sound_slots"] = int(shown)


def generation_values(name: str, data: dict, clips: Path) -> list[Any]:
    """A saved project as ``generate_drama3``'s inputs, in the order the Speak step sends them."""
    values: list[Any] = [speech_value(data, "text"), *voice_inputs(data, lambda value: _clip(clips, value))]
    values += [speech_value(data, key) for key in ("temperature", "top_p", "top_k", "max_frames", "parallel",
                                                   "retries", "script_mode")]
    for slot in range(DRAMA3_SOUND_SLOTS):
        values += [data.get(f"sound_{slot}_preset") or UPLOAD_SOUND, _clip(clips, data.get(f"sound_{slot}_audio")),
                   data.get(f"sound_{slot}_save_name") or "", data.get(f"sound_{slot}_description") or "",
                   data.get(f"sound_{slot}_duration") or 5]
    values.append(speech_value(data, "title"))
    values += [data.get(f"voice_{slot}_save_name") or "" for slot in range(DRAMA3_VOICE_SLOTS)]
    values += [bool(speech_value(data, "word_timing")), name]
    return values


