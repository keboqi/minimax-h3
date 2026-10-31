"""The Guided Drama 3 writer: several short stage calls whose answers code compiles into a script.

See docs/plans/drama3-guided-writer.md. The content stage settles who says what; voices,
direction and sound then run side by side on the locked lines. The result is a writer
reply in the Single-pass JSON shape, so it goes through the same parser, lint and checks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Callable

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.model import NARRATOR, Draft, Scene, Utterance
from h3_app.drama3.guided.schema import (
    CHARACTER_EFFECTS, StageError, canonical, parse_content_brief, parse_content_preserve, parse_direction,
    parse_staging, parse_voices, scenes_from_units,
)
from h3_app.drama3.guided.segment import BRIEF, attribute_units, input_kind, segment_source
from h3_app.drama3.speakers import identity, normalize_name
from h3_app.errors import H3Error
from h3_app.prompt_writers.drama3 import (
    DRAMA3_PROMPT_FAILED as FAILED, DRAMA3_SPEAKER_ADAPT, DRAMA3_SPEAKER_SINGLE, drama3_language_instruction,
)

GUIDED_PROMPTS = {
    "preserve": "prompt_drama3_guided_preserve.txt",
    "brief": "prompt_drama3_guided_brief.txt",
    "voices": "prompt_drama3_guided_voices.txt",
    "direction": "prompt_drama3_guided_direction.txt",
    "sound": "prompt_drama3_guided_sound.txt",
}
MAX_BATCH_LINES = 40
MAX_SPEAKERS = 16

Ask = Callable[[Path, str], tuple[str, str]]


def guided_prompt_paths(prompts_dir: Path) -> dict[str, Path]:
    return {stage: prompts_dir / name for stage, name in GUIDED_PROMPTS.items()}


def slot_identities(cast_context: str) -> list[dict]:
    """The Cast's slots (speaker_id, name, description, has_reference) from the writer's cast context."""
    for line in str(cast_context or "").splitlines():
        if line.startswith("[{") or line == "[]":
            try:
                entries = json.loads(line)
            except ValueError:
                continue
            if all(isinstance(entry, dict) and "speaker_id" in entry for entry in entries):
                return entries
    return []


class _Calls:
    """Stage calls with one targeted retry; counts calls for the status."""

    def __init__(self, ask: Ask):
        self.ask = ask
        self.count = 0

    def _call(self, path: Path, request: str, stage: str) -> str:
        self.count += 1
        reply, status = self.ask(path, request)
        if str(status).startswith(FAILED):
            raise H3Error(f"The {stage} step failed: {status[len(FAILED):].strip()}")
        return reply

    def run(self, path: Path, request: str, parse, stage: str):
        reply = self._call(path, request, stage)
        try:
            return parse(reply)
        except StageError as exc:
            problems = "\n".join(f"- {problem}" for problem in exc.problems[:12])
            retry = (f"{request}\n\nYour previous reply had these problems:\n{problems}\n"
                     "Reply again with the complete corrected JSON object.")
            try:
                return parse(self._call(path, retry, stage))
            except StageError as again:
                raise H3Error(f"The {stage} step gave an invalid answer twice ({again}).") from None


def _line_listing(utterances: list[Utterance]) -> str:
    marks = {"reaction": " [reaction]", "interrupt": " [interrupt]", "heading": " [title]"}
    return "\n".join(f"{u.id} {u.speaker}{marks.get(u.kind, '')}: {u.text}" for u in utterances)


def _batches(scenes: list[Scene]) -> list[list[Scene]]:
    batches: list[list[Scene]] = [[]]
    for scene in scenes:
        if batches[-1] and sum(len(s.utterances) for s in batches[-1]) + len(scene.utterances) > MAX_BATCH_LINES:
            batches.append([])
        batches[-1].append(scene)
    return [batch for batch in batches if batch]


def _scene_listing(batch: list[Scene], by_id: dict[str, Utterance]) -> str:
    parts = []
    for scene in batch:
        parts.append(f"Scene {scene.id}. Place: {scene.place or 'unspecified'}. Change: {scene.change or 'unspecified'}.")
        parts.append(_line_listing([by_id[uid] for uid in scene.utterances]))
    return "\n".join(parts)


def _available_names(voices, speaker_mode: str) -> dict[str, int] | None:
    """For Adapt and Single, the names the cast must use, by speaker ID; None when any name goes."""
    named = {(normalize_name(name) if str(name or "").strip() else f"Voice {speaker}"): int(speaker)
             for speaker, name in voices}
    if speaker_mode == DRAMA3_SPEAKER_SINGLE:
        speaker, name = min(((int(s), n) for s, n in voices), default=(0, ""))
        return {normalize_name(name) if str(name or "").strip() else NARRATOR: speaker}
    if speaker_mode == DRAMA3_SPEAKER_ADAPT:
        if not named:
            raise H3Error("Adapt to voices needs at least one reference voice.")
        return named
    return None


def _assign_ids(names: list[str], slots: list[dict], fixed: dict[str, int] | None) -> dict[str, int]:
    """Speaker IDs by name: a fixed voice, else a slot with that name, else the lowest free ID."""
    if fixed is not None:
        lookup = {identity(name): speaker for name, speaker in fixed.items()}
        return {name: lookup[identity(name)] for name in names}
    by_name = {identity(slot["name"]): int(slot["speaker_id"]) for slot in slots if str(slot.get("name") or "").strip()}
    reserved = {int(slot["speaker_id"]) for slot in slots}
    ids: dict[str, int] = {}
    for name in names:
        if identity(name) in by_name:
            ids[name] = by_name[identity(name)]
    free = (speaker for speaker in range(MAX_SPEAKERS) if speaker not in reserved | set(ids.values()))
    for name in names:
        if name not in ids:
            try:
                ids[name] = next(free)
            except StopIteration:
                raise H3Error(f"The script has more than {MAX_SPEAKERS} speakers.") from None
    return ids


def _ordered_effects(effects) -> tuple[str, ...]:
    """Character, then place and distance, then device: the order Drama 3 chains them."""
    devices = {"phone_call", "walkie_talkie", "radio_broadcast", "megaphone", "old_gramophone", "bad_signal",
               "metallic_helmet"}
    unique = list(dict.fromkeys(effects))
    return tuple(sorted(unique, key=lambda e: 0 if e in CHARACTER_EFFECTS else 2 if e in devices else 1))[:2]


def write_drama3_guided(
    text: str,
    ask: Ask,
    prompts: dict[str, Path],
    *,
    voices=(),
    speaker_mode: str,
    advanced: bool,
    cast_context: str = "",
    sound_context: str = "",
    notes: list[str] | None = None,
) -> tuple[str, str, int]:
    """A writer reply (JSON, as the Single-pass writer returns), its source mode, and the call count.

    ``ask(system_path, request)`` returns ``(reply, status)``. Raises ``H3Error`` when a
    required stage fails twice; optional direction and sound that fail are left out with a note.
    """
    notes = notes if notes is not None else []
    calls = _Calls(ask)
    language = drama3_language_instruction(text)
    slots = slot_identities(cast_context)
    fixed = _available_names(tuple(voices), speaker_mode)
    allowed = {name.casefold() for name in fixed} if fixed is not None else None
    known = sorted({slot["name"] for slot in slots if str(slot.get("name") or "").strip()})
    voices_line = (f"Available voices (cast only these names): {', '.join(fixed)}" if fixed is not None
                   else f"Characters already in the Cast (keep these names): {', '.join(known)}" if known else "")
    kind = input_kind(text)
    roles: dict[str, str] = {}
    single = speaker_mode == DRAMA3_SPEAKER_SINGLE

    if kind == BRIEF:
        source_mode = "brief"
        request = "\n".join(filter(None, [language, voices_line, f"Script mode: {'Advanced' if advanced else 'Basic'}.",
                                          f"Brief:\n{text.strip()}"]))
        title, utterances, scenes, roles = calls.run(
            prompts["brief"], request, lambda reply: parse_content_brief(reply, allowed, notes), "writing")
    else:
        source_mode = "preserve"
        source = segment_source(text, kind)
        title = next((unit.text for unit in source.units if unit.kind == "heading"), "")
        if single:
            utterances = attribute_units(source, {unit.id: NARRATOR for unit in source.units if unit.kind == "quote"})
            scenes = scenes_from_units(utterances, [])
        else:
            listing = "\n".join(
                f"{unit.id} [{unit.kind}{', label ' + unit.label if unit.label else ''}] {unit.text}"
                for unit in source.units if unit.kind != "heading")
            request = "\n".join(filter(None, [language, voices_line, f"Units:\n{listing}"]))
            answer = calls.run(prompts["preserve"], request,
                               lambda reply: parse_content_preserve(reply, source, allowed), "casting")
            title = title or answer["title"]
            utterances = attribute_units(source, answer["speakers"])
            scenes = scenes_from_units(utterances, answer["scenes"])
    if fixed is not None and not single and NARRATOR not in {identity(name) for name in fixed}             and any(u.speaker == NARRATOR for u in utterances):
        # Adapt to voices: narration goes to the first voice rather than a speaker with no voice.
        reader = min(fixed, key=fixed.get)
        notes.append(f"The narration is read by {reader}, as no voice is named narrator.")
        utterances = [Utterance(u.id, reader, u.text, u.kind, u.effects, u.sources) if u.speaker == NARRATOR else u
                      for u in utterances]
    if single:
        name = next(iter(fixed))
        utterances = [Utterance(u.id, name, u.text, u.kind if u.kind == "heading" else "line", u.effects, u.sources)
                      for u in utterances]

    names = list(dict.fromkeys(u.speaker for u in utterances))
    ids = _assign_ids(names, slots, fixed)
    by_id = {u.id: u for u in utterances}

    def voices_stage():
        listing = []
        for name in names:
            lines = " / ".join(u.text for u in utterances if u.speaker == name)[:300]
            role = roles.get(name) or ("the story's narrator" if name == NARRATOR else "")
            listing.append(f"- {name}{f' ({role})' if role else ''}: {lines}")
        request = "\n".join([language, "Characters:", *listing])
        return calls.run(prompts["voices"], request, lambda reply: parse_voices(reply, names), "voice design")

    def direction_stage(batch):
        cast = ", ".join(f"{name} ({roles[name]})" if roles.get(name) else name for name in names)
        request = "\n".join([language, f"Cast: {cast}", _scene_listing(batch, by_id)])
        ids_in_batch = {uid for scene in batch for uid in scene.utterances}
        try:
            return calls.run(prompts["direction"], request,
                             lambda reply: parse_direction(reply, ids_in_batch, notes), "direction")
        except H3Error as exc:
            notes.append(f"{exc} Those lines are left undirected.")
            return {}

    def sound_stage(batch):
        request = "\n".join(filter(None, [language, sound_context, _scene_listing(batch, by_id)]))
        scene_ids = {scene.id for scene in batch}
        ids_in_batch = {uid for scene in batch for uid in scene.utterances}
        try:
            return calls.run(prompts["sound"], request,
                             lambda reply: parse_staging(reply, scene_ids, ids_in_batch, notes), "sound design")
        except H3Error as exc:
            notes.append(f"{exc} Those scenes play without added sound.")
            return {}, {}, {}

    batches = _batches(scenes)
    with ThreadPoolExecutor(max_workers=4) as pool:
        voices_future = pool.submit(voices_stage)
        direction_futures = [pool.submit(direction_stage, batch) for batch in batches]
        sound_futures = [pool.submit(sound_stage, batch) for batch in batches] if advanced else []
        designed = voices_future.result()
        directions = {key: value for future in direction_futures for key, value in future.result().items()}
        sounds, staging, effects = {}, {}, {}
        for future in sound_futures:
            batch_sounds, batch_staging, batch_effects = future.result()
            for key, sound in batch_sounds.items():
                sounds.setdefault(key, sound)
            staging.update(batch_staging)
            effects.update(batch_effects)

    # A character effect is who the character is, so it goes on every line they speak.
    character = {}
    for uid, chosen in effects.items():
        for effect in chosen:
            if effect in CHARACTER_EFFECTS:
                character.setdefault(by_id[uid].speaker, effect)
    utterances = [Utterance(u.id, u.speaker, u.text, u.kind,
                            _ordered_effects([*u.effects, *effects.get(u.id, ()),
                                              *([character[u.speaker]] if u.speaker in character else [])]),
                            u.sources) if u.speaker != NARRATOR else u for u in utterances]

    draft = Draft(utterances, scenes, title, directions, sounds, staging)
    script = compile_script(draft, advanced=advanced, notes=notes)
    slot_by_id = {int(slot["speaker_id"]): slot for slot in slots}
    speakers = []
    for name in names:
        description, preview = designed[name]
        slot = slot_by_id.get(ids[name], {})
        speakers.append({"speaker_id": ids[name], "name": name,
                         "description": str(slot.get("description") or "").strip() or description,
                         "preview_text": preview})
    reply = json.dumps({"title": " ".join(str(title or "").split())[:60], "source_mode": source_mode, "script": script,
                        "speakers": speakers}, ensure_ascii=False)
    return reply, source_mode, calls.count
