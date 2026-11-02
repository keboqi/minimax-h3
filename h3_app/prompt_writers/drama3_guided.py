"""The Guided Drama 3 writer: several short stage calls whose answers code compiles into a script.

See docs/plans/drama3-guided-writer.md. The content stage settles who says what; voices,
direction and sound then run side by side on the locked lines. The result is a writer
reply in the Single-pass JSON shape, so it goes through the same parser, lint and checks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import replace
from pathlib import Path
import re
from typing import Callable

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.batching import continuation_staging, merge_staging, prepare_batches, scene_batches
from h3_app.drama3.guided.model import NARRATOR, Draft, Scene, Utterance
from h3_app.drama3.guided.enhance import apply_content_annotations
from h3_app.drama3.guided.fidelity import words
from h3_app.drama3.guided.local_staging import localize_existing_cues
from h3_app.drama3.guided.schema import (
    CHARACTER_EFFECTS, StageError, canonical, parse_content_brief, parse_content_preserve, parse_direction,
    parse_sound_palette, parse_staging, parse_voices, scenes_from_units,
)
from h3_app.drama3.guided.segment import (
    BRIEF, attribute_units, input_kind, is_control_line, replace_speaker_tag_line, restore_structure, segment_source,
    source_request,
)
from h3_app.drama3.speakers import identity, normalize_name
from h3_app.errors import H3Error
from h3_app.prompt_writers.drama3 import (
    DRAMA3_PROMPT_FAILED as FAILED, DRAMA3_SPEAKER_ADAPT, DRAMA3_SPEAKER_SINGLE, drama3_language_instruction,
)

GUIDED_PROMPTS = {
    "plan": "prompt_drama3_guided_plan.txt",
    "preserve": "prompt_drama3_guided_preserve.txt",
    "brief": "prompt_drama3_guided_brief.txt",
    "voices": "prompt_drama3_guided_voices.txt",
    "direction": "prompt_drama3_guided_direction.txt",
    "sound": "prompt_drama3_guided_sound.txt",
}
MAX_BATCH_LINES = 40
MIN_BATCH_LINES = 10
MAX_BATCH_LINES_LIMIT = 100
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


def reusable_voices(cast_context: str) -> list[dict]:
    """Saved speaker presets (name, description) listed in the writer's cast context."""
    for line in str(cast_context or "").splitlines():
        if line.startswith("[{"):
            try:
                entries = json.loads(line)
            except ValueError:
                continue
            if all(isinstance(entry, dict) and "name" in entry and "speaker_id" not in entry for entry in entries):
                return [entry for entry in entries if str(entry.get("name") or "").strip()]
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
    return "\n".join(
        f"{u.id} {u.speaker}{marks.get(u.kind, '')}{f' (cue hint: {u.hint})' if u.hint else ''}: {u.text}"
        for u in utterances
    )


def _batches(scenes: list[Scene], max_batch_lines: int = MAX_BATCH_LINES) -> list[list[Scene]]:
    limit = max(MIN_BATCH_LINES, min(MAX_BATCH_LINES_LIMIT, int(max_batch_lines or MAX_BATCH_LINES)))
    batches: list[list[Scene]] = [[]]
    for scene in scenes:
        if batches[-1] and sum(len(s.utterances) for s in batches[-1]) + len(scene.utterances) > limit:
            batches.append([])
        batches[-1].append(scene)
    return [batch for batch in batches if batch]


def _scene_listing(batch: list[Scene], by_id: dict[str, Utterance]) -> str:
    parts = []
    for scene in batch:
        parts.append(f"Scene {scene.id}. Place: {scene.place or 'unspecified'}. Change: {scene.change or 'unspecified'}.")
        if scene.beats:
            parts.append("Beats: " + "; ".join(
                f'at "{at}"' + (f" ({focus})" if focus else "") + (f": {intent}" if intent else "")
                for at, focus, intent in scene.beats))
        parts.append(_line_listing([by_id[uid] for uid in scene.utterances]))
    return "\n".join(parts)


def _available_names(voices, speaker_mode: str, slots: list[dict] = ()) -> dict[str, int] | None:
    """For Adapt and Single, the names the cast must use, by speaker ID; None when any name goes."""
    if speaker_mode == DRAMA3_SPEAKER_SINGLE:
        speaker, name = min(((int(s), n) for s, n in voices), default=(0, ""))
        if not str(name or "").strip():
            name = next((slot.get("name", "") for slot in slots if int(slot["speaker_id"]) == speaker), "")
        return {normalize_name(name) if str(name or "").strip() else NARRATOR: speaker}
    if speaker_mode == DRAMA3_SPEAKER_ADAPT:
        if not voices:
            raise H3Error("Adapt to voices needs at least one reference voice.")
        named: dict[str, int] = {}
        seen: set[str] = set()
        for speaker, raw_name in voices:
            base = normalize_name(raw_name) if str(raw_name or "").strip() else f"Voice {speaker}"
            candidate = base if identity(base) not in seen else f"{base} ({speaker})"
            seen.add(identity(candidate))
            named[candidate] = int(speaker)
        return named
    return None


def _assign_ids(names: list[str], slots: list[dict], fixed: dict[str, int] | None) -> dict[str, int]:
    """Speaker IDs by name: a fixed voice, else a slot with that name, else the lowest free ID."""
    if fixed is not None:
        lookup = {identity(name): speaker for name, speaker in fixed.items()}
        return {name: lookup[identity(name)] for name in names}
    by_name = {identity(slot["name"]): int(slot["speaker_id"]) for slot in slots if str(slot.get("name") or "").strip()}
    reserved = {int(slot["speaker_id"]) for slot in slots}
    ids: dict[str, int] = {name: by_name[identity(name)] for name in names if identity(name) in by_name}
    for name in names:
        # Named slots go first, so "Speaker 2" never takes the ID of a slot named in this cast.
        if name not in ids and (match := re.fullmatch(r"speaker (\d+)", name.strip(), re.IGNORECASE)):
            speaker_id = int(match.group(1))
            if 0 <= speaker_id < MAX_SPEAKERS and speaker_id not in set(ids.values()):
                ids[name] = speaker_id
    free = (speaker for speaker in range(MAX_SPEAKERS) if speaker not in reserved | set(ids.values()))
    for name in names:
        if name not in ids:
            try:
                ids[name] = next(free)
            except StopIteration:
                raise H3Error(f"The script has more than {MAX_SPEAKERS} speakers.") from None
    return ids


_REFERENCE_PREVIEWS = {
    "zh": "今天的路很安静，我们可以慢慢往前走。等到了前面的路口，我会再看看地图，"
          "然后找个合适的地方坐一会儿，把接下来的事情安排好。",
    "ja": "今朝は時間に余裕があります。出かける前に地図を見て、道を確かめておきましょう。"
          "途中で静かな場所を見つけたら、少し休んでから次の予定を考えたいと思います。",
    "ko": "오늘 아침에는 시간이 충분합니다. 출발하기 전에 지도를 보고 길을 확인해 두겠습니다. "
          "가는 길에 조용한 곳을 찾으면 잠시 앉아서 쉬고 다음 일정을 천천히 생각해 보려고 합니다.",
    "en": "We have plenty of time to get there this morning. I will bring the map and check "
          "the route before we leave. We can stop somewhere quiet along the way and decide what to do next.",
}


def _reference_preview(name: str, text: str, language: str, own_lines: str) -> str:
    """Synthesis text for a voice that already has a reference clip, never a transcript of it."""
    if re.search(r"[\u3040-\u30ff]", text):
        return _REFERENCE_PREVIEWS["ja"]
    if re.search(r"[\uac00-\ud7af]", text):
        return _REFERENCE_PREVIEWS["ko"]
    if re.search(r"[\u3400-\u9fff]", text):
        return _REFERENCE_PREVIEWS["zh"]
    if "English" in language:
        return _REFERENCE_PREVIEWS["en"]
    # Another language: the character's own lines, without cues, tags or notes.
    preview = re.sub(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}", "", own_lines)[:1000].strip()
    if not words(preview):
        raise H3Error(f"{name}: no spoken words are available for a reference preview.")
    return preview


def _merge_batch_sounds(sounds: dict, staging: dict, batch_sounds: dict, batch_staging: dict,
                        notes: list[str], palette: dict | None = None) -> None:
    """Add one batch's sounds and staging, renaming a clashing asset rather than substituting the first.

    The story's ``palette`` assets are authoritative: a batch that redefines one keeps the shared
    score, so a recurring theme stays one asset that resumes across scenes.
    """
    palette = palette or {}
    for key, sound in batch_sounds.items():
        if key in palette:
            if sound != palette[key]:
                notes.append(f"Kept the story palette's {key!r} instead of a batch's redefinition.")
            continue
        if key not in sounds or sounds[key] == sound:
            sounds[key] = sound
            continue
        suffix = 2
        while (renamed := f"{key[:27]}_{suffix}") in sounds or renamed in batch_sounds:
            suffix += 1
        batch_staging = {sid: replace(plan,
            ambience=renamed if plan.ambience == key else plan.ambience,
            music=renamed if plan.music == key else plan.music,
            events=tuple(replace(event, sound=renamed) if event.sound == key else event for event in plan.events))
            for sid, plan in batch_staging.items()}
        sounds[renamed] = replace(sound, name=renamed)
        notes.append(f"Renamed conflicting sound {key!r} to {renamed!r}.")
    staging.update(batch_staging)


def _drop_performed_sounds(staging: dict, sounds: dict, directions: dict, by_id: dict, notes: list[str]) -> None:
    """Remove a sound event that duplicates what its speaker already performs (a sigh both acted and played)."""
    for scene_id, plan in list(staging.items()):
        kept = []
        for event in plan.events:
            direction = directions.get(event.at)
            sound = sounds.get(event.sound)
            performed = direction.event if direction else ""
            actor = by_id[event.at].speaker if event.at in by_id else ""
            if (performed and sound and sound.role != "group_reaction" and actor.casefold() in sound.prompt.casefold()
                    and re.search(r"\b" + re.escape(performed.rstrip("s")) + r"\w*\b", sound.prompt, re.I)):
                notes.append(f"{event.at}: omitted a sound asset duplicating {actor} performing {performed}.")
            else:
                kept.append(event)
        staging[scene_id] = replace(plan, events=tuple(kept))


def _ordered_effects(effects) -> tuple[str, ...]:
    """Character, then place and distance, then device: the order Drama 3 chains them."""
    devices = {"phone_call", "walkie_talkie", "radio_broadcast", "megaphone", "old_gramophone", "bad_signal",
               "metallic_helmet"}
    unique = list(dict.fromkeys(effects))
    return tuple(sorted(unique, key=lambda e: 0 if e in CHARACTER_EFFECTS else 2 if e in devices else 1))


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
    max_batch_lines: int = MAX_BATCH_LINES,
    smart: bool = False,
    smart_budget=None,
) -> tuple[str, str, int]:
    """A writer reply (JSON, as the Single-pass writer returns), its source mode, and the call count.

    ``ask(system_path, request)`` returns ``(reply, status)``. Raises ``H3Error`` when a
    required stage fails twice; optional direction and sound that fail are left out with a note.
    """
    notes = notes if notes is not None else []
    calls = _Calls(ask)
    language = drama3_language_instruction(text)
    slots = slot_identities(cast_context)
    reusable = reusable_voices(cast_context)
    reusable_by_id = {identity(entry["name"]): entry for entry in reusable}
    slot_names = {int(slot["speaker_id"]): normalize_name(slot["name"])
                  for slot in slots if str(slot.get("name") or "").strip()}
    text = "\n".join(replace_speaker_tag_line(line, slot_names) for line in str(text or "").splitlines())
    text, expansion_request = source_request(text)
    fixed = _available_names(tuple(voices), speaker_mode, slots)
    allowed = {canonical(name).casefold() for name in fixed} if fixed is not None else None
    known = sorted({
        *(slot["name"] for slot in slots if str(slot.get("name") or "").strip()),
        *(entry["name"] for entry in reusable),
    })
    voices_line = (f"Available voices (cast only these names): {', '.join(fixed)}" if fixed is not None
                   else f"Characters already in the Cast (keep these names): {', '.join(known)}" if known else "")
    kind = input_kind(text)
    roles: dict[str, str] = {}
    palette_style, palette = "", {}
    source = None
    structured = False
    single = speaker_mode == DRAMA3_SPEAKER_SINGLE
    smart_plan = None
    if smart:
        from h3_app.drama3.guided.smart import brief_content, context, parse_plan, preserve_content
        from h3_app.drama3.guided.budget import TOKEN_BUDGET
        smart_budget = smart_budget or TOKEN_BUDGET
        smart_limit = min(smart_budget.lines, max(MIN_BATCH_LINES, int(max_batch_lines or MAX_BATCH_LINES)))

    if kind != BRIEF:
        source_mode = "preserve"
        source = segment_source(text, kind)
        structured = any(is_control_line(line) for line in source.paragraphs)
        title = next((unit.text for unit in source.units if unit.kind == "heading"), "")
        if not source.units:
            raise H3Error("The source contains no spoken text.")
        listing = "\n".join(
            f"{unit.id} [{unit.kind}{', label ' + unit.label if unit.label else ''}] {unit.text}"
            for unit in source.units if unit.kind != "heading")
        content_context = "\n".join(filter(None, [language, voices_line,
            f"Script mode: {'Advanced' if advanced else 'Basic'}.",
            f"Explicit expansion instructions: {expansion_request}" if expansion_request else
            "Preserve source; no inserted lines are authorized."]))
        request = f"{content_context}\nUnits:\n{listing}"
        if smart:
            plan_request = content_context + "\nUnits:\n" + "\n".join(
                f"{unit.id} [{unit.kind}, label {unit.label}] {unit.text}" for unit in source.units)
            plan_request += f"\n{cast_context}" if cast_context else ""
            smart_plan, (palette_style, palette) = calls.run(prompts["plan"], plan_request,
                lambda reply: (parse_plan(reply, source, allowed), parse_sound_palette(reply)), "story planning")
            answer = preserve_content(source, smart_plan, calls, prompts["preserve"],
                content_context, bool(expansion_request), smart_limit,
                allowed, smart_budget)
        else:
            answer, (palette_style, palette) = calls.run(prompts["preserve"], request,
                lambda reply: (parse_content_preserve(reply, source, allowed, expansion=bool(expansion_request)),
                               parse_sound_palette(reply)), "casting")
        title = title or answer["title"]
        utterances = attribute_units(source, answer["speakers"],
            break_after={after for after, _speaker, _text in answer.get("insertions", [])})
        utterances = apply_content_annotations(utterances, answer)
        scenes = scenes_from_units(utterances, answer["scenes"])
        roles = answer.get("roles", {})
    if kind == BRIEF:
        source_mode = "brief"
        brief_voices = "" if single else voices_line
        request = "\n".join(filter(None, [language, brief_voices, f"Script mode: {'Advanced' if advanced else 'Basic'}.",
                                          f"Brief:\n{text.strip()}"]))
        if smart:
            smart_plan, (palette_style, palette) = calls.run(prompts["plan"],
                request + f"\n{cast_context}\nPlan parts for at most {smart_limit} short lines per writing request.",
                lambda reply: (parse_plan(reply, allowed=None if single else allowed), parse_sound_palette(reply)),
                "story planning")
            title, utterances, scenes, roles = brief_content(
                smart_plan, calls, prompts["brief"], request, smart_limit, notes, smart_budget)
            if allowed is not None and not single:
                used = {identity(u.speaker) for u in utterances}
                if len(allowed - used) > (1 if NARRATOR in used and NARRATOR not in allowed else 0):
                    raise H3Error("The Smart scene writing did not use every available voice.")
        else:
            (title, utterances, scenes, roles), (palette_style, palette) = calls.run(
                prompts["brief"], request,
                lambda reply: (parse_content_brief(reply, None if single else allowed, notes),
                               parse_sound_palette(reply)), "writing")
    if fixed is not None and not single and NARRATOR not in {identity(name) for name in fixed} \
            and any(u.speaker == NARRATOR for u in utterances):
        # Adapt to voices: narration goes to an unused voice first, else the first voice.
        spoken = {identity(u.speaker) for u in utterances if u.speaker != NARRATOR}
        unused = {name: speaker for name, speaker in fixed.items() if identity(name) not in spoken}
        pool = unused or fixed
        reader = min(pool, key=pool.get)
        notes.append(f"The narration is read by {reader}, as no voice is named narrator.")
        utterances = [Utterance(u.id, reader, u.text, u.kind, u.effects, u.sources, u.hint)
                      if u.speaker == NARRATOR else u for u in utterances]
    if single:
        name = next(iter(fixed))
        utterances = [Utterance(u.id, name, u.text, u.kind if u.kind == "heading" else "line",
                                u.effects, u.sources, u.hint)
                      for u in utterances]

    names = list(dict.fromkeys(u.speaker for u in utterances))
    ids = _assign_ids(names, slots, fixed)
    slot_by_id = {int(slot["speaker_id"]): slot for slot in slots}
    saved = {name: str(slot_by_id.get(ids[name], {}).get("description")
                       or reusable_by_id.get(identity(name), {}).get("description") or "").strip()
             for name in names}
    for name, description in saved.items():
        if description:
            roles[name] = f"{roles.get(name, '')} Authoritative voice description: {description}".strip()
    by_id = {u.id: u for u in utterances}
    referenced_ids = {int(speaker) for speaker, _ in voices}

    def voices_stage():
        listing = []
        existing = {}
        required = []
        for name in names:
            slot = slot_by_id.get(ids[name], {})
            if slot.get("has_reference", ids[name] in referenced_ids):
                preview = str(slot.get("preview_text") or "") or _reference_preview(
                    name, text, language, " ".join(u.text for u in utterances if u.speaker == name))
                existing[name] = (saved[name] or "The existing reference defines the timbre; "
                                  "use calm natural delivery.", preview)
                continue
            required.append(name)
            lines = " / ".join(u.text for u in utterances if u.speaker == name)[:300]
            role = roles.get(name) or ("the story's narrator" if name == NARRATOR else "")
            listing.append(f"- {name}{f' ({role})' if role else ''}: {lines}")
        if not required:
            return existing
        story_head = " ".join(u.text for u in utterances[:6])[:240]
        context_line = f"Story context: {title + ' — ' if title else ''}{story_head}" if story_head else ""
        size = smart_budget.voices if smart else len(required)
        designed = dict(existing)
        for start in range(0, len(required), size):
            selected = required[start:start + size]
            request = "\n".join(filter(None, [language, context_line, "Characters:", *listing[start:start + size]]))
            designed.update(calls.run(prompts["voices"], request,
                lambda reply, selected=selected: parse_voices(reply, selected), "voice design"))
        return designed

    def batch_context(batch):
        if not smart_plan:
            return ""
        local_scenes = [{"place": scene.place, "change": scene.change} for scene in batch.scenes]
        # Stage-specific cast/profiles below already carry authoritative descriptions.
        return (context(smart_plan, smart_budget, scenes=local_scenes,
                        names=set() if smart_budget.compact else None)
                + "\nNeighboring lines (context only; do not output entries for these):\n"
                + _line_listing(batch.neighbors) + "\nReturn decisions only for the TARGET lines below. "
                "This may be a window within a scene; do not restart the scene or repeat an earlier event. "
                "Use shared assets consistently. Music start/end anchors must both be in TARGET lines.")

    def direction_stage(batch):
        local_names = (batch.speaker_names
                       if smart and smart_budget.compact else set(names))
        cast = ", ".join(f"{name} ({roles[name]})" if roles.get(name) else name for name in names if name in local_names)
        request = "\n".join([language, f"Cast: {cast}", batch_context(batch), "TARGET:", _scene_listing(batch.scenes, by_id)])
        try:
            return calls.run(prompts["direction"], request,
                             lambda reply: parse_direction(reply, batch.line_ids, notes), "direction")
        except H3Error as exc:
            notes.append(f"{exc} Those lines are left undirected.")
            return {}

    palette_line = ""
    if palette_style or palette:
        shared = [{"name": s.name, "kind": s.kind, "prompt": s.prompt} for s in palette.values()]
        palette_line = (f"Story sound palette: {palette_style or 'unspecified'}. Shared score assets (select by "
                        f"name; do not redefine): {json.dumps(shared, ensure_ascii=False)}")

    def sound_stage(batch):
        continuation = ""
        if smart and batch.continued:
            continuation = (f"Continuation windows: {', '.join(batch.continued)}. For these scene IDs ONLY, "
                            "return ambience/music null and omit room/transition: their first windows own those settings. "
                            "Add local events, pauses, positions and voice effects; use anchored music for local score changes. "
                            "For other scene IDs in this request, choose scene beds, room and transition normally.")
        local_roles = ({name: role for name, role in roles.items() if name in batch.speaker_names}
                       if smart and smart_budget.compact else roles)
        request = "\n".join(filter(None, [language, f"Character profiles: {json.dumps(local_roles, ensure_ascii=False)}",
                                          palette_line, sound_context, batch_context(batch), continuation, "TARGET:",
                                          _scene_listing(batch.scenes, by_id)]))
        try:
            scene_lines = batch.scene_lines
            # Only Smart windows need anchors kept inside their own scene; regular Guided drops bad events.
            sounds, staging, effects = calls.run(prompts["sound"], request,
                lambda reply: parse_staging(reply, set(scene_lines), batch.line_ids, notes,
                                           scene_lines=scene_lines if smart else None), "sound design")
            if smart:
                staging = continuation_staging(staging, batch.continued, notes)
            return sounds, staging, effects
        except H3Error as exc:
            notes.append(f"{exc} Those scenes play without added sound.")
            return {}, {}, {}

    batches = _batches(scenes, max_batch_lines)
    if smart:
        batches = scene_batches(scenes, by_id, smart_limit, smart_budget)
    batches = prepare_batches(batches, scenes, utterances, neighbors=smart_budget.neighbors if smart else 0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        voices_future = pool.submit(voices_stage)
        direction_futures = [pool.submit(direction_stage, batch) for batch in batches]
        sound_futures = [pool.submit(sound_stage, batch) for batch in batches] if advanced and not structured else []
        designed = voices_future.result()
        directions = {key: value for future in direction_futures for key, value in future.result().items()}
        sounds, staging, effects = dict(palette), {}, {}
        for future in sound_futures:
            batch_sounds, batch_staging, batch_effects = future.result()
            previous = {sid: staging[sid] for sid in batch_staging if smart and sid in staging}
            _merge_batch_sounds(sounds, staging, batch_sounds, batch_staging, notes, palette)
            for sid, earlier in previous.items():
                staging[sid] = merge_staging(earlier, staging[sid], notes, scene_id=sid)
            effects.update(batch_effects)

    _drop_performed_sounds(staging, sounds, directions, by_id, notes)
    # A character effect is who the character is, so it goes on every line they speak.
    character = {}
    for uid, chosen in effects.items():
        for effect in chosen:
            if effect in CHARACTER_EFFECTS:
                character.setdefault(by_id[uid].speaker, effect)
    utterances = [Utterance(u.id, u.speaker, u.text, u.kind,
                            _ordered_effects([*u.effects, *effects.get(u.id, ()),
                                              *([character[u.speaker]] if u.speaker in character else [])]),
                            u.sources, u.hint) if u.speaker != NARRATOR else u for u in utterances]

    draft = Draft(utterances, scenes, title, directions, sounds, staging)
    if structured:
        script = restore_structure(source, utterances, compile_script(draft, advanced=False, notes=[]))
        script = localize_existing_cues(script)
        notes.append("Existing sound assets were retained; anchored cues were grouped with their speech.")
    else:
        script = compile_script(draft, advanced=advanced, notes=notes)
    speakers = []
    for name in names:
        description, preview = designed[name]
        speakers.append({"speaker_id": ids[name], "name": name,
                         "description": saved[name] or description, "preview_text": preview})
    title = " ".join(str(title or "").split())[:100 if source_mode == "preserve" else 60]
    reply = json.dumps({"title": title, "source_mode": source_mode, "script": script,
                        "speakers": speakers}, ensure_ascii=False)
    return reply, source_mode, calls.count
