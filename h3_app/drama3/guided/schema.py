"""Read and check each Guided stage's JSON answer.

A required stage's problems raise ``StageError`` with one line per problem, which the
orchestrator sends back for a single retry. Optional details that are wrong (an unknown
event, a bad cue) are dropped here or by the compiler and reported as notes instead.
"""

from __future__ import annotations

import json
import re

from h3_app.drama3.serving.planner import AUDIBLE_EVENTS
from h3_app.drama3.serving.rooms import SCENE_ROOMS
from h3_app.drama3.serving.speaker_effects import SPEAKER_TAGS

from .compile import SCOPED_TAGS, sound_name
from .model import (
    EVENT_TIMES, INTERRUPT, LINE, NARRATOR, PLACES, REACTION, SOUND_KINDS,
    Direction, Scene, Sound, SoundEvent, Staging, Utterance,
)
from .segment import NARRATOR_ALIASES, QUOTE, Source

# Effects that are part of who a character is, so they go on every line that character speaks.
CHARACTER_EFFECTS = frozenset({"robotic_voice", "monster_voice", "ghost_spirit", "giant_titan", "fairy_tiny",
                               "alien_voice", "ancient_god"})
MAX_REACTION_WORDS = 4
MAX_REACTION_WIDE = 6
_CONTROL_RE = re.compile(r"[\[\]<>{}|]")
_WIDE_RE = re.compile(r"[^\x00-\x7f]")
_SEX_RE = re.compile(r"^\s*an? (?:(?:young |old |elderly |middle-aged )?(?:fe)?male|girl's|boy's|androgynous|woman's|man's)"
                     r"\b", re.IGNORECASE)


class StageError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def load_answer(reply: str) -> dict:
    """The JSON object in a stage reply, allowing a code fence or a sentence around it."""
    text = str(reply or "").strip()
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{") and "{" in text and "}" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise StageError([f"the reply is not valid JSON ({exc.msg} at character {exc.pos})"]) from exc
    if not isinstance(data, dict):
        raise StageError(["the reply must be one JSON object"])
    return data


def _text(value) -> str:
    return " ".join(str(value or "").split()) if isinstance(value, (str, int, float)) else ""


def canonical(name: str) -> str:
    name = _text(name)
    return NARRATOR if name.casefold() in NARRATOR_ALIASES else name


def _cast(data: dict, problems: list[str], allowed_names: set[str] | None) -> dict[str, str]:
    """Alias (casefolded) to canonical name, for the stage's cast list."""
    aliases: dict[str, str] = {}
    entries = data.get("cast")
    if not isinstance(entries, list) or not entries:
        problems.append('"cast" must list every speaking character')
        return aliases
    for entry in entries:
        name = canonical(entry.get("name") if isinstance(entry, dict) else entry)
        if not name or _CONTROL_RE.search(name) or ":" in name or "：" in name or len(name) > 64:
            problems.append(f"cast name {name!r} must be 1–64 characters without brackets or colons")
            continue
        if allowed_names is not None and name != NARRATOR and name.casefold() not in allowed_names:
            problems.append(f"{name!r} is not one of the available voices: {', '.join(sorted(allowed_names))}")
            continue
        aliases[name.casefold()] = name
        for alias in entry.get("aliases", []) if isinstance(entry, dict) and isinstance(entry.get("aliases"), list) else []:
            if _text(alias):
                aliases.setdefault(_text(alias).casefold(), name)
    return aliases


def parse_content_preserve(reply: str, source: Source, allowed_names: set[str] | None = None) -> dict:
    """The preserve content stage: ``{"title", "speakers": {unit: name}, "scenes": [(unit, place, change)]}``.

    Script labels resolve through the cast's aliases; every quote needs a known speaker.
    """
    data = load_answer(reply)
    problems: list[str] = []
    aliases = _cast(data, problems, allowed_names)
    aliases.setdefault(NARRATOR, NARRATOR)
    units = {unit.id: unit for unit in source.units}
    speakers: dict[str, str] = {}
    raw = data.get("speakers")
    if not isinstance(raw, dict):
        problems.append('"speakers" must map unit IDs to cast names')
        raw = {}
    for unit_id, name in raw.items():
        if unit_id not in units:
            problems.append(f"{unit_id} is not a unit ID")
        elif canonical(name).casefold() not in aliases:
            problems.append(f"{unit_id}: {name!r} is not in the cast")
        else:
            speakers[unit_id] = aliases[canonical(name).casefold()]
    for unit in source.units:
        if unit.kind == "line" and unit.id not in speakers:
            label = canonical(unit.label)
            resolved = aliases.get(label.casefold())
            if resolved is None:
                problems.append(f"{unit.id}: the label {unit.label!r} matches no cast name or alias")
            else:
                speakers[unit.id] = resolved
    missing = [unit.id for unit in source.units if unit.kind == QUOTE and unit.id not in speakers]
    if missing:
        problems.append(f"no speaker for quote unit(s) {', '.join(missing)}")
    scenes = []
    for entry in data.get("scenes") if isinstance(data.get("scenes"), list) else []:
        if isinstance(entry, dict) and entry.get("from") in units:
            scenes.append((entry["from"], _text(entry.get("place")), _text(entry.get("change"))))
    if problems:
        raise StageError(problems)
    return {"title": _text(data.get("title")), "speakers": speakers, "scenes": scenes}


def scenes_from_units(utterances: list[Utterance], starts: list[tuple[str, str, str]]) -> list[Scene]:
    """Scenes that split the utterances where the stage said a scene starts (a unit ID)."""
    order = {}
    for index, utterance in enumerate(utterances):
        for unit in utterance.sources:
            order.setdefault(unit, index)
    begins = sorted({order[unit]: (place, change) for unit, place, change in starts if unit in order}.items())
    if not begins or begins[0][0] != 0:
        begins.insert(0, (0, ("", "")))
    scenes = []
    for number, (start, (place, change)) in enumerate(begins):
        end = begins[number + 1][0] if number + 1 < len(begins) else len(utterances)
        scenes.append(Scene(f"c{number + 1}", tuple(u.id for u in utterances[start:end]), place, change))
    return scenes


def _spoken_words(text: str) -> int:
    return len(_WIDE_RE.findall(text)) // 2 + len(_WIDE_RE.sub(" ", text).split())


def parse_content_brief(reply: str, allowed_names: set[str] | None = None,
                        notes: list[str] | None = None) -> tuple[str, list[Utterance], list[Scene], dict[str, str]]:
    """The brief content stage: title, locked utterances, scenes, and each character's role."""
    data = load_answer(reply)
    problems: list[str] = []
    aliases = _cast(data, problems, allowed_names)
    aliases.setdefault(NARRATOR, NARRATOR)
    roles = {aliases[canonical(entry.get("name")).casefold()]: _text(entry.get("role"))
             for entry in data.get("cast", []) if isinstance(entry, dict)
             and canonical(entry.get("name")).casefold() in aliases}
    utterances: list[Utterance] = []
    scenes: list[Scene] = []
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        problems.append('"scenes" must list at least one scene with its lines')
        raw_scenes = []
    for number, scene in enumerate(raw_scenes, 1):
        lines = scene.get("lines") if isinstance(scene, dict) else None
        if not isinstance(lines, list) or not lines:
            problems.append(f"scene {number} has no lines")
            continue
        ids = []
        for index, line in enumerate(lines, 1):
            where = f"scene {number} line {index}"
            if not isinstance(line, dict):
                problems.append(f"{where} must be an object with speaker and text")
                continue
            name = canonical(line.get("speaker"))
            text = _text(line.get("text")).strip("\"“”「」")
            if name.casefold() not in aliases:
                problems.append(f"{where}: speaker {line.get('speaker')!r} is not in the cast")
                continue
            if not text or _CONTROL_RE.search(text):
                problems.append(f"{where}: text must be spoken words only, without brackets, tags or braces")
                continue
            kind = line.get("type") if line.get("type") in (REACTION, INTERRUPT) else LINE
            name = aliases[name.casefold()]
            previous = utterances[-1] if utterances else None
            if kind == REACTION and (previous is None or previous.speaker in (NARRATOR, name) or name == NARRATOR
                                     or len(_WIDE_RE.findall(text)) > MAX_REACTION_WIDE
                                     or len(_WIDE_RE.sub(" ", text).split()) > MAX_REACTION_WORDS):
                if notes is not None:
                    notes.append(f"{where}: kept the reaction as an ordinary line")
                kind = LINE
            if kind == INTERRUPT and (previous is None or previous.speaker == name):
                kind = LINE
            utterance = Utterance(f"u{len(utterances) + 1}", name, text, kind)
            utterances.append(utterance)
            ids.append(utterance.id)
        if ids:
            scenes.append(Scene(f"c{len(scenes) + 1}", tuple(ids), _text(scene.get("place")), _text(scene.get("change"))))
    if not utterances and not problems:
        problems.append("the script has no lines")
    if problems:
        raise StageError(problems)
    return _text(data.get("title")), utterances, scenes, roles


def parse_voices(reply: str, names: list[str]) -> dict[str, tuple[str, str]]:
    """Name to (description, preview_text) for every name asked for."""
    data = load_answer(reply)
    voices = {}
    problems = []
    for entry in data.get("voices") if isinstance(data.get("voices"), list) else []:
        if not isinstance(entry, dict):
            continue
        name = canonical(entry.get("name"))
        description = _text(entry.get("description"))
        preview = _text(entry.get("preview_text"))
        match = next((wanted for wanted in names if wanted.casefold() == name.casefold()), None)
        if match is None:
            continue
        if not _SEX_RE.match(description) or len(description) > 2000:
            problems.append(f"{match}: description must start with \"A female voice\", \"A male voice\", "
                            "\"A girl's voice\", \"A boy's voice\" or \"An androgynous voice\"")
        elif not preview or len(preview) > 1000 or _CONTROL_RE.search(preview):
            problems.append(f"{match}: preview_text must be 1–1000 characters of plain spoken words")
        else:
            voices[match] = (description, preview)
    missing = [name for name in names if name not in voices and not any(name in p for p in problems)]
    problems.extend(f"{name}: no voice given" for name in missing)
    if problems:
        raise StageError(problems)
    return voices


def parse_direction(reply: str, ids: set[str], notes: list[str]) -> dict[str, Direction]:
    data = load_answer(reply)
    directions = {}
    for entry in data.get("lines") if isinstance(data.get("lines"), list) else []:
        if not isinstance(entry, dict) or entry.get("id") not in ids:
            notes.append(f"dropped direction for unknown line {entry.get('id') if isinstance(entry, dict) else entry!r}")
            continue
        tag = _text(entry.get("tag")).lower().strip("<>/") or "emphasis"
        event = _text(entry.get("event")).lower().strip("[]")
        directions[entry["id"]] = Direction(
            cue=_text(entry.get("cue")), phrase=_text(entry.get("stress")),
            tag=tag if tag in SCOPED_TAGS else "emphasis", event=event if event in AUDIBLE_EVENTS else "")
    return directions


def parse_staging(reply: str, scene_ids: set[str], utterance_ids: set[str], notes: list[str]
                  ) -> tuple[dict[str, Sound], dict[str, Staging], dict[str, tuple[str, ...]]]:
    """Sounds by name, staging by scene ID, and voice effects by utterance ID."""
    data = load_answer(reply)
    sounds = {}
    for entry in data.get("sounds") if isinstance(data.get("sounds"), list) else []:
        if not isinstance(entry, dict):
            continue
        name, kind, prompt = sound_name(entry.get("name")), _text(entry.get("kind")).lower(), _text(entry.get("prompt"))
        if kind in SOUND_KINDS and prompt:
            sounds.setdefault(name, Sound(name, kind, prompt))
        else:
            notes.append(f"dropped sound {entry.get('name')!r}")
    staging, effects = {}, {}
    for entry in data.get("scenes") if isinstance(data.get("scenes"), list) else []:
        if not isinstance(entry, dict) or entry.get("scene") not in scene_ids:
            continue
        events = []
        for event in entry.get("events") if isinstance(entry.get("events"), list) else []:
            if not isinstance(event, dict):
                continue
            when = _text(event.get("when")).lower() or "with"
            place = _text(event.get("place")).lower()
            if when not in EVENT_TIMES or event.get("at") not in utterance_ids:
                notes.append(f"dropped sound event {event.get('sound')!r} at {event.get('at')!r}")
                continue
            events.append(SoundEvent(sound_name(event.get("sound")), event["at"], when, place if place in PLACES else ""))
        positions = []
        raw_positions = entry.get("positions") if isinstance(entry.get("positions"), dict) else {}
        for name, value in raw_positions.items():
            try:
                positions.append((canonical(name), max(-1.0, min(float(value), 1.0))))
            except (TypeError, ValueError):
                continue
        pauses = []
        for pause in entry.get("pauses") if isinstance(entry.get("pauses"), list) else []:
            if isinstance(pause, dict) and pause.get("before") in utterance_ids:
                pauses.append((pause["before"], "long" if _text(pause.get("length")).lower() == "long" else "short"))
        room = _text(entry.get("room")).lower()
        staging[entry["scene"]] = Staging(
            ambience=sound_name(entry["ambience"]) if _text(entry.get("ambience")) else "",
            music=sound_name(entry["music"]) if _text(entry.get("music")) else "",
            room=room if room in SCENE_ROOMS else "", positions=tuple(positions), events=tuple(events),
            pauses=tuple(pauses))
        for voice in entry.get("voices") if isinstance(entry.get("voices"), list) else []:
            if not isinstance(voice, dict) or voice.get("id") not in utterance_ids:
                continue
            raw = voice.get("effects") if isinstance(voice.get("effects"), list) else [voice.get("effect")]
            chosen = tuple(effect for effect in (_text(item).lower().strip("<>") for item in raw)
                           if effect in SPEAKER_TAGS)
            if chosen:
                effects[voice["id"]] = chosen
    return sounds, staging, effects
