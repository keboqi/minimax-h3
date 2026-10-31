"""Stage 5: write the Drama 3 script from a Guided draft.

The stages chose words, cues, sound kinds and anchors; this module places every cue, picks
every volume, duration and delay, and writes all the markup, so a small model never has to.
Its output is an ordinary ``Name: line`` script for the existing parser and lint.
"""

from __future__ import annotations

import re

from h3_app.drama3.serving.planner import AUDIBLE_EVENTS, expected_seconds, insert_cue
from h3_app.drama3.serving.rooms import scene_room_name
from h3_app.drama3.serving.speaker_effects import SPEAKER_TAGS

from .model import (
    AFTER, AMBIENCE, BEFORE, DURING, EFFECT, HEADING, INTERRUPT, MUSIC, NARRATOR, REACTION, STINGER, WITH,
    Direction, Draft, Sound, Staging, Utterance,
)

SCOPED_TAGS = ("whisper", "soft", "emphasis", "stress", "forceful", "fast", "slow")
MAX_CUE_WORDS = 5
SCENE_PAUSE = "<#0.8#>"
BARGE_IN = "<#-0.3#>"
PAUSE_SECONDS = {"short": 0.6, "long": 1.2}
MAX_EVENTS_PER_SCENE = 4
MAX_ATTRIBUTION_WORDS = 6
MAX_ATTRIBUTION_WIDE = 12
# (loop, volume, duration) by sound kind, within the ranges the Single-pass writer is given.
SOUND_SETTINGS = {AMBIENCE: (True, 0.4, 25), MUSIC: (True, 0.2, 40), EFFECT: (False, 1.0, 4), STINGER: (False, 0.4, 6)}
PLACEMENT = {"left": 'pan="-0.6"', "right": 'pan="0.6"', "near": 'distance="near"', "far": 'distance="far"',
             "moving_left_to_right": 'pan="-0.8,0.8"', "moving_right_to_left": 'pan="0.8,-0.8"'}

_WIDE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_META_CUE_RE = re.compile(r"\b(?:narrator|narrative|narration|description|transition|follow-up)\b", re.IGNORECASE)


def _clean_cue(cue: str) -> str:
    """The cue's words, or "" when it is not a short plain-English brief."""
    words = " ".join(str(cue or "").strip().strip("[]").split())
    if not words or _WIDE_RE.search(words) or re.search(r"[\[\]<>{}|]", words):
        return ""
    cleaned = _META_CUE_RE.sub("", words)
    cleaned = re.sub(r"\s*,\s*,+\s*", ", ", cleaned).strip(" ,;-")
    cleaned = " ".join(cleaned.split())
    if not re.search(r"[A-Za-z]{2,}", cleaned):
        return ""
    return cleaned if len(cleaned.split()) <= MAX_CUE_WORDS else ""


def _place_cue(text: str, bracketed: str, cast_names: tuple[str, ...] = ()) -> str | None:
    """Place ``bracketed`` after the opening phrase, avoiding splitting a leading 3+ char CJK name."""
    for name in cast_names:
        if len(name) > 2 and text.startswith(name) and _WIDE_RE.search(name):
            rest = text[len(name):]
            if rest.strip() and not any(ch in text[:len(name) + 1] for ch in ",，、;；:：.!?。！？"):
                placed = insert_cue(text, bracketed)
                if placed is not None and placed.startswith(f"{name[:2]} {bracketed}"):
                    return f"{name} {bracketed} {rest.lstrip()}"
    return insert_cue(text, bracketed)


def _wrap_phrase(text: str, phrase: str, tag: str) -> str | None:
    """``text`` with its one occurrence of ``phrase`` in ``<tag>…</tag>``, or None when that is unsafe."""
    phrase = phrase.strip()
    if tag not in SCOPED_TAGS or not phrase or text.count(phrase) != 1 or re.search(r"[\[\]<>{}]", phrase):
        return None
    start = text.index(phrase)
    end = start + len(phrase)
    # Never open the line with a tag (so never wrap all of it), and never split a word from its {note}.
    if not text[:start].strip() or text[end:end + 1] == "{":
        return None
    return f"{text[:start]}<{tag}>{phrase}</{tag}>{text[end:]}"


def is_attribution(utterance: Utterance, previous: Utterance | None) -> bool:
    """A short narrator line right after a character's line ("she said."), which reads plain."""
    if utterance.speaker != NARRATOR or previous is None or previous.speaker == NARRATOR:
        return False
    return len(_WIDE_RE.findall(utterance.text)) <= MAX_ATTRIBUTION_WIDE         and len(_WIDE_RE.sub(" ", utterance.text).split()) <= MAX_ATTRIBUTION_WORDS


def direct_line(utterance: Utterance, direction: Direction | None, previous_cue: str, notes: list[str],
                *, attribution: bool = False, cast_names: tuple[str, ...] = ()) -> tuple[str, str]:
    """The utterance's spoken body with its direction applied, and the cue it ends up with.

    An ``attribution`` keeps its words plain apart from a scoped phrase: a cue on "she said."
    would make the narrator act out the character's line.
    """
    text = utterance.text
    if utterance.kind == HEADING:
        return text, ""
    if direction is None and utterance.hint and _clean_cue(utterance.hint):
        direction = Direction(cue=utterance.hint)
    if direction is None:
        return text, ""
    raw_cue = direction.cue or (utterance.hint if _clean_cue(utterance.hint) else "")
    cue = "" if attribution else _clean_cue(raw_cue)
    if direction.cue and not cue and not attribution:
        notes.append(f"{utterance.id}: dropped cue [{direction.cue}]")
    if cue and cue.casefold() == previous_cue.casefold():
        notes.append(f"{utterance.id}: dropped cue [{cue}] repeating the line before")
        cue = ""
    if cue:
        placed = _place_cue(text, f"[{cue}]", cast_names)
        if placed is None:
            notes.append(f"{utterance.id}: no place for cue [{cue}]")
            cue = ""
        else:
            text = placed
    if direction.phrase:
        wrapped = _wrap_phrase(text, direction.phrase, direction.tag)
        # A phrase that opens the line is skipped quietly: a scoped tag may never open one.
        if wrapped is None and not utterance.text.lstrip().startswith(direction.phrase.strip()):
            notes.append(f"{utterance.id}: could not mark “{direction.phrase}”")
        elif wrapped is not None:
            text = wrapped
    event = direction.event.strip().strip("[]").lower()
    if event:
        if event in AUDIBLE_EVENTS and not text.rstrip().endswith("—"):
            text = f"{text.rstrip()} [{event}]"
        else:
            notes.append(f"{utterance.id}: dropped event [{event}]")
    return text, cue


def _speaker_line(utterance: Utterance, body: str, notes: list[str]) -> str:
    effects = [effect for effect in utterance.effects if effect in SPEAKER_TAGS]
    if utterance.speaker == NARRATOR and effects:
        notes.append(f"{utterance.id}: the narrator takes no voice effect")
        effects = []
    tags = "".join(f"<{effect}>" for effect in effects[:2])
    return f"{utterance.speaker}: {tags + ' ' if tags else ''}{body}"


def sound_name(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", str(name or "").strip().lower()).strip("_")[:32] or "sound"


def sound_line(sound: Sound, place: str = "") -> str:
    loop, volume, duration = SOUND_SETTINGS[sound.kind]
    attrs = [f'name="{sound_name(sound.name)}"']
    if loop:
        attrs.append('loop="true"')
    if volume != 1.0:
        attrs.append(f'volume="{volume:g}"')
    attrs.append(f'duration="{duration}"')
    if not loop and place in PLACEMENT:
        attrs.append(PLACEMENT[place])
    prompt = " ".join(re.sub(r"[<>]", " ", sound.prompt).split())
    return f"<sound {' '.join(attrs)}>{prompt}</sound>"


def _delay(seconds: float) -> str:
    return f"<#{max(0.3, round(seconds, 1)):g}#>"


def compile_script(draft: Draft, *, advanced: bool, notes: list[str] | None = None) -> str:
    """The Drama 3 script for ``draft``, with ``Name: line`` speaker lines.

    Basic scripts are only directed lines. Advanced scripts give a scene with loops, a room or
    positions its master ``<parallel>``, and anchor one-shots, reactions and interruptions to
    the exact utterance they belong to, so a speaker's passage stays one take elsewhere.
    What could not be applied is appended to ``notes``.
    """
    notes = notes if notes is not None else []
    by_id = {utterance.id: utterance for utterance in draft.utterances}
    cast_names = tuple(sorted({u.speaker for u in draft.utterances if u.speaker != NARRATOR and len(u.speaker) > 2},
                              key=len, reverse=True))
    bodies: dict[str, str] = {}
    previous_cue = ""
    for index, utterance in enumerate(draft.utterances):
        attribution = is_attribution(utterance, draft.utterances[index - 1] if index else None)
        bodies[utterance.id], cue = direct_line(utterance, draft.directions.get(utterance.id), previous_cue, notes,
                                                attribution=attribution, cast_names=cast_names)
        previous_cue = cue or previous_cue

    def line(utterance: Utterance) -> str:
        return _speaker_line(utterance, bodies[utterance.id], notes)

    if not advanced:
        return "\n".join(line(utterance) for utterance in draft.utterances)

    sounds = {sound_name(name): sound for name, sound in draft.sounds.items()}
    out: list[str] = []
    for scene_index, scene in enumerate(draft.scenes):
        staging = draft.staging.get(scene.id, Staging())
        events = list(staging.events)
        if len(events) > MAX_EVENTS_PER_SCENE:
            notes.append(f"{scene.id}: kept the first {MAX_EVENTS_PER_SCENE} of {len(events)} sound events")
            events = events[:MAX_EVENTS_PER_SCENE]
        anchored: dict[tuple[str, str], list[str]] = {}
        for event in events:
            sound = sounds.get(sound_name(event.sound))
            if sound is None or sound.kind not in (EFFECT, STINGER) or event.at not in scene.utterances:
                notes.append(f"{scene.id}: dropped sound event {event.sound!r} at {event.at}")
                continue
            anchored.setdefault((event.at, event.when), []).append(sound_line(sound, event.place))
        pauses = dict(staging.pauses)

        items: list[str] = []
        previous: Utterance | None = None
        ids = list(scene.utterances)
        position = 0
        joint_sources: tuple[str, ...] = ()
        while position < len(ids):
            utterance = by_id[ids[position]]
            position += 1
            # Attribution expands a joint quote into adjacent lines sharing source IDs.
            # Keep each member's anchored sounds in its own branch, starting together.
            next_is_joint = (bool(utterance.sources) and position < len(ids)
                             and by_id[ids[position]].sources == utterance.sources)
            if next_is_joint and not joint_sources:
                joint_sources = utterance.sources
                items.append("<parallel>")
            if joint_sources:
                items.append("<sequential>")
            reactions: list[Utterance] = []
            if not joint_sources and utterance.speaker != NARRATOR and utterance.kind != HEADING:
                while position < len(ids) and by_id[ids[position]].kind == REACTION \
                        and by_id[ids[position]].speaker not in (NARRATOR, utterance.speaker):
                    reactions.append(by_id[ids[position]])
                    position += 1
            same_passage = previous is not None and previous.speaker == utterance.speaker
            if utterance.id in pauses:
                if same_passage:
                    notes.append(f"{utterance.id}: no pause inside one speaker's passage")
                else:
                    items.append(_delay(PAUSE_SECONDS.get(pauses[utterance.id], 0.6)))
            # A line cut off with a dash is already talked over; one that ends normally gets barged in on.
            if utterance.kind == INTERRUPT and previous is not None and not same_passage \
                    and not previous.text.rstrip().endswith("—"):
                items.append(BARGE_IN)
            items.extend(anchored.get((utterance.id, BEFORE), []))
            withs = anchored.get((utterance.id, WITH), [])
            durings = anchored.get((utterance.id, DURING), [])
            if withs or durings or reactions:
                spoken = expected_seconds(bodies[utterance.id])
                items.append("<parallel>")
                items.extend(withs if len(withs) == 1 else ["<sequential>", *withs, "</sequential>"] if withs else [])
                items.append(line(utterance))
                for sound in durings:
                    items.extend(["<sequential>", _delay(spoken * 0.5), sound, "</sequential>"])
                if reactions:
                    items.append("<sequential>")
                    items.append(_delay(spoken * 0.6))
                    for index, reaction in enumerate(reactions):
                        if index:
                            items.append(_delay(0.4))
                        items.append(line(reaction))
                    items.append("</sequential>")
                items.append("</parallel>")
            else:
                items.append(line(utterance))
            items.extend(anchored.get((utterance.id, AFTER), []))
            if joint_sources:
                items.append("</sequential>")
                if not next_is_joint:
                    items.append("</parallel>")
                    joint_sources = ()
            previous = reactions[-1] if reactions else utterance

        loops = []
        for name, kind in ((staging.ambience, AMBIENCE), (staging.music, MUSIC)):
            if not name:
                continue
            sound = sounds.get(sound_name(name))
            if sound is None or sound.kind != kind:
                notes.append(f"{scene.id}: dropped {kind} {name!r}")
            else:
                loops.append(sound_line(sound))
        room = scene_room_name(staging.room) if staging.room else ""
        if staging.room and not room:
            notes.append(f"{scene.id}: unknown room {staging.room!r}")
        speakers = {by_id[uid].speaker for uid in scene.utterances} - {NARRATOR}
        stage = ", ".join(f"{name}={max(-1.0, min(float(value), 1.0)):g}"
                          for name, value in staging.positions if name in speakers)
        if scene_index:
            out.append(SCENE_PAUSE)
        if loops or room or stage:
            attrs = (f' room="{room}"' if room else "") + (f' stage="{stage}"' if stage else "")
            out.extend([f"<parallel{attrs}>", *loops, "<sequential>", *items, "</sequential>", "</parallel>"])
        else:
            out.extend(items)
    return "\n".join(out)
