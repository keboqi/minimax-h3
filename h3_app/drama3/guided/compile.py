"""Stage 5: write the Drama 3 script from a Guided draft.

The stages chose words, cues, sound kinds and anchors; this module places every cue, picks
every volume, duration and delay, and writes all the markup, so a small model never has to.
Its output is an ordinary ``Name: line`` script for the existing parser and lint.
"""

from __future__ import annotations

import re

from h3_app.drama3.serving.planner import AUDIBLE_EVENTS, expected_seconds, insert_cue, spoken_text
from h3_app.drama3.serving.rooms import scene_room_name
from h3_app.drama3.serving.speaker_effects import SPEAKER_TAGS
from h3_app.drama3.sound_policy import (
    DURATION_CLASSES, SOUND_DEFAULTS, TRANSITIONS, event_budget, gain_envelope, sound_volume,
)
from h3_app.drama3.phonemes import strip_notes
from .local_staging import local_staging, phrase_count

from .model import (
    AFTER, AMBIENCE, BEFORE, DURING, EFFECT, HEADING, INTERRUPT, MUSIC, NARRATOR, REACTION, STINGER, WITH,
    Direction, Draft, Sound, Staging, Utterance,
)

SCOPED_TAGS = ("whisper", "soft", "emphasis", "stress", "forceful", "fast", "slow")
MAX_CUE_WORDS = 5
BARGE_IN = "<#-0.3#>"
PAUSE_SECONDS = {"short": 0.6, "long": 1.2}
MAX_ATTRIBUTION_WORDS = 6
MAX_ATTRIBUTION_WIDE = 12
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


# A speech verb, not a compound ending in one (回答, 小说, 学问).
_ATTRIBUTION_RE = re.compile(
    r"\b(?:said|asked|replied|answered|whispered|shouted|murmured|added|cried)\b"
    r"|(?:说道|问道|答道|应道|喊道|低声道|(?<![小听据])说|(?<![学疑提追])问|(?<![回解])答|喊)[。！？，,:：.!?]*$", re.I)


def is_attribution(utterance: Utterance, previous: Utterance | None) -> bool:
    """A short narrator line right after a character's line ("she said."), which reads plain."""
    if utterance.speaker != NARRATOR or previous is None or previous.speaker == NARRATOR:
        return False
    short = (len(_WIDE_RE.findall(utterance.text)) <= MAX_ATTRIBUTION_WIDE
             and len(_WIDE_RE.sub(" ", utterance.text).split()) <= MAX_ATTRIBUTION_WORDS)
    return short and _ATTRIBUTION_RE.search(utterance.text) is not None


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
    tags = "".join(f"<{effect}>" for effect in effects)
    return f"{utterance.speaker}: {tags + ' ' if tags else ''}{body}"


def sound_name(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", str(name or "").strip().lower()).strip("_")[:32] or "sound"


def sound_line(sound: Sound, place: str = "", *, level: str = "", anchor: str = "", end_anchor: str = "",
               edge: str = "end", occurrence: int = 1, end_occurrence: int = 1) -> str:
    """The ``<sound>`` tag for one use of ``sound``; ``level`` sets this use's volume, not the asset's."""
    loop, _volume, duration = SOUND_DEFAULTS[sound.kind]
    volume = sound_volume(sound.kind, level)
    duration = DURATION_CLASSES.get(sound.duration_class, duration)
    attrs = [f'name="{sound_name(sound.name)}"']
    envelope = gain_envelope(sound.envelope)
    if envelope:
        attrs.append('envelope="' + ','.join(f'{t:g}:{g:g}' for t, g in envelope) + '"')
    for key, value in (("anchor", anchor), ("end_anchor", end_anchor)):
        if value:
            safe = re.sub(r'[<>";~]', " ", value)
            attrs.append(f'{key}="{safe}"')
    if anchor:
        attrs.extend([f'edge="{edge}"', f'occurrence="{occurrence}"'])
    if end_anchor and end_occurrence != 1:
        attrs.append(f'end_occurrence="{end_occurrence}"')
    if loop:
        attrs.append('resume="true"')
    if sound.role:
        attrs.append(f'role="{sound.role}"')
    if sound.perspective:
        attrs.append(f'perspective="{sound.perspective}"')
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
    group_cue = ""  # the cue before a line spoken together, which each member may act
    for index, utterance in enumerate(draft.utterances):
        before = draft.utterances[index - 1] if index else None
        if not (before is not None and utterance.sources and before.sources == utterance.sources):
            group_cue = previous_cue
        attribution = is_attribution(utterance, before)
        bodies[utterance.id], cue = direct_line(utterance, draft.directions.get(utterance.id), group_cue, notes,
                                                attribution=attribution, cast_names=cast_names)
        previous_cue = cue or previous_cue

    def line(utterance: Utterance) -> str:
        return _speaker_line(utterance, bodies[utterance.id], notes)

    if not advanced:
        return "\n".join(line(utterance) for utterance in draft.utterances)

    sounds = {sound_name(name): sound for name, sound in draft.sounds.items()}
    blocks: list[tuple[str, list[str], list[str], list[str]]] = []  # (transition, master attrs, loops, items)
    for scene in draft.scenes:
        staging = draft.staging.get(scene.id, Staging())
        anchored, local_cues = _scene_events(scene, staging, sounds, by_id, notes)
        pauses = dict(staging.pauses)
        holds = []

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
            joined = bool(joint_sources)  # a later member of a line spoken together
            reactions: list[Utterance] = []
            if not joined and not next_is_joint and utterance.speaker != NARRATOR and utterance.kind != HEADING:
                while position < len(ids) and by_id[ids[position]].kind == REACTION \
                        and by_id[ids[position]].speaker not in (NARRATOR, utterance.speaker):
                    reactions.append(by_id[ids[position]])
                    position += 1
            same_passage = previous is not None and previous.speaker == utterance.speaker
            if joined:
                # The group's pause and barge-in come before it opens, so every member starts together.
                if utterance.id in pauses:
                    notes.append(f"{utterance.id}: no pause inside a line spoken together")
            elif utterance.id in pauses:
                if same_passage:
                    # A full long turn may span several synthesis chunks. Anchor its ending.
                    phrase = re.sub(r'[<>";~]', " ", strip_notes(spoken_text(previous.text))).split()
                    phrase = " ".join(phrase[-6:])
                    if _WIDE_RE.search(phrase):
                        phrase = phrase[-12:]
                    holds.append((previous.id, phrase, PAUSE_SECONDS.get(pauses[utterance.id], 0.6)))
                else:
                    items.append(_delay(PAUSE_SECONDS.get(pauses[utterance.id], 0.6)))
            # A line cut off with a dash is already talked over; one that ends normally gets barged in on.
            if utterance.kind == INTERRUPT and not joined and previous is not None and not same_passage \
                    and not previous.text.rstrip().endswith("—"):
                items.append(BARGE_IN)
            if next_is_joint and not joined:
                joint_sources = utterance.sources
                items.append("<parallel>")
            if joint_sources:
                items.append("<sequential>")
            items.extend(anchored.get((utterance.id, BEFORE), []))
            withs = anchored.get((utterance.id, WITH), [])
            durings = anchored.get((utterance.id, DURING), [])
            if withs or durings or reactions:
                spoken = expected_seconds(bodies[utterance.id])
                items.append("<parallel>")
                items.extend(withs)
                items.append(line(utterance))
                for sound in durings:
                    items.extend(["<sequential>", _delay(spoken * 0.5), sound, "</sequential>"])
                if reactions:
                    items.extend(['<parallel reaction="true">', '<sequential>'])
                    for index, reaction in enumerate(reactions):
                        if index:
                            items.append(_delay(0.4))
                        items.append(line(reaction))
                    items.append("</sequential>")
                    items.append("</parallel>")
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

        items = local_staging(items, [(uid, line(by_id[uid])) for uid in ids], local_cues, holds)
        attrs, loops = _scene_master(scene, staging, sounds, by_id, notes)
        blocks.append((staging.transition or "cut", attrs, loops, items))
    return _join_scenes(blocks)


def _scene_events(scene, staging: Staging, sounds: dict[str, Sound], by_id: dict[str, Utterance],
                  notes: list[str]):
    """A scene's one-shots by (line, when), and its phrase-anchored cues for ``local_staging``."""
    events = list(staging.events)
    budget = event_budget(len(scene.utterances))
    if len(events) > budget:
        notes.append(f"{scene.id}: kept the first {budget} of {len(events)} sound events")
        events = events[:budget]
    anchored: dict[tuple[str, str], list[str]] = {}
    local_cues = []
    comic_used = False
    for event in events:
        sound = sounds.get(sound_name(event.sound))
        if sound is None or sound.kind not in (EFFECT, STINGER, MUSIC) or event.at not in scene.utterances:
            notes.append(f"{scene.id}: dropped sound event {event.sound!r} at {event.at}")
            continue
        if sound.role == "comic":
            if comic_used:
                continue
            comic_used = True
        if event.phrase or sound.kind == MUSIC:
            phrase = event.phrase or strip_notes(spoken_text(by_id[event.at].text))
            if phrase_count(phrase, by_id[event.at].text) < event.occurrence:
                notes.append(f'{event.at}: dropped unmatched sound anchor {phrase!r}')
                continue
            end_at = event.at
            if event.end_phrase:
                start = scene.utterances.index(event.at)
                candidates = [uid for uid in scene.utterances[start:]
                              if (not event.end_at or uid == event.end_at)
                              and phrase_count(event.end_phrase, by_id[uid].text) >= event.end_occurrence]
                if not candidates:
                    notes.append(f'{event.at}: dropped unmatched ending anchor {event.end_phrase!r}')
                    continue
                end_at = candidates[0]
            elif sound.kind == MUSIC:
                end_at = scene.utterances[-1]
            def format_cue(number, end_number, sound=sound, event=event, phrase=phrase):
                return sound_line(sound, event.place, level=event.level, anchor=phrase,
                    end_anchor=event.end_phrase, edge="start" if event.when in (BEFORE, WITH) else "end",
                    occurrence=number, end_occurrence=end_number)
            local_cues.append((event.at, end_at, phrase, event.end_phrase,
                               event.occurrence, event.end_occurrence, format_cue))
        else:
            anchored.setdefault((event.at, event.when), []).append(
                sound_line(sound, event.place, level=event.level))
    return anchored, local_cues


def _scene_master(scene, staging: Staging, sounds: dict[str, Sound], by_id: dict[str, Utterance],
                  notes: list[str]) -> tuple[list[str], list[str]]:
    """The scene master's room/stage attributes and its looped beds."""
    loops = []
    for name, kind, level in ((staging.ambience, AMBIENCE, staging.ambience_level),
                              (staging.music, MUSIC, staging.music_level)):
        if not name:
            continue
        sound = sounds.get(sound_name(name))
        if sound is None or sound.kind != kind:
            notes.append(f"{scene.id}: dropped {kind} {name!r}")
        else:
            loops.append(sound_line(sound, level=level))
    room = scene_room_name(staging.room) if staging.room else ""
    if staging.room and not room:
        notes.append(f"{scene.id}: unknown room {staging.room!r}")
    speakers = {by_id[uid].speaker for uid in scene.utterances} - {NARRATOR}
    stage = ", ".join(f"{name}={max(-1.0, min(float(value), 1.0)):g}"
                      for name, value in staging.positions if name in speakers)
    attrs = ([f'room="{room}"'] if room else []) + ([f'stage="{stage}"'] if stage else [])
    return attrs, loops


def _join_scenes(blocks: list[tuple[str, list[str], list[str], list[str]]]) -> str:
    """Write the scenes in order, applying each one's transition from the scene before it.

    An establishing scene lets its bed play longer before the first line. A bridge crossfades
    beds: the previous scene's beds ring on while this scene's beds lead in, so its first line
    starts as the old bed fades out rather than over it. A bridge needs beds on both sides;
    otherwise it is a cut.
    """
    def effective(index: int) -> str:
        transition = blocks[index][0] if blocks[index][0] in TRANSITIONS else "cut"
        if transition == "bridge" and not (index and blocks[index - 1][2] and blocks[index][2]):
            return "cut"
        return transition

    out: list[str] = []
    for index, (_transition, attrs, loops, items) in enumerate(blocks):
        lead_in, _tail, gap = TRANSITIONS[effective(index)]
        next_tail = TRANSITIONS[effective(index + 1)][1] if index + 1 < len(blocks) else None
        if index:
            out.append(f"<#{gap:g}#>")
        attrs = list(attrs)
        if loops and lead_in is not None:
            attrs.append(f'lead_in="{lead_in:g}"')
        if loops and next_tail is not None:
            attrs.append(f'tail="{next_tail:g}"')
        if loops or attrs:
            opening = "<parallel" + "".join(f" {attr}" for attr in attrs) + ">"
            out.extend([opening, *loops, "<sequential>", *items, "</sequential>", "</parallel>"])
        else:
            out.extend(items)
    return "\n".join(out)
