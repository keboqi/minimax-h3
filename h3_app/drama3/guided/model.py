"""The Guided writer's intermediate representation: what each stage settled, before any markup."""

from __future__ import annotations

from dataclasses import dataclass, field

NARRATOR = "narrator"

# Utterance kinds. A reaction is heard under the previous speaker's line; an interrupt cuts in as
# the previous line ends; a heading is the narrator reading the title, never directed or scored.
LINE, REACTION, INTERRUPT, HEADING = "line", "reaction", "interrupt", "heading"
UTTERANCE_KINDS = (LINE, REACTION, INTERRUPT, HEADING)

# Sound kinds the staging stage chooses from; code sets loop, volume and duration from the kind.
AMBIENCE, MUSIC, EFFECT, STINGER = "ambience", "music", "effect", "stinger"
SOUND_KINDS = (AMBIENCE, MUSIC, EFFECT, STINGER)

# When a one-shot plays relative to the utterance it is anchored to.
BEFORE, WITH, DURING, AFTER = "before", "with", "during", "after"
EVENT_TIMES = (BEFORE, WITH, DURING, AFTER)

# Where a one-shot is heard; code maps each to pan and distance.
PLACES = ("left", "right", "near", "far", "moving_left_to_right", "moving_right_to_left")


@dataclass(frozen=True)
class Utterance:
    """One locked line: who says it and its exact words, with no controls."""

    id: str
    speaker: str
    text: str
    kind: str = LINE
    effects: tuple[str, ...] = ()  # speaker effect and distance tags, from the source or staging
    sources: tuple[str, ...] = ()  # the source units it reads, for preserved text


@dataclass(frozen=True)
class Scene:
    id: str
    utterances: tuple[str, ...]  # utterance IDs in order
    place: str = ""
    change: str = ""  # what turns in the scene, for the direction and staging stages


@dataclass(frozen=True)
class Direction:
    """A delivery cue (placed by code), an optional phrase to wrap, and an optional closing sound."""

    cue: str = ""
    phrase: str = ""
    tag: str = "emphasis"
    event: str = ""


@dataclass(frozen=True)
class Sound:
    name: str
    kind: str
    prompt: str


@dataclass(frozen=True)
class SoundEvent:
    sound: str
    at: str  # utterance ID
    when: str = WITH
    place: str = ""


@dataclass(frozen=True)
class Staging:
    """One scene's sound design and acoustics."""

    ambience: str = ""
    music: str = ""
    room: str = ""
    positions: tuple[tuple[str, float], ...] = ()
    events: tuple[SoundEvent, ...] = ()
    pauses: tuple[tuple[str, str], ...] = ()  # (utterance ID, "short" | "long") before that utterance


@dataclass
class Draft:
    """Everything the stages settled for one script; ``compile.compile_script`` writes it out."""

    utterances: list[Utterance]
    scenes: list[Scene]
    title: str = ""
    directions: dict[str, Direction] = field(default_factory=dict)
    sounds: dict[str, Sound] = field(default_factory=dict)
    staging: dict[str, Staging] = field(default_factory=dict)  # by scene ID
