"""Recursive AST parser and sound/tag utilities for Drama 3 scripts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Union

from h3_app.workflows.drama3 import NAME_LINE

from .rooms import SCENE_ROOMS, scene_room_name
from .speaker_effects import extract_speaker_effects, has_speaker_effects, strip_speaker_effects

SPEAKER_LINE_RE = re.compile(r"^<\|speaker:(\d+)\|>\s*(.*)$")
# An opening tag may carry attributes: <parallel room="small_room" stage="Mia=-0.4">.
CONTAINER_TAG_RE = re.compile(r"(</?(?:parallel|sequential)\b[^<>]*>)", re.IGNORECASE)
SOUND_BLOCK_RE = re.compile(
    r"(<sound\b[^>]*>.*?</sound\s*>)",
    re.IGNORECASE | re.DOTALL,
)
SOUND_PARSE_RE = re.compile(
    r"^<sound(?P<attrs>\b[^>]*)>(?P<body>.*?)</sound\s*>$",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))')
PAUSE_TAG_RE = re.compile(r"<#\s*(\d+(?:\.\d+)?)\s*#>")


@dataclass(frozen=True)
class SpeechTurn:
    speaker: str  # the Cast name, or "Speaker N" for an unnamed slot
    text: str
    speaker_effects: tuple[str, ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class SpeechNode:
    turns: tuple[SpeechTurn, ...]


@dataclass(frozen=True)
class SoundNode:
    description: str
    name: str = ""
    loop: bool = False
    volume: float = 1.0
    duration: int = 5
    pan: tuple[float, float] | None = None  # (start, end) position, -1 left to 1 right
    distance: str = ""  # "near", "far", or "" for as recorded


@dataclass(frozen=True)
class PauseNode:
    seconds: float


@dataclass(frozen=True)
class SequentialNode:
    children: tuple["TreeNode", ...]


@dataclass(frozen=True)
class ParallelNode:
    children: tuple["TreeNode", ...]
    room: str = ""  # the scene's room (``rooms.SCENE_ROOMS``), heard by its in-world voices
    stage: tuple[tuple[str, float], ...] = ()  # (speaker, position -1 left to 1 right) in this scene


TreeNode = Union[
    SpeechNode,
    SoundNode,
    PauseNode,
    SequentialNode,
    ParallelNode,
]


def parse_sound_tag(raw_tag: str) -> SoundNode | None:
    match = SOUND_PARSE_RE.match(raw_tag.strip())
    if not match:
        return None
    attrs_raw = match.group("attrs") or ""
    body = " ".join((match.group("body") or "").split()).strip()
    if not body:
        return None
    attrs: dict[str, str] = {}
    for attr_match in ATTR_RE.finditer(attrs_raw):
        key = attr_match.group(1).lower()
        val = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or ""
        attrs[key] = val.strip()

    loop = attrs.get("loop", "").lower() in ("true", "1", "yes")
    try:
        volume = float(attrs["volume"]) if "volume" in attrs else 1.0
    except ValueError:
        volume = 1.0
    volume = max(0.0, min(volume, 2.0))

    default_duration = 10 if loop else 5
    body_dur_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b", body, re.IGNORECASE)
    if "duration" in attrs:
        try:
            duration = int(round(float(attrs["duration"])))
        except ValueError:
            duration = default_duration
    elif body_dur_match:
        try:
            duration = int(round(float(body_dur_match.group(1))))
        except ValueError:
            duration = default_duration
    else:
        duration = default_duration
    duration = max(1, min(duration, 47))

    name = " ".join(attrs.get("name", "").split()).strip()
    if not name:
        name = default_sound_name(body)
    distance = attrs.get("distance", "").lower()
    return SoundNode(
        description=body,
        name=name,
        loop=loop,
        volume=volume,
        duration=duration,
        pan=parse_pan(attrs.get("pan", "")),
        distance=distance if distance in ("near", "far") else "",
    )


PAN_RE = re.compile(r"^\s*(-?\d*\.?\d+)\s*(?:(?:,|to|->|→)\s*(-?\d*\.?\d+)\s*)?$", re.IGNORECASE)


def parse_pan(value: str) -> tuple[float, float] | None:
    """``pan="-0.5"`` holds a position; ``pan="-0.8,0.2"`` moves from one to the other."""
    match = PAN_RE.match(str(value or ""))
    if match is None:
        return None
    start = max(-1.0, min(float(match.group(1)), 1.0))
    end = max(-1.0, min(float(match.group(2)), 1.0)) if match.group(2) else start
    return start, end


def sound_placement_attrs(node: SoundNode) -> list[str]:
    """The ``pan`` and ``distance`` attributes that place ``node``, as written in a tag."""
    attrs = []
    if node.pan is not None:
        start, end = node.pan
        attrs.append(f'pan="{start:g}"' if start == end else f'pan="{start:g},{end:g}"')
    if node.distance:
        attrs.append(f'distance="{node.distance}"')
    return attrs


STAGE_ENTRY_RE = re.compile(r"\s*([^=,;]+?)\s*=\s*(-?\d*\.?\d+)\s*(?:[,;]|$)")


def tag_attrs(tag: str) -> dict[str, str]:
    """A tag's attributes by lowercase name."""
    return {match.group(1).lower(): (match.group(2) or match.group(3) or match.group(4) or "").strip()
            for match in ATTR_RE.finditer(tag)}


def staging_problems(tag: str) -> list[str]:
    """What is wrong with a ``<parallel ...>`` tag's ``room`` and ``stage``, for the writer's status."""
    attrs, problems = tag_attrs(tag), []
    if "room" in attrs and not scene_room_name(attrs["room"]):
        problems.append(f'unknown room "{attrs["room"]}" (rooms: {", ".join(SCENE_ROOMS)})')
    if "stage" in attrs:
        entries = [entry for entry in re.split(r"[,;]", attrs["stage"]) if entry.strip()]
        bad = [entry.strip() for entry in entries if not STAGE_ENTRY_RE.fullmatch(entry)]
        if bad or not entries:
            problems.append(f'stage "{attrs["stage"]}" is not written Name=position, ...')
    return problems


def parse_parallel_attrs(tag: str, resolve: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    """The ``room`` and ``stage`` of a ``<parallel ...>`` tag; invalid values are left out.

    ``stage="Mia=-0.4, Leo=0.5"`` names speakers as the script's lines do and places each
    from -1 (left) to 1 (right).
    """
    attrs = tag_attrs(tag)
    stage = []
    for match in STAGE_ENTRY_RE.finditer(attrs.get("stage", "")):
        name = " ".join(match.group(1).split())
        label = (resolve(name) if resolve else None) or name
        stage.append((label, max(-1.0, min(float(match.group(2)), 1.0))))
    return {"room": scene_room_name(attrs.get("room", "")), "stage": tuple(stage)}


def default_sound_name(description: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", description.lower())[:4]
    return " ".join(words) if words else "sound effect"


def _normalize_script_tokens(script: str) -> list[str]:
    text = (script or "").replace("\r\n", "\n")
    text = SOUND_BLOCK_RE.sub(r"\n\1\n", text)
    text = CONTAINER_TAG_RE.sub(r"\n\1\n", text)
    tokens: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line:
            tokens.append(line)
    return tokens


def _expand_speaker_line(speaker: str, raw_body: str) -> list[SpeechTurn | PauseNode]:
    """Expand a single speaker line's body into SpeechTurn and PauseNode items."""
    if not PAUSE_TAG_RE.search(raw_body):
        body, effects = extract_speaker_effects(raw_body)
        cleaned = body.strip()
        if not cleaned:
            return []
        return [
            SpeechTurn(
                speaker=speaker,
                text=cleaned,
                speaker_effects=effects,
                raw_text=raw_body.strip(),
            )
        ]

    items: list[SpeechTurn | PauseNode] = []
    active_effects: tuple[str, ...] = ()
    last_end = 0
    for match in PAUSE_TAG_RE.finditer(raw_body):
        raw_seg = raw_body[last_end : match.start()]
        seg_body, seg_effects = extract_speaker_effects(raw_seg)
        if seg_effects:
            active_effects = tuple(dict.fromkeys((*active_effects, *seg_effects)))
        before = seg_body.strip()
        if before:
            items.append(
                SpeechTurn(
                    speaker=speaker,
                    text=before,
                    speaker_effects=active_effects,
                    raw_text=before,
                )
            )
        try:
            seconds = float(match.group(1))
        except ValueError:
            seconds = 0.0
        if seconds > 0:
            items.append(PauseNode(seconds=min(seconds, 30.0)))
        last_end = match.end()
    raw_tail = raw_body[last_end:]
    tail_body, tail_effects = extract_speaker_effects(raw_tail)
    if tail_effects:
        active_effects = tuple(dict.fromkeys((*active_effects, *tail_effects)))
    tail = tail_body.strip()
    if tail:
        items.append(
            SpeechTurn(
                speaker=speaker,
                text=tail,
                speaker_effects=active_effects,
                raw_text=tail,
            )
        )
    return items


def parse_drama3_tree(script: str, *, default_speaker: str = "Speaker 0",
                      resolve: Callable[[str], str | None] | None = None) -> SequentialNode:
    """Parse a Drama 3 script with arbitrary `<parallel>` and `<sequential>` nesting.

    Speaker lines start with a name ("Mia: Hello.") or, in older scripts, a
    ``<|speaker:N|>`` tag, which is "Speaker N". ``resolve`` maps a name to its
    speaker label, or None when the line does not start with a speaker; without
    it every ``Name:`` prefix is a speaker. Text before the first speaker line
    belongs to ``default_speaker``.
    """
    tokens = _normalize_script_tokens(script)
    stack: list[tuple[str, list[Any]]] = [("sequential", [])]

    def flush_turns(mode: str, items: list[Any], pending_turns: list[SpeechTurn]) -> None:
        if not pending_turns:
            return
        if mode == "parallel":
            for turn in pending_turns:
                items.append(SpeechNode(turns=(turn,)))
        else:
            items.append(SpeechNode(turns=tuple(pending_turns)))
        pending_turns.clear()

    pending_stack: list[list[SpeechTurn]] = [[]]
    attrs_stack: list[dict[str, Any]] = [{}]  # each open <parallel>'s room and stage
    active_speaker = default_speaker

    for token in tokens:
        lower = token.lower()
        if CONTAINER_TAG_RE.fullmatch(token) and lower.startswith("<parallel"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            stack.append(("parallel", []))
            pending_stack.append([])
            attrs_stack.append(parse_parallel_attrs(token, resolve))
            continue
        if lower.startswith("</parallel") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            if len(stack) > 1 and mode == "parallel":
                stack.pop()
                pending_stack.pop()
                attrs = attrs_stack.pop()
                parent_mode, parent_items = stack[-1]
                flush_turns(parent_mode, parent_items, pending_stack[-1])
                if items:
                    parent_items.append(ParallelNode(children=tuple(items), **attrs))
            continue
        if CONTAINER_TAG_RE.fullmatch(token) and lower.startswith("<sequential"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            stack.append(("sequential", []))
            pending_stack.append([])
            attrs_stack.append({})
            continue
        if lower.startswith("</sequential") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            if len(stack) > 1 and mode == "sequential":
                stack.pop()
                pending_stack.pop()
                attrs_stack.pop()
                parent_mode, parent_items = stack[-1]
                flush_turns(parent_mode, parent_items, pending_stack[-1])
                if items:
                    parent_items.append(SequentialNode(children=tuple(items)))
            continue

        sound_node = parse_sound_tag(token)
        if sound_node is not None:
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            items.append(sound_node)
            continue

        pause_only = PAUSE_TAG_RE.fullmatch(token)
        if pause_only:
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            seconds = float(pause_only.group(1))
            if seconds > 0:
                items.append(PauseNode(seconds=min(seconds, 30.0)))
            continue

        raw_body = token
        speaker_match = SPEAKER_LINE_RE.match(token)
        named = None if speaker_match else NAME_LINE.match(token)
        name = f"Speaker {int(speaker_match.group(1))}" if speaker_match else named["name"] if named else None
        label = (resolve(name) if resolve else " ".join(name.split())) if name else None
        if label:
            active_speaker = label
            raw_body = speaker_match.group(2) if speaker_match else named["body"]
        elif speaker_match:  # a tag for a slot the caller cannot resolve is still not spoken
            raw_body = speaker_match.group(2)

        expanded = _expand_speaker_line(active_speaker, raw_body)
        mode, items = stack[-1]
        if mode == "parallel" and len(expanded) > 1:
            seq_children: list[TreeNode] = [
                SpeechNode(turns=(entry,)) if isinstance(entry, SpeechTurn) else entry
                for entry in expanded
            ]
            items.append(SequentialNode(children=tuple(seq_children)))
            continue
        for entry in expanded:
            if isinstance(entry, SpeechTurn):
                if mode == "parallel":
                    items.append(SpeechNode(turns=(entry,)))
                else:
                    pending_stack[-1].append(entry)
            elif isinstance(entry, PauseNode):
                flush_turns(mode, items, pending_stack[-1])
                items.append(entry)

    while len(stack) > 1:
        mode, items = stack.pop()
        pending = pending_stack.pop()
        attrs = attrs_stack.pop()
        flush_turns(mode, items, pending)
        parent_mode, parent_items = stack[-1]
        flush_turns(parent_mode, parent_items, pending_stack[-1])
        if items:
            node = (ParallelNode(children=tuple(items), **attrs) if mode == "parallel"
                    else SequentialNode(children=tuple(items)))
            parent_items.append(node)

    root_mode, root_items = stack[0]
    flush_turns(root_mode, root_items, pending_stack[0])
    return SequentialNode(children=tuple(root_items))


def is_sound_only_node(node: TreeNode) -> bool:
    """Return True if a node contains only SoundNode (and optional PauseNode) leaves."""
    if isinstance(node, SoundNode):
        return True
    if isinstance(node, (SpeechNode, PauseNode)):
        return False
    has_sound = False
    for child in node.children:
        if isinstance(child, SoundNode):
            has_sound = True
        elif isinstance(child, PauseNode):
            continue
        elif isinstance(child, (SequentialNode, ParallelNode)):
            if not is_sound_only_node(child):
                return False
            has_sound = True
        else:
            return False
    return has_sound


def iter_sound_nodes(node: TreeNode) -> list[SoundNode]:
    """Return all SoundNode leaves in depth-first order."""
    if isinstance(node, SoundNode):
        return [node]
    if isinstance(node, (SequentialNode, ParallelNode)):
        result: list[SoundNode] = []
        for child in node.children:
            result.extend(iter_sound_nodes(child))
        return result
    return []


def iter_speech_nodes(node: TreeNode) -> list[SpeechNode]:
    """Return all SpeechNode leaves in depth-first order."""
    if isinstance(node, SpeechNode):
        return [node]
    if isinstance(node, (SequentialNode, ParallelNode)):
        result: list[SpeechNode] = []
        for child in node.children:
            result.extend(iter_speech_nodes(child))
        return result
    return []


def script_speakers(script: str, *, default_speaker: str = "Speaker 0",
                    resolve: Callable[[str], str | None] | None = None) -> list[str]:
    """The labels of every speaker with a line in the script, in order of first line."""
    tree = parse_drama3_tree(script, default_speaker=default_speaker, resolve=resolve)
    return list(dict.fromkeys(turn.speaker for node in iter_speech_nodes(tree) for turn in node.turns))


def extract_script_sounds(script: str) -> list[dict[str, Any]]:
    """Extract unique `<sound>` definitions in first-appearance order."""
    tree = parse_drama3_tree(script)
    seen: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for sound in iter_sound_nodes(tree):
        key = (sound.name or sound.description).strip().lower()
        if not key:
            continue
        if key in seen:
            existing = seen[key]
            if sound.loop and not existing["loop"]:
                existing["loop"] = True
                existing["duration"] = max(existing["duration"], sound.duration)
            if not existing["description"] and sound.description:
                existing["description"] = sound.description
            continue
        item = {
            "name": sound.name or default_sound_name(sound.description),
            "description": sound.description,
            "loop": sound.loop,
            "volume": sound.volume,
            "duration": sound.duration,
        }
        seen[key] = item
        ordered.append(item)
    return ordered


def has_drama3_sound_features(script: str) -> bool:
    """Return True if the script contains any staging, sound, pause, or speaker effect tags."""
    text = script or ""
    if CONTAINER_TAG_RE.search(text):
        return True
    if SOUND_BLOCK_RE.search(text):
        return True
    if PAUSE_TAG_RE.search(text):
        return True
    if has_speaker_effects(text):
        return True
    return False


def strip_drama3_sound_tags(script: str) -> str:
    """Remove all `<parallel>`, `<sequential>`, `<sound>`, `<#...#>`, and `<speaker_effect>` tags."""
    if not script:
        return ""
    text = script.replace("\r\n", "\n")
    text = SOUND_BLOCK_RE.sub("", text)
    text = CONTAINER_TAG_RE.sub("", text)
    text = PAUSE_TAG_RE.sub("", text)
    text = strip_speaker_effects(text)
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r" {2,}", " ", raw_line).strip()
        if not line:
            continue
        match = SPEAKER_LINE_RE.match(line)
        if match and not match.group(2).strip():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
