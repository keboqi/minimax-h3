"""Recursive AST parser and sound/tag utilities for Drama 3 scripts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Union

from .speaker_effects import extract_speaker_effects, has_speaker_effects, strip_speaker_effects

SPEAKER_LINE_RE = re.compile(r"^<\|speaker:(\d+)\|>\s*(.*)$")
CONTAINER_TAG_RE = re.compile(r"(</?(?:parallel|sequential)\s*>)", re.IGNORECASE)
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
    speaker: int
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


@dataclass(frozen=True)
class PauseNode:
    seconds: float


@dataclass(frozen=True)
class SequentialNode:
    children: tuple["TreeNode", ...]


@dataclass(frozen=True)
class ParallelNode:
    children: tuple["TreeNode", ...]


TreeNode = Union[
    SpeechNode,
    SoundNode,
    PauseNode,
    SequentialNode,
    ParallelNode,
]


def _parse_sound_tag(raw_tag: str) -> SoundNode | None:
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
        name = _default_sound_name(body)
    return SoundNode(
        description=body,
        name=name,
        loop=loop,
        volume=volume,
        duration=duration,
    )


def _default_sound_name(description: str) -> str:
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


def _expand_speaker_line(speaker: int, raw_body: str) -> list[SpeechTurn | PauseNode]:
    """Expand a single `<|speaker:N|>` line into SpeechTurn and PauseNode items."""
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


def parse_drama3_tree(script: str, *, default_speaker: int = 0) -> SequentialNode:
    """Parse a Drama 3 script with arbitrary `<parallel>` and `<sequential>` nesting."""
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
    active_speaker = default_speaker

    for token in tokens:
        lower = token.lower()
        if lower.startswith("<parallel") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            stack.append(("parallel", []))
            pending_stack.append([])
            continue
        if lower.startswith("</parallel") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            if len(stack) > 1 and mode == "parallel":
                stack.pop()
                pending_stack.pop()
                parent_mode, parent_items = stack[-1]
                flush_turns(parent_mode, parent_items, pending_stack[-1])
                if items:
                    parent_items.append(ParallelNode(children=tuple(items)))
            continue
        if lower.startswith("<sequential") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            stack.append(("sequential", []))
            pending_stack.append([])
            continue
        if lower.startswith("</sequential") and lower.endswith(">"):
            mode, items = stack[-1]
            flush_turns(mode, items, pending_stack[-1])
            if len(stack) > 1 and mode == "sequential":
                stack.pop()
                pending_stack.pop()
                parent_mode, parent_items = stack[-1]
                flush_turns(parent_mode, parent_items, pending_stack[-1])
                if items:
                    parent_items.append(SequentialNode(children=tuple(items)))
            continue

        sound_node = _parse_sound_tag(token)
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

        speaker_match = SPEAKER_LINE_RE.match(token)
        if speaker_match:
            active_speaker = int(speaker_match.group(1))
            raw_body = speaker_match.group(2)
        else:
            raw_body = token

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
        flush_turns(mode, items, pending)
        parent_mode, parent_items = stack[-1]
        flush_turns(parent_mode, parent_items, pending_stack[-1])
        if items:
            node = ParallelNode(children=tuple(items)) if mode == "parallel" else SequentialNode(children=tuple(items))
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
            "name": sound.name or _default_sound_name(sound.description),
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
