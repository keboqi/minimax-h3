"""How densely a Drama 3 script is sound-designed, for comparing writers and prompt changes.

Pure text analysis of an Advanced script: no audio is generated or measured.
"""

from __future__ import annotations

import re

from h3_app.drama3.serving.tree import ParallelNode, SequentialNode, SoundNode, SpeechNode, parse_drama3_tree

_WIDE_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def spoken_words(text: str) -> int:
    """English words, counting two CJK characters as one word."""
    return len(_WIDE_RE.findall(text)) // 2 + len(_WIDE_RE.sub(" ", text).split())


def _is_master(node) -> bool:
    """A scene master: a parallel holding a looped bed directly, or setting a room or stage."""
    return isinstance(node, ParallelNode) and (
        bool(node.room or node.stage)
        or any(isinstance(child, SoundNode) and child.loop and not child.anchor for child in node.children))


def _walk(node, scene: dict) -> None:
    if isinstance(node, SpeechNode):
        for turn in node.turns:
            scene["lines"] += 1
            scene["words"] += spoken_words(turn.text)
    elif isinstance(node, SoundNode):
        scene["names"].add(node.name or node.description)
        if node.loop and not node.anchor:
            scene["beds"] += 1
        else:
            scene["events"] += 1
            key = node.role or "unlabelled"
            scene["roles"][key] = scene["roles"].get(key, 0) + 1
    elif isinstance(node, (ParallelNode, SequentialNode)):
        if isinstance(node, ParallelNode) and node.room:
            scene["room"] = node.room
        for child in node.children:
            _walk(child, scene)


def sound_density_report(script: str) -> dict:
    """Per-scene and total counts of lines, words, sound events, beds and assets.

    Scenes are the script's top-level scene masters; dry top-level content between them is
    counted as its own scene.
    """
    def new_scene():
        return {"lines": 0, "words": 0, "events": 0, "beds": 0, "room": "", "roles": {}, "names": set()}

    scenes: list[dict] = []
    dry = None
    for node in parse_drama3_tree(script).children:
        if _is_master(node):
            dry = None
            scene = new_scene()
            _walk(node, scene)
            scenes.append(scene)
        else:
            if dry is None:
                dry = new_scene()
                scenes.append(dry)
            _walk(node, dry)
    scenes = [scene for scene in scenes if scene["lines"] or scene["events"] or scene["beds"]]
    names = set().union(*(scene.pop("names") for scene in scenes)) if scenes else set()
    lines = sum(scene["lines"] for scene in scenes)
    words = sum(scene["words"] for scene in scenes)
    events = sum(scene["events"] for scene in scenes)
    return {
        "scenes": scenes,
        "lines": lines,
        "words": words,
        "events": events,
        "scenes_with_bed": sum(1 for scene in scenes if scene["beds"]),
        "assets": len(names),
        "events_per_10_lines": round(10 * events / lines, 1) if lines else 0.0,
        "events_per_100_words": round(100 * events / words, 1) if words else 0.0,
    }


def density_summary(script: str) -> str:
    """One status sentence summarizing ``sound_density_report``, or "" for a script without sounds."""
    try:
        report = sound_density_report(script)
    except Exception:  # a report must never fail the writer
        return ""
    if not report["events"] and not report["scenes_with_bed"]:
        return ""
    count = len(report["scenes"])
    return (f"Sound design: {report['events']} cues ({report['events_per_10_lines']:g} per 10 lines), "
            f"beds in {report['scenes_with_bed']} of {count} scene{'s' if count != 1 else ''}, "
            f"{report['assets']} assets.")
