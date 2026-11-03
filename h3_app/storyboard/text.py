"""Scene boundaries in the editable Create storyboard text.

A standalone ``---`` line separates scenes. ``---N---`` also gives the scene after it
a length of N seconds, so the first scene can carry one too; scenes without one use
the storyboard's usual length.
"""

import re

SCENE_SEPARATOR = "\n\n---\n\n"
SEPARATOR_LINE = re.compile(r"^[ \t]*---(?:[ \t]*(\d+)[ \t]*---)?[ \t]*\r?$", re.MULTILINE)


def timed_separator(seconds: int) -> str:
    return f"---{int(seconds)}---"


def parse_scenes(text: str) -> list[tuple[str, int | None]]:
    """(prompt, seconds or None) per scene; blank lines belong to a prompt."""
    scenes, seconds, start = [], None, 0
    text = str(text or "")
    for match in SEPARATOR_LINE.finditer(text):
        prompt = text[start:match.start()].strip()
        if prompt:
            scenes.append((prompt, seconds))
            seconds = None
        # A timing line times the scene after it, even with no prompt before it.
        if match.group(1) is not None:
            seconds = int(match.group(1))
        start = match.end()
    prompt = text[start:].strip()
    if prompt:
        scenes.append((prompt, seconds))
    return scenes


def split_scenes(text: str) -> list[str]:
    """Only a standalone --- or ---N--- line starts a new scene."""
    return [prompt for prompt, _ in parse_scenes(text)]


def join_scenes(scenes) -> str:
    """Scene text with each scene's length on the separator line before it."""
    return "\n\n".join(f"{timed_separator(seconds)}\n\n{prompt.strip()}" for prompt, seconds in scenes)
