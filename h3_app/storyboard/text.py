"""Scene boundaries in the editable Create storyboard text."""

import re

SCENE_SEPARATOR = "\n\n---\n\n"
SEPARATOR_LINE = re.compile(r"^[ \t]*---[ \t]*\r?$", re.MULTILINE)


def split_scenes(text: str) -> list[str]:
    """Blank lines belong to a prompt; only a standalone --- line starts a new scene."""
    return [scene.strip() for scene in SEPARATOR_LINE.split(str(text or "")) if scene.strip()]
