"""View-only links to finished dubs, kept apart from the sessions so links outlive them.

A share is a file named after its unguessable token, in the private workspace. It names
the dub's output, so the link follows that dub's player: a later word alignment shows up,
and deleting the dub from Media ends the link. A dub has at most one link.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Iterator

from h3_app.errors import H3Error
from h3_models import write_json_atomic

from . import session as sessions
from .results import Player, load_player

_LOCK = threading.RLock()
TOKEN = re.compile(r"[A-Za-z0-9_-]{32}")


@dataclass(frozen=True)
class Share:
    player: Player
    title: str


def _root(runtime) -> Path:
    return sessions.sessions_dir(runtime).parent / "dubbing-shares"


def _output(runtime, name: str) -> Path:
    """Dubs are found by file name, so a volume mounted at another path still works."""
    return Path(runtime.outputs_dir) / "dubbing" / Path(str(name)).name


def _entries(runtime) -> Iterator[tuple[Path, dict]]:
    for path in _root(runtime).glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("output"), str):
            yield path, data


def find(runtime, output: str | Path) -> str | None:
    """The token of this dub's link, if it has one."""
    name = Path(str(output)).name
    return next((path.stem for path, data in _entries(runtime) if data["output"] == name), None)


def create(runtime, output: str | Path, title: str) -> str:
    """This dub's link token, made on first use."""
    name = Path(str(output)).name
    if load_player(_output(runtime, name)) is None:
        raise H3Error("This dub has no saved player to share; generate it again.")
    with _LOCK:
        token = find(runtime, name)
        if token is None:
            token = secrets.token_urlsafe(24)
            root = _root(runtime)
            root.mkdir(parents=True, exist_ok=True)
            write_json_atomic(root / f"{token}.json", {"output": name, "title": str(title or "")})
        return token


def revoke(runtime, output: str | Path) -> bool:
    """End this dub's link; True when it had one."""
    with _LOCK:
        token = find(runtime, output)
        if token is None:
            return False
        (_root(runtime) / f"{token}.json").unlink(missing_ok=True)
        return True


def load(runtime, token: str) -> Share | None:
    """What a link shows, or None when the link is unknown, ended, or its dub is gone."""
    if not TOKEN.fullmatch(str(token or "")):
        return None
    try:
        data = json.loads((_root(runtime) / f"{token}.json").read_text(encoding="utf-8"))
        player = load_player(_output(runtime, data["output"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return Share(player, str(data.get("title") or "")) if player is not None else None
