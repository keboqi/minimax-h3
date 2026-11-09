"""One lock per saved Soundstage project, shared by every part of the page that changes it.

A project is saved whole: each change reads it, edits it and writes all of it back. Cast jobs
write clips into it from their own threads while the page saves settings, slots, the script or
the video choices, so every change takes the project's lock; otherwise a change that read the
project before a clip arrived would write it back without that clip.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import Lock, RLock
from typing import Any, Callable, Iterator

from .controls import DRAMA3_VOICE_SLOTS, DRAMA3_SOUND_SLOTS

_LOCKS: dict[str, RLock] = {}
_GUARD = Lock()


def project_lock(name: str) -> RLock:
    """The lock for project ``name`` (re-entrant, so a change may make another change)."""
    with _GUARD:
        return _LOCKS.setdefault(str(name), RLock())


@contextmanager
def editing(store, name: str) -> Iterator[dict]:
    """Hold the project's lock, yield its saved data to edit in place, then save it.

    Nothing is saved when the block raises.
    """
    with project_lock(name):
        with store.edit_soundstage_project(name) as data:
            before = deepcopy(data)
            yield data
            for kind, count, shown in (("voice", DRAMA3_VOICE_SLOTS, "shown_slots"),
                                       ("sound", DRAMA3_SOUND_SLOTS, "shown_sound_slots")):
                for slot in range(count):
                    prefix = f"{kind}_{slot}_"
                    old = {key: val for key, val in before.items() if key.startswith(prefix)}
                    new = {key: val for key, val in data.items() if key.startswith(prefix)}
                    visibility_changed = (slot < int(before.get(shown) or 1)) != (slot < int(data.get(shown) or 1))
                    if old != new or visibility_changed:
                        versions = data.setdefault("_slot_versions", {})
                        key = f"{kind}:{slot}"
                        versions[key] = versions.get(key, 0) + 1


def edit_project(store, name: str, change: Callable[[dict], Any]) -> Any:
    """Read, change and save project ``name`` under its lock; returns what ``change`` returns."""
    with editing(store, name) as data:
        return change(data)
