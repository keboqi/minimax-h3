"""A long Remake: its windows, each one's frames, prompt and take, saved beside the renders."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from h3_app.catalog import VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.media_review import review_path
from h3_app.provenance import snapshot_path
from h3_app.staging import file_content_sha256

from .model import FPS, MAX_PICTURES
from .windows import carried_frame

FOLDER = "h3/.remake"
PENDING, DONE, CHANGED, OUTDATED, FAILED = "pending", "done", "changed", "outdated", "failed"
STATE_NAMES = {PENDING: "Not generated", DONE: "Done", CHANGED: "Changed", OUTDATED: "Earlier window changed",
               FAILED: "Failed"}
RENDERED = frozenset({DONE, OUTDATED})
OPTION_DEFAULTS = {"mode": "Swap", "pinned": True, "smart": True, "auto_prompts": True, "request": ""}
_LOCK = threading.Lock()


def _copy(source: str | Path, target: Path) -> None:
    """Copy whole or not at all: an existing file is taken to be complete."""
    partial = target.with_name(f"{target.name}.{uuid.uuid4().hex}.partial")
    shutil.copy2(source, partial)
    partial.replace(target)


def fingerprint(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode("utf-8")).hexdigest()[:20]


def new_window(index: int, plan: dict[str, Any]) -> dict[str, Any]:
    return {**plan, "index": index, "state": PENDING, "pictures": [], "pins": [], "prompt": "", "prompt_auto": False,
            "prompt_pictures": 0, "take": None, "carried": None, "carried_pin": None, "use_carried": True}


class Projects:
    """Saved Remake projects shared across the workspace, keyed by selection.

    Workshop and Studio run on different origins and therefore have different
    browser owners, so projects are listed and loaded across owners just like
    Storyboards, Dubbing projects, and Media. ``owner`` records the creator and
    keeps existing ``<owner_hash>/<key>`` folders in place.
    """

    def __init__(self, output_dir: Path):
        self.root = Path(output_dir) / FOLDER

    def folder(self, owner: str, key: str) -> Path:
        return self.root / hashlib.sha256(str(owner).encode("utf-8")).hexdigest()[:16] / key

    def load(self, owner: str | None, key: str) -> dict[str, Any] | None:
        projects = self.list(owner)
        match = next((project for project in projects if project.get("key") == key), None)
        if match is not None:
            return match
        return next((project for project in projects if key in project.get("aliases", ())), None)

    def list(self, owner: str | None = None) -> list[dict[str, Any]]:
        """Saved projects in this workspace, most recently changed first."""
        if not self.root.is_dir():
            return []
        found: list[tuple[float, Path]] = []
        for path in self.root.glob("*/*/project.json"):
            try:
                found.append((path.stat().st_mtime, path))
            except OSError:
                continue
        found.sort(key=lambda item: item[0], reverse=True)
        projects: list[dict[str, Any]] = []
        seen: set[str] = set()
        for mtime, path in found:
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            key = data.get("key") or path.parent.name
            if key in seen:
                continue
            seen.add(key)
            projects.append({**data, "folder": str(path.parent), "updated": mtime})
        return projects

    def delete(self, project: dict[str, Any]) -> None:
        folder = Path(project["folder"]) if project.get("folder") else self.folder(project["owner"], project["key"])
        with _LOCK:
            shutil.rmtree(folder)

    def clear_history_takes(self, project: dict[str, Any]) -> dict[str, Any]:
        """Remove superseded or orphaned render takes in ``project``, keeping active window takes."""
        with _LOCK:
            folder = (
                Path(project["folder"]).resolve()
                if project.get("folder")
                else self.folder(project["owner"], project["key"]).resolve()
            )
            path = folder / "project.json"
            if not path.is_file():
                raise H3Error(f'Remake project "{project.get("key") or folder.name}" no longer exists.')
            current = {**json.loads(path.read_text("utf-8")), "folder": str(folder)}

            keep_files: set[Path] = {path.resolve()}
            if current.get("source"):
                try:
                    keep_files.add(Path(str(current["source"])).resolve())
                except OSError:
                    pass
            kept_takes = 0
            for window in current.get("windows") or ():
                if not isinstance(window, dict) or not window.get("take"):
                    continue
                kept_takes += 1
                try:
                    tp = Path(str(window["take"])).resolve()
                    keep_files.add(tp)
                    keep_files.add(snapshot_path(tp).resolve())
                    keep_files.add(review_path(tp).resolve())
                except OSError:
                    continue

            removed_takes = 0
            removed_files = 0
            freed_bytes = 0
            video_exts = VIDEO_EXTENSIONS | {".m4v"}

            def prune_dir(directory: Path) -> None:
                nonlocal removed_takes, removed_files, freed_bytes
                if not directory.is_dir() or directory.is_symlink():
                    return
                try:
                    entries = list(directory.iterdir())
                except OSError:
                    return
                for entry in entries:
                    try:
                        if entry.is_dir() or entry.is_symlink():
                            continue
                        if entry.resolve() in keep_files:
                            continue
                        size = entry.stat().st_size
                        entry.unlink(missing_ok=True)
                        removed_files += 1
                        freed_bytes += size
                        if entry.suffix.lower() in video_exts:
                            removed_takes += 1
                    except OSError:
                        continue

            prune_dir(folder)
            takes_dir = folder / "takes"
            prune_dir(takes_dir)
            if takes_dir.is_dir():
                try:
                    takes_dir.rmdir()
                except OSError:
                    pass

            return {
                "name": str(current.get("name") or current.get("key") or folder.name),
                "kept_takes": kept_takes,
                "removed_takes": removed_takes,
                "removed_files": removed_files,
                "freed_bytes": freed_bytes,
            }

    def create(self, owner: str, key: str, source: str, data: dict[str, Any], plans: list[dict]) -> dict[str, Any]:
        folder = self.folder(owner, key)
        (folder / "pictures").mkdir(parents=True, exist_ok=True)
        stored = folder / f"source{Path(source).suffix.lower() or '.mp4'}"
        if not stored.exists():
            _copy(source, stored)
        project = {**data, "key": key, "owner": owner, "source": str(stored), "folder": str(folder),
                   "windows": [new_window(index, plan) for index, plan in enumerate(plans)]}
        self._write(project)
        return project

    @staticmethod
    def write(project: dict[str, Any]) -> None:
        folder = Path(project["folder"])
        partial = folder / f"project.{uuid.uuid4().hex}.tmp"
        partial.write_text(
            json.dumps({k: v for k, v in project.items() if k not in {"folder", "updated"}}, indent=1),
            "utf-8",
        )
        partial.replace(folder / "project.json")

    _write = write

    def edit(self, project: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Apply ``change`` to the saved project: the run and the tab both write while a job runs."""
        with _LOCK:
            folder = Path(project["folder"]) if project.get("folder") else self.folder(project["owner"], project["key"])
            path = folder / "project.json"
            if path.is_file():
                current = {**json.loads(path.read_text("utf-8")), "folder": str(folder)}
            else:
                current = self.load(project["owner"], project["key"])
            change(current)
            self._write(current)
            return current


def with_selection(projects: Projects, project: dict[str, Any]) -> dict[str, Any]:
    """Make sure a project records the selection it reopens with.

    Earlier projects kept only its fingerprint, so they gain these values and answer to the new
    fingerprint too. Their folders don't move, because their renders are recorded by path.
    """
    if "selection" in project:
        return project
    lengths = [window["render"][1] - window["render"][0] for window in project["windows"]]
    selection = {
        "sha": file_content_sha256(Path(project["source"])), "start": project["start"],
        "end": project["start"] + project["frames"] / FPS, "short_edge": min(project["width"], project["height"]),
        "mix": project["mix"], "max_window": max(lengths),
    }
    alias = fingerprint(selection["sha"], float(selection["start"]), float(selection["end"]),
                        int(selection["short_edge"]), int(selection["mix"]), int(selection["max_window"]))

    def adopt(current):
        current.update(selection=selection, aliases=[alias], name=Path(project["source"]).stem)

    return projects.edit(project, adopt)


def store_picture(project: dict[str, Any], path: str | Path) -> str:
    """Keep a picture with the project, named by content so an unchanged one compares equal."""
    source = Path(path)
    target = Path(project["folder"]) / "pictures" / f"{file_content_sha256(source)[:24]}{source.suffix.lower()}"
    if not target.exists():
        _copy(source, target)
    return str(target)


def save_editor(window: dict[str, Any], pictures: list[str], pins: list[int | None], prompt: str,
                use_carried: bool) -> None:
    """Record the editor's values; a rendered window that really changed needs generating again."""
    # The editor may not have received a prompt Smart wrote; empty means "as it is" there.
    prompt = prompt or (window["prompt"] if window["prompt_auto"] else "")
    changed = (pictures != window["pictures"] or pins != window["pins"] or prompt != window["prompt"]
               or bool(use_carried) != window["use_carried"])
    if not changed:
        return
    if prompt != window["prompt"]:
        window["prompt_auto"] = False
    window.update(pictures=pictures, pins=pins, prompt=prompt, use_carried=bool(use_carried))
    if window["state"] in RENDERED:
        window["state"] = CHANGED


def save_options(project: dict[str, Any], options: dict[str, Any]) -> None:
    """Queue takes affected by changed conditioning or automatic prompt settings."""
    changed = {key for key, default in OPTION_DEFAULTS.items()
               if key in options and options[key] != project.get(key, default)}
    conditioning = bool(changed & {"mode", "pinned", "smart"})
    for window in project["windows"]:
        if window["state"] in RENDERED and (conditioning or (changed and window["prompt_auto"])):
            window["state"] = CHANGED
    project.update(options)


def record_take(project: dict[str, Any], index: int, take: str, *, current_inputs: bool) -> None:
    """A new take: what continues from it no longer matches, but stays usable until regenerated."""
    windows = project["windows"]
    windows[index].update(take=take, state=DONE if current_inputs else CHANGED)
    following = index + 1
    if following < len(windows) and windows[following]["state"] in RENDERED:
        windows[following]["state"] = OUTDATED


def window_inputs(project: dict[str, Any], index: int, *, pinned: bool, smart: bool) -> tuple[list[str], list]:
    """The pictures and pins a window renders with: yours first, so their tags stay put.

    Smart adds the frame carried from the previous window and, when a window has nothing of
    its own, the first window's picture to hold the identity across a chain.
    """
    window = project["windows"][index]
    pictures = list(window["pictures"])
    pins = [pin if pinned else None for pin in window["pins"]]
    if smart and index:
        if window["use_carried"] and window["carried"]:
            pictures.append(window["carried"])
            pins.append(window["carried_pin"])
        first = project["windows"][0]["pictures"]
        if not window["pictures"] and first:
            pictures.append(first[0])
            pins.append(None)
    return pictures[:MAX_PICTURES], pins[:MAX_PICTURES]


def carry_position(project: dict[str, Any], index: int) -> tuple[int, int | None]:
    return carried_frame(project["windows"][index - 1], project["windows"][index])
