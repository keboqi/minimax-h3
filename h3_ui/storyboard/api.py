"""API-only Storyboard endpoints, for scripted clients that plan a storyboard elsewhere.

The page's own controls are browser-only. A client imports a whole storyboard in one
call (scenes with their modes, frames, references and cast pictures), then lists
storyboards or reads one's scene states and what each scene still needs. Generating
stays on the page: open the storyboard on the GPU endpoint and generate its scenes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from h3_app import storyboard as sb
from h3_app.errors import H3Error
from h3_ui.job_admission import require_owner

SCENE_KEYS = frozenset({"title", "prompt", "seconds", "link", "mode", "cast", "overlap", "seed",
                        "takes_per_run", *sb.MEDIA_FIELDS})
BOARD_KEYS = frozenset({"name", "settings", "cast", "scenes"})


def _uploads(files) -> dict[str, str]:
    """Uploaded files by their file name, the name the spec refers to them by."""
    named = {}
    for item in files or []:
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        path = Path(str(item))
        if path.name in named:
            raise H3Error(f"Two uploaded files are named {path.name}; give each file its own name.")
        named[path.name] = str(path)
    return named


def _unknown(keys, allowed, where):
    extra = sorted(set(keys) - allowed)
    if extra:
        raise H3Error(f"Unknown {where} field(s): {', '.join(extra)}.")


class StoryboardApiHandlers:
    """Import a storyboard, list storyboards and read one's scene states."""

    def storyboard_import(self, spec, files, request: gr.Request = None):
        """Create a storyboard from a JSON spec; media fields name files uploaded in ``files``."""
        owner = require_owner(request)
        try:
            return self._import(owner, spec, files)
        except (H3Error, ValueError, TypeError, KeyError) as exc:
            raise gr.Error(f"Couldn't import the storyboard: {exc}") from exc

    def _import(self, owner, spec, files):
        data = json.loads(spec) if isinstance(spec, str) else spec
        if not isinstance(data, dict) or not data.get("scenes"):
            raise H3Error("The spec is a JSON object with at least one scene.")
        _unknown(data, BOARD_KEYS, "storyboard")
        uploads = _uploads(files)

        def upload(name):
            name = Path(str(name)).name
            if name not in uploads:
                raise H3Error(f"{name} is named in the spec but was not uploaded.")
            return uploads[name]

        settings = data.get("settings") or {}
        _unknown(settings, set(sb.PROFILE_FIELDS), "settings")
        defaults = self.services.request_defaults()
        profile = sb.make_profile({**defaults, **settings}, defaults)
        cast = [(member["name"], upload(member["image"])) for member in data.get("cast") or []]
        if len(cast) > sb.model.MAX_CAST:
            raise H3Error(f"A storyboard has at most {sb.model.MAX_CAST} cast pictures.")
        names = {str(name).strip() for name, _ in cast}
        scenes, media = [], []
        for number, item in enumerate(data["scenes"], 1):
            _unknown(item, SCENE_KEYS, f"scene {number}")
            prompt = str(item.get("prompt") or "").strip()
            missing = sorted(set(item.get("cast") or []) - names)
            if missing:
                raise H3Error(f"Scene {number} casts {', '.join(missing)}, who is not in the cast.")
            seconds = item.get("seconds", 5)
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds != int(seconds):
                raise H3Error(f"Scene {number}: a scene is {sb.SCENE_SECONDS[0]} to {sb.SCENE_SECONDS[-1]} "
                              "whole seconds long.")
            scene = sb.new_scene(item.get("title") or sb.title_from(prompt) or f"Scene {number}", prompt,
                                 seconds=int(seconds), link=item.get("link", sb.FLOW),
                                 cast=item.get("cast") or (), mode=item.get("mode", sb.TEXT))
            for field in ("overlap", "seed", "takes_per_run"):
                if item.get(field) is not None:
                    scene[field] = int(item[field])
            try:
                sb.model.validate_scene(scene)
            except ValueError as exc:
                raise H3Error(f"Scene {number}: {exc}") from exc
            scenes.append(scene)
            # Frames are one file; reference fields are lists of files.
            media.append({field: ([upload(name) for name in item[field]] if isinstance(item[field], list)
                                  else upload(item[field]))
                          for field in sb.MEDIA_FIELDS if item.get(field)})
        storyboard_id = self.service.create(owner, str(data.get("name") or ""), profile, scenes=scenes, cast=cast)
        try:
            for scene, values in zip(scenes, media):
                for field, value in values.items():
                    self.service.set_scene_media(owner, storyboard_id, scene["id"], field, value)
        except Exception:
            # Nothing half-imported is left behind.
            self.service.delete(owner, storyboard_id)
            raise
        return self._details(owner, storyboard_id)

    def storyboard_list(self, request: gr.Request = None):
        """Every storyboard on the deployment, newest first."""
        owner = require_owner(request)
        return [{"id": row["id"], "name": row["name"], "updated": row["updated"]} for row in self.service.list(owner)]

    def storyboard_details(self, storyboard_id, request: gr.Request = None):
        """A storyboard's scenes: how each starts, its state and what it still needs."""
        owner = require_owner(request)
        try:
            return self._details(owner, str(storyboard_id or ""))
        except (H3Error, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    def _details(self, owner, storyboard_id) -> dict[str, Any]:
        board = self.service.load(owner, storyboard_id)
        states = sb.scene_states(board)
        scenes = [{
            "id": scene["id"], "number": index + 1, "title": scene["title"], "seconds": int(scene["seconds"]),
            "link": scene["link"] if index else None, "runs_as": sb.runs_as(board, index),
            "cast": list(scene["cast"]), "state": states[index], "issues": sb.scene_issues(board, index),
            "takes": len(scene["takes"]),
        } for index, scene in enumerate(board["scenes"])]
        return {"id": storyboard_id, "name": board["name"], "settings": sb.profile_summary(board["profile"]),
                "cast": [member["name"] for member in board["cast"]],
                "total_seconds": sb.total_seconds(board), "scenes": scenes}
