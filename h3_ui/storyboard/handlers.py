"""Storyboard callbacks: render the page, edit scenes, run takes on the GPU, join and export.

Edit callbacks return a dict keyed by ``view.RENDER_FIELDS`` names (plus any extra
fields an event resets); the bindings turn it into that event's outputs.
"""

from __future__ import annotations

import json
import random
import struct
from time import monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

import gradio as gr

from h3_app import storyboard as sb
from h3_app.errors import H3Error
from h3_app.generation.construct_graph import StoryboardGraph
from h3_app.jobs import CURRENT_JOB
from h3_app.storyboard.text import parse_scenes
from h3_app.storyboard.writer import rewrite_story
from h3_ui.job_admission import require_owner

from .api import StoryboardApiHandlers
from .presentation import cast_html, clock, scene_list_html, seconds_label, timeline_html

TAKE_NOTE = "Every take is kept. Generate again for another version, then choose which one to use."
# The prompt writer settings come from the MiniMax H3 tab, in enhance_h3_prompt's order.
REFERENCE_FIELDS = (
    *(f"ref_image_{number}" for number in range(1, 10)),
    *(f"ref_video_{number}" for number in range(1, 4)),
    *(f"ref_audio_{number}" for number in range(1, 4)),
    *(f"fl2va_audio_{number}" for number in range(1, 4)),
)
WRITER_FIELDS = (
    "prompt_writer_backend", "local_prompt_base_model", "local_prompt_max_tokens", "local_prompt_temperature",
    "local_prompt_top_p", "local_prompt_greedy", "local_prompt_seed", "gemini_prompt_model", "gemini_api_key",
    "openai_api_key", "openai_model", "openai_base_url",
)
STORY_WRITER_FIELDS = (
    "prompt_writer_backend", "gemini_prompt_model", "gemini_api_key",
    "openai_api_key", "openai_model", "openai_base_url",
)


@dataclass(frozen=True)
class StoryboardServices:
    """The storyboard store, one H3 take generator, request defaults, Media and the prompt writer."""

    service: Callable[[], sb.StoryboardService]
    # (generate values, StoryboardGraph, outputs dict) → GenerationUpdate iterator
    generate_take: Callable[..., Iterator[Any]]
    request_defaults: Callable[[], dict[str, Any]]
    publish: Callable[[Path, dict[str, Any]], None]
    # enhance_h3_prompt(**values) → (prompt, status)
    enhance_prompt: Callable[..., tuple[str, str]] | None = None


class Selection(NamedTuple):
    """The storyboard and scene an action leaves selected."""

    storyboard_id: str | None
    scene_id: str | None


def latent_prefix_frames(path: Path) -> int:
    """``h3_prefix_frames`` from a safetensors header, without loading tensors."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > 1 << 24:
            raise H3Error(f"{path.name} is not a storyboard latent.")
        header = json.loads(handle.read(length))
    return int(header.get("__metadata__", {}).get("h3_prefix_frames", 0))


def take_label(number: int, take: dict[str, Any]) -> str:
    return f"Take {number} · seed {take['seed']}"


def checklist(issues: list[str], flow: bool, previous: int) -> str:
    lines = ["**Before generating**"]
    if flow:
        lines.append(f"- ✓ Starts from the end of scene {previous} (locked)")
    lines += [f"- {issue}" for issue in issues] or ["- ✓ Everything this scene needs is filled in"]
    return "\n".join(lines)


class StoryboardHandlers(StoryboardApiHandlers):
    def __init__(self, services: StoryboardServices):
        self.services = services

    @property
    def service(self) -> sb.StoryboardService:
        return self.services.service()

    # Rendering -------------------------------------------------------------
    def choices(self, owner):
        return [(row["name"], row["id"]) for row in self.service.list(owner)]

    def render(self, owner, storyboard_id, scene_id=None, status="", *, previous_scene_id=None) -> dict[str, Any]:
        choices = self.choices(owner)
        if not any(value == storyboard_id for _, value in choices):
            storyboard_id = choices[0][1] if choices else None
        if storyboard_id is None:
            return {
                "start_group": gr.update(visible=True), "board_group": gr.update(visible=False),
                "storyboard": gr.update(choices=[], value=None), "scene_id": "", "status": status,
                "confirm_group": gr.update(visible=False),
            }
        board = self.service.load(owner, storyboard_id)
        ids = [scene["id"] for scene in board["scenes"]]
        scene_id = scene_id if scene_id in ids else ids[0]
        index = ids.index(scene_id)
        scene = board["scenes"][index]
        media = sb.scene_media(scene)
        states = sb.scene_states(board)
        flow = sb.flows_from_previous(board, index)
        previous = board["scenes"][index - 1] if index else None
        issues = sb.scene_issues(board, index, states)
        cast_names = [member["name"] for member in board["cast"]]
        take_choices = [
            (take_label(number, board["takes"][take_id]), take_id)
            for number, take_id in enumerate(scene["takes"], 1) if take_id in board["takes"]
        ]
        in_use = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
        if index == 0:
            context = "This is the opening scene."
        elif flow:
            context = f"Comes after scene {index} “{previous['title']}” and continues from its last frames."
        else:
            context = f"Comes after scene {index} “{previous['title']}” as a fresh start (a new shot)."
        tags = sb.picture_tags(board, scene)
        if media["mode"] == sb.REFERENCES:
            prompt_hint = "Refer to references by name or tag, for example " + (
                f"“{scene['cast'][0]}” or <Picture 1>." if scene["cast"] else "<Picture 1>.")
        elif flow:
            prompt_hint = f"Describe what happens after the hand-over from scene {index}."
        else:
            prompt_hint = "Describe the action, the camera and the sound."
        added = sb.new_seconds(board, index)
        if flow:
            new_seconds = (f"Adds about {seconds_label(added)} of new footage; the first "
                           f"{seconds_label(scene['seconds'] - added)} repeats scene {index}.")
        else:
            new_seconds = f"Adds {seconds_label(added)}."
        note = TAKE_NOTE if in_use else "No take yet."
        if states[index] == sb.OUTDATED:
            note = ("This take is out of date: the scene, or the scene it continues from, changed since it "
                    "was made. Generate it again to update it.")
        if board["queue"]:
            note += (f"\n\n{len(board['queue'])} scene(s) remain in the generation queue. "
                     "Press **Generate all remaining scenes** to continue.")
        pending = len(states) - states.count(sb.DONE)
        pictures = []
        players = {}
        if media["mode"] == sb.REFERENCES:
            _, cast_images = sb.model.cast_prompt(board, scene)
            pictures = list(zip([*cast_images, *media["ref_images"]], tags))
        voiced = media["mode"] in (sb.FRAMES, sb.REFERENCES)
        for kind in ("video", "audio"):
            shown = media["mode"] == sb.REFERENCES if kind == "video" else voiced
            paths = media[f"ref_{kind}s"] if shown else []
            for i in range(3):
                players[f"{kind}_preview_{i}"] = gr.update(value=paths[i] if i < len(paths) else None,
                                                           visible=i < len(paths))
        updates = {
            "start_group": gr.update(visible=False),
            "board_group": gr.update(visible=True),
            "storyboard": gr.update(choices=choices, value=storyboard_id),
            "scene_id": scene_id,
            "profile_summary": (f"Uses your MiniMax H3 settings: {sb.profile_summary(board['profile'])} · "
                                f"{len(ids)} scenes, about {clock(sb.total_seconds(board))} in total"),
            "status": status,
            "scene_list": scene_list_html(board, scene_id, states),
            "scene_heading": (f"### Scene {index + 1} of {len(ids)} · {scene['title'] or 'Untitled scene'}\n"
                              f"Runs as **{sb.runs_as(board, index)}**"),
            "scene_context": context,
            "start": gr.update(
                choices=[(f"Continue scene {index}: picks up from its last frames, so motion and sound carry on",
                          sb.FLOW),
                         ("Start fresh: a new shot that works like a MiniMax H3 clip, with every input", sb.CUT)],
                value=scene["link"] if index else sb.FLOW, visible=index > 0),
            "opening_note": gr.update(visible=index == 0),
            "mode": gr.update(choices=[
                (sb.TEXT, sb.TEXT),
                ("Last frame" if flow else sb.FRAMES, sb.FRAMES),
                (f"{sb.REFERENCES} (recommended)" if flow else sb.REFERENCES, sb.REFERENCES),
            ], value=media["mode"]),
            "frames_group": gr.update(visible=media["mode"] == sb.FRAMES),
            "first_image": media["first_image"],
            "first_frame_box": gr.update(visible=not flow),
            "no_first_note": gr.update(
                value=(f"**No first frame.** This scene opens on scene {index}'s last frames. "
                       "To open on a picture of your own, choose **Start fresh** above."),
                visible=flow),
            "last_image": gr.update(value=media["last_image"],
                                    label="Last frame (required)" if flow else "Last frame (optional)"),
            "refs_group": gr.update(visible=media["mode"] == sb.REFERENCES),
            "cast": gr.update(choices=cast_names, value=[name for name in scene["cast"] if name in cast_names],
                              info=None if cast_names else "No cast yet: add people under Cast below."),
            "ref_images": media["ref_images"] or None,
            "ref_videos": media["ref_videos"] or None,
            "voices_group": gr.update(visible=voiced),
            "ref_audios": gr.update(
                value=media["ref_audios"] or None,
                label=("Voices (up to 3): <Audio 1> to <Audio 3>, for new dialogue with these frames"
                       if media["mode"] == sb.FRAMES else "Reference audio (up to 3)")),
            "image_previews": gr.update(value=pictures, visible=bool(pictures)),
            **players,
            "tag_hint": "In the prompt: " + ", ".join(tags) if tags else "Tick cast or add an image, video or audio.",
            "prompt": scene["prompt"],
            "prompt_hint": prompt_hint,
            "title": scene["title"],
            "seconds": scene["seconds"],
            "added_seconds": new_seconds,
            "checklist": checklist(issues, flow, index),
            "generate": gr.update(
                value="Generate another take" if scene["takes"] else "Generate this scene",
                interactive=not issues),
            "overlap": gr.update(value=scene["overlap"], interactive=flow),
            "seed": scene["seed"],
            "takes_per_run": scene["takes_per_run"],
            "take_video": in_use["video"] if in_use and Path(in_use["video"]).is_file() else None,
            "takes": gr.update(choices=take_choices, value=scene["in_use"] if in_use else None,
                               visible=len(take_choices) > 1),
            "use_take": gr.update(visible=len(take_choices) > 1),
            "take_note": note,
            "confirm_group": gr.update(visible=False),
            "confirm_text": "",
            "timeline": timeline_html(board, scene_id, states),
            "generate_remaining": gr.update(
                value=f"Generate all remaining scenes ({pending})" if pending else "All scenes are done",
                interactive=bool(pending)),
            "cast_list": cast_html(board),
            "remove_cast_name": gr.update(choices=cast_names, value=None),
            "rename_name": board["name"],
            "versions": gr.update(choices=[(v["label"], v["id"]) for v in reversed(board["versions"])], value=None),
        }

        # Gradio postprocesses file outputs and fires their change listeners even when
        # two scenes use the same media. Only transfer fields that changed on navigation.
        if previous_scene_id in ids:
            old_scene = board["scenes"][ids.index(previous_scene_id)]
            old_media = sb.scene_media(old_scene)
            for field in sb.MEDIA_FIELDS:
                if media[field] == old_media[field]:
                    if field in ("last_image", "ref_audios"):
                        updates[field].pop("value", None)  # its label can still change
                    else:
                        updates.pop(field)
            if old_scene["in_use"] == scene["in_use"]:
                updates.pop("take_video")
        return updates

    def refresh(self, storyboard_id, scene_id, request: gr.Request):
        return self.render(require_owner(request), storyboard_id, scene_id)

    def select_storyboard(self, storyboard_id, request: gr.Request):
        return self.render(require_owner(request), storyboard_id)

    def show_start(self):
        return {"start_group": gr.update(visible=True), "new_story": "", "new_name": "", "rewrite_status": ""}

    def _act(self, storyboard_id, scene_id, request, action, done="", reset=None):
        """Run ``action(owner)`` and re-render; ``reset`` fields follow only on success.

        The action may return a ``Selection`` to show, a message replacing ``done``,
        or ``False`` when nothing changed, which leaves the page as it is.
        """
        owner = require_owner(request)
        try:
            result = action(owner)
        except (H3Error, ValueError) as exc:
            # A toast, so the reason shows wherever the page is scrolled.
            gr.Warning(str(exc))
            return self.render(owner, storyboard_id, scene_id, f"**Couldn't do that:** {exc}")
        if result is False:
            return {}
        if isinstance(result, Selection):
            storyboard_id, scene_id = result
        elif isinstance(result, str):
            done = result
        if done:
            gr.Info(done)
        return {**self.render(owner, storyboard_id, scene_id, done), **(reset or {})}

    # Storyboards -----------------------------------------------------------
    def rewrite_story(self, story, connect, seconds, count, *writer_and_request):
        *writer, request = writer_and_request
        require_owner(request)
        settings = dict(zip(STORY_WRITER_FIELDS, writer, strict=True))
        settings["backend"] = settings.pop("prompt_writer_backend")
        try:
            rewritten = rewrite_story(story, connect, seconds, count,
                                      runtime=self.service.runtime(), **settings)
        except Exception as exc:
            gr.Warning(f"Couldn't rewrite the story: {exc}")
            return {"rewrite_status": f"Story unchanged. {exc}"}
        return {"new_story": rewritten,
                "rewrite_status": f"Wrote {int(count)} scenes; each ---N--- line gives the next scene's length. "
                                 "Review or edit them, then choose Create storyboard."}

    def create(self, name, story, connect, seconds, profile_values, request: gr.Request):
        def action(owner):
            defaults = self.services.request_defaults()
            profile = sb.make_profile({**defaults, **profile_values}, defaults)
            link = sb.CUT if connect == sb.CUT else sb.FLOW
            scenes = []
            for number, (text, length) in enumerate(parse_scenes(story), 1):
                if length is not None and length not in sb.SCENE_SECONDS:
                    raise H3Error(f"Scene {number} asks for {length} s; a scene is {sb.SCENE_SECONDS[0]} "
                                  f"to {sb.SCENE_SECONDS[-1]} seconds.")
                scenes.append(sb.new_scene(sb.title_from(text), text, seconds=length or int(seconds), link=link))
            return Selection(self.service.create(owner, name, profile, scenes=scenes), None)
        return self._act(None, None, request, action,
                         "Storyboard created. Check each scene, then press Generate this scene.")

    def apply_profile(self, storyboard_id, scene_id, profile_values, request: gr.Request):
        def action(owner):
            defaults = self.services.request_defaults()
            profile = sb.make_profile({**defaults, **profile_values}, defaults)
            self.service.apply_profile(owner, storyboard_id, profile)
        return self._act(storyboard_id, scene_id, request, action,
                         "Applied your current MiniMax H3 settings. Scenes made with the old ones are out of date.")

    def rename(self, storyboard_id, scene_id, name, request: gr.Request):
        def action(owner):
            self.service.rename(owner, storyboard_id, name)
        return self._act(storyboard_id, scene_id, request, action, "Renamed.")

    def duplicate(self, storyboard_id, scene_id, request: gr.Request):
        return self._act(storyboard_id, scene_id, request,
                         lambda owner: Selection(self.service.branch(owner, storyboard_id), None), "Duplicated.")

    def delete(self, storyboard_id, scene_id, confirmed, request: gr.Request):
        def action(owner):
            if not confirmed:
                raise H3Error("Tick the delete confirmation first.")
            self.service.delete(owner, storyboard_id)
            return Selection(None, None)
        return self._act(storyboard_id, scene_id, request, action, "Storyboard deleted.",
                         reset={"confirm_delete": False})

    def add_cast(self, storyboard_id, scene_id, name, image, request: gr.Request):
        def action(owner):
            if not image:
                raise H3Error("Choose a picture first.")
            self.service.add_cast(owner, storyboard_id, name, image)
        return self._act(storyboard_id, scene_id, request, action, "Added to the cast.",
                         reset={"cast_name": "", "cast_image": None})

    def remove_cast(self, storyboard_id, scene_id, name, request: gr.Request):
        def action(owner):
            self.service.remove_cast(owner, storyboard_id, name)
        return self._act(storyboard_id, scene_id, request, action, "Removed from the cast.")

    # Scenes ----------------------------------------------------------------
    def list_action(self, storyboard_id, scene_id, evt: gr.EventData, request: gr.Request):
        """A click in the scene list or timeline: select a scene, or add one."""
        action = getattr(evt, "action", "select")
        target = getattr(evt, "scene_id", "") or scene_id

        if action == "add":
            return self._act(storyboard_id, scene_id, request, lambda owner: Selection(
                storyboard_id, self.service.add_scene(owner, storyboard_id, after=None)[1]))
        return self.render(require_owner(request), storyboard_id, target, previous_scene_id=scene_id)

    def save_text(self, storyboard_id, scene_id, prompt, title, seconds, overlap, seed, takes_per_run,
                  request: gr.Request):
        if not storyboard_id or not scene_id:
            return {}

        def action(owner):
            fields = {"prompt": str(prompt or ""), "title": str(title or "").strip()[:120],
                      "seconds": int(seconds), "overlap": int(overlap),
                      "seed": int(seed if seed is not None else -1), "takes_per_run": int(takes_per_run)}
            board = self.service.load(owner, storyboard_id)
            scene = board["scenes"][self.service.scene_index(board, scene_id)]
            if all(scene.get(key) == value for key, value in fields.items()):
                return False
            self.service.update_scene(owner, storyboard_id, scene_id, **fields)
        return self._act(storyboard_id, scene_id, request, action)

    def set_start(self, storyboard_id, scene_id, link, request: gr.Request):
        def action(owner):
            self.service.update_scene(owner, storyboard_id, scene_id, link=link)
        return self._act(storyboard_id, scene_id, request, action)

    def set_mode(self, storyboard_id, scene_id, mode, request: gr.Request):
        def action(owner):
            self.service.set_mode(owner, storyboard_id, scene_id, mode)
        return self._act(storyboard_id, scene_id, request, action)

    def set_cast(self, storyboard_id, scene_id, cast, request: gr.Request):
        def action(owner):
            self.service.update_scene(owner, storyboard_id, scene_id, cast=list(cast or []))
        return self._act(storyboard_id, scene_id, request, action)

    def set_media(self, field):
        def change(storyboard_id, scene_id, value, request: gr.Request):
            if not storyboard_id or not scene_id:
                return {}
            return self._act(storyboard_id, scene_id, request, lambda owner: self.service.set_scene_media(
                owner, storyboard_id, scene_id, field, value) or False)
        change.__name__ = f"set_{field}"
        return change

    def duplicate_scene(self, storyboard_id, scene_id, request: gr.Request):
        return self._act(storyboard_id, scene_id, request, lambda owner: Selection(
            storyboard_id, self.service.duplicate_scene(owner, storyboard_id, scene_id)[1]), "Scene duplicated.")

    def move_scene(self, offset):
        def move(storyboard_id, scene_id, request: gr.Request):
            def action(owner):
                self.service.move_scene(owner, storyboard_id, scene_id, offset)
            return self._act(storyboard_id, scene_id, request, action)
        move.__name__ = "move_earlier" if offset < 0 else "move_later"
        return move

    def delete_scene(self, storyboard_id, scene_id, request: gr.Request):
        def action(owner):
            self.service.delete_scene(owner, storyboard_id, scene_id)
            return Selection(storyboard_id, None)
        return self._act(storyboard_id, scene_id, request, action, "Scene deleted.")

    def enhance(self, storyboard_id, scene_id, *writer_and_request):
        """Improve the scene's prompt with the MiniMax H3 prompt writer and its settings."""
        *writer, request = writer_and_request
        owner = require_owner(request)
        if self.services.enhance_prompt is None or not storyboard_id or not scene_id:
            return {}
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        values = sb.scene_request_values(board, index, self.services.request_defaults(), seed=0)
        settings = dict(zip(WRITER_FIELDS, writer, strict=True))
        try:
            prompt, status = self.services.enhance_prompt(
                prompt=values["prompt"], backend=settings["prompt_writer_backend"],
                local_base_model=settings["local_prompt_base_model"],
                local_max_new_tokens=settings["local_prompt_max_tokens"],
                local_temperature=settings["local_prompt_temperature"], local_top_p=settings["local_prompt_top_p"],
                local_greedy=settings["local_prompt_greedy"], local_seed=settings["local_prompt_seed"],
                gemini_model=settings["gemini_prompt_model"], gemini_api_key=settings["gemini_api_key"],
                openai_api_key=settings["openai_api_key"], mode=values["mode"],
                first_image=values["first_image"], last_image=values["last_image"],
                **{key: values[key] for key in REFERENCE_FIELDS},
                duration=values["duration"], width=values["width"], height=values["height"],
                openai_model=settings["openai_model"], openai_base_url=settings["openai_base_url"],
            )
        except Exception as exc:
            gr.Warning(f"The prompt writer failed: {exc}")
            return {}
        # The writer reports failures in its status and hands back the original prompt.
        if "failed" in str(status).lower() or not str(prompt).strip():
            gr.Warning(str(status) or "The prompt writer returned nothing.")
            return {"status": f"**Prompt unchanged.** {status}"}
        # Cast tags are added at generation; keep the writer's text without them.
        for tag in (f"{name} is <Picture {n}>." for n, name in enumerate(board["scenes"][index]["cast"], 1)):
            prompt = prompt.replace(tag, "").strip()
        self.service.update_scene(owner, storyboard_id, scene_id, prompt=prompt)
        gr.Info("Prompt improved. Edit it further if you like.")
        return self.render(owner, storyboard_id, scene_id, status)

    # Takes -----------------------------------------------------------------
    def depends_on(self, board, index) -> list[int]:
        """Later scenes that have a take made from this scene's take, through unbroken continues."""
        return [later for later in sb.affected_by(board, index) if board["scenes"][later]["in_use"]]

    def use_take(self, storyboard_id, scene_id, take_id, request: gr.Request):
        """Switch straight away, or ask first when later scenes continue from this one."""
        owner = require_owner(request)
        if not take_id:
            gr.Warning("Choose a take first.")
            return {}
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        if board["scenes"][index]["in_use"] == take_id:
            return {}
        affected = self.depends_on(board, index)
        if not affected:
            return self._act(storyboard_id, scene_id, request, lambda owner: (
                self.service.use_take(owner, storyboard_id, scene_id, take_id) and "Take in use changed."))
        numbers = " and ".join(str(later + 1) for later in affected)
        after = affected[-1] + 1
        stop = (f" Scene {after + 1} starts fresh, so it and the scenes after it stay as they are."
                if after < len(board["scenes"]) else "")
        return {
            "confirm_group": gr.update(visible=True),
            "confirm_text": (f"**Use this take for scene {index + 1}?**\n\nScene {numbers} "
                             f"continue{'s' if len(affected) == 1 else ''} from this scene and will need "
                             f"regenerating.{stop}"),
        }

    def use_take_later(self, storyboard_id, scene_id, take_id, request: gr.Request):
        def action(owner):
            self.service.use_take(owner, storyboard_id, scene_id, take_id)
            return "Take in use changed. The scenes that continue from it are out of date."
        return self._act(storyboard_id, scene_id, request, action)

    def cancel_take(self, storyboard_id, scene_id, request: gr.Request):
        return self.render(require_owner(request), storyboard_id, scene_id)

    # Versions --------------------------------------------------------------
    def save_version(self, storyboard_id, scene_id, label, request: gr.Request):
        def action(owner):
            self.service.save_version(owner, storyboard_id, label)
        return self._act(storyboard_id, scene_id, request, action, "Version saved.", reset={"version_label": ""})

    def branch(self, storyboard_id, scene_id, version_id, request: gr.Request):
        def action(owner):
            if not version_id:
                raise H3Error("Choose a saved version first.")
            return Selection(self.service.branch(owner, storyboard_id, version_id), None)
        return self._act(storyboard_id, scene_id, request, action, "Branched into a new storyboard.")

    # Joining ---------------------------------------------------------------
    def play_all(self, storyboard_id, include_outdated, request: gr.Request):
        owner = require_owner(request)
        if not storyboard_id:
            raise gr.Error("Create a storyboard first.")
        board = self.service.load(owner, storyboard_id)
        try:
            path = self.service.render(board, self.service.preview_path(storyboard_id),
                                       include_outdated=bool(include_outdated))
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        return str(path), "Full video ready. Continuing scenes drop their repeated opening; fresh starts cut."

    def export(self, storyboard_id, include_outdated, request: gr.Request):
        owner = require_owner(request)
        if not storyboard_id:
            raise gr.Error("Create a storyboard first.")
        board = self.service.load(owner, storyboard_id)
        try:
            path = self.service.render(board, self.service.export_path(board),
                                       include_outdated=bool(include_outdated))
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        self.services.publish(path, {
            "storyboard": {"id": storyboard_id, "name": board["name"], "scenes": len(board["scenes"])},
            "settings": board["profile"],
        })
        self.service.save_version(owner, storyboard_id, f"Exported {path.name}")
        gr.Info(f"Exported {path.name} to Media.")
        return str(path), f"Exported **{path.name}** to Media and saved a version."

    # Generation ------------------------------------------------------------
    def _take(self, owner, storyboard_id, index, seed):
        """Generate one take of scene ``index``; yields status text."""
        service = self.service
        board = service.load(owner, storyboard_id)
        scene = board["scenes"][index]
        sb.check_scene(board, index)
        source = None
        if sb.flows_from_previous(board, index):
            source = board["takes"][board["scenes"][index - 1]["in_use"]]
        take_id = sb.new_id()
        prefix = service.take_prefix(storyboard_id, scene["id"], take_id)
        graph = StoryboardGraph(
            output_prefix=prefix,
            latent_output_prefix=prefix,
            continuation_source=source["latent"] if source else None,
            continuation_stage_source=source.get("stage_latent") if source else None,
            continuation_frames=int(scene["overlap"]),
        )
        inputs = sb.scene_inputs(board, index)
        values = sb.scene_request_values(board, index, self.services.request_defaults(), seed)
        outputs: dict[str, str] = {}
        video = None
        for update in self.services.generate_take(values, graph, outputs):
            if update.status.startswith("Error:"):
                raise H3Error(update.status.removeprefix("Error:").strip().split("\n\n")[0])
            video = update.output or video
            yield f"Scene {index + 1} · {update.status.splitlines()[0]}"
        if video is None or "latent" not in outputs:
            raise H3Error(f"Scene {index + 1} finished without a saved take.")
        outputs["prefix_frames"] = latent_prefix_frames(service.output_path(outputs["latent"])) if source else 0
        job = CURRENT_JOB.get()
        service.record_take(owner, storyboard_id, scene["id"], take_id, video=video, latents=outputs,
                            seed=seed, inputs=inputs, job_id=job.id if job else None)

    def _run_scene(self, owner, storyboard_id, index):
        board = self.service.load(owner, storyboard_id)
        scene = board["scenes"][index]
        count = int(scene["takes_per_run"])
        base = int(scene["seed"])
        seeds = [random.randrange(0, 2**63 - 1) if base < 0 else base + number for number in range(count)]
        for number, seed in enumerate(seeds, 1):
            job = CURRENT_JOB.get()
            if job is not None:
                job.check()
            for status in self._take(owner, storyboard_id, index, seed):
                yield (f"Take {number}/{count} · " if count > 1 else "") + status

    def generate_scene(self, storyboard_id, scene_id, request: gr.Request = None):
        owner = require_owner(request)
        if not storyboard_id or not scene_id:
            raise H3Error("Select a scene first.")
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        for status in self._run_scene(owner, storyboard_id, index):
            yield status, gr.update()
        board = self.service.load(owner, storyboard_id)
        take = board["takes"].get(board["scenes"][index]["in_use"])
        yield f"Scene {index + 1} generated.", take["video"] if take else None

    def generate_remaining(self, storyboard_id, request: gr.Request = None):
        """Every scene without a current take, in order; scenes that still need input are skipped."""
        owner = require_owner(request)
        if not storyboard_id:
            raise H3Error("Create a storyboard first.")
        board = self.service.load(owner, storyboard_id)
        started = monotonic()
        queue, skipped = [], []
        for index in sb.pending_scenes(board):
            own = [issue for issue in sb.scene_issues(board, index) if not issue.startswith("Generate scene")]
            (skipped if own else queue).append(index if own else board["scenes"][index]["id"])
        if not queue:
            yield ("Nothing to generate. " + (f"Scene {', '.join(str(i + 1) for i in skipped)} still need input."
                                              if skipped else "Every scene is done.")), gr.update()
            return
        # Saved so a restart shows what was waiting; nothing resumes on its own.
        self.service.set_queue(owner, storyboard_id, queue)
        while queue:
            board = self.service.load(owner, storyboard_id)
            index = sb.next_runnable(board, queue)
            if index is None:
                break
            for status in self._run_scene(owner, storyboard_id, index):
                yield f"{len(queue)} left · {status}", gr.update()
            queue.remove(board["scenes"][index]["id"])
            self.service.set_queue(owner, storyboard_id, queue)
            yield f"Scene {index + 1} generated · {len(queue)} left.", sb.new_id()
        waiting = f" {len(queue)} scene(s) wait for a scene that needs input." if queue else ""
        if skipped:
            waiting += f" Skipped scene {', '.join(str(i + 1) for i in skipped)}: still need input."
        yield f"Generated the remaining scenes. Total generation time {clock(monotonic() - started)}.{waiting}", gr.update()

    def switch_and_regenerate(self, storyboard_id, scene_id, take_id, request: gr.Request = None):
        """Use the chosen take, then regenerate the scenes that continue from it."""
        owner = require_owner(request)
        self.service.use_take(owner, storyboard_id, scene_id, take_id)
        yield from self.generate_remaining(storyboard_id, request)
