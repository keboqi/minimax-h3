"""Storyboard callbacks: render a storyboard, edit it, run takes on the GPU, join and export."""

from __future__ import annotations

import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

import gradio as gr

from h3_app import storyboard as sb
from h3_app.errors import H3Error
from h3_app.generation.construct_graph import StoryboardGraph
from h3_app.jobs import CURRENT_JOB
from h3_ui.job_admission import require_owner

from .presentation import cast_html, strip_html, timeline_html


@dataclass(frozen=True)
class StoryboardServices:
    """The storyboard store, one H3 take generator, request defaults and Media registration."""

    service: Callable[[], sb.StoryboardService]
    # (generate values, StoryboardGraph, outputs dict) → GenerationUpdate iterator
    generate_take: Callable[..., Iterator[Any]]
    request_defaults: Callable[[], dict[str, Any]]
    publish: Callable[[Path, dict[str, Any]], None]


def latent_prefix_frames(path: Path) -> int:
    """``h3_prefix_frames`` from a safetensors header, without loading tensors."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > 1 << 24:
            raise H3Error(f"{path.name} is not a storyboard latent.")
        header = json.loads(handle.read(length))
    return int(header.get("__metadata__", {}).get("h3_prefix_frames", 0))


class Selection(NamedTuple):
    """The storyboard and scene an action leaves selected."""

    storyboard_id: str | None
    scene_id: str | None


def take_label(number: int, take: dict[str, Any]) -> str:
    return f"Take {number} · seed {take['seed']}"


class StoryboardHandlers:
    def __init__(self, services: StoryboardServices):
        self.services = services

    @property
    def service(self) -> sb.StoryboardService:
        return self.services.service()

    # Rendering -------------------------------------------------------------
    def choices(self, owner):
        return [(row["name"], row["id"]) for row in self.service.list(owner)]

    def render(self, owner, storyboard_id, scene_id=None, status=""):
        """Updates for ``StoryboardView.render_outputs``."""
        choices = self.choices(owner)
        board = None
        if storyboard_id and any(value == storyboard_id for _, value in choices):
            board = self.service.load(owner, storyboard_id)
        else:
            storyboard_id = choices[0][1] if choices else None
            board = self.service.load(owner, storyboard_id) if storyboard_id else None
        if board is None:
            empty = gr.update(value=None, choices=[])
            return (
                gr.update(choices=choices, value=None), "", "Render settings come from MiniMax H3 when you create a storyboard.",
                cast_html(None), empty, strip_html(None, None), "### Scene", "", "", 5, "flow",
                gr.update(choices=[], value=[]), 39, -1, 1, empty, None,
                "Every take is kept. The one in use is what later scenes continue from.",
                timeline_html(None, None), empty, gr.update(value="Generate remaining"), status,
            )
        ids = [scene["id"] for scene in board["scenes"]]
        scene_id = scene_id if scene_id in ids else ids[0]
        index = ids.index(scene_id)
        scene = board["scenes"][index]
        states = sb.scene_states(board)
        take_choices = [
            (take_label(number, board["takes"][take_id]), take_id)
            for number, take_id in enumerate(scene["takes"], 1) if take_id in board["takes"]
        ]
        in_use = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
        pending = sb.pending_scenes(board)
        note = "Every take is kept. The one in use is what later scenes continue from."
        if states[index] == sb.OUTDATED:
            note = "This scene changed since its take was made, or the scene it flows from did. Generate it again to update it."
        if board["queue"]:
            note += f"\n\n{len(board['queue'])} scene(s) were waiting when generation stopped. Press **Generate remaining** to continue."
        heading = f"### Scene {index + 1} · {states[index].capitalize()}"
        return (
            gr.update(choices=choices, value=storyboard_id),
            scene_id,
            f"**Render settings from MiniMax H3:** {sb.profile_summary(board['profile'])}",
            cast_html(board),
            gr.update(choices=[member["name"] for member in board["cast"]], value=None),
            strip_html(board, scene_id),
            heading,
            scene["title"],
            scene["prompt"],
            scene["seconds"],
            scene["link"] if index else "flow",
            gr.update(choices=[member["name"] for member in board["cast"]], value=scene["cast"]),
            scene["overlap"],
            scene["seed"],
            scene["takes_per_run"],
            gr.update(choices=take_choices, value=scene["in_use"] if in_use else None),
            in_use["video"] if in_use and Path(in_use["video"]).is_file() else None,
            note,
            timeline_html(board, scene_id),
            gr.update(choices=[(v["label"], v["id"]) for v in reversed(board["versions"])], value=None),
            gr.update(value=f"Generate remaining ({len(pending)})" if pending else "All scenes done"),
            status,
        )

    def refresh(self, storyboard_id, scene_id, request: gr.Request):
        return self.render(require_owner(request), storyboard_id, scene_id)

    def select_storyboard(self, storyboard_id, request: gr.Request):
        return self.render(require_owner(request), storyboard_id)

    def _act(self, storyboard_id, scene_id, request, action, done="", reset=()):
        """Run ``action(owner)`` and re-render; ``reset`` values follow only on success.

        The action may return a ``Selection`` to show, or a message replacing ``done``.
        """
        owner = require_owner(request)
        try:
            result = action(owner)
        except (H3Error, ValueError) as exc:
            failed = self.render(owner, storyboard_id, scene_id, f"**Couldn't do that:** {exc}")
            return (*failed, *(gr.skip() for _ in reset))
        if isinstance(result, Selection):
            storyboard_id, scene_id = result
        elif isinstance(result, str):
            done = result
        return (*self.render(owner, storyboard_id, scene_id, done), *reset)

    # Storyboards -----------------------------------------------------------
    def create(self, name, story, seconds, profile_values, request: gr.Request):
        def action(owner):
            profile = sb.make_profile(profile_values, self.services.request_defaults())
            paragraphs = [part.strip() for part in str(story or "").split("\n\n") if part.strip()]
            scenes = [
                sb.new_scene(f"Scene {number}", text, seconds=int(seconds), link=sb.FLOW)
                for number, text in enumerate(paragraphs, 1)
            ]
            return Selection(self.service.create(owner, name, profile, scenes=scenes), None)
        return self._act(None, None, request, action, "Storyboard created.")

    def apply_profile(self, storyboard_id, scene_id, profile_values, request: gr.Request):
        def action(owner):
            profile = sb.make_profile(profile_values, self.services.request_defaults())
            self.service.apply_profile(owner, storyboard_id, profile)
        return self._act(storyboard_id, scene_id, request, action,
                         "Applied the current MiniMax H3 settings. Scenes made with the old settings are outdated.")

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
        return self._act(storyboard_id, scene_id, request, action, "Storyboard deleted.", reset=(False,))

    def add_cast(self, storyboard_id, scene_id, name, image, request: gr.Request):
        def action(owner):
            if not image:
                raise H3Error("Choose a cast picture first.")
            self.service.add_cast(owner, storyboard_id, name, image)
        return self._act(storyboard_id, scene_id, request, action, "Added to the cast.", reset=("", None))

    def remove_cast(self, storyboard_id, scene_id, name, request: gr.Request):
        def action(owner):
            self.service.remove_cast(owner, storyboard_id, name)
        return self._act(storyboard_id, scene_id, request, action, "Removed from the cast.")

    # Scenes ----------------------------------------------------------------
    def save_scene(self, storyboard_id, scene_id, title, prompt, seconds, link, cast, overlap, seed,
                   takes_per_run, request: gr.Request):
        if not storyboard_id or not scene_id:
            return self.refresh(storyboard_id, scene_id, request)

        def action(owner):
            self.service.update_scene(
                owner, storyboard_id, scene_id, title=str(title or "").strip()[:120], prompt=str(prompt or ""),
                seconds=int(seconds), link=link, cast=list(cast or []), overlap=int(overlap),
                seed=int(seed if seed is not None else -1), takes_per_run=int(takes_per_run),
            )
        return self._act(storyboard_id, scene_id, request, action)

    def strip_action(self, storyboard_id, scene_id, evt: gr.EventData, request: gr.Request):
        """A click in the scene strip or timeline: select, add a scene, or switch a link."""
        action = getattr(evt, "action", "select")
        target = getattr(evt, "scene_id", "") or scene_id

        def run(owner):
            if action == "add":
                return Selection(storyboard_id, self.service.add_scene(owner, storyboard_id)[1])
            if action == "link":
                self.service.toggle_link(owner, storyboard_id, target)
            return Selection(storyboard_id, target)
        return self._act(storyboard_id, scene_id, request, run)

    def duplicate_scene(self, storyboard_id, scene_id, request: gr.Request):
        return self._act(storyboard_id, scene_id, request, lambda owner: Selection(
            storyboard_id, self.service.duplicate_scene(owner, storyboard_id, scene_id)[1]), "Scene duplicated.")

    def move_scene(self, offset):
        def move(storyboard_id, scene_id, request: gr.Request):
            def action(owner):
                self.service.move_scene(owner, storyboard_id, scene_id, offset)
            return self._act(storyboard_id, scene_id, request, action)
        return move

    def delete_scene(self, storyboard_id, scene_id, request: gr.Request):
        def action(owner):
            self.service.delete_scene(owner, storyboard_id, scene_id)
            return Selection(storyboard_id, None)
        return self._act(storyboard_id, scene_id, request, action, "Scene deleted.")

    def use_take(self, storyboard_id, scene_id, take_id, request: gr.Request):
        def action(owner):
            if not take_id:
                raise H3Error("Choose a take first.")
            board = self.service.use_take(owner, storyboard_id, scene_id, take_id)
            affected = sb.affected_by(board, self.service.scene_index(board, scene_id))
            if not affected:
                return "Take in use changed."
            numbers = ", ".join(str(index + 1) for index in affected)
            return (f"Take in use changed. Scene {numbers} flowed from the old take and is now outdated. "
                    "Use **Generate remaining** to update it; scenes after a cut stay as they are.")
        return self._act(storyboard_id, scene_id, request, action)

    # Versions --------------------------------------------------------------
    def save_version(self, storyboard_id, scene_id, label, request: gr.Request):
        def action(owner):
            self.service.save_version(owner, storyboard_id, label)
        return self._act(storyboard_id, scene_id, request, action, "Version saved.", reset=("",))

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
        return str(path), "Joined preview ready. Flow joins drop the repeated overlap; cuts join hard."

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
        return str(path), f"Exported **{path.name}** to Media and saved a version."

    # Generation ------------------------------------------------------------
    def _take(self, owner, storyboard_id, index, seed):
        """Generate one take of scene ``index``; yields status text."""
        service = self.service
        board = service.load(owner, storyboard_id)
        scene = board["scenes"][index]
        source = None
        if sb.flows_from_previous(board, index):
            previous = board["scenes"][index - 1]
            source = board["takes"].get(previous["in_use"]) if previous["in_use"] else None
            if source is None or sb.scene_states(board)[index - 1] != sb.DONE:
                raise H3Error(f"Scene {index} must be done before scene {index + 1} can flow on from it.")
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
        owner = require_owner(request)
        if not storyboard_id:
            raise H3Error("Create a storyboard first.")
        board = self.service.load(owner, storyboard_id)
        queue = [board["scenes"][index]["id"] for index in sb.pending_scenes(board)]
        if not queue:
            yield "Every scene is done.", gr.update()
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
        if queue:
            yield f"{len(queue)} scene(s) are waiting for an earlier scene.", gr.update()
        else:
            yield "All queued scenes generated.", gr.update()
