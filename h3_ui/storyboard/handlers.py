"""Storyboard actions behind the page: create, write prompts, switch takes, join and run takes on the GPU."""

from __future__ import annotations

import json
import random
import struct
from time import monotonic
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from h3_app import storyboard as sb
from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, DEFAULT_PROMPT_WRITER_BACKEND,
)
from h3_app.errors import H3Error
from h3_app import media_review
from h3_app.generation.construct_graph import StoryboardGraph
from h3_app.jobs import CURRENT_JOB
from h3_app.storyboard.text import parse_scenes
from h3_app.storyboard.writer import rewrite_story
from h3_ui.job_admission import require_owner

from .api import StoryboardApiHandlers
from .payload import clock

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
# What a writer setting is when neither the MiniMax H3 tab nor the page gave one.
WRITER_DEFAULTS = {
    "prompt_writer_backend": DEFAULT_PROMPT_WRITER_BACKEND, "gemini_prompt_model": DEFAULT_GEMINI_PROMPT_MODEL,
    "gemini_api_key": "", "openai_api_key": "", "openai_model": DEFAULT_OPENAI_PROMPT_MODEL, "openai_base_url": None,
    "local_prompt_base_model": "", "local_prompt_max_tokens": 512, "local_prompt_temperature": 0.7,
    "local_prompt_top_p": 0.9, "local_prompt_greedy": False, "local_prompt_seed": 0,
}
GENERATE_SCENE, GENERATE_REMAINING, SWITCH_AND_REGENERATE = "scene", "remaining", "switch"


@dataclass(frozen=True)
class StoryboardServices:
    """The storyboard store, one H3 take generator, request defaults, Media and the prompt writer."""

    service: Callable[[], sb.StoryboardService]
    # (generate values, StoryboardGraph, outputs dict) → GenerationUpdate iterator
    generate_take: Callable[..., Iterator[Any]] | None
    request_defaults: Callable[[], dict[str, Any]]
    publish: Callable[[Path, dict[str, Any]], None]
    # enhance_h3_prompt(**values) → (prompt, status)
    enhance_prompt: Callable[..., tuple[str, str]] | None = None


@dataclass(frozen=True)
class MediaLibrary:
    """The media library's image index, for picking scene pictures."""

    index: Any
    refresh_page: Callable[..., Any]
    validate_path: Callable[..., Any]


def latent_prefix_frames(path: Path) -> int:
    """``h3_prefix_frames`` from a safetensors header, without loading tensors."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > 1 << 24:
            raise H3Error(f"{path.name} is not a storyboard latent.")
        header = json.loads(handle.read(length))
    return int(header.get("__metadata__", {}).get("h3_prefix_frames", 0))


class StoryboardHandlers(StoryboardApiHandlers):
    def __init__(self, services: StoryboardServices):
        self.services = services
        self.library: MediaLibrary | None = None
        # owner → the MiniMax H3 tab's values when that browser last opened Storyboard. The page
        # sits outside Gradio, so it can't read them; API keys here stay in memory only.
        self.settings: dict[str, dict[str, dict[str, Any]]] = {}
        # (owner, storyboard id) → the scene a run is generating now.
        self.running_scene: dict[tuple[str, str], str] = {}

    @property
    def service(self) -> sb.StoryboardService:
        return self.services.service()

    def capture_settings(self, owner, profile_values: dict[str, Any], writer_values: dict[str, Any]):
        self.settings[owner] = {"profile": profile_values, "writer": writer_values}

    def _profile_for(self, owner) -> dict[str, Any]:
        defaults = self.services.request_defaults()
        captured = self.settings.get(owner, {}).get("profile", {})
        return sb.make_profile({**defaults, **captured}, defaults)

    # Storyboards -----------------------------------------------------------
    def rewrite_story(self, story, connect, seconds, count, writer: dict[str, Any]) -> str:
        return rewrite_story(story, connect, seconds, count, runtime=self.service.runtime(), **writer)

    def create(self, owner, name, story, connect, seconds) -> str:
        link = sb.CUT if connect == sb.CUT else sb.FLOW
        scenes = []
        for number, (text, length) in enumerate(parse_scenes(story), 1):
            if length is not None and length not in sb.SCENE_SECONDS:
                raise H3Error(f"Scene {number} asks for {length} s; a scene is {sb.SCENE_SECONDS[0]} "
                              f"to {sb.SCENE_SECONDS[-1]} seconds.")
            scenes.append(sb.new_scene(sb.title_from(text), text, seconds=length or int(seconds), link=link))
        return self.service.create(owner, name, self._profile_for(owner), scenes=scenes)

    def apply_profile(self, owner, storyboard_id):
        if "profile" not in self.settings.get(owner, {}):
            raise H3Error("Open Storyboard from the Studio's tab so it can read your MiniMax H3 settings.")
        self.service.apply_profile(owner, storyboard_id, self._profile_for(owner))

    def enhance(self, owner, storyboard_id, scene_id, writer: dict[str, Any] | None = None) -> str:
        """Improve the scene's prompt with the MiniMax H3 prompt writer and its settings."""
        if self.services.enhance_prompt is None:
            raise H3Error("The prompt writer isn't available here.")
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        values = sb.scene_request_values(board, index, self.services.request_defaults(), seed=0)
        settings = dict(WRITER_DEFAULTS)
        # The MiniMax H3 tab's choices win; the page's fill what it left blank.
        for chosen in (writer or {}, self.settings.get(owner, {}).get("writer") or {}):
            settings.update({name: value for name, value in chosen.items() if value not in (None, "")})
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
        # The writer reports failures in its status and hands back the original prompt.
        if "failed" in str(status).lower() or not str(prompt).strip():
            raise H3Error(f"Prompt unchanged. {status or 'The prompt writer returned nothing.'}")
        # Cast tags are added at generation; keep the writer's text without them.
        for tag in (f"{name} is <Picture {n}>." for n, name in enumerate(board["scenes"][index]["cast"], 1)):
            prompt = prompt.replace(tag, "").strip()
        self.service.update_scene(owner, storyboard_id, scene_id, prompt=prompt)
        return str(status)

    # Takes -----------------------------------------------------------------
    def depends_on(self, board, index) -> list[int]:
        """Later scenes that have a take made from this scene's take, through unbroken continues."""
        return [later for later in sb.affected_by(board, index) if board["scenes"][later]["in_use"]]

    def take_switch_question(self, board, index) -> str | None:
        """What to ask before switching this scene's take, or None when nothing depends on it."""
        affected = self.depends_on(board, index)
        if not affected:
            return None
        numbers = " and ".join(str(later + 1) for later in affected)
        after = sb.affected_by(board, index)[-1] + 1
        stop = (f" Scene {after + 1} starts fresh, so it and the scenes after it stay as they are."
                if after < len(board["scenes"]) else "")
        return (f"Scene {numbers} continue{'s' if len(affected) == 1 else ''} from this scene and will need "
                f"regenerating.{stop}")

    def use_take(self, owner, storyboard_id, scene_id, take_id, *, confirmed=False) -> str | None:
        """Switch, or return the question to ask first when later scenes continue from this one."""
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        if board["scenes"][index]["in_use"] == take_id:
            return None
        question = None if confirmed else self.take_switch_question(board, index)
        if question is None:
            self.service.use_take(owner, storyboard_id, scene_id, take_id)
        return question

    # Joining ---------------------------------------------------------------
    def play_all(self, owner, storyboard_id, include_outdated) -> Path:
        board = self.service.load(owner, storyboard_id)
        return self.service.render(board, self.service.preview_path(storyboard_id),
                                   include_outdated=bool(include_outdated))

    def export(self, owner, storyboard_id, include_outdated) -> Path:
        board = self.service.load(owner, storyboard_id)
        path = self.service.render(board, self.service.export_path(board), include_outdated=bool(include_outdated))
        self.services.publish(path, {
            "storyboard": {"id": storyboard_id, "name": board["name"], "scenes": len(board["scenes"])},
            "settings": board["profile"],
        })
        self.service.save_version(owner, storyboard_id, f"Exported {path.name}")
        return path

    # Generation ------------------------------------------------------------
    def _take(self, owner, storyboard_id, scene_id, seed):
        """Generate one take of the selected scene and yield status text."""
        service = self.service
        board = service.load(owner, storyboard_id)
        index = service.scene_index(board, scene_id)
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
        # Scored in the background with the MiniMax H3 page's option, so the next take starts at once.
        captured = self.settings.get(owner, {})
        media_review.schedule(video, values.get("prompt") or scene["prompt"],
                              api_key=(captured.get("writer") or {}).get("gemini_api_key") or "",
                              enabled=bool(captured.get("review", True)))

    def _run_scene(self, owner, storyboard_id, scene_id):
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        scene = board["scenes"][index]
        self.running_scene[(owner, storyboard_id)] = scene["id"]
        count = int(scene["takes_per_run"])
        base = int(scene["seed"])
        seeds = [random.randrange(0, 2**63 - 1) if base < 0 else base + number for number in range(count)]
        for number, seed in enumerate(seeds, 1):
            job = CURRENT_JOB.get()
            if job is not None:
                job.check()
            for status in self._take(owner, storyboard_id, scene_id, seed):
                yield (f"Take {number}/{count} · " if count > 1 else "") + status

    def generate_scene(self, storyboard_id, scene_id, request=None):
        """Yields (status, the new take's video once done)."""
        owner = require_owner(request)
        if not storyboard_id or not scene_id:
            raise H3Error("Select a scene first.")
        for status in self._run_scene(owner, storyboard_id, scene_id):
            yield status, None
        board = self.service.load(owner, storyboard_id)
        index = self.service.scene_index(board, scene_id)
        take = board["takes"].get(board["scenes"][index]["in_use"])
        yield f"Scene {index + 1} generated.", take["video"] if take else None

    def generate_remaining(self, storyboard_id, request=None):
        """Every scene without a current take, in order; scenes that still need input are skipped.

        Yields (status, a new value each time a scene finishes).
        """
        owner = require_owner(request)
        if not storyboard_id:
            raise H3Error("Create a storyboard first.")
        board = self.service.load(owner, storyboard_id)
        yield from self._generate_pending(owner, storyboard_id, board, sb.pending_scenes(board))

    def _generate_pending(self, owner, storyboard_id, board, indices):
        started = monotonic()
        queue, skipped = [], []
        for index in indices:
            own = [issue for issue in sb.scene_issues(board, index) if not issue.startswith("Generate scene")]
            (skipped if own else queue).append(index if own else board["scenes"][index]["id"])
        if not queue:
            yield ("Nothing to generate. " + (f"Scene {', '.join(str(i + 1) for i in skipped)} still need input."
                                              if skipped else "All selected scenes are done.")), None
            return
        # Saved so a restart shows what was waiting; nothing resumes on its own.
        self.service.set_queue(owner, storyboard_id, queue)
        while queue:
            board = self.service.load(owner, storyboard_id)
            index = sb.next_runnable(board, queue)
            if index is None:
                break
            scene_id = board["scenes"][index]["id"]
            for status in self._run_scene(owner, storyboard_id, scene_id):
                yield f"{len(queue)} left · {status}", None
            queue.remove(scene_id)
            self.service.set_queue(owner, storyboard_id, queue)
            yield f"Scene {index + 1} generated · {len(queue)} left.", sb.new_id()
        waiting = f" {len(queue)} scene(s) wait for a scene that needs input." if queue else ""
        if skipped:
            waiting += f" Skipped scene {', '.join(str(i + 1) for i in skipped)}: still need input."
        yield f"Generated the remaining scenes. Total generation time {clock(monotonic() - started)}.{waiting}", None

    def switch_and_regenerate(self, storyboard_id, scene_id, take_id, request=None):
        """Use the chosen take, then regenerate the scenes that continue from it."""
        owner = require_owner(request)
        board = self.service.use_take(owner, storyboard_id, scene_id, take_id)
        index = self.service.scene_index(board, scene_id)
        pending = set(sb.pending_scenes(board))
        affected = [later for later in self.depends_on(board, index) if later in pending]
        yield from self._generate_pending(owner, storyboard_id, board, affected)

    def run_job(self, action, storyboard_id, scene_id, take_id, request=None):
        """The storyboard job family's one callback; its values are what the Jobs tab retries."""
        if action == GENERATE_SCENE:
            yield from self.generate_scene(storyboard_id, scene_id, request)
        elif action == GENERATE_REMAINING:
            yield from self.generate_remaining(storyboard_id, request)
        elif action == SWITCH_AND_REGENERATE:
            yield from self.switch_and_regenerate(storyboard_id, scene_id, take_id, request)
        else:
            raise H3Error(f"Unknown storyboard action {action!r}.")
