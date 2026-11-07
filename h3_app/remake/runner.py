"""Rendering a long selection window by window, and joining the takes."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Sequence

from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.media_tools import probe_video_metadata
from h3_app.prompt_writers.remake import enhance_remake_prompt
from h3_app.provenance import write_snapshot

from .media import detect_cuts, extract_frame, frame_seconds, held_clip, join_windows, trim_source
from .model import FPS, RemakeGraph, RenderSize, check_pins, frames_for_seconds, render_size, request_overrides
from .project import (
    DONE, FAILED, RENDERED, Projects, carry_position, fingerprint, record_take, save_editor, store_picture,
    window_inputs,
)
from .windows import needs_windows, plan_windows, selection_frames


class Selection(NamedTuple):
    """What identifies a long Remake: the same values reopen the same windows.

    ``mix`` is 0 now. Only a project made before held frames has another value, which it keeps
    so that it reopens and joins as it was split.
    """

    sha: str | None
    start: float
    end: float
    short_edge: int
    mix: int
    max_window: int

    @property
    def key(self) -> str:
        return fingerprint(self.sha, float(self.start or 0), float(self.end or 0), int(self.short_edge),
                           int(self.mix), int(self.max_window))


SELECTION_FIELDS = len(Selection._fields)


@dataclass(frozen=True)
class RunOptions:
    mode: str
    pinned: bool
    smart: bool
    request: str
    writer_model: str
    writer_backend: str
    openai_model: str
    # In memory only: never part of a job's saved values.
    writer_keys: Mapping[str, Any] = field(default_factory=dict)
    h3_settings: Mapping[str, Any] = field(default_factory=dict)
    # Off: later windows reuse window 1's prompt instead of getting their own.
    auto_prompts: bool = True


# With no edit request, a later window is written from the prompt before it.
CONTINUE_REQUEST = "Continue the same edit through this window."


def _scratch_clip() -> Path:
    return Path(tempfile.gettempdir()) / f"h3-remake-{uuid.uuid4().hex}.mp4"


def active_pins(pins: Sequence[int | None]) -> tuple[int | None, ...]:
    return tuple(pins) if any(pin is not None for pin in pins) else ()


@dataclass(frozen=True)
class WindowRunner:
    runtime: Callable[[], Any]
    stage_file: Callable[..., str]
    take: Callable[..., Iterator[GenerationUpdate]]
    defaults: Callable[[], dict]

    @property
    def projects(self) -> Projects:
        return Projects(self.runtime().output_dir)

    def saved(self, owner: str, selection: Selection) -> dict[str, Any] | None:
        return self.projects.load(owner, selection.key) if selection.sha else None

    def plan(self, owner: str, video: str, selection: Selection) -> dict[str, Any]:
        """This selection's project, resumed if it was split before."""
        saved = self.saved(owner, selection)
        if saved is not None:
            return saved
        start, total = float(selection.start or 0), selection_frames(selection.start, selection.end)
        if not needs_windows(selection.start, selection.end, selection.max_window):
            total = min(frames_for_seconds(total / FPS), selection.max_window)
        metadata = probe_video_metadata(Path(video))
        if total > int(metadata.duration * FPS) + 1:
            raise H3Error("The selection runs past the end of the source.")
        size = render_size(selection.short_edge, metadata.width, metadata.height)
        plans = plan_windows(total, detect_cuts(video, start, total / FPS), max_frames=selection.max_window)
        data = {"start": start, "frames": total, "width": size.width, "height": size.height,
                "latent_upscale": size.latent_upscale, "mix": 0,
                "has_audio": bool(metadata.has_audio), "name": Path(video).stem, "selection": selection._asdict()}
        return self.projects.create(owner, selection.key, video, data, plans)

    def store_editor(self, project, index, prompt, use_carried, pictures, pins, *,
                     options=None, originals=None) -> dict[str, Any]:
        """Save the editor's filled slots to a window."""
        kept = [(path, pin) for path, pin in zip(pictures, pins) if path]
        stored = [store_picture(project, path) for path, _pin in kept]

        def update(current):
            window = current["windows"][int(index)]
            save_editor(window, stored, [None if pin is None else int(pin) for _path, pin in kept],
                        prompt, use_carried)
            if originals is not None:
                window["originals"] = [originals.get(str(slot)) for slot, path in enumerate(pictures, 1) if path]
            if options is not None:
                current.update(options)

        return self.projects.edit(project, update)

    def run(self, project: dict[str, Any], options: RunOptions, progress) -> Iterator[tuple[str | None, str]]:
        """Render every window that needs it, in order, then join them. Finished windows are kept."""
        owner = project["owner"]
        for index in [window["index"] for window in project["windows"] if window["state"] not in RENDERED]:
            yield from self._render(project, index, options, progress)
            project = self.projects.load(owner, project["key"])
        if all(window["take"] for window in project["windows"]):
            yield None, "Joining the windows"
            yield self.combine(project), self.combined_status(project)
        else:
            yield None, "Some windows still need a frame or an earlier window before they can be combined."

    def _carry(self, project: dict[str, Any], index: int) -> dict[str, Any]:
        """Refresh the frame a window takes from the take before it."""
        if not index or not project["windows"][index - 1]["take"]:
            return project
        frame, pin = carry_position(project, index)
        path = Path(project["folder"]) / "pictures" / f"carried_{index + 1:02d}.png"
        extract_frame(project["windows"][index - 1]["take"], frame_seconds(0, frame), path)
        return self.projects.edit(project, lambda current: current["windows"][index].update(
            carried=str(path), carried_pin=pin,
        ))

    def _render(self, project, index, options: RunOptions, progress):
        windows = project["windows"]
        label = f"Window {index + 1} of {len(windows)}"
        project = self._carry(project, index)
        pictures, pins = window_inputs(project, index, pinned=options.pinned, smart=options.smart)
        if not pictures:
            yield None, f"{label} needs a frame. Skipped."
            return
        window = project["windows"][index]
        hold = window.get("hold", 0)
        if hold and not project["windows"][index - 1]["take"]:
            yield None, f"{label} continues window {index}, which needs generating first. Skipped."
            return
        first, last = window["render"]
        frames = last - first
        pinned = active_pins(pins)
        check_pins(pinned, len(pictures), frames)
        clip = _scratch_clip()
        try:
            yield None, f"{label}: trimming the source"
            trim_source(project["source"], clip, start=project["start"] + first / FPS, frames=frames,
                        width=project["width"], height=project["height"])
            text = window["prompt"]
            # An auto prompt's <Picture N> tags only fit the pictures it was written for.
            stale = window["prompt_auto"] and window["prompt_pictures"] != len(pictures)
            shared = project["windows"][0]["prompt"]
            if index and not options.auto_prompts and shared.strip() and (window["prompt_auto"] or not text.strip()):
                text = shared
            elif stale or not text.strip():
                yield None, f"{label}: writing the prompt"
                text = self._write_prompt(project, index, str(clip), pictures, pins, options)
            if text != window["prompt"]:
                self.projects.edit(project, lambda current: current["windows"][index].update(
                    prompt=text, prompt_auto=True, prompt_pictures=len(pictures),
                ))
            staged = self.stage_file(str(clip), "remake_sources", reuse=True)
            held = self._stage_held(project, index) if hold else None
        finally:
            clip.unlink(missing_ok=True)
        values = {**self.defaults(), **options.h3_settings}
        values.update(request_overrides(
            prompt=text, first_picture=str(pictures[0]), width=project["width"], height=project["height"],
            frames=frames, latent_upscale=RenderSize(project["width"], project["height"],
                                                     project.get("latent_upscale")).upscale(
                values.get("latent_upscale")),
        ))
        # Upscaling would change the size windows join at. Face refine keeps it, so it runs per window.
        values["postprocess"] = "None"
        prefix = Path(project["folder"]).relative_to(Path(self.runtime().output_dir)).as_posix()
        remake = RemakeGraph(
            mode=options.mode, source_video=staged,
            pictures=tuple(self.stage_file(str(path), "remake_pictures", reuse=True) for path in pictures),
            pins=pinned, frames=frames, output_prefix=f"{prefix}/w{index + 1:02d}_{uuid.uuid4().hex[:8]}",
            held_frames=held,
        )
        take = None
        for update in self.take(values, remake, progress):
            if str(update.status).startswith("Error:"):
                self.projects.edit(project, lambda current: current["windows"][index].update(state=FAILED))
                raise H3Error(f"{label}: " + str(update.status).removeprefix("Error:").strip().split("\n\n")[0])
            take = update.output or take
            yield None, f"{label}: {str(update.status).splitlines()[0]}"
        if take is None:
            raise H3Error(f"{label} finished without a video.")
        previous_take = window["take"]
        project = self.projects.edit(project, lambda current: record_take(current, index, str(take)))
        if previous_take and previous_take != str(take):
            Path(previous_take).unlink(missing_ok=True)
        if index + 1 < len(windows):
            self._carry(project, index + 1)
        yield str(take), f"{label} done"

    def _stage_held(self, project: dict[str, Any], index: int) -> str:
        """The frames of the previous render that this window starts on."""
        previous, window = project["windows"][index - 1], project["windows"][index]
        clip = _scratch_clip()
        try:
            held_clip(previous["take"], clip, first=window["render"][0] - previous["render"][0], frames=window["hold"])
            return self.stage_file(str(clip), "remake_sources", reuse=True)
        finally:
            clip.unlink(missing_ok=True)

    def _write_prompt(self, project, index, clip, pictures, pins, options: RunOptions) -> str:
        previous = project["windows"][index - 1]["prompt"] if index else ""
        request = options.request.strip() or (CONTINUE_REQUEST if previous.strip() else "")
        if not request:
            raise H3Error(f"Window {index + 1} needs a prompt, or an edit request for Smart to write one.")
        window = project["windows"][index]
        keys = options.writer_keys
        text, status = enhance_remake_prompt(
            request, options.mode, clip, pictures, pins, window["render"][1] - window["render"][0],
            project["width"], project["height"], options.writer_model, keys.get("gemini") or "",
            options.writer_backend, keys.get("openai") or "", options.openai_model, keys.get("url"),
            runtime=self.runtime(),
            previous_prompt=previous,
            carried_picture=pictures.index(window["carried"]) + 1 if window["carried"] in pictures else None,
        )
        if "failed" in status:
            raise H3Error(f"Window {index + 1}: {status}")
        return text

    def combine(self, project: dict[str, Any]) -> str:
        """Join the latest takes into a video in Media."""
        windows = project["windows"]
        missing = [window["index"] + 1 for window in windows if not window["take"]]
        if missing:
            raise H3Error("Generate these windows first: " + ", ".join(map(str, missing)) + ".")
        pieces = [{"path": window["take"], "first": window["own"][0] - window["render"][0],
                   "frames": window["own"][1] - window["own"][0], "continues": window["continues"]}
                  for window in windows]
        output = Path(self.runtime().output_dir) / "h3" / f"remake_{project['key'][:8]}_{uuid.uuid4().hex[:8]}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        join_windows(pieces, project["source"], start=project["start"], frames=project["frames"], mix=project["mix"],
                     output=output, has_audio=project["has_audio"])
        write_snapshot(output, {"family": "Remake", "remake": {
            "windows": len(windows), "frames": project["frames"], "mix": project["mix"],
        }})
        self.projects.edit(project, lambda current: current.update(combined={
            "output": str(output), "takes": [window["take"] for window in windows],
        }))
        return str(output)

    @staticmethod
    def combined_status(project: dict[str, Any]) -> str:
        stale = [window["index"] + 1 for window in project["windows"] if window["state"] != DONE]
        note = f" Windows {', '.join(map(str, stale))} follow changed windows or edits." if stale else ""
        return f"Combined {len(project['windows'])} windows.{note}"
