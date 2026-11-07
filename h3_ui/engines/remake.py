"""Remake in Create: edit an existing video with Viggle Remake on the MiniMax H3 settings.

The tab is a page of its own in a frame (:mod:`h3_ui.remake_page`). This module holds what the page
calls, plus a hidden Gradio bridge for what still lives in Gradio: the ``generate_remake`` API, the
H3 settings, and passing pictures to and from Media and the Image task.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import gradio as gr

from h3_app.capabilities import engines_for_task
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, QWEN_IMAGE21_DEFAULTS
from h3_app.errors import H3Error
from h3_app.generation.requests import QwenImage21Request
from h3_app.jobs import CURRENT_JOB
from h3_app.media_tools import probe_video_metadata
from h3_app.progress import no_progress
from h3_app.prompt_writers.remake import enhance_remake_prompt
from h3_app.remake import (
    AUTO_SHORT_EDGE, DEFAULT_SHORT_EDGE, EDIT, FPS, MAX_PICTURES, MIN_FRAMES, RECOMMENDED_MAX_FRAMES, SHORT_EDGES, SWAP,
    RemakeGraph, RenderSize, check_pins, frames_for_seconds, missing_sections, output_size, render_size,
    request_overrides,
)
from h3_app.remake.align import align_image, comparison, load_image, measure_drift
from h3_app.remake.media import extract_frame, filmstrip, frame_seconds, trim_source
from h3_app.remake.project import CHANGED, RENDERED, STATE_NAMES, with_selection
from h3_app.remake.runner import RunOptions, Selection, WindowRunner
from h3_app.remake.windows import DEFAULT_WINDOW_FRAMES, needs_windows, selection_frames
from h3_app.staging import file_content_sha256
from h3_app.storyboard import PROFILE_FIELDS

from ..image_library import build_image_input
from ..job_admission import family_busy, require_owner
from .controls import QUIET, bind_generation, generation_actions
from .feature import EngineDeps, EngineFeature

SLOTS = range(1, MAX_PICTURES + 1)
FINISHING_CONTROLS = {
    "postprocess": "generation_postprocess",
    "upscale_force_offload": "generation_force_offload",
    "upscale_resolution": "generation_upscale_resolution",
    "seedvr2_model": "generation_seedvr2_model",
    "ltx25_model": "ltx25_model",
}
DOCUMENTED_PROFILE = "Original"
READY = "Ready to generate."
RUN_BUTTON = "remake-generate"
# The engines that can repaint a frame. Qwen comes first, as the default: it edits frames best.
EDIT_MODELS = tuple(sorted((engine for engine in engines_for_task("Image") if engine.id in ("h3", "qwen")),
                           key=lambda engine: engine.id != "qwen"))
EDIT_BRIDGE = "remake-edit-bridge-{}"
H3_MODE = "h3-conditioning-mode"
SETTINGS_BRIDGE = "remake-settings-bridge"
FRAME = ('<iframe class="h3-remake-frame" src="/remake/" title="Remake" '
         'allow="fullscreen" scrolling="no"></iframe>')

# Gradio has copied a scratch file into its own cache long before this.
SCRATCH_LIFETIME = 3600
# Uploads and extracted frames of selections that never became a project.
UPLOAD_LIFETIME = 7 * 24 * 3600


def quick_edit_request(prompt: str, picture: str) -> QwenImage21Request:
    """Qwen Image 2.1's Fast preset editing ``picture`` at its own size."""
    defaults = QWEN_IMAGE21_DEFAULTS
    return QwenImage21Request(
        mode="Image edit", model_choice=defaults["model"], text_encoder_choice=defaults["text_encoder"], prompt=prompt,
        negative_prompt="", reference_images=(picture,), width=defaults["width"], height=defaults["height"],
        reference_resolution=defaults["reference_resolution"], match_input_size=True, seed=defaults["seed"],
        steps=defaults["steps"], cfg=defaults["cfg"], sampler_name=defaults["sampler"],
        scheduler=defaults["scheduler"], cache_device=defaults["cache_device"], cache_dtype=defaults["cache_dtype"],
        attention_backend=defaults["attention_backend"], accelerator=defaults["accelerator"],
        turbo_variant=defaults["turbo_variant"],
    )


def open_engine_js(task: str, engine: str, h3_mode: str | None = None) -> str:
    """Browser code that opens ``engine`` under ``task``, as clicking them in the top bar does.

    ``h3_mode`` is then clicked too: as with a person's click, the mode's own inputs appear.
    """
    choose_mode = f"""
  const mode = (tries) => {{
    const label = [...document.querySelectorAll("#{H3_MODE} label")]
      .find((item) => item.textContent.trim() === {h3_mode!r});
    if (label?.offsetParent) {{
      const input = label.querySelector("input");
      if (!input.checked) input.click();
    }} else if (tries > 0) setTimeout(() => mode(tries - 1), 100);
  }};
  mode(50);""" if h3_mode else ""
    return f"""() => {{
  const pick = (selector, text) => [...document.querySelectorAll(selector)]
    .find((label) => label.textContent.trim() === text)?.querySelector("input")?.click();
  pick(".h3-task-picker label", {task!r});
  // The task's engines arrive from the server, so an engine that isn't the default needs waiting for.
  const engine = (tries) => {{
    const label = [...document.querySelectorAll(".h3-engine-bar label")]
      .find((item) => item.textContent.trim() === {engine!r});
    if (label) label.querySelector("input")?.click();
    else if (tries > 0) setTimeout(() => engine(tries - 1), 100);
  }};
  requestAnimationFrame(() => requestAnimationFrame(() => engine(50)));{choose_mode}
}}"""


@dataclass(frozen=True)
class RemakeView:
    """The hidden bridge: nothing here is seen; the page in the frame is the tab."""

    mode: gr.Radio
    source_video: gr.Video
    start: gr.Number
    end: gr.Number
    picture_1: gr.Image
    picture_2: gr.Image
    picture_3: gr.Image
    picture_4: gr.Image
    pin_1: gr.Number
    pin_2: gr.Number
    pin_3: gr.Number
    pin_4: gr.Number
    pin_frames: gr.Checkbox
    request: gr.Textbox
    prompt: gr.Textbox
    short_edge: gr.Dropdown
    max_window: gr.Dropdown
    smart: gr.Checkbox
    auto_prompts: gr.Checkbox
    mix: gr.Number
    window_index: gr.Number
    use_carried: gr.Checkbox
    writer_model: gr.Textbox
    writer_backend: gr.Textbox
    openai_model: gr.Textbox
    source_sha: gr.Textbox
    output: gr.Video
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    edit_bridges: dict[str, gr.Button]
    settings_bridge: gr.Button

    def pictures(self):
        return [getattr(self, f"picture_{slot}") for slot in SLOTS]

    def pins(self):
        return [getattr(self, f"pin_{slot}") for slot in SLOTS]


def _scratch_path(suffix: str) -> Path:
    """A new temporary path; every scrub writes a frame, so old ones are cleared as we go."""
    directory = Path(tempfile.gettempdir()) / "h3-remake"
    directory.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - SCRATCH_LIFETIME
    for path in directory.iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass  # Another request got there first, or the file is open on Windows.
    return directory / f"{uuid.uuid4().hex}{suffix}"


def _render_size(video, short_edge) -> RenderSize:
    metadata = probe_video_metadata(Path(video))
    return render_size(short_edge, metadata.width, metadata.height)


def selected_frames(start, end) -> int:
    return frames_for_seconds(float(end or 0) - float(start or 0))


def clock(frames: int) -> str:
    return f"{frames // (60 * FPS):02d}:{frames / FPS % 60:04.1f}"



def selection_text(start, end, max_window: int = 345) -> str:
    if needs_windows(start, end, max_window):
        seconds = selection_frames(start, end) / FPS
        return (f"**{seconds:.2f} s** · longer than one window, so this renders in windows of up to "
                f"{int(max_window) / FPS:.1f} s")
    try:
        frames = selected_frames(start, end)
    except H3Error:
        return f"Select at least {MIN_FRAMES / FPS:.2f} seconds."
    text = f"**{frames / FPS:.2f} s** · {frames} frames at {FPS} fps · frames come in steps of 17"
    if frames > RECOMMENDED_MAX_FRAMES:
        text += f" · Viggle recommends {RECOMMENDED_MAX_FRAMES} or fewer"
    return text


def filled_slots(pictures: Iterable[Any], pins: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    """The filled pictures, in slot order, with their pins: tags follow this order."""
    kept = [(picture, pin) for picture, pin in zip(pictures, pins) if picture]
    return [picture for picture, _ in kept], [pin for _, pin in kept]


def resolved_pins(pin_frames: bool, pins: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(pin) for pin in pins) if pin_frames and all(pin is not None for pin in pins) else ()


def unedited_slot(pictures: Iterable[Any], originals: Mapping[str, str] | None) -> int | None:
    """The first slot still holding the frame exactly as it was extracted."""
    for slot, picture in zip(SLOTS, pictures):
        original = (originals or {}).get(str(slot))
        if picture and original and file_content_sha256(Path(picture)) == original:
            return slot
    return None


def readiness_message(video, start, end, prompt, pin_frames, originals, pictures, pins, *, long=False, window=0,
                      smart=True, request="", window_frames=None, auto_prompts=True) -> str:
    """Why Generate is unavailable, or a note beginning with :data:`READY`."""
    if not video:
        return "Add a source video to start."
    try:
        frames = window_frames or selected_frames(start, end)
    except H3Error as exc:
        return str(exc)
    filled, filled_pins = filled_slots(pictures, pins)
    # Smart windows after the first start from the frame carried to them.
    carried = long and smart and int(window or 0) > 0
    if not filled and not carried:
        return "Extract a frame, then edit it to show the change."
    slot = unedited_slot(pictures, originals)
    if slot is not None:
        return f"Edit Picture {slot}, or remove it."
    if pin_frames:
        if any(pin is None for pin in filled_pins):
            return "Give every picture a frame number, or turn pinning off."
        if any(not 0 <= int(pin) < frames for pin in filled_pins):
            return f"Pins must be frames from 0 to {frames - 1} of the selection."
    if not str(prompt or "").strip():
        if long and int(window or 0) > 0 and not auto_prompts:
            return f"{READY} This window reuses window 1's prompt."
        if long and smart and str(request or "").strip():
            return f"{READY} Smart writes each window's prompt from your edit request."
        if long and smart and int(window or 0) > 0:
            return f"{READY} Smart writes this window's prompt from the one before it."
        return "Rewrite your request, then accept the six-section prompt."
    if missing_sections(prompt):
        return f"{READY} This prompt isn't in the six-section format, which Remake hasn't been tested with."
    return READY


def next_run_details(captured: Mapping[str, Any], mode, video, start, end, short_edge, picture_count, pin_frames,
                     max_window: int = 345) -> dict[str, Any]:
    """What the page's Next run line shows."""
    profile = str(captured.get("model_profile") or "Current selection")
    upscale = bool(captured.get("latent_upscale")) if int(short_edge) else False
    size = "Set by the source"
    duration = "Set by the selection"
    windowed = needs_windows(start, end, max_window)
    try:
        duration = f"{selection_frames(start, end) / FPS:.2f} s" if windowed else (
            f"{min(selected_frames(start, end), int(max_window)) / FPS:.2f} s")
        if video:
            sampled = _render_size(video, short_edge)
            upscale = sampled.upscale(upscale)
            size = "{} × {}".format(*output_size(sampled.width, sampled.height, upscale))
    except H3Error:
        pass
    context = [f"Remake {mode} LoRA", f"{picture_count} edited frame{'' if picture_count == 1 else 's'}",
               "frames pinned" if pin_frames else "not pinned", "source audio kept"]
    if windowed:
        context.append("rendered in windows")
    notes = []
    if profile != DOCUMENTED_PROFILE:
        notes.append(f"Untested pairing: {profile} isn't the stock FL2VA base Remake documents.")
    if upscale:
        notes.append("Untested: latent upscale refines with the Remake LoRA, which Viggle doesn't document.")
    return {"profile": profile, "output": f"{size} · {duration}", "sampling": "6 steps · Euler",
            "context": context, "notes": notes}


def readiness_step(video, message: str) -> str | None:
    """The step of the page's checklist that ``message`` is about, or None when it's ready."""
    if message.startswith(READY):
        return None
    if not video or message.startswith("Select"):
        return "source"
    return "prompt" if message.startswith("Rewrite") else "frames"


def window_payload(window: Mapping[str, Any]) -> dict[str, Any]:
    kind = "continues" if window["continues"] else "new shot" if window["index"] else "start"
    return {
        "index": window["index"], "own": window["own"], "render": window["render"], "hold": window.get("hold", 0),
        "kind": kind, "state": window["state"], "state_name": STATE_NAMES[window["state"]],
        "pictures": window["pictures"], "pins": window["pins"], "prompt": window["prompt"],
        "originals": window.get("originals", []),
        "prompt_auto": window["prompt_auto"], "take": window["take"], "carried": window["carried"],
        "carried_pin": window["carried_pin"], "use_carried": window["use_carried"],
    }


def project_payload(project: Mapping[str, Any]) -> dict[str, Any]:
    windows = project["windows"]
    done = sum(window["state"] in RENDERED for window in windows)
    cuts = sum(1 for window in windows[1:] if not window["continues"])
    combined = project.get("combined") or {}
    current = combined.get("takes") == [window["take"] for window in windows]
    return {
        "key": project["key"], "name": project.get("name") or Path(project["source"]).stem,
        "source": project["source"], "start": project["start"], "frames": project["frames"],
        "selection": project.get("selection"), "windows": [window_payload(window) for window in windows],
        "summary": f"{len(windows)} windows · {cuts} at cuts, {len(windows) - 1 - cuts} continuing · "
                   f"{done} of {len(windows)} rendered",
        "rendered": done, "output": combined.get("output") if current else None,
        "mode": project.get("mode") or SWAP, "request": project.get("request") or "",
        "smart": project.get("smart", True), "pinned": project.get("pinned", True),
        "auto_prompts": project.get("auto_prompts", True),
    }


class RemakeHandlers:
    """What the page calls; the parameter names of ``generate_remake`` are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps
        self.settings: dict[str, dict[str, Any]] = {}
        # Temporary writer keys stay in memory: job values are saved, so they can't carry them.
        self.writer_keys: dict[str, dict[str, Any]] = {}
        # Pictures on their way between the page and Gradio.
        self.inbox: dict[str, dict[int, str]] = {}
        self.outgoing: dict[str, str] = {}
        # Files outside this owner's Remake folder that the page may show, such as joined videos.
        self.granted: dict[str, set[str]] = {}
        # The Media library, set once Media is built.
        self.library = None

    def captured(self, owner) -> dict[str, Any]:
        return self.settings.get(owner, {})

    def set_writer_keys(self, owner, gemini_key, openai_key, openai_url):
        self.writer_keys[owner] = {"gemini": gemini_key, "openai": openai_key, "url": openai_url}

    @property
    def runner(self) -> WindowRunner:
        return WindowRunner(self.deps.runtime_config, self.deps.stage_file, self.deps.h3_take, self.deps.h3_defaults)

    # Files -----------------------------------------------------------------
    def _owner_root(self, owner) -> Path:
        return self.runner.projects.folder(owner, "x").parent

    def _upload_folder(self, owner) -> Path:
        folder = self._owner_root(owner) / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def upload_path(self, owner, suffix: str) -> Path:
        """A new path for a picture or frame of this owner; week-old ones are cleared as we go."""
        folder = self._upload_folder(owner)
        cutoff = time.time() - UPLOAD_LIFETIME
        for path in folder.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass  # Another request got there first, or the file is open on Windows.
        return folder / f"{uuid.uuid4().hex}{suffix}"

    def grant(self, owner, path) -> str:
        self.granted.setdefault(owner, set()).add(str(path))
        return str(path)

    def may_read(self, owner, path) -> bool:
        if str(path) in self.granted.get(owner, ()):
            return True
        root = self._owner_root(owner).resolve()
        return root in Path(path).resolve().parents

    def require_readable(self, owner, *paths) -> None:
        """Refuse paths the browser sent that aren't this owner's; empty ones mean nothing is set."""
        if not all(self.may_read(owner, path) for path in paths if path):
            raise H3Error("That file isn't available to you.")

    @staticmethod
    def frame_preview(video, base, frame) -> Path:
        """One frame for a source the browser can't play; scrubbing writes many, so they're scratch."""
        return extract_frame(video, frame_seconds(base, frame), _scratch_path(".png"))

    def send_to_image(self, owner, path, engine) -> None:
        """Leave a copy where Gradio will take it, for the bridge button the page presses next."""
        self.require_readable(owner, path)
        if engine not in {model.id for model in EDIT_MODELS}:
            raise H3Error("Pick an edit model.")
        target = _scratch_path(Path(path).suffix or ".png")
        shutil.copy2(path, target)
        self.outgoing[owner] = str(target)

    def adopt_picture(self, owner, path) -> str:
        """Keep a copy of a picture from outside, since the original may be a Gradio temp file."""
        source = Path(path)
        target = self.upload_path(owner, source.suffix.lower() or ".png")
        shutil.copy2(source, target)
        return str(target)

    # Source ----------------------------------------------------------------
    def add_source(self, owner, path, name) -> dict[str, Any]:
        """Keep an uploaded video by its content hash under its own name, and describe it."""
        sha = file_content_sha256(Path(path))
        stem = re.sub(r"[^\w. -]", "_", Path(str(name)).stem).strip(" .") or "source"
        target = self._upload_folder(owner) / sha[:24] / f"{stem}{Path(path).suffix.lower()}"
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(path), target)
        target.touch()
        return self.source_payload(target, sha)

    @staticmethod
    def source_payload(video, sha=None, name=None) -> dict[str, Any]:
        path = Path(video)
        metadata = probe_video_metadata(path)
        return {
            "path": str(path), "sha": sha or file_content_sha256(path), "name": name or path.stem,
            "info": f"{metadata.width}×{metadata.height} · {metadata.fps:.2f} fps · {metadata.duration:.2f} s",
            "audio_note": ("Source audio found. It stays in the result and guides mouth movement."
                           if metadata.has_audio else "No audio in this source, so the result stays silent."),
            "duration": metadata.duration,
            # Viggle's workflows use this length, so a new source starts on it.
            "end": round(min(metadata.duration, DEFAULT_WINDOW_FRAMES / FPS), 2),
        }

    def strip(self, owner, video, start, seconds) -> Path:
        """A filmstrip of the selection, kept while it is in use."""
        folder = self._upload_folder(owner) / "strips"
        folder.mkdir(exist_ok=True)
        stat = Path(video).stat()
        key = hashlib.sha256(f"{video}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
        target = folder / f"{key}_{float(start):.3f}_{float(seconds):.3f}.jpg"
        return target if target.is_file() else filmstrip(video, float(start), float(seconds), target)

    def selection_state(self, owner, selection: Selection) -> dict[str, Any]:
        """The selection's line, and its saved windows if it was split before."""
        long = needs_windows(selection.start, selection.end, selection.max_window)
        project = self.runner.saved(owner, selection)
        return {"text": selection_text(selection.start, selection.end, selection.max_window), "long": long,
                "project": project_payload(project) if project is not None else None}

    # Projects --------------------------------------------------------------
    def project_rows(self, owner) -> list[dict[str, Any]]:
        """Every saved Remake, newest first."""
        rows = []
        for project in self.runner.projects.list(owner):
            windows = project["windows"]
            done = sum(window["state"] in RENDERED for window in windows)
            rows.append({
                "key": project["key"], "name": project.get("name") or Path(project["source"]).stem,
                "meta": f"{project['frames'] / FPS:.1f} s · {done} of {len(windows)} windows · "
                        + time.strftime("%Y-%m-%d %H:%M", time.localtime(project["updated"])),
                "thumb": next((window["pictures"][0] for window in windows if window["pictures"]), None),
            })
        return rows

    def _named_project(self, owner, key):
        # A key from a request body, unlike a route segment, could hold a path.
        named = isinstance(key, str) and key != ".." and Path(key).name == key
        project = self.runner.projects.load(owner, key) if named else None
        if project is None:
            raise H3Error("Pick a project first.")
        return project

    def load_project(self, owner, key) -> dict[str, Any]:
        """Reopen a project from its own copy of the source, so nothing needs uploading."""
        project = with_selection(self.runner.projects, self._named_project(owner, key))
        if not Path(project["source"]).is_file():
            raise H3Error("This project's copy of the source is gone.")
        chosen = project["selection"]
        data = project_payload(project)
        data["source_info"] = self.source_payload(project["source"], chosen["sha"], data["name"])
        data["note"] = ("It keeps its crossfades. Upload the source again for seamless windows."
                        if chosen["mix"] else "")
        return data

    def project_data(self, owner, key) -> dict[str, Any]:
        return project_payload(self._named_project(owner, key))

    def rename_project(self, owner, key, name) -> None:
        name = str(name or "").strip()
        if not name:
            raise H3Error("Enter a name.")
        self.runner.projects.edit(self._named_project(owner, key), lambda current: current.update(name=name))

    def delete_project(self, owner, key) -> None:
        if family_busy(owner, "remake"):
            raise H3Error("Wait for the current Remake to finish.")
        self.runner.projects.delete(self._named_project(owner, key))

    # Windows ---------------------------------------------------------------
    def _project(self, owner, selection):
        project = self.runner.saved(owner, selection)
        if project is None:
            raise H3Error("Split the selection into windows first.")
        return project

    def split(self, owner, selection, video) -> dict[str, Any]:
        if not video:
            raise H3Error("Add a source video first.")
        return project_payload(self.runner.plan(owner, video, selection))

    def save_project(self, owner, selection, video, index, prompt, use_carried, pictures, pins, options):
        """Keep preparation for any selection so Studio can reopen it.

        ``project`` in ``options`` names an open project for autosave, which only
        ever updates it; without one, the selection's project is created if needed.
        """
        if options.get("mode", SWAP) not in (SWAP, EDIT):
            raise H3Error("Remake type must be Swap or Edit.")
        if not video:
            raise H3Error("Add a source video first.")
        runner = self.runner
        key = options.get("project")
        project = self._named_project(owner, key) if key else runner.plan(owner, video, selection)
        if not 0 <= index < len(project["windows"]):
            raise H3Error("That window is no longer part of this project.")
        project = runner.store_editor(
            project, index, prompt, use_carried, pictures, pins,
            options={"mode": options.get("mode", SWAP), "request": str(options.get("request") or ""),
                     "smart": bool(options.get("smart", True)), "pinned": bool(options.get("pinned", True)),
                     "auto_prompts": bool(options.get("auto_prompts", True))},
            originals=options.get("originals") or {},
        )
        return project_payload(project)

    def save_window(self, owner, selection, index, prompt, use_carried, pictures, pins) -> dict[str, Any]:
        """Keep a window's edits, as the page does before it leaves that window."""
        return project_payload(self.runner.store_editor(self._project(owner, selection), index, prompt, use_carried,
                                                        pictures, pins))

    def copy_carried(self, owner, selection, index) -> dict[str, Any]:
        """The carried frame and where it belongs, to edit it like any other."""
        window = self._project(owner, selection)["windows"][int(index)]
        if not window["carried"]:
            raise H3Error("This window has no carried frame yet. Generate the window before it.")
        # Under held frames it is the last of them; repainted, it still belongs at that moment.
        pin = window["hold"] - 1 if window.get("hold") else window["carried_pin"]
        return {"path": window["carried"], "pin": pin}

    def mark_window(self, owner, selection, index, prompt, use_carried, pictures, pins) -> dict[str, Any]:
        """Queue a window for the next run, keeping its edits."""
        runner = self.runner
        project = runner.store_editor(self._project(owner, selection), index, prompt, use_carried, pictures, pins)
        project = runner.projects.edit(project, lambda current: current["windows"][int(index)].update(state=CHANGED))
        return project_payload(project)

    def combine(self, owner, selection) -> dict[str, Any]:
        project = self._project(owner, selection)
        runner = self.runner
        output = self.grant(owner, runner.combine(project))
        return {"output": output, "status": runner.combined_status(project)}

    # Frames ----------------------------------------------------------------
    def extract(self, owner, video, base, frame) -> dict[str, Any]:
        """The frame at ``frame``, with what tells an untouched copy from an edit."""
        if not video:
            raise H3Error("Add a source video first.")
        path = extract_frame(video, frame_seconds(base, frame), self.upload_path(owner, ".png"))
        return {"path": str(path), "pin": int(frame), "original": file_content_sha256(path)}

    # Alignment -------------------------------------------------------------
    @staticmethod
    def _picture_images(video, base, picture, pin):
        """The source frame a picture is pinned to, and the picture."""
        if not video or not picture or pin is None:
            raise H3Error("Pick a picture with a frame number from your source.")
        frame = extract_frame(video, frame_seconds(base, pin), _scratch_path(".png"))
        return load_image(frame), load_image(picture)

    def check_alignment(self, owner, video, base, picture, pin, view_name="Overlay") -> dict[str, Any]:
        source, edited = self._picture_images(video, base, picture, pin)
        drift = measure_drift(source, edited)
        preview = self.upload_path(owner, ".png")
        comparison(source, edited, difference=view_name == "Difference").save(preview)
        if not drift.plausible(*source.size):
            return {"preview": str(preview), "state": "unknown",
                    "note": "Can't measure this one. Too little of the frame is unchanged to compare, "
                            "so judge it from the comparison."}
        if drift.visible:
            return {"preview": str(preview), "state": "shifted",
                    "note": f"Shifted. Move {drift.dx:+d} px across, {drift.dy:+d} px down, zoom {drift.zoom:.0%}. "
                            "Align it before pinning, because a pin holds this composition at that frame. "
                            "Alignment can't fix a changed pose."}
        return {"preview": str(preview), "state": "lined_up",
                "note": "Lined up with the source. No global shift or zoom found. Alignment can't fix a changed pose."}

    def apply_alignment(self, owner, video, base, picture, pin) -> str:
        """Measure the picture now, so a choice changed after a check can't misplace it."""
        source, edited = self._picture_images(video, base, picture, pin)
        drift = measure_drift(source, edited)
        if not drift.plausible(*source.size):
            raise H3Error("Too little of this frame is unchanged to align it.")
        aligned = self.upload_path(owner, ".png")
        align_image(source, edited, drift).save(aligned)
        return str(aligned)

    # Prompt ----------------------------------------------------------------
    def rewrite_prompt(self, owner, selection, index, request, mode, video, pin_frames, pictures, pins, writer):
        """The writer's suggestion for the open window; the current prompt stays until it's accepted."""
        pictures, pins = filled_slots(pictures, pins)
        if not video:
            return "", "Add a source video first."
        if not pictures:
            return "", "Add at least one edited frame first."
        project = self.runner.saved(owner, selection)
        try:
            if project is not None:
                first, last = project["windows"][int(index or 0)]["render"]
                start, frames = project["start"] + first / FPS, last - first
            else:
                start, frames = float(selection.start or 0), min(selected_frames(selection.start, selection.end),
                                                                 int(selection.max_window))
            size = _render_size(video, selection.short_edge)
            width, height = size.width, size.height
            clip = _scratch_path(".mp4")
            trim_source(video, clip, start=start, frames=frames, width=width, height=height)
        except H3Error as exc:
            return "", f"Prompt enhancement failed: {exc}"
        try:
            suggestion, status = enhance_remake_prompt(
                request, mode, str(clip), pictures, resolved_pins(pin_frames, pins), frames, width, height,
                writer.get("model") or DEFAULT_GEMINI_PROMPT_MODEL, writer.get("gemini_key") or "",
                writer.get("backend") or "Gemini", writer.get("openai_key") or "",
                writer.get("openai_model") or DEFAULT_OPENAI_PROMPT_MODEL, writer.get("openai_url"),
                runtime=self.deps.runtime_config(),
            )
        finally:
            clip.unlink(missing_ok=True)
        # A failed writer hands the request back; it isn't a prompt to accept.
        return ("" if "failed" in status else suggestion), status

    # Quick edit ------------------------------------------------------------
    def quick_edit(self, picture, prompt, progress=no_progress):
        """Repaint a picture from text with Qwen Image 2.1, keeping its size. Yields (image, status)."""
        if not picture or not str(prompt or "").strip():
            raise H3Error("Write what to change in the frame.")
        staged = self.deps.stage_file(str(picture), "qwen_image21_references", reuse=True)
        result = None
        for update in self.deps.qwen_take(quick_edit_request(prompt, staged), progress):
            status = str(update.status).split("\n\n")[0]
            if status.startswith("Error:"):
                raise H3Error(status.removeprefix("Error:").strip())
            output = update.output[0] if isinstance(update.output, list) else update.output
            result = output or result
            yield None, status
        if result is None:
            raise H3Error("The edit finished without an image.")
        yield result, "Frame edited."

    # Generation ------------------------------------------------------------
    def generate_remake(self, mode, source_video, start, end, short_edge, prompt, pin_frames,
                        picture_1, picture_2, picture_3, picture_4, pin_1, pin_2, pin_3, pin_4,
                        max_window=DEFAULT_WINDOW_FRAMES, edit_request="", smart=True, mix=0,
                        window_index=0, use_carried=True, writer_model=DEFAULT_GEMINI_PROMPT_MODEL,
                        writer_backend="Gemini", openai_model=DEFAULT_OPENAI_PROMPT_MODEL, source_sha="",
                        auto_prompts=True, progress=gr.Progress(track_tqdm=False)):
        slots = (picture_1, picture_2, picture_3, picture_4), (pin_1, pin_2, pin_3, pin_4)
        pictures, pins = filled_slots(*slots)
        if mode not in (SWAP, EDIT):
            raise H3Error("Remake type must be Swap or Edit.")
        if not source_video:
            raise H3Error("Add a source video.")
        job = CURRENT_JOB.get()
        owner = job.owner if job is not None else "local"
        sha = source_sha or file_content_sha256(Path(source_video))
        selection = Selection(sha, start, end, short_edge, mix, max_window)
        if needs_windows(start, end, max_window) or self.runner.saved(owner, selection) is not None:
            yield None, "Scanning the source for cuts"
            runner = self.runner
            project = runner.plan(owner, source_video, selection)
            project = runner.store_editor(project, window_index, prompt, use_carried, *slots)
            project = runner.projects.edit(project, lambda current: current.update(
                mode=mode, request=str(edit_request or ""), smart=bool(smart), pinned=bool(pin_frames),
                auto_prompts=bool(auto_prompts),
            ))
            yield from runner.run(project, RunOptions(
                mode, bool(pin_frames), bool(smart), str(edit_request or ""), writer_model, writer_backend,
                openai_model, self.writer_keys.get(owner, {}), self.captured(owner), bool(auto_prompts),
            ), progress)
            return
        if not pictures:
            raise H3Error("Add at least one edited frame.")
        frames = min(selected_frames(start, end), int(max_window))
        pinned = resolved_pins(pin_frames, pins)
        if pin_frames and not pinned:
            raise H3Error("Give every picture a frame number, or turn pinning off.")
        check_pins(pinned, len(pictures), frames)
        size = _render_size(source_video, short_edge)
        width, height = size.width, size.height

        yield None, "Trimming the source to the selection"
        captured = self.captured(owner) if job is not None else {}
        values = {**self.deps.h3_defaults(), **captured}
        values.update(request_overrides(
            prompt=prompt, first_picture=str(pictures[0]), width=width, height=height, frames=frames,
            latent_upscale=size.upscale(values.get("latent_upscale")),
        ))
        if int(short_edge) == AUTO_SHORT_EDGE:
            # Finishing can change the size, and Auto's size is the source's.
            values["postprocess"] = "None"
        clip = _scratch_path(".mp4")
        try:
            trim_source(source_video, clip, start=float(start or 0), frames=frames, width=width, height=height)
            staged = self.deps.stage_file(str(clip), "remake_sources", reuse=True)
        finally:
            clip.unlink(missing_ok=True)
        remake = RemakeGraph(
            mode=mode, source_video=staged,
            pictures=tuple(self.deps.stage_file(str(path), "remake_pictures", reuse=True) for path in pictures),
            pins=pinned, frames=frames,
        )
        yield from self.deps.h3_take(values, remake, progress)



def build_remake_view(root) -> RemakeView:
    """The frame, and the bridge beside it. It is hidden by CSS, not ``visible=False``, so that
    picture changes fire, the bridge buttons exist in the DOM and the API stays registered."""
    with root:
        gr.HTML(FRAME)
        with gr.Column(elem_classes=["h3-remake-bridge"]):
            mode = gr.Radio([SWAP, EDIT], value=SWAP, label="Remake type")
            source_video = gr.Video(label="Source video", sources=["upload"])
            start = gr.Number(value=0, minimum=0, label="Start (s)")
            end = gr.Number(value=0, minimum=0, label="End (s)")
            pictures = [build_image_input(type="filepath", label=f"Picture {slot}") for slot in SLOTS]
            pins = [gr.Number(value=None, precision=0, label=f"Picture {slot} frame") for slot in SLOTS]
            pin_frames = gr.Checkbox(value=True, label="Pin every picture to its frame")
            request = gr.Textbox(label="Edit request")
            prompt = gr.Textbox(label="Remake prompt")
            short_edge = gr.Dropdown([("Auto", AUTO_SHORT_EDGE), *((str(edge), edge) for edge in SHORT_EDGES)],
                                     value=DEFAULT_SHORT_EDGE, label="Short edge")
            max_window = gr.Number(value=DEFAULT_WINDOW_FRAMES, precision=0, label="Longest window")
            smart = gr.Checkbox(value=True, label="Smart")
            auto_prompts = gr.Checkbox(value=True, label="Write a prompt for each window")
            mix = gr.Number(value=0, precision=0, label="Mix")
            window_index = gr.Number(value=0, precision=0, label="Window")
            use_carried = gr.Checkbox(value=True, label="Use the carried frame")
            writer_model = gr.Textbox(value=DEFAULT_GEMINI_PROMPT_MODEL, label="Writer model")
            writer_backend = gr.Textbox(value="Gemini", label="Writer service")
            openai_model = gr.Textbox(value=DEFAULT_OPENAI_PROMPT_MODEL, label="OpenAI model")
            source_sha = gr.Textbox(value="", label="Source hash")
            output = gr.Video(label="Generated remake", interactive=False)
            output.h3_metadata_root = gr.Column()
            run, stop, status = generation_actions("Generate remake", elem_id=RUN_BUTTON)
            edit_bridges = {model.id: gr.Button(f"Edit in {model.label}", elem_id=EDIT_BRIDGE.format(model.id))
                            for model in EDIT_MODELS}
            settings_bridge = gr.Button("Open MiniMax H3 settings", elem_id=SETTINGS_BRIDGE)
    return RemakeView(
        mode, source_video, start, end, *pictures, *pins, pin_frames, request, prompt, short_edge, max_window, smart,
        auto_prompts, mix, window_index, use_carried, writer_model, writer_backend, openai_model, source_sha, output,
        run, stop, status, edit_bridges, settings_bridge,
    )


def _with_request(function, inputs: int):
    """``function(*values, request)`` with the signature Gradio reads to inject the request."""
    function.__signature__ = inspect.Signature([
        *(inspect.Parameter(f"value_{index}", inspect.Parameter.POSITIONAL_OR_KEYWORD) for index in range(inputs)),
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=gr.Request, default=None),
    ])
    # Gradio injects the request from annotations; without this it passes None.
    function.__annotations__ = {"request": gr.Request}
    return function


def generation_inputs(view: RemakeView) -> list:
    """``generate_remake``'s inputs, in the order of its parameters."""
    return [view.mode, view.source_video, view.start, view.end, view.short_edge, view.prompt, view.pin_frames,
            *view.pictures(), *view.pins(), view.max_window, view.request, view.smart, view.mix, view.window_index,
            view.use_carried, view.writer_model, view.writer_backend, view.openai_model, view.source_sha,
            view.auto_prompts]


def _receiver(handlers: RemakeHandlers, slot: int):
    """Keep a picture set from Media until the page's next pulse collects it."""
    def received(path, request):
        if path:
            owner = require_owner(request)
            handlers.inbox.setdefault(owner, {})[slot] = handlers.adopt_picture(owner, path)

    return received


def bind_remake(view: RemakeView, deps: EngineDeps) -> RemakeHandlers:
    handlers = RemakeHandlers(deps)

    for slot, picture in zip(SLOTS, view.pictures()):
        picture.change(_with_request(_receiver(handlers, slot), 1), inputs=picture, outputs=None, **QUIET)
    view.settings_bridge.click(None, js=open_engine_js("Video", "MiniMax H3"), queue=False, api_name=False)
    bind_generation(view, "remake", handlers.generate_remake, deps.interrupt, generation_inputs(view))
    return handlers


@dataclass(frozen=True)
class EditTarget:
    """Where an edit model takes the frame: its input, and the mode that shows that input.

    ``h3_mode`` is clicked in the browser; ``mode`` is a control set to ``mode_value``.
    """

    picture: Any
    multiple: bool = True
    h3_mode: str | None = None
    mode: Any = None
    mode_value: str | None = None


def _bind_edit_bridge(button, target: EditTarget, label: str, handlers: RemakeHandlers) -> None:
    def outgoing(request):
        path = handlers.outgoing.pop(require_owner(request), None)
        if not path:
            return gr.update()
        return [path] if target.multiple else path

    step = button.click(_with_request(outgoing, 0), outputs=target.picture, api_name=False, show_progress="hidden")
    if target.mode is not None:
        step = step.then(lambda: target.mode_value, outputs=target.mode, queue=False, api_name=False,
                         show_progress="hidden")
    step.then(None, js=open_engine_js("Image", label, target.h3_mode), queue=False, api_name=False)


def bind_h3_settings(view: RemakeView, handlers: RemakeHandlers, h3_values: Mapping[str, Any], triggers,
                     edit_targets: Mapping[str, EditTarget] | None = None) -> None:
    """Follow the MiniMax H3 selection, and hand frames to ``edit_targets`` for editing.

    Settings are read on ``triggers``. Changing them means opening MiniMax H3 and coming back,
    which fires the task change.
    """
    names = [name for name in PROFILE_FIELDS if name in h3_values]
    finishing = {name: control for name, control in FINISHING_CONTROLS.items() if control in h3_values}
    controls = [h3_values[name] for name in names] + [h3_values[control] for control in finishing.values()]

    def capture(*args):
        *values, request = args
        handlers.settings[require_owner(request)] = dict(zip([*names, *finishing], values, strict=True))

    gr.on(triggers=list(triggers), fn=_with_request(capture, len(controls)), inputs=controls, outputs=None,
          queue=False, api_name=False, show_progress="hidden")
    for model in EDIT_MODELS:
        if model.id in (edit_targets or {}):
            _bind_edit_bridge(view.edit_bridges[model.id], edit_targets[model.id], model.label, handlers)


FEATURE = EngineFeature("remake", lambda root, deps: build_remake_view(root), bind_remake, library_label="Remake",
                        settings=lambda view: {}, library_inputs=lambda view: tuple(view.pictures()))
