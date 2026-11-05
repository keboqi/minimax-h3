"""Remake in Create: edit an existing video with Viggle Remake on the MiniMax H3 settings."""

from __future__ import annotations

import html
import inspect
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import gradio as gr

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from h3_app.errors import H3Error
from h3_app.jobs import CURRENT_JOB
from h3_app.media_tools import probe_video_metadata
from h3_app.prompt_writers.remake import enhance_remake_prompt
from h3_app.remake import (
    DEFAULT_SHORT_EDGE, EDIT, FPS, MAX_PICTURES, MIN_FRAMES, RECOMMENDED_MAX_FRAMES, SHORT_EDGES, SWAP,
    RemakeGraph, canvas_size, check_pins, frames_for_seconds, missing_sections, output_size, request_overrides,
)
from h3_app.remake.align import align_image, comparison, load_image, measure_drift
from h3_app.remake.media import extract_frame, frame_seconds, trim_source
from h3_app.remake.project import CHANGED, RENDERED, STATE_NAMES
from h3_app.remake.runner import SELECTION_FIELDS, RunOptions, Selection, WindowRunner
from h3_app.remake.windows import (
    DEFAULT_MIX_FRAMES, DEFAULT_WINDOW_FRAMES, LENGTHS, needs_windows, selection_frames,
)
from h3_app.staging import file_content_sha256
from h3_app.storyboard import PROFILE_FIELDS

from ..h3.view import panel_heading
from ..image_library import build_image_input
from ..job_admission import require_owner
from ..job_bindings import bind_prompt_action
from ..prompt_writer_controls import build_remote_prompt_writer_controls
from .controls import QUIET, bind_generation, generation_actions
from .feature import EngineDeps, EngineFeature

SLOTS = range(1, MAX_PICTURES + 1)
# H3 settings that Storyboard leaves out but a Remake run follows: finishing, by H3 view control.
FINISHING_CONTROLS = {
    "postprocess": "generation_postprocess",
    "upscale_force_offload": "generation_force_offload",
    "upscale_resolution": "generation_upscale_resolution",
    "seedvr2_model": "generation_seedvr2_model",
    "ltx25_model": "ltx25_model",
}
DOCUMENTED_PROFILE = "Original"
READY = "Ready to generate."
DEFAULT_SELECTION_FRAMES = DEFAULT_WINDOW_FRAMES
RUN_BUTTON = "remake-generate"
# What opening a window fills: its index, slots, frame numbers, eight editor values, then the list.
WINDOW_OUTPUTS = 1 + 2 * MAX_PICTURES + 8 + 2

# Gradio has copied a scratch file into its own cache long before this.
SCRATCH_LIFETIME = 3600


def open_engine_js(task: str, engine: str) -> str:
    """Browser code that opens ``engine`` under ``task``, as clicking them in the top bar does."""
    return f"""() => {{
  const pick = (selector, text) => [...document.querySelectorAll(selector)]
    .find((label) => label.textContent.trim() === text)?.querySelector("input")?.click();
  pick(".h3-task-picker label", {task!r});
  requestAnimationFrame(() => requestAnimationFrame(() => pick(".h3-engine-bar label", {engine!r})));
}}"""


@dataclass(frozen=True)
class RemakeView:
    mode: gr.Radio
    source_video: gr.Video
    source_info: gr.Markdown
    start: gr.Number
    end: gr.Number
    selection_info: gr.Markdown
    frame: gr.Slider
    frame_preview: gr.Image
    extract: gr.Button
    picture_1: gr.Image
    picture_2: gr.Image
    picture_3: gr.Image
    picture_4: gr.Image
    pin_1: gr.Number
    pin_2: gr.Number
    pin_3: gr.Number
    pin_4: gr.Number
    edit_1: gr.Button
    edit_2: gr.Button
    edit_3: gr.Button
    edit_4: gr.Button
    pin_frames: gr.Checkbox
    originals: gr.State
    last_start: gr.State
    align_slot: gr.Radio
    align_view: gr.Radio
    align_check: gr.Button
    align_apply: gr.Button
    align_preview: gr.Image
    align_status: gr.Markdown
    request: gr.Textbox
    writer: Any
    rewrite: gr.Button
    rewrite_status: gr.Textbox
    suggestion: gr.Textbox
    accept: gr.Button
    keep: gr.Button
    undo: gr.Button
    previous_prompt: gr.State
    prompt: gr.Textbox
    short_edge: gr.Dropdown
    max_window: gr.Dropdown
    window_group: Any
    smart: gr.Checkbox
    mix: gr.Slider
    split: gr.Button
    window_summary: gr.Markdown
    window: gr.Dropdown
    window_index: gr.Number
    window_take: gr.Video
    carried: gr.Image
    use_carried: gr.Checkbox
    copy_carried: gr.Button
    regenerate: gr.Button
    combine: gr.Button
    source_sha: gr.State
    frame_base: gr.State
    readiness: gr.HTML
    output: gr.Video
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    next_run: gr.HTML
    open_settings: gr.Button

    def pictures(self):
        return [getattr(self, f"picture_{slot}") for slot in SLOTS]

    def pins(self):
        return [getattr(self, f"pin_{slot}") for slot in SLOTS]

    def edits(self):
        return [getattr(self, f"edit_{slot}") for slot in SLOTS]


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


def _canvas(video, short_edge) -> tuple[int, int]:
    metadata = probe_video_metadata(Path(video))
    return canvas_size(short_edge, metadata.width, metadata.height)


def _slot_images(video, start, slot_label, values):
    """The chosen slot's index, its source frame and its edited picture."""
    pictures, pins = values[:MAX_PICTURES], values[MAX_PICTURES:]
    index = int(str(slot_label).rsplit(" ", 1)[-1]) - 1
    if not video or not pictures[index] or pins[index] is None:
        raise gr.Error("Pick a picture with a frame number from your source.")
    frame = extract_frame(video, frame_seconds(start, pins[index]), _scratch_path(".png"))
    return index, load_image(frame), load_image(pictures[index])


def selected_frames(start, end) -> int:
    return frames_for_seconds(float(end or 0) - float(start or 0))


def clock(frames: int) -> str:
    return f"{frames // (60 * FPS):02d}:{frames / FPS % 60:04.1f}"


def window_label(window: Mapping[str, Any]) -> str:
    first, last = window["own"]
    kind = "continues" if window["continues"] else "new shot" if window["index"] else "start"
    return (f"{window['index'] + 1} · {clock(first)}–{clock(last)} · {kind} · {STATE_NAMES[window['state']]}")


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
                      smart=True, request="", window_frames=None) -> str:
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
        if long and smart and str(request or "").strip():
            return f"{READY} Smart writes each window's prompt from your edit request."
        if long and smart and int(window or 0) > 0:
            return f"{READY} Smart writes this window's prompt from the one before it."
        return "Rewrite your request, then accept the six-section prompt."
    if missing_sections(prompt):
        return f"{READY} This prompt isn't in the six-section format, which Remake hasn't been tested with."
    return READY


def next_run_card(captured: Mapping[str, Any], mode, video, start, end, short_edge, pictures, pin_frames,
                  max_window: int = 345) -> str:
    """The Next run card; only ``h3-setup-*`` markup, so it restyles with the H3 card."""
    profile = str(captured.get("model_profile") or "Current selection")
    upscale = bool(captured.get("latent_upscale"))
    size = "Set by the source"
    duration = "Set by the selection"
    windowed = needs_windows(start, end, max_window)
    try:
        duration = f"{selection_frames(start, end) / FPS:.2f} s" if windowed else (
            f"{min(selected_frames(start, end), int(max_window)) / FPS:.2f} s")
        metadata = probe_video_metadata(Path(video)) if video else None
        width, height = canvas_size(short_edge, metadata.width, metadata.height) if metadata else canvas_size(
            short_edge, 16, 9)
        width, height = output_size(width, height, upscale)
        size = f"{width} × {height}"
    except H3Error:
        pass
    context = [f"Remake {mode} LoRA", f"{len(pictures)} edited frame{'' if len(pictures) == 1 else 's'}",
               "frames pinned" if pin_frames else "not pinned", "source audio kept"]
    if windowed:
        context.append("rendered in windows")
    notes = []
    if profile != DOCUMENTED_PROFILE:
        notes.append(f"Untested pairing: {profile} isn't the stock FL2VA base Remake documents.")
    if upscale:
        notes.append("Untested: latent upscale refines with the Remake LoRA, which Viggle doesn't document.")
    alert = "".join(f'<p role="alert">{html.escape(note)}</p>' for note in notes)
    metric = '<div class="h3-setup-detail"><dt>{}</dt><dd>{}</dd></div>'
    return (
        '<section class="h3-setup-card" aria-label="Next run">'
        f'<div class="h3-setup-heading"><strong>Next run</strong><span>Remake · {html.escape(str(mode))}</span></div>'
        '<dl class="h3-setup-metrics">'
        + metric.format("Base model", html.escape(profile))
        + metric.format("Output", f"{html.escape(size)} · {html.escape(duration)}")
        + metric.format("Sampling", "6 steps · Euler")
        + f'</dl><div class="h3-setup-context">{"".join(f"<span>{html.escape(item)}</span>" for item in context)}</div>'
        + alert + "</section>"
    )


class RemakeHandlers:
    """Generation and its helpers; the parameter names are the public API's."""

    def __init__(self, deps: EngineDeps):
        self.deps = deps
        self.settings: dict[str, dict[str, Any]] = {}
        # Temporary writer keys stay in memory: job values are saved, so they can't carry them.
        self.writer_keys: dict[str, dict[str, Any]] = {}

    def captured(self, owner) -> dict[str, Any]:
        return self.settings.get(owner, {})

    def set_writer_keys(self, owner, gemini_key, openai_key, openai_url):
        self.writer_keys[owner] = {"gemini": gemini_key, "openai": openai_key, "url": openai_url}

    @property
    def runner(self) -> WindowRunner:
        return WindowRunner(self.deps.runtime_config, self.deps.stage_file, self.deps.h3_take, self.deps.h3_defaults)

    # Source ----------------------------------------------------------------
    def source_changed(self, video):
        """Select the start of a new source, at the length Viggle's workflows use."""
        if not video:
            return "", 0, 0, gr.update(value=0, maximum=1), None, 0, None
        metadata = probe_video_metadata(Path(video))
        end = round(min(metadata.duration, DEFAULT_SELECTION_FRAMES / FPS), 2)
        canvas_note = "" if abs(metadata.width / metadata.height - 16 / 9) < 0.02 else (
            " · Remake's canvas is 16:9, so other shapes are center-cropped")
        audio_note = "Source audio found. It stays in the result and guides mouth movement." if metadata.has_audio else (
            "No audio in this source, so the result stays silent.")
        info = (f"{metadata.width}×{metadata.height} · {metadata.fps:.2f} fps · {metadata.duration:.2f} s"
                f"{canvas_note}\n\n{audio_note}")
        return (info, 0, end, self._frame_range(0, end, 0), self._preview(video, 0, 0), 0,
                file_content_sha256(Path(video)))

    @staticmethod
    def _frame_range(start, end, frame, max_window=345):
        try:
            frames = min(selected_frames(start, end), int(max_window))
        except H3Error:
            return gr.update(value=0, maximum=1)
        return gr.update(value=min(int(frame or 0), frames - 1), minimum=0, maximum=frames - 1)

    def selection_changed(self, owner, selection, frame, index=0):
        """The selection line and its windows; a saved selection opens on its first window."""
        start, end = selection.start, selection.end
        long = needs_windows(start, end, selection.max_window)
        project = self.runner.saved(owner, selection) if long else None
        heading = (selection_text(start, end, selection.max_window), gr.update(visible=long),
                   *self._window_list(project, 0))
        if project is not None:
            return (*heading, *self._open_window(project, 0))
        # A later window's content belongs to its old plan; window 1's is the new selection's.
        clear = [None if index else gr.update() for _ in range(2 * MAX_PICTURES + 1)]
        return (*heading, 0, *clear, gr.update(), gr.update(), gr.update(), float(start or 0),
                self._frame_range(start, end, frame, selection.max_window), gr.update(),
                {} if index else gr.update())

    def _open_window(self, project, index):
        """A window's values, in the order of the editor's outputs."""
        window = project["windows"][index]
        slots = [*window["pictures"], *[None] * MAX_PICTURES][:MAX_PICTURES]
        frames = [*window["pins"], *[None] * MAX_PICTURES][:MAX_PICTURES]
        base = project["start"] + window["render"][0] / FPS
        length = window["render"][1] - window["render"][0]
        return (int(index), *slots, *frames, window["prompt"], window["use_carried"], window["carried"],
                window["take"], base, gr.update(value=0, maximum=length - 1),
                self._preview(project["source"], base, 0), {})

    # Long selections -------------------------------------------------------
    @staticmethod
    def _window_list(project, value=None):
        if project is None:
            return "Windows are planned when you split or generate.", gr.update(choices=[], value=None)
        windows = project["windows"]
        done = sum(window["state"] in RENDERED for window in windows)
        cuts = sum(1 for window in windows[1:] if not window["continues"])
        summary = (f"**{len(windows)} windows** · {cuts} at cuts, {len(windows) - 1 - cuts} continuing · "
                   f"{done} of {len(windows)} rendered")
        choices = [(window_label(window), window["index"]) for window in windows]
        return summary, gr.update(choices=choices, value=value) if value is not None else gr.update(choices=choices)

    def _project(self, owner, selection):
        project = self.runner.saved(owner, selection)
        if project is None:
            raise gr.Error("Split the selection into windows first.")
        return project

    def split(self, owner, selection, video):
        if not video:
            raise gr.Error("Add a source video first.")
        return self._window_list(self.runner.plan(owner, video, selection), 0)

    def window_list(self, owner, selection):
        return self._window_list(self.runner.saved(owner, selection))

    def refresh_windows(self, owner, selection, index):
        """The window list and the open window, after a run changed them."""
        project = self.runner.saved(owner, selection)
        if project is None:
            return (*self._window_list(None), gr.update(), gr.update(), gr.update())
        window = project["windows"][int(index or 0)]
        return (*self._window_list(project, int(index or 0)), window["take"], window["carried"],
                window["prompt"] if window["prompt_auto"] else gr.update())

    def window_selected(self, owner, selection, new, previous, prompt, use_carried, *values):
        """Keep the open window's edits, then open ``new`` in the same editor."""
        if new is None:  # Opening the list without choosing a window.
            return tuple(gr.update() for _ in range(WINDOW_OUTPUTS))
        project = self.runner.store_editor(self._project(owner, selection), previous, prompt, use_carried,
                                           values[:MAX_PICTURES], values[MAX_PICTURES:])
        return (*self._open_window(project, int(new)), *self._window_list(project, int(new)))

    def copy_carried(self, owner, selection, index, *pictures):
        """Put the carried frame in an empty slot, to edit it like any other."""
        window = self._project(owner, selection)["windows"][int(index)]
        if not window["carried"]:
            raise gr.Error("This window has no carried frame yet. Generate the window before it.")
        slot = next((number for number, picture in zip(SLOTS, pictures) if not picture), None)
        if slot is None:
            raise gr.Error("All four slots are full. Remove a picture first.")
        return (*(gr.update(value=window["carried"]) if number == slot else gr.update() for number in SLOTS),
                *(gr.update(value=window["carried_pin"]) if number == slot else gr.update() for number in SLOTS),
                False)

    def mark_window(self, owner, selection, index, prompt, use_carried, *values):
        """Queue the open window for the next run, keeping its edits."""
        runner = self.runner
        project = runner.store_editor(self._project(owner, selection), index, prompt, use_carried,
                                      values[:MAX_PICTURES], values[MAX_PICTURES:])
        project = runner.projects.edit(project, lambda current: current["windows"][int(index)].update(state=CHANGED))
        return self._window_list(project, int(index))

    def combine(self, owner, selection):
        project = self._project(owner, selection)
        return self.runner.combine(project), self.runner.combined_status(project)

    def start_moved(self, start, last_start, *pins):
        """Keep each pin on its source frame: pins count from the start of the selection."""
        moved = round((float(start or 0) - float(last_start or 0)) * FPS)
        return (start, float(start or 0), *(
            pin if pin is None or not moved else (int(pin) - moved if int(pin) >= moved else None) for pin in pins
        ))

    @staticmethod
    def _preview(video, start, frame):
        return str(extract_frame(video, frame_seconds(start, frame), _scratch_path(".png")))

    def preview_frame(self, video, start, frame):
        return self._preview(video, start, frame) if video else None

    def extract(self, video, base, frame, originals, *pictures):
        """Put the frame at ``frame`` in the first empty slot, pinned to where it came from."""
        if not video:
            raise gr.Error("Add a source video first.")
        slot = next((index for index, picture in zip(SLOTS, pictures) if not picture), None)
        if slot is None:
            raise gr.Error("All four slots are full. Remove a picture first.")
        path = self._preview(video, base, frame)
        slots = dict(originals or {})
        slots[str(slot)] = file_content_sha256(Path(path))
        return (
            *(gr.update(value=path) if index == slot else gr.update() for index in SLOTS),
            *(gr.update(value=int(frame)) if index == slot else gr.update() for index in SLOTS),
            slots,
        )

    # Alignment -------------------------------------------------------------
    def check_alignment(self, video, start, slot_label, view_name, *values):
        _index, source, edited = _slot_images(video, start, slot_label, values)
        drift = measure_drift(source, edited)
        preview = _scratch_path(".png")
        comparison(source, edited, difference=view_name == "Difference").save(preview)
        if not drift.plausible(*source.size):
            return str(preview), ("**Can't measure this one.** Too little of the frame is unchanged to compare, "
                                  "so judge it from the comparison.")
        if drift.visible:
            note = (f"**Shifted.** Move {drift.dx:+d} px across, {drift.dy:+d} px down, zoom {drift.zoom:.0%}. "
                    "Align it before pinning, because a pin holds this composition at that frame.")
        else:
            note = "**Lined up with the source.** No global shift or zoom found."
        return str(preview), note + " Alignment can't fix a changed pose."

    def apply_alignment(self, video, start, slot_label, *values):
        """Measure the selected picture now, so a choice changed after a check can't misplace it."""
        index, source, edited = _slot_images(video, start, slot_label, values)
        drift = measure_drift(source, edited)
        if not drift.plausible(*source.size):
            raise gr.Error("Too little of this frame is unchanged to align it.")
        aligned = _scratch_path(".png")
        align_image(source, edited, drift).save(aligned)
        return (*(gr.update(value=str(aligned)) if slot == index + 1 else gr.update() for slot in SLOTS),
                "**Aligned.** The picture now sits on the source frame.")

    # Prompt ----------------------------------------------------------------
    def rewrite_prompt(self, owner, selection, index, request, mode, video, pin_frames, *values):
        """The writer's suggestion for the open window; the current prompt stays until it's accepted."""
        pictures, pins = filled_slots(values[:MAX_PICTURES], values[MAX_PICTURES:2 * MAX_PICTURES])
        model, gemini_key, backend, openai_key, openai_model, openai_url = values[2 * MAX_PICTURES:]
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
            width, height = _canvas(video, selection.short_edge)
            clip = _scratch_path(".mp4")
            trim_source(video, clip, start=start, frames=frames, width=width, height=height)
        except H3Error as exc:
            return "", f"Prompt enhancement failed: {exc}"
        try:
            suggestion, status = enhance_remake_prompt(
                request, mode, str(clip), pictures, resolved_pins(pin_frames, pins), frames, width, height,
                model, gemini_key, backend, openai_key, openai_model, openai_url,
                runtime=self.deps.runtime_config(),
            )
        finally:
            clip.unlink(missing_ok=True)
        # A failed writer hands the request back; it isn't a prompt to accept.
        return ("" if "failed" in status else suggestion), status

    # Generation ------------------------------------------------------------
    def generate_remake(self, mode, source_video, start, end, short_edge, prompt, pin_frames,
                        picture_1, picture_2, picture_3, picture_4, pin_1, pin_2, pin_3, pin_4,
                        max_window=DEFAULT_WINDOW_FRAMES, edit_request="", smart=True, mix=DEFAULT_MIX_FRAMES,
                        window_index=0, use_carried=True, writer_model=DEFAULT_GEMINI_PROMPT_MODEL,
                        writer_backend="Gemini", openai_model=DEFAULT_OPENAI_PROMPT_MODEL,
                        progress=gr.Progress(track_tqdm=False)):
        slots = (picture_1, picture_2, picture_3, picture_4), (pin_1, pin_2, pin_3, pin_4)
        pictures, pins = filled_slots(*slots)
        if mode not in (SWAP, EDIT):
            raise H3Error("Remake type must be Swap or Edit.")
        if not source_video:
            raise H3Error("Add a source video.")
        job = CURRENT_JOB.get()
        owner = job.owner if job is not None else "local"
        if needs_windows(start, end, max_window):
            yield None, "Scanning the source for cuts"
            runner = self.runner
            selection = Selection(file_content_sha256(Path(source_video)), start, end, short_edge, mix, max_window)
            project = runner.plan(owner, source_video, selection)
            project = runner.store_editor(project, window_index, prompt, use_carried, *slots)
            yield from runner.run(project, RunOptions(
                mode, bool(pin_frames), bool(smart), str(edit_request or ""), writer_model, writer_backend,
                openai_model, self.writer_keys.get(owner, {}), self.captured(owner),
            ), progress)
            return
        if not pictures:
            raise H3Error("Add at least one edited frame.")
        frames = min(selected_frames(start, end), int(max_window))
        pinned = resolved_pins(pin_frames, pins)
        if pin_frames and not pinned:
            raise H3Error("Give every picture a frame number, or turn pinning off.")
        check_pins(pinned, len(pictures), frames)
        width, height = _canvas(source_video, short_edge)

        yield None, "Trimming the source to the selection"
        captured = self.captured(owner) if job is not None else {}
        values = {**self.deps.h3_defaults(), **captured}
        values.update(request_overrides(
            prompt=prompt, first_picture=str(pictures[0]), width=width, height=height, frames=frames,
            latent_upscale=bool(values.get("latent_upscale")),
        ))
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


def _settings_map() -> str:
    groups = (
        ("Follows MiniMax H3", (
            ("Base model", True), ("Text encoder", False), ("Video encoder and decoder", False), ("Seed", False),
            ("Attention and memory", False), ("Acceleration", True), ("Latent upscale", True),
            ("Finishing, on single clips", False),
        )),
        ("Set by Remake", (
            ("Swap or Edit LoRA", False), ("Six-step Euler schedule", False), ("Size and window length", False),
            ("Reference conditioning and pins", False), ("Clean source audio during sampling", False),
        )),
        ("Not used here", (
            ("Turbo LoRAs", False), ("Steps and scheduler", False), ("Semantic Bridge", False),
            ("Refinement LoRA choice", False), ("Variations", False),
        )),
    )
    columns = []
    for title, items in groups:
        rows = "".join(
            f"<li>{html.escape(label)}{' <span>Untested</span>' if untested else ''}</li>"
            for label, untested in items
        )
        columns.append(f"<div><h3>{html.escape(title)}</h3><ul>{rows}</ul></div>")
    return f'<div class="h3-remake-settings">{"".join(columns)}</div>'


def build_remake_view(root) -> RemakeView:
    with root:
        with gr.Row(equal_height=False, elem_classes=["h3-generator-shell"]):
            with gr.Column(scale=6, min_width=320, elem_classes=["h3-composer"]):
                gr.HTML(panel_heading("01", "Compose"))
                mode = gr.Radio([SWAP, EDIT], value=SWAP, label="Remake type",
                                info="Swap replaces a character or subject. Edit changes objects, background, "
                                     "weather or light.", elem_classes=["h3-recipe-cards"])
                with gr.Column(elem_classes=["h3-essentials"]):
                    gr.Markdown("### Source video")
                    source_video = gr.Video(label="Source video", sources=["upload"])
                    source_info = gr.Markdown()
                    with gr.Row():
                        start = gr.Number(value=0, minimum=0, step=0.5, label="Start (s)")
                        end = gr.Number(value=0, minimum=0, step=0.5, label="End (s)")
                    selection_info = gr.Markdown()
                    frame = gr.Slider(0, 1, value=0, step=1, label="Frame in selection")
                    frame_preview = gr.Image(label="Frame preview", interactive=False, height=260)
                    extract = gr.Button("Extract frame")
                with gr.Column(elem_classes=["h3-essentials"]):
                    gr.Markdown("### Reference frames\nUp to four. Edit a frame to show the target in the same "
                                "pose and framing. One is usually enough for a single shot.")
                    with gr.Row(elem_classes=["h3-remake-frames"]):
                        controls = []
                        for slot in SLOTS:
                            with gr.Column(min_width=220):
                                picture = build_image_input(type="filepath", label=f"Picture {slot}")
                                pin = gr.Number(value=None, precision=0, minimum=0, label=f"Picture {slot} frame",
                                                visible=False)
                                edit = gr.Button("Edit in Image", size="sm", visible=False)
                                controls.append((picture, pin, edit))
                    pin_frames = gr.Checkbox(
                        value=True, label="Pin every picture to its frame",
                        info="Holds each picture at its frame for the first four steps. Remake needs all pictures "
                             "pinned or none. Good for cuts, or when the composition matters.",
                    )
                    with gr.Accordion("Check alignment", open=False):
                        gr.Markdown("Compares a picture with its source frame. Image editors often shift or zoom "
                                    "the whole frame slightly.")
                        with gr.Row():
                            align_slot = gr.Radio([f"Picture {slot}" for slot in SLOTS], value="Picture 1",
                                                  label="Picture")
                            align_view = gr.Radio(["Overlay", "Difference"], value="Overlay", label="View")
                        with gr.Row(elem_classes=["h3-settings-links"]):
                            align_check = gr.Button("Check alignment")
                            align_apply = gr.Button("Align to source")
                        align_preview = gr.Image(label="Comparison", interactive=False, height=300)
                        align_status = gr.Markdown()
                with gr.Column(elem_classes=["h3-essentials"]):
                    gr.Markdown("### Prompt")
                    request = gr.Textbox(label="Edit request", lines=3,
                                         placeholder="Turn the dancer into a chrome robot. Keep the choreography.")
                    rewrite = gr.Button("Rewrite / enhance")
                    with gr.Accordion("Writer service & keys", open=False):
                        gr.Markdown("Gemini reads the source video and your frames. OpenAI-compatible services "
                                    "only see the frames.")
                        writer = build_remote_prompt_writer_controls(GEMINI_PROMPT_MODELS, DEFAULT_GEMINI_PROMPT_MODEL)
                    rewrite_status = gr.Textbox(label="Writer status", lines=2, interactive=False)
                    suggestion = gr.Textbox(label="Suggested prompt", lines=6, interactive=False)
                    with gr.Row(elem_classes=["h3-settings-links"]):
                        accept = gr.Button("Accept suggested prompt")
                        keep = gr.Button("Keep original")
                        undo = gr.Button("Undo accepted prompt")
                    prompt = gr.Textbox(label="Remake prompt", lines=10, max_lines=30,
                                        placeholder="subject_definitions:\n…\n\nsummary:\n…",
                                        info="The six sections Remake reads. You can edit them directly.")
                with gr.Column(elem_classes=["h3-essentials"]):
                    gr.Markdown("### Output essentials")
                    short_edge = gr.Dropdown(list(SHORT_EDGES), value=DEFAULT_SHORT_EDGE, label="Short edge",
                                             info="480 gives 832×480. 736 gives 1280×736 and needs more memory. "
                                                  "Latent upscale in MiniMax H3 doubles either.")
                    max_window = gr.Dropdown(
                        [(f"{length / FPS:.1f} s ({length} frames)", length) for length in LENGTHS],
                        value=DEFAULT_WINDOW_FRAMES, label="Longest window",
                        info="The longest clip your GPU renders in one go. Lower it if long clips run out of memory. "
                             "A longer selection renders in windows.")
                    with gr.Column(visible=False) as window_group:
                        gr.Markdown("### Windows\nEach window renders on its own, then they're joined. The editors "
                                    "above follow the window you pick here.")
                        smart = gr.Checkbox(
                            value=True, label="Smart: continue automatically",
                            info="Carries the last frame of each render into the next window and writes its prompt "
                                 "from your edit request. Turn off to give every window its own frames.")
                        mix = gr.Slider(0, 24, value=DEFAULT_MIX_FRAMES, step=1, label="Mix frames",
                                        info="Frames crossfaded between continuing windows.")
                        split = gr.Button("Split into windows")
                        window_summary = gr.Markdown()
                        window = gr.Dropdown([], value=None, label="Window")
                        window_index = gr.Number(value=0, precision=0, visible=False)
                        with gr.Row():
                            window_take = gr.Video(label="This window's render", interactive=False)
                            carried = gr.Image(label="Carried from the previous window", interactive=False,
                                               type="filepath")
                        use_carried = gr.Checkbox(value=True, label="Use the carried frame")
                        with gr.Row(elem_classes=["h3-settings-links"]):
                            copy_carried = gr.Button("Copy carried frame to a slot")
                            regenerate = gr.Button("Regenerate this window")
                            combine = gr.Button("Combine final video")
            with gr.Column(scale=6, min_width=320, elem_classes=["h3-preview-panel"]):
                gr.HTML(panel_heading("02", "Preview & generate",
                                      "Your latest result stays here as you refine your next idea."))
                output = gr.Video(label="Generated remake", interactive=False, elem_classes=["h3-result-player"])
                output.h3_metadata_root = gr.Column()
                with gr.Column(elem_classes=["h3-action-dock"]):
                    readiness = gr.HTML('<p class="h3-readiness" role="alert">Add a source video to start.</p>')
                    run, stop, status = generation_actions("Generate remake", readiness=readiness, elem_id=RUN_BUTTON)
                next_run = gr.HTML(next_run_card({}, SWAP, None, 0, 0, DEFAULT_SHORT_EDGE, [], True),
                                   elem_classes=["h3-settings-summary"])
                gr.Markdown("Powered by [Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake), a model "
                            "derivative of MiniMax H3 under the MiniMax H3 Community License.")
        gr.HTML(panel_heading("03", "Fine-tune", "Remake keeps no second copy of your settings. "
                                                 "Change them once in MiniMax H3."))
        gr.HTML(_settings_map())
        with gr.Row(elem_classes=["h3-settings-links"]):
            open_settings = gr.Button("Open MiniMax H3 settings →", elem_classes=["h3-settings-link"])
        originals = gr.State({})
        last_start = gr.State(0)
        source_sha = gr.State(None)
        frame_base = gr.State(0.0)
        previous_prompt = gr.State("")
    pictures, pins, edits = zip(*controls)
    return RemakeView(
        mode, source_video, source_info, start, end, selection_info, frame, frame_preview, extract,
        *pictures, *pins, *edits, pin_frames, originals, last_start, align_slot, align_view, align_check,
        align_apply, align_preview, align_status, request, writer, rewrite, rewrite_status, suggestion, accept, keep,
        undo, previous_prompt, prompt, short_edge, max_window, window_group, smart, mix, split, window_summary,
        window, window_index, window_take, carried, use_carried, copy_carried, regenerate, combine, source_sha,
        frame_base, readiness, output, run, stop, status, next_run, open_settings,
    )


def _bind_readiness(view: RemakeView):
    inputs = [view.source_video, view.start, view.end, view.prompt, view.pin_frames, view.originals, view.max_window,
              view.smart, view.request, view.window_index, *view.pictures(), *view.pins()]

    def readiness(video, start, end, prompt, pin_frames, originals, max_window, smart, request, window, *values):
        long = needs_windows(start, end, max_window)
        message = readiness_message(video, start, end, prompt, pin_frames, originals,
                                    values[:MAX_PICTURES], values[MAX_PICTURES:], long=long, window=window,
                                    smart=smart, request=request, window_frames=max_window if long else None)
        ready = message.startswith(READY)
        return (f'<p class="h3-readiness" role="{"status" if ready else "alert"}">{html.escape(message)}</p>',
                gr.update(interactive=ready))

    gr.on(
        triggers=[component.change for component in inputs],
        fn=readiness, inputs=inputs, outputs=[view.readiness, view.run],
        queue=False, trigger_mode="always_last", show_progress="hidden", api_name=False,
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


def _owned(function, inputs: int):
    """Call ``function(owner, *values)`` from an event whose request names the owner."""
    def call(*args):
        *values, request = args
        return function(require_owner(request), *values)

    return _with_request(call, inputs)


def _on_selection(function, inputs: int):
    """Call ``function(owner, selection, *rest)`` from an event whose inputs start with the selection."""
    return _owned(lambda owner, *values: function(
        owner, Selection(*values[:SELECTION_FIELDS]), *values[SELECTION_FIELDS:],
    ), inputs)


def bind_remake(view: RemakeView, deps: EngineDeps) -> RemakeHandlers:
    handlers = RemakeHandlers(deps)
    pictures, pins = view.pictures(), view.pins()
    selection = [view.source_sha, view.start, view.end, view.short_edge, view.mix, view.max_window]
    editor = [view.prompt, view.use_carried, *pictures, *pins]
    window_list = [view.window_summary, view.window]
    opened = [view.window_index, *pictures, *pins, view.prompt, view.use_carried, view.carried, view.window_take,
              view.frame_base, view.frame, view.frame_preview, view.originals]
    for picture, pin, edit in zip(pictures, pins, view.edits()):
        # Replacing a picture keeps its frame number: that's how an edit comes back to its slot.
        picture.change(lambda path: (gr.update(visible=bool(path)), gr.update(visible=bool(path))),
                       inputs=picture, outputs=[pin, edit], **QUIET)
    view.source_video.change(
        handlers.source_changed, inputs=view.source_video,
        outputs=[view.source_info, view.start, view.end, view.frame, view.frame_preview, view.last_start,
                 view.source_sha],
        show_progress="hidden", api_name=False,
    )
    gr.on(
        triggers=[view.source_sha.change, view.start.change, view.end.change, view.max_window.change,
                  view.short_edge.change, view.mix.change],
        fn=_on_selection(handlers.selection_changed, len(selection) + 2),
        inputs=[*selection, view.frame, view.window_index],
        outputs=[view.selection_info, view.window_group, *window_list, *opened], **QUIET,
    )
    preview = {"fn": handlers.preview_frame, "inputs": [view.source_video, view.frame_base, view.frame],
               "outputs": view.frame_preview, "show_progress": "hidden", "api_name": False}
    view.start.change(handlers.start_moved, inputs=[view.start, view.last_start, *pins],
                      outputs=[view.last_start, view.frame_base, *pins], **QUIET).then(**preview)
    view.frame.release(**preview)
    view.extract.click(handlers.extract,
                       inputs=[view.source_video, view.frame_base, view.frame, view.originals, *pictures],
                       outputs=[*pictures, *pins, view.originals], show_progress="hidden", api_name=False)
    view.split.click(_on_selection(handlers.split, len(selection) + 1), inputs=[*selection, view.source_video],
                     outputs=window_list, show_progress="minimal", api_name=False)
    assert len(opened) + len(window_list) == WINDOW_OUTPUTS
    view.window.input(
        _on_selection(handlers.window_selected, len(selection) + 2 + len(editor)),
        inputs=[*selection, view.window, view.window_index, *editor], outputs=[*opened, *window_list],
        show_progress="hidden", api_name=False,
    )
    view.copy_carried.click(_on_selection(handlers.copy_carried, len(selection) + 1 + len(pictures)),
                            inputs=[*selection, view.window_index, *pictures],
                            outputs=[*pictures, *pins, view.use_carried], show_progress="hidden", api_name=False)
    view.regenerate.click(
        _on_selection(handlers.mark_window, len(selection) + 1 + len(editor)),
        inputs=[*selection, view.window_index, *editor], outputs=window_list, show_progress="hidden", api_name=False,
    ).success(None, js=f"() => document.getElementById({RUN_BUTTON!r})?.click()", queue=False, api_name=False)
    view.combine.click(_on_selection(handlers.combine, len(selection)), inputs=selection,
                       outputs=[view.output, view.status], show_progress="minimal", api_name=False)
    # A run plans the windows itself, so the list fills in as it goes.
    view.status.change(_on_selection(handlers.window_list, len(selection)), inputs=selection, outputs=window_list,
                       trigger_mode="always_last", **QUIET)
    view.output.change(_on_selection(handlers.refresh_windows, len(selection) + 1),
                       inputs=[*selection, view.window_index],
                       outputs=[*window_list, view.window_take, view.carried, view.prompt], **QUIET)
    keys = [view.writer.gemini_api_key, view.writer.openai_api_key, view.writer.openai_base_url]
    gr.on(triggers=[key.change for key in keys], fn=_owned(handlers.set_writer_keys, len(keys)), inputs=keys,
          outputs=None, **QUIET)
    view.align_check.click(
        handlers.check_alignment,
        inputs=[view.source_video, view.frame_base, view.align_slot, view.align_view, *pictures, *pins],
        outputs=[view.align_preview, view.align_status], show_progress="hidden", api_name=False,
    )
    view.align_apply.click(
        handlers.apply_alignment,
        inputs=[view.source_video, view.frame_base, view.align_slot, *pictures, *pins],
        outputs=[*pictures, view.align_status], show_progress="hidden", api_name=False,
    )
    writer = view.writer
    rewrite_inputs = [*selection, view.window_index, view.request, view.mode, view.source_video, view.pin_frames,
                      *pictures, *pins, writer.model, writer.gemini_api_key, writer.backend, writer.openai_api_key,
                      writer.openai_model, writer.openai_base_url]
    bind_prompt_action(
        view.rewrite.click, _on_selection(handlers.rewrite_prompt, len(rewrite_inputs)), inputs=rewrite_inputs,
        outputs=[view.suggestion, view.rewrite_status], show_progress="minimal", api_name=False,
    )
    view.accept.click(lambda suggested, current: (suggested or current, current, ""),
                      inputs=[view.suggestion, view.prompt], outputs=[view.prompt, view.previous_prompt, view.suggestion],
                      **QUIET)
    view.keep.click(lambda: "", outputs=view.suggestion, **QUIET)
    view.undo.click(lambda previous: previous, inputs=view.previous_prompt, outputs=view.prompt, **QUIET)
    view.open_settings.click(None, js=open_engine_js("Video", "MiniMax H3"), queue=False, api_name=False)
    _bind_readiness(view)
    bind_generation(view, "remake", handlers.generate_remake, deps.interrupt, [
        view.mode, view.source_video, view.start, view.end, view.short_edge, view.prompt, view.pin_frames,
        *pictures, *pins, view.max_window, view.request, view.smart, view.mix, view.window_index, view.use_carried,
        view.writer.model, view.writer.backend, view.writer.openai_model,
    ])
    return handlers


def bind_h3_settings(view: RemakeView, handlers: RemakeHandlers, h3_values: Mapping[str, Any], triggers,
                     image_target=None) -> None:
    """Follow the MiniMax H3 selection, and hand frames to ``image_target`` for editing.

    Settings are read on ``triggers``. Changing them means opening MiniMax H3 and coming back,
    which fires the task change.
    """
    names = [name for name in PROFILE_FIELDS if name in h3_values]
    finishing = {name: control for name, control in FINISHING_CONTROLS.items() if control in h3_values}
    controls = [h3_values[name] for name in names] + [h3_values[control] for control in finishing.values()]
    summary_inputs = [view.mode, view.source_video, view.start, view.end, view.short_edge, view.pin_frames,
                      view.max_window, *view.pictures()]

    def card(owner, mode, video, start, end, short_edge, pin_frames, max_window, *pictures):
        return next_run_card(handlers.captured(owner), mode, video, start, end, short_edge,
                             [picture for picture in pictures if picture], pin_frames, max_window)

    def capture(*args):
        *values, request = args
        owner = require_owner(request)
        settings, rest = values[:len(controls)], values[len(controls):]
        handlers.settings[owner] = dict(zip([*names, *finishing], settings, strict=True))
        return card(owner, *rest)

    def refresh(*args):
        *values, request = args
        return card(require_owner(request), *values)

    gr.on(triggers=list(triggers), fn=_with_request(capture, len(controls) + len(summary_inputs)),
          inputs=[*controls, *summary_inputs], outputs=view.next_run,
          queue=False, api_name=False, show_progress="hidden")
    gr.on(triggers=[component.change for component in summary_inputs],
          fn=_with_request(refresh, len(summary_inputs)), inputs=summary_inputs,
          outputs=view.next_run, queue=False, trigger_mode="always_last", api_name=False, show_progress="hidden")
    if image_target is not None:
        for button, picture in zip(view.edits(), view.pictures()):
            button.click(lambda path: [path] if path else gr.update(), inputs=picture, outputs=image_target,
                         api_name=False, show_progress="hidden").then(
                None, js=open_engine_js("Image", "Qwen Image 2.1"), queue=False, api_name=False,
            )


FEATURE = EngineFeature("remake", lambda root, deps: build_remake_view(root), bind_remake, library_label="Remake",
                        library_inputs=lambda view: tuple(view.pictures()))
