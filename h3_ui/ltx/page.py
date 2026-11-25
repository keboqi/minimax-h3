"""The LTX 2.5 page and its JSON API, outside Gradio, as the Image and Song pages are.

One draft per owner: the mode, the prompt, the keyframes or reference pictures, and the options the
unseen Gradio view declares (their defaults and limits come from it). A run takes the job path the
Gradio Generate button took, in a thread here, and the page polls it. Official workflow models
download from here too, as the old "Official workflows and model downloads" accordion did.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock
from typing import Any, Callable, Iterator, Mapping
import uuid

from fastapi import Body, FastAPI, File, Form, Request, UploadFile

from h3_app import media_review
from h3_app.capabilities import engines_for_task
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.policy import ltx25_frame_length, validate_resolution
from h3_app.provenance import render_snapshot
from h3_app.reference_metadata import describe_reference

from ..job_admission import execute_accepted
from ..page_fields import (
    FieldSpec, change_writer, coerce, library_items, library_source, option_row, owned_file,
    path_of as _path_of, plain, quote_path as _quote, spool_image, update_value, writer_rows,
)
from ..page_routes import install_static_page, interrupt_active_run, owner_of, session_of, validate_ui_patch
from ..page_runs import Run, Runs, guarded
from ..page_state import OwnerPageState, private_page_root

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/ltx"
API = f"{ROUTE}/api"
RUN_KEY = "ltx"
FAMILY = "ltx"
GENERATE, ENHANCE, MODELS = "generate", "enhance", "models"
BUSY = {GENERATE: "A video is already being made.", ENHANCE: "The prompt writer is already running.",
        MODELS: "Models are already downloading."}

TEXT, IMAGE, REFERENCE = "Text to video", "Image to video", "Reference images"
MODES = (TEXT, IMAGE, REFERENCE)
MODE_HELP = {
    TEXT: "Prompt only. Video and its audio are generated together.",
    IMAGE: "Animate from a start keyframe; add middle and end keyframes to steer the shot.",
    REFERENCE: "Characters, props and settings from a reference sheet (IC-LoRA).",
}
# What the Ingredients IC-LoRA is trained for; Reference images locks these.
REFERENCE_LOCK = {"width": 768, "height": 448, "duration": 5, "fps": 24}
REFERENCE_LIMIT = 9
KEYFRAMES = ("first_image", "middle_image", "end_image")
RESULT_LIMIT = 6

# Every name the generation job takes, in the order ``LtxHandlers.generate_ltx25`` takes them.
JOB_ORDER = ("mode", "model_choice", "prompt", "negative_prompt", "first_image", "duration", "fps", "width",
             "height", "seed", "cfg", "sampler_name", "image_strength", "middle_image", "middle_time",
             "middle_strength", "end_image", "end_strength", "reference_images")
MEDIA_INDICES = {name: JOB_ORDER.index(name) for name in (*KEYFRAMES, "reference_images")}
# The options by job name, and the attribute of ``LtxView`` that declares each.
VIEW_FIELDS = {
    "model_choice": "model", "negative_prompt": "negative", "duration": "duration", "fps": "fps",
    "width": "width", "height": "height", "seed": "seed", "cfg": "cfg", "sampler_name": "sampler",
    "image_strength": "image_strength", "middle_time": "middle_time", "middle_strength": "middle_strength",
    "end_strength": "end_strength",
}

GROUPS = (
    ("Sampling", "The distilled model is tuned for 8 steps at CFG 1; raise CFG only for stronger prompt adherence."),
    ("Timing & exact size", "Dimensions snap to multiples of 32; the frame count snaps to 8n + 1."),
    ("Model", "Both run the single-stage distilled transformer on the shared ComfyUI backend."),
    ("Prompt writer", "What Generate / enhance prompt uses."),
    ("Workflows & models", "Gated assets download on first use. Accept the LTX 2.5 model license first; "
                           "some repositories need separate acceptance."),
)
GROUP_OF = {"seed": 0, "cfg": 0, "sampler_name": 0, "fps": 1, "width": 1, "height": 1, "model_choice": 2}
# Changes in these groups count as changes from the defaults; an exact size is a choice, not a change.
COUNTED = {"seed", "cfg", "sampler_name", "fps", "model_choice"}
WORKFLOW_GROUP = 4

# Aspect ratio → size tier → (width, height); every side is a multiple of 32.
SIZES = {
    "16:9": {"Small": (768, 448), "Medium": (960, 544), "Large": (1280, 704)},
    "9:16": {"Small": (448, 768), "Medium": (544, 960), "Large": (704, 1280)},
    "1:1": {"Small": (576, 576), "Medium": (736, 736), "Large": (960, 960)},
    "4:3": {"Small": (640, 480), "Medium": (832, 640), "Large": (1088, 832)},
}
TIERS = ("Small", "Medium", "Large")


def fields_of(view) -> dict[str, FieldSpec]:
    """The options as the unseen Gradio view declares them, by the name the job takes each under."""
    from ..page_fields import spec_of

    return {name: spec_of(getattr(view, attribute)) for name, attribute in VIEW_FIELDS.items()}


def private_ltx_root(runtime, workspace: Path) -> Path:
    """Keep owner drafts outside the served roots."""
    return private_page_root(runtime, workspace, "ltx")


@dataclass(frozen=True)
class LtxPageServices:
    """What the page needs from the application; ``root`` is where owners' drafts and pictures live."""

    root: Callable[[], Path]
    fields: Mapping[str, FieldSpec]
    # The Gradio prompt writer: (prompt, model, key, mode, start, middle, end, duration, width, height, ...).
    enhance: Callable[..., tuple[str, str]]
    # Each official workflow with what it still needs (``ModelController.ltx25_workflow_status``).
    workflows: Callable[[], list[dict[str, Any]]] = lambda: []
    # Downloads for one workflow by label, or every missing model: each yields (status, inventory).
    prepare: Callable[[str], Iterator] = lambda label: iter(())
    prepare_all: Callable[[], Iterator] = lambda: iter(())
    interrupt: Callable[..., str] | None = None
    library: Callable[[], Any] = lambda: None


@dataclass
class LtxRun(Run):
    """A step of the page: what it is doing, and what it left for the page to fetch."""

    result: str | None = None
    suggestion: str | None = None
    original: str | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "suggestion": self.suggestion, "original": self.original}


def blank_media() -> dict[str, Any]:
    return {"first_image": None, "middle_image": None, "end_image": None, "references": []}


class LtxPageState(OwnerPageState):
    """Each owner's draft and options, saved as one JSON file."""

    @classmethod
    def from_bootstrap(
        cls,
        services,
        view: Any,
        handlers: Any,
        *,
        library: Callable[[], Any],
    ) -> LtxPageState:
        return cls(
            LtxPageServices(
                root=lambda: private_ltx_root(services.engine_deps.runtime_config(), JOBS.store.root),
                fields=fields_of(view),
                enhance=handlers.enhance_ltx25_prompt,
                workflows=handlers.ltx25_workflow_status,
                prepare=handlers.prepare_ltx25_official_workflow,
                prepare_all=handlers.prepare_all_ltx25_official_models,
                interrupt=services.interrupt,
                library=library,
            )
        )

    def load(self, owner: str) -> dict[str, Any]:
        saved = self.read_saved(owner)
        options = self.load_options(saved)
        writer = self.load_writer(owner, saved)
        uploads = self.owner_root(owner) / "uploads"
        # Only the owner's own copies count; anything else on disk is ignored.
        own = lambda path: str(uploads / Path(path).name) if path and (uploads / Path(path).name).is_file() else None
        stored = saved.get("media") if isinstance(saved.get("media"), dict) else {}
        media = blank_media()
        for slot in KEYFRAMES:
            media[slot] = own(stored.get(slot))
        media["references"] = [path for path in map(own, stored.get("references") or []) if path]
        return {
            "mode": saved.get("mode") if saved.get("mode") in MODES else TEXT,
            "prompt": str(saved.get("prompt") or ""), "options": options, "writer": writer, "media": media,
            "results": [str(path) for path in saved.get("results") or [] if path and Path(path).is_file()],
            "ui": saved.get("ui") if isinstance(saved.get("ui"), dict) else {},
        }

    # Edits -----------------------------------------------------------------
    def set_mode(self, state: dict[str, Any], mode: str) -> None:
        if mode not in MODES:
            raise H3Error("Choose Text to video, Image to video or Reference images.")
        state["mode"] = mode
        if mode == REFERENCE:
            # As the Gradio view did: Ingredients only runs at the size and length it was trained for.
            state["options"].update(REFERENCE_LOCK)

    def apply(self, state: dict[str, Any], action: str, changes: Mapping[str, Any], name: str | None = None) -> None:
        options = state["options"]
        if action == "mode":
            self.set_mode(state, str(changes.get("mode") or ""))
        elif action == "size":
            ratio, _, tier = str(name or "").partition(" ")
            if ratio not in SIZES or tier not in SIZES[ratio]:
                raise H3Error("Choose one of the listed sizes.")
            self.unlocked(state, "The size")
            options["width"], options["height"] = SIZES[ratio][tier]
        elif action == "restore":
            defaults = self.defaults()
            for option in COUNTED:
                options[option] = defaults[option]
            if state["mode"] == REFERENCE:
                options.update(REFERENCE_LOCK)
        else:
            for changed, value in changes.items():
                if changed not in self.services.fields:
                    raise H3Error(f"Unknown setting: {changed}")
                if changed in REFERENCE_LOCK:
                    self.unlocked(state, self.services.fields[changed].label)
                value = coerce(self.services.fields[changed], value)
                if changed in ("width", "height"):
                    value = validate_resolution(value, value)[0]
                    if not 256 <= value <= 2048:
                        raise H3Error(f"{self.services.fields[changed].label} must be between 256 and 2048.")
                options[changed] = value
            if "duration" in changes and options["middle_time"] >= options["duration"]:
                # Keep the middle keyframe inside the shorter video.
                options["middle_time"] = round(max(0.1, options["duration"] / 2), 1)

    def unlocked(self, state: Mapping[str, Any], label: str) -> None:
        if state["mode"] == REFERENCE:
            raise H3Error(f"{label} is set by Reference images: 768×448, 5 seconds at 24 fps.")

    # What the page shows -----------------------------------------------------
    def issue(self, state: Mapping[str, Any]) -> str:
        mode, media, options = state["mode"], state["media"], state["options"]
        if not state["prompt"].strip():
            return "Write a video prompt to start."
        if mode == IMAGE and not media["first_image"]:
            return "Add a start keyframe to generate this video."
        if mode == IMAGE and media["middle_image"] and not 0 < options["middle_time"] < options["duration"]:
            return "Move the middle keyframe inside the video's length."
        if mode == REFERENCE and not media["references"]:
            return "Add reference images to generate this video."
        return ""

    def picture(self, owner: str, path: str | None, label: str) -> dict[str, Any] | None:
        if not path:
            return None
        return {"path": path, "url": f"{API}/file?path={_quote(self.grant(owner, path))}",
                "detail": re.sub(r"^[0-9a-f]{12}-", "", describe_reference(path, label))}

    def fine(self, options: Mapping[str, Any], writer: Mapping[str, Any]) -> list[dict[str, Any]]:
        defaults = self.defaults()
        rows = []
        for name, group in GROUP_OF.items():
            changed = name in COUNTED and options[name] != defaults[name]
            rows.append(option_row(name, self.services.fields[name], options[name], group=group, changed=changed,
                                   preset=defaults[name] if changed else None))
        return [*rows, *writer_rows(writer, group=3)]

    def describe(self, owner: str, state: dict[str, Any]) -> dict[str, Any]:
        options, media, mode = state["options"], state["media"], state["mode"]
        issue = self.issue(state)
        rows = self.fine(options, state["writer"])
        counts: dict[str, int] = {}
        for row in rows:
            if row.get("changed"):
                counts[str(row["group"])] = counts.get(str(row["group"]), 0) + 1
        compose = lambda name: option_row(name, self.services.fields[name], options[name])
        paths = [self.grant(owner, path) for path in state["results"]]
        width, height = options["width"], options["height"]
        size = next(((ratio, tier) for ratio, tiers in SIZES.items() for tier, value in tiers.items()
                     if value == (width, height)), ("Custom", "Exact"))
        return {
            "mode": mode, "mode_help": MODE_HELP[mode], "locked": mode == REFERENCE,
            "prompt": state["prompt"], "ui": state["ui"], "options": options,
            "keyframes": {slot: self.picture(owner, media[slot], "Picture") for slot in KEYFRAMES},
            "references": [self.picture(owner, path, "Picture") for path in media["references"]],
            "reference_limit": REFERENCE_LIMIT,
            "fields": rows, "changed": counts,
            "compose": {name: compose(name) for name in ("duration", "negative_prompt", "image_strength",
                                                          "middle_time", "middle_strength", "end_strength")},
            "width": width, "height": height, "ratio": size[0], "tier": size[1],
            "frames": ltx25_frame_length(options["duration"], options["fps"]),
            "readiness": {"ready": not issue, "message": issue or "Ready to generate with LTX 2.5."},
            "writer": state["writer"]["backend"],
            "results": {"videos": [{"path": path, "name": Path(path).name, "url": f"{API}/file?path={_quote(path)}",
                                    "review": media_review.public(path)} for path in reversed(paths)],
                        "settings": render_snapshot(paths[-1:]) if paths else ""},
        }

    def job_values(self, state: Mapping[str, Any]) -> list[Any]:
        """The generation job's inputs, in its order; media the mode does not use is left out."""
        mode, media, options = state["mode"], state["media"], state["options"]
        keyframes = media if mode == IMAGE else blank_media()
        values = {**options, "mode": mode, "prompt": state["prompt"],
                  **{slot: keyframes[slot] for slot in KEYFRAMES},
                  "reference_images": list(media["references"]) if mode == REFERENCE else None}
        return [values[name] for name in JOB_ORDER]

    # Handoffs from other pages ---------------------------------------------------
    def receive(self, owner: str, slot: str, path: str) -> str:
        """Media hands over one picture for a keyframe or the reference sheet."""
        with self.lock:
            state = self.load(owner)
            if slot == "references":
                if len(state["media"]["references"]) >= REFERENCE_LIMIT:
                    raise H3Error("The reference sheet is full. Remove a picture on the LTX 2.5 page first.")
                state["media"]["references"].append(self.keep(owner, path))
                self.set_mode(state, REFERENCE)
                where = "LTX · Reference images"
            elif slot in KEYFRAMES:
                state["media"][slot] = self.keep(owner, path)
                state["mode"] = IMAGE
                where = f"LTX · {SLOT_LABELS[slot]}"
            else:
                raise H3Error("Choose an LTX 2.5 input.")
            self.save(owner, state)
        return f"Added image to {where}. Open LTX 2.5 to use it."


SLOT_LABELS = {"first_image": "Start keyframe", "middle_image": "Middle keyframe", "end_image": "End keyframe"}


def media_targets(pages: LtxPageState) -> dict[str, Callable[[str, str], str]]:
    """The Media page's Set as destinations on this page."""
    targets = {f"LTX · {label}": (lambda owner, path, slot=slot: pages.receive(owner, slot, path))
               for slot, label in SLOT_LABELS.items()}
    targets["LTX · Reference images"] = lambda owner, path: pages.receive(owner, "references", path)
    return targets


def install_ltx_page(app: FastAPI, pages: LtxPageState, *, gpu: bool = True) -> LtxPageState:
    """Serve the page at ``/ltx/`` and its API under ``/ltx/api``."""
    services = pages.services
    lock = RLock()

    def blocked(run: Run) -> str:
        # Downloads and generation both fetch model files; one at a time per owner.
        other = MODELS if run.kind == GENERATE else GENERATE if run.kind == MODELS else None
        if other and runs[other].active(run.owner, RUN_KEY):
            return ("Wait for the model downloads to finish." if other == MODELS
                    else "Wait for the video to finish before downloading models.")
        return ""

    runs = {kind: Runs(busy=message, lock=lock, blocked=blocked) for kind, message in BUSY.items()}

    def can_generate() -> bool:
        return gpu and FAMILY in JOBS.callbacks

    def require_gpu():
        if not can_generate():
            raise H3Error("Open LTX 2.5 in Studio to generate on the GPU.")

    def run_payloads(owner: str) -> dict[str, Any]:
        found = {}
        for kind, items in runs.items():
            run = items.latest(owner, RUN_KEY)
            found[kind] = run.public() if run is not None else None
        return found

    def answer(owner: str, state: dict[str, Any], **extra) -> dict[str, Any]:
        return {"state": pages.describe(owner, state), "runs": run_payloads(owner), **extra}

    def edit(request: Request, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
            change(state)
            pages.save(owner, state)
            return answer(owner, state)

    def workflows() -> dict[str, Any]:
        rows = services.workflows()
        return {"workflows": rows, "missing": sum(1 for row in rows if row["missing"])}

    # The page --------------------------------------------------------------
    install_static_page(app, ROUTE, STATIC, STATIC_TYPES)

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        with pages.lock:
            described = answer(owner, pages.load(owner))
        return {
            **described, **workflows(), "gpu": can_generate(),
            "engines": [
                {"id": engine.id, "label": engine.label, "description": engine.description}
                for engine in engines_for_task("Video")
            ],
            "modes": list(MODES), "groups": [list(group) for group in GROUPS], "workflow_group": WORKFLOW_GROUP,
            "sizes": {ratio: [[tier, *tiers[tier]] for tier in TIERS] for ratio, tiers in SIZES.items()},
            "reference_lock": REFERENCE_LOCK,
        }

    # Draft and options -------------------------------------------------------
    @app.post(f"{API}/settings", include_in_schema=False)
    @guarded
    def change_settings(request: Request, body: dict = Body(...)):
        changes = body.get("changes") or {}
        if not isinstance(changes, dict):
            raise H3Error("Send the changed settings by name.")
        return edit(request, lambda state: pages.apply(state, str(body.get("action") or "edit"), changes,
                                                       body.get("name")))

    @app.post(f"{API}/writer", include_in_schema=False)
    @guarded
    def writer_settings(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: change_writer(state["writer"], body.get("changes")))

    @app.post(f"{API}/prompt", include_in_schema=False)
    @guarded
    def change_prompt(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: state.update(prompt=str(body.get("prompt") or "")))

    @app.post(f"{API}/empty", include_in_schema=False)
    @guarded
    def empty_draft(request: Request):
        return edit(request, lambda state: state.update(prompt="", media=blank_media()))

    @app.post(f"{API}/ui", include_in_schema=False)
    @guarded
    def change_ui(request: Request, body: dict = Body(...)):
        """What only the page reads: whether Fine-tune is open, and its group."""
        ui = validate_ui_patch(body)
        return edit(request, lambda state: state.update(ui=ui))

    # Keyframes and reference pictures ---------------------------------------
    def place(owner: str, slot: str, paths: list[str], rollback: ExitStack) -> dict[str, Any]:
        with pages.lock:
            state = pages.load(owner)
            if slot == "references":
                if len(state["media"]["references"]) + len(paths) > REFERENCE_LIMIT:
                    raise H3Error(f"The reference sheet takes at most {REFERENCE_LIMIT} pictures.")
                state["media"]["references"] += paths
            elif slot in KEYFRAMES:
                if len(paths) != 1:
                    raise H3Error("A keyframe takes one picture.")
                state["media"][slot] = paths[0]
            else:
                raise H3Error("Choose a keyframe or the reference sheet.")
            pages.save(owner, state)
            rollback.pop_all()
            return answer(owner, state)

    @app.post(f"{API}/media/upload", include_in_schema=False)
    @guarded
    def upload_media(request: Request, slot: str = Form(...), files: list[UploadFile] = File(...)):
        owner = owner_of(request)
        folder = pages.uploads(owner)
        with ExitStack() as rollback:
            paths = [str(spool_image(upload, folder, rollback)) for upload in files]
            return place(owner, slot, paths, rollback)

    @app.post(f"{API}/media/library", include_in_schema=False)
    @guarded
    def media_from_library(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        source = library_source(services.library(), str(body.get("source") or ""))
        with ExitStack() as rollback:
            path = pages.keep(owner, source, rollback)
            return place(owner, str(body.get("slot") or ""), [path], rollback)

    @app.post(f"{API}/media/remove", include_in_schema=False)
    @guarded
    def remove_media(request: Request, body: dict = Body(...)):
        slot = str(body.get("slot") or "")

        def change(state):
            if slot in KEYFRAMES:
                state["media"][slot] = None
                return
            references = state["media"]["references"]
            index = body.get("index")
            if slot != "references" or isinstance(index, bool) or not isinstance(index, int) \
                    or not 0 <= index < len(references):
                raise H3Error("Choose one of the pictures.")
            references.pop(index)

        return edit(request, change)

    @app.get(f"{API}/library", include_in_schema=False)
    @guarded
    def library_page(request: Request, query: str = "", limit: int = 24):
        owner_of(request)
        return library_items(services.library(), query, limit)

    @app.get(f"{API}/file", include_in_schema=False)
    def read_file(path: str, request: Request, download: bool = False):
        return owned_file(lambda: owner_of(request), pages.may_read, path, download)

    # Long work -------------------------------------------------------------
    def start(owner: str, kind: str, work: Callable[[LtxRun], None], status: str = "Starting…") -> dict[str, Any]:
        run = LtxRun(owner, RUN_KEY, kind)
        run.status = status
        runs[kind].start(run, work)
        return {"runs": run_payloads(owner)}

    @app.post(f"{API}/generate", include_in_schema=False)
    @guarded
    def generate(request: Request):
        require_gpu()
        owner = owner_of(request)
        session = session_of(request)
        with pages.lock:
            state = pages.load(owner)
            if issue := pages.issue(state):
                raise H3Error(issue)
            values = pages.job_values(state)
            prompt, writer = state["prompt"], dict(state["writer"])
        media_indices = tuple(index for name, index in MEDIA_INDICES.items() if values[index])

        def work(run: LtxRun):
            job = JOBS.accept(owner, FAMILY, values, callback=JOBS.callbacks[FAMILY], key=uuid.uuid4().hex,
                              media_indices=media_indices, output_indices=(0,))
            run.job_id = job.id
            for update in execute_accepted(job.id, session):
                sets, value = update_value(update[0])
                if sets and value and (path := _path_of(value)):
                    run.result = path
                status = update_value(update[1])[1] if len(update) > 1 else None
                if isinstance(status, str) and status:
                    run.status = plain(status)
                run.revision += 1
            if run.result:
                with pages.lock:
                    latest = pages.load(owner)
                    latest["results"] = [*latest["results"], run.result][-RESULT_LIMIT:]
                    pages.save(owner, latest)
                # In the background: the video plays before its score and tags arrive.
                media_review.schedule(run.result, prompt, api_key=writer.get("gemini_api_key") or "",
                                      enabled=bool(writer.get("review", True)), detailed=True)
            if run.error is None and not run.result and run.status.startswith("Error"):
                run.error = run.status

        return start(owner, GENERATE, work, "Waiting for the GPU…")

    @app.post(f"{API}/enhance", include_in_schema=False)
    @guarded
    def enhance(request: Request):
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
        options, writer, media, mode = state["options"], state["writer"], state["media"], state["mode"]
        prompt = state["prompt"]
        keyframes = [media[slot] if mode == IMAGE else None for slot in KEYFRAMES]

        def work(run: LtxRun):
            run.status = "Writing…"
            suggestion, status = services.enhance(
                prompt, writer["model"], writer["gemini_api_key"], mode, *keyframes, options["duration"],
                options["width"], options["height"], backend=writer["backend"],
                openai_api_key=writer["openai_api_key"], openai_model=writer["openai_model"],
                openai_base_url=writer["openai_base_url"] or None)
            run.original = prompt
            if not suggestion or suggestion == prompt:
                run.status = plain(status) or "The writer returned the same prompt."
                return
            run.suggestion = suggestion
            run.status = f"{plain(status)} Preview ready; choose Accept or Keep original.".strip()

        return start(owner, ENHANCE, work)

    @app.get(f"{API}/models", include_in_schema=False)
    @guarded
    def read_models(request: Request):
        owner_of(request)
        return workflows()

    @app.post(f"{API}/models/prepare", include_in_schema=False)
    @guarded
    def prepare_models(request: Request, body: dict = Body(default={})):
        require_gpu()
        owner = owner_of(request)
        label = body.get("workflow")
        if label is not None and label not in {row["label"] for row in services.workflows()}:
            raise H3Error("Choose one of the official workflows.")

        def work(run: LtxRun):
            for status, _inventory in (services.prepare(label) if label else services.prepare_all()):
                # Bold markers would otherwise leave a space before the closing period.
                run.status = plain(str(status).replace("**", ""))
                run.revision += 1
            if run.status.startswith("Error"):
                run.error = run.status

        return start(owner, MODELS, work, "Checking model files…")

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, state: bool = False, models: bool = False):
        owner = owner_of(request)
        found: dict[str, Any] = {"runs": run_payloads(owner)}
        if state:
            with pages.lock:
                found["state"] = pages.describe(owner, pages.load(owner))
        if models:
            found.update(workflows())
        return found

    @app.post(f"{API}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request):
        require_gpu()
        owner = owner_of(request)
        run = runs[GENERATE].active(owner, RUN_KEY)
        message = interrupt_active_run(request, run, FAMILY, services.interrupt)
        return {"message": message, "runs": run_payloads(owner)}

    return pages
