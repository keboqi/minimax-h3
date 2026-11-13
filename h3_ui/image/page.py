"""The Image page and its JSON API, outside Gradio, as MiniMax H3's is.

One draft (task, prompt, images to edit) is shared by the five image models; each model keeps its
own options. Every edit goes through the same transitions the Gradio views bound, and the answer
says what the page shows: the options, which are hidden or locked, and whether a run may start.
A run takes the job path the Gradio Generate button took, in a thread here, and the page polls it.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import errno
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from h3_app.h3_image import H3_IMAGE_OFFICIAL, H3_IMAGE_PRUNA
from h3_app import catalog, media_review
from h3_app.capabilities import engines_for_task
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.provenance import render_snapshot
from h3_app.reference_metadata import describe_reference
from h3_app.workflows.flux3 import flux3_sampling_defaults
from h3_app.workflows.moss_image_prompt import is_layout_prompt
from h3_app.workspace_store import request_owner

from ..engines.controls import resolution_preset_values
from ..engines.moss_image import moss_image_turbo_defaults, moss_image_turbo_size_steps
from ..engines.qwen.bindings import qwen_turbo_defaults
from ..engines.qwen.view import QWEN_IMAGE21_LIGHTNING_MODES
from ..h3.page import plain
from ..image_controller import IMAGE_MODELS, draft_issue, image_mode, reference_limit, reference_tag
from ..job_admission import execute_accepted
from ..page_fields import (
    SECRET_FIELDS, UI_LIMIT, FieldSpec, change_writer, coerce, library_items, library_source, option_row, owned_file,
    review_row, spool_image, update_value, writer_defaults, writer_rows,
)
from ..page_runs import Run, Runs, guarded

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/image"
API = f"{ROUTE}/api"
RUN_KEY = "image"
GENERATE, ENHANCE = "generate", "enhance"
BUSY = {GENERATE: "An image is already being made.", ENHANCE: "The prompt writer is already running."}
GENERATE_MODE, EDIT_MODE = "Text to image", "Image edit"
MANUAL = catalog.QWEN_EDIT_SIZE_MANUAL
WRITER_ENGINES = ("h3", "qwen", "flux3", "moss_image")
REFERENCE_LIMIT = 24  # Kept in the draft; each model's own limit is checked when it runs.

GROUPS = (
    ("Model & sampling", "Which weights load, Turbo, and how the image is sampled."),
    ("Size & seed", "Exact width and height, and a fixed seed."),
    ("Edit inputs", "How the images to edit are read. These apply to edits."),
    ("Speed", "Attention and sampling acceleration."),
    ("Prompt writer", "What Suggest a prompt uses, and prompt help that runs with the model."),
)
# Where Fine-tune shows a model's option, by the name its generation takes it under.
GROUP_OF = {
    "backend": 0, "model_choice": 0, "precision": 0, "turbo_variant": 0, "lightning_strength": 0, "text_encoder_choice": 0,
    "vae_choice": 0, "steps": 0, "cfg": 0, "negative_prompt": 0, "sampler_name": 0, "scheduler": 0,
    "width": 1, "height": 1, "seed": 1,
    "reference_resolution": 2, "cache_device": 2, "cache_dtype": 2,
    "attention_backend": 3, "accelerator": 3, "accelerate": 3,
    "rewrite_mode": 4, "prompt_enhancer": 4, "expand_prompt": 4, "upsampler_choice": 4,
}
# Options Compose shows itself; a change to one is not a change to the preset's settings.
COMPOSE = ("preset", "batch_count", "batch_edit_inputs", "edit_size", "layout",
           "output_resolution_1k", "output_resolution_2k")
EDIT_ONLY = ("reference_resolution", "cache_device", "cache_dtype")


@dataclass(frozen=True)
class ImagePageServices:
    """What the page needs from the application; ``root`` is where owners' drafts and uploads live."""

    root: Callable[[], Path]
    # Each model's options by job name (``qwen__steps``); MiniMax H3's own are ``image__``.
    fields: Mapping[str, FieldSpec]
    # Every name a generation job takes after the draft, in its order.
    job_fields: tuple[str, ...]
    # Each model's resolution tiers, smaller first.
    presets: Mapping[str, tuple[tuple[str, ...], ...]]
    suggest: Callable[..., tuple[str, str]]
    h3_values: Callable[[str], Mapping[str, Any]] = lambda owner: {}
    h3_defaults: Mapping[str, Any] = field(default_factory=dict)
    interrupt: Callable[..., str] | None = None
    library: Callable[[], Any] = lambda: None
    followers: tuple = ()


@dataclass
class ImageRun(Run):
    """A step of the page: what it is doing, and what it left for the page to fetch."""

    results: list[str] = field(default_factory=list)
    suggestion: str | None = None
    original: str | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "suggestion": self.suggestion, "original": self.original}


class Live(dict):
    """Each owner's options as a follower reads them: always what is saved now."""

    def __init__(self, load: Callable[[str], Any]):
        super().__init__()
        self.load = load

    def get(self, key, default=None):
        return self.load(key)


def private_image_root(runtime, workspace: Path) -> Path:
    """Keep owner drafts outside the served roots, including on upgrades."""
    outputs = runtime.outputs_dir.resolve()
    comfy = runtime.output_dir.resolve()
    root = (workspace / "image-page").resolve()
    if any(root.is_relative_to(public) for public in (outputs, comfy)):
        outer = comfy if outputs.is_relative_to(comfy) else outputs
        root = (outer.with_name(f".{outer.name}-image-drafts") / outputs.relative_to(outer)).resolve()
        if any(root.is_relative_to(public) for public in (outputs, comfy)):
            raise ValueError("Choose an H3_WORKSPACE_DIR outside the output directories for private image drafts.")
    previous = outputs / ".image-page"
    if previous.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            previous.rename(root)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            # The workspace may be on a different volume. Publish only a complete copy.
            with TemporaryDirectory(dir=root.parent) as staging:
                copy = Path(staging) / "image-page"
                shutil.copytree(previous, copy)
                copy.rename(root)
            shutil.rmtree(previous)
    return root


def key_of(engine: str, parameter: str) -> str:
    return f"{'image' if engine == 'h3' else engine}__{parameter}"


def _path_of(value) -> str | None:
    if isinstance(value, dict):
        return _path_of(value.get("path") or value.get("image"))
    if isinstance(value, (list, tuple)):
        return _path_of(value[0]) if value else None
    return str(value) if value else None


def _quote(path: str) -> str:
    return quote(str(path), safe="")


def _size(name: str) -> dict[str, Any]:
    width, height = resolution_preset_values(name)
    parts = [part.strip() for part in name.split("·")]
    return {"name": name, "ratio": parts[1] if len(parts) > 2 else f"{width}:{height}", "width": width,
            "height": height}


class ImagePageState:
    """Each owner's draft and every model's options, saved as one JSON file."""

    def __init__(self, services: ImagePageServices):
        self.services = services
        self.lock = RLock()
        self.secrets: dict[str, dict[str, str]] = {}
        self.granted: dict[str, set[str]] = {}
        self.labels = {engine.id: engine.label for engine in engines_for_task("Image")}
        for follower in services.followers:
            follower.settings = Live(self.job_options)

    # Storage ---------------------------------------------------------------
    def owner_root(self, owner: str) -> Path:
        folder = self.services.root() / re.sub(r"[^\w-]", "_", owner)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def uploads(self, owner: str) -> Path:
        folder = self.owner_root(owner) / "uploads"
        folder.mkdir(exist_ok=True)
        return folder

    def defaults(self) -> dict[str, Any]:
        return {name: spec.default for name, spec in self.services.fields.items()}

    def load(self, owner: str) -> dict[str, Any]:
        saved: dict[str, Any] = {}
        try:
            saved = json.loads((self.owner_root(owner) / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        saved = saved if isinstance(saved, dict) else {}
        options = self.defaults()
        for name, value in (saved.get("options") or {}).items():
            if name in options:
                try:
                    options[name] = coerce(self.services.fields[name], value)
                except H3Error:
                    continue
        writer = writer_defaults()
        writer.update({name: value for name, value in (saved.get("writer") or {}).items()
                       if name in writer and name not in SECRET_FIELDS})
        writer.update(self.secrets.get(owner, {}))
        files = lambda paths: [str(path) for path in paths or [] if path and Path(path).is_file()]
        # Earlier drafts stored absolute paths under the old output directory.
        uploads = self.owner_root(owner) / "uploads"
        references = [uploads / Path(path).name for path in saved.get("references") or [] if path]
        return {
            "engine": saved.get("engine") if saved.get("engine") in IMAGE_MODELS else IMAGE_MODELS[0],
            "mode": EDIT_MODE if saved.get("mode") == EDIT_MODE else GENERATE_MODE,
            "prompt": str(saved.get("prompt") or ""), "references": files(references),
            "options": options, "writer": writer, "results": files(saved.get("results")),
            "ui": saved.get("ui") if isinstance(saved.get("ui"), dict) else {},
        }

    def save(self, owner: str, state: dict[str, Any]) -> None:
        self.secrets[owner] = {name: state["writer"].get(name) or "" for name in SECRET_FIELDS}
        kept = {**state, "writer": {name: value for name, value in state["writer"].items()
                                    if name not in SECRET_FIELDS}}
        target = self.owner_root(owner) / "settings.json"
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(kept, default=str), encoding="utf-8")
        scratch.replace(target)

    def may_read(self, owner: str, path) -> bool:
        if str(path) in self.granted.get(owner, ()):
            return True
        return self.owner_root(owner).resolve() in Path(path).resolve().parents

    def grant(self, owner: str, path) -> str:
        self.granted.setdefault(owner, set()).add(str(path))
        return str(path)

    def keep(self, owner: str, source, rollback: ExitStack | None = None) -> str:
        """A copy of ``source`` among the owner's uploads, so the draft outlives where it came from."""
        target = self.uploads(owner) / f"{uuid.uuid4().hex[:12]}-{Path(source).name}"
        if rollback is not None:
            rollback.callback(target.unlink, missing_ok=True)
        shutil.copy2(source, target)
        return str(target)

    def receive(self, owner: str, engine: str, path: str) -> None:
        """Another page hands over one picture to edit with ``engine``."""
        with self.lock:
            state = self.load(owner)
            state.update(engine=engine if engine in IMAGE_MODELS else state["engine"],
                         references=[self.keep(owner, path)], mode=EDIT_MODE)
            self.save(owner, state)

    # Options ---------------------------------------------------------------
    def job_options(self, owner: str) -> dict[str, Any]:
        """Every option a generation job takes, by name: this page's, and MiniMax H3's own."""
        return self.job_values(owner, self.load(owner)["options"])

    def job_values(self, owner: str, options: Mapping[str, Any]) -> dict[str, Any]:
        h3 = {**self.services.h3_defaults, **self.services.h3_values(owner)}
        return {name: h3.get(name.removeprefix("h3__")) if name.startswith("h3__") else options.get(name)
                for name in self.services.job_fields}

    def has(self, engine: str, parameter: str) -> bool:
        return key_of(engine, parameter) in self.services.fields

    def cascade(self, options: dict[str, Any], name: str) -> None:
        """What a change to ``name`` sets with it, as the model's own view bound it."""
        engine, _, parameter = name.partition("__")
        at = lambda other: f"{engine}__{other}"
        if engine == "qwen":
            if parameter == "preset":
                before = options[at("turbo_variant")]
                (options[at("model_choice")], options[at("turbo_variant")], options[at("steps")],
                 options[at("accelerator")]) = catalog.QWEN_IMAGE21_PRESETS[options[name]]
                if options[at("turbo_variant")] != before:
                    parameter = "turbo_variant"
            if parameter == "turbo_variant":
                (options[at("steps")], options[at("cfg")], options[at("sampler_name")],
                 options[at("accelerator")]) = qwen_turbo_defaults(options[at("turbo_variant")], options[at("preset")])
            elif parameter == "edit_size" and options[name] != MANUAL:
                # A named size would otherwise override the choice just made.
                options[at("output_resolution_1k")] = options[at("output_resolution_2k")] = None
        elif engine == "flux3" and parameter == "model_choice":
            options[at("steps")], options[at("cfg")] = flux3_sampling_defaults(options[name])
        elif engine == "moss_image":
            size = options[at("width")], options[at("height")]
            if parameter == "preset":
                (options[at("turbo_variant")], options[at("steps")], options[at("cfg")], options[at("model_choice")],
                 options[at("vae_choice")]) = catalog.MOSS_IMAGE_PRESETS[options[name]]
                parameter = "width"
            elif parameter == "turbo_variant":
                options[at("steps")], options[at("cfg")] = moss_image_turbo_defaults(
                    options[name], options[at("preset")], *size)
            if parameter in ("width", "height"):
                options[at("steps")] = moss_image_turbo_size_steps(options[at("turbo_variant")], options[at("steps")], *size)
        elif engine == "fibo2" and parameter == "model_choice":
            turbo = options[name] == "Edit Turbo"
            options[at("steps")], options[at("cfg")] = (4, 1) if turbo else (30, 5)
        if parameter in ("width", "height") and engine == "qwen":
            options[at("output_resolution_1k")] = options[at("output_resolution_2k")] = None

    def apply(self, state: dict[str, Any], action: str, changes: dict[str, Any], name: str | None = None) -> None:
        """One edit of the draft or of a model's options."""
        engine, options = state["engine"], state["options"]
        if action == "engine":
            if changes.get("engine") not in IMAGE_MODELS:
                raise H3Error("Choose an image model.")
            state["engine"] = changes["engine"]
        elif action == "mode":
            if changes.get("mode") not in (GENERATE_MODE, EDIT_MODE):
                raise H3Error("Choose Generate or Edit.")
            state["mode"] = changes["mode"]
        elif action == "resolution":
            self.choose_size(engine, options, str(name or ""))
        elif action == "restore":
            self.restore(engine, options)
        else:
            for changed, value in changes.items():
                if changed not in self.services.fields:
                    raise H3Error(f"Unknown setting: {changed}")
                options[changed] = coerce(self.services.fields[changed], value)
                self.cascade(options, changed)

    def choose_size(self, engine: str, options: dict[str, Any], name: str) -> None:
        tiers = self.services.presets.get(engine, ())
        if not any(name in tier for tier in tiers):
            raise H3Error("Choose one of this model's sizes.")
        width, height = resolution_preset_values(name)
        options[key_of(engine, "width")], options[key_of(engine, "height")] = width, height
        if engine == "qwen":
            for index, field in enumerate(("qwen__output_resolution_1k", "qwen__output_resolution_2k")):
                options[field] = name if index < len(tiers) and name in tiers[index] else None
        if self.has(engine, "edit_size"):
            # As in the Gradio views, a chosen size takes priority over the first image's.
            options[key_of(engine, "edit_size")] = MANUAL
        if engine == "moss_image":
            self.cascade(options, "moss_image__width")

    def baseline(self, engine: str, options: Mapping[str, Any]) -> dict[str, Any]:
        """The model's options as its defaults and the chosen preset set them, at the chosen size."""
        clean = self.defaults()
        for parameter in ("preset", "width", "height"):
            if self.has(engine, parameter):
                clean[key_of(engine, parameter)] = options[key_of(engine, parameter)]
        if self.has(engine, "preset"):
            self.cascade(clean, key_of(engine, "preset"))
        return clean

    def restore(self, engine: str, options: dict[str, Any]) -> None:
        clean = self.baseline(engine, options)
        for name in self.fine_names(engine):
            if GROUP_OF[name.partition("__")[2]] != 1:
                options[name] = clean[name]

    def fine_names(self, engine: str) -> list[str]:
        prefix = key_of(engine, "")
        return [name for name in self.services.fields
                if name.startswith(prefix) and name.partition("__")[2] in GROUP_OF]

    # What the page shows -----------------------------------------------------
    def option(self, name: str, options: Mapping[str, Any], **extra) -> dict[str, Any]:
        return option_row(name, self.services.fields[name], options[name], **extra)

    def fine(self, engine: str, options: Mapping[str, Any], editing: bool) -> list[dict[str, Any]]:
        clean = self.baseline(engine, options)
        value = lambda parameter: options.get(key_of(engine, parameter))
        hidden = set()
        if engine == "h3" and value("backend") == H3_IMAGE_PRUNA:
            hidden.add("expand_prompt")
        if engine == "qwen":
            if value("cfg") <= 1:
                hidden.add("negative_prompt")
            if value("turbo_variant") not in QWEN_IMAGE21_LIGHTNING_MODES:
                hidden.add("lightning_strength")
        if engine == "flux3" and not str(value("model_choice")).startswith("Base "):
            hidden.add("cfg")
        locked = {"steps", "cfg"} if engine == "fibo2" and value("model_choice") == "Edit Turbo" else set()
        rows = []
        for name in self.fine_names(engine):
            parameter = name.partition("__")[2]
            if parameter in hidden:
                continue
            group = GROUP_OF[parameter]
            idle = parameter in EDIT_ONLY and not editing
            changed = group != 1 and not idle and options[name] != clean[name]
            rows.append(self.option(name, options, group=group, disabled=parameter in locked or idle,
                                    note="applies to edits" if idle else "set by the model" if parameter in locked else "",
                                    preset=clean[name] if changed else None, changed=changed))
        return rows

    def issue(self, state: Mapping[str, Any]) -> str:
        engine, options = state["engine"], state["options"]
        return draft_issue(engine, image_mode(engine, state["mode"]), state["prompt"], state["references"],
                           options.get("qwen__turbo_variant", "Off"), options.get("fibo2__model_choice", "Dev"),
                           bool(options.get("qwen__batch_edit_inputs")),
                           options.get("image__backend", H3_IMAGE_OFFICIAL))

    def guidance(self, state: Mapping[str, Any], editing: bool) -> str:
        engine, count = state["engine"], len(state["references"])
        note = ""
        if editing:
            limit = reference_limit(engine, state["options"].get("qwen__turbo_variant", "Off"),
                                    state["options"].get("image__backend", H3_IMAGE_OFFICIAL))
            note = f"Up to {limit} images per edit, in this order."
        if editing and engine == "qwen":
            note += " The first image is the target; name several as <image1>, <image2>, …"
        if engine == "fibo2":
            note += " Plain text or VGL JSON" + ("; refer to images as <image_1>, <image_2>, …" if editing else ".")
        if editing and engine == "h3":
            note += " Refer to images as <Picture 1>, <Picture 2>, …"
        if editing and engine == "flux3":
            note += " Layout boxes can name <ref_image_0>, <ref_image_1>, …; a tag adds its name there."
        if not editing and count:
            note += f" {count} saved image{'s are' if count > 1 else ' is'} inactive in Generate."
        return note.strip()

    def describe(self, owner: str, state: dict[str, Any]) -> dict[str, Any]:
        engine, options = state["engine"], state["options"]
        mode = image_mode(engine, state["mode"])
        editing = mode == EDIT_MODE
        separate = bool(options.get("qwen__batch_edit_inputs"))
        count = len(state["references"])
        issue = self.issue(state)
        rows = self.fine(engine, options, editing)
        if engine in WRITER_ENGINES:
            rows += writer_rows(state["writer"], group=4)
        else:
            rows.append(review_row(state["writer"], group=4))
        if engine not in WRITER_ENGINES and state["writer"]["review"]:
            # Models without a prompt writer still need a key for result scores and tags.
            rows.append({"name": "gemini_api_key", "label": "Temporary Gemini API key", "kind": "password",
                         "value": state["writer"]["gemini_api_key"], "store": "writer", "group": 4, "choices": [],
                         "placeholder": "Uses GEMINI_API_KEY when blank",
                         "hint": "Only for result scores and tags. The server does not store keys entered here."})
        counts: dict[str, int] = {}
        for row in rows:
            if row.get("changed"):
                counts[str(row["group"])] = counts.get(str(row["group"]), 0) + 1
        compose = lambda parameter, shown=True: (
            self.option(key_of(engine, parameter), options) if shown and self.has(engine, parameter) else None)
        paths = [self.grant(owner, path) for path in state["results"]]
        return {
            "engine": engine, "mode": mode, "editing": editing, "generate_only": engine == "moss_image",
            "prompt": state["prompt"], "ui": state["ui"], "options": options,
            "references": [{"path": path, "url": f"{API}/file?path={_quote(path)}",
                            "detail": re.sub(r"^[0-9a-f]{12}-", "", describe_reference(path, "Picture")),
                            "tag": reference_tag(engine, index, count, separate)}
                           for index, path in enumerate(state["references"], 1)],
            "fields": rows, "changed": counts,
            "preset": compose("preset"), "batch": compose("batch_count"),
            "edit_size": compose("edit_size", editing), "separate": compose("batch_edit_inputs", editing),
            "layout": compose("layout"),
            "width": options[key_of(engine, "width")], "height": options[key_of(engine, "height")],
            "seed": options[key_of(engine, "seed")],
            "guidance": self.guidance(state, editing),
            "readiness": {"ready": not issue,
                          "message": issue or f"Ready to {'edit' if editing else 'generate'} with {self.labels[engine]}."},
            "can_enhance": engine in WRITER_ENGINES,
            "writer": state["writer"]["backend"],
            "results": {"images": [{"path": path, "name": Path(path).name, "url": f"{API}/file?path={_quote(path)}",
                                    "review": media_review.public(path)} for path in paths],
                        "settings": render_snapshot(paths) if paths else ""},
        }


def install_image_page(app: FastAPI, pages: ImagePageState, *, gpu: bool = True) -> ImagePageState:
    """Serve the page at ``/image/`` and its API under ``/image/api``."""
    services = pages.services
    runs = {kind: Runs(busy=message) for kind, message in BUSY.items()}

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        """What outlives ``request`` and names its owner to the job path."""
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None, headers=dict(request.headers))

    def require_gpu():
        if not gpu or "image" not in JOBS.callbacks:
            raise H3Error("Open Image in Studio to generate on the GPU.")

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

    # The page --------------------------------------------------------------
    @app.get(ROUTE, include_in_schema=False)
    def image_redirect():
        return RedirectResponse(f"{ROUTE}/")

    @app.get(f"{ROUTE}/", include_in_schema=False)
    def image_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get(ROUTE + "/{name}", include_in_schema=False)
    def image_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        with pages.lock:
            described = answer(owner, pages.load(owner))
        sizes = {}
        for engine in IMAGE_MODELS:
            sizes[engine] = [{"label": names[0].split("·")[0].strip(), "sizes": [_size(name) for name in names]}
                             for names in services.presets.get(engine, ()) if names]
        return {
            **described, "gpu": gpu and "image" in JOBS.callbacks,
            "engines": [{"id": engine.id, "label": engine.label, "description": engine.description}
                        for engine in sorted(engines_for_task("Image"), key=lambda item: IMAGE_MODELS.index(item.id))],
            "groups": [list(group) for group in GROUPS], "sizes": sizes,
            "modes": [["Generate", GENERATE_MODE], ["Edit", EDIT_MODE]],
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

    @app.post(f"{API}/empty", include_in_schema=False)
    @guarded
    def empty_draft(request: Request):
        def change(state):
            state.update(prompt="", references=[], mode=GENERATE_MODE)
            for engine in IMAGE_MODELS:
                name = key_of(engine, "layout")
                if name in state["options"]:
                    state["options"][name] = ""

        return edit(request, change)

    @app.post(f"{API}/prompt", include_in_schema=False)
    @guarded
    def change_prompt(request: Request, body: dict = Body(...)):
        def change(state):
            if "layout" in body:
                layout = body["layout"]
                if layout["name"] != key_of(state["engine"], "layout"):
                    raise H3Error("The image model changed. Refresh the draft before saving.")
                pages.apply(state, "edit", {layout["name"]: layout["value"]})
            state["prompt"] = str(body.get("prompt") or "")
            if body.get("accepted"):
                if state["engine"] == "h3":
                    state["options"]["image__expand_prompt"] = False
                # A remote rewrite is already expanded, and a rewritten Layout carries its own boxes.
                if state["engine"] == "flux3":
                    state["options"]["flux3__expand_prompt"] = False
                if state["engine"] == "moss_image" and is_layout_prompt(state["prompt"]):
                    state["options"]["moss_image__layout"] = ""

        return edit(request, change)

    @app.post(f"{API}/ui", include_in_schema=False)
    @guarded
    def change_ui(request: Request, body: dict = Body(...)):
        """What only the page reads: the open Fine-tune group."""
        ui = body.get("ui")
        if not isinstance(ui, dict) or len(json.dumps(ui)) > UI_LIMIT:
            raise H3Error("That is too much to remember.")
        return edit(request, lambda state: state.update(ui=ui))

    # Images to edit ----------------------------------------------------------
    def add(owner: str, paths: list[str], rollback: ExitStack) -> dict[str, Any]:
        with pages.lock:
            state = pages.load(owner)
            if len(state["references"]) + len(paths) > REFERENCE_LIMIT:
                raise H3Error(f"Keep at most {REFERENCE_LIMIT} images in the draft.")
            state["references"] += paths
            state["mode"] = EDIT_MODE
            pages.save(owner, state)
            rollback.pop_all()
            return answer(owner, state)

    @app.post(f"{API}/references/upload", include_in_schema=False)
    @guarded
    def upload_references(request: Request, files: list[UploadFile] = File(...)):
        owner = owner_of(request)
        folder = pages.uploads(owner)
        with ExitStack() as rollback:
            paths = [str(spool_image(upload, folder, rollback)) for upload in files]
            return add(owner, paths, rollback)

    @app.post(f"{API}/references/library", include_in_schema=False)
    @guarded
    def reference_from_library(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        source = library_source(services.library(), str(body.get("source") or ""))
        with ExitStack() as rollback:
            path = pages.keep(owner, source, rollback)
            return add(owner, [path], rollback)

    def position(state: dict[str, Any], value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(state["references"]):
            raise H3Error("Choose one of the images in the list.")
        return value

    @app.post(f"{API}/references/remove", include_in_schema=False)
    @guarded
    def remove_reference(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: state["references"].pop(position(state, body.get("index"))))

    @app.post(f"{API}/references/clear", include_in_schema=False)
    @guarded
    def clear_references(request: Request):
        return edit(request, lambda state: state["references"].clear())

    @app.post(f"{API}/references/move", include_in_schema=False)
    @guarded
    def move_reference(request: Request, body: dict = Body(...)):
        def change(state):
            source, target = position(state, body.get("from")), position(state, body.get("to"))
            state["references"].insert(target, state["references"].pop(source))

        return edit(request, change)

    @app.post(f"{API}/references/use-result", include_in_schema=False)
    @guarded
    def use_result(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        path = str(body.get("path") or "")

        def change(state):
            if state["engine"] == "moss_image":
                raise H3Error("MOSS-Image supports generation only. Choose an edit-capable model first.")
            if path not in state["results"] or not Path(path).is_file():
                raise H3Error("Select an image result first.")
            state.update(references=[pages.keep(owner, path)], mode=EDIT_MODE)

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
    def start(owner: str, kind: str, work: Callable[[ImageRun], None]) -> dict[str, Any]:
        run = ImageRun(owner, RUN_KEY, kind)
        run.status = "Starting…"
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
            engine = state["engine"]
            mode = image_mode(engine, state["mode"])
            references = list(state["references"]) if mode == EDIT_MODE else []
            prompt, writer = state["prompt"], dict(state["writer"])
            values = [engine, mode, state["prompt"], references,
                      *pages.job_values(owner, state["options"]).values()]
        label = pages.labels[engine]

        def work(run: ImageRun):
            job = JOBS.accept(owner, "image", values, callback=JOBS.callbacks["image"], key=uuid.uuid4().hex,
                              media_indices=(3,) if references else (), output_indices=(0,))
            run.job_id = job.id
            try:
                for update in execute_accepted(job.id, session):
                    sets, value = update_value(update[0])
                    if sets and value:
                        items = value if isinstance(value, (list, tuple)) else [value]
                        run.results = [path for path in map(_path_of, items) if path]
                    status = update_value(update[1])[1] if len(update) > 1 else None
                    if isinstance(status, str) and status:
                        run.status = plain(status)
                    run.revision += 1
            finally:
                # Interrupted or failed later variations still leave usable images.
                # Read the latest draft so edits made during generation survive.
                if run.results:
                    with pages.lock:
                        latest = pages.load(owner)
                        latest["results"] = list(run.results)
                        pages.save(owner, latest)
            for path in run.results:
                # In the background: the score and tags arrive after the image is shown.
                media_review.schedule(path, prompt, api_key=writer.get("gemini_api_key") or "",
                                      enabled=bool(writer.get("review", True)), edited=bool(references))
            if not run.status.startswith(label):
                run.status = f"{label} · {run.status}"

        return start(owner, GENERATE, work)

    @app.post(f"{API}/enhance", include_in_schema=False)
    @guarded
    def enhance(request: Request):
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
        engine, options, writer = state["engine"], state["options"], state["writer"]
        mode = image_mode(engine, state["mode"])
        prompt, references = state["prompt"], list(state["references"])
        arguments = {
            "h3_width": options.get("image__width"), "h3_height": options.get("image__height"),
            "h3_edit_size": options.get("image__edit_size"), "h3_backend": options.get("image__backend"),
            "moss_rewrite_mode": options.get("moss_image__rewrite_mode"), "qwen_edit_size": options.get("qwen__edit_size"),
            **{f"{model}_{side}": options.get(f"{model}__{side}") for model in WRITER_ENGINES if model != "h3" for side in ("width", "height")},
            "writer_model": writer["model"], "writer_key": writer["gemini_api_key"], "writer_backend": writer["backend"],
            "writer_openai_key": writer["openai_api_key"], "writer_openai_model": writer["openai_model"],
            "writer_base_url": writer["openai_base_url"] or None,
        }

        def work(run: ImageRun):
            run.status = "Writing…"
            suggestion, status = services.suggest(engine, mode, prompt, references, arguments)
            run.original = prompt
            if not suggestion or suggestion == prompt:
                run.status = plain(status)
                return
            run.suggestion = suggestion
            run.status = f"{plain(status)} Preview ready; choose Accept or Keep original."

        return start(owner, ENHANCE, work)

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, state: bool = False):
        owner = owner_of(request)
        found: dict[str, Any] = {"runs": run_payloads(owner)}
        generation = runs[GENERATE].latest(owner, RUN_KEY)
        if generation is not None:
            found["progress"] = [{"path": path, "name": Path(path).name,
                                  "url": f"{API}/file?path={_quote(pages.grant(owner, path))}"}
                                 for path in generation.results]
        if state:
            with pages.lock:
                found["state"] = pages.describe(owner, pages.load(owner))
        return found

    @app.post(f"{API}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request):
        require_gpu()
        owner = owner_of(request)
        run = runs[GENERATE].active(owner, RUN_KEY)
        if services.interrupt is None:
            raise H3Error("Nothing can run here.")
        if run is None:
            message = "Nothing is running."
        elif run.job_id is None:
            message = "The run is still starting. Try again in a moment."
        else:
            message = services.interrupt(session_of(request), family="image", job_id=run.job_id)
        return {"message": message, "runs": run_payloads(owner)}

    return pages
