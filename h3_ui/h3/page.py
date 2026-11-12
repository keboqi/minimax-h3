"""The MiniMax H3 page and its JSON API, outside Gradio, as Remake's and Storyboard's are.

The page holds no policy. Every settings change goes through the same pure transitions the old
Gradio view used (``transition_modes`` and ``resolve_settings``), kept per owner on disk so a
browser reopens where it left off. A run takes the same job path as every other generation, in a
thread here, and the page polls it. Remake and Storyboard follow the settings saved here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from h3_app import catalog
from h3_models import (
    H3_LATENT_UPSCALER_MODEL_CHOICES, H3_TEXT_ENCODER_CHOICES, LTX25_MODEL_CHOICES, SEEDVR2_MODEL_CHOICES,
)
from h3_app.contracts import GENERATION_FIELDS
from h3_app.decoder_intent import VideoDecoder
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app import media_review
from h3_app.provenance import render_snapshot
from h3_app.reference_bindings import KINDS, ReferenceBinding, ReferenceMap
from h3_app.reference_metadata import describe_reference
from h3_app.settings import transition_modes
from h3_app.storyboard import PROFILE_FIELDS
from h3_app.storyboard.model import request_defaults
from h3_app.workspace_store import request_owner

from ..job_admission import execute_accepted
from ..page_runs import Run, Runs, guarded
from ..presentation import generation_readiness
from ..storyboard.handlers import WRITER_FIELDS
from . import layout

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/h3"
API = f"{ROUTE}/api"
RUN_KEY = "h3"
GENERATE, ENHANCE, UPSCALE, COMPILE = "generate", "enhance", "input-upscale", "compile"
BUSY = {GENERATE: "A generation is already running.", ENHANCE: "The prompt writer is already running.",
        UPSCALE: "Inputs are already being upscaled.", COMPILE: "The TensorRT engine is already compiling."}

IMAGE_SLOTS = ("first_image", "last_image", *(f"ref_image_{i}" for i in range(1, 10)))
VIDEO_SLOTS = tuple(f"ref_video_{i}" for i in range(1, 4))
AUDIO_SLOTS = (*(f"ref_audio_{i}" for i in range(1, 4)), *(f"fl2va_audio_{i}" for i in range(1, 4)))
MEDIA_FIELDS = (*IMAGE_SLOTS, *VIDEO_SLOTS, *AUDIO_SLOTS)
FRAME_SLOTS = ("first_image", "last_image", "fl2va_audio_1", "fl2va_audio_2", "fl2va_audio_3")
REFERENCE_SLOTS = tuple(name for name in MEDIA_FIELDS if name.startswith("ref_"))
# Everything a run takes that is neither the prompt nor media, plus the recipe and the variant count.
SETTING_FIELDS = (*(name for name in GENERATION_FIELDS if name not in MEDIA_FIELDS and name != "prompt"),
                  "preset", "batch_count")
SECRET_FIELDS = ("gemini_api_key", "openai_api_key")
INTEGER_FIELDS = frozenset({"width", "height", "steps", "seed", "sol_dense_steps", "sol_sink_tokens",
                            "fbcache_max_hits", "latent_upscale_refine_steps", "batch_count", "image_frames",
                            "local_prompt_max_tokens", "local_prompt_seed"})
DECODER_FLAGS = ("use_int8_vae", "use_lynnreal_vae", "use_trt_vae", "lightvae_decoder", "use_sol_vae")
FBCACHE_VALUES = ("fbcache_threshold", "fbcache_start", "fbcache_end", "fbcache_max_hits")
UI_LIMIT = 4096
# ``upscale_selected_input_images`` answers with these eleven pictures, then the files and a status.
UPSCALE_OUTPUTS = IMAGE_SLOTS


@dataclass(frozen=True)
class H3PageServices:
    """What the page needs from the application; ``root`` is where owners' uploads and settings live."""

    root: Callable[[], Path]
    resolve: Callable[[dict], Any]
    describe: Callable[[dict], str]
    enhance: Callable[..., tuple[str, str]]
    interrupt: Callable[..., str] | None = None
    compile_trt: Callable[[], str] | None = None
    library: Callable[[], Any] = lambda: None
    followers: tuple = ()
    attention_backend: str = "sol"


@dataclass
class H3Run(Run):
    """A step of the page: what it is doing, and what it left for the page to fetch."""

    results: dict[int, str] = field(default_factory=dict)
    suggestion: str | None = None
    original: str | None = None
    files: list[str] = field(default_factory=list)

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "suggestion": self.suggestion, "original": self.original}


class Followed(dict):
    """Another page's per-owner copy of these settings, read from disk for an owner not yet seen."""

    def __init__(self, load: Callable[[str], Any]):
        super().__init__()
        self.load = load

    def get(self, key, default=None):
        if key not in self and (loaded := self.load(key)) is not None:
            self[key] = loaded
        return super().get(key, default)


def writer_defaults() -> dict[str, Any]:
    try:
        from h3_prompt_rewriter import DEFAULT_BASE_MODEL_LABEL as local_model
    except Exception:  # The local writer is optional where no GPU runs.
        local_model = ""
    return {
        "prompt_writer_backend": catalog.DEFAULT_PROMPT_WRITER_BACKEND, "local_prompt_base_model": local_model,
        "local_prompt_max_tokens": 4096, "local_prompt_temperature": 0.7, "local_prompt_top_p": 0.8,
        "local_prompt_greedy": True, "local_prompt_seed": 42,
        "gemini_prompt_model": catalog.DEFAULT_GEMINI_PROMPT_MODEL, "gemini_api_key": "", "openai_api_key": "",
        "openai_model": catalog.DEFAULT_OPENAI_PROMPT_MODEL, "openai_base_url": "",
        # Score each video against its prompt and tag it with Gemini Decision; Storyboard follows it.
        "review": True,
    }


def setting_defaults() -> dict[str, Any]:
    values = {**request_defaults(), "preset": "Singularity",
              "batch_count": catalog.DEFAULT_VIDEO_BATCH_COUNT, "result_format": "Video"}
    return {name: values[name] for name in SETTING_FIELDS}


def coerce(name: str, value: Any, default: Any) -> Any:
    """``value`` as the kind its default has; a browser sends 6 for 6.0 and may send text for a number."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise H3Error(f"{name} must be on or off.")
        return value
    if isinstance(default, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise H3Error(f"{name} must be a number.") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise H3Error(f"{name} must be a number.")
        return int(round(number)) if name in INTEGER_FIELDS else number
    return "" if value is None else str(value)


def plain(text: str) -> str:
    """Markdown or HTML status text as the plain line the page shows."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|\*\*", " ", str(text or ""))).strip()


def readiness_line(ready) -> str:
    """The readiness card as one line: what is missing, or that nothing is."""
    if ready.ready:
        return "Ready to generate."
    missing = re.search(r"<span>(.*?)</span>", ready.html)
    return f"Before generating: {plain(missing[1]) if missing else plain(ready.html)}"


def _map_from(saved) -> ReferenceMap:
    try:
        return ReferenceMap(int(saved["version"]), tuple(ReferenceBinding(**item) for item in saved["bindings"]))
    except (KeyError, TypeError, ValueError):
        return ReferenceMap()


def _map_to(mapping: ReferenceMap) -> dict:
    return {"version": mapping.version,
            "bindings": [{"asset_id": b.asset_id, "kind": b.kind, "slot": b.slot, "path": b.path,
                          "engine_ordinal": b.engine_ordinal, "unresolved": b.unresolved} for b in mapping.bindings]}


def _spool(upload: UploadFile, folder: Path, allowed) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise H3Error(f"{upload.filename} isn't a supported file ({', '.join(sorted(allowed))}).")
    stem = re.sub(r"[^\w. -]", "_", Path(upload.filename or "").stem).strip(" .")[:60] or "upload"
    path = folder / f"{uuid.uuid4().hex[:12]}-{stem}{suffix}"
    with path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return path


def _update_value(update) -> tuple[bool, Any]:
    """Whether a Gradio update sets a value, and the value."""
    if isinstance(update, dict) and update.get("__type__") == "update":
        return ("value" in update, update.get("value"))
    return (True, update)


def _path_of(value) -> str | None:
    if isinstance(value, dict):
        return _path_of(value.get("path") or value.get("video"))
    if isinstance(value, (list, tuple)):
        return _path_of(value[0]) if value else None
    return str(value) if value else None


class H3PageState:
    """Each owner's settings, prompt and media, saved as one JSON file."""

    def __init__(self, services: H3PageServices):
        self.services = services
        self.lock = RLock()
        self.secrets: dict[str, dict[str, str]] = {}
        self.granted: dict[str, set[str]] = {}

    # Storage ---------------------------------------------------------------
    def owner_root(self, owner: str) -> Path:
        folder = self.services.root() / re.sub(r"[^\w-]", "_", owner)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def uploads(self, owner: str) -> Path:
        folder = self.owner_root(owner) / "uploads"
        folder.mkdir(exist_ok=True)
        return folder

    def load(self, owner: str) -> dict[str, Any]:
        defaults = setting_defaults()
        saved: dict[str, Any] = {}
        try:
            saved = json.loads((self.owner_root(owner) / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        saved = saved if isinstance(saved, dict) else {}
        values = dict(defaults)
        for name, value in (saved.get("values") or {}).items():
            if name in defaults:
                try:
                    values[name] = coerce(name, value, defaults[name])
                except H3Error:
                    continue
        values["result_format"] = "Video"
        writer = writer_defaults()
        writer.update({name: value for name, value in (saved.get("writer") or {}).items()
                       if name in writer and name not in SECRET_FIELDS})
        writer.update(self.secrets.get(owner, {}))
        media = {name: path for name, path in (saved.get("media") or {}).items()
                 if name in MEDIA_FIELDS and path and Path(path).is_file()}
        return {"values": values, "memory": saved.get("memory") or None, "writer": writer, "media": media,
                "prompt": str(saved.get("prompt") or ""), "references": saved.get("references") or {},
                "results": [path for path in saved.get("results") or [] if path and Path(path).is_file()],
                "ui": saved.get("ui") if isinstance(saved.get("ui"), dict) else {}}

    def save(self, owner: str, state: dict[str, Any]) -> None:
        self.secrets[owner] = {name: state["writer"].get(name) or "" for name in SECRET_FIELDS}
        kept = {**state, "writer": {name: value for name, value in state["writer"].items()
                                    if name not in SECRET_FIELDS}}
        target = self.owner_root(owner) / "settings.json"
        scratch = target.with_suffix(".tmp")
        scratch.write_text(json.dumps(kept, default=str), encoding="utf-8")
        scratch.replace(target)
        self.follow(owner, state)

    def follow(self, owner: str, state: dict[str, Any]) -> None:
        """Hand Remake and Storyboard the settings they render with."""
        for follower in self.services.followers:
            follower.settings[owner] = self.followed(follower, state)

    @staticmethod
    def followed(follower, state: dict[str, Any]):
        profile = {name: state["values"][name] for name in PROFILE_FIELDS if name in state["values"]}
        if hasattr(follower, "capture_settings"):
            return {"profile": profile, "writer": {name: state["writer"].get(name) for name in WRITER_FIELDS},
                    "review": bool(state["writer"].get("review", True))}
        return profile

    def may_read(self, owner: str, path) -> bool:
        if str(path) in self.granted.get(owner, ()):
            return True
        return self.owner_root(owner).resolve() in Path(path).resolve().parents

    def grant(self, owner: str, path) -> str:
        self.granted.setdefault(owner, set()).add(str(path))
        return str(path)

    # Requests --------------------------------------------------------------
    @staticmethod
    def request_values(state: dict[str, Any]) -> dict[str, Any]:
        """Every ``generate`` value by name, for the resolver, the writer and a run."""
        values = {**request_defaults(), **state["values"], **{name: None for name in MEDIA_FIELDS},
                  **state["media"], "prompt": state["prompt"]}
        values["result_format"] = "Video"
        return values

    def references(self, state: dict[str, Any], *, repair=False) -> ReferenceMap:
        mapping = _map_from(state.get("references")).update(state["media"], repair=repair)
        state["references"] = _map_to(mapping)
        return mapping

    # Transitions -----------------------------------------------------------
    def apply(self, state: dict[str, Any], action: str, changes: dict[str, Any], size: dict | None = None) -> None:
        """One settings edit, with the same policy the Gradio view applied."""
        defaults = setting_defaults()
        values = state["values"]
        before = dict(values)
        for name, value in changes.items():
            if name not in defaults or name == "result_format":
                raise H3Error(f"Unknown setting: {name}")
            values[name] = coerce(name, value, defaults[name])
        if action.startswith("decoder:"):
            values.update(VideoDecoder(action.removeprefix("decoder:")).workflow_flags())
        memory = state.get("memory") or {"active": before["generation_mode"], "modes": {}}
        if action == "generation_mode":
            # The outgoing mode keeps the settings it had before this edit.
            _memory, _current = transition_modes(memory, before, "edit")
            memory = {**memory, **_memory}
        new_memory, current = transition_modes(memory, values, action)
        if action == "text_encoder":
            if current["text_encoder"] == "BF16":
                new_memory["offload_preference"] = before["stage_model_offload"]
                current["stage_model_offload"] = True
            else:
                current["stage_model_offload"] = memory.get("offload_preference", current["stage_model_offload"])
        elif "offload_preference" in memory:
            new_memory["offload_preference"] = memory["offload_preference"]
        if action == "fbcache_preset":
            for name, update in zip(FBCACHE_VALUES, layout.fbcache_preset_defaults(current["fbcache_preset"])):
                if "value" in update:
                    current[name] = update["value"]
        latent = bool(current["latent_upscale"])
        width, height = current["width"], current["height"]
        first = state["media"].get("first_image") if current["mode"] == "First / last frame" else None
        if size and size.get("tier"):
            width, height, _info = layout.resolution_choice_updates(size.get("name"), size["tier"], latent, "Video")
        elif action in {"first_image", "auto_megapixels", "mode"} and first:
            width, height, info = layout.auto_resolution_from_start_frame(
                first, width, height, "Video", latent, current["auto_megapixels"])
            if info.startswith("⚠"):
                raise H3Error(plain(info))
        elif action == "latent_upscale":
            _group, width, height, _info = layout.latent_upscale_layout_updates(latent, width, height, "Video")
        elif action in {"width", "height"}:
            width, height, _info = layout.resolution_control_updates(width, height, latent, "Video")
        current.update(width=int(width), height=int(height), result_format="Video")
        state["values"] = {name: current[name] for name in SETTING_FIELDS}
        state["memory"] = new_memory

    def describe(self, owner: str, state: dict[str, Any]) -> dict[str, Any]:
        """What the page shows: the values, what policy made of them, and the references."""
        values = state["values"]
        request = {**self.request_values(state), "_settings_ready": True}
        locked, inactive, issues, differences = {}, [], [], {}
        try:
            plan = self.services.resolve(request)
        except (ValueError, TypeError, RuntimeError, H3Error) as exc:
            issues = [str(exc)]
        else:
            inactive, issues = sorted(plan.inactive), list(plan.issues)
            differences = {key: [before, after] for key, (before, after) in plan.differences().items()}
            for name in ("steps", "scheduler"):
                if name in plan.inactive:
                    locked[name] = getattr(plan.effective.sampling, name)
            if "stage_model_offload" in plan.inactive:
                locked["stage_model_offload"] = True
            if "semantic_bridge" in plan.inactive:
                locked["semantic_bridge"] = False
        references = [name for name in REFERENCE_SLOTS if state["media"].get(name)]
        ready = generation_readiness(values["mode"], state["prompt"], state["media"].get("first_image"),
                                     state["media"].get("last_image"), references)
        mapping = self.references(state)
        try:
            decoder = VideoDecoder.from_flags(values["use_int8_vae"], values["use_lynnreal_vae"],
                                              values["use_trt_vae"], values["lightvae_decoder"],
                                              values["use_sol_vae"]).value
        except ValueError:
            decoder = ""
        latent = bool(values["latent_upscale"])
        return {
            "values": values, "writer": state["writer"], "prompt": state["prompt"], "ui": state["ui"],
            "media": {name: self.media_item(name, path) for name, path in state["media"].items()},
            "references": [{"tag": b.tag, "engine": b.engine_tag, "kind": b.kind, "slot": b.slot,
                            "unresolved": b.unresolved, "active": bool(b.engine_ordinal)}
                           for b in mapping.bindings],
            "reference_version": mapping.version,
            "locked": locked, "inactive": inactive, "issues": issues, "differences": differences,
            "decoder": decoder, "summary": self.services.describe(request),
            "resolution": ("Latent upscale 64-pixel alignment · " if latent else "")
            + plain(layout.resolution_summary(values["width"], values["height"])),
            "readiness": {"ready": ready.ready and not issues,
                          "message": " ".join(issues) or readiness_line(ready)},
            "results": self.results(owner, state),
        }

    @staticmethod
    def media_item(name: str, path: str) -> dict[str, Any]:
        kind = "Picture" if name in IMAGE_SLOTS else "Video" if name in VIDEO_SLOTS else "Audio"
        # Uploads are kept under a random prefix; people see the name they chose.
        detail = re.sub(r"^[0-9a-f]{12}-", "", describe_reference(path, kind))
        return {"path": path, "kind": kind, "detail": detail,
                "url": f"{API}/file?path={_quote(path)}"}

    def results(self, owner: str, state: dict[str, Any]) -> dict[str, Any]:
        paths = [self.grant(owner, path) for path in state["results"]]
        return {"videos": [{"path": path, "name": Path(path).name, "url": f"{API}/file?path={_quote(path)}",
                            "review": media_review.public(path)} for path in paths],
                "settings": render_snapshot(paths) if paths else ""}


def _quote(path: str) -> str:
    from urllib.parse import quote

    return quote(str(path), safe="")


def install_h3_page(app: FastAPI, services: H3PageServices, *, gpu: bool = True) -> H3PageState:
    """Serve the page at ``/h3/`` and its API under ``/h3/api``."""
    pages = H3PageState(services)
    runs = {kind: Runs(busy=message) for kind, message in BUSY.items()}
    for follower in services.followers:
        def load(owner, follower=follower):
            if not (pages.services.root() / re.sub(r"[^\w-]", "_", owner) / "settings.json").is_file():
                return None
            return pages.followed(follower, pages.load(owner))

        follower.settings = Followed(load)

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        """What outlives ``request`` and names its owner to the job path."""
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None, headers=dict(request.headers))

    def require_gpu():
        if not gpu:
            raise H3Error("Open MiniMax H3 in Studio to generate on the GPU.")

    def answer(owner: str, state: dict[str, Any], **extra) -> dict[str, Any]:
        return {"state": pages.describe(owner, state), "runs": run_payloads(owner), **extra}

    def run_payloads(owner: str) -> dict[str, Any]:
        found = {}
        for kind, items in runs.items():
            run = items.latest(owner, RUN_KEY)
            found[kind] = run.public() if run is not None else None
        return found

    def edit(request: Request, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
            change(state)
            # References are rebuilt before saving so a replaced card is remembered.
            pages.references(state)
            pages.save(owner, state)
            return answer(owner, state)

    # The page --------------------------------------------------------------
    @app.get(ROUTE, include_in_schema=False)
    def h3_redirect():
        return RedirectResponse(f"{ROUTE}/")

    @app.get(f"{ROUTE}/", include_in_schema=False)
    def h3_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get(ROUTE + "/{name}", include_in_schema=False)
    def h3_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
            pages.follow(owner, state)
            described = answer(owner, state)
        try:
            from h3_prompt_rewriter import BASE_MODEL_CHOICES as local_models
        except Exception:
            local_models = ()
        return {
            **described, "gpu": gpu, "defaults": setting_defaults(),
            "choices": {
                "preset": ["Singularity", "Quality", "Balanced", "Fast"],
                "mode": ["Text to video", "First / last frame", "Reference media"],
                "model_profile": list(catalog.MODEL_PROFILE_CHOICES),
                "generation_mode": ["Normal", "Turbo"],
                "turbo_variant": list(catalog.TURBO_SETTINGS),
                "scheduler": ["simple", "beta", "normal"],
                "auto_megapixels": list(catalog.AUTO_RESOLUTION_MEGAPIXEL_PRESETS),
                "attention_mode": ["Sage 2", "Kitchen", "SLA", "Veda", "Sol-Attn", "Auto"],
                "sla_preset": list(catalog.SLA_PRESET_INPUTS),
                "sol_thresh_type": ["diag", "exact"],
                "sol_exact_mode": ["off", "exact_kv", "exact_kv_and_rows"],
                "cache_mode": ["Spectrum", "FirstBlockCache", "EasyCache", "Off"],
                "fbcache_preset": ["Safe", "Fast", "Aggressive", "Custom"],
                "text_encoder": list(H3_TEXT_ENCODER_CHOICES),
                "video_encoder": ["Original", "LightVAE Lite"],
                "video_decoder": [item.value for item in VideoDecoder],
                "latent_upscaler_model": list(H3_LATENT_UPSCALER_MODEL_CHOICES),
                "latent_upscale_refine_lora": list(catalog.REFINEMENT_LORA_CHOICES),
                "latent_upscale_method": list(catalog.H3_LATENT_UPSCALE_METHODS),
                "postprocess": list(catalog.GENERATION_POSTPROCESS_OPTIONS),
                "upscale_resolution": list(catalog.UPSCALE_RESOLUTION_PRESETS),
                "seedvr2_model": list(SEEDVR2_MODEL_CHOICES),
                "ltx25_model": list(LTX25_MODEL_CHOICES),
                "ref_image_size": ["match", "max"],
                "prompt_writer_backend": list(catalog.PROMPT_WRITER_BACKENDS),
                "local_prompt_base_model": list(local_models),
                "gemini_prompt_model": list(catalog.GEMINI_PROMPT_MODELS),
                "openai_model": list(catalog.OPENAI_PROMPT_MODELS),
                "input_slots": list(catalog.INPUT_IMAGE_UPSCALE_SLOTS),
                "input_preset": list(catalog.INPUT_IMAGE_FRAME_PRESETS),
            },
            "resolutions": {"draft": list(catalog.DRAFT_RESOLUTIONS), "fast": list(catalog.FAST_RESOLUTIONS),
                            "large": list(catalog.LARGE_RESOLUTIONS)},
            "input_presets": {name: size for name, size in catalog.INPUT_IMAGE_FRAME_PRESETS.items()},
            "limits": {"batch": [catalog.MIN_VIDEO_BATCH_COUNT, catalog.MAX_VIDEO_BATCH_COUNT],
                       "face_refine_strength": list(catalog.FACE_REFINE_STRENGTH_RANGE)},
            "postprocess_kinds": {"ai": sorted(catalog.AI_POSTPROCESS_OPTIONS), "seedvr2": catalog.SEEDVR2_UPSCALE,
                                  "ltx": [catalog.LTX25_UPSCALE, catalog.LTX25_TILED_UPSCALE],
                                  "regenerate": list(catalog.H3_REGENERATE_MODELS)},
            "attention_locked": services.attention_backend != "sol",
            "default_seedvr2": request_defaults().get("seedvr2_model"),
        }

    # Settings --------------------------------------------------------------
    @app.post(f"{API}/settings", include_in_schema=False)
    @guarded
    def change_settings(request: Request, body: dict = Body(...)):
        action = str(body.get("action") or "edit")
        changes = body.get("changes") or {}
        if not isinstance(changes, dict):
            raise H3Error("Send the changed settings by name.")
        return edit(request, lambda state: pages.apply(state, action, changes, body.get("size")))

    @app.post(f"{API}/writer", include_in_schema=False)
    @guarded
    def change_writer(request: Request, body: dict = Body(...)):
        defaults = writer_defaults()

        def change(state):
            for name, value in (body.get("changes") or {}).items():
                if name not in defaults:
                    raise H3Error(f"Unknown writer setting: {name}")
                state["writer"][name] = coerce(name, value, defaults[name])

        return edit(request, change)

    @app.post(f"{API}/empty", include_in_schema=False)
    @guarded
    def empty_draft(request: Request):
        def change(state):
            state.update(prompt="", media={}, references={})
            pages.apply(state, "mode", {"mode": "Text to video"})

        return edit(request, change)

    @app.post(f"{API}/prompt", include_in_schema=False)
    @guarded
    def change_prompt(request: Request, body: dict = Body(...)):
        return edit(request, lambda state: state.update(prompt=str(body.get("prompt") or "")))

    @app.post(f"{API}/ui", include_in_schema=False)
    @guarded
    def change_ui(request: Request, body: dict = Body(...)):
        """What only the page reads: the open Fine-tune group, the chosen ratio and size."""
        ui = body.get("ui")
        if not isinstance(ui, dict) or len(json.dumps(ui)) > UI_LIMIT:
            raise H3Error("That is too much to remember.")
        return edit(request, lambda state: state.update(ui=ui))

    # Media -----------------------------------------------------------------
    def slot_of(name) -> str:
        if name not in MEDIA_FIELDS:
            raise H3Error("Unknown media slot.")
        return name

    def allowed(slot: str):
        if slot in IMAGE_SLOTS:
            return catalog.IMAGE_EXTENSIONS
        return catalog.VIDEO_EXTENSIONS if slot in VIDEO_SLOTS else catalog.AUDIO_EXTENSIONS

    def forget_references(state, tags):
        mapping = _map_from(state.get("references"))
        state["references"] = _map_to(ReferenceMap(mapping.version + 1,
            tuple(binding for binding in mapping.bindings if binding.tag not in tags)))
        for tag in tags:
            state["prompt"] = state["prompt"].replace(tag, "")
        state["prompt"] = state["prompt"].strip()

    def leave_empty_reference_mode(state):
        if state["values"]["mode"] == "Reference media" and not any(state["media"].get(slot) for slot in REFERENCE_SLOTS):
            pages.apply(state, "mode", {"mode": "Text to video"})

    def place(request: Request, slot: str, path: str | None) -> dict[str, Any]:
        def change(state):
            if path:
                state["media"][slot] = path
            else:
                state["media"].pop(slot, None)
                for kind, prefix, _limit in KINDS:
                    if slot.startswith(f"{prefix}_"):
                        forget_references(state, {f"<{kind} {slot.removeprefix(prefix + '_')}>"})
                        leave_empty_reference_mode(state)
            pages.apply(state, slot, {})

        return edit(request, change)

    @app.post(f"{API}/media/upload", include_in_schema=False)
    @guarded
    def upload_media(request: Request, slot: str, file: UploadFile = File(...)):
        owner = owner_of(request)
        name = slot_of(slot)
        return place(request, name, str(_spool(file, pages.uploads(owner), allowed(name))))

    @app.post(f"{API}/media/library", include_in_schema=False)
    @guarded
    def media_from_library(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        name = slot_of(body.get("slot"))
        library = services.library()
        if library is None or name not in IMAGE_SLOTS:
            raise H3Error("The media library isn't available here.")
        source = str(body.get("source") or "")
        library.validate_path(source, "Image")
        if not Path(source).is_file():
            raise H3Error("This image is unavailable. Refresh the library and choose another.")
        target = pages.uploads(owner) / f"{uuid.uuid4().hex[:12]}-{Path(source).name}"
        shutil.copy2(source, target)
        return place(request, name, str(target))

    @app.post(f"{API}/media/clear", include_in_schema=False)
    @guarded
    def clear_media(request: Request, body: dict = Body(...)):
        return place(request, slot_of(body.get("slot")), None)

    @app.post(f"{API}/references/confirm", include_in_schema=False)
    @guarded
    def confirm_references(request: Request):
        def change(state):
            mapping = pages.references(state)
            missing = [binding.tag for binding in mapping.bindings
                       if not binding.engine_ordinal and binding.tag in state["prompt"]]
            if missing:
                forget_references(state, set(missing))
                leave_empty_reference_mode(state)
            pages.references(state, repair=True)

        return edit(request, change)

    @app.get(f"{API}/library", include_in_schema=False)
    @guarded
    def library_page(request: Request, query: str = "", limit: int = 24):
        owner_of(request)
        library = services.library()
        if library is None:
            raise H3Error("The media library isn't available here.")
        limit = max(1, min(limit, 240))
        page = library.refresh_page("Image", limit, paths=library.index.inventory("Image", query, limit=limit))
        version = library.index.versions.get("Image", 0)
        return {"items": [{"path": str(path), "caption": caption,
                           "thumb": f"/media-previews/{Path(thumb).name}?v={version}"}
                          for path, (thumb, caption) in zip(page.paths, page.items)]}

    @app.get(f"{API}/file", include_in_schema=False)
    def read_file(path: str, request: Request, download: bool = False):
        try:
            owner = owner_of(request)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        file = Path(path)
        if not pages.may_read(owner, path) or not file.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file, filename=file.name if download else None,
                            headers={"Cache-Control": "private, max-age=3600"})

    # Long work -------------------------------------------------------------
    def start(owner: str, kind: str, work: Callable[[H3Run], None]) -> dict[str, Any]:
        run = H3Run(owner, RUN_KEY, kind)
        run.status = "Starting…"
        runs[kind].start(run, work)
        return {"runs": run_payloads(owner)}

    def job_values(state: dict[str, Any]) -> tuple[list[Any], tuple[int, ...]]:
        """The positional request and its media positions, as the Generate button captured them."""
        names = ("batch_count", *GENERATION_FIELDS, "preset")
        captured = pages.request_values(state)
        mode = captured["mode"]
        if mode == "Reference media":
            captured = pages.references(state).dense_values(captured)
        active = REFERENCE_SLOTS if mode == "Reference media" else FRAME_SLOTS if mode == "First / last frame" else ()
        for name in MEDIA_FIELDS:
            if name not in active:
                captured[name] = None
        plan = services.resolve(captured)
        if plan.issues:
            raise H3Error(" ".join(plan.issues))
        ready = generation_readiness(mode, captured["prompt"], captured["first_image"], captured["last_image"],
                                     [captured[name] for name in REFERENCE_SLOTS])
        if not ready.ready:
            raise H3Error(readiness_line(ready))
        return ([captured[name] for name in names],
                tuple(index for index, name in enumerate(names) if name in active and captured[name]))

    @app.post(f"{API}/generate", include_in_schema=False)
    @guarded
    def generate(request: Request):
        require_gpu()
        owner = owner_of(request)
        session = session_of(request)
        with pages.lock:
            state = pages.load(owner)
            values, media_indices = job_values(state)
        count = int(values[0])

        def work(run: H3Run):
            job = JOBS.accept(owner, "h3", values, callback=JOBS.callbacks["h3"], key=uuid.uuid4().hex,
                              media_indices=media_indices, output_indices=(0, 1, 2, 3))
            run.job_id = job.id
            for update in execute_accepted(job.id, session):
                for index in range(min(4, len(update))):
                    sets, value = _update_value(update[index])
                    if sets:
                        path = _path_of(value)
                        if path:
                            run.results[index] = path
                        else:
                            run.results.pop(index, None)
                status = _update_value(update[7])[1] if len(update) > 7 else None
                if isinstance(status, str) and status:
                    run.status = status
                run.revision += 1
            finished = [run.results[index] for index in sorted(run.results) if index < count]
            for path in finished:
                # In the background: the score and tags arrive after the run has finished.
                media_review.schedule(path, state["prompt"], api_key=state["writer"].get("gemini_api_key") or "",
                                      enabled=bool(state["writer"].get("review", True)))
            if finished:
                with pages.lock:
                    latest = pages.load(owner)
                    latest["results"] = finished
                    pages.save(owner, latest)

        return start(owner, GENERATE, work)

    @app.post(f"{API}/enhance", include_in_schema=False)
    @guarded
    def enhance(request: Request):
        owner = owner_of(request)
        with pages.lock:
            state = pages.load(owner)
            original = pages.request_values(state)
            mapping = pages.references(state)
            writer = dict(state["writer"])
        current = mapping.dense_values(original) if original["mode"] == "Reference media" else original
        arguments = [
            current["prompt"], writer["prompt_writer_backend"], writer["local_prompt_base_model"],
            writer["local_prompt_max_tokens"], writer["local_prompt_temperature"], writer["local_prompt_top_p"],
            writer["local_prompt_greedy"], writer["local_prompt_seed"], writer["gemini_prompt_model"],
            writer["gemini_api_key"], writer["openai_api_key"], current["mode"], current["first_image"],
            current["last_image"], *(current[name] for name in REFERENCE_SLOTS), current["duration"],
            current["width"], current["height"], "Video", current["image_frames"],
            *(current[f"fl2va_audio_{i}"] for i in range(1, 4)),
        ]

        def work(run: H3Run):
            run.status = "Writing…"
            suggestion, status = services.enhance(*arguments, openai_model=writer["openai_model"],
                                                  openai_base_url=writer["openai_base_url"] or None)
            if original["mode"] == "Reference media":
                suggestion = mapping.restore_prompt(suggestion)
            run.original = original["prompt"]
            if suggestion == original["prompt"]:
                run.status = status
                return
            run.suggestion = suggestion
            run.status = f"{status} Preview ready; choose Accept or Keep original."

        return start(owner, ENHANCE, work)

    @app.post(f"{API}/input-upscale", include_in_schema=False)
    @guarded
    def input_upscale(request: Request, body: dict = Body(...)):
        require_gpu()
        owner = owner_of(request)
        session = session_of(request)
        with pages.lock:
            media = pages.load(owner)["media"]
        slots = [slot for slot in body.get("slots") or [] if slot in catalog.INPUT_IMAGE_UPSCALE_SLOTS]
        if not slots:
            raise H3Error("Choose at least one image to upscale.")
        values = [slots, str(body.get("model") or request_defaults().get("seedvr2_model")),
                  int(body.get("seed", -1)), bool(body.get("force_offload")), int(body.get("width") or 1920),
                  int(body.get("height") or 1920), *(media.get(name) for name in IMAGE_SLOTS)]
        media_indices = tuple(6 + index for index, name in enumerate(IMAGE_SLOTS) if media.get(name))

        def work(run: H3Run):
            job = JOBS.accept(owner, "h3-input", values, callback=JOBS.callbacks["h3-input"], key=uuid.uuid4().hex,
                              media_indices=media_indices, output_indices=tuple(range(len(UPSCALE_OUTPUTS))))
            run.job_id = job.id
            replaced: dict[str, str] = {}
            for update in execute_accepted(job.id, session):
                for index, name in enumerate(UPSCALE_OUTPUTS):
                    sets, value = _update_value(update[index]) if index < len(update) else (False, None)
                    if sets and (path := _path_of(value)) and path != media.get(name):
                        replaced[name] = path
                if len(update) > len(UPSCALE_OUTPUTS):
                    sets, files = _update_value(update[len(UPSCALE_OUTPUTS)])
                    if sets and files:
                        run.files = [pages.grant(owner, _path_of(item)) for item in files if _path_of(item)]
                status = _update_value(update[-1])[1]
                if isinstance(status, str) and status:
                    run.status = plain(status)
                run.revision += 1
            if replaced:
                with pages.lock:
                    latest = pages.load(owner)
                    for name, path in replaced.items():
                        # A completed job may update only the input it started with.
                        if not media.get(name) or latest["media"].get(name) != media[name]:
                            continue
                        target = pages.uploads(owner) / f"{uuid.uuid4().hex[:12]}-{Path(path).name}"
                        shutil.copy2(path, target)
                        latest["media"][name] = str(target)
                    pages.references(latest)
                    pages.save(owner, latest)

        return start(owner, UPSCALE, work)

    @app.post(f"{API}/compile-trt", include_in_schema=False)
    @guarded
    def compile_trt(request: Request):
        require_gpu()
        owner = owner_of(request)
        if services.compile_trt is None:
            raise H3Error("TensorRT isn't available here.")

        def work(run: H3Run):
            run.status = "Compiling the TensorRT VAE engine…"
            with JOBS.maintenance("trt-compile"):
                run.status = plain(services.compile_trt())

        return start(owner, COMPILE, work)

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, state: bool = False):
        owner = owner_of(request)
        found: dict[str, Any] = {"runs": run_payloads(owner)}
        generation = runs[GENERATE].latest(owner, RUN_KEY)
        if generation is not None:
            found["progress"] = [{"index": index, "path": path, "name": Path(path).name,
                                  "url": f"{API}/file?path={_quote(pages.grant(owner, path))}"}
                                 for index, path in sorted(generation.results.items())]
        upscaled = runs[UPSCALE].latest(owner, RUN_KEY)
        if upscaled is not None:
            found["files"] = [{"name": Path(path).name, "url": f"{API}/file?path={_quote(path)}&download=true"}
                              for path in upscaled.files]
        if state:
            with pages.lock:
                found["state"] = pages.describe(owner, pages.load(owner))
        return found

    @app.post(f"{API}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request, body: dict = Body(default={})):
        require_gpu()
        owner = owner_of(request)
        kind = body.get("kind") or GENERATE
        if kind not in (GENERATE, UPSCALE):
            raise H3Error("Unknown step.")
        run = runs[kind].active(owner, RUN_KEY)
        if services.interrupt is None:
            raise H3Error("Nothing can run here.")
        if run is None:
            message = "Nothing is running."
        elif run.job_id is None:
            message = "The run is still starting. Try again in a moment."
        else:
            message = services.interrupt(session_of(request), family="h3" if kind == GENERATE else "h3-input",
                                         job_id=run.job_id)
        return {"message": message, "runs": run_payloads(owner)}

    return pages
