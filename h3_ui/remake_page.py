"""The Remake page and its JSON API, outside Gradio, as Storyboard's is.

The browser holds the editor, so scrubbing asks the server for nothing. A long selection's windows
stay in their project. A run takes the same job path as every other generation, in a thread here,
and the page polls it, so it outlives the page that started it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from h3_app.catalog import (
    DEFAULT_GEMINI_PROMPT_MODEL, DEFAULT_OPENAI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, IMAGE_EXTENSIONS,
    OPENAI_PROMPT_MODELS, VIDEO_EXTENSIONS,
)
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.remake import DEFAULT_SHORT_EDGE, FPS, MAX_PICTURES, SHORT_EDGES
from h3_app.remake.model import SECTIONS
from h3_app.remake.runner import Selection
from h3_app.remake.windows import DEFAULT_WINDOW_FRAMES, LENGTHS, needs_windows

from .engines.remake import EDIT_MODELS, READY, RemakeHandlers, next_run_details, readiness_message, readiness_step
from .job_admission import execute_accepted
from .job_bindings import owned_generation
from .page_fields import library_items, library_source, spool_upload as _spool
from .page_routes import install_static_page, interrupt_active_run, owner_of, session_of
from .page_runs import Run, Runs, guarded
from .player_fullscreen import FULLSCREEN_SCRIPT

STATIC = Path(__file__).with_name("remake_static")
STATIC_TYPES = {"app.js": "application/javascript", "helpers.js": "application/javascript", "app.css": "text/css"}
# ``generate_remake``'s positions of the source and the four pictures.
MEDIA_INDICES = (1, 7, 8, 9, 10)
RUN_KEY = "remake"
EDIT_KEY = "remake-quick-edit"


def _selection(body: dict[str, Any]) -> Selection:
    chosen = body["selection"]
    return Selection(chosen.get("sha"), float(chosen.get("start") or 0), float(chosen.get("end") or 0),
                     int(chosen.get("short_edge") or DEFAULT_SHORT_EDGE), int(chosen.get("mix") or 0),
                     int(chosen.get("max_window") or DEFAULT_WINDOW_FRAMES))


def _slots(body: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """The pictures and pins of all four slots, in order."""
    slots = [*(body.get("slots") or []), *[None] * MAX_PICTURES][:MAX_PICTURES]
    return ([slot["path"] if slot else None for slot in slots],
            [slot.get("pin") if slot else None for slot in slots])


def install_remake_page(app: FastAPI, handlers: RemakeHandlers, *, interrupt=None, gpu: bool = True,
                        image_pages=None) -> None:
    runs = Runs(busy="A Remake is already running.")
    edits = Runs(busy="A quick edit is already running.")
    callback = owned_generation(handlers.generate_remake, "remake")
    # The Gradio bridge registers an equal callback; without it, retries still need one.
    if gpu:
        JOBS.callbacks.setdefault("remake", callback)
    edit_callback = owned_generation(handlers.quick_edit, "remake_edit")
    # The Jobs tab retries a quick edit through this.
    JOBS.register_callback("remake_edit", edit_callback)
    api = "/remake/api"

    def require_gpu():
        if not gpu:
            raise H3Error("Open this project in Studio to render or edit frames on the GPU.")

    def run_payload(owner):
        run = runs.latest(owner, RUN_KEY)
        return run.public(None) if run is not None else None

    def edit_payload(owner):
        run = edits.latest(owner, EDIT_KEY)
        return run.public(None) if run is not None else None

    def interrupted(request: Request, run: Run | None, family: str) -> str:
        return interrupt_active_run(request, run, family, interrupt)

    def readable(request: Request, path: str) -> tuple[str, Path]:
        """The owner and the file a GET names; anything else is simply not found."""
        try:
            owner = owner_of(request)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        file = Path(path)
        if not handlers.may_read(owner, path) or not file.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return owner, file

    def editor_of(request: Request, body: dict[str, Any]) -> tuple[str, list[Any], list[Any]]:
        """The owner and four slots of a request, once every path in it is known to be theirs."""
        owner = owner_of(request)
        pictures, pins = _slots(body)
        handlers.require_readable(owner, body.get("video"), *pictures)
        return owner, pictures, pins

    def library_picture(owner, path: str) -> str:
        return handlers.adopt_picture(owner, library_source(handlers.library, path))

    # The page --------------------------------------------------------------
    install_static_page(
        app,
        "/remake",
        STATIC,
        STATIC_TYPES,
        extra_files={"fullscreen.js": (FULLSCREEN_SCRIPT, "application/javascript")},
    )

    @app.get(f"{api}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        return {
            "gpu": gpu, "projects": handlers.project_rows(owner), "run": run_payload(owner), "fps": FPS, "sections": list(SECTIONS),
            "short_edges": list(SHORT_EDGES), "short_edge": DEFAULT_SHORT_EDGE,
            "edit_models": [{"id": model.id, "label": model.label} for model in EDIT_MODELS
                            if gpu or model.id == "gpt_image"],
            "lengths": list(LENGTHS), "max_window": DEFAULT_WINDOW_FRAMES,
            "writer": {"gemini_models": list(GEMINI_PROMPT_MODELS), "gemini_model": DEFAULT_GEMINI_PROMPT_MODEL,
                       "openai_models": list(OPENAI_PROMPT_MODELS), "openai_model": DEFAULT_OPENAI_PROMPT_MODEL},
        }

    @app.get(f"{api}/pulse", include_in_schema=False)
    @guarded
    def pulse(request: Request, collect: bool = False):
        """What can change from outside the page: Media, the H3 settings and a run.

        Pictures from Media wait until the page has an editor to ``collect`` them into.
        """
        owner = owner_of(request)
        waiting = handlers.inbox.pop(owner, {}) if collect else {}
        settings = json.dumps(handlers.captured(owner), sort_keys=True, default=str)
        return {"inbox": [{"slot": slot, "path": path} for slot, path in waiting.items()],
                "settings": hashlib.sha256(settings.encode()).hexdigest()[:12], "run": run_payload(owner)}

    # Source and selection --------------------------------------------------
    @app.post(f"{api}/source", include_in_schema=False)
    @guarded
    def add_source(request: Request, file: UploadFile = File(...)):
        owner = owner_of(request)
        with TemporaryDirectory(prefix="h3-remake-") as directory:
            return handlers.add_source(owner, _spool(file, directory, VIDEO_EXTENSIONS), file.filename)

    @app.post(f"{api}/selection", include_in_schema=False)
    @guarded
    def selection(request: Request, body: dict = Body(...)):
        return handlers.selection_state(owner_of(request), _selection(body))

    @app.post(f"{api}/check", include_in_schema=False)
    @guarded
    def check(request: Request, body: dict = Body(...)):
        """Whether the open editor can run, and what the run would be."""
        owner, pictures, pins = editor_of(request, body)
        chosen = _selection(body)
        video = body.get("video")
        long = needs_windows(chosen.start, chosen.end, chosen.max_window)
        message = readiness_message(
            video, chosen.start, chosen.end, body.get("prompt", ""), bool(body.get("pinned")),
            body.get("originals") or {}, pictures, pins, long=long, window=int(body.get("window") or 0),
            smart=bool(body.get("smart")), request=body.get("request", ""),
            window_frames=chosen.max_window if long else None, auto_prompts=bool(body.get("auto_prompts")),
        )
        return {
            "ready": message.startswith(READY), "message": message, "step": readiness_step(video, message),
            "next_run": next_run_details(handlers.captured(owner), body.get("mode"), video, chosen.start, chosen.end,
                                         chosen.short_edge, sum(1 for path in pictures if path),
                                         bool(body.get("pinned")), chosen.max_window),
        }

    # Projects --------------------------------------------------------------
    @app.get(f"{api}/projects", include_in_schema=False)
    @guarded
    def projects(request: Request):
        return {"projects": handlers.project_rows(owner_of(request))}

    @app.post(f"{api}/projects/{{key}}/load", include_in_schema=False)
    @guarded
    def load_project(key: str, request: Request):
        return handlers.load_project(owner_of(request), key)

    @app.get(f"{api}/projects/{{key}}", include_in_schema=False)
    @guarded
    def read_project(key: str, request: Request):
        return handlers.project_data(owner_of(request), key)

    @app.patch(f"{api}/projects/{{key}}", include_in_schema=False)
    @guarded
    def rename_project(key: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        handlers.rename_project(owner, key, body.get("name"))
        return {"projects": handlers.project_rows(owner)}

    @app.delete(f"{api}/projects/{{key}}", include_in_schema=False)
    @guarded
    def delete_project(key: str, request: Request):
        owner = owner_of(request)
        handlers.delete_project(owner, key)
        return {"projects": handlers.project_rows(owner)}

    @app.post(f"{api}/save", include_in_schema=False)
    @guarded
    def save_project(request: Request, body: dict = Body(...)):
        owner, pictures, pins = editor_of(request, body)
        return handlers.save_project(owner, _selection(body), body.get("video"),
                                     int(body.get("window") or 0), body.get("prompt", ""),
                                     bool(body.get("use_carried")), pictures, pins, body)

    # Windows ---------------------------------------------------------------
    @app.post(f"{api}/split", include_in_schema=False)
    @guarded
    def split(request: Request, body: dict = Body(...)):
        owner, _pictures, _pins = editor_of(request, body)
        return handlers.split(owner, _selection(body), body.get("video"))

    def window_action(action, request, body):
        owner, pictures, pins = editor_of(request, body)
        return action(owner, _selection(body), int(body.get("window") or 0), body.get("prompt", ""),
                      bool(body.get("use_carried")), pictures, pins)

    @app.post(f"{api}/window/save", include_in_schema=False)
    @guarded
    def save_window(request: Request, body: dict = Body(...)):
        return window_action(handlers.save_window, request, body)

    @app.post(f"{api}/window/mark", include_in_schema=False)
    @guarded
    def mark_window(request: Request, body: dict = Body(...)):
        return window_action(handlers.mark_window, request, body)

    @app.post(f"{api}/window/carried", include_in_schema=False)
    @guarded
    def copy_carried(request: Request, body: dict = Body(...)):
        return handlers.copy_carried(owner_of(request), _selection(body), int(body.get("window") or 0))

    @app.post(f"{api}/combine", include_in_schema=False)
    @guarded
    def combine(request: Request, body: dict = Body(...)):
        return handlers.combine(owner_of(request), _selection(body))

    # Frames and pictures ---------------------------------------------------
    @app.post(f"{api}/extract", include_in_schema=False)
    @guarded
    def extract(request: Request, body: dict = Body(...)):
        owner, _pictures, _pins = editor_of(request, body)
        return handlers.extract(owner, body.get("video"), body.get("base"), body.get("frame"))

    @app.post(f"{api}/picture", include_in_schema=False)
    @guarded
    def upload_picture(request: Request, file: UploadFile = File(...)):
        owner = owner_of(request)
        with TemporaryDirectory(prefix="h3-remake-") as directory:
            return {"path": handlers.adopt_picture(owner, _spool(file, directory, IMAGE_EXTENSIONS))}

    @app.post(f"{api}/picture/library", include_in_schema=False)
    @guarded
    def picture_from_library(request: Request, body: dict = Body(...)):
        return {"path": library_picture(owner_of(request), body.get("source", ""))}

    @app.get(f"{api}/library", include_in_schema=False)
    @guarded
    def library(request: Request, query: str = "", limit: int = 24):
        owner_of(request)
        return library_items(handlers.library, query, limit)

    @app.post(f"{api}/align/check", include_in_schema=False)
    @guarded
    def align_check(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        handlers.require_readable(owner, body.get("video"), body.get("picture"))
        return handlers.check_alignment(owner, body.get("video"), body.get("base"), body.get("picture"),
                                        body.get("pin"), body.get("view", "Overlay"))

    @app.post(f"{api}/align/apply", include_in_schema=False)
    @guarded
    def align_apply(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        handlers.require_readable(owner, body.get("video"), body.get("picture"))
        return {"path": handlers.apply_alignment(owner, body.get("video"), body.get("base"), body.get("picture"),
                                                 body.get("pin"))}

    @app.post(f"{api}/send-to-image", include_in_schema=False)
    @guarded
    def send_to_image(request: Request, body: dict = Body(...)):
        if not gpu:
            if image_pages is None or body.get("model") != "gpt_image":
                raise H3Error("Choose GPT Image to edit frames in Workshop.")
            owner = owner_of(request)
            path = str(body.get("path") or "")
            handlers.require_readable(owner, path)
            image_pages.receive(owner, "gpt_image", path)
            return {"ok": True, "url": "/image/"}
        handlers.send_to_image(owner_of(request), str(body.get("path") or ""), body.get("model"))
        return {"ok": True}

    # Prompt ----------------------------------------------------------------
    @app.post(f"{api}/rewrite", include_in_schema=False)
    @guarded
    def rewrite(request: Request, body: dict = Body(...)):
        owner, pictures, pins = editor_of(request, body)
        suggestion, status = handlers.rewrite_prompt(
            owner, _selection(body), int(body.get("window") or 0), body.get("request", ""),
            body.get("mode"), body.get("video"), bool(body.get("pinned")), pictures, pins, body.get("writer") or {},
            smart=bool(body.get("smart")), use_carried=body.get("use_carried", True) is not False,
        )
        return {"suggestion": suggestion, "status": status}

    # Files -----------------------------------------------------------------
    @app.get(f"{api}/file", include_in_schema=False)
    def read_file(path: str, request: Request):
        return FileResponse(readable(request, path)[1], headers={"Cache-Control": "private, max-age=3600"})

    @app.get(f"{api}/frame", include_in_schema=False)
    def read_frame(path: str, base: float, frame: int, request: Request):
        """One frame, for a source the browser can't play."""
        _owner, file = readable(request, path)
        try:
            return FileResponse(handlers.frame_preview(file, base, frame),
                                headers={"Cache-Control": "private, max-age=3600"})
        except H3Error as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(f"{api}/strip", include_in_schema=False)
    def read_strip(path: str, start: float, seconds: float, request: Request):
        owner, file = readable(request, path)
        try:
            return FileResponse(handlers.strip(owner, file, start, seconds),
                                headers={"Cache-Control": "private, max-age=86400"})
        except H3Error as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Long work -------------------------------------------------------------
    @app.get(f"{api}/run", include_in_schema=False)
    @guarded
    def read_run(request: Request):
        return {"run": run_payload(owner_of(request))}

    @app.post(f"{api}/generate", include_in_schema=False)
    @guarded
    def generate(request: Request, body: dict = Body(...)):
        require_gpu()
        owner, pictures, pins = editor_of(request, body)
        chosen = _selection(body)
        writer = body.get("writer") or {}
        handlers.set_writer_keys(owner, writer.get("gemini_key"), writer.get("openai_key"), writer.get("openai_url"))
        values = [
            body.get("mode"), body.get("video"), chosen.start, chosen.end, chosen.short_edge, body.get("prompt", ""),
            bool(body.get("pinned")), *pictures, *pins, chosen.max_window, body.get("request", ""),
            bool(body.get("smart")), chosen.mix, int(body.get("window") or 0), bool(body.get("use_carried")),
            writer.get("model") or DEFAULT_GEMINI_PROMPT_MODEL, writer.get("backend") or "Gemini",
            writer.get("openai_model") or DEFAULT_OPENAI_PROMPT_MODEL, chosen.sha or "",
            bool(body.get("auto_prompts")),
        ]
        session = session_of(request)

        def work(run: Run):
            job = JOBS.accept(owner, "remake", values, callback=JOBS.callbacks["remake"], key=uuid.uuid4().hex,
                              media_indices=tuple(index for index in MEDIA_INDICES if values[index]),
                              output_indices=(0,))
            run.job_id = job.id
            for output, status in execute_accepted(job.id, session):
                if status:
                    run.status = status
                if output:
                    run.result_url = handlers.grant(owner, output)
                run.revision += 1

        runs.start(Run(owner, RUN_KEY, "generate", project=body.get("project")), work)
        return {"run": run_payload(owner)}

    @app.post(f"{api}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request):
        require_gpu()
        owner = owner_of(request)
        message = interrupted(request, runs.active(owner, RUN_KEY, "generate"), "remake")
        return {"message": message, "run": run_payload(owner)}

    # Quick edit ------------------------------------------------------------
    @app.post(f"{api}/quick-edit", include_in_schema=False)
    @guarded
    def quick_edit(request: Request, body: dict = Body(...)):
        engine = body.get("engine") or ("qwen" if gpu else "gpt_image")
        if not gpu and engine != "gpt_image":
            raise H3Error("Choose GPT Image to edit frames in Workshop.")
        owner = owner_of(request)
        picture, prompt = body.get("picture"), str(body.get("prompt") or "").strip()
        handlers.require_readable(owner, picture)
        if not picture or not prompt:
            raise H3Error("Write what to change in the frame.")
        session = session_of(request)

        def work(run: Run):
            job = JOBS.accept(owner, "remake_edit", [picture, prompt, engine], callback=edit_callback, key=uuid.uuid4().hex,
                              media_indices=(0,), output_indices=(0,))
            run.job_id = job.id
            for output, status in execute_accepted(job.id, session):
                run.status = status
                if output:
                    run.result_url = handlers.adopt_picture(owner, output)
                run.revision += 1

        edits.start(Run(owner, EDIT_KEY, "edit"), work)
        return {"run": edit_payload(owner)}

    @app.get(f"{api}/quick-edit/run", include_in_schema=False)
    @guarded
    def read_quick_edit(request: Request):
        return {"run": edit_payload(owner_of(request))}

    @app.post(f"{api}/quick-edit/interrupt", include_in_schema=False)
    @guarded
    def stop_quick_edit(request: Request):
        require_gpu()
        owner = owner_of(request)
        message = interrupted(request, edits.active(owner, EDIT_KEY, "edit"), "remake_edit")
        return {"message": message, "run": edit_payload(owner)}
