"""The Storyboard page and its JSON API, outside Gradio.

The browser holds the whole storyboard, so moving between scenes asks the server for
nothing. Edits save one field in the background. A GPU run takes the same job path as
every other generation, in a thread here, and the page polls it, so progress needs no
long-lived connection and a run outlives the page that started it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Lock, Thread
import time
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from h3_app import storyboard as sb
from h3_app.catalog import (
    AUDIO_EXTENSIONS, DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, IMAGE_EXTENSIONS,
    OPENAI_PROMPT_MODELS, VIDEO_EXTENSIONS,
)
from h3_app.errors import H3Error
from h3_app.jobs import JOBS, JobCancelled
from h3_app.storyboard.model import MAX_TAKES_PER_RUN, OVERLAP_FRAMES
from h3_app.workspace_store import request_owner

from ..job_admission import execute_accepted
from .handlers import GENERATE_REMAINING, GENERATE_SCENE, SWITCH_AND_REGENERATE, StoryboardHandlers
from .payload import board_payload, file_url, referenced_paths

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
SCENE_FIELDS = ("title", "prompt", "seconds", "link", "overlap", "seed", "takes_per_run", "cast")
INTEGER_FIELDS = ("seconds", "overlap", "seed", "takes_per_run")
UPLOAD_TYPES = {"first_image": IMAGE_EXTENSIONS, "last_image": IMAGE_EXTENSIONS, "ref_images": IMAGE_EXTENSIONS,
                "ref_videos": VIDEO_EXTENSIONS, "ref_audios": AUDIO_EXTENSIONS}
SINGLE_FIELDS = ("first_image", "last_image")
THUMB_WIDTH = 480
# A finished run stays readable this long, for a page that polls or reopens afterwards.
RUN_RETENTION_SECONDS = 600


@dataclass
class Run:
    """Long work on a storyboard: GPU generation, or joining the takes."""

    owner: str
    storyboard_id: str
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str | None = None  # The generation job, once accepted.
    project: str | None = None  # The Remake project it renders, once it has one.
    status: str = "Waiting for the GPU…"
    # Rises each time the storyboard changed, so the page knows to read it again.
    revision: int = 0
    finished_at: float | None = None
    error: str | None = None
    result_url: str | None = None
    message: str | None = None

    def public(self, scene_id) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status, "revision": self.revision,
                "active": self.finished_at is None, "error": self.error, "scene_id": scene_id,
                "project": self.project, "result_url": self.result_url, "message": self.message}


class Runs:
    def __init__(self, busy="This storyboard is already doing that."):
        self.lock = Lock()
        self.busy = busy
        self.items: dict[str, Run] = {}

    def start(self, run: Run, work: Callable[[Run], None]) -> Run:
        with self.lock:
            cutoff = time.time() - RUN_RETENTION_SECONDS
            for key in [key for key, item in self.items.items()
                        if item.finished_at is not None and item.finished_at < cutoff]:
                del self.items[key]
            if self.active(run.owner, run.storyboard_id, run.kind):
                raise H3Error(self.busy)
            self.items[run.id] = run

        def execute():
            try:
                work(run)
            except JobCancelled:
                run.status = "Interrupted."
            except Exception as exc:  # The page shows every failure; nothing else reads it.
                run.error = str(exc)
                run.status = "Stopped."
            finally:
                run.revision += 1
                run.finished_at = time.time()

        Thread(target=execute, name=f"storyboard-{run.kind}", daemon=True).start()
        return run

    def active(self, owner, storyboard_id, kind=None) -> Run | None:
        return next((run for run in list(self.items.values())
                     if run.owner == owner and run.storyboard_id == storyboard_id and run.finished_at is None
                     and (kind is None or run.kind == kind)), None)

    def latest(self, owner, storyboard_id) -> Run | None:
        with self.lock:
            active = self.active(owner, storyboard_id)
            if active is not None:
                return active
            return next((run for run in reversed(self.items.values())
                         if run.owner == owner and run.storyboard_id == storyboard_id), None)


def guarded(endpoint):
    """Every refusal reaches the page as ``{"error": message}``."""

    @functools.wraps(endpoint)
    def run(*args, **kwargs):
        try:
            return endpoint(*args, **kwargs)
        except (H3Error, ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return run


def _image_thumbnail(path: Path, root: Path) -> Path | None:
    """A small cached JPEG of a picture, or None when it can't be read as one."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    stat = path.stat()
    key = hashlib.sha256(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:24]
    thumbnail = root / f"storyboard_{key}.jpg"
    if thumbnail.is_file():
        return thumbnail
    root.mkdir(parents=True, exist_ok=True)
    partial = thumbnail.with_suffix(f".{uuid.uuid4().hex}.tmp.jpg")
    try:
        with Image.open(path) as image:
            upright = ImageOps.exif_transpose(image)
            upright.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
            upright.convert("RGB").save(partial, "JPEG", quality=82)
        partial.replace(thumbnail)
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError):
        partial.unlink(missing_ok=True)
        return None
    return thumbnail


def install_storyboard_page(app: FastAPI, handlers: StoryboardHandlers, *, gpu: bool, interrupt=None) -> None:
    runs = Runs()
    # The Jobs tab retries a storyboard job through this.
    JOBS.register_callback("storyboard", handlers.run_job)
    api = "/storyboard/api"

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        """What outlives ``request`` and names its owner to the job path."""
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None)

    def payload(owner, storyboard_id):
        board = handlers.service.load(owner, storyboard_id)
        data = board_payload(storyboard_id, board)
        data["run"] = run_payload(owner, storyboard_id)
        return data

    def run_payload(owner, storyboard_id):
        run = runs.latest(owner, storyboard_id)
        if run is None:
            return None
        return run.public(handlers.running_scene.get((owner, storyboard_id)) if run.kind == "generate" else None)

    @app.get("/storyboard", include_in_schema=False)
    def storyboard_redirect():
        return RedirectResponse("/storyboard/")

    @app.get("/storyboard/", include_in_schema=False)
    def storyboard_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/storyboard/{name}", include_in_schema=False)
    def storyboard_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{api}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        return {
            "gpu": gpu,
            "boards": [{"id": row["id"], "name": row["name"], "generated": bool(row["generated"])}
                       for row in handlers.service.list(owner)],
            "seconds": [sb.SCENE_SECONDS[0], sb.SCENE_SECONDS[-1]],
            "overlaps": list(OVERLAP_FRAMES), "max_takes": MAX_TAKES_PER_RUN,
            "library": handlers.library is not None,
            "enhance": handlers.services.enhance_prompt is not None,
            "writer": {"gemini_models": list(GEMINI_PROMPT_MODELS), "gemini_default": DEFAULT_GEMINI_PROMPT_MODEL,
                       "openai_models": list(OPENAI_PROMPT_MODELS)},
        }

    # Storyboards -----------------------------------------------------------
    @app.post(f"{api}/rewrite", include_in_schema=False)
    def rewrite(request: Request, body: dict = Body(...)):
        owner_of(request)
        writer = body.get("writer") or {}
        try:
            story = handlers.rewrite_story(
                body.get("story", ""), body.get("connect", sb.FLOW), int(body.get("seconds", 10)),
                int(body.get("count", 5)),
                {"backend": writer.get("prompt_writer_backend", "Gemini"),
                 "gemini_prompt_model": writer.get("gemini_prompt_model", DEFAULT_GEMINI_PROMPT_MODEL),
                 "gemini_api_key": writer.get("gemini_api_key", ""),
                 "openai_api_key": writer.get("openai_api_key", ""),
                 "openai_model": writer.get("openai_model") or OPENAI_PROMPT_MODELS[0],
                 "openai_base_url": writer.get("openai_base_url", "")},
            )
        except Exception as exc:  # The writer calls out to hosted services and can fail any way.
            return JSONResponse({"error": f"Story unchanged. {exc}"}, status_code=400)
        return {"story": story}

    @app.post(f"{api}/boards", include_in_schema=False)
    @guarded
    def create(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        storyboard_id = handlers.create(owner, str(body.get("name") or ""), body.get("story", ""),
                                        body.get("connect", sb.FLOW), int(body.get("seconds", 10)))
        return payload(owner, storyboard_id)

    @app.get(f"{api}/boards/{{storyboard_id}}", include_in_schema=False)
    @guarded
    def read(storyboard_id: str, request: Request):
        return payload(owner_of(request), storyboard_id)

    @app.patch(f"{api}/boards/{{storyboard_id}}", include_in_schema=False)
    @guarded
    def rename(storyboard_id: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        handlers.service.rename(owner, storyboard_id, body.get("name", ""))
        return payload(owner, storyboard_id)

    @app.delete(f"{api}/boards/{{storyboard_id}}", include_in_schema=False)
    @guarded
    def delete(storyboard_id: str, request: Request):
        handlers.service.delete(owner_of(request), storyboard_id)
        return {"deleted": True}

    @app.post(f"{api}/boards/{{storyboard_id}}/duplicate", include_in_schema=False)
    @guarded
    def duplicate(storyboard_id: str, request: Request):
        owner = owner_of(request)
        return payload(owner, handlers.service.branch(owner, storyboard_id))

    @app.post(f"{api}/boards/{{storyboard_id}}/profile", include_in_schema=False)
    @guarded
    def apply_profile(storyboard_id: str, request: Request):
        owner = owner_of(request)
        handlers.apply_profile(owner, storyboard_id)
        return payload(owner, storyboard_id)

    # Scenes ----------------------------------------------------------------
    @app.patch(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}", include_in_schema=False)
    @guarded
    def edit_scene(storyboard_id: str, scene_id: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        fields = {name: body[name] for name in SCENE_FIELDS if name in body}
        for name in INTEGER_FIELDS:
            if name in fields:
                fields[name] = int(fields[name])
        if "title" in fields:
            fields["title"] = str(fields["title"]).strip()[:120]
        if fields:
            handlers.service.update_scene(owner, storyboard_id, scene_id, **fields)
        if "mode" in body:
            handlers.service.set_mode(owner, storyboard_id, scene_id, body["mode"])
        return payload(owner, storyboard_id)

    @app.post(f"{api}/boards/{{storyboard_id}}/scenes", include_in_schema=False)
    @guarded
    def add_scene(storyboard_id: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        _, scene_id = handlers.service.add_scene(owner, storyboard_id, after=body.get("after"))
        return {**payload(owner, storyboard_id), "scene_id": scene_id}

    @app.post(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/duplicate", include_in_schema=False)
    @guarded
    def duplicate_scene(storyboard_id: str, scene_id: str, request: Request):
        owner = owner_of(request)
        _, copy_id = handlers.service.duplicate_scene(owner, storyboard_id, scene_id)
        return {**payload(owner, storyboard_id), "scene_id": copy_id}

    @app.post(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/move", include_in_schema=False)
    @guarded
    def move_scene(storyboard_id: str, scene_id: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        handlers.service.move_scene(owner, storyboard_id, scene_id, int(body.get("offset", 0)))
        return payload(owner, storyboard_id)

    @app.delete(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}", include_in_schema=False)
    @guarded
    def delete_scene(storyboard_id: str, scene_id: str, request: Request):
        owner = owner_of(request)
        handlers.service.delete_scene(owner, storyboard_id, scene_id)
        return payload(owner, storyboard_id)

    @app.post(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/enhance", include_in_schema=False)
    def enhance(storyboard_id: str, scene_id: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        try:
            status = handlers.enhance(owner, storyboard_id, scene_id, body.get("writer"))
        except Exception as exc:  # The writer calls out to hosted services and can fail any way.
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {**payload(owner, storyboard_id), "message": status}

    # Media -----------------------------------------------------------------
    def stored_media(board, scene_id, field_name):
        scene = board["scenes"][handlers.service.scene_index(board, scene_id)]
        current = sb.scene_media(scene)[field_name]
        return [current] if isinstance(current, str) else list(current or [])

    def set_media(owner, storyboard_id, scene_id, field_name, added):
        with handlers.service.lock:
            board = handlers.service.load(owner, storyboard_id)
            current = stored_media(board, scene_id, field_name)
            value = added[-1] if field_name in SINGLE_FIELDS else [*current, *added]
            handlers.service.set_scene_media(owner, storyboard_id, scene_id, field_name, value)

    def spool(files: list[UploadFile], directory: str, allowed) -> list[str]:
        """Saved uploads under unique names: a stored file with the same name is another file."""
        paths = []
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in allowed:
                raise H3Error(f"{upload.filename} isn't a supported file ({', '.join(sorted(allowed))}).")
            path = Path(directory) / f"{uuid.uuid4().hex}{suffix}"
            with path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            paths.append(str(path))
        return paths

    def library_source(path: str) -> str:
        if handlers.library is None:
            raise H3Error("The media library isn't available here.")
        handlers.library.validate_path(path, "Image")
        if not Path(path).is_file():
            raise H3Error("This image is unavailable. Refresh the library and choose another.")
        return path

    @app.post(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/media/{{field_name}}", include_in_schema=False)
    @guarded
    def upload_media(storyboard_id: str, scene_id: str, field_name: str, request: Request,
                     files: list[UploadFile] = File(...)):
        owner = owner_of(request)
        if field_name not in UPLOAD_TYPES:
            raise H3Error(f"Unknown scene media {field_name!r}.")
        with TemporaryDirectory(prefix="h3-storyboard-") as directory:
            set_media(owner, storyboard_id, scene_id, field_name, spool(files, directory, UPLOAD_TYPES[field_name]))
        return payload(owner, storyboard_id)

    @app.put(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/media/{{field_name}}", include_in_schema=False)
    @guarded
    def library_media(storyboard_id: str, scene_id: str, field_name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        if UPLOAD_TYPES.get(field_name) is not IMAGE_EXTENSIONS:
            raise H3Error("The library holds pictures for this scene's picture inputs.")
        set_media(owner, storyboard_id, scene_id, field_name, [library_source(body.get("source", ""))])
        return payload(owner, storyboard_id)

    @app.delete(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/media/{{field_name}}", include_in_schema=False)
    @guarded
    def remove_media(storyboard_id: str, scene_id: str, field_name: str, path: str, request: Request):
        owner = owner_of(request)
        with handlers.service.lock:
            board = handlers.service.load(owner, storyboard_id)
            kept = [item for item in stored_media(board, scene_id, field_name) if item != path]
            value = (kept[0] if kept else None) if field_name in SINGLE_FIELDS else kept
            handlers.service.set_scene_media(owner, storyboard_id, scene_id, field_name, value)
        return payload(owner, storyboard_id)

    @app.get(f"{api}/library", include_in_schema=False)
    @guarded
    def library(request: Request, query: str = "", limit: int = 24):
        owner_of(request)
        if handlers.library is None:
            raise H3Error("The media library isn't available here.")
        limit = max(1, min(limit, 240))
        index = handlers.library.index
        page = handlers.library.refresh_page("Image", limit, paths=index.inventory("Image", query, limit=limit))
        version = index.versions.get("Image", 0)
        return {
            "total": page.total, "more": page.next_cursor is not None,
            "items": [{"path": str(path), "caption": caption, "thumb": f"/media-previews/{Path(thumb).name}?v={version}"}
                      for path, (thumb, caption) in zip(page.paths, page.items)],
        }

    # Cast ------------------------------------------------------------------
    @app.post(f"{api}/boards/{{storyboard_id}}/cast", include_in_schema=False)
    @guarded
    def add_cast(storyboard_id: str, request: Request, name: str = Form(...),
                 file: UploadFile | None = File(default=None), source: str = Form(default="")):
        owner = owner_of(request)
        if source:
            handlers.service.add_cast(owner, storyboard_id, name, library_source(source))
        elif file is not None:
            with TemporaryDirectory(prefix="h3-storyboard-") as directory:
                handlers.service.add_cast(owner, storyboard_id, name, spool([file], directory, IMAGE_EXTENSIONS)[0])
        else:
            raise H3Error("Choose a picture first.")
        return payload(owner, storyboard_id)

    @app.delete(f"{api}/boards/{{storyboard_id}}/cast/{{name:path}}", include_in_schema=False)
    @guarded
    def remove_cast(storyboard_id: str, name: str, request: Request):
        owner = owner_of(request)
        handlers.service.remove_cast(owner, storyboard_id, name)
        return payload(owner, storyboard_id)

    # Takes and versions ----------------------------------------------------
    @app.post(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/use-take", include_in_schema=False)
    @guarded
    def use_take(storyboard_id: str, scene_id: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        question = handlers.use_take(owner, storyboard_id, scene_id, body.get("take_id"),
                                     confirmed=bool(body.get("confirmed")))
        return {**payload(owner, storyboard_id), "question": question}

    @app.delete(f"{api}/boards/{{storyboard_id}}/scenes/{{scene_id}}/takes/{{take_id}}", include_in_schema=False)
    @guarded
    def delete_take(storyboard_id: str, scene_id: str, take_id: str, request: Request):
        owner = owner_of(request)
        # A running scene may be reading this take's latent, whichever take is in use by now.
        if runs.active(owner, storyboard_id, "generate") is not None:
            raise H3Error("Wait for generation to finish before deleting a take.")
        handlers.service.delete_take(owner, storyboard_id, scene_id, take_id)
        return payload(owner, storyboard_id)

    @app.post(f"{api}/boards/{{storyboard_id}}/versions", include_in_schema=False)
    @guarded
    def save_version(storyboard_id: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        handlers.service.save_version(owner, storyboard_id, body.get("label", ""))
        return payload(owner, storyboard_id)

    @app.post(f"{api}/boards/{{storyboard_id}}/versions/{{version_id}}/branch", include_in_schema=False)
    @guarded
    def branch(storyboard_id: str, version_id: str, request: Request):
        owner = owner_of(request)
        return payload(owner, handlers.service.branch(owner, storyboard_id, version_id))

    # Files -----------------------------------------------------------------
    def owned_file(request: Request, storyboard_id, path) -> Path:
        try:
            board = handlers.service.load(owner_of(request), storyboard_id)
            allowed = referenced_paths(board) | {str(handlers.service.preview_path(storyboard_id))}
        except (H3Error, ValueError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        file = Path(path)
        if path not in allowed or not file.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return file

    @app.get(f"{api}/boards/{{storyboard_id}}/file", include_in_schema=False)
    def storyboard_file(storyboard_id: str, path: str, request: Request):
        return FileResponse(owned_file(request, storyboard_id, path),
                            headers={"Cache-Control": "private, max-age=3600"})

    @app.get(f"{api}/boards/{{storyboard_id}}/thumb", include_in_schema=False)
    def storyboard_thumb(storyboard_id: str, path: str, request: Request):
        file = owned_file(request, storyboard_id, path)
        thumbnail = _image_thumbnail(file, handlers.service.runtime().gallery_thumbnails_dir)
        return FileResponse(thumbnail or file, headers={"Cache-Control": "private, max-age=86400"})

    # Long work -------------------------------------------------------------
    @app.get(f"{api}/boards/{{storyboard_id}}/run", include_in_schema=False)
    @guarded
    def read_run(storyboard_id: str, request: Request):
        return {"run": run_payload(owner_of(request), storyboard_id)}

    @app.post(f"{api}/boards/{{storyboard_id}}/generate", include_in_schema=False)
    @guarded
    def generate(storyboard_id: str, request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Scenes are generated on the GPU endpoint.")
        owner = owner_of(request)
        action = body.get("action")
        if action not in (GENERATE_SCENE, GENERATE_REMAINING, SWITCH_AND_REGENERATE):
            raise H3Error("Unknown generation request.")
        board = handlers.service.load(owner, storyboard_id)
        if action == GENERATE_SCENE:
            sb.check_scene(board, handlers.service.scene_index(board, body.get("scene_id")))
        session = session_of(request)

        def work(run):
            # Accepted here, once this is the storyboard's only run: a refused run leaves no job.
            job = JOBS.accept(owner, "storyboard",
                              [action, storyboard_id, body.get("scene_id"), body.get("take_id")],
                              callback=handlers.run_job, output_indices=(1,), key=uuid.uuid4().hex)
            run.job_id = job.id
            try:
                for status, changed in execute_accepted(job.id, session):
                    run.status = status
                    if changed:
                        run.revision += 1
            finally:
                handlers.running_scene.pop((owner, storyboard_id), None)

        runs.start(Run(owner, storyboard_id, "generate"), work)
        return {"run": run_payload(owner, storyboard_id)}

    @app.post(f"{api}/boards/{{storyboard_id}}/interrupt", include_in_schema=False)
    @guarded
    def stop(storyboard_id: str, request: Request):
        if interrupt is None:
            raise H3Error("Nothing can run here.")
        owner = owner_of(request)
        run = runs.active(owner, storyboard_id, "generate")
        if run is None:
            message = "This storyboard isn't generating."
        elif run.job_id is None:
            message = "The run is still starting. Try again in a moment."
        else:
            # One owner can run several storyboards; only this one stops.
            message = interrupt(session_of(request), family="storyboard", job_id=run.job_id)
        return {"message": message, "run": run_payload(owner, storyboard_id)}

    @app.post(f"{api}/boards/{{storyboard_id}}/join", include_in_schema=False)
    @guarded
    def join(storyboard_id: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        include = bool(body.get("include_outdated", True))
        export = bool(body.get("export"))

        def work(run):
            run.status = "Exporting to Media…" if export else "Joining the takes…"
            if export:
                path = handlers.export(owner, storyboard_id, include)
                run.message = f"Exported {path.name} to Media and saved a version."
            else:
                path = handlers.play_all(owner, storyboard_id, include)
                run.result_url = file_url(storyboard_id, str(path), v=path.stat().st_mtime_ns)
                run.message = "Full video ready. Continuing scenes drop their repeated opening; fresh starts cut."
            run.status = "Done."

        runs.start(Run(owner, storyboard_id, "export" if export else "join"), work)
        return {"run": run_payload(owner, storyboard_id)}
