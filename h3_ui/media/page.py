"""The Media page and its JSON API, outside Gradio, as the Song and Image pages are.

Everything the Gradio Media library did, against the same services: browse by type with search,
favorites, engine and sort; preview the selected item with its Soundstage timeline, dub player or
illustrated video; tags, favorites and lineage; the settings snapshot; compare two images or two
videos; Set as an image input; enhance a copy on the GPU; import a file or download a link with
saved cookies; scan historical media; and delete with a confirmation that names what goes. The
Gradio view stays built but unseen for the published API and the modules that still read it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response

import h3_models as models
from h3_app import catalog
from h3_app.catalog import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.provenance import read_snapshot
from h3_app.workspace_store import request_owner

from ..h3.page import plain
from ..job_admission import execute_accepted
from ..library_tools import lineage_steps
from ..media_actions import DeletionIntent, file_fingerprints
from ..page_runs import Run, Runs, guarded

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
ROUTE = "/media"
API = f"{ROUTE}/api"
KINDS = ("Video", "Image", "Audio")
PAGE_SIZE = 48
IMPORT_LIMIT = 100  # Files in one Add media batch.
SORTS = (("Newest first", "newest"), ("Oldest first", "oldest"), ("Name", "name"))
EXTENSIONS = {"Video": VIDEO_EXTENSIONS, "Image": IMAGE_EXTENSIONS, "Audio": AUDIO_EXTENSIONS}
NOUNS = {"Video": "video", "Image": "image", "Audio": "audio"}
SCAN, DOWNLOAD, ENHANCE = "scan", "download", "enhance"
BUSY = {SCAN: "A historical scan is already running.", DOWNLOAD: "A download is already running.",
        ENHANCE: "An enhancement is already running."}
# The enhance methods each type offers, as the Gradio view switched them.
IMAGE_METHODS = (catalog.SEEDVR2_UPSCALE, *catalog.RELAYVSR_OPTIONS, catalog.LTX25_CQ_IMAGE_ENHANCER)
LTX_OPTIONS = {catalog.LTX25_UPSCALE, catalog.LTX25_TILED_UPSCALE} | set(catalog.LTX25_SAME_RESOLUTION_OPTIONS)
METHOD_HELP = ("H3 refinement uses the current Minimax H3 settings, including text encoder, prompt and seed. "
               "Target resolution is set here.")
SEEDVR2_HELP = ("Downloaded on first use. 7B INT8 is the default quality/VRAM balance; FP16 favors fidelity, 7B Sharp "
                "favors stronger detail, and MXFP8/NVFP4 are experimental speed options.")


def method_fields(value: str) -> dict[str, Any]:
    """Which enhance settings a method uses, and the scene prompt's hint, as the Gradio view showed them."""
    ai = value in catalog.AI_POSTPROCESS_OPTIONS or value == catalog.LTX25_CQ_IMAGE_ENHANCER
    grouped = ai and value != catalog.H3_REFINEMENT
    prompt = (value in LTX_OPTIONS | set(catalog.H3_REGENERATE_MODELS)
              and value not in {catalog.LTX25_CQ_ENHANCER, catalog.LTX25_SDR_TO_HDR})
    hint = (
        "Uses the current Minimax H3 native upscale settings: method, upscaler precision, refinement steps and "
        "LoRA. Preserves source timing and audio." if value == catalog.H3_REFINEMENT else
        "Experimental H3 refiner + LightX2V, three Euler steps. Describe the source scene. Preserves frames, fps "
        "and audio; best tested on short 24 fps drafts. Uses the default H3 model profile."
        if value in catalog.H3_REGENERATE_MODELS else
        "Describe texture, lighting and sharpness only; named subjects can repeat across tiles. Uses spatial "
        "tiles and temporal streaming." if value == catalog.LTX25_TILED_UPSCALE else
        "Describe the source scene; focus restoration instructions are added automatically. Preserves source "
        "resolution. Uses the finishing model selected here." if value == catalog.LTX25_DEBLUR else
        "Describe the source scene; compression artifact removal instructions are added automatically. "
        "Preserves source resolution. Uses the finishing model selected here."
        if value in catalog.LTX25_RESTORATION_OPTIONS else
        "Optional but recommended. Uses the finishing model selected here."
    )
    return {
        "seed": grouped,
        "seedvr2": grouped and value == catalog.SEEDVR2_UPSCALE,
        "ltx_model": grouped and value in LTX_OPTIONS,
        "prompt": grouped and prompt,
        "prompt_hint": hint,
        "resolution": ai and value not in catalog.LTX25_SAME_RESOLUTION_OPTIONS
        and value not in catalog.RELAYVSR_OPTIONS,
    }


def choice(value, choices, label: str, default):
    """A dropdown's value as its Gradio control allowed it: one of ``choices``, or the default when unset."""
    if value in (None, ""):
        return default
    if value not in choices:
        raise H3Error(f"Choose a {label} from the list.")
    return value


def result_css() -> str:
    """The Gradio app's rules for the dub player, illustrated video and Soundstage view, for this page."""
    from ..styles import H3_WORKSPACE_CSS as WORKSPACE

    start = WORKSPACE.find("/* The dub result player")
    end = WORKSPACE.find("/* Narrower windows", start)
    if start < 0 or end < 0:
        return ""
    block = WORKSPACE[start:end].rstrip()
    if block.endswith("}") and block.count("{") < block.count("}"):
        block = block[:-1]
    return block.replace(".dark .h3-workspace ", ':root[data-theme="dark"] ').replace(".h3-workspace ", "")


@dataclass(frozen=True)
class MediaPageServices:
    """What the page needs from the application.

    ``controller`` is the Media controller the Gradio library used; ``index`` its asset index.
    """

    controller: Any
    index: Callable[[], Any]
    # Download from a URL with saved cookies (the Dubbing handlers), or None.
    downloads: Any = None
    # Set as: ``names`` and ``queue(owner, name, path)``, or None.
    set_as: Callable[[], Any] = lambda: None
    # GPU enhancement; off on GPU-free deployments.
    enhance: bool = False
    # The MiniMax H3 page's values by name, for H3 refinement.
    h3_values: Callable[[str], dict[str, Any]] | None = None
    interrupt: Callable[..., str] | None = None
    seedvr2_default: str | None = None


@dataclass
class MediaRun(Run):
    """A scan, download or enhancement, and what it left for the page."""

    result: str | None = None
    mode: str | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "result": self.result, "mode": self.mode}


def media_url(path, mode: str, *, download: bool = False) -> str:
    return f"{API}/file?mode={mode}&path={quote(str(path), safe='')}" + ("&download=1" if download else "")


class MediaPageState:
    """The page's server side: listing, selection, annotations and deletion intents per owner."""

    def __init__(self, services: MediaPageServices):
        self.services = services
        self.lock = RLock()
        self.intents: dict[str, DeletionIntent] = {}

    @property
    def controller(self):
        return self.services.controller

    # Paths --------------------------------------------------------------------
    def mode_of(self, mode) -> str:
        if mode not in KINDS:
            raise H3Error("Choose Video, Image or Audio.")
        return mode

    def managed(self, path: str, mode: str) -> Path:
        """The library file ``path``, or why it isn't one."""
        mode = self.mode_of(mode)
        if not path:
            raise H3Error(f"Select {'an' if mode != 'Video' else 'a'} {NOUNS[mode]} in the library first.")
        try:
            # The same check the download links make: only managed library files pass.
            self.controller.gallery_media_download_path(path, mode)
        except (H3Error, OSError, ValueError) as exc:
            raise H3Error("This item is unavailable in the current library. Refresh and choose another.") from exc
        found = Path(path)
        if not found.is_file():
            raise H3Error("This item is unavailable in the current library. Refresh and choose another.")
        return found

    def asset_row(self, path: Path) -> dict[str, Any] | None:
        with JOBS.store.connect() as db:
            row = db.execute(
                "SELECT a.*, COALESCE(n.tags, '[]') AS tags, COALESCE(n.favorite,0) AS favorite "
                "FROM assets a LEFT JOIN annotations n ON n.asset_id=a.id WHERE a.path=?",
                (str(path.resolve()),),
            ).fetchone()
        return dict(row) if row else None

    # Listing --------------------------------------------------------------------
    def summary(self, mode: str, engine: str) -> dict[str, Any]:
        counts, engines = self.services.index().summary()
        choices = list(engines.get(mode, []))
        return {"counts": {kind: counts.get(kind, 0) for kind in KINDS}, "engines": choices,
                "engine": engine if engine in choices else ""}

    def items(self, mode: str, query: str, favorite: bool, engine: str, sort: str, limit: int) -> dict[str, Any]:
        mode = self.mode_of(mode)
        index = self.services.index()
        summary = self.summary(mode, engine)
        limit = max(1, min(int(limit), 2000))
        sort = sort if sort in {value for _label, value in SORTS} else "newest"
        matching = index.inventory(mode, query or "", bool(favorite), limit=limit, engine=summary["engine"], sort=sort)
        page = self.controller.refresh_media_page(mode, limit, paths=matching)
        rows = {str(Path(row["path"])): row for row in getattr(matching, "records", ())}
        version = index.versions.get(mode, 0)
        items = []
        for path, (thumb, caption) in zip(page.paths, page.items):
            row = rows.get(str(Path(path))) or {}
            items.append({"path": str(path), "name": Path(path).name, "caption": caption,
                          "favorite": bool(row.get("favorite")),
                          "thumb": f"/media-previews/{Path(thumb).name}?v={version}"})
        status = f"{page.total} matching assets · {page.status}" + (
            " · Use Scan historical media to add older or externally created outputs." if page.total == 0 else "")
        return {"items": items, "total": page.total, "more": page.next_cursor is not None, "status": status,
                **summary}

    # The selected item -----------------------------------------------------------
    def describe(self, mode: str, path: str) -> dict[str, Any]:
        file = self.managed(path, mode)
        row = self.asset_row(file)
        snapshot = read_snapshot(file)
        found: dict[str, Any] = {
            "path": str(file), "name": file.name, "mode": mode,
            "url": media_url(file, mode), "download": media_url(file, mode, download=True),
            "settings": json.dumps(snapshot, indent=2, ensure_ascii=False) if snapshot
            else "Settings unavailable for this output.",
            "asset_id": None, "tags": "", "favorite": False, "lineage": [], "identity": None,
        }
        if row is not None:
            metadata = json.loads(row["metadata"])
            sources, copies = JOBS.store.asset_lineage(row["id"])
            found.update(
                asset_id=row["id"], tags=", ".join(json.loads(row["tags"])), favorite=bool(row["favorite"]),
                lineage=[{"name": name, "note": note, "current": current}
                         for name, note, current in lineage_steps(row["path"], metadata, sources, copies)],
                identity=json.dumps({"asset_id": row["id"], "filename": file.name,
                                     "settings_and_lineage": {key: value for key, value in metadata.items()
                                                              if key != "_media"}},
                                    indent=2, ensure_ascii=False),
            )
        found["views"] = self.views(file, mode, snapshot)
        return found

    def views(self, item: Path, mode: str, snapshot: dict | None) -> dict[str, Any]:
        """The Soundstage view, dub player and illustrated video for the selected item, as Media showed them.

        The same views as ``result_view_values``, with their media served by this page for this item only.
        """
        from h3_app.drama3.results import DRAMA3_FAMILY
        from h3_app.dubbing.results import DUB_FAMILY, TRACKS, load_player
        from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY, latest_export, stage_video_html
        from h3_ui.dubbing.player import player_html
        from ..result_views import soundstage_html

        found = {"soundstage": "", "dub": "", "stage": "", "switch": False}
        family = (snapshot or {}).get("family")
        item_query = f"mode={mode}&path={quote(str(item), safe='')}"

        def stage_url(_path) -> str:
            # The item names its illustrated video, so one address serves it.
            return f"{API}/stage?{item_query}"

        try:
            if family == DRAMA3_FAMILY:
                latest = latest_export(item)
                found.update(soundstage=soundstage_html(item, snapshot.get("settings") or {}),
                             stage=stage_video_html(latest, url=stage_url) if latest else "",
                             switch=latest is not None)
            elif family == ILLUSTRATED_FAMILY:
                mix = item.with_name(Path(str((snapshot.get("settings") or {}).get("source") or "")).name)
                found.update(soundstage=soundstage_html(mix) if mix.name and mix.is_file() else "",
                             stage=stage_video_html(item, url=stage_url))
            elif family == DUB_FAMILY and (player := load_player(item)) is not None:
                urls = {getattr(player, track): f"{API}/track?{item_query}&name={track}"
                        for track in TRACKS if getattr(player, track, None)}
                found["dub"] = player_html(player, key=f"media-{player.original.parent.name}", url=urls.__getitem__)
        except (OSError, TypeError, ValueError, KeyError):
            return {"soundstage": "", "dub": "", "stage": "", "switch": False}
        return found

    def track(self, mode: str, path: str, name: str) -> Path:
        from h3_app.dubbing.results import TRACKS, load_player

        file = self.managed(path, mode)
        player = load_player(file)
        location = getattr(player, name, None) if player is not None and name in TRACKS else None
        if not location or not Path(location).is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return Path(location)

    def stage(self, mode: str, path: str) -> Path:
        from h3_ui.drama3.stage_video import latest_export

        file = self.managed(path, mode)
        found = latest_export(file) if mode == "Audio" else file if mode == "Video" else None
        if not found or not Path(found).is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return Path(found)

    # Annotations ------------------------------------------------------------------
    def annotate(self, mode: str, path: str, tags: str | None, favorite: bool | None) -> dict[str, Any]:
        file = self.managed(path, mode)
        row = self.asset_row(file)
        if row is None:
            raise H3Error("This item isn't in the library index yet. Use Scan historical media first.")
        current_tags = json.loads(row["tags"])
        JOBS.store.annotate(
            row["id"],
            tags=str(tags).split(",") if tags is not None else current_tags,
            favorite=bool(row["favorite"]) if favorite is None else bool(favorite),
        )
        return self.describe(mode, path)

    # Deletion -------------------------------------------------------------------
    def prepare_deletion(self, owner: str, mode: str, selected: str | None, action: str) -> str:
        mode = self.mode_of(mode)
        paths = tuple(str(path) for path in self.controller.list_media_paths(mode))
        if action == "selected":
            if not selected or str(selected) not in paths:
                raise H3Error("Select an item in this library first.")
            paths = (str(selected),)
        elif action != "library":
            raise H3Error("Choose what to delete.")
        if not paths:
            raise H3Error("This library is empty.")
        names = ", ".join(Path(path).name for path in paths[:3])
        remainder = f" and {len(paths) - 3} more" if len(paths) > 3 else ""
        with self.lock:
            self.intents[owner] = DeletionIntent(mode, selected, action, paths, file_fingerprints(paths))
        return (f"Permanently delete {len(paths)} {mode.lower()} item(s) from the generated library? "
                f"{names}{remainder}. Settings sidecars and previews are removed too.")

    def confirm_deletion(self, owner: str, mode: str, selected: str | None) -> str:
        with self.lock:
            intent = self.intents.pop(owner, None)
        if intent is None or intent.mode != mode or intent.selected != selected:
            raise H3Error("The selection changed. Review the deletion again.")
        paths = tuple(str(path) for path in self.controller.list_media_paths(mode))
        current = (str(selected),) if intent.action == "selected" and selected in paths else paths
        if current != intent.paths:
            raise H3Error("The library changed. Review the deletion again.")
        try:
            unchanged = file_fingerprints(current) == intent.fingerprints
        except OSError:
            unchanged = False
        if not unchanged:
            raise H3Error("The media changed. Review the deletion again.")
        if any(JOBS.references_path(path) for path in intent.paths):
            raise H3Error("A queued or active job uses this media. Cancel or finish that job first.")
        result = (self.controller.delete_selected_gallery_media(mode, selected, True) if intent.action == "selected"
                  else self.controller.empty_generated_media_gallery(mode, selected, True))
        return plain(result[2])

    def cancel_deletion(self, owner: str) -> None:
        with self.lock:
            self.intents.pop(owner, None)

    # Import ---------------------------------------------------------------------
    def import_file(self, upload: Path, filename: str) -> dict[str, Any]:
        """One uploaded file into the library under its own type, or why not."""
        name = Path(filename).name
        suffix = Path(name).suffix.lower()
        mode = next((kind for kind, extensions in EXTENSIONS.items() if suffix in extensions), None)
        if mode is None:
            raise H3Error(f"{name or 'This file'} is not a video, image, or audio file.")
        named = upload.with_name(name)
        upload.replace(named)
        result = self.controller.import_gallery_media(mode, str(named))
        selected = result[7]
        if not selected or not isinstance(selected, str):
            raise H3Error(plain(result[2]))
        return {"name": name, "mode": mode, "selected": selected, "status": plain(result[2])}

    def import_files(self, uploads: list[tuple[Path, str]]) -> dict[str, Any]:
        """Each uploaded file under its own type; one that fails leaves the others imported."""
        if not uploads:
            raise H3Error("Choose a video, image, or audio file.")
        imported, failed = [], []
        for upload, filename in uploads:
            try:
                imported.append(self.import_file(upload, filename))
            except (H3Error, OSError, ValueError) as exc:
                failed.append({"name": Path(filename).name, "error": str(exc)})
        if not imported:
            raise H3Error(" ".join(item["error"] for item in failed))
        counts = {}
        for item in imported:
            counts[item["mode"]] = counts.get(item["mode"], 0) + 1
        added = ", ".join(f"{count} {NOUNS[mode]}{'s' if count > 1 and mode != 'Audio' else ''}"
                          for mode, count in counts.items())
        status = (imported[0]["status"] if len(imported) == 1 and not failed
                  else f"Added {len(imported)} file{'s' if len(imported) > 1 else ''} ({added}).")
        if failed:
            status += f" Not added: {', '.join(item['name'] for item in failed)}."
        # The page shows the last file added, under its type.
        return {"imported": imported, "failed": failed, "status": status,
                "mode": imported[-1]["mode"], "selected": imported[-1]["selected"]}


def install_media_page(app: FastAPI, pages: MediaPageState) -> MediaPageState:
    """Serve the page at ``/media/`` and its API under ``/media/api``."""
    services = pages.services
    runs = {kind: Runs(busy=message) for kind, message in BUSY.items()}
    css = result_css()

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request, owner: str):
        # Downloads stop by this key; enhancement builds its download link from ``request``.
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=f"media-{owner}",
                               headers=dict(request.headers), request=request, client=request.client)

    def run_payloads(owner: str) -> dict[str, Any]:
        found = {}
        for kind, group in runs.items():
            run = group.latest(owner, kind)
            found[kind] = run.public() if run else None
        return found

    def serve(file: Path, download: bool) -> FileResponse:
        return FileResponse(file, filename=file.name if download else None,
                            headers={"Cache-Control": "private, max-age=3600"})

    def checked(request: Request, action: Callable[[], Path], download: bool = False) -> FileResponse:
        try:
            owner_of(request)
            return serve(action(), download)
        except (H3Error, ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc

    # The page --------------------------------------------------------------------
    @app.get(ROUTE, include_in_schema=False)
    def media_redirect():
        return RedirectResponse(f"{ROUTE}/")

    @app.get(f"{ROUTE}/", include_in_schema=False)
    def media_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get(f"{ROUTE}/results.css", include_in_schema=False)
    def media_result_css():
        return Response(css, media_type="text/css", headers={"Cache-Control": "no-cache"})

    @app.get(f"{ROUTE}/results.js", include_in_schema=False)
    def media_result_js():
        """The result views' scripts, as a module the page calls on each view it shows."""
        from h3_ui.drama3.result_view import DRAMA3_SUBTITLES_JS
        from h3_ui.drama3.stage_video import STAGE_VIDEO_JS
        from h3_ui.dubbing.player import PLAYER_JS

        return Response(
            f"export const followSubtitles = {DRAMA3_SUBTITLES_JS.strip()};\n"
            f"export function bindPlayer(element) {{\n{PLAYER_JS}\n}}\n"
            f"export function bindStage(element) {{\n{STAGE_VIDEO_JS}\n}}\n",
            media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    @app.get(ROUTE + "/{name}", include_in_schema=False)
    def media_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner = owner_of(request)
        set_as = services.set_as()
        downloads = services.downloads
        video_methods = list(catalog.POSTPROCESS_OPTIONS)
        from h3_app.video_download import DEFAULT_QUALITY, QUALITIES

        return {
            "kinds": list(KINDS), "page_size": PAGE_SIZE, "sorts": [list(sort) for sort in SORTS],
            "set_as": list(set_as.names) if set_as is not None else [],
            "downloads": downloads is not None,
            "qualities": list(QUALITIES), "quality": DEFAULT_QUALITY,
            "enhance": {
                "on": bool(services.enhance),
                "methods": {"Video": video_methods, "Image": list(IMAGE_METHODS)},
                "fields": {method: method_fields(method) for method in {*video_methods, *IMAGE_METHODS}},
                "method_help": METHOD_HELP, "seedvr2_help": SEEDVR2_HELP,
                "resolutions": list(catalog.UPSCALE_RESOLUTION_PRESETS),
                "resolution": catalog.DEFAULT_UPSCALE_RESOLUTION,
                "seedvr2": list(models.SEEDVR2_MODEL_CHOICES),
                "seedvr2_default": services.seedvr2_default or models.DEFAULT_SEEDVR2_MODEL,
                "ltx": list(models.LTX25_MODEL_CHOICES), "ltx_default": models.DEFAULT_LTX25_MODEL,
            },
            "runs": run_payloads(owner),
        }

    # Browsing ----------------------------------------------------------------------
    @app.get(f"{API}/items", include_in_schema=False)
    @guarded
    def list_items(request: Request, mode: str = "Video", query: str = "", favorite: bool = False,
                   engine: str = "", sort: str = "newest", limit: int = PAGE_SIZE):
        owner_of(request)
        return pages.items(mode, query, favorite, engine, sort, limit)

    @app.get(f"{API}/item", include_in_schema=False)
    @guarded
    def read_item(request: Request, mode: str, path: str):
        owner_of(request)
        return pages.describe(mode, path)

    @app.get(f"{API}/file", include_in_schema=False)
    def read_file(request: Request, mode: str, path: str, download: bool = False):
        return checked(request, lambda: pages.managed(path, mode), download)

    @app.get(f"{API}/track", include_in_schema=False)
    def read_track(request: Request, mode: str, path: str, name: str):
        return checked(request, lambda: pages.track(mode, path, name))

    @app.get(f"{API}/stage", include_in_schema=False)
    def read_stage(request: Request, mode: str, path: str):
        return checked(request, lambda: pages.stage(mode, path))

    # Annotations ----------------------------------------------------------------------
    @app.post(f"{API}/annotate", include_in_schema=False)
    @guarded
    def annotate(request: Request, body: dict = Body(...)):
        owner_of(request)
        favorite = body.get("favorite")
        return pages.annotate(str(body.get("mode")), str(body.get("path") or ""),
                              body.get("tags"), None if favorite is None else bool(favorite))

    # Set as --------------------------------------------------------------------------
    @app.post(f"{API}/set-as", include_in_schema=False)
    @guarded
    def set_as(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        destinations = services.set_as()
        if destinations is None:
            raise H3Error("Set as isn't available here.")
        path = str(pages.managed(str(body.get("path") or ""), "Image"))
        return {"message": plain(destinations.queue(owner, str(body.get("name") or ""), path))}

    # Deletion --------------------------------------------------------------------------
    @app.post(f"{API}/delete/prepare", include_in_schema=False)
    @guarded
    def prepare_deletion(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        return {"message": pages.prepare_deletion(owner, str(body.get("mode")), body.get("path"),
                                                  str(body.get("action") or ""))}

    @app.post(f"{API}/delete/confirm", include_in_schema=False)
    @guarded
    def confirm_deletion(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        return {"message": pages.confirm_deletion(owner, str(body.get("mode")), body.get("path"))}

    @app.post(f"{API}/delete/cancel", include_in_schema=False)
    @guarded
    def cancel_deletion(request: Request):
        pages.cancel_deletion(owner_of(request))
        return {"ok": True}

    # Import ---------------------------------------------------------------------------
    @app.post(f"{API}/import", include_in_schema=False)
    @guarded
    def import_media(request: Request, files: list[UploadFile] = File(...)):
        owner_of(request)
        if len(files) > IMPORT_LIMIT:
            raise H3Error(f"Add up to {IMPORT_LIMIT} files at a time.")
        folder = Path(tempfile.mkdtemp(prefix="h3-media-import-"))
        try:
            uploads = []
            for file in files:
                # Each upload gets its own folder, so two files with the same name stay apart.
                upload = folder / uuid.uuid4().hex / uuid.uuid4().hex
                upload.parent.mkdir()
                with upload.open("wb") as handle:
                    shutil.copyfileobj(file.file, handle)
                uploads.append((upload, file.filename or ""))
            return pages.import_files(uploads)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    # Long work -------------------------------------------------------------------------
    def start(owner: str, kind: str, work: Callable[[MediaRun], None], status: str, mode: str | None = None):
        run = MediaRun(owner, kind, kind, mode=mode)
        run.status = status
        runs[kind].start(run, work)
        return {"runs": run_payloads(owner)}

    @app.post(f"{API}/scan", include_in_schema=False)
    @guarded
    def scan(request: Request):
        owner = owner_of(request)

        def work(run: MediaRun):
            for message in services.index().rebuild():
                run.status = plain(message)
                run.revision += 1

        return start(owner, SCAN, work, "Scanning historical media…")

    def downloads():
        if services.downloads is None:
            raise H3Error("Downloads aren't available here.")
        return services.downloads

    @app.post(f"{API}/download", include_in_schema=False)
    @guarded
    def download(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        actions = downloads()
        session = session_of(request, owner)
        url, quality = str(body.get("url") or "").strip(), str(body.get("quality") or "")

        def work(run: MediaRun):
            for status, path in actions.download_video(url, quality, request=session):
                run.status = plain(status)
                if path:
                    run.result, run.mode = str(path), "Video"
                run.revision += 1
            if run.status.startswith("Error:"):
                raise H3Error(run.status.removeprefix("Error:").strip())

        return start(owner, DOWNLOAD, work, "Starting the download…", "Video")

    @app.post(f"{API}/download/info", include_in_schema=False)
    @guarded
    def download_info(request: Request, body: dict = Body(...)):
        owner_of(request)
        return {"message": plain(downloads().video_info(str(body.get("url") or "").strip()))}

    @app.post(f"{API}/download/stop", include_in_schema=False)
    @guarded
    def download_stop(request: Request):
        owner = owner_of(request)
        return {"message": plain(downloads().stop_downloads(session_of(request, owner))),
                "runs": run_payloads(owner)}

    def cookie_answer(actions, message: str = "") -> dict[str, Any]:
        """The saved sites after a cookie action, and what the action said."""
        sites = list(actions.cookie_site_choices().get("choices") or [])
        return {"message": plain(message), "summary": plain(actions.cookie_sites()),
                "sites": [site[0] if isinstance(site, (list, tuple)) else site for site in sites]}

    @app.get(f"{API}/cookies", include_in_schema=False)
    @guarded
    def cookies(request: Request):
        owner_of(request)
        return cookie_answer(downloads())

    @app.post(f"{API}/cookies/curl", include_in_schema=False)
    @guarded
    def cookies_curl(request: Request, body: dict = Body(...)):
        owner_of(request)
        actions = downloads()
        message = actions.import_curl_cookies(str(body.get("curl") or ""), str(body.get("domain") or ""))[0]
        return {**cookie_answer(actions, message), "saved": not plain(message).startswith("Not saved")}

    @app.post(f"{API}/cookies/file", include_in_schema=False)
    @guarded
    def cookies_file(request: Request, file: UploadFile = File(...), domain: str = Form("")):
        owner_of(request)
        actions = downloads()
        folder = Path(tempfile.mkdtemp(prefix="h3-media-cookies-"))
        try:
            target = folder / "cookies.txt"
            with target.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            message = actions.import_cookie_file(str(target), domain)[0]
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return cookie_answer(actions, message)

    @app.post(f"{API}/cookies/delete", include_in_schema=False)
    @guarded
    def cookies_delete(request: Request, body: dict = Body(...)):
        owner_of(request)
        actions = downloads()
        return cookie_answer(actions, actions.delete_cookies(str(body.get("domain") or ""))[0])

    @app.post(f"{API}/enhance", include_in_schema=False)
    @guarded
    def enhance(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        if not services.enhance or "gallery" not in JOBS.callbacks:
            raise H3Error("Enhancing needs the GPU. Open Media in Studio.")
        mode = pages.mode_of(str(body.get("mode")))
        if mode == "Audio":
            raise H3Error("Audio gallery outputs are available for playback and download.")
        path = str(pages.managed(str(body.get("path") or ""), mode))
        method = str(body.get("method") or "")
        allowed = IMAGE_METHODS if mode == "Image" else tuple(catalog.POSTPROCESS_OPTIONS)
        if method not in allowed:
            raise H3Error("Choose an enhancement method.")
        try:
            seed = int(body.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise H3Error("Seed must be a whole number.") from exc
        values = [mode, path, method, seed,
                  choice(body.get("seedvr2_model"), models.SEEDVR2_MODEL_CHOICES, "SeedVR2 model",
                         services.seedvr2_default or models.DEFAULT_SEEDVR2_MODEL),
                  choice(body.get("ltx_model"), models.LTX25_MODEL_CHOICES, "finishing LTX model",
                         models.DEFAULT_LTX25_MODEL),
                  str(body.get("prompt") or ""), bool(body.get("force_offload")),
                  choice(body.get("resolution"), catalog.UPSCALE_RESOLUTION_PRESETS, "target resolution",
                         catalog.DEFAULT_UPSCALE_RESOLUTION)]
        if services.h3_values is not None:
            from h3_app.refinement import H3_REFINEMENT_FIELDS

            h3 = services.h3_values(owner)
            values.extend(h3.get(name) for name in H3_REFINEMENT_FIELDS)
        session = session_of(request, owner)

        def work(run: MediaRun):
            job = JOBS.accept(owner, "gallery", values, callback=JOBS.callbacks["gallery"], key=uuid.uuid4().hex,
                              media_indices=(1,), output_indices=(3, 4))
            run.job_id = job.id
            for update in execute_accepted(job.id, session):
                status = update[-1] if update else None
                if isinstance(status, str) and status:
                    run.status = plain(status)
                selected = update[7] if len(update) > 7 else None
                if isinstance(selected, str) and selected and selected != path:
                    run.result = selected
                run.revision += 1

        return start(owner, ENHANCE, work, "Waiting for the GPU…", mode)

    @app.post(f"{API}/enhance/stop", include_in_schema=False)
    @guarded
    def enhance_stop(request: Request):
        owner = owner_of(request)
        run = runs[ENHANCE].active(owner, ENHANCE)
        if services.interrupt is None:
            raise H3Error("Nothing can run here.")
        if run is None:
            message = "Nothing is running."
        elif run.job_id is None:
            message = "The enhancement is still starting. Try again in a moment."
        else:
            message = services.interrupt(session_of(request, owner), family="gallery", job_id=run.job_id)
        return {"message": plain(message), "runs": run_payloads(owner)}

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request):
        return {"runs": run_payloads(owner_of(request))}

    return pages
