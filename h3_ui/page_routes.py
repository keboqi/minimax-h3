"""Shared FastAPI route mounting, owner/session extraction, and run interruption for standalone pages."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.workspace_store import request_owner

from .page_fields import UI_LIMIT
from .page_runs import Run

COMMON_STATIC = Path(__file__).with_name("page_static")
COMMON_STATIC_TYPES = {"common.js": "application/javascript", "score_radar.css": "text/css"}


def install_common_static(app: FastAPI) -> None:
    """Mount `/page-common/{name}` once per FastAPI application."""
    if getattr(app.state, "h3_common_static_installed", False):
        return
    app.state.h3_common_static_installed = True

    @app.get("/page-common/{name}", include_in_schema=False)
    def common_page_asset(name: str):
        if name not in COMMON_STATIC_TYPES or not (COMMON_STATIC / name).is_file():
            raise HTTPException(status_code=404)
        return FileResponse(
            COMMON_STATIC / name,
            media_type=COMMON_STATIC_TYPES[name],
            headers={"Cache-Control": "no-cache"},
        )


def install_static_page(
    app: FastAPI,
    route: str,
    static_dir: Path,
    static_types: Mapping[str, str],
    *,
    extra_files: Mapping[str, tuple[Path, str]] | None = None,
) -> None:
    """Mount `GET {route}`, `GET {route}/`, and `GET {route}/{name}` for a standalone page."""
    install_common_static(app)
    prefix = route.rstrip("/")

    @app.get(prefix, include_in_schema=False)
    def page_redirect():
        return RedirectResponse(f"{prefix}/")

    @app.get(f"{prefix}/", include_in_schema=False)
    def page_index():
        return FileResponse(static_dir / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    for asset_name, (asset_path, media_type) in (extra_files or {}).items():
        def register_extra(name: str, path: Path, mime: str) -> None:
            @app.get(f"{prefix}/{name}", include_in_schema=False)
            def page_extra_asset():
                return FileResponse(path, media_type=mime, headers={"Cache-Control": "no-cache"})

        register_extra(asset_name, asset_path, media_type)

    @app.get(f"{prefix}/{{name}}", include_in_schema=False)
    def page_asset(name: str):
        if name in static_types:
            return FileResponse(static_dir / name, media_type=static_types[name], headers={"Cache-Control": "no-cache"})
        if name in COMMON_STATIC_TYPES and (COMMON_STATIC / name).is_file():
            return FileResponse(
                COMMON_STATIC / name,
                media_type=COMMON_STATIC_TYPES[name],
                headers={"Cache-Control": "no-cache"},
            )
        raise HTTPException(status_code=404)


def owner_of(request: Request) -> str:
    return request_owner(request, JOBS.store)


def session_of(request: Request, *, session_hash: str | None = None, include_request: bool = False) -> SimpleNamespace:
    """What outlives ``request`` and names its owner to the job path."""
    values: dict[str, Any] = {
        "cookies": dict(request.cookies),
        "session_hash": session_hash,
        "headers": dict(request.headers),
    }
    if include_request:
        values.update(request=request, client=request.client)
    return SimpleNamespace(**values)


def validate_ui_patch(body: Mapping[str, Any]) -> dict[str, Any]:
    ui = body.get("ui")
    if not isinstance(ui, dict) or len(json.dumps(ui)) > UI_LIMIT:
        raise H3Error("That is too much to remember.")
    return ui


def interrupt_active_run(
    request: Request,
    run: Run | None,
    family: str,
    interrupt: Callable[..., str] | None,
    *,
    idle: str = "Nothing is running.",
    starting: str = "The run is still starting. Try again in a moment.",
    session: Any = None,
) -> str:
    if interrupt is None:
        raise H3Error("Nothing can run here.")
    if run is None:
        return idle
    if run.job_id is None:
        return starting
    return interrupt(session or session_of(request), family=family, job_id=run.job_id)
