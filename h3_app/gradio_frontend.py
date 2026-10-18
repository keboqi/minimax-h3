"""Serve Gradio's frontend bundle with a fix for its per-event loading-status cost.

On every server event, Gradio's frontend (checked against 6.30.0) re-sends the
loading status of each component that any earlier server event used as an
input or output, and each send searches the whole layout tree for that
component. In a UI this large, the cost grows with every event and reaches
seconds for streamed jobs, on every tab. The patched bundle re-sends only
statuses that changed. A bundle without the expected code is served unchanged,
as is every bundle when H3_GRADIO_FRONTEND_PATCH=0.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.resources
import os
import re

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response

_ASSETS = importlib.resources.files("gradio").joinpath("templates/frontend/assets")
_BUNDLE = re.compile(r"Blocks-[A-Za-z0-9_-]+\.js")
LOADING_STATI_ORIGINAL = (
    "async update_loading_stati_state(){await Promise.all(Object.entries(this.loading_stati.current)"
    ".map(([e,t])=>this.update_state_cb(Number(e),{loading_status:t},!1)))}"
)
LOADING_STATI_PATCHED = (
    "async update_loading_stati_state(){let h3_sent=this.h3_sent_stati??=new Map;"
    "await Promise.all(Object.entries(this.loading_stati.current).map(([e,t])=>{"
    "let h3_key=JSON.stringify(t);if(h3_sent.get(e)===h3_key)return;h3_sent.set(e,h3_key);"
    "return this.update_state_cb(Number(e),{loading_status:t},!1)}))}"
)


@lru_cache(maxsize=4)
def patched_bundle(name: str) -> bytes | None:
    """The bundle with the loading-status fix, or None when disabled or this Gradio build differs."""
    if os.getenv("H3_GRADIO_FRONTEND_PATCH", "1") == "0":
        return None
    source = _ASSETS.joinpath(name).read_text(encoding="utf-8")
    if source.count(LOADING_STATI_ORIGINAL) != 1:
        print(f"[h3-ui] Gradio bundle {name} has no known loading-status code; serving it unchanged.",
              flush=True)
        return None
    return source.replace(LOADING_STATI_ORIGINAL, LOADING_STATI_PATCHED).encode("utf-8")


def install_gradio_frontend_patch(app: FastAPI) -> None:
    """Register before mounting Gradio so this route answers first."""

    @app.get("/assets/Blocks-{bundle_hash}.js", include_in_schema=False)
    async def gradio_bundle(bundle_hash: str, request: Request):
        name = f"Blocks-{bundle_hash}.js"
        path = _ASSETS.joinpath(name)
        if not _BUNDLE.fullmatch(name) or not path.is_file():
            return Response(status_code=404)
        body = patched_bundle(name)
        if body is None:
            return FileResponse(str(path), media_type="text/javascript")
        etag = '"h3-' + hashlib.sha256(body).hexdigest()[:32] + '"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(body, media_type="text/javascript", headers=headers)
