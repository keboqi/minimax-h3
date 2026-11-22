"""Serve Gradio's frontend bundle with a fix for its per-event loading-status cost.

On every server event, Gradio's frontend (checked against 6.31.0) re-sends the
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
_BUNDLE = re.compile(r"(Blocks|core)-[A-Za-z0-9_-]+\.js")
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
HANDLE_DATA_ORIGINAL = (
    "async handle_data(e,t){await Promise.all(e.map(async(n,r)=>{let i=t[r]===void 0?qe:t[r];"
    "if(i!==qe)if(Xe(i)){let t=!1,n=null;for(let[a,o]of Object.entries(i))if(a!==`__type__`)"
    "{if(a===`visible`){t=!0,n=o;continue}await this.update_state_cb(e[r],{[a]:o},!1)}"
    "t&&await this.update_state_cb(e[r],{visible:n},!0)}else await this.update_state_cb(n,{value:i},!1)}))}"
)
HANDLE_DATA_PATCHED = (
    "async handle_data(e,t){await Promise.all(e.map(async(n,r)=>{let i=t[r]===void 0?qe:t[r];"
    "if(i!==qe)if(Xe(i)){let t=!1,n=null,s={},u=!1;for(let[a,o]of Object.entries(i))if(a!==`__type__`)"
    "{if(a===`visible`){t=!0,n=o;continue}s[a]=o,u=!0}"
    "if(`selected`in s&&typeof window<`u`){if(s.selected===`h3`||s.selected===`ltx`)window.__h3VideoEngine??=s.selected;"
    "if(window.__h3DesiredEngineTab)s.selected=window.__h3DesiredEngineTab}"
    "u&&await this.update_state_cb(e[r],s,!1);"
    "t&&await this.update_state_cb(e[r],{visible:n},!0)}else await this.update_state_cb(n,{value:i},!1)}))}"
)
CLEAR_VALIDATION_ORIGINAL = (
    "if(o===0&&r.inputs.forEach(e=>{this.update_state_cb(e,{loading_status:{validation_error:null}},!1)}),o+=1,s!==null)"
)
CLEAR_VALIDATION_PATCHED = (
    "if(o===0&&this.h3_invalid_inputs?.size&&r.inputs.forEach(e=>{"
    "this.h3_invalid_inputs.delete(e)&&this.update_state_cb(e,{loading_status:{validation_error:null}},!1)}),o+=1,s!==null)"
)
SET_VALIDATION_ORIGINAL = (
    "s.message.forEach((e,t)=>{this.update_state_cb(r.inputs[t],"
    "{loading_status:{validation_error:e.is_valid?null:e.message,show_validation_error:!0}},!1)})"
)
SET_VALIDATION_PATCHED = (
    "s.message.forEach((e,t)=>{!e.is_valid&&(this.h3_invalid_inputs??=new Set).add(r.inputs[t]);"
    "this.update_state_cb(r.inputs[t],"
    "{loading_status:{validation_error:e.is_valid?null:e.message,show_validation_error:!0}},!1)})"
)
CORE_SYNC_ORIGINAL = "r&&!(`value`in t)&&(await this.#m(e,r),await this.#h(r));"
CORE_SYNC_PATCHED = "r&&n&&!(`value`in t)&&(await this.#m(e,r),await this.#h(r));"
CORE_FIND_ORIGINAL = (
    "function Z(e,t){if(e.id===t)return e;if(e.children)for(let n of e.children)"
    "{let e=Z(n,t);if(e)return e}return null}"
)
CORE_FIND_PATCHED = (
    "function Z(e,t){let c=e._h3_nodes;if(c){let r=c.get(t);if(r?.id===t)return r}"
    "else if(e.children)c=e._h3_nodes=new Map;if(e.id===t)return e;if(e.children)"
    "for(let n of e.children){let r=Z(n,t);if(r)return c?.set(t,r),r}return null}"
)
CORE_RERENDER_ORIGINAL = "a.children=i.children,this.#f(i)}"
CORE_RERENDER_PATCHED = "a.children=i.children,this.root._h3_nodes?.clear(),this.#f(i)}"

_BLOCKS_PATCHES = (
    (LOADING_STATI_ORIGINAL, LOADING_STATI_PATCHED),
    (HANDLE_DATA_ORIGINAL, HANDLE_DATA_PATCHED),
    (CLEAR_VALIDATION_ORIGINAL, CLEAR_VALIDATION_PATCHED),
    (SET_VALIDATION_ORIGINAL, SET_VALIDATION_PATCHED),
)
_CORE_PATCHES = (
    (CORE_SYNC_ORIGINAL, CORE_SYNC_PATCHED),
    (CORE_FIND_ORIGINAL, CORE_FIND_PATCHED),
    (CORE_RERENDER_ORIGINAL, CORE_RERENDER_PATCHED),
)


@lru_cache(maxsize=8)
def patched_bundle(name: str) -> bytes | None:
    """The bundle with the frontend performance fixes, or None when disabled or this Gradio build differs."""
    if os.getenv("H3_GRADIO_FRONTEND_PATCH", "1") == "0":
        return None
    patches = _BLOCKS_PATCHES if name.startswith("Blocks-") else _CORE_PATCHES if name.startswith("core-") else ()
    if not patches:
        return None
    source = _ASSETS.joinpath(name).read_text(encoding="utf-8")
    if any(source.count(original) != 1 for original, _ in patches):
        print(f"[h3-ui] Gradio bundle {name} has no known patch target; serving it unchanged.",
              flush=True)
        return None
    for original, patched in patches:
        source = source.replace(original, patched)
    return source.encode("utf-8")


def install_gradio_frontend_patch(app: FastAPI) -> None:
    """Register before mounting Gradio so these routes answer first."""

    async def _serve_bundle(name: str, request: Request):
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

    @app.get("/assets/Blocks-{bundle_hash}.js", include_in_schema=False)
    async def gradio_blocks_bundle(bundle_hash: str, request: Request):
        return await _serve_bundle(f"Blocks-{bundle_hash}.js", request)

    @app.get("/assets/core-{bundle_hash}.js", include_in_schema=False)
    async def gradio_core_bundle(bundle_hash: str, request: Request):
        return await _serve_bundle(f"core-{bundle_hash}.js", request)
