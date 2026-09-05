"""FastAPI routes and transparent ComfyUI HTTP/WebSocket proxy."""

from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import quote
import aiohttp
import httpx
import gradio as gr
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)

COMFY_PROXY_PATH = "/comfyui"


@dataclass(frozen=True)
class ServerConfig:
    comfy_url: str
    output_dir: Path
    outputs_dir: Path
    workflows: dict
    workflow_dir: Path
    video_extensions: frozenset[str]
    css: str


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_COMFY_REWRITE_TYPES = (
    "text/html",
    "text/css",
)


def _proxy_headers(headers: Any) -> dict[str, str]:
    connection = next(
        (value for key, value in headers.items() if key.lower() == "connection"),
        "",
    )
    connection_tokens = {
        token.strip().lower() for token in connection.split(",") if token.strip()
    }
    blocked = _HOP_BY_HOP_HEADERS | connection_tokens | {"host", "content-length"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _rewrite_comfy_text(content: str, content_type: str) -> str:
    """Keep ComfyUI's static browser assets inside the proxy prefix."""
    prefix = COMFY_PROXY_PATH
    if content_type.startswith("text/html"):
        content = re.sub(
            r"(?i)(\b(?:href|src|action)\s*=\s*[\"']?)/(?!/)",
            rf"\1{prefix}/",
            content,
        )
        if not re.search(r"(?i)<base(?:\s|>)", content):
            content = re.sub(
                r"(?i)(<head(?:\s[^>]*)?>)",
                rf'\1<base href="{prefix}/">',
                content,
                count=1,
            )
    elif content_type.startswith("text/css"):
        content = re.sub(
            r"(?i)(url\(\s*[\"']?)/(?!/)",
            rf"\1{prefix}/",
            content,
        )
    return content


def _comfy_upstream_path(path: str, raw_path: bytes | None) -> str:
    """Preserve encoded userdata separators consumed by ASGI route matching."""
    prefix = f"{COMFY_PROXY_PATH}/".encode("ascii")
    if isinstance(raw_path, bytes) and raw_path.startswith(prefix):
        try:
            return raw_path[len(prefix) :].decode("ascii")
        except UnicodeDecodeError:
            pass
    return quote(path, safe="/:@")


async def _close_websocket(socket: WebSocket, code: int = 1000) -> None:
    try:
        await socket.close(code=code)
    except (RuntimeError, WebSocketDisconnect):
        # The browser may already have completed its side of the close handshake.
        pass


async def _relay_comfy_websocket(
    socket: WebSocket,
    upstream: aiohttp.ClientWebSocketResponse,
) -> None:
    async def browser_to_comfy() -> None:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("text")
            if payload is not None:
                await upstream.send_str(payload)
            else:
                await upstream.send_bytes(message.get("bytes", b""))

    async def comfy_to_browser() -> None:
        async for message in upstream:
            if message.type == aiohttp.WSMsgType.TEXT:
                await socket.send_text(message.data)
            elif message.type == aiohttp.WSMsgType.BINARY:
                await socket.send_bytes(message.data)
            elif message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"ComfyUI websocket failed: {upstream.exception()}")
            elif message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
            }:
                return

    tasks = {
        asyncio.create_task(browser_to_comfy()),
        asyncio.create_task(comfy_to_browser()),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(
            result,
            asyncio.CancelledError,
        ):
            raise result


def _append_set_cookies(response: Response, headers: httpx.Headers) -> Response:
    for cookie in headers.get_list("set-cookie"):
        response.headers.append("set-cookie", cookie)
    return response


def build_server(
    demo: gr.Blocks, allowed_paths: list[str], config: ServerConfig
) -> FastAPI:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=30),
        follow_redirects=False,
        trust_env=False,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get(
        "/ltx25-workflows/{workflow_id}.json",
        name="download_ltx25_workflow",
        include_in_schema=False,
    )
    async def download_ltx25_workflow(workflow_id: str) -> FileResponse:
        entry = next(
            (
                candidate
                for candidate in config.workflows.values()
                if candidate["id"] == workflow_id
            ),
            None,
        )
        if entry is None:
            raise HTTPException(status_code=404, detail="Workflow not found")
        root = config.workflow_dir.resolve()
        candidate = (root / entry["filename"]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(
                status_code=404,
                detail="Workflow template is not installed; re-run setup_h3.py",
            )
        return FileResponse(
            candidate,
            filename=entry["filename"],
            media_type="application/json",
        )

    @app.get(COMFY_PROXY_PATH, include_in_schema=False)
    async def comfy_slash_redirect() -> RedirectResponse:
        return RedirectResponse(f"{COMFY_PROXY_PATH}/", status_code=307)

    @app.get(
        "/downloads/{bucket}/{file_path:path}",
        name="download_generated_video",
        include_in_schema=False,
    )
    async def download_generated_video(
        bucket: str,
        file_path: str,
        download: bool = False,
    ) -> FileResponse:
        roots = {
            "comfy": config.output_dir.resolve(),
            "gradio": config.outputs_dir.resolve(),
        }
        root = roots.get(bucket)
        if root is None:
            raise HTTPException(status_code=404, detail="Video not found")
        candidate = (root / file_path).resolve()
        if (
            not candidate.is_relative_to(root)
            or candidate.suffix.lower() not in config.video_extensions
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(
            candidate,
            filename=candidate.name if download else None,
        )

    @app.api_route(
        f"{COMFY_PROXY_PATH}/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def comfy_http_proxy(path: str, request: Request) -> Response:
        upstream_path = _comfy_upstream_path(
            path,
            request.scope.get("raw_path"),
        )
        target = f"{config.comfy_url}/{upstream_path}"
        if request.url.query:
            target += f"?{request.url.query}"
        upstream_request = client.build_request(
            request.method,
            target,
            headers=_proxy_headers(request.headers),
            content=request.stream(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            print(f"[h3-ui] ComfyUI HTTP proxy error: {exc}", flush=True)
            return PlainTextResponse(
                "ComfyUI backend is unavailable",
                status_code=502,
            )
        headers = _proxy_headers(upstream.headers)
        headers.pop("set-cookie", None)
        location = headers.get("location")
        if location and location.startswith("/") and not location.startswith("//"):
            headers["location"] = f"{COMFY_PROXY_PATH}{location}"

        content_type = upstream.headers.get("content-type", "")
        if any(content_type.startswith(kind) for kind in _COMFY_REWRITE_TYPES):
            body = await upstream.aread()
            await upstream.aclose()
            encoding = upstream.encoding or "utf-8"
            rewritten = _rewrite_comfy_text(
                body.decode(encoding, errors="replace"),
                content_type,
            )
            for name in ("content-encoding", "content-length", "etag"):
                headers.pop(name, None)
            response = Response(
                content=rewritten.encode(encoding),
                status_code=upstream.status_code,
                headers=headers,
                media_type=None,
            )
            return _append_set_cookies(response, upstream.headers)

        async def stream_response() -> Any:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(
            stream_response(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=None,
        )
        return _append_set_cookies(response, upstream.headers)

    @app.websocket(f"{COMFY_PROXY_PATH}/{{path:path}}")
    async def comfy_websocket_proxy(socket: WebSocket, path: str) -> None:
        query = f"?{socket.url.query}" if socket.url.query else ""
        upstream_url = re.sub(r"^http", "ws", config.comfy_url, count=1)
        upstream_path = _comfy_upstream_path(
            path,
            socket.scope.get("raw_path"),
        )
        upstream_url += f"/{upstream_path}{query}"
        protocols = [
            value.strip()
            for value in socket.headers.get("sec-websocket-protocol", "").split(",")
            if value.strip()
        ]
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
            async with aiohttp.ClientSession(
                timeout=timeout,
                trust_env=False,
            ) as session:
                async with session.ws_connect(
                    upstream_url,
                    protocols=protocols,
                    max_msg_size=0,
                    autoping=True,
                ) as upstream:
                    await socket.accept(subprotocol=upstream.protocol or None)
                    await _relay_comfy_websocket(socket, upstream)
                    await _close_websocket(socket, upstream.close_code or 1000)
        except WebSocketDisconnect:
            # The browser or an outer deployment proxy completed the close.
            return
        except Exception as exc:
            print(
                f"[h3-ui] ComfyUI websocket proxy error: {type(exc).__name__}: {exc!r}",
                flush=True,
            )
            await _close_websocket(socket, code=1011)

    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        allowed_paths=allowed_paths,
        show_error=True,
        css=config.css,
    )
