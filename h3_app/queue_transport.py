"""Carry Gradio's existing queue stream over WebSockets through public proxies."""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


QUEUE_TRANSPORT_HEAD = '<script src="/h3/queue-transport.js"></script>'


class QueueTransportHeadMiddleware:
    """Install the fetch bridge before Gradio starts page-load queue events."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        start = None
        chunks = []

        async def send_html(message):
            nonlocal start
            if message["type"] == "http.response.start":
                headers = dict(message["headers"])
                if b"text/html" in headers.get(b"content-type", b"") and b"content-encoding" not in headers:
                    start = message
                    return
            if start is not None and message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body = re.sub(
                        rb"(<head(?:\s[^>]*)?>)",
                        lambda match: match.group(0) + QUEUE_TRANSPORT_HEAD.encode("utf-8"),
                        b"".join(chunks), count=1, flags=re.IGNORECASE,
                    )
                    headers = [(key, value) for key, value in start["headers"] if key != b"content-length"]
                    headers.append((b"content-length", str(len(body)).encode("ascii")))
                    await send({**start, "headers": headers})
                    await send({"type": "http.response.body", "body": body, "more_body": False})
                return
            await send(message)

        await self.app(scope, receive, send_html)


async def relay_queue_stream(app, socket: WebSocket) -> None:
    # Browser cookies must not grant a different website access to this stream.
    origin = urlsplit(socket.headers.get("origin", ""))
    if origin.scheme not in {"http", "https"} or origin.netloc.lower() != socket.headers.get("host", "").lower():
        await socket.close(code=1008)
        return
    await socket.accept()
    tasks = set()
    try:
        options = await asyncio.wait_for(socket.receive_json(), timeout=10)
        if not isinstance(options, dict):
            await socket.close(code=1008)
            return
        authorization = options.get("authorization")
        if authorization is not None and not isinstance(authorization, str):
            await socket.close(code=1008)
            return
        headers = [
            (key, value)
            for key, value in socket.scope["headers"]
            if key not in {b"connection", b"upgrade", b"accept", b"accept-encoding"}
            and not (key == b"authorization" and authorization)
            and not key.startswith(b"sec-websocket-")
        ]
        headers.append((b"accept", b"text/event-stream"))
        if authorization:
            headers.append((b"authorization", authorization.encode("latin1")))
        scope = dict(socket.scope)
        scope.update(
            type="http",
            method="GET",
            scheme="https" if socket.url.scheme == "wss" else "http",
            headers=headers,
            asgi={"version": "3.0", "spec_version": "2.3"},
        )
        for key in ("endpoint", "route", "path_params"):
            scope.pop(key, None)
        disconnected = asyncio.Event()
        request_sent = False

        async def receive_http():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send_http(message):
            if message["type"] == "http.response.start":
                content_type = next(
                    (value.decode("latin1") for key, value in message["headers"] if key == b"content-type"),
                    "text/event-stream",
                )
                await socket.send_json({"status": message["status"], "content_type": content_type})
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    await socket.send_bytes(message["body"])

        async def watch_disconnect():
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    disconnected.set()
                    return

        stream = asyncio.create_task(app(scope, receive_http, send_http))
        watcher = asyncio.create_task(watch_disconnect())
        tasks = {stream, watcher}
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if stream in done and not disconnected.is_set():
            await socket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[h3-ui] Queue websocket error: {type(exc).__name__}: {exc}", flush=True)
        try:
            await socket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def install_queue_transport(app: FastAPI) -> None:
    @app.get("/h3/queue-transport.js", include_in_schema=False)
    async def transport_script():
        return FileResponse(
            Path(__file__).with_suffix(".js"),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/gradio_api/queue/data")
    @app.websocket("/gradio_api/heartbeat/{session_hash}")
    async def queue_websocket(socket: WebSocket):
        await relay_queue_stream(app, socket)
