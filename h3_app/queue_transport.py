"""Carry Gradio's existing queue stream over WebSockets through public proxies."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


QUEUE_TRANSPORT_HEAD = '<script src="/h3/queue-transport.js"></script>'
# A proxy can drop a WebSocket in the middle of a long job. Gradio treats a
# dropped queue stream as a lost server (the page reloads) and a dropped
# heartbeat as a closed tab (the session's running events stop), so a dropped
# stream stays open here, keeping its unread bytes, until the browser resumes it.
RESUME_SECONDS = 60.0
# A stream whose browser stops acknowledging bytes is abandoned past this size.
MAX_UNACKED_BYTES = 32 * 1024 * 1024
_STREAM_ID = re.compile(r"[0-9a-f]{32}")
_STREAMS: dict[str, "_Stream"] = {}
_SOCKET_ERRORS = (WebSocketDisconnect, RuntimeError, OSError)


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


class _Stream:
    """One Gradio HTTP stream that outlives the WebSocket carrying it."""

    def __init__(self, key, target):
        self.key, self.target = key, target
        self.start = None
        self.buffer = bytearray()
        self.base = 0  # stream offset of buffer[0]; earlier bytes were acknowledged
        self.finished = self.failed = False
        self.disconnected = asyncio.Event()
        self.changed = asyncio.Event()
        self.socket = None
        self.task = None
        self.expiry = None

    @property
    def end(self):
        return self.base + len(self.buffer)

    def acknowledge(self, offset):
        offset = min(max(offset, self.base), self.end)
        del self.buffer[:offset - self.base]
        self.base = offset

    def attach(self, socket):
        if self.expiry is not None:
            self.expiry.cancel()
            self.expiry = None
        self.socket = socket
        # Wakes a replaced socket's delivery so it stops.
        self.changed.set()

    def detach(self, socket, *, resumable):
        if self.socket is not socket:
            return
        self.socket = None
        if resumable and not self.disconnected.is_set():
            self.expiry = asyncio.get_running_loop().call_later(RESUME_SECONDS, self.close)
        else:
            self.close()

    def close(self):
        """Allow no further resume; Gradio then sees the browser disconnect."""
        if self.expiry is not None:
            self.expiry.cancel()
            self.expiry = None
        self.disconnected.set()
        if _STREAMS.get(self.key) is self:
            del _STREAMS[self.key]
        self.changed.set()

    async def run(self, app, scope):
        request_sent = False

        async def receive_http():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await self.disconnected.wait()
            return {"type": "http.disconnect"}

        async def send_http(message):
            if self.disconnected.is_set():
                return
            if message["type"] == "http.response.start":
                content_type = next(
                    (value.decode("latin1") for key, value in message["headers"] if key == b"content-type"),
                    "text/event-stream",
                )
                self.start = {"status": message["status"], "content_type": content_type}
            elif message["type"] == "http.response.body" and message.get("body"):
                self.buffer += message["body"]
                if len(self.buffer) > MAX_UNACKED_BYTES:
                    print("[h3-ui] Queue websocket stopped: the browser stopped reading progress.", flush=True)
                    self.failed = True
                    self.close()
            self.changed.set()

        try:
            await app(scope, receive_http, send_http)
        except Exception as exc:
            print(f"[h3-ui] Queue websocket error: {type(exc).__name__}: {exc}", flush=True)
            self.failed = True
        finally:
            self.finished = True
            self.changed.set()

    async def deliver(self, socket, offset):
        """Send from ``offset`` on; the close code once done, or None when the socket drops or is replaced."""
        sent, started = offset, False
        try:
            while self.socket is socket:
                if self.start is not None and not started:
                    started = True
                    await socket.send_json(self.start)
                elif sent < self.end:
                    chunk = bytes(self.buffer[max(sent, self.base) - self.base:])
                    sent = self.end
                    await socket.send_bytes(chunk)
                elif self.failed or self.disconnected.is_set():
                    return 1011
                elif self.finished:
                    return 1000
                else:
                    self.changed.clear()
                    await self.changed.wait()
        except _SOCKET_ERRORS:
            pass
        return None


async def _listen(stream, socket):
    """Apply acknowledgements until the socket closes; return its close code."""
    try:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return message.get("code", 1005)
            try:
                ack = json.loads(message.get("text") or "null")
            except ValueError:
                continue
            if isinstance(ack, dict) and type(ack.get("ack")) is int:
                stream.acknowledge(ack["ack"])
    except _SOCKET_ERRORS:
        return 1006


def _http_scope(socket, authorization):
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
    return scope


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
        key = options.get("stream")
        offset = options.get("offset", 0)
        if (
            (authorization is not None and not isinstance(authorization, str))
            or (key is not None and not (isinstance(key, str) and _STREAM_ID.fullmatch(key)))
            or type(offset) is not int or offset < 0
        ):
            await socket.close(code=1008)
            return
        target = (socket.url.path, socket.url.query)
        stream = _STREAMS.get(key) if key else None
        if stream is None:
            if offset:
                # The stream already ended or its resume window passed.
                await socket.close(code=4404)
                return
            stream = _Stream(key, target)
            if key:
                _STREAMS[key] = stream
            stream.task = asyncio.create_task(stream.run(app, _http_scope(socket, authorization)))
        elif stream.target != target or not stream.base <= offset <= stream.end:
            await socket.close(code=1008)
            return
        stream.attach(socket)
        delivery = asyncio.create_task(stream.deliver(socket, offset))
        listener = asyncio.create_task(_listen(stream, socket))
        tasks = {delivery, listener}
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        code = delivery.result() if delivery in done else None
        if code is not None:
            stream.close()
            await socket.close(code=code)
        elif stream.socket is not socket:
            await socket.close(code=4000)  # resumed on a newer socket
        else:
            dropped = listener.result() if listener in done else 1006
            # Only the browser's own close (1000) ends the stream; a proxy drop can resume.
            stream.detach(socket, resumable=key is not None and dropped != 1000)
    except _SOCKET_ERRORS:
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
