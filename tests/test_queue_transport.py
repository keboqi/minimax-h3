"""Queue WebSocket routing preserves Gradio's HTTP auth and stream cleanup."""

import asyncio
from threading import Event
import unittest

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from h3_app.queue_transport import install_queue_transport


class QueueTransportTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        install_queue_transport(self.app)
        self.headers = {"origin": "http://testserver"}

    def test_streams_each_chunk_and_preserves_auth_and_query(self):
        observed = {}

        @self.app.get("/gradio_api/queue/data")
        async def stream(request: Request):
            observed.update(
                session=request.query_params["session_hash"],
                authorization=request.headers["authorization"],
                cookie=request.cookies["owner"],
                compressed="accept-encoding" in request.headers,
            )

            async def chunks():
                yield b'data: {"msg":"progress"}\n'
                await asyncio.sleep(0.01)
                yield b'\ndata: {"msg":"process_completed"}\n\n'

            return StreamingResponse(chunks(), media_type="text/event-stream")

        with TestClient(self.app) as client:
            with client.websocket_connect(
                "/gradio_api/queue/data?session_hash=fixture",
                headers={**self.headers, "cookie": "owner=browser-owner"},
            ) as socket:
                socket.send_json({"authorization": "Bearer fixture"})
                self.assertEqual(socket.receive_json()["status"], 200)
                self.assertEqual(socket.receive_bytes(), b'data: {"msg":"progress"}\n')
                self.assertEqual(socket.receive_bytes(), b'\ndata: {"msg":"process_completed"}\n\n')
                self.assertEqual(socket.receive()["code"], 1000)
        self.assertEqual(observed, {"session": "fixture", "authorization": "Bearer fixture", "cookie": "browser-owner", "compressed": False})

    def test_unauthorized_request_keeps_http_error_status(self):
        @self.app.get("/gradio_api/queue/data")
        async def stream():
            raise HTTPException(status_code=401, detail="Login required")

        with TestClient(self.app) as client:
            with client.websocket_connect("/gradio_api/queue/data?session_hash=fixture", headers=self.headers) as socket:
                socket.send_json({})
                self.assertEqual(socket.receive_json()["status"], 401)
                self.assertIn(b"Login required", socket.receive_bytes())

    def test_different_origin_cannot_read_browser_queue(self):
        with TestClient(self.app) as client:
            with self.assertRaises(WebSocketDisconnect) as error:
                with client.websocket_connect("/gradio_api/queue/data", headers={"origin": "https://other.example"}):
                    pass
            self.assertEqual(error.exception.code, 1008)

    def test_browser_disconnect_closes_the_http_stream(self):
        cleaned = Event()

        @self.app.get("/gradio_api/queue/data")
        async def stream():
            async def chunks():
                try:
                    yield b"data: first\n\n"
                    while True:
                        await asyncio.sleep(1)
                finally:
                    cleaned.set()

            return StreamingResponse(chunks(), media_type="text/event-stream")

        with TestClient(self.app) as client:
            with client.websocket_connect("/gradio_api/queue/data?session_hash=fixture", headers=self.headers) as socket:
                socket.send_json({})
                socket.receive_json()
                self.assertEqual(socket.receive_bytes(), b"data: first\n\n")
            self.assertTrue(cleaned.wait(timeout=2))

    def test_heartbeat_uses_the_same_transport(self):
        @self.app.get("/gradio_api/heartbeat/{session_hash}")
        async def heartbeat(session_hash: str):
            async def chunks():
                yield f"data: {session_hash}\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")

        with TestClient(self.app) as client:
            with client.websocket_connect("/gradio_api/heartbeat/fixture", headers=self.headers) as socket:
                socket.send_json({})
                socket.receive_json()
                self.assertEqual(socket.receive_bytes(), b"data: fixture\n\n")

    def test_transport_script_is_served_without_caching(self):
        with TestClient(self.app) as client:
            response = client.get("/h3/queue-transport.js")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("new WebSocket", response.text)
