"""The served Gradio bundle re-sends only loading statuses that changed."""

import asyncio
import importlib.resources
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from gradio.ranged_response import OpenRange, RangedFileResponse
from starlette.exceptions import HTTPException

from h3_app.gradio_frontend import (
    LOADING_STATI_ORIGINAL, LOADING_STATI_PATCHED, install_gradio_frontend_patch, patched_bundle)
from tests.node_js import run_js

ASSETS = importlib.resources.files("gradio").joinpath("templates/frontend/assets")


def blocks_bundle():
    return next(item.name for item in ASSETS.iterdir()
                if item.name.startswith("Blocks-") and item.name.endswith(".js")
                and LOADING_STATI_ORIGINAL in item.read_text(encoding="utf-8"))


class GradioFrontendPatchTests(unittest.TestCase):
    def test_pinned_gradio_bundle_has_the_patched_code(self):
        # A Gradio upgrade that changes this code fails here instead of silently losing the fix.
        name = blocks_bundle()
        body = patched_bundle(name).decode("utf-8")
        self.assertNotIn(LOADING_STATI_ORIGINAL, body)
        self.assertEqual(body.count(LOADING_STATI_PATCHED), 1)

    def test_route_serves_the_patched_bundle_with_revalidation(self):
        app = FastAPI()
        install_gradio_frontend_patch(app)
        client = TestClient(app)
        name = blocks_bundle()
        response = client.get(f"/assets/{name}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("javascript", response.headers["content-type"])
        self.assertIn(LOADING_STATI_PATCHED, response.text)
        cached = client.get(f"/assets/{name}", headers={"if-none-match": response.headers["etag"]})
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/Blocks-missing.js").status_code, 404)
        self.assertEqual(client.get("/assets/Blocks-..%2Fevil.js").status_code, 404)

    def test_only_changed_statuses_are_sent(self):
        method = LOADING_STATI_PATCHED
        sent = run_js(f"""() => {{
            const manager = {{
                calls: [],
                loading_stati: {{current: {{1: {{status: "pending"}}, 2: {{status: "complete"}}}}}},
                update_state_cb(id, update) {{ this.calls.push([id, update.loading_status.status]); }},
                {method},
            }};
            manager.update_loading_stati_state();
            manager.loading_stati.current = {{1: {{status: "complete"}}, 2: {{status: "complete"}}, 3: {{status: "pending"}}}};
            manager.update_loading_stati_state();
            manager.update_loading_stati_state();
            return manager.calls;
        }}""")
        self.assertEqual(sent, [[1, "pending"], [2, "complete"], [1, "complete"], [3, "pending"]])


class GradioMediaRangeTests(unittest.IsolatedAsyncioTestCase):
    async def respond(self, path, byte_range, **kwargs):
        messages = []

        async def send(message):
            messages.append(message)

        async def receive():
            return {"type": "http.disconnect"}

        response = RangedFileResponse(path, byte_range, **kwargs)
        await asyncio.wait_for(response({}, receive, send), timeout=2)
        return messages

    async def test_range_beyond_eof_returns_only_available_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.mp4"
            path.write_bytes(b"0123456789")
            messages = await self.respond(path, OpenRange(3, 100_000))
        self.assertEqual(messages[0]["status"], 206)
        headers = dict(messages[0]["headers"])
        self.assertEqual(headers[b"content-length"], b"7")
        self.assertEqual(headers[b"content-range"], b"bytes 3-9/10")
        self.assertEqual(b"".join(m["body"] for m in messages[1:]), b"3456789")
        self.assertFalse(messages[-1]["more_body"])

    async def test_start_past_eof_is_rejected_including_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.mp4"
            for body in (b"", b"0123456789"):
                path.write_bytes(body)
                with self.assertRaises(HTTPException) as raised:
                    await self.respond(path, OpenRange(len(body), None))
                self.assertEqual(raised.exception.status_code, 416)
                self.assertEqual(raised.exception.headers["Content-Range"], f"bytes */{len(body)}")

    async def test_truncated_media_aborts_instead_of_looping_on_empty_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.mp4"
            path.write_bytes(b"x" * 8192)
            original_stat = path.stat()
            path.write_bytes(b"x" * 1024)
            with self.assertRaisesRegex(RuntimeError, "File ended"):
                await self.respond(path, OpenRange(0, 8191), stat_result=original_stat)


if __name__ == "__main__":
    unittest.main()
