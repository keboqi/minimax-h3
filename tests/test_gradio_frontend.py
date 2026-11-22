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
    CORE_FIND_ORIGINAL, CORE_FIND_PATCHED, CORE_SYNC_ORIGINAL, CORE_SYNC_PATCHED,
    HANDLE_DATA_ORIGINAL, HANDLE_DATA_PATCHED,
    LOADING_STATI_ORIGINAL, LOADING_STATI_PATCHED, install_gradio_frontend_patch, patched_bundle)
from tests.node_js import run_js

ASSETS = importlib.resources.files("gradio").joinpath("templates/frontend/assets")


def blocks_bundle():
    return next(item.name for item in ASSETS.iterdir()
                if item.name.startswith("Blocks-") and item.name.endswith(".js")
                and LOADING_STATI_ORIGINAL in item.read_text(encoding="utf-8"))


def core_bundle():
    return next(item.name for item in ASSETS.iterdir()
                if item.name.startswith("core-") and item.name.endswith(".js")
                and CORE_SYNC_ORIGINAL in item.read_text(encoding="utf-8"))


class GradioFrontendPatchTests(unittest.TestCase):
    def test_pinned_gradio_bundle_has_the_patched_code(self):
        # A Gradio upgrade that changes this code fails here instead of silently losing the fix.
        name = blocks_bundle()
        body = patched_bundle(name).decode("utf-8")
        self.assertNotIn(LOADING_STATI_ORIGINAL, body)
        self.assertEqual(body.count(LOADING_STATI_PATCHED), 1)
        self.assertNotIn(HANDLE_DATA_ORIGINAL, body)
        self.assertEqual(body.count(HANDLE_DATA_PATCHED), 1)
        core_name = core_bundle()
        core_body = patched_bundle(core_name).decode("utf-8")
        self.assertNotIn(CORE_SYNC_ORIGINAL, core_body)
        self.assertEqual(core_body.count(CORE_SYNC_PATCHED), 1)
        self.assertNotIn(CORE_FIND_ORIGINAL, core_body)
        self.assertEqual(core_body.count(CORE_FIND_PATCHED), 1)

    def test_route_serves_the_patched_bundle_with_revalidation(self):
        app = FastAPI()
        install_gradio_frontend_patch(app)
        client = TestClient(app)
        for name, expected in ((blocks_bundle(), LOADING_STATI_PATCHED), (core_bundle(), CORE_SYNC_PATCHED)):
            response = client.get(f"/assets/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("javascript", response.headers["content-type"])
            self.assertIn(expected, response.text)
            cached = client.get(f"/assets/{name}", headers={"if-none-match": response.headers["etag"]})
            self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/Blocks-missing.js").status_code, 404)
        self.assertEqual(client.get("/assets/Blocks-..%2Fevil.js").status_code, 404)
        self.assertEqual(client.get("/assets/core-missing.js").status_code, 404)

    def test_handle_data_batches_non_visible_properties_per_component(self):
        method = HANDLE_DATA_PATCHED.replace("await ", "").replace("async ", "")
        calls = run_js(f"""() => {{
            const qe = Symbol("skip");
            const Xe = (v) => typeof v === "object" && v !== null && v.__type__ === "update";
            const manager = {{
                calls: [],
                update_state_cb(id, update, isVisible) {{
                    this.calls.push([id, update, isVisible]);
                }},
                {method},
            }};
            manager.handle_data(
                [10, 20, 30],
                [
                    {{__type__: "update", choices: ["a", "b"], value: "b", interactive: true, visible: false}},
                    {{__type__: "update", visible: true}},
                    "plain",
                ]
            );
            return manager.calls;
        }}""")
        self.assertEqual(calls, [
            [10, {"choices": ["a", "b"], "value": "b", "interactive": True}, False],
            [10, {"visible": False}, True],
            [20, {"visible": True}, True],
            [30, {"value": "plain"}, False],
        ])

    def test_tree_lookup_caches_found_nodes_on_root(self):
        result = run_js(f"""() => {{
            {CORE_FIND_PATCHED}
            const leaf = {{id: 42, children: []}};
            const root = {{id: 0, children: [{{id: 1, children: [leaf]}}]}};
            const first = Z(root, 42);
            const cached = root._h3_nodes.get(42);
            const second = Z(root, 42);
            return [first === leaf, cached === leaf, second === leaf, Z(root, 99)];
        }}""")
        self.assertEqual(result, [True, True, True, None])

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
