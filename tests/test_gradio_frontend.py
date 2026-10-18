"""The served Gradio bundle re-sends only loading statuses that changed."""

import importlib.resources
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

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


if __name__ == "__main__":
    unittest.main()
