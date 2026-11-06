"""Backend-free acceptance of the shared image composer and model dispatch.

Run with a local Chrome or Playwright Chromium installation.
"""

from contextlib import ExitStack
from functools import wraps
import inspect
import os
from pathlib import Path
import socket
import time
from threading import Thread
from tempfile import TemporaryDirectory

import requests
import uvicorn
from unittest.mock import patch

from PIL import Image
from playwright.sync_api import expect, sync_playwright

from h3_app.generation.results import GenerationUpdate
from h3_ui import application as app
from h3_ui.engines.fibo2 import Fibo2Handlers
from h3_ui.engines.flux3 import Flux3Handlers
from h3_ui.engines.moss_image import MossImageHandlers
from h3_ui.engines.qwen import QwenHandlers
from tests.browser_navigation import choose_engine


ARTIFACTS = Path(__file__).resolve().parents[1] / ".cache/ui-redesign/image-workspace"


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    captures = []
    fixture = ARTIFACTS / "result.png"
    Image.new("RGB", (512, 512), "#a5c9ba").save(fixture)

    def fake(original, family):
        @wraps(original)
        def generate(*args, **kwargs):
            values = dict(inspect.signature(original).bind(*args, **kwargs).arguments)
            values.pop("self", None)
            captures.append((family, values))
            yield GenerationUpdate(None, "Fixture preparing")
            time.sleep(1.2)
            yield GenerationUpdate([str(fixture)], "Fixture completed")
        return generate

    def h3_generate(request, *args, **kwargs):
        captures.append(("h3", request.values()))
        yield GenerationUpdate([str(fixture)], "Fixture completed")

    with ExitStack() as stack:
        state = stack.enter_context(TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {"H3_WORKSPACE_DIR": state}))
        stack.enter_context(patch.object(app, "backend_status", return_value="Image browser fixture"))
        for cls, method, family in (
            (QwenHandlers, "generate_qwen_image21", "qwen"),
            (Flux3Handlers, "generate_flux3", "flux3"),
            (MossImageHandlers, "generate_moss_image", "moss_image"),
            (Fibo2Handlers, "generate_fibo2", "fibo2"),
        ):
            stack.enter_context(patch.object(cls, method, fake(getattr(cls, method), family)))
        stack.enter_context(patch("h3_app.generation.ph3_image.generate_ph3_image", h3_generate))
        demo = app.build_ui()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            app.build_server(demo, [str(ARTIFACTS)]), host="127.0.0.1", port=port, log_level="warning"))
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            try:
                if requests.get(f"http://127.0.0.1:{port}", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)
        try:
            with sync_playwright() as playwright:
                chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                browser = playwright.chromium.launch(
                    executable_path=os.getenv("H3_BROWSER_EXECUTABLE") or (str(chrome) if chrome.exists() else None),
                    headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1080})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{port}")
                choose_engine(page, "Image", "Qwen Image 2.1")
                prompt = page.locator('textarea[placeholder="Describe the image you want to create."]:visible')
                prompt.fill("A paper bird")
                button = page.get_by_role("button", name="Generate image", exact=True)
                expect(button).to_be_enabled()
                button.click()
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "Qwen Image 2.1 · Fixture preparing", timeout=20000)
                choose_engine(page, "Image", "FLUX 3 Image Klein")
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "Qwen Image 2.1 · Fixture completed", timeout=20000)
                assert captures[-1][0] == "qwen" and captures[-1][1]["prompt"] == "A paper bird"
                assert captures[-1][1]["reference_images"] == []
                expect(prompt).to_have_value("A paper bird")
                button.click()
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "FLUX 3 Image Klein · Fixture completed", timeout=20000)
                assert captures[-1][0] == "flux3"
                # A result stays available after a model switch and becomes the common edit input.
                page.locator("#h3-image-results button.thumbnail-item").first.click()
                page.get_by_role("button", name="Use selected result as edit input").click()
                edit_prompt = page.locator('textarea[placeholder="Describe what to change and what to preserve."]:visible')
                expect(edit_prompt).to_have_value("A paper bird")
                choose_engine(page, "Image", "MOSS-Image")
                expect(page.get_by_role("radio", name="Edit", exact=True)).to_have_count(0)
                expect(page.get_by_role("button", name="Edit image", exact=True)).to_have_count(0)
                expect(page.get_by_role("button", name="Use selected result as edit input")).to_have_count(0)
                expect(page.get_by_text("Reference images, in order", exact=True)).not_to_be_visible()
                expect(prompt).to_have_value("A paper bird")
                expect(page.get_by_role("button", name="Generate image", exact=True)).to_be_enabled()
                page.get_by_role("button", name="Generate image", exact=True).click()
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "MOSS-Image · Fixture completed", timeout=20000)
                assert captures[-1][0] == "moss_image"
                assert "reference_images" not in captures[-1][1]
                choose_engine(page, "Image", "Fibo 2.0")
                expect(page.get_by_label("Edit", exact=True)).to_be_checked()
                expect(page.get_by_role("button", name="Use selected result as edit input")).to_be_visible()
                page.get_by_role("button", name="Edit image", exact=True).click()
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "Fibo 2.0 · Fixture completed", timeout=20000)
                assert captures[-1][0] == "fibo2" and len(captures[-1][1]["reference_images"]) == 1
                choose_engine(page, "Image", "MiniMax H3")
                page.get_by_role("button", name="Edit image", exact=True).click()
                expect(page.get_by_label("Generation progress", exact=True).locator("visible=true")).to_have_value(
                    "MiniMax H3 · Fixture completed", timeout=20000)
                assert captures[-1][0] == "h3" and captures[-1][1]["ref_image_1"]
                page.screenshot(path=str(ARTIFACTS / "desktop.png"), full_page=True)
                page.get_by_text("Model settings", exact=True).click()
                page.get_by_role("button", name="Open Video → MiniMax H3 settings").click()
                expect(page.locator(".h3-task-picker").get_by_label("Video", exact=True)).to_be_checked()
                choose_engine(page, "Image", "Qwen Image 2.1")
                expect(edit_prompt).to_have_value("A paper bird")
                cache = page.get_by_label("Edit KV cache device", exact=True).locator("visible=true")
                expect(cache).to_be_enabled()
                page.get_by_label("Generate", exact=True).check()
                expect(cache).to_be_disabled()
                page.get_by_label("Edit", exact=True).check()
                expect(cache).to_be_enabled()
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2")
                page.get_by_text("Prompt assistance", exact=True).click()
                page.get_by_label("OpenAI-compatible", exact=True).locator("visible=true").check()
                page.get_by_label("Temporary OpenAI API key", exact=True).locator("visible=true").fill("fixture-secret")
                page.wait_for_timeout(600)
                page.reload()
                expect(page.locator(".h3-engine-bar").get_by_label("Qwen Image 2.1", exact=True)).to_be_checked(timeout=20000)
                page.get_by_text("Model settings", exact=True).click()
                expect(cache).to_be_enabled()
                page.get_by_text("Prompt assistance", exact=True).click()
                expect(page.get_by_label("OpenAI-compatible", exact=True).locator("visible=true")).to_be_checked()
                expect(page.get_by_label("Temporary OpenAI API key", exact=True).locator("visible=true")).to_have_value("")
                assert not errors, errors
                browser.close()
                print("Shared image browser acceptance passed; screenshots:", ARTIFACTS)
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            demo.close()


if __name__ == "__main__":
    run()
