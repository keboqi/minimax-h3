"""Browser acceptance for library frames, reference slots and multi-image inputs."""

import os
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import time

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import requests


ROOT = Path(__file__).resolve().parents[1]


def run():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with TemporaryDirectory() as directory, TemporaryFile(mode="w+b") as log:
        output = Path(directory) / "outputs"
        output.mkdir()
        Image.new("RGB", (768, 1152), "red").save(output / "portrait.png")
        Image.new("RGB", (640, 960), "blue").save(output / "second.png")
        captured = Path(directory) / "captured.json"
        source = f"""
import json
import os
from pathlib import Path
import gradio as gr
from h3_ui import application as app
from h3_app.contracts import GENERATION_FIELDS
app.backend_status = lambda: 'Connected browser test fixture'
def generate(batch_count, *args):
    values = dict(zip(GENERATION_FIELDS, args, strict=True))
    Path(os.environ['H3_TEST_CAPTURE']).write_text(json.dumps(values), encoding='utf-8')
    yield (*(gr.skip() for _ in range(11)), 'Library generation fixture completed: ' + values['prompt'])
app.generate_for_ui = generate
demo = app.build_ui().queue()
for name in ('portrait.png', 'second.png'):
    app.write_snapshot(app.OUTPUTS_DIR / name, {{'family': 'Browser fixture'}})
import uvicorn
server = app.build_server(demo, [str(app.OUTPUT_DIR.resolve()), str(app.OUTPUTS_DIR.resolve())])
uvicorn.run(server, host='127.0.0.1', port={port}, log_level='warning')
"""
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source], cwd=ROOT,
            env={**os.environ, "GRADIO_OUTPUT_DIR": str(output),
                 "H3_WORKSPACE_DIR": str(Path(directory) / "workspace"),
                 "H3_TEST_CAPTURE": str(captured)},
            stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("UI fixture exited")
                try:
                    if requests.get(url + "/config", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("UI fixture did not start")
            with sync_playwright() as playwright:
                executable = os.getenv("H3_BROWSER_EXECUTABLE")
                chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                if not executable and chrome.exists():
                    executable = str(chrome)
                browser = playwright.chromium.launch(
                    headless=True,
                    **({"executable_path": executable} if executable else {}),
                )
                try:
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                    page.get_by_label("First / last frame", exact=True).check()
                    first = page.locator(".h3-image-library-input").filter(
                        has=page.locator("#first-frame-image")
                    )
                    first.get_by_text("Choose from media library", exact=True).click()
                    first.get_by_label("Search library images").fill("portrait.png")
                    first.get_by_role("button", name="Search / refresh images", exact=True).click()
                    expect(first.locator(".thumbnail-item")).to_have_count(1)
                    first.locator(".thumbnail-item").click()
                    expect(page.locator(".h3-setup-metrics")).to_contain_text("768×1152", timeout=15000)
                    page.locator("#h3-composer").get_by_label("Prompt", exact=True).fill("A slow camera move.")
                    generate = page.get_by_role("button", name="Generate video", exact=True)
                    expect(generate).to_be_enabled()
                    generate.click()
                    expect(page.get_by_label("Generation progress", exact=True).first).to_have_value("Library generation fixture completed: A slow camera move.", timeout=15000)
                    values = json.loads(captured.read_text(encoding="utf-8"))
                    assert (values["width"], values["height"]) == (768, 1152), values
                    assert values["first_image"] and values["prompt"] == "A slow camera move.", values
                    first.get_by_label("Search library images").fill("second.png")
                    first.get_by_role("button", name="Search / refresh images", exact=True).click()
                    expect(first.locator(".thumbnail-item")).to_have_count(1)
                    first.locator(".thumbnail-item").click()
                    expect(page.locator(".h3-setup-metrics")).to_contain_text("640×960", timeout=15000)
                    page.locator("#h3-composer").get_by_label("Prompt", exact=True).fill("A replacement frame.")
                    expect(generate).to_be_enabled()
                    generate.click()
                    expect(page.get_by_label("Generation progress", exact=True).first).to_have_value("Library generation fixture completed: A replacement frame.", timeout=15000)
                    values = json.loads(captured.read_text(encoding="utf-8"))
                    assert (values["width"], values["height"]) == (640, 960), values
                    assert values["first_image"] and values["prompt"] == "A replacement frame.", values
                    last = page.locator(".h3-image-library-input").filter(
                        has=page.get_by_text("Last frame", exact=True)
                    )
                    last.get_by_text("Choose from media library", exact=True).click()
                    last.get_by_label("Search library images").fill("second.png")
                    last.get_by_role("button", name="Search / refresh images", exact=True).click()
                    expect(last.locator(".thumbnail-item")).to_have_count(1)
                    last.locator(".thumbnail-item").click()
                    expect(last.locator("img").first).to_be_visible()
                    page.get_by_label("Reference media", exact=True).check()
                    page.get_by_role("button", name="Add image reference", exact=True).click()
                    picture = page.locator(".h3-image-library-input").filter(
                        has=page.get_by_text("Picture 2", exact=True)
                    )
                    expect(picture).to_be_visible()
                    picture.get_by_text("Choose from media library", exact=True).click()
                    picture.get_by_label("Search library images").fill("portrait.png")
                    picture.get_by_role("button", name="Search / refresh images", exact=True).click()
                    expect(picture.locator(".thumbnail-item")).to_have_count(1)
                    picture.locator(".thumbnail-item").click()
                    expect(picture.locator("img").first).to_be_visible()
                    page.locator(".h3-task-picker").get_by_label("Image", exact=True).check()
                    page.get_by_label("Engine", exact=True).click()
                    page.get_by_role("option", name="Qwen Image 2.1", exact=True).click()
                    page.get_by_text("Reference images and editing", exact=True).click()
                    inputs = page.locator(".h3-image-library-input").filter(
                        has=page.get_by_text("Input / reference images", exact=True)
                    )
                    inputs.get_by_text("Choose from media library", exact=True).click()
                    for filename in ("portrait.png", "second.png"):
                        inputs.get_by_label("Search library images").fill(filename)
                        inputs.get_by_role("button", name="Search / refresh images", exact=True).click()
                        expect(inputs.locator(".thumbnail-item")).to_have_count(1)
                        inputs.locator(".thumbnail-item").click()
                        expect(inputs.get_by_role("cell", name=filename, exact=True)).to_be_visible()
                    expect(inputs.get_by_role("cell", name="portrait.png", exact=True)).to_be_visible()
                    expect(inputs.get_by_role("row")).to_have_count(2)
                    page.get_by_role("tab", name="Media", exact=True).click()
                    media = page.locator(".h3-gallery-shell")
                    media.locator("#h3-library-kind").get_by_label("Image", exact=True).check()
                    expect(media.get_by_role("button", name="Set image", exact=True)).to_be_disabled()
                    media.get_by_label("Search media", exact=True).fill("second.png")
                    media.get_by_role("button", name="Search library", exact=True).click()
                    expect(media.locator("#generated-video-gallery .thumbnail-item")).to_have_count(1)
                    media.locator("#generated-video-gallery .thumbnail-item").click()
                    expect(media.get_by_role("button", name="Set image", exact=True)).to_be_enabled()

                    def set_as(name):
                        media.get_by_label("Set as", exact=True).click()
                        page.get_by_role("option", name=name, exact=True).click()
                        media.get_by_role("button", name="Set image", exact=True).click()

                    set_as("H3 · First frame (auto resolution)")
                    expect(media.get_by_text("Set image as H3 · First frame (auto resolution). Open Create to use it.", exact=True)).to_be_visible()
                    set_as("H3 · Reference images")
                    expect(media.get_by_text("Added image to H3 · Reference images · Picture 1. Open Create to use it.", exact=True)).to_be_visible()
                    media.get_by_role("button", name="Set image", exact=True).click()
                    expect(media.get_by_text("Added image to H3 · Reference images · Picture 3. Open Create to use it.", exact=True)).to_be_visible()
                    set_as("Qwen · Input / reference images")
                    expect(media.get_by_text("Added image to Qwen · Input / reference images. Open Create to use it.", exact=True)).to_be_visible()
                    page.get_by_role("tab", name="Create", exact=True).click()
                    expect(inputs.get_by_role("row")).to_have_count(3)
                    page.locator(".h3-task-picker").get_by_label("Video", exact=True).check()
                    page.get_by_label("Engine", exact=True).click()
                    page.get_by_role("option", name="MiniMax H3", exact=True).click()
                    page.get_by_label("First / last frame", exact=True).check()
                    expect(page.locator(".h3-setup-card")).to_contain_text("640×960", timeout=15000)
                    page.get_by_label("Reference media", exact=True).check()
                    picture3 = page.locator(".h3-image-library-input").filter(
                        has=page.get_by_text("Picture 3", exact=True)
                    )
                    expect(picture3).to_be_visible()
                    expect(picture3.locator("img").first).to_be_visible()
                    expect(picture.locator("img").first).to_be_visible()
                    log.seek(0)
                    assert b"Traceback" not in log.read(), "Server callback failed"
                finally:
                    browser.close()
        except Exception:
            log.seek(0)
            print(log.read().decode("utf-8", errors="replace")[-6000:])
            raise
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    run()
