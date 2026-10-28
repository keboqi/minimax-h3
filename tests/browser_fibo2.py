"""Focused browser smoke for Fibo controls and Spectrum preference; no GPU calls."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from tests.browser_navigation import choose_engine, engine_pill


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".cache/fibo2-browser"
SERVER = '''
import inspect, runpy
import gradio as gr
from h3_ui.engines.fibo2 import Fibo2Handlers
from h3_ui.engines.moss_image import MossImageHandlers
def fake(original):
    def generate(*args, **kwargs):
        yield gr.skip(), "Image integration fixture complete"
    generate.__signature__ = inspect.signature(original)
    return generate
Fibo2Handlers.generate_fibo2 = fake(Fibo2Handlers.generate_fibo2)
MossImageHandlers.generate_moss_image = fake(MossImageHandlers.generate_moss_image)
runpy.run_module("tests.workspace_fixture", run_name="__main__")
'''


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="h3-fibo-browser-") as temp:
        root = Path(temp)
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "GRADIO_ANALYTICS_ENABLED": "False",
               "H3_WORKSPACE_DIR": str(root / "workspace"), "GRADIO_OUTPUT_DIR": str(root / "media"),
               "COMFY_DIR": str(root / "comfy")}
        reference = root / "reference.png"
        Image.new("RGB", (64, 64), "red").save(reference)
        with (ARTIFACTS / "server.log").open("wb") as log:
            server = subprocess.Popen([sys.executable, "-c", SERVER, str(port)], cwd=ROOT, env=env,
                stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError((ARTIFACTS / "server.log").read_text())
                    try:
                        if requests.get(url + "/test/jobs", timeout=1).ok:
                            break
                    except requests.ConnectionError:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError("Browser fixture did not start")
                with sync_playwright() as playwright:
                    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                    executable = os.getenv("H3_BROWSER_EXECUTABLE") or (str(chrome) if chrome.exists() else None)
                    browser = playwright.chromium.launch(headless=True, **({"executable_path": executable} if executable else {}))
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url, wait_until="domcontentloaded")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for(state="attached")
                    choose_engine(page, "Image", "Fibo 2.0")
                    view = page.locator('#h3-engine-tabs > [role="tabpanel"]:visible')
                    view.get_by_label("Fibo model", exact=True).click()
                    page.get_by_role("option", name="Edit Turbo", exact=True).click()
                    view.get_by_text("Sampling", exact=True).click()
                    steps = view.get_by_role("spinbutton", name="number input for Steps", exact=True)
                    expect(steps).to_have_value("4")
                    expect(steps).to_be_disabled()
                    expect(view.get_by_role("spinbutton", name="number input for CFG", exact=True)).to_have_value("1")
                    view.locator('input[type="file"]').first.set_input_files(str(reference))
                    view.get_by_label("Image prompt / edit instruction / VGL JSON").fill("Make the square blue")
                    view.get_by_role("button", name="Generate / edit image", exact=True).click()
                    expect(view.get_by_label("Generation progress", exact=True)).to_have_value("Image integration fixture complete")
                    page.screenshot(path=str(ARTIFACTS / "fibo2.png"), full_page=True)
                    jobs = requests.get(url + "/test/jobs", timeout=5).json()
                    job = next(job for job in jobs if job["family"] == "fibo2")
                    assert job["values"][1] == "Edit Turbo" and job["values"][8:10] == [4, 1]
                    engine_pill(page, "MOSS-Image").check()
                    accelerate = view.get_by_label("Accelerate with Spectrum")
                    expect(accelerate).to_be_checked()
                    saved = page.evaluate("JSON.stringify(localStorage)")
                    accelerate.uncheck()
                    page.wait_for_function("before => JSON.stringify(localStorage) !== before", arg=saved)
                    page.reload(wait_until="domcontentloaded")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for(state="attached")
                    expect(view.get_by_label("Accelerate with Spectrum")).not_to_be_checked()
                    assert not errors, errors
                    browser.close()
                print("Fibo model defaults, Turbo job capture, and Spectrum default/persistence passed.")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    run()
