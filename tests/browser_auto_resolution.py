"""Browser acceptance check for H3 first-frame resolution with Output essentials closed."""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def run():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    source = (
        "import gradio_app as app;"
        "app.backend_status=lambda:'Connected browser test fixture';"
        "app.build_ui().queue(default_concurrency_limit=1,max_size=8).launch("
        f"server_name='127.0.0.1',server_port={port},inbrowser=False,ssr_mode=False,css=app.H3_SETUP_CSS)"
    )
    with tempfile.TemporaryFile(mode="w+b") as log, tempfile.TemporaryDirectory() as images:
        first = Path(images) / "first.png"
        second = Path(images) / "second.png"
        Image.new("RGB", (768, 1152)).save(first)
        Image.new("RGB", (640, 960)).save(second)
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            cwd=ROOT,
            stdout=log,
            stderr=log,
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
                    card = page.locator(".h3-setup-card")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                    decoder = card.locator(".h3-setup-detail").filter(has_text="Decoder")
                    expect(decoder).to_contain_text("INT8 ConvRot")
                    presets = page.locator("fieldset").filter(
                        has=page.get_by_text("Generation preset", exact=True)
                    )
                    presets.get_by_label("Quality", exact=True).check()
                    expect(decoder).to_contain_text("FP16")
                    presets.get_by_label("Fast", exact=True).check()
                    expect(decoder).to_contain_text("INT8 ConvRot")
                    presets.get_by_label("Singularity", exact=True).check()
                    expect(decoder).to_contain_text("INT8 ConvRot")
                    page.get_by_label("First / last frame", exact=True).check()
                    output = page.get_by_text("Output essentials", exact=True)
                    expect(output).to_be_visible()
                    file_input = page.locator('#first-frame-image input[type="file"]')
                    file_input.set_input_files(str(first))
                    expect(card).to_contain_text("768×1152", timeout=15000)
                    file_input.set_input_files(str(second))
                    expect(card).to_contain_text("640×960", timeout=15000)
                    page.wait_for_timeout(500)
                    page.reload(wait_until="domcontentloaded")
                    page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                    expect(card).to_contain_text("640×960", timeout=15000)
                finally:
                    browser.close()
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    run()
