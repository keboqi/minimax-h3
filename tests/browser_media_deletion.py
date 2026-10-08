"""Delete selected and empty library through the Media confirmation flow."""

import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep

import requests
from playwright.sync_api import expect, sync_playwright


def run():
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="h3-media-deletion-") as directory:
        state = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {
            **os.environ, "HF_HUB_OFFLINE": "1", "H3_WORKSPACE_DIR": str(state / "workspace"),
            "GRADIO_OUTPUT_DIR": str(state / "media"), "COMFY_DIR": str(state / "comfy"),
        }
        # Add playable audio to the ordinary backend-free browser fixture.
        code = """
import runpy, sys, wave
from h3_ui import application as app
original_server = app.build_server
def build_server(*args, **kwargs):
    for name in ('alpha.wav', 'beta.wav'):
        source = app.OUTPUTS_DIR / name
        with wave.open(str(source), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b'\\0' * 16000)
        app.write_snapshot(source, {'family': 'Audio fixture'})
    return original_server(*args, **kwargs)
app.build_server = build_server
runpy.run_module('tests.workspace_fixture', run_name='__main__')
"""
        with (state / "server.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", code, str(port)], cwd=root, env=env,
                stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(200):
                    if process.poll() is not None:
                        raise RuntimeError((state / "server.log").read_text())
                    try:
                        if requests.get(url + "/test/jobs", timeout=1).ok:
                            break
                    except requests.RequestException:
                        pass
                    sleep(0.2)
                else:
                    raise TimeoutError("Media deletion fixture unavailable:\n" + (state / "server.log").read_text())
                with sync_playwright() as playwright:
                    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                    executable = os.getenv("H3_BROWSER_EXECUTABLE")
                    browser = playwright.chromium.launch(
                        headless=True,
                        **({"executable_path": executable or str(chrome)} if executable or chrome.exists() else {}),
                    )
                    try:
                        page = browser.new_page(viewport={"width": 1440, "height": 1000})
                        page.goto(url, wait_until="domcontentloaded")
                        page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                        page.get_by_role("tab", name="Media", exact=True).click()
                        thumbs = page.locator("#generated-video-gallery .thumbnail-item")
                        expect(thumbs).to_have_count(2)
                        page.get_by_text("Manage library", exact=True).click()
                        delete = page.get_by_role("button", name="Delete selected", exact=True)
                        confirm = page.get_by_role("button", name="Confirm permanent deletion", exact=True)
                        status = page.locator(".h3-gallery-status").first
                        for mode, extension in (("Video", ".mp4"), ("Image", ".png"), ("Audio", ".wav")):
                            page.locator("#h3-library-kind").get_by_label(mode, exact=True).check()
                            expect(thumbs).to_have_count(2)
                            thumbs.filter(has_text="alpha" + extension).click()
                            expect(delete).to_be_enabled()
                            source = state / "media" / ("alpha" + extension)
                            delete.click()
                            expect(confirm).to_be_visible()
                            page.get_by_role("button", name="Keep media", exact=True).click()
                            expect(confirm).to_be_hidden()
                            assert source.is_file(), source
                            delete.click()
                            confirm.click()
                            expect(confirm).to_be_hidden()
                            expect(thumbs).to_have_count(1)
                            expect(thumbs.first).to_contain_text("beta" + extension)
                            expect(status).to_contain_text("Deleted")
                            expect(delete).to_be_disabled()
                            assert not source.exists(), source
                            assert not source.with_name(source.name + ".settings.json").exists()
                            page.get_by_role("button", name="Empty generated library", exact=True).click()
                            expect(confirm).to_be_visible()
                            confirm.click()
                            expect(thumbs).to_have_count(0)
                            expect(status).to_contain_text("Deleted 1 generated")
                            assert not (state / "media" / ("beta" + extension)).exists()
                        # A fresh browsing session must read the updated catalog.
                        page.reload(wait_until="domcontentloaded")
                        page.locator('.h3-setup-card[data-settings-ready="true"]').wait_for()
                        page.get_by_role("tab", name="Media", exact=True).click()
                        expect(thumbs).to_have_count(0)
                        print("Confirmed selection and library deletion passed for video, image and audio.")
                    finally:
                        browser.close()
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    run()
