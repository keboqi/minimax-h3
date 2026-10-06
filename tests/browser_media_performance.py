"""Persisted previews bypass scanning and hashing; selection keeps originals."""

import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import requests
from playwright.sync_api import expect, sync_playwright


def run():
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="h3-media-performance-") as directory:
        state = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "H3_WORKSPACE_DIR": str(state),
               "GRADIO_OUTPUT_DIR": str(state / "media"), "COMFY_DIR": str(state / "comfy")}
        code = f"""
import sys,runpy
from h3_ui import application as app
from gradio import processing_utils
from PIL import Image
original_new = Image.new
def full_resolution_fixture(mode, size, *args, **kwargs):
    return original_new(mode, (960,768) if size == (320,256) else size, *args, **kwargs)
Image.new = full_resolution_fixture
original_hash = processing_utils.hash_file
def forbid_scan(*args, **kwargs):
    raise AssertionError('Browsing must not scan the filesystem')
def forbid_thumbnail_hash(path, *args, **kwargs):
    if '.gallery_thumbnails' in str(path):
        raise AssertionError('Thumbnail must be served directly')
    return original_hash(path, *args, **kwargs)
app.gallery_image_paths = app.gallery_video_paths = app.gallery_audio_paths = forbid_scan
app.gallery_store.gallery_image_paths = app.gallery_store.gallery_video_paths = app.gallery_store.gallery_audio_paths = forbid_scan
processing_utils.hash_file = forbid_thumbnail_hash
sys.argv=['fixture','{port}']
runpy.run_module('tests.workspace_fixture',run_name='__main__')
"""
        with (state / "server.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", code], cwd=root, env=env,
                stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError((state / "server.log").read_text())
                    try:
                        if requests.get(url + "/test/jobs", timeout=1).ok:
                            break
                    except requests.RequestException:
                        pass
                    sleep(0.2)
                else:
                    raise TimeoutError("Media fixture unavailable")
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
                        started = monotonic()
                        page.get_by_role("tab", name="Media", exact=True).click()
                        thumbs = page.locator("#generated-video-gallery .thumbnail-item")
                        expect(thumbs).to_have_count(2, timeout=2500)
                        first_page = monotonic() - started
                        started = monotonic()
                        thumbs.filter(has_text="alpha.mp4").click()
                        player = page.locator(".h3-gallery-player video")
                        expect(player).to_have_attribute("src", url + "/downloads/gradio/alpha.mp4", timeout=1500)
                        selection = monotonic() - started
                        page.locator("#h3-library-kind").get_by_label("Image", exact=True).check()
                        expect(thumbs).to_have_count(2, timeout=2500)
                        expect(thumbs.first).to_contain_text(".png")
                        thumbs.filter(has_text="alpha.png").click()
                        original = page.locator('.h3-gallery-player img[src*="/downloads/"]').first
                        expect(original).to_be_visible(timeout=2000)
                        expect(original).to_have_js_property("naturalWidth", 960)
                        expect(original).to_have_js_property("naturalHeight", 768)
                        assert requests.get(original.get_attribute("src"), timeout=5).content == (state / "media" / "alpha.png").read_bytes()
                        expect(thumbs.first.locator("img")).to_have_attribute("src", __import__("re").compile(r"/media-previews/"))
                        page.get_by_role("button", name="Add to compare A", exact=True).click()
                        expect(page.locator("#h3-compare-a")).to_contain_text("alpha.png")
                        expect(page.locator("#h3-compare-a img")).to_have_attribute("src", __import__("re").compile(r"/media-previews/"))
                        # Wait beyond the old video decode: it must not replace images.
                        page.wait_for_timeout(4500)
                        expect(thumbs.first).to_contain_text(".png")
                        page.locator("#h3-library-kind").get_by_label("Video", exact=True).check()
                        page.locator("#h3-library-kind").get_by_label("Audio", exact=True).check()
                        expect(thumbs).to_have_count(0, timeout=2500)
                        expect(page.locator(".h3-gallery-status").first).to_contain_text("audio files")
                        print(f"Cold media page: {first_page:.3f}s; thumbnail selection: {selection:.3f}s; mode cancellation passed")
                    finally:
                        browser.close()
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    run()
