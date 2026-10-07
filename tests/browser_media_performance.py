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
                        # Ordinary catalog pages, including empty media types, must
                        # arrive in one direct callback without a queue stream.
                        config = requests.get(url + "/config", timeout=5).json()
                        grid_id = next(component["id"] for component in config["components"]
                                       if component["props"].get("elem_id") == "generated-video-gallery")
                        page_events = {event["id"] for event in config["dependencies"]
                                       if grid_id in event["outputs"]}
                        page_requests = []

                        def record_page_request(request):
                            if request.method != "POST" or "/gradio_api/" not in request.url:
                                return
                            data = request.post_data_json
                            if isinstance(data, dict) and data.get("fn_index") in page_events:
                                page_requests.append(request.url)

                        page.on("request", record_page_request)
                        switches = []
                        for mode, kind, extension, count in (
                            ("Audio", "audio files", None, 0),
                            ("Video", "videos", ".mp4", 2),
                            ("Image", "images", ".png", 2),
                        ):
                            page_requests.clear()
                            started = monotonic()
                            page.locator("#h3-library-kind").get_by_label(mode, exact=True).check()
                            expect(page.locator(".h3-gallery-status").first).to_contain_text(
                                f"generated {kind}", timeout=2000)
                            expect(thumbs).to_have_count(count, timeout=2000)
                            if extension:
                                expect(thumbs.first).to_contain_text(extension)
                            elapsed = monotonic() - started
                            assert len(page_requests) == 1, page_requests
                            assert "/queue/join" not in page_requests[0], page_requests
                            switches.append(f"{mode}: {elapsed:.3f}s")
                        # An older response must not overwrite the last selected type.
                        page.locator("#h3-library-kind").get_by_label("Video", exact=True).check()
                        page.locator("#h3-library-kind").get_by_label("Audio", exact=True).check()
                        expect(thumbs).to_have_count(0, timeout=2500)
                        expect(page.locator(".h3-gallery-status").first).to_contain_text("audio files")
                        # Search and type changes share the same latest-request
                        # handling, so a previous search cannot replace a new type.
                        search = page.get_by_label("Search media", exact=True)
                        search.fill("no-matching-media")
                        search.press("Enter")
                        page.locator("#h3-library-kind").get_by_label("Image", exact=True).check()
                        page.locator("#h3-library-kind").get_by_label("Video", exact=True).check()
                        expect(page.locator(".h3-gallery-status").first).to_contain_text("generated videos")
                        expect(thumbs).to_have_count(0)
                        search.fill("")
                        page.get_by_role("button", name="Search library", exact=True).click()
                        expect(thumbs).to_have_count(2)
                        expect(thumbs.first).to_contain_text(".mp4")
                        print(f"Cold media page: {first_page:.3f}s; thumbnail selection: {selection:.3f}s; "
                              f"media switches: {', '.join(switches)}; latest mode preserved")
                    finally:
                        browser.close()
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    run()
