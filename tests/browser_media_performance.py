"""Cold poster generation must not block browsing, selection, or mode changes."""

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
        code = (
            "import sys,time,runpy; from h3_ui import application as app; "
            "original=app.gallery_thumbnail; "
            "app.gallery_thumbnail=lambda path: (time.sleep(4),original(path))[1]; "
            f"sys.argv=['fixture','{port}']; "
            "runpy.run_module('tests.workspace_fixture',run_name='__main__')"
        )
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
                        expect(page.locator(".h3-gallery-status").first).to_contain_text("Preparing")
                        started = monotonic()
                        thumbs.filter(has_text="alpha.mp4").click()
                        player = page.locator(".h3-gallery-player video")
                        expect(player).to_have_attribute("src", url + "/downloads/gradio/alpha.mp4", timeout=1500)
                        selection = monotonic() - started
                        page.locator("#h3-library-kind").get_by_label("Image", exact=True).check()
                        expect(thumbs).to_have_count(2, timeout=2500)
                        expect(thumbs.first).to_contain_text(".png")
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
