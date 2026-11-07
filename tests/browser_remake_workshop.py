"""CPU Workshop acceptance: prepare, reopen, externally edit, and delete a Remake."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import requests


SERVER = """
from pathlib import Path
import sys
from h3_ui.media_console import media_runtime
from h3_app.workspace_store import WorkspaceStore
from h3_workshop import build_ui, _server
data = Path(sys.argv[1])
runtime = media_runtime(data, Path.cwd())
for folder in (runtime.input_dir, runtime.output_dir, runtime.outputs_dir):
    folder.mkdir(parents=True, exist_ok=True)
ui = build_ui(None, (runtime, WorkspaceStore(data / 'state')))
_server(ui, runtime, host='127.0.0.1', port=int(sys.argv[2])).run()
"""


def main():
    with tempfile.TemporaryDirectory(prefix="remake-workshop-") as folder, socket.socket() as sock:
        root = Path(folder)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        source, edited = root / "source.mp4", root / "edited.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=320x192:rate=24:duration=6", str(source)], check=True)
        Image.new("RGB", (320, 192), "orange").save(edited)
        process = subprocess.Popen([sys.executable, "-c", SERVER, str(root / "data"), str(port)],
                                   env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"})
        try:
            url = f"http://127.0.0.1:{port}"
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("Workshop exited before starting")
                try:
                    if requests.get(f"{url}/remake/", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("Workshop did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                page.get_by_role("tab", name="Remake", exact=True).click(timeout=60000)
                view = page.frame_locator(".h3-remake-frame")
                # The page binds its controls only after booting; an earlier file would be lost.
                expect(view.locator("#no-projects")).to_contain_text("Save any selection")
                view.locator("#video-input").set_input_files(str(source))
                expect(view.locator("#workshop-note")).to_be_visible()
                expect(view.locator("#primary")).to_be_disabled()
                view.get_by_role("button", name="Extract frame", exact=True).click()
                expect(view.get_by_role("button", name="Quick edit", exact=True)).to_have_count(0)
                expect(view.get_by_role("button", name="Edit in Image", exact=True)).to_have_count(0)
                with page.expect_download() as downloaded:
                    view.get_by_role("link", name="Download frame", exact=True).click()
                assert downloaded.value.suggested_filename == "frame-1.png"
                view.get_by_role("button", name="Save project", exact=True).click()
                expect(view.locator("#toast")).to_contain_text("Project saved")
                view.locator("#project-button").click()
                view.get_by_role("button", name="New from a video", exact=True).click()
                expect(view.locator("#start")).to_be_visible()
                view.locator(".project-row").first.click()
                expect(view.locator("#readiness")).to_contain_text("Edit Picture 1")
                # Use the replacement control, keeping the extracted frame's pin.
                with page.expect_file_chooser() as chooser:
                    view.get_by_role("button", name="Upload edited frame", exact=True).click()
                chooser.value.set_files(str(edited))
                expect(view.locator("#readiness")).not_to_contain_text("Edit Picture 1")
                expect(view.get_by_role("spinbutton", name="Picture 1 frame", exact=True)).to_have_value("0")
                view.get_by_role("button", name="Save project", exact=True).click()
                expect(view.locator("#toast")).to_contain_text("Project saved")
                # A save that keeps failing must not trap anyone in the project.
                page.route("**/remake/api/save", lambda route: route.abort())
                view.locator("#prompt").fill("not saved")
                view.locator("#prompt").blur()
                view.locator("#project-button").click()
                view.get_by_role("button", name="New from a video", exact=True).click()
                expect(view.locator("#start")).to_be_visible()
                expect(view.locator("#toast")).to_contain_text("weren't saved")
                view.locator(".project-row").first.click()
                expect(view.locator("#prompt")).to_be_visible()
                view.locator("#prompt").fill("not saved either")
                view.locator("#prompt").blur()
                view.locator("#project-button").click()
                view.get_by_role("button", name="Delete", exact=True).click()
                view.get_by_role("button", name="Delete project", exact=True).click()
                expect(view.locator("#start")).to_be_visible()
                expect(view.locator(".project-row")).to_have_count(0)
                assert not errors, errors
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=20)


if __name__ == "__main__":
    main()
