"""Browser acceptance for the Remake task, on the backend-free workspace fixture."""

import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import requests
from PIL import Image
from playwright.sync_api import expect, sync_playwright


def wait_for(url, process):
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError("The workspace fixture exited early.")
        try:
            if requests.get(url, timeout=1).ok:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError("The workspace fixture did not start.")


def wait_for_remake_jobs(url, count):
    """Wait until ``count`` Remake jobs have finished, so a stale status can't pass for this run."""
    for _ in range(120):
        jobs = [job for job in requests.get(f"{url}test/jobs", timeout=5).json() if job["family"] == "remake"]
        if len(jobs) == count and all(job["state"] == "completed" for job in jobs):
            return
        time.sleep(0.5)
    raise AssertionError(f"Expected {count} finished Remake jobs, found {jobs}")


def main(shots=None):
    with tempfile.TemporaryDirectory(prefix="h3-remake-browser-") as folder, socket.socket() as sock:
        root = Path(folder)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        source = root / "source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=6",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", str(source)],
            check=True,
        )
        edited = root / "edited.png"
        extra = root / "extra.png"
        Image.new("RGB", (640, 360), (200, 120, 40)).save(extra)
        env = {**os.environ, "H3_WORKSPACE_DIR": str(root / "state")}
        env.pop("H3_UI_LAYOUT", None)
        process = subprocess.Popen([sys.executable, "-m", "tests.workspace_fixture", str(port)], env=env)
        try:
            url = f"http://127.0.0.1:{port}/"
            wait_for(url, process)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)

                picker = page.locator(".h3-task-picker label")
                expect(picker.first).to_be_visible(timeout=60000)
                names = [text.strip() for text in picker.all_inner_texts()]
                assert names == ["Video", "Image", "Song", "Remake", "Storyboard", "Soundstage", "Dub & Translate"], names
                picker.filter(has_text="Remake").click()
                # The tab is a page of its own, in a frame.
                remake = page.frame_locator(".h3-remake-frame:visible")
                expect(remake.get_by_role("heading", name="Drop the video you want to change")).to_be_visible(
                    timeout=30000)
                remake.locator("#video-input").set_input_files(str(source))
                readiness = remake.locator("#readiness")
                expect(readiness).to_contain_text("Extract a frame", timeout=30000)
                expect(remake.locator("#summary")).to_contain_text("frames at 24 fps")
                primary = remake.locator("#primary")
                expect(primary).to_have_text("Generate remake")
                expect(primary).to_be_disabled()
                expect(remake.get_by_label("Next run")).to_contain_text("Remake Swap")
                expect(remake.get_by_label("Next run")).to_contain_text("Singularity")
                expect(remake.get_by_label("Next run")).to_contain_text("Untested pairing")

                remake.get_by_role("button", name="Extract frame").click()
                expect(readiness).to_contain_text("Edit Picture 1", timeout=20000)
                expect(remake.get_by_label("Picture 1 frame")).to_have_value("0")
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-extracted.png"), full_page=True)

                # Qwen Image 2.1 is the default.
                expect(remake.get_by_label("Edit with")).to_have_value("qwen")

                # The edit round trip: MiniMax H3 edits on the frame as its first frame.
                remake.get_by_label("Edit with").select_option(label="MiniMax H3")
                remake.get_by_role("button", name="Edit in Image").first.click()
                expect(page.locator(".h3-task-picker label:has(input:checked)")).to_have_text("Image", timeout=10000)
                expect(page.locator(".h3-engine-bar label:has(input:checked):visible")).to_have_text(
                    "MiniMax H3", timeout=10000)
                expect(page.get_by_label("First / last frame")).to_be_checked(timeout=10000)
                # The edit keeps the frame's size, not the start-frame cap's.
                # The frame shows, and the edit keeps its size on the grid, not the start-frame cap's.
                expect(page.locator("#first-frame-image img").first).to_be_visible(timeout=10000)
                expect(page.get_by_text("1280×704", exact=True).locator("visible=true").first).to_be_visible(
                    timeout=10000)
                picker.filter(has_text="Remake").click()

                # Any other model still opens with the frame.
                remake.get_by_label("Edit with").select_option(label="Qwen Image 2.1")
                remake.get_by_role("button", name="Edit in Image").first.click()
                expect(page.locator(".h3-task-picker label:has(input:checked)")).to_have_text("Image", timeout=10000)
                expect(page.locator(".h3-engine-bar label:has(input:checked):visible")).to_have_text(
                    "Qwen Image 2.1", timeout=10000)
                # It opens ready to edit, with the frame itself, not just the task.
                expect(page.locator(".h3-engine-view:visible").get_by_label("Mode", exact=True)).to_have_value(
                    "Image edit", timeout=10000)
                page.locator(".h3-engine-view:visible").get_by_text("Reference images and editing").click()
                references = page.locator(".h3-engine-view:visible .h3-image-library-input").filter(
                    has=page.locator(".h3-input-thumbnails"))
                expect(references.locator(".h3-input-thumbnails img").first).to_be_visible(timeout=10000)
                expect(references.locator(".h3-input-files tbody tr")).to_have_count(1, timeout=10000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "qwen-thumbnails.png"), full_page=True)
                # Adding one and clicking a thumbnail away both reach the input the request reads.
                references.locator('.h3-input-files input[type="file"]').first.set_input_files(str(extra))
                expect(references.locator(".h3-input-files tbody tr")).to_have_count(2, timeout=10000)
                expect(references.locator(".h3-input-thumbnails img")).to_have_count(2, timeout=10000)
                references.locator(".h3-input-thumbnails .thumbnail-item").first.click()
                expect(references.locator(".h3-input-files tbody tr")).to_have_count(1, timeout=10000)
                picker.filter(has_text="Remake").click()
                expect(readiness).to_contain_text("Edit Picture 1")

                # An edit replaces the raw frame.
                Image.new("RGB", (1280, 720), (40, 60, 80)).save(edited)
                # A quick edit repaints the picture where it is, and can be undone.
                remake.get_by_label("Quick edit for picture 1").fill("Make the sky orange.")
                remake.get_by_role("button", name="Quick edit").click()
                expect(readiness).to_contain_text("Rewrite your request", timeout=30000)
                remake.get_by_role("button", name="Undo edit").click()
                expect(readiness).to_contain_text("Edit Picture 1", timeout=10000)

                remake.get_by_label("Remove picture 1").click()
                expect(readiness).to_contain_text("Extract a frame", timeout=10000)
                remake.locator("#picture-input").set_input_files(str(edited))
                expect(readiness).to_contain_text("Rewrite your request", timeout=20000)

                remake.get_by_label("Remake prompt").fill("summary:\nfree text")
                expect(readiness).to_contain_text("six-section", timeout=10000)
                expect(primary).to_be_enabled()

                remake.get_by_label("Picture 1 frame").fill("0")
                remake.get_by_label("Picture 1 frame").press("Tab")
                remake.get_by_role("button", name="Check alignment").click()
                expect(remake.locator("#align-note")).to_contain_text("Can't measure this one", timeout=20000)
                remake.get_by_role("button", name="Source", exact=True).click()

                # A selection one window can hold shows no windows; a longer one does.
                expect(remake.get_by_role("button", name="Split into windows")).to_be_hidden()
                remake.get_by_label("Longest window").select_option(label="3.8 s (90 frames)")
                split = remake.get_by_role("button", name="Split into windows")
                expect(split).to_be_visible(timeout=10000)
                split.click()
                expect(remake.locator("#summary")).to_contain_text("rendered", timeout=20000)
                # Window 1's prompt is enough: no edit request.
                remake.get_by_label("Remake prompt").fill("summary:\nA robot dances.")
                expect(primary).to_be_enabled(timeout=10000)
                primary.click()
                status = remake.get_by_label("Generation progress")
                expect(status).to_contain_text("Combined", timeout=60000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-windows.png"), full_page=True)

                expect(remake.get_by_label("Prompt per window")).to_be_checked()
                # The manual side: open a window, take its carried frame, regenerate it alone, combine.
                remake.get_by_role("button", name=re.compile("^Window 2,")).click()
                expect(remake.get_by_label("Remake prompt")).to_have_value(re.compile("Fixture window"), timeout=15000)
                expect(remake.get_by_label("Picture 1 frame")).to_be_hidden()
                remake.get_by_role("button", name="Copy to a slot").click()
                expect(remake.get_by_label("Picture 1 frame")).to_be_visible(timeout=10000)
                expect(remake.get_by_label("Picture 1 frame")).to_have_value(re.compile("[0-9]+"))
                expect(remake.get_by_label("Use the carried frame")).not_to_be_checked()
                remake.get_by_label("Remake prompt").fill("Edited window two.")
                regenerate = remake.get_by_role("button", name="Regenerate this window")
                expect(regenerate).to_be_enabled(timeout=10000)
                regenerate.click()
                wait_for_remake_jobs(url, 2)
                expect(status).to_contain_text("Combined", timeout=30000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-window2.png"), full_page=True)

                # A project reopens after a reload, with nothing to upload and nothing to render again.
                page.reload()
                expect(picker.first).to_be_visible(timeout=60000)
                # An edit comes back from Media, even before Remake has been opened.
                page.get_by_role("tab", name="Media", exact=True).click()
                media = page.locator(".h3-gallery-shell")
                media.locator("#h3-library-kind").get_by_label("Image", exact=True).check()
                media.get_by_label("Search media", exact=True).fill("alpha.png")
                media.get_by_role("button", name="Search library", exact=True).click()
                expect(media.locator("#generated-video-gallery .thumbnail-item")).to_have_count(1)
                media.locator("#generated-video-gallery .thumbnail-item").click()
                media.get_by_label("Set as", exact=True).click()
                page.get_by_role("option", name="Remake · Picture 2", exact=True).click()
                media.get_by_role("button", name="Set image", exact=True).click()
                picker.filter(has_text="Remake").click()
                remake.locator(".project-row", has_text="source").first.click()
                expect(remake.locator("#summary")).to_contain_text("rendered", timeout=20000)
                expect(remake.locator(".pic strong", has_text="Picture 2")).to_be_visible(timeout=20000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-loaded.png"), full_page=True)
                remake.get_by_role("button", name=re.compile("^Window 2,")).click()
                expect(remake.get_by_role("button", name="Result", exact=True)).to_be_visible()

                remake.get_by_text("What follows MiniMax H3").click()
                remake.get_by_role("button", name="Open MiniMax H3 settings").click()
                expect(page.locator(".h3-task-picker label:has(input:checked)")).to_have_text("Video", timeout=10000)
                if shots:
                    picker.filter(has_text="Remake").click()
                    page.set_viewport_size({"width": 390, "height": 844})
                    page.wait_for_timeout(600)
                    overflow = page.evaluate("document.documentElement.scrollWidth - innerWidth")
                    assert overflow <= 1, overflow
                    page.screenshot(path=str(Path(shots) / "remake-phone.png"), full_page=True)
                assert not errors, errors
                browser.close()
        finally:
            process.terminate()
            process.wait(timeout=20)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
