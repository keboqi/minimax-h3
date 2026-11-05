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
                page.locator(".h3-task-picker label", has_text="Remake").click()
                readiness = page.locator(".h3-readiness:visible")
                expect(readiness).to_have_text("Add a source video to start.")
                expect(page.get_by_role("button", name="Generate remake")).to_be_disabled()
                next_run = page.locator('[aria-label="Next run"]:visible')
                expect(next_run).to_contain_text("Remake · Swap")
                expect(next_run).to_contain_text("Singularity")
                expect(next_run).to_contain_text("Untested pairing")

                page.locator('input[type="file"][accept*="video"]:visible, '
                             '.h3-composer:visible input[type="file"]').first.set_input_files(str(source))
                expect(page.get_by_text("Source audio found")).to_be_visible(timeout=30000)
                expect(page.get_by_text("frames at 24 fps")).to_be_visible()
                expect(readiness).to_contain_text("Extract a frame")

                page.get_by_role("button", name="Extract frame").click()
                expect(readiness).to_contain_text("Edit Picture 1", timeout=20000)
                expect(page.get_by_label("Picture 1 frame")).to_have_value("0")
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-extracted.png"), full_page=True)

                # The edit round trip's landing spot: Edit in Image opens that task.
                page.get_by_role("button", name="Edit in Image").first.click()
                expect(page.locator(".h3-task-picker label:has(input:checked)")).to_have_text("Image", timeout=10000)
                expect(page.locator(".h3-engine-bar label:has(input:checked):visible")).to_have_text(
                    "Qwen Image 2.1", timeout=10000)
                page.locator(".h3-task-picker label", has_text="Remake").click()
                expect(readiness).to_contain_text("Edit Picture 1")

                # An edit moved by a few pixels replaces the raw frame.
                image = Image.new("RGB", (1280, 720), (40, 60, 80))
                image.save(edited)
                slot = page.locator(".h3-remake-frames").locator('input[type="file"]').first
                page.locator(".h3-remake-frames").get_by_role("button", name=re.compile("clear|remove", re.I)).first.click()
                expect(slot).to_be_attached(timeout=10000)
                slot.set_input_files(str(edited))
                expect(readiness).to_contain_text("Rewrite your request", timeout=20000)

                page.get_by_label("Remake prompt").fill("summary:\nfree text")
                expect(readiness).to_contain_text("six-section", timeout=10000)
                expect(page.get_by_role("button", name="Generate remake")).to_be_enabled()

                page.get_by_text("Check alignment", exact=True).first.click()
                page.get_by_role("button", name="Check alignment").last.click()
                expect(page.get_by_text(re.compile("Can't measure this one"))).to_be_visible(timeout=20000)

                # A selection one window can hold shows no windows; a longer one does.
                expect(page.get_by_role("button", name="Split into windows")).to_be_hidden()
                page.get_by_label("Longest window").click()
                page.get_by_role("option", name="3.8 s (90 frames)").click()
                expect(page.get_by_role("button", name="Split into windows")).to_be_visible(timeout=10000)
                page.get_by_role("button", name="Split into windows").click()
                expect(page.get_by_text("windows", exact=False).filter(has_text="rendered").first).to_be_visible(
                    timeout=20000)
                # Window 1's prompt is enough: no edit request.
                page.get_by_label("Edit request").fill("")
                page.get_by_label("Remake prompt").fill("summary:\nA robot dances.")
                page.get_by_role("button", name="Generate remake").click()
                status = page.get_by_label("Generation progress").locator("visible=true").first
                expect(status).to_have_value(re.compile("Combined"), timeout=60000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-windows.png"), full_page=True)

                # The manual side: open a window, take its carried frame, regenerate it alone, combine.
                choices = page.get_by_label("Window", exact=True)
                choices.click()
                page.get_by_role("option", name=re.compile(r"^2 ")).click()
                expect(page.get_by_label("Remake prompt")).to_have_value(re.compile("Fixture window"), timeout=15000)
                expect(page.get_by_label("Picture 1 frame")).to_be_hidden()
                page.get_by_role("button", name="Copy carried frame to a slot").click()
                expect(page.get_by_label("Picture 1 frame")).to_be_visible(timeout=10000)
                expect(page.get_by_label("Picture 1 frame")).to_have_value(re.compile("[0-9]+"))
                expect(page.get_by_label("Use the carried frame")).not_to_be_checked()
                page.get_by_label("Remake prompt").fill("Edited window two.")
                page.get_by_role("button", name="Regenerate this window").click()
                wait_for_remake_jobs(url, 2)
                expect(status).to_have_value(re.compile("Combined"), timeout=30000)
                page.get_by_role("button", name="Combine final video").click()
                expect(status).to_have_value(re.compile("Combined"), timeout=30000)
                if shots:
                    page.screenshot(path=str(Path(shots) / "remake-window2.png"), full_page=True)

                page.get_by_role("button", name="Open MiniMax H3 settings →").click()
                expect(page.locator(".h3-task-picker label:has(input:checked)")).to_have_text("Video", timeout=10000)
                if shots:
                    page.locator(".h3-task-picker label", has_text="Remake").click()
                    expect(readiness).to_be_visible()
                    page.screenshot(path=str(Path(shots) / "remake-ready.png"), full_page=True)
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
