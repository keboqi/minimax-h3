"""Verify the Storyboard page in a real browser: instant switching, saving and a GPU run."""

import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from playwright.sync_api import expect, sync_playwright
import requests

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".cache" / "storyboard-page"
# What a switch may take: no network, so this is the browser's own paint and little else.
SWITCH_LIMIT_MS = 100


def run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS / "server.log"
    with TemporaryDirectory(prefix="storyboard-page-", ignore_cleanup_errors=True) as directory:
        with socket.socket() as socket_port:
            socket_port.bind(("127.0.0.1", 0))
            port = socket_port.getsockname()[1]
        with log_path.open("wb") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "tests.storyboard_fixture", str(port), directory],
                                       cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE="1"), stdout=log, stderr=log)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError(log_path.read_text(errors="replace"))
                    try:
                        if requests.get(url + "/test/ready", timeout=1).ok:
                            break
                    except requests.RequestException:
                        time.sleep(0.2)
                else:
                    raise TimeoutError("Storyboard fixture unavailable")
                with sync_playwright() as playwright:
                    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
                    browser = playwright.chromium.launch(
                        headless=True, **({"executable_path": str(chrome)} if chrome.exists() else {}))
                    check(browser.new_context(viewport={"width": 1440, "height": 1000}), url)
                    browser.close()
                print("PASS: scene switching makes no server request, edits save alone, a run survives a reload, "
                      "and takes, interrupt and cast work")
            except BaseException:
                print(log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
                raise
            finally:
                process.terminate()
                process.wait(timeout=10)


def check(context, url):
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url + "/storyboard/", wait_until="domcontentloaded")
    # A first visit with no storyboards opens on the start screen, with a live scene preview.
    story = page.get_by_label("Story", exact=True)
    story.fill("---8---\n\nOne.\n\n---\n\nTwo.")
    expect(page.get_by_role("heading", name="2 scenes")).to_be_visible()
    page.get_by_label("Name", exact=True).fill("Started here")
    page.get_by_role("button", name="Create storyboard").click()
    expect(page.locator("#scene-heading")).to_contain_text("Scene 1")
    assert context.request.post(url + "/test/seed").ok

    # The tab shows the page in a frame.
    page.goto(url, wait_until="domcontentloaded")
    frame = page.frame_locator("iframe.h3-storyboard-frame")
    expect(frame.locator("#scene-heading")).to_be_visible(timeout=20000)

    page.goto(url + "/storyboard/", wait_until="domcontentloaded")
    picker = page.get_by_label("Storyboard", exact=True)
    picker.select_option(label="Board 0")
    expect(page.locator("#strip .frame")).to_have_count(8)
    page.wait_for_load_state("networkidle")
    page.screenshot(path=str(ARTIFACTS / "editor.png"))

    # Moving between scenes asks the server for nothing.
    api = []
    page.on("request", lambda request: api.append(request.url) if "/storyboard/api/" in request.url and "/thumb" not in request.url and "/file" not in request.url else None)
    timings = page.evaluate("""async () => {
      const heading = document.getElementById('scene-heading');
      const frames = [...document.querySelectorAll('#strip .frame')];
      const timings = [];
      for (const frame of [...frames, ...frames.reverse()]) {
        const start = performance.now();
        frame.click();
        await new Promise(requestAnimationFrame);
        timings.push(performance.now() - start);
        if (!heading.isConnected && !document.getElementById('scene-heading')) throw new Error('heading lost');
      }
      return timings;
    }""")
    slowest = max(timings)
    print(f"scene switch: {sum(timings) / len(timings):.1f} ms average, {slowest:.1f} ms slowest, {len(timings)} switches")
    assert slowest < SWITCH_LIMIT_MS, timings
    assert not api, api
    page.get_by_role("button", name=re.compile(r"Scene 3, ")).click()
    expect(page.locator("#scene-heading")).to_contain_text("Scene 3")

    # Typing stays local and saves one field, surviving a move to another scene.
    prompt = page.get_by_label("Prompt", exact=True)
    with page.expect_request(lambda request: request.method == "PATCH") as saved:
        prompt.fill("Typed in scene three.")
        page.get_by_role("button", name=re.compile(r"Scene 4, ")).click()
    assert saved.value.post_data_json == {"prompt": "Typed in scene three."}, saved.value.post_data
    expect(page.locator("#save-state")).to_have_text("All changes saved")
    page.get_by_role("button", name=re.compile(r"Scene 3, ")).click()
    expect(prompt).to_have_value("Typed in scene three.")

    # Choices apply and fix what the checklist asks for.
    page.get_by_role("button", name=re.compile(r"Scene 1, ")).click()
    page.get_by_role("button", name="Generate this scene").wait_for()
    page.get_by_role("button", name="Text", exact=True).click()
    expect(page.locator("#checklist")).to_contain_text("Ready to generate")

    # A run goes on while the page is reloaded and fills in the filmstrip.
    assert context.request.post(url + "/test/hold").ok
    page.get_by_role("button", name="Generate this scene").click()
    expect(page.locator("#run-status")).to_contain_text("Scene 1", timeout=10000)
    page.get_by_role("button", name=re.compile(r"Scene 5, ")).click()  # Browsing stays free during a run.
    expect(page.locator("#scene-heading")).to_contain_text("Scene 5")
    page.screenshot(path=str(ARTIFACTS / "generating.png"))
    page.reload(wait_until="domcontentloaded")
    expect(page.locator("#run-status")).to_contain_text("Scene 1", timeout=10000)
    assert context.request.post(url + "/test/release").ok
    expect(page.locator('#strip [aria-label^="Scene 1,"] .meta')).to_contain_text("Done", timeout=15000)

    # A second take asks before it changes the scene that continues from the first.
    page.get_by_role("button", name=re.compile(r"Scene 2, ")).click()
    page.get_by_role("button", name="Text", exact=True).click()
    expect(page.locator("#checklist")).to_contain_text("Ready to generate")
    page.get_by_role("button", name="Generate this scene").click()
    expect(page.locator('#strip [aria-label^="Scene 2,"] .meta')).to_contain_text("Done", timeout=15000)
    page.get_by_role("button", name=re.compile(r"Scene 1, ")).click()
    page.get_by_role("button", name="Generate another take").click()
    second_take = page.get_by_role("button", name="Take 2", exact=True)
    expect(second_take).to_be_visible(timeout=15000)
    expect(page.locator("#run-status")).to_have_count(0, timeout=15000)
    second_take.click()
    page.get_by_role("button", name="Use this take").click()
    expect(page.get_by_role("alertdialog")).to_contain_text("Scene 2")
    page.get_by_role("button", name="Switch, regenerate later").click()
    expect(page.locator('#strip [aria-label^="Scene 2,"] .meta')).to_contain_text("Out of date")

    # Interrupt stops a run and hands the buttons back.
    assert context.request.post(url + "/test/hold").ok
    page.get_by_role("button", name=re.compile(r"Generate all remaining scenes")).click()
    page.get_by_role("button", name="Interrupt").click()
    assert context.request.post(url + "/test/release").ok
    expect(page.locator("#run-status")).to_have_count(0, timeout=15000)
    expect(page.get_by_role("button", name=re.compile(r"Generate all remaining scenes"))).to_be_enabled()

    # Cast pictures upload, and a scene ticks them.
    page.get_by_role("button", name="Cast", exact=True).click()
    page.get_by_label("Name", exact=True).fill("Mei")
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Upload picture").click()
    chooser.value.set_files(picture())
    expect(page.locator("#inspector").get_by_text("Mei", exact=True)).to_be_visible(timeout=10000)
    page.get_by_role("button", name=re.compile(r"Scene 7, ")).click()
    page.get_by_role("button", name="References", exact=False).first.click()
    page.get_by_role("button", name="Mei", exact=True).click()
    expect(page.locator("#tag-hint")).to_contain_text("Mei = <Picture 1>")
    assert not errors, errors


def picture():
    from PIL import Image

    path = ARTIFACTS / "portrait.png"
    Image.new("RGB", (64, 64), "teal").save(path)
    return str(path)


if __name__ == "__main__":
    run()
