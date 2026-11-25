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
                        headless=True, ignore_default_args=["--hide-scrollbars"],
                        **({"executable_path": str(chrome)} if chrome.exists() else {}))
                    context = browser.new_context(viewport={"width": 1440, "height": 1000})
                    check(context, url)
                    check_full_preview(context, url)
                    check_workshop_auto(context, url)
                    browser.close()
                print("PASS: scene switching, saves, generation, takes, interrupt, cast and full preview")
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
    subtitles = page.get_by_role("button", name="Generate word-aligned subtitles")
    expect(subtitles).to_be_disabled()
    page.get_by_role("button", name="Settings", exact=True).click()
    auto_subtitles = page.get_by_role("checkbox", name="Automatically generate word-aligned subtitles after each take")
    expect(auto_subtitles).not_to_be_checked()
    auto_subtitles.check()
    expect(auto_subtitles).to_be_checked()
    expect(auto_subtitles).to_be_enabled()
    auto_score = page.get_by_role("checkbox", name="Automatically score each take in the background after generation")
    expect(auto_score).not_to_be_checked()
    auto_score.check()
    expect(auto_score).to_be_checked()
    expect(auto_score).to_be_enabled()
    assert context.request.post(url + "/test/seed").ok

    # The tab shows the page in a frame.
    page.goto(url, wait_until="domcontentloaded")
    frame = page.frame_locator("iframe.h3-storyboard-frame")
    expect(frame.locator("#scene-heading")).to_be_visible(timeout=20000)

    page.goto(url + "/storyboard/", wait_until="domcontentloaded")
    picker = page.get_by_label("Storyboard", exact=True)
    picker.select_option(label="Board 0")
    expect(page.locator("#strip .frame")).to_have_count(8)
    stage_top = page.locator("main").bounding_box()["y"]
    for region in ("Filmstrip", "Subtitles"):
        bounds = page.get_by_role("region", name=region, exact=True).bounding_box()
        assert bounds["y"] + bounds["height"] <= stage_top
    actions = page.get_by_text("Scene actions", exact=True)
    duplicate = page.get_by_role("button", name="Duplicate scene")
    expect(duplicate).to_be_hidden()
    actions.click()
    expect(duplicate).to_be_visible()
    page.keyboard.press("Escape")
    expect(duplicate).to_be_hidden()
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

    stage = page.get_by_role("main")
    inspector = page.locator("#inspector")
    prompt_tab = page.get_by_role("tab", name="Prompt & Flow")
    media_tab = page.get_by_role("tab", name=re.compile(r"^Media & Cast"))
    timing_tab = page.get_by_role("tab", name="Timing & Seed")
    prompt_tab.focus()
    page.keyboard.press("ArrowRight")
    expect(media_tab).to_be_focused()
    expect(media_tab).to_have_attribute("aria-selected", "true")
    expect(page.locator("#scene-heading")).to_contain_text("Scene 3")
    page.keyboard.press("Home")
    expect(prompt_tab).to_be_focused()

    # Typing stays local and saves one field, surviving a move to another scene.
    prompt = page.get_by_label("Prompt", exact=True)
    with page.expect_request(lambda request: request.method == "PATCH") as saved:
        prompt.fill("Typed in scene three.")
        page.get_by_role("button", name=re.compile(r"Scene 4, ")).click()
    assert saved.value.post_data_json == {"prompt": "Typed in scene three."}, saved.value.post_data
    expect(page.locator("#save-state")).to_have_text("All changes saved")
    page.get_by_role("button", name=re.compile(r"Scene 3, ")).click()
    expect(prompt).to_have_value("Typed in scene three.")

    # A dropped connection loses nothing: the edit is sent again.
    dropped = []
    def drop_once(route):
        dropped.append(route.request.post_data)
        route.abort() if len(dropped) == 1 else route.continue_()
    page.route("**/storyboard/api/boards/*/scenes/*", drop_once)
    prompt.fill("Typed through a dropped connection.")
    expect(page.locator("#save-state")).to_have_text("All changes saved", timeout=10000)
    assert len(dropped) == 2, dropped
    page.unroute("**/storyboard/api/boards/*/scenes/*")

    # Seed 0 is a seed, not "random".
    prompt_tab.focus()
    page.keyboard.press("End")
    page.keyboard.press("Enter")
    expect(timing_tab).to_be_focused()
    seed = page.get_by_label("Seed (-1 is random)")
    with page.expect_request(lambda request: request.method == "PATCH") as saved:
        seed.fill("0")
        seed.blur()
    assert saved.value.post_data_json == {"seed": 0}, saved.value.post_data

    # Choices apply and fix what the checklist asks for.
    page.get_by_role("button", name=re.compile(r"Scene 1, ")).click()
    prompt_tab.click()
    page.get_by_role("button", name="Text", exact=True).click()
    prompt.fill("A traveler says <d>[English] Hello there.</d>")
    expect(page.locator("#checklist")).to_contain_text("Ready to generate")

    # A run goes on while the page is reloaded and fills in the filmstrip.
    assert context.request.post(url + "/test/hold").ok
    inspector.get_by_role("button", name="Generate this scene").click()
    expect(page.locator("#run-status")).to_contain_text("Scene 1", timeout=10000)
    page.get_by_role("button", name=re.compile(r"Scene 5, ")).click()  # Browsing stays free during a run.
    expect(page.locator("#scene-heading")).to_contain_text("Scene 5")
    page.screenshot(path=str(ARTIFACTS / "generating.png"))
    page.reload(wait_until="domcontentloaded")
    expect(page.locator("#run-status")).to_contain_text("Scene 1", timeout=10000)
    assert context.request.post(url + "/test/release").ok
    expect(page.locator('#strip [aria-label^="Scene 1,"] .meta')).to_contain_text("Done", timeout=15000)
    expect(subtitles).to_be_enabled()

    # The scene toggle controls the actual text track and remembers the choice across scene switches.
    scene_one = page.get_by_role("button", name=re.compile(r"Scene 1, "))
    scene_two = page.get_by_role("button", name=re.compile(r"Scene 2, "))
    show_subtitles = page.get_by_role("checkbox", name="Show subtitles")
    burn_subtitles = page.get_by_role("checkbox", name="Burn in subtitles")
    scene_one.click()
    subtitle_controls = page.get_by_role("region", name="Subtitles", exact=True)
    subtitle_toggle = page.get_by_role("button", name="Subtitles", exact=True)
    expect(subtitle_controls).to_be_visible()
    subtitle_toggle.click()
    expect(subtitle_controls).to_be_hidden()
    subtitle_toggle.click()
    expect(subtitle_toggle).to_have_attribute("aria-expanded", "true")
    expect(show_subtitles).to_be_checked()
    expect(burn_subtitles).to_be_enabled()
    expect(burn_subtitles).not_to_be_checked()
    page.wait_for_function("""() => document.querySelector('#preview video')
      .textTracks[0]?.cues?.[0]?.text === 'Hello there.'""")
    show_subtitles.uncheck()
    assert page.locator("#preview video").evaluate("(video) => video.textTracks[0].mode") == "disabled"
    scene_two.click()
    expect(page.locator("#subtitle-status")).to_contain_text("Scene 2 · No subtitles")
    scene_one.click()
    expect(show_subtitles).not_to_be_checked()
    show_subtitles.check()
    assert page.locator("#preview video").evaluate("(video) => video.textTracks[0].mode") == "showing"
    page.keyboard.press("Escape")
    expect(subtitle_controls).to_be_hidden()
    expect(subtitle_toggle).to_be_focused()

    # A second take asks before it changes the scene that continues from the first.
    scene_two.click()
    page.get_by_role("button", name="Text", exact=True).click()
    expect(page.locator("#checklist")).to_contain_text("Ready to generate")
    inspector.get_by_role("button", name="Generate this scene").click()
    expect(page.locator('#strip [aria-label^="Scene 2,"] .meta')).to_contain_text("Done", timeout=15000)
    page.get_by_role("button", name=re.compile(r"Scene 1, ")).click()
    inspector.get_by_role("button", name="Generate another take").click()
    second_take = stage.get_by_role("button", name=re.compile(r"^Take 2\b"))
    expect(second_take).to_be_visible(timeout=15000)
    expect(page.locator("#run-status")).to_have_count(0, timeout=15000)
    second_take.click()
    page.get_by_role("button", name="Use this take").click()
    expect(page.get_by_role("alertdialog")).to_contain_text("Scene 2")
    page.get_by_role("button", name="Switch, regenerate later").click()
    expect(page.locator('#strip [aria-label^="Scene 2,"] .meta')).to_contain_text("Out of date")

    # An unused take can be previewed and deleted, and its seed is shown.
    first_take = stage.get_by_role("button", name=re.compile(r"\bTake 1\b"))
    first_take.click()
    removed_video = page.locator("#preview video").get_attribute("src")
    expect(page.get_by_role("button", name="Delete take")).to_be_visible()
    page.get_by_role("button", name="Delete take").click()
    expect(page.get_by_role("alertdialog", name="Delete take")).to_contain_text("Scene 2")
    page.get_by_role("button", name="Delete permanently").click()
    expect(stage.locator(".take-card")).to_have_count(1)
    expect(page.locator("#preview video")).not_to_have_attribute("src", removed_video)

    # Interrupt stops a run and hands the buttons back.
    assert context.request.post(url + "/test/hold").ok
    page.get_by_role("button", name=re.compile(r"Generate all remaining scenes")).click()
    page.get_by_role("button", name="Interrupt").click()
    assert context.request.post(url + "/test/release").ok
    expect(page.locator("#run-status")).to_have_count(0, timeout=15000)
    expect(page.get_by_role("button", name=re.compile(r"Generate all remaining scenes"))).to_be_enabled()

    # Filmstrip connector toggles how a scene starts directly from the strip.
    connector = page.get_by_role("button", name=re.compile(r"Transition to scene 2:"))
    connector.click()
    expect(page.get_by_role("button", name=re.compile(r"Transition to scene 2: fresh start"))).to_be_visible()
    connector.click()
    expect(page.get_by_role("button", name=re.compile(r"Transition to scene 2: continues"))).to_be_visible()

    # Cast pictures upload, Escape returns to the scene, and clicking a tag chip inserts into the prompt.
    page.get_by_role("button", name="Cast", exact=True).click()
    expect(page.get_by_role("region", name="Filmstrip")).to_be_visible()
    page.get_by_label("Name", exact=True).fill("Mei")
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Upload picture").click()
    chooser.value.set_files(picture())
    expect(page.locator("#inspector").get_by_text("Mei", exact=True)).to_be_visible(timeout=10000)
    page.keyboard.press("Escape")
    expect(page.get_by_label("Scene title")).to_be_visible()
    expect(page.get_by_role("region", name="Filmstrip")).to_be_visible()
    page.get_by_role("button", name="Auto render", exact=True).click()
    expect(page.get_by_role("heading", name="Queue storyboards")).to_be_visible()
    pick = inspector.get_by_role("checkbox", name=re.compile(r"^Board 0\b"))
    pick.check()
    expect(pick).to_be_focused()
    expect(inspector.get_by_role("button", name="Render and export 1 storyboard")).to_be_enabled()
    inspector.get_by_role("button", name="Clear", exact=True).click()
    expect(pick).not_to_be_checked()
    page.keyboard.press("Escape")
    expect(page.get_by_role("region", name="Filmstrip")).to_be_visible()
    page.get_by_role("button", name=re.compile(r"Scene 7, ")).click()
    page.get_by_role("button", name="References", exact=False).first.click()
    media_tab.click()
    page.get_by_role("button", name="Mei", exact=True).click()
    page.get_by_role("button", name="Mei = <Picture 1>", exact=True).click()
    expect(page.locator("#tag-hint")).to_contain_text("Mei = <Picture 1>")
    expect(prompt).to_have_value(re.compile(r"<Picture 1>"))

    # The tabs fit on a narrow screen, and a scrolling textarea wraps at the same width as its mirror.
    page.set_viewport_size({"width": 320, "height": 800})
    assert timing_tab.evaluate("(tab) => tab.getBoundingClientRect().right <= document.documentElement.clientWidth")
    prompt.fill(("A long prompt line with <Picture 1> and dialogue.\n" * 24).rstrip())
    page.wait_for_function("""() => {
      const input = document.getElementById('scene-prompt');
      const mirror = document.getElementById('prompt-mirror');
      return input.scrollHeight > input.clientHeight && input.clientWidth === mirror.clientWidth
        && input.scrollHeight === mirror.scrollHeight;
    }""")
    prompt.evaluate("(textarea) => { textarea.scrollTop = textarea.scrollHeight; }")
    page.wait_for_function("""() => {
      const input = document.getElementById('scene-prompt');
      return input.scrollTop > 0 && input.scrollTop === document.getElementById('prompt-mirror').scrollTop;
    }""")
    expect(page.locator("#save-state")).to_have_text("All changes saved")
    assert not errors, errors


def check_full_preview(context, url):
    seeded = context.request.post(url + "/test/seed-preview")
    assert seeded.ok
    page = context.new_page()
    errors = []
    joins = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: joins.append(request.url) if request.url.endswith("/join") else None)
    page.goto(url + "/storyboard/", wait_until="domcontentloaded")
    page.get_by_label("Storyboard", exact=True).select_option(seeded.json()["id"])
    expect(page.locator("#strip .frame")).to_have_count(3)
    toolbar = page.get_by_role("region", name="Subtitles", exact=True)
    language = toolbar.get_by_role("combobox", name="Translation language", exact=True)
    expect(language).to_be_visible()
    with page.expect_response(lambda response: response.url.endswith("/subtitle-settings")) as saved:
        language.select_option("French")
    assert saved.value.ok
    page.reload(wait_until="domcontentloaded")
    expect(language).to_have_value("French")
    toolbar.get_by_text("Translation options", exact=True).click()
    model = toolbar.get_by_role("combobox", name="Translation model", exact=True)
    expect(model).to_have_value("gemma-4-26b-a4b-it")
    model.select_option("gemini-3.5-flash-lite")
    toolbar.get_by_text("Translation options", exact=True).click()
    status = toolbar.locator("#subtitle-status")
    expect(status).to_contain_text("Scene 1 · Subtitles available, translation missing")
    translation = toolbar.get_by_role("checkbox", name="Show translation")
    expect(translation).to_be_disabled()
    with page.expect_request(lambda request: request.url.endswith("/subtitle-translations")) as generated:
        page.get_by_role("button", name="Translate this take").click()
    assert generated.value.post_data_json["force"] is False
    assert len(generated.value.post_data_json["take_ids"]) == 1
    page.wait_for_function("""() => document.querySelector('#preview video').textTracks[0]
      ?.cues?.[0]?.text.includes('<c.translation>French Line 1</c>')""")
    expect(status).to_contain_text("Scene 1 · Subtitles available, translation ready")
    with page.expect_request(lambda request: request.url.endswith("/subtitle-translations")) as regenerated:
        toolbar.get_by_role("button", name="Regenerate this take").click()
    assert regenerated.value.post_data_json["force"] is True
    assert regenerated.value.post_data_json["model"] == "gemini-3.5-flash-lite"
    expect(page.locator("#translate-scene")).to_be_enabled()
    page.reload(wait_until="domcontentloaded")
    expect(status).to_contain_text("Scene 1 · Subtitles available, translation ready")
    requests = []
    page.on("request", lambda request: requests.append(request.post_data_json)
            if request.url.endswith("/subtitle-translations") else None)
    expect(translation).to_be_checked()
    translation.uncheck()
    translation.check()
    page.wait_for_function("""() => document.querySelector('#preview video').textTracks[0]
      ?.cues?.[0]?.text.includes('<c.translation>French Line 1</c>')""")
    assert not requests, requests
    translation.uncheck()
    page.wait_for_function("""() => document.querySelector('#preview video').textTracks[0]
      ?.cues?.[0]?.text === 'Line 1'""")
    toolbar.get_by_role("checkbox", name="Translate on export", exact=True).check()
    page.get_by_role("button", name="Play full video").click()
    preview = page.locator("#full-video")
    expect(status).to_contain_text("1 of 3 current takes translated")
    assert not requests, requests
    position = preview.get_by_role("status")
    subtitles = toolbar.get_by_role("checkbox", name="Show subtitles")
    active = preview.locator("video:not(.standby)")
    expect(position).to_have_text("Scene 1 of 3")
    active.evaluate("video => video.pause()")
    toolbar.get_by_role("button", name="Translate missing takes (2)", exact=True).click()
    expect(status).to_contain_text("3 of 3 current takes translated")
    assert len(requests) == 2 and all(len(request["take_ids"]) == 1 and not request["force"] for request in requests)
    page.wait_for_function("""() => {
      const next = document.querySelector('#full-video video.standby');
      return next.readyState >= 2 && !next.seeking && next.currentTime === 1;
    }""")
    expect(subtitles).to_be_checked()
    expect(translation).to_be_checked()
    page.wait_for_function("""() => document.querySelector('#full-video video:not(.standby)')
      .textTracks[0]?.cues?.[0]?.text.includes('<c.translation>French Line 1</c>')""")
    subtitles.uncheck()
    preview.get_by_role("button", name="Fullscreen", exact=True).click()
    expect(preview.get_by_role("button", name="Exit fullscreen")).to_be_visible()
    expect(preview.get_by_role("region", name="Subtitles", exact=True)).to_be_visible()
    active.evaluate("video => video.play()")
    expect(position).to_have_text("Scene 2 of 3")
    assert active.evaluate("video => video.currentTime >= 1 && video.textTracks[0].mode === 'disabled'")
    subtitles.check()
    expect(position).to_have_text("Scene 3 of 3")
    page.wait_for_function("""() => document.querySelector('#full-video video:not(.standby)')
      .textTracks[0]?.activeCues?.[0]?.text.includes('<c.translation>French Line 3</c>')""")
    translation.uncheck()
    page.wait_for_function("""() => document.querySelector('#full-video video:not(.standby)')
      .textTracks[0]?.activeCues?.[0]?.text === 'Line 3'""")
    expect(preview.get_by_role("button", name="Exit fullscreen")).to_be_visible()
    preview.get_by_role("button", name="Close").click()
    expect(preview.locator("video")).to_have_count(0)
    expect(language).to_be_visible()
    assert not joins, joins
    assert not errors, errors
    page.close()


def check_workshop_auto(context, url):
    page = context.new_page()
    auto_requests = []

    def workshop_boot(route):
        response = route.fetch()
        route.fulfill(response=response, json={**response.json(), "gpu": False, "auto": None})

    page.route("**/storyboard/api/boot", workshop_boot)
    page.on("request", lambda request: auto_requests.append(request.url) if "/storyboard/api/auto" in request.url else None)
    page.goto(url + "/storyboard/", wait_until="domcontentloaded")
    page.get_by_role("button", name="Auto render", exact=True).click()
    expect(page.get_by_text("Open Storyboard on a GPU Studio endpoint", exact=False)).to_be_visible()
    expect(page.get_by_role("heading", name="Queue storyboards")).to_have_count(0)
    assert not auto_requests, auto_requests
    page.close()


def picture():
    from PIL import Image

    path = ARTIFACTS / "portrait.png"
    Image.new("RGB", (64, 64), "teal").save(path)
    return str(path)


if __name__ == "__main__":
    run()
