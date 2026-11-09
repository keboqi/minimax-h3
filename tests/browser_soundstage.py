"""Soundstage module and save smoke with fake media; no GPU or hosted calls."""
from pathlib import Path
import socket
from threading import Thread
import time

from playwright.sync_api import expect, sync_playwright
import uvicorn

from tests.test_drama3_page_video import Drama3PageVideoTests


def run():
    fixture = Drama3PageVideoTests()
    fixture.setUp()
    list(fixture.fake_export(fixture.mix.name, "qwen", "ink wash", 2, True))
    fixture.store.save_soundstage_project("Second", {"text": "Mia: Another draft.", "shown_slots": 1})
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(fixture.client.app, log_level="error"))
    thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        with sync_playwright() as playwright:
            chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
            browser = playwright.chromium.launch(headless=True, **({"executable_path": str(chrome)} if chrome.exists() else {}))
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}/soundstage/", wait_until="networkidle")
            expect(page.locator("#app")).to_be_visible()
            page.locator("#step-script").click()
            page.locator("#script-text").fill("Mia: A newer line.")
            # Switching projects flushes even a still-debounced edit.
            page.locator("#project").select_option("Second")
            expect(page.locator("#script-text")).to_have_value("Mia: Another draft.")
            assert fixture.store.soundstage_project("Night Market")[0]["text"] == "Mia: A newer line."
            page.locator("#step-cast").click()
            expect(page.locator("#cast-grid")).to_be_visible()
            page.locator("#open-library").click()
            expect(page.locator("#library-dialog")).to_be_visible()
            page.locator("#library-close").click()
            page.locator("#project").select_option("Night Market")
            page.locator("#step-mix").click()
            expect(page.locator("#player .player-title")).to_be_visible()
            expect(page.get_by_role("heading", name="Illustrated video", exact=True)).to_be_visible()
            page.get_by_role("button", name="Video", exact=True).click()
            expect(page.locator("#still-back")).to_be_visible()
            page.get_by_role("button", name="Larger subtitles", exact=True).click()
            page.locator("#project").select_option("Second")
            expect(page.locator("#project")).to_have_value("Second")
            expect(page.locator("#project-state")).to_have_text("Draft")
            assert fixture.store.soundstage_project("Night Market")[0]["video_subtitle_size"] == 110
            assert "video_subtitle_size" not in fixture.store.soundstage_project("Second")[0]
            # Failed autosaves retain the draft and prevent switching; retry saves it.
            page.locator("#step-script").click()
            page.route("**/soundstage/api/project/script?*", lambda route: route.fulfill(status=503, json={"error": "offline"}))
            page.locator("#script-text").fill("Mia: Keep this unsaved draft.")
            expect(page.locator("#save-state")).to_have_text("Not saved")
            page.locator("#project").select_option("Night Market")
            expect(page.locator("#project")).to_have_value("Second")
            expect(page.locator("#script-text")).to_have_value("Mia: Keep this unsaved draft.")
            page.unroute("**/soundstage/api/project/script?*")
            page.locator("#project").select_option("Night Market")
            expect(page.locator("#project-state")).to_have_text("Generated")
            assert fixture.store.soundstage_project("Second")[0]["text"] == "Mia: Keep this unsaved draft."
            # A same-project refresh must refetch if edits saved after its snapshot.
            page.locator("#step-script").click()
            delayed = []

            def edit_during_refresh(route):
                if delayed:
                    route.continue_()
                    return
                delayed.append(True)
                response = route.fetch()
                page.locator("#script-text").fill("Mia: Edited during refresh.")
                route.fulfill(response=response)

            page.route("**/soundstage/api/project?*", edit_during_refresh)
            page.locator("#project").select_option("Night Market")
            expect(page.locator("#script-text")).to_have_value("Mia: Edited during refresh.")
            expect(page.locator("#save-state")).to_have_text("All changes saved")
            page.wait_for_load_state("networkidle")
            expect(page.locator("#script-text")).to_have_value("Mia: Edited during refresh.")
            assert fixture.store.soundstage_project("Night Market")[0]["text"] == "Mia: Edited during refresh."
            page.unroute("**/soundstage/api/project?*")
            assert not errors, errors
            browser.close()
        print("PASS: modules, all page steps, still playback/subtitles, save-on-switch, failure/retry, and delayed refresh")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        fixture.doCleanups()


if __name__ == "__main__":
    run()
