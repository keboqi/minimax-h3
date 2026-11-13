"""Soundstage module and save smoke with fake media; no GPU or hosted calls."""
from pathlib import Path
import socket
from threading import Thread
import time

from playwright.sync_api import expect, sync_playwright
import uvicorn

from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED
from tests.test_drama3_page_video import Drama3PageVideoTests


def run():
    fixture = Drama3PageVideoTests()
    fixture.setUp()
    list(fixture.fake_export(fixture.mix.name, "qwen", "ink wash", 2, True))
    fixture.store.save_soundstage_project("Second", {"text": "Mia: Another draft.", "shown_slots": 1})
    fixture.store.save_soundstage_project("Scenes", {"script_mode": DRAMA3_SCRIPT_ADVANCED, "text": "\n".join([
        "Mia: Plums, two for one!",
        '<parallel room="small_room" stage="Mia=-0.4, Leo=0.4">',
        '  <sound loop="true" duration="20" name="rain">Steady rain</sound>',
        "  <parallel>",
        "    Leo: Who's there?",
        "    <sequential>",
        "      <#1.0#>",
        "      Mia: Only me. [laughs]",
        "    </sequential>",
        "  </parallel>",
        "  <parallel>",
        '    <sound name="door" duration="2">A door opens</sound>',
        "    Leo: [warmly] Come in.",
        "  </parallel>",
        "</parallel>",
    ])})
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
            # The visual view reads the script as scenes; the timeline and both views find each other's lines.
            page.locator("#project").select_option("Scenes")
            expect(page.locator("#estimate .clip[data-line]")).to_have_count(6)
            expect(page.locator(".scene-band")).to_have_count(1)
            caret = "(() => { const a = document.getElementById('script-text'); return a.value.slice(0, a.selectionStart).split('\\n').length; })()"
            page.locator("#estimate .clip[data-line='13']").click()
            assert page.evaluate(caret) == 13
            expect(page.locator("#estimate .clip.active")).to_have_count(1)
            page.get_by_role("button", name="Visual", exact=True).click()
            expect(page.locator("#visual-view .vscene")).to_have_count(1)
            expect(page.locator("#visual-view .vtogether .vbranch")).to_have_count(2)
            expect(page.locator("#visual-view .vbranch-note")).to_have_text("Starts 1 s into Leo's line")
            expect(page.locator("#visual-view .vbed")).to_contain_text("rain")
            door_row = page.locator("#visual-view .vrow", has=page.locator(".vcue", has_text="door"))
            # Delivery cues show as chips beside the words, never as raw [brackets].
            expect(door_row.locator(".vcue-inline")).to_have_text("warmly")
            expect(door_row.locator(".vsaid")).to_contain_text("Come in.")
            expect(page.locator("#visual-view")).not_to_contain_text("[laughs]")
            # The outline lists the opening line and the scene, and the layout toggles hide and show its column.
            expect(page.locator("#outline-scenes .rail-scene")).to_have_count(2)
            expect(page.locator("#speakers .rail-speaker")).to_have_count(2)
            page.locator("#toggle-outline").click()
            expect(page.locator("#outline-col")).to_be_hidden()
            page.locator("#toggle-outline").click()
            expect(page.locator("#outline-col")).to_be_visible()
            page.locator("#writer-tab-paste").click()
            expect(page.locator("#apply-reply")).to_be_visible()
            page.locator("#writer-tab-built").click()
            page.locator("#visual-view .vline", has_text="Only me.").dblclick()
            expect(page.locator("#code-view")).to_be_visible()
            assert page.evaluate(caret) == 8
            # Rewrapping after columns appear or the window narrows must grow the editor.
            original_script = page.locator("#script-text").input_value()
            page.locator("#toggle-outline").click()
            page.locator("#toggle-writer").click()
            page.locator("#script-text").fill(("Mia: " + "Some longer dialogue. " * 12 + "\n") * 30)
            fits = "() => { const a = document.getElementById('script-text'); return a.clientHeight >= a.scrollHeight; }"
            page.wait_for_function(fits)
            wide_height = page.locator("#script-text").evaluate("a => a.clientHeight")
            page.locator("#toggle-outline").click()
            page.locator("#toggle-writer").click()
            page.wait_for_function(fits)
            assert page.locator("#script-text").evaluate("a => a.clientHeight") > wide_height
            page.set_viewport_size({"width": 800, "height": 1000})
            page.wait_for_function(fits)
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator("#script-text").fill(original_script)
            page.locator("#project").select_option("Night Market")
            expect(page.locator("#script-text")).to_have_value("Mia: Plums!")
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
            # Add sound lists the new, empty slot and selects it.
            page.locator("#add-sound").click()
            expect(page.locator('#sound-table tr[aria-selected="true"]')).to_contain_text("Sound 1")
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
