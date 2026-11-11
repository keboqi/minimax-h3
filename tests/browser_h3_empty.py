"""Check the Video Empty button against the real page API, including autosave."""

from pathlib import Path
import re
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

from tests.test_h3_page import H3PageTests


def run():
    fixture = H3PageTests()
    fixture.setUp()
    try:
        fixture.change("mode", mode="Reference media")
        fixture.upload("ref_image_1")
        fixture.upload("ref_image_1", "replacement.png")
        fixture.upload("first_image", "first.png")
        fixture.client.post("/h3/api/prompt", json={"prompt": "Animate <Picture 1>"})
        static = Path(__file__).resolve().parents[1] / "h3_ui/h3/page"
        held_prompts = []
        delay_prompt = False
        writes = []

        def route(request):
            url = urlsplit(request.request.url)
            path = url.path
            if path.startswith("/h3/api/"):
                if path == "/h3/api/prompt" and delay_prompt:
                    held_prompts.append(request)
                    return
                if request.request.method == "POST":
                    writes.append(path)
                target = path + (f"?{url.query}" if url.query else "")
                response = fixture.client.request(request.request.method, target,
                                                  content=request.request.post_data,
                                                  headers={"Content-Type": "application/json"})
                request.fulfill(status=response.status_code, body=response.content,
                                content_type=response.headers.get("content-type", "application/json"))
            elif path in ("/h3/", "/h3/app.js", "/h3/app.css"):
                file = static / (Path(path).name if not path.endswith("/") else "index.html")
                content_type = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}[file.suffix]
                request.fulfill(body=file.read_bytes(), content_type=content_type)
            else:
                request.abort()

        with sync_playwright() as playwright:
            chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
            browser = playwright.chromium.launch(executable_path=str(chrome) if chrome.exists() else None)
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", route)
            page.goto("http://test.local/h3/")
            expect(page.locator("#prompt")).to_have_value("Animate <Picture 1>")
            expect(page.locator("#run")).to_be_disabled()
            expect(page.locator("#readiness")).to_contain_text("Confirm the replaced reference")
            page.locator("#mode").get_by_role("button", name="First / last frame", exact=True).click()
            expect(page.locator("#run")).to_be_enabled()
            expect(page.locator("#readiness")).not_to_contain_text("Confirm the replaced reference")
            # The old reference still needs confirmation if that mode is selected again.
            page.locator("#mode").get_by_role("button", name="Reference media", exact=True).click()
            expect(page.locator("#run")).to_be_disabled()
            # Hold the blur autosave and ensure confirmation waits for it.
            delay_prompt = True
            before = writes.count("/h3/api/references/confirm")
            page.locator("#prompt").fill("Animate <Picture 1> with wind")
            page.locator("#confirm-references").click()
            page.wait_for_timeout(100)
            assert held_prompts, "Expected a pending blur autosave"
            assert writes.count("/h3/api/references/confirm") == before
            delay_prompt = False
            route(held_prompts.pop(0))
            expect(page.locator("#run")).to_be_enabled()
            assert writes.count("/h3/api/references/confirm") == before + 1
            page.locator("#prompt").fill("Animate <Picture 1>")
            page.locator("#reference-list").get_by_role("button", name="Remove", exact=True).click()
            expect(page.locator("#prompt")).to_have_value("Animate")
            expect(page.locator("#reference-confirm")).to_be_hidden()
            expect(page.locator("#run")).to_be_enabled()
            page.reload()
            expect(page.locator("#run")).to_be_enabled()
            expect(page.locator("#prompt")).to_have_value("Animate")
            page.locator("#prompt").fill("Unsaved draft")
            page.get_by_role("button", name="Empty", exact=True).click()
            expect(page.locator("#prompt")).to_have_value("")
            state = fixture.client.get("/h3/api/boot").json()["state"]
            assert state["media"] == {}, state["media"]
            page.reload()
            expect(page.locator("#prompt")).to_have_value("")

            fixture.upload("first_image", "context-first.png")
            fixture.change("mode", mode="First / last frame")
            fixture.client.post("/h3/api/prompt", json={"prompt": "Animate this"})
            page.reload()

            def suggest():
                page.locator("#enhance").click()
                expect(page.locator("#suggestion-box")).to_be_visible(timeout=10000)

            def reject_stale():
                page.locator("#accept").click()
                expect(page.locator("#enhance-status")).to_contain_text("changed")
                expect(page.locator("#prompt")).to_have_value("Animate this")

            suggest()
            fixture.upload("first_image", "context-replacement.png")
            page.locator("#refresh").click()
            expect(page.locator("#frame-list img").first).to_have_attribute("src", re.compile("context-replacement"))
            reject_stale()

            suggest()
            page.locator("#mode").get_by_role("button", name="Text to video", exact=True).click()
            expect(page.locator("#mode").get_by_role("button", name="Text to video", exact=True)).to_have_attribute("aria-pressed", "true")
            reject_stale()

            page.locator("#mode").get_by_role("button", name="First / last frame", exact=True).click()
            expect(page.locator("#frames")).to_be_visible()
            suggest()
            # Inactive Ref2VA changes must not invalidate a first-frame suggestion.
            fixture.upload("ref_image_1", "inactive-reference.png")
            page.locator("#refresh").click()
            page.locator("#accept").click()
            expect(page.locator("#prompt")).to_have_value("Animate this In golden light.")
            assert not errors, errors
            browser.close()
        print("Video checks passed: mode readiness, autosave/confirmation ordering, frame/mode suggestion invalidation, inactive inputs, Empty and reload.")
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    run()
