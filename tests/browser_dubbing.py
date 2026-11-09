"""Browser regressions for dubbing edits and completion handling. Run with python -m tests.browser_dubbing."""

import json
import os
import unittest
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright

from tests.test_dub_page import PageTests


class DubbingBrowserTests(PageTests):
    gpu = True

    def serve(self, route):
        request = route.request
        url = urlsplit(request.url)
        path = url.path + ("?" + url.query if url.query else "")
        response = self.client.request(request.method, path, content=request.post_data,
                                       headers={"content-type": "application/json"})
        route.fulfill(status=response.status_code, content_type=response.headers.get("content-type"),
                      body=response.content)

    def test_target_language_picker(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            page.goto("http://dubbing.test/dubbing/")
            picker = page.get_by_label("Dub into", exact=True)
            picker.select_option("Chinese")
            picker.select_option("French")
            expect(picker).to_have_value("French")
            picker.select_option(label="Other language…")
            custom = page.get_by_label("Language name")
            expect(custom).to_be_focused()
            custom.fill("Swahili")
            page.reload()
            expect(custom).to_have_value("Swahili")
            picker.select_option("Same as spoken")
            expect(page.get_by_text("New text starts as")).to_be_visible()
            picker.select_option("Japanese")
            expect(page.get_by_text("New text starts as")).to_have_count(0)
            expect(picker).to_have_value("Japanese")
            browser.close()

    def test_delayed_edits_and_fast_completion(self):
        session = self.session([{"start": 1.234, "end": 2.345, "speaker": "speaker1",
                                 "source": "Hello", "translation": "Bonjour"}])
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()

            page.route("http://dubbing.test/**", self.serve)
            state = {"screen": "review", "sessionId": session.id}
            page.add_init_script("if (!localStorage.getItem('h3-dubbing-page')) "
                                 f"localStorage.setItem('h3-dubbing-page', {json.dumps(json.dumps(state))})")
            page.add_init_script("""{
                const fetch = window.fetch;
                window.fetch = async (...args) => {
                    const response = await fetch(...args);
                    if (window.delayEdit && ["PATCH", "PUT"].includes(args[1]?.method)) {
                        window.delayEdit = false;
                        await new Promise(resolve => window.releaseEdit = resolve);
                    }
                    return response;
                };
            }""")
            page.goto("http://dubbing.test/dubbing/")
            for first, second, original, translation in (
                ('[data-line-field="original"]', '[data-line-field="translation"]', "Hi", "Salut"),
                ('[data-cell="4"]', '[data-cell="5"]', "Good day", "Bonne journée"),
            ):
                if "cell" in first:
                    page.locator('[data-action="table-open"]').click()
                page.evaluate("window.delayEdit = true")
                page.locator(first).fill(original)
                page.locator(second).focus()
                page.wait_for_function("typeof window.releaseEdit === 'function'")
                page.locator(second).fill(translation)
                page.evaluate("window.releaseEdit(); delete window.releaseEdit")
                expect(page.locator(second)).to_have_value(translation)
                with page.expect_response(lambda res: res.request.method in ("PATCH", "PUT")):
                    page.locator(second).press("Tab")
                saved = self.client.get("/dubbing/api/session", params={"id": session.id}).json()["session"]
                line = saved["board"]["lines"][0]
                self.assertEqual((line["source"], line["translation"], line["s"], line["e"]),
                                 (original, translation, 1.234, 2.345))

            completed = {"id": "cached", "kind": "lines", "task": "preview", "active": False, "audio": True}
            page.route("**/api/preview", lambda route: route.fulfill(json={"run": completed}))
            page.evaluate("""async () => {
                const {app} = await import("./app.js");
                await app.startRun("preview", {session: app.sessionId, line: 0});
            }""")
            expect(page.locator('audio[src*="cached"]')).to_have_count(1)
            analysis = {"id": "analysis", "kind": "analyze", "task": "analyze", "active": True}
            page.route("**/api/analyze", lambda route: route.fulfill(json={"run": analysis}))
            page.route("**/api/run?*", lambda route: route.fulfill(json={"runs": {"analyze": analysis}}))
            page.evaluate("""async () => {
                const {app} = await import("./app.js");
                app.go("source");
                await app.startRun("analyze", {});
            }""")
            analysis.update(active=False, session=session.id, project="Reopened")
            page.reload()
            expect(page.locator("#screen")).to_have_attribute("data-screen", "review")
            state = page.evaluate("JSON.parse(localStorage.getItem('h3-dubbing-page'))")
            self.assertEqual(state["project"], {"name": "Reopened", "session": session.id})
            self.assertEqual(state["pendingRuns"], {})

            # Different revisions with equal-sized markup within the same minute.
            page.evaluate("""async () => {
                const {afterResult} = await import("./result.js");
                const slot = document.createElement("div");
                slot.id = "player-slot";
                document.body.append(slot);
                for (const revision of ["first", "later"]) {
                    const result = {html: `<audio data-test src="${revision}.mp3" preload="none"></audio>`,
                        when: "Nov 9, 08:00", timed: 0};
                    await afterResult({session: {id: "same", result}});
                }
            }""")
            expect(page.locator("#player-slot audio")).to_have_attribute("src", "later.mp3")
            browser.close()


if __name__ == "__main__":
    unittest.main()
