"""Browser regressions for dubbing edits and completion handling. Run with python -m tests.browser_dubbing."""

import json
import os
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
from playwright.sync_api import expect, sync_playwright

from h3_app.dubbing import pipeline, projects, session as sessions
from h3_app.dubbing.results import Player, player_folder, save_player
from tests.test_dub_page import PageTests


class DubbingBrowserTests(PageTests):
    gpu = True

    def test_workshop_automatically_loads_the_open_project_and_studio_dub(self):
        draft = self.client.post("/dubbing/api/project/save", json={
            "name": "Interview", "source": str(self.media), "settings": {"target_language": "French"},
        }).json()["project"]
        analyzed = self.session([{"start": 0, "end": 1, "source": "Hi", "translation": "Bonjour"}],
                                target_language="French")
        self.runtime.outputs_dir.mkdir()
        output = self.runtime.outputs_dir / "interview.mp3"
        output.write_bytes(b"dub")
        sessions.modify(analyzed, lambda session: session.data.update(last_output=str(output)))

        reloads = []

        def reload_from_studio(route):
            reloads.append(route.request.url)
            if len(reloads) == 1:
                route.fulfill(status=409, json={"detail": "Workshop is using the shared volume."})
                return
            if len(reloads) == 2:
                self.assertTrue(projects.attach(self.runtime, "Interview", analyzed.id,
                                                expected_session=draft["session"]))
            else:
                def update_translation(session):
                    lines = session.segments
                    lines[0]["translation"] = "Salut"
                    session.segments = lines

                sessions.modify(analyzed, update_translation)
                if len(reloads) == 3:
                    projects.save(self.runtime, "Interview", analyzed.id, None,
                                  {"target_language": "French", "notes": "Saved in Studio"})
            route.fulfill(json={"reloaded": True})

        boot = self.client.get("/dubbing/api/boot").json()
        boot.update(gpu=False, reload_volume=True, pipelines=[item for item in boot["pipelines"] if item["cloud"]])
        state = {"screen": "source", "sessionId": draft["session"], "project": draft,
                 "settings": {"target_language": "Japanese"}}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            page.route("**/api/boot", lambda route: route.fulfill(json=boot))
            page.route("**/api/reload-volume", reload_from_studio)
            page.add_init_script(f"localStorage.setItem('h3-dubbing-page', {json.dumps(json.dumps(state))})")
            page.route("**/workshop-test", lambda route: route.fulfill(
                content_type="text/html", body='<iframe id="dub" src="/dubbing/" hidden></iframe>',
            ))
            page.goto("http://dubbing.test/workshop-test")
            frame = page.frame_locator("#dub")
            self.assertEqual(reloads, [])
            page.locator("#dub").evaluate("(frame) => { frame.hidden = false; }")
            expect(frame.locator(".line-row.selected")).to_contain_text("Bonjour")
            expect(frame.get_by_role("button", name="Retry Studio sync")).to_be_hidden()
            self.assertEqual(page.evaluate(
                "JSON.parse(localStorage.getItem('h3-dubbing-page')).settings.target_language"
            ), "French")
            self.assertEqual(len(reloads), 2)
            frame.get_by_role("button", name="Dub", exact=True).click()
            expect(frame.locator(".latest")).to_contain_text("Latest dub")

            page.locator("#dub").evaluate(
                "(frame) => new Promise(resolve => { frame.hidden = true; "
                "requestAnimationFrame(() => requestAnimationFrame(resolve)); })"
            )
            with page.expect_response("**/api/reload-volume"):
                page.locator("#dub").evaluate("(frame) => { frame.hidden = false; }")
            self.assertEqual(len(reloads), 3)
            frame.get_by_role("button", name="Review lines", exact=True).click()
            expect(frame.locator(".line-row.selected")).to_contain_text("Salut")
            self.assertEqual(page.evaluate(
                "JSON.parse(localStorage.getItem('h3-dubbing-page')).settings.notes"
            ), "Saved in Studio")
            browser.close()

    def test_language_change_and_new_takes_apply_to_all_requested_lines(self):
        session = self.session([
            {"start": i * 2, "end": i * 2 + 1, "source": f"Line {i + 1}", "translation": f"French {i + 1}"}
            for i in range(3)], target_language="French")
        generated = []

        def preview(request, index, **_kwargs):
            generated.append(index)
            yield pipeline.LinePreview(pipeline.RATE, np.zeros(100, dtype=np.float32), "Fresh take ready.")

        with patch("h3_ui.dubbing.page.owned_generation", lambda callback, family: callback), \
                patch("h3_ui.dubbing.page.run_job", lambda run, owner, session, family, values, callback: callback(*values)), \
                patch.object(self.handlers, "_hf_token", return_value=None), \
                patch.object(pipeline.gemini, "translate", return_value=["German 1", "German 2", "German 3"]) as translate, \
                patch.object(pipeline, "preview_line", side_effect=preview), sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            state = {"screen": "source", "sessionId": session.id,
                     "settings": {"target_language": "French", "translator": pipeline.TRANSLATOR_GEMINI}}
            page.add_init_script(f"localStorage.setItem('h3-dubbing-page', {json.dumps(json.dumps(state))})")
            page.goto("http://dubbing.test/dubbing/")
            page.get_by_label("Dub into", exact=True).select_option("German")
            page.locator('[data-screen="review"]').click()
            page.locator('[data-action="pick"][data-line="1"]').check()
            page.locator(".inspector details.sub summary").click()
            with page.expect_request("**/api/lines-prompt") as prompt_request:
                page.locator('.inspector [data-action="copy-lines-prompt"]').click()
            self.assertEqual(prompt_request.value.post_data_json["target_language"], "German")
            self.assertEqual(prompt_request.value.post_data_json["picked"], [])
            page.locator('.inspector [data-action="translate"]').click()
            expect(page.locator(".readiness-title")).to_contain_text("German")
            self.assertEqual(translate.call_args.kwargs["target_language"], "German")
            saved = self.client.get("/dubbing/api/session", params={"id": session.id}).json()["session"]
            self.assertEqual([line["translation"] for line in saved["board"]["lines"]],
                             ["German 1", "German 2", "German 3"])

            page.locator('[data-action="pick"][data-line="2"]').check()
            with page.expect_request("**/api/preview") as request:
                page.get_by_role("button", name="New takes", exact=True).click()
            self.assertEqual(request.value.post_data_json["lines"], [1, 2])
            expect(page.locator('.inspector [data-run-status]')).to_contain_text("Generated fresh takes for 2 lines")
            self.assertEqual(generated, [1, 2])
            browser.close()

    def test_timeline_scrolls_to_line_in_board_and_table(self):
        session = self.session([
            {"start": i * 2, "end": i * 2 + 1, "speaker": "speaker1",
             "source": f"Line {i + 1}", "translation": f"Translation {i + 1}"}
            for i in range(40)
        ])
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.route("http://dubbing.test/**", self.serve)
            state = {"screen": "review", "sessionId": session.id}
            page.add_init_script(f"localStorage.setItem('h3-dubbing-page', {json.dumps(json.dumps(state))})")
            page.goto("http://dubbing.test/dubbing/")
            for table in (False, True):
                if table:
                    page.locator('[data-action="table-open"]').click()
                else:
                    # Timeline navigation must reveal a line hidden by the current filter.
                    page.locator('[data-filter="untranslated"]').click()
                    expect(page.locator(".line-list .line-row")).to_have_count(0)
                page.locator('.timeline [data-line="35"]').click()
                row = page.locator('.lines-table tr[data-row="35"]' if table else '.line-row.selected')
                expect(row).to_be_in_viewport(ratio=1)
                expect(page.locator(".inspector h2")).to_have_text("Line 36")
                self.assertGreater(page.evaluate("window.scrollY"), 500)
                if not table:
                    expect(row).to_contain_text("Translation 36")
                    expect(page.locator('[data-filter="all"]')).to_have_attribute("aria-pressed", "true")
            browser.close()

    def test_auto_controls_and_batch_generation(self):
        session = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hello",
                                 "translation": "Bonjour"}])
        self.client.post("/dubbing/api/project/save", json={"name": "Interview", "session": session.id})

        def generate(*values):
            yield None, None, None, None, "**Error:** Test take failed."

        with patch.object(self.handlers, "generate_dub", generate), sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            page.goto("http://dubbing.test/dubbing/")
            page.get_by_label("Auto mode", exact=False).check()
            expect(page.get_by_role("button", name="Auto-dub source", exact=True)).to_be_visible()
            page.reload()
            expect(page.get_by_label("Auto mode", exact=False)).to_be_checked()
            page.locator("#project-list").select_option("Interview")
            page.locator("#project-load").click()
            page.locator("#auto-dub").click()
            expect(page.locator("#auto-message")).to_contain_text("Test take failed.")
            page.locator("#batch-open").click()
            page.get_by_label("Interview", exact=True).check()
            page.locator("#batch-start").click()
            expect(page.locator("#batch-progress")).to_contain_text("failed")
            expect(page.locator("#batch-progress")).to_contain_text("Test take failed.")
            browser.close()

    def test_fast_auto_completion_restores_its_project_settings_and_source(self):
        previous = self.session([])
        completed = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hello",
                                   "translation": "Bonjour"}], target_language="French")
        saved = self.client.post("/dubbing/api/project/save", json={
            "name": "Batch project", "session": completed.id,
            "settings": {"preset": "Strict", "target_language": "French"},
        }).json()
        run = {"id": "fast-auto", "kind": "auto", "task": "auto", "active": False,
               "session": completed.id, "project": "Batch project", "status": "Done."}
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            state = {"sessionId": previous.id, "source": {"path": str(previous.path("source"))},
                     "settings": {"preset": "Expressive", "target_language": "English"}}
            page.add_init_script("if (!localStorage.getItem('h3-dubbing-page')) "
                                 f"localStorage.setItem('h3-dubbing-page', {json.dumps(json.dumps(state))})")
            page.route("**/api/run?*", lambda route: route.fulfill(json={"runs": {"auto": run}}))
            page.goto("http://dubbing.test/dubbing/")
            expect(page.locator("#project-status")).to_have_text("Project: Batch project")
            expect(page.locator("#screen")).to_have_attribute("data-screen", "result")
            page.wait_for_function("JSON.parse(localStorage.getItem('h3-dubbing-page')).completedAuto === 'fast-auto'")
            state = page.evaluate("JSON.parse(localStorage.getItem('h3-dubbing-page'))")
            self.assertEqual(state["settings"]["preset"], "Strict")
            self.assertEqual(state["settings"]["target_language"], "French")
            self.assertEqual(state["source"]["path"], saved["session"]["source"]["path"])
            # A single Auto keeps current edits; a batch follows the saved settings it used.
            for batch, expected in ((False, "Expressive"), (True, "Strict")):
                run.update(id=f"next-auto-{batch}", batch=batch)
                page.evaluate("""async () => {
                    const {app} = await import('./app.js');
                    app.settings.preset = 'Expressive';
                    app.watchRuns();
                }""")
                page.wait_for_function("id => JSON.parse(localStorage.getItem('h3-dubbing-page')).completedAuto === id",
                                       arg=run["id"])
                state = page.evaluate("JSON.parse(localStorage.getItem('h3-dubbing-page'))")
                self.assertEqual(state["settings"]["preset"], expected)
            page.reload()
            expect(page.locator("#project-status")).to_have_text("Project: Batch project")
            page.locator("#project-new").click()
            expect(page.locator("#project-status")).to_contain_text("Unsaved draft")
            expect(page.locator("#screen")).to_have_attribute("data-screen", "source")
            browser.close()

    def serve(self, route):
        request = route.request
        url = urlsplit(request.url)
        path = url.path + ("?" + url.query if url.query else "")
        response = self.client.request(request.method, path, content=request.post_data,
                                       headers={"content-type": "application/json"})
        route.fulfill(status=response.status_code, content_type=response.headers.get("content-type"),
                      body=response.content)

    def test_new_project_from_saved_project(self):
        saved = self.client.post("/dubbing/api/project/save", json={
            "name": "Original", "source": str(self.media), "settings": {"target_language": "French"},
        }).json()["project"]
        replacement = self.root / "replacement.wav"
        replacement.write_bytes(b"other audio")
        copied = sessions.create(self.runtime, replacement, video=False)
        uploaded = self.client.get("/dubbing/api/source", params={"path": str(copied.path("source"))}).json()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            page.route("**/api/upload", lambda route: route.fulfill(json=uploaded))
            page.goto("http://dubbing.test/dubbing/")
            page.locator("#project-list").select_option("Original")
            page.get_by_role("button", name="Load", exact=True).click()
            expect(page.locator("#project-status")).to_have_text("Project: Original")
            unsaved = page.locator("#project-unsaved")
            target = page.get_by_label("Dub into", exact=True)
            expect(unsaved).to_be_hidden()
            target.select_option("Japanese")
            expect(unsaved).to_be_visible()
            target.select_option("French")
            expect(unsaved).to_be_hidden()
            target.select_option("Japanese")
            page.reload()
            expect(target).to_have_value("Japanese")
            expect(unsaved).to_be_visible()
            page.get_by_role("button", name="Save project", exact=True).click()
            expect(unsaved).to_be_hidden()
            page.get_by_role("button", name="New project").click()
            expect(page.locator("#project-status")).to_contain_text("Unsaved draft")
            expect(unsaved).to_be_hidden()
            expect(page.get_by_label("Dub into", exact=True)).to_have_value("English")
            state = page.evaluate("JSON.parse(localStorage.getItem('h3-dubbing-page'))")
            self.assertEqual([state[key] for key in ("sessionId", "source", "project")], [None] * 3)
            page.locator("#project-list").select_option("Original")
            page.get_by_role("button", name="Load", exact=True).click()
            expect(target).to_have_value("Japanese")
            self.assertEqual(page.evaluate("JSON.parse(localStorage.getItem('h3-dubbing-page')).sessionId"),
                             saved["session"])
            page.locator('[data-action="replace-file"]').click()
            page.locator('[data-action="upload"]').set_input_files(replacement)
            expect(unsaved).to_be_visible()
            page.get_by_role("button", name="Save project", exact=True).click()
            expect(unsaved).to_be_hidden()
            self.assertNotEqual(projects.load(self.runtime, "Original")[1].id, saved["session"])
            browser.close()

    def test_opening_a_finished_project_defers_player_until_result(self):
        session = self.session([{"start": 0, "end": 1, "source": "Hello", "translation": "Bonjour"}])
        output = self.root / "dub.mp3"
        output.write_bytes(b"mix")
        tracks = player_folder(output)
        tracks.mkdir(parents=True)
        dub, original = tracks / "dub.mp3", tracks / "original.mp3"
        dub.write_bytes(b"dub")
        original.write_bytes(b"original")
        save_player(tracks, Player(None, dub, original, None, 1, []))
        sessions.modify(session, lambda current: current.data.update(last_output=str(output)))
        self.client.post("/dubbing/api/project/save", json={"name": "Interview", "session": session.id})

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel=os.environ.get("H3_BROWSER_CHANNEL"))
            page = browser.new_page()
            page.route("http://dubbing.test/**", self.serve)
            player_requests = []
            page.on("request", lambda request: player_requests.append(request.url) if "/api/result-player" in request.url else None)
            page.goto("http://dubbing.test/dubbing/")
            page.locator("#project-list").select_option("Interview")
            page.get_by_role("button", name="Load", exact=True).click()
            expect(page.locator("#screen")).to_have_attribute("data-screen", "review")
            self.assertEqual(player_requests, [])
            page.locator('button[data-screen="result"]').click()
            expect(page.locator("#player-slot .h3-dub-player")).to_be_visible()
            self.assertEqual(len(player_requests), 1)
            browser.close()

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
