"""The Soundstage page's Script step: saving the script, the writer, pasted replies, examples and the preview.

The tab's own handlers run against a temporary workspace; only the hosted writer is replaced.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from h3_app.jobs import JOBS
from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED, DRAMA3_SCRIPT_BASIC, DRAMA3_WORKFLOWS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.drama3.bindings import Drama3Actions
from h3_ui.drama3.handlers import Drama3Handlers, Drama3Services
from h3_ui.drama3.page import install_soundstage_page

API = "/soundstage/api"
CAST = [{"speaker_id": 0, "name": "Mia", "description": "A bright young vendor.",
         "preview_text": "Fresh plums, two for one, only while the rain holds off."},
        {"speaker_id": 1, "name": "Leo", "description": "A dry, patient umbrella seller.",
         "preview_text": "Someone has to keep this corner dry on a night like this one."}]


class Drama3PageScriptTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        system = self.root / "drama3_system.md"
        system.write_text("You write Drama 3 scripts.", encoding="utf-8")
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs" / "h3",
                                       input_dir=self.root / "input", prompt_systems={"Drama 3": system})
        workspace = patch.dict("os.environ", {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        workspace.start()
        self.addCleanup(workspace.stop)
        handlers = Drama3Handlers(Drama3Services(runtime_config=lambda: self.runtime, stage_file=lambda *args: "",
                                                 generation_services=lambda: None, free_gpu=lambda: None))
        self.actions = Drama3Actions.of(handlers, interrupt=lambda *args, **kwargs: "")
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, self.actions, store=self.store, output_dir=self.root / "outputs",
                                takes_dir=lambda: self.root / "takes")
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertNotIn("error", data)
        return data

    def create(self, name="Night Market"):
        return self.ok(self.client.post(f"{API}/projects", json={"name": name}))

    def test_boot_offers_the_writer_choices(self):
        boot = self.ok(self.client.get(f"{API}/boot"))
        self.assertEqual([workflow["id"] for workflow in boot["workflows"]], list(DRAMA3_WORKFLOWS))
        self.assertTrue(all(workflow["note"] for workflow in boot["workflows"]))
        self.assertEqual(boot["advanced"], DRAMA3_SCRIPT_ADVANCED)
        self.assertIn("narrator", boot["narrators"])
        self.assertEqual([example["id"] for example in boot["examples"]], ["en", "zh", "zh-latest"])

    def test_a_new_project_starts_as_an_advanced_draft(self):
        project = self.create()
        self.assertEqual(project["script"]["script_mode"], DRAMA3_SCRIPT_ADVANCED)
        self.assertEqual(project["script"]["text"], "")
        self.assertIsNone(project["result"])
        self.assertIn("already exists", self.client.post(f"{API}/projects", json={"name": "Night  Market"}).json()["error"])
        self.assertIn("name", self.client.post(f"{API}/projects", json={"name": " "}).json()["error"])

    def test_saving_the_script_keeps_writer_choices_and_stages_its_sounds(self):
        self.create()
        script = 'Mia: Plums!\n<sound name="rain" duration="8" loop="true">Steady rain on canvas awnings</sound>'
        project = self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={
            "text": script, "title": "Night Market", "writer_workflow": DRAMA3_WORKFLOWS[1], "strict_source": True}))
        self.assertEqual(project["script"]["text"], script)
        self.assertEqual(project["script"]["writer"]["writer_workflow"], DRAMA3_WORKFLOWS[1])
        self.assertTrue(project["script"]["writer"]["strict_source"])
        self.assertFalse(project["script"]["prepared"])
        # An Advanced script's sound tag gets a sound slot, as the tab gives it.
        self.assertEqual([sound["name"] for sound in project["sounds"]], ["rain"])
        refused = self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={"script_mode": "Loud"})
        self.assertIn("Unknown script mode", refused.json()["error"])

    def test_a_rewrite_fills_the_script_cast_and_title(self):
        self.create()
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"},
                                  json={"text": "Mia called out to Leo across the wet market."}))
        answer = ("<|speaker:0|>Leo! You kept the stall open?\n<|speaker:1|>Someone has to.", "Rewrote the script.",
                  CAST, "Night Market")
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", return_value=answer) as writer:
            data = self.ok(self.client.post(f"{API}/project/rewrite", params={"name": "Night Market"},
                                            json={"writer": {"gemini_api_key": "test-key"}}))
        self.assertEqual(writer.call_args.args[2], "test-key")
        self.assertFalse(data["failed"])
        self.assertEqual(data["previous"], "Mia called out to Leo across the wet market.")
        self.assertEqual(data["script"]["text"], "Mia: Leo! You kept the stall open?\nLeo: Someone has to.")
        self.assertEqual(data["script"]["title"], "Night Market")
        self.assertTrue(data["script"]["prepared"])
        self.assertEqual([(speaker["name"], speaker["ready"]) for speaker in data["cast"]], [("Mia", False), ("Leo", False)])
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual(saved["voice_1_description"], CAST[1]["description"])
        self.assertEqual(saved["shown_slots"], 2)

    def test_a_failed_rewrite_changes_nothing_but_keeps_the_writer_choices(self):
        self.create()
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={"text": "A story."}))
        answer = ("", "Prompt enhancement failed: no key", [], "")
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", return_value=answer):
            data = self.ok(self.client.post(f"{API}/project/rewrite", params={"name": "Night Market"},
                                            json={"speaker_mode": "Single"}))
        self.assertTrue(data["failed"])
        self.assertEqual(data["script"]["text"], "A story.")
        self.assertEqual(data["script"]["writer"]["speaker_mode"], "Single")
        self.create("Empty")
        empty = self.client.post(f"{API}/project/rewrite", params={"name": "Empty"}, json={})
        self.assertIn("text first", empty.json()["error"])

    def test_another_chat_model_gets_the_full_prompt_and_its_reply_is_checked(self):
        self.create()
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"},
                                  json={"text": "Hello, Leo.", "script_mode": DRAMA3_SCRIPT_BASIC}))
        prompt = self.ok(self.client.post(f"{API}/project/writer-prompt", params={"name": "Night Market"}, json={}))["prompt"]
        self.assertIn("You write Drama 3 scripts.", prompt)
        self.assertIn("Hello, Leo.", prompt)
        reply = json.dumps({"script": "<|speaker:0|>Hello, Leo.", "speakers": CAST[:1], "title": "Greeting"})
        data = self.ok(self.client.post(f"{API}/project/reply", params={"name": "Night Market"}, json={"reply": reply}))
        self.assertFalse(data["failed"], data["status"])
        self.assertEqual(data["script"]["text"], "Mia: Hello, Leo.")
        bad = self.ok(self.client.post(f"{API}/project/reply", params={"name": "Night Market"}, json={"reply": "not json"}))
        self.assertTrue(bad["failed"])
        self.assertEqual(bad["script"]["text"], "Mia: Hello, Leo.")

    def test_a_pasted_reply_that_fixes_the_source_is_kept_with_a_warning(self):
        self.create()
        source = "现在他们感应到妖兽的可拍，有人大叫起来。"
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"},
                                  json={"text": source, "script_mode": DRAMA3_SCRIPT_BASIC}))
        reply = json.dumps({"script": "<|speaker:0|>现在他们感应到妖兽的可怕，有人大叫起来。",
                            "speakers": [{"speaker_id": 0, "name": "narrator", "description": "A calm storyteller.",
                                          "preview_text": "这片海域离琼英岛足有数万里之遥，四周海面看似平静，实则暗藏玄机。"}]})
        data = self.ok(self.client.post(f"{API}/project/reply", params={"name": "Night Market"}, json={"reply": reply}))
        self.assertFalse(data["failed"], data["status"])
        self.assertIn("妖兽的可[拍] → 妖兽的可[怕]", data["status"])
        self.assertEqual(data["script"]["text"], "narrator: 现在他们感应到妖兽的可怕，有人大叫起来。")
        # "Keep my exact words" refuses a reply that changes them.
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"},
                                  json={"text": source, "strict_source": True}))
        strict = self.ok(self.client.post(f"{API}/project/reply", params={"name": "Night Market"}, json={"reply": reply}))
        self.assertTrue(strict["failed"])
        self.assertEqual(strict["script"]["text"], source)

    def test_an_example_loads_its_script_cast_and_sounds(self):
        self.create()
        data = self.ok(self.client.post(f"{API}/project/example", params={"name": "Night Market"}, json={"example": "en"}))
        self.assertEqual(data["script"]["script_mode"], DRAMA3_SCRIPT_ADVANCED)
        self.assertTrue(data["script"]["text"])
        self.assertTrue(data["script"]["title"])
        self.assertGreater(len(data["cast"]), 1)
        self.assertTrue(data["sounds"])
        self.assertIn("Unknown", self.client.post(f"{API}/project/example", params={"name": "Night Market"},
                                                  json={"example": "fr"}).json()["error"])

    def test_the_preview_estimates_a_timeline_without_audio(self):
        preview = self.ok(self.client.post(f"{API}/preview", json={
            "text": 'Mia: Plums, two for one!\n<sound name="thunder" duration="3">Distant thunder</sound>\nLeo: Someone has to.',
            "script_mode": DRAMA3_SCRIPT_ADVANCED}))
        self.assertEqual([line["speaker"] for line in preview["lines"]], ["Mia", "Leo"])
        self.assertGreater(preview["duration"], 0)
        self.assertEqual([sound["name"] for sound in preview["sounds"]], ["thunder"])
        basic = self.ok(self.client.post(f"{API}/preview", json={"text": 'Mia: Plums!\n<sound name="thunder" duration="3">Thunder</sound>',
                                                                "script_mode": DRAMA3_SCRIPT_BASIC}))
        self.assertEqual(basic["sounds"], [])


if __name__ == "__main__":
    unittest.main()
