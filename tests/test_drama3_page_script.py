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

    def test_changes_saved_while_the_writer_works_are_kept(self):
        """The writer can take minutes; a clip or setting saved meanwhile outlives its answer."""
        self.create()
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"},
                                  json={"text": "Mia called out to Leo across the wet market."}))
        answer = ("<|speaker:0|>Leo! You kept the stall open?\n<|speaker:1|>Someone has to.", "Rewrote the script.",
                  CAST, "Night Market")

        def writer(*args, **kwargs):
            # As a cast job writing a clip, and a settings save, would while the writer works.
            data, _ = self.store.soundstage_project("Night Market")
            data["voice_0_audio"] = "made-meanwhile.wav"
            self.store.save_soundstage_project("Night Market", data)
            self.ok(self.client.patch(f"{API}/project/settings", params={"name": "Night Market"},
                                      json={"temperature": 1.2}))
            return answer

        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", side_effect=writer):
            data = self.ok(self.client.post(f"{API}/project/rewrite", params={"name": "Night Market"}, json={}))
        self.assertEqual(data["script"]["text"], "Mia: Leo! You kept the stall open?\nLeo: Someone has to.")
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual((saved["voice_0_audio"], saved["temperature"]), ("made-meanwhile.wav", 1.2))
        self.assertEqual(saved["voice_1_description"], CAST[1]["description"])

    def test_a_late_writer_answer_is_kept_for_explicit_review(self):
        self.create()
        self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={"text": "Mia sells plums."}))
        def writer(*args, **kwargs):
            self.ok(self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={"text": "Mia sells apples."}))
            return ("<|speaker:0|>Fresh plums!", "Rewrote.", CAST, "Market")
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", side_effect=writer):
            answer = self.ok(self.client.post(f"{API}/project/rewrite", params={"name": "Night Market"}, json={}))
        self.assertTrue(answer["failed"])
        self.assertEqual(answer["script"]["text"], "Mia sells apples.")
        self.assertTrue(answer["script"]["proposal"])
        proposal = self.ok(self.client.get(f"{API}/project/proposal", params={"name": "Night Market"}))
        self.assertIn("Fresh plums", proposal["text"])
        applied = self.ok(self.client.post(f"{API}/project/proposal", params={"name": "Night Market"}, json={"id": proposal["id"]}))
        self.assertFalse(applied["script"]["proposal"])
        self.assertIn("Fresh plums", applied["script"]["text"])

    def test_replaced_writer_proposal_requires_new_review(self):
        self.create()
        params = {"name": "Night Market"}
        self.ok(self.client.patch(f"{API}/project/script", params=params, json={"text": "Mia sells plums."}))

        def conflict(words, newer_text):
            def writer(*args, **kwargs):
                self.ok(self.client.patch(f"{API}/project/script", params=params, json={"text": newer_text}))
                return (f"<|speaker:0|>{words}", "Rewrote.", CAST, "Market")
            with patch("h3_app.prompt_writers.drama3.enhance_drama3_text", side_effect=writer):
                self.assertTrue(self.ok(self.client.post(f"{API}/project/rewrite", params=params, json={}))['failed'])

        conflict("Reviewed answer.", "First newer draft.")
        reviewed = self.ok(self.client.get(f"{API}/project/proposal", params=params))
        conflict("Unreviewed answer.", "Second newer draft.")
        for body in ({}, {"id": reviewed["id"]}):
            response = self.client.post(f"{API}/project/proposal", params=params, json=body)
            self.assertEqual(response.status_code, 400)
        current = self.ok(self.client.get(f"{API}/project/proposal", params=params))
        self.assertNotEqual(current["id"], reviewed["id"])
        self.assertEqual(self.store.soundstage_project("Night Market")[0]["text"], "Second newer draft.")
        applied = self.ok(self.client.post(f"{API}/project/proposal", params=params, json={"id": current["id"]}))
        self.assertIn("Unreviewed answer.", applied["script"]["text"])

    def test_invalid_patch_values_do_not_coerce_or_partially_save(self):
        self.create()
        for patch in ({"strict_source": "false"}, {"title": None}, {"writer_workflow": "unknown"}):
            answer = self.client.patch(f"{API}/project/script", params={"name": "Night Market"}, json={"text": "changed", **patch})
            self.assertEqual(answer.status_code, 400)
            self.assertEqual(self.store.soundstage_project("Night Market")[0]["text"], "")
        answer = self.client.patch(f"{API}/project/settings", params={"name": "Night Market"}, json={"temperature": "nan"})
        self.assertEqual(answer.status_code, 400)

    def test_a_settings_save_waits_for_a_clip_being_written(self):
        """Every page save takes the project's lock, as a cast job writing a clip does."""
        from threading import Thread

        from h3_ui.drama3.page_store import project_lock

        self.create()
        replies = []
        with project_lock("Night Market"):
            saving = Thread(target=lambda: replies.append(self.client.patch(
                f"{API}/project/settings", params={"name": "Night Market"}, json={"top_k": 50})))
            saving.start()
            saving.join(0.3)
            self.assertTrue(saving.is_alive(), "The settings save didn't wait for the project.")
            data, _ = self.store.soundstage_project("Night Market")
            data["voice_0_audio"] = "made-meanwhile.wav"
            self.store.save_soundstage_project("Night Market", data)
        saving.join(10)
        self.ok(replies[0])
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertEqual((saved["voice_0_audio"], saved["top_k"]), ("made-meanwhile.wav", 50))

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

    def test_an_example_lets_go_of_the_old_clips_and_saved_answers(self):
        """The example's speakers get voices of their own, not the clips the old cast left in their slots."""
        self.create()
        clips = self.root / "outputs" / "h3" / ".soundstage" / "clips"
        clips.mkdir(parents=True)
        (clips / "old-voice.wav").write_bytes(b"RIFF")
        data, _ = self.store.soundstage_project("Night Market")
        data.update({"voice_0_audio": "old-voice.wav", "voice_0_transcript": "Old words.",
                     "sound_0_audio": "old-voice.wav", "_writer_proposal": {"id": "x", "outputs": [], "source": ""},
                     "_cast_proposals": {"voice:0": {"id": "y", "patch": {}}}})
        self.store.save_soundstage_project("Night Market", data)
        loaded = self.ok(self.client.post(f"{API}/project/example", params={"name": "Night Market"}, json={"example": "en"}))
        self.assertFalse(any(speaker["ready"] for speaker in loaded["speakers"]))
        self.assertFalse(any(sound["ready"] for sound in loaded["sounds"]))
        self.assertFalse(loaded["script"]["proposal"])
        saved, _ = self.store.soundstage_project("Night Market")
        self.assertNotIn("voice_0_audio", saved)
        self.assertNotIn("voice_0_transcript", saved)
        self.assertNotIn("_cast_proposals", saved)

    def test_a_pasted_reply_replaces_a_saved_writer_answer(self):
        self.create()
        params = {"name": "Night Market"}
        self.ok(self.client.patch(f"{API}/project/script", params=params,
                                  json={"text": "Hello, Leo.", "script_mode": DRAMA3_SCRIPT_BASIC}))
        data, _ = self.store.soundstage_project("Night Market")
        data["_writer_proposal"] = {"id": "old", "outputs": [], "source": "Hello, Leo."}
        self.store.save_soundstage_project("Night Market", data)
        reply = json.dumps({"script": "<|speaker:0|>Hello, Leo.", "speakers": CAST[:1], "title": "Greeting"})
        applied = self.ok(self.client.post(f"{API}/project/reply", params=params, json={"reply": reply}))
        self.assertFalse(applied["failed"], applied["status"])
        self.assertFalse(applied["script"]["proposal"])

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

    def test_the_preview_numbers_each_line_and_sound_with_its_script_line(self):
        script = "\n".join([
            "Mia: Plums! <#0.4#> Two for one.",              # 1: two turns, one line
            '<parallel room="small_room" stage="Mia=-0.4, Leo=0.4">',
            '  <sound loop="true" duration="20" name="rain">Steady rain on a tin roof</sound>',  # 3
            "",
            "  <parallel>",
            '    <sound name="door" duration="2">A door</sound>',  # 6
            "    Leo: Who's there?",                           # 7
            "    <sequential>",
            "      <#1.0#>",
            "      Mia: Only me. [laughs]",                     # 10, overlapping Leo
            "    </sequential>",
            "  </parallel>",
            "</parallel>",
            "Leo: [sighs]",                                    # 14: nothing to hear, left off the timeline
            "Leo: Come in.",                                   # 15
        ])
        preview = self.ok(self.client.post(f"{API}/preview", json={"text": script, "script_mode": DRAMA3_SCRIPT_ADVANCED}))
        self.assertEqual([(line["text"], line["line"]) for line in preview["lines"]],
                         [("Plums!", 1), ("Two for one.", 1), ("Who's there?", 7), ("Only me.", 10), ("Come in.", 15)])
        self.assertEqual([(sound["name"], sound["line"]) for sound in preview["sounds"]], [("rain", 3), ("door", 6)])
        basic = self.ok(self.client.post(f"{API}/preview", json={"text": script, "script_mode": DRAMA3_SCRIPT_BASIC}))
        self.assertEqual([line["line"] for line in basic["lines"]], [1, 7, 10, 15])


if __name__ == "__main__":
    unittest.main()
