"""Auto orchestration and batch lifecycle without model inference."""
from threading import Event
import time
from unittest.mock import patch

from h3_app.dubbing import pipeline, projects, session as sessions
from tests.test_dub_page import API, PageTests


class AutoTests(PageTests):
    gpu = True

    def setUp(self):
        super().setUp()
        for target, replacement in (
            ("h3_ui.dubbing.page.owned_generation", lambda callback, family: callback),
            ("h3_ui.dubbing.page.run_job", lambda run, owner, session, family, values, callback: callback(*values)),
        ):
            context = patch(target, replacement)
            context.start()
            self.addCleanup(context.stop)

    def prepared(self):
        return self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi",
                              "translation": "Salut"}])

    def generate(self, session_id, *settings):
        session = sessions.load(self.runtime, session_id)
        session.data["last_output"] = "finished.mp3"
        session.save()
        yield None, None, "finished.mp3", None, "Dub finished."

    def test_source_analyzes_then_dubs_and_keeps_settings(self):
        prepared = self.prepared()
        order = []

        def analyze(*values):
            order.append("analyze")
            yield prepared.id, {}, [], "Found lines.", None

        def generate(session_id, *settings):
            order.append("dub")
            self.assertEqual(session_id, prepared.id)
            self.assertEqual(settings[1], "Strict")
            yield from self.generate(session_id, *settings)

        with patch.object(self.handlers, "analyze_source", analyze), patch.object(self.handlers, "generate_dub", generate):
            response = self.client.post(f"{API}/auto", json={"source": str(self.media), "preset": "Strict"})
            self.assertEqual(response.status_code, 200)
            run = self.wait("auto")
        self.assertEqual(order, ["analyze", "dub"])
        self.assertIsNone(run["error"])
        self.assertEqual(run["session"], prepared.id)

    def test_prepared_project_preserves_edits_without_analysis(self):
        prepared = self.prepared()
        with patch.object(self.handlers, "analyze_source") as analyze, patch.object(self.handlers, "generate_dub", self.generate):
            self.client.post(f"{API}/auto", json={"session": prepared.id})
            self.assertIsNone(self.wait("auto")["error"])
            analyze.assert_not_called()
        self.assertEqual(sessions.load(self.runtime, prepared.id).segments[0]["translation"], "Salut")

    def test_previous_output_is_not_mistaken_for_a_completed_generation(self):
        prepared = self.prepared()
        prepared.data["last_output"] = "older.mp3"
        prepared.save()

        def generate(*values):
            yield None, None, None, None, "Starting…"

        with patch.object(self.handlers, "generate_dub", generate):
            self.client.post(f"{API}/auto", json={"session": prepared.id})
            self.assertIn("without a dub", self.wait("auto")["error"])

    def test_changed_target_language_requires_review(self):
        prepared = self.prepared()
        prepared.data["target_language"] = "French"
        prepared.save()
        with patch.object(self.handlers, "generate_dub") as generate:
            response = self.client.post(f"{API}/auto", json={"session": prepared.id, "target_language": "German"})
            self.assertIn("Translate again", response.json()["error"])
            generate.assert_not_called()

    def test_inactive_hand_rewrite_setting_does_not_block_analysis(self):
        prepared = self.prepared()

        def analyze(*values):
            yield prepared.id, {}, [], "Found lines.", None

        with patch.object(self.handlers, "analyze_source", analyze), patch.object(self.handlers, "generate_dub", self.generate):
            response = self.client.post(f"{API}/auto", json={"source": str(self.media), "rewrite": False,
                                                             "rewrite_by_hand": True})
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(self.wait("auto")["error"])
        response = self.client.post(f"{API}/auto", json={"source": str(self.media), "rewrite": True,
                                                         "rewrite_by_hand": True, "target_language": pipeline.SAME_LANGUAGE})
        self.assertIn("rewrite by hand", response.json()["error"])

    def test_analysis_failure_never_generates(self):
        def analyze(*values):
            yield None, None, None, "**Error:** Bad transcript", None

        with patch.object(self.handlers, "analyze_source", analyze), patch.object(self.handlers, "generate_dub") as generate:
            self.client.post(f"{API}/auto", json={"source": str(self.media)})
            self.assertEqual(self.wait("auto")["error"], "Bad transcript")
            generate.assert_not_called()

    def test_stop_between_analysis_and_dub(self):
        entered, release = Event(), Event()
        prepared = self.prepared()

        def analyze(*values):
            entered.set()
            release.wait(5)
            yield prepared.id, {}, [], "Found lines.", None

        with patch.object(self.handlers, "analyze_source", analyze), patch.object(self.handlers, "generate_dub") as generate:
            self.client.post(f"{API}/auto", json={"source": str(self.media)})
            self.assertTrue(entered.wait(5))
            try:
                refused = self.client.post(f"{API}/dub", json={"session": prepared.id})
                self.assertIn("Auto mode is running", refused.json()["error"])
                self.client.post(f"{API}/interrupt", json={"kind": "auto"})
            finally:
                release.set()
            self.assertEqual(self.wait("auto")["status"], "Interrupted.")
            generate.assert_not_called()

    def test_batch_continues_after_failure_and_deduplicates(self):
        prepared = self.prepared()
        projects.save(self.runtime, "Good", prepared.id, None, {"preset": "Strict"})
        with patch.object(self.handlers, "generate_dub", self.generate):
            response = self.client.post(f"{API}/batch", json={"names": ["Missing", "Good", "Good"]})
            self.assertEqual(response.status_code, 200)
            batch = self.wait_batch()
        self.assertEqual([item["state"] for item in batch["items"]], ["failed", "done"])

    def test_project_io_failure_does_not_stop_the_batch(self):
        prepared = self.prepared()
        projects.save(self.runtime, "Good", prepared.id, None, {})
        original_load = projects.load

        def load(runtime, name):
            if name == "Broken":
                raise OSError("Project storage failed")
            return original_load(runtime, name)

        with patch.object(projects, "load", load), patch.object(self.handlers, "generate_dub", self.generate):
            self.client.post(f"{API}/batch", json={"names": ["Broken", "Good"]})
            batch = self.wait_batch()
        self.assertEqual([item["state"] for item in batch["items"]], ["failed", "done"])

    def test_batch_stop_keeps_remaining_projects_from_starting(self):
        prepared = self.prepared()
        for name in ("First", "Second"):
            projects.save(self.runtime, name, prepared.id, None, {})
        entered, release = Event(), Event()
        calls = []

        def generate(session_id, *settings):
            calls.append(session_id)
            entered.set()
            release.wait(5)
            yield from self.generate(session_id, *settings)

        with patch.object(self.handlers, "generate_dub", generate):
            self.client.post(f"{API}/batch", json={"names": ["First", "Second"]})
            self.assertTrue(entered.wait(5))
            try:
                self.client.post(f"{API}/batch/stop")
            finally:
                release.set()
            batch = self.wait_batch()
        self.assertEqual(len(calls), 1)
        self.assertEqual([item["state"] for item in batch["items"]], ["stopped", "stopped"])

    def test_missing_text_and_manual_steps_stop_for_review(self):
        prepared = self.prepared()
        prepared.segments[0]["translation"] = ""
        prepared.save()
        with patch.object(self.handlers, "generate_dub") as generate:
            self.client.post(f"{API}/auto", json={"session": prepared.id})
            self.assertIn("no new text", self.wait("auto")["error"])
            generate.assert_not_called()
        response = self.client.post(f"{API}/auto", json={"source": str(self.media),
                                                         "pipeline": pipeline.PIPELINE_MANUAL})
        self.assertIn("Paste", response.json()["error"])

    def wait_batch(self):
        for _ in range(100):
            batch = self.client.get(f"{API}/batch").json()["batch"]
            if not batch["active"]:
                return batch
            time.sleep(.05)
        self.fail("Batch never finished")


class WorkshopAutoTests(PageTests):
    def test_auto_and_batch_require_studio(self):
        for route in ("auto", "batch"):
            self.assertIn("Studio", self.client.post(f"{API}/{route}", json={}).json()["error"])
