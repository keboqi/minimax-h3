"""The Dubbing page's API: sources, edits, runs, results and the Workshop's limits."""

from __future__ import annotations

import os
from pathlib import Path
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from h3_app.dubbing import pipeline, session as sessions
from h3_app.dubbing.results import Player, player_folder, save_player
from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.dubbing.handlers import DubbingHandlers, DubbingServices
from h3_ui.dubbing.page import install_dubbing_page

API = "/dubbing/api"


class PageTests(unittest.TestCase):
    gpu = False

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs")
        env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        env.start()
        self.addCleanup(env.stop)
        store = WorkspaceStore(self.root / "store")
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        JOBS.store = store
        self.media = self.root / "media.wav"
        self.media.write_bytes(b"media")
        self.handlers = DubbingHandlers(DubbingServices(lambda: self.runtime, None,
                                                        lambda mode: [self.media] if mode == "Audio" else [],
                                                        gpu=self.gpu))
        app = FastAPI()
        install_owner_cookie(app, store)
        install_dubbing_page(app, self.handlers, gpu=self.gpu)
        self.client = TestClient(app)

    def session(self, lines, **data):
        created = sessions.create(self.runtime, self.media, video=False)
        created.data.update(data)
        created.segments = lines
        created.save()
        return created

    def wait(self, kind, session=""):
        for _ in range(100):
            run = self.client.get(f"{API}/run", params={"session": session}).json()["runs"][kind]
            if run and not run["active"]:
                return run
            time.sleep(0.05)
        self.fail(f"{kind} never finished")


class WorkshopPageTests(PageTests):
    def test_page_and_assets_are_served(self):
        self.assertEqual(self.client.get("/dubbing/").status_code, 200)
        self.assertEqual(self.client.get("/dubbing/app.js").headers["content-type"], "application/javascript")
        self.assertIn("export function bindPlayers", self.client.get("/dubbing/player.js").text)
        self.assertEqual(self.client.get("/dubbing/secret.py").status_code, 404)

    def test_boot_offers_only_cloud_steps_and_the_same_language(self):
        boot = self.client.get(f"{API}/boot").json()
        self.assertFalse(boot["gpu"])
        self.assertEqual(boot["same"], pipeline.SAME_LANGUAGE)
        self.assertTrue(all(p["cloud"] for p in boot["pipelines"]))
        self.assertIn("Children", boot["rewrite_styles"])

    def test_sources_are_uploads_media_or_sessions_never_other_paths(self):
        uploaded = self.client.post(f"{API}/upload", files={"file": ("talk.mp3", b"x")}).json()["source"]
        self.assertEqual((uploaded["name"], uploaded["kind"]), ("talk.mp3", "Audio"))
        refused = self.client.post(f"{API}/upload", files={"file": ("notes.txt", b"x")}).json()
        self.assertIn("isn't a video or audio file", refused["error"])
        self.assertEqual(self.client.get(f"{API}/source", params={"path": str(self.media)}).json()["source"]["name"],
                         "media.wav")
        secret = self.root / "secret.wav"
        secret.write_bytes(b"x")
        self.assertIn("Pick the source again", self.client.get(f"{API}/source", params={"path": str(secret)}).json()["error"])
        self.assertIn("Pick the source again",
                      self.client.post(f"{API}/analyze", json={"source": str(secret)}).json()["error"])

    def test_line_edits_follow_the_same_language_rule(self):
        created = self.session([
            {"start": 1, "end": 2, "speaker": "speaker1", "source": "Add three grams.", "translation": "Add three grams.",
             "dub": False}], target_language=pipeline.SAME_LANGUAGE, source_language="English")
        key = {"index": 0, "start": 1, "speaker": "speaker1", "source": "Add three grams."}
        body = {"session": created.id, "key": key, "start": "0:01.0", "end": "0:02.0", "speaker": "speaker1",
                "original": "Add three grams.", "translation": "Add two grams.", "dub": False}
        line = self.client.patch(f"{API}/line", json=body).json()["session"]["board"]["lines"][0]
        self.assertEqual((line["translation"], line["dub"], line["edited"]), ("Add two grams.", True, True))
        session = self.client.get(f"{API}/session", params={"id": created.id}).json()["session"]
        self.assertTrue(session["same"])
        self.assertEqual(session["label"], "English · edited")
        self.assertEqual(session["readiness"]["voiced"], 1)
        bad = self.client.patch(f"{API}/line", json={**body, "start": "soon"}).json()
        self.assertIn("Start and end need times", bad["error"])

    def test_switch_add_delete_and_table(self):
        created = self.session([{"start": 1, "end": 2, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        off = self.client.post(f"{API}/line/dub", json={"session": created.id, "line": 0, "dub": False}).json()
        self.assertFalse(off["session"]["board"]["lines"][0]["dub"])
        added = self.client.post(f"{API}/line/add", json={"session": created.id, "focus": 0}).json()
        self.assertEqual(len(added["session"]["board"]["lines"]), 2)
        self.assertEqual(added["focus"], 1)
        removed = self.client.post(f"{API}/line/delete", json={"session": created.id, "focus": 1}).json()
        self.assertEqual(len(removed["session"]["board"]["lines"]), 1)
        rows = [[1, "0:01.0", "0:02.0", "speaker1", "Hi", "Bonjour", True]]
        table = self.client.put(f"{API}/lines", json={"session": created.id, "rows": rows}).json()
        self.assertEqual(table["session"]["board"]["lines"][0]["translation"], "Bonjour")

    def test_combining_picked_lines(self):
        created = self.session([
            {"start": 1, "end": 2, "speaker": "speaker1", "source": "赫恩篇", "translation": "Horn story"},
            {"start": 3, "end": 4.3, "speaker": "speaker1", "source": "盗版危机", "translation": "Piracy crisis",
             "dub": False},
            {"start": 5, "end": 6, "speaker": "speaker1", "source": "第三集", "translation": ""},
            {"start": 8, "end": 9, "speaker": "speaker2", "source": "Hi", "translation": "Salut"}])
        merged = self.client.post(f"{API}/line/merge", json={"session": created.id, "picked": [2, 0, 1]}).json()
        lines = merged["session"]["board"]["lines"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(merged["focus"], 0)
        self.assertEqual((lines[0]["s"], lines[0]["e"], lines[0]["source"], lines[0]["translation"], lines[0]["dub"]),
                         (1, 6, "赫恩篇盗版危机第三集", "Horn story Piracy crisis", True))
        self.assertIn("Combined 3 lines", merged["status"])
        other = self.client.post(f"{API}/line/merge", json={"session": created.id, "picked": [0, 1]}).json()
        self.assertIn("one speaker", other["error"])
        self.assertIn("two or more", self.client.post(f"{API}/line/merge",
                                                      json={"session": created.id, "picked": [0]}).json()["error"])

    def test_runs_report_progress_and_open_the_analyzed_session(self):
        analyzed = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])

        def analyze(*values):
            self.assertTrue(values[-1])  # Rewrite reaches the handler.
            yield None, None, None, "Transcribing with Gemini", None
            yield analyzed.id, {}, [], "Found 1 line(s).", None

        with patch.object(self.handlers, "analyze_source", analyze):
            started = self.client.post(f"{API}/analyze", json={"source": str(self.media), "rewrite": True,
                                                               "target_language": pipeline.SAME_LANGUAGE}).json()
            self.assertEqual(started["run"]["task"], "analyze")
            run = self.wait("analyze")
        self.assertEqual((run["session"], run["status"], run["error"]), (analyzed.id, "Found 1 line(s).", None))

        def failing(*values):
            yield None, None, None, "**Error:** Gemini said no.", None

        with patch.object(self.handlers, "analyze_source", failing):
            self.client.post(f"{API}/analyze", json={"source": str(self.media)})
            self.assertEqual(self.wait("analyze")["error"], "Gemini said no.")

    def test_analysis_advances_only_the_attached_project(self):
        saved = self.client.post(f"{API}/project/save", json={"name": "Example", "source": str(self.media)}).json()
        project = saved["project"]
        for attempt, expected in enumerate(("Example", "Example", None)):
            if attempt == 2:
                # Another browser still holds the original selection.
                project = saved["project"]
            analyzed = self.session([])
            with patch.object(pipeline, "analyze", return_value=iter([pipeline.Analysis(analyzed, "Done.")])):
                self.client.post(f"{API}/analyze", json={"source": str(self.media), "project": project,
                                                         "pipeline": pipeline.PIPELINE_MANUAL})
                run = self.wait("analyze")
            self.assertEqual((run["project"], run["session"], run["error"]), (expected, analyzed.id, None))
            if expected:
                project = {"name": run["project"], "session": run["session"]}

    def test_prepared_audio_download_with_unicode_filename(self):
        speech = self.root / "日本語-speech.mp3"
        speech.write_bytes(b"vocals")
        def prepare(*values):
            yield str(speech), "Ready."

        with patch.object(self.handlers, "prepare_speech", prepare):
            self.client.post(f"{API}/prepare-audio", json={"source": str(self.media)})
            run = self.wait("analyze")
        self.assertIsNone(run["error"])
        download = self.client.get(f"{API}/run-file", params={"id": run["id"]})
        self.assertEqual(download.content, b"vocals")
        self.assertIn("filename*=utf-8''", download.headers["content-disposition"])

    def test_studio_only_steps_say_where_to_go(self):
        created = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        for path, body in (("preview", {"session": created.id, "line": 0}), ("dub", {"session": created.id}),
                           ("align", {"session": created.id})):
            self.assertIn("Studio", self.client.post(f"{API}/{path}", json=body).json()["error"])

    def test_prompts_by_hand(self):
        created = self.session([{"start": 0, "end": 1.5, "speaker": "speaker1", "source": "Hello there.",
                                 "translation": "Hello there.", "dub": False}],
                               target_language=pipeline.SAME_LANGUAGE, source_language="English")
        prompt = self.client.post(f"{API}/source-prompt", json={"target_language": pipeline.SAME_LANGUAGE,
                                                                "translate": True}).json()["prompt"]
        self.assertIn("without translating it", prompt)
        copied = self.client.post(f"{API}/lines-prompt", json={"session": created.id, "picked": [],
                                                               "target_language": pipeline.SAME_LANGUAGE,
                                                               "style": "Children"}).json()
        self.assertIn("Rewrite each line", copied["prompt"])
        applied = self.client.post(f"{API}/lines-reply", json={"session": created.id, "request": copied["request"],
                                                              "reply": '["Hi, friends."]'}).json()
        line = applied["session"]["board"]["lines"][0]
        self.assertEqual((line["translation"], line["dub"]), ("Hi, friends.", True))
        plan = self.client.post(f"{API}/plan", json={"target_language": pipeline.SAME_LANGUAGE, "rewrite": True,
                                                     "rewrite_style": "Teens", "pipeline": pipeline.PIPELINE_GEMINI,
                                                     "translator": pipeline.TRANSLATOR_GEMINI}).json()["steps"]
        self.assertEqual([step["title"] for step in plan][-1], "Rewrite for Teens")

    def test_result_files_and_share_links(self):
        created = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        folder = self.runtime.outputs_dir / "dubbing"
        folder.mkdir(parents=True)
        output = folder / "日本語.mp3"
        output.write_bytes(b"mix")
        (folder / "日本語.french.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSalut\n", encoding="utf-8")
        tracks = player_folder(output)
        tracks.mkdir(parents=True)
        dub, original = tracks / "dub.mp3", tracks / "original.mp3"
        dub.write_bytes(b"dub")
        original.write_bytes(b"original")
        cue = {"start": 0, "end": 1, "shown_start": 0, "shown_end": 1, "speaker": "speaker1", "source": "Hi",
               "translation": "Salut", "translation_spoken": True, "line_number": 1, "take_problem": "",
               "source_words": [], "translation_words": []}
        save_player(tracks, Player(None, dub, original, None, 1, [cue], "French", "English"))
        sessions.modify(created, lambda current: current.data.update(last_output=str(output)))
        result = self.client.get(f"{API}/session", params={"id": created.id}).json()["session"]["result"]
        self.assertIn("h3-dub-player", result["html"])
        self.assertEqual([f["name"] for f in result["files"]], ["日本語.mp3", "日本語.french.srt"])
        self.assertFalse(result["separated"])
        self.assertEqual(self.client.get(f"{API}/track", params={"session": created.id, "name": "dub"}).content, b"dub")
        self.assertEqual(self.client.get(f"{API}/track", params={"session": created.id, "name": "secrets"}).status_code, 404)
        download = self.client.get(f"{API}/output", params={"session": created.id, "name": output.name})
        self.assertEqual(download.content, b"mix")
        self.assertIn("filename*=utf-8''", download.headers["content-disposition"])
        self.assertEqual(self.client.get(f"{API}/output", params={"session": created.id, "name": "../x"}).status_code, 404)
        shared = self.client.post(f"{API}/share", json={"session": created.id}).json()["session"]
        self.assertIn("/dub-share/", shared["share"])
        stopped = self.client.delete(f"{API}/share", params={"session": created.id}).json()["session"]
        self.assertIsNone(stopped["share"])


class StudioPageTests(PageTests):
    gpu = True

    def test_studio_shares_from_workshop_only(self):
        created = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        self.assertIn("Workshop", self.client.post(f"{API}/share", json={"session": created.id}).json()["error"])
        self.assertTrue(self.client.get(f"{API}/boot").json()["gpu"])

    def test_preview_needs_a_line_of_the_session(self):
        created = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        self.assertIn("Pick a line", self.client.post(f"{API}/preview", json={"session": created.id, "line": 5}).json()["error"])

    def test_interrupt_with_nothing_running(self):
        self.assertEqual(self.client.post(f"{API}/interrupt", json={"kind": "dub", "session": "x"}).json()["message"],
                         "Nothing is running.")
        self.assertIn("Unknown step", self.client.post(f"{API}/interrupt", json={"kind": "other"}).json()["error"])


if __name__ == "__main__":
    unittest.main()
