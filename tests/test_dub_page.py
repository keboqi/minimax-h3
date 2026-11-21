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
        self.handlers = DubbingHandlers(DubbingServices(
            lambda: self.runtime, None,
            lambda mode: [self.media] if mode == "Audio" else [],
            gpu=self.gpu,
            generation_services=(lambda: SimpleNamespace()) if self.gpu else None,
        ))
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

    def test_source_duration_is_reused_until_the_file_changes(self):
        def duration():
            return self.client.get(f"{API}/source", params={"path": str(self.media)}).json()["source"]["seconds"]

        with patch("h3_app.media_tools.probe_audio_seconds", side_effect=[None, 2.0, 3.0]) as probe:
            self.assertIsNone(duration())
            self.assertEqual(duration(), 2.0)
            self.assertEqual(duration(), 2.0)
            self.media.write_bytes(b"replacement")
            self.assertEqual(duration(), 3.0)
            self.assertEqual(probe.call_count, 3)

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
            # Rewrite and the Gemini prompt options reach the handler.
            self.assertEqual(values[-3:], (True, False, True))
            yield None, None, None, "Transcribing with Gemini", None
            yield analyzed.id, {}, [], "Found 1 line(s).", None

        with patch.object(self.handlers, "analyze_source", analyze):
            started = self.client.post(f"{API}/analyze", json={"source": str(self.media), "rewrite": True,
                                                               "readings": True,
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

    def test_analysis_of_the_saved_project_source_recovers_its_context(self):
        saved = self.client.post(f"{API}/project/save", json={"name": "Interview", "source": str(self.media)}).json()
        source = saved["session"]["source"]["path"]
        analyzed = self.session([{"start": 0, "end": 1, "source": "Hi", "translation": "Bonjour"}])
        with patch.object(pipeline, "analyze", return_value=iter([pipeline.Analysis(analyzed, "Done.")])):
            self.client.post(f"{API}/analyze", json={"source": source, "pipeline": pipeline.PIPELINE_MANUAL})
            run = self.wait("analyze")
        self.assertEqual((run["project"], run["session"], run["error"]), ("Interview", analyzed.id, None))
        stale = self.client.post(f"{API}/project/save", json={
            "name": "Interview", "session": saved["session"]["id"], "expected_session": saved["session"]["id"],
            "settings": {"target_language": "Japanese"},
        })
        self.assertEqual(stale.status_code, 400)
        self.assertIn("changed", stale.json()["error"])
        loaded = self.client.post(f"{API}/project/load", json={"name": "Interview"}).json()
        self.assertEqual(loaded["session"]["id"], analyzed.id)
        self.assertEqual(loaded["session"]["board"]["lines"][0]["translation"], "Bonjour")

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
        self.assertIn("attached audio or video", prompt)
        self.assertIn('For an audio-only attachment, omit "visual_speaker"', prompt)
        self.assertIn('"face_center": [x, y]', prompt)
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
        self.client.post(f"{API}/project/save", json={"name": "Interview", "session": created.id})
        with patch("h3_ui.dubbing.page.player_html", side_effect=AssertionError("Rendered the deferred player")):
            reopened = self.client.post(f"{API}/project/load", json={"name": "Interview", "include_player": False})
            self.assertEqual(reopened.status_code, 200)
            self.assertEqual(reopened.json()["session"]["result"], {**result, "html": None})
            restored = self.client.get(f"{API}/session", params={"id": created.id, "include_player": False})
            self.assertEqual(restored.json()["session"]["result"], {**result, "html": None})
        player = self.client.get(f"{API}/result-player", params={"session": created.id})
        self.assertEqual(player.json(), {"html": result["html"]})
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

    def test_new_takes_passes_all_selected_lines_to_the_handler(self):
        created = self.session([
            {"start": i, "end": i + 1, "source": f"Line {i}", "translation": f"Text {i}"}
            for i in range(3)])
        calls = []

        def generate(session_id, numbers, *settings):
            calls.append((session_id, numbers))
            yield None, "Fresh takes ready."

        with patch("h3_ui.dubbing.page.owned_generation", lambda callback, family: callback), \
                patch("h3_ui.dubbing.page.run_job", lambda run, owner, session, family, values, callback: callback(*values)), \
                patch.object(self.handlers, "preview_new_take", generate):
            response = self.client.post(f"{API}/preview", json={"session": created.id, "fresh": True,
                                                                 "lines": [2, 0, 2]})
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(self.wait("lines", created.id)["error"])
        self.assertEqual(calls, [(created.id, "1,3")])
        for invalid in ([], [True], [3], "1,3"):
            response = self.client.post(f"{API}/preview", json={"session": created.id, "fresh": True,
                                                                 "lines": invalid})
            self.assertIn("Pick a line", response.json()["error"])

    def test_interrupt_with_nothing_running(self):
        self.assertEqual(self.client.post(f"{API}/interrupt", json={"kind": "dub", "session": "x"}).json()["message"],
                         "Nothing is running.")
        self.assertIn("Unknown step", self.client.post(f"{API}/interrupt", json={"kind": "other"}).json()["error"])

    def test_one_click_lip_sync_and_optional_shot_overrides(self):
        import json as _json
        from h3_app.dubbing.lip_sync import PLAN_FILE, REPORT_FILE
        from h3_app.dubbing.results import player_folder

        created = self.session([{"start": 0, "end": 1, "speaker": "speaker1", "source": "Hi", "translation": "Salut"}])
        out_dir = self.runtime.outputs_dir / "dubbing"
        out_dir.mkdir(parents=True)
        dub_video = out_dir / "clip.mp4"
        dub_video.write_bytes(b"mp4-dub")
        folder = player_folder(dub_video)
        folder.mkdir(parents=True)
        (folder / "alignment.json").write_text("{}", encoding="utf-8")
        (folder / "alignment.npz").write_bytes(b"npz")
        sessions.modify(created, lambda current: current.data.update(last_output=str(dub_video)))

        self.assertTrue(self.client.get(f"{API}/boot").json()["lip_sync"])
        initial = self.client.get(f"{API}/session", params={"id": created.id}).json()["session"]["result"]["lip_sync"]
        self.assertTrue(initial["ready"])
        self.assertIsNone(initial["output"])

        calls = []

        def fake_render(*, dub, manual_overrides, reviewed_plan, **_kwargs):
            self.assertEqual(dub, dub_video)
            calls.append((manual_overrides, reviewed_plan["revision"] if reviewed_plan else None))
            out = dub_video.with_name("clip.lipsync.mp4")
            out.write_bytes(b"mp4-lipsync")
            (folder / REPORT_FILE).write_text(_json.dumps({
                "output": out.name, "backend": "lr-asd", "total_windows": 2,
                "cached_windows": 1 if manual_overrides else 0, "rendered_windows": 1 if manual_overrides else 2,
                "uncertain_count": 0 if manual_overrides else 1,
                "assignments": [] if manual_overrides else [{
                    "shot_index": 0, "speaker": "speaker1", "status": "uncertain",
                    "track_id": None, "face_center": None, "reference_time": 0.5,
                    "source": "lr-asd", "confidence": 0.3, "reason": "Ambiguous match.",
                }],
            }), encoding="utf-8")
            yield out

        def run_job(_run, _owner, _session, _family, values, callback):
            return callback(*_json.loads(_json.dumps(values)))

        with patch("h3_ui.dubbing.page.owned_generation", lambda callback, family: callback), \
                patch("h3_ui.dubbing.page.run_job", run_job), \
                patch("h3_ui.dubbing.handlers.render_dub_lip_sync", fake_render):
            res = self.client.post(f"{API}/lip-sync", json={"session": created.id})
            self.assertEqual(res.status_code, 200)
            done = self.wait("lip-sync", created.id)
            self.assertIsNone(done["error"])
            self.assertEqual(done["file"], "clip.lipsync.mp4")

            after = self.client.get(f"{API}/session", params={"id": created.id}).json()["session"]["result"]["lip_sync"]
            self.assertEqual(after["output"]["name"], "clip.lipsync.mp4")
            self.assertEqual(len(after["review_shots"]), 1)
            self.assertEqual(after["review_shots"][0]["speaker"], "speaker1")

            # Serve lip-synced output both inline and as download.
            dl = self.client.get(f"{API}/output", params={"session": created.id, "name": "clip.lipsync.mp4"})
            self.assertEqual(dl.content, b"mp4-lipsync")
            self.assertIn("content-disposition", dl.headers)
            inline = self.client.get(f"{API}/output", params={"session": created.id, "name": "clip.lipsync.mp4", "inline": "true"})
            self.assertEqual(inline.content, b"mp4-lipsync")
            self.assertNotIn("content-disposition", inline.headers)

            # Apply an optional shot override.
            res2 = self.client.post(f"{API}/lip-sync", json={
                "session": created.id,
                "overrides": [{"shot_index": 0, "speaker": "speaker1", "track_id": 1}],
            })
            self.assertEqual(res2.status_code, 200)
            self.assertIsNone(self.wait("lip-sync", created.id)["error"])

            target = {"session": created.id, "video": initial["video"], "video_version": initial["video_version"]}
            (folder / PLAN_FILE).write_text(_json.dumps({"video": dub_video.name, "revision": "reviewed"}), encoding="utf-8")
            stale = self.client.post(f"{API}/lip-sync", json={**target, "plan_revision": "old"})
            self.assertEqual(stale.status_code, 400)
            confirmed = self.client.post(f"{API}/lip-sync", json={**target, "plan_revision": "reviewed"})
            self.assertEqual(confirmed.status_code, 200)
            self.assertIsNone(self.wait("lip-sync", created.id)["error"])

            new_dub = out_dir / "new-dub.mp4"
            new_dub.write_bytes(b"new-mp4")
            sessions.modify(created, lambda current: current.data.update(last_output=str(new_dub)))
            for action in ("lip-sync-overrides", "lip-sync-detect", "lip-sync"):
                response = self.client.post(f"{API}/{action}", json={**target, "plan_revision": "reviewed"})
                self.assertEqual(response.status_code, 400)
                self.assertIn("dub changed", response.json()["error"])
            self.assertEqual(self.client.get(f"{API}/lip-sync-plan", params=target).status_code, 400)

        self.assertEqual(calls, [
            (None, None),
            ({(0, "speaker1"): 1}, None),
            (None, "reviewed"),
        ])


if __name__ == "__main__":
    unittest.main()
