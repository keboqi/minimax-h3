"""The Song page's JSON API: each model's draft and options, the writer's review, takes and runs."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui import application as gradio_app
from h3_ui.job_bindings import owned_generation
from h3_ui.song.page import JOB_ORDER, SongPageState, install_song_page, score_and_mode

API = "/song/api"


class SongPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The page takes its options, their defaults and their limits from the built views.
        cls.demo = gradio_app.build_ui()
        cls.pages = next(install.keywords["pages"] for install in cls.demo.h3_routes
                         if getattr(install, "func", None) is install_song_page)
        cls.services = cls.pages.services

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.generated, self.written, self.stopped = [], [], []
        self.writer_answer = None
        self.hold = None
        for engine in ("music", "yue2"):
            previous = JOBS.callbacks.get(engine)
            JOBS.callbacks[engine] = owned_generation(self.generator(engine), engine, input_names=JOB_ORDER[engine])
            self.addCleanup(lambda engine=engine, previous=previous:
                            JOBS.callbacks.update({engine: previous}) if previous else JOBS.callbacks.pop(engine, None))
        self.state = SongPageState(replace(
            self.services, root=lambda: self.root / "pages",
            enhance={"music": self.writer("music"), "yue2": self.writer("yue2")},
            take_facts={"music": self.facts, "yue2": self.facts},
            interrupt=self.interrupt, library=lambda: None))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_song_page(app, self.state)
        self.client = TestClient(app)
        self.boot = self.ok(self.client.get(f"{API}/boot"))

    # Fakes -----------------------------------------------------------------
    def generator(self, engine):
        def generate(*values):
            self.generated.append((engine, dict(zip(JOB_ORDER[engine], values, strict=True))))
            target = self.root / "out" / f"{engine}-{len(self.generated)}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"RIFF0000WAVE")
            yield {"__type__": "update"}, "Sampling"
            if self.hold is not None:
                self.hold.wait(5)
            yield str(target), "<b>Done</b> in 3 s"
        return generate

    def writer(self, engine):
        def enhance(*arguments):
            self.written.append((engine, arguments))
            if self.writer_answer is not None:
                return self.writer_answer
            return arguments[0] + " Warm guitars.", "[Verse]\nNew line", "Song draft ready."
        return enhance

    def facts(self, path):
        return {"seed": 42, "seconds": 125, "score": "X:1\nK:G\nGABc|" if "yue2" in str(path) else None,
                "score_mode": "melody"}

    def interrupt(self, request, family, job_id):
        self.stopped.append((family, job_id))
        return "Stopping."

    # Helpers ---------------------------------------------------------------
    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def refused(self, response, words):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(words, response.json()["error"])

    def post(self, route, **body):
        return self.ok(self.client.post(f"{API}{route}", json=body))

    def draft(self, engine="music", **changes):
        return self.post("/draft", engine=engine, changes=changes)["state"]

    def change(self, action="edit", **changes):
        return self.post("/settings", action=action, changes=changes)["state"]

    def settle(self, kind):
        for _ in range(200):
            answer = self.ok(self.client.get(f"{API}/run?state=true"))
            run = answer["runs"][kind]
            if run and not run["active"]:
                return answer
            time.sleep(0.02)
        self.fail(f"The {kind} run did not finish.")

    # Tests -----------------------------------------------------------------
    def test_boot_draws_both_models_with_their_view_defaults(self):
        self.assertEqual([engine["id"] for engine in self.boot["engines"]], ["music", "yue2"])
        state = self.boot["state"]
        self.assertEqual(state["engine"], "music")
        self.assertEqual(state["brief"]["duration"]["value"], 120)
        self.assertEqual(state["brief"]["duration"]["maximum"], 300)
        self.assertNotIn("mode", state["brief"])
        self.assertFalse(state["readiness"]["ready"])
        self.assertIn("Describe your song", state["readiness"]["message"])
        self.assertIn("h3-song-check", state["check"])
        self.assertEqual({row["name"] for row in state["fields"]},
                         {f"music__{name}" for name in ("model", "seed", "steps", "cfg", "ar_cfg", "top_k", "tiled")})
        self.assertEqual(len(self.boot["starters"]["yue2"]), 3)
        self.assertEqual(self.boot["durations"]["yue2"][-1], [300, "5 min"])
        self.assertIn("Stronger chorus", self.boot["quick_fixes"])

    def test_each_model_keeps_its_own_draft_and_readiness(self):
        state = self.draft(lyrics="[Verse]\nHello")
        self.assertIn("Add a music description", state["readiness"]["message"])
        state = self.draft(description="Warm indie pop")
        self.assertTrue(state["readiness"]["ready"])
        state = self.change("engine", engine="yue2")
        self.assertEqual(state["draft"]["description"], "")
        self.assertIn("mode", state["brief"])
        self.assertEqual(state["brief"]["duration"]["maximum"], 900)
        state = self.draft("yue2", description="Indie rock")
        self.assertIn("Add lyrics", state["readiness"]["message"])
        state = self.change("engine", engine="music")
        self.assertEqual(state["draft"]["lyrics"], "[Verse]\nHello")

    def test_options_are_checked_against_the_view_limits(self):
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"music__duration": 301}}), "between 1 and 300")
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"music__vocals": "Robot"}}), "listed options")
        self.refused(self.client.post(f"{API}/settings", json={"changes": {"nope": 1}}), "Unknown setting")
        state = self.change(music__steps=12, music__vocals="Duet")
        self.assertEqual(state["changed"], {"1": 1})
        self.assertEqual(state["brief"]["vocals"]["value"], "Duet")
        state = self.change("restore")
        self.assertEqual(state["changed"], {})
        self.assertEqual(state["brief"]["vocals"]["value"], "Duet")  # The brief is not a generation setting.

    def test_write_previews_then_accept_and_undo(self):
        self.draft(description="Last train home", notes="Stronger chorus")
        self.change(music__vocals="Female lead", music__lyrics_policy="Polish my lyrics")
        self.post("/write")
        answer = self.settle("write")
        run = answer["runs"]["write"]
        self.assertEqual(run["suggestion"], {"description": "Last train home Warm guitars.", "lyrics": "[Verse]\nNew line"})
        self.assertIn("Preview ready", run["status"])
        self.assertEqual(answer["state"]["draft"]["description"], "Last train home")
        engine, arguments = self.written[-1]
        self.assertEqual(engine, "music")
        # The writer gets what the Gradio Write button sent, in its order.
        self.assertEqual(arguments[0], "Last train home")
        self.assertEqual(arguments[4:7], (None, None, None))
        self.assertEqual(arguments[11:16], ("Female lead", "Auto", "Auto", "Polish my lyrics", "Stronger chorus"))
        state = self.post("/accept")["state"]
        self.assertEqual(state["draft"]["lyrics"], "[Verse]\nNew line")
        self.assertTrue(state["undo"])
        self.refused(self.client.post(f"{API}/accept"), "no suggestion")
        state = self.post("/undo")["state"]
        self.assertEqual(state["draft"]["description"], "Last train home")
        self.assertFalse(state["undo"])

    def test_keep_original_and_a_draft_changed_while_writing(self):
        self.draft(description="Idea")
        self.post("/write")
        self.settle("write")
        state = self.post("/keep")["state"]
        self.assertEqual(state["draft"]["description"], "Idea")
        self.post("/write")
        self.settle("write")
        self.draft(description="Edited meanwhile")
        self.refused(self.client.post(f"{API}/accept"), "draft changed")

    def test_a_failed_write_is_reported(self):
        self.writer_answer = ("Idea", "", "Prompt enhancement failed: no key.")
        self.draft(description="Idea")
        self.post("/write")
        run = self.settle("write")["runs"]["write"]
        self.assertIn("no key", run["error"])
        self.assertIsNone(run["suggestion"])

    def test_generate_records_a_take_with_the_draft_that_made_it(self):
        self.draft(description="Warm pop", lyrics="[Chorus]\nLa")
        self.change(music__seed=7, music__duration=60)
        self.post("/generate")
        answer = self.settle("generate")
        engine, values = self.generated[-1]
        self.assertEqual(engine, "music")
        self.assertEqual(values["caption"], "Warm pop")
        self.assertEqual(values["duration"], 60)
        self.assertEqual(values["seed"], 7)
        self.assertEqual(answer["runs"]["generate"]["status"], "MiniMax Music 3 · Done in 3 s")
        takes = answer["state"]["takes"]
        self.assertEqual(len(takes["items"]), 1)
        self.assertEqual(takes["items"][0]["label"], "Take 1 · 2:05 · seed 42")
        self.assertEqual(takes["a"], 0)
        self.assertIn("1 take this session", takes["summary"])
        self.assertEqual(self.client.get(takes["items"][0]["url"]).content, b"RIFF0000WAVE")
        # A second take, then compare: the details name what changed.
        self.draft(lyrics="[Chorus]\nLa la")
        self.post("/generate")
        takes = self.settle("generate")["state"]["takes"]
        self.assertEqual((takes["a"], takes["b"]), (1, 0))
        self.assertIn("lyrics", takes["details"])
        # Back to take 1's draft, and its seed.
        self.post("/takes/select", a=0, b=1)
        state = self.post("/takes/reuse-draft", index=0)["state"]
        self.assertEqual(state["draft"]["lyrics"], "[Chorus]\nLa")
        self.assertEqual(state["brief"]["duration"]["value"], 60)
        state = self.post("/takes/use-seed", index=0)["state"]
        self.assertEqual(state["seed"], 42)

    def test_another_owner_cannot_read_a_take(self):
        self.draft(description="Warm pop")
        self.post("/generate")
        url = self.settle("generate")["state"]["takes"]["items"][0]["url"]
        stranger = TestClient(self.client.app)
        self.assertEqual(stranger.get(url).status_code, 404)

    def test_write_and_generate_applies_without_review_and_starts_a_take(self):
        self.draft(description="Idea")
        self.post("/write-generate")
        run = self.settle("write")["runs"]["write"]
        self.assertIn("Applied without review. Starting a take.", run["status"])
        answer = self.settle("generate")
        self.assertEqual(self.generated[-1][1]["caption"], "Idea Warm guitars.")
        self.assertTrue(answer["state"]["undo"])
        self.assertEqual(len(answer["state"]["takes"]["items"]), 1)

    def test_yue2_reuses_a_score_in_the_mode_it_was_planned(self):
        self.change("engine", engine="yue2")
        self.draft("yue2", description="Indie rock", lyrics="[Verse]\nRoad")
        self.change(yue2__mode="off")
        self.post("/generate")
        state = self.settle("generate")["state"]
        self.assertEqual(self.generated[-1][1]["mode"], "off")
        self.assertTrue(state["takes"]["items"][0]["has_score"])
        self.assertEqual(state["takes"]["last_score"], "X:1\nK:G\nGABc|")
        state = self.post("/takes/reuse-score", index="latest")["state"]
        self.assertEqual(state["draft"]["abc"], "X:1\nK:G\nGABc|")
        self.assertEqual(state["brief"]["mode"]["value"], "melody")
        self.assertEqual(score_and_mode({"score": "X", "draft": {}}, "off")[1], "full")

    def test_pictures_are_music_3_only_and_at_most_three(self):
        files = []
        for index in range(4):
            path = self.root / f"p{index}.png"
            Image.new("RGB", (32, 32), "red").save(path)
            files.append(("files", (path.name, path.read_bytes(), "image/png")))
        self.refused(self.client.post(f"{API}/references/upload", files=files), "at most 3")
        state = self.ok(self.client.post(f"{API}/references/upload", files=files[:2]))["state"]
        self.assertEqual(len(state["draft"]["references"]), 2)
        self.draft(description="Idea")
        self.post("/write")
        self.settle("write")
        self.assertTrue(all(self.written[-1][1][4:6]))
        state = self.post("/references/remove", index=0)["state"]
        self.assertEqual(len(state["draft"]["references"]), 1)
        self.change("engine", engine="yue2")
        self.refused(self.client.post(f"{API}/references/upload", files=files[:1]), "Music 3")

    def test_media_hands_a_picture_to_music_3(self):
        from types import SimpleNamespace
        from h3_app.workspace_store import request_owner

        name = request_owner(SimpleNamespace(cookies=dict(self.client.cookies), headers={}), self.store)
        source = self.root / "media.png"
        Image.new("RGB", (32, 32), "blue").save(source)
        self.change("engine", engine="yue2")
        self.assertIn("Visual inspiration", self.state.receive(name, str(source)))
        state = self.ok(self.client.get(f"{API}/run?state=true"))["state"]
        self.assertEqual(state["engine"], "music")
        self.assertEqual(len(state["draft"]["references"]), 1)

    def test_empty_clears_only_the_current_model(self):
        self.draft(description="Idea", lyrics="La", notes="Calmer")
        self.draft("yue2", description="Rock")
        state = self.post("/empty")["state"]
        self.assertEqual((state["draft"]["description"], state["draft"]["lyrics"], state["draft"]["notes"]), ("", "", ""))
        state = self.change("engine", engine="yue2")
        self.assertEqual(state["draft"]["description"], "Rock")

    def test_a_running_take_stays_visible_and_stoppable_from_the_other_model(self):
        self.assertEqual(self.post("/interrupt")["message"], "Nothing is running.")
        self.hold = threading.Event()
        self.addCleanup(self.hold.set)
        self.draft(description="Warm pop")
        self.post("/generate")
        for _ in range(200):
            run = self.ok(self.client.get(f"{API}/run"))["runs"]["generate"]
            if run["status"] == "Sampling":
                break
            time.sleep(0.02)
        self.change("engine", engine="yue2")
        run = self.ok(self.client.get(f"{API}/run"))["runs"]["generate"]
        self.assertTrue(run["active"])
        self.assertEqual(run["engine"], "music")
        self.draft("yue2", description="Rock", lyrics="[Verse]\nRoad")
        self.refused(self.client.post(f"{API}/generate"), "already being made")
        self.assertEqual(self.post("/interrupt")["message"], "Stopping.")
        self.assertEqual(self.stopped[-1][0], "music")
        self.hold.set()
        state = self.settle("generate")["state"]
        self.assertEqual(state["takes"]["items"], [])  # The take belongs to Music 3's history.
        self.assertEqual(len(self.change("engine", engine="music")["takes"]["items"]), 1)

    def test_keys_are_not_written_to_disk(self):
        self.post("/writer", changes={"gemini_api_key": "secret-key", "backend": "OpenAI-compatible"})
        saved = next((self.root / "pages").rglob("settings.json")).read_text(encoding="utf-8")
        self.assertNotIn("secret-key", saved)
        self.assertIn("OpenAI-compatible", saved)
        state = self.ok(self.client.get(f"{API}/run?state=true"))["state"]
        self.assertEqual(state["writer"]["gemini_api_key"], "secret-key")
        self.assertIn("openai_base_url", {row["name"] for row in state["writer_rows"]})


if __name__ == "__main__":
    unittest.main()
