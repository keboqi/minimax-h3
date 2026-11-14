"""The Soundstage page's illustrated video: settings, planning, pasted plans, drawing, its stills and saving.

The planner runs for real with its writer replaced; drawing is replaced by a fake that keeps two
stills, and saving to Media renders with FFmpeg replaced.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
import gradio as gr

from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.jobs import JOBS
from h3_app.provenance import write_snapshot
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.drama3.bindings import Drama3Actions
from h3_ui.drama3.handlers import Drama3Handlers, Drama3Services
from h3_ui.drama3.page import install_soundstage_page
from h3_app.drama3.visuals import Shot
from h3_ui.drama3 import visuals
from h3_ui.drama3.visuals import DEFAULT_STYLE, STYLE_PRESETS, VideoExport

API = "/soundstage/api"
NAME = {"name": "Night Market"}
PLAN = {"style": "ink wash", "continuity": {"Mia": "yellow raincoat"},
        "shots": [{"start": 0, "transition": "cut", "beat": "The stall", "references": ["Mia"],
                   "prompt": "Mia at her plum stall in the rain."},
                  {"start": 4, "transition": "dissolve", "beat": "The radio", "references": [],
                   "prompt": "An old radio glowing on a shelf."}]}


class Drama3PageVideoTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        drama3 = self.root / "outputs" / "h3" / "drama3"
        drama3.mkdir(parents=True)
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs" / "h3",
                                       input_dir=self.root / "input",
                                       prompt_systems={"Drama 3 visuals": Path("prompts/prompt_drama3_visuals.txt")})
        self.mix = drama3 / f"night_20261108-120000_{uuid.uuid4().hex}.mp3"
        self.mix.write_bytes(b"ID3")
        write_snapshot(self.mix, {"family": DRAMA3_FAMILY, "settings": {
            "duration": 8.0, "chunks": [{"name": "Mia", "text": "Plums!", "take": "a" * 32, "start": 0, "end": 2,
                                         "lines": [{"text": "Plums!", "start": 0, "end": 2}]}], "sounds": []}})
        self.renders = []
        deps = SimpleNamespace(resolve_result=lambda name: drama3 / Path(name).name, runtime_config=lambda: self.runtime)
        controller = SimpleNamespace(fields={"qwen__steps": SimpleNamespace(value=8), "qwen__seed": SimpleNamespace(value=-1)})
        self.export = VideoExport(controller, deps, self.fake_export, [("Qwen Image", "qwen"), ("FLUX 3", "flux3")])
        handlers = Drama3Handlers(Drama3Services(runtime_config=lambda: self.runtime, stage_file=lambda *args: "",
                                                 generation_services=lambda: None, free_gpu=lambda: None))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, Drama3Actions.of(handlers, interrupt=lambda *args, **kwargs: ""), store=self.store,
                                output_dir=self.root / "outputs", takes_dir=lambda: self.root / "takes",
                                runtime_config=lambda: self.runtime, video_export=self.export)
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")
        # An 8-second mix holds at most four shots; the sample plan has two.
        self.store.save_soundstage_project("Night Market", {"text": "Mia: Plums!", "shown_slots": 1, "video_limit": 2})
        self.store.mark_soundstage_generated("Night Market", str(self.mix))

    def fake_export(self, result_name, engine, style, limit, use_references, *values):
        """Draw two images (the second dissolving in at 4 s) and keep them as the mix's stills."""
        from PIL import Image

        self.renders.append((result_name, engine, style, limit, use_references, values))
        yield gr.skip(), "Shot 1 of 2"
        images = []
        for index, color in enumerate(("red", "blue")):
            image = self.root / f"drawn-{len(self.renders)}-{index}.png"
            Image.new("RGB", (64, 36), color).save(image)
            images.append(image)
        shots = [Shot(0, 96, False, "Mia at her stall.", "The stall"), Shot(96, 96, True, "The radio.", "The radio")]
        record = visuals.keep_stills(self.mix, self.runtime.output_dir, shots, images, {"model": engine})
        yield str(record), "Drew 2 images."

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertNotIn("error", data)
        return data

    def project(self):
        return self.ok(self.client.get(f"{API}/project", params=NAME))

    def wait(self):
        for _ in range(300):
            run = self.project()["video_run"]
            if run and not run["active"]:
                return run
            time.sleep(0.02)
        self.fail("The video run did not finish.")

    def test_boot_and_defaults(self):
        boot = self.ok(self.client.get(f"{API}/boot"))["video"]
        self.assertEqual([engine["id"] for engine in boot["engines"]], ["qwen", "flux3"])
        self.assertEqual(boot["default_style"], DEFAULT_STYLE)
        video = self.project()["video"]
        self.assertEqual((video["engine"], video["preset"], video["limit"], video["continuity"]),
                         ("qwen", DEFAULT_STYLE, 2, True))
        self.assertEqual(video["plan"], {"rows": [], "stale": False, "saved": False})
        self.assertIsNone(video["latest"])

    def test_settings_are_saved_and_kept_in_range(self):
        video = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={
            "video_style": "Oil paint, thick impasto", "video_limit": 500, "video_engine": "flux3",
            "video_continuity": False}))["video"]
        self.assertEqual((video["style"], video["preset"], video["limit"], video["engine"], video["continuity"]),
                         ("Oil paint, thick impasto", "Custom", 64, "flux3", False))

    def test_subtitles_are_shown_by_default_and_their_size_is_kept_in_range(self):
        self.assertEqual(self.ok(self.client.get(f"{API}/boot"))["video"]["subtitle_sizes"], [50, 200, 10])
        video = self.project()["video"]
        self.assertEqual((video["subtitles"], video["subtitle_size"]), (True, 100))
        video = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={
            "video_subtitles": False, "video_subtitle_size": 130}))["video"]
        self.assertEqual((video["subtitles"], video["subtitle_size"]), (False, 130))
        for asked, kept in ((10, 50), (900, 200)):
            video = self.ok(self.client.patch(f"{API}/project/video", params=NAME,
                                              json={"video_subtitle_size": asked}))["video"]
            self.assertEqual(video["subtitle_size"], kept)

    def test_planning_saves_a_plan_for_these_settings(self):
        with patch("h3_app.drama3.visual_plan.write_prompt", return_value=(json.dumps(PLAN), "ok")) as writer:
            run = self.ok(self.client.post(f"{API}/project/video/plan", params=NAME,
                                           json={"writer": {"gemini_api_key": "k"}}))["video_run"]
            self.assertEqual(run["kind"], "plan")
            run = self.wait()
        self.assertIsNone(run["error"])
        self.assertTrue(run["status"].startswith("Planned 2 shots"))
        writer.assert_called_once()
        rows = self.project()["video"]["plan"]["rows"]
        self.assertEqual([(row["beat"], row["transition"]) for row in rows], [("The stall", "Cut"), ("The radio", "Dissolve")])
        # Another Art direction has no plan yet: the table empties and says why.
        plan = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_style": "Pixel art"}))["video"]["plan"]
        self.assertEqual((plan["rows"], plan["stale"]), ([], True))

    def test_drawing_waits_for_planning_on_the_same_mix(self):
        """Drawing uses the plan and saving the stills, so a mix runs one of them at a time."""
        from threading import Event

        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        release = Event()

        def slow_writer(*args, **kwargs):
            release.wait(10)
            return json.dumps(PLAN), "ok"

        with patch("h3_app.drama3.visual_plan.write_prompt", side_effect=slow_writer):
            self.ok(self.client.post(f"{API}/project/video/plan", params=NAME, json={}))
            refused = self.client.post(f"{API}/project/video/draw", params=NAME)
            self.assertIn("already being planned", refused.json()["error"])
            release.set()
            self.wait()

    def test_a_pasted_plan_is_checked_then_saved(self):
        refused = self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": "not json"})
        self.assertIn("error", refused.json())
        data = self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        self.assertEqual(data["message"], "Applied the pasted 2-shot plan.")
        self.assertEqual(len(data["video"]["plan"]["rows"]), 2)
        prompt = self.ok(self.client.post(f"{API}/project/video/prompt", params=NAME))["prompt"]
        self.assertIn("Plums!", prompt)

    def draw(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        return self.wait()

    def test_drawing_keeps_stills_that_play_with_the_mix_and_renders_nothing(self):
        refused = self.client.post(f"{API}/project/video/draw", params=NAME)
        self.assertIn("no shot plan", refused.json()["error"])
        run = self.draw()
        self.assertIsNone(run["error"])
        self.assertEqual(run["status"], "Drew 2 images.")
        name, engine, style, limit, continuity, values = self.renders[-1]
        self.assertEqual((name, engine, style, limit, continuity), (self.mix.name, "qwen", STYLE_PRESETS[DEFAULT_STYLE], 2, True))
        self.assertEqual(values, (8, -1))  # The Image tab's defaults until the studio hands its settings over.
        video = self.project()["video"]
        self.assertIsNone(video["latest"])  # Nothing is rendered until it is saved to Media.
        self.assertEqual(list(self.mix.parent.glob("*.mp4")), [])
        shots = video["stills"]["shots"]
        self.assertEqual([(shot["start"], shot["end"], shot["fade"], shot["beat"]) for shot in shots],
                         [(0.0, 4.0, 0.0, "The stall"), (4.0, 8.0, 0.5, "The radio")])
        still = self.client.get(shots[1]["url"])
        self.assertEqual((still.status_code, still.headers["content-type"]), (200, "image/jpeg"))
        self.assertIn("immutable", still.headers["cache-control"])
        self.assertEqual(self.client.get(f"{API}/video/still", params={**NAME, "file": "../secret.jpg"}).status_code, 404)

    def test_drawing_again_replaces_the_stills(self):
        first = self.draw() and self.project()["video"]["stills"]
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        self.wait()
        second = self.project()["video"]["stills"]
        self.assertNotEqual(first["revision"], second["revision"])
        self.assertEqual(self.client.get(first["shots"][0]["url"]).status_code, 404)
        folder = visuals.stills_dir(self.mix, self.runtime.output_dir)
        self.assertEqual(len(list(folder.glob("*.jpg"))), 2)

    def test_saving_to_media_renders_the_stills_on_request(self):
        refused = self.client.post(f"{API}/project/video/save", params=NAME)
        self.assertIn("Draw the images first", refused.json()["error"])
        self.draw()
        rendered = []

        def fake_render(audio, shots, images, output_dir, subtitles=None):
            rendered.append(([(shot.first, shot.frames, shot.dissolve) for shot in shots], images, output_dir,
                             subtitles))
            video = Path(output_dir) / f"{Path(audio).stem}_illustrated_test_{len(rendered)}.mp4"
            video.write_bytes(b"MP4")
            return video

        def save():
            with patch.object(visuals, "render_video", fake_render), patch.object(visuals, "which", lambda name: name):
                self.ok(self.client.post(f"{API}/project/video/save", params=NAME))
                return self.wait()

        run = save()
        self.assertIsNone(run["error"])
        self.assertIn("to Media", run["status"])
        shots, images, output_dir, subtitles = rendered[0]
        self.assertEqual(shots, [(0, 96, False), (96, 96, True)])
        self.assertTrue(all(Path(image).suffix == ".jpg" and Path(image).is_file() for image in images))
        self.assertEqual(Path(output_dir), self.mix.parent)
        # The subtitles shown on the page are burned in: on by default, at 100%.
        self.assertIn("{\\b1}Mia:{\\b0} Plums!", subtitles)
        self.assertIn(",28,", subtitles)
        latest = self.project()["video"]["latest"]
        self.assertTrue(latest["current"])
        self.assertEqual(latest["subtitles"], 100)  # the page asks for a new video when its own differ
        # Larger subtitles, then none, are what the next videos get.
        self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_subtitle_size": 150}))
        save()
        self.assertIn(",42,", rendered[1][3])
        self.assertEqual(self.project()["video"]["latest"]["subtitles"], 150)
        self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_subtitles": False}))
        save()
        self.assertIsNone(rendered[2][3])
        latest = self.project()["video"]["latest"]
        self.assertEqual((latest["subtitles"], latest["current"]), (0, True))
        self.assertEqual(self.client.get(latest["url"]).content, b"MP4")
        from h3_app.provenance import read_snapshot

        snapshot = read_snapshot(self.mix.parent / latest["file"])
        self.assertEqual((snapshot["settings"]["source"], snapshot["settings"]["model"]), (self.mix.name, "qwen"))
        # New stills make the saved video an older one, to save again.
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        self.wait()
        self.assertFalse(self.project()["video"]["latest"]["current"])

    def test_captured_image_settings_are_used_for_their_owner(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        from h3_app.workspace_store import COOKIE

        owner = self.store.verify_owner(self.client.cookies.get(COOKIE))
        self.export.settings[owner] = {"qwen__steps": 20, "qwen__seed": 7}
        self.export.settings["someone-else"] = {"qwen__steps": 99}
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        self.wait()
        self.assertEqual(self.renders[-1][5], (20, 7))

    def test_without_a_mix_there_is_nothing_to_draw(self):
        self.store.save_soundstage_project("Draft", {"text": "Mia: Hi.", "shown_slots": 1})
        refused = self.client.post(f"{API}/project/video/plan", params={"name": "Draft"}, json={})
        self.assertIn("Speak the project first", refused.json()["error"])
        self.assertEqual(self.client.get(f"{API}/video", params={"name": "Draft"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
