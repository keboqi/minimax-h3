"""The Soundstage page's illustrated video: settings, planning, pasted plans, drawing, its stills and saving.

The planner runs for real with its writer replaced; drawing is replaced by a fake that keeps two
stills, and saving to Media renders with FFmpeg replaced.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
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

    def fake_export(self, result_name, engine, style, limit, use_references, orientation, only, *values):
        """Draw two images (the second dissolving in at 4 s) and keep them as the mix's stills."""
        from PIL import Image

        self.renders.append((result_name, engine, style, limit, use_references, values))
        self.drawn_orientation = orientation
        self.drawn_only = json.loads(only)
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
                         ("qwen", DEFAULT_STYLE, 2, False))
        self.assertEqual(video["plan"], {"rows": [], "subjects": [], "art_style": "", "counts": {}, "stale": False, "saved": False, "saved_settings": None})
        self.assertIsNone(video["latest"])

    def test_workshop_draws_gpt_images_before_speech_and_carries_them_to_the_mix(self):
        from PIL import Image
        from h3_ui.drama3.project_visuals import carry_project_stills
        from h3_ui.drama3.visuals import video_export

        drawn = []
        def generate(engine, mode, prompt, refs, captured):
            from h3_app.jobs import CURRENT_JOB
            self.assertFalse(CURRENT_JOB.get().has_gpu)
            path = self.runtime.output_dir / f"api-{uuid.uuid4().hex}.png"
            Image.new("RGB", (8, 8)).save(path)
            drawn.append(path)
            yield [str(path)], "Complete"

        controller = SimpleNamespace(fields={}, generate=generate)
        export = video_export(controller, self.export.deps)
        export.engines = [("GPT Image", "gpt_image")]
        handlers = Drama3Handlers(Drama3Services(runtime_config=lambda: self.runtime, stage_file=None,
                                                 generation_services=None, free_gpu=None))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, Drama3Actions.of(handlers, interrupt=None), store=self.store,
                                output_dir=self.root / "outputs", takes_dir=lambda: self.root / "takes",
                                runtime_config=lambda: self.runtime, video_export=export, gpu=False)
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")
        self.store.mark_soundstage_generated("Night Market", "")
        plan = {"shots": [dict(cue=0, offset=offset, transition="cut", beat="Plums", prompt="A plum stall.")
                          for offset in (0, 0.5)]}
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(plan)}))
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME, json={"only": [0]}))
        self.assertIsNone(self.wait()["error"])
        before = self.project()["video"]
        self.assertEqual(len(drawn), 1)
        self.assertEqual(before["plan"]["counts"]["drawn"], 1)
        for row in before["plan"]["rows"]:
            if row["image"]:
                self.assertEqual(self.client.get(row["image"]).status_code, 200)
        self.store.mark_soundstage_generated("Night Market", str(self.mix))
        carry_project_stills(self.store, "Night Market", None, str(self.mix), export)
        after = self.project()["video"]
        self.assertEqual(after["plan"]["counts"]["drawn"], 1)
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME, json={"only": "missing"}))
        self.assertIsNone(self.wait()["error"])
        self.assertEqual(self.project()["video"]["plan"]["counts"]["drawn"], 2)
        self.assertEqual(len(drawn), 2)

    def test_style_previews_are_drawn_with_the_projects_model_and_served(self):
        from PIL import Image

        boot = self.ok(self.client.get(f"{API}/boot"))["video"]
        self.assertEqual([name for group in boot["style_groups"] for name in group["presets"]], list(boot["styles"]))
        self.assertEqual(set(boot["style_uses"]), set(boot["styles"]))
        drawn = []

        def fake_preview(engine, presets, *values):
            drawn.append((engine, json.loads(presets), values))
            image = self.root / "preview.png"
            Image.new("RGB", (64, 36), "green").save(image)
            file = visuals.keep_preview(self.runtime.output_dir, engine, "Anime", image)
            yield file, "Drew 1 style preview."

        self.export.preview = fake_preview
        self.assertEqual(self.project()["video"]["previews"], {})
        run = self.ok(self.client.post(f"{API}/project/video/preview", params=NAME, json={"presets": ["Anime"]}))["video_run"]
        self.assertEqual(run["kind"], "preview")
        self.assertIsNone(self.wait()["error"])
        self.assertEqual(drawn, [("qwen", ["Anime"], (8, -1))])
        previews = self.project()["video"]["previews"]
        self.assertEqual(list(previews), ["Anime"])
        response = self.client.get(previews["Anime"])
        self.assertEqual((response.status_code, response.headers["content-type"]), (200, "image/jpeg"))
        self.assertEqual(self.client.get(f"{API}/video/preview", params={"engine": "qwen", "file": "x.jpg"}).status_code, 404)
        # Another model has its own previews.
        self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_engine": "flux3"}))
        self.assertEqual(self.project()["video"]["previews"], {})
        refused = self.client.post(f"{API}/project/video/preview", params=NAME, json={"presets": ["Nope"]})
        self.assertIn("Pick style presets", refused.text)

    def test_settings_are_saved_and_kept_in_range(self):
        video = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={
            "video_style": "Oil paint, thick impasto", "video_limit": 500, "video_engine": "flux3",
            "video_continuity": False}))["video"]
        self.assertEqual((video["style"], video["preset"], video["limit"], video["engine"], video["continuity"]),
                         ("Oil paint, thick impasto", "Custom", 64, "flux3", False))
        self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_continuity": True}))
        self.assertTrue(self.project()["video"]["continuity"])

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
        self.assertTrue(rows[0]["prompt"])
        # Another style and frame use the same plan: only the drawing prompt changes.
        video = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={
            "video_style": STYLE_PRESETS["Pixel art"], "video_orientation": "portrait"}))["video"]
        self.assertEqual((video["preset"], video["orientation"]), ("Pixel art", "portrait"))
        self.assertEqual([(row["beat"], row["prompt"]) for row in video["plan"]["rows"]],
                         [(row["beat"], row["prompt"]) for row in rows])
        self.assertFalse(video["plan"]["stale"])
        refused = self.client.patch(f"{API}/project/video", params=NAME, json={"video_orientation": "square"})
        self.assertIn("Unknown video_orientation", refused.text)
        # Another Minimum shots has no plan yet: the table empties and names the saved plan's.
        plan = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_limit": 3}))["video"]["plan"]
        self.assertEqual((plan["rows"], plan["stale"], plan["saved_settings"]), ([], True, {"limit": 2}))
        plan = self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_limit": 2}))["video"]["plan"]
        self.assertEqual((len(plan["rows"]), plan["stale"], plan["saved_settings"]), (2, False, None))

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
            batch = self.ok(self.client.get(f"{API}/batch"))
            self.assertTrue(next(row for row in batch["projects"] if row["name"] == NAME["name"])["busy"])
            refused = self.client.post(f"{API}/project/video/draw", params=NAME)
            self.assertIn("already being planned", refused.json()["error"])
            refused = self.client.post(f"{API}/project/video/plan", params=NAME, json={})
            self.assertIn("already being planned", refused.json()["error"])
            release.set()
            self.assertIsNone(self.wait()["error"])  # A refused request cannot invalidate the active planner.

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
        self.assertEqual((name, engine, style, limit, continuity), (self.mix.name, "qwen", STYLE_PRESETS[DEFAULT_STYLE], 2, False))
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

    def test_completed_images_are_served_while_the_next_shot_is_generating(self):
        from PIL import Image

        next_started, release = Event(), Event()
        calls = []

        def generate(*args):
            index = len(calls)
            calls.append(args)
            if index:
                next_started.set()
                if not release.wait(5):
                    raise RuntimeError("Timed out waiting for the intermediate-state check.")
            path = self.root / f"progressive-{index}.png"
            Image.new("RGB", (64, 36), "blue").save(path)
            yield [str(path)], "Complete"

        self.export.controller.generate = generate

        def progressive(result_name, engine, style, limit, continuity, orientation, only, *values):
            yield from visuals.draw_stills(self.export.controller, self.export.deps, result_name, engine,
                                          style, limit, continuity, orientation, *values, only=json.loads(only))

        self.export.generate = progressive
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        try:
            self.assertTrue(next_started.wait(5))
            project = self.project()
            self.assertTrue(project["video_run"]["active"])
            shots = project["video"]["stills"]["shots"]
            self.assertEqual(len(shots), 1)
            self.assertEqual(self.client.get(shots[0]["url"]).status_code, 200)
        finally:
            release.set()
        self.assertIsNone(self.wait()["error"])
        self.assertEqual(len(self.project()["video"]["stills"]["shots"]), 2)

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

        def fake_render(audio, shots, images, output_dir, subtitles=None, size=None):
            self.assertEqual(size, (1280, 720))  # drawn in the default landscape frame
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
        self.assertEqual(self.drawn_orientation, "landscape")

    def test_without_a_mix_there_is_nothing_to_draw(self):
        self.store.save_soundstage_project("Draft", {"text": "Mia: Hi.", "shown_slots": 1})
        refused = self.client.post(f"{API}/project/video/draw", params={"name": "Draft"}, json={})
        self.assertIn("Speak the project", refused.json()["error"])
        self.assertEqual(self.client.get(f"{API}/video", params={"name": "Draft"}).status_code, 404)

    def test_workshop_plan_is_shared_edited_and_adapted_after_speech(self):
        from h3_ui.drama3.project_visuals import planning_export

        self.store.save_soundstage_project("Draft", {"text": "Mia: Plums!", "shown_slots": 1, "video_limit": 2})
        app = FastAPI()
        install_owner_cookie(app, self.store)
        handlers = Drama3Handlers(Drama3Services(lambda: self.runtime, None, None, None))
        install_soundstage_page(app, Drama3Actions.of(handlers, interrupt=None), store=self.store,
                                output_dir=self.runtime.output_dir, takes_dir=lambda: self.root / "takes",
                                runtime_config=lambda: self.runtime, gpu=False,
                                video_export=planning_export(self.runtime))
        client = TestClient(app)
        self.assertIsNotNone(client.get(f"{API}/boot").json()["video"])
        plan = {"shots": [{"cue": 0, "offset": offset, "transition": "cut", "beat": "Plums", "prompt": "Plums on a stall."}
                          for offset in (0, 0.5)]}
        args = {"name": "Draft"}
        applied = self.ok(client.post(f"{API}/project/video/reply", params=args, json={"reply": json.dumps(plan)}))
        self.assertTrue(applied["video"]["timing_pending"])
        self.assertEqual(len(applied["video"]["plan"]["rows"]), 2)
        self.ok(client.patch(f"{API}/project/video/shot", params=args, json={"index": 1, "prompt": "A bright plum."}))
        saved = self.store.soundstage_project("Draft")[0]["visual_plan"]
        self.assertEqual(saved["plan"]["shots"][1]["prompt"], "A bright plum.")
        # A newer Workshop project plan replaces an older local Studio sidecar too.
        visuals.write_plan(self.mix, "old style", 2, PLAN)
        self.store.mark_soundstage_generated("Draft", str(self.mix))
        project = self.ok(self.client.get(f"{API}/project", params=args))
        self.assertEqual(project["video"]["plan"]["rows"][1]["start"], 1)
        self.assertEqual(visuals.read_plan(self.mix)["plan"]["shots"][1]["prompt"], "A bright plum.")

    def auto_client(self, speak_callback=None):
        from dataclasses import replace
        from h3_app.generation.results import GenerationUpdate
        from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS

        def speak(*values):
            self.store.mark_soundstage_generated(values[-1], str(self.mix))
            yield GenerationUpdate(str(self.mix), "Done.")

        def prepare(*values):
            yield (*[gr.skip()] * (3 * DRAMA3_VOICE_SLOTS), "Ready.",
                   *[gr.skip()] * (2 * DRAMA3_SOUND_SLOTS), "Ready.", "Ready.")

        handlers = Drama3Handlers(Drama3Services(lambda: self.runtime, None, None, None))
        actions = replace(Drama3Actions.of(handlers, interrupt=None), generate_drama3=speak_callback or speak,
                          prepare_drama3_cast=prepare)
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_soundstage_page(app, actions, store=self.store, output_dir=self.runtime.output_dir,
                                takes_dir=lambda: self.root / "takes", runtime_config=lambda: self.runtime,
                                video_export=self.export)
        return TestClient(app)

    def anchored_images(self):
        from PIL import Image
        plan = {"shots": [dict(cue=0, offset=offset, transition="cut", beat="Plums", prompt="A plum stall.")
                          for offset in (0, .5)]}
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(plan)}))
        shots = visuals.plan_shots_for(visuals.read_plan(self.mix), 8, STYLE_PRESETS[DEFAULT_STYLE])
        images = []
        for index in range(2):
            file = self.root / f"kept-{index}.png"
            Image.new("RGB", (32, 18), "red").save(file)
            images.append(file)
        visuals.keep_stills(self.mix, self.runtime.output_dir, shots, images, {}, total_frames=8 * 24)
        return visuals.read_stills(self.mix, self.runtime.output_dir)

    def test_auto_wording_fix_reuses_the_storyboard_and_images_on_the_new_mix(self):
        from h3_app.generation.results import GenerationUpdate
        record = self.anchored_images()
        original_ids = [row["id"] for row in self.store.soundstage_project("Night Market")[0]["visual_plan"]["plan"]["shots"]]
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(text="Mia: Fresh plums!", prepared=False, auto_illustrate=True)
        updated_mix = self.mix.with_name("corrected.mp3")
        updated_mix.write_bytes(b"ID3")
        write_snapshot(updated_mix, {"family": DRAMA3_FAMILY, "settings": {"duration": 6,
            "chunks": [dict(name="Mia", text="Fresh plums!", start=0, end=4)]}})

        def speak(*values):
            self.store.mark_soundstage_generated(values[-1], str(updated_mix))
            yield GenerationUpdate(str(updated_mix), "Done.")

        client = self.auto_client(speak)
        with patch("h3_ui.drama3.page_video.plan_shots") as planner, \
                patch("h3_app.prompt_writers.drama3.enhance_drama3_text") as writer:
            self.ok(client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_auto(client)
        self.assertIsNone(run["error"], run)
        planner.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(self.renders, [])
        migrated = visuals.read_stills(updated_mix, self.runtime.output_dir)
        self.assertEqual([item["shot_id"] for item in migrated["shots"]], original_ids)
        self.assertEqual(migrated["total_frames"], 6 * 24)
        for before, after in zip(record["shots"], migrated["shots"]):
            self.assertEqual((visuals.stills_dir(self.mix, self.runtime.output_dir) / before["image"]).read_bytes(),
                             (visuals.stills_dir(updated_mix, self.runtime.output_dir) / after["image"]).read_bytes())

    def test_alignment_repair_moves_existing_images_without_redrawing(self):
        record = self.anchored_images()
        write_snapshot(self.mix, {"family": DRAMA3_FAMILY, "settings": {"duration": 8,
            "chunks": [dict(name="Mia", text="Plums!", start=0, end=4)]}})
        project = self.project()
        moved = visuals.read_stills(self.mix, self.runtime.output_dir)
        self.assertEqual([item["first_frame"] for item in moved["shots"]], [0, 48])
        self.assertNotEqual(moved["revision"], record["revision"])
        self.assertEqual([item["image"] for item in moved["shots"]], [item["image"] for item in record["shots"]])
        self.assertEqual(project["video"]["plan"]["counts"], {"drawn": 2, "changed": 0, "todo": 0})

    def test_auto_wording_fix_without_illustrations_does_not_rewrite_the_script(self):
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(text="Mia: Fresh plums!", prepared=False, auto_illustrate=False)
        client = self.auto_client()
        with patch("h3_app.prompt_writers.drama3.enhance_drama3_text") as writer:
            self.ok(client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_auto(client)
        self.assertIsNone(run["error"], run)
        writer.assert_not_called()

    def test_numeric_tags_and_cast_names_share_cue_identity(self):
        from h3_ui.drama3.project_visuals import can_reuse_script, project_cues
        from h3_app.drama3.visual_plan import check_plan_reply
        from h3_app.drama3.visual_timing import preview_settings
        data = {"text": "<|speaker:0|> Plums!", "voice_0_save_name": "Mia", "shown_slots": 1}
        cues = project_cues(data, lambda: self.runtime)
        self.assertEqual(cues, [dict(speaker="Mia", text="Plums!")])
        plan, _ = check_plan_reply(json.dumps({"shots": [dict(cue=0, offset=0, beat="Market",
            prompt="A plum stall.", transition="cut")]}), preview_settings(data["text"], cues=cues), 1)
        data.update(text="<|speaker:0|> Fresh plums!", visual_plan={"plan": plan})
        self.assertTrue(can_reuse_script(data, runtime=lambda: self.runtime))

    def test_speaker_edits_do_not_relabel_the_plan_playing_with_older_audio(self):
        self.anchored_images()
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(text="Leo: Plums!", prepared=False)
        project = self.project()
        self.assertNotIn("error", project["video"]["plan"])
        self.assertEqual(project["result"]["lines"][0]["speaker"], "Mia")
        saved = self.store.soundstage_project("Night Market")[0]["visual_plan"]["plan"]
        self.assertEqual(saved["script_cues"], [dict(speaker="Mia", text="Plums!")])

    def test_invalid_saved_plan_does_not_replace_the_last_valid_sidecar(self):
        self.anchored_images()
        before = visuals.read_plan(self.mix)
        with self.store.edit_soundstage_project("Night Market") as data:
            data["visual_plan"]["plan"]["shots"][0]["prompt"] = ""
        project = self.project()
        self.assertTrue(project["video"]["plan"]["error"])
        self.assertEqual(project["video"]["plan"]["rows"], [])
        self.assertEqual(visuals.read_plan(self.mix), before)

    def wait_auto(self, client, batch=False):
        for _ in range(300):
            data = client.get(f"{API}/batch" if batch else f"{API}/run", params=NAME).json()
            run = data.get("batch" if batch else "run")
            if run and not run["active"]:
                return run
            time.sleep(0.02)
        self.fail("Auto did not finish.")

    def test_auto_illustrations_reuse_existing_plan_even_with_another_minimum(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(prepared=True, auto_illustrate=True, video_limit=64)
        client = self.auto_client()
        with patch("h3_ui.drama3.page_video.plan_shots") as planner:
            self.ok(client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_auto(client)
        self.assertIsNone(run["error"])
        planner.assert_not_called()
        self.assertEqual(len(self.renders), 1)
        self.assertEqual(self.drawn_only, "missing")
        self.assertEqual(self.store.soundstage_project("Night Market")[0]["video_limit"], 2)

    def test_batch_option_plans_and_draws_after_speech(self):
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(prepared=True, auto_illustrate=False)
        client = self.auto_client()
        with patch("h3_ui.drama3.page_video.plan_shots", return_value=(PLAN, [None, None])) as planner:
            self.ok(client.post(f"{API}/batch", json={"names": ["Night Market"], "illustrate": True}))
            batch = self.wait_auto(client, batch=True)
        self.assertEqual(batch["items"][0]["state"], "done", batch)
        planner.assert_called_once()
        self.assertEqual(len(self.renders), 1)

    def test_auto_preserves_the_script_of_an_untimed_workshop_plan(self):
        self.store.mark_soundstage_generated("Night Market", "")
        self.store.save_soundstage_project("Night Market", {"text": "Mia: Plums!", "video_limit": 2,
                                                            "auto_illustrate": True, "prepared": False})
        plan = {"shots": [dict(cue=0, offset=offset, transition="cut", beat="Plums", prompt="A plum stall.")
                          for offset in (0, 0.5)]}
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(plan)}))
        client = self.auto_client()
        with patch("h3_ui.drama3.page_video.plan_shots") as planner, \
                patch("h3_app.prompt_writers.drama3.enhance_drama3_text") as writer:
            self.ok(client.post(f"{API}/auto", params=NAME, json={}))
            run = self.wait_auto(client)
        self.assertIsNone(run["error"])
        planner.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(len(self.renders), 1)

    def test_changed_or_cleared_workshop_script_stays_editable_and_rejects_stale_plan_edits(self):
        self.store.mark_soundstage_generated("Night Market", "")
        plan = {"shots": [dict(cue=0, offset=offset, transition="cut", beat="Plums", prompt="A plum stall.")
                          for offset in (0, 0.5)]}
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(plan)}))
        for text in ("Mia: A different story.", ""):
            with self.subTest(text=text):
                with self.store.edit_soundstage_project("Night Market") as data:
                    data["text"] = text
                project = self.project()
                self.assertEqual(project["video"]["plan"]["rows"], [])
                self.assertTrue(project["video"]["plan"]["error"])
                reply = self.client.patch(f"{API}/project/video/shot", params=NAME,
                                          json={"index": 0, "prompt": "Changed image."})
                self.assertEqual(reply.status_code, 400)
                saved = self.store.soundstage_project("Night Market")[0]["visual_plan"]
                self.assertEqual(saved["plan"]["shots"][0]["prompt"], "A plum stall.")

    def test_auto_blocks_manual_planning_pasting_and_edits_until_drawing_finishes(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        with self.store.edit_soundstage_project("Night Market") as data:
            data.update(prepared=True, auto_illustrate=True)
        started, release = Event(), Event()
        generate = self.export.generate

        def slow_draw(*args):
            started.set()
            release.wait(10)
            yield from generate(*args)

        self.export.generate = slow_draw
        client = self.auto_client()
        try:
            self.ok(client.post(f"{API}/auto", params=NAME, json={}))
            self.assertTrue(started.wait(5))
            for route, body in (("plan", {}), ("reply", {"reply": json.dumps(PLAN)})):
                reply = client.post(f"{API}/project/video/{route}", params=NAME, json=body)
                self.assertEqual(reply.status_code, 400, reply.text)
                self.assertIn("Wait for speech or Auto", reply.json()["error"])
            reply = client.patch(f"{API}/project/video/shot", params=NAME,
                                 json={"index": 0, "prompt": "An unrelated image."})
            self.assertEqual(reply.status_code, 400)
        finally:
            release.set()
        self.assertIsNone(self.wait_auto(client)["error"])

    def test_refresh_with_an_old_snapshot_cannot_restore_a_superseded_plan(self):
        from h3_ui.drama3.project_visuals import planning_mix, save_project_plan

        save_project_plan(self.store, "Night Market", self.mix, "comic", 2, PLAN)
        old_data = self.store.soundstage_project("Night Market")[0]
        updated = {**PLAN, "shots": [{**row, "prompt": "A newly chosen image."} for row in PLAN["shots"]]}
        save_project_plan(self.store, "Night Market", self.mix, "comic", 2, updated)
        planning_mix("Night Market", old_data, str(self.mix), self.export, store=self.store)
        self.assertEqual([row["prompt"] for row in visuals.read_plan(self.mix)["plan"]["shots"]],
                         [row["prompt"] for row in updated["shots"]])

    def test_plan_rows_carry_status_and_edits_mark_shots_changed(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        plan = self.project()["video"]["plan"]
        self.assertEqual([(row["index"], row["status"], row["image"], row["subjects"]) for row in plan["rows"]],
                         [(0, "todo", None, ["Mia"]), (1, "todo", None, [])])
        self.assertEqual(plan["counts"], {"drawn": 0, "changed": 0, "todo": 2})
        self.assertEqual([(item["kind"], item["key"], item["anchor"], item["uses"]) for item in plan["subjects"]],
                         [("characters", "Mia", "yellow raincoat", 1)])

        edited = self.ok(self.client.patch(f"{API}/project/video/shot", params=NAME,
                                           json={"index": 1, "prompt": " A radio  glowing. ", "transition": "Cut"}))
        row = edited["video"]["plan"]["rows"][1]
        self.assertEqual((row["prompt"], row["transition"]), ("A radio glowing.", "Cut"))
        looks = self.ok(self.client.patch(f"{API}/project/video/subject", params=NAME,
                                          json={"kind": "characters", "key": "Mia", "anchor": "green coat"}))
        self.assertEqual(looks["video"]["plan"]["subjects"][0]["anchor"], "green coat")
        refused = self.client.patch(f"{API}/project/video/shot", params=NAME, json={"index": 5, "prompt": "x"})
        self.assertIn("not in the current shot plan", refused.json()["error"])

    def test_effect_edits_persist_without_invalidating_images(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME))
        self.wait()
        before = self.project()["video"]
        edited = self.ok(self.client.patch(f"{API}/project/video/shot", params=NAME,
                                          json={"index": 0, "effects": ["push_in", "rain"]}))["video"]
        self.assertEqual(edited["plan"]["rows"][0]["effects"], ["push_in", "rain"])
        self.assertEqual(edited["stills"], before["stills"])
        self.assertEqual(edited["plan"]["counts"], before["plan"]["counts"])
        self.assertEqual(self.project()["video"]["plan"]["rows"][0]["effects"], ["push_in", "rain"])
        self.assertEqual(self.client.get("/soundstage/effects.js").status_code, 200)
        self.assertEqual(self.client.get("/soundstage/effects_webgpu.js").status_code, 200)
        self.assertTrue(edited["effects_enabled"])
        self.ok(self.client.patch(f"{API}/project/video", params=NAME, json={"video_effects": False}))
        self.assertFalse(self.project()["video"]["effects_enabled"])
        self.assertFalse(self.project()["video"]["effects_override_reduced_motion"])
        self.ok(self.client.patch(f"{API}/project/video", params=NAME,
                                 json={"video_effects": True, "video_effects_override_reduced_motion": True}))
        self.assertTrue(self.project()["video"]["effects_override_reduced_motion"])

    def test_script_effects_follow_the_spoken_or_estimated_clock(self):
        from h3_app.drama3.results import result_settings, script_fingerprint

        source = "Mia: Plums!"
        settings = {**result_settings(self.mix), "script_mode": "Basic",
                    "script_fingerprint": script_fingerprint(source)}
        write_snapshot(self.mix, {"family": DRAMA3_FAMILY, "settings": settings})
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))

        def preview(text):
            return self.ok(self.client.post(f"{API}/preview", params=NAME,
                                            json={"text": text, "script_mode": "Basic"}))

        self.assertNotIn("effects", preview(source))
        self.ok(self.client.patch(f"{API}/project/video/shot", params=NAME, json={"index": 1, "effects": ["rain"]}))
        actual = preview(source)
        self.assertTrue(actual["actual"])
        self.assertEqual(actual["effects"], [{"index": 1, "start": 4, "end": 8, "effects": ["rain"]}])
        estimated = preview("Mia: Fresh plums!")
        self.assertNotIn("actual", estimated)
        effect, = estimated["effects"]
        self.assertEqual(effect["end"], estimated["duration"])
        self.assertGreater(effect["start"], 0)
        self.assertLess(effect["start"], 4)
        self.assertNotIn("effects", preview(source + "\nLeo: Thanks!"))

    def test_examples_replace_the_previous_plan_and_load_effects_before_speech(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        old_plan = visuals.read_plan(self.mix)
        for lang, opening_effect in (("en", "rain"), ("zh", "snow"), ("zh-latest", "rain")):
            with self.subTest(lang=lang):
                loaded = self.ok(self.client.post(f"{API}/project/example", params=NAME, json={"example": lang}))
                self.assertIsNone(loaded["result"])
                video = loaded["video"]
                self.assertTrue(video["timing_pending"])
                self.assertEqual(video["limit"], 16)
                rows = video["plan"]["rows"]
                self.assertEqual(len(rows), 16)
                self.assertEqual(rows[0]["effects"], [opening_effect, "push_in"])
                self.assertEqual(self.project()["video"]["plan"], video["plan"])
                preview = self.ok(self.client.post(f"{API}/preview", params=NAME, json=loaded["script"]))
                self.assertEqual([clip["index"] for clip in preview["effects"]],
                                 [row["index"] for row in rows if row["effects"]])
        self.assertTrue(self.mix.is_file())
        self.assertEqual(visuals.read_plan(self.mix), old_plan)

    def test_workshop_finds_studio_audio_and_illustrations_through_shared_media_links(self):
        from h3_workshop import link_studio_media

        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        list(self.fake_export(self.mix.name, "qwen", "ink wash", 2, True, "landscape", '"all"'))
        self.runtime.comfy_dir = self.root / "cpu-app" / "ComfyUI"
        alias = self.runtime.comfy_dir / "output" / self.mix.relative_to(self.runtime.output_dir)
        self.store.mark_soundstage_generated("Night Market", str(alias))
        self.assertIsNone(self.project()["result"])
        try:
            link_studio_media(self.runtime)
        except OSError as exc:
            self.skipTest(f"Directory symlinks unavailable: {exc}")
        link_studio_media(self.runtime)  # Startup may prepare the same paths more than once.
        project = self.project()
        self.assertEqual(project["result"]["file"], self.mix.name)
        self.assertEqual(len(project["video"]["stills"]["shots"]), 2)
        self.assertEqual(self.client.get(f"{API}/audio", params=NAME).content, self.mix.read_bytes())
        self.assertEqual(self.client.get(project["video"]["stills"]["shots"][0]["url"]).status_code, 200)

    def test_drawing_some_shots_passes_which_ones(self):
        self.ok(self.client.post(f"{API}/project/video/reply", params=NAME, json={"reply": json.dumps(PLAN)}))
        self.ok(self.client.post(f"{API}/project/video/draw", params=NAME, json={"only": [1]}))
        self.wait()
        self.assertEqual(self.drawn_only, [1])
        for only in ("some", []):
            refused = self.client.post(f"{API}/project/video/draw", params=NAME, json={"only": only})
            self.assertIn("missing ones", refused.json()["error"])


if __name__ == "__main__":
    unittest.main()
