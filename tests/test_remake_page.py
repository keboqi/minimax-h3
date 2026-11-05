"""The Remake page's JSON API: sources, selections, windows, files and runs."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import JOBS
from h3_app.remake import SECTIONS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_app.catalog import QWEN_IMAGE21_DEFAULTS
from h3_ui.engines.feature import EngineDeps
from h3_ui.engines.remake import RemakeHandlers
from h3_ui.h3.handlers import default_request
from h3_ui.remake_page import install_remake_page

API = "/remake/api"
SIX_SECTIONS = "\n".join(f"{name}:\ntext" for name in SECTIONS)


class RemakePageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = TemporaryDirectory()
        cls.addClassCleanup(cls.shared.cleanup)
        cls.video = Path(cls.shared.name) / "dance.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=14",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=14", "-shortest", str(cls.video)], check=True,
        )

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.runtime = SimpleNamespace(output_dir=self.output_dir, prompt_systems={"Viggle Remake": Path("x")})
        self.interrupted = []
        self.edits = []
        writer = patch("h3_app.remake.runner.enhance_remake_prompt", return_value=(SIX_SECTIONS, "Enhanced."))
        writer.start()
        self.addCleanup(writer.stop)
        self.handlers = RemakeHandlers(EngineDeps(
            runtime_config=lambda: self.runtime, generation_services=lambda: None, stage_file=self.stage,
            interrupt=lambda: None, h3_take=self.take, h3_defaults=lambda: default_request(""),
            qwen_take=self.qwen_take,
        ))
        self.addCleanup(JOBS.callbacks.pop, "remake", None)
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_remake_page(app, self.handlers, interrupt=lambda request, family, job_id: self.interrupted.append(
            (family, job_id)) or "Stopping.")
        self.client = TestClient(app)
        self.client.get(f"{API}/boot")
        picture = self.root / "edited.png"
        Image.new("RGB", (832, 480), "red").save(picture)
        with picture.open("rb") as handle:
            self.picture = self.ok(self.client.post(f"{API}/picture", files={"file": ("edited.png", handle)}))["path"]

    def stage(self, path, category, **_options):
        target = self.input_dir / category / f"{uuid.uuid4().hex}{Path(path).suffix}"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        return str(target.relative_to(self.input_dir))

    def qwen_take(self, request, progress):
        self.edits.append(request)
        if "fail" in request.prompt:
            yield GenerationUpdate([], "Error: no memory\n\ntimings")
            return
        target = self.output_dir / "qwen" / "edited.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.input_dir / request.reference_images[0], target)
        yield GenerationUpdate([str(target)], "Qwen Image 2.1 image 1/1 completed")

    def take(self, values, remake, progress):
        target = self.output_dir / f"{remake.output_prefix or 'h3/single'}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.input_dir / remake.source_video, target)
        yield GenerationUpdate(str(target), "done")

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def source(self):
        with self.video.open("rb") as handle:
            return self.ok(self.client.post(f"{API}/source", files={"file": ("dance.mp4", handle)}))

    def selection(self, source, end=12, max_window=226):
        return {"sha": source["sha"], "start": 0, "end": end, "short_edge": 0, "mix": 0, "max_window": max_window}

    def editor(self, source, **changes):
        return {"video": source["path"], "selection": self.selection(source, end=8), "prompt": SIX_SECTIONS,
                "request": "robot", "pinned": True, "smart": True, "auto_prompts": True, "window": 0, "mode": "Swap",
                "slots": [{"path": self.picture, "pin": 12}, None, None, None], "originals": {}, "use_carried": True,
                **changes}

    def finished_run(self):
        for _ in range(200):
            run = self.ok(self.client.get(f"{API}/run"))["run"]
            if not run["active"]:
                return run
            time.sleep(0.05)
        self.fail("The run never finished.")

    def test_the_edit_models_are_h3_and_qwen(self):
        models = self.ok(self.client.get(f"{API}/boot"))["edit_models"]
        self.assertEqual([model["id"] for model in models], ["qwen", "h3"])

    def finished_edit(self):
        for _ in range(200):
            run = self.ok(self.client.get(f"{API}/quick-edit/run"))["run"]
            if not run["active"]:
                return run
            time.sleep(0.05)
        self.fail("The edit never finished.")

    def test_a_quick_edit_runs_qwen_on_the_picture_and_keeps_its_size(self):
        self.ok(self.client.post(f"{API}/quick-edit", json={"picture": self.picture, "prompt": "make it blue"}))
        run = self.finished_edit()
        self.assertIsNone(run["error"])
        self.assertEqual(self.client.get(f"{API}/file", params={"path": run["result_url"]}).status_code, 200)
        self.assertNotEqual(run["result_url"], self.picture)
        edit = self.edits[0]
        self.assertEqual((edit.mode, edit.prompt), ("Image edit", "make it blue"))
        self.assertTrue(edit.match_input_size)
        self.assertEqual(edit.turbo_variant, QWEN_IMAGE21_DEFAULTS["turbo_variant"])
        self.assertTrue(any(job.family == "remake_edit" for job in JOBS.records.values()))

    def test_a_failed_quick_edit_reports_why(self):
        self.ok(self.client.post(f"{API}/quick-edit", json={"picture": self.picture, "prompt": "fail"}))
        run = self.finished_edit()
        self.assertEqual(run["error"], "no memory")
        self.assertIsNone(run["result_url"])

    def test_a_quick_edit_needs_a_prompt_and_the_owners_picture(self):
        for body in ({"picture": self.picture, "prompt": "  "},
                     {"picture": str(self.root / "secret.png"), "prompt": "x"}):
            self.assertEqual(self.client.post(f"{API}/quick-edit", json=body).status_code, 400)
        self.assertEqual(self.edits, [])

    def test_a_source_is_kept_and_described(self):
        source = self.source()
        self.assertEqual(source["end"], 9.42)
        self.assertIn("Source audio found", source["audio_note"])
        self.assertTrue(Path(source["path"]).is_file())
        self.assertEqual(self.client.get(f"{API}/file", params={"path": source["path"]}).status_code, 200)

    def test_files_outside_the_owners_folder_are_refused(self):
        secret = self.root / "secret.txt"
        secret.write_text("no")
        self.assertEqual(self.client.get(f"{API}/file", params={"path": str(secret)}).status_code, 404)
        other = TestClient(self.client.app)
        other.get(f"{API}/boot")
        source = self.source()
        self.assertEqual(other.get(f"{API}/file", params={"path": source["path"]}).status_code, 404)

    def test_paths_from_the_browser_must_be_the_owners(self):
        source = self.source()
        secret = self.root / "secret.png"
        Image.new("RGB", (64, 64), "blue").save(secret)
        foreign = {"slots": [{"path": str(secret), "pin": 0}, None, None, None]}
        selection = self.selection(source)
        project = self.ok(self.client.post(f"{API}/split", json={"selection": selection, "video": source["path"]}))
        for route in ("check", "generate", "window/save", "window/mark", "rewrite"):
            with self.subTest(route=route):
                response = self.client.post(f"{API}/{route}", json=self.editor(source, selection=selection, **foreign))
                self.assertEqual(response.status_code, 400, response.text)
        for route, body in (("extract", {"video": str(self.video), "base": 0, "frame": 0}),
                            ("split", {"selection": selection, "video": str(self.video)}),
                            ("align/check", {"video": source["path"], "base": 0, "picture": str(secret), "pin": 0}),
                            ("align/apply", {"video": str(self.video), "base": 0, "picture": self.picture, "pin": 0})):
            with self.subTest(route=route):
                self.assertEqual(self.client.post(f"{API}/{route}", json=body).status_code, 400)
        saved = self.ok(self.client.get(f"{API}/projects/{project['key']}"))
        self.assertEqual(saved["windows"][0]["pictures"], [])

    def test_a_frame_past_the_end_is_not_found(self):
        source = self.source()
        frame = self.client.get(f"{API}/frame", params={"path": source["path"], "base": 900, "frame": 0})
        self.assertEqual(frame.status_code, 404)

    def test_the_strip_and_a_frame_come_from_the_source(self):
        source = self.source()
        strip = self.client.get(f"{API}/strip", params={"path": source["path"], "start": 0, "seconds": 8})
        self.assertEqual(strip.status_code, 200)
        self.assertEqual(strip.headers["content-type"], "image/jpeg")
        frame = self.client.get(f"{API}/frame", params={"path": source["path"], "base": 1, "frame": 12})
        self.assertEqual(frame.status_code, 200)

    def test_check_walks_through_the_steps(self):
        source = self.source()
        data = self.ok(self.client.post(f"{API}/check", json=self.editor(source, slots=[], prompt="")))
        self.assertEqual((data["ready"], data["step"]), (False, "frames"))
        data = self.ok(self.client.post(f"{API}/check", json=self.editor(source, prompt="")))
        self.assertEqual(data["step"], "prompt")
        data = self.ok(self.client.post(f"{API}/check", json=self.editor(source)))
        self.assertTrue(data["ready"])
        self.assertIsNone(data["step"])
        self.assertEqual(data["next_run"]["sampling"], "6 steps · Euler")

    def test_a_short_selection_runs_on_the_job_path(self):
        source = self.source()
        self.ok(self.client.post(f"{API}/generate", json=self.editor(source)))
        run = self.finished_run()
        self.assertIsNone(run["error"])
        self.assertTrue(Path(run["result_url"]).is_file())
        self.assertEqual(self.client.get(f"{API}/file", params={"path": run["result_url"]}).status_code, 200)
        self.assertTrue(any(job.family == "remake" for job in JOBS.records.values()))

    def test_windows_split_save_render_and_combine(self):
        source = self.source()
        selection = self.selection(source)
        self.assertIsNone(self.ok(self.client.post(f"{API}/selection", json={"selection": selection}))["project"])
        project = self.ok(self.client.post(f"{API}/split", json={"selection": selection, "video": source["path"]}))
        self.assertGreater(len(project["windows"]), 1)
        body = self.editor(source, selection=selection)
        saved = self.ok(self.client.post(f"{API}/window/save", json=body))
        self.assertEqual(saved["windows"][0]["pins"], [12])
        self.ok(self.client.post(f"{API}/generate", json=body))
        run = self.finished_run()
        self.assertIsNone(run["error"])
        self.assertIn("Combined", run["status"])
        done = self.ok(self.client.get(f"{API}/projects/{project['key']}"))
        self.assertTrue(all(window["take"] for window in done["windows"]))
        self.assertIsNotNone(done["output"])
        self.assertEqual(self.client.get(f"{API}/file", params={"path": done["output"]}).status_code, 200)

        marked = self.ok(self.client.post(f"{API}/window/mark", json=body))
        self.assertEqual(marked["windows"][0]["state"], "changed")
        self.assertEqual(self.ok(self.client.post(f"{API}/window/carried", json={**body, "window": 1}))["pin"], 16)
        joined = self.ok(self.client.post(f"{API}/combine", json={"selection": selection}))
        self.assertIn("Combined", joined["status"])

    def test_projects_load_rename_and_delete(self):
        source = self.source()
        selection = self.selection(source)
        project = self.ok(self.client.post(f"{API}/split", json={"selection": selection, "video": source["path"]}))
        key = project["key"]
        rows = self.ok(self.client.patch(f"{API}/projects/{key}", json={"name": "Robot"}))["projects"]
        self.assertEqual(rows[0]["name"], "Robot")
        loaded = self.ok(self.client.post(f"{API}/projects/{key}/load"))
        self.assertEqual(loaded["source_info"]["sha"], source["sha"])
        self.assertEqual(self.ok(self.client.delete(f"{API}/projects/{key}"))["projects"], [])
        self.assertEqual(self.client.post(f"{API}/projects/{key}/load").status_code, 400)

    def test_pictures_from_media_reach_the_page_on_a_pulse(self):
        self.assertEqual(self.ok(self.client.get(f"{API}/pulse"))["inbox"], [])
        self.handlers.inbox = SimpleNamespace(pop=lambda owner, default: {2: self.picture})
        self.assertEqual(self.ok(self.client.get(f"{API}/pulse"))["inbox"], [])
        self.assertEqual(self.ok(self.client.get(f"{API}/pulse", params={"collect": True}))["inbox"],
                         [{"slot": 2, "path": self.picture}])

    def test_a_picture_can_only_be_sent_from_the_owners_folder(self):
        self.ok(self.client.post(f"{API}/send-to-image", json={"path": self.picture, "model": "h3"}))
        sent = next(iter(self.handlers.outgoing.values()))
        self.assertNotEqual(sent, self.picture)
        self.assertTrue(Path(sent).is_file())
        for body in ({"path": str(self.root / "x.png"), "model": "h3"}, {"path": self.picture, "model": "nope"}):
            self.assertEqual(self.client.post(f"{API}/send-to-image", json=body).status_code, 400)

    def test_an_interrupt_with_nothing_running_says_so(self):
        self.assertEqual(self.ok(self.client.post(f"{API}/interrupt"))["message"], "Nothing is running.")
        self.assertEqual(self.interrupted, [])


if __name__ == "__main__":
    unittest.main()
