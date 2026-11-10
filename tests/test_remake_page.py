"""The Remake page's JSON API: sources, selections, windows, files and runs."""

from dataclasses import replace
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
from h3_app.media_tools import probe_video_metadata
from h3_app.remake import SECTIONS
from h3_app.remake.media import join_windows
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

    def workshop_client(self):
        from h3_ui.remake_workshop import build_remake_workshop
        import gradio as gr

        with gr.Blocks() as ui:
            handlers = build_remake_workshop(gr.Tab("Remake"), ui, self.runtime)
        app = FastAPI()
        install_owner_cookie(app, self.store)
        for install in ui.h3_routes:
            install(app)
        client = TestClient(app)
        client.cookies.update(self.client.cookies)
        return client, handlers

    def test_workshop_refuses_gpu_actions_without_submitting_jobs(self):
        client, _ = self.workshop_client()
        boot = self.ok(client.get(f"{API}/boot"))
        self.assertFalse(boot["gpu"])
        self.assertEqual(boot["edit_models"], [])
        jobs = len(JOBS.records)
        for route in ("generate", "quick-edit", "send-to-image", "interrupt", "quick-edit/interrupt"):
            response = client.post(f"{API}/{route}", json={})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("Studio", response.text)
        self.assertEqual(len(JOBS.records), jobs)

    def test_workshop_short_project_reopens_in_studio_and_renders(self):
        client, _ = self.workshop_client()
        with self.video.open("rb") as handle:
            source = self.ok(client.post(f"{API}/source", files={"file": ("dance.mp4", handle)}))
        body = self.editor(source, mode="Edit", smart=False, pinned=False, auto_prompts=False)
        frame = self.ok(client.post(f"{API}/extract", json={**body, "base": 0, "frame": 12}))
        self.assertEqual(client.get(f"{API}/file", params={"path": frame["path"]}).status_code, 200)
        with Path(self.picture).open("rb") as handle:
            edited = self.ok(client.post(f"{API}/picture", files={"file": ("edited.png", handle)}))["path"]
        body["slots"][0]["path"] = edited
        saved = self.ok(client.post(f"{API}/save", json=body))
        self.assertEqual(len(saved["windows"]), 1)
        Path(source["path"]).unlink()
        Path(edited).unlink()
        loaded = self.ok(self.client.post(f"{API}/projects/{saved['key']}/load"))
        self.assertEqual(loaded["mode"], "Edit")
        self.assertEqual(loaded["request"], "robot")
        self.assertFalse(loaded["smart"])
        self.assertFalse(loaded["pinned"])
        self.assertFalse(loaded["auto_prompts"])
        window = loaded["windows"][0]
        self.assertEqual(window["prompt"], SIX_SECTIONS)
        self.assertEqual(window["pins"], [12])
        self.assertTrue(Path(window["pictures"][0]).is_file())
        body.update(video=loaded["source_info"]["path"], selection=loaded["selection"],
                    slots=[{"path": window["pictures"][0], "pin": 12}, None, None, None])
        selected = self.ok(self.client.post(f"{API}/selection", json=body))
        self.assertEqual(selected["project"]["key"], saved["key"])
        self.ok(self.client.post(f"{API}/generate", json=body))
        run = self.finished_run()
        self.assertIsNone(run["error"])
        rendered = self.ok(self.client.get(f"{API}/projects/{saved['key']}"))
        self.assertEqual(rendered["rendered"], 1)

    def test_workshop_keeps_each_windows_materials(self):
        client, _ = self.workshop_client()
        source = self.source()
        body = self.editor(source, selection=self.selection(source, end=12))
        first = self.ok(client.post(f"{API}/save", json=body))
        self.assertGreater(len(first["windows"]), 1)
        body.update(window=1, prompt="second window", use_carried=False)
        self.ok(client.post(f"{API}/save", json=body))
        loaded = self.ok(self.client.post(f"{API}/projects/{first['key']}/load"))
        self.assertEqual(loaded["windows"][0]["prompt"], SIX_SECTIONS)
        self.assertEqual(loaded["windows"][1]["prompt"], "second window")
        self.assertFalse(loaded["windows"][1]["use_carried"])
        self.assertEqual(loaded["windows"][1]["pins"], [12])
        self.assertTrue(Path(loaded["windows"][1]["pictures"][0]).is_file())

    def test_autosave_updates_the_open_project_and_never_creates_one(self):
        source = self.source()
        body = self.editor(source)
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        projects = self.ok(self.client.get(f"{API}/projects"))["projects"]
        moved = self.selection(source, end=6)
        self.ok(self.client.post(f"{API}/save", json={
            **body, "project": saved["key"], "selection": moved, "prompt": "still the open project"}))
        rows = self.ok(self.client.get(f"{API}/projects"))["projects"]
        self.assertEqual([row["key"] for row in rows], [row["key"] for row in projects])
        loaded = self.ok(self.client.post(f"{API}/projects/{saved['key']}/load"))
        self.assertEqual(loaded["windows"][0]["prompt"], "still the open project")

    def test_autosave_does_not_recreate_a_deleted_project(self):
        source = self.source()
        body = self.editor(source)
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        self.ok(self.client.delete(f"{API}/projects/{saved['key']}"))
        response = self.client.post(f"{API}/save", json={**body, "project": saved["key"]})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.ok(self.client.get(f"{API}/projects"))["projects"], [])

    def test_saving_a_prompt_during_generation_keeps_the_window_queued(self):
        body = self.editor(self.source())
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        body["project"] = saved["key"]
        original_take = self.handlers.deps.h3_take

        def edit_during_take(*args):
            self.ok(self.client.post(f"{API}/save", json={**body, "prompt": "Revised prompt"}))
            yield from original_take(*args)

        with patch.object(self.handlers, "deps", replace(self.handlers.deps, h3_take=edit_during_take)):
            self.ok(self.client.post(f"{API}/generate", json=body))
            self.assertIsNone(self.finished_run()["error"])
        window = self.ok(self.client.get(f"{API}/projects/{saved['key']}"))["windows"][0]
        self.assertEqual((window["prompt"], window["state"]), ("Revised prompt", "changed"))
        self.assertTrue(window["take"])

    def test_join_keeps_every_frame_when_the_soundtrack_ends_early(self):
        audio = self.root / "short.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(audio)], check=True)
        output = self.root / "joined.mp4"
        join_windows([{"path": self.video, "first": 0, "frames": 192, "continues": False}],
                     audio, start=0, frames=192, mix=0, output=output, has_audio=True)
        metadata = probe_video_metadata(output)
        self.assertEqual(metadata.frame_count, 192)
        self.assertTrue(metadata.has_audio)

    def test_save_refuses_a_path_as_project_key(self):
        source = self.source()
        body = self.editor(source)
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        for key in (f"../{saved['key']}", "..", f"x/../{saved['key']}", 7):
            with self.subTest(key=key):
                response = self.client.post(f"{API}/save", json={**body, "project": key})
                self.assertEqual(response.status_code, 400, response.text)

    def test_save_refuses_a_window_outside_the_project(self):
        source = self.source()
        body = self.editor(source)
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        for window in (len(saved["windows"]), -1):
            with self.subTest(window=window):
                response = self.client.post(f"{API}/save", json={
                    **body, "project": saved["key"], "window": window})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("no longer part", response.text)

    def test_saved_short_selections_use_supported_frames_from_the_selection_start(self):
        source = self.source()
        for seconds, frames in ((6, 141), (8.7, 192), (10, 226)):
            with self.subTest(seconds=seconds):
                chosen = {**self.selection(source, end=seconds + 1), "start": 1}
                body = self.editor(source, selection=chosen)
                saved = self.ok(self.client.post(f"{API}/save", json=body))
                self.assertEqual(saved["frames"], frames)
                self.assertEqual(saved["windows"][0]["render"], [0, frames])
                self.assertEqual(saved["selection"], chosen)

    def test_saving_and_reopening_preserves_unedited_frame_markers(self):
        source = self.source()
        body = self.editor(source)
        frame = self.ok(self.client.post(f"{API}/extract", json={**body, "base": 0, "frame": 12}))
        # Empty slots are compressed when saving, so markers must follow their pictures.
        body.update(slots=[None, {"path": frame["path"], "pin": 12}, None, None],
                    originals={"2": frame["original"]})
        saved = self.ok(self.client.post(f"{API}/save", json=body))
        loaded = self.ok(self.client.post(f"{API}/projects/{saved['key']}/load"))
        window = loaded["windows"][0]
        self.assertEqual(window["originals"], [frame["original"]])
        body.update(slots=[{"path": window["pictures"][0], "pin": 12}, None, None, None],
                    originals={"1": window["originals"][0]})
        checked = self.ok(self.client.post(f"{API}/check", json=body))
        self.assertFalse(checked["ready"])
        self.assertIn("Edit Picture 1", checked["message"])
        body.update(slots=[{"path": self.picture, "pin": 12}, None, None, None], originals={})
        replaced = self.ok(self.client.post(f"{API}/save", json=body))
        self.assertEqual(replaced["windows"][0]["originals"], [None])

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
        for route in ("check", "generate", "save", "window/save", "window/mark", "rewrite"):
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
        self.ok(self.client.post(f"{API}/generate", json={**body, "project": project["key"]}))
        run = self.finished_run()
        self.assertIsNone(run["error"])
        self.assertIn("Combined", run["status"])
        self.assertEqual(run["project"], project["key"])
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
