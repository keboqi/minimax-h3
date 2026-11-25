"""The Storyboard page's JSON API: editing, media, take switching, files and GPU runs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from threading import Barrier, Event
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app import storyboard as sb
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, UI_DEFAULTS
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS
from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import JOBS
from h3_app.workspace_store import COOKIE, WorkspaceStore, install_owner_cookie
from h3_ui.storyboard.handlers import MediaLibrary, StoryboardHandlers, StoryboardServices
from h3_ui.storyboard.page import _image_thumbnail, install_storyboard_page

API = "/storyboard/api"
PROFILE_VALUES = {name: UI_DEFAULTS.get(name, f"default-{name}") for name in GENERATION_FIELDS}


def write_latent(path: Path):
    header = json.dumps({"__metadata__": {"h3_prefix_frames": "39"}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class StoryboardPageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.release = None
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=self.fake_take,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
            free_gpu=lambda: None,
        ))
        self.interrupted = []
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_storyboard_page(app, self.handlers, gpu=True,
                                interrupt=lambda request, family, job_id: self.interrupted.append((family, job_id))
                                or "Stopping.")
        self.client = TestClient(app)

    def fake_take(self, values, graph, outputs):
        yield GenerationUpdate(None, "Sampling")
        video = self.runtime.output_dir / f"{graph.output_prefix}_00001_.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mp4")
        latent = f"{graph.latent_output_prefix}_00001_.safetensors"
        write_latent(self.runtime.output_dir / latent)
        outputs["latent"] = latent
        yield GenerationUpdate(str(video), "Completed")

    def create(self, story="One.\n\n---\n\nTwo.\n\n---\n\nThree."):
        response = self.client.post(f"{API}/boards", json={"name": "Chase", "story": story})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def scene_url(self, board, index, path=""):
        return f"{API}/boards/{board['id']}/scenes/{board['scenes'][index]['id']}{path}"

    def picture(self, name="pic.png"):
        path = self.root / name
        Image.new("RGB", (800, 800), (10, 120, 90)).save(path)
        return path

    def wait_for_run(self, board):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = self.client.get(f"{API}/boards/{board['id']}/run").json()["run"]
            if run and not run["active"]:
                return run
            time.sleep(0.01)
        self.fail("The run never finished.")

    def test_cpu_studio_does_not_advertise_storyboard_generation(self):
        self.assertTrue(self.client.get(f"{API}/boot").json()["gpu"])
        with patch("h3_ui.storyboard.page.is_cpu_only", return_value=True):
            self.assertFalse(self.client.get(f"{API}/boot").json()["gpu"])

    def test_one_read_carries_every_scene_the_page_will_show(self):
        board = self.create()
        self.assertEqual([scene["prompt"] for scene in board["scenes"]], ["One.", "Two.", "Three."])
        self.assertEqual(board["scenes"][1]["state_label"], "Not generated")
        self.assertEqual(self.client.get(f"{API}/boot").json()["boards"], [{"id": board["id"], "name": "Chase", "generated": False}])

    def test_a_field_edit_saves_alone_and_returns_what_it_changed(self):
        board = self.create()
        changed = self.client.patch(self.scene_url(board, 0), json={"prompt": "", "seconds": "8"}).json()
        self.assertEqual(changed["scenes"][0]["seconds"], 8)
        self.assertEqual(changed["scenes"][0]["issues"], ["Write a prompt"])
        self.assertEqual(changed["scenes"][1]["prompt"], "Two.")

    def test_invalid_edit_is_refused_with_a_reason(self):
        board = self.create()
        response = self.client.patch(self.scene_url(board, 0), json={"seconds": 99})
        self.assertEqual(response.status_code, 400)
        self.assertIn("seconds", response.json()["error"])

    def test_scene_list_edits_return_the_new_scene(self):
        board = self.create()
        added = self.client.post(f"{API}/boards/{board['id']}/scenes").json()
        self.assertEqual(added["scene_id"], added["scenes"][-1]["id"])
        inserted = self.client.post(f"{API}/boards/{board['id']}/scenes",
                                    json={"after": board["scenes"][0]["id"]}).json()
        self.assertEqual(inserted["scene_id"], inserted["scenes"][1]["id"])
        copied = self.client.post(self.scene_url(board, 0, "/duplicate")).json()
        self.assertEqual(copied["scenes"][1]["id"], copied["scene_id"])
        moved = self.client.post(self.scene_url(board, 0, "/move"), json={"offset": 2}).json()
        self.assertEqual(moved["scenes"][2]["id"], board["scenes"][0]["id"])
        left = self.client.delete(self.scene_url(board, 0)).json()
        self.assertNotIn(board["scenes"][0]["id"], [scene["id"] for scene in left["scenes"]])

    def test_uploads_append_replace_remove_and_refuse_the_wrong_type(self):
        board = self.create()
        url = self.scene_url(board, 0, "/media/ref_images")
        with self.picture().open("rb") as handle:
            data = self.client.post(url, files=[("files", ("a.png", handle, "image/png"))]).json()
        with self.picture("two.png").open("rb") as handle:
            data = self.client.post(url, files=[("files", ("b.png", handle.read() + b"x", "image/png"))]).json()
        images = data["scenes"][0]["ref_images"]
        self.assertEqual(len(images), 2)
        data = self.client.delete(url, params={"path": images[0]["path"]}).json()
        self.assertEqual([item["path"] for item in data["scenes"][0]["ref_images"]], [images[1]["path"]])
        refused = self.client.post(url, files=[("files", ("a.exe", b"x", "application/octet-stream"))])
        self.assertEqual(refused.status_code, 400)

    def test_only_this_storyboards_files_are_served(self):
        board = self.create()
        with self.picture().open("rb") as handle:
            data = self.client.post(self.scene_url(board, 0, "/media/first_image"),
                                    files=[("files", ("a.png", handle, "image/png"))]).json()
        image = data["scenes"][0]["first_image"]
        self.assertEqual(self.client.get(image["url"]).status_code, 200)
        thumb = self.client.get(image["thumb"])
        self.assertEqual(thumb.status_code, 200)
        with Image.open(BytesIO(thumb.content)) as small:
            self.assertEqual(small.size, (480, 480))
        outside = self.root / "secret.txt"
        outside.write_text("secret")
        response = self.client.get(f"{API}/boards/{board['id']}/file", params={"path": str(outside)})
        self.assertEqual(response.status_code, 404)

    def test_takes_stream_in_ranges_so_players_can_seek(self):
        board = self.create()
        self.client.post(f"{API}/boards/{board['id']}/generate",
                         json={"action": "scene", "scene_id": board["scenes"][0]["id"]})
        self.wait_for_run(board)
        take = self.client.get(f"{API}/boards/{board['id']}").json()["scenes"][0]["takes"][0]
        response = self.client.get(take["video"], headers={"Range": "bytes=1-2"})
        self.assertEqual((response.status_code, response.content), (206, b"p4"))

    def test_library_pictures_are_validated(self):
        board = self.create()
        picture = self.picture("library.png")
        checked = []
        self.handlers.library = MediaLibrary(None, None, lambda path, mode: checked.append((path, mode)))
        data = self.client.put(self.scene_url(board, 0, "/media/last_image"), json={"source": str(picture)}).json()
        self.assertEqual(checked, [(str(picture), "Image")])
        self.assertTrue(data["scenes"][0]["last_image"])

    def test_manual_take_upload_library_and_overlap_routes_work_without_gpu(self):
        from tests.test_storyboard_media import make_video

        app = FastAPI()
        install_owner_cookie(app, self.store)
        checked = []
        workshop_handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=None,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
        ))
        workshop_handlers.library = MediaLibrary(None, None, lambda path, mode: checked.append((path, mode)))
        install_storyboard_page(app, workshop_handlers, gpu=False)
        client = TestClient(app)

        created = client.post(f"{API}/boards", json={"name": "Manual", "story": "One.\n\n---\n\nTwo."}).json()
        board_id = created["id"]
        s1_id, s2_id = created["scenes"][0]["id"], created["scenes"][1]["id"]

        clip1 = make_video(self.root / "take1.mp4", width=320, height=192, seconds=3.5, audio=True, rate=24)
        with clip1.open("rb") as handle:
            res = client.post(f"{API}/boards/{board_id}/scenes/{s1_id}/takes",
                              files={"file": ("take1.mp4", handle, "video/mp4")})
        self.assertEqual(res.status_code, 200, res.text)
        data1 = res.json()
        t1 = data1["scenes"][0]["takes"][0]
        self.assertEqual((t1["id"], t1["source"], t1["source_name"]), (data1["take_id"], "upload", "take1.mp4"))
        self.assertEqual(data1["scenes"][0]["in_use"], t1["id"])

        clip2 = make_video(self.root / "lib2.mp4", width=320, height=192, seconds=3.5, audio=True, rate=24)
        res2 = client.put(f"{API}/boards/{board_id}/scenes/{s2_id}/takes", json={"source": str(clip2)})
        self.assertEqual(res2.status_code, 200, res2.text)
        self.assertEqual(checked, [(str(clip2), "Video")])
        data2 = res2.json()
        t2 = data2["scenes"][1]["takes"][0]
        self.assertEqual((t2["source"], t2["prefix_frames"]), ("library", 0))

        toggled = client.patch(f"{API}/boards/{board_id}/scenes/{s2_id}/takes/{t2['id']}",
                               json={"trim_overlap": True})
        self.assertEqual(toggled.status_code, 200, toggled.text)
        self.assertEqual(toggled.json()["scenes"][1]["takes"][0]["prefix_frames"], 39)

        bad_ext = client.post(f"{API}/boards/{board_id}/scenes/{s1_id}/takes",
                              files={"file": ("bad.txt", b"hello", "text/plain")})
        self.assertEqual(bad_ext.status_code, 400)

    def test_cast_needs_a_picture_and_scenes_lose_a_removed_member(self):
        board = self.create()
        url = f"{API}/boards/{board['id']}/cast"
        self.assertEqual(self.client.post(url, data={"name": "Mei"}).status_code, 400)
        with self.picture().open("rb") as handle:
            data = self.client.post(url, data={"name": "Mei"}, files={"file": ("mei.png", handle, "image/png")}).json()
        self.assertEqual(data["cast"][0]["name"], "Mei")
        self.client.patch(self.scene_url(board, 0), json={"cast": ["Mei"]})
        data = self.client.delete(f"{url}/Mei").json()
        self.assertEqual((data["cast"], data["scenes"][0]["cast"]), ([], []))

    def test_settings_captured_from_the_h3_tab_shape_new_storyboards(self):
        board = self.create()
        owner = self.store.verify_owner(self.client.cookies[COOKIE])
        self.assertEqual(self.client.post(f"{API}/boards/{board['id']}/profile").status_code, 400)
        self.handlers.capture_settings(owner, {"width": 1234}, {})
        applied = self.client.post(f"{API}/boards/{board['id']}/profile")
        self.assertEqual(applied.status_code, 200)
        self.assertEqual(self.service.load(owner, board["id"])["profile"]["width"], 1234)

    def test_a_run_finishes_in_the_background_and_reports_its_takes(self):
        board = self.create()
        started = self.client.post(f"{API}/boards/{board['id']}/generate", json={"action": "remaining"})
        self.assertEqual(started.status_code, 200, started.text)
        run = self.wait_for_run(board)
        self.assertIsNone(run["error"])
        self.assertGreater(run["revision"], 0)
        after = self.client.get(f"{API}/boards/{board['id']}").json()
        self.assertEqual(after["done"], 3)
        self.assertEqual(after["run"]["id"], run["id"])
        job = next(job for job in JOBS.records.values() if job.family == "storyboard" and job.id in JOBS.records)
        self.assertEqual(job.state, "completed")

    def test_a_scene_that_is_not_ready_is_refused_before_the_gpu(self):
        board = self.create()
        response = self.client.post(f"{API}/boards/{board['id']}/generate",
                                    json={"action": "scene", "scene_id": board["scenes"][1]["id"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Generate scene 1 first", response.json()["error"])

    def test_switching_a_take_asks_then_switches(self):
        board = self.create()
        url = f"{API}/boards/{board['id']}"
        self.client.post(f"{url}/generate", json={"action": "remaining"})
        self.wait_for_run(board)
        self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": board["scenes"][0]["id"]})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            first = self.client.get(url).json()["scenes"][0]
            if len(first["takes"]) == 2:
                break
            time.sleep(0.01)
        self.wait_for_run(board)
        other = first["takes"][1]["id"]
        asked = self.client.post(self.scene_url(board, 0, "/use-take"), json={"take_id": other}).json()
        self.assertIn("Scene 2 and 3 continue", asked["question"])
        self.assertNotEqual(asked["scenes"][0]["in_use"], other)
        done = self.client.post(self.scene_url(board, 0, "/use-take"), json={"take_id": other, "confirmed": True}).json()
        self.assertEqual(done["scenes"][0]["in_use"], other)
        self.assertEqual(done["scenes"][1]["state"], "outdated")
        old_take = first["takes"][0]["id"]
        self.assertEqual(done["scenes"][0]["takes"][0]["continued_by"], [2])
        after_delete = self.client.delete(self.scene_url(board, 0, f"/takes/{old_take}")).json()
        self.assertEqual([t["id"] for t in after_delete["scenes"][0]["takes"]], [other])

    def enhancing_handlers(self):
        """Handlers whose writer checks each call against the real one and keeps its settings."""
        import inspect

        from h3_ui.prompt_controller import PromptController

        signature = inspect.signature(PromptController.enhance_h3_prompt)
        calls = []

        def writer(**kwargs):
            signature.bind(None, **kwargs)
            calls.append(kwargs)
            return "Better prompt", "Prompt enhanced."

        self.handlers.services = replace(self.handlers.services, enhance_prompt=writer)
        return calls

    def test_prompt_writer_works_before_any_writer_settings_are_chosen(self):
        calls = self.enhancing_handlers()
        board = self.create()
        response = self.client.post(self.scene_url(board, 0, "/enhance"), json={"writer": {}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["scenes"][0]["prompt"], "Better prompt")
        self.assertEqual((calls[0]["backend"], calls[0]["gemini_model"]), ("Gemini", DEFAULT_GEMINI_PROMPT_MODEL))

    def test_prompt_writer_uses_the_h3_tabs_settings_over_the_pages(self):
        calls = self.enhancing_handlers()
        board = self.create()
        owner = self.store.verify_owner(self.client.cookies[COOKIE])
        self.handlers.capture_settings(owner, {}, {"prompt_writer_backend": "OpenAI-compatible", "gemini_api_key": ""})
        self.client.post(self.scene_url(board, 0, "/enhance"),
                         json={"writer": {"prompt_writer_backend": "Gemini", "gemini_api_key": "page-key"}})
        self.assertEqual((calls[0]["backend"], calls[0]["gemini_api_key"]), ("OpenAI-compatible", "page-key"))

    def test_a_second_generate_is_refused_and_leaves_no_job_behind(self):
        board = self.create()
        url = f"{API}/boards/{board['id']}/generate"
        owner = self.store.verify_owner(self.client.cookies[COOKIE])
        release = Event()
        self.addCleanup(release.set)

        def waiting_take(values, graph, outputs):
            release.wait(timeout=10)
            yield from self.fake_take(values, graph, outputs)

        self.handlers.services = replace(self.handlers.services, generate_take=waiting_take)
        self.assertEqual(self.client.post(url, json={"action": "remaining"}).status_code, 200)
        self.assertEqual(self.client.post(url, json={"action": "remaining"}).status_code, 400)
        release.set()
        self.wait_for_run(board)
        jobs = JOBS.list_owned(owner)
        self.assertEqual(len(jobs), 1)
        self.assertIsNotNone(jobs[0].finished_at)

    def test_a_cast_name_with_a_slash_can_be_removed(self):
        board = self.create()
        url = f"{API}/boards/{board['id']}/cast"
        with self.picture().open("rb") as handle:
            self.client.post(url, data={"name": "Mei/Tan"}, files={"file": ("mei.png", handle, "image/png")})
        self.assertEqual(self.client.delete(f"{url}/Mei%2FTan").json()["cast"], [])

    def test_interrupt_stops_only_this_storyboards_job(self):
        started, release = Event(), Event()
        finish = self.fake_take

        def held(values, graph, outputs):
            started.set()
            release.wait(10)
            yield from finish(values, graph, outputs)

        self.handlers.services = replace(self.handlers.services, generate_take=held)
        board, other = self.create(), self.create()
        self.assertEqual(self.client.post(f"{API}/boards/{other['id']}/interrupt").json()["message"],
                         "This storyboard isn't generating.")
        self.client.post(f"{API}/boards/{board['id']}/generate",
                         json={"action": "scene", "scene_id": board["scenes"][0]["id"]})
        self.assertTrue(started.wait(10))
        try:
            response = self.client.post(f"{API}/boards/{board['id']}/interrupt")
            self.assertEqual(response.json()["message"], "Stopping.")
            [(family, job_id)] = self.interrupted
            self.assertEqual(family, "storyboard")
            self.assertEqual(JOBS.records[job_id].family, "storyboard")
        finally:
            release.set()
        self.wait_for_run(board)

    def test_a_take_stays_while_this_storyboard_generates(self):
        started, release = Event(), Event()
        finish = self.fake_take

        def held(values, graph, outputs):
            if started.is_set():
                started.clear()
                release.wait(10)
            yield from finish(values, graph, outputs)

        board = self.create()
        url = f"{API}/boards/{board['id']}"
        self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": board["scenes"][0]["id"]})
        self.wait_for_run(board)
        take = self.client.get(url).json()["scenes"][0]["takes"][0]["id"]
        self.handlers.services = replace(self.handlers.services, generate_take=held)
        started.set()
        self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": board["scenes"][0]["id"]})
        deadline = time.monotonic() + 10
        while started.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            response = self.client.delete(self.scene_url(board, 0, f"/takes/{take}"))
            self.assertEqual(response.status_code, 400)
            self.assertIn("Wait for the current run", response.json()["error"])
            # One run per storyboard of any kind, and it can't be deleted from under the run.
            joined = self.client.post(f"{url}/join", json={"include_outdated": True})
            self.assertEqual(joined.status_code, 400)
            self.assertIn("current run", joined.json()["error"])
            deleted = self.client.delete(url)
            self.assertEqual(deleted.status_code, 400)
            self.assertIn("run to finish", deleted.json()["error"])
        finally:
            release.set()
        self.wait_for_run(board)

    def test_thumbnails_follow_the_pictures_rotation(self):
        path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # Shown rotated 90 degrees.
        Image.new("RGB", (800, 400)).save(path, exif=exif)
        thumbnail = _image_thumbnail(path, self.root / "thumbs")
        with Image.open(thumbnail) as image:
            self.assertGreater(image.height, image.width)

    def test_joining_runs_in_the_background_and_offers_the_video(self):
        board = self.create()
        joined = []

        def fake_render(document, destination, *, include_outdated=True, subtitle_preview=None):
            joined.append(include_outdated)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"mp4")
            return destination

        with patch.object(self.service, "render", fake_render):
            self.client.post(f"{API}/boards/{board['id']}/join", json={"include_outdated": False})
            run = self.wait_for_run(board)
        self.assertEqual(joined, [False])
        self.assertEqual(self.client.get(run["result_url"]).content, b"mp4")

    def test_export_burns_subtitles_when_requested(self):
        board = self.create()
        url = f"{API}/boards/{board['id']}/join"
        self.assertEqual(self.client.post(url, json={"export": True, "burn_subtitles": "false"}).status_code, 400)
        published = []
        self.handlers.services = replace(self.handlers.services,
                                         publish=lambda path, metadata: published.append((path, metadata)))
        with patch.object(self.service, "render", side_effect=self.fake_render) as render:
            self.assertEqual(self.client.post(url, json={"export": True, "burn_subtitles": True}).status_code, 200)
            self.assertIsNone(self.wait_for_run(board)["error"])
        self.assertIs(render.call_args.kwargs["burn_subtitles"], True)
        path, metadata = published[0]
        self.assertTrue(path.name.endswith("_subtitled.mp4"))
        self.assertIs(metadata["settings"]["burned_subtitles"], True)

    def test_subtitle_setting_and_project_action(self):
        board = self.create("A traveler says <d>[English] Scripted line.</d>")
        url = f"{API}/boards/{board['id']}"
        scene_id = board["scenes"][0]["id"]
        lines = [{"text": "Spoken line", "start": 1.0, "end": 2.0, "words": [
            {"text": "Spoken", "start": 1.0, "end": 1.4}, {"text": "line", "start": 1.5, "end": 2.0},
        ]}]

        def generate():
            response = self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": scene_id})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(self.wait_for_run(board)["error"])

        with patch("h3_app.storyboard.subtitles.transcribe", return_value=lines) as transcribe:
            self.assertFalse(board["auto_subtitles"])
            generate()
            transcribe.assert_not_called()
            take = self.client.get(url).json()["scenes"][0]["takes"][0]
            self.assertIn("Scripted line.", self.client.get(take["subtitles"]).text)

            self.assertEqual(self.client.post(f"{url}/subtitles").status_code, 200)
            self.assertIsNone(self.wait_for_run(board)["error"])
            take = self.client.get(url).json()["scenes"][0]["takes"][0]
            track = self.client.get(take["subtitles"])
            self.assertIn("<c>Spoken </c><00:00:01.500><c>line</c>", track.text)
            self.assertTrue(track.headers["content-type"].startswith("text/vtt"))

            self.assertEqual(self.client.post(f"{url}/subtitles").status_code, 200)
            self.wait_for_run(board)
            transcribe.assert_called_once()

            settings = self.client.put(f"{url}/subtitle-settings", json={"auto_subtitles": True}).json()
            self.assertTrue(settings["auto_subtitles"])
            generate()
            self.assertEqual(transcribe.call_count, 2)
            takes = self.client.get(url).json()["scenes"][0]["takes"]
            self.assertIn("<c>Spoken", self.client.get(takes[1]["subtitles"]).text)

    def test_subtitle_language_is_saved_per_storyboard(self):
        board, other = self.create(), self.create()
        url = f"{API}/boards/{board['id']}"
        settings = f"{url}/subtitle-settings"
        default = self.client.get(f"{API}/boot").json()["subtitle_languages"][0]
        self.assertEqual(board["subtitle_language"], default)
        self.assertEqual(self.client.put(settings, json={"auto_subtitles": True}).status_code, 200)
        updated = self.client.put(settings, json={"subtitle_language": "French"})
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertTrue(updated.json()["auto_subtitles"])
        self.assertEqual(self.client.put(settings, json={"subtitle_language": "Klingon"}).status_code, 400)
        self.assertEqual(self.client.get(url).json()["subtitle_language"], "French")
        self.assertEqual(self.client.get(f"{API}/boards/{other['id']}").json()["subtitle_language"], default)
        copied = self.client.post(f"{url}/duplicate")
        self.assertEqual(copied.status_code, 200, copied.text)
        self.assertEqual(copied.json()["subtitle_language"], "French")

    def test_subtitle_translation_runs_parallel_batches_and_preserves_caption_order(self):
        from h3_app.storyboard.subtitles import Translation

        board = self.create("One scene.")
        url = f"{API}/boards/{board['id']}"
        scene_id = board["scenes"][0]["id"]
        self.assertEqual(self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": scene_id}).status_code, 200)
        self.assertIsNone(self.wait_for_run(board)["error"])
        take = self.client.get(url).json()["scenes"][0]["takes"][0]
        owner = self.store.verify_owner(self.client.cookies.get(COOKIE))
        lines = [{"text": str(index), "start": index / 4, "end": (index + 1) / 4} for index in range(11)]
        self.service.save_subtitles(owner, board["id"], take["id"], lines)

        first_wave = Barrier(10, timeout=5)
        last_started = Event()

        def translate(translation, texts):
            index = int(texts[0])
            if index < 10:
                first_wave.wait()
                if index == 0 and not last_started.wait(timeout=5):
                    raise AssertionError("The queued batch did not start while the first batch was still running.")
            else:
                last_started.set()
            return [f"Translated {index}"]

        with (patch("h3_app.storyboard.subtitles.TRANSLATION_BATCH", 1),
              patch.object(Translation, "translate", translate)):
            document = self.service.load(owner, board["id"])
            translated = self.service.translate_subtitles(owner, board["id"], list(document["takes"].values()),
                                                          Translation("French"))

        self.assertEqual(translated[take["id"]], [f"Translated {index}" for index in range(11)])

    def test_scene_translation_can_finish_while_another_take_is_translating(self):
        from h3_app.storyboard.subtitles import Translation

        board = self.create("<d>First.</d>\n---\n<d>Second.</d>")
        url = f"{API}/boards/{board['id']}"
        self.assertEqual(self.client.post(f"{url}/generate", json={"action": "remaining"}).status_code, 200)
        self.assertIsNone(self.wait_for_run(board)["error"])
        owner = self.store.verify_owner(self.client.cookies.get(COOKIE))
        document = self.service.load(owner, board["id"])
        first, second = [document["takes"][scene["in_use"]] for scene in document["scenes"]]
        first_started, release_first = Event(), Event()

        def translate(_translation, texts):
            if texts == ["First."]:
                first_started.set()
                release_first.wait(timeout=5)
            return [f"Translated {text}" for text in texts]

        with patch.object(Translation, "translate", translate), ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(self.service.translate_subtitles, owner, board["id"], [first], Translation("French"))
            try:
                self.assertTrue(first_started.wait(timeout=3))
                other = pool.submit(self.service.translate_subtitles, owner, board["id"], [second], Translation("French"))
                self.assertEqual(other.result(timeout=3), {second["id"]: ["Translated Second."]})
                self.assertFalse(pending.done())
            finally:
                release_first.set()
            self.assertEqual(pending.result(timeout=3), {first["id"]: ["Translated First."]})

    def test_subtitle_translation_is_reused_for_export_and_refreshed_when_the_source_changes(self):
        board = self.create("A traveler says <d>Original line.</d>")
        url = f"{API}/boards/{board['id']}"
        scene_id = board["scenes"][0]["id"]
        self.assertEqual(self.client.post(f"{url}/generate", json={"action": "scene", "scene_id": scene_id}).status_code, 200)
        self.assertIsNone(self.wait_for_run(board)["error"])
        take = self.client.get(url).json()["scenes"][0]["takes"][0]
        options = {"language": "French", "gemini_api_key": "test-key"}

        def translate(**overrides):
            response = self.client.post(f"{url}/subtitle-translations",
                                        json={**options, "take_ids": [take["id"]], **overrides})
            self.assertEqual(response.status_code, 200, response.text)
            return response.json()["tracks"][take["id"]]

        def reply(text):
            return {"candidates": [{"content": {"parts": [{"text": json.dumps([text])}]}}]}

        with patch("h3_app.storyboard.subtitles.gemini.Session") as session:
            generate = session.return_value.__enter__.return_value.generate
            generate.side_effect = [reply("Texte traduit."), reply("Texto traducido."), reply("Autre modèle."),
                                    reply("Flash Lite."), reply("Nouveau texte.")]
            french = translate()
            self.assertIn("Original line.\n<c.translation>Texte traduit.</c>", self.client.get(french).text)
            self.assertNotIn("Texte traduit.", self.client.get(take["subtitles"]).text)
            self.service = sb.StoryboardService(WorkspaceStore(self.store.root), lambda: self.runtime)
            restored = self.client.get(url).json()["scenes"][0]["takes"][0]
            self.assertEqual(restored["subtitle_translations"], {"French": french})
            self.assertEqual(translate(), french)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(generate.call_args.kwargs["generation_config"]["thinkingConfig"], {"thinkingLevel": "MINIMAL"})
            spanish = translate(language="Spanish")
            self.assertIn("Texto traducido.", self.client.get(spanish).text)
            self.assertEqual(translate(), french)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args.args[0], "gemma-4-26b-a4b-it")
            self.assertEqual(translate(model="gemma-4-31b-it"), french)
            self.assertEqual(generate.call_count, 2)
            other = translate(model="gemma-4-31b-it", force=True)
            self.assertNotEqual(other, french)
            self.assertIn("Autre modèle.", self.client.get(other).text)
            self.assertEqual(generate.call_args.args[0], "gemma-4-31b-it")
            flash = translate(model="gemini-3.5-flash-lite", force=True)
            self.assertIn("Flash Lite.", self.client.get(flash).text)
            self.assertEqual(generate.call_args.args[0], "gemini-3.5-flash-lite")
            self.assertEqual(translate(), flash)
            self.assertEqual(generate.call_count, 4)

            with patch.object(self.service, "render", return_value=self.root / "export.mp4") as render:
                self.assertEqual(self.client.post(f"{url}/join", json={"export": True, "translation": options}).status_code, 200)
                self.assertIsNone(self.wait_for_run(board)["error"])
                self.assertEqual(render.call_args.kwargs["translations"], {take["id"]: ["Flash Lite."]})
                self.assertEqual(render.call_args.kwargs["translation_language"], "French")
            self.assertEqual(generate.call_count, 4)

            owner = self.store.verify_owner(self.client.cookies.get(COOKIE))
            self.service.save_subtitles(owner, board["id"], take["id"],
                                        [{"text": "Edited.", "start": 0, "end": 1}])
            self.assertEqual(self.client.get(url).json()["scenes"][0]["takes"][0]["subtitle_translations"], {})
            self.assertNotIn("Flash Lite.", self.client.get(flash).text)
            self.assertIn("Edited.\n<c.translation>Nouveau texte.</c>", self.client.get(translate()).text)
            self.assertEqual(generate.call_count, 5)
            self.assertNotIn("test-key", json.dumps(self.service.load(owner, board["id"])))

    def wait_for_auto(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            auto = self.client.get(f"{API}/auto?boards=false").json()["auto"]
            if auto and not auto["active"]:
                return auto
            time.sleep(0.01)
        self.fail("Auto render never finished.")

    def fake_render(self, document, destination, *, include_outdated=True, burn_subtitles=False):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mp4")
        return destination

    def test_auto_render_generates_and_exports_each_selected_storyboard_in_turn(self):
        first, second, unselected = self.create(), self.create("Alpha.\n\n---\n\nBeta."), self.create()
        published = []
        self.handlers.services = replace(self.handlers.services,
                                         publish=lambda path, metadata: published.append(metadata["storyboard"]["id"]))
        listed = {row["id"]: row for row in self.client.get(f"{API}/auto").json()["boards"]}
        self.assertEqual((listed[second["id"]]["pending"], listed[second["id"]]["needs_input"]), (2, []))
        with patch.object(self.service, "render", self.fake_render):
            started = self.client.post(f"{API}/auto", json={"ids": [first["id"], second["id"]]})
            self.assertEqual(started.status_code, 200, started.text)
            again = self.client.post(f"{API}/auto", json={"ids": [unselected["id"]]})
            auto = self.wait_for_auto()
        self.assertEqual(again.status_code, 400)
        self.assertEqual([item["state"] for item in auto["items"]], ["exported", "exported"])
        self.assertEqual(published, [first["id"], second["id"]])
        self.assertEqual(self.client.get(f"{API}/boards/{second['id']}").json()["done"], 2)
        self.assertEqual(self.client.get(f"{API}/boards/{unselected['id']}").json()["done"], 0)
        self.assertEqual(self.client.get(f"{API}/boot").json()["auto"]["id"], auto["id"])

    def test_auto_render_does_not_export_a_storyboard_with_scenes_that_need_input(self):
        blocked, ready = self.create(), self.create("Alpha.")
        self.client.patch(self.scene_url(blocked, 1), json={"prompt": ""})
        published = []
        self.handlers.services = replace(self.handlers.services,
                                         publish=lambda path, metadata: published.append(metadata["storyboard"]["id"]))
        listed = {row["id"]: row for row in self.client.get(f"{API}/auto").json()["boards"]}
        self.assertEqual(listed[blocked["id"]]["needs_input"], [2])
        self.assertEqual((listed[blocked["id"]]["runnable"], listed[ready["id"]]["runnable"]), (1, 1))
        with patch.object(self.service, "render", self.fake_render):
            self.client.post(f"{API}/auto", json={"ids": [blocked["id"], ready["id"]]})
            auto = self.wait_for_auto()
        self.assertEqual([item["state"] for item in auto["items"]], ["incomplete", "exported"])
        self.assertIn("scene 2, 3", auto["items"][0]["detail"])
        self.assertEqual(published, [ready["id"]])
        self.assertEqual(self.client.get(f"{API}/boards/{blocked['id']}").json()["done"], 1)

    def test_stopping_auto_render_interrupts_the_current_storyboard_and_skips_the_rest(self):
        started, release = Event(), Event()
        finish = self.fake_take

        def held(values, graph, outputs):
            started.set()
            release.wait(10)
            yield from finish(values, graph, outputs)

        self.handlers.services = replace(self.handlers.services, generate_take=held)
        first, second = self.create(), self.create()
        self.client.post(f"{API}/auto", json={"ids": [first["id"], second["id"]]})
        self.assertTrue(started.wait(10))
        try:
            deadline = time.monotonic() + 10
            while (self.client.get(f"{API}/boards/{first['id']}/run").json()["run"] or {}).get("status", "").startswith(
                    "Waiting") and time.monotonic() < deadline:
                time.sleep(0.01)
            stopped = self.client.post(f"{API}/auto/stop").json()
            self.assertTrue(stopped["auto"]["stopping"])
            self.assertEqual([family for family, _ in self.interrupted], ["storyboard"])
        finally:
            release.set()
        auto = self.wait_for_auto()
        self.assertEqual([item["state"] for item in auto["items"]], ["stopped", "stopped"])
        self.assertEqual(self.client.get(f"{API}/boards/{second['id']}").json()["done"], 0)

    def test_page_and_assets_are_served(self):
        self.assertIn("app.js", self.client.get("/storyboard/").text)
        self.assertEqual(self.client.get("/storyboard/app.js").status_code, 200)
        self.assertEqual(self.client.get("/storyboard/app.css").status_code, 200)
        self.assertEqual(self.client.get("/storyboard/other.txt").status_code, 404)

    def test_clear_history_takes_removes_unused_takes_and_preserves_in_use_and_branches(self):
        created = self.create("Scene one.")
        board_id = created["id"]
        scene_id = created["scenes"][0]["id"]
        self.ok_board = self.client.patch(
            f"{API}/boards/{board_id}/scenes/{scene_id}",
            json={"prompt": "Shot prompt"},
        )
        self.assertEqual(self.ok_board.status_code, 200)

        def add_take(tid: str) -> Path:
            vid = self.service.output_path(f"{self.service.take_prefix(board_id, scene_id, tid)}.mp4")
            vid.parent.mkdir(parents=True, exist_ok=True)
            vid.write_bytes(f"video-{tid}".encode())
            lat = self.service.output_path(f"{self.service.take_prefix(board_id, scene_id, tid)}.safetensors")
            write_latent(lat)
            board = self.service.load(created_owner, board_id)
            self.service.record_take(
                created_owner, board_id, scene_id, tid,
                video=str(vid), latents={"latent": str(lat), "prefix_frames": 39}, seed=1,
                inputs=sb.model.scene_inputs(board, 0),
            )
            return vid

        created_owner = self.store.verify_owner(self.client.cookies[COOKIE])
        vid_branch_only = add_take("tk_branch")

        # Duplicate board while only tk_branch exists on it, then add tk_old + tk_cur on the original.
        self.service.use_take(created_owner, board_id, scene_id, "tk_branch")
        dup = self.client.post(f"{API}/boards/{board_id}/duplicate")
        self.assertEqual(dup.status_code, 200, dup.text)
        dup_id = dup.json()["id"]

        vid_old = add_take("tk_old")
        vid_cur = add_take("tk_cur")
        self.service.use_take(created_owner, board_id, scene_id, "tk_cur")
        published = self.runtime.output_dir / "h3" / "Chase_media.mp4"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes(b"media-export")

        res = self.client.post(f"{API}/boards/{board_id}/clear-history-takes")
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["outcome"]["removed_takes"], 2)
        self.assertEqual(data["outcome"]["kept_takes"], 1)
        self.assertEqual([t["id"] for t in data["scenes"][0]["takes"]], ["tk_cur"])
        self.assertEqual(data["scenes"][0]["in_use"], "tk_cur")
        self.assertEqual(set(self.service.load(created_owner, board_id)["takes"].keys()), {"tk_cur"})
        self.assertFalse(vid_old.exists())
        self.assertTrue(vid_cur.is_file())
        # Shared with duplicated board so its video file remains intact.
        self.assertTrue(vid_branch_only.is_file())
        self.assertTrue(published.is_file())
        self.assertIn("tk_branch", self.service.load(created_owner, dup_id)["takes"])
