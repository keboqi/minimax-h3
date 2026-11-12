"""The Media page's JSON API: browsing, the selected item, annotations, deletion, import, Set as and enhancement."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import h3_models as models
from h3_app import catalog
from h3_app.jobs import JOBS
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.job_bindings import owned_generation
from h3_ui.media.page import (
    IMAGE_METHODS, MediaPageServices, MediaPageState, install_media_page, method_fields, result_css,
)

API = "/media/api"
SUFFIX = {"Video": ".mp4", "Image": ".png", "Audio": ".wav"}
ENHANCE_NAMES = ("mode", "selected_media", "option", "seed", "seedvr2_model", "ltx25_model", "ltx25_prompt",
                 "force_offload", "upscale_resolution")


class FakeController:
    """The Media controller's surface, over one folder per type."""

    def __init__(self, root: Path):
        self.root = root
        for kind in SUFFIX:
            (root / kind).mkdir(parents=True, exist_ok=True)

    def add(self, kind, name, data=b"media"):
        path = self.root / kind / name
        path.write_bytes(data)
        return path

    def gallery_media_download_path(self, media, mode):
        path = Path(media).resolve()
        if not path.is_relative_to((self.root / mode).resolve()):
            raise ValueError("Not a managed file.")
        return f"/files/{path.name}"

    def list_media_paths(self, mode):
        return sorted((self.root / mode).iterdir())

    def refresh_media_page(self, mode, limit, *, paths=None):
        listed = list(paths.paths)[:limit]
        return SimpleNamespace(items=tuple((str(self.root / f"{path.stem}.webp"), f"Fake · {path.name}") for path in listed),
                               paths=tuple(str(path) for path in listed), total=len(paths.paths),
                               next_cursor=limit if len(paths.paths) > limit else None, status=f"{len(listed)} shown")

    def result(self, mode, message, selected=None):
        return ([], [], message, None, None, None, "", selected, False)

    def import_gallery_media(self, mode, uploaded):
        source = Path(uploaded)
        target = self.root / mode / f"import_{source.name}"
        shutil.copy(source, target)
        return self.result(mode, f"Imported `{source.name}`", str(target))

    def delete_selected_gallery_media(self, mode, selected, confirmed):
        Path(selected).unlink()
        return self.result(mode, f"Deleted **{Path(selected).name}**")

    def empty_generated_media_gallery(self, mode, selected, confirmed):
        count = 0
        for path in self.list_media_paths(mode):
            path.unlink()
            count += 1
        return self.result(mode, f"Deleted {count} item(s)")


class FakeIndex:
    def __init__(self, controller):
        self.controller = controller
        self.versions = {}
        self.favorites = set()

    def inventory(self, mode, query="", favorite=False, *, limit=None, engine="", sort="newest"):
        paths = [path for path in self.controller.list_media_paths(mode)
                 if query.lower() in path.name.lower() and (not favorite or str(path) in self.favorites)]
        records = tuple({"path": str(path), "favorite": int(str(path) in self.favorites)} for path in paths)
        return SimpleNamespace(paths=tuple(paths), records=records, total=len(paths), kind=mode)

    def summary(self):
        return ({kind: len(self.controller.list_media_paths(kind)) for kind in SUFFIX},
                {"Video": ["MiniMax H3"], "Image": ["FLUX 3"]})

    def rebuild(self):
        yield "Scanning historical video files…"
        yield "Historical scan complete: 0 files."


class MediaPageTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.controller = FakeController(self.root / "library")
        self.index = FakeIndex(self.controller)
        self.queued, self.enhanced = [], []
        set_as = SimpleNamespace(names=["LTX · First frame", "Song · Visual inspiration"], queue=self.queue)
        self.pages = MediaPageState(MediaPageServices(
            controller=self.controller, index=lambda: self.index, set_as=lambda: set_as, enhance=True,
            interrupt=lambda *args, **kwargs: "Interrupting."))
        previous = JOBS.callbacks.get("gallery")
        JOBS.callbacks["gallery"] = owned_generation(self.enhancer, "gallery", input_names=ENHANCE_NAMES)
        self.addCleanup(lambda: JOBS.callbacks.update({"gallery": previous}) if previous
                        else JOBS.callbacks.pop("gallery", None))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_media_page(app, self.pages)
        self.client = TestClient(app)
        self.boot = self.ok(self.client.get(f"{API}/boot"))

    # Helpers ------------------------------------------------------------------
    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def refused(self, response, text):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(text, response.json()["error"])

    def queue(self, owner, name, path):
        self.queued.append((name, path))
        return f"Set image as {name}. Open Create to use it."

    def enhancer(self, *values):
        self.enhanced.append(dict(zip(ENHANCE_NAMES, values)))
        source = Path(values[1])
        result = source.with_name(f"{source.stem}_seedvr2{source.suffix}")
        result.write_bytes(b"enhanced")
        yield ([], [], "Preparing", None, None, None, "", None, False, "Preparing SeedVR2 2x")
        yield ([], [], "Completed", str(result), None, None, "", str(result), False, "Completed SeedVR2 2x in 1.0s")

    def indexed(self, kind, name, metadata=None):
        path = self.controller.add(kind, name)
        self.store.index_assets([(path, kind, metadata or {"family": "flux3"})])
        return path

    def wait(self, kind):
        for _ in range(100):
            run = self.ok(self.client.get(f"{API}/run"))["runs"][kind]
            if run and not run["active"]:
                return run
            time.sleep(0.05)
        self.fail(f"The {kind} run did not finish.")

    # The page --------------------------------------------------------------------
    def test_page_and_result_view_assets_are_served(self):
        self.assertIn("app.js", self.client.get("/media/").text)
        self.assertEqual(self.client.get("/media/app.css").status_code, 200)
        self.assertEqual(self.client.get("/media/other.txt").status_code, 404)
        css = self.client.get("/media/results.css").text
        self.assertIn(".h3-dub-player", css)
        self.assertNotIn(".h3-workspace", css)
        script = self.client.get("/media/results.js").text
        for name in ("followSubtitles", "bindPlayer", "bindStage"):
            self.assertIn(name, script)
        self.assertTrue(result_css())

    def test_boot_offers_what_the_gradio_library_offered(self):
        enhance = self.boot["enhance"]
        self.assertEqual(enhance["methods"]["Video"], list(catalog.POSTPROCESS_OPTIONS))
        self.assertEqual(enhance["methods"]["Image"], list(IMAGE_METHODS))
        self.assertEqual(enhance["resolution"], catalog.DEFAULT_UPSCALE_RESOLUTION)
        self.assertEqual(self.boot["set_as"], ["LTX · First frame", "Song · Visual inspiration"])
        self.assertEqual([value for _label, value in self.boot["sorts"]], ["newest", "oldest", "name"])

    def test_method_fields_follow_the_gradio_rules(self):
        seedvr2 = method_fields(catalog.SEEDVR2_UPSCALE)
        self.assertTrue(seedvr2["seedvr2"] and seedvr2["resolution"] and seedvr2["seed"])
        self.assertFalse(seedvr2["prompt"] or seedvr2["ltx_model"])
        refine = method_fields(catalog.H3_REFINEMENT)
        self.assertTrue(refine["resolution"])
        self.assertFalse(refine["seed"] or refine["seedvr2"])
        relay = method_fields(next(iter(catalog.RELAYVSR_OPTIONS)))
        self.assertFalse(relay["resolution"])
        deblur = method_fields(catalog.LTX25_DEBLUR)
        self.assertTrue(deblur["ltx_model"] and deblur["prompt"])
        self.assertFalse(deblur["resolution"])
        self.assertIn("focus restoration", deblur["prompt_hint"])
        interpolation = method_fields("48 fps interpolation")
        self.assertFalse(any(value for key, value in interpolation.items() if key != "prompt_hint"))

    # Browsing --------------------------------------------------------------------
    def test_items_list_one_type_with_counts_engines_and_favorites(self):
        first = self.controller.add("Image", "a.png")
        self.controller.add("Image", "b.png")
        self.controller.add("Video", "clip.mp4")
        self.index.favorites.add(str(first))
        found = self.ok(self.client.get(f"{API}/items", params={"mode": "Image"}))
        self.assertEqual([item["name"] for item in found["items"]], ["a.png", "b.png"])
        self.assertEqual(found["counts"], {"Video": 1, "Image": 2, "Audio": 0})
        self.assertEqual(found["engines"], ["FLUX 3"])
        self.assertTrue(found["items"][0]["favorite"])
        self.assertTrue(found["items"][0]["thumb"].startswith("/media-previews/a.webp"))
        starred = self.ok(self.client.get(f"{API}/items", params={"mode": "Image", "favorite": True}))
        self.assertEqual([item["name"] for item in starred["items"]], ["a.png"])
        paged = self.ok(self.client.get(f"{API}/items", params={"mode": "Image", "limit": 1}))
        self.assertTrue(paged["more"])
        self.refused(self.client.get(f"{API}/items", params={"mode": "Other"}), "Video, Image or Audio")

    def test_unknown_engine_filter_shows_every_engine(self):
        self.controller.add("Video", "clip.mp4")
        found = self.ok(self.client.get(f"{API}/items", params={"mode": "Video", "engine": "Gone"}))
        self.assertEqual(found["engine"], "")

    # The selected item ------------------------------------------------------------
    def test_item_describes_settings_lineage_and_identity(self):
        path = self.indexed("Image", "portrait.png", {"family": "flux3", "job_id": "abcdef1234"})
        item = self.ok(self.client.get(f"{API}/item", params={"mode": "Image", "path": str(path)}))
        self.assertEqual(item["name"], "portrait.png")
        self.assertEqual(item["settings"], "Settings unavailable for this output.")
        self.assertTrue(item["asset_id"])
        self.assertIn("portrait.png", item["identity"])
        current = [step for step in item["lineage"] if step["current"]]
        self.assertEqual(current[0]["name"], "portrait.png")
        self.assertIn("Job abcdef12", [step["name"] for step in item["lineage"]])
        self.assertEqual(item["views"], {"soundstage": "", "dub": "", "stage": "", "switch": False})
        self.assertEqual(self.client.get(item["url"]).content, b"media")
        download = self.client.get(item["download"])
        self.assertIn("attachment", download.headers["content-disposition"])

    def test_files_outside_the_library_are_refused(self):
        outside = self.root / "secret.png"
        outside.write_bytes(b"secret")
        self.refused(self.client.get(f"{API}/item", params={"mode": "Image", "path": str(outside)}), "unavailable")
        self.assertEqual(self.client.get(f"{API}/file", params={"mode": "Image", "path": str(outside)}).status_code, 404)
        clip = self.controller.add("Video", "clip.mp4")
        # A video is not an image, and an item with no dub has no tracks.
        self.assertEqual(self.client.get(f"{API}/file", params={"mode": "Image", "path": str(clip)}).status_code, 404)
        self.assertEqual(self.client.get(f"{API}/track", params={"mode": "Video", "path": str(clip),
                                                                  "name": "dub"}).status_code, 404)

    def test_tags_and_favorite_are_saved(self):
        path = self.indexed("Image", "portrait.png")
        item = self.ok(self.client.post(f"{API}/annotate", json={"mode": "Image", "path": str(path),
                                                                   "tags": "keeper, harbor"}))
        self.assertEqual(item["tags"], "harbor, keeper")
        self.assertFalse(item["favorite"])
        item = self.ok(self.client.post(f"{API}/annotate", json={"mode": "Image", "path": str(path), "favorite": True}))
        self.assertTrue(item["favorite"])
        self.assertEqual(item["tags"], "harbor, keeper")
        loose = self.controller.add("Image", "loose.png")
        self.refused(self.client.post(f"{API}/annotate", json={"mode": "Image", "path": str(loose), "favorite": True}),
                     "Scan historical media")

    # Deletion ------------------------------------------------------------------------
    def test_deleting_the_selection_names_it_and_confirms(self):
        path = self.controller.add("Video", "clip.mp4")
        message = self.ok(self.client.post(f"{API}/delete/prepare", json={
            "mode": "Video", "path": str(path), "action": "selected"}))["message"]
        self.assertIn("1 video item(s)", message)
        self.assertIn("clip.mp4", message)
        self.refused(self.client.post(f"{API}/delete/confirm", json={"mode": "Video", "path": None}),
                     "selection changed")
        self.ok(self.client.post(f"{API}/delete/prepare", json={"mode": "Video", "path": str(path), "action": "selected"}))
        answer = self.ok(self.client.post(f"{API}/delete/confirm", json={"mode": "Video", "path": str(path)}))
        self.assertIn("Deleted clip.mp4", answer["message"])
        self.assertFalse(path.exists())

    def test_emptying_a_library_lists_names_and_refuses_changed_media(self):
        for index in range(5):
            self.controller.add("Audio", f"take{index}.wav")
        message = self.ok(self.client.post(f"{API}/delete/prepare", json={
            "mode": "Audio", "path": None, "action": "library"}))["message"]
        self.assertIn("5 audio item(s)", message)
        self.assertIn("and 2 more", message)
        (self.controller.root / "Audio" / "take0.wav").write_bytes(b"changed media")
        self.refused(self.client.post(f"{API}/delete/confirm", json={"mode": "Audio", "path": None}), "media changed")
        self.ok(self.client.post(f"{API}/delete/prepare", json={"mode": "Audio", "path": None, "action": "library"}))
        self.ok(self.client.post(f"{API}/delete/confirm", json={"mode": "Audio", "path": None}))
        self.assertEqual(self.controller.list_media_paths("Audio"), [])

    def test_an_empty_library_has_nothing_to_delete(self):
        self.refused(self.client.post(f"{API}/delete/prepare", json={"mode": "Image", "action": "library"}),
                     "This library is empty.")
        self.refused(self.client.post(f"{API}/delete/prepare", json={"mode": "Image", "path": "x", "action": "selected"}),
                     "Select an item")

    # Import, Set as, scan -----------------------------------------------------------------
    def test_import_files_the_upload_under_its_type(self):
        answer = self.ok(self.client.post(f"{API}/import", files={"file": ("photo.png", b"png", "image/png")}))
        self.assertEqual(answer["mode"], "Image")
        self.assertTrue(Path(answer["selected"]).is_file())
        self.assertIn("photo.png", answer["status"])
        self.refused(self.client.post(f"{API}/import", files={"file": ("notes.txt", b"x", "text/plain")}),
                     "video, image, or audio")

    def test_set_as_queues_a_library_image(self):
        image = self.controller.add("Image", "frame.png")
        answer = self.ok(self.client.post(f"{API}/set-as", json={"path": str(image), "name": "LTX · First frame"}))
        self.assertEqual(answer["message"], "Set image as LTX · First frame. Open Create to use it.")
        self.assertEqual(self.queued, [("LTX · First frame", str(image))])
        clip = self.controller.add("Video", "clip.mp4")
        self.refused(self.client.post(f"{API}/set-as", json={"path": str(clip), "name": "LTX · First frame"}),
                     "unavailable")

    def test_scan_reports_progress_until_done(self):
        self.ok(self.client.post(f"{API}/scan"))
        run = self.wait("scan")
        self.assertEqual(run["status"], "Historical scan complete: 0 files.")

    def test_downloads_are_off_without_their_handlers(self):
        self.assertFalse(self.boot["downloads"])
        self.refused(self.client.post(f"{API}/download", json={"url": "https://example.com/v"}), "aren't available")

    # Enhancement ---------------------------------------------------------------------------
    def test_enhance_runs_the_gallery_job_and_reports_the_copy(self):
        clip = self.controller.add("Video", "clip.mp4")
        self.ok(self.client.post(f"{API}/enhance", json={
            "mode": "Video", "path": str(clip), "method": catalog.SEEDVR2_UPSCALE, "seed": 7,
            "seedvr2_model": next(iter(models.SEEDVR2_MODEL_CHOICES)), "ltx_model": None, "prompt": "",
            "force_offload": True, "resolution": catalog.DEFAULT_UPSCALE_RESOLUTION}))
        run = self.wait("enhance")
        self.assertIsNone(run["error"], run)
        self.assertEqual(run["status"], "Completed SeedVR2 2x in 1.0s")
        self.assertTrue(str(run["result"]).endswith("_seedvr2.mp4"), run)
        self.assertEqual(run["mode"], "Video")
        values = self.enhanced[0]
        self.assertEqual((values["option"], values["seed"], values["force_offload"]),
                         (catalog.SEEDVR2_UPSCALE, 7, True))
        # An unset dropdown takes its default, as the Gradio control started on it.
        self.assertEqual(values["ltx25_model"], models.DEFAULT_LTX25_MODEL)

    def test_enhance_refuses_values_its_dropdowns_never_offered(self):
        clip = self.controller.add("Video", "clip.mp4")
        base = {"mode": "Video", "path": str(clip), "method": catalog.SEEDVR2_UPSCALE}
        for name, label in (("seedvr2_model", "SeedVR2 model"), ("ltx_model", "finishing LTX model"),
                            ("resolution", "target resolution")):
            with self.subTest(name=name):
                self.refused(self.client.post(f"{API}/enhance", json={**base, name: "Made up"}), label)
        self.refused(self.client.post(f"{API}/enhance", json={**base, "seed": "x"}), "whole number")
        self.assertEqual(self.enhanced, [])

    def test_enhance_refuses_audio_and_methods_of_another_type(self):
        take = self.controller.add("Audio", "take.wav")
        self.refused(self.client.post(f"{API}/enhance", json={"mode": "Audio", "path": str(take),
                                                              "method": catalog.SEEDVR2_UPSCALE}), "playback")
        image = self.controller.add("Image", "frame.png")
        self.refused(self.client.post(f"{API}/enhance", json={"mode": "Image", "path": str(image),
                                                              "method": catalog.H3_REFINEMENT}), "Choose an enhancement")

    def test_interrupt_with_nothing_running(self):
        self.assertEqual(self.ok(self.client.post(f"{API}/enhance/stop"))["message"], "Nothing is running.")


if __name__ == "__main__":
    unittest.main()
