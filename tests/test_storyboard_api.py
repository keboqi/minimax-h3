"""API-only Storyboard endpoints: import a planned storyboard, list storyboards and read one."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gradio as gr

from h3_app import storyboard as sb
from h3_app.catalog import UI_DEFAULTS
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS
from h3_app.workspace_store import WorkspaceStore
from h3_ui.storyboard import api as api_module
from h3_ui.storyboard.handlers import StoryboardHandlers, StoryboardServices

PROFILE_VALUES = {name: UI_DEFAULTS.get(name, f"default-{name}") for name in GENERATION_FIELDS}


class StoryboardApiTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(script_dir=self.root, comfy_url="http://comfy", comfy_dir=self.root / "comfy",
                                     models_config=self.root / "models.json", outputs_dir=self.root / "outputs")
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service, generate_take=None,
            request_defaults=lambda: dict(PROFILE_VALUES), publish=lambda path, metadata: None,
        ))
        patcher = patch.object(api_module, "require_owner", lambda request: "api:client")
        patcher.start()
        self.addCleanup(patcher.stop)
        uploads = self.root / "uploads"
        uploads.mkdir()
        self.files = {}
        for name in ("mei.png", "door.png", "end.png", "voice.wav"):
            (uploads / name).write_bytes(name.encode())
            self.files[name] = str(uploads / name)

    def spec(self, **changes):
        spec = {
            "name": "Lighthouse",
            "settings": {"width": 1920, "height": 1088},
            "cast": [{"name": "Mei", "image": "mei.png"}],
            "scenes": [
                {"title": "Arrival", "prompt": "Mei climbs the stairs.", "seconds": 10, "mode": "Reference media",
                 "cast": ["Mei"], "ref_images": ["door.png"], "ref_audios": ["voice.wav"]},
                {"prompt": "She opens the door.", "link": "flow", "mode": "First / last frame",
                 "last_image": "end.png", "overlap": 22},
                {"prompt": "Dawn over the sea.", "link": "cut", "seconds": 5, "takes_per_run": 2},
            ],
        }
        spec.update(changes)
        return json.dumps(spec)

    def test_import_keeps_modes_media_cast_and_settings(self):
        details = self.handlers.storyboard_import(self.spec(), list(self.files.values()))
        self.assertEqual((details["name"], details["cast"]), ("Lighthouse", ["Mei"]))
        self.assertEqual([scene["runs_as"] for scene in details["scenes"]],
                         ["Reference media", "Last frame", "Text to video"])
        self.assertEqual([scene["link"] for scene in details["scenes"]], [None, "flow", "cut"])
        self.assertEqual([scene["state"] for scene in details["scenes"]], ["draft"] * 3)
        # Only the continuing scene waits; the opening and the fresh start are ready.
        self.assertEqual([bool(scene["issues"]) for scene in details["scenes"]], [False, True, False])
        self.assertIn("Generate scene 1 first", details["scenes"][1]["issues"][0])
        board = self.service.load("gpu-browser", details["id"])
        self.assertEqual((board["profile"]["width"], board["profile"]["height"]), (1920, 1088))
        first, second, third = board["scenes"]
        self.assertEqual(Path(first["ref_images"][0]).read_bytes(), b"door.png")
        self.assertTrue(Path(first["ref_images"][0]).is_relative_to(self.runtime.output_dir.resolve()))
        self.assertEqual(len(first["ref_audios"]), 1)
        self.assertEqual((second["overlap"], Path(second["last_image"]).read_bytes()), (22, b"end.png"))
        self.assertEqual((third["takes_per_run"], third["title"]), (2, "Dawn over the sea"))
        self.assertEqual(self.handlers.storyboard_list()[0]["id"], details["id"])
        self.assertEqual(details["changes"]["added"], ["Arrival", "She opens the door", "Dawn over the sea"])
        self.assertEqual({**self.handlers.storyboard_details(details["id"]), "changes": details["changes"]}, details)

    def keyed_board(self):
        """An imported storyboard of keyed scenes a, b, c, d whose takes are all in use."""
        spec = {"name": "Keys", "scenes": [
            {"key": "a", "prompt": "A.", "mode": "Reference media", "ref_images": ["door.png"]},
            {"key": "b", "prompt": "B.", "link": "flow"},
            {"key": "c", "prompt": "C.", "link": "cut"},
            {"key": "d", "prompt": "D.", "link": "cut"},
        ]}
        storyboard_id = self.handlers.storyboard_import(json.dumps(spec), [self.files["door.png"]])["id"]
        board = self.service.load("a", storyboard_id)
        for index, scene in enumerate(board["scenes"]):
            board = self.service.record_take("a", storyboard_id, scene["id"], f"t{index}", video="v.mp4",
                                             latents={"latent": "l.safetensors"}, seed=1,
                                             inputs=sb.scene_inputs(board, index))
        self.assertEqual(sb.scene_states(board), ["done"] * 4)
        return storyboard_id, spec, board

    def test_update_keeps_takes_of_unchanged_scenes_and_follows_the_plan(self):
        storyboard_id, spec, before = self.keyed_board()
        # The same picture under another upload name is matched by content, not copied again.
        renamed = self.root / "uploads" / "door-again.png"
        renamed.write_bytes(b"door.png")
        folder = self.runtime.output_dir / "h3/.storyboard" / storyboard_id / "media"
        media = sorted(folder.iterdir())
        a, b, c, _ = spec["scenes"]
        revised = {"scenes": [dict(a, ref_images=["door-again.png"]), dict(b, prompt="B, revised."),
                              {"key": "e", "prompt": "E.", "link": "cut"}, c]}
        details = self.handlers.storyboard_update(storyboard_id, json.dumps(revised), [str(renamed)])
        self.assertEqual(details["changes"], {"added": ["e"], "updated": ["b"], "removed": ["d"],
                                              "settings_changed": False})
        self.assertEqual([(scene["key"], scene["state"]) for scene in details["scenes"]],
                         [("a", "done"), ("b", "outdated"), ("e", "draft"), ("c", "done")])
        board = self.service.load("a", storyboard_id)
        self.assertEqual([scene["id"] for scene in board["scenes"][:2]],
                         [scene["id"] for scene in before["scenes"][:2]])
        self.assertEqual(board["scenes"][1]["takes"], ["t1"])
        self.assertEqual(board["scenes"][0]["ref_images"], before["scenes"][0]["ref_images"])
        self.assertEqual(sorted(folder.iterdir()), media)
        # The removed scene's take stays recorded, and the profile is untouched without settings.
        self.assertIn("t3", board["takes"])
        self.assertEqual(board["profile"], before["profile"])

    def test_update_with_new_settings_outdates_every_scene(self):
        storyboard_id, spec, _ = self.keyed_board()
        details = self.handlers.storyboard_update(
            storyboard_id, json.dumps({**spec, "settings": {"width": 1920}}), [self.files["door.png"]])
        self.assertTrue(details["changes"]["settings_changed"])
        self.assertEqual({scene["state"] for scene in details["scenes"]}, {"outdated"})

    def test_failed_update_changes_nothing_and_removes_its_copies(self):
        storyboard_id, spec, before = self.keyed_board()
        folder = self.runtime.output_dir / "h3/.storyboard" / storyboard_id / "media"
        media = sorted(folder.iterdir())
        revised = {"scenes": [dict(spec["scenes"][0], ref_images=["mei.png"])]}
        with patch.object(self.service, "save", side_effect=RuntimeError("disk full")), \
                self.assertRaisesRegex(RuntimeError, "disk full"):
            self.handlers.storyboard_update(storyboard_id, json.dumps(revised), [self.files["mei.png"]])
        self.assertEqual(sorted(folder.iterdir()), media)
        self.assertEqual(self.service.load("a", storyboard_id), before)
        bad = ((json.dumps({"scenes": [{"key": "a", "prompt": "x"}, {"key": "a", "prompt": "y"}]}), "already used"),
               (json.dumps({"scenes": [{"prompt": "x", "cast": ["Mei"]}]}), "not in the cast"))
        for spec_text, message in bad:
            with self.subTest(message=message), self.assertRaisesRegex(gr.Error, message):
                self.handlers.storyboard_update(storyboard_id, spec_text, [])
        with self.assertRaisesRegex(gr.Error, "no longer exists"):
            self.handlers.storyboard_update("missing", json.dumps(spec), [])

    def test_bad_specs_are_refused_before_anything_is_stored(self):
        cases = (
            (self.spec(scenes=[{"prompt": "x", "seconds": 16}]), "5 to 15 whole seconds"),
            (self.spec(scenes=[{"prompt": "x", "seconds": 7.5}]), "5 to 15 whole seconds"),
            (self.spec(scenes=[{"prompt": "x", "cast": ["Leo"]}]), "Leo, who is not in the cast"),
            (self.spec(scenes=[{"prompt": "x", "first_image": "missing.png"}]), "missing.png is named"),
            (self.spec(scenes=[{"prompt": "x", "camera": "dolly"}]), "Unknown scene 1 field"),
            (self.spec(settings={"prompt": "no"}), "Unknown settings field"),
            (json.dumps({"name": "Empty", "scenes": []}), "at least one scene"),
        )
        for spec, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(gr.Error, message):
                self.handlers.storyboard_import(spec, list(self.files.values()))
        self.assertEqual(self.handlers.storyboard_list(), [])

    def test_duplicate_upload_names_are_refused(self):
        other = self.root / "other"
        other.mkdir()
        (other / "mei.png").write_bytes(b"other")
        with self.assertRaisesRegex(gr.Error, "Two uploaded files are named mei.png"):
            self.handlers.storyboard_import(self.spec(), [*self.files.values(), str(other / "mei.png")])

    def test_workshop_publishes_the_endpoints(self):
        from h3_ui.storyboard.workshop import build_storyboard_workshop

        with gr.Blocks() as ui, gr.Tabs():
            build_storyboard_workshop(gr.Tab("Storyboard"), ui, self.runtime, self.store)
        names = {dep.api_name for dep in ui.fns.values() if dep.api_visibility == "public"}
        self.assertTrue({"storyboard_import", "storyboard_update", "storyboard_list", "storyboard_details"} <= names)


if __name__ == "__main__":
    unittest.main()
