"""API-only Soundstage library endpoints: saved speakers by group, saved sounds and result details."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr


class Drama3LibraryApiTests(unittest.TestCase):
    def setUp(self):
        from h3_ui.application import drama3_handlers

        self.handlers = drama3_handlers
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = SimpleNamespace(outputs_dir=self.root, output_dir=self.root / "output")
        env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        env.start()
        self.addCleanup(env.stop)
        runtime = patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime)
        runtime.start()
        self.addCleanup(runtime.stop)
        self.clip = self.root / "upload.wav"
        self.clip.write_bytes(b"RIFF" + b"v" * 120)

    def test_speakers_round_trip_through_a_story_group(self):
        saved = self.handlers.drama3_library_save_speaker("My Story", "林夕", str(self.clip), "我来了。",
                                                          "A female voice: bright and quick.")
        self.assertEqual((saved["name"], saved["group"]), ("林夕", "My Story"))
        self.assertEqual(saved["sha256"], hashlib.sha256(self.clip.read_bytes()).hexdigest())
        self.assertEqual(self.handlers.drama3_library_groups(), ["Global", "My Story"])
        # A group's listing holds its own speakers only.
        self.assertEqual([entry["name"] for entry in self.handlers.drama3_library_speakers("my story")], ["林夕"])
        self.assertEqual(self.handlers.drama3_library_speakers("Global"), [])
        path, entry = self.handlers.drama3_library_speaker_clip("My Story", "林夕")
        self.assertEqual(Path(path).read_bytes(), self.clip.read_bytes())
        # The download is a served copy, never the private workspace file.
        self.assertTrue(Path(path).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertEqual((entry["transcript"], entry["description"]), ("我来了。", "A female voice: bright and quick."))

    def test_saving_again_replaces_and_errors_reach_the_client(self):
        self.handlers.drama3_library_save_speaker("Story", "Mia", str(self.clip), "One.", "")
        other = self.root / "other.wav"
        other.write_bytes(b"RIFF" + b"w" * 40)
        replaced = self.handlers.drama3_library_save_speaker("Story", "mia", str(other), "Two.", "")
        self.assertEqual(replaced["transcript"], "Two.")
        self.assertEqual(len(self.handlers.drama3_library_speakers("Story")), 1)
        with self.assertRaisesRegex(gr.Error, "transcript"):
            self.handlers.drama3_library_save_speaker("Story", "Leo", str(self.clip), "", "")
        with self.assertRaisesRegex(gr.Error, "no longer available"):
            self.handlers.drama3_library_speaker_clip("Story", "Nobody")

    def test_sounds_round_trip(self):
        saved = self.handlers.drama3_library_save_sound("rain", str(self.clip), "Steady rain on a window", 25)
        self.assertEqual((saved["name"], saved["duration"]), ("rain", 25))
        self.assertEqual([entry["name"] for entry in self.handlers.drama3_library_sounds()], ["rain"])
        path, entry = self.handlers.drama3_library_sound_clip("rain")
        self.assertEqual(Path(path).read_bytes(), self.clip.read_bytes())
        self.assertEqual(entry["description"], "Steady rain on a window")

    def test_result_details_find_the_original_by_its_id(self):
        result_id = "0123456789abcdef0123456789abcdef"
        settings = {"title": "Ch 1", "chunks": [{"take": "abc", "text": "Hello.", "start": 0.0}]}
        with patch.object(self.handlers, "_result_settings", return_value=settings) as lookup:
            details = self.handlers.drama3_result_details(f"https://host/gradio_api/file=/tmp/x/Ch_1_{result_id}.mp3")
        self.assertEqual(details, settings)
        lookup.assert_called_once_with(f"Ch_1_{result_id}.mp3")
        with self.assertRaisesRegex(gr.Error, "file name"):
            self.handlers.drama3_result_details("story.mp3")
        with patch.object(self.handlers, "_result_settings", return_value={}), \
                self.assertRaisesRegex(gr.Error, "No Drama 3 result"):
            self.handlers.drama3_result_details(f"x_{result_id}.mp3")

    def test_endpoints_are_published_and_hidden_from_the_page(self):
        from h3_ui.application import _drama3_actions
        from h3_ui.drama3.bindings import bind_drama3
        from h3_ui.drama3.view import build_drama3_view

        with gr.Blocks() as demo:
            view = build_drama3_view(gr.Group())
            bind_drama3(view, _drama3_actions())
        api = demo.get_api_info()["named_endpoints"]
        parameters = {name: [row["parameter_name"] for row in api[name]["parameters"]] for name in (
            "/drama3_library_groups", "/drama3_library_speakers", "/drama3_library_save_speaker",
            "/drama3_library_speaker_clip", "/drama3_library_sounds", "/drama3_library_save_sound",
            "/drama3_library_sound_clip", "/drama3_result_details")}
        self.assertEqual(parameters, {
            "/drama3_library_groups": [],
            "/drama3_library_speakers": ["group"],
            "/drama3_library_save_speaker": ["group", "name", "audio", "transcript", "description"],
            "/drama3_library_speaker_clip": ["group", "name"],
            "/drama3_library_sounds": [],
            "/drama3_library_save_sound": ["name", "audio", "description", "duration"],
            "/drama3_library_sound_clip": ["name"],
            "/drama3_result_details": ["result"],
        })
        config = json.loads(json.dumps(demo.get_config_file(), default=str))
        triggers = {dep["api_name"]: dep["targets"][0][0] for dep in config["dependencies"]
                    if str(dep.get("api_name", "")).startswith(("drama3_library", "drama3_result"))}
        components = {component["id"]: component for component in config["components"]}
        self.assertEqual(len(triggers), 8)
        self.assertTrue(all(components[target]["props"].get("visible") is False for target in triggers.values()))


if __name__ == "__main__":
    unittest.main()
