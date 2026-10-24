"""API-only Soundstage batch endpoints: many voices or sounds per GPU job, and a model preload."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr


def _fake_worker(clips):
    """Stand-in for audio_tools.run: writes a clip per item, failing the ones named in ``clips``."""
    def run(model, payload, runtime, **_kwargs):
        items = payload["items"]
        errors = []
        for item in items:
            target = Path(item["output"])
            if "FAIL" in str(item.get("text", "")):
                errors.append("RuntimeError: boom")
                continue
            target.write_bytes(b"RIFF" + b"x" * 80)
            errors.append("")
        clips.append((model, len(items)))
        return {"outputs": [item["output"] for item in items], "errors": errors}
    return run


class Drama3BatchApiTests(unittest.TestCase):
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
        for target, value in (("_runtime_config", self.runtime),):
            patcher = patch.object(drama3_handlers, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_voices_share_one_worker_request_and_failures_stay_with_their_item(self):
        calls = []
        briefs = [
            {"name": "林夕", "description": "A female voice: bright.", "preview_text": "你好，今天天气不错。"},
            {"name": "老周", "description": "A male voice: gravelly.", "preview_text": "FAIL 这一句会失败。"},
            {"name": "坏", "description": "", "preview_text": "没有描述。"},
        ]
        with patch("h3_app.audio_tools.run", _fake_worker(calls)), \
                patch("h3_models.resolve_hf_token", return_value=None):
            files, listing = self.handlers.generate_drama3_voices_batch(json.dumps(briefs, ensure_ascii=False))
        self.assertEqual(calls, [("voice_design", 2)])  # the invalid brief never reaches the GPU
        self.assertEqual([entry["ok"] for entry in listing], [True, False, False])
        self.assertEqual(listing[0]["file"], 0)
        self.assertEqual(listing[0]["transcript"], "你好，今天天气不错。")
        self.assertIn("boom", listing[1]["error"])
        self.assertIn("Describe the voice", listing[2]["error"])
        self.assertEqual(len(files), 1)
        self.assertTrue(Path(files[0]).is_file())
        self.assertTrue(Path(files[0]).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()))

    def test_sounds_run_as_one_request(self):
        seen = []

        def run(model, payload, runtime, **_kwargs):
            seen.append((model, [item["prompt"] for item in payload["items"]]))
            for item in payload["items"]:
                Path(item["output"]).write_bytes(b"RIFF" + b"y" * 80)
            return {}

        sounds = [{"name": "rain", "description": "Steady rain", "duration": 25},
                  {"name": "bad", "description": "x", "duration": 999},
                  {"name": "door", "description": "Wooden door", "duration": 3}]
        with patch("h3_app.audio_tools.run", run), patch("h3_models.resolve_hf_token", return_value=None):
            files, listing = self.handlers.generate_drama3_sounds_batch(sounds)
        self.assertEqual(seen, [("stable_audio", ["Steady rain", "Wooden door"])])
        self.assertEqual([(entry["name"], entry["ok"], entry["file"]) for entry in listing],
                         [("rain", True, 0), ("bad", False, None), ("door", True, 1)])
        self.assertEqual(len(files), 2)

    def test_a_failed_sound_stays_with_its_item(self):
        payloads = []

        def run(model, payload, runtime, **_kwargs):
            payloads.append(payload)
            errors = []
            for item in payload["items"]:
                if item["prompt"] == "Broken":
                    errors.append("RuntimeError: invalid audio")
                    continue
                Path(item["output"]).write_bytes(b"RIFF" + b"y" * 80)
                errors.append("")
            return {"outputs": [None if error else item["output"] for item, error in zip(payload["items"], errors)],
                    "errors": errors}

        sounds = [{"name": "rain", "description": "Steady rain", "duration": 5},
                  {"name": "broken", "description": "Broken", "duration": 5}]
        with patch("h3_app.audio_tools.run", run), patch("h3_models.resolve_hf_token", return_value=None):
            files, listing = self.handlers.generate_drama3_sounds_batch(sounds)
        self.assertTrue(payloads[0]["keep_going"])
        self.assertEqual([(entry["ok"], entry["file"]) for entry in listing], [(True, 0), (False, None)])
        self.assertIn("invalid audio", listing[1]["error"])
        self.assertEqual(len(files), 1)
        self.assertEqual(len(list((self.root / "output" / "drama3_sounds").iterdir())), 1)

    def test_missing_language_defaults_to_auto_and_short_replies_fail_per_item(self):
        seen = []

        def run(model, payload, runtime, **_kwargs):
            seen.extend(item["language"] for item in payload["items"])
            Path(payload["items"][0]["output"]).write_bytes(b"RIFF" + b"x" * 80)
            return {"outputs": [payload["items"][0]["output"]], "errors": [""]}  # no entry for the second item

        briefs = [{"name": "a", "description": "A voice.", "preview_text": "One."},
                  {"name": "b", "description": "A voice.", "preview_text": "Two."}]
        with patch("h3_app.audio_tools.run", run), patch("h3_models.resolve_hf_token", return_value=None):
            files, listing = self.handlers.generate_drama3_voices_batch(json.dumps(briefs), None)
        self.assertEqual(seen, ["Auto", "Auto"])
        self.assertEqual([entry["ok"] for entry in listing], [True, False])
        self.assertIn("no audio", listing[1]["error"])
        self.assertEqual(len(files), 1)

    def test_old_served_copies_are_pruned(self):
        from h3_ui.drama3 import library_api

        clip = self.root / "clip.wav"
        clip.write_bytes(b"RIFF" + b"x" * 80)
        stale = Path(tempfile.mkdtemp(prefix=library_api.EXPORT_PREFIXES[0]))
        self.addCleanup(lambda: stale.exists() and stale.rmdir())
        old = stale.stat().st_mtime - library_api.EXPORT_MAX_AGE_SECONDS - 60
        os.utime(stale, (old, old))
        fresh = Path(library_api._export(clip, "clip"))
        self.addCleanup(lambda: (fresh.unlink(missing_ok=True), fresh.parent.rmdir()))
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.is_file())

    def test_malformed_requests_reach_the_client(self):
        with self.assertRaisesRegex(gr.Error, "JSON list"):
            self.handlers.generate_drama3_voices_batch("{not json")
        with self.assertRaisesRegex(gr.Error, "non-empty"):
            self.handlers.generate_drama3_sounds_batch([])

    def test_endpoints_are_published_and_hidden_from_the_page(self):
        from h3_ui.application import _drama3_actions
        from h3_ui.drama3.bindings import bind_drama3
        from h3_ui.drama3.view import build_drama3_view

        with gr.Blocks() as demo:
            view = build_drama3_view(gr.Group())
            bind_drama3(view, _drama3_actions())
        api = demo.get_api_info()["named_endpoints"]
        parameters = {name: [row["parameter_name"] for row in api[name]["parameters"]] for name in (
            "/generate_drama3_voices_batch", "/generate_drama3_sounds_batch", "/preload_drama3_models")}
        self.assertEqual(parameters, {
            "/generate_drama3_voices_batch": ["briefs", "language"],
            "/generate_drama3_sounds_batch": ["sounds"],
            "/preload_drama3_models": ["voice_engine", "script_mode"],
        })
        config = json.loads(json.dumps(demo.get_config_file(), default=str))
        wanted = {name.lstrip("/") for name in parameters}
        triggers = {dep["api_name"]: dep["targets"][0][0] for dep in config["dependencies"]
                    if dep.get("api_name") in wanted}
        components = {component["id"]: component for component in config["components"]}
        self.assertEqual(len(triggers), 3)
        self.assertTrue(all(components[target]["props"].get("visible") is False for target in triggers.values()))


if __name__ == "__main__":
    unittest.main()
