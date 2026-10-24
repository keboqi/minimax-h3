"""API-only Soundstage batch endpoints: many voices or sounds per GPU job, and a model preload."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

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

    def _saved_cast(self):
        from h3_app.drama3 import sounds, speakers

        clip = self.root / "voice.wav"
        clip.write_bytes(b"RIFF" + b"v" * 80)
        speakers.save_speaker("林夕", clip, "你好，今天天气不错。", self.runtime, "A female voice.", "雨夜")
        speakers.save_speaker("narrator", clip, "故事从这里开始。", self.runtime, "A male voice.", "雨夜")
        sounds.save_sound("night_rain", clip, "Steady rain", 20, self.runtime)

    def _fake_render(self, requests):
        """Stand-in for the Drama 3 run: a result with a snapshot whose second take fails until regenerated."""
        from h3_app.generation.results import GenerationUpdate, fresh_audio
        from h3_app.provenance import write_snapshot

        def run(request, services, runtime, **_kwargs):
            requests.append(request)
            yield GenerationUpdate(None, "Planned 2 chunk(s)")
            if "BROKEN" in request.text:
                yield GenerationUpdate(None, "Error: Script rejected\n\ntimings")
                return
            result = self.runtime.output_dir / "drama3" / f"{request.title}_{uuid.uuid4().hex}.mp3"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_bytes(b"ID3" + b"m" * 80)
            fixed = "b" * 32 in request.regenerate
            write_snapshot(result, {"family": "Drama 3", "settings": {"title": request.title, "chunks": [
                {"name": "narrator", "take": "a" * 32, "check": "ok", "ok": True},
                {"name": "林夕", "take": "b" * 32, "check": "ok" if fixed else "no speech", "ok": fixed}]}})
            yield GenerationUpdate(fresh_audio(result), "Drama 3 complete · 2 chunks")
        return run

    def test_batch_render_uses_saved_voices_fixes_failed_takes_and_isolates_broken_jobs(self):
        import dataclasses

        self._saved_cast()
        self.runtime.input_dir = self.root / "input"
        requests = []
        jobs = [
            {"id": "ch01", "title": "第一章", "text": "narrator: 雨夜。\n林夕: 有人吗？", "script_mode": "Advanced (sound enhanced)",
             "temperature": 2.0, "top_p": 1.0, "top_k": 30, "speakers": [
                 {"slot": 0, "name": "narrator", "group": "雨夜", "voice": "narrator"},
                 {"slot": 1, "name": "林夕", "group": "雨夜", "voice": "林夕"}], "sounds": [{"name": "night_rain"}]},
            {"id": "ch02", "title": "第二章", "text": "BROKEN", "speakers": [
                {"slot": 0, "name": "narrator", "group": "雨夜", "voice": "narrator"}]},
            {"id": "ch03", "title": "第三章", "text": "narrator: 天亮了。", "speakers": [
                {"slot": 0, "name": "narrator", "group": "雨夜", "voice": "nobody"}]},
        ]
        services = dataclasses.replace(self.handlers.services, resolve_result=lambda path: path)
        with patch("h3_app.generation.drama3.generate_drama3", self._fake_render(requests)), \
                patch.object(self.handlers, "services", services), \
                patch.object(self.handlers, "stage_file", side_effect=lambda path, *_a, **_k: Path(path).name):
            updates = list(self.handlers.generate_drama3_batch(json.dumps(jobs, ensure_ascii=False), 1))
        listing, status = updates[-1]
        self.assertEqual(status, "Batch finished: 1/3 job(s) rendered.")
        self.assertEqual([(entry["id"], entry["ok"]) for entry in listing],
                         [("ch01", True), ("ch02", False), ("ch03", False)])
        first = listing[0]
        self.assertEqual((first["failed"], first["rounds"]), ([], 1))
        self.assertRegex(first["result"], r"[0-9a-f]{32}\.mp3$")
        self.assertIn("Script rejected", listing[1]["error"])
        self.assertIn("nobody", listing[2]["error"])
        # ch01 ran twice (the second time regenerating only its failed take); ch02 once; ch03 never.
        self.assertEqual([request.title for request in requests], ["第一章", "第一章", "第二章"])
        self.assertEqual(requests[1].regenerate, ("b" * 32,))
        self.assertEqual(requests[0].voice_names, ((0, "narrator"), (1, "林夕")))
        self.assertEqual([voice[2] for voice in requests[0].voices], ["故事从这里开始。", "你好，今天天气不错。"])
        self.assertEqual([(name, duration) for name, _path, _desc, duration in requests[0].sounds], [("night_rain", 20)])
        self.assertTrue(all(Path(path).parent == self.runtime.input_dir for _s, path, _t in requests[0].voices))
        with self.assertRaisesRegex(gr.Error, "0 to 3"):
            list(self.handlers.generate_drama3_batch(json.dumps(jobs), 9))

        # The listed result's audio and details come back by file name.
        found = next(self.runtime.output_dir.rglob(first["result"]))
        services = dataclasses.replace(self.handlers.services, resolve_result=lambda path: found)
        with patch.object(self.handlers, "services", services):
            audio = Path(self.handlers.drama3_result_audio(first["result"]))
            self.addCleanup(lambda: (audio.unlink(missing_ok=True), audio.parent.rmdir()))
            self.assertEqual(audio.read_bytes(), found.read_bytes())
            self.assertEqual(self.handlers.drama3_result_details(first["result"])["chunks"][1]["ok"], True)
        with self.assertRaisesRegex(gr.Error, "file name"):
            self.handlers.drama3_result_audio("nothing.mp3")

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
            "/generate_drama3_voices_batch", "/generate_drama3_sounds_batch", "/generate_drama3_batch",
            "/preload_drama3_models", "/drama3_result_audio")}
        self.assertEqual(parameters, {
            "/generate_drama3_voices_batch": ["briefs", "language"],
            "/generate_drama3_sounds_batch": ["sounds"],
            "/generate_drama3_batch": ["jobs", "fix_rounds"],
            "/drama3_result_audio": ["result"],
            "/preload_drama3_models": ["voice_engine", "script_mode"],
        })
        config = json.loads(json.dumps(demo.get_config_file(), default=str))
        wanted = {name.lstrip("/") for name in parameters}
        triggers = {dep["api_name"]: dep["targets"][0][0] for dep in config["dependencies"]
                    if dep.get("api_name") in wanted}
        components = {component["id"]: component for component in config["components"]}
        self.assertEqual(len(triggers), 5)
        self.assertTrue(all(components[target]["props"].get("visible") is False for target in triggers.values()))


if __name__ == "__main__":
    unittest.main()
