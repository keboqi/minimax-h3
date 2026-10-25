"""Snapshot preload: models load before the workspace opens, then the UI waits for restore."""
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app.drama3.serving import server
from h3_ui import snapshot_preload


class RunBeforeServingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ready = Path(self.directory.name) / "ready"
        self.app = SimpleNamespace(HTTP=Mock())
        self.seen = {}

        def preload(app):
            self.seen.update(capture=server.SNAPSHOT_CAPTURE.is_set(),
                             commit=os.environ.get("H3_MODAL_COMMIT_VOLUME"), app=app)

        self.preload = Mock(side_effect=preload)
        patcher = patch.dict(snapshot_preload.PRELOADERS, {"video": self.preload})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, env, stdin="serve\n"):
        with patch.dict(os.environ, env, clear=False), patch("sys.stdin", io.StringIO(stdin)):
            snapshot_preload.run_before_serving(self.app)
            return dict(os.environ)

    def test_without_preload_serves_at_once(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(snapshot_preload.PRELOAD_ENV, None)
            with patch("sys.stdin", io.StringIO("")):
                snapshot_preload.run_before_serving(self.app)
        self.preload.assert_not_called()
        self.app.HTTP.close.assert_not_called()

    def test_preload_runs_in_capture_mode_then_waits_for_restore(self):
        env = self.run_with({"H3_SNAPSHOT_PRELOAD": "Video", "H3_SNAPSHOT_READY_FILE": str(self.ready),
                             "H3_MODAL_COMMIT_VOLUME": "minimax-h3-data"})
        self.assertEqual(self.seen, {"capture": True, "commit": None, "app": self.app})
        self.assertFalse(server.SNAPSHOT_CAPTURE.is_set())
        self.assertEqual(env["H3_MODAL_COMMIT_VOLUME"], "minimax-h3-data")
        self.assertNotIn("H3_SNAPSHOT_PRELOAD", env)
        self.assertEqual(self.ready.read_text(encoding="utf-8"), "video")
        self.app.HTTP.close.assert_called_once()

    def test_closed_stdin_stops_before_serving(self):
        with self.assertRaisesRegex(SystemExit, "stdin closed"):
            self.run_with({"H3_SNAPSHOT_PRELOAD": "video", "H3_SNAPSHOT_READY_FILE": str(self.ready)}, stdin="")

    def test_failed_preload_reports_nothing_ready(self):
        self.preload.side_effect = RuntimeError("no room")
        with self.assertRaisesRegex(RuntimeError, "no room"):
            self.run_with({"H3_SNAPSHOT_PRELOAD": "video", "H3_SNAPSHOT_READY_FILE": str(self.ready),
                           "H3_MODAL_COMMIT_VOLUME": "minimax-h3-data"})
        self.assertFalse(self.ready.exists())
        self.assertFalse(server.SNAPSHOT_CAPTURE.is_set())

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "must be one of"):
            self.run_with({"H3_SNAPSHOT_PRELOAD": "music"})
        self.preload.assert_not_called()


class PreloaderTests(unittest.TestCase):
    def test_video_warmup_runs_defaults_and_removes_its_result(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "h3_t2va_1.mp4"
            sidecar = Path(directory) / "h3_t2va_1.mp4.settings.json"

            def generate(**values):
                self.assertEqual(values["model_profile"], "Singularity")
                self.assertEqual(values["prompt"], snapshot_preload.VIDEO_WARMUP_PROMPT)
                yield None, "Validating request"
                video.write_bytes(b"clip")
                sidecar.write_text("{}", encoding="utf-8")
                yield str(video), "Done"

            snapshot_preload.preload_video(SimpleNamespace(generate=generate))
            self.assertFalse(video.exists())
            self.assertFalse(sidecar.exists())

    def test_video_warmup_without_result_fails(self):
        def generate(**values):
            yield None, "Stopped"

        with self.assertRaisesRegex(RuntimeError, "no result: Stopped"):
            snapshot_preload.preload_video(SimpleNamespace(generate=generate))

    def test_soundstage_requires_every_default_model(self):
        handlers = SimpleNamespace(preload_drama3_models=Mock(return_value=iter(["Preload finished."])))
        app = SimpleNamespace(drama3_handlers=handlers)
        with patch("h3_app.audio_tools.loaded_models", return_value={"voice_design"}), \
                patch.object(server, "server_ready", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "did not load stable_audio, SGLang-Omni"):
                snapshot_preload.preload_soundstage(app)
        handlers.preload_drama3_models.assert_called_once_with(
            "Qwen3-TTS VoiceDesign", "Advanced (sound enhanced)")

    def test_soundstage_with_every_model_loaded(self):
        handlers = SimpleNamespace(preload_drama3_models=Mock(return_value=iter(["Preload finished."])))
        with patch("h3_app.audio_tools.loaded_models", return_value={"voice_design", "stable_audio"}), \
                patch.object(server, "server_ready", return_value=True):
            snapshot_preload.preload_soundstage(SimpleNamespace(drama3_handlers=handlers))


if __name__ == "__main__":
    unittest.main()
