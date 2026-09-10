"""Behavioral regressions for resource selection, maintenance, and execution."""

import inspect
import json
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.config import RuntimeConfig
from h3_app.execution import Submission, ExecutionRunner
from h3_app.errors import H3Error
from h3_app.jobs import JobCoordinator, CURRENT_JOB, JobCancelled
from h3_app.processes import run_process
from h3_app.resources import resolve_decoders
from h3_app.settings import GenerationRequest, OutputSettings, resolve_settings
import websocket


class DecoderTests(unittest.TestCase):
    def test_output_decoders_are_independent_of_inactive_preferences(self):
        for fmt, image, needs_video in (
            ("Audio", "", False),
            ("Image", "Single-frame 500K", False),
            ("Image", "Official video VAE", True),
            ("Video", "", True),
        ):
            for trt, int8 in ((False, False), (True, False), (False, True)):
                with self.subTest(fmt=fmt, image=image, trt=trt, int8=int8):
                    plan = resolve_settings(
                        GenerationRequest(
                            output=OutputSettings(result_format=fmt, image_vae=image),
                            use_trt_vae=trt,
                            use_int8_vae=int8,
                        )
                    )
                    self.assertEqual(plan.effective.use_trt_vae, trt and needs_video)
                    self.assertEqual(plan.effective.use_int8_vae, int8 and needs_video)
                    self.assertEqual(plan.requested.use_trt_vae, trt)
        self.assertEqual(
            resolve_decoders(
                "Audio", "", use_trt_vae=True, use_int8_vae=True
            ).optional_model_keys,
            (),
        )
        with self.assertRaises(ValueError):
            resolve_decoders("Video", "", use_trt_vae=True, use_int8_vae=True)

    def test_generation_does_not_prepare_unused_video_decoder(self):
        import gradio_app as app

        for fmt in ("Audio", "Image"):
            kwargs = {
                name: app.UI_DEFAULTS.get(name)
                for name, p in inspect.signature(app.generate).parameters.items()
                if p.default is inspect.Parameter.empty
            }
            kwargs.update(
                prompt="Fixture",
                result_format=fmt,
                image_vae=app.SINGLE_FRAME_IMAGE_VAE,
                latent_upscale=False,
                use_trt_vae=True,
                image_frames=1,
                sol_step_off=0.0,
                sol_sink_tokens=0,
                progress=lambda *a, **k: None,
            )
            with (
                patch.object(app, "unload_prompt_rewriter"),
                patch.object(
                    app,
                    "load_model_config",
                    return_value=app.ModelConfig(
                        {
                            "speed": app.ModelProfile(
                                "Speed", "fl.safetensors", "ref.safetensors"
                            )
                        },
                        "speed",
                        "text",
                        "video",
                        "audio",
                        image_vae_500k="image",
                    ),
                ),
                patch.object(
                    app,
                    "h3_text_encoder_settings",
                    return_value=("text_encoder", "encoder", False),
                ),
                patch.object(app, "model_file_is_ready", return_value=True),
                patch.object(
                    app, "ensure_h3_text_encoder", return_value=("encoder", False)
                ),
                patch.object(app, "ensure_single_frame_image_vae"),
                patch.object(
                    app,
                    "ensure_trt_video_vae_engine",
                    side_effect=AssertionError("unused TRT"),
                ) as trt,
                patch.object(
                    app,
                    "ensure_int8_video_vae",
                    side_effect=AssertionError("unused INT8"),
                ) as int8,
                patch.object(
                    app,
                    "ensure_profile_model",
                    side_effect=H3Error("Reached model preparation"),
                ),
            ):
                updates = list(app.generate(**kwargs))
            trt.assert_not_called()
            int8.assert_not_called()
            self.assertNotIn("unused", updates[-1][1])
            self.assertIn("Reached model preparation", updates[-1][1])


class MaintenanceTests(unittest.TestCase):
    def test_maintenance_waits_and_nested_generation_does_not_relock(self):
        jobs = JobCoordinator()
        waiting, acquired = Event(), Event()

        def maintenance():
            waiting.set()
            with jobs.maintenance("compile"):
                acquired.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with jobs.run("session", "h3") as job:
                token = CURRENT_JOB.set(job)
                try:
                    with jobs.maintenance("automatic"):
                        self.assertIs(CURRENT_JOB.get(), job)
                    future = pool.submit(maintenance)
                    self.assertTrue(waiting.wait(2))
                    self.assertFalse(acquired.is_set())
                finally:
                    CURRENT_JOB.reset(token)
            future.result(timeout=3)
        self.assertTrue(acquired.is_set())
        self.assertFalse(jobs.gpu.locked())


class ProcessTests(unittest.TestCase):
    def test_cancel_reaps_child_cleans_output_and_releases_lease(self):
        jobs = JobCoordinator()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "partial.mp4"

            def worker():
                with jobs.run("session", "gallery") as job:
                    return run_process(
                        [
                            sys.executable,
                            "-c",
                            "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text('partial');time.sleep(30)",
                            str(output),
                        ],
                        timeout=10,
                        check_cancelled=job.check,
                        partial_outputs=(output,),
                    )

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(worker)
                deadline = time.monotonic() + 5
                while not output.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(output.exists())
                jobs.cancel("session", "gallery", Mock(), Mock())
                with self.assertRaises(JobCancelled):
                    future.result(timeout=4)
            self.assertFalse(output.exists())
            self.assertFalse(jobs.gpu.locked())

    def test_timeout_and_bounded_diagnostics(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_process(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                timeout=0.2,
                check_cancelled=lambda: None,
            )
        result = run_process(
            [
                sys.executable,
                "-c",
                "import sys;sys.stdout.write('x'*200000);sys.stderr.write('y'*200000)",
            ],
            timeout=5,
            check_cancelled=lambda: None,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout), 65536)
        self.assertEqual(len(result.stderr), 65536)


class ExecutionTests(unittest.TestCase):
    def test_websocket_fallback_does_not_restart_deadline(self):
        now = [0.0]
        client = Mock(timeout=60)
        client.get.return_value.json.return_value = {}
        socket = Mock()

        def disconnect():
            now[0] = 9.0
            raise websocket.WebSocketException("lost")

        socket.recv.side_effect = disconnect
        submission = Submission(
            "id",
            {},
            client,
            10,
            10,
            1,
            socket=socket,
            clock=lambda: now[0],
            sleeper=lambda dt: now.__setitem__(0, now[0] + dt),
        )
        with self.assertRaisesRegex(H3Error, "timed out"):
            list(submission.progress())
        self.assertEqual(now[0], 10)
        socket.close.assert_called_once()
        calls = client.get.call_count
        with self.assertRaisesRegex(H3Error, "timed out"):
            submission.history()
        self.assertEqual(client.get.call_count, calls)

    def test_failed_submission_closes_socket_and_success_keeps_identity(self):
        config = RuntimeConfig.from_environment(Path("."), {})
        client = Mock(timeout=60)
        client.websocket_url.return_value = "ws://fixture/ws"
        client.post.return_value.json.return_value = {"prompt_id": "prompt"}
        socket = Mock()
        jobs = JobCoordinator()
        runner = ExecutionRunner(
            config, client, jobs, connect=Mock(return_value=socket)
        )
        with jobs.run("session", "h3") as job:
            handle = runner.submit(
                {"1": {"inputs": {"filename_prefix": "h3/output"}}}, "client", job
            )
            self.assertIsInstance(handle, str)
            self.assertEqual(handle, "prompt")
            self.assertIn(
                job.output_token,
                handle.submission.graph["1"]["inputs"]["filename_prefix"],
            )
            self.assertEqual(job.prompt_id, "prompt")
        socket.close.assert_called_once()
        socket.reset_mock()
        client.post.side_effect = RuntimeError("submission failed")
        with self.assertRaises(RuntimeError):
            runner.submit({}, "client")
        socket.close.assert_called_once()

    def test_domain_modules_import_without_gradio(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys;import h3_app.model_service,h3_app.workflows.h3,h3_app.workflows.ltx,h3_app.workflows.music,h3_app.generation.h3,h3_app.generation.ltx,h3_app.generation.music,h3_app.media_tools,h3_app.prompt_service;assert 'gradio' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
