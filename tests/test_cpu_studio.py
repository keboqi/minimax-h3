"""Full Prism Studio UI runs on CPU with H3_CPU_ONLY=1 and fails fast on GPU actions."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import h3_gpu
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error


ROOT = Path(__file__).resolve().parents[1]

_SUBPROCESS_BOOT_SCRIPT = """
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

class Blocked:
    NAMES = {
        "scipy", "torch", "torchaudio", "torchvision", "transformers", "diffusers",
        "accelerate", "peft", "safetensors", "soundfile", "librosa", "qwen_asr",
        "audio_separator", "tensorrt", "sageattention", "kornia", "ultralytics",
        "pedalboard", "decord", "einops", "timm", "imageio",
    }
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in self.NAMES:
            raise ModuleNotFoundError(f"Blocked GPU package: {name}", name=name)

sys.meta_path.insert(0, Blocked())

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

with TemporaryDirectory() as directory:
    root = Path(directory)
    os.environ["COMFY_DIR"] = str(root / "ComfyUI")
    os.environ["GRADIO_OUTPUT_DIR"] = str(root / "output")
    os.environ["H3_WORKSPACE_DIR"] = str(root / "h3-workspace")
    os.environ["MODELS_CONFIG"] = str(root / "h3_models.json")
    os.environ["COMFY_URL"] = "http://127.0.0.1:1"
    os.environ["H3_CPU_ONLY"] = "1"

    from h3_gpu import CPU_ONLY_MESSAGE
    from h3_ui import application as app

    demo = app.build_ui().queue(default_concurrency_limit=1, max_size=8)
    try:
        server = app.build_server(
            demo, [str(app.OUTPUT_DIR.resolve()), str(app.OUTPUTS_DIR.resolve())],
        )
        client = TestClient(server)

        for route in (
            "/", "/h3/", "/image/", "/song/", "/ltx/",
            "/media/", "/remake/", "/soundstage/", "/storyboard/", "/dubbing/",
        ):
            response = client.get(route)
            assert response.status_code == 200, (route, response.status_code)

        boots = {}
        for route in (
            "/h3/api/boot", "/image/api/boot", "/song/api/boot", "/ltx/api/boot",
            "/media/api/boot", "/remake/api/boot", "/soundstage/api/boot",
            "/storyboard/api/boot", "/dubbing/api/boot",
        ):
            response = client.get(route)
            assert response.status_code == 200, (route, response.status_code, response.text)
            boots[route] = response.json()

        assert boots["/storyboard/api/boot"]["gpu"] is True
        assert boots["/remake/api/boot"]["gpu"] is True
        assert boots["/soundstage/api/boot"]["gpu"] is True
        assert boots["/dubbing/api/boot"]["gpu"] is True
        assert "Remake · Picture 1" in boots["/media/api/boot"]["set_as"]

        redirect = client.get("/comfyui", follow_redirects=False)
        assert redirect.status_code == 307, redirect.status_code

        for comfy_route in ("/comfyui/", "/comfyui/system_stats"):
            response = client.get(comfy_route)
            assert response.status_code == 503, (comfy_route, response.status_code)
            assert response.text == CPU_ONLY_MESSAGE

        try:
            with client.websocket_connect("/comfyui/ws"):
                raise AssertionError("Expected CPU-only /comfyui/ws to close immediately")
        except WebSocketDisconnect:
            pass
    finally:
        demo.close()
        from h3_app.jobs import JOBS
        JOBS.close()

print("CPU_STUDIO_BOOT_OK")
"""


class CpuStudioTests(unittest.TestCase):
    def test_is_cpu_only_flag_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(h3_gpu.is_cpu_only())
        for value in ("1", "true", "TRUE", "yes", "on", " 1 "):
            with patch.dict(os.environ, {h3_gpu.CPU_ONLY_ENV: value}, clear=True):
                self.assertTrue(h3_gpu.is_cpu_only(), value)
        for value in ("", "0", "false", "no", "off"):
            with patch.dict(os.environ, {h3_gpu.CPU_ONLY_ENV: value}, clear=True):
                self.assertFalse(h3_gpu.is_cpu_only(), value)

    def test_full_studio_ui_boots_without_gpu_packages(self):
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_BOOT_SCRIPT],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr[-2000:]}",
        )
        self.assertIn("CPU_STUDIO_BOOT_OK", completed.stdout)

    def test_backend_status_reports_cpu_preview_without_http_requests(self):
        from h3_ui import application as app

        with patch.dict(os.environ, {h3_gpu.CPU_ONLY_ENV: "1"}), \
                patch.object(app.HTTP, "get", side_effect=AssertionError("unexpected ComfyUI HTTP GET")):
            detail = app.backend_status()
            compact = app.compact_backend_status(detail)
        self.assertTrue(detail.startswith("CPU UI preview"))
        self.assertIn("CPU preview", compact)
        self.assertIn("No GPU backend", compact)
        self.assertIn("h3-system-status", compact)
        self.assertNotIn("Backend needs attention", compact)

    def test_gpu_only_actions_fail_fast_under_cpu_only(self):
        import h3_prompt_rewriter
        from h3_app import audio_tools, model_service
        from h3_app.drama3.serving import environment as drama3_env, generate as drama3_generate
        from h3_app.dubbing import moss, nemotron, tools as dub_tools
        from h3_app.model_types import ModelConfig
        from h3_app.workers import env as worker_env
        from h3_ui import application as app

        expected = h3_gpu.CPU_ONLY_MESSAGE
        with TemporaryDirectory() as directory, \
                patch.dict(os.environ, {h3_gpu.CPU_ONLY_ENV: "1"}), \
                patch.object(app.HTTP, "get", side_effect=AssertionError("unexpected GET")), \
                patch.object(app.HTTP, "post", side_effect=AssertionError("unexpected POST")):
            root = Path(directory)
            runtime = RuntimeConfig.from_environment(ROOT, {
                "COMFY_DIR": str(root / "ComfyUI"),
                "GRADIO_OUTPUT_DIR": str(root / "output"),
                "COMFY_URL": "http://127.0.0.1:1",
            })
            dummy_models = ModelConfig(
                profiles={},
                default_profile="singularity",
                text_encoder="text.safetensors",
                video_vae="video.safetensors",
                audio_vae="audio.safetensors",
            )
            no_free = Mock(side_effect=AssertionError("unexpected free_gpu"))

            for fn in (
                lambda: app.wait_for_comfy(timeout=1.0),
                lambda: app.submit_prompt({}, "client"),
                lambda: app.object_info(),
                lambda: app.free_comfy_vram(timeout=1.0),
                lambda: model_service.ensure_models(("text_encoder",), runtime=runtime),
                lambda: model_service.ensure_ltx25_models(runtime=runtime),
                lambda: model_service.ensure_trt_video_vae_engine(dummy_models, runtime=runtime, release_backend=no_free),
                lambda: worker_env.prepare_venv(
                    root / "venv", root / "setup.log",
                    name="Test", label="Test", fingerprint="fp", ready_marker=".ready",
                    install_commands=lambda *_args: [], timeout=10,
                ),
                lambda: audio_tools.run("voice_design", {}, runtime, free_gpu=no_free),
                lambda: audio_tools.preload("voice_design", runtime, free_gpu=no_free),
                lambda: dub_tools.run("separate", {}, runtime, free_gpu=no_free),
                lambda: drama3_env.prepare(root / "sglang-venv"),
                lambda: list(drama3_generate._ready_server(None, None, "SGLang-Omni", no_free, None)),
                lambda: moss.transcribe(runtime, root / "clip.wav", duration=1.0, free_gpu=no_free),
                lambda: nemotron.prepare(root / "nemotron-venv"),
                lambda: nemotron.transcribe(runtime, root / "clip.wav", duration=1.0, free_gpu=no_free),
            ):
                with self.assertRaises(H3Error) as raised:
                    fn()
                self.assertEqual(str(raised.exception), expected)

            message, status = app.unload_all_models()
            self.assertEqual(message, expected)
            self.assertTrue(status.startswith("CPU UI preview"))

            ltx_updates = list(app.MODELS.prepare_ltx25_official_workflow("Text / image to video — single stage"))
            self.assertEqual(len(ltx_updates), 1)
            self.assertEqual(ltx_updates[0][0], expected)

            with self.assertRaises(RuntimeError) as raised_rewriter:
                h3_prompt_rewriter.rewrite_prompt(
                    prompt="A fox runs.", task="t2av", resolution="16:9", duration=5,
                )
            self.assertEqual(str(raised_rewriter.exception), expected)


if __name__ == "__main__":
    unittest.main()
