"""Integration regressions for extracted configuration, media, and generation."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import gradio_app as app
import h3_models
from h3_app.config import RuntimeConfig
from h3_app.gallery_store import gallery_video_paths
from h3_app.jobs import JobCancelled
from h3_app.media_tools import postprocess_video
from h3_app.provenance import copy_media, read_snapshot, write_snapshot


class ConfigurationTests(unittest.TestCase):
    def test_environment_is_captured_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            env = {
                "COMFY_DIR": str(root / "backend"),
                "COMFY_URL": "http://fixture:8188/",
                "GRADIO_OUTPUT_DIR": str(root / "results"),
                "GENERATION_TIMEOUT": "42",
            }
            config = RuntimeConfig.from_environment(root, env)
            env["GENERATION_TIMEOUT"] = "99"
            self.assertEqual(config.generation_timeout, 42)
            self.assertEqual(config.comfy_url, "http://fixture:8188")
            self.assertEqual(config.input_dir, root / "backend/input")
            with self.assertRaises(FrozenInstanceError):
                config.poll_seconds = 2

    def test_local_and_modal_import_the_same_source_catalog(self):
        root = Path(__file__).resolve().parents[1]
        catalog = ast.parse((root / "h3_sources.py").read_text(encoding="utf-8"))
        names = {
            target.id
            for node in catalog.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(len(names), 21)
        for filename in ("setup_h3.py", "modal_h3.py"):
            tree = ast.parse((root / filename).read_text(encoding="utf-8"))
            imports = {
                alias.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module == "h3_sources"
                for alias in node.names
            }
            self.assertLessEqual(names, imports)
        modal = (root / "modal_h3.py").read_text(encoding="utf-8")
        self.assertIn("(LOCAL_SHARED_SOURCES, SHARED_SOURCES)", modal)
        self.assertIn('LOCAL / "h3_app"', modal)
        self.assertIn("LOCAL_UI_PACKAGE,", modal)


class ManifestTests(unittest.TestCase):
    def test_sync_transactions_preserve_disjoint_updates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"

            def plan(**kwargs):
                # Without transaction locking both callers read the empty manifest
                # before either writes its distinct model entry.
                time.sleep(0.05)
                key, spec = kwargs["key"], kwargs["spec"]
                dest = root / key
                dest.write_bytes(b"fixture")
                return dict(
                    key=key,
                    spec=spec,
                    dest=dest,
                    manifest_key=h3_models.model_manifest_key(spec),
                    remote_ok=True,
                    needs_download=False,
                    revision="fixture",
                    blob_id=None,
                    sha256=None,
                    size=7,
                    identity=key,
                )

            with (
                patch.object(h3_models, "_fetch_repositories", return_value={}),
                patch.object(h3_models, "_plan_model", side_effect=plan),
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            h3_models.sync_models,
                            root=root,
                            manifest_path=manifest,
                            model_keys=[key],
                        )
                        for key in ("text_encoder", "audio_vae")
                    ]
                    for future in futures:
                        future.result(timeout=5)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(data["files"]), 2)
            self.assertFalse(list(root.glob("*.partial")))

    def test_manifest_lock_coordinates_separate_processes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            script = (
                "import sys,time;from pathlib import Path;"
                "from h3_models import model_manifest_lock,read_json,write_json_atomic;"
                "p=Path(sys.argv[1]);"
                "\nfor i in range(3):"
                "\n with model_manifest_lock(p):"
                "\n  data=read_json(p,{});time.sleep(.03);"
                "data[sys.argv[2]+str(i)]=i;write_json_atomic(p,data)"
            )
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(path), label],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for label in ("a", "b")
            ]
            try:
                for process in processes:
                    _, error = process.communicate(timeout=15)
                    self.assertEqual(process.returncode, 0, error.decode())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), 6)


class MediaPublicationTests(unittest.TestCase):
    def test_processed_and_copied_media_retain_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"complete source")
            write_snapshot(source, {"settings": {"seed": 12}})
            config = replace(
                RuntimeConfig.from_environment(root, {}), outputs_dir=root / "results"
            )

            def process(command, **kwargs):
                pending = Path(command[-1])
                self.assertEqual(pending.suffix, ".partial")
                pending.write_bytes(b"processed")
                self.assertEqual(gallery_video_paths(runtime=config), [])
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                patch("h3_app.media_tools.has_encoder", return_value=False),
                patch("h3_app.media_tools.run_media_process", side_effect=process),
            ):
                result = postprocess_video(
                    source, "48 fps interpolation", runtime=config
                )
            self.assertEqual(result.read_bytes(), b"processed")
            self.assertEqual(read_snapshot(result)["settings"]["seed"], 12)
            copied = root / "copied.mp4"
            copy_media(result, copied)
            self.assertEqual(read_snapshot(copied)["output"], copied.name)
            self.assertEqual(read_snapshot(copied)["settings"]["seed"], 12)

    def test_input_upscale_resolves_slot_token_with_separate_submission_scope(self):
        from h3_app.outputs import OutputContext, resolve_seedvr2_input_upscale_outputs

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(RuntimeConfig.from_environment(root, {}), output_root=root)
            outputs = root / "h3/input_upscale"
            outputs.mkdir(parents=True)
            image = outputs / "request_first_submission_00001.png"
            image.write_bytes(b"fixture")
            unrelated = outputs / "request_first_other_00002.png"
            unrelated.write_bytes(b"another job")
            context = OutputContext(config, "submission")
            history = {
                "outputs": {
                    "save": {
                        "images": [
                            {
                                "filename": image.name,
                                "subfolder": "h3/input_upscale",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
            for saved in (history, {}):
                resolved = resolve_seedvr2_input_upscale_outputs(
                    saved, time.time(), "request", ["first"], context=context
                )
                self.assertEqual(resolved, {"first": image})

    def test_interrupted_process_does_not_publish_partial_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"complete source")
            config = replace(
                RuntimeConfig.from_environment(root, {}), outputs_dir=root / "results"
            )

            def interrupted(command, **kwargs):
                Path(command[-1]).write_bytes(b"partial")
                raise JobCancelled("Interrupted")

            with (
                patch("h3_app.media_tools.has_encoder", return_value=False),
                patch("h3_app.media_tools.run_media_process", side_effect=interrupted),
            ):
                with self.assertRaises(JobCancelled):
                    postprocess_video(source, "48 fps interpolation", runtime=config)
            self.assertEqual(source.read_bytes(), b"complete source")
            self.assertEqual(list(config.outputs_dir.iterdir()), [])


class GenerationIntegrationTests(unittest.TestCase):
    def test_h3_formats_and_cancelled_finishing_preserve_completed_source(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.mp4"
            source.write_bytes(b"complete")
            models = app.ModelConfig(
                {
                    "speed": app.ModelProfile(
                        "Speed", "fl2va.safetensors", "ref2va.safetensors"
                    )
                },
                "speed",
                "text.safetensors",
                "video.safetensors",
                "audio.safetensors",
                image_vae_500k="image.safetensors",
                turbo_lora="turbo.safetensors",
            )
            for fmt, cancel in (
                ("Video", False),
                ("Image", False),
                ("Audio", False),
                ("Video", True),
            ):
                with self.subTest(fmt=fmt, cancel=cancel), ExitStack() as stack:
                    returns = dict(
                        load_model_config=models,
                        model_file_is_ready=True,
                        ensure_h3_text_encoder=("text.safetensors", False),
                        ensure_profile_model=False,
                        ensure_single_frame_image_vae=False,
                        object_info={},
                        build_fl2va_graph={},
                        submit_prompt="fixture-job",
                        poll_comfy_progress=[],
                        wait_for_history={"outputs": {}},
                        resolve_output=source,
                        resolve_image_outputs=[source],
                        resolve_audio_output=source,
                        postprocess_video=source,
                        unload_prompt_rewriter=None,
                    )
                    for name, value in returns.items():
                        stack.enter_context(patch.object(app, name, return_value=value))
                    stack.enter_context(
                        patch(
                            "h3_app.generation.preparation.required_nodes_for",
                            return_value=set(),
                        )
                    )
                    if cancel:
                        stack.enter_context(
                            patch.object(
                                app,
                                "postprocess_video",
                                side_effect=JobCancelled("Generation interrupted."),
                            )
                        )
                    kwargs = {
                        name: app.UI_DEFAULTS.get(name)
                        for name, param in inspect.signature(
                            app.generate
                        ).parameters.items()
                        if param.default is inspect.Parameter.empty
                    }
                    kwargs.update(
                        prompt="Private prompt",
                        generation_mode="Normal",
                        steps=15,
                        sol_step_off=0.0,
                        sol_sink_tokens=0,
                        attention_mode="Kitchen",
                        latent_upscale=False,
                        cache_mode="Off",
                        seed=123,
                        result_format=fmt,
                        image_vae=app.SINGLE_FRAME_IMAGE_VAE,
                        image_frames=1,
                        use_trt_vae=False,
                        use_int8_vae=False,
                        semantic_bridge=False,
                        postprocess="None",
                        progress=lambda *args, **kwargs: None,
                    )
                    updates = list(app.generate(**kwargs))
                    self.assertIn(
                        "Error:" if cancel else "Completed", updates[-1].status
                    )
                    if cancel:
                        self.assertEqual(updates[-1].output, str(source))
                        self.assertIn("still available", updates[-1].status)
                    self.assertTrue(source.is_file())

    def test_ltx_and_music_sidecars_exclude_private_inputs(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            result = Path(directory) / "result.mp4"
            result.write_bytes(b"fixture")
            for name in (
                "unload_prompt_rewriter",
                "ensure_ltx25_models",
                "ensure_music3_models",
            ):
                stack.enter_context(patch.object(app, name))
            for name in (
                "missing_ltx25_model_names",
                "missing_music3_model_names",
                "poll_comfy_progress",
            ):
                stack.enter_context(patch.object(app, name, return_value=[]))
            stack.enter_context(
                patch.object(
                    app,
                    "object_info",
                    return_value={
                        key: {}
                        for key in app.required_ltx25_nodes(image_to_video=True)
                        | app.required_music3_nodes(tiled_decode=True)
                    },
                )
            )
            stack.enter_context(patch.object(app, "build_ltx25_graph", return_value={}))
            stack.enter_context(
                patch.object(app, "submit_prompt", return_value="fixture")
            )
            stack.enter_context(patch.object(app, "wait_for_history", return_value={}))
            for name in ("resolve_output", "resolve_audio_output"):
                stack.enter_context(patch.object(app, name, return_value=result))
            ltx = list(
                app.generate_ltx25(
                    "Text to video",
                    app.DEFAULT_LTX25_MODEL,
                    "Private prompt",
                    "Private negative",
                    None,
                    5,
                    24,
                    864,
                    480,
                    123,
                    1,
                    "euler",
                    1,
                    progress=lambda *args, **kwargs: None,
                )
            )
            self.assertIn("completed", ltx[-1].status)
            self.assertNotIn("Private", json.dumps(read_snapshot(result)))
            # Use actual defaults to exercise Music graph construction with no backend.
            args = dict(app.MUSIC3_DEFAULTS)
            required = {
                key: args.get(key)
                for key in inspect.signature(app.generate_music3).parameters
                if key != "progress"
            }
            required.update(
                model_choice=app.DEFAULT_MUSIC3_MODEL,
                caption="Private caption",
                lyrics="Private lyrics",
                max_duration=120,
                seed=123,
                progress=lambda *args, **kwargs: None,
            )
            music = list(app.generate_music3(**required))
            self.assertIn("completed", music[-1].status)
            self.assertNotIn("Private", json.dumps(read_snapshot(result)))


if __name__ == "__main__":
    unittest.main()
