"""Fibo request, queue, UI and isolated-worker contracts without model downloads."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from custom_nodes.H3Fibo2 import NODE_CLASS_MAPPINGS, worker
from custom_nodes.H3Fibo2 import nodes as bridge
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import (
    generate_optional_engine,
    validate_request,
)
from h3_app.generation.requests import Fibo2Request
from h3_app.workflows.fibo2 import build_fibo2_graph


def inputs(graph, kind):
    return [node["inputs"] for node in graph.values() if node["class_type"] == kind]


class FiboRequestTests(unittest.TestCase):
    def test_invalid_requests_fail_before_queue_or_downloads(self):
        request = Fibo2Request("A fox")
        for changes, message in (
            ({"prompt": " "}, "prompt"), ({"prompt": "{bad"}, "JSON"),
            ({"prompt": "[]"}, "object"), ({"model_choice": "unknown"}, "Choose"),
            ({"reference_images": ("a",) * 5}, "four"),
            ({"mask_image": "mask"}, "exactly one"),
            ({"mask_image": "mask", "reference_images": ("a", "b")}, "exactly one"),
            ({"model_choice": "Edit Turbo"}, "requires a reference"),
            ({"model_choice": "Edit Turbo", "reference_images": ("a",)}, "4 steps"),
            ({"cfg": float("nan")}, "CFG"), ({"width": 1000}, "Width"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(H3Error, message):
                validate_request(replace(request, **changes))
        validate_request(request)
        validate_request(replace(request, model_choice="Edit Turbo", reference_images=("a",), steps=4, cfg=1))

    def test_generation_and_edit_graphs_match_the_bundled_node_contract(self):
        for model, refs, mask, steps, cfg in (
            ("Dev", (), None, 50, 5), ("Dev", ("a.png", "b.png"), None, 50, 5),
            ("Edit Turbo", ("a.png",), "mask.png", 4, 1),
        ):
            graph = build_fibo2_graph(prompt="change the sky", model_choice=model,
                reference_images=refs, mask_image=mask, steps=steps, cfg=cfg,
                output_stamp="123", output_nonce="abcd")
            node, = inputs(graph, "H3Fibo2Generate")
            contract = NODE_CLASS_MAPPINGS["H3Fibo2Generate"].INPUT_TYPES()
            self.assertLessEqual(set(contract["required"]), set(node))
            self.assertLessEqual(set(node), set(contract["required"]) | set(contract["optional"]))
            for index, path in enumerate(refs, 1):
                self.assertEqual(graph[node[f"image_{index}"][0]]["inputs"]["image"], path)
            if mask:
                self.assertEqual(node["mask_image"][1], 0, "Use white/black pixels, not inverted alpha")
            self.assertEqual(inputs(graph, "SaveImage")[0]["filename_prefix"],
                             "h3/image_staging/fibo2_123_abcd")

    def test_queue_requires_only_bridge_nodes_and_records_edit_metadata(self):
        request = Fibo2Request("change sky", "Edit Turbo", ("a.png",), "mask.png", seed=7, steps=4, cfg=1)
        models = Mock()
        execution = SimpleNamespace(
            object_info=lambda: {name: {} for name in ("H3Fibo2Generate", "LoadImage", "SaveImage")},
            submit_prompt=Mock(return_value="job"), poll_comfy_progress=lambda *args: iter(()),
            wait_for_history=lambda *args: {},
        )
        services = SimpleNamespace(models=models, execution=execution, media=SimpleNamespace(
            resolve_image_outputs=lambda *args: [Path("fibo2_result.png")]))
        with patch("h3_app.generation.optional_engines.write_snapshot") as snapshot:
            updates = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("Fibo 2.0 complete", updates[-1].status)
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["seed"], settings["reference_count"], settings["masked_edit"]), (7, 1, True))
        self.assertNotIn("mask_image", settings)
        models.ensure_model_keys.assert_called_once_with(())
        execution.object_info = dict
        execution.submit_prompt.reset_mock()
        updates = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("Missing nodes", updates[-1].status)
        execution.submit_prompt.assert_not_called()


class FiboWorkerTests(unittest.TestCase):
    def fake_dependencies(self):
        pipe = Mock()
        pipeline = Mock()
        pipeline.from_pretrained.return_value.to.return_value = pipe
        generator = Mock()
        torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True),
                                Generator=Mock(return_value=generator))
        modules = {"torch": torch, "diffusers": SimpleNamespace(BriaFibo2Pipeline=pipeline),
                   "huggingface_hub": SimpleNamespace(get_token=lambda: None)}
        return modules, pipeline, pipe, torch

    def test_pipeline_receives_edit_references_mask_seed_and_turbo_parameters(self):
        modules, pipeline, pipe, torch = self.fake_dependencies()
        with tempfile.TemporaryDirectory() as root:
            image, mask = Path(root) / "input.png", Path(root) / "mask.png"
            Image.new("RGB", (32, 32), "red").save(image)
            Image.new("L", (32, 32), 255).save(mask)
            request = dict(model_choice="Edit Turbo", prompt="make blue", references=[str(image)], mask=str(mask),
                           steps=4, cfg=1, seed=19, width=1024, height=1024, cache_dir=root,
                           output=str(Path(root) / "output.png"))
            pipe.return_value.images = [Image.new("RGB", (32, 32), "blue")]
            progress = Mock()
            with patch.dict(sys.modules, modules), patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
                worker.run(request, on_step=progress)
            self.assertTrue(Path(request["output"]).is_file())
        self.assertEqual(pipeline.from_pretrained.call_args.args, ("briaai/Fibo-2.0-edit-turbo",))
        self.assertEqual(pipeline.from_pretrained.call_args.kwargs["token"], "test-token")
        kwargs = pipe.call_args.kwargs
        self.assertEqual((kwargs["num_inference_steps"], kwargs["guidance_scale"]), (4, 1))
        self.assertEqual((len(kwargs["image"]), kwargs["mask"].getpixel((0, 0))), (1, 255))
        torch.Generator.return_value.manual_seed.assert_called_once_with(19)
        tensors = {"latents": object()}
        self.assertIs(kwargs["callback_on_step_end"](None, 2, None, tensors), tensors)
        progress.assert_called_once_with(3)

    def test_mask_size_is_validated_before_loading_weights(self):
        modules, pipeline, _, _ = self.fake_dependencies()
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, modules):
            image, mask = Path(root) / "input.png", Path(root) / "mask.png"
            Image.new("RGB", (32, 32)).save(image)
            Image.new("L", (16, 32)).save(mask)
            with self.assertRaisesRegex(ValueError, "same dimensions"):
                worker.run(dict(model_choice="Dev", references=[str(image)], mask=str(mask)), on_step=Mock())
        pipeline.from_pretrained.assert_not_called()

    def test_interrupt_terminates_and_reaps_worker(self):
        process = Mock()
        process.poll.return_value = None
        check = Mock(side_effect=[None, RuntimeError("cancelled")])
        with tempfile.TemporaryDirectory() as root, patch.object(bridge.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                bridge.run_worker("python", Path(root) / "request.json", check_cancelled=check, progress=Mock())
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=2)

    def test_progress_write_survives_windows_sharing_errors(self):
        real_replace, attempts = Path.replace, []

        def replace_while_read(self, target):
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError(13, "The process cannot access the file")
            return real_replace(self, target)

        def run(_request, *, on_step):
            on_step(1)
            on_step(2)

        with tempfile.TemporaryDirectory() as root:
            request = Path(root) / "request.json"
            request.write_text("{}", encoding="utf-8")
            with patch.object(worker, "run", run), patch.object(sys, "argv", ["worker.py", str(request)]), \
                    patch.object(Path, "replace", replace_while_read):
                worker.main()
            self.assertEqual(len(attempts), 2)
            self.assertEqual(request.with_suffix(".progress").read_text(encoding="utf-8"), "2")

    def test_progress_read_survives_windows_sharing_errors(self):
        process = Mock(returncode=0)
        process.poll.side_effect = [None, None, 0, 0]
        real_read, reads = Path.read_text, []

        def read_while_replaced(self, *args, **kwargs):
            reads.append(self)
            if len(reads) == 1:
                raise PermissionError(13, "The process cannot access the file")
            return real_read(self, *args, **kwargs)

        progress = Mock()
        with tempfile.TemporaryDirectory() as root:
            request = Path(root) / "request.json"
            request.with_suffix(".progress").write_text("3", encoding="utf-8")
            with patch.object(bridge.subprocess, "Popen", return_value=process), \
                    patch.object(bridge.time, "sleep"), patch.object(Path, "read_text", read_while_replaced):
                bridge.run_worker("python", request, check_cancelled=Mock(), progress=progress)
        self.assertEqual(len(reads), 2)
        progress.assert_called_once_with(3)


class FiboProvisionTests(unittest.TestCase):
    def test_installer_targets_isolated_python_and_pins_diffusers(self):
        from h3_requirements import fibo2_install_commands
        from h3_sources import FIBO2_DIFFUSERS_REF
        for root, suffix in ((PurePosixPath("/comfy/.fibo2-venv"), "bin/python"),
                             (PureWindowsPath("C:/comfy/.fibo2-venv"), "Scripts/python.exe")):
            commands = fibo2_install_commands(root)
            install = commands[1]
            self.assertEqual(install[install.index("--python") + 1], str(root / suffix))
            self.assertNotIn("--system", install)
            self.assertTrue(any(FIBO2_DIFFUSERS_REF in arg for arg in install))
            self.assertIn("BriaFibo2Pipeline", commands[-1][-1])

    def test_navigation_and_persistence_include_fibo_and_spectrum(self):
        from h3_app.capabilities import engines_for_task
        from h3_ui.fibo2_view import fibo2_sampling_defaults
        from h3_ui.persistence import PERSISTED_NAMES
        self.assertIn("fibo2", {engine.id for engine in engines_for_task("Image")})
        self.assertNotIn("fibo2", {engine.id for engine in engines_for_task("Video")})
        self.assertIn("fibo2.model", PERSISTED_NAMES)
        self.assertIn("moss_image.accelerate", PERSISTED_NAMES)
        steps, cfg = fibo2_sampling_defaults("Edit Turbo")
        self.assertEqual((steps["value"], cfg["value"], steps["interactive"]), (4, 1, False))


if __name__ == "__main__":
    unittest.main()
