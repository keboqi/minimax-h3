"""Fibo request, native ComfyUI graph, queue and UI contracts without model downloads."""

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.errors import H3Error
from h3_app.generation.optional_engines import (
    generate_optional_engine,
    validate_request,
)
from h3_app.generation.requests import Fibo2Request
from h3_app.workflows.fibo2 import build_fibo2_graph, fibo2_model_keys
from h3_models import FIBO2_MODEL_KEYS, FIBO2_PRECISION_INT8, LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS


def inputs(graph, kind):
    return [node["inputs"] for node in graph.values() if node["class_type"] == kind]


def source(graph, link):
    return graph[link[0]]


class FiboRequestTests(unittest.TestCase):
    def test_invalid_requests_fail_before_queue_or_downloads(self):
        request = Fibo2Request("A fox")
        for changes, message in (
            ({"prompt": " "}, "prompt"), ({"prompt": "{bad"}, "JSON"),
            ({"prompt": "[]"}, "object"), ({"model_choice": "unknown"}, "Choose"),
            ({"precision": "FP4"}, "precision"),
            ({"reference_images": ("a",) * 5}, "four"),
            ({"model_choice": "Edit Turbo"}, "requires a reference"),
            ({"model_choice": "Edit Turbo", "reference_images": ("a",)}, "4 steps"),
            ({"cfg": float("nan")}, "CFG"), ({"width": 1000}, "Width"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(H3Error, message):
                validate_request(replace(request, **changes))
        validate_request(request)
        validate_request(replace(request, model_choice="Edit Turbo", reference_images=("a",), steps=4, cfg=1))

    def test_defaults_follow_the_native_recipe(self):
        request = Fibo2Request("A fox")
        self.assertEqual((request.model_choice, request.precision, request.steps, request.cfg), ("Dev", "BF16", 30, 5.0))


class FiboGraphTests(unittest.TestCase):
    def test_text_to_image_uses_native_nodes_and_the_shared_int8_encoder(self):
        graph = build_fibo2_graph(prompt="a fox", width=1536, height=1024, seed=3,
                                  output_stamp="123", output_nonce="abcd")
        self.assertNotIn("H3Fibo2Generate", {node["class_type"] for node in graph.values()})
        encode, = inputs(graph, "TextEncodeFibo2")
        self.assertEqual((encode["width"], encode["height"], encode["negative_prompt"]), (1536, 1024, ""))
        self.assertFalse([key for key in encode if key.startswith("images.")])
        clip, = inputs(graph, "CLIPLoader")
        self.assertEqual((clip["type"], clip["clip_name"]),
                         ("fibo2", MODEL_SPECS["flux3_text_encoder"].local_name))
        self.assertEqual(inputs(graph, "UNETLoader")[0]["unet_name"], "fibo2_dev_bf16.safetensors")
        self.assertEqual(inputs(graph, "VAELoader")[0]["vae_name"], MODEL_SPECS["flux3_vae"].local_name)
        sampler, = inputs(graph, "KSampler")
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]),
                         (30, 5.0, "euler", "simple"))
        self.assertEqual([source(graph, sampler[name])["class_type"] for name in ("positive", "negative", "latent_image")],
                         ["TextEncodeFibo2"] * 3)
        self.assertEqual([sampler[name][1] for name in ("positive", "negative", "latent_image")], [0, 1, 2])
        self.assertEqual(inputs(graph, "SaveImage")[0]["filename_prefix"], "h3/image_staging/fibo2_123_abcd")

    def test_edits_wire_ordered_references_bf16_encoder_and_auto_size(self):
        graph = build_fibo2_graph(prompt="put <image_2> on <image_1>", model_choice="Edit Turbo",
                                  precision=FIBO2_PRECISION_INT8, reference_images=("a.png", "b.png"),
                                  width=1536, height=1024, steps=4, cfg=1)
        encode, = inputs(graph, "TextEncodeFibo2")
        self.assertEqual([source(graph, encode[f"images.image_{i}"])["inputs"]["image"] for i in (1, 2)],
                         ["a.png", "b.png"])
        self.assertEqual(source(graph, encode["vae"])["class_type"], "VAELoader")
        self.assertEqual((encode["width"], encode["height"]), (0, 0))
        self.assertEqual(inputs(graph, "CLIPLoader")[0]["clip_name"], "qwen3vl_4b_bf16.safetensors")
        self.assertEqual(inputs(graph, "UNETLoader")[0]["unet_name"], "fibo2_edit_turbo_int8_convrot.safetensors")
        fixed = build_fibo2_graph(prompt="edit", reference_images=("a.png",), width=1536, height=1024,
                                  auto_size=False)
        self.assertEqual((inputs(fixed, "TextEncodeFibo2")[0]["width"], inputs(fixed, "TextEncodeFibo2")[0]["height"]),
                         (1536, 1024))

    def test_model_keys_are_ungated_and_registered(self):
        self.assertEqual(fibo2_model_keys("Dev", "BF16", False), ("fibo2_dev_bf16", "flux3_text_encoder", "flux3_vae"))
        self.assertEqual(fibo2_model_keys("Edit Turbo", "BF16", True)[:2], ("fibo2_edit_turbo_bf16", "fibo2_text_bf16"))
        for key in FIBO2_MODEL_KEYS:
            self.assertIn(key, LAZY_OPTIONAL_MODEL_KEYS)
            if key.startswith("fibo2_"):
                self.assertEqual(MODEL_SPECS[key].repo_id, "Comfy-Org/Fibo-2.0")
                self.assertEqual(len(MODEL_SPECS[key].expected_sha256), 64)


class FiboQueueTests(unittest.TestCase):
    def services(self, info):
        execution = SimpleNamespace(
            object_info=lambda: info, submit_prompt=Mock(return_value="job"),
            poll_comfy_progress=lambda *args: iter(()), wait_for_history=lambda *args: {},
        )
        return SimpleNamespace(models=Mock(), execution=execution, media=SimpleNamespace(
            resolve_image_outputs=lambda *args: [Path("fibo2_result.png")]))

    def test_queue_downloads_edit_weights_and_records_metadata(self):
        request = Fibo2Request("change sky", "Edit Turbo", reference_images=("a.png",), seed=7, steps=4, cfg=1)
        info = {name: {} for name in ("UNETLoader", "VAELoader", "LoadImage", "TextEncodeFibo2",
                                      "KSampler", "VAEDecode", "SaveImage")}
        info["CLIPLoader"] = {"input": {"required": {"type": [["flux3", "fibo2"]]}}}
        services = self.services(info)
        with patch("h3_app.generation.optional_engines.write_snapshot") as snapshot:
            updates = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("Fibo 2.0 complete", updates[-1].status)
        self.assertTrue(any("auto size" in update.status for update in updates))
        services.models.ensure_model_keys.assert_called_once_with(
            ("fibo2_edit_turbo_bf16", "fibo2_text_bf16", "flux3_vae"))
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["seed"], settings["reference_count"], settings["text_encoder"]),
                         (7, 1, "qwen3vl_4b_bf16.safetensors"))

    def test_backend_without_native_fibo_support_is_refused_before_queueing(self):
        request = Fibo2Request("a fox")
        info = {name: {} for name in ("UNETLoader", "VAELoader", "KSampler", "VAEDecode", "SaveImage")}
        services = self.services(info)
        updates = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("TextEncodeFibo2", updates[-1].status)
        info["TextEncodeFibo2"] = {}
        info["CLIPLoader"] = {"input": {"required": {"type": [["flux3"]]}}}
        updates = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("Fibo 2.0 text encoder", updates[-1].status)
        services.execution.submit_prompt.assert_not_called()
        services.models.ensure_model_keys.assert_not_called()


class FiboIntegrationTests(unittest.TestCase):
    def test_no_bridge_node_or_isolated_runtime_remains(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "custom_nodes" / "H3Fibo2").exists())
        for filename in ("setup_h3.py", "modal_h3.py", "h3_requirements.py", "h3_sources.py"):
            text = (root / filename).read_text(encoding="utf-8")
            self.assertNotIn("H3Fibo2", text)
            self.assertNotIn("fibo2-venv", text)

    def test_storage_console_groups_and_presets_fibo(self):
        from h3_storage_console import catalog_groups, model_presets
        self.assertEqual(set(catalog_groups()["Fibo 2.0"]), {key for key in FIBO2_MODEL_KEYS if key.startswith("fibo2_")})
        presets = model_presets()
        self.assertEqual(presets["Fibo 2.0 defaults"], ("fibo2_dev_bf16", "flux3_text_encoder", "flux3_vae"))
        self.assertIn("fibo2_text_bf16", presets["Fibo 2.0 + editing"])

    def test_navigation_and_persistence_include_fibo_and_spectrum(self):
        from h3_app.capabilities import engines_for_task
        from h3_ui.fibo2_view import fibo2_sampling_defaults
        from h3_ui.persistence import PERSISTED_NAMES
        self.assertIn("fibo2", {engine.id for engine in engines_for_task("Image")})
        self.assertNotIn("fibo2", {engine.id for engine in engines_for_task("Video")})
        for name in ("model", "precision", "auto_size"):
            self.assertIn(f"fibo2.{name}", PERSISTED_NAMES)
        self.assertIn("moss_image.accelerate", PERSISTED_NAMES)
        steps, cfg = fibo2_sampling_defaults("Edit Turbo")
        self.assertEqual((steps["value"], cfg["value"], steps["interactive"]), (4, 1, False))
        steps, cfg = fibo2_sampling_defaults("Dev")
        self.assertEqual((steps["value"], cfg["value"]), (30, 5))


if __name__ == "__main__":
    unittest.main()
