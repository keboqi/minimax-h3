"""Fibo request, native ComfyUI graph, queue and UI contracts without model downloads."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import (
    generate_optional_engine,
    resolve_fibo2_dimensions,
    validate_request,
)
from h3_app.generation.requests import Fibo2Request
from h3_app.workflows.fibo2 import (
    FIBO2_1M_RESOLUTION_PRESETS,
    FIBO2_2M_RESOLUTION_PRESETS,
    FIBO2_EDIT_SIZE_MAX,
    FIBO2_EDIT_SIZE_TRAINED,
    FIBO2_MAX_PIXELS,
    FIBO2_NEGATIVE_T2I,
    build_fibo2_graph,
    fibo2_model_keys,
)
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
        self.assertEqual((encode["width"], encode["height"]), (1536, 1024))
        self.assertEqual(encode["negative_prompt"], FIBO2_NEGATIVE_T2I)
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

    def test_edits_wire_ordered_references_and_the_bf16_encoder(self):
        graph = build_fibo2_graph(prompt="put <image_2> on <image_1>", model_choice="Edit Turbo",
                                  precision=FIBO2_PRECISION_INT8, reference_images=("a.png", "b.png"),
                                  width=0, height=0, steps=4, cfg=1)
        encode, = inputs(graph, "TextEncodeFibo2")
        self.assertEqual([source(graph, encode[f"images.image_{i}"])["inputs"]["image"] for i in (1, 2)],
                         ["a.png", "b.png"])
        self.assertEqual(source(graph, encode["vae"])["class_type"], "VAELoader")
        self.assertEqual((encode["width"], encode["height"]), (0, 0))
        self.assertEqual(inputs(graph, "CLIPLoader")[0]["clip_name"], "qwen3vl_4b_bf16.safetensors")
        self.assertEqual(inputs(graph, "UNETLoader")[0]["unet_name"], "fibo2_edit_turbo_int8_convrot.safetensors")

    def test_text_to_image_uses_bria_t2i_negative_and_edits_keep_the_null_caption(self):
        negative = json.loads(FIBO2_NEGATIVE_T2I)
        self.assertEqual(negative["objects"][0]["description"], "the main subject")
        self.assertIn("motion blur", negative["aesthetics"]["artifacts"])
        dev_edit = build_fibo2_graph(prompt="make it night", reference_images=("a.png",), width=0, height=0)
        turbo_edit = build_fibo2_graph(prompt="make it night", model_choice="Edit Turbo",
                                       reference_images=("a.png",), width=0, height=0, steps=4, cfg=1)
        for graph in (dev_edit, turbo_edit):
            self.assertEqual(inputs(graph, "TextEncodeFibo2")[0]["negative_prompt"], "")


class FiboSizeTests(unittest.TestCase):
    def resolve(self, edit_size, source=(1920, 1080), refs=("a.png",), width=1248, height=832):
        request = Fibo2Request("edit", reference_images=refs, width=width, height=height, edit_size=edit_size)
        with patch("h3_app.generation.optional_engines.first_reference_dimensions", return_value=source):
            return resolve_fibo2_dimensions(request)

    def test_edit_sizes_keep_the_first_reference_aspect_ratio(self):
        self.assertEqual(self.resolve(FIBO2_EDIT_SIZE_TRAINED), (0, 0))  # the node's trained ~1 MP size
        self.assertEqual(self.resolve(QWEN_EDIT_SIZE_MATCH), (1920, 1072))  # own area, under the 2 MP cap
        self.assertEqual(self.resolve(QWEN_EDIT_SIZE_MATCH, source=(4000, 3000)), (1680, 1248))
        self.assertEqual(self.resolve(QWEN_EDIT_SIZE_MATCH, source=(640, 480)), (640, 480))
        self.assertEqual(self.resolve(FIBO2_EDIT_SIZE_MAX, source=(640, 480)), (1680, 1248))
        self.assertEqual(self.resolve(FIBO2_EDIT_SIZE_MAX, source=(1024, 1024)), (1456, 1456))
        self.assertEqual(self.resolve(QWEN_EDIT_SIZE_MANUAL), (1248, 832))
        self.assertEqual(self.resolve(FIBO2_EDIT_SIZE_MAX, refs=()), (1248, 832))  # generation uses width/height
        for size in (self.resolve(FIBO2_EDIT_SIZE_MAX, source=source) for source in ((3000, 1000), (1000, 3000))):
            self.assertLessEqual(size[0] * size[1], FIBO2_MAX_PIXELS)
            self.assertLessEqual(max(size), 2048)
        with self.assertRaisesRegex(H3Error, "edit output size"):
            validate_request(Fibo2Request("edit", edit_size="Huge"))

    def test_resolution_presets_are_trained_1m_and_2m_sizes_on_the_grid(self):
        from h3_ui.qwen_bindings import qwen_resolution_preset_values
        self.assertIn("1M · 16:9 · 1424×800", FIBO2_1M_RESOLUTION_PRESETS)
        for presets, low, high in ((FIBO2_1M_RESOLUTION_PRESETS, 0.9, 1.15), (FIBO2_2M_RESOLUTION_PRESETS, 1.9, 2.2)):
            for name in presets:
                width, height = qwen_resolution_preset_values(name)
                self.assertEqual((width % 16, height % 16), (0, 0), name)
                self.assertTrue(low <= width * height / 1024**2 <= high, name)
                self.assertLessEqual(width * height, FIBO2_MAX_PIXELS, name)
                validate_request(Fibo2Request("a fox", width=width, height=height))

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
            updates = list(generate_optional_engine(request, services, SimpleNamespace(input_dir=None)))
        self.assertIn("Fibo 2.0 complete", updates[-1].status)
        self.assertTrue(any("trained size" in update.status for update in updates))
        self.assertEqual(inputs(services.execution.submit_prompt.call_args.args[0], "TextEncodeFibo2")[0]["width"], 0)
        services.models.ensure_model_keys.assert_called_once_with(
            ("fibo2_edit_turbo_bf16", "fibo2_text_bf16", "flux3_vae"))
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["seed"], settings["reference_count"], settings["text_encoder"]),
                         (7, 1, "qwen3vl_4b_bf16.safetensors"))

    def test_backend_without_native_fibo_support_is_refused_before_queueing(self):
        request = Fibo2Request("a fox")
        info = {name: {} for name in ("UNETLoader", "VAELoader", "KSampler", "VAEDecode", "SaveImage")}
        services = self.services(info)
        updates = list(generate_optional_engine(request, services, SimpleNamespace(input_dir=None)))
        self.assertIn("TextEncodeFibo2", updates[-1].status)
        info["TextEncodeFibo2"] = {}
        info["CLIPLoader"] = {"input": {"required": {"type": [["flux3"]]}}}
        updates = list(generate_optional_engine(request, services, SimpleNamespace(input_dir=None)))
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
        for name in ("model", "precision", "edit_size"):
            self.assertIn(f"fibo2.{name}", PERSISTED_NAMES)
        self.assertIn("moss_image.accelerate", PERSISTED_NAMES)
        steps, cfg = fibo2_sampling_defaults("Edit Turbo")
        self.assertEqual((steps["value"], cfg["value"], steps["interactive"]), (4, 1, False))
        steps, cfg = fibo2_sampling_defaults("Dev")
        self.assertEqual((steps["value"], cfg["value"]), (30, 5))


if __name__ == "__main__":
    unittest.main()
