from __future__ import annotations

import unittest

from h3_app.workflows.qwen import (
    build_qwen_image21_graph,
    required_qwen_image21_nodes,
)
from h3_models import (
    DEFAULT_QWEN_IMAGE21_MODEL,
    DEFAULT_QWEN_IMAGE21_TEXT_ENCODER,
    MODEL_SPECS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
)
from h3_ui.bindings import qwen_resolution_preset_values


class QwenImage21WorkflowTests(unittest.TestCase):
    def _build(self, references=(), match_input_size=True):
        return build_qwen_image21_graph(
            model_choice="INT8 ConvRot (lower VRAM)",
            text_encoder_choice="INT8 ConvRot (recommended)",
            prompt="A red fox reading a book",
            negative_prompt="blurry",
            reference_images=references,
            width=1024,
            height=768,
            reference_resolution=0,
            match_input_size=match_input_size,
            seed=123,
            steps=25,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            output_stamp="1234",
            output_nonce="abcd",
        )

    @staticmethod
    def _by_type(graph, class_type):
        return [
            (node_id, node)
            for node_id, node in graph.items()
            if node["class_type"] == class_type
        ]

    def test_text_to_image_uses_official_native_nodes(self):
        graph = self._build()
        self.assertFalse(self._by_type(graph, "LoadImage"))
        self.assertFalse(self._by_type(graph, "QwenImage21Cache"))
        encoder = self._by_type(graph, "TextEncodeQwenImage21")[0][1]
        self.assertNotIn("vae", encoder["inputs"])
        latent_id, latent = self._by_type(graph, "EmptyLatentImage")[0]
        self.assertEqual((latent["inputs"]["width"], latent["inputs"]["height"]), (1024, 768))
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["latent_image"], [latent_id, 0])
        self.assertEqual(sampler["inputs"]["cfg"], 1.0)
        saved = self._by_type(graph, "SaveImage")[0][1]
        self.assertTrue(
            saved["inputs"]["filename_prefix"].startswith(
                "h3/image_staging/qwen_image21_"
            )
        )

    def test_edit_uses_references_conditioner_latent_and_cache(self):
        graph = self._build(("target.png", "style.png"))
        loads = self._by_type(graph, "LoadImage")
        self.assertEqual(
            [node["inputs"]["image"] for _, node in loads],
            ["target.png", "style.png"],
        )
        conditioner_id, conditioner = self._by_type(
            graph, "TextEncodeQwenImage21"
        )[0]
        self.assertIn("vae", conditioner["inputs"])
        self.assertEqual(conditioner["inputs"]["images.image_1"], [loads[0][0], 0])
        self.assertEqual(conditioner["inputs"]["images.image_2"], [loads[1][0], 0])
        cache_id, cache = self._by_type(graph, "QwenImage21Cache")[0]
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["model"], [cache_id, 0])
        self.assertEqual(sampler["inputs"]["latent_image"], [conditioner_id, 2])
        self.assertEqual(cache["inputs"]["device"], "auto")

    def test_model_registry_matches_published_repository_layout(self):
        selected = {
            *QWEN_IMAGE21_MODEL_CHOICES.values(),
            *QWEN_IMAGE21_TEXT_ENCODER_CHOICES.values(),
            "qwen_image21_vae",
        }
        self.assertTrue(selected)
        for key in selected:
            self.assertEqual(MODEL_SPECS[key].repo_id, "Comfy-Org/Qwen-Image-2.1")
        self.assertEqual(
            MODEL_SPECS["qwen_image21_vae"].filename,
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        )

    def test_required_nodes_add_edit_only_nodes(self):
        generate = required_qwen_image21_nodes(editing=False)
        edit = required_qwen_image21_nodes(editing=True)
        self.assertNotIn("LoadImage", generate)
        self.assertNotIn("QwenImage21Cache", generate)
        self.assertTrue({"LoadImage", "QwenImage21Cache"} <= edit)

    def test_bf16_is_the_quality_default(self):
        self.assertEqual(DEFAULT_QWEN_IMAGE21_MODEL, "BF16")
        self.assertEqual(DEFAULT_QWEN_IMAGE21_TEXT_ENCODER, "BF16")

    def test_native_resolution_presets(self):
        self.assertEqual(
            qwen_resolution_preset_values("1K · 1:1 · 1024×1024"),
            (1024, 1024),
        )
        self.assertEqual(
            qwen_resolution_preset_values("1K · 16:9 · 1824×1024"),
            (1824, 1024),
        )
        self.assertEqual(
            qwen_resolution_preset_values("2K · 9:16 · 1536×2752"),
            (1536, 2752),
        )

    def test_optional_kitchen_attention_wraps_the_model(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="Transparent glass sculpture",
            negative_prompt="",
            reference_images=(),
            width=2048,
            height=2048,
            reference_resolution=0,
            match_input_size=True,
            seed=42,
            steps=40,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="comfy kitchen attention",
            output_stamp="1234",
            output_nonce="abcd",
        )
        backend_id, backend = self._by_type(graph, "ModelAttentionBackend")[0]
        self.assertEqual(
            backend["inputs"]["attention"], "comfy kitchen attention"
        )
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["model"], [backend_id, 0])


if __name__ == "__main__":
    unittest.main()
