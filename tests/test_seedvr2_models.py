"""SeedVR2 model choices and shared image/video defaults."""

import unittest

from h3_app.model_types import ModelConfig, seedvr2_upscale_model_names
from h3_app.workflows.upscale import (
    build_seedvr2_image_upscale_graph,
    build_seedvr2_upscale_graph,
)
from h3_models import DEFAULT_SEEDVR2_MODEL, MODEL_SPECS, SEEDVR2_MODEL_CHOICES


class SeedVR2ModelTests(unittest.TestCase):
    def setUp(self):
        self.models = ModelConfig(
            profiles={},
            default_profile="",
            text_encoder="text.safetensors",
            video_vae="video_vae.safetensors",
            audio_vae="audio_vae.safetensors",
            seedvr2_dit="seedvr2_7b_nvfp4.safetensors",
            seedvr2_models={
                "7B NVFP4": "seedvr2_7b_nvfp4.safetensors",
            },
            seedvr2_vae="seedvr2_ema_vae_fp16.safetensors",
        )

    def test_catalog_defaults_to_7b_int8_and_resolves_new_legacy_config_choices(self):
        self.assertEqual(DEFAULT_SEEDVR2_MODEL, "7B INT8")
        expected_choices = {
            "7B FP16",
            "7B INT8",
            "3B FP16",
            "3B INT8",
            "7B Sharp FP16",
            "7B MXFP8",
            "3B NVFP4",
            "7B NVFP4",
            "7B Sharp NVFP4",
        }
        self.assertEqual(set(SEEDVR2_MODEL_CHOICES), expected_choices)
        for label, key in SEEDVR2_MODEL_CHOICES.items():
            with self.subTest(label=label):
                resolved = seedvr2_upscale_model_names(self.models, label)
                self.assertEqual(resolved["seedvr2_dit"], MODEL_SPECS[key].local_name)

    def test_image_and_video_graphs_share_7b_int8_default(self):
        graphs = (
            build_seedvr2_image_upscale_graph(
                source_images=[("first", "input.png", 2.0)],
                seed=7,
                models=self.models,
                output_token="test",
            ),
            build_seedvr2_upscale_graph(
                source_video="input.mp4",
                seed=7,
                models=self.models,
            ),
        )
        for graph in graphs:
            with self.subTest():
                loader = next(
                    node for node in graph.values() if node["class_type"] == "UNETLoader"
                )
                self.assertEqual(
                    loader["inputs"]["unet_name"],
                    "seedvr2_7b_int8_convrot.safetensors",
                )


if __name__ == "__main__":
    unittest.main()