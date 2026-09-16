"""FastH3 8-Step V2 model inventory and profile routing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h3_models
from h3_app.model_service import load_model_config
from h3_app.model_types import ModelConfig, ModelProfile


class FastH3ModelTests(unittest.TestCase):
    def test_official_comfy_checkpoint_is_a_lazy_profile(self):
        key = "fasth3_8step_v2"
        spec = h3_models.MODEL_SPECS[key]
        self.assertEqual(spec.repo_id, "FastVideo/FastVideo-FastH3-Comfy")
        self.assertEqual(
            spec.filename,
            "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        )
        self.assertEqual(
            h3_models.PROFILE_LABELS["fasth3_8step_v2"], "FastH3 8-Step V2"
        )
        self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)

    def test_profile_label_resolves_to_internal_key(self):
        profile = ModelProfile(
            label="FastH3 8-Step V2",
            fl2va="fastvideo.safetensors",
            ref2va="fastvideo.safetensors",
        )
        models = ModelConfig(
            profiles={"fasth3_8step_v2": profile},
            default_profile="fasth3_8step_v2",
            text_encoder="text.safetensors",
            video_vae="video.safetensors",
            audio_vae="audio.safetensors",
        )
        self.assertEqual(models.profile_key("FastH3 8-Step V2"), "fasth3_8step_v2")

    def test_old_model_config_gains_lazy_fasth3_profile_in_memory(self):
        with TemporaryDirectory() as temp:
            config_path = Path(temp) / "h3_models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "speed": {
                                "label": "Speed",
                                "fl2va": "speed-fl2va.safetensors",
                                "ref2va": "speed-ref2va.safetensors",
                            }
                        },
                        "default_profile": "speed",
                        "text_encoder": "text.safetensors",
                        "video_vae": "video.safetensors",
                        "audio_vae": "audio.safetensors",
                    }
                ),
                encoding="utf-8",
            )
            models = load_model_config(runtime=SimpleNamespace(models_config=config_path))

        profile = models.profile("FastH3 8-Step V2")
        self.assertEqual(
            profile.fl2va,
            "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        )
        self.assertEqual(models.profiles["speed"].fl2va, "speed-fl2va.safetensors")


if __name__ == "__main__":
    unittest.main()
