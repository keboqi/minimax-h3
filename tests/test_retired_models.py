"""Retired models cannot return through old configs or browser preferences."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h3_models
from h3_app.catalog import (
    DEFAULT_TURBO, MODEL_PROFILE_CHOICES, REFINEMENT_LORA_CHOICES, TURBO_SETTINGS,
)
from h3_app.errors import H3Error
from h3_app.generation.preparation import _validate_sampling_steps
from h3_app.model_service import load_model_config
from h3_app.policy import normalize_turbo_variant
from h3_ui.persistence import restore_preferences


class RetiredModelsTests(unittest.TestCase):
    def test_retired_weights_are_absent_from_inventory_and_generated_config(self):
        config = h3_models._build_config("manifest.json")
        self.assertNotIn("fasth3_8step_v2", config["profiles"])
        for key in ("fasth3_8step_v2", "taomate_turbo_lora", "pdmd_2step_lora"):
            self.assertNotIn(key, h3_models.MODEL_SPECS)
            self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
            self.assertNotIn(key, config)
        self.assertNotIn("taomate_turbo_source", config)

    def test_old_config_drops_retired_profiles_and_repairs_default(self):
        config = h3_models._build_config("manifest.json")
        speed_filename = config["profiles"]["speed"]["fl2va"] = "custom-speed.safetensors"
        retired = {
            "label": "FastH3 8-Step V2",
            "fl2va": "retired.safetensors", "ref2va": "retired.safetensors",
        }
        config["profiles"]["fasth3_8step_v2"] = retired
        config["profiles"]["custom-retired"] = retired
        config["default_profile"] = "fasth3_8step_v2"
        config["taomate_turbo_lora"] = "retired-taomate.safetensors"
        config["pdmd_2step_lora"] = "retired-pdmd.safetensors"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            models = load_model_config(runtime=SimpleNamespace(models_config=path))
        self.assertNotIn("fasth3_8step_v2", models.profiles)
        self.assertNotIn("custom-retired", models.profiles)
        self.assertEqual(models.default_profile, "singularity")
        self.assertEqual(models.profiles["speed"].fl2va, speed_filename)
        self.assertFalse(hasattr(models, "taomate_turbo_lora"))
        self.assertFalse(hasattr(models, "pdmd_2step_lora"))

    def test_saved_retired_choices_and_mode_memory_restore_supported_defaults(self):
        components = {
            "h3.model_profile": SimpleNamespace(value="Singularity", choices=MODEL_PROFILE_CHOICES),
            "h3.turbo_variant": SimpleNamespace(value=DEFAULT_TURBO, choices=list(TURBO_SETTINGS)),
            "h3.latent_upscale_refine_lora": SimpleNamespace(value="Same as generation", choices=REFINEMENT_LORA_CHOICES),
            "h3.steps": SimpleNamespace(value=4, minimum=4, maximum=30),
        }
        for variant, steps in (("TaoMate-H3 / 3-step", 3), ("PDMD / 2-step", 2)):
            with self.subTest(variant=variant):
                saved = {
                    "values": {
                        "h3.model_profile": "FastH3 8-Step V2",
                        "h3.turbo_variant": variant,
                        "h3.latent_upscale_refine_lora": variant,
                        "h3.steps": steps,
                    },
                    "mode_memory": {"modes": {"Turbo": {"turbo_variant": variant, "steps": steps}}},
                }
                restored, memory = restore_preferences(saved, components)
                self.assertEqual(restored["h3.model_profile"], "Singularity")
                self.assertEqual(restored["h3.turbo_variant"], DEFAULT_TURBO)
                self.assertEqual(restored["h3.latent_upscale_refine_lora"], "Same as generation")
                self.assertEqual(restored["h3.steps"], 4)
                self.assertEqual(memory["modes"]["Turbo"]["turbo_variant"], DEFAULT_TURBO)
                self.assertEqual(memory["modes"]["Turbo"]["steps"], 4)
                self.assertEqual(normalize_turbo_variant(variant), DEFAULT_TURBO)
                with self.assertRaisesRegex(H3Error, "at least 4 steps"):
                    _validate_sampling_steps("speed", True, variant, steps)

    def test_normal_generation_has_no_distilled_eight_step_exception(self):
        for profile in h3_models.PROFILE_MODEL_KEYS:
            with self.subTest(profile=profile), self.assertRaisesRegex(H3Error, "at least 10 steps"):
                _validate_sampling_steps(profile, False, DEFAULT_TURBO, 8)
