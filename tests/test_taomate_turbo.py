"""TaoMate selection, shared FL2VA/Ref2VA routing, and lazy provisioning."""

import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import gradio_app as app
import h3_models
from h3_app import model_service
from h3_app.catalog import TAOMATE_3STEP_TURBO, TURBO_SETTINGS
from h3_app.config import RuntimeConfig
from h3_app.settings import (
    GenerationRequest, SamplingSettings, resolve_settings, transition_modes,
)
from h3_ui.persistence import restore_preferences


class TaoMateTurboTests(unittest.TestCase):
    def models(self):
        return app.ModelConfig(
            {}, "speed", "text.safetensors", "video.safetensors", "audio.safetensors",
            taomate_turbo_lora=h3_models.MODEL_SPECS["taomate_turbo_lora"].local_name,
        )

    def test_selection_sets_three_steps_and_survives_restore(self):
        _, values = transition_modes(None, {
            "generation_mode": "Turbo", "turbo_variant": TAOMATE_3STEP_TURBO,
            "steps": 8, "scheduler": "beta",
        }, "turbo_variant")
        self.assertEqual((values["steps"], values["scheduler"]), (3, "simple"))
        restored, _ = restore_preferences(
            {"h3.steps": 3, "h3.turbo_variant": TAOMATE_3STEP_TURBO},
            {
                "h3.steps": SimpleNamespace(value=4, minimum=3, maximum=30),
                "h3.turbo_variant": SimpleNamespace(
                    value=app.DEFAULT_TURBO, choices=list(TURBO_SETTINGS)
                ),
            },
        )
        self.assertEqual(restored["h3.steps"], 3)
        self.assertEqual(restored["h3.turbo_variant"], TAOMATE_3STEP_TURBO)
        for mode in ("Text to video", "First / last frame", "Reference media"):
            request = GenerationRequest(
                mode=mode,
                sampling=SamplingSettings(steps=3, turbo_variant=TAOMATE_3STEP_TURBO),
            )
            self.assertEqual(resolve_settings(request).issues, ())
            self.assertEqual(
                self.models().turbo_lora_for(mode, TAOMATE_3STEP_TURBO),
                self.models().taomate_turbo_lora,
            )
            self.assertTrue(resolve_settings(replace(
                request, sampling=replace(request.sampling, steps=2)
            )).issues)
        self.assertTrue(resolve_settings(GenerationRequest(
            sampling=SamplingSettings(steps=3)
        )).issues)

    def test_existing_config_gets_on_demand_adapter_for_both_bases(self):
        self.assertNotIn("taomate_turbo_lora", h3_models.PRELOAD_MODEL_KEYS)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(root, "http://fixture", root / "ComfyUI", root / "models.json", root / "outputs")
            config = h3_models._build_config("manifest.json")
            self.assertEqual(config.pop("taomate_turbo_lora"), self.models().taomate_turbo_lora)
            config.pop("taomate_turbo_source")
            config["schema_version"] = 16
            runtime.models_config.write_text(json.dumps(config), encoding="utf-8")
            models = model_service.load_model_config(runtime=runtime)
            self.assertEqual(models.taomate_turbo_lora, self.models().taomate_turbo_lora)
            for mode in ("Text to video", "Reference media"):
                with (
                    patch.object(model_service, "stale_model_keys", return_value=["taomate_turbo_lora"]),
                    patch.object(model_service, "sync_models") as sync,
                    patch.object(model_service, "model_file_is_ready", return_value=True),
                    patch.object(model_service, "resolve_hf_token", return_value=None),
                ):
                    self.assertTrue(model_service.ensure_turbo_lora(models, TAOMATE_3STEP_TURBO, mode, runtime=runtime))
                    self.assertEqual(sync.call_args.kwargs["model_keys"], ("taomate_turbo_lora",))
                with (
                    patch.object(model_service, "stale_model_keys", return_value=[]),
                    patch.object(model_service, "sync_models") as sync,
                ):
                    self.assertFalse(model_service.ensure_turbo_lora(models, TAOMATE_3STEP_TURBO, mode, runtime=runtime))
                    sync.assert_not_called()

    def test_both_workflows_use_standard_lora_euler_and_three_step_schedule(self):
        self.assertEqual(TURBO_SETTINGS[TAOMATE_3STEP_TURBO].strength, 0.7)
        for build in (app.build_fl2va_graph, app.build_ref2va_graph):
            with self.subTest(workflow=build.__name__):
                args = {
                    name: app.UI_DEFAULTS.get(name, 0)
                    for name, parameter in inspect.signature(build).parameters.items()
                    if parameter.default is inspect.Parameter.empty
                }
                args.update(
                    prompt="A bird takes flight", width=864, height=480, duration=5,
                    steps=3, scheduler="simple", seed=7, model_name="base.safetensors",
                    models=self.models(), turbo_variant=TAOMATE_3STEP_TURBO,
                    turbo_lora_name=self.models().taomate_turbo_lora,
                    turbo_strength=TURBO_SETTINGS[TAOMATE_3STEP_TURBO].strength,
                    use_sol=False, cache_mode="Off",
                    available_nodes=app.turbo_required_nodes(TAOMATE_3STEP_TURBO),
                )
                if build is app.build_fl2va_graph:
                    args.update(first_image=None, last_image=None)
                else:
                    args.update(reference_images=[], reference_videos=[], reference_audios=[])
                graph = build(**args)
                nodes = list(graph.values())
                loader = next(n for n in nodes if n["class_type"] == "LoraLoaderModelOnly")
                self.assertEqual(loader["inputs"]["lora_name"], self.models().taomate_turbo_lora)
                self.assertEqual(loader["inputs"]["strength_model"], 0.7)
                sampler = next(n for n in nodes if n["class_type"] == "KSamplerSelect")
                self.assertEqual(sampler["inputs"]["sampler_name"], "euler")
                schedule = next(n for n in nodes if n["class_type"] == "BasicScheduler")
                self.assertEqual((schedule["inputs"]["steps"], schedule["inputs"]["scheduler"]), (3, "simple"))
                self.assertTrue(any(n["class_type"] == "BasicGuider" for n in nodes))
                self.assertFalse(any(n["class_type"] in (app.LIGHTX2V_BYPASS_LORA_NODE, app.H3_SIGMA_SHIFT_NODE, "CFGGuider") for n in nodes))
