"""LightX2V Turbo Fusion v1.0 Full contracts across T2VA, I2VA/FL2VA, Ref2VA, presets, and refinement."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h3_models
from h3_app import model_service
from h3_app.catalog import (
    DEFAULT_TURBO,
    FUSED_MODULATION_NODE,
    H3_REFINEMENT,
    H3_SIGMA_SHIFT_NODE,
    LIGHTX2V_BYPASS_LORA_NODE,
    LIGHTX2V_FUSION_4STEP_FULL,
    LIGHTX2V_FUSION_8STEP_FULL,
    PIKA_FLEX,
    REFINEMENT_LORA_SETTINGS,
    TURBO_SETTINGS,
    UI_DEFAULTS,
)
from h3_app.errors import H3Error
from h3_app.generation.h3 import lightx2v_fusion_sampling_profile
from h3_app.policy import (
    is_lightx2v_turbo_lora,
    lightx2v_uses_768p_schedule,
    normalize_turbo_variant,
    turbo_sampler_name,
    turbo_sigma_shifts,
    turbo_steps_for,
    turbo_strength_for,
    turbo_uses_custom_nodes,
)
from h3_app.settings import GenerationRequest, preset_settings, resolve_settings, transition_modes
from h3_app.workflows import face_refine, h3, regenerate
from h3_ui.persistence import restore_preferences
from tests.workflow_cases import H3_DEFAULTS, MODELS

FUSION_CASES = (
    (LIGHTX2V_FUSION_4STEP_FULL, "turbo_fusion_4step_full_lora", 4),
    (LIGHTX2V_FUSION_8STEP_FULL, "turbo_fusion_8step_full_lora", 8),
)


class LightX2VTurboFusionTests(unittest.TestCase):
    def graph_args(self, variant: str, steps: int, **values):
        filename = MODELS.turbo_lora_for("Text to video", variant)
        return {
            **H3_DEFAULTS,
            "turbo_variant": variant,
            "turbo_lora_name": filename,
            "turbo_strength": 1.0,
            "available_nodes": set(H3_DEFAULTS["available_nodes"]) | h3.turbo_required_nodes(variant),
            "width": 832,
            "steps": steps,
            **values,
        }

    def test_presets_and_retired_lite_migration(self):
        self.assertEqual(DEFAULT_TURBO, LIGHTX2V_FUSION_4STEP_FULL)
        self.assertEqual(UI_DEFAULTS["turbo_variant"], LIGHTX2V_FUSION_4STEP_FULL)
        self.assertEqual(UI_DEFAULTS["steps"], 4)
        for name in ("Singularity", "Fast"):
            cfg = preset_settings(name, "Turbo")
            self.assertEqual((cfg.turbo_variant, cfg.steps), (LIGHTX2V_FUSION_4STEP_FULL, 4))
        for name in ("Balanced", "Quality"):
            cfg = preset_settings(name, "Turbo")
            self.assertEqual((cfg.turbo_variant, cfg.steps), (LIGHTX2V_FUSION_8STEP_FULL, 8))
        for retired, replacement in (
            ("LightX2V Turbo Fusion v1.0 / 4-step Lite r32", LIGHTX2V_FUSION_4STEP_FULL),
            ("LightX2V Turbo Fusion v1.0 / 8-step Lite r32", LIGHTX2V_FUSION_8STEP_FULL),
        ):
            self.assertNotIn(retired, TURBO_SETTINGS)
            self.assertNotIn(retired, REFINEMENT_LORA_SETTINGS)
            self.assertEqual(normalize_turbo_variant(retired), replacement)
            restored, _ = restore_preferences(
                {"h3.turbo_variant": retired, "h3.latent_upscale_refine_lora": retired},
                {
                    "h3.turbo_variant": SimpleNamespace(value=DEFAULT_TURBO, choices=list(TURBO_SETTINGS)),
                    "h3.latent_upscale_refine_lora": SimpleNamespace(
                        value="Same as generation",
                        choices=["Same as generation", *REFINEMENT_LORA_SETTINGS],
                    ),
                },
            )
            self.assertEqual(restored["h3.turbo_variant"], replacement)
            self.assertEqual(restored["h3.latent_upscale_refine_lora"], replacement)
        self.assertNotIn("turbo_fusion_4step_lite_lora", h3_models.MODEL_SPECS)
        self.assertNotIn("turbo_fusion_8step_lite_lora", h3_models.MODEL_SPECS)

    def test_policy_settings_and_preferences_for_both_full_variants(self):
        for variant, key, steps in FUSION_CASES:
            with self.subTest(variant=variant):
                filename = h3_models.MODEL_SPECS[key].local_name
                self.assertIn(variant, TURBO_SETTINGS)
                self.assertIn(variant, REFINEMENT_LORA_SETTINGS)
                self.assertEqual(normalize_turbo_variant(variant), variant)
                self.assertEqual(turbo_steps_for(variant), steps)
                self.assertEqual(turbo_strength_for(variant), 1.0)
                self.assertFalse(turbo_uses_custom_nodes(variant))
                self.assertTrue(is_lightx2v_turbo_lora(variant, filename))
                self.assertFalse(lightx2v_uses_768p_schedule(variant, filename))
                self.assertEqual(turbo_sampler_name(variant, filename), "euler")
                self.assertEqual(turbo_sampler_name(variant, None), "res_multistep")
                self.assertEqual(turbo_sigma_shifts(variant, filename), (12.0, 3.0))
                self.assertIsNone(turbo_sigma_shifts(variant, None))

                _, values = transition_modes(
                    None,
                    {
                        "generation_mode": "Turbo",
                        "turbo_variant": variant,
                        "steps": 12,
                        "scheduler": "beta",
                        "latent_upscale": True,
                    },
                    "turbo_variant",
                )
                self.assertEqual((values["steps"], values["scheduler"], values["latent_upscale"]), (steps, "simple", True))
                plan = resolve_settings(GenerationRequest.from_values({
                    "turbo_variant": variant,
                    "steps": steps,
                    "scheduler": "simple",
                    "latent_upscale": True,
                    "latent_upscale_refine_steps": 2,
                }))
                self.assertFalse(plan.issues)
                self.assertNotIn("steps", plan.inactive)
                self.assertNotIn("scheduler", plan.inactive)

                custom_steps = resolve_settings(GenerationRequest.from_values({
                    "turbo_variant": variant,
                    "steps": 6,
                    "scheduler": "beta",
                    "latent_upscale": False,
                }))
                self.assertFalse(custom_steps.issues)
                self.assertEqual((custom_steps.effective.sampling.steps, custom_steps.effective.sampling.scheduler), (6, "beta"))

                restored, _ = restore_preferences(
                    {"h3.turbo_variant": variant, "h3.latent_upscale_refine_lora": variant},
                    {
                        "h3.turbo_variant": SimpleNamespace(value=PIKA_FLEX, choices=list(TURBO_SETTINGS)),
                        "h3.latent_upscale_refine_lora": SimpleNamespace(
                            value="Same as generation",
                            choices=["Same as generation", *REFINEMENT_LORA_SETTINGS],
                        ),
                    },
                )
                self.assertEqual(restored["h3.turbo_variant"], variant)
                self.assertEqual(restored["h3.latent_upscale_refine_lora"], variant)

    def test_both_full_variants_work_for_t2va_i2va_and_ref2va_graphs(self):
        task_cases = (
            ("Text to video", h3.build_fl2va_graph, {"first_image": None, "last_image": None}),
            ("First / last frame", h3.build_fl2va_graph, {"first_image": "first.png", "last_image": "last.png"}),
            (
                "Reference media",
                h3.build_ref2va_graph,
                {
                    "reference_images": ["ref.png"],
                    "reference_videos": ["ref.mp4"],
                    "reference_audios": ["ref.wav"],
                    "ref_image_size": "match",
                },
            ),
        )
        for variant, key, steps in FUSION_CASES:
            expected_file = h3_models.MODEL_SPECS[key].local_name
            for mode, build, media in task_cases:
                with self.subTest(variant=variant, mode=mode):
                    filename = MODELS.turbo_lora_for(mode, variant)
                    self.assertEqual(filename, expected_file)
                    args = self.graph_args(variant, steps, turbo_lora_name=filename)
                    if build is h3.build_ref2va_graph:
                        for name in ("first_image", "last_image", "voice_reference_audios", *h3.FL2VA_ONLY_OPTIONS):
                            args.pop(name, None)
                    graph = build(**{**args, **media})
                    types = [node["class_type"] for node in graph.values()]
                    self.assertIn(LIGHTX2V_BYPASS_LORA_NODE, types)
                    self.assertIn(FUSED_MODULATION_NODE, types)
                    self.assertIn(H3_SIGMA_SHIFT_NODE, types)
                    self.assertIn("BasicScheduler", types)
                    self.assertNotIn("LoraLoaderModelOnly", types)
                    self.assertNotIn("MiniMaxH3TurboLoRA", types)
                    self.assertNotIn("MiniMaxH3TurboSampler", types)
                    bypass_nodes = [
                        node["inputs"]
                        for node in graph.values()
                        if node["class_type"] == LIGHTX2V_BYPASS_LORA_NODE
                    ]
                    self.assertEqual(
                        bypass_nodes,
                        [{"model": bypass_nodes[0]["model"], "lora_name": expected_file, "strength": 1.0}],
                    )
                    shifts = [
                        (node["inputs"]["shift_video"], node["inputs"]["shift_audio"])
                        for node in graph.values()
                        if node["class_type"] == H3_SIGMA_SHIFT_NODE
                    ]
                    self.assertEqual(shifts, [(12.0, 3.0)])
                    sample = next(
                        node["inputs"]
                        for node in graph.values()
                        if node["class_type"] == "SamplerCustomAdvanced"
                    )
                    self.assertEqual(graph[sample["sampler"][0]]["inputs"], {"sampler_name": "euler"})
                    sched = graph[sample["sigmas"][0]]
                    self.assertEqual(sched["class_type"], "BasicScheduler")
                    self.assertEqual(sched["inputs"]["steps"], steps)

    def test_missing_required_nodes_fail_before_sampling(self):
        variant, _, steps = FUSION_CASES[0]
        for missing in (LIGHTX2V_BYPASS_LORA_NODE, FUSED_MODULATION_NODE, H3_SIGMA_SHIFT_NODE):
            with self.subTest(missing=missing):
                args = self.graph_args(variant, steps)
                args["available_nodes"].remove(missing)
                with self.assertRaisesRegex(H3Error, missing):
                    h3.build_fl2va_graph(**args)

    def test_lazy_provisioning_and_old_model_config_fallback(self):
        runtime = Mock()
        for variant, key, _ in FUSION_CASES:
            spec = h3_models.MODEL_SPECS[key]
            self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
            self.assertEqual(spec.repo_id, h3_models.TURBO_FUSION_REPO)
            self.assertEqual(spec.revision, h3_models.TURBO_FUSION_REVISION)
            self.assertIsNotNone(spec.expected_sha256)
            self.assertEqual(h3_models.build_model_config("manifest.json")[key], spec.local_name)
            for mode in ("Text to video", "First / last frame", "Reference media"):
                with self.subTest(variant=variant, mode=mode), patch.object(
                    model_service, "ensure_models", return_value=True
                ) as ensure:
                    self.assertTrue(model_service.ensure_turbo_lora(MODELS, variant, mode, runtime=runtime))
                    self.assertEqual(ensure.call_args.args[0], (key,))
                    self.assertEqual(MODELS.turbo_lora_for(mode, variant), spec.local_name)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "h3_models.json"
            config = h3_models.build_model_config("manifest.json")
            for _, key, _ in FUSION_CASES:
                config.pop(key)
            path.write_text(json.dumps(config), encoding="utf-8")
            models = model_service.load_model_config(runtime=SimpleNamespace(models_config=path))
            for _, key, _ in FUSION_CASES:
                self.assertEqual(getattr(models, key), h3_models.MODEL_SPECS[key].local_name)

    def test_native_upscale_refinement_and_face_refine_with_fusion(self):
        # 1. Two-stage native latent upscale: Pika base refined with Fusion 8-step Full.
        fusion_variant = LIGHTX2V_FUSION_8STEP_FULL
        fusion_file = MODELS.turbo_lora_for("Text to video", fusion_variant)
        graph = h3.build_fl2va_graph(**{
            **H3_DEFAULTS,
            "turbo_variant": PIKA_FLEX,
            "turbo_lora_name": MODELS.pika_flex_fl2va_lora,
            "width": 1664,
            "height": 960,
            "steps": 6,
            "latent_upscale_model_name": "upscale.pth",
            "latent_upscale_refine_steps": 2,
            "refinement_lora_name": fusion_file,
            "refinement_variant": fusion_variant,
            "available_nodes": (
                set(H3_DEFAULTS["available_nodes"])
                | h3.turbo_required_nodes(PIKA_FLEX)
                | h3.turbo_required_nodes(fusion_variant)
            ),
        })
        samples = [n["inputs"] for n in graph.values() if n["class_type"] == "SamplerCustomAdvanced"]
        self.assertEqual(len(samples), 2)
        draft, refined = samples
        self.assertEqual(graph[draft["sampler"][0]]["inputs"], {"sampler_name": "lcm"})
        self.assertEqual(graph[refined["sampler"][0]]["inputs"], {"sampler_name": "euler"})
        split = graph[refined["sigmas"][0]]["inputs"]
        self.assertEqual(split["step"], 6)
        self.assertEqual(graph[split["sigmas"][0]]["inputs"]["steps"], 8)

        # 2. Existing-video refinement and face refine with Fusion 4-step Full.
        refinement_cfg = {
            "mode": "Reference media",
            "model_profile": "Speed",
            "generation_mode": "Turbo",
            "turbo_variant": LIGHTX2V_FUSION_4STEP_FULL,
            "steps": 4,
            "scheduler": "simple",
            "latent_upscale_refine_steps": 1,
        }
        regen_graph = regenerate.build_h3_regenerate_graph(
            option=H3_REFINEMENT,
            source_video="clip.mp4",
            seed=7,
            models=MODELS,
            target_width=1280,
            target_height=768,
            refinement=refinement_cfg,
        )
        self.assertTrue(any(n["class_type"] == LIGHTX2V_BYPASS_LORA_NODE for n in regen_graph.values()))
        face_graph = face_refine.build_h3_face_refine_graph(
            source_video="clip.mp4",
            seed=7,
            models=MODELS,
            prompt="close-up",
            fps=24.0,
            strength=0.4,
            refinement=refinement_cfg,
        )
        self.assertTrue(any(n["class_type"] == LIGHTX2V_BYPASS_LORA_NODE for n in face_graph.values()))

    def test_provenance_records_pinned_identity_only_for_pinned_file(self):
        for variant, key, steps in FUSION_CASES:
            spec = h3_models.MODEL_SPECS[key]
            profile = lightx2v_fusion_sampling_profile(variant, spec.local_name, steps)
            self.assertEqual(
                (profile["sampler"], profile["video_shift"], profile["audio_shift"], profile["steps"]),
                ("euler", 12.0, 3.0, steps),
            )
            self.assertEqual(
                (profile["repo_id"], profile["revision"], profile["sha256"]),
                (spec.repo_id, spec.revision, spec.expected_sha256),
            )
            custom = lightx2v_fusion_sampling_profile(variant, "custom_fusion.safetensors", steps)
            self.assertIsNone(custom["repo_id"])
            self.assertIsNone(custom["revision"])
            self.assertIsNone(custom["sha256"])


if __name__ == "__main__":
    unittest.main()
