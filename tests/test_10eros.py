"""10Eros v1.0 (Built-in Turbo, no-LoRA), SLA compatibility, and preset base-model contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import h3_models
from h3_app import model_service
from h3_app.catalog import (
    BUILTIN_TURBO,
    BUILTIN_TURBO_SETTINGS,
    CORE_LORA_LOADER_NODE,
    DEFAULT_PRESET,
    DEFAULT_TURBO,
    FUSED_MODULATION_NODE,
    H3_REFINEMENT,
    H3_REGENERATE_16,
    H3_SIGMA_SHIFT_NODE,
    LARRY_TURBO_LORA_NODE,
    LARRY_TURBO_SAMPLER_NODE,
    LIGHTX2V_BYPASS_LORA_NODE,
    LIGHTX2V_FUSION_4STEP_FULL,
    LIGHTX2V_FUSION_8STEP_FULL,
    MODEL_PROFILE_CHOICES,
    REFINEMENT_LORA_CHOICES,
    REFINEMENT_LORA_SETTINGS,
    SLA_ATTENTION_NODE,
    TURBO_SETTINGS,
    UI_DEFAULTS,
)
from h3_app.errors import H3Error
from h3_app.generation.h3 import builtin_turbo_sampling_profile, generate as generate_h3
from h3_app.generation.ph3_image import generate_ph3_image
from h3_app.generation.preparation import _validate_sampling_steps
from h3_app.generation.requests import H3Request
from h3_app.jobs import JOBS
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.policy import (
    is_builtin_turbo_variant,
    normalize_turbo_variant,
    turbo_sampler_name,
    turbo_sigma_shifts,
    turbo_steps_for,
)
from h3_app.refinement import resolve_face_refinement, resolve_refinement
from h3_app.remake import RemakeGraph, SWAP
from h3_app.settings import (
    GenerationRequest,
    PRESET_MODEL_PROFILES,
    preset_settings,
    resolve_settings,
    transition_modes,
)
from h3_app.storyboard.model import request_defaults
from h3_app.workflows import face_refine, h3, regenerate
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.engines.remake import next_run_details
from h3_ui.h3.handlers import H3Handlers, H3Services
from h3_ui.h3.page import H3PageServices, install_h3_page
from h3_ui.settings_presentation import render_settings
from tests.workflow_cases import H3_DEFAULTS, MODELS


EROS_MODELS = ModelConfig(
    **{
        **MODELS.__dict__,
        "profiles": {
            **MODELS.profiles,
            "10eros": ModelProfile(
                label="10Eros v1.0",
                fl2va=h3_models.MODEL_SPECS["eros_max_v1_fl2va"].local_name,
                ref2va=h3_models.MODEL_SPECS["eros_max_v1_ref2va"].local_name,
                fl2va_source=h3_models.MODEL_SPECS["eros_max_v1_fl2va"].source,
                ref2va_source=h3_models.MODEL_SPECS["eros_max_v1_ref2va"].source,
            ),
        },
    }
)


class TenErosTests(unittest.TestCase):
    def graph_args(self, **values):
        return {
            **H3_DEFAULTS,
            "model_name": h3_models.MODEL_SPECS["eros_max_v1_fl2va"].local_name,
            "models": EROS_MODELS,
            "turbo_variant": BUILTIN_TURBO,
            "turbo_lora_name": None,
            "turbo_strength": 1.0,
            "use_sla": True,
            "sla_preset": "Balanced",
            "steps": 8,
            "scheduler": "simple",
            "available_nodes": set(H3_DEFAULTS["available_nodes"]) | h3.turbo_required_nodes(BUILTIN_TURBO),
            **values,
        }

    def test_10eros_model_spec_is_lazy_pinned_and_shared_across_modes(self):
        fl2va = h3_models.MODEL_SPECS["eros_max_v1_fl2va"]
        ref2va = h3_models.MODEL_SPECS["eros_max_v1_ref2va"]
        self.assertEqual(fl2va, ref2va)
        self.assertEqual(fl2va.repo_id, "TenStrip/10Eros-Max")
        self.assertEqual(fl2va.revision, "875036a9e9c363483095c82ee5da18fee3cef08b")
        self.assertEqual(fl2va.local_name, "10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")
        self.assertEqual(
            fl2va.expected_sha256,
            "776194990379ecaf81f8c56925164894f78f3ab27d905c23d85c1164f20ec5bc",
        )
        self.assertIn("10Eros v1.0", MODEL_PROFILE_CHOICES)
        self.assertNotIn("eros_max_v1_fl2va", h3_models.PRELOAD_MODEL_KEYS)
        self.assertNotIn("eros_max_v1_ref2va", h3_models.PRELOAD_MODEL_KEYS)
        self.assertTrue(h3_models.is_baked_turbo_profile("10eros"))
        self.assertTrue(h3_models.is_baked_turbo_profile("10Eros v1.0"))
        self.assertFalse(h3_models.is_baked_turbo_profile("Quality"))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "h3_models.json"
            config = h3_models.build_model_config("manifest.json")
            config["profiles"].pop("10eros")
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = model_service.load_model_config(runtime=SimpleNamespace(models_config=path))
            self.assertEqual(loaded.profile("10Eros v1.0").fl2va, fl2va.local_name)
            self.assertEqual(loaded.profile("10Eros v1.0").ref2va, ref2va.local_name)

    def test_presets_select_requested_base_models_and_default_is_balanced(self):
        self.assertEqual(DEFAULT_PRESET, "Balanced")
        self.assertEqual(GenerationRequest().preset, "Balanced")
        self.assertEqual(GenerationRequest().model_profile, "Quality")
        self.assertEqual(UI_DEFAULTS["model_profile"], "Quality")
        self.assertEqual(UI_DEFAULTS["text_encoder"], "INT8 ConvRot")
        self.assertEqual(UI_DEFAULTS["turbo_variant"], LIGHTX2V_FUSION_8STEP_FULL)
        self.assertEqual(DEFAULT_TURBO, LIGHTX2V_FUSION_8STEP_FULL)
        self.assertEqual(UI_DEFAULTS["steps"], 8)
        self.assertEqual(UI_DEFAULTS["sla_preset"], "Balanced")
        self.assertTrue(UI_DEFAULTS["use_int8_vae"])
        self.assertFalse(UI_DEFAULTS["use_sol_vae"])
        self.assertEqual(UI_DEFAULTS["video_encoder"], "Original")

        expected_profiles = {
            "Balanced": "Quality",
            "Quality": "Original",
            "Fast": "Speed",
            "Singularity": "Singularity",
            "10Eros": "10Eros v1.0",
        }
        self.assertEqual(PRESET_MODEL_PROFILES, expected_profiles)
        for preset, expected_model in expected_profiles.items():
            for action in ("preset", "restore"):
                with self.subTest(preset=preset, action=action):
                    _, values = transition_modes(
                        None,
                        {
                            "preset": preset,
                            "generation_mode": "Normal",
                            "model_profile": "Singularity",
                            "latent_upscale_refine_lora": LIGHTX2V_FUSION_4STEP_FULL,
                            "latent_upscale_method": H3_REGENERATE_16,
                        },
                        action,
                    )
                    self.assertEqual(values["model_profile"], expected_model)
                    if preset == "10Eros":
                        self.assertEqual(values["generation_mode"], "Turbo")
                        self.assertEqual(values["turbo_variant"], BUILTIN_TURBO)
                        self.assertEqual(values["steps"], 8)
                        self.assertEqual(values["scheduler"], "simple")
                        self.assertEqual(values["sla_preset"], "Balanced")
                        self.assertTrue(values["use_sol_vae"])
                        self.assertEqual(values["video_encoder"], "Original")
                        self.assertEqual(values["latent_upscale_refine_lora"], "Same as generation")
                        self.assertEqual(values["latent_upscale_method"], "Full-frame refinement")

        # Switching base model manually to and away from 10Eros v1.0 transitions Turbo cleanly.
        memory, to_eros = transition_modes(
            None,
            {
                "preset": "Balanced",
                "model_profile": "10Eros v1.0",
                "generation_mode": "Normal",
                "turbo_variant": LIGHTX2V_FUSION_8STEP_FULL,
                "steps": 18,
                "scheduler": "beta",
                "latent_upscale_refine_lora": LIGHTX2V_FUSION_4STEP_FULL,
                "latent_upscale_method": H3_REGENERATE_16,
            },
            "model_profile",
        )
        self.assertEqual(to_eros["generation_mode"], "Turbo")
        self.assertEqual(to_eros["turbo_variant"], BUILTIN_TURBO)
        self.assertEqual(to_eros["steps"], 8)
        self.assertEqual(to_eros["scheduler"], "simple")
        self.assertEqual(to_eros["latent_upscale_refine_lora"], "Same as generation")
        self.assertEqual(to_eros["latent_upscale_method"], "Full-frame refinement")

        _, back_to_quality = transition_modes(
            memory,
            {**to_eros, "model_profile": "Quality"},
            "model_profile",
        )
        self.assertEqual(back_to_quality["turbo_variant"], LIGHTX2V_FUSION_8STEP_FULL)
        self.assertEqual(back_to_quality["steps"], 8)

    def test_mode_switch_restores_preset_and_manual_base_profile(self):
        for profile in ("Quality", "Speed"):
            with self.subTest(profile=profile):
                turbo = dict(UI_DEFAULTS, preset="Balanced", model_profile=profile)
                memory, values = transition_modes(None, turbo, "edit")
                memory, values = transition_modes(memory, dict(values, generation_mode="Normal"), "generation_mode")
                memory, values = transition_modes(memory, dict(values, preset="Quality"), "preset")
                self.assertEqual(values["model_profile"], "Original")
                _, values = transition_modes(memory, dict(values, generation_mode="Turbo"), "generation_mode")
                self.assertEqual((values["preset"], values["model_profile"], values["steps"]),
                                 ("Balanced", profile, 8))

    def test_first_normal_mode_after_switching_away_from_10eros(self):
        for profile, preset in (("Quality", "Balanced"), ("Speed", "Fast"), ("Original", "Quality")):
            with self.subTest(profile=profile):
                memory, values = transition_modes(None, dict(UI_DEFAULTS, preset="10Eros"), "preset")
                memory, values = transition_modes(memory, dict(values, model_profile=profile), "model_profile")
                _, values = transition_modes(memory, dict(values, generation_mode="Normal"), "generation_mode")
                self.assertEqual((values["preset"], values["model_profile"], values["steps"]),
                                 (preset, profile, preset_settings(preset, "Normal").steps))
                self.assertEqual(resolve_settings(GenerationRequest.from_values(values)).issues, ())

    def test_saved_lite_refinement_resolves_to_matching_full_adapter(self):
        for steps, variant in ((4, LIGHTX2V_FUSION_4STEP_FULL), (8, LIGHTX2V_FUSION_8STEP_FULL)):
            for mode in ("Turbo", "Normal"):
                with self.subTest(steps=steps, mode=mode):
                    values = dict(UI_DEFAULTS, generation_mode=mode, steps=18,
                                  latent_upscale_refine_lora=f"LightX2V Turbo Fusion v1.0 / {steps}-step Lite r32")
                    config = resolve_refinement(values, EROS_MODELS)
                    self.assertEqual(config["variant"], variant)
                    self.assertEqual(config["lora"], EROS_MODELS.turbo_lora_for(values["mode"], variant))

    def test_builtin_turbo_policy_steps_locks_and_guards(self):
        self.assertIn(BUILTIN_TURBO, TURBO_SETTINGS)
        self.assertIn(BUILTIN_TURBO, BUILTIN_TURBO_SETTINGS)
        self.assertNotIn(BUILTIN_TURBO, REFINEMENT_LORA_SETTINGS)
        self.assertNotIn(BUILTIN_TURBO, REFINEMENT_LORA_CHOICES)
        self.assertTrue(is_builtin_turbo_variant(BUILTIN_TURBO))
        self.assertEqual(normalize_turbo_variant(BUILTIN_TURBO), BUILTIN_TURBO)
        self.assertEqual(turbo_steps_for(BUILTIN_TURBO), 8)
        self.assertEqual(turbo_sampler_name(BUILTIN_TURBO, None), "euler")
        self.assertEqual(turbo_sigma_shifts(BUILTIN_TURBO, None), (12.0, 3.0))
        self.assertIsNone(EROS_MODELS.turbo_lora_for("Text to video", BUILTIN_TURBO))
        self.assertIsNone(EROS_MODELS.turbo_lora_for("Reference media", BUILTIN_TURBO))
        with patch.object(model_service, "ensure_models") as ensure:
            self.assertFalse(model_service.ensure_turbo_lora(EROS_MODELS, BUILTIN_TURBO, "Text to video", runtime=Mock()))
            ensure.assert_not_called()

        for steps in (6, 7, 8, 12):
            with self.subTest(steps=steps):
                plan = resolve_settings(GenerationRequest.from_values({
                    "preset": "10Eros",
                    "model_profile": "10Eros v1.0",
                    "generation_mode": "Normal",
                    "turbo_variant": LIGHTX2V_FUSION_4STEP_FULL,
                    "steps": steps,
                    "scheduler": "simple",
                    "sla_preset": "Balanced",
                    "latent_upscale": True,
                    "latent_upscale_refine_steps": 2,
                    "latent_upscale_refine_lora": LIGHTX2V_FUSION_4STEP_FULL,
                    "latent_upscale_method": H3_REGENERATE_16,
                }))
                self.assertEqual(plan.issues, ())
                self.assertEqual(plan.effective.generation_mode, "Turbo")
                self.assertEqual(plan.effective.sampling.turbo_variant, BUILTIN_TURBO)
                self.assertEqual(plan.effective.sampling.steps, steps)
                self.assertEqual(plan.effective.finishing.latent_upscale_refine_lora, "Same as generation")
                self.assertEqual(plan.effective.finishing.latent_upscale_method, "Full-frame refinement")
                self.assertTrue(
                    {"generation_mode", "turbo_variant", "latent_upscale_refine_lora", "latent_upscale_method"}
                    <= plan.inactive
                )
                self.assertNotIn("steps", plan.inactive)
                rendered = render_settings(plan)
                self.assertIn("Built-in Turbo", rendered)
                self.assertIn("shifts 12/3", rendered)
                self.assertIn("no LoRA", rendered)

        too_few = resolve_settings(GenerationRequest.from_values({
            "preset": "10Eros",
            "model_profile": "10Eros v1.0",
            "generation_mode": "Turbo",
            "turbo_variant": BUILTIN_TURBO,
            "steps": 5,
            "latent_upscale": False,
        }))
        self.assertTrue(any("at least 6" in issue for issue in too_few.issues))
        with self.assertRaisesRegex(H3Error, "at least 6"):
            _validate_sampling_steps("10eros", True, BUILTIN_TURBO, 5)

        wrong_base = resolve_settings(GenerationRequest.from_values({
            "model_profile": "Quality",
            "generation_mode": "Turbo",
            "turbo_variant": BUILTIN_TURBO,
            "steps": 8,
            "latent_upscale": False,
        }))
        self.assertTrue(any("fused Turbo base model" in issue for issue in wrong_base.issues))

    def test_builtin_turbo_graphs_for_t2va_fl2va_ref2va_sla_and_latent_upscale(self):
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
        for mode, build, media in task_cases:
            for steps in (8, 6):
                with self.subTest(mode=mode, steps=steps):
                    args = self.graph_args(steps=steps)
                    if build is h3.build_ref2va_graph:
                        for name in ("first_image", "last_image", "voice_reference_audios", *h3.FL2VA_ONLY_OPTIONS):
                            args.pop(name, None)
                    graph = build(**{**args, **media})
                    types = {node["class_type"] for node in graph.values()}
                    self.assertNotIn(CORE_LORA_LOADER_NODE, types)
                    self.assertNotIn(LIGHTX2V_BYPASS_LORA_NODE, types)
                    self.assertNotIn(LARRY_TURBO_LORA_NODE, types)
                    self.assertNotIn(LARRY_TURBO_SAMPLER_NODE, types)
                    self.assertNotIn(FUSED_MODULATION_NODE, types)
                    self.assertNotIn("MiniMaxH3ChunkFeedForward", types)
                    self.assertIn(SLA_ATTENTION_NODE, types)
                    self.assertIn(H3_SIGMA_SHIFT_NODE, types)
                    shifts = [
                        (node["inputs"]["shift_video"], node["inputs"]["shift_audio"])
                        for node in graph.values()
                        if node["class_type"] == H3_SIGMA_SHIFT_NODE
                    ]
                    self.assertEqual(shifts, [(12.0, 3.0)])
                    sla_nodes = [node["inputs"] for node in graph.values() if node["class_type"] == SLA_ATTENTION_NODE]
                    self.assertEqual(len(sla_nodes), 1)
                    self.assertEqual(sla_nodes[0]["sparsity_ratio"], 0.85)
                    self.assertEqual(sla_nodes[0]["block_size"], "64")
                    self.assertTrue(sla_nodes[0]["protect_audio"])
                    sample = next(
                        node["inputs"]
                        for node in graph.values()
                        if node["class_type"] == "SamplerCustomAdvanced"
                    )
                    self.assertEqual(graph[sample["sampler"][0]]["inputs"], {"sampler_name": "euler"})
                    sched = graph[sample["sigmas"][0]]
                    self.assertEqual(sched["class_type"], "BasicScheduler")
                    self.assertEqual(sched["inputs"]["steps"], steps)
                    self.assertEqual(sched["inputs"]["scheduler"], "simple")

        # 2-stage native latent upscale with SLA and built-in Turbo (no LoRA loader).
        upscaled = h3.build_fl2va_graph(**self.graph_args(
            width=1664,
            height=960,
            steps=8,
            latent_upscale_model_name="upscale.pth",
            latent_upscale_refine_steps=2,
        ))
        types = {node["class_type"] for node in upscaled.values()}
        self.assertNotIn(CORE_LORA_LOADER_NODE, types)
        self.assertNotIn(LIGHTX2V_BYPASS_LORA_NODE, types)
        samples = [n["inputs"] for n in upscaled.values() if n["class_type"] == "SamplerCustomAdvanced"]
        self.assertEqual(len(samples), 2)
        draft, refined = samples
        self.assertEqual(upscaled[draft["sampler"][0]]["inputs"], {"sampler_name": "euler"})
        self.assertEqual(upscaled[refined["sampler"][0]]["inputs"], {"sampler_name": "euler"})
        self.assertEqual(upscaled[refined["sigmas"][0]]["inputs"]["step"], 6)
        sla_nodes = [n["inputs"] for n in upscaled.values() if n["class_type"] == SLA_ATTENTION_NODE]
        self.assertEqual([n["dense_steps"] for n in sla_nodes], ["0", ""])

        missing_shift = self.graph_args()
        missing_shift["available_nodes"].remove(H3_SIGMA_SHIFT_NODE)
        with self.assertRaisesRegex(H3Error, H3_SIGMA_SHIFT_NODE):
            h3.build_fl2va_graph(**missing_shift)

    def test_existing_video_refinement_face_refine_remake_and_ph3_guards(self):
        refinement_cfg = {
            "mode": "Text to video",
            "model_profile": "10Eros v1.0",
            "generation_mode": "Turbo",
            "turbo_variant": BUILTIN_TURBO,
            "steps": 8,
            "scheduler": "simple",
            "attention_mode": "SLA",
            "sla_preset": "Balanced",
            "latent_upscale_refine_steps": 2,
        }
        face_cfg = resolve_face_refinement(refinement_cfg, EROS_MODELS)
        self.assertEqual(face_cfg["variant"], BUILTIN_TURBO)
        self.assertIsNone(face_cfg["lora"])
        video_cfg = resolve_refinement(refinement_cfg, EROS_MODELS)
        self.assertEqual(video_cfg["variant"], BUILTIN_TURBO)
        self.assertIsNone(video_cfg["lora"])

        regen_graph = regenerate.build_h3_regenerate_graph(
            option=H3_REFINEMENT,
            source_video="clip.mp4",
            seed=7,
            models=EROS_MODELS,
            target_width=1280,
            target_height=768,
            refinement=refinement_cfg,
        )
        self.assertFalse(any(n["class_type"] in {CORE_LORA_LOADER_NODE, LIGHTX2V_BYPASS_LORA_NODE} for n in regen_graph.values()))
        self.assertTrue(any(n["class_type"] == H3_SIGMA_SHIFT_NODE for n in regen_graph.values()))
        self.assertTrue(any(n["class_type"] == SLA_ATTENTION_NODE for n in regen_graph.values()))

        face_graph = face_refine.build_h3_face_refine_graph(
            source_video="clip.mp4",
            seed=7,
            models=EROS_MODELS,
            prompt="close-up",
            fps=24.0,
            strength=0.4,
            refinement=refinement_cfg,
        )
        self.assertFalse(any(n["class_type"] in {CORE_LORA_LOADER_NODE, LIGHTX2V_BYPASS_LORA_NODE} for n in face_graph.values()))
        self.assertTrue(any(n["class_type"] == H3_SIGMA_SHIFT_NODE for n in face_graph.values()))

        # Viggle Remake rejects 10Eros v1.0 before touching ComfyUI.
        request = H3Request.from_values({**request_defaults(), "model_profile": "10Eros v1.0", "prompt": "test"})
        services = SimpleNamespace(models=SimpleNamespace(unload_prompt_rewriter=Mock()))
        remake_updates = list(generate_h3(
            request,
            services,
            SimpleNamespace(),
            run_context={"preset": "10Eros"},
            remake=RemakeGraph(SWAP, "src.mp4", ("pic.png",), (0,), 90),
        ))
        self.assertIn("Viggle Remake requires a non-turbo base model", remake_updates[-1].status)
        services.models.unload_prompt_rewriter.assert_not_called()

        details = next_run_details({"model_profile": "10Eros v1.0"}, SWAP, None, 0, 4.6, 480, 1, False)
        self.assertTrue(any("Unsupported base model: 10Eros v1.0" in note for note in details["notes"]))

        # P-H3-Image rejects 10Eros v1.0 cleanly.
        ph3_request = H3Request.from_values({
            **request_defaults(),
            "model_profile": "10Eros v1.0",
            "result_format": "Image",
            "prompt": "a portrait",
        })
        ph3_services = SimpleNamespace(
            policy=SimpleNamespace(resolve_request_settings=lambda values: resolve_settings(GenerationRequest.from_values(values))),
            models=SimpleNamespace(load_model_config=lambda: EROS_MODELS),
        )
        ph3_updates = list(generate_ph3_image(ph3_request, ph3_services, SimpleNamespace()))
        self.assertIn("P-H3-Image requires a non-turbo FL2VA base model", ph3_updates[-1].status)

        # Provenance pins upstream identity only for the pinned 10Eros v1.0 filename.
        spec = h3_models.MODEL_SPECS["eros_max_v1_fl2va"]
        profile = builtin_turbo_sampling_profile(spec.local_name, 7, "simple")
        self.assertEqual(
            (profile["sampler"], profile["video_shift"], profile["audio_shift"], profile["steps"], profile["adapter"]),
            ("euler", 12.0, 3.0, 7, None),
        )
        self.assertEqual(
            (profile["repo_id"], profile["revision"], profile["sha256"]),
            (spec.repo_id, spec.revision, spec.expected_sha256),
        )
        custom = builtin_turbo_sampling_profile("custom_10eros.safetensors", 8, "simple")
        self.assertIsNone(custom["repo_id"])
        self.assertIsNone(custom["revision"])
        self.assertIsNone(custom["sha256"])

    def test_h3_page_api_10eros_preset_and_preset_base_models(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_store = JOBS.store
            JOBS.store = WorkspaceStore(root / "state")
            try:
                runtime = SimpleNamespace(memory_profile="unknown", output_dir=root / "out", outputs_dir=root / "out")
                handlers = H3Handlers(H3Services(runtime_config=lambda: runtime, generate=lambda **_: iter(()), download_url=None))
                app = FastAPI()
                install_owner_cookie(app, JOBS.store)
                install_h3_page(app, H3PageServices(
                    root=lambda: root / "pages",
                    resolve=handlers.resolve_request_settings,
                    describe=handlers.describe_settings,
                    enhance=lambda *a, **k: ("", ""),
                ))
                client = TestClient(app)
                boot = client.get("/h3/api/boot").json()
                self.assertEqual(boot["choices"]["preset"], ["Balanced", "Quality", "Fast", "Singularity", "10Eros"])
                self.assertEqual(boot["state"]["values"]["preset"], "Balanced")
                self.assertEqual(boot["state"]["values"]["model_profile"], "Quality")
                self.assertEqual(boot["state"]["values"]["steps"], 8)
                self.assertEqual(boot["state"]["locked"], {})

                def pick(action, **changes):
                    resp = client.post("/h3/api/settings", json={"action": action, "changes": changes})
                    self.assertEqual(resp.status_code, 200, resp.text)
                    return resp.json()["state"]

                eros = pick("preset", preset="10Eros")
                self.assertEqual(eros["values"]["model_profile"], "10Eros v1.0")
                self.assertEqual(eros["values"]["turbo_variant"], BUILTIN_TURBO)
                self.assertEqual(eros["values"]["steps"], 8)
                self.assertEqual(eros["values"]["sla_preset"], "Balanced")
                self.assertEqual(eros["differences"], {})
                self.assertEqual(eros["issues"], [])
                self.assertEqual(eros["locked"]["generation_mode"], "Turbo")
                self.assertEqual(eros["locked"]["turbo_variant"], BUILTIN_TURBO)
                self.assertEqual(eros["locked"]["latent_upscale_refine_lora"], "Same as generation")
                self.assertEqual(eros["locked"]["latent_upscale_method"], "Full-frame refinement")
                self.assertNotIn("steps", eros["locked"])

                six_step = pick("steps", steps=6)
                self.assertEqual(six_step["values"]["steps"], 6)
                self.assertEqual(six_step["issues"], [])
                self.assertEqual(six_step["differences"]["steps"], [8, 6])

                for preset, expected_profile in (
                    ("Fast", "Speed"),
                    ("Balanced", "Quality"),
                    ("Quality", "Original"),
                    ("Singularity", "Singularity"),
                ):
                    state = pick("preset", preset=preset)
                    self.assertEqual(state["values"]["model_profile"], expected_profile)
                    self.assertEqual(state["locked"], {})
                    self.assertEqual(state["issues"], [])
            finally:
                JOBS.store = previous_store


if __name__ == "__main__":
    unittest.main()
