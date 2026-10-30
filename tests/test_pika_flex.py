"""Pika integration contracts without model downloads or GPU inference."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h3_models
from h3_app import model_service
from h3_app.catalog import PIKA_FLEX, PIKA_FLEX_SIGMAS, REFINEMENT_LORA_SETTINGS
from h3_app.errors import H3Error
from h3_app.policy import frame_length, turbo_sampler_name
from h3_app.settings import GenerationRequest, resolve_settings, transition_modes
from h3_app.workflows import h3
from h3_ui.persistence import restore_preferences
from h3_ui.settings_presentation import render_settings
from tests.workflow_cases import H3_DEFAULTS, MODELS


class PikaFlexTests(unittest.TestCase):
    def request(self, **values):
        return GenerationRequest.from_values({
            "turbo_variant": PIKA_FLEX, "latent_upscale": False, **values,
        })

    def graph_args(self, **values):
        return {
            **H3_DEFAULTS,
            "turbo_variant": PIKA_FLEX,
            "turbo_lora_name": MODELS.pika_flex_fl2va_lora,
            "available_nodes": set(H3_DEFAULTS["available_nodes"]) | h3.turbo_required_nodes(PIKA_FLEX),
            "width": 832, "steps": 6,
            **values,
        }

    def test_api_resolves_fixed_schedule_without_changing_other_features(self):
        plan = resolve_settings(self.request(steps=20, scheduler="beta"))
        self.assertFalse(plan.issues)
        self.assertEqual((plan.effective.sampling.steps, plan.effective.sampling.scheduler), (6, "simple"))
        self.assertEqual((plan.requested.sampling.steps, plan.requested.sampling.scheduler), (20, "beta"))
        self.assertEqual((plan.effective.output.width, plan.effective.output.height), (864, 480))
        self.assertEqual(plan.effective.cache_mode, plan.requested.cache_mode)
        self.assertTrue(plan.effective.semantic_bridge)
        self.assertEqual(frame_length(5), 124)
        self.assertEqual(frame_length(15), 362)
        self.assertIn("Pika fixed sigmas", render_settings(plan))
        normal = resolve_settings(self.request(generation_mode="Normal", steps=20, scheduler="beta"))
        self.assertEqual((normal.effective.sampling.steps, normal.effective.sampling.scheduler), (20, "beta"))
        self.assertEqual(turbo_sampler_name(PIKA_FLEX, None), "res_multistep")

    def test_unknown_turbo_label_resolves_to_the_adapter_that_runs(self):
        # The workflow loads Pika for unknown labels, so settings must enforce its schedule.
        plan = resolve_settings(self.request(
            generation_mode="Turbo", turbo_variant="Removed adapter", steps=20, scheduler="beta",
        ))
        self.assertEqual(plan.effective.sampling.turbo_variant, PIKA_FLEX)
        self.assertEqual((plan.effective.sampling.steps, plan.effective.sampling.scheduler), (6, "simple"))
        self.assertIn("turbo_variant", {item.field for item in plan.adjustments})
        legacy = resolve_settings(self.request(
            generation_mode="Turbo", turbo_variant="LightX2V v1.0 / 8-step 768p", steps=8,
        ))
        self.assertEqual(legacy.effective.sampling.turbo_variant, "LightX2V v1.1 / 8-step 768p")
        self.assertEqual(legacy.effective.sampling.steps, 8)

    def test_provenance_claims_upstream_identity_only_for_pinned_adapter(self):
        from h3_app.generation.h3 import pika_sampling_profile
        for mode, key in (("Reference media", "pika_flex_ref2va_lora"), ("reference MEDIA ", "pika_flex_ref2va_lora"),
                          ("Text to video", "pika_flex_fl2va_lora")):
            spec = h3_models.MODEL_SPECS[key]
            profile = pika_sampling_profile(mode, spec.local_name)
            self.assertEqual((profile["repo_id"], profile["revision"], profile["sha256"]),
                             (spec.repo_id, spec.revision, spec.expected_sha256))
        custom = pika_sampling_profile("Text to video", "my_pika_finetune.safetensors")
        self.assertEqual(custom["adapter"], "my_pika_finetune.safetensors")
        self.assertIsNone(custom["sha256"])
        self.assertIsNone(custom["repo_id"])

    def test_existing_output_and_refinement_features_remain_available(self):
        cases = (
            {"latent_upscale": True}, {"result_format": "Image"}, {"result_format": "Audio"},
            {"duration": 4}, {"duration": 5.5},
            {"mode": "First / last frame", "fl2va_audio_1": "voice.wav"},
            {"latent_upscale": True, "latent_upscale_refine_lora": PIKA_FLEX},
        )
        for values in cases:
            with self.subTest(values=values):
                plan = resolve_settings(self.request(**values))
                self.assertFalse(plan.issues)
        self.assertIn(PIKA_FLEX, REFINEMENT_LORA_SETTINGS)

    def test_ui_selection_and_saved_preference_keep_pika(self):
        _, values = transition_modes(None, {"generation_mode": "Turbo", "turbo_variant": PIKA_FLEX,
                                            "steps": 20, "scheduler": "beta", "latent_upscale": True}, "turbo_variant")
        self.assertEqual((values["steps"], values["scheduler"], values["latent_upscale"]), (6, "simple", True))
        restored, _ = restore_preferences({"h3.turbo_variant": PIKA_FLEX}, {
            "h3.turbo_variant": SimpleNamespace(value="old", choices=[PIKA_FLEX]),
        })
        self.assertEqual(restored["h3.turbo_variant"], PIKA_FLEX)

    def test_both_tasks_use_lcm_manual_sigmas_and_one_matching_adapter(self):
        for build, media in (
            (h3.build_fl2va_graph, {"first_image": None, "last_image": None}),
            (h3.build_fl2va_graph, {"first_image": "first.png", "last_image": "last.png"}),
            (h3.build_ref2va_graph, {"reference_images": ["ref.png"], "reference_videos": ["ref.mp4"], "reference_audios": ["ref.wav"], "ref_image_size": "match"}),
        ):
            with self.subTest(build=build.__name__, media=media):
                mode = "Reference media" if build is h3.build_ref2va_graph else "Text to video"
                filename = MODELS.turbo_lora_for(mode, PIKA_FLEX)
                args = self.graph_args(turbo_lora_name=filename)
                if build is h3.build_ref2va_graph:
                    for name in ("first_image", "last_image", "voice_reference_audios", *h3.FL2VA_ONLY_OPTIONS):
                        args.pop(name, None)
                graph = build(**{**args, **media})
                types = [node["class_type"] for node in graph.values()]
                self.assertNotIn("BasicScheduler", types)
                self.assertNotIn("H3LightX2VBypassLoRA", types)
                self.assertNotIn("MiniMaxH3TurboSampler", types)
                loras = [node["inputs"] for node in graph.values() if node["class_type"] == "LoraLoaderModelOnly"]
                self.assertEqual(loras, [{"model": loras[0]["model"], "lora_name": filename, "strength_model": 1.0}])
                sample = next(node["inputs"] for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced")
                self.assertEqual(graph[sample["sampler"][0]]["inputs"], {"sampler_name": "lcm"})
                sigma_node = graph[sample["sigmas"][0]]
                self.assertEqual(sigma_node, {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_FLEX_SIGMAS}})
                sigmas = [float(value) for value in PIKA_FLEX_SIGMAS.split(",")]
                expected = [12 * t / (1 + 11 * t) for t in (1, .75, .5, .25, .1, .04, 0)]
                for sigma, wanted in zip(sigmas, expected, strict=True):
                    self.assertAlmostEqual(sigma, wanted, places=6)
                    # Packed AV uses scale video_shift/audio_shift, yielding the audio ladder.
                    audio_sigma = sigma / (4 - 3 * sigma)
                    base = wanted / (12 - 11 * wanted)
                    self.assertAlmostEqual(audio_sigma, 3 * base / (1 + 2 * base), places=5)
                self.assertEqual(len(sigmas) - 1, 6)
                self.assertEqual(graph[sample["guider"][0]]["class_type"], "BasicGuider")
                shifts = [node["inputs"] for node in graph.values() if node["class_type"] == "MiniMaxH3SigmaShift"]
                self.assertEqual([(node["shift_video"], node["shift_audio"]) for node in shifts], [(12.0, 3.0)])

    def test_missing_nodes_fail_before_sampling(self):
        for missing in ("ManualSigmas", "MiniMaxH3SigmaShift", "LoraLoaderModelOnly"):
            args = self.graph_args()
            args["available_nodes"].remove(missing)
            with self.assertRaisesRegex(H3Error, missing):
                h3.build_fl2va_graph(**args)

    def test_native_upscale_keeps_full_pika_draft_and_refines_a_separate_branch(self):
        from h3_app.catalog import DMAD_4STEP_LORA

        for variant, filename in ((None, None), (DMAD_4STEP_LORA, "dmad.safetensors")):
            with self.subTest(refinement=variant):
                graph = h3.build_fl2va_graph(**self.graph_args(
                    width=1664, height=960, latent_upscale_model_name="upscale.pth",
                    refinement_lora_name=filename, refinement_variant=variant,
                ))
                samples = [node["inputs"] for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]
                self.assertEqual(len(samples), 2)
                draft, refined = samples
                self.assertEqual(graph[draft["sigmas"][0]]["inputs"], {"sigmas": PIKA_FLEX_SIGMAS})
                split = graph[refined["sigmas"][0]]["inputs"]
                if filename:
                    schedule = graph[split["sigmas"][0]]
                    self.assertEqual(schedule["class_type"], "BasicScheduler")
                    self.assertEqual(schedule["inputs"]["steps"], 4)
                    self.assertEqual(split["step"], 2)
                else:
                    self.assertEqual(split["sigmas"], draft["sigmas"])
                    self.assertEqual(split["step"], 4)
                self.assertEqual(graph[refined["sampler"][0]]["inputs"], {"sampler_name": "lcm"})
                upscaler = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3LatentUpscaleWithModel")
                self.assertIsNotNone(upscaler)
        plan = resolve_settings(self.request(latent_upscale=True, width=1664, height=960))
        self.assertFalse(plan.issues)
        self.assertEqual((plan.effective.output.width, plan.effective.output.height), (1664, 960))

    def test_adapter_is_lazy_and_only_selected_mode_downloads(self):
        runtime = Mock()
        for mode, key in (("Text to video", "pika_flex_fl2va_lora"), ("First / last frame", "pika_flex_fl2va_lora"), ("Reference media", "pika_flex_ref2va_lora")):
            with self.subTest(mode=mode), patch.object(model_service, "ensure_models", return_value=True) as ensure:
                self.assertTrue(model_service.ensure_turbo_lora(MODELS, PIKA_FLEX, mode, runtime=runtime))
                self.assertEqual(ensure.call_args.args[0], (key,))
                self.assertEqual(MODELS.turbo_lora_for(mode, PIKA_FLEX), h3_models.MODEL_SPECS[key].local_name)
                self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
                self.assertEqual(h3_models.build_model_config("manifest.json")[key], h3_models.MODEL_SPECS[key].local_name)

    def test_old_model_config_uses_default_pika_filenames(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "h3_models.json"
            config = h3_models.build_model_config("manifest.json")
            for key in ("pika_flex_fl2va_lora", "pika_flex_ref2va_lora"):
                config.pop(key)
            path.write_text(json.dumps(config), encoding="utf-8")
            models = model_service.load_model_config(runtime=SimpleNamespace(models_config=path))
            self.assertEqual(models.pika_flex_fl2va_lora, MODELS.pika_flex_fl2va_lora)
            self.assertEqual(models.pika_flex_ref2va_lora, MODELS.pika_flex_ref2va_lora)

    def test_pika_can_refine_another_base_schedule(self):
        from h3_app.catalog import LIGHTX2V_8STEP_TURBO

        graph = h3.build_fl2va_graph(**self.graph_args(
            steps=8, turbo_variant=LIGHTX2V_8STEP_TURBO,
            turbo_lora_name=MODELS.turbo_8step_lora,
            latent_upscale_model_name="upscale.pth",
            refinement_lora_name=MODELS.pika_flex_fl2va_lora, refinement_variant=PIKA_FLEX,
            available_nodes=set(H3_DEFAULTS["available_nodes"]) | h3.turbo_required_nodes(PIKA_FLEX),
        ))
        samples = [node["inputs"] for node in graph.values() if node["class_type"] == "SamplerCustomAdvanced"]
        draft, refined = samples
        self.assertEqual(graph[draft["sigmas"][0]]["class_type"], "BasicScheduler")
        split = graph[refined["sigmas"][0]]["inputs"]
        self.assertEqual(split["step"], 4)
        self.assertEqual(graph[split["sigmas"][0]]["inputs"], {"sigmas": PIKA_FLEX_SIGMAS})
        self.assertEqual(graph[refined["sampler"][0]]["inputs"], {"sampler_name": "lcm"})
        bad_replacement = resolve_settings(self.request(
            latent_upscale=True, latent_upscale_refine_lora="DMAD / 4-step",
            latent_upscale_refine_steps=4,
        ))
        self.assertIn(
            "Refinement steps must be positive and smaller than the selected refinement adapter's schedule.",
            bad_replacement.issues,
        )
        pika_refiner = resolve_settings(self.request(
            preset="Fast", turbo_variant="LightX2V / 4-step 768p (FL2V v1.3 · Ref2V v1.0)",
            steps=4, latent_upscale=True, latent_upscale_refine_lora=PIKA_FLEX,
            latent_upscale_refine_steps=5,
        ))
        self.assertFalse(pika_refiner.issues)


class PikaDownloadTests(unittest.TestCase):
    def test_sync_keeps_pinned_and_main_models_from_the_same_repo_separate(self):
        with TemporaryDirectory() as directory, patch.object(h3_models, "MIN_VALID_MODEL_BYTES", 0):
            root = Path(directory)
            cached = root / "cache"
            cached.write_bytes(b"weights")
            digest = hashlib.sha256(b"weights").hexdigest()
            specs = {
                "test_pinned": h3_models.ModelSpec("same/repo", "loras", "pinned", "test", expected_sha256=digest, revision="pin"),
                "test_main": h3_models.ModelSpec("same/repo", "loras", "main", "test"),
            }
            def metadata(repo, token, revision="main"):
                name = "pinned" if revision == "pin" else "main"
                return revision, {name: SimpleNamespace(lfs={"sha256": digest}, size=7, blob_id="blob")}

            manifest = root / "manifest.json"
            with patch.dict(h3_models.MODEL_SPECS, specs), patch.object(h3_models, "_fetch_repo", side_effect=metadata), patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as download, patch.object(h3_models, "_remove_hf_cache"):
                h3_models.sync_models(root=root, manifest_path=manifest, token="test", model_keys=specs)
            self.assertEqual({call.kwargs["revision"] for call in download.call_args_list}, {"pin", "main"})
            entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
            self.assertEqual(entries["loras/pinned"]["repo_revision"], "pin")
            self.assertEqual(entries["loras/main"]["repo_revision"], "main")
            self.assertEqual((root / "loras/pinned").read_bytes(), b"weights")

    def test_metadata_resolves_main_and_pinned_revision_independently(self):
        fake_api = Mock()
        fake_api.model_info.return_value = SimpleNamespace(sha="resolved", siblings=[])
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            result = h3_models._fetch_repositories(["repo", ("repo", "pinned")], None, 2)
        self.assertEqual(set(result), {"repo", ("repo", "pinned")})
        self.assertEqual({call.kwargs["revision"] for call in fake_api.model_info.call_args_list}, {"main", "pinned"})

    def test_corrupt_download_preserves_existing_destination(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached, dest = root / "cached", root / "dest"
            cached.write_bytes(b"wrong")
            dest.write_bytes(b"previous")
            spec = h3_models.ModelSpec("repo", "loras", "weights", "test", expected_sha256=hashlib.sha256(b"right").hexdigest(), revision="pinned")
            plan = {"spec": spec, "dest": dest, "revision": "pinned", "size": 5}
            with patch("huggingface_hub.hf_hub_download", return_value=str(cached)) as download:
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    h3_models._download_model(plan, None, "[test]", False)
            self.assertEqual(download.call_args.kwargs["revision"], "pinned")
            self.assertEqual(dest.read_bytes(), b"previous")

    def test_pin_and_hash_required_for_manifest_and_offline_reuse(self):
        with TemporaryDirectory() as tmp, patch.object(h3_models, "MIN_VALID_MODEL_BYTES", 0):
            root = Path(tmp)
            (root / "loras").mkdir()
            path = root / "loras/weights"
            path.write_bytes(b"weights")
            digest = hashlib.sha256(b"weights").hexdigest()
            spec = h3_models.ModelSpec("repo", "loras", "weights", "test", expected_sha256=digest, revision="pinned")
            entry = {"repo_id": "repo", "filename": "weights", "size": 7, "sha256": digest, "repo_revision": "pinned"}
            manifest = {"files": {"loras/weights": entry}}
            self.assertTrue(h3_models.model_file_matches_manifest(root, manifest, spec))
            entry["repo_revision"] = "older"
            self.assertFalse(h3_models.model_file_matches_manifest(root, manifest, spec))
            with self.assertRaisesRegex(RuntimeError, "Cannot verify pinned"):
                h3_models._plan_model(root=root, manifest=manifest, repo_results={("repo", "pinned"): OSError("offline")}, key="test", spec=spec, log_prefix="[test]")
            entry["repo_revision"] = "pinned"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                h3_models._plan_model(root=root, manifest=manifest, repo_results={("repo", "pinned"): OSError("offline")}, key="test", spec=spec, log_prefix="[test]")


if __name__ == "__main__":
    unittest.main()
