"""P-H3-Image selected-base routing and native workflow contracts (CPU only)."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app.catalog import UI_DEFAULTS
from h3_app.contracts import GENERATION_FIELDS
from h3_app.generation.h3 import generate
from h3_app.generation.requests import H3Request
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.workflows.ph3_image import PH3_IMAGE_KEYS, PH3_IMAGE_SIGMAS, build_ph3_image_graph
from h3_models import LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS, PRELOAD_MODEL_KEYS


def inputs(graph, kind):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == kind)


class PH3ImageTests(unittest.TestCase):
    def request(self, **changes):
        return H3Request.from_values({**dict.fromkeys(GENERATION_FIELDS), **UI_DEFAULTS, "result_format": "Image",
                                      "width": 2912, "height": 2912, "seed": 42,
                                      "prompt": "a fox", **changes})

    def services(self, missing=()):
        profiles = {name: ModelProfile(name.title(), f"custom_{name}_fl2va.safetensors",
                                       f"custom_{name}_ref2va.safetensors")
                    for name in ("speed", "quality", "original", "singularity")}
        models = ModelConfig(profiles, "quality", "old_text", "old_video", "old_audio")
        graph = build_ph3_image_graph(model_name="dummy", prompt="test", reference_images=("a", "b"))
        info = {node["class_type"]: {} for node in graph.values()
                if node["class_type"] not in missing}
        info["CLIPLoader"] = {"input": {"required": {"type": [["minimax"]]}}}
        return SimpleNamespace(
            models=SimpleNamespace(load_model_config=Mock(return_value=models),
                                   unload_prompt_rewriter=Mock(), ensure_model_keys=Mock(),
                                   ensure_profile_model=Mock()),
            execution=SimpleNamespace(object_info=Mock(return_value=info),
                                      submit_prompt=Mock(return_value="job-1"),
                                      poll_comfy_progress=Mock(return_value=[]),
                                      wait_for_history=Mock(return_value={})),
            media=SimpleNamespace(resolve_image_outputs=Mock(return_value=[Path("fox.png")]),
                                  stage_file=Mock(side_effect=lambda path, category: "staged/" + path)),
        )

    def test_all_selected_profiles_reuse_their_configured_fl2va_base(self):
        for profile in ("Speed", "Quality", "Original", "Singularity"):
            with self.subTest(profile=profile), patch("h3_app.generation.image_jobs.write_snapshot") as snapshot:
                services = self.services()
                # Reference mode must also use the selected FL2VA, never Ref2VA.
                request = self.request(model_profile=profile)
                results = list(generate(request, services, SimpleNamespace(), run_context={}))
                self.assertEqual(results[-1].output, ["fox.png"])
                graph = services.execution.submit_prompt.call_args.args[0]
                self.assertEqual(inputs(graph, "UNETLoader")["unet_name"],
                                 f"custom_{profile.lower()}_fl2va.safetensors")
                services.models.ensure_profile_model.assert_called_once_with(
                    profile.lower(), services.models.load_model_config().profiles[profile.lower()], "Text to video")
                services.models.ensure_model_keys.assert_called_once_with(PH3_IMAGE_KEYS)
                settings = snapshot.call_args.args[1]["settings"]
                self.assertEqual((settings["steps"], settings["seed"]), (8, 42))

    def test_native_schedule_lora_and_upsampler_reach_sampling(self):
        graph = build_ph3_image_graph(model_name="selected.safetensors", prompt="fox", seed=9)
        lora = inputs(graph, "LoraLoaderBypassModelOnly")
        self.assertEqual(lora["strength_model"], 1.0)
        self.assertEqual(lora["lora_name"], MODEL_SPECS["ph3_image_lora"].local_name)
        self.assertEqual(inputs(graph, "ManualSigmas")["sigmas"], PH3_IMAGE_SIGMAS)
        self.assertEqual(inputs(graph, "KSamplerSelect")["sampler_name"], "euler")
        self.assertEqual(inputs(graph, "MiniMaxH3SigmaShift")["shift_video"], 1.0)
        cond = inputs(graph, "PH3ImageConditioning")
        self.assertEqual(graph[cond["prompt"][0]]["class_type"], "TextGenerate")
        self.assertEqual(inputs(graph, "TextGenerate")["prompt"], "Aspect ratio: 1:1\nRequest: fox")
        sample = inputs(graph, "SamplerCustomAdvanced")
        self.assertEqual(sample["latent_image"][1], 1)
        self.assertEqual(graph[sample["latent_image"][0]]["inputs"], cond)

    def test_eight_edit_images_keep_order_in_writer_and_conditioning(self):
        graph = build_ph3_image_graph(model_name="selected", prompt="combine",
                                      reference_images=[f"ref{i}.png" for i in range(8)])
        cond, batch = inputs(graph, "PH3ImageConditioning"), inputs(graph, "BatchImagesNode")
        self.assertEqual(len(batch), 8)
        for i in range(8):
            image_ref = cond[f"images.image_{i + 1}"]
            self.assertEqual(graph[image_ref[0]]["inputs"]["image"], f"ref{i}.png")
            writer_image = graph[batch[f"images.image{i}"][0]]["inputs"]["image"]
            self.assertEqual(writer_image, image_ref)
        self.assertIn("<Picture 8>", inputs(graph, "TextGenerate")["prompt"])
        with self.assertRaises(ValueError):
            build_ph3_image_graph(model_name="selected", prompt="test", reference_images=["x"] * 9)

    def test_old_backend_fails_before_any_download(self):
        services = self.services(missing=("PH3ImageConditioning",))
        results = list(generate(self.request(), services, SimpleNamespace(), run_context={}))
        self.assertIn("PH3ImageConditioning", results[-1].status)
        services.models.ensure_model_keys.assert_not_called()
        services.models.ensure_profile_model.assert_not_called()
        services.execution.submit_prompt.assert_not_called()

    def test_reference_edit_stages_images_and_still_selects_fl2va(self):
        services = self.services()
        with patch("h3_app.generation.image_jobs.write_snapshot"):
            result = list(generate(self.request(mode="Reference media", model_profile="Speed",
                                                ref_image_1="target.png", ref_image_2="style.png"),
                                   services, SimpleNamespace(), run_context={}))[-1]
        self.assertEqual(result.output, ["fox.png"])
        graph = services.execution.submit_prompt.call_args.args[0]
        self.assertEqual(inputs(graph, "UNETLoader")["unet_name"], "custom_speed_fl2va.safetensors")
        loaded = [node["inputs"]["image"] for node in graph.values() if node["class_type"] == "LoadImage"]
        self.assertEqual(loaded, ["staged/target.png", "staged/style.png"])
        self.assertEqual(services.models.ensure_profile_model.call_args.args[2], "Text to video")

    def test_invalid_edit_references_do_not_download(self):
        for changes in ({"mode": "Reference media", "ref_video_1": "clip.mp4"},
                        {"mode": "Reference media", **{f"ref_image_{i}": f"{i}.png" for i in range(1, 10)}}):
            services = self.services()
            result = list(generate(self.request(**changes), services, SimpleNamespace(), run_context={}))[-1]
            self.assertTrue(result.status.startswith("Error:"))
            services.models.ensure_model_keys.assert_not_called()

    def test_weights_are_lazy_and_no_image_transformer_is_provisioned(self):
        self.assertTrue(set(PH3_IMAGE_KEYS) <= set(LAZY_OPTIONAL_MODEL_KEYS))
        self.assertFalse(set(PH3_IMAGE_KEYS) & set(PRELOAD_MODEL_KEYS))
        self.assertFalse(any(MODEL_SPECS[key].folder == "diffusion_models" for key in PH3_IMAGE_KEYS))


if __name__ == "__main__":
    unittest.main()
