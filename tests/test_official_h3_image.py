"""Official image routing, native sampling and lazy provisioning contracts."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app.capabilities import DEFAULT_ENGINES
from h3_app.errors import H3Error
from h3_app.generation.h3_image import H3ImageRequest, generate_h3_image, validate_h3_image
from h3_app.workflows.h3_image import H3_IMAGE_KEYS, build_h3_image_graph
from h3_app.h3_image import H3_IMAGE_OFFICIAL
from h3_models import LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS, PRELOAD_MODEL_KEYS
from tests import test_image_workspace as workspace


def inputs(graph, kind):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == kind)


class OfficialH3ImageTests(unittest.TestCase):
    def services(self, request, missing=()):
        graph = build_h3_image_graph(**request.__dict__)
        info = {node["class_type"]: {} for node in graph.values() if node["class_type"] not in missing}
        info["CLIPLoader"] = {"input": {"required": {"type": [["minimax"]]}}}
        return SimpleNamespace(
            models=SimpleNamespace(unload_prompt_rewriter=Mock(), ensure_model_keys=Mock(),
                                   ensure_profile_model=Mock(), load_model_config=Mock()),
            execution=SimpleNamespace(object_info=Mock(return_value=info), submit_prompt=Mock(return_value="job"),
                                      poll_comfy_progress=Mock(return_value=[]), wait_for_history=Mock(return_value={})),
            media=SimpleNamespace(stage_file=Mock(side_effect=lambda path, _: "staged/" + path),
                                  resolve_image_outputs=Mock(return_value=[Path("official.png")])) )

    def test_native_model_sampling_and_vae_are_independent_of_video(self):
        graph = build_h3_image_graph(prompt="fox", expand_prompt=False, seed=123)
        kinds = {node["class_type"] for node in graph.values()}
        self.assertFalse(kinds & {"LoraLoaderBypassModelOnly", "MiniMaxH3SigmaShift", "PH3ImageConditioning"})
        self.assertEqual(inputs(graph, "UNETLoader")["unet_name"], MODEL_SPECS["h3_image_dit"].local_name)
        self.assertEqual(inputs(graph, "VAELoader")["vae_name"], MODEL_SPECS["h3_image_vae"].local_name)
        sample = inputs(graph, "KSampler")
        self.assertEqual((sample["steps"], sample["cfg"], sample["sampler_name"], sample["scheduler"]),
                         (6, 1.0, "euler", "simple"))
        cond = inputs(graph, "MiniMaxH3ImageConditioning")
        self.assertEqual((cond["width"], cond["height"], cond["prompt"]), (4096, 4096, "fox"))
        self.assertEqual(graph[sample["latent_image"][0]]["inputs"], cond)
        self.assertEqual(sample["seed"], 123)

    def test_sixteen_references_keep_order_in_both_consumers(self):
        graph = build_h3_image_graph(prompt="combine", reference_images=tuple(f"{i}.png" for i in range(16)))
        cond = inputs(graph, "MiniMaxH3ImageConditioning")
        writer = inputs(graph, "TextGenerateV2")
        for i in range(1, 17):
            original = cond[f"images.image_{i}"]
            self.assertEqual(graph[original[0]]["inputs"]["image"], f"{i - 1}.png")
            self.assertEqual(graph[writer[f"images.image{i}"][0]]["inputs"]["image"], original)
        self.assertIn("<Picture 16>", writer["prompt"])
        self.assertFalse(writer["use_default_template"])
        self.assertEqual(graph[cond["prompt"][0]]["class_type"], "TextGenerateV2")

    def test_job_stages_all_references_without_loading_a_video_profile(self):
        request = H3ImageRequest("combine", seed=42, reference_images=tuple(f"{i}.png" for i in range(16)))
        services = self.services(request)
        with patch("h3_app.generation.image_jobs.write_snapshot") as snapshot:
            results = list(generate_h3_image(request, services, SimpleNamespace()))
        self.assertEqual(results[-1].output, ["official.png"])
        services.models.ensure_model_keys.assert_called_once_with(H3_IMAGE_KEYS + ("h3_image_upsampler",))
        services.models.ensure_profile_model.assert_not_called()
        services.models.load_model_config.assert_not_called()
        graph = services.execution.submit_prompt.call_args.args[0]
        loaded = [node["inputs"]["image"] for node in graph.values() if node["class_type"] == "LoadImage"]
        self.assertEqual(loaded, ["staged/" + path for path in request.reference_images])
        self.assertEqual(snapshot.call_args.args[1]["settings"]["backend"], H3_IMAGE_OFFICIAL)

    def test_missing_core_support_fails_before_download(self):
        request = H3ImageRequest("fox", expand_prompt=False)
        services = self.services(request, missing=("MiniMaxH3ImageConditioning",))
        results = list(generate_h3_image(request, services, SimpleNamespace()))
        self.assertIn("MiniMaxH3ImageConditioning", results[-1].status)
        services.models.ensure_model_keys.assert_not_called()
        services.execution.submit_prompt.assert_not_called()

    def test_disabled_expansion_does_not_download_writer(self):
        request = H3ImageRequest("fox", expand_prompt=False)
        services = self.services(request)
        with patch("h3_app.generation.image_jobs.write_snapshot"):
            list(generate_h3_image(request, services, SimpleNamespace()))
        services.models.ensure_model_keys.assert_called_once_with(H3_IMAGE_KEYS)

    def test_local_expansion_honors_the_runtime_system_prompt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            system = Path(directory, "custom.txt")
            system.write_text("Custom H3 Image instructions.", encoding="utf-8")
            request = H3ImageRequest("fox")
            services = self.services(request)
            runtime = SimpleNamespace(prompt_systems={"MiniMax H3 Image": system})
            with patch("h3_app.generation.image_jobs.write_snapshot"):
                result = list(generate_h3_image(request, services, runtime))[-1]
        self.assertEqual(result.output, ["official.png"])
        graph = services.execution.submit_prompt.call_args.args[0]
        self.assertIn("Custom H3 Image instructions.", inputs(graph, "TextGenerateV2")["prompt"])

    def test_limits_are_validated_before_backend_or_provisioning(self):
        for request in (H3ImageRequest(""), H3ImageRequest("fox", reference_images=("x",) * 17),
                        H3ImageRequest("fox", width=8192, height=8192), H3ImageRequest("fox", width=4097)):
            with self.subTest(request=request), self.assertRaises(H3Error):
                validate_h3_image(request)
        validate_h3_image(H3ImageRequest("fox", width=6272, height=2688))

    def test_defaults_and_lazy_inventory(self):
        self.assertEqual(DEFAULT_ENGINES["Image"], "h3")
        keys = {"h3_image_dit", "h3_image_vae", "h3_image_upsampler"}
        self.assertTrue(keys <= set(LAZY_OPTIONAL_MODEL_KEYS))
        self.assertFalse(keys & set(PRELOAD_MODEL_KEYS))
        self.assertTrue(all(MODEL_SPECS[key].revision for key in keys))

    def test_legacy_image_result_format_also_defaults_to_official(self):
        from h3_app.generation.h3 import generate
        from tests.test_ph3_image import PH3ImageTests

        fixture = PH3ImageTests()
        request = fixture.request()
        services = self.services(H3ImageRequest("fox"))
        services.policy = fixture.services().policy
        with patch("h3_app.generation.image_jobs.write_snapshot"):
            result = list(generate(request, services, SimpleNamespace(), run_context={}))[-1]
        self.assertEqual(result.output, ["official.png"])
        graph = services.execution.submit_prompt.call_args.args[0]
        self.assertEqual(inputs(graph, "UNETLoader")["unet_name"], MODEL_SPECS["h3_image_dit"].local_name)
        services.models.ensure_profile_model.assert_not_called()

    def test_controller_defaults_to_official_and_preserves_all_sixteen_inputs(self):
        fixture = workspace.ImageGenerationControllerTests()
        fixture.setUp()
        captured = {**fixture.values, "image__width": 4096, "image__height": 4096,
                    "image__seed": 42}
        captured.pop("image__backend")
        with patch("h3_app.generation.h3_image.generate_h3_image", return_value=iter(())) as generate:
            fixture.controller.generate("h3", "Image edit", "combine", ["x"] * 16, captured)
        self.assertEqual(len(generate.call_args.args[0].reference_images), 16)
        self.assertEqual(generate.call_args.args[0].width, 4096)


if __name__ == "__main__":
    unittest.main()
