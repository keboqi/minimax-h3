"""P-H3-Image selected-base routing and native workflow contracts (CPU only)."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from h3_app.catalog import H3_EDIT_SIZE_CAP, H3_EDIT_SIZE_MATCH
from unittest.mock import Mock, patch

from h3_app.catalog import UI_DEFAULTS
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.generation.ph3_image import generate_ph3_image as generate
from h3_app.generation.requests import H3Request
from h3_app.image_canvas import apply_job_canvas
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.settings import GenerationRequest, ResolutionContext, resolve_settings
from h3_app.workflows.ph3_image import PH3_IMAGE_KEYS, PH3_IMAGE_SIGMAS, build_ph3_image_graph
from h3_models import LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS, PRELOAD_MODEL_KEYS
from h3_ui.h3.layout import auto_resolution_from_start_frame, start_frame_generation_resolution
from PIL import Image


def inputs(graph, kind):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == kind)


class PH3ImageTests(unittest.TestCase):
    context = ResolutionContext()

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
            policy=SimpleNamespace(resolve_request_settings=lambda values: resolve_settings(
                GenerationRequest.from_values(values), self.context)),
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
                results = list(generate(request, services, SimpleNamespace()))
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
        results = list(generate(self.request(), services, SimpleNamespace()))
        self.assertIn("PH3ImageConditioning", results[-1].status)
        services.models.ensure_model_keys.assert_not_called()
        services.models.ensure_profile_model.assert_not_called()
        services.execution.submit_prompt.assert_not_called()

    def test_reference_edit_stages_images_and_still_selects_fl2va(self):
        services = self.services()
        with patch("h3_app.generation.image_jobs.write_snapshot"):
            result = list(generate(self.request(mode="Reference media", model_profile="Speed",
                                                ref_image_1="target.png", ref_image_2="style.png"),
                                   services, SimpleNamespace()))[-1]
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
            result = list(generate(self.request(**changes), services, SimpleNamespace()))[-1]
            self.assertTrue(result.status.startswith("Error:"))
            services.models.ensure_model_keys.assert_not_called()

    def test_the_graph_uses_the_size_the_plan_resolved(self):
        for context, changes, size in (
            (ResolutionContext(), {"width": 1000, "height": 1000}, (992, 992)),
            (ResolutionContext(start_dimensions=(1280, 704)),
             {"mode": "First / last frame", "first_image": "start.png"}, (1280, 704)),
        ):
            with self.subTest(size=size), patch("h3_app.generation.image_jobs.write_snapshot"), \
                    patch.object(self, "context", context):
                services = self.services()
                results = list(generate(self.request(**changes), services, SimpleNamespace()))
                self.assertEqual(results[-1].output, ["fox.png"])
                graph = services.execution.submit_prompt.call_args.args[0]
                conditioning = inputs(graph, "PH3ImageConditioning")
                self.assertEqual((conditioning["width"], conditioning["height"]), size)

    def test_first_frame_edit_scales_to_the_selected_pixel_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            first = str(Path(directory) / "first.png")
            Image.new("RGB", (3000, 2000)).save(first)
            for cap, size in (("1 MP", (1152, 768)), ("4 MP", (2432, 1632)), ("8 MP", (3456, 2304))):
                with self.subTest(cap=cap):
                    self.assertEqual(start_frame_generation_resolution(
                        first, alignment=32, auto_megapixels=cap, edit_size=H3_EDIT_SIZE_CAP), size)
                    self.assertEqual(auto_resolution_from_start_frame(
                        first, 1, 1, "Image", False, cap, H3_EDIT_SIZE_CAP)[:2], size)
            self.assertEqual(self.request(first_image=first, auto_megapixels="4 MP").values()["auto_megapixels"], "4 MP")

            small = str(Path(directory) / "small.png")
            Image.new("RGB", (640, 640)).save(small)
            self.assertEqual(start_frame_generation_resolution(
                small, alignment=32, auto_megapixels="4 MP", edit_size=H3_EDIT_SIZE_CAP), (1984, 1984))
            # Video keeps its cap as a ceiling and never upscales.
            self.assertEqual(auto_resolution_from_start_frame(small, 1, 1, "Video", False, "4 MP")[:2], (640, 640))

    def test_a_first_frame_edit_matches_its_image_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            for source, size in (((1280, 736), (1280, 736)), ((640, 640), (640, 640)), ((1920, 1080), (1920, 1088)),
                                 ((6000, 4000), (3456, 2304))):
                with self.subTest(source=source):
                    first = str(Path(directory) / f"{source[0]}.png")
                    Image.new("RGB", source).save(first)
                    # Whatever the cap says: 1 MP would shrink every one of these.
                    self.assertEqual(start_frame_generation_resolution(first, alignment=32, auto_megapixels="1 MP"), size)
                    self.assertEqual(auto_resolution_from_start_frame(first, 1, 1, "Image", False, "1 MP")[:2], size)
            legacy = [None] * GENERATION_FIELDS.index("image_edit_size")
            self.assertEqual(GenerationArguments.from_positional(legacy).values["image_edit_size"], H3_EDIT_SIZE_MATCH)

    def test_old_positional_calls_default_the_cap(self):
        legacy = [None] * GENERATION_FIELDS.index("auto_megapixels")
        self.assertEqual(GenerationArguments.from_positional(legacy).values["auto_megapixels"], "1 MP")

    def test_a_canvas_transform_keeps_its_exact_size_past_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            Image.new("RGB", (640, 320)).save(first)
            job = SimpleNamespace(
                canvas_request={"mode": "Fit / pad", "width": 1024, "height": 512}, family="h3",
                input_lease=directory, callback=SimpleNamespace(job_input_names=(
                    "result_format", "mode", "first_image", "last_image", "auto_megapixels")))
            values = apply_job_canvas(job, ["Image", "First / last frame", str(first), None, "8 MP"])
            self.assertEqual(start_frame_generation_resolution(values[2], alignment=32, auto_megapixels=values[4]),
                             (1024, 512))

    def test_video_only_settings_cannot_block_an_image(self):
        plan = resolve_settings(GenerationRequest.from_values({
            **UI_DEFAULTS, "result_format": "Image", "prompt": "a fox", "steps": 1, "image_frames": 99,
            "image_vae": "Unknown", "latent_upscale": True, "latent_upscale_refine_steps": 99,
        }))
        self.assertEqual(plan.issues, ())
        self.assertFalse(plan.effective.finishing.latent_upscale)

    def test_weights_are_lazy_and_no_image_transformer_is_provisioned(self):
        self.assertTrue(set(PH3_IMAGE_KEYS) <= set(LAZY_OPTIONAL_MODEL_KEYS))
        self.assertFalse(set(PH3_IMAGE_KEYS) & set(PRELOAD_MODEL_KEYS))
        self.assertFalse(any(MODEL_SPECS[key].folder == "diffusion_models" for key in PH3_IMAGE_KEYS))


if __name__ == "__main__":
    unittest.main()
