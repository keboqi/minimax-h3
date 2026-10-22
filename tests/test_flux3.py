import ast
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.errors import H3Error
from h3_app.generation.flux3 import (
    flux3_edit_dimensions,
    resolve_flux3_dimensions,
    validate_flux3 as validate_request,
)
from h3_app.generation.requests import Flux3Request
from h3_app.prompt_writers import flux3 as flux3_writer
from h3_app.workflows.flux3 import UPSAMPLER_MAX_LENGTH, build_flux3_graph, upsampler_system
from h3_models import (
    FLUX3_MODEL_CHOICES, MODEL_SPECS, MIN_VALID_MODEL_BYTES,
    _plan_model, model_file_matches_manifest, model_manifest_key,
)

ROOT = Path(__file__).resolve().parents[1]


class Flux3OfficialUpdateTests(unittest.TestCase):
    def test_local_edit_expansion_sees_the_same_ten_ordered_references_as_the_dit(self):
        paths = tuple(f"ref{i}.png" for i in range(10))
        validate_request(Flux3Request("change the sky", reference_images=paths, expand_prompt=True))
        graph = build_flux3_graph(model_choice="BF16", prompt="change the sky", reference_images=paths)
        writer_id, writer = next((k, n) for k, n in graph.items()
                                 if n["class_type"] == "H3Flux3EditPrompt")
        refs = next(n["inputs"] for n in graph.values() if n["class_type"] == "ReferenceMultiLatent")
        for i, path in enumerate(paths):
            image = writer["inputs"][f"image{i}"]
            self.assertEqual(image, refs[f"images.image{i}"])
            self.assertEqual(graph[image[0]]["inputs"]["image"], path)
        self.assertEqual(writer["inputs"]["system_prompt"], (ROOT / "prompts" / "prompt_flux3_i2i.txt").read_text(encoding="utf-8").strip())
        encoded = next(n["inputs"] for n in graph.values() if n["class_type"] == "CLIPTextEncode")
        self.assertEqual(encoded["text"], [writer_id, 0])

    def test_bfl_alpha_16_reaches_the_sampler_at_every_canvas_size(self):
        # BFL 62e90b9278: alpha*t/(1+(alpha-1)*t). Comfy takes mu=ln(alpha),
        # then interpolates by canvas size; equal anchors make that constant.
        for size in ((256, 256), (1024, 1024), (2048, 2048), (4096, 1024)):
            graph = build_flux3_graph(model_choice="BF16", prompt="a fox",
                                      width=size[0], height=size[1], expand_prompt=False)
            node_id, node = next((k, n) for k, n in graph.items()
                                 if n["class_type"] == "ModelSamplingFlux")
            values = node["inputs"]
            slope = (values["max_shift"] - values["base_shift"]) / (4096 - 256)
            mu = size[0] * size[1] / 256 * slope + values["base_shift"] - 256 * slope
            self.assertAlmostEqual(math.exp(mu), 16)
            self.assertAlmostEqual(math.exp(mu) / (math.exp(mu) + 1), 16 / 17)
            sampler = next(n["inputs"] for n in graph.values() if n["class_type"] == "KSampler")
            self.assertEqual(sampler["model"], [node_id, 0])
            self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"]), (16, 1.0, "euler"))

    def test_corrected_workflow_disables_speculative_caption_decoding(self):
        graph = build_flux3_graph(model_choice="BF16", prompt="a fox")
        writer = next(n["inputs"] for n in graph.values() if n["class_type"] == "TextGenerate")
        self.assertEqual(writer["mtp"], "off")
        self.assertEqual(writer["sampling_mode.top_k"], 1)

    def test_renamed_pinned_weights_reuse_old_local_files_and_download_identity(self):
        for key in ("flux3_nvfp4", "flux3_bf16"):
            spec = MODEL_SPECS[key]
            old_filename = "diffusion_models/" + spec.local_name
            self.assertNotEqual(old_filename, spec.filename)
            self.assertTrue(spec.filename.startswith("diffusion_models/flux3-image-klein-"))
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                dest = root / spec.folder / spec.local_name
                dest.parent.mkdir()
                size = MIN_VALID_MODEL_BYTES + 1
                with dest.open("wb") as stream:
                    stream.truncate(size)
                entry = dict(repo_id=spec.repo_id, filename=old_filename, size=size,
                             sha256=spec.expected_sha256, remote_identity=spec.expected_sha256)
                manifest = {"files": {model_manifest_key(spec): entry}}
                self.assertTrue(model_file_matches_manifest(root, manifest, spec))
                sibling = SimpleNamespace(size=size, lfs={"sha256": spec.expected_sha256})
                plan = _plan_model(root=root, manifest=manifest, key=key, spec=spec,
                    repo_results={spec.repo_id: ("new-revision", {spec.filename: sibling})}, log_prefix="[test]")
                self.assertFalse(plan["needs_download"])
                self.assertEqual(plan["dest"], dest)
                entry["sha256"] = "wrong"
                self.assertFalse(model_file_matches_manifest(root, manifest, spec))
                entry["sha256"] = spec.expected_sha256
                entry["repo_id"] = "different/repo"
                self.assertFalse(model_file_matches_manifest(root, manifest, spec))
                entry["repo_id"] = spec.repo_id
                self.assertFalse(model_file_matches_manifest(root, manifest, replace(spec, expected_sha256=None)))

    def test_int8_selection_uses_distilled_checkpoint_and_cfg_one(self):
        choice = "INT8 ConvRot (lower VRAM)"
        self.assertEqual(FLUX3_MODEL_CHOICES[choice], "flux3_int8")
        graph = build_flux3_graph(model_choice=choice, prompt="a fox", expand_prompt=False)
        loader = next(n["inputs"] for n in graph.values() if n["class_type"] == "UNETLoader")
        self.assertEqual(loader["unet_name"], "flux3-image-klein-int8-convrot.safetensors")
        self.assertEqual(next(n["inputs"]["cfg"] for n in graph.values()
                              if n["class_type"] == "KSampler"), 1.0)


class Flux3EditUpsamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "custom_nodes/H3Acceleration/__init__.py"
        node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "H3Flux3EditPrompt")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls.upsampler = namespace["H3Flux3EditPrompt"]()

    def test_unequal_size_rgba_references_reach_the_native_tokenizer_in_order(self):
        images = [np.zeros((1, 48, 96, 4)), np.ones((1, 120, 60, 3))]
        clip = Mock()
        clip.decode.return_value = "<think>analysis</think> Replace the sky. "
        result = self.upsampler.generate(clip, "change sky", "edit instructions", 3584, 0,
                                        images[0], image1=images[1])
        self.assertEqual(result, ("Replace the sky.",))
        params = clip.tokenize.call_args.kwargs
        self.assertEqual([x.shape for x in params["images"]], [(1, 48, 96, 3), (1, 120, 60, 3)])
        for actual, original in zip(params["images"], images):
            self.assertTrue(np.shares_memory(actual, original))
            np.testing.assert_array_equal(actual, original[:, :, :, :3])
        self.assertEqual(params["system_prompt"], "edit instructions")
        generation = clip.generate.call_args
        self.assertEqual(generation.args, (clip.tokenize.return_value,))
        self.assertEqual((generation.kwargs["top_k"], generation.kwargs["presence_penalty"],
                          generation.kwargs["mtp"]), (1, 0.0, False))

    def test_empty_or_truncated_reasoning_never_becomes_an_unconditional_edit(self):
        for text in ("", "   ", "<think>unfinished reasoning", "<think>x</think> "):
            clip = Mock()
            clip.decode.return_value = text
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "no instruction"):
                self.upsampler.generate(clip, "edit", "rules", 3584, 0, np.zeros((1, 4, 4, 3)))


class Flux3EditSizeTests(unittest.TestCase):
    def test_request_defaults_to_matching_the_first_reference(self):
        self.assertEqual(Flux3Request("x").edit_size, QWEN_EDIT_SIZE_MATCH)

    def test_match_keeps_native_size_on_the_grid_and_caps_at_4mp(self):
        self.assertEqual(flux3_edit_dimensions(1024, 1024), (1024, 1024))
        self.assertEqual(flux3_edit_dimensions(640, 480), (640, 480))
        self.assertEqual(flux3_edit_dimensions(1920, 1080), (1920, 1072))
        self.assertEqual(flux3_edit_dimensions(1080, 1920), (1072, 1920))
        self.assertEqual(flux3_edit_dimensions(200, 200), (256, 256))

    def test_max_scales_toward_4mp(self):
        self.assertEqual(flux3_edit_dimensions(1024, 1024, maximize=True), (2048, 2048))
        self.assertEqual(flux3_edit_dimensions(1920, 1080, maximize=True), (2720, 1536))

    def test_sizes_respect_grid_bounds_and_aspect(self):
        for source in ((1000, 750), (6000, 4000), (3840, 2160), (1023, 767), (750, 1000)):
            for maximize in (False, True):
                width, height = flux3_edit_dimensions(*source, maximize=maximize)
                self.assertEqual((width % 16, height % 16), (0, 0))
                self.assertLessEqual(width * height, 4_194_304)
                self.assertAlmostEqual(width / height, source[0] / source[1], delta=0.02 * width / height)

    def test_resolution_follows_mode_and_reference(self):
        with tempfile.TemporaryDirectory() as root:
            Image.new("RGB", (1536, 1024)).save(Path(root) / "ref.png")
            request = Flux3Request("x", reference_images=("ref.png",), width=1024, height=1024,
                                   expand_prompt=False)
            self.assertEqual(resolve_flux3_dimensions(request, input_dir=Path(root)), (1536, 1024))
            maxed = replace(request, edit_size=QWEN_EDIT_SIZE_MAX)
            self.assertEqual(resolve_flux3_dimensions(maxed, input_dir=Path(root)), (2496, 1680))
            manual = replace(request, edit_size=QWEN_EDIT_SIZE_MANUAL)
            self.assertEqual(resolve_flux3_dimensions(manual, input_dir=Path(root)), (1024, 1024))
        text_only = Flux3Request("x", width=1280, height=768)
        self.assertEqual(resolve_flux3_dimensions(text_only), (1280, 768))

    def test_invalid_edit_size_is_rejected(self):
        with self.assertRaises(H3Error):
            validate_request(Flux3Request("x", edit_size="Stretch"))

    def test_graph_canvas_uses_resolved_size(self):
        graph = build_flux3_graph(model_choice="BF16", prompt="x", reference_images=("ref.png",),
                                  width=1536, height=1024, expand_prompt=False)
        latent = next(n for n in graph.values() if n["class_type"] == "EmptyFlux2LatentImage")
        self.assertEqual((latent["inputs"]["width"], latent["inputs"]["height"]), (1536, 1024))


    def test_upsampler_seed_is_constant_so_rerolls_reuse_the_expansion(self):
        def upsampler(seed):
            graph = build_flux3_graph(model_choice="BF16", prompt="a fox", seed=seed)
            return next(n for n in graph.values() if n["class_type"] == "TextGenerate")["inputs"]
        self.assertEqual(upsampler(1), upsampler(987654321))
        self.assertEqual(upsampler(1)["sampling_mode.top_k"], 1)

    def test_image_is_saved_where_the_shared_resolver_and_gallery_look(self):
        graph = build_flux3_graph(model_choice="BF16", prompt="x", expand_prompt=False,
                                  output_stamp="123", output_nonce="abcd")
        save = next(n for n in graph.values() if n["class_type"] == "SaveImage")
        self.assertEqual(save["inputs"]["filename_prefix"], "h3/image_staging/flux3_123_abcd")


class Flux3WriterTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(prompt_systems={
            "FLUX 3 Image Klein": ROOT / "prompts" / "prompt_flux3_t2i.txt",
            "FLUX 3 Image Klein edit": ROOT / "prompts" / "prompt_flux3_i2i.txt",
        })

    @staticmethod
    def completion(text):
        return Mock(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def test_local_upsampler_uses_the_official_prompt_and_template_length(self):
        graph = build_flux3_graph(model_choice="BF16", prompt="a fox")
        writer = next(n for n in graph.values() if n["class_type"] == "TextGenerate")["inputs"]
        self.assertTrue(writer["system_prompt"].startswith("You are the prompt upsampler of a text-to-image model."))
        self.assertEqual(writer["system_prompt"], upsampler_system())
        self.assertEqual(writer["max_length"], UPSAMPLER_MAX_LENGTH)

    @patch("openai.OpenAI")
    def test_text_to_image_rewrite_sends_the_target_canvas(self, client):
        client.return_value.chat.completions.create.return_value = self.completion("Full-color outdoor photograph, ...")
        caption, status = flux3_writer.enhance_flux3_prompt(
            "a fox in snow", "unused", "", [], 1920, 1072, "Lightning AI", "key",
            "openai/gpt-6.1-sol", runtime=self.runtime,
        )
        self.assertEqual(caption, "Full-color outdoor photograph, ...")
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-6.1-sol")
        self.assertEqual(request["messages"][0]["content"], upsampler_system())
        self.assertEqual(request["messages"][1]["content"][0]["text"], "TARGET CANVAS: 16:9\na fox in snow")

    @patch("openai.OpenAI")
    def test_edit_rewrite_uses_the_edit_prompt_with_images_in_order(self, client):
        client.return_value.chat.completions.create.return_value = self.completion("Change the sky to dusk ...")
        with tempfile.TemporaryDirectory() as root:
            images = []
            for name in ("a.png", "b.png"):
                Image.new("RGB", (8, 8)).save(Path(root) / name)
                images.append(str(Path(root) / name))
            flux3_writer.enhance_flux3_prompt(
                "make it dusk", "unused", "", images, 1024, 1024, "Lightning AI", "key", runtime=self.runtime,
            )
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertTrue(request["messages"][0]["content"].startswith("You are the prompt upsampler of an image-editing model."))
        content = request["messages"][1]["content"]
        self.assertEqual(content[0]["text"], "make it dusk")
        self.assertEqual([part["text"] for part in content if part["type"] == "text"][1:],
                         ["The next uploaded image is the first image.", "The next uploaded image is the second image."])

    def test_successful_rewrite_turns_local_expansion_off(self):
        from unittest.mock import Mock
        from h3_ui.engines.flux3 import Flux3Handlers
        handlers = Flux3Handlers(Mock())
        with patch.object(flux3_writer, "enhance_flux3_prompt",
                          return_value=("Full caption", "Enhanced FLUX 3 Image Klein prompt with x")):
            self.assertEqual(handlers.enhance_flux3_prompt("fox", [], 1024, 1024, "m", "")[2], False)
        with patch.object(flux3_writer, "enhance_flux3_prompt",
                          return_value=("fox", "Prompt enhancement failed: no key")):
            self.assertNotEqual(handlers.enhance_flux3_prompt("fox", [], 1024, 1024, "m", "")[2], False)


if __name__ == "__main__":
    unittest.main()
