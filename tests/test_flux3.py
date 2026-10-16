import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import (
    flux3_edit_dimensions,
    resolve_flux3_dimensions,
    validate_request,
)
from h3_app.generation.requests import Flux3Request
from h3_app import prompt_service
from h3_app.workflows.flux3 import UPSAMPLER_MAX_LENGTH, build_flux3_graph, upsampler_system

ROOT = Path(__file__).resolve().parents[1]


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
            "FLUX 3 Image Klein": ROOT / "prompt_flux3_t2i.txt",
            "FLUX 3 Image Klein edit": ROOT / "prompt_flux3_i2i.txt",
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
        caption, status = prompt_service.enhance_flux3_prompt(
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
            prompt_service.enhance_flux3_prompt(
                "make it dusk", "unused", "", images, 1024, 1024, "Lightning AI", "key", runtime=self.runtime,
            )
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertTrue(request["messages"][0]["content"].startswith("You are the prompt upsampler of an image-editing model."))
        content = request["messages"][1]["content"]
        self.assertEqual(content[0]["text"], "make it dusk")
        self.assertEqual([part["text"] for part in content if part["type"] == "text"][1:],
                         ["The next uploaded image is the first image.", "The next uploaded image is the second image."])

    def test_successful_rewrite_turns_local_expansion_off(self):
        from h3_ui import application
        with patch.object(application.prompt_service, "enhance_flux3_prompt",
                          return_value=("Full caption", "Enhanced FLUX 3 Image Klein prompt with x")):
            self.assertEqual(application.enhance_flux3_prompt("fox", [], 1024, 1024, "m", "")[2], False)
        with patch.object(application.prompt_service, "enhance_flux3_prompt",
                          return_value=("fox", "Prompt enhancement failed: no key")):
            self.assertNotEqual(application.enhance_flux3_prompt("fox", [], 1024, 1024, "m", "")[2], False)


if __name__ == "__main__":
    unittest.main()
