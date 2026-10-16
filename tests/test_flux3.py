import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from PIL import Image

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import (
    flux3_edit_dimensions,
    resolve_flux3_dimensions,
    validate_request,
)
from h3_app.generation.requests import Flux3Request
from h3_app.workflows.flux3 import build_flux3_graph


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


if __name__ == "__main__":
    unittest.main()
