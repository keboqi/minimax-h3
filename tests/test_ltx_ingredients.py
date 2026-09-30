"""Ingredients reference-sheet generation graph and model contract."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from h3_app.workflows.ltx import build_ltx25_graph, required_ltx25_nodes
from h3_models import MODEL_SPECS
import h3_ui.application as app


class LtxIngredientsTests(unittest.TestCase):
    def test_multiple_uploads_become_distinct_sheet_panels(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / "red.png", Path(directory) / "blue.png"]
            for path, color in zip(paths, ("red", "blue")):
                Image.new("RGB", (64, 64), color).save(path)

            captured = []

            def stage(path, category):
                self.assertEqual(category, "ltx25_references")
                with Image.open(path) as sheet:
                    captured.extend((sheet.size, sheet.getpixel((192, 224)), sheet.getpixel((576, 224))))
                return "ltx25_references/sheet.png"

            with patch.object(app, "stage_file", side_effect=stage):
                app.build_ltx25_graph(
                    prompt="Reference sheet: red and blue panels. Generated video: motion.",
                    negative_prompt="",
                    first_image=None,
                    width=768,
                    height=448,
                    duration=5,
                    fps=24,
                    seed=1,
                    cfg=1,
                    sampler_name="euler_ancestral",
                    image_strength=0.7,
                    reference_images=tuple(map(str, paths)),
                )
            self.assertEqual(captured, [(768, 448), (255, 0, 0), (0, 0, 255)])

    def test_reference_sheet_uses_ingredients_adapter_and_crops_control_frames(self):
        graph = build_ltx25_graph(
            prompt="Reference sheet: rabbit and garden. Generated video: rabbit hops.",
            negative_prompt="",
            first_image=None,
            width=768,
            height=448,
            duration=5,
            fps=24,
            seed=1,
            cfg=1,
            sampler_name="euler_ancestral",
            image_strength=0.7,
            reference_sheet="ltx25_references/sheet.png",
        )

        def node(kind):
            return next((key, value["inputs"]) for key, value in graph.items() if value["class_type"] == kind)

        model_id, adapter = node("LTXICLoRALoaderModelOnly")
        self.assertEqual(adapter["lora_name"], MODEL_SPECS["ltx25_iclora_ingredients"].local_name)
        repeated_id, repeated = node("RepeatImageBatch")
        self.assertEqual(repeated["amount"], 121)
        self.assertEqual(graph[repeated["image"][0]]["inputs"]["image"], "ltx25_references/sheet.png")
        _, guide = node("LTXAddVideoICLoRAGuide")
        self.assertEqual(guide["image"], [repeated_id, 0])
        self.assertEqual(guide["latent_downscale_factor"], [model_id, 1])
        cropped_id, _ = node("LTXVCropGuides")
        self.assertEqual(node("VAEDecodeTiled")[1]["samples"], [cropped_id, 2])
        self.assertTrue(required_ltx25_nodes(reference_images=True) <= {n["class_type"] for n in graph.values()})


if __name__ == "__main__":
    unittest.main()
