"""SAM misses must leave the other concepts in a shot available."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from h3_ui.drama3 import effect_regions


class SoundStageEffectRegionsTests(unittest.TestCase):
    def test_missing_concept_does_not_invoke_comfy_image_saver(self):
        source = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        output_node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == effect_regions.MASK_OUTPUT_NODE)
        namespace = {}
        exec(compile(ast.Module(body=[output_node], type_ignores=[]), str(source), "exec"), namespace)
        saver = namespace[effect_regions.MASK_OUTPUT_NODE]()
        missing = Mock(shape=(0, 32, 48))
        found = Mock(shape=(2, 32, 48))
        save_image = Mock()
        image_list = [{"filename": "1.png"}, {"filename": "2.png"}]
        save_image.return_value.save_images.return_value = {"ui": {"images": image_list}, "result": (None,)}
        with patch.dict("sys.modules", {"nodes": SimpleNamespace(SaveImage=save_image)}):
            self.assertEqual(saver.save(missing, "miss"), {"ui": {"images": []}})
            save_image.assert_not_called()
            missing.unsqueeze.assert_not_called()
            self.assertEqual(saver.save(found, "match"), {"ui": {"images": image_list}})
            found.unsqueeze.assert_called_once_with(-1)
            found.unsqueeze.return_value.expand.assert_called_once_with(-1, -1, -1, 3)
            save_image.return_value.save_images.assert_called_once_with(
                found.unsqueeze.return_value.expand.return_value, "match")

    def test_segment_keeps_a_match_after_a_miss_and_caches_no_matches(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            still = root / "shot.jpg"
            Image.new("RGB", (48, 32)).save(still)

            def submit(graph, client):
                outputs = {}
                for node, spec in graph.items():
                    if spec["class_type"] != effect_regions.MASK_OUTPUT_NODE:
                        continue
                    inputs = spec["inputs"]
                    mask = graph[inputs["masks"][0]]
                    self.assertEqual(mask["class_type"], "SAM3_Detect")
                    self.assertTrue(mask["inputs"]["individual_masks"])
                    text = graph[mask["inputs"]["conditioning"][0]]["inputs"]["text"]
                    outputs[node] = {"images": []}
                    if text == "window":
                        path = root / "h3" / "soundstage_masks" / client / "mask.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pixels = Image.new("L", (48, 32))
                        ImageDraw.Draw(pixels).rectangle((0, 0, 23, 31), fill=255)
                        pixels.save(path)
                        outputs[node]["images"].append({"filename": path.name})
                self.assertEqual(len(outputs), sum(spec["class_type"] == "SAM3_Detect" for spec in graph.values()))
                return {"outputs": outputs}

            services = SimpleNamespace(
                media=SimpleNamespace(stage_file=Mock(return_value="shot.jpg")),
                execution=SimpleNamespace(submit_prompt=submit, poll_comfy_progress=Mock(return_value=[]),
                                          wait_for_history=lambda prompt: prompt),
            )
            runtime = SimpleNamespace(output_dir=root)
            labels = ["person", "window"]
            regions = effect_regions.segment(still, labels, services, runtime)
            self.assertEqual([(item["label"], item["box"]) for item in regions], [("window", [0, 0, 0.5, 1])])
            self.assertEqual(effect_regions.read_regions(root, still.name, labels), regions)

            self.assertEqual(effect_regions.segment(still, ["person"], services, runtime), [])
            self.assertEqual(effect_regions.read_regions(root, still.name, ["person"]), [])
