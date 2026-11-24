"""SAM misses must leave the other concepts in a shot available."""

import ast
import json
import os
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

    def test_concepts_skip_locations_and_unconditional_screen(self):
        row = {"subjects": ["elena", "brass_console", "watchroom"], "prompt": "Inside @watchroom, @elena checks @brass_console.", "beat": "Inspection"}
        subjects = {
            "elena": {"kind": "characters", "anchor": "Elena, a woman in her thirties, navy jacket"},
            "brass_console": {"kind": "props", "anchor": "heavy riveted brass console with analog dial"},
            "watchroom": {"kind": "locations", "anchor": "circular stone watchroom"},
        }
        labels = effect_regions.concepts(row, subjects)
        self.assertIn("brass console", labels)
        self.assertNotIn("circular stone watchroom", labels)
        self.assertNotIn("screen", labels)

    def test_merges_multiple_windows_and_skips_full_frame_noise(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            still = root / "shot.jpg"
            Image.new("RGB", (48, 32)).save(still)

            def submit(graph, client):
                outputs = {}
                for node, spec in graph.items():
                    if spec["class_type"] != effect_regions.MASK_OUTPUT_NODE:
                        continue
                    mask = graph[spec["inputs"]["masks"][0]]
                    text = graph[mask["inputs"]["conditioning"][0]]["inputs"]["text"]
                    outputs[node] = {"images": []}
                    folder = root / "h3" / "soundstage_masks" / client
                    folder.mkdir(parents=True, exist_ok=True)
                    if text == "window":
                        for i, rect in enumerate(((0, 0, 9, 15), (38, 0, 47, 15))):
                            p = folder / f"w{i}.png"
                            img = Image.new("L", (48, 32))
                            ImageDraw.Draw(img).rectangle(rect, fill=255)
                            img.save(p)
                            outputs[node]["images"].append({"filename": p.name})
                    elif text == "door":
                        p = folder / "full.png"
                        Image.new("L", (48, 32), 255).save(p)
                        outputs[node]["images"].append({"filename": p.name})
                return {"outputs": outputs}

            services = SimpleNamespace(
                media=SimpleNamespace(stage_file=Mock(return_value="shot.jpg")),
                execution=SimpleNamespace(submit_prompt=submit, poll_comfy_progress=Mock(return_value=[]),
                                          wait_for_history=lambda prompt: prompt),
            )
            regions = effect_regions.segment(still, ["window", "door"], services, SimpleNamespace(output_dir=root))
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0]["label"], "window")
            self.assertEqual(regions[0]["box"], [0.0, 0.0, 1.0, 0.5])

    def test_read_regions_falls_back_to_legacy_manifest_and_prefers_exact_match(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / effect_regions.MASK_FOLDER
            masks.mkdir()
            (masks / "shot.jpg.aaaa111122223333.region_1.png").write_bytes(b"old-mask")
            (masks / "shot.jpg.bbbb111122223333.region_1.png").write_bytes(b"new-mask")
            (masks / "other.jpg.cccc111122223333.region_1.png").write_bytes(b"other-mask")

            older_regions = [{"id": "region_1", "label": "old_lamp", "mask": "shot.jpg.aaaa111122223333.region_1.png",
                              "box": [0.0, 0.0, 0.4, 0.4]}]
            newer_regions = [{"id": "region_1", "label": "window", "mask": "shot.jpg.bbbb111122223333.region_1.png",
                              "box": [0.1, 0.1, 0.9, 0.9]}]
            broken_regions = [{"id": "region_1", "label": "door", "mask": "shot.jpg.missing.region_1.png",
                               "box": [0.0, 0.0, 1.0, 1.0]}]
            other_regions = [{"id": "region_1", "label": "person", "mask": "other.jpg.cccc111122223333.region_1.png",
                              "box": [0.2, 0.2, 0.8, 0.8]}]

            older = masks / "shot.jpg.0123456789abcdef.json"
            newer = masks / "shot.jpg.fedcba9876543210.json"
            broken = masks / "shot.jpg.1111222233334444.json"
            other = masks / "other.jpg.9999888877776666.json"
            older.write_text(json.dumps(older_regions), encoding="utf-8")
            newer.write_text(json.dumps(newer_regions), encoding="utf-8")
            broken.write_text(json.dumps(broken_regions), encoding="utf-8")
            other.write_text(json.dumps(other_regions), encoding="utf-8")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            os.utime(broken, (3000, 3000))
            os.utime(other, (4000, 4000))

            labels = ["person", "window", "door", "lamp"]
            self.assertEqual(effect_regions.read_regions(root, "shot.jpg", labels), newer_regions)

            exact = effect_regions.manifest_path(root, "shot.jpg", labels)
            exact.write_text("[]", encoding="utf-8")
            os.utime(exact, (500, 500))
            self.assertEqual(effect_regions.read_regions(root, "shot.jpg", labels), [])

    def test_rebind_stills_copies_matching_effect_masks(self):
        from h3_app.drama3.visuals import FPS, Shot
        from h3_ui.drama3 import visuals

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_audio = root / "old.mp3"
            new_audio = root / "new.mp3"
            old_audio.write_bytes(b"old")
            new_audio.write_bytes(b"new")
            still = root / "s1.png"
            Image.new("RGB", (32, 18), "navy").save(still)
            old_shots = [Shot(0, FPS, False, "Opening", "Beat 1", id="shot_1")]
            visuals.keep_stills(old_audio, root, old_shots, [still], {}, total_frames=FPS)
            old_folder = visuals.stills_dir(old_audio, root)
            image_name = visuals.read_stills(old_audio, root)["shots"][0]["image"]

            mask_dir = old_folder / effect_regions.MASK_FOLDER
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_name = f"{image_name}.{'a' * 32}.region_1.png"
            (mask_dir / mask_name).write_bytes(b"mask-png")
            regions = [{"id": "region_1", "label": "person", "mask": mask_name, "box": [0.0, 0.0, 0.5, 0.5]}]
            effect_regions.manifest_path(old_folder, image_name, ["person"]).write_text(
                json.dumps(regions), encoding="utf-8"
            )
            (mask_dir / "other.jpg.1234567890abcdef.json").write_text("[]", encoding="utf-8")

            new_shots = [Shot(0, 2 * FPS, False, "Opening", "Beat 1", id="shot_1")]
            visuals.rebind_stills(new_audio, root, new_shots, previous_audio=old_audio, previous_shots=old_shots)
            new_folder = visuals.stills_dir(new_audio, root)
            self.assertEqual(effect_regions.read_regions(new_folder, image_name, ["person"]), regions)
            self.assertFalse((new_folder / effect_regions.MASK_FOLDER / "other.jpg.1234567890abcdef.json").exists())

    def test_concepts_extract_cjk_objects_and_map_non_ascii_characters_to_english_sam_labels(self):
        row = {
            "subjects": ["lin", "chen", "lantern_prop"],
            "prompt": "@lin 站在木窗户旁，手提纸灯笼望向火焰。",
            "beat": "雨夜窗边",
        }
        subjects = {
            "lin": {"kind": "characters", "anchor": "林晚，二十多岁的年轻女子，身穿深蓝风衣"},
            "chen": {"kind": "characters", "anchor": "陈默，中年男子，灰色大衣"},
            "lantern_prop": {"kind": "props", "anchor": "旧纸灯笼"},
        }
        labels = effect_regions.concepts(row, subjects)
        self.assertIn("woman", labels)
        self.assertIn("man", labels)
        self.assertIn("window", labels)
        self.assertIn("lamp", labels)
        self.assertIn("fire", labels)
        self.assertTrue(all(label.isascii() for label in labels), labels)

    def test_segment_new_regions_hybrid_selection_dedup_and_preview_render(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            still = root / "shot.jpg"
            Image.new("RGB", (80, 48), (60, 80, 100)).save(still)
            mask_dir = root / effect_regions.MASK_FOLDER
            mask_dir.mkdir(parents=True, exist_ok=True)

            # Existing region_1 covers left person [0, 0, 24, 47].
            existing_mask_name = f"shot.jpg.{'a' * 32}.region_1.png"
            existing_img = Image.new("RGBA", (80, 48), (0, 0, 0, 0))
            ImageDraw.Draw(existing_img).rectangle((0, 0, 23, 47), fill=(255, 255, 255, 255))
            existing_img.save(mask_dir / existing_mask_name)
            existing_regions = [
                {"id": "region_1", "label": "person", "mask": existing_mask_name, "box": [0.0, 0.0, 0.3, 1.0]}
            ]

            submitted = []

            def submit(graph, client):
                submitted.append(graph)
                outputs = {}
                out_folder = root / "h3" / "soundstage_masks" / client
                out_folder.mkdir(parents=True, exist_ok=True)
                for node, spec in graph.items():
                    if spec["class_type"] != effect_regions.MASK_OUTPUT_NODE:
                        continue
                    sam = graph[spec["inputs"]["masks"][0]]
                    outputs[node] = {"images": []}
                    if "conditioning" in sam["inputs"]:
                        text = graph[sam["inputs"]["conditioning"][0]]["inputs"]["text"]
                        if text == "brass lantern":
                            # Two lanterns: upper-middle (28..40, 6..18) and lower-right (56..74, 26..42).
                            for idx, rect in enumerate(((28, 6, 40, 18), (56, 26, 74, 42))):
                                p = out_folder / f"lantern_{idx}.png"
                                m = Image.new("L", (80, 48), 0)
                                ImageDraw.Draw(m).rectangle(rect, fill=255)
                                m.save(p)
                                outputs[node]["images"].append({"filename": p.name})
                        elif text == "woman":
                            # Same silhouette as existing region_1 -> should deduplicate to region_1!
                            p = out_folder / "dup_person.png"
                            m = Image.new("L", (80, 48), 0)
                            ImageDraw.Draw(m).rectangle((0, 0, 23, 47), fill=255)
                            m.save(p)
                            outputs[node]["images"].append({"filename": p.name})
                    elif "positive_coords" in sam["inputs"]:
                        coords = json.loads(sam["inputs"]["positive_coords"])
                        # Fallback point segmentation for "stone archway" where text missed.
                        if coords and coords[0]["x"] == 40 and coords[0]["y"] == 24:
                            p = out_folder / "arch_pt.png"
                            m = Image.new("L", (80, 48), 0)
                            ImageDraw.Draw(m).rectangle((32, 8, 48, 40), fill=255)
                            m.save(p)
                            outputs[node]["images"].append({"filename": p.name})
                return {"outputs": outputs}

            services = SimpleNamespace(
                media=SimpleNamespace(stage_file=Mock(return_value="shot.jpg")),
                execution=SimpleNamespace(
                    submit_prompt=submit,
                    poll_comfy_progress=Mock(return_value=[]),
                    wait_for_history=lambda prompt: prompt,
                ),
            )
            requests = [
                # Picks the lower-right lantern matching include/box instead of upper-middle one.
                {
                    "id": "new_1",
                    "label": "table lantern",
                    "prompt": "brass lantern",
                    "box": [0.68, 0.50, 0.95, 0.90],
                    "include": [[0.81, 0.71]],
                    "exclude": [],
                    "merge": False,
                },
                # Text misses "stone archway", falls back to point branch in the same ComfyUI submission.
                {
                    "id": "new_2",
                    "label": "archway",
                    "prompt": "stone archway",
                    "box": [0.40, 0.16, 0.60, 0.84],
                    "include": [[0.50, 0.50]],
                    "exclude": [],
                    "merge": False,
                },
                # Deduplicates against existing region_1.
                {
                    "id": "new_3",
                    "label": "hero",
                    "prompt": "woman",
                    "include": [[0.15, 0.50]],
                    "exclude": [],
                    "merge": False,
                },
                # Neither text nor point matches -> reported in skipped.
                {
                    "id": "new_4",
                    "label": "distant bird",
                    "prompt": "tiny bird",
                    "include": [[0.95, 0.05]],
                    "exclude": [],
                    "merge": False,
                },
            ]

            created, id_map, skipped = effect_regions.segment_new_regions(
                still, requests, existing_regions, services, SimpleNamespace(output_dir=root)
            )
            self.assertEqual(len(submitted), 1)
            self.assertEqual(skipped, ["distant bird"])
            self.assertEqual(id_map, {"new_1": "custom_1", "new_2": "custom_2", "new_3": "region_1"})
            self.assertEqual(set(created), {"custom_1", "custom_2"})
            self.assertEqual(created["custom_1"]["label"], "table lantern")
            self.assertGreater(created["custom_1"]["box"][0], 0.65)
            self.assertTrue((mask_dir / created["custom_1"]["mask"]).is_file())
            self.assertTrue((mask_dir / created["custom_2"]["mask"]).is_file())

            # Verify annotated segment preview image renders all regions cleanly.
            all_regions = [*existing_regions, *created.values()]
            vis_out = root / "vis" / "segments.png"
            rendered = effect_regions.render_regions_preview(still, mask_dir, all_regions, vis_out, max_side=64)
            self.assertEqual(rendered, vis_out)
            with Image.open(vis_out) as vis_img:
                self.assertEqual(vis_img.size, (64, 38))
                self.assertEqual(vis_img.mode, "RGB")
