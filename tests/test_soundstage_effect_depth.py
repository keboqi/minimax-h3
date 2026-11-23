"""Unit tests for Depth Pro 2 artifact packing, cache management, and SAM region enrichment."""

import ast
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from h3_app.errors import H3Error
from h3_models import DEFAULT_SOUNDSTAGE_DEPTH_MODEL
from h3_ui.drama3 import effect_depth, visuals


def _load_encoder():
    source = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_encode_depthpro2_artifacts"
    )
    namespace = {"math": math}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_encode_depthpro2_artifacts"]


ENCODE_DEPTH = _load_encoder()


class SoundStageEffectDepthTests(unittest.TestCase):
    def test_encode_depthpro2_artifacts_roundtrip_sky_and_anchors(self):
        height, width = 72, 128
        depth = np.full((height, width), 25.0, dtype=np.float32)
        depth[0:14, :] = np.inf  # top sky band
        depth[18:54, 16:48] = 1.5  # near foreground
        depth[20:52, 48:84] = 4.0  # subject midrange

        rgb = np.full((height, width, 3), 64, dtype=np.uint8)
        # Bright warm flame spot at lower-right (u ~ 0.80, v ~ 0.72, z = 4.0m)
        depth[46:58, 94:112] = 3.2
        rgb[46:58, 94:112] = [255, 175, 30]

        png_img, meta = ENCODE_DEPTH(depth, rgb, 140.0, DEFAULT_SOUNDSTAGE_DEPTH_MODEL)
        self.assertEqual(png_img.mode, "RGB")
        self.assertEqual(png_img.size, (width, height))
        self.assertTrue(meta["is_metric"])
        self.assertEqual(meta["preset"], DEFAULT_SOUNDSTAGE_DEPTH_MODEL)
        self.assertEqual(meta["f_px"], 140.0)
        self.assertGreater(meta["auto_strength"], 0.4)

        arr = np.asarray(png_img, dtype=np.uint16)
        q = (arr[..., 0] << 8) | arr[..., 1]
        inv_min = float(meta["inv_min"])
        inv_max = float(meta["inv_max"])
        inv_rec = inv_min + (q.astype(np.float32) / 65535.0) * (inv_max - inv_min)

        for y, x, expected_z in ((30, 28, 1.5), (30, 64, 4.0), (36, 96, 25.0)):
            z_rec = 1.0 / max(float(inv_rec[y, x]), 1e-4)
            rel_err = abs(z_rec - expected_z) / expected_z
            self.assertLess(rel_err, 0.015, (expected_z, z_rec, rel_err))

        # Infinite depth top row encodes sky mask = 255 in B channel.
        self.assertEqual(int(arr[4, 64, 2]), 255)
        self.assertLess(meta["sky_dir"]["v"], 0.45)
        self.assertGreater(meta["warm_anchor"]["u"], 0.65)
        self.assertGreater(meta["warm_anchor"]["strength"], 0.15)
        self.assertAlmostEqual(meta["rack_start_z"], 1.5, delta=0.8)

    def test_encode_flat_depth_sets_zero_auto_strength(self):
        depth = np.full((36, 64), 5.0, dtype=np.float32)
        rgb = np.full((36, 64, 3), 120, dtype=np.uint8)
        _, meta = ENCODE_DEPTH(depth, rgb, None, "depthpro2-b")
        self.assertEqual(meta["auto_strength"], 0.0)

    def test_cache_lookup_and_style_scaling(self):
        self.assertEqual(effect_depth.normalize_depth_model("unknown"), DEFAULT_SOUNDSTAGE_DEPTH_MODEL)
        self.assertEqual(effect_depth.normalize_depth_model("depthpro2-l"), "depthpro2-l")

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            png_b, json_b = effect_depth.artifact_paths(folder, "shot_01.jpg", "depthpro2-b")
            png_b.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 18), (64, 0, 0)).save(png_b)
            json_b.write_text(json.dumps({"inv_min": 0.0, "inv_max": 0.5, "auto_strength": 0.9}), encoding="utf-8")

            self.assertIsNone(effect_depth.read_depth(folder, "shot_01.jpg", "depthpro2-s", exact=True))
            fallback = effect_depth.read_depth(folder, "shot_01.jpg", "depthpro2-s", exact=False)
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback["png"], png_b.name)

            meta = fallback["meta"]
            self.assertEqual(effect_depth.effective_meta(meta, "Pixel art")["depth_strength"], 0.0)
            self.assertEqual(effect_depth.effective_meta(meta, "Pixel art")["parallax_scale"], 0.0)
            self.assertEqual(effect_depth.effective_meta(meta, "Anime")["parallax_scale"], 0.65)
            self.assertEqual(effect_depth.effective_meta(meta, "Ink wash")["parallax_scale"], 1.0)

    def test_region_depth_stats_and_ensure_stills_depth(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            still_a = folder / "shot_a.jpg"
            still_b = folder / "shot_b.jpg"
            Image.new("RGB", (64, 36), "gray").save(still_a)
            Image.new("RGB", (64, 36), "blue").save(still_b)

            depth = np.full((36, 64), 24.0, dtype=np.float32)
            depth[:, :32] = 1.5
            rgb = np.full((36, 64, 3), 100, dtype=np.uint8)
            png_img, meta = ENCODE_DEPTH(depth, rgb, 90.0, DEFAULT_SOUNDSTAGE_DEPTH_MODEL)

            png_a, json_a = effect_depth.artifact_paths(folder, still_a.name, DEFAULT_SOUNDSTAGE_DEPTH_MODEL)
            png_a.parent.mkdir(parents=True, exist_ok=True)
            png_img.save(png_a)
            json_a.write_text(json.dumps({**meta, "png": png_a.name}), encoding="utf-8")

            mask_dir = folder / ".effect-masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            fg_mask = Image.new("RGBA", (64, 36), (0, 0, 0, 0))
            ImageDraw.Draw(fg_mask).rectangle((2, 2, 28, 32), fill=(255, 255, 255, 255))
            fg_mask.save(mask_dir / "fg.png")

            bg_mask = Image.new("RGBA", (64, 36), (0, 0, 0, 0))
            ImageDraw.Draw(bg_mask).rectangle((36, 2, 60, 32), fill=(255, 255, 255, 255))
            bg_mask.save(mask_dir / "bg.png")

            depth_entry = effect_depth.read_depth(folder, still_a.name, DEFAULT_SOUNDSTAGE_DEPTH_MODEL, exact=True)
            fg_stats = effect_depth.region_depth_stats(folder, depth_entry, "fg.png")
            bg_stats = effect_depth.region_depth_stats(folder, depth_entry, "bg.png")
            self.assertEqual(fg_stats["depth_band"], "foreground")
            self.assertAlmostEqual(fg_stats["depth_m"], 1.5, delta=0.1)
            self.assertEqual(bg_stats["depth_band"], "background")
            self.assertAlmostEqual(bg_stats["depth_m"], 24.0, delta=0.6)

            seen = []
            with patch.object(effect_depth, "prepare"), patch.object(
                effect_depth, "estimate_still", side_effect=lambda p, m, s, r: seen.append((Path(p).name, m))
            ):
                done = effect_depth.ensure_stills_depth(
                    folder, [still_a.name, still_b.name, still_b.name], "depthpro2-s", Mock(), SimpleNamespace(output_dir=folder)
                )
            self.assertEqual(done, 1)
            self.assertEqual(seen, [("shot_b.jpg", "depthpro2-s")])

            with patch.object(effect_depth, "prepare", side_effect=H3Error("missing node")):
                self.assertEqual(
                    effect_depth.ensure_stills_depth(
                        folder, [still_b.name], "depthpro2-s", Mock(), SimpleNamespace(output_dir=folder)
                    ),
                    0,
                )

    def test_rebind_stills_preserves_depth_sidecars(self):
        from h3_app.drama3.visuals import Shot

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_mix = root / "mix_a.mp3"
            new_mix = root / "mix_b.mp3"
            src_img = root / "img.png"
            Image.new("RGB", (32, 18), "red").save(src_img)
            shots = [Shot(0, 48, False, "Prompt 1", "Beat 1", id="shot_1")]
            visuals.keep_stills(old_mix, root, shots, [src_img], {"model": "qwen"}, total_frames=48)
            record = visuals.read_stills(old_mix, root)
            image_name = record["shots"][0]["image"]

            old_folder = visuals.stills_dir(old_mix, root)
            png_path, json_path = effect_depth.artifact_paths(old_folder, image_name, "depthpro2-s")
            png_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 18), (40, 0, 0)).save(png_path)
            json_path.write_text(json.dumps({"inv_min": 0.0, "inv_max": 0.5, "png": png_path.name}), encoding="utf-8")

            visuals.rebind_stills(new_mix, root, shots, previous_audio=old_mix)
            new_folder = visuals.stills_dir(new_mix, root)
            loaded = effect_depth.read_depth(new_folder, image_name, "depthpro2-s", exact=True)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["png"], png_path.name)


if __name__ == "__main__":
    unittest.main()
