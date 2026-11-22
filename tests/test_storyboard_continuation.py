"""Storyboard continuation nodes: tail geometry, masks and latent files, on CPU tensors."""

import ast
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
NAMES = {
    "H3_FRAMES_PER_STEP", "H3_VIDEO_FPS", "H3_AUDIO_HZ", "H3_CONTINUATION_FRAMES",
    "h3_latent_frame_boundaries", "h3_continuation_plan", "h3_av_streams",
    "h3_apply_continuation", "H3ContinuationPrefix", "h3_storyboard_latent_path",
    "H3SaveAVLatent", "H3LoadAVLatent",
    "H3_TAIL_FRAMES", "h3_tail_latent_steps", "h3_tail_pixels", "h3_tail_wave", "H3EncodeAVTail", "H3LipSyncMask",
    "H3LowerFaceMask", "H3LipSyncPrepareCrop", "H3LipSyncBlend",
}


class NestedTensor:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = tuple(tensors)


def load_nodes(output_dir):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES
        or isinstance(node, ast.Assign) and any(getattr(t, "id", None) in NAMES for t in node.targets)
    ]

    def save_path(prefix, root):
        subfolder, name = os.path.split(prefix)
        folder = os.path.join(root, subfolder)
        os.makedirs(folder, exist_ok=True)
        return folder, name, 1, subfolder, prefix

    def common_upscale(samples, width, height, method, crop):
        return torch.nn.functional.interpolate(samples, size=(height, width))

    namespace = {
        "torch": torch, "logging": logging, "os": os, "F": torch.nn.functional,
        "comfy": SimpleNamespace(nested_tensor=SimpleNamespace(NestedTensor=NestedTensor),
                                 utils=SimpleNamespace(common_upscale=common_upscale)),
        "folder_paths": SimpleNamespace(
            get_output_directory=lambda: str(output_dir), get_save_image_path=save_path,
        ),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def av_latent(steps, ticks, *, fill=0.0, height=4, width=6):
    video = torch.full((1, 24, steps, height, width), fill)
    audio = torch.full((1, 32, 2, ticks), fill)
    return {"samples": NestedTensor((video, audio))}


class ContinuationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.nodes = load_nodes(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_frame_grid_and_context_sizes(self):
        boundaries = self.nodes["h3_latent_frame_boundaries"]
        self.assertEqual(boundaries(37)[-1], 141 - 17)  # 5 s → 124 frames
        plan = self.nodes["h3_continuation_plan"]
        for requested, frames, steps in ((22, 22, 7), (39, 39, 12), (56, 56, 17), (40, 39, 12)):
            result = plan(37, 37, requested)
            self.assertEqual((result["frames"], result["context_steps"]), (frames, steps))
            self.assertEqual(result["start_step"] % 5, 0)
            self.assertEqual(result["end_step"] % 5, 2)

    def test_tail_is_the_latest_aligned_window_and_audio_matches_it(self):
        result = self.nodes["h3_continuation_plan"](37, 37, 39)
        boundaries = self.nodes["h3_latent_frame_boundaries"](37)
        self.assertEqual(result["end_step"], 37 - (37 - 2) % 5)
        self.assertEqual(boundaries[result["end_step"]] - boundaries[result["start_step"]], 39)
        self.assertEqual(result["audio_steps"], 65)
        self.assertEqual(result["audio_end"], round(boundaries[result["end_step"]] * 40 / 24))

    def test_lip_sync_mask_covers_whole_video_tokens_and_keeps_audio_fixed(self):
        latent = av_latent(2, 8, height=4, width=4)
        frames = torch.zeros((5, 64, 64, 3))
        mouth = torch.zeros((5, 64, 64))
        mouth[2, 42, 18] = 0.1
        mouth[3, 42, 18] = 1.0
        mouth[3, 43, 18] = 0.25
        apply = self.nodes["H3LipSyncMask"]().apply
        masked, composite = apply(latent, frames, mouth)
        video, audio = masked["noise_mask"].tensors
        self.assertIs(masked["samples"], latent["samples"])
        self.assertFalse(torch.any(audio))
        self.assertFalse(torch.any(video[:, :, 0]))
        expected = torch.zeros((4, 4))
        expected[2:, :2] = 1
        self.assertTrue(torch.equal(video[0, 0, 1], expected))
        self.assertFalse(torch.any(composite[2]))
        self.assertTrue(torch.equal(composite[3], mouth[3]))
        with self.assertRaisesRegex(ValueError, "matching crop"):
            apply(latent, frames, mouth[:-1])
        with self.assertRaisesRegex(ValueError, "no editable pixels"):
            apply(latent, frames, torch.zeros_like(mouth))

    def test_lower_face_mask_targets_mouth_and_respects_active_interval_and_absent_frames(self):
        crops = torch.zeros((6, 128, 128, 3))
        transform = {
            "face_rect": [(32.0, 20.0, 64.0, 80.0)] * 6,
            "detected": [True, True, False, True, True, True],
            "absent": [False, False, False, True, False, False],
            "weights": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
        build = self.nodes["H3LowerFaceMask"]().build
        denoise, composite, _ = build(crops, transform, active_start=1, active_end=5)
        self.assertEqual(tuple(denoise.shape), (6, 128, 128))
        self.assertEqual(tuple(composite.shape), (6, 128, 128))
        # Outside active interval: frame 0 and frame 5 composite must be zero.
        self.assertFalse(torch.any(composite[0]))
        self.assertFalse(torch.any(composite[5]))
        # Undetected (frame 2) and absent (frame 3) must be zero in both masks.
        self.assertFalse(torch.any(denoise[2]))
        self.assertFalse(torch.any(composite[2]))
        self.assertFalse(torch.any(denoise[3]))
        self.assertFalse(torch.any(composite[3]))
        # Active detected frame 1: upper face (forehead y=25) is untouched; mouth (y=82, x=64) is masked.
        self.assertEqual(float(composite[1, 25, 64]), 0.0)
        self.assertAlmostEqual(float(composite[1, 82, 64]), 0.5)
        self.assertEqual(float(denoise[1, 82, 64]), 1.0)
        # Denoise mask is strictly wider than composite mask.
        self.assertGreater(int((denoise[1] > 0.5).sum()), int((composite[1] > 0.5).sum()))
        _, faded, _ = build(crops, {"face_rect": transform["face_rect"]}, active_start=0, active_end=6)
        torch.testing.assert_close(faded[:, 82, 64], torch.tensor([1/3, 2/3, 1., 1., 2/3, 1/3]))
        _, single, _ = build(crops, {"face_rect": transform["face_rect"]}, active_start=2, active_end=3)
        self.assertEqual(float(single[2, 82, 64]), 1.0)
        _, stitch_mask, spatial = build(crops, transform, active_start=1, active_end=5, separate_opacity=True)
        self.assertEqual(float(spatial[1, 82, 64]), 1.0)
        self.assertEqual(float(stitch_mask[1, 82, 64]), 0.5)
        self.assertFalse(torch.any(spatial[0]))
        # All absent raises clearly.
        with self.assertRaisesRegex(ValueError, "No detected face frames"):
            build(crops, {**transform, "absent": [True] * 6})

    def test_lip_sync_prepare_crop_neutralizes_core_and_preserves_context(self):
        crops = torch.full((2, 96, 96, 3), 0.2)
        crops[:, :, ::2, :] = 0.8
        mask = torch.zeros((2, 96, 96))
        mask[1, 24:72, 24:72] = 1.0
        (prepared,) = self.nodes["H3LipSyncPrepareCrop"]().prepare(crops, mask, strength=0.85)
        self.assertTrue(torch.equal(prepared[0], crops[0]))
        self.assertTrue(torch.equal(prepared[1, :24, :], crops[1, :24, :]))
        self.assertTrue(torch.equal(prepared[1, 72:, :], crops[1, 72:, :]))
        core_before_std = float(crops[1, 36:60, 36:60].std())
        core_after_std = float(prepared[1, 36:60, 36:60].std())
        self.assertLess(core_after_std, core_before_std * 0.35)

    def test_lip_sync_blend_skin_ring_color_match_pyramid_and_outside_preservation(self):
        from unittest.mock import patch
        import builtins

        src = torch.zeros((2, 95, 97, 3), dtype=torch.float32)
        src[..., 0] = 0.75
        src[..., 1] = 0.55
        src[..., 2] = 0.45
        # Add subtle skin texture so high-pass sharpening path executes non-trivially.
        src[:, ::2, ::2, :] += 0.03
        mask = torch.zeros((2, 95, 97), dtype=torch.float32)
        mask[1, 18:78, 18:78] = 0.5
        mask[1, 24:72, 24:72] = 1.0

        gen = src.clone()
        # Shift generated crop cooler/darker across the masked area and add a dark inner mouth cavity.
        gen[1, :, :, 0] = (src[1, :, :, 0] - 0.18).clamp(0.0, 1.0)
        gen[1, :, :, 1] = (src[1, :, :, 1] + 0.06).clamp(0.0, 1.0)
        gen[1, :, :, 2] = (src[1, :, :, 2] + 0.12).clamp(0.0, 1.0)
        gen[1, 40:56, 40:56, :] = 0.02

        blend_fn = self.nodes["H3LipSyncBlend"]().blend
        (out,) = blend_fn(src, gen, mask, color_match=1.0, sharpen=0.35, pyramid_levels=3)
        self.assertEqual(tuple(out.shape), (2, 95, 97, 3))
        self.assertTrue(torch.equal(out[0], src[0]))
        self.assertTrue(torch.equal(out[1, :18, :], src[1, :18, :]))
        # Outer skin inside the mask stays close to source skin tone despite the dark center cavity.
        ring_mean = out[1, 26:32, 26:32, :].mean(dim=(0, 1))
        self.assertAlmostEqual(float(ring_mean[0]), float(src[1, 26:32, 26:32, 0].mean()), delta=0.05)
        self.assertAlmostEqual(float(ring_mean[2]), float(src[1, 26:32, 26:32, 2].mean()), delta=0.05)
        # Inner mouth cavity remains dark (not washed out).
        self.assertLess(float(out[1, 46:50, 46:50, :].mean()), 0.25)

        # Also exercise the pure-PyTorch fallback when cv2 is unavailable.
        real_import = builtins.__import__

        def no_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("cv2 disabled")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_cv2):
            (out_torch,) = blend_fn(src, gen, mask, color_match=1.0, sharpen=0.35, pyramid_levels=3)
        self.assertEqual(tuple(out_torch.shape), (2, 95, 97, 3))
        self.assertTrue(torch.equal(out_torch[1, :18, :], src[1, :18, :]))
        ring_mean_torch = out_torch[1, 26:32, 26:32, :].mean(dim=(0, 1))
        self.assertAlmostEqual(float(ring_mean_torch[0]), float(src[1, 26:32, 26:32, 0].mean()), delta=0.05)

    def test_short_source_or_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            self.nodes["h3_continuation_plan"](1, 37, 39)
        with self.assertRaisesRegex(ValueError, "at least 5"):
            self.nodes["h3_continuation_plan"](37, 1, 39)

    def test_prefix_copies_tail_and_masks_only_the_head(self):
        source = av_latent(37, 207, fill=1.0)
        source_video, source_audio = source["samples"].tensors
        source_video[:, :, 30:32] = 7.0
        target = {**av_latent(37, 207), "noise_mask": "stale low-resolution mask", "other": 1}
        (out,) = self.nodes["H3ContinuationPrefix"]().apply(target, source, 39)
        video, audio = out["samples"].tensors
        video_mask, audio_mask = out["noise_mask"].tensors
        self.assertEqual(out["h3_prefix_frames"], 39)
        self.assertEqual(out["other"], 1)
        self.assertTrue(torch.equal(video[:, :, :12], source_video[:, :, 25:37]))
        self.assertTrue(torch.all(video[:, :, 12:] == 0))
        self.assertTrue(torch.all(video_mask[:, :, :12] == 0) and torch.all(video_mask[:, :, 12:] == 1))
        self.assertTrue(torch.all(audio_mask[..., :65] == 0) and torch.all(audio_mask[..., 65:] == 1))
        self.assertTrue(torch.all(audio[..., :65] == 1) and torch.all(audio[..., 65:] == 0))
        self.assertEqual(video_mask.shape, video.shape)
        self.assertEqual(audio_mask.shape, audio.shape)
        self.assertEqual(target["noise_mask"], "stale low-resolution mask")

    def test_mismatched_size_or_batch_fails_clearly(self):
        prefix = self.nodes["H3ContinuationPrefix"]()
        with self.assertRaisesRegex(ValueError, "matching video latent sizes"):
            prefix.apply(av_latent(37, 207), av_latent(37, 207, height=8), 39)
        video = torch.zeros((2, 24, 37, 4, 6))
        batch = {"samples": NestedTensor((video, torch.zeros((2, 32, 2, 207))))}
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            prefix.apply(batch, av_latent(37, 207), 39)

    def test_saved_latent_round_trips_with_prefix_metadata(self):
        from safetensors import safe_open

        latent = {**av_latent(37, 207, fill=2.0), "h3_prefix_frames": 39}
        result = self.nodes["H3SaveAVLatent"]().save(latent, "h3/.storyboard/sb/take")
        saved = result["ui"]["h3_latents"][0]
        relative = os.path.join(saved["subfolder"], saved["filename"])
        with safe_open(os.path.join(self.temp.name, relative), "pt") as handle:
            self.assertEqual(handle.metadata()["h3_prefix_frames"], "39")
        (loaded,) = self.nodes["H3LoadAVLatent"]().load(relative)
        video, audio = loaded["samples"].tensors
        self.assertTrue(torch.all(video == 2.0) and audio.shape == (1, 32, 2, 207))

    def test_loading_outside_the_output_directory_is_refused(self):
        load = self.nodes["H3LoadAVLatent"]()
        for path in ("../escape.safetensors", "h3/clip.mp4"):
            with self.assertRaisesRegex(ValueError, "inside the output directory"):
                load.load(path)



class EncodeTailTests(unittest.TestCase):
    """A clip from Media, encoded so a scene can continue it, with stand-in VAEs."""

    class VideoVAE:
        def encode(self, pixels):
            frames, height, width = pixels.shape[0], pixels.shape[1], pixels.shape[2]
            steps = 2 + 5 * ((frames - 5) // 17)
            # Each latent step carries the mean of its first pixel frame, so the test can tell which frames went in.
            first, frame = [], 0
            for step in range(steps):
                first.append(frame)
                frame += (1, 4, 4, 4, 4)[step % 5]
            values = pixels[first].mean(dim=(1, 2, 3))
            return values.view(1, 1, steps, 1, 1).expand(1, 24, steps, height // 16, width // 16).clone()

    class AudioVAE:
        def __init__(self):
            self.waves = []

        def encode(self, wave):
            self.waves.append(wave)
            ticks = wave.shape[1] // 800
            return torch.ones(1, 32, 2, ticks + 3)  # Longer than asked: the node cuts it.

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.nodes = load_nodes(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_tail_steps_end_on_a_latent_group_that_continuation_can_take(self):
        steps = self.nodes["h3_tail_latent_steps"](self.nodes["H3_TAIL_FRAMES"])
        self.assertEqual(steps, 22)
        self.assertEqual(self.nodes["h3_latent_frame_boundaries"](steps)[-1], 73)
        for overlap in self.nodes["H3_CONTINUATION_FRAMES"]:
            plan = self.nodes["h3_continuation_plan"](steps, 37, overlap)
            self.assertEqual((plan["end_step"], plan["frames"]), (22, overlap))
            self.assertLessEqual(plan["audio_end"], round(73 * 40 / 24))
        with self.assertRaisesRegex(ValueError, "5 \\+ 17k"):
            self.nodes["h3_tail_latent_steps"](72)

    def test_encodes_the_last_frames_at_the_canvas_with_the_matching_sound(self):
        frames = torch.arange(100, dtype=torch.float32).view(100, 1, 1, 1).expand(100, 36, 64, 3).clone()
        rate = 32000  # H3's own rate, so no resampling (which needs ComfyUI).
        wave = torch.arange(100 * rate // 24, dtype=torch.float32).view(1, 1, -1)
        audio_vae = self.AudioVAE()
        (latent,) = self.nodes["H3EncodeAVTail"]().encode(
            frames, self.VideoVAE(), audio_vae, 96, 48, 73, {"waveform": wave, "sample_rate": rate})
        video, sound = latent["samples"].tensors
        self.assertEqual(tuple(video.shape), (1, 24, 22, 3, 6))
        self.assertEqual(float(video[0, 0, 0, 0, 0]), 27.0)  # The tail starts at frame 100 - 73.
        self.assertEqual(float(video[0, 0, -1, 0, 0]), 27.0 + 69)  # The last step's first frame.
        self.assertEqual(tuple(sound.shape), (1, 32, 2, 122))
        (sent,) = audio_vae.waves
        self.assertEqual(tuple(sent.shape), (1, 122 * 800, 2))

    def test_silence_without_a_soundtrack_and_short_clips_refused(self):
        frames = torch.zeros(80, 32, 32, 3)
        audio_vae = self.AudioVAE()
        self.nodes["H3EncodeAVTail"]().encode(frames, self.VideoVAE(), audio_vae, 32, 32, 73)
        self.assertTrue(torch.all(audio_vae.waves[0] == 0))
        with self.assertRaisesRegex(ValueError, "last 73"):
            self.nodes["H3EncodeAVTail"]().encode(frames[:60], self.VideoVAE(), audio_vae, 32, 32, 73)

    def test_wave_starts_where_the_tail_does(self):
        rate = 32000
        wave = torch.arange(100 * rate // 24, dtype=torch.float32).repeat(2, 1).unsqueeze(0)
        cut = self.nodes["h3_tail_wave"]({"waveform": wave, "sample_rate": rate}, 100, 73, 122)
        self.assertEqual(float(cut[0, 0]), 27 * rate / 24)
        self.assertEqual(tuple(cut.shape), (2, 122 * 800))


if __name__ == "__main__":
    unittest.main()
