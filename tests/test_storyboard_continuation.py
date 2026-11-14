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
    "H3_TAIL_FRAMES", "h3_tail_latent_steps", "h3_tail_pixels", "h3_tail_wave", "H3EncodeAVTail",
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
