"""Tiled LTX routing and frame/audio preservation without GPU inference."""

from tests.app_targets import owner
import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.catalog import LTX25_TILED_UPSCALE
from h3_app.generation.finish_video import finish_video
from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.status import node_stage
from h3_app.workflows.upscale import build_upscale_graph, required_upscale_nodes
from h3_models import DEFAULT_LTX25_MODEL, LTX25_MODEL_CHOICES, MODEL_SPECS
from h3_ui import application as app


class TiledUpscaleTests(unittest.TestCase):
    def test_shared_canvas_streaming_guides_and_original_audio(self):
        for model in LTX25_MODEL_CHOICES:
            for size in ((3840, 2176), (2176, 3840), (1920, 1920)):
                with self.subTest(model=model, size=size):
                    graph, steps = build_upscale_graph(
                        option=LTX25_TILED_UPSCALE, source_video="source.mp4",
                        seed=42, models=Mock(), ltx25_model=model,
                        width=960, height=544, target_width=size[0],
                        target_height=size[1], fps=23.976,
                    )

                    def node(kind):
                        return next((key, n["inputs"]) for key, n in graph.items()
                                    if n["class_type"] == kind)

                    classes = {n["class_type"] for n in graph.values()}
                    self.assertEqual(classes, required_upscale_nodes(LTX25_TILED_UPSCALE))
                    self.assertEqual(steps, 8)
                    pad_id, pad = node("H3LTXPadVideoFrames")
                    components_id, _ = node("GetVideoComponents")
                    self.assertEqual(pad["images"], [components_id, 0])
                    _, latent = node("EmptyLTXVLatentVideo")
                    self.assertEqual((latent["width"], latent["height"]), size)
                    self.assertEqual(latent["length"], [pad_id, 1])
                    guide_id, guide = node("LTXAddVideoICLoRAGuide")
                    self.assertEqual(guide["image"], [pad_id, 0])
                    self.assertTrue(guide["use_streaming"])
                    self.assertFalse(guide["use_tiled_encode"])
                    self.assertEqual(guide["latent_downscale_factor"], 1)
                    self.assertEqual(guide["frame_idx"], 0)
                    _, sampler = node("LTXVTiledFusionSampler")
                    self.assertEqual(sampler["latents"], [guide_id, 2])
                    self.assertEqual(sampler["tile_frames"], guide["tile_frames"])
                    self.assertEqual(sampler["tile_frames"], 97)
                    self.assertEqual(sampler["canvas_device"], "auto")
                    self.assertEqual(sampler["overlap_frac"], 0.5)
                    self.assertEqual(max(sampler["tile_width"], sampler["tile_height"]), 1280)
                    self.assertEqual(node("KSamplerSelect")[1]["sampler_name"], "euler")
                    adapter_id, adapter = node("LTXICLoRALoaderModelOnly")
                    self.assertEqual(adapter["lora_name"], MODEL_SPECS["ltx25_refine_details"].local_name)
                    self.assertEqual(sampler["model"], [adapter_id, 0])
                    trim_id, trim = node("ImageFromBatch")
                    self.assertEqual(trim["length"], [pad_id, 2])
                    self.assertEqual(trim["batch_index"], 0)
                    _, video = node("CreateVideo")
                    self.assertEqual(video["images"], [trim_id, 0])
                    self.assertEqual(video["audio"], [components_id, 1])
                    self.assertEqual(video["fps"], 23.976)
                    self.assertNotIn("LTXVCropGuides", classes)
                    self.assertNotIn("LTXVConcatAVLatent", classes)
                    self.assertEqual(node_stage("LTXVTiledFusionSampler", classes),
                                     "Upscaling video with LTX tiled fusion")

    def test_media_and_h3_keep_splitting_as_explicit_fallback(self):
        source, result = Path("source.mp4"), Path("upscaled.mp4")
        metadata = VideoMetadata(23.976, 960, 544, 5, 120, True)
        for split in (False, True):
            for entry in ("media", "h3"):
                with self.subTest(entry=entry, split=split), ExitStack() as stack:
                    batch = UpscaleClipBatch(
                        ("clip1.mp4", "clip2.mp4") if split else ("source.mp4",)
                    )
                    values = {
                        "managed_video_path": source, "load_model_config": Mock(),
                        "object_info": required_upscale_nodes(LTX25_TILED_UPSCALE),
                        "ensure_ltx25_upscale_models": False,
                        "probe_video_metadata": metadata,
                        "prepare_upscale_clip_batch": batch,
                        "submit_prompt": "job", "poll_comfy_progress": [],
                        "wait_for_history": {}, "resolve_output": result,
                        "concat_upscaled_clips": result,
                        "cleanup_upscale_clip_batch": None,
                        "gallery_media_processed_result": "complete", "write_snapshot": None,
                    }
                    mocks = {name: stack.enter_context(patch.object(owner(app, name), name, return_value=value))
                             for name, value in values.items()}
                    build = stack.enter_context(patch.object(app, "build_upscale_graph", wraps=app.build_upscale_graph))
                    if entry == "media":
                        updates = list(app.MEDIA.postprocess_selected_gallery_media(
                            "Video", str(source), LTX25_TILED_UPSCALE, 42, "unused",
                            DEFAULT_LTX25_MODEL, "Subtle natural texture", False,
                            split, 5.0, "3840 × 3840", request=Mock(), progress=Mock(),
                        ))
                        self.assertEqual(updates[-1], "complete")
                    else:
                        request = SimpleNamespace(
                            media=SimpleNamespace(prompt="A named subject in a narrative scene"),
                            finishing=SimpleNamespace(
                                postprocess=LTX25_TILED_UPSCALE,
                                ltx25_model=DEFAULT_LTX25_MODEL, seedvr2_model="unused",
                                upscale_resolution="3840 × 3840", upscale_split_enabled=split,
                                upscale_split_seconds=5.0, upscale_force_offload=False,
                            ),
                        )
                        prepared = SimpleNamespace(models=Mock(), actual_seed=42,
                                                   resolved_width=960, resolved_height=544)
                        services = SimpleNamespace(
                            models=SimpleNamespace(ensure_ltx25_upscale_models=mocks["ensure_ltx25_upscale_models"]),
                            media=SimpleNamespace(**mocks),
                            workflows=SimpleNamespace(build_upscale_graph=build),
                            execution=SimpleNamespace(**mocks),
                        )
                        generator = finish_video(request, prepared, services, source,
                                                 "client", 0.0, Mock(), Mock())
                        while True:
                            try:
                                next(generator)
                            except StopIteration as finished:
                                self.assertEqual(finished.value, result)
                                break
                    self.assertEqual(build.call_count, 2 if split else 1)
                    for index, call in enumerate(build.call_args_list):
                        self.assertEqual(call.kwargs["option"], LTX25_TILED_UPSCALE)
                        self.assertEqual(call.kwargs["target_width"], 3840)
                        self.assertEqual(call.kwargs["seed"], 42 + index)
                        self.assertEqual(call.kwargs["prompt"], "" if entry == "h3" else "Subtle natural texture")
                    mocks["ensure_ltx25_upscale_models"].assert_called_once_with(
                        DEFAULT_LTX25_MODEL, option=LTX25_TILED_UPSCALE)
                    self.assertEqual(mocks["prepare_upscale_clip_batch"].call_args.kwargs["split_enabled"], split)
                    self.assertEqual(mocks["concat_upscaled_clips"].call_count, int(split))
                    mocks["cleanup_upscale_clip_batch"].assert_called_once()


class FramePaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the node without importing CUDA-dependent Comfy modules.
        class ImageBatch(np.ndarray):
            def expand(self, *shape):
                shape = tuple(old if new == -1 else new for old, new in zip(self.shape, shape))
                return np.broadcast_to(self, shape)

        cls.batch_type = ImageBatch
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "H3LTXPadVideoFrames")
        namespace = {"torch": SimpleNamespace(cat=np.concatenate)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        cls.node_type = namespace["H3LTXPadVideoFrames"]

    def test_arbitrary_counts_keep_all_frames_and_repeat_only_the_tail(self):
        for count in (1, 8, 9, 96, 97, 98, 120, 121, 124, 241):
            with self.subTest(count=count):
                images = np.arange(count * 12).reshape(count, 2, 2, 3).view(self.batch_type)
                padded, padded_count, source_count = self.node_type().pad(images)
                self.assertEqual(source_count, count)
                self.assertEqual(padded.shape[0], padded_count)
                self.assertEqual((padded_count - 1) % 8, 0)
                self.assertLess(padded_count - count, 8)
                np.testing.assert_array_equal(padded[:source_count], images)
                for frame in padded[source_count:]:
                    np.testing.assert_array_equal(frame, images[-1])

    def test_empty_or_non_video_inputs_are_rejected(self):
        for shape in ((0, 2, 2, 3), (2, 2, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.node_type().pad(np.empty(shape))


if __name__ == "__main__":
    unittest.main()
