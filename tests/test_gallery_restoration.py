"""Gallery restoration routing, adapter downloads, and Comfy graph contracts."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import model_service
from h3_app.catalog import (
    GENERATION_POSTPROCESS_OPTIONS,
    LTX25_DECOMPRESSION,
    LTX25_POSTPROCESS_MODELS,
    LTX25_RESTORATION_OPTIONS,
)
from h3_app.errors import H3Error
from h3_app.workflows.upscale import build_upscale_graph, required_upscale_nodes
from h3_models import (
    DEFAULT_LTX25_MODEL,
    LAZY_POSTPROCESS_MODEL_KEYS,
    LTX25_MODEL_CHOICES,
    MODEL_SPECS,
    PRELOAD_MODEL_KEYS,
)
import h3_ui.application as app


class GalleryRestorationTests(unittest.TestCase):
    def test_restoration_uses_selected_adapter_and_same_resolution_reference(self):
        for option in LTX25_RESTORATION_OPTIONS:
            for model in LTX25_MODEL_CHOICES:
                for width, height in ((960, 544), (544, 960), (1920, 1080)):
                    with self.subTest(option=option, model=model, size=(width, height)):
                        graph, steps = build_upscale_graph(
                            option=option,
                            source_video="source.mp4",
                            seed=42,
                            models=Mock(),
                            ltx25_model=model,
                            prompt="a rabbit.",
                            width=width,
                            height=height,
                            target_width=3840,
                            target_height=2160,
                            fps=23.976,
                        )

                        def node(kind):
                            return next(
                                (key, n["inputs"])
                                for key, n in graph.items()
                                if n["class_type"] == kind
                            )

                        self.assertEqual(steps, 8)
                        self.assertEqual(
                            set(n["class_type"] for n in graph.values()),
                            required_upscale_nodes(option),
                        )
                        adapter_id, adapter = node("LTXICLoRALoaderModelOnly")
                        key = LTX25_POSTPROCESS_MODELS[option]
                        self.assertEqual(
                            adapter["lora_name"], MODEL_SPECS[key].local_name
                        )
                        self.assertEqual(adapter["strength_model"], 1.0)
                        _, guide = node("LTXAddVideoICLoRAGuide")
                        self.assertEqual(guide["latent_downscale_factor"], 1)
                        self.assertTrue(guide["use_tiled_encode"])
                        _, latent = node("EmptyLTXVLatentVideo")
                        reference = graph[guide["image"][0]]["inputs"]
                        self.assertEqual(
                            (reference["width"], reference["height"]),
                            (latent["width"], latent["height"]),
                        )
                        self.assertEqual(latent["length"], [node("GetImageSize")[0], 2])
                        _, video = node("CreateVideo")
                        self.assertEqual(
                            video["audio"], [node("GetVideoComponents")[0], 1]
                        )
                        self.assertEqual(video["fps"], 23.976)
                        if height == 1080:
                            final_images = graph[video["images"][0]]["inputs"]
                            self.assertEqual(
                                (final_images["width"], final_images["height"]),
                                (width, height),
                            )
                        self.assertEqual(node("CFGGuider")[1]["model"], [adapter_id, 0])
                        text = node("CLIPTextEncode")[1]["text"]
                        self.assertIn("a rabbit", text)
                        self.assertIn(
                            "ENHANCE QUALITY"
                            if option == LTX25_DECOMPRESSION
                            else "DEBLUR",
                            text,
                        )
                        self.assertIn(
                            key.removeprefix("ltx25_"),
                            node("SaveVideo")[1]["filename_prefix"],
                        )

    def test_source_dimensions_are_required(self):
        for option in LTX25_RESTORATION_OPTIONS:
            with self.subTest(option=option), self.assertRaises(H3Error):
                build_upscale_graph(
                    option=option, source_video="source.mp4", seed=42, models=Mock()
                )

    def test_downloads_only_selected_adapter_and_uses_cached_models(self):
        runtime = SimpleNamespace(
            models_config=Path("models/config.json"), comfy_dir=Path("comfy")
        )
        for option in LTX25_RESTORATION_OPTIONS:
            key = LTX25_POSTPROCESS_MODELS[option]
            self.assertIn(key, LAZY_POSTPROCESS_MODEL_KEYS)
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            self.assertNotIn(option, GENERATION_POSTPROCESS_OPTIONS)
            for stale in (False, True):
                with (
                    self.subTest(option=option, stale=stale),
                    patch.object(
                        model_service, "ensure_ltx25_models", return_value=False
                    ),
                    patch.object(
                        model_service,
                        "stale_model_keys",
                        return_value=[key] if stale else [],
                    ),
                    patch.object(model_service, "sync_models") as sync,
                    patch.object(
                        model_service, "resolve_hf_token", return_value="test-token"
                    ),
                    patch.object(
                        model_service, "model_file_is_ready", return_value=True
                    ),
                ):
                    self.assertEqual(
                        model_service.ensure_ltx25_upscale_models(
                            option=option, runtime=runtime
                        ),
                        stale,
                    )
                    if stale:
                        self.assertEqual(sync.call_args.kwargs["model_keys"], (key,))
                        self.assertEqual(sync.call_args.kwargs["token"], "test-token")
                    else:
                        sync.assert_not_called()

    def test_gallery_dispatches_restoration_and_split_clips(self):
        source = Path("source.mp4")
        result = Path("restored.mp4")
        metadata = SimpleNamespace(
            width=1920, height=1080, fps=24.0, duration=10.0, frame_count=241
        )
        for option in LTX25_RESTORATION_OPTIONS:
            for split in (False, True):
                with self.subTest(option=option, split=split), ExitStack() as stack:
                    batch = SimpleNamespace(
                        sources=["clip1.mp4", "clip2.mp4"] if split else ["source.mp4"],
                        temporary_inputs=split,
                    )
                    mocks = {}
                    returns = {
                        "managed_video_path": source,
                        "load_model_config": Mock(),
                        "object_info": required_upscale_nodes(option),
                        "ensure_ltx25_upscale_models": False,
                        "probe_video_metadata": metadata,
                        "prepare_upscale_clip_batch": batch,
                        "submit_prompt": "job",
                        "poll_comfy_progress": [],
                        "wait_for_history": {},
                        "resolve_output": result,
                        "concat_upscaled_clips": result,
                        "cleanup_upscale_clip_batch": None,
                        "unload_comfy_models": None,
                        "gallery_processed_result": "complete",
                        "postprocess_video": None,
                        "upscale_target_dimensions": None,
                    }
                    for name, value in returns.items():
                        mocks[name] = stack.enter_context(
                            patch.object(app, name, return_value=value)
                        )
                    build = stack.enter_context(
                        patch.object(
                            app, "build_upscale_graph", wraps=app.build_upscale_graph
                        )
                    )
                    updates = list(
                        app.postprocess_selected_gallery_video(
                            str(source),
                            option,
                            42,
                            "unused",
                            DEFAULT_LTX25_MODEL,
                            "a rabbit",
                            True,
                            split,
                            5.0,
                            "unused resolution",
                            request=Mock(),
                            progress=Mock(),
                        )
                    )
                    self.assertEqual(updates[-1], "complete", updates)
                    mocks["postprocess_video"].assert_not_called()
                    mocks["upscale_target_dimensions"].assert_not_called()
                    mocks["ensure_ltx25_upscale_models"].assert_called_once_with(
                        DEFAULT_LTX25_MODEL, option=option
                    )
                    mocks["unload_comfy_models"].assert_called_once()
                    self.assertEqual(build.call_count, 2 if split else 1)
                    for i, call in enumerate(build.call_args_list):
                        self.assertEqual(call.kwargs["option"], option)
                        self.assertEqual(call.kwargs["target_width"], 1920)
                        self.assertEqual(call.kwargs["target_height"], 1080)
                        self.assertEqual(call.kwargs["seed"], 42 + i)
                    self.assertEqual(
                        mocks["concat_upscaled_clips"].call_count, int(split)
                    )
                    mocks["cleanup_upscale_clip_batch"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
