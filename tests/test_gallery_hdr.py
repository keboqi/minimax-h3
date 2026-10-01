"""HDR gallery routing, color graph, conditioning and lazy resource selection."""

import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gradio_app as app
from h3_app import model_service
from h3_app.catalog import (
    GENERATION_POSTPROCESS_OPTIONS, LTX25_SDR_TO_HDR, POSTPROCESS_OPTIONS,
)
from h3_app.workflows.upscale import build_upscale_graph, required_upscale_nodes
from h3_models import LAZY_POSTPROCESS_MODEL_KEYS, LTX25_MODEL_CHOICES, MODEL_SPECS, PRELOAD_MODEL_KEYS


class Tensor(np.ndarray):
    """Only CPU array operations used by the conditioning loader."""

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis).view(Tensor)

    def new_zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype).view(Tensor)

    def to(self, other):
        return self.astype(other.dtype).view(Tensor)


class GalleryHdrTests(unittest.TestCase):
    def test_hdr_graph_uses_acescct_fixed_conditioning_full_vae_and_hlg(self):
        self.assertIn(LTX25_SDR_TO_HDR, POSTPROCESS_OPTIONS)
        self.assertNotIn(LTX25_SDR_TO_HDR, GENERATION_POSTPROCESS_OPTIONS)
        for choice in LTX25_MODEL_CHOICES:
            graph, steps = build_upscale_graph(
                option=LTX25_SDR_TO_HDR, source_video="source.mp4", seed=42,
                models=Mock(), ltx25_model=choice, width=1920, height=1080,
                target_width=3840, target_height=2160, fps=23.976,
                output_stamp="fixture", output_nonce="nonce",
            )
            def node(kind):
                return next((key, n["inputs"]) for key, n in graph.items() if n["class_type"] == kind)
            self.assertEqual(steps, 8)
            self.assertEqual({n["class_type"] for n in graph.values()}, required_upscale_nodes(LTX25_SDR_TO_HDR))
            working, _ = node("LTXVSDRToHDRWorkingSpace")
            self.assertEqual(node("ImageScale")[1]["image"], [working, 0])
            self.assertEqual(node("VAELoader")[1]["vae_name"], MODEL_SPECS["ltx25_video_vae_full"].local_name)
            forced, _ = node("LTXVVAEForceFloat32")
            self.assertEqual(node("LTXAddVideoICLoRAGuide")[1]["vae"], [forced, 0])
            self.assertEqual(node("VAEDecodeTiled")[1]["vae"], [forced, 0])
            self.assertEqual(node("LTXAddVideoICLoRAGuide")[1]["latent_downscale_factor"], 1)
            self.assertEqual(node("KSamplerSelect")[1]["sampler_name"], "euler")
            hdr, decoded = node("LTXVHDRDecodePostprocess")
            self.assertEqual(decoded["transfer"], "acescct")
            scale = graph[decoded["image"][0]]["inputs"]
            self.assertEqual((scale["width"], scale["height"]), (1920, 1080))
            output = node("LTXVSaveHLG")[1]
            self.assertEqual(output["hdr_linear"], [hdr, 1])
            self.assertEqual(output["linear_primaries"], "acescg")
            self.assertEqual(output["frame_rate"], 23.976)
            self.assertEqual(output["audio"], [node("GetVideoComponents")[0], 1])

    def test_hdr_download_has_no_text_encoder_or_audio_vae(self):
        runtime = SimpleNamespace(models_config=Path("models/config.json"), comfy_dir=Path("comfy"))
        for choice, base in LTX25_MODEL_CHOICES.items():
            keys = (base, "ltx25_video_vae_full", "ltx25_sdr_to_hdr", "ltx25_sdr_to_hdr_conditioning")
            for key in keys[2:]:
                self.assertIn(key, LAZY_POSTPROCESS_MODEL_KEYS)
                self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            with (
                patch.object(model_service, "ensure_ltx25_models") as normal,
                patch.object(model_service, "stale_model_keys", return_value=list(keys)),
                patch.object(model_service, "sync_models") as sync,
                patch.object(model_service, "resolve_hf_token", return_value="fixture-token"),
                patch.object(model_service, "model_file_is_ready", return_value=True),
            ):
                self.assertTrue(model_service.ensure_ltx25_upscale_models(choice, option=LTX25_SDR_TO_HDR, runtime=runtime))
                self.assertEqual(sync.call_args.kwargs["model_keys"], keys)
                normal.assert_not_called()
            with (
                patch.object(model_service, "stale_model_keys", return_value=[]),
                patch.object(model_service, "sync_models") as sync,
            ):
                self.assertFalse(model_service.ensure_ltx25_upscale_models(choice, option=LTX25_SDR_TO_HDR, runtime=runtime))
                sync.assert_not_called()

    def test_gallery_hdr_dispatch_preserves_dimensions_and_split_flow(self):
        for split in (False, True):
            with self.subTest(split=split), ExitStack() as stack:
                result = Path("hdr.mp4")
                batch = SimpleNamespace(sources=["clip1.mp4", "clip2.mp4"] if split else ["source.mp4"], temporary_inputs=split)
                values = {
                    "managed_video_path": Path("source.mp4"), "load_model_config": Mock(),
                    "object_info": required_upscale_nodes(LTX25_SDR_TO_HDR),
                    "ensure_ltx25_upscale_models": False,
                    "probe_video_metadata": SimpleNamespace(width=1920, height=1080, fps=24.0, duration=10.0, frame_count=241),
                    "prepare_upscale_clip_batch": batch, "submit_prompt": "job", "poll_comfy_progress": [],
                    "wait_for_history": {}, "resolve_output": result, "concat_upscaled_clips": result,
                    "cleanup_upscale_clip_batch": None, "unload_comfy_models": None,
                    "gallery_processed_result": "complete", "upscale_target_dimensions": None,
                }
                mocks = {name: stack.enter_context(patch.object(app, name, return_value=value)) for name, value in values.items()}
                build = stack.enter_context(patch.object(app, "build_upscale_graph", wraps=app.build_upscale_graph))
                updates = list(app.postprocess_selected_gallery_video(
                    "source.mp4", LTX25_SDR_TO_HDR, 42, "unused", "BF16", "ignored",
                    False, split, 5.0, "unused", request=Mock(), progress=Mock(),
                ))
                self.assertEqual(updates[-1], "complete", updates)
                mocks["upscale_target_dimensions"].assert_not_called()
                self.assertEqual(build.call_count, 2 if split else 1)
                self.assertEqual(build.call_args.kwargs["target_height"], 1080)
                self.assertEqual(mocks["concat_upscaled_clips"].call_count, int(split))

    def test_video_only_pipeline_embeddings_get_neutral_audio_padding(self):
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "H3LTXHDRConditioning")
        state = {"video_context": np.ones((2, 4096), dtype=np.float32).view(Tensor)}
        namespace = {
            "torch": SimpleNamespace(cat=lambda parts, dim: np.concatenate(parts, axis=dim).view(Tensor)),
            "folder_paths": SimpleNamespace(get_full_path_or_raise=lambda *a: "fixture"),
            "comfy": SimpleNamespace(utils=SimpleNamespace(load_torch_file=lambda *a, **k: state)),
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        loader = namespace["H3LTXHDRConditioning"]()
        result = loader.load("fixture")[0][0][0]
        self.assertEqual(tuple(result.shape), (1, 2, 6144))
        self.assertTrue(np.array_equal(result[..., :4096], state["video_context"].unsqueeze(0)))
        self.assertEqual(np.count_nonzero(result[..., 4096:]), 0)
        state["audio_context"] = np.ones((2, 2048), dtype=np.float32).view(Tensor)
        self.assertTrue(np.all(loader.load("fixture")[0][0][0] == 1))
        state["video_context"] = np.ones((2, 5000), dtype=np.float32).view(Tensor)
        with self.assertRaises(ValueError):
            loader.load("fixture")
