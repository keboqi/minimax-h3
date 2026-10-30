"""Regenerate adapter isolation, video preservation and lazy entry-point routing."""

import ast
import inspect
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app import catalog, model_service
from h3_app.generation.finish_video import finish_video
from h3_app.generation.preparation import _prepare_latent_upscale
from h3_app.media_types import UpscaleClipBatch, VideoMetadata
from h3_app.policy import resolve_h3_latent_upscale_method, resolve_h3_split_upscale_config
from h3_app.workflows import h3, upscale
from h3_models import MODEL_SPECS, PRELOAD_MODEL_KEYS
from h3_ui import application as app
from tests.app_targets import owner
from h3_app.model_types import ModelConfig, ModelProfile
from h3_app.errors import H3Error


class RegenerateGraphTests(unittest.TestCase):
    def models(self):
        return ModelConfig(
            {"speed": ModelProfile("Test", "base.safetensors", "ref.safetensors")},
            "speed", "text.safetensors", "video.safetensors", "audio.safetensors",
        )

    def test_existing_video_keeps_timing_audio_and_source_frame_count(self):
        for option, key in catalog.H3_REGENERATE_MODELS.items():
            graph, steps = upscale.build_upscale_graph(
                option=option, source_video="source.mp4", seed=42, models=self.models(),
                width=960, height=544, target_width=2560, target_height=1440, fps=23.976,
            )
            by_type = lambda kind: [(n, v["inputs"]) for n, v in graph.items() if v["class_type"] == kind]
            self.assertEqual({v["class_type"] for v in graph.values()}, upscale.required_upscale_nodes(option))
            self.assertEqual(steps, 3)
            self.assertEqual(by_type("ManualSigmas")[0][1]["sigmas"], catalog.H3_REGENERATE_SIGMAS)
            self.assertEqual(by_type("KSamplerSelect")[0][1]["sampler_name"], "euler")
            self.assertEqual([v["lora_name"] for _, v in by_type("LoraLoaderModelOnly")],
                             [MODEL_SPECS["turbo_lora"].local_name, MODEL_SPECS[key].local_name])
            encode, _ = by_type("H3RegenerateEncodeVideo")[0]
            native = by_type(catalog.H3_LATENT_UPSCALER_NODE)[0][1]
            self.assertEqual(native["samples"], [encode, 0])
            self.assertEqual((native["width"], native["height"]), (2560, 1408))
            self.assertEqual(graph[native["model"][0]]["inputs"]["precision"], "bf16")
            self.assertEqual(by_type("ImageFromBatch")[0][1]["length"], [encode, 2])
            self.assertEqual(by_type("MiniMaxH3ImageToVideo")[0][1]["length"], [encode, 1])
            video = by_type("CreateVideo")[0][1]
            self.assertEqual(video["fps"], 23.976)
            self.assertEqual(video["audio"], [by_type("GetVideoComponents")[0][0], 1])
            final = graph[video["images"][0]]["inputs"]
            self.assertEqual((final["width"], final["height"]), (2560, 1440))

    def test_native_refinement_uses_its_own_schedule_and_adapter_stack(self):
        for build in (app.build_fl2va_graph, app.build_ref2va_graph):
            for option, key in catalog.H3_REGENERATE_MODELS.items():
                args = {name: app.UI_DEFAULTS.get(name, 0) for name, param in inspect.signature(build).parameters.items()
                        if param.default is inspect.Parameter.empty}
                args.update(
                    prompt="A bird flies", width=864, height=480, duration=5, seed=7,
                    steps=2, scheduler="simple", model_name="base.safetensors", models=self.models(),
                    turbo_variant=catalog.LIGHTX2V_8STEP_TURBO, turbo_lora_name="draft.safetensors",
                    use_sol=False, cache_mode="Off", latent_upscale_model_name="upscaler.pth",
                    refinement_lora_name=MODEL_SPECS[key].local_name, refinement_variant=option,
                    available_nodes=(h3.turbo_required_nodes(catalog.LIGHTX2V_8STEP_TURBO, "draft.safetensors")
                                     | h3.turbo_required_nodes(option)),
                )
                if build is app.build_fl2va_graph:
                    args.update(first_image=None, last_image=None)
                else:
                    args.update(reference_images=[], reference_videos=[], reference_audios=[])
                graph = build(**args)
                samplers = [v["inputs"] for v in graph.values() if v["class_type"] == "SamplerCustomAdvanced"]
                initial, refine = samplers
                def chain(sample):
                    model = graph[sample["guider"][0]]["inputs"]["model"]
                    nodes = []
                    while model:
                        node = graph[model[0]]
                        nodes.append(node)
                        model = node["inputs"].get("model")
                    return nodes
                self.assertEqual([n["inputs"]["lora_name"] for n in chain(initial) if "lora_name" in n["inputs"]],
                                 ["draft.safetensors"])
                self.assertEqual([n["inputs"]["lora_name"] for n in chain(refine) if "lora_name" in n["inputs"]],
                                 [MODEL_SPECS[key].local_name, MODEL_SPECS["turbo_lora"].local_name])
                self.assertEqual(graph[refine["sigmas"][0]]["inputs"]["sigmas"], catalog.H3_REGENERATE_SIGMAS)
                self.assertEqual(refine["sigmas"][1], 0)
                self.assertEqual(graph[refine["latent_image"][0]]["class_type"], "H3RegeneratePrepareLatent")
                self.assertEqual(graph[refine["sampler"][0]]["inputs"]["sampler_name"], "euler")

    def test_modes_do_not_activate_split_and_weights_stay_lazy(self):
        for option, key in catalog.H3_REGENERATE_MODELS.items():
            self.assertEqual(resolve_h3_latent_upscale_method(option), option)
            self.assertIsNone(resolve_h3_split_upscale_config(
                option, tile_width=0, tile_height=0, overlap_ratio=0, fade_ratio=0,
                chunk_frames=0, temporal_overlap_frames=0, seam_denoise=0, seam_polish="off"))
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            self.assertIn(option, catalog.GENERATION_POSTPROCESS_OPTIONS)
            self.assertIn(option, catalog.POSTPROCESS_OPTIONS)

    def test_media_convrot_model_uses_chunking_and_checks_its_dependency(self):
        models = self.models()
        models.profiles["speed"].fl2va = "base_int8_convrot.safetensors"
        graph, _ = upscale.build_upscale_graph(
            option=catalog.H3_REGENERATE_16, source_video="source.mp4", seed=42,
            models=models, width=960, height=544,
        )
        classes = {node["class_type"] for node in graph.values()}
        self.assertIn(catalog.CHUNK_FEED_FORWARD_NODE, classes)
        self.assertEqual(classes, upscale.required_upscale_nodes(catalog.H3_REGENERATE_16, models))

    def test_native_preparation_accepts_short_draft_schedule_and_downloads_only_selected_rank(self):
        for option, key in catalog.H3_REGENERATE_MODELS.items():
            finishing = SimpleNamespace(**app.UI_DEFAULTS)
            finishing.latent_upscale_method = option
            finishing.latent_upscale_refine_steps = 6
            finishing.latent_upscale_refine_lora = catalog.DMAD_4STEP_LORA
            effects = Mock()
            effects.model_file_is_ready.return_value = True
            generator = _prepare_latent_upscale(
                SimpleNamespace(finishing=finishing, media=SimpleNamespace(mode="Reference media")),
                SimpleNamespace(models=effects), SimpleNamespace(comfy_dir=Path("ComfyUI")), self.models(),
                width=864, height=480, effective_steps=2, started=0, progress=Mock())
            while True:
                try:
                    next(generator)
                except StopIteration as finished:
                    prepared = finished.value
                    break
            effects.ensure_model_keys.assert_called_once_with((key, "turbo_lora"))
            effects.ensure_turbo_lora.assert_not_called()
            self.assertEqual(finishing.latent_upscale_refine_steps, 3)
            self.assertEqual(prepared.refinement_variant, option)
            self.assertIsNone(prepared.split_config)

    def test_media_provisioning_keeps_other_rank_lazy(self):
        for option, key in catalog.H3_REGENERATE_MODELS.items():
            models = self.models()
            models.text_encoder = MODEL_SPECS["text_encoder"].local_name
            with ExitStack() as stack:
                effects = {name: stack.enter_context(patch.object(model_service, name, return_value=False))
                           for name in ("ensure_profile_model", "ensure_base_video_vae", "ensure_audio_vae", "ensure_models")}
                self.assertFalse(model_service.ensure_h3_regenerate_models(models, option, runtime=Mock()))
                downloaded = [k for call in effects["ensure_models"].call_args_list for k in call.args[0]]
                self.assertIn(key, downloaded)
                self.assertEqual(set(downloaded) & set(catalog.H3_REGENERATE_MODELS.values()), {key})
                effects["ensure_profile_model"].assert_called_once()

    def test_missing_custom_encoder_fails_before_downloading_models(self):
        models = self.models()
        runtime = SimpleNamespace(comfy_dir=Path("ComfyUI"))
        with (patch.object(model_service, "model_file_is_ready", return_value=False),
              patch.object(model_service, "ensure_profile_model") as download):
            with self.assertRaisesRegex(H3Error, "Configured H3 text encoder is missing"):
                model_service.ensure_h3_regenerate_models(models, catalog.H3_REGENERATE_16, runtime=runtime)
            download.assert_not_called()

    def test_both_video_entry_points_use_h3_provisioning(self):
        models = self.models()
        source, result = Path("source.mp4"), Path("upscaled.mp4")
        metadata = VideoMetadata(23.976, 960, 544, 5, 120, True)
        for option in catalog.H3_REGENERATE_MODELS:
            for entry in ("media", "h3"):
                with self.subTest(option=option, entry=entry), ExitStack() as stack:
                    values = {
                        "managed_video_path": source, "load_model_config": models,
                        "object_info": upscale.required_upscale_nodes(option),
                        "ensure_h3_regenerate_models": False, "ensure_ltx25_upscale_models": False,
                        "probe_video_metadata": metadata,
                        "prepare_upscale_clip_batch": UpscaleClipBatch(("source.mp4",)),
                        "submit_prompt": "job", "poll_comfy_progress": [], "wait_for_history": {},
                        "resolve_output": result, "cleanup_upscale_clip_batch": None,
                        "gallery_media_processed_result": "complete",
                    }
                    mocks = {name: stack.enter_context(patch.object(owner(app, name), name, return_value=value))
                             for name, value in values.items()}
                    build = stack.enter_context(patch.object(app, "build_upscale_graph", wraps=app.build_upscale_graph))
                    if entry == "media":
                        updates = list(app.MEDIA.postprocess_selected_gallery_media(
                            "Video", str(source), option, 42, "unused", "unused", "A bird flies",
                            False, False, 5.0, "2560 × 2560", request=Mock(), progress=Mock()))
                        self.assertEqual(updates[-1], "complete")
                    else:
                        request = SimpleNamespace(media=SimpleNamespace(prompt="A bird flies"), finishing=SimpleNamespace(
                            postprocess=option, ltx25_model="unused", seedvr2_model="unused",
                            upscale_resolution="2560 × 2560", upscale_split_enabled=False,
                            upscale_split_seconds=5, upscale_force_offload=False))
                        services = SimpleNamespace(
                            models=SimpleNamespace(**mocks), media=SimpleNamespace(**mocks),
                            execution=SimpleNamespace(**mocks), workflows=SimpleNamespace(build_upscale_graph=build))
                        generator = finish_video(request, SimpleNamespace(models=models, actual_seed=42,
                                                 resolved_width=960, resolved_height=544),
                                                 services, source, "client", 0, Mock(), Mock())
                        while True:
                            try:
                                next(generator)
                            except StopIteration as finished:
                                self.assertEqual(finished.value, result)
                                break
                    mocks["ensure_h3_regenerate_models"].assert_called_once_with(models, option)
                    mocks["ensure_ltx25_upscale_models"].assert_not_called()
                    self.assertEqual(build.call_count, 1)


class RegenerateNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Tensor(np.ndarray):
            def expand(self, *shape):
                return np.broadcast_to(self, tuple(old if new == -1 else new for old, new in zip(self.shape, shape))).view(Tensor)
            def to(self, other):
                return self.astype(other.dtype).view(Tensor)
        cls.tensor = Tensor
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("H3Regenerate")]
        cls.nested = lambda tensors: SimpleNamespace(is_nested=True, tensors=tensors)
        namespace = {"torch": SimpleNamespace(cat=lambda xs, dim=0: np.concatenate(xs, axis=dim).view(Tensor),
                    zeros=lambda shape, **kw: np.zeros(shape).view(Tensor), float32=np.float32,
                    ones_like=np.ones_like, zeros_like=np.zeros_like),
                     "comfy": SimpleNamespace(nested_tensor=SimpleNamespace(NestedTensor=cls.nested))}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
        cls.encode = namespace["H3RegenerateEncodeVideo"]
        cls.prepare = namespace["H3RegeneratePrepareLatent"]

    def test_encode_keeps_every_frame_and_only_pads_the_tail(self):
        for count in (1, 5, 6, 120, 121, 124, 125, 360):
            frames = np.arange(count * 12).reshape(count, 2, 2, 3).view(self.tensor)
            vae = Mock()
            _, padded, original = self.encode().encode(frames, vae)
            pixels = vae.encode_tiled.call_args.args[0]
            self.assertEqual(original, count)
            self.assertEqual(padded % 17, 5)
            self.assertLess(padded - count, 17)
            np.testing.assert_array_equal(pixels[:count], frames)
            for tail in pixels[count:]:
                np.testing.assert_array_equal(tail, frames[-1])

    def test_preparation_preserves_video_and_freezes_encoded_silence(self):
        video = np.full((1, 24, 2, 2, 2), 3.0).view(self.tensor)
        audio = np.full((1, 32, 2, 8), 9.0).view(self.tensor)
        for encoded_frames in (6, 8, 10):
            with self.subTest(encoded_frames=encoded_frames):
                vae = Mock()
                vae.encode.return_value = np.full((1, 32, 2, encoded_frames), 0.25).view(self.tensor)
                out, = self.prepare().prepare({"samples": type(self).nested((video, audio))}, vae)
                self.assertEqual(vae.encode.call_args.args[0].shape, (1, 6400, 2))
                self.assertTrue(np.all(vae.encode.call_args.args[0] == 0))
                np.testing.assert_array_equal(out["samples"].tensors[0], video)
                self.assertTrue(np.all(out["samples"].tensors[1] == 0.25))
                self.assertEqual(out["samples"].tensors[1].shape, audio.shape)
                self.assertTrue(np.all(out["noise_mask"].tensors[0] == 1))
                self.assertTrue(np.all(out["noise_mask"].tensors[1] == 0))
                self.assertTrue(np.all(audio == 9))


if __name__ == "__main__":
    unittest.main()
