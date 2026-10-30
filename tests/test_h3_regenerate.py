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
from h3_app.media_types import VideoMetadata
from h3_app.policy import resolve_h3_latent_upscale_method
from h3_app.workflows import h3, upscale
from h3_models import MODEL_SPECS, PRELOAD_MODEL_KEYS, H3_LATENT_UPSCALER_MODEL_CHOICES
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

    def test_modes_resolve_and_weights_stay_lazy(self):
        for option, key in catalog.H3_REGENERATE_MODELS.items():
            self.assertEqual(resolve_h3_latent_upscale_method(option), option)
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            self.assertIn(option, catalog.GENERATION_POSTPROCESS_OPTIONS)
            self.assertNotIn(option, catalog.POSTPROCESS_OPTIONS)
            self.assertIn(option, catalog.H3_LATENT_UPSCALE_METHODS)

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
                        "stage_file": "source.mp4",
                        "submit_prompt": "job", "poll_comfy_progress": [], "wait_for_history": {},
                        "resolve_output": result,
                        "gallery_media_processed_result": "complete", "write_snapshot": None,
                    }
                    mocks = {name: stack.enter_context(patch.object(owner(app, name), name, return_value=value))
                             for name, value in values.items()}
                    build = stack.enter_context(patch.object(app, "build_upscale_graph", wraps=app.build_upscale_graph))
                    if entry == "media":
                        updates = list(app.MEDIA.postprocess_selected_gallery_media(
                            "Video", str(source), option, 42, "unused", "unused", "A bird flies",
                            False, "2560 × 2560", request=Mock(), progress=Mock()))
                        self.assertEqual(updates[-1], "complete")
                    else:
                        request = SimpleNamespace(media=SimpleNamespace(prompt="A bird flies"), finishing=SimpleNamespace(
                            postprocess=option, ltx25_model="unused", seedvr2_model="unused",
                            upscale_resolution="2560 × 2560", upscale_force_offload=False))
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


class SharedRefinementTests(unittest.TestCase):
    models = RegenerateGraphTests.models
    def test_shared_settings_drive_precision_steps_and_lora(self):
        models = self.models()
        models.dmad_4step_lora = "dmad.safetensors"
        precision_choice = next(name for name, key in H3_LATENT_UPSCALER_MODEL_CHOICES.items()
                                if key == "h3_latent_upscaler_3d_fp32")
        settings = dict(steps=4, latent_upscale_method=catalog.H3_LATENT_UPSCALE_STANDARD,
                        latent_upscaler_model=precision_choice, latent_upscale_refine_steps=1,
                        latent_upscale_refine_lora=catalog.DMAD_4STEP_LORA)
        graph, steps = upscale.build_upscale_graph(
            option=catalog.H3_REFINEMENT, source_video="source.mp4", seed=42, models=models,
            width=960, height=544, fps=29.97, refinement=settings,
        )
        nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
        self.assertEqual(steps, 1)
        self.assertEqual(nodes[catalog.H3_LATENT_UPSCALER_LOADER_NODE]["precision"], "fp32")
        self.assertEqual(nodes["LoraLoaderModelOnly"]["lora_name"], "dmad.safetensors")
        self.assertEqual(nodes["KSamplerSelect"]["sampler_name"], "lcm")
        self.assertEqual(nodes["SplitSigmas"]["step"], 3)
        self.assertEqual(nodes["CreateVideo"]["fps"], 29.97)
        self.assertEqual(set(nodes), upscale.required_upscale_nodes(catalog.H3_REFINEMENT, models, settings))

    def test_full_frame_and_regenerate_choices_use_shared_settings(self):
        models = self.models()
        for method in (catalog.H3_LATENT_UPSCALE_STANDARD, *catalog.H3_REGENERATE_MODELS):
            with self.subTest(method=method):
                graph, steps = app.build_upscale_graph(
                    option=catalog.H3_REFINEMENT, source_video="source.mp4", seed=42, models=models,
                    width=960, height=544, refinement=dict(latent_upscale_method=method,
                        generation_mode="Normal", steps=10, latent_upscale_refine_steps=1),
                )
                nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
                self.assertIn("SamplerCustomAdvanced", nodes)
                if method in catalog.H3_REGENERATE_MODELS:
                    self.assertEqual(steps, 3)
                    self.assertEqual(nodes["ManualSigmas"]["sigmas"], catalog.H3_REGENERATE_SIGMAS)
                else:
                    self.assertEqual(steps, 1)
                    self.assertNotIn("LoraLoaderModelOnly", nodes)
                    self.assertEqual(nodes["SplitSigmas"]["step"], 9)

    def test_shared_model_downloads_follow_selected_profile_upscaler_and_lora(self):
        models = self.models()
        models.text_encoder = MODEL_SPECS["text_encoder"].local_name
        models.profiles["other"] = ModelProfile("Other", "other.safetensors", "other-ref.safetensors")
        models.dmad_4step_lora = "dmad.safetensors"
        precision_choice = next(name for name, key in H3_LATENT_UPSCALER_MODEL_CHOICES.items()
                                if key == "h3_latent_upscaler_3d_fp16")
        settings = dict(model_profile="Other", latent_upscaler_model=precision_choice,
                        latent_upscale_refine_lora=catalog.DMAD_4STEP_LORA)
        with ExitStack() as stack:
            effects = {name: stack.enter_context(patch.object(model_service, name, return_value=False))
                       for name in ("ensure_profile_model", "ensure_base_video_vae", "ensure_audio_vae",
                                    "ensure_models", "ensure_turbo_lora")}
            model_service.ensure_h3_regenerate_models(models, catalog.H3_REFINEMENT,
                                                     runtime=Mock(), refinement=settings)
            self.assertEqual(effects["ensure_profile_model"].call_args.args[0], "other")
            keys = effects["ensure_models"].call_args.args[0]
            self.assertIn("h3_latent_upscaler_3d_fp16", keys)
            self.assertFalse(set(keys) & set(catalog.H3_REGENERATE_MODELS.values()))
            self.assertEqual(effects["ensure_turbo_lora"].call_args.args,
                             (models, catalog.DMAD_4STEP_LORA, "Text to video"))

    def test_media_uses_native_encoder_decoder_stack_for_every_selection(self):
        from h3_app.decoder_intent import VideoDecoder
        models = self.models()
        models.video_vae_int8 = "official-int8.safetensors"
        models.video_vae_lynnreal_int8 = "lynnreal.safetensors"
        models.video_vae_trt_decoder = "decoder.onnx"
        for encoder in ("Original", "LightVAE Lite"):
            for decoder in VideoDecoder:
                with self.subTest(encoder=encoder, decoder=decoder):
                    settings = dict(video_encoder=encoder, **decoder.workflow_flags(),
                                    latent_upscale_method=catalog.H3_REGENERATE_16)
                    graph, _ = upscale.build_upscale_graph(
                        option=catalog.H3_REFINEMENT, source_video="source.mp4", seed=42, models=models,
                        width=960, height=544, target_width=1600, target_height=900, refinement=settings)
                    encode = next(n["inputs"] for n in graph.values() if n["class_type"] == "H3RegenerateEncodeVideo")
                    encode_node = graph[encode["vae"][0]]
                    self.assertEqual(encode_node["class_type"], "VAELoader" if encoder == "Original" else "LightVAELoader")
                    if encoder == "Original":
                        self.assertEqual(encode_node["inputs"]["vae_name"], models.video_vae)
                    decode = next(n["inputs"] for n in graph.values() if n["class_type"] == "VAEDecode")
                    decode_node = graph[decode["vae"][0]]
                    if decoder == VideoDecoder.TENSORRT:
                        self.assertEqual(decode_node["class_type"], "MiniMaxH3TRTVAELoader")
                    elif decoder in {VideoDecoder.LIGHTVAE, VideoDecoder.LIGHTVAE_INT8}:
                        self.assertEqual(decode_node["class_type"], "LightVAELoader")
                        self.assertEqual(decode_node["inputs"]["encoder"], "none")
                    else:
                        key = {VideoDecoder.STANDARD: "video_vae", VideoDecoder.OFFICIAL_INT8: "video_vae_int8",
                               VideoDecoder.SOL_INT8: "video_vae_sol_int8", VideoDecoder.LYNNREAL_INT8: "video_vae_lynnreal_int8"}[decoder]
                        self.assertEqual(decode_node["inputs"]["vae_name"], getattr(models, key))
                    video = next(n["inputs"] for n in graph.values() if n["class_type"] == "CreateVideo")
                    final = graph[video["images"][0]]["inputs"]
                    self.assertEqual((final["width"], final["height"]), (1600, 900))
                    self.assertIn(catalog.H3_NVENC_SAVE_NODE, {n["class_type"] for n in graph.values()})

    def test_attention_conditioning_and_memory_follow_native_policy(self):
        for attention, node in (("SLA", catalog.SLA_ATTENTION_NODE), ("Veda", catalog.VEDA_ATTENTION_NODE),
                                ("Sage 2", catalog.SAGE_ATTENTION_NODE), ("Sol-Attn", catalog.SOL_ATTENTION_NODE)):
            with self.subTest(attention=attention):
                settings = dict(latent_upscale_method=catalog.H3_REGENERATE_16,
                    attention_mode=attention, sla_preset="Quality", semantic_bridge=True, semantic_bridge_alpha=0.23,
                    encoder_small_input=True, reuse_unchanged_inputs=False, stage_model_offload=True, cache_mode="Spectrum")
                graph, _ = upscale.build_upscale_graph(option=catalog.H3_REFINEMENT, source_video="source.mp4",
                    seed=42, models=self.models(), width=960, height=544, refinement=settings)
                nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
                self.assertIn(node, nodes)
                self.assertNotIn("SpectrumApplyMiniMaxH3", nodes)  # H3 disables caches for native refinement.
                self.assertEqual(nodes[catalog.H3_SEMANTIC_BRIDGE_NODE]["alpha"], 0.23)
                cache = nodes[catalog.H3_CONDITIONING_CACHE_NODE]
                self.assertEqual((cache["encoder_small_input"], cache["reuse_conditioning"]), (True, False))
                self.assertIn(catalog.H3_STAGE_OFFLOAD_NODE, nodes)
                upscaler = nodes[catalog.H3_LATENT_UPSCALER_NODE]
                self.assertEqual(graph[upscaler["samples"][0]]["class_type"], catalog.H3_STAGE_OFFLOAD_NODE)
                if attention == "SLA":
                    self.assertEqual(nodes[node]["dense_steps"], "")
                if attention == "Veda":
                    self.assertEqual(nodes[node]["full_attention_steps"], "2")
        graph, _ = upscale.build_upscale_graph(option=catalog.H3_REFINEMENT, source_video="source.mp4",
            seed=42, models=self.models(), width=960, height=544,
            refinement=dict(text_encoder="BF16", latent_upscale_method=catalog.H3_REGENERATE_16, stage_model_offload=False))
        self.assertIn(catalog.H3_STAGE_OFFLOAD_POLICY_NODE, {n["class_type"] for n in graph.values()})

    def test_selected_reference_model_and_lora_are_preserved(self):
        models = self.models()
        models.turbo_ref_lora = "reference-turbo.safetensors"
        settings = dict(mode="Reference media", turbo_variant=catalog.LIGHTX2V_4STEP_TURBO,
                        latent_upscale_refine_steps=1, steps=4)
        graph, _ = upscale.build_upscale_graph(option=catalog.H3_REFINEMENT, source_video="source.mp4",
            seed=42, models=models, width=960, height=544, refinement=settings)
        nodes = {n["class_type"]: n["inputs"] for n in graph.values()}
        self.assertEqual(nodes["UNETLoader"]["unet_name"], "ref.safetensors")
        self.assertTrue(any(n["inputs"].get("lora_name") == models.turbo_ref_lora for n in graph.values()))
        self.assertIn("MiniMaxH3ReferenceToVideo", nodes)
        self.assertNotIn(catalog.H3_SEMANTIC_BRIDGE_NODE, nodes)

    def test_selected_nvfp4_encoder_is_used_instead_of_configured_int8(self):
        models = self.models()
        models.text_encoder = MODEL_SPECS["text_encoder_int8"].local_name
        settings = dict(text_encoder="NVFP4 / AWQ", latent_upscale_method=catalog.H3_REGENERATE_16)
        with ExitStack() as stack:
            effects = {name: stack.enter_context(patch.object(model_service, name, return_value=False))
                       for name in ("ensure_profile_model", "ensure_base_video_vae", "ensure_audio_vae", "ensure_models")}
            model_service.ensure_h3_regenerate_models(models, catalog.H3_REFINEMENT,
                                                     runtime=Mock(), refinement=settings)
            keys = effects["ensure_models"].call_args.args[0]
            self.assertIn("text_encoder", keys)
            self.assertNotIn("text_encoder_int8", keys)
        graph, _ = app.build_upscale_graph(
            option=catalog.H3_REFINEMENT, source_video="source.mp4", seed=42, models=models,
            width=960, height=544, refinement=settings,
        )
        encoder = next(n["inputs"] for n in graph.values() if n["class_type"] == "CLIPLoader")
        self.assertEqual(encoder["clip_name"], MODEL_SPECS["text_encoder"].local_name)
        self.assertEqual(models.text_encoder, MODEL_SPECS["text_encoder_int8"].local_name)

    def test_shared_refinement_routes_settings_through_media_and_downloads(self):
        models = self.models()
        source, result = Path("source.mp4"), Path("upscaled.mp4")
        settings = dict(latent_upscale_method=catalog.H3_REGENERATE_384, text_encoder="NVFP4 / AWQ",
                        prompt="H3 scene prompt", seed=123, upscale_force_offload=True)
        with ExitStack() as stack:
            values = {
                "managed_video_path": source, "load_model_config": models,
                "object_info": upscale.required_upscale_nodes(catalog.H3_REFINEMENT, models, settings),
                "ensure_h3_regenerate_models": False, "unload_comfy_models": None,
                "probe_video_metadata": VideoMetadata(24, 960, 544, 5, 120, True),
                "stage_file": "source.mp4",
                "submit_prompt": "job", "poll_comfy_progress": [], "wait_for_history": {},
                "resolve_output": result,
                "gallery_media_processed_result": "complete", "write_snapshot": None,
            }
            mocks = {name: stack.enter_context(patch.object(owner(app, name), name, return_value=value))
                     for name, value in values.items()}
            build = stack.enter_context(patch.object(app, "build_upscale_graph", wraps=app.build_upscale_graph))
            updates = list(app.MEDIA.postprocess_selected_gallery_media(
                "Video", str(source), catalog.H3_REFINEMENT, 42, "unused", "unused", "A bird flies",
                False, "2560 × 2560", request=Mock(), progress=Mock(), h3_refinement=settings))
            self.assertEqual(updates[-1], "complete")
            received = build.call_args.kwargs["refinement"]
            self.assertEqual({key: received[key] for key in settings}, settings)
            self.assertEqual(build.call_args.kwargs["seed"], 123)
            self.assertEqual(build.call_args.kwargs["prompt"], "H3 scene prompt")
            self.assertEqual(build.call_args.kwargs["target_width"], 2560)
            mocks["unload_comfy_models"].assert_called_once()
            mocks["ensure_h3_regenerate_models"].assert_called_once_with(models, catalog.H3_REFINEMENT, refinement=received)


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
