"""Independent video encoding/decoding, lazy components and API compatibility."""

import inspect
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import model_service
from h3_app.catalog import SINGLE_FRAME_IMAGE_VAE
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.decoder_intent import VideoDecoder
from h3_app.errors import H3Error
from h3_app.generation.preparation import prepare_h3
from h3_app.generation.requests import H3Request
from h3_app.resources import resolve_decoders
from h3_app.settings import (
    GenerationRequest,
    OutputSettings,
    resolve_settings,
    transition_modes,
)
from h3_app.workflows import h3
from h3_models import MODEL_SPECS, PRELOAD_MODEL_KEYS
from h3_ui.persistence import restore_preferences
from tests.workflow_cases import H3_DEFAULTS, MODELS


class LightVAETests(unittest.TestCase):
    def test_local_setup_copies_the_complete_pinned_adapter(self):
        import setup_h3

        with TemporaryDirectory() as directory:
            comfy = Path(directory)
            setup_h3.install_bundled_nodes(comfy)
            source = Path(__file__).resolve().parents[1] / "custom_nodes" / "H3LightVAE"
            destination = comfy / "custom_nodes" / "H3LightVAE"
            for name in ("__init__.py", "nodes.py", "h3_vae.py", "UPSTREAM.md"):
                self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())

    def test_every_encoder_decoder_pair_in_both_workflows_and_refinement(self):
        for encoder in ("Original", "LightVAE Lite"):
            for decoder in VideoDecoder:
                for mode in ("fl2va", "ref2va"):
                    with self.subTest(encoder=encoder, decoder=decoder, mode=mode):
                        args = dict(H3_DEFAULTS, **decoder.workflow_flags())
                        args.update(
                            video_encoder=encoder,
                            models=replace(
                                MODELS, video_vae_lynnreal_int8="lynnreal.safetensors"
                            ),
                            available_nodes=H3_DEFAULTS["available_nodes"]
                            | {"LightVAELoader", "MiniMaxH3TRTVAELoader"},
                            latent_upscale_model_name="upscale.safetensors",
                            latent_upscale_refine_steps=2,
                        )
                        if mode == "fl2va":
                            args.update(first_image="first.png", last_image="last.png")
                            graph = h3.build_fl2va_graph(**args)
                        else:
                            args.pop("first_image")
                            args.pop("last_image")
                            args.update(
                                reference_images=["ref.png"],
                                reference_videos=[],
                                reference_audios=[],
                                ref_image_size="Original",
                            )
                            args = {
                                key: value
                                for key, value in args.items()
                                if key
                                in inspect.signature(h3.build_ref2va_graph).parameters
                            }
                            graph = h3.build_ref2va_graph(**args)
                        conditioning = [
                            n
                            for n in graph.values()
                            if n["class_type"]
                            in {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"}
                        ]
                        self.assertGreaterEqual(len(conditioning), 2)
                        encode_refs = [n["inputs"]["vae"] for n in conditioning]
                        self.assertTrue(
                            all(ref == encode_refs[0] for ref in encode_refs)
                        )
                        encode_node = graph[encode_refs[0][0]]
                        if encoder == "Original":
                            self.assertEqual(
                                encode_node["inputs"], {"vae_name": MODELS.video_vae}
                            )
                        else:
                            self.assertEqual(
                                encode_node["class_type"], "LightVAELoader"
                            )
                            self.assertEqual(
                                encode_node["inputs"]["encoder"],
                                MODEL_SPECS["lightvae_encoder"].local_name,
                            )
                            self.assertEqual(encode_node["inputs"]["decoder"], "none")
                        decode = next(
                            n for n in graph.values() if n["class_type"] == "VAEDecode"
                        )
                        decode_node = graph[decode["inputs"]["vae"][0]]
                        if decoder in {
                            VideoDecoder.LIGHTVAE,
                            VideoDecoder.LIGHTVAE_INT8,
                        }:
                            key = (
                                "lightvae_decoder"
                                if decoder == VideoDecoder.LIGHTVAE
                                else "lightvae_decoder_int8"
                            )
                            self.assertEqual(
                                decode_node["inputs"]["decoder"],
                                MODEL_SPECS[key].local_name,
                            )
                            self.assertEqual(decode_node["inputs"]["encoder"], "none")
                        elif decoder == VideoDecoder.TENSORRT:
                            self.assertEqual(
                                decode_node["class_type"], "MiniMaxH3TRTVAELoader"
                            )
                        else:
                            key = {
                                VideoDecoder.STANDARD: "video_vae",
                                VideoDecoder.OFFICIAL_INT8: "video_vae_int8",
                                VideoDecoder.LYNNREAL_INT8: "video_vae_lynnreal_int8",
                            }[decoder]
                            self.assertEqual(
                                decode_node["inputs"]["vae_name"],
                                getattr(args["models"], key),
                            )
                        audio = next(
                            n
                            for n in graph.values()
                            if n["class_type"] == "VAEDecodeAudio"
                        )
                        self.assertEqual(
                            graph[audio["inputs"]["vae"][0]]["inputs"]["vae_name"],
                            MODELS.audio_vae,
                        )

    def test_encoder_changes_conditioning_cache_and_missing_node_fails(self):
        args = dict(
            H3_DEFAULTS,
            first_image="first.png",
            available_nodes=H3_DEFAULTS["available_nodes"] | {"LightVAELoader"},
        )
        original = h3.build_fl2va_graph(**args)
        lite = h3.build_fl2va_graph(**args, video_encoder="LightVAE Lite")

        def cache(graph):
            return next(
                n["inputs"]["cache_key"]
                for n in graph.values()
                if n["class_type"] == "H3ConditioningCache"
            )

        self.assertNotEqual(cache(original), cache(lite))
        args["available_nodes"] = H3_DEFAULTS["available_nodes"]
        with self.assertRaisesRegex(H3Error, "LightVAE is unavailable"):
            h3.build_fl2va_graph(**args, lightvae_decoder="lite")

    def test_lazy_component_selection_and_output_gating(self):
        for encoder in ("Original", "LightVAE Lite"):
            for decoder, key in (
                ("none", None),
                ("lite", "lightvae_decoder"),
                ("int8", "lightvae_decoder_int8"),
            ):
                resources = resolve_decoders(
                    "Video",
                    "",
                    use_int8_vae=False,
                    use_trt_vae=False,
                    video_encoder=encoder,
                    lightvae_decoder=decoder,
                )
                expected = (() if key is None else (key,)) + (
                    ("lightvae_encoder",) if encoder != "Original" else ()
                )
                self.assertEqual(resources.optional_model_keys, expected)
                self.assertTrue(set(expected).isdisjoint(PRELOAD_MODEL_KEYS))
                for fmt, image in (("Audio", ""), ("Image", "Single-frame 500K")):
                    plan = resolve_settings(
                        GenerationRequest(
                            output=OutputSettings(result_format=fmt, image_vae=image),
                            video_encoder=encoder,
                            lightvae_decoder=decoder,
                        )
                    )
                    self.assertEqual(plan.effective.lightvae_decoder, "none")
                    self.assertEqual(plan.effective.video_encoder, encoder)
        with self.assertRaises(ValueError):
            resolve_decoders(
                "Video",
                "",
                use_int8_vae=True,
                use_trt_vae=False,
                lightvae_decoder="lite",
            )

    def test_downloads_only_selected_components_and_reuses_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(
                root,
                "http://fixture",
                root / "ComfyUI",
                root / "models.json",
                root / "outputs",
            )
            with (
                patch.object(
                    model_service,
                    "stale_model_keys",
                    side_effect=lambda **kw: list(kw["model_keys"]),
                ),
                patch.object(model_service, "sync_models") as sync,
                patch.object(model_service, "model_file_is_ready", return_value=True),
                patch.object(model_service, "resolve_hf_token", return_value=None),
            ):
                model_service.ensure_lightvae("LightVAE Lite", "int8", runtime=runtime)
                self.assertEqual(
                    sync.call_args.kwargs["model_keys"],
                    ("lightvae_decoder_int8", "lightvae_encoder"),
                )
            with (
                patch.object(model_service, "stale_model_keys", return_value=[]),
                patch.object(model_service, "sync_models") as sync,
            ):
                self.assertFalse(
                    model_service.ensure_lightvae("Original", "lite", runtime=runtime)
                )
                sync.assert_not_called()

    def test_preparation_skips_unused_components_and_provisions_original_encoding(self):
        from h3_ui import application as app

        for mode, fmt, image, encoder, decoder, selected in (
            (
                "Text to video",
                "Video",
                "Official video VAE",
                "LightVAE Lite",
                "lite",
                ("Original", "lite"),
            ),
            (
                "First / last frame",
                "Video",
                "Official video VAE",
                "LightVAE Lite",
                "int8",
                ("LightVAE Lite", "int8"),
            ),
            (
                "First / last frame",
                "Audio",
                "Official video VAE",
                "LightVAE Lite",
                "int8",
                ("LightVAE Lite", "none"),
            ),
            (
                "First / last frame",
                "Image",
                SINGLE_FRAME_IMAGE_VAE,
                "LightVAE Lite",
                "lite",
                ("LightVAE Lite", "none"),
            ),
            (
                "Text to video",
                "Audio",
                "Official video VAE",
                "LightVAE Lite",
                "lite",
                None,
            ),
            (
                "First / last frame",
                "Video",
                "Official video VAE",
                "Original",
                "none",
                None,
            ),
        ):
            with self.subTest(mode=mode, fmt=fmt, encoder=encoder, decoder=decoder):
                values = {
                    name: app.UI_DEFAULTS.get(name)
                    if p.default is inspect.Parameter.empty
                    else p.default
                    for name, p in inspect.signature(app.generate).parameters.items()
                    if name != "progress"
                }
                values.update(
                    mode=mode,
                    prompt="Fixture",
                    model_profile="Speed",
                    generation_mode="Normal",
                    steps=15,
                    attention_mode="Kitchen",
                    cache_mode="Off",
                    sol_step_off=0.0,
                    semantic_bridge=False,
                    latent_upscale=False,
                    postprocess="None",
                    seed=123,
                    first_image="first.png" if mode == "First / last frame" else None,
                    result_format=fmt,
                    image_vae=image,
                    image_frames=1,
                    video_encoder=encoder,
                    lightvae_decoder=decoder,
                    use_lynnreal_vae=decoder == "none",
                )
                request = H3Request.from_values(values)
                models = replace(MODELS, video_vae_lynnreal_int8="lynnreal.safetensors")
                effects = SimpleNamespace(
                    load_model_config=lambda: models,
                    h3_text_encoder_settings=lambda *a: (
                        "text_encoder",
                        "text.safetensors",
                        False,
                    ),
                    ensure_h3_text_encoder=lambda *a: ("text.safetensors", False),
                    ensure_profile_model=Mock(),
                    ensure_lightvae=Mock(),
                    ensure_int8_video_vae=Mock(),
                    ensure_base_video_vae=Mock(),
                    ensure_audio_vae=Mock(),
                    ensure_single_frame_image_vae=Mock(),
                    model_file_is_ready=lambda path: path.name != models.video_vae,
                )
                services = SimpleNamespace(
                    models=effects,
                    execution=SimpleNamespace(
                        object_info=lambda: dict.fromkeys(
                            H3_DEFAULTS["available_nodes"]
                            | {"LightVAELoader"}
                            | h3.required_nodes_for(
                                mode, False, "Off", result_format=fmt, image_vae=image
                            ),
                            {},
                        )
                    ),
                    policy=SimpleNamespace(
                        resolve_request_settings=lambda v: resolve_settings(
                            GenerationRequest.from_values(v)
                        ),
                        resolve_sol_policy=app.resolve_sol_policy,
                    ),
                    workflows=h3,
                )
                list(
                    prepare_h3(
                        request,
                        services,
                        app.RUNTIME,
                        values,
                        time.monotonic(),
                        lambda *a, **kw: None,
                    )
                )
                if selected:
                    effects.ensure_lightvae.assert_called_once_with(*selected)
                else:
                    effects.ensure_lightvae.assert_not_called()
                effects.ensure_base_video_vae.assert_called_once_with(models)

    def test_old_api_and_preferences_and_presets(self):
        old = [None] * GENERATION_FIELDS.index("video_encoder")
        migrated = GenerationArguments.from_positional(old).values
        self.assertEqual(migrated["video_encoder"], "Original")
        self.assertEqual(migrated["lightvae_decoder"], "none")
        from h3_ui.application import generate

        self.assertEqual(
            tuple(inspect.signature(generate).parameters)[:-1], GENERATION_FIELDS
        )
        components = {
            "h3.video_encoder": SimpleNamespace(
                value="Original",
                choices=[("Original", "Original"), ("LightVAE Lite", "LightVAE Lite")],
            ),
            "h3.lightvae_decoder": SimpleNamespace(
                value="none", choices=[(s, s) for s in ("none", "lite", "int8")]
            ),
        }
        restored, _ = restore_preferences(
            {"h3.video_encoder": "LightVAE Lite", "h3.lightvae_decoder": "int8"},
            components,
        )
        self.assertEqual(restored["h3.lightvae_decoder"], "int8")
        self.assertEqual(restored["h3.video_encoder"], "LightVAE Lite")
        values = {
            "preset": "Fast",
            "video_encoder": "LightVAE Lite",
            **VideoDecoder.LIGHTVAE.workflow_flags(),
        }
        _, preset = transition_modes(None, values, "preset")
        self.assertEqual(preset["lightvae_decoder"], "none")
        self.assertEqual(preset["video_encoder"], "LightVAE Lite")
        self.assertTrue(preset["use_lynnreal_vae"])


if __name__ == "__main__":
    unittest.main()
