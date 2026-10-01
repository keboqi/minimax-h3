"""LynnReal decoder selection, graph routing and lazy provisioning."""

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h3_models
from h3_app import model_service
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.graph import Graph
from h3_app.model_types import ModelConfig
from h3_app.resources import resolve_decoders
from h3_app.settings import GenerationRequest, OutputSettings, resolve_settings, transition_modes
from h3_app.workflows.h3 import add_model_stack
from h3_ui.persistence import restore_preferences


class LynnRealVaeTests(unittest.TestCase):
    def test_selection_excludes_other_vaes_and_survives_restore(self):
        memory, values = transition_modes(None, {
            "use_int8_vae": True, "use_trt_vae": True, "use_lynnreal_vae": True,
        }, "use_lynnreal_vae")
        self.assertTrue(values["use_lynnreal_vae"])
        self.assertFalse(values["use_int8_vae"])
        self.assertFalse(values["use_trt_vae"])
        for field in ("use_int8_vae", "use_trt_vae"):
            _, selected = transition_modes(memory, {**values, field: True}, field)
            self.assertFalse(selected["use_lynnreal_vae"])
        restored, _ = restore_preferences(
            {"h3.use_lynnreal_vae": True},
            {"h3.use_lynnreal_vae": SimpleNamespace(value=False)},
        )
        self.assertTrue(restored["h3.use_lynnreal_vae"])
        for fmt, image, active in (
            ("Video", "", True), ("Audio", "", False),
            ("Image", "Single-frame 500K", False),
        ):
            plan = resolve_settings(GenerationRequest(
                output=OutputSettings(result_format=fmt, image_vae=image),
                use_lynnreal_vae=True,
            ))
            self.assertEqual(plan.effective.use_lynnreal_vae, active)
        with self.assertRaises(ValueError):
            resolve_decoders("Video", "", use_trt_vae=False, use_int8_vae=True, use_lynnreal_vae=True)
        legacy = [None] * (len(GENERATION_FIELDS) - 1)
        self.assertFalse(GenerationArguments.from_positional(legacy).values["use_lynnreal_vae"])

    def test_old_config_migrates_and_downloads_only_selected_vae(self):
        key = "video_vae_lynnreal_int8"
        self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
        self.assertEqual(resolve_decoders(
            "Video", "", use_trt_vae=False, use_int8_vae=False, use_lynnreal_vae=True,
        ).optional_model_keys, (key,))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(root, "http://fixture", root / "ComfyUI", root / "models.json", root / "outputs")
            config = h3_models._build_config("manifest.json")
            config.pop(key)
            config.pop(key + "_source")
            runtime.models_config.write_text(json.dumps(config), encoding="utf-8")
            models = model_service.load_model_config(runtime=runtime)
            self.assertEqual(models.video_vae_lynnreal_int8, h3_models.MODEL_SPECS[key].local_name)
            with (
                patch.object(model_service, "stale_model_keys", return_value=[key]),
                patch.object(model_service, "sync_models") as sync,
                patch.object(model_service, "model_file_is_ready", return_value=True),
                patch.object(model_service, "resolve_hf_token", return_value=None),
            ):
                self.assertTrue(model_service.ensure_int8_video_vae(models, runtime=runtime, lynnreal=True))
                self.assertEqual(sync.call_args.kwargs["model_keys"], (key,))
            with (
                patch.object(model_service, "stale_model_keys", return_value=[]),
                patch.object(model_service, "sync_models") as sync,
            ):
                self.assertFalse(model_service.ensure_int8_video_vae(models, runtime=runtime, lynnreal=True))
                sync.assert_not_called()

    def test_graph_routes_selected_vae_and_preserves_audio(self):
        models = ModelConfig({}, "speed", "text", "fp16", "audio",
                             video_vae_int8="int8", video_vae_lynnreal_int8="lynnreal")
        args = {
            name: None for name, parameter in inspect.signature(add_model_stack).parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        args.update(model_name="base", models=models, turbo_lora_name=None,
                    use_sol=False, cache_mode="None", available_nodes=set())
        for selection, expected in (({}, "fp16"), ({"use_int8_vae": True}, "int8"),
                                    ({"use_lynnreal_vae": True}, "lynnreal")):
            graph = Graph()
            args["graph"] = graph
            _, _, video, audio = add_model_stack(**args, **selection)
            self.assertEqual(graph.nodes[video[0]]["inputs"]["vae_name"], expected)
            self.assertEqual(graph.nodes[audio[0]]["inputs"]["vae_name"], "audio")
