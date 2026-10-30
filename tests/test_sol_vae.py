"""Sol decoder provisioning, compatibility and selection boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h3_models
from h3_app import model_service
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.decoder_intent import VideoDecoder
from h3_app.settings import GenerationRequest, OutputSettings, resolve_settings, transition_modes
from h3_ui.persistence import restore_preferences


class SolVaeTests(unittest.TestCase):
    def test_preparation_downloads_only_sol_for_text_only_video(self):
        from h3_app.generation.preparation import _ensure_decoders, _ensure_base_vaes

        output = SimpleNamespace(**VideoDecoder.SOL_INT8.workflow_flags(),
                                 video_encoder="LightVAE Lite", result_format="Video")
        request = SimpleNamespace(output=output)
        effects = SimpleNamespace(
            ensure_int8_video_vae=Mock(), ensure_lightvae=Mock(),
            ensure_base_video_vae=Mock(), model_file_is_ready=lambda path: True,
        )
        services = SimpleNamespace(models=effects)
        runtime = SimpleNamespace(comfy_dir=Path("fixture"))
        models = SimpleNamespace(audio_vae="audio.safetensors")
        list(_ensure_decoders(request, services, runtime, models,
                             has_visual_conditioning=False, use_lightvae=False,
                             selected_image_vae="", started=0, progress=Mock()))
        effects.ensure_int8_video_vae.assert_called_once_with(models, sol=True)
        effects.ensure_lightvae.assert_not_called()
        list(_ensure_base_vaes(request, services, runtime, models,
                              has_visual_conditioning=False, started=0, progress=Mock()))
        effects.ensure_base_video_vae.assert_not_called()

    def test_exclusive_selection_and_inactive_output_formats(self):
        for other in (VideoDecoder.OFFICIAL_INT8, VideoDecoder.LYNNREAL_INT8,
                      VideoDecoder.TENSORRT, VideoDecoder.LIGHTVAE_INT8):
            with self.subTest(other=other):
                _, values = transition_modes(None, {
                    **other.workflow_flags(), "use_sol_vae": True,
                }, "use_sol_vae")
                self.assertEqual(values, VideoDecoder.SOL_INT8.workflow_flags())
                plan = resolve_settings(GenerationRequest(
                    **{**other.workflow_flags(), "use_sol_vae": True}))
                self.assertIn("Select only one video decoder.", plan.issues)
        for fmt, image, active in (("Video", "", True), ("Audio", "", False),
                                   ("Image", "Single-frame 500K", False)):
            plan = resolve_settings(GenerationRequest(
                use_sol_vae=True, output=OutputSettings(result_format=fmt, image_vae=image)))
            self.assertEqual(plan.effective.use_sol_vae, active)
        legacy = [None] * GENERATION_FIELDS.index("use_sol_vae")
        self.assertFalse(GenerationArguments.from_positional(legacy).values["use_sol_vae"])

    def test_old_decoder_preferences_override_new_default(self):
        components = {
            "h3." + name: SimpleNamespace(value=value)
            for name, value in VideoDecoder.SOL_INT8.workflow_flags().items()
        }
        components["h3.lightvae_decoder"].choices = [(s, s) for s in ("none", "lite", "int8")]
        for decoder in VideoDecoder:
            flags = decoder.workflow_flags()
            flags.pop("use_sol_vae")
            saved = {"h3." + k: v for k, v in flags.items()}
            restored, _ = restore_preferences(saved, components)
            self.assertFalse(restored["h3.use_sol_vae"])
        restored, _ = restore_preferences({}, components)
        self.assertTrue(restored["h3.use_sol_vae"])

    def test_old_model_config_and_lazy_download(self):
        key = "video_vae_sol_int8"
        self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
        self.assertEqual(len(h3_models.MODEL_SPECS[key].expected_sha256), 64)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(root, "http://fixture", root / "ComfyUI", root / "models.json", root / "outputs")
            config = h3_models._build_config("manifest.json")
            config.pop(key)
            runtime.models_config.write_text(json.dumps(config), encoding="utf-8")
            models = model_service.load_model_config(runtime=runtime)
            self.assertEqual(models.video_vae_sol_int8, h3_models.MODEL_SPECS[key].local_name)
            with patch.object(model_service, "ensure_models", return_value=True) as ensure:
                self.assertTrue(model_service.ensure_int8_video_vae(models, runtime=runtime, sol=True))
                self.assertEqual(ensure.call_args.args[0], (key,))
                self.assertEqual(ensure.call_args.kwargs["files"], {key: models.video_vae_sol_int8})


if __name__ == "__main__":
    unittest.main()
