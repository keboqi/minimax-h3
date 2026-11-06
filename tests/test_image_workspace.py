"""Shared image draft validation and H3 settings isolation."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

from h3_app.catalog import UI_DEFAULTS
from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.generation.requests import H3Request
from h3_models import QWEN_IMAGE21_OFFICIAL_TURBO, QWEN_IMAGE21_TURBO_MODES
from h3_ui.image_controller import (
    GENERATION_METHODS, GENERATION_OPTIONS, ImageGenerationController,
    draft_issue, h3_image_values, reference_limit, suggest_prompt,
)
from h3_ui.image_bindings import move_reference
from h3_ui.persistence import restore_preferences


class ImageWorkspaceTests(unittest.TestCase):
    def test_generation_does_not_validate_inactive_references(self):
        self.assertEqual(draft_issue("moss_image", "Text to image", "fox", ["a"] * 15), "")
        self.assertEqual(draft_issue("qwen", "Text to image", "fox", ["a"] * 15), "")

    def test_edit_capabilities_and_variant_limits(self):
        self.assertIn("generation only", draft_issue("moss_image", "Image edit", "fox", ["a"]))
        self.assertIn("at least one", draft_issue("h3", "Image edit", "fox", []))
        self.assertIn("8 references", draft_issue("h3", "Image edit", "fox", ["a"] * 9))
        self.assertIn("4 references", draft_issue("fibo2", "Image edit", "fox", ["a"] * 5))
        self.assertIn("Edit mode", draft_issue("fibo2", "Text to image", "fox", [], fibo_model="Edit Turbo"))
        self.assertEqual(reference_limit("qwen", QWEN_IMAGE21_OFFICIAL_TURBO), 10)
        for variant in QWEN_IMAGE21_TURBO_MODES:
            if reference_limit("qwen", variant) == 3:
                self.assertIn("3 references", draft_issue("qwen", "Image edit", "fox", ["a"] * 4, variant))
                self.assertEqual(draft_issue("qwen", "Image edit", "fox", ["a"] * 12, variant,
                                             separate=True), "")

    def test_reordering_preserves_all_inputs_and_moves_the_target(self):
        paths = ["a", "b", "c"]
        self.assertEqual(move_reference(paths, 3, 1), ["c", "a", "b"])
        self.assertEqual(paths, ["a", "b", "c"])

    def test_reordering_rejects_empty_nonfinite_or_noninteger_positions(self):
        for position in (None, "1", True, float("nan"), float("inf"), 0, 1.5, 4):
            with self.subTest(position=position), self.assertRaises(gr.Error):
                move_reference(["a", "b", "c"], position, 1)

    def test_h3_adapter_reads_current_settings_and_clears_video_conditioning(self):
        settings = {**dict.fromkeys(GENERATION_FIELDS), **UI_DEFAULTS,
                    "model_profile": "Quality", "first_image": "old.png", "ref_audio_1": "old.wav",
                    "ref_video_1": "old.mp4", "ref_image_9": "old-ref.png"}
        original = deepcopy(settings)
        values = h3_image_values(settings, "edit this", ["a.png", "b.png"], 1248, 832, 17)
        request = H3Request.from_values(values)
        self.assertEqual(settings, original)
        self.assertEqual(request.sampling.model_profile, "Quality")
        self.assertEqual(request.output.result_format, "Image")
        self.assertEqual(values["mode"], "Reference media")
        self.assertEqual((values["ref_image_1"], values["ref_image_2"]), ("a.png", "b.png"))
        self.assertIsNone(values["ref_image_9"])
        self.assertIsNone(values["first_image"])
        self.assertIsNone(values["ref_video_1"])
        self.assertIsNone(values["ref_audio_1"])
        settings["model_profile"] = "Speed"
        self.assertEqual(h3_image_values(settings, "fox", [], 1024, 1024, -1)["model_profile"], "Speed")
        # The UI deliberately does not capture Video media inputs for an Image job.
        only_settings = {name: value for name, value in settings.items()
                         if not name.startswith(("ref_image_", "ref_audio_", "ref_video_", "fl2va_audio_"))}
        request = H3Request.from_values(h3_image_values(only_settings, "fox", ["a.png"], 1024, 1024, -1))
        self.assertIsNone(request.media.ref_image_2)

    def test_old_image_preferences_migrate_without_restoring_the_hidden_qwen_mode(self):
        components = {
            "image.mode": SimpleNamespace(value="Text to image", choices=["Text to image", "Image edit"]),
            "image.prompt_backend": SimpleNamespace(value="Gemini", choices=["Gemini", "OpenAI-compatible"]),
            "qwen_image21.mode": SimpleNamespace(value="Text to image"),
        }
        saved = {"schema_version": 11, "values": {
            "workspace.task": "Image", "workspace.engine": "qwen", "qwen_image21.mode": "Image edit",
            "qwen_image21.prompt_backend": "OpenAI-compatible", "image.writer_key": "secret",
        }}
        restored, _ = restore_preferences(saved, components)
        self.assertEqual(restored["image.mode"], "Image edit")
        self.assertEqual(restored["image.prompt_backend"], "OpenAI-compatible")
        self.assertNotIn("qwen_image21.mode", restored)
        self.assertNotIn("image.writer_key", restored)
        saved["values"]["image.mode"] = "Text to image"
        restored, _ = restore_preferences(saved, components)
        self.assertEqual(restored["image.mode"], "Text to image")


class ImageGenerationControllerTests(unittest.TestCase):
    def setUp(self):
        self.views = {
            engine: SimpleNamespace(**{component: f"{engine}:{component}" for _, component in fields})
            for engine, fields in GENERATION_OPTIONS.items()
        }
        self.handlers = {engine: SimpleNamespace(**{method: Mock(return_value=iter(()))})
                         for engine, method in GENERATION_METHODS.items()}
        h3 = SimpleNamespace(**{component: UI_DEFAULTS.get(field)
                               for field, component in zip(GENERATION_FIELDS, GENERATION_COMPONENTS, strict=True)})
        image = SimpleNamespace(h3_width=1024, h3_height=1024, h3_seed=27)
        self.controller = ImageGenerationController(self.views, self.handlers, h3, image, Mock())
        self.values = {**self.controller.fields, "qwen__turbo_variant": "Off", "qwen__batch_edit_inputs": False,
                       "fibo2__model_choice": "Dev"}

    def test_dispatch_preserves_selected_options_and_excludes_inactive_references(self):
        self.controller.generate("qwen", "Text to image", "a bird", ["saved.png"], self.values)
        kwargs = self.handlers["qwen"].generate_qwen_image21.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a bird")
        self.assertEqual(kwargs["model_choice"], "qwen:model")
        self.assertEqual(kwargs["width"], "qwen:width")
        self.assertEqual(kwargs["reference_images"], [])
        self.assertNotIn("output_resolution", kwargs)  # No duplicate deprecated preset alias.
        self.assertNotIn("layout", kwargs)
        self.controller.generate("flux3", "Image edit", "edit", ["a", "b"], self.values)
        kwargs = self.handlers["flux3"].generate_flux3.call_args.kwargs
        self.assertEqual(kwargs["width"], "flux3:width")
        self.assertEqual(kwargs["reference_images"], ["a", "b"])

    def test_moss_generation_never_receives_an_edit_parameter(self):
        self.controller.generate("moss_image", "Text to image", "bird", ["inactive"], self.values)
        kwargs = self.handlers["moss_image"].generate_moss_image.call_args.kwargs
        self.assertNotIn("reference_images", kwargs)
        self.assertNotIn("edit_size", kwargs)
        with self.assertRaisesRegex(H3Error, "generation only"):
            self.controller.generate("moss_image", "Image edit", "edit", ["a"], self.values)
        self.handlers["moss_image"].generate_moss_image.assert_called_once()

    def test_h3_dispatch_uses_named_current_settings_without_video_media(self):
        self.values["h3__model_profile"] = "Original"
        with patch("h3_app.generation.ph3_image.generate_ph3_image", return_value=iter(())) as generate:
            self.controller.generate("h3", "Image edit", "edit", ["a.png"], self.values)
        request = generate.call_args.args[0]
        self.assertEqual(request.sampling.model_profile, "Original")
        self.assertEqual(request.output.seed, 27)
        self.assertEqual(request.media.ref_image_1, "a.png")
        self.assertIsNone(request.media.ref_audio_1)
        self.assertFalse(any(name.startswith("option_") for name in self.controller.fields))

    def test_prompt_assistance_uses_selected_size_and_excludes_inactive_references(self):
        enhance = Mock(return_value=("suggestion", "ready"))
        options = {
            "writer_model": "writer", "writer_key": "temporary", "writer_backend": "Gemini",
            "writer_openai_key": "", "writer_openai_model": "model", "writer_base_url": None,
            "qwen_width": 1536, "qwen_height": 1024, "qwen_edit_size": "Use width and height above",
        }
        result = suggest_prompt({"qwen": SimpleNamespace(enhance_qwen_image21_prompt=enhance)},
                                "qwen", "Text to image", "bird", ["inactive"], options)
        self.assertEqual(result, ("suggestion", "ready"))
        self.assertEqual(enhance.call_args.args[4:], ([], 1536, 1024))
        self.assertEqual(enhance.call_args.kwargs["edit_size"], "Use width and height above")


if __name__ == "__main__":
    unittest.main()
