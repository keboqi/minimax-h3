"""Shared image draft validation and H3 settings isolation."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from h3_app.catalog import (
    H3_EDIT_SIZE_MATCH, H3_IMAGE_4M_RESOLUTION_PRESETS, H3_IMAGE_8M_RESOLUTION_PRESETS, H3_IMAGE_EDIT_SIZE_MAX,
    QWEN_EDIT_SIZE_MANUAL, UI_DEFAULTS,
)
from h3_app.contracts import GENERATION_COMPONENTS, GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.generation.requests import H3Request
from h3_models import QWEN_IMAGE21_OFFICIAL_TURBO, QWEN_IMAGE21_TURBO_MODES
from h3_ui.image_controller import (
    GENERATION_METHODS, GENERATION_OPTIONS, ImageGenerationController,
    draft_issue, h3_image_size, h3_image_values, image_mode, reference_limit, reference_tag, suggest_prompt,
)
from h3_ui.engines.controls import resolution_preset_values
from h3_ui.image_bindings import move_reference
from h3_ui.image_library import NAME_THUMBNAILS, REMOVE_THUMBNAILS, _thumbnail_items, _thumbnail_update
from h3_ui.persistence import restore_preferences


class ImageWorkspaceTests(unittest.TestCase):
    def test_moss_keeps_the_edit_draft_inactive_while_other_models_preserve_intent(self):
        self.assertEqual(image_mode("moss_image", "Image edit"), "Text to image")
        self.assertEqual(image_mode("qwen", "Image edit"), "Image edit")
        self.assertEqual(image_mode("fibo2", "Text to image"), "Text to image")

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


    def test_image_preferences_migrate_when_another_task_was_last_open(self):
        components = {
            "image.mode": SimpleNamespace(value="Text to image", choices=["Text to image", "Image edit"]),
            "image.prompt_backend": SimpleNamespace(value="Gemini", choices=["Gemini", "OpenAI-compatible"]),
        }
        restored, _ = restore_preferences({"schema_version": 11, "values": {
            "workspace.task": "Video", "workspace.engine": "ltx", "qwen_image21.mode": "Image edit",
            "qwen_image21.prompt_backend": "OpenAI-compatible",
        }}, components)
        self.assertEqual(restored["image.mode"], "Image edit")
        self.assertEqual(restored["image.prompt_backend"], "OpenAI-compatible")


class ReferenceTagTests(unittest.TestCase):
    def test_each_model_names_a_reference_its_own_way(self):
        for engine, tags in (("h3", ("<Picture 1>", "<Picture 2>")), ("qwen", ("<image1>", "<image2>")),
                             ("fibo2", ("<image_1>", "<image_2>")), ("flux3", ("<ref_image_0>", "<ref_image_1>"))):
            with self.subTest(engine=engine):
                self.assertEqual(tuple(reference_tag(engine, i, 2) for i in (1, 2)), tags)

    def test_a_tag_shows_only_where_a_prompt_can_use_it(self):
        self.assertIsNone(reference_tag("qwen", 1, 1))
        self.assertIsNone(reference_tag("qwen", 1, 3, True))
        self.assertEqual(reference_tag("h3", 1, 1), "<Picture 1>")
        self.assertIsNone(reference_tag("moss_image", 1, 2))
        tag = lambda index, count, model, separate: reference_tag(model, index, count, separate)
        self.assertEqual(_thumbnail_update(tag, ["a.png"], "qwen", False)["label"], REMOVE_THUMBNAILS)
        self.assertEqual(_thumbnail_update(tag, ["a.png"], "h3", False)["label"], NAME_THUMBNAILS)
        self.assertIsNone(reference_tag("ltx", 1, 2))

    def test_thumbnails_keep_order_and_follow_the_model(self):
        tag = lambda index, count, model, separate: reference_tag(model, index, count, separate)
        self.assertEqual(_thumbnail_items(["a", "b"], tag, "h3", False),
                         [("a", "<Picture 1>"), ("b", "<Picture 2>")])
        self.assertEqual(_thumbnail_items(None, tag, "h3", False), [])
        self.assertEqual(_thumbnail_items(["a"]), ["a"])


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
        image = SimpleNamespace(h3_width=1024, h3_height=1024, h3_seed=27, h3_edit_size=QWEN_EDIT_SIZE_MANUAL)
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

    def test_h3_edit_size_follows_the_first_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory, "first.png"), Path(directory, "second.png")
            Image.new("RGB", (1920, 1080)).save(first)
            Image.new("RGB", (100, 900)).save(second)
            references, sliders = [str(first), str(second)], (1024, 1024)
            self.assertEqual(h3_image_size(H3_EDIT_SIZE_MATCH, references, *sliders), (1920, 1088))
            self.assertEqual(h3_image_size(QWEN_EDIT_SIZE_MANUAL, references, *sliders), sliders)
            width, height = h3_image_size(H3_IMAGE_EDIT_SIZE_MAX, references, *sliders)
            self.assertEqual((width % 32, height % 32), (0, 0))
            self.assertAlmostEqual(width / height, 16 / 9, delta=0.02)
            self.assertAlmostEqual(width * height / 1_000_000, 8, delta=0.5)
        # Generation has no first image, so every choice uses the sliders.
        self.assertEqual(h3_image_size(H3_EDIT_SIZE_MATCH, [], *sliders), sliders)

    def test_h3_edit_dispatch_uses_the_resolved_size(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("h3_app.generation.ph3_image.generate_ph3_image", return_value=iter(())) as generate:
            first = Path(directory, "first.png")
            Image.new("RGB", (1920, 1080)).save(first)
            self.controller.generate("h3", "Image edit", "edit", [str(first)], {
                **self.values, "image__width": 1024, "image__height": 1024, "image__seed": 27,
                "image__edit_size": H3_EDIT_SIZE_MATCH,
            })
        request = generate.call_args.args[0]
        self.assertEqual((request.output.width, request.output.height), (1920, 1088))

    def test_h3_resolution_presets_stay_on_the_grid_and_near_their_size(self):
        for presets, megapixels in ((H3_IMAGE_4M_RESOLUTION_PRESETS, 4), (H3_IMAGE_8M_RESOLUTION_PRESETS, 8)):
            for name in presets:
                with self.subTest(name=name):
                    width, height = resolution_preset_values(name)
                    ratio_width, ratio_height = map(int, name.split(" · ")[1].split(":"))
                    self.assertEqual((width % 32, height % 32), (0, 0))
                    self.assertAlmostEqual(width / height, ratio_width / ratio_height, delta=0.01)
                    self.assertAlmostEqual(width * height / 1_000_000, megapixels, delta=0.5)

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
