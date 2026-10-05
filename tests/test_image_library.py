"""Image-library selection preserves originals and multi-image input order."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import gradio as gr

from h3_app.gallery_store import AssetPage
from h3_ui.image_library import (
    bind_image_library, build_image_input, build_image_file_input,
    build_media_image_destinations, select_library_image,
)


class ImageLibraryTests(unittest.TestCase):
    def test_media_assignment_updates_only_target_and_reveals_reference_slots(self):
        with TemporaryDirectory() as directory:
            source = str(Path(directory) / "original.png")
            Path(source).touch()
            with gr.Blocks() as demo:
                first = build_image_input(type="filepath", label="First")
                refs = [build_image_input(type="filepath", label=f"Picture {i}") for i in (1, 2)]
                files = build_image_file_input(file_count="multiple", type="filepath", label="Refs")
                shown = gr.State(1)
                add = gr.Button("Add reference")
                for slot, ref in enumerate(refs, start=1):
                    ref.h3_reference_slot = slot
                    ref.h3_reference_shown = shown
                    ref.h3_reference_add = add
                view = SimpleNamespace(
                    inspector=gr.Column(), mode=gr.Radio(["Image", "Video"]),
                    selected=gr.State(None),
                )
                build_media_image_destinations(
                    view, {"First": first, "Picture 1": refs[0], "Picture 2": refs[1],
                           "References": tuple(refs), "Refs": files},
                    validate_path=Mock(),
                )
            assign = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "assign")
            values = ["previous.png", None, None, ["uploaded.png"], 1]
            updates = assign("Image", source, "First", *values)
            self.assertEqual(updates[first]["value"], source)
            self.assertNotIn(files, updates)
            self.assertNotIn(refs[0], updates)
            updates = assign("Image", source, "Refs", *values)
            self.assertEqual(updates[files]["value"], ["uploaded.png", source])
            self.assertNotIn(first, updates)
            updates = assign("Image", source, "Picture 2", *values)
            self.assertEqual(updates[shown], 2)
            self.assertTrue(updates[refs[0].h3_library_container]["visible"])
            self.assertTrue(updates[refs[1].h3_library_container]["visible"])
            self.assertFalse(updates[add]["interactive"])
            occupied = ["previous.png", "occupied.png", None, ["uploaded.png"], 1]
            updates = assign("Image", source, "References", *occupied)
            self.assertEqual(updates[refs[1]]["value"], source)
            self.assertNotIn("value", updates[refs[0]])
            with self.assertRaisesRegex(gr.Error, "full"):
                assign("Image", source, "References", "previous.png", "occupied.png", "also-occupied.png", [], 2)
            for mode, selected, name in (("Video", source, "First"), ("Image", None, "First"), ("Image", source, "Other")):
                with self.subTest(mode=mode, selected=selected, name=name), self.assertRaises(gr.Error):
                    assign(mode, selected, name, *values)

    def test_select_original_and_append_without_replacing_uploads(self):
        with TemporaryDirectory() as directory:
            source = str(Path(directory) / "original.png")
            Path(source).touch()
            validate = Mock()
            event = SimpleNamespace(index=0)
            self.assertEqual(
                select_library_image([source], None, event, validate_path=validate),
                source,
            )
            uploads = ["uploaded.png"]
            self.assertEqual(
                select_library_image(
                    [source], uploads, event, validate_path=validate, multiple=True
                ),
                ["uploaded.png", source],
            )
            self.assertEqual(uploads, ["uploaded.png"])
            self.assertEqual(
                select_library_image(
                    [source], [source], event, validate_path=validate, multiple=True
                ),
                [source],
            )
            validate.assert_called_with(source, "Image")

    def test_invalid_and_deleted_selections_are_rejected(self):
        for index in (-1, 1, None, "0", True, []):
            with self.subTest(index=index), self.assertRaises(gr.Error):
                select_library_image(
                    ["missing.png"], None, SimpleNamespace(index=index),
                    validate_path=Mock(),
                )
        with self.assertRaises(gr.Error):
            select_library_image(
                ["missing.png"], None, SimpleNamespace(index=0), validate_path=Mock()
            )
        validate = Mock(side_effect=ValueError("outside library"))
        with self.assertRaisesRegex(ValueError, "outside library"):
            select_library_image(
                ["outside.png"], None, SimpleNamespace(index=0), validate_path=validate
            )

    def test_picker_is_lazy_searches_images_and_loads_more(self):
        index = Mock()
        inventory = index.inventory.return_value
        refresh = Mock(return_value=AssetPage(
            (("thumbnail.jpg", "Original"),), ("original.png",), 30, 24, "images"
        ))
        with gr.Blocks() as demo:
            image = build_image_input(type="filepath", label="First frame")
            bind_image_library(
                image, index=index, refresh_page=refresh, validate_path=Mock()
            )
        refresh.assert_not_called()
        index.inventory.assert_not_called()
        picker = image.h3_library_picker
        browse = next(
            fn.fn for fn in demo.fns.values()
            if fn.outputs == [picker.grid, picker.paths, picker.shown, picker.more, picker.status]
        )
        result = browse("favorite")
        index.inventory.assert_called_once_with("Image", "favorite")
        refresh.assert_called_once_with("Image", 24, paths=inventory)
        self.assertEqual(result[0], [("thumbnail.jpg", "Original")])
        self.assertEqual(result[1], ["original.png"])
        self.assertTrue(result[3]["interactive"])
        more = next(fn.fn for fn in demo.fns.values() if fn.inputs == [picker.query, picker.shown])
        more("favorite", 24)
        refresh.assert_called_with("Image", 48, paths=inventory)
        browse(None)
        index.inventory.assert_called_with("Image", "")


if __name__ == "__main__":
    unittest.main()
