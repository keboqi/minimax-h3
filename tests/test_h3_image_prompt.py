"""Image-specific writing recipes, source metadata, and UI routing."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from h3_app.prompt_writers.h3_image import enhance_h3_image_prompt
from h3_app.workflows.h3_image import build_h3_image_graph
from h3_app.h3_image import h3_image_prompt_path
from h3_ui.image_controller import suggest_prompt


class H3ImagePromptTests(unittest.TestCase):
    def test_local_expansion_selects_the_same_image_specific_recipe(self):
        for references in ((), ("source.png",)):
            with self.subTest(editing=bool(references)):
                graph = build_h3_image_graph(prompt="requested change", reference_images=references,
                                             width=4736, height=3552)
                writer = next(node["inputs"] for node in graph.values() if node["class_type"] == "TextGenerateV2")
                self.assertIn(h3_image_prompt_path(bool(references)).read_text(encoding="utf-8").strip(), writer["prompt"])
                self.assertIn("Selected target canvas: 4736x3552; aspect ratio 4:3.", writer["prompt"])

    def test_text_writer_uses_t2i_recipe_and_selected_canvas(self):
        with patch("h3_app.prompt_writers.h3_image.common.write_prompt", return_value=("caption", "ok")) as write:
            result = enhance_h3_image_prompt("three bowls", "model", "key", [], 4096, 4096,
                                             backend="OpenAI-compatible", openai_api_key="temporary-key",
                                             openai_model="vision-model", openai_base_url="https://example.test/v1")
        self.assertEqual(result, ("caption", "ok"))
        values = write.call_args.kwargs
        self.assertEqual(values["system_path"], h3_image_prompt_path(False))
        self.assertEqual(values["media_values"], ())
        self.assertIn("4096x4096; aspect ratio 1:1", values["request_text"])
        self.assertEqual(values["openai_api_key"], "temporary-key")
        self.assertEqual(values["openai_base_url"], "https://example.test/v1")

    def test_edit_writer_binds_all_sources_with_original_geometry_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(16):
                path = Path(directory, f"{i}.png")
                Image.new("RGB", (120, 80) if i % 2 == 0 else (80, 120)).save(path)
                paths.append(str(path))
            with patch("h3_app.prompt_writers.h3_image.common.write_prompt", return_value=("edit", "ok")) as write:
                enhance_h3_image_prompt("change only the cover", "model", "key", paths, 5472, 3072)
        values = write.call_args.kwargs
        self.assertEqual(values["system_path"], h3_image_prompt_path(True))
        self.assertEqual([str(path) for _, path in values["media_values"]], paths)
        self.assertEqual(values["media_values"][0][0], "<Picture 1> (original display aspect ratio 3:2)")
        self.assertEqual(values["media_values"][-1][0], "<Picture 16> (original display aspect ratio 2:3)")

    def test_invalid_inputs_preserve_original_prompt_without_requesting_a_rewrite(self):
        with patch("h3_app.prompt_writers.h3_image.common.write_prompt") as write:
            for references, width in ((["missing.png"], 4096), (["x"] * 17, 4096), ([], 0),
                                      ([], None), ([], 4096.5), ([], True), ([None], 4096)):
                original, status = enhance_h3_image_prompt("original", "model", "key", references, width, 4096)
                self.assertEqual(original, "original")
                self.assertIn("failed", status)
            write.assert_not_called()

    def test_shared_image_action_routes_to_image_writer_and_ignores_inactive_references(self):
        options = dict(h3_width=4096, h3_height=4096, writer_backend="Gemini", writer_openai_key="",
                       writer_openai_model="model", writer_base_url=None, writer_model="model", writer_key="key")
        with patch("h3_app.prompt_writers.h3_image.enhance_h3_image_prompt", return_value=("new", "ok")) as write:
            self.assertEqual(suggest_prompt({}, "h3", "Text to image", "request", ["inactive.png"], options),
                             ("new", "ok"))
        self.assertEqual(write.call_args.args[3:6], ([], 4096, 4096))

    def test_legacy_image_prompt_action_also_uses_the_image_recipe(self):
        import inspect
        from h3_app.prompt_writers.h3 import enhance_h3_prompt

        values = {name: None for name, parameter in inspect.signature(enhance_h3_prompt).parameters.items()
                  if parameter.default is inspect.Parameter.empty}
        values.update(prompt="three bowls", backend="OpenAI-compatible", mode="Text to video",
                      result_format="Image", width=4096, height=4096, runtime=object())
        with patch("h3_app.prompt_writers.h3_image.enhance_h3_image_prompt", return_value=("still", "ok")) as write:
            self.assertEqual(enhance_h3_prompt(**values), ("still", "ok"))
        self.assertEqual(write.call_args.args[0], "three bowls")
        self.assertEqual(write.call_args.args[3:6], ([], 4096, 4096))


if __name__ == "__main__":
    unittest.main()
