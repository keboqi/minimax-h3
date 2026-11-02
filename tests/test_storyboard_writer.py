"""Storyboard planning contracts, without network calls or GPU dependencies."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr

from h3_app.errors import H3Error
from h3_app.storyboard import request_defaults, writer
from h3_app.storyboard.text import SCENE_SEPARATOR, split_scenes
from h3_ui.storyboard.handlers import STORY_WRITER_FIELDS, WRITER_FIELDS, StoryboardHandlers


class StoryboardWriterTests(unittest.TestCase):
    def settings(self, backend="Gemini"):
        return dict(backend=backend, gemini_prompt_model="gemini-test", gemini_api_key="gemini-key",
                    openai_api_key="openai-key", openai_model="custom-model",
                    openai_base_url="https://example.test/v1",
                    runtime=SimpleNamespace(prompt_system_path=Path(__file__).resolve().parents[1]
                                            / "prompts" / "prompt.txt"))

    def test_gemini_writes_paragraphs_with_timing_and_continuity(self):
        with patch.object(writer.gemini, "Session") as session:
            session.return_value.__enter__.return_value.generate.return_value = {
                "candidates": [{"content": {"parts": [{"text": json.dumps([
                    "Mei climbs.\n\nWind whistles.", "Mei reaches the roof."])}]}}]}
            result = writer.rewrite_story("Mei escapes", "flow", 10, 2, **self.settings())
        self.assertEqual(split_scenes(result), ["Mei climbs.\n\nWind whistles.", "Mei reaches the roof."])
        call = session.return_value.__enter__.return_value.generate.call_args
        self.assertIn("exactly 2 scenes, each 10 seconds", call.args[1][0]["text"])
        self.assertIn("flowing continuous sequence", call.args[1][0]["text"])
        guidelines = self.settings()["runtime"].prompt_system_path.read_text(encoding="utf-8").strip()
        self.assertTrue(call.kwargs["system_prompt"].startswith(guidelines + "\n\n"))
        self.assertIn("Apply the MiniMax H3 guidelines above to EACH scene", call.kwargs["system_prompt"])

    def test_openai_uses_selected_model_and_endpoint(self):
        with patch.object(writer.openai_compatible, "complete", return_value='```json\n["One", "Two"]\n```') as complete:
            result = writer.rewrite_story("A chase", "cut", 5, 2, **self.settings("OpenAI-compatible"))
        self.assertEqual(result, "One" + SCENE_SEPARATOR + "Two")
        self.assertEqual(complete.call_args.args[:2], ("openai-key", "custom-model"))
        self.assertEqual(complete.call_args.args[4], "https://example.test/v1")
        self.assertIn("separate shots joined by cuts", complete.call_args.args[3][0]["text"])
        guidelines = self.settings()["runtime"].prompt_system_path.read_text(encoding="utf-8").strip()
        self.assertTrue(complete.call_args.args[2].startswith(guidelines + "\n\n"))

    def test_missing_or_empty_guidelines_fail_before_calling_the_writer(self):
        with TemporaryDirectory() as directory, patch.object(writer.openai_compatible, "complete") as complete:
            path = Path(directory) / "prompt.txt"
            settings = {**self.settings("OpenAI-compatible"),
                        "runtime": SimpleNamespace(prompt_system_path=path)}
            with self.assertRaisesRegex(H3Error, "Missing MiniMax H3 system prompt"):
                writer.rewrite_story("Idea", "flow", 5, 2, **settings)
            path.write_text("  ", encoding="utf-8")
            with self.assertRaisesRegex(H3Error, "system prompt is empty"):
                writer.rewrite_story("Idea", "flow", 5, 2, **settings)
            complete.assert_not_called()

    def test_h3_prompts_preserve_blank_lines_through_rewrite_and_creation(self):
        scene = ("integrated_multimodal_description: [Shot 1] Mei climbs.\n\n"
                 "overall_soundscape: Wind whistles.\n\nnon_diegetic_music: N/A")
        with patch.object(writer.openai_compatible, "complete", return_value=json.dumps([scene, scene])):
            result = writer.rewrite_story("Idea", "flow", 5, 2, **self.settings("OpenAI-compatible"))
        self.assertEqual(split_scenes(result), [scene] * 2)
        service = SimpleNamespace(create=lambda owner, name, profile, **kwargs: "created")
        handlers = StoryboardHandlers(SimpleNamespace(
            service=lambda: service, request_defaults=request_defaults))
        with patch.object(service, "create", wraps=service.create) as create, \
                patch.object(handlers, "_act", side_effect=lambda a, b, c, action, *args: action("owner")):
            handlers.create("Story", result, "flow", 10, {}, SimpleNamespace())
        self.assertEqual([item["prompt"] for item in create.call_args.kwargs["scenes"]], [scene] * 2)

    def test_only_standalone_separator_lines_split_scenes(self):
        first = "Description with --- in dialogue.\n\nSoundscape."
        self.assertEqual(split_scenes(first), [first])
        self.assertEqual(split_scenes(first + "\r\n \t--- \t\r\nSecond\r\n\r\nSound."),
                         [first, "Second\r\n\r\nSound."])
        self.assertEqual(split_scenes("---\nFirst\n---\n---\nSecond\n---"), ["First", "Second"])

    def test_bad_replies_and_empty_input_are_rejected(self):
        for reply in ('[]', '["Only one"]', '["One", ""]', '["One", 2]', 'not JSON', '{"scenes": []}',
                      json.dumps(["One\n---\nHidden scene", "Two"])):
            with self.subTest(reply=reply), patch.object(writer.openai_compatible, "complete", return_value=reply):
                with self.assertRaises(H3Error):
                    writer.rewrite_story("Idea", "flow", 5, 2, **self.settings("OpenAI-compatible"))
        with patch.object(writer.gemini, "Session") as session:
            with self.assertRaises(H3Error):
                writer.rewrite_story(" ", "flow", 5, 2, **self.settings())
            session.assert_not_called()

    def test_handler_keeps_input_on_failure_and_only_updates_draft_on_success(self):
        runtime = self.settings()["runtime"]
        handlers = StoryboardHandlers(SimpleNamespace(
            service=lambda: SimpleNamespace(runtime=lambda: runtime)))
        settings = {**self.settings(), "prompt_writer_backend": "Gemini"}
        values = [settings.get(field) for field in STORY_WRITER_FIELDS]
        with patch("h3_ui.storyboard.handlers.require_owner", return_value="owner"), \
                patch("h3_ui.storyboard.handlers.gr.Warning"), \
                patch("h3_ui.storyboard.handlers.rewrite_story") as rewrite:
            rewrite.side_effect = H3Error("Service unavailable")
            result = handlers.rewrite_story("Original", "flow", 10, 2, *values, SimpleNamespace())
            self.assertNotIn("new_story", result)
            self.assertIn("Story unchanged", result["rewrite_status"])
            rewrite.side_effect = None
            rewrite.return_value = "One" + SCENE_SEPARATOR + "Two"
            result = handlers.rewrite_story("Original", "flow", 10, 2, *values, SimpleNamespace())
            self.assertEqual(result["new_story"], "One" + SCENE_SEPARATOR + "Two")
            self.assertIs(rewrite.call_args.kwargs["runtime"], runtime)

    def test_workshop_writer_is_available_before_creating_a_board_and_bound(self):
        from h3_ui.storyboard.bindings import bind_storyboard
        from h3_ui.storyboard.view import build_storyboard_view

        with gr.Blocks() as ui:
            view = build_storyboard_view(gr.Column(), gpu=False)
            bind_storyboard(view, StoryboardHandlers(None), gpu=False)
        settings = view.writer["prompt_writer_backend"]
        parent = settings.parent
        while parent is not None:
            self.assertIsNot(parent, view.board_group)
            parent = getattr(parent, "parent", None)
        event = next(dep for dep in ui.fns.values() if (view.rewrite_story._id, "click") in dep.targets)
        self.assertEqual(event.inputs[:4], [view.new_story, view.new_connect, view.new_seconds, view.new_scene_count])
        self.assertEqual(event.outputs, [view.new_story, view.rewrite_status])

    def test_studio_rewrite_uses_its_hosted_writer_when_h3_uses_local(self):
        from h3_app.storyboard import PROFILE_FIELDS, request_defaults
        from h3_ui.storyboard.bindings import bind_storyboard
        from h3_ui.storyboard.view import build_storyboard_view

        with gr.Blocks() as ui:
            defaults = request_defaults()
            h3_values = {name: gr.State(defaults[name]) for name in PROFILE_FIELDS}
            h3_values.update({name: gr.State(None) for name in WRITER_FIELDS})
            h3_values["prompt_writer_backend"] = gr.State("Local MiniMax-H3 8B")
            view = build_storyboard_view(gr.Column(), gpu=True)
            bind_storyboard(view, StoryboardHandlers(None), h3_values, interrupt=lambda: None)
        rewrite = next(dep for dep in ui.fns.values() if (view.rewrite_story._id, "click") in dep.targets)
        enhance = next(dep for dep in ui.fns.values() if (view.enhance._id, "click") in dep.targets)
        self.assertTrue(view.rewrite_story.visible)
        self.assertEqual(rewrite.inputs[4:], [view.writer[name] for name in STORY_WRITER_FIELDS])
        self.assertEqual(view.writer["prompt_writer_backend"].value, "Gemini")
        self.assertEqual(enhance.inputs[2:], [h3_values[name] for name in WRITER_FIELDS])


if __name__ == "__main__":
    unittest.main()
