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
from h3_app.storyboard.text import SCENE_SEPARATOR, parse_scenes, split_scenes
from h3_ui.storyboard.handlers import WRITER_FIELDS, StoryboardHandlers


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
                    {"seconds": 12, "prompt": "Mei climbs.\n\nWind whistles."},
                    {"seconds": 6, "prompt": "Mei reaches the roof."}])}]}}]}
            result = writer.rewrite_story("Mei escapes", "flow", 10, 2, **self.settings())
        # Each scene's chosen length is on the separator line before it.
        self.assertEqual(result, "---12---\n\nMei climbs.\n\nWind whistles.\n\n---6---\n\nMei reaches the roof.")
        self.assertEqual(parse_scenes(result), [("Mei climbs.\n\nWind whistles.", 12), ("Mei reaches the roof.", 6)])
        call = session.return_value.__enter__.return_value.generate.call_args
        self.assertIn("exactly 2 scenes", call.args[1][0]["text"])
        self.assertIn("usually about 10 seconds", call.args[1][0]["text"])
        self.assertIn("whole\nnumber of seconds from 5 to 15", call.kwargs["system_prompt"])
        self.assertIn("flowing continuous sequence", call.args[1][0]["text"])
        guidelines = self.settings()["runtime"].prompt_system_path.read_text(encoding="utf-8").strip()
        self.assertTrue(call.kwargs["system_prompt"].startswith(guidelines + "\n\n"))
        self.assertIn("Apply the MiniMax H3 guidelines above to EACH scene", call.kwargs["system_prompt"])

    def test_openai_uses_selected_model_and_endpoint(self):
        with patch.object(writer.openai_compatible, "complete", return_value='```json\n["One", "Two"]\n```') as complete:
            result = writer.rewrite_story("A chase", "cut", 5, 2, **self.settings("OpenAI-compatible"))
        # Plain strings get the usual length.
        self.assertEqual(parse_scenes(result), [("One", 5), ("Two", 5)])
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
        reply = json.dumps([{"seconds": 4, "prompt": scene}, {"seconds": 20, "prompt": scene}])
        with patch.object(writer.openai_compatible, "complete", return_value=reply):
            result = writer.rewrite_story("Idea", "flow", 5, 2, **self.settings("OpenAI-compatible"))
        # Lengths past the limits become the limits.
        self.assertEqual(parse_scenes(result), [(scene, 5), (scene, 15)])
        service = SimpleNamespace(create=lambda owner, name, profile, **kwargs: "created")
        handlers = StoryboardHandlers(SimpleNamespace(
            service=lambda: service, request_defaults=request_defaults))
        with patch.object(service, "create", wraps=service.create) as create:
            handlers.create("owner", "Story", result, "flow", 10)
        created = create.call_args.kwargs["scenes"]
        self.assertEqual([(item["prompt"], item["seconds"]) for item in created], [(scene, 5), (scene, 15)])

    def test_only_standalone_separator_lines_split_scenes(self):
        first = "Description with --- in dialogue.\n\nSoundscape."
        self.assertEqual(split_scenes(first), [first])
        self.assertEqual(split_scenes(first + "\r\n \t--- \t\r\nSecond\r\n\r\nSound."),
                         [first, "Second\r\n\r\nSound."])
        self.assertEqual(split_scenes("---\nFirst\n---\n---\nSecond\n---"), ["First", "Second"])
        self.assertEqual(split_scenes("Rate ---6--- inline.\n---six---"), ["Rate ---6--- inline.\n---six---"])

    def test_timed_separators_set_the_next_scene_length(self):
        text = "---8---\nFirst\n\n---\nSecond\n --- 12 --- \r\nThird\n---6---\n---\nFourth"
        self.assertEqual(parse_scenes(text), [("First", 8), ("Second", None), ("Third", 12), ("Fourth", 6)])
        service = SimpleNamespace(create=lambda owner, name, profile, **kwargs: "created")
        handlers = StoryboardHandlers(SimpleNamespace(service=lambda: service, request_defaults=request_defaults))
        with patch.object(service, "create", wraps=service.create) as create:
            handlers.create("owner", "Story", text, "flow", 10)
            self.assertEqual([item["seconds"] for item in create.call_args.kwargs["scenes"]], [8, 10, 12, 6])
            with self.assertRaisesRegex(H3Error, "Scene 2 asks for 20 s"):
                handlers.create("owner", "Story", "One\n---20---\nTwo", "flow", 10)

    def test_bad_replies_and_empty_input_are_rejected(self):
        for reply in ('[]', '["Only one"]', '["One", ""]', '["One", 2]', 'not JSON', '{"scenes": []}',
                      json.dumps(["One\n---\nHidden scene", "Two"]), json.dumps(["One\n---7---\nHidden", "Two"]),
                      json.dumps([{"seconds": 6.5, "prompt": "One"}, "Two"]), json.dumps([{"seconds": 6}, "Two"])):
            with self.subTest(reply=reply), patch.object(writer.openai_compatible, "complete", return_value=reply):
                with self.assertRaises(H3Error):
                    writer.rewrite_story("Idea", "flow", 5, 2, **self.settings("OpenAI-compatible"))
        with patch.object(writer.gemini, "Session") as session:
            with self.assertRaises(H3Error):
                writer.rewrite_story(" ", "flow", 5, 2, **self.settings())
            session.assert_not_called()

    def test_handler_passes_the_runtime_and_the_page_reports_a_failed_rewrite(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from h3_ui.storyboard.page import install_storyboard_page

        runtime = self.settings()["runtime"]
        handlers = StoryboardHandlers(SimpleNamespace(
            service=lambda: SimpleNamespace(runtime=lambda: runtime), enhance_prompt=None))
        app = FastAPI()
        install_storyboard_page(app, handlers, gpu=False)
        client = TestClient(app)
        with patch("h3_ui.storyboard.page.request_owner", return_value="owner"),                 patch("h3_ui.storyboard.handlers.rewrite_story") as rewrite:
            rewrite.side_effect = OSError("Service unavailable")
            failed = client.post("/storyboard/api/rewrite", json={"story": "Original", "count": 2})
            self.assertEqual(failed.status_code, 400)
            self.assertIn("Service unavailable", failed.json()["error"])
            rewrite.side_effect = None
            rewrite.return_value = "One" + SCENE_SEPARATOR + "Two"
            result = client.post("/storyboard/api/rewrite", json={"story": "Original", "count": 2})
            self.assertEqual(result.json()["story"], "One" + SCENE_SEPARATOR + "Two")
            self.assertIs(rewrite.call_args.kwargs["runtime"], runtime)

    def test_studio_enhance_uses_the_h3_writer_settings_it_captured(self):
        from h3_app.storyboard import PROFILE_FIELDS
        from h3_ui.storyboard.tab import build_storyboard_tab

        captured = {}
        with gr.Blocks() as ui:
            defaults = request_defaults()
            h3_values = {name: gr.State(defaults[name]) for name in PROFILE_FIELDS}
            h3_values.update({name: gr.State(None) for name in WRITER_FIELDS})
            handlers = StoryboardHandlers(None)
            handlers.capture_settings = lambda owner, profile, written: captured.update(writer=written, profile=profile)
            build_storyboard_tab(gr.Column(), ui, handlers, h3_values=h3_values, triggers=(ui.load,))
        capture = next(dep for dep in ui.fns.values() if dep.fn.__name__ == "capture")
        self.assertEqual(capture.inputs[-len(WRITER_FIELDS):], [h3_values[name] for name in WRITER_FIELDS])
        self.assertIs(capture.queue, False)


if __name__ == "__main__":
    unittest.main()
