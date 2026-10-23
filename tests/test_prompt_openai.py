"""OpenAI-compatible prompt writer contracts shared across the generation tabs."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.prompt_writers import ltx as ltx_writer, qwen as qwen_writer, song as song_writer
from h3_app.catalog import OPENAI_API_ROOT, OPENAI_PROMPT_MODEL, OPENAI_PROMPT_MODELS


class RemotePromptOpenAITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.systems = {}
        for target in ("Qwen Image 2.1", "LTX-2.5", "MiniMax Music 3", "YuE2"):
            path = root / f"{target.replace(' ', '_')}.txt"
            path.write_text(f"Instructions for {target}", encoding="utf-8")
            self.systems[target] = path
        self.runtime = SimpleNamespace(prompt_systems=self.systems)
        self.image = root / "reference.png"
        self.image.write_bytes(b"image bytes")
        environment = patch.dict(os.environ, {"OPENAI_BASE_URL": ""})
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def completion(text: str) -> Mock:
        return Mock(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    @patch("openai.OpenAI")
    def test_qwen_uses_single_image_context_and_openai_key(self, client: Mock) -> None:
        client.return_value.chat.completions.create.return_value = self.completion(
            "Rewritten edit instruction"
        )
        result, status = qwen_writer.enhance_qwen_image21_prompt(
            "change the sky", "unused Gemini model", "", "Image edit",
            [str(self.image)], 1024, 1024, "OpenAI-compatible", "temporary-key",
            runtime=self.runtime,
        )
        self.assertEqual(result, "Rewritten edit instruction")
        self.assertIn(OPENAI_PROMPT_MODEL, status)
        client.assert_called_once_with(
            base_url=OPENAI_API_ROOT, api_key="temporary-key", timeout=600.0
        )
        request = client.return_value.chat.completions.create.call_args.kwargs
        # OpenAI's own API takes the router name without its openai/ prefix.
        self.assertEqual(request["model"], OPENAI_PROMPT_MODEL.removeprefix("openai/"))
        self.assertEqual(request["messages"][0]["content"], "Instructions for Qwen Image 2.1")
        content = request["messages"][1]["content"]
        self.assertIn("do not use an <image1> tag", content[0]["text"])
        self.assertEqual(content[2]["type"], "image_url")
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("openai.OpenAI")
    def test_ltx_preserves_keyframe_labels(self, client: Mock) -> None:
        client.return_value.chat.completions.create.return_value = self.completion(
            "A detailed camera move"
        )
        result, status = ltx_writer.enhance_ltx25_prompt(
            "street scene", "unused", "", "Image to video", str(self.image),
            None, None, 5.0, 768, 512, "OpenAI-compatible", "temporary-key",
            runtime=self.runtime,
        )
        self.assertEqual(result, "A detailed camera move")
        self.assertIn("1 image(s)", status)
        content = client.return_value.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Start keyframe", content[1]["text"])
        self.assertIn("Duration: 5.00 seconds", content[0]["text"])

    @patch("openai.OpenAI")
    def test_music_and_yue_keep_sectioned_output_parsing(self, client: Mock) -> None:
        client.return_value.chat.completions.create.return_value = self.completion(
            "CAPTION: Dream pop with layered vocals\nLYRICS: [Verse] New words"
        )
        caption, lyrics, status = song_writer.enhance_music3_prompt(
            "dream pop", "unused", "", "", str(self.image), None, None,
            "OpenAI-compatible", "temporary-key", runtime=self.runtime,
        )
        self.assertEqual((caption, lyrics), ("Dream pop with layered vocals", "[Verse] New words"))
        self.assertIn("MiniMax Music 3", status)
        client.return_value.chat.completions.create.return_value = self.completion(
            "STYLE: Warm acoustic folk\nLYRICS: [Chorus] Keep singing"
        )
        style, lyrics, status = song_writer.enhance_yue2_prompt(
            "folk", "unused", "", "", "full", 90,
            "OpenAI-compatible", "temporary-key", runtime=self.runtime,
        )
        self.assertEqual((style, lyrics), ("Warm acoustic folk", "[Chorus] Keep singing"))
        self.assertIn("YuE2", status)
        client.return_value.chat.completions.create.return_value = self.completion(
            "STYLE: Lo-fi instrumental\nLYRICS: N/A"
        )
        style, lyrics, _status = song_writer.enhance_yue2_prompt(
            "lo-fi, no vocals", "unused", "", "", "full", 90,
            "OpenAI-compatible", "temporary-key", runtime=self.runtime,
        )
        self.assertEqual(style, "Lo-fi instrumental")
        self.assertIn("[Instrumental]", lyrics)

    @patch("openai.OpenAI")
    def test_base_url_env_selects_compatible_endpoint_and_models(self, client: Mock) -> None:
        self.assertEqual(OPENAI_PROMPT_MODELS[2:], (
            "deepseek-v4.1-flash", "deepseek-v4.1-flash-uncensored", "glm-5.3-flash"))
        client.return_value.chat.completions.create.return_value = self.completion("A rewritten shot")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": " https://llm.example.test/v1 "}):
            result, status = ltx_writer.enhance_ltx25_prompt(
                "street scene", "unused", "", "Text to video", None,
                None, None, 5.0, 768, 512, "OpenAI-compatible", "temporary-key", "glm-5.3-flash",
                runtime=self.runtime,
            )
        self.assertEqual(result, "A rewritten shot")
        self.assertIn("glm-5.3-flash", status)
        client.assert_called_once_with(
            base_url="https://llm.example.test/v1", api_key="temporary-key", timeout=600.0
        )
        self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs["model"], "glm-5.3-flash")
        client.reset_mock()
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://llm.example.test/v1"}):
            ltx_writer.enhance_ltx25_prompt(
                "street scene", "unused", "", "Text to video", None,
                None, None, 5.0, 768, 512, "OpenAI-compatible", "temporary-key", "deepseek-v4.1-flash",
                " https://ui.example.test/v1 ", runtime=self.runtime,
            )
        client.assert_called_once_with(
            base_url="https://ui.example.test/v1", api_key="temporary-key", timeout=600.0
        )

    @patch("openai.OpenAI")
    def test_temporary_endpoint_never_receives_the_server_key(self, client: Mock) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "server-key"}):
            result, status = ltx_writer.enhance_ltx25_prompt(
                "street scene", "unused", "", "Text to video", None,
                None, None, 5.0, 768, 512, "OpenAI-compatible", "", "glm-5.3-flash",
                "https://attacker.example.test/v1", runtime=self.runtime,
            )
            self.assertEqual(result, "street scene")
            self.assertIn("temporary API key", status)
            client.assert_not_called()
            ltx_writer.enhance_ltx25_prompt(
                "street scene", "unused", "", "Text to video", None,
                None, None, 5.0, 768, 512, "OpenAI-compatible", "", "glm-5.3-flash",
                runtime=self.runtime,
            )
        client.assert_called_once_with(base_url=OPENAI_API_ROOT, api_key="server-key", timeout=600.0)

    def test_missing_key_preserves_original_text(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            style, lyrics, status = song_writer.enhance_yue2_prompt(
                "Original style", "unused", "", "Original lyrics", "full", 90,
                "OpenAI-compatible", "", runtime=self.runtime,
            )
        self.assertEqual((style, lyrics), ("Original style", "Original lyrics"))
        self.assertIn("Prompt enhancement failed", status)

    def test_saved_lightning_writer_preferences_carry_over(self) -> None:
        from h3_ui.persistence import restore_preferences

        backends = ("OpenAI-compatible", "Gemini")
        components = {
            "flux3.prompt_backend": SimpleNamespace(value="Gemini", choices=backends),
            "h3.prompt_writer_backend": SimpleNamespace(value="Gemini", choices=("Local MiniMax-H3 8B", *backends)),
            "flux3.openai_model": SimpleNamespace(value=OPENAI_PROMPT_MODEL, choices=OPENAI_PROMPT_MODELS),
            "drama3.openai_model": SimpleNamespace(value=OPENAI_PROMPT_MODEL, choices=OPENAI_PROMPT_MODELS,
                                                   allow_custom_value=True),
        }
        restored, _ = restore_preferences({"values": {
            "flux3.prompt_backend": "Lightning AI",
            "h3.prompt_writer_backend": "Lightning AI",
            "flux3.lightning_model": "openai/gpt-6.1-sol",
            "drama3.openai_model": "my-model",
        }}, components)
        self.assertEqual(restored, {
            "flux3.prompt_backend": "OpenAI-compatible",
            "h3.prompt_writer_backend": "OpenAI-compatible",
            "flux3.openai_model": "openai/gpt-6.1-sol",
            "drama3.openai_model": "my-model",  # typed models survive a reload
        })


if __name__ == "__main__":
    unittest.main()
