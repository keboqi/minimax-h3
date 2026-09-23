"""Lightning prompt writer contracts shared across the generation tabs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app import prompt_service
from h3_app.catalog import LIGHTNING_API_ROOT, LIGHTNING_PROMPT_MODEL


class RemotePromptLightningTests(unittest.TestCase):
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

    @staticmethod
    def completion(text: str) -> Mock:
        return Mock(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    @patch("openai.OpenAI")
    def test_qwen_uses_single_image_context_and_lightning_key(self, client: Mock) -> None:
        client.return_value.chat.completions.create.return_value = self.completion(
            "Rewritten edit instruction"
        )
        result, status = prompt_service.enhance_qwen_image21_prompt(
            "change the sky", "unused Gemini model", "", "Image edit",
            [str(self.image)], 1024, 1024, "Lightning AI", "temporary-key",
            runtime=self.runtime,
        )
        self.assertEqual(result, "Rewritten edit instruction")
        self.assertIn(LIGHTNING_PROMPT_MODEL, status)
        client.assert_called_once_with(
            base_url=LIGHTNING_API_ROOT, api_key="temporary-key", timeout=600.0
        )
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], LIGHTNING_PROMPT_MODEL)
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
        result, status = prompt_service.enhance_ltx25_prompt(
            "street scene", "unused", "", "Image to video", str(self.image),
            None, None, 5.0, 768, 512, "Lightning AI", "temporary-key",
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
        caption, lyrics, status = prompt_service.enhance_music3_prompt(
            "dream pop", "unused", "", "", str(self.image), None, None,
            "Lightning AI", "temporary-key", runtime=self.runtime,
        )
        self.assertEqual((caption, lyrics), ("Dream pop with layered vocals", "[Verse] New words"))
        self.assertIn("MiniMax Music 3", status)
        client.return_value.chat.completions.create.return_value = self.completion(
            "STYLE: Warm acoustic folk\nLYRICS: [Chorus] Keep singing"
        )
        style, lyrics, status = prompt_service.enhance_yue2_prompt(
            "folk", "unused", "", "", "full", 90,
            "Lightning AI", "temporary-key", runtime=self.runtime,
        )
        self.assertEqual((style, lyrics), ("Warm acoustic folk", "[Chorus] Keep singing"))
        self.assertIn("YuE2", status)

    def test_missing_key_preserves_original_text(self) -> None:
        with patch.dict(prompt_service.os.environ, {"LIGHTNING_API_KEY": ""}):
            style, lyrics, status = prompt_service.enhance_yue2_prompt(
                "Original style", "unused", "", "Original lyrics", "full", 90,
                "Lightning AI", "", runtime=self.runtime,
            )
        self.assertEqual((style, lyrics), ("Original style", "Original lyrics"))
        self.assertIn("Prompt enhancement failed", status)


if __name__ == "__main__":
    unittest.main()
