import base64
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from h3_app.errors import H3Error
from h3_app.gpt_image import GptImageEndpoint
from h3_app.hosted_image import MAI_IMAGE, NANO_BANANA
from h3_app.generation.hosted_image import _gemini_image, _mai_image, generate_hosted_image


class HostedImageTests(unittest.TestCase):
    def test_gemini_reads_final_image_and_requests_native_tier(self):
        endpoint = GptImageEndpoint("Gemini", "https://gemini.example", "key")
        response = MagicMock(status_code=200, ok=True)
        response.json.return_value = {"candidates": [{"content": {"parts": [
            {"thought": True, "inlineData": {"data": base64.b64encode(b"preview").decode()}},
            {"inlineData": {"data": base64.b64encode(b"final").decode()}},
        ]}}]}
        with patch("h3_app.generation.hosted_image.requests.post", return_value=response) as post:
            data = _gemini_image(endpoint, "gemini-nano-banana-2.1", "edit", "4K",
                                 [("reference.png", b"reference", "image/png")])
        self.assertEqual(data, b"final")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["generationConfig"]["imageConfig"]["imageSize"], "4K")
        self.assertEqual(body["contents"][0]["parts"][1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "key")

    def test_mai_gateway_uses_openai_image_operations_and_disables_sdk_retries(self):
        endpoint = GptImageEndpoint("primary", "https://gateway.example/v1", "key")
        result = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"image").decode())])
        references = [("first.png", b"a", "image/png"), ("second.png", b"b", "image/png")]
        with patch("openai.OpenAI") as client_type:
            images = client_type.return_value.__enter__.return_value.images
            images.generate.return_value = images.edit.return_value = result
            model = "MAI-Image-2.7-Flash"
            self.assertEqual(_mai_image(endpoint, model, "create", "1K", []), b"image")
            self.assertEqual(_mai_image(endpoint, model, "edit", "2K", references), b"image")
        images.generate.assert_called_once_with(model=model, prompt="create", size="1024x1024", n=1)
        images.edit.assert_called_once_with(model=model, prompt="edit", size="2048x2048", n=1, image=references)
        self.assertEqual(client_type.call_args.kwargs, {"base_url": endpoint.url, "api_key": "key",
                                                        "timeout": 600, "max_retries": 0})

    def test_image_job_uses_provider_quota_and_keeps_provider_output_dimensions(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buffer, format="PNG")
        endpoint = GptImageEndpoint("test", "https://image.example", "key")
        cases = ((NANO_BANANA, "gemini-3.1-flash-lite-image", "_gemini_image", "1K"),
                 (NANO_BANANA, "gemini-nano-banana-2.1", "_gemini_image", "4K"),
                 (MAI_IMAGE, "MAI-Image-2.7-Flash", "_mai_image", "2K"),
                 (MAI_IMAGE, "MAI-Image-2.7", "_mai_image", "2K"))
        with TemporaryDirectory() as directory, patch("h3_app.hosted_image.hosted_image_endpoint", return_value=endpoint):
            root = Path(directory)
            with patch("h3_app.generation.hosted_image.RATE_LIMITERS") as limiters:
                limiter = limiters.__getitem__.return_value
                for engine, model, method, requested in cases:
                    with self.subTest(engine=engine, model=model):
                        with patch(f"h3_app.generation.hosted_image.{method}", return_value=buffer.getvalue()) as call:
                            updates = list(generate_hosted_image(engine, model, requested, 1, "test", [],
                                                                 SimpleNamespace(output_dir=root)))
                        call.assert_called_once_with(endpoint, model, "test", requested, [])
                        self.assertEqual(limiter.acquire.call_args.args, (endpoint.url, engine))
                        self.assertEqual(limiter.acquire.call_args.kwargs["api_key"], endpoint.api_key)
                        with Image.open(updates[-1][0][0]) as image:
                            self.assertEqual(image.size, (64, 48))

    def test_unsupported_sizes_are_rejected_before_an_api_call(self):
        for engine, model, tier in (
            (NANO_BANANA, "gemini-3.1-flash-lite-image", "2K"),
            (MAI_IMAGE, "MAI-Image-2.7-Flash", "4K"),
            (MAI_IMAGE, "MAI-Image-2.7", "4K"),
        ):
            with self.subTest(model=model), self.assertRaisesRegex(H3Error, "native output sizes"):
                list(generate_hosted_image(engine, model, tier, 1, "test", [], None))
