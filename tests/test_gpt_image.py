from __future__ import annotations

import base64
import dataclasses
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.gallery_store import generated_image_family
from h3_app.generation.gpt_image import (
    GptImageRateLimiter,
    GptImageRequest,
    generate_gpt_image,
    plan_gpt_image_slots,
    validate_gpt_image,
)
from h3_app.gpt_image import (
    DEFAULT_GPT_IMAGE_MODEL,
    GPT_IMAGE_ENV_NAMES,
    GPT_IMAGE_MODELS,
    GPT_IMAGE_QUALITY,
    GptImageCredentials,
    gpt_image_available,
    gpt_image_credentials,
    gpt_image_env,
    gpt_image_failback_available,
    gpt_image_max_concurrency,
    normalize_gpt_image_models,
    normalize_gpt_image_url,
    read_dotenv,
    validate_gpt_image_canvas,
    validate_gpt_image_quality,
)
from h3_app.jobs import CURRENT_JOB, Job, JobCancelled
from h3_app.provenance import read_snapshot


def _png_b64(width: int = 16, height: int = 16, color: tuple[int, int, int] = (20, 90, 160)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_png(path: Path, width: int = 32, height: int = 32, color: tuple[int, int, int] = (200, 60, 40)) -> Path:
    Image.new("RGB", (width, height), color).save(path, format="PNG")
    return path


def _test_runtime(root: Path) -> RuntimeConfig:
    base = RuntimeConfig.from_environment(root, {})
    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(base, output_root=output_root)


class GptImageConfigTests(unittest.TestCase):
    def test_dotenv_loading_and_env_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text(
                "\n".join(
                    [
                        "# comment line",
                        "gpt_image_api_url = https://primary.example/v1",
                        'export GPT_IMAGE_API_KEY=\'pri#mary-secret\' # trailing comment',
                        "GPT_IMAGE_API_FAILBACK_URL=https://backup.example/v1#frag # inline note",
                        'GPT_IMAGE_API_FAILBACK_KEY="backup-secret" # inline note',
                    ]
                ),
                encoding="utf-8",
            )
            parsed = read_dotenv(dotenv)
            self.assertEqual(parsed["GPT_IMAGE_API_URL"], "https://primary.example/v1")
            self.assertEqual(parsed["GPT_IMAGE_API_KEY"], "pri#mary-secret")
            self.assertEqual(parsed["GPT_IMAGE_API_FAILBACK_URL"], "https://backup.example/v1#frag")
            self.assertEqual(parsed["GPT_IMAGE_API_FAILBACK_KEY"], "backup-secret")

            creds = gpt_image_credentials(env={}, dotenv_path=dotenv)
            self.assertTrue(creds.primary_available)
            self.assertTrue(creds.failback_available)
            self.assertTrue(creds.available)
            self.assertEqual(len(creds.endpoints(False)), 1)
            self.assertEqual(len(creds.endpoints(True)), 2)

            self.assertEqual(
                gpt_image_env("GPT_IMAGE_API_KEY", env={"GPT_IMAGE_API_KEY": "override"}, dotenv_path=dotenv),
                "override",
            )
            self.assertEqual(
                gpt_image_env("GPT_IMAGE_API_KEY", env={"gpt_image_api_key": "lower-override"}, dotenv_path=dotenv),
                "lower-override",
            )
            self.assertEqual(
                gpt_image_env("GPT_IMAGE_API_KEY", env={"GPT_IMAGE_API_KEY": ""}, dotenv_path=dotenv),
                "",
            )
            with patch.dict(os.environ, {"H3_DOTENV_PATH": str(dotenv)}, clear=True):
                self.assertTrue(gpt_image_available())
                self.assertTrue(gpt_image_failback_available())

    def test_missing_credentials_mark_gpt_image_unavailable(self):
        blank = {name: "" for name in GPT_IMAGE_ENV_NAMES}
        with tempfile.TemporaryDirectory() as tmp:
            missing_dotenv = Path(tmp) / "missing.env"
            self.assertFalse(gpt_image_available(env=blank, dotenv_path=missing_dotenv))
            self.assertFalse(gpt_image_failback_available(env=blank, dotenv_path=missing_dotenv))
            partial = {**blank, "GPT_IMAGE_API_URL": "https://primary.example"}
            self.assertFalse(gpt_image_available(env=partial, dotenv_path=missing_dotenv))

    def test_url_normalization(self):
        self.assertEqual(normalize_gpt_image_url("https://api.example.com/"), "https://api.example.com")
        self.assertEqual(normalize_gpt_image_url(" https://api.example.com/v1/ "), "https://api.example.com/v1")

    def test_models_concurrency_canvas_and_quality_rules(self):
        with self.assertRaises(H3Error):
            normalize_gpt_image_models(None)
        with self.assertRaises(H3Error):
            normalize_gpt_image_models(())
        with self.assertRaises(H3Error):
            normalize_gpt_image_models(("unknown-model",))
        self.assertEqual(normalize_gpt_image_models("gpt-image-2"), ("gpt-image-2",))
        self.assertEqual(
            set(normalize_gpt_image_models(["gpt-image-2.5-flare", "gpt-image-2", "gpt-image-2"])),
            {"gpt-image-2", "gpt-image-2.5-flare"},
        )
        self.assertEqual(len(normalize_gpt_image_models(["gpt-image-2.5-flare", "gpt-image-2", "gpt-image-2"])), 2)

        self.assertEqual(gpt_image_max_concurrency(["gpt-image-2.5-flare"], False), 2)
        self.assertEqual(gpt_image_max_concurrency(["gpt-image-2.5-sunburst", "gpt-image-2.5-flare"], False), 4)
        self.assertEqual(gpt_image_max_concurrency(GPT_IMAGE_MODELS, False), 6)
        self.assertEqual(gpt_image_max_concurrency(["gpt-image-2.5-flare"], True), 4)
        self.assertEqual(gpt_image_max_concurrency(["gpt-image-2.5-sunburst", "gpt-image-2.5-flare"], True), 8)
        self.assertEqual(gpt_image_max_concurrency(GPT_IMAGE_MODELS, True), 12)

        self.assertEqual(validate_gpt_image_canvas(1920, 1088), (1920, 1088))
        self.assertEqual(validate_gpt_image_canvas(1088, 1920), (1088, 1920))
        with self.assertRaises(H3Error):
            validate_gpt_image_canvas(1024, 1024)
        with self.assertRaises(H3Error):
            validate_gpt_image_canvas(1920, 1080)

        self.assertEqual(validate_gpt_image_quality(GPT_IMAGE_QUALITY), "low")
        with self.assertRaises(H3Error):
            validate_gpt_image_quality("high")


class GptImageRateLimiterTests(unittest.TestCase):
    def test_same_url_credentials_have_independent_rate_limits_and_slots(self):
        creds = GptImageCredentials(
            api_url="https://shared.example/v1", api_key="primary-key",
            failback_url="https://shared.example/v1", failback_key="backup-key",
        )
        now = [0.0]
        limiter = GptImageRateLimiter(
            clock=lambda: now[0], sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
        )
        model = DEFAULT_GPT_IMAGE_MODEL
        endpoints = creds.endpoints(True)
        plan = plan_gpt_image_slots((model,), endpoints, 4, limiter=limiter)
        self.assertEqual([ep.name for ep, _ in plan], ["primary", "primary", "failback", "failback"])
        for endpoint, selected_model in plan:
            self.assertEqual(limiter.acquire(
                endpoint.url, selected_model, api_key=endpoint.api_key, check_cancel=lambda: None,
            ), 0.0)
        self.assertEqual(now[0], 0.0)
        for endpoint in endpoints:
            self.assertEqual(limiter.available_slots(endpoint.url, model, api_key=endpoint.api_key), 0)
        self.assertGreater(limiter.acquire(
            endpoints[0].url, model, api_key=endpoints[0].api_key, check_cancel=lambda: None,
        ), 0.0)

    def test_duplicate_credentials_do_not_advertise_extra_planner_capacity(self):
        creds = GptImageCredentials(
            api_url="https://shared.example/v1", api_key="same-key",
            failback_url="https://shared.example/v1", failback_key="same-key",
        )
        self.assertEqual(
            gpt_image_max_concurrency((DEFAULT_GPT_IMAGE_MODEL,), True, credentials=creds),
            2,
        )
        with self.assertRaisesRegex(H3Error, "between 1 and 2"):
            validate_gpt_image(
                GptImageRequest(
                    prompt="fox",
                    models=(DEFAULT_GPT_IMAGE_MODEL,),
                    use_failback=True,
                    batch_count=4,
                ),
                credentials=creds,
            )
        with self.assertRaisesRegex(H3Error, "at most 2"):
            plan_gpt_image_slots((DEFAULT_GPT_IMAGE_MODEL,), creds.endpoints(True), 4)

    def test_enforces_2_per_minute_per_model_per_endpoint_and_isolates_buckets(self):
        now = [100.0]
        sleeps: list[float] = []

        def fake_clock() -> float:
            return now[0]

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=fake_clock, sleeper=fake_sleep)
        ep1 = "https://primary.example/v1"
        ep2 = "https://failback.example/v1"
        flare = "gpt-image-2.5-flare"
        sunburst = "gpt-image-2.5-sunburst"

        self.assertEqual(limiter.available_slots(ep1, flare), 2)
        self.assertEqual(limiter.acquire(ep1, flare, check_cancel=lambda: None), 0.0)
        self.assertEqual(limiter.available_slots(ep1, flare), 1)
        self.assertEqual(limiter.acquire(ep1, flare, check_cancel=lambda: None), 0.0)
        self.assertEqual(limiter.available_slots(ep1, flare), 0)

        # Other models and other endpoints stay completely independent.
        self.assertEqual(limiter.available_slots(ep1, sunburst), 2)
        self.assertEqual(limiter.available_slots(ep2, flare), 2)
        self.assertEqual(limiter.acquire(ep1, sunburst, check_cancel=lambda: None), 0.0)
        self.assertEqual(limiter.acquire(ep2, flare, check_cancel=lambda: None), 0.0)

        # Third request on (ep1, flare) waits for the 60s window to expire.
        waited = limiter.acquire(ep1, flare, check_cancel=lambda: None)
        self.assertGreater(waited, 0.0)
        self.assertTrue(sleeps)
        self.assertGreaterEqual(now[0], 160.0)

    def test_wait_honors_job_cancellation(self):
        now = [0.0]

        def fake_sleep(seconds: float) -> None:
            now[0] += min(seconds, 1.0)

        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=lambda: now[0], sleeper=fake_sleep)
        limiter.acquire("https://p", "gpt-image-2", check_cancel=lambda: None)
        limiter.acquire("https://p", "gpt-image-2", check_cancel=lambda: None)

        checks = [0]

        def cancel_after_one() -> None:
            checks[0] += 1
            if checks[0] >= 2:
                raise JobCancelled("cancelled")

        with self.assertRaises(JobCancelled):
            limiter.acquire("https://p", "gpt-image-2", check_cancel=cancel_after_one)

    def test_slot_planner_scales_across_models_and_failback(self):
        creds = GptImageCredentials(
            api_url="https://primary.example/v1",
            api_key="pk",
            failback_url="https://backup.example/v1",
            failback_key="fk",
        )
        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=lambda: 0.0, sleeper=lambda _: None)
        endpoints = creds.endpoints(use_failback=True)
        plan = plan_gpt_image_slots(
            ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"),
            endpoints,
            8,
            limiter=limiter,
        )
        self.assertEqual(len(plan), 8)
        counts: dict[tuple[str, str], int] = {}
        for endpoint, model in plan:
            counts[(endpoint.name, model)] = counts.get((endpoint.name, model), 0) + 1
        self.assertEqual(
            counts,
            {
                ("primary", "gpt-image-2.5-sunburst"): 2,
                ("primary", "gpt-image-2.5-flare"): 2,
                ("failback", "gpt-image-2.5-sunburst"): 2,
                ("failback", "gpt-image-2.5-flare"): 2,
            },
        )


class GptImageGenerationTests(unittest.TestCase):
    def test_validate_gpt_image_rejects_missing_creds_and_over_concurrency(self):
        missing = GptImageCredentials()
        with self.assertRaises(H3Error):
            validate_gpt_image(GptImageRequest(prompt="fox"), credentials=missing)

        primary_only = GptImageCredentials(api_url="https://primary.example/v1", api_key="pk")
        with self.assertRaises(H3Error):
            validate_gpt_image(
                GptImageRequest(
                    prompt="fox",
                    models=("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"),
                    use_failback=True,
                ),
                credentials=primary_only,
            )
        with self.assertRaises(H3Error):
            validate_gpt_image(
                GptImageRequest(
                    prompt="fox",
                    models=("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"),
                    use_failback=False,
                    batch_count=5,
                ),
                credentials=primary_only,
            )

    def test_concurrent_generate_writes_png_provenance_and_gallery_family(self):
        creds = GptImageCredentials(
            api_url="https://primary.example/v1",
            api_key="pk",
            failback_url="https://backup.example/v1",
            failback_key="fk",
        )
        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=lambda: 100.0, sleeper=lambda _: None)
        calls: list[dict[str, object]] = []

        class FakeImages:
            def __init__(self, base_url: str, api_key: str) -> None:
                self.base_url = base_url
                self.api_key = api_key

            def generate(self, **kwargs):
                calls.append({"mode": "generate", "base_url": self.base_url, "api_key": self.api_key, **kwargs})
                return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])

        class FakeOpenAI:
            def __init__(self, *, api_key: str, base_url: str, timeout: float | None = None, max_retries: int = 2) -> None:
                if max_retries != 0:
                    raise AssertionError("SDK retries bypass the rate limiter")
                self.images = FakeImages(base_url, api_key)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = _test_runtime(Path(tmp))
            request = GptImageRequest(
                prompt="neon harbor at dusk",
                models=("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"),
                use_failback=False,
                quality="low",
                width=1920,
                height=1088,
                batch_count=4,
                seed=7,
            )
            fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
            token = CURRENT_JOB.set(Job(owner="test", family="GPT Image"))
            try:
                with patch.dict("sys.modules", {"openai": fake_openai}):
                    updates = list(generate_gpt_image(request, runtime, credentials=creds, limiter=limiter))
            finally:
                CURRENT_JOB.reset(token)

            self.assertEqual(len(calls), 4)
            self.assertTrue(all(call["size"] == "1920x1088" for call in calls))
            self.assertTrue(all(call["quality"] == "low" for call in calls))
            models_used = {call["model"] for call in calls}
            self.assertEqual(models_used, {"gpt-image-2.5-sunburst", "gpt-image-2.5-flare"})

            final_images = updates[-1].output
            self.assertEqual(len(final_images), 4)
            for path_str in final_images:
                path = Path(path_str)
                self.assertTrue(path.exists())
                self.assertTrue(path.name.startswith("gpt_image_"))
                self.assertEqual(generated_image_family(path, runtime=runtime), "GPT Image")
                snap = read_snapshot(path)
                self.assertIsNotNone(snap)
                self.assertEqual(snap["family"], "GPT Image")
                self.assertEqual(snap["settings"]["quality"], "low")

    def test_edit_uses_reference_images_and_failback_retries_primary_failure(self):
        creds = GptImageCredentials(
            api_url="https://primary.example/v1",
            api_key="pk",
            failback_url="https://backup.example/v1",
            failback_key="fk",
        )
        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=lambda: 100.0, sleeper=lambda _: None)
        calls: list[dict[str, object]] = []

        class FakeImages:
            def __init__(self, base_url: str, api_key: str) -> None:
                self.base_url = base_url
                self.api_key = api_key

            def edit(self, **kwargs):
                calls.append({"base_url": self.base_url, "api_key": self.api_key, **kwargs})
                if "primary.example" in self.base_url:
                    raise RuntimeError("primary overloaded")
                return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])

        class FakeOpenAI:
            def __init__(self, *, api_key: str, base_url: str, timeout: float | None = None, max_retries: int = 2) -> None:
                if max_retries != 0:
                    raise AssertionError("SDK retries bypass the rate limiter")
                self.images = FakeImages(base_url, api_key)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runtime = _test_runtime(tmp_path)
            ref1 = _write_png(tmp_path / "ref1.png")
            ref2 = _write_png(tmp_path / "ref2.png")
            request = GptImageRequest(
                prompt="combine <image_1> and <image_2>",
                models=("gpt-image-2.5-flare",),
                use_failback=True,
                quality="low",
                width=1088,
                height=1920,
                batch_count=1,
                reference_images=(str(ref1), str(ref2)),
            )
            fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
            with patch.dict("sys.modules", {"openai": fake_openai}):
                updates = list(generate_gpt_image(request, runtime, credentials=creds, limiter=limiter))

            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(calls[0]["base_url"], "https://primary.example/v1")
            self.assertEqual(calls[-1]["base_url"], "https://backup.example/v1")
            self.assertEqual(calls[-1]["size"], "1088x1920")
            self.assertEqual(calls[-1]["quality"], "low")
            self.assertEqual(len(updates[-1].output), 1)
            self.assertTrue(Path(updates[-1].output[0]).exists())

    def test_partial_batch_failure_preserves_completed_image_and_reports_error(self):
        import threading

        creds = GptImageCredentials(api_url="https://primary.example/v1", api_key="pk")
        limiter = GptImageRateLimiter(limit=2, window_seconds=60.0, clock=lambda: 100.0, sleeper=lambda _: None)
        lock = threading.Lock()
        counter = [0]

        class FakeImages:
            def generate(self, **kwargs):
                with lock:
                    counter[0] += 1
                    idx = counter[0]
                if idx == 2:
                    raise RuntimeError("second request failed")
                return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])

        class FakeOpenAI:
            def __init__(self, *, api_key: str, base_url: str, timeout: float | None = None, max_retries: int = 2) -> None:
                if max_retries != 0:
                    raise AssertionError("SDK retries bypass the rate limiter")
                self.images = FakeImages()

        with tempfile.TemporaryDirectory() as tmp:
            runtime = _test_runtime(Path(tmp))
            request = GptImageRequest(
                prompt="studio portrait",
                models=("gpt-image-2.5-flare",),
                use_failback=False,
                quality="low",
                width=1920,
                height=1088,
                batch_count=2,
            )
            fake_openai = SimpleNamespace(OpenAI=FakeOpenAI)
            job = Job(owner="test", family="GPT Image")
            token = CURRENT_JOB.set(job)
            try:
                with patch.dict("sys.modules", {"openai": fake_openai}):
                    updates = list(generate_gpt_image(request, runtime, credentials=creds, limiter=limiter))
            finally:
                CURRENT_JOB.reset(token)

            self.assertEqual(len(updates[-1].output), 1)
            self.assertTrue(Path(updates[-1].output[0]).exists())
            self.assertTrue(updates[-1].status.startswith("Error:"))
            self.assertIn("second request failed", updates[-1].status)
            self.assertIn("second request failed", job.error or "")


if __name__ == "__main__":
    unittest.main()
