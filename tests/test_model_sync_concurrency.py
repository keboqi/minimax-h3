"""Transfers coordinate by destination while manifest commits remain independent."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h3_models as models


class ModelSyncConcurrencyTests(unittest.TestCase):
    def test_unrelated_transfer_commits_while_another_is_blocked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            slow_started, release = Event(), Event()
            specs = {key: models.ModelSpec("test/repo", "weights", key, "test")
                     for key in ("slow", "fast")}
            files = {key: SimpleNamespace(size=7, blob_id=key, lfs=None) for key in specs}
            caches = []

            def download(**kwargs):
                if kwargs["filename"] == "slow":
                    slow_started.set()
                    if not release.wait(5):
                        raise TimeoutError("test did not release slow transfer")
                cache = Path(kwargs["cache_dir"])
                caches.append(cache)
                cache.mkdir(parents=True, exist_ok=True)
                result = cache / "blob"
                result.write_bytes(b"weights")
                return str(result)

            def sync(key):
                return models.sync_models(root=root, manifest_path=root / "manifest.json",
                                          token="test", model_keys=[key])

            with (patch.dict(models.MODEL_SPECS, specs),
                  patch.object(models, "MIN_VALID_MODEL_BYTES", 0),
                  patch.object(models, "_fetch_repo", return_value=("revision", files)),
                  patch("huggingface_hub.hf_hub_download", side_effect=download),
                  ThreadPoolExecutor(max_workers=2) as pool):
                slow = pool.submit(sync, "slow")
                try:
                    self.assertTrue(slow_started.wait(5))
                    pool.submit(sync, "fast").result(timeout=3)
                    entries = json.loads((root / "manifest.json").read_text())["files"]
                    self.assertIn("weights/fast", entries)
                    self.assertNotIn("weights/slow", entries)
                finally:
                    release.set()
                slow.result(timeout=5)
            entries = json.loads((root / "manifest.json").read_text())["files"]
            self.assertEqual(set(entries), {"weights/slow", "weights/fast"})
            self.assertEqual(len(set(caches)), 2)

    def test_overlapping_requests_and_aliases_download_one_destination_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            started, release, second_metadata = Event(), Event(), Event()
            spec = models.ModelSpec("test/repo", "weights", "shared", "test")
            fetches = []
            calls = []

            def metadata(*args):
                fetches.append(args)
                if len(fetches) == 2:
                    second_metadata.set()
                return "revision", {"shared": SimpleNamespace(size=7, blob_id="blob", lfs=None)}

            def download(**kwargs):
                calls.append(kwargs)
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release transfer")
                cache = Path(kwargs["cache_dir"])
                cache.mkdir(parents=True, exist_ok=True)
                result = cache / "blob"
                result.write_bytes(b"weights")
                return str(result)

            def sync(key):
                return models.sync_models(root=root, manifest_path=root / "manifest.json",
                                          token="test", model_keys=[key])

            with (patch.dict(models.MODEL_SPECS, {"first": spec, "alias": spec}),
                  patch.object(models, "MIN_VALID_MODEL_BYTES", 0),
                  patch.object(models, "_fetch_repo", side_effect=metadata),
                  patch("huggingface_hub.hf_hub_download", side_effect=download),
                  ThreadPoolExecutor(max_workers=2) as pool):
                first = pool.submit(sync, "first")
                try:
                    self.assertTrue(started.wait(5))
                    second = pool.submit(sync, "alias")
                    self.assertTrue(second_metadata.wait(5))
                finally:
                    release.set()
                first.result(timeout=5)
                second.result(timeout=5)
            self.assertEqual(len(calls), 1)

    def test_failed_sibling_does_not_erase_a_successful_commit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            specs = {key: models.ModelSpec("test/repo", "weights", key, "test")
                     for key in ("good", "bad")}
            files = {key: SimpleNamespace(size=7, blob_id=key, lfs=None) for key in specs}

            def download(**kwargs):
                if kwargs["filename"] == "bad":
                    raise OSError("interrupted")
                result = Path(kwargs["cache_dir"]) / "blob"
                result.parent.mkdir(parents=True, exist_ok=True)
                result.write_bytes(b"weights")
                return str(result)

            with (patch.dict(models.MODEL_SPECS, specs),
                  patch.object(models, "MIN_VALID_MODEL_BYTES", 0),
                  patch.object(models, "_fetch_repo", return_value=("revision", files)),
                  patch("huggingface_hub.hf_hub_download", side_effect=download),
                  self.assertRaisesRegex(RuntimeError, "interrupted")):
                models.sync_models(root=root, manifest_path=root / "manifest.json",
                                   token="test", model_keys=specs)
            entries = json.loads((root / "manifest.json").read_text())["files"]
            self.assertEqual(set(entries), {"weights/good"})


if __name__ == "__main__":
    unittest.main()
