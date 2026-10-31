"""Model checkpoints kept in RAM (H3_RAM_MODELS_DIR, set by the Colab notebook)."""

import contextlib
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import h3_models

GIB = 1024**3


def symlinks_work(folder: Path) -> bool:
    try:
        (folder / "probe-link").symlink_to(folder)
        (folder / "probe-link").unlink()
        return True
    except OSError:
        return False


class RamModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory(prefix="h3-ram-")
        self.root = Path(self._tmp.name)
        if not symlinks_work(self.root):
            self.skipTest("symlinks unavailable")
        self.ram = self.root / "shm"
        self.disk = self.root / "disk"
        self.addCleanup(self._tmp.cleanup)
        env = patch.dict(os.environ, {h3_models.RAM_MODELS_ENV: str(self.ram)})
        env.start()
        self.addCleanup(env.stop)
        keys = patch.object(h3_models, "_ram_keys", set())
        keys.start()
        self.addCleanup(keys.stop)
        h3_models._ram_reserved = 0
        self.memory = patch.object(h3_models, "_available_memory", return_value=100 * GIB)
        self.memory.start()
        self.addCleanup(self.memory.stop)
        usage = patch.object(h3_models.shutil, "disk_usage",
                             return_value=h3_models.shutil._ntuple_diskusage(200 * GIB, 0, 80 * GIB))
        usage.start()
        self.addCleanup(usage.stop)

    def plan(self, key="h3_model", payload=b"weights"):
        dest = self.disk / "models" / "diffusion_models" / "model.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        spec = h3_models.ModelSpec(repo_id="org/repo", folder="diffusion_models",
                                   filename="model.safetensors", source="test")
        return {"key": key, "spec": spec, "dest": dest, "revision": "rev1", "size": len(payload)}

    def download(self, payload=b"weights"):
        """A stand-in for hf_hub_download that writes into whichever cache it is given."""
        calls = []

        def fake(repo_id, filename, revision, token, cache_dir=None):
            base = Path(cache_dir) if cache_dir else self.disk / "hf-cache"
            blob = base / "models--org--repo" / "blobs" / "abc"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(payload)
            snapshot = base / "models--org--repo" / "snapshots" / revision / filename
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.symlink_to(blob)
            calls.append(cache_dir)
            return str(snapshot)

        hub = types.SimpleNamespace(hf_hub_download=fake)
        return patch.dict(sys.modules, {"huggingface_hub": hub}), calls

    def run_download(self, plan):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            h3_models._download_model(plan, token=None, log_prefix="[test]")
        return output.getvalue()

    def test_a_routed_model_lands_in_ram_behind_a_symlink(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        fake, calls = self.download()
        with fake:
            output = self.run_download(plan)
        ram_file = self.ram / "diffusion_models" / "model.safetensors"
        self.assertTrue(plan["dest"].is_symlink())
        self.assertEqual(plan["dest"].resolve(), ram_file.resolve())
        self.assertEqual(plan["dest"].read_bytes(), b"weights")
        self.assertTrue(calls[0].startswith(str(self.ram)))  # the Hub cache was in RAM too
        self.assertFalse((self.ram / ".hf-download-cache" / "h3_model").exists())
        self.assertIn("into RAM", output)
        self.assertEqual(h3_models._ram_reserved, 0)

    def test_qwen_keys_always_qualify_and_other_keys_stay_on_disk(self):
        plan = self.plan(key="qwen_image21_transformer")
        fake, _calls = self.download()
        with fake:
            self.run_download(plan)
        self.assertTrue(plan["dest"].is_symlink())
        other = self.plan(key="ltx25_model")
        other["dest"].unlink()
        fake, calls = self.download()
        with fake:
            self.run_download(other)
        self.assertFalse(other["dest"].is_symlink())
        self.assertEqual(calls, [None])

    def test_short_of_ram_it_falls_back_to_disk_and_drops_an_old_ram_copy(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        old = self.ram / "diffusion_models" / "model.safetensors"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"old")
        plan["dest"].symlink_to(old)
        self.memory.stop()
        with patch.object(h3_models, "_available_memory", return_value=10 * GIB):
            fake, calls = self.download()
            with fake:
                output = self.run_download(plan)
        self.memory.start()
        self.assertIn("not enough free RAM; downloading to disk", output)
        self.assertFalse(plan["dest"].is_symlink())
        self.assertEqual(plan["dest"].read_bytes(), b"weights")
        self.assertFalse(old.exists())
        self.assertEqual(calls, [None])

    def test_parallel_downloads_reserve_space_before_any_bytes_arrive(self):
        h3_models.route_to_ram(["a", "b"])
        size = 40 * GIB  # 100 GiB available - 32 GiB headroom fits one, not two
        first = {"key": "a", "size": size}
        second = {"key": "b", "size": size}
        self.assertEqual(h3_models._reserve_ram(first), self.ram)
        self.assertIsNone(h3_models._reserve_ram(second))
        h3_models._release_ram(size)
        self.assertEqual(h3_models._reserve_ram(second), self.ram)

    def test_without_the_setting_nothing_goes_to_ram(self):
        h3_models.route_to_ram(["h3_model"])
        with patch.dict(os.environ, {h3_models.RAM_MODELS_ENV: ""}):
            self.assertIsNone(h3_models._reserve_ram(self.plan()))


if __name__ == "__main__":
    unittest.main()
