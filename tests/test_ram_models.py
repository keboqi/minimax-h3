"""Model checkpoints kept in RAM (H3_RAM_MODELS_DIR, set by the Colab notebook)."""

import contextlib
import io
import os
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import h3_models

GIB = 1024**3


def symlinks_work(folder: Path) -> bool:
    try:
        (folder / "probe-link").symlink_to(folder)
        (folder / "probe-link").unlink()
        return True
    except OSError:
        return False


class RamModelFixture:
    requires_symlinks = True
    def setUp(self):
        self._tmp = TemporaryDirectory(prefix="h3-ram-")
        self.root = Path(self._tmp.name)
        if self.requires_symlinks and not symlinks_work(self.root):
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



class RamModelTests(RamModelFixture, unittest.TestCase):
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

    def test_a_folder_download_lands_in_ram_unless_short_or_already_on_disk(self):
        target = self.disk / "models" / "drama3_serving" / "fishaudio--drama-3--e0a9"
        with h3_models.ram_directory(target, 40 * GIB, log=lambda _message: None) as in_ram:
            self.assertTrue(in_ram)
            (target / "model.safetensors").write_bytes(b"x")
        self.assertTrue(target.is_symlink())
        self.assertTrue((self.ram / "drama3_serving" / target.name / "model.safetensors").is_file())
        self.assertEqual(h3_models._ram_reserved, 0)
        # A runtime restart empties RAM: the dangling link goes, and a short RAM keeps it on disk.
        shutil.rmtree(self.ram)
        self.memory.stop()
        with patch.object(h3_models, "_available_memory", return_value=10 * GIB), \
                h3_models.ram_directory(target, 40 * GIB, log=lambda _message: None) as in_ram:
            self.assertFalse(in_ram)
            self.assertFalse(target.is_symlink() or target.exists())
        self.memory.start()
        target.mkdir()
        with h3_models.ram_directory(target, 40 * GIB, log=lambda _message: None) as in_ram:
            self.assertFalse(in_ram)  # a partial disk download keeps resuming there

    def test_an_unfinished_ram_download_resumes_only_while_the_rest_fits(self):
        target = self.disk / "models" / "drama3_serving" / "fishaudio--drama-3--e0a9"
        quiet = {"log": lambda _message: None}
        with h3_models.ram_directory(target, 40 * GIB, **quiet):
            (target / "part.safetensors").write_bytes(b"x" * 10)
        # The on-demand download (new=False) resumes in RAM while the rest fits.
        with h3_models.ram_directory(target, 40 * GIB, new=False, **quiet) as in_ram:
            self.assertTrue(in_ram)
            self.assertEqual(h3_models._ram_reserved, 40 * GIB - 10)
        self.assertEqual(h3_models._ram_reserved, 0)
        # Short of room for the rest, the partial RAM copy goes and the download moves to disk.
        with patch.object(h3_models, "_available_memory", return_value=10 * GIB), \
                h3_models.ram_directory(target, 40 * GIB, new=False, **quiet) as in_ram:
            self.assertFalse(in_ram)
            self.assertFalse(target.is_symlink() or target.exists())
        self.assertFalse((self.ram / "drama3_serving" / target.name).exists())

    def test_the_drama3_preload_downloads_its_checkpoint_into_ram(self):
        from h3_app.drama3.serving import server

        def snapshot_download(repo, revision, local_dir, token):
            (Path(local_dir) / "config.json").write_text("{}")

        settings = server.ServerSettings(venv=self.root / "venv", data_dir=self.root / "data",
                                         model_root=self.disk / "models" / "drama3_serving")
        hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            target = server.ensure_model(settings, log=lambda _message: None, ram=True)
        self.assertTrue(target.is_symlink())
        self.assertTrue((target / ".h3-complete").is_file())
        self.assertTrue(target.resolve().is_relative_to(self.ram.resolve()))

    def test_without_the_setting_nothing_goes_to_ram(self):
        h3_models.route_to_ram(["h3_model"])
        with patch.dict(os.environ, {h3_models.RAM_MODELS_ENV: ""}):
            self.assertIsNone(h3_models._reserve_ram(self.plan()))


class RamResumeTests(RamModelFixture, unittest.TestCase):
    requires_symlinks = False

    def test_invalid_completed_blob_is_discarded_for_a_clean_retry(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        cache = h3_models._model_download_cache(self.ram, plan)

        def bad_download(**kwargs):
            blob = Path(kwargs["cache_dir"]) / "blob"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"bad")
            return str(blob)

        hub = types.SimpleNamespace(hf_hub_download=bad_download)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.run_download(plan)
        self.assertFalse(cache.exists())
        self.assertEqual(h3_models._ram_reserved, 0)

    def plain_download(self):
        """Hub stand-in without symlinks, which Windows may not permit."""
        def fake(**_kwargs):
            blob = self.disk / "hf-cache" / "blob"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"weights")
            return str(blob)

        return patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=fake)})

    def shared_hub(self, shared_bytes):
        """A shared cache holding ``shared_bytes``, plus a network download of the right bytes."""
        transfers = []
        shared = self.disk / "hf-cache" / "shared-blob"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(shared_bytes)

        def fake(**kwargs):
            if kwargs.get("local_files_only"):
                return str(shared)
            transfers.append(kwargs["cache_dir"])
            blob = Path(kwargs["cache_dir"]) / "blob"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"weights")
            return str(blob)

        hub = types.SimpleNamespace(hf_hub_download=fake)
        return patch.dict(sys.modules, {"huggingface_hub": hub}), transfers

    def test_a_complete_shared_cache_file_is_reused_without_a_transfer(self):
        plan = self.plan()
        plan["cache_dir"] = h3_models._model_download_cache(self.disk / "models", plan)
        hub, transfers = self.shared_hub(b"weights")
        with hub:
            self.run_download(plan)
        self.assertEqual(transfers, [])
        self.assertEqual(plan["dest"].read_bytes(), b"weights")

    def test_an_invalid_shared_cache_file_is_ignored_and_kept(self):
        plan = self.plan()
        plan["cache_dir"] = h3_models._model_download_cache(self.disk / "models", plan)
        hub, transfers = self.shared_hub(b"short")
        with hub:
            self.run_download(plan)
        self.assertEqual(transfers, [str(plan["cache_dir"])])
        self.assertEqual(plan["dest"].read_bytes(), b"weights")
        self.assertTrue((self.disk / "hf-cache" / "shared-blob").exists())

    def partial_ram_transfer(self, plan):
        cache = h3_models._model_download_cache(self.ram, plan)
        partial = cache / "models--org--repo" / "blobs" / "abc.incomplete"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial")
        return cache

    def test_invalid_blob_in_private_disk_cache_is_discarded_for_a_clean_retry(self):
        plan = self.plan()
        plan["cache_dir"] = h3_models._model_download_cache(self.disk / "models", plan)

        def bad_download(**kwargs):
            blob = Path(kwargs["cache_dir"]) / "blob"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"bad")
            return str(blob)

        hub = types.SimpleNamespace(hf_hub_download=bad_download)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.run_download(plan)
        self.assertFalse(plan["cache_dir"].exists())

    def test_invalid_blob_in_shared_hub_cache_is_left_alone(self):
        plan = self.plan()
        blob = self.disk / "hf-cache" / "blob"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"bad")
        hub = types.SimpleNamespace(hf_hub_download=lambda **_kwargs: str(blob))
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.run_download(plan)
        self.assertTrue(blob.exists())

    def test_disk_fallback_releases_the_partial_ram_transfer_it_replaced(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        cache = self.partial_ram_transfer(plan)
        self.memory.stop()
        with patch.object(h3_models, "_available_memory", return_value=10 * GIB):
            with self.plain_download():
                self.run_download(plan)
        self.memory.start()
        self.assertEqual(plan["dest"].read_bytes(), b"weights")
        self.assertFalse(cache.parent.exists())

    def test_failed_disk_fallback_keeps_the_partial_ram_transfer(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        cache = self.partial_ram_transfer(plan)
        self.memory.stop()
        hub = types.SimpleNamespace(hf_hub_download=Mock(side_effect=OSError("network")))
        with (patch.object(h3_models, "_available_memory", return_value=10 * GIB),
              patch.dict(sys.modules, {"huggingface_hub": hub}),
              self.assertRaisesRegex(OSError, "network")):
            self.run_download(plan)
        self.memory.start()
        self.assertTrue(cache.exists())

    def test_a_changed_source_drops_superseded_partial_transfers_before_reserving(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        superseded = self.partial_ram_transfer(plan)
        plan["revision"] = "rev2"
        current = self.partial_ram_transfer(plan)
        self.assertEqual(superseded.parent, current.parent)

        def reserve(_plan):
            self.assertFalse(superseded.exists())
            self.assertTrue(current.exists())
            return None

        with patch.object(h3_models, "_reserve_ram", side_effect=reserve), self.plain_download():
            self.run_download(plan)

    def test_reservations_in_other_processes_reduce_available_headroom(self):
        @contextlib.contextmanager
        def shared(_root):
            yield {str(os.getpid() + 1000): 40 * GIB}

        with patch.object(h3_models, "_shared_ram_reservations", shared):
            self.assertIsNone(h3_models.reserve_ram_bytes(40 * GIB))
            self.assertEqual(h3_models._ram_reserved, 0)

    def test_linux_reservation_ledger_drops_dead_processes(self):
        self.ram.mkdir()
        ledger = self.ram / ".h3-ram-reservations.json"
        h3_models.write_json_atomic(ledger, {"123": 100, "456": 200, "invalid": 1})
        exists = Path.exists

        def process_exists(path):
            if path.as_posix().startswith("/proc/"):
                return path.as_posix() == "/proc/123"
            return exists(path)

        # Exercise the Linux ledger on Windows too; the OS lock itself has
        # separate cross-process regression coverage.
        linux_os = types.SimpleNamespace(name="posix", replace=os.replace)
        with (patch.object(h3_models, "os", linux_os),
              patch.object(h3_models, "model_manifest_lock", return_value=contextlib.nullcontext()),
              patch.object(Path, "exists", process_exists)):
            with h3_models._shared_ram_reservations(self.ram) as reservations:
                self.assertEqual(reservations, {"123": 100})
                reservations["789"] = 300
        self.assertEqual(h3_models.read_json(ledger, {}), {"123": 100, "789": 300})

    def test_interrupted_transfer_keeps_partial_cache_and_retry_reuses_it(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan(payload=b"complete weights")
        cache = h3_models._model_download_cache(self.ram, plan)
        partial = cache / "models--org--repo" / "blobs" / "abc.incomplete"

        def interrupted(**kwargs):
            self.assertEqual(Path(kwargs["cache_dir"]), cache)
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"complete ")
            raise KeyboardInterrupt()

        hub = types.SimpleNamespace(hf_hub_download=interrupted)
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaises(KeyboardInterrupt):
            self.run_download(plan)
        self.assertTrue(partial.is_file())
        self.assertFalse(plan["dest"].exists())
        self.assertEqual(h3_models._ram_reserved, 0)

        def resumed(**kwargs):
            self.assertEqual(Path(kwargs["cache_dir"]), cache)
            self.assertEqual(partial.read_bytes(), b"complete ")
            partial.write_bytes(partial.read_bytes() + b"weights")
            return str(partial)

        hub.hf_hub_download = resumed
        # Exercise retry/publication even on Windows without symlink rights.
        def link_file(path, target, target_is_directory=False):
            os.link(target, path)

        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(Path, "symlink_to", link_file):
            self.run_download(plan)
        self.assertEqual(plan["dest"].read_bytes(), b"complete weights")
        self.assertFalse(cache.exists())
        self.assertEqual(h3_models._ram_reserved, 0)

    def test_different_revisions_never_resume_each_others_partial_bytes(self):
        plan = self.plan()
        cache = h3_models._model_download_cache(self.ram, plan)
        plan["revision"] = "rev2"
        self.assertNotEqual(cache, h3_models._model_download_cache(self.ram, plan))

    def test_partial_cache_reserves_only_remaining_ram(self):
        h3_models.route_to_ram(["h3_model"])
        plan = self.plan()
        plan["size"] = 40 * GIB
        cache = h3_models._model_download_cache(self.ram, plan)
        # Simulate already allocated bytes without writing gigabytes of test data.
        with patch.object(h3_models, "_directory_bytes", return_value=10 * GIB) as count:
            self.assertEqual(h3_models._reserve_ram(plan), self.ram)
        count.assert_called_once_with(cache)
        self.assertEqual(h3_models._ram_reserved, 30 * GIB)
        h3_models._release_ram(plan["ram_reserved_bytes"])
        self.assertEqual(h3_models._ram_reserved, 0)



if __name__ == "__main__":
    unittest.main()
