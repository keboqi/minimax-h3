import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import h3_models
from h3_app.swiftvr import ensure_swiftvr_checkpoint


class HfCacheCleanupTests(unittest.TestCase):
    def test_remove_hf_cache_cleans_snapshot_and_blob_without_touching_dest(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            hub = temp / "hub"
            repo_dir = hub / "models--testorg--testmodel"
            blobs_dir = repo_dir / "blobs"
            snapshots_dir = repo_dir / "snapshots" / "commit123"
            blobs_dir.mkdir(parents=True)
            snapshots_dir.mkdir(parents=True)
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text("commit123")
            (repo_dir / ".no_exist" / "commit123").mkdir(parents=True)
            (repo_dir / ".no_exist" / "commit123" / "missing.bin").touch()

            blob_sha = "aabbccddeeff11223344"
            blob_file = blobs_dir / blob_sha
            blob_file.write_bytes(b"model-weights-content")

            snapshot_file = snapshots_dir / "model.safetensors"
            try:
                snapshot_file.symlink_to(blob_file)
            except OSError:
                # Fallback to hardlink if symlinks not permitted
                os.link(blob_file, snapshot_file)

            comfy_models = temp / "ComfyUI" / "models" / "diffusion_models"
            comfy_models.mkdir(parents=True)
            dest = comfy_models / "model.safetensors"
            dest.write_bytes(b"model-weights-content")

            plan = {
                "dest": dest,
                "sha256": blob_sha,
                "blob_id": blob_sha,
            }

            self.assertTrue(snapshot_file.exists() or snapshot_file.is_symlink())
            self.assertTrue(blob_file.exists())
            self.assertTrue(dest.exists())

            h3_models._remove_hf_cache(snapshot_file, plan=plan)

            self.assertFalse(snapshot_file.exists())
            self.assertFalse(snapshot_file.is_symlink())
            self.assertFalse(blob_file.exists())
            self.assertFalse(repo_dir.exists())

            # Crucial: destination file in ComfyUI models must remain untouched!
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"model-weights-content")

    def test_download_model_invokes_cache_removal(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cached_dir = temp / "hub" / "models--org--repo" / "snapshots" / "rev1"
            cached_dir.mkdir(parents=True)
            cached_file = cached_dir / "weights.safetensors"
            cached_file.write_bytes(b"downloaded-data")

            dest = temp / "comfy" / "weights.safetensors"
            dest.parent.mkdir(parents=True)

            spec = h3_models.ModelSpec(
                repo_id="org/repo",
                folder="test",
                filename="weights.safetensors",
                source="test",
            )
            plan = {
                "key": "test_key",
                "spec": spec,
                "dest": dest,
                "revision": "rev1",
                "size": len(b"downloaded-data"),
                "sha256": None,
                "blob_id": None,
            }

            with patch("huggingface_hub.hf_hub_download", return_value=str(cached_file)):
                result = h3_models._download_model(plan, token=None, log_prefix="[test]")

            self.assertEqual(result, "test_key")
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), b"downloaded-data")
            # Cached file must have been removed immediately
            self.assertFalse(cached_file.exists())

    def test_failed_copy_keeps_download_for_retry_and_removes_partial(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cached = temp / "hub" / "models--org--repo" / "snapshots" / "rev1" / "weights.bin"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"complete download")
            dest = temp / "comfy" / "weights.bin"
            dest.parent.mkdir()
            spec = h3_models.ModelSpec(
                repo_id="org/repo", folder="test", filename="weights.bin", source="test"
            )
            plan = {
                "key": "test_key", "spec": spec, "dest": dest,
                "revision": "rev1", "size": cached.stat().st_size,
            }

            def fail_copy(_source, target):
                Path(target).write_bytes(b"partial")
                raise OSError("disk full")

            with (
                patch("huggingface_hub.hf_hub_download", return_value=str(cached)),
                patch.object(h3_models.shutil, "copy2", side_effect=fail_copy),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    h3_models._download_model(plan, token=None, log_prefix="[test]")

            self.assertTrue(cached.exists())
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_suffix(".bin.partial").exists())

    def test_remove_hf_cache_preserves_blob_used_by_another_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            repo = temp / "hub" / "models--org--repo"
            blob = repo / "blobs" / "shared"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"shared weights")
            first = repo / "snapshots" / "rev1" / "weights.bin"
            second = repo / "snapshots" / "rev2" / "weights.bin"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            try:
                first.symlink_to(blob)
                second.symlink_to(blob)
            except OSError:
                os.link(blob, first)
                os.link(blob, second)

            dest = temp / "models" / "weights.bin"
            dest.parent.mkdir()
            shutil.copy2(first, dest)
            h3_models._remove_hf_cache(first, plan={"dest": dest, "blob_id": "shared"})

            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(second.read_bytes(), b"shared weights")
            self.assertTrue(blob.exists())

    def test_prune_hf_cache_cleans_target_repos(self):
        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            target_repo = "org/model-target"
            other_repo = "other/model-keep"

            target_folder = temp / f"models--{target_repo.replace('/', '--')}"
            other_folder = temp / f"models--{other_repo.replace('/', '--')}"

            for f in (target_folder / "snapshots" / "revA", other_folder / "snapshots" / "revB"):
                f.mkdir(parents=True)
                (f / "file.bin").write_bytes(b"dummy")

            with patch("huggingface_hub.constants.HF_HUB_CACHE", str(temp)):
                freed = h3_models.prune_hf_cache(repo_ids=[target_repo], log_prefix="[test]")

            self.assertGreater(freed, 0)
            self.assertFalse(target_folder.exists())
            self.assertTrue(other_folder.exists())

    def test_swiftvr_clears_local_cache_only_after_checkpoint_is_complete(self):
        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "ComfyUI" / "models" / "swiftvr"
            runtime = SimpleNamespace(comfy_dir=checkpoint.parents[1])

            def incomplete_download(**kwargs):
                local = Path(kwargs["local_dir"])
                (local / ".cache" / "huggingface").mkdir(parents=True)
                (local / "reae.safetensors").write_bytes(b"model")

            with patch("huggingface_hub.snapshot_download", side_effect=incomplete_download):
                with self.assertRaisesRegex(Exception, "required files"):
                    ensure_swiftvr_checkpoint(runtime=runtime)
            self.assertTrue((checkpoint / ".cache").exists())

            def complete_download(**kwargs):
                local = Path(kwargs["local_dir"])
                for relative in (
                    "prompt_embedding.safetensors", "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                ):
                    path = local / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"model")

            with patch("huggingface_hub.snapshot_download", side_effect=complete_download):
                actual, downloaded = ensure_swiftvr_checkpoint(runtime=runtime)
            self.assertEqual(actual, checkpoint)
            self.assertTrue(downloaded)
            self.assertFalse((checkpoint / ".cache").exists())

            (checkpoint / ".cache" / "huggingface").mkdir(parents=True)
            actual, downloaded = ensure_swiftvr_checkpoint(runtime=runtime)
            self.assertEqual(actual, checkpoint)
            self.assertFalse(downloaded)
            self.assertFalse((checkpoint / ".cache").exists())


if __name__ == "__main__":
    unittest.main()
