import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h3_models


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


if __name__ == "__main__":
    unittest.main()
