"""CPU manager behavior without downloading models or allocating a GPU."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import h3_model_manager as manager_module
from h3_model_manager import ModelManager, model_presets, preset_selection
from h3_models import MODEL_SPECS, model_manifest_key, read_json, write_json_atomic


class ModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.volume = Mock()
        self.manager = ModelManager(Path(self.temp.name), self.volume)

    def test_presets_cover_defaults_and_optional_loras(self):
        presets = model_presets()
        for keys in presets.values():
            self.assertFalse(set(keys) - MODEL_SPECS.keys())
        h3 = presets["MiniMax H3 defaults"]
        self.assertTrue({"singularity_fl2va", "text_encoder", "turbo_lora",
                         "turbo_ref_lora", "video_vae_lynnreal_int8"}.issubset(h3))
        self.assertEqual(presets["Qwen Image 2.1 defaults"], (
            "qwen_image21_dit_bf16", "qwen_image21_text_bf16", "qwen_image21_vae",
        ))
        self.assertIn("ltx25_distilled_int8", presets["LTX 2.5 defaults"])
        self.assertIn("ltx25_iclora_ingredients", presets["LTX 2.5 + IC-LoRAs"])
        combined = preset_selection(["LTX 2.5 defaults", "LTX 2.5 + IC-LoRAs"])
        self.assertEqual(len(combined), len(set(combined)))

    def test_batch_continues_after_failure_and_commits_success(self):
        keys = ["qwen_image21_vae", "ltx25_audio_vae", "text_encoder"]
        attempted = []

        def download(**kwargs):
            key, = kwargs["model_keys"]
            attempted.append(key)
            if key == keys[1]:
                raise RuntimeError("access denied")
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"downloaded")

        with patch.object(manager_module, "sync_models", side_effect=download):
            message, rows = self.manager.download(keys)
        self.assertEqual(attempted, keys)
        self.assertIn("2/3", message)
        self.assertIn("ltx25_audio_vae: access denied", message)
        self.assertTrue(self.manager._path(keys[2]).is_file())
        self.assertTrue(self.manager.config.is_file())
        self.volume.reload.assert_called_once()
        self.assertEqual(self.volume.commit.call_count, len(keys) + 2)
        self.assertEqual(len(rows), len(MODEL_SPECS))

    def test_remove_only_selected_and_clear_manifest(self):
        selected, retained = "turbo_lora", "text_encoder"
        manifest = {"files": {}}
        for key in (selected, retained):
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            manifest["files"][model_manifest_key(MODEL_SPECS[key])] = {"size": 5}
        write_json_atomic(self.manager.manifest, manifest)
        self.manager.remove([selected], False)
        self.assertTrue(self.manager._path(selected).exists())
        self.volume.commit.assert_not_called()
        self.manager.remove([selected], True)
        self.assertFalse(self.manager._path(selected).exists())
        self.assertTrue(self.manager._path(retained).exists())
        remaining = read_json(self.manager.manifest, {})["files"]
        self.assertNotIn(model_manifest_key(MODEL_SPECS[selected]), remaining)
        self.assertIn(model_manifest_key(MODEL_SPECS[retained]), remaining)
        self.volume.commit.assert_called_once()

    def test_remove_failure_keeps_unremoved_entries_and_commits_progress(self):
        first, second = "turbo_lora", "text_encoder"
        manifest = {"files": {}}
        for key in (first, second):
            path = self.manager._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            manifest["files"][model_manifest_key(MODEL_SPECS[key])] = {"size": 5}
        write_json_atomic(self.manager.manifest, manifest)
        real_unlink = Path.unlink

        def flaky(path, *args, **kwargs):
            if path.name == MODEL_SPECS[second].local_name:
                raise OSError("busy")
            return real_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", flaky), self.assertRaises(OSError):
            self.manager.remove([first, second], True)
        remaining = read_json(self.manager.manifest, {})["files"]
        self.assertNotIn(model_manifest_key(MODEL_SPECS[first]), remaining)
        self.assertIn(model_manifest_key(MODEL_SPECS[second]), remaining)
        self.assertTrue(self.manager._path(second).exists())
        self.volume.commit.assert_called_once()

    def test_unknown_keys_rejected_before_mutation(self):
        with self.assertRaises(ValueError):
            self.manager.download(["../anything"])
        with self.assertRaises(ValueError):
            self.manager.remove(["../anything"], True)
        self.volume.assert_not_called()
        self.assertFalse(self.manager.config.exists())

    def test_aliases_share_one_download_and_removal_target(self):
        self.assertEqual(self.manager._keys([
            "singularity_fl2va", "singularity_ref2va",
        ]), ("singularity_fl2va",))

    def test_symlink_outside_models_rejected(self):
        root = self.manager.root
        root.mkdir(parents=True)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        spec = MODEL_SPECS["turbo_lora"]
        try:
            (root / spec.folder).symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        with self.assertRaises(ValueError):
            self.manager._path("turbo_lora")

    def test_cpu_endpoint_has_default_resources_and_independent_image(self):
        source = Path(__file__).resolve().parents[1] / "modal_h3.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("provision_models", functions)
        manager = functions["manage_models"]
        decorator = next(d for d in manager.decorator_list
                         if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                         and d.func.attr == "function")
        arguments = {kw.arg: kw.value for kw in decorator.keywords}
        self.assertEqual(arguments["image"].id, "manager_image")
        for resource in ("gpu", "cpu", "memory"):
            self.assertNotIn(resource, arguments)
        self.assertNotIn("sync_models", ast.unparse(functions["prepare_runtime_models"]))


if __name__ == "__main__":
    unittest.main()
