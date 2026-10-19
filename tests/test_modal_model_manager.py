"""CPU manager behavior without downloading models or allocating a GPU."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import h3_model_manager as manager_module
from h3_model_manager import (
    AUDIO_TOOLS_FAMILY, SNAPSHOT_SPECS, ModelManager, SnapshotSpec, catalog_groups, model_presets, preset_selection,
)
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
            self.assertFalse(set(keys) - MODEL_SPECS.keys() - SNAPSHOT_SPECS.keys())
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
        self.assertEqual(len(rows), len(MODEL_SPECS) + len(SNAPSHOT_SPECS))

    def test_snapshots_use_runtime_layouts(self):
        from h3_app import audio_tools_worker
        from h3_app.drama3_serving.server import ServerSettings

        root = self.manager.root
        self.assertEqual(self.manager._path("qwen_voice_design"),
                         audio_tools_worker.voice_design_cache(root))
        self.assertEqual(self.manager._path("stable_audio3_medium"),
                         audio_tools_worker.stable_audio_dir(root))
        runtime = ServerSettings(venv=Path(), data_dir=Path(self.temp.name) / "drama3-serving",
                                 model_root=root / "drama3_serving")
        self.assertEqual(self.manager._path("drama3_sglang_checkpoint"), runtime.model_dir)
        groups = catalog_groups()
        self.assertIn("drama3_sglang_checkpoint", groups["Drama 3"])
        self.assertEqual(groups[AUDIO_TOOLS_FAMILY], ("qwen_voice_design", "stable_audio3_medium"))
        self.assertFalse(SNAPSHOT_SPECS.keys() & MODEL_SPECS.keys())

    def test_snapshot_inventory_reports_runtime_readiness(self):
        statuses = lambda: {row[0]: row[3] for row in self.manager._inventory()}
        self.assertEqual({statuses()[key] for key in SNAPSHOT_SPECS}, {"Missing"})
        stable = self.manager._path("stable_audio3_medium")
        (stable / "model.safetensors").parent.mkdir(parents=True)
        (stable / "model.safetensors").write_bytes(b"x")
        self.assertEqual(statuses()["stable_audio3_medium"], "Unverified / incomplete")
        drama = self.manager._path("drama3_sglang_checkpoint")
        drama.mkdir(parents=True)
        (drama / ".h3-complete").write_text("rev", encoding="utf-8")
        row = next(row for row in self.manager._inventory() if row[0] == "drama3_sglang_checkpoint")
        self.assertEqual((row[1], row[2], row[3]), ("drama3_serving", drama.name + "/", "Installed"))

    def test_snapshot_download_calls_runtime_downloaders(self):
        from h3_app import audio_tools_worker
        from h3_app.drama3_serving import server

        with patch.object(audio_tools_worker, "download") as audio, \
                patch.object(server, "ensure_model") as drama, \
                patch.object(manager_module, "sync_models") as files, \
                patch.dict("os.environ", {"HF_TOKEN": "token"}):
            message, _rows = self.manager.download(list(SNAPSHOT_SPECS))
        self.assertIn("3/3", message)
        files.assert_not_called()
        self.assertEqual([call.args for call in audio.call_args_list], [
            ("voice_design", str(self.manager.root)), ("stable_audio", str(self.manager.root)),
        ])
        settings, = drama.call_args.args
        self.assertEqual(settings.model_root, self.manager.root / "drama3_serving")
        self.assertEqual(drama.call_args.kwargs["token"], "token")
        self.assertEqual(self.volume.commit.call_count, len(SNAPSHOT_SPECS) + 2)

    def test_remove_snapshot_deletes_only_its_directory(self):
        voice = self.manager._path("qwen_voice_design")
        (voice / "models--Qwen").mkdir(parents=True)
        (voice / "models--Qwen" / "blob").write_bytes(b"model")
        stable = self.manager._path("stable_audio3_medium")
        stable.mkdir(parents=True)
        self.manager.remove(["qwen_voice_design"], True)
        self.assertFalse(voice.exists())
        self.assertTrue(stable.is_dir())
        self.volume.commit.assert_called_once()

    def test_snapshot_paths_must_stay_below_models(self):
        escaping = SnapshotSpec(
            "Drama 3", "repo", "test", lambda root: root, lambda root: False, lambda root: None,
        )
        with patch.dict(SNAPSHOT_SPECS, {"qwen_voice_design": escaping}), \
                self.assertRaises(ValueError):
            self.manager._path("qwen_voice_design")

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
