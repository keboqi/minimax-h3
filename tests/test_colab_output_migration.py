import ast
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


class ColabOutputMigrationTests(unittest.TestCase):
    def test_only_accessible_existing_mount_skips_authorization(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8")
        )
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "def setup_google_drive_outputs(" in "".join(c["source"]))
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "setup_google_drive_outputs")
        google = ModuleType("google")
        colab = ModuleType("google.colab")
        mount = Mock()
        colab.drive = SimpleNamespace(mount=mount)
        google.colab = colab
        for mounted, accessible in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(mounted=mounted, accessible=accessible), TemporaryDirectory() as directory:
                root = Path(directory)
                if accessible:
                    (root / "drive" / "MyDrive").mkdir(parents=True)
                namespace = dict(Path=Path, shutil=shutil, os=os, NOTEBOOK_DIR=root)
                exec(compile(ast.Module(body=[function], type_ignores=[]), "<colab>", "exec"), namespace)
                mount.reset_mock()
                with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                     patch.object(os.path, "ismount", return_value=mounted), \
                     patch.object(Path, "symlink_to", return_value=None):
                    namespace["setup_google_drive_outputs"](True, workspace_dir=str(root / "workspace"))
                if mounted and accessible:
                    mount.assert_not_called()
                else:
                    mount.assert_called_once_with(str(root / "drive"))

    def test_existing_drive_name_preserves_both_outputs(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8")
        )
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if "def setup_google_drive_outputs(" in "".join(cell["source"])
        )
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_google_drive_outputs"
        )
        namespace = {"Path": Path, "shutil": shutil, "os": os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<colab>", "exec"), namespace)

        google = ModuleType("google")
        colab = ModuleType("google.colab")
        colab.drive = SimpleNamespace(mount=lambda _: None)
        google.colab = colab

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            local = workspace / "h3" / "ComfyUI" / "output"
            local.mkdir(parents=True)
            (local / "clip.mp4").write_bytes(b"new clip")
            drive_root = root / "drive"
            drive_output = drive_root / "output"
            drive_output.mkdir(parents=True)
            (drive_output / "clip.mp4").write_bytes(b"old clip")

            with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
                with patch.object(Path, "symlink_to", return_value=None):
                    namespace["setup_google_drive_outputs"](
                        True, str(drive_root), str(workspace)
                    )

            self.assertEqual((drive_output / "clip.mp4").read_bytes(), b"old clip")
            self.assertEqual(
                (drive_output / "clip_from_colab_1.mp4").read_bytes(), b"new clip"
            )


if __name__ == "__main__":
    unittest.main()
