"""Check the web UI Drive cell without a live Colab account."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ColabDriveAuthTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8"))
        self.source = next("".join(c["source"]) for c in notebook["cells"]
                           if "def setup_google_drive_outputs(" in "".join(c["source"]))
        google, colab = ModuleType("google"), ModuleType("google.colab")
        self.mount = Mock()
        colab.drive = SimpleNamespace(mount=self.mount)
        google.colab = colab
        self.modules = {"google": google, "google.colab": colab}

    def run_cell(self, root, mounted=False, enabled=True):
        namespace = dict(NOTEBOOK_DIR=root, WORKSPACE_DIR=str(root))
        source = self.source if enabled else self.source.replace("MOUNT_GOOGLE_DRIVE = True", "MOUNT_GOOGLE_DRIVE = False")
        with patch.dict(sys.modules, self.modules), patch("os.path.ismount", return_value=mounted), patch.object(Path, "symlink_to", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "<Drive cell>", "exec"), namespace)
        return namespace

    def test_run_all_uses_native_confirmation_and_links_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = self.run_cell(root)
            self.mount.assert_called_once_with(str(root / "drive"))
            self.assertTrue(namespace["MOUNT_GOOGLE_DRIVE"])
            self.assertTrue((root / "drive" / "MyDrive" / "MiniMax-H3" / "output").is_dir())
            self.assertTrue((root / "drive" / "MyDrive" / "MiniMax-H3" / "gradio_outputs").is_dir())

    def test_existing_mount_does_not_request_confirmation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "drive" / "MyDrive").mkdir(parents=True)
            self.run_cell(root, mounted=True)
            self.mount.assert_not_called()

    def test_mount_error_stops_before_output_migration(self):
        self.mount.side_effect = RuntimeError("Google authorization declined")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Drive setup stopped"):
                self.run_cell(root)
            self.assertFalse((root / "h3").exists())

    def test_disabled_drive_skips_authorization(self):
        with TemporaryDirectory() as directory:
            self.run_cell(Path(directory), enabled=False)
        self.mount.assert_not_called()


if __name__ == "__main__":
    unittest.main()
