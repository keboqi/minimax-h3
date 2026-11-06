"""A successful setup can be reused only while its sources and packages match."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import setup_h3 as setup


class SetupReuseTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.install = Path(temp.name)
        self.comfy = self.install / "ComfyUI"
        self.comfy.mkdir()
        self.git("init", "-q")
        (self.comfy / "main.py").write_text("# fixture\n")
        (self.comfy / "requirements.txt").write_text("fixture==1\n")
        self.git("add", "main.py", "requirements.txt")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture")
        bundled = self.comfy / "custom_nodes" / "H3Acceleration" / "__init__.py"
        bundled.parent.mkdir(parents=True)
        bundled.write_text("# installed node\n")
        self.packages = [["fixture", "1", "site-packages"]]
        for mocked in (
            patch.object(setup, "_environment_fingerprint", side_effect=lambda: self.packages),
            patch.object(setup, "comfy_frontend_package_is_ready", return_value=True),
            patch.dict(setup._SETUP_REPOSITORIES,
                       {str(self.comfy): {"required_paths": ("main.py",)}}, clear=True),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)
        setup.record_setup_state(self.install)

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.comfy), *args], check=True,
                       capture_output=True, text=True)

    def test_unchanged_installation_passes_without_fetching(self):
        with patch.object(setup, "run") as mutation:
            self.assertTrue(setup.setup_is_current(self.install))
        mutation.assert_not_called()

    def test_source_edit_or_missing_tracked_file_invalidates_record(self):
        (self.comfy / "main.py").write_text("# changed source\n")
        self.assertFalse(setup.setup_is_current(self.install))
        setup.record_setup_state(self.install)
        (self.comfy / "requirements.txt").unlink()
        self.assertFalse(setup.setup_is_current(self.install))

    def test_dependency_drift_invalidates_record(self):
        self.packages[0][1] = "2"
        self.assertFalse(setup.setup_is_current(self.install))

    def test_changed_setup_inputs_invalidates_record(self):
        with patch.object(setup, "_setup_inputs", return_value={"changed": "pin"}):
            self.assertFalse(setup.setup_is_current(self.install))

    def test_missing_bundled_node_or_frontend_invalidates_record(self):
        with patch.object(setup, "comfy_frontend_package_is_ready", return_value=False):
            self.assertFalse(setup.setup_is_current(self.install))
        (self.comfy / "custom_nodes" / "H3Acceleration" / "__init__.py").unlink()
        self.assertFalse(setup.setup_is_current(self.install))

    def test_malformed_record_triggers_repair(self):
        (self.install / ".setup-state.json").write_text("[]")
        self.assertFalse(setup.setup_is_current(self.install))

    def test_main_reuses_setup_and_refresh_forces_repair(self):
        arguments = ["setup_h3.py", "--install-dir", str(self.install)]
        with (patch.object(sys, "argv", arguments),
              patch.object(setup, "ensure_root_entrypoints"),
              patch.object(setup, "sync_model_inventory") as inventory,
              patch.object(setup, "install_environment") as install):
            setup.main()
        inventory.assert_called_once_with(self.install, self.comfy)
        install.assert_not_called()
        with (patch.object(sys, "argv", arguments + ["--refresh"]),
              patch.object(setup, "ensure_root_entrypoints"),
              patch.object(setup, "install_environment", side_effect=RuntimeError("repair failed")),
              self.assertRaisesRegex(RuntimeError, "repair failed")):
            setup.main()
        self.assertFalse((self.install / ".setup-state.json").exists())


if __name__ == "__main__":
    unittest.main()
