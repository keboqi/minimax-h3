"""A successful setup can be reused only while its sources and packages match."""

import contextlib
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error

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
        for name in setup.BUNDLED_NODE_PACKAGES:
            shutil.copytree(setup.SCRIPT_DIR / "custom_nodes" / name, self.comfy / "custom_nodes" / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
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

    def test_gpu_reassignment_invalidates_setup_in_both_directions(self):
        for source, target in (("default", "a100"), ("a100", "default")):
            with patch.dict(os.environ, H3_GPU_PROFILE=source):
                setup.record_setup_state(self.install)
                self.assertTrue(setup.setup_is_current(self.install))
            with patch.dict(os.environ, H3_GPU_PROFILE=target):
                self.assertFalse(setup.setup_is_current(self.install))

    def test_missing_bundled_node_or_frontend_invalidates_record(self):
        with patch.object(setup, "comfy_frontend_package_is_ready", return_value=False):
            self.assertFalse(setup.setup_is_current(self.install))
        (self.comfy / "custom_nodes" / "H3Acceleration" / "__init__.py").unlink()
        self.assertFalse(setup.setup_is_current(self.install))

    def test_stale_or_missing_bundled_package_invalidates_record(self):
        for name in setup.BUNDLED_NODE_PACKAGES:
            with self.subTest(name=name):
                installed = self.comfy / "custom_nodes" / name / "__init__.py"
                original = installed.read_bytes()
                installed.write_bytes(original + b"# older copy\n")
                self.assertFalse(setup.setup_is_current(self.install))
                installed.unlink()
                self.assertFalse(setup.setup_is_current(self.install))
                installed.write_bytes(original)
                self.assertTrue(setup.setup_is_current(self.install))

    def test_tracked_path_absent_at_record_time_does_not_fail_setup(self):
        (self.comfy / "requirements.txt").unlink()
        setup.record_setup_state(self.install)
        self.assertTrue(setup.setup_is_current(self.install))

    def test_dangling_tracked_symlink_does_not_fail_setup(self):
        link = self.comfy / "dangling"
        try:
            os.symlink("absent-target", link)
        except OSError:
            self.skipTest("symlinks are unavailable here")
        self.git("add", "dangling")
        setup.record_setup_state(self.install)
        self.assertTrue(setup.setup_is_current(self.install))

    def test_malformed_record_triggers_repair(self):
        (self.install / ".setup-state.json").write_text("[]")
        self.assertFalse(setup.setup_is_current(self.install))

    def test_main_reuses_setup_and_refresh_forces_repair(self):
        arguments = ["setup_h3.py", "--install-dir", str(self.install)]
        with (patch.object(sys, "argv", arguments),
              patch.object(setup, "ensure_root_entrypoints"),
              patch.object(setup, "ensure_cjk_font"),
              patch.object(setup, "sync_model_inventory") as inventory,
              patch.object(setup, "install_environment") as install):
            setup.main()
        inventory.assert_called_once_with(self.install, self.comfy)
        install.assert_not_called()
        with (patch.object(sys, "argv", arguments + ["--refresh"]),
              patch.object(setup, "ensure_root_entrypoints"),
              patch.object(setup, "ensure_cjk_font"),
              patch.object(setup, "install_environment", side_effect=RuntimeError("repair failed")),
              self.assertRaisesRegex(RuntimeError, "repair failed")):
            setup.main()
        self.assertFalse((self.install / ".setup-state.json").exists())


class _FakeUrlResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class CjkFontInstallTests(unittest.TestCase):
    def test_non_linux_skips_fc_list_and_download(self):
        with (patch.object(setup.sys, "platform", "win32"),
              patch.object(setup.shutil, "which") as which,
              patch.object(setup.urllib.request, "urlopen") as urlopen):
            setup.ensure_cjk_font()
        which.assert_not_called()
        urlopen.assert_not_called()

    def test_existing_font_files_skip_fc_list_and_download(self):
        with TemporaryDirectory() as directory:
            font_dir = Path(directory)
            for name, _ in setup.CJK_FONT_FILES:
                (font_dir / name).write_bytes(b"OTTO-fixture")
            with (patch.object(setup.sys, "platform", "linux"),
                  patch.object(setup.shutil, "which") as which,
                  patch.object(setup.urllib.request, "urlopen") as urlopen):
                setup.ensure_cjk_font(font_dir)
        which.assert_not_called()
        urlopen.assert_not_called()

    def test_existing_system_cjk_font_from_fc_list_skips_download(self):
        with TemporaryDirectory() as directory:
            font_dir = Path(directory)
            fc_list = subprocess.CompletedProcess(["fc-list", ":lang=zh", "family"], 0, stdout="Noto Sans CJK SC\n")
            with (patch.object(setup.sys, "platform", "linux"),
                  patch.object(setup.shutil, "which", side_effect=lambda cmd: "/usr/bin/fc-list" if cmd == "fc-list" else None),
                  patch.object(setup.subprocess, "run", return_value=fc_list) as run,
                  patch.object(setup.urllib.request, "urlopen") as urlopen):
                setup.ensure_cjk_font(font_dir)
            run.assert_called_once_with(
                ["fc-list", ":lang=zh", "family"],
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=30,
            )
            urlopen.assert_not_called()

    def test_missing_fonts_download_verify_and_refresh_font_cache_without_apt(self):
        regular = b"regular-otf-bytes"
        bold = b"bold-otf-bytes"
        specs = (
            ("NotoSansCJKsc-Regular.otf", hashlib.sha256(regular).hexdigest()),
            ("NotoSansCJKsc-Bold.otf", hashlib.sha256(bold).hexdigest()),
        )
        payloads = {"NotoSansCJKsc-Regular.otf": regular, "NotoSansCJKsc-Bold.otf": bold}

        def fake_urlopen(req, timeout=None):
            name = req.full_url.rsplit("/", 1)[-1]
            return _FakeUrlResponse(payloads[name])

        with TemporaryDirectory() as directory:
            font_dir = Path(directory)
            output = io.StringIO()
            with (patch.object(setup, "CJK_FONT_FILES", specs),
                  patch.object(setup.sys, "platform", "linux"),
                  patch.object(setup.shutil, "which", side_effect=lambda cmd: "/usr/bin/fc-cache" if cmd == "fc-cache" else None),
                  patch.object(setup.urllib.request, "urlopen", side_effect=fake_urlopen),
                  patch.object(setup.subprocess, "run") as run,
                  contextlib.redirect_stdout(output)):
                setup.ensure_cjk_font(font_dir)

            self.assertEqual((font_dir / "NotoSansCJKsc-Regular.otf").read_bytes(), regular)
            self.assertEqual((font_dir / "NotoSansCJKsc-Bold.otf").read_bytes(), bold)
            self.assertEqual(sorted(p.name for p in font_dir.iterdir()),
                             ["NotoSansCJKsc-Bold.otf", "NotoSansCJKsc-Regular.otf"])
            run.assert_called_once_with(
                ["fc-cache", "-f", str(font_dir)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                check=False,
            )
            for call in run.call_args_list:
                self.assertNotIn("apt-get", call.args[0])
            self.assertIn("Noto Sans CJK SC fonts ready", output.getvalue())

    def test_checksum_mismatch_or_network_error_warns_and_leaves_no_partial_files(self):
        regular = b"regular-otf-bytes"
        bold = b"bold-otf-bytes"
        specs = (
            ("NotoSansCJKsc-Regular.otf", hashlib.sha256(regular).hexdigest()),
            ("NotoSansCJKsc-Bold.otf", hashlib.sha256(bold).hexdigest()),
        )

        for failure in ("bad-checksum", "network-error"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                font_dir = Path(directory)

                def fake_urlopen(req, timeout=None, mode=failure):
                    name = req.full_url.rsplit("/", 1)[-1]
                    if name == "NotoSansCJKsc-Bold.otf":
                        if mode == "bad-checksum":
                            return _FakeUrlResponse(b"corrupted-bytes")
                        raise urllib.error.URLError("connection reset")
                    return _FakeUrlResponse(regular)

                output = io.StringIO()
                with (patch.object(setup, "CJK_FONT_FILES", specs),
                      patch.object(setup.sys, "platform", "linux"),
                      patch.object(setup.shutil, "which", return_value=None),
                      patch.object(setup.urllib.request, "urlopen", side_effect=fake_urlopen),
                      contextlib.redirect_stdout(output)):
                    setup.ensure_cjk_font(font_dir)

                self.assertEqual(list(font_dir.iterdir()), [])
                self.assertIn("Could not install CJK fonts", output.getvalue())

    def test_main_cjk_font_flags(self):
        with (patch.object(sys, "argv", ["setup_h3.py", "--cjk-font-only"]),
              patch.object(setup, "ensure_cjk_font") as ensure_font,
              patch.object(setup, "ensure_root_entrypoints") as entrypoints,
              patch.object(setup, "install_environment") as install):
            setup.main()
        ensure_font.assert_called_once_with()
        entrypoints.assert_not_called()
        install.assert_not_called()

        with TemporaryDirectory() as directory:
            with (patch.object(sys, "argv", ["setup_h3.py", "--install-dir", directory, "--skip-cjk-font"]),
                  patch.object(setup, "ensure_root_entrypoints"),
                  patch.object(setup, "ensure_cjk_font") as ensure_font,
                  patch.object(setup, "setup_is_current", return_value=True),
                  patch.object(setup, "sync_model_inventory") as inventory,
                  patch.object(setup, "install_environment") as install):
                setup.main()
            ensure_font.assert_not_called()
            inventory.assert_called_once()
            install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
