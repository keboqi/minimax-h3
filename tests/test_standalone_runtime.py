"""Standalone hosts must select Python 3.12 before installing GPU dependencies."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import NamedTuple
import unittest
from unittest.mock import patch

import h3_runtime as runtime
import setup_h3


class Version(NamedTuple):
    major: int
    minor: int


class StandaloneRuntimeTests(unittest.TestCase):
    def test_active_python312_is_reused_without_installing_anything(self):
        with (
            patch.object(runtime.sys, "version_info", Version(3, 12)),
            patch.object(runtime, "run") as run,
            TemporaryDirectory() as directory,
        ):
            root = Path(directory) / "h3"
            self.assertEqual(runtime.ensure_python312(root), Path(sys.executable))
            run.assert_not_called()
            self.assertFalse(root.exists())

    def test_python314_provisions_isolated312_and_keeps_host_interpreter(self):
        host_python = sys.executable
        with (
            TemporaryDirectory() as directory,
            patch.object(runtime.sys, "version_info", Version(3, 14)),
            patch.object(runtime.shutil, "which", return_value="uv"),
            patch.object(runtime, "run") as run,
            patch.object(runtime, "python_version", return_value="3.12"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            root = Path(directory) / "h3"
            selected = runtime.ensure_python312(root)
            self.assertTrue(selected.is_relative_to(root / ".venv"))
            run.assert_called_once_with("uv", "venv", "--python", "3.12", "--seed", root / ".venv")
            self.assertEqual(sys.executable, host_python)

    def test_managed312_is_reused_after_restart_without_another_install(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(runtime.sys, "version_info", Version(3, 14)),
            patch.object(runtime, "python_version", return_value="3.12"),
            patch.object(runtime, "run") as run,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            root = Path(directory)
            (root / ".venv").mkdir()
            selected = runtime.ensure_python312(root)
            self.assertTrue(selected.is_relative_to(root / ".venv"))
            run.assert_not_called()

    def test_incompatible_existing_environment_is_preserved(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(runtime.sys, "version_info", Version(3, 14)),
            patch.object(runtime, "python_version", return_value="3.14"),
            patch.object(runtime, "run") as run,
        ):
            root = Path(directory)
            existing = root / ".venv"
            existing.mkdir()
            valuable = existing / "keep.txt"
            valuable.write_text("existing installation")
            with self.assertRaisesRegex(RuntimeError, "has been preserved"):
                runtime.ensure_python312(root)
            self.assertEqual(valuable.read_text(), "existing installation")
            run.assert_not_called()

    def test_missing_uv_bootstraps_into_local_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / ".bootstrap/uv" / ("uv.exe" if os.name == "nt" else "uv")

            def install(*args):
                if "pip" in args:
                    cached.parent.mkdir(parents=True)
                    cached.touch()

            with (
                patch.object(runtime.sys, "version_info", Version(3, 14)),
                patch.object(runtime.shutil, "which", return_value=None),
                patch.object(runtime, "python_version", return_value="3.12"),
                patch.object(runtime, "run", side_effect=install) as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                runtime.ensure_python312(root)
                run.assert_any_call(sys.executable, "-m", "pip", "install", "--upgrade",
                                    "--target", root / ".bootstrap", "uv>=0.8")
                run.assert_any_call(str(cached), "venv", "--python", "3.12", "--seed", root / ".venv")

    def test_failed_python_probe_is_unusable(self):
        for error in (OSError(), subprocess.TimeoutExpired("python", 15)):
            with self.subTest(error=type(error)), patch.object(runtime.subprocess, "run", side_effect=error):
                self.assertIsNone(runtime.python_version(Path("broken-python")))

    def test_direct_setup_rejects314_before_modifying_environment(self):
        with (
            patch.object(setup_h3.sys, "version_info", Version(3, 14)),
            patch.object(setup_h3.sys, "argv", ["setup_h3.py", "--skip-env"]),
            patch.object(setup_h3, "ensure_root_entrypoints") as recover,
            patch.object(setup_h3, "install_environment") as install,
            contextlib.redirect_stderr(io.StringIO()) as error,
            self.assertRaises(SystemExit) as stopped,
        ):
            setup_h3.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("requires Python 3.12", error.getvalue())
        self.assertIn("bash run_h3.sh", error.getvalue())
        recover.assert_not_called()
        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
