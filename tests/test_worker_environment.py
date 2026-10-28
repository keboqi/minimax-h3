"""Setup output remains live without losing the persistent log or environment reuse."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app.workers import env


class WorkerEnvironmentTests(unittest.TestCase):
    def test_setup_output_reaches_console_and_ui_before_process_exits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log, done = root / "setup.log", root / "done"
            updates, observed_live = [], []

            def report(message):
                updates.append(message)
                if "Downloading packages" in message:
                    observed_live.append(not done.exists())

            command = [sys.executable, "-u", "-c",
                "import sys,time; from pathlib import Path; print('Downloading packages', flush=True); "
                "time.sleep(0.8); Path(sys.argv[1]).touch(); print('Installed',flush=True)", str(done)]
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                env.run_process(command, log, timeout=10, label="Tools setup", report=report)
            self.assertEqual(observed_live, [True])
            self.assertIn("Installed", updates[-1])
            self.assertIn("Downloading packages", console.getvalue())
            self.assertIn("Installed", log.read_text())

    def test_failed_setup_keeps_diagnostics_and_reports_log_path(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "setup.log"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(H3Error) as raised:
                env.run_process([sys.executable, "-c", "print('build failed'); raise SystemExit(2)"],
                    log, timeout=10, label="Tools setup", report=lambda text: None)
            self.assertIn(str(log), str(raised.exception))
            self.assertIn("build failed", log.read_text())

    def test_cancelling_live_setup_reaps_the_installer(self):
        with TemporaryDirectory() as directory:
            processes = []
            popen = subprocess.Popen

            def launch(*args, **kwargs):
                process = popen(*args, **kwargs)
                processes.append(process)
                return process

            def cancel(_message):
                raise JobCancelled("stop")

            with patch.object(env.subprocess, "Popen", side_effect=launch), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(JobCancelled):
                env.run_process([sys.executable, "-u", "-c", "import time; print('Ready'); time.sleep(30)"],
                    Path(directory) / "setup.log", timeout=10, label="Tools setup", report=cancel)
            self.assertIsNotNone(processes[0].poll())

    def test_ready_environment_is_reused_without_installing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            python = env.python_path(root / "venv")
            python.parent.mkdir(parents=True)
            python.touch()
            (root / "venv" / "ready").write_text("current")
            updates = []
            with patch.object(env, "run_process") as install, contextlib.redirect_stdout(io.StringIO()):
                result = env.prepare_venv(root / "venv", root / "setup.log", name="Tools", label="Tools",
                    fingerprint="current", ready_marker="ready", install_commands=lambda *a: [],
                    timeout=10, report=updates.append)
            self.assertEqual(result, python)
            install.assert_not_called()
            self.assertIn("Using the prepared", updates[-1])

    def test_setup_reports_install_and_import_steps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            python = env.python_path(root / "venv")
            python.parent.mkdir(parents=True)
            python.touch()
            updates = []
            with patch.object(env, "uv_command", return_value=["uv"]), \
                    patch.object(env, "run_process") as install, contextlib.redirect_stdout(io.StringIO()):
                env.prepare_venv(root / "venv", root / "setup.log", name="Tools", label="Tools",
                    fingerprint="current", ready_marker="ready", install_commands=lambda *a: [
                        ["uv", "venv"], ["uv", "pip", "install", "packages"], [str(python), "-c", "import packages"]],
                    timeout=10, report=updates.append)
            self.assertEqual(install.call_count, 2)
            self.assertTrue(any("Installing Python packages" in text for text in updates))
            self.assertTrue(any("Checking installed package imports" in text for text in updates))
            self.assertTrue(all(call.kwargs["report"] == updates.append for call in install.call_args_list))
