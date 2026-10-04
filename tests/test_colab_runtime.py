"""Exercise notebook runtime fixes without a Colab GPU or network downloads."""

import ast
import contextlib
import io
import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import unittest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import h3_models

NOTEBOOK = Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb"


def cell_source(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"] if marker in "".join(cell["source"]))


def load_function(name, namespace):
    tree = ast.parse(cell_source(f"def {name}("))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace[name]


class ColabRamRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = h3_models._download_model
        self.addCleanup(setattr, h3_models, "_download_model", self.original)
        tree = ast.parse(cell_source("RAM_ROUTER_SOURCE ="))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RAM_ROUTER_SOURCE" for t in node.targets))
        self.router = {}
        exec(ast.literal_eval(assignment.value), self.router)
        self.router["_ram_root"] = self.root / "ram"
        self.router["_available_memory"] = lambda: 10**12
        self.fake_hf = ModuleType("huggingface_hub")
        self.fake_hf.hf_hub_download = self.download
        self.plans = []

    def download(self, **kwargs):
        cache = Path(kwargs.get("cache_dir", self.root / "disk-cache"))
        cached = cache / Path(kwargs["filename"]).name
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"data")
        return str(cached)

    def plan(self, **kwargs):
        spec = kwargs["spec"]
        dest = self.root / "models" / spec.folder / spec.local_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        plan = dict(key=kwargs["key"], spec=spec, dest=dest, manifest_key=h3_models.model_manifest_key(spec), remote_ok=True, revision="test", sha256=None, size=4, blob_id="blob", identity="identity", needs_download=True)
        self.plans.append(plan)
        return plan

    def batch(self, key, free=10**12):
        # Hard links exercise the file publication and cleanup on Windows too.
        with patch.dict(sys.modules, {"huggingface_hub": self.fake_hf}), patch.object(h3_models, "_fetch_repositories", return_value={}), patch.object(h3_models, "_plan_model", side_effect=self.plan), patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=free)), patch.object(Path, "symlink_to", lambda path, target: os.link(target, path)), contextlib.redirect_stdout(io.StringIO()):
            h3_models.sync_models(root=self.root / "models", manifest_path=self.root / "manifest.json", token="fixture", model_keys=[key])
        return self.plans[-1]

    def test_non_qwen_download_works_with_real_batch_caller(self):
        plan = self.batch("text_encoder")
        self.assertEqual(plan["dest"].read_bytes(), b"data")
        self.assertIn("cached_path", plan)

    def test_ram_download_records_manifest_and_survives_private_cache_cleanup(self):
        plan = self.batch("qwen_image21_vae")
        self.assertTrue(plan["dest"].is_relative_to(self.router["_ram_root"]))
        self.assertEqual(plan["dest"].read_bytes(), b"data")
        self.assertEqual((self.root / "models" / plan["manifest_key"]).read_bytes(), b"data")
        self.assertFalse(plan["cached_path"].exists())
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(manifest["files"][plan["manifest_key"]]["size"], 4)

    def test_low_ram_falls_back_to_disk_with_real_batch_caller(self):
        plan = self.batch("qwen_image21_vae", free=0)
        self.assertTrue(plan["dest"].is_relative_to(self.root / "models"))
        self.assertEqual(plan["dest"].read_bytes(), b"data")
        self.assertIn("cached_path", plan)

    def test_non_qwen_hook_forwards_deferred_cleanup_flag(self):
        original = Mock(return_value="text_encoder")
        self.router["_original_download_model"] = original
        plan = {"key": "text_encoder"}
        self.router["_download_model"](plan, "token", "prefix", False)
        original.assert_called_once_with(plan, "token", "prefix", False)


class ColabEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.namespace = dict(Path=Path, subprocess=subprocess, shutil=shutil, sys=sys)
        self.ensure = load_function("ensure_python_environment", self.namespace)

    def make_environment(self, root):
        venv = root / "venv_h3"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python3").touch()
        (venv / "pyvenv.cfg").write_text("version = 3.12")
        marker = venv / "installed-package"
        marker.write_text("preserve")
        return venv, marker

    def test_reruns_preserve_installed_environment_and_skip_uv_venv(self):
        with TemporaryDirectory() as directory:
            venv, marker = self.make_environment(Path(directory))
            with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="3.12\n")) as run, patch.object(shutil, "which", return_value="/usr/bin/uv"), contextlib.redirect_stdout(io.StringIO()):
                self.ensure(str(venv))
                self.ensure(str(venv))
            self.assertEqual(marker.read_text(), "preserve")
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.args[0][0] == str(venv / "bin" / "python3") for call in run.call_args_list))

    def test_wrong_python_version_is_preserved_and_rejected(self):
        with TemporaryDirectory() as directory:
            venv, marker = self.make_environment(Path(directory))
            with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="3.11\n")) as run:
                with self.assertRaisesRegex(RuntimeError, "not a working Python 3.12"):
                    self.ensure(str(venv))
            self.assertEqual(marker.read_text(), "preserve")
            self.assertEqual(run.call_count, 1)

    def test_missing_environment_is_created_with_python_312(self):
        with TemporaryDirectory() as directory:
            venv = Path(directory) / "venv_h3"
            with patch.object(subprocess, "run") as run, patch.object(shutil, "which", return_value="/usr/bin/uv"), contextlib.redirect_stdout(io.StringIO()):
                self.ensure(str(venv))
            run.assert_called_once_with(["/usr/bin/uv", "venv", "--python", "3.12", str(venv)], check=True)

    def test_unrecognised_existing_directory_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            venv = Path(directory) / "venv_h3"
            venv.mkdir()
            marker = venv / "keep"
            marker.write_text("preserve")
            with patch.object(subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "not a usable virtual environment"):
                    self.ensure(str(venv))
            run.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")


class ColabDriveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "workspace"
        self.namespace = dict(Path=Path, shutil=shutil, os=os, NOTEBOOK_DIR=self.root)
        self.setup = load_function("setup_google_drive_outputs", self.namespace)
        self.mount = Mock()
        google, colab = ModuleType("google"), ModuleType("google.colab")
        colab.drive = SimpleNamespace(mount=self.mount)
        google.colab = colab
        self.modules = {"google": google, "google.colab": colab}
        self.gradio = self.workspace / "h3" / "gradio_outputs"
        self.gradio.mkdir(parents=True)
        self.clip = self.gradio / "clip.mp4"
        self.clip.write_bytes(b"valuable output")

    def call(self, target):
        with patch.dict(sys.modules, self.modules), patch.object(Path, "symlink_to", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            self.setup(True, str(target), str(self.workspace))

    def test_source_equal_to_destination_preserves_existing_outputs(self):
        self.call(self.workspace / "h3")
        self.assertEqual(self.clip.read_bytes(), b"valuable output")

    def test_nested_destination_is_rejected_before_mount_or_file_moves(self):
        target = self.gradio / "nested"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.call(target)
        self.mount.assert_not_called()
        self.assertEqual(self.clip.read_bytes(), b"valuable output")
        self.assertFalse(target.exists())

    def test_ancestor_destination_is_rejected_before_any_mutation(self):
        self.namespace["WORKSPACE_DIR"] = str(self.root / "output")
        nested_clip = self.root / "output" / "h3" / "ComfyUI" / "output" / "clip.mp4"
        nested_clip.parent.mkdir(parents=True)
        nested_clip.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "overlap"), patch.dict(sys.modules, self.modules):
            self.setup(True, str(self.root))
        self.mount.assert_not_called()
        self.assertEqual(nested_clip.read_bytes(), b"keep")

    def test_spaces_in_mount_path_are_rejected_before_mount(self):
        self.namespace["NOTEBOOK_DIR"] = self.root / "folder with spaces"
        with self.assertRaisesRegex(ValueError, "cannot contain spaces"):
            self.call(self.root / "drive-output")
        self.mount.assert_not_called()
        self.assertEqual(self.clip.read_bytes(), b"valuable output")

    def test_disabled_drive_does_not_reject_notebook_path_with_spaces(self):
        self.namespace["NOTEBOOK_DIR"] = self.root / "folder with spaces"
        with contextlib.redirect_stdout(io.StringIO()):
            self.setup(False)
        self.mount.assert_not_called()

    def test_file_at_output_path_is_preserved_and_rejected(self):
        comfy = self.workspace / "h3" / "ComfyUI" / "output"
        comfy.parent.mkdir(parents=True)
        comfy.write_bytes(b"keep file")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            self.call(self.root / "drive-output")
        self.mount.assert_not_called()
        self.assertEqual(comfy.read_bytes(), b"keep file")

    def test_new_file_appearing_during_migration_is_not_deleted(self):
        target = self.root / "drive-output"
        original_move = shutil.move
        late_file = self.gradio / "late.mp4"
        def move_and_add_file(*args):
            result = original_move(*args)
            late_file.write_bytes(b"late output")
            return result
        with patch.object(shutil, "move", side_effect=move_and_add_file):
            with self.assertRaises(OSError):
                self.call(target)
        self.assertEqual((target / "gradio_outputs" / "clip.mp4").read_bytes(), b"valuable output")
        self.assertEqual(late_file.read_bytes(), b"late output")


class ColabLauncherTests(unittest.TestCase):
    def test_terminal_launcher_uses_selected_venv_with_spaces(self):
        git = shutil.which("git")
        bash = Path(git).resolve().parent.parent / "bin" / "bash.exe" if os.name == "nt" and git else Path(shutil.which("bash") or "")
        if not bash.is_file():
            self.skipTest("Bash unavailable")
        with TemporaryDirectory(prefix="launcher spaces ") as directory:
            root = Path(directory).resolve()
            venv = root / "env with spaces"
            (venv / "bin").mkdir(parents=True)
            python = venv / "bin" / "python3"
            python.write_text("#!/usr/bin/env bash\nprintf 'selected-venv\\n'\n", encoding="utf-8")
            python.chmod(0o755)
            run_script = root / "run_h3.sh"
            run_script.write_text('printf "%s\\n" "$VIRTUAL_ENV"\npython3 --version\nprintf "%s\\n" "${PYTHONPATH-unset}"\n', encoding="utf-8")
            tree = ast.parse(cell_source("launcher_source ="))
            conditional = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "USE_RAM_FOR_QWEN")
            command_assignment = next(node for node in conditional.orelse if isinstance(node, ast.Assign))
            source_assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "launcher_source" for t in node.targets))
            def shell_path(path):
                value = path.as_posix()
                return "/" + value[0].lower() + value[2:] if os.name == "nt" else value
            namespace = dict(Path=PurePosixPath, shlex=shlex, VENV_DIR=shell_path(venv), workspace=PurePosixPath(shell_path(root)))
            exec(compile(ast.Module(body=[command_assignment, source_assignment], type_ignores=[]), "<launcher>", "exec"), namespace)
            launcher = root / "run_h3_colab.sh"
            launcher.write_text(namespace["launcher_source"], encoding="utf-8")
            env = dict(os.environ, VIRTUAL_ENV="wrong-environment", PYTHONPATH="inherited-conflict")
            result = subprocess.run([str(bash), launcher.as_posix()], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [shell_path(venv), "selected-venv", "unset"])
            python.unlink()
            failed = subprocess.run([str(bash), launcher.as_posix()], env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Python environment missing", failed.stderr)


class InterruptingOutput:
    closed = False

    def __iter__(self):
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


class ColabLogStreamingTests(unittest.TestCase):
    def setUp(self):
        self.run = load_function("run_colab_launcher", dict(os=os, subprocess=subprocess, signal=signal))

    def run_child_fixture(self, script, output):
        original_popen = subprocess.Popen
        def open_fixture(command, **kwargs):
            self.assertEqual(command, ["bash", "fixture.sh"])
            self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
            return original_popen([sys.executable, "-u", "-c", script], **kwargs)
        with patch.object(subprocess, "Popen", side_effect=open_fixture), contextlib.redirect_stdout(output):
            self.run("fixture.sh")

    def test_stdout_and_stderr_are_visible_before_child_exits(self):
        with TemporaryDirectory() as directory:
            release = Path(directory) / "log-seen"
            class NotebookOutput(io.StringIO):
                def write(self, text):
                    if "server ready" in text:
                        release.touch()
                    return super().write(text)
            output = NotebookOutput()
            script = f"""import sys, time
from pathlib import Path
print('server ready', flush=True)
deadline = time.monotonic() + 5
while not Path({str(release)!r}).exists():
    if time.monotonic() >= deadline:
        sys.exit('Startup output was not relayed while the process was running')
    time.sleep(0.01)
print('Public URL: https://fixture.gradio.live', file=sys.stderr, flush=True)
sys.stdout.buffer.write(b'\\xffinvalid byte\\n')
sys.stdout.buffer.flush()
"""
            self.run_child_fixture(script, output)
            self.assertTrue(release.exists())
            self.assertIn("server ready", output.getvalue())
            self.assertIn("https://fixture.gradio.live", output.getvalue())
            self.assertIn("\ufffdinvalid byte", output.getvalue())
            self.assertIn("This cell stays running", output.getvalue())

    def test_startup_failure_keeps_error_log_and_reports_exit_code(self):
        output = io.StringIO()
        script = "import sys; print('setup failed', file=sys.stderr, flush=True); sys.exit(7)"
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_child_fixture(script, output)
        self.assertEqual(raised.exception.returncode, 7)
        self.assertIn("setup failed", output.getvalue())

    def test_interrupt_signals_linux_process_group_and_closes_log_pipe(self):
        proc = Mock(pid=12345, stdout=InterruptingOutput())
        proc.poll.return_value = None
        proc.wait.return_value = 0
        with patch.object(subprocess, "Popen", return_value=proc) as popen, patch.object(os, "name", "posix"), patch.object(os, "killpg", create=True) as killpg, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self.run("fixture.sh")
            killpg.assert_called_once_with(proc.pid, signal.SIGTERM)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
        proc.wait.assert_called_once_with(timeout=10)
        self.assertTrue(proc.stdout.closed)

    def test_unresponsive_process_group_is_force_killed(self):
        proc = Mock(pid=12345, stdout=InterruptingOutput())
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("bash", 10), 0]
        with patch.object(subprocess, "Popen", return_value=proc), patch.object(os, "name", "posix"), patch.object(os, "killpg", create=True) as killpg, patch.object(signal, "SIGKILL", 9, create=True), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self.run("fixture.sh")
            self.assertEqual(killpg.call_count, 2)
            self.assertEqual(killpg.call_args.args, (proc.pid, signal.SIGKILL))
        self.assertEqual(proc.wait.call_count, 2)
        self.assertTrue(proc.stdout.closed)

    def test_interrupt_uses_process_termination_on_windows(self):
        proc = Mock(stdout=InterruptingOutput())
        proc.poll.return_value = None
        proc.wait.return_value = 0
        with patch.object(subprocess, "Popen", return_value=proc), patch.object(os, "name", "nt"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self.run("fixture.sh")
        proc.terminate.assert_called_once()
        self.assertTrue(proc.stdout.closed)


class ColabCloudflareTests(unittest.TestCase):
    def setUp(self):
        self.namespace = dict(Path=Path, subprocess=subprocess, threading=threading, re=re, os=os)
        self.stop = load_function("stop_cloudflare", self.namespace)
        self.drain = load_function("drain_cloudflare_logs", self.namespace)
        self.launch = load_function("launch_cloudflare", self.namespace)

    def test_reader_drains_logs_after_first_url_and_announces_once(self):
        class LogStream(io.StringIO):
            count = 0
            def __next__(self):
                line = super().__next__()
                self.count += 1
                return line
        stream = LogStream("https://fixture.trycloudflare.com\nhttps://fixture.trycloudflare.com\nlater log\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.drain(SimpleNamespace(stdout=stream))
        self.assertEqual(stream.count, 3)
        self.assertTrue(stream.closed)
        self.assertEqual(output.getvalue().count("Cloudflare Tunnel Public URL:"), 1)

    def test_unresponsive_tunnel_is_killed_and_reaped(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("cloudflared", 5), 0]
        self.stop(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 2)

    def test_launch_merges_streams_and_starts_drain_thread(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / ".colab_bin" / "cloudflared"
            binary.parent.mkdir()
            binary.touch()
            self.namespace["NOTEBOOK_DIR"] = root
            proc = Mock()
            with patch.object(subprocess, "Popen", return_value=proc) as popen, patch.object(threading, "Thread") as thread:
                self.assertIs(self.launch(), proc)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            thread.assert_called_once_with(target=self.drain, args=(proc,), daemon=True)
            thread.return_value.start.assert_called_once()

    def test_interruption_while_starting_log_reader_stops_tunnel(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / ".colab_bin" / "cloudflared"
            binary.parent.mkdir()
            binary.touch()
            self.namespace["NOTEBOOK_DIR"] = root
            proc = Mock()
            proc.poll.return_value = None
            with patch.object(subprocess, "Popen", return_value=proc), patch.object(threading, "Thread") as thread:
                thread.return_value.start.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    self.launch()
            proc.terminate.assert_called_once()
            proc.wait.assert_called_once_with(timeout=5)

    def test_interrupted_app_cleans_up_old_and_new_tunnels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("h3_ui/application.py", "h3_models.py", ".colab_bin/cloudflared"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            old, new = Mock(), Mock()
            old.poll.return_value = new.poll.return_value = None
            namespace = dict(Path=Path, NOTEBOOK_DIR=root, WORKSPACE_DIR=str(root), COLAB_LAUNCHER=str(root / "run_h3_colab.sh"), _CLOUDFLARE_PROCESS=old)
            tree = ast.parse(cell_source("def launch_cloudflare("))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENABLE_CLOUDFLARE_TUNNEL" for t in node.targets):
                    node.value = ast.Constant(True)
            app = Mock(pid=12345, stdout=InterruptingOutput())
            app.poll.return_value = None
            app.wait.return_value = 0
            with patch.object(os, "chdir"), patch.object(subprocess, "Popen", side_effect=[new, app]), patch.object(threading, "Thread"), patch.object(os, "killpg", create=True), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    exec(compile(ast.fix_missing_locations(tree), "<launch cell>", "exec"), namespace)
            self.assertTrue(app.stdout.closed)
            old.terminate.assert_called_once()
            new.terminate.assert_called_once()
            self.assertIsNone(namespace["_CLOUDFLARE_PROCESS"])


if __name__ == "__main__":
    unittest.main()
