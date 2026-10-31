"""Background default-model preparation launched by run_h3.sh."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_prepare
from h3_models import MODEL_SPECS

ROOT = Path(__file__).resolve().parents[1]


def bash_path():
    git = shutil.which("git")
    if os.name == "nt" and git:
        return Path(git).resolve().parent.parent / "bin" / "bash.exe"
    return Path(shutil.which("bash") or "")


class GroupTests(unittest.TestCase):
    def test_groups_accept_aliases_all_and_none(self):
        self.assertEqual(h3_prepare.parse_groups("minimax_h3, qwen,h3"), ("h3", "qwen_image"))
        self.assertEqual(h3_prepare.parse_groups("all"), h3_prepare.GROUPS)
        for off in ("none", "off", "", "0"):
            self.assertEqual(h3_prepare.parse_groups(off), ())
        with self.assertRaises(argparse.ArgumentTypeError):
            h3_prepare.parse_groups("h3,ltx")

    def test_default_keys_are_known_model_files(self):
        h3 = h3_prepare.h3_model_keys()
        qwen = h3_prepare.qwen_image_model_keys()
        for keys in (h3, qwen, h3_prepare.drama3_model_keys()):
            self.assertTrue(keys)
            self.assertTrue(set(keys) <= set(MODEL_SPECS), keys)
        self.assertIn("singularity_fl2va", h3)
        self.assertIn("pika_flex_fl2va_lora", h3)
        self.assertFalse(any(key.startswith("qwen_image21_") for key in h3))
        self.assertTrue(all(key.startswith("qwen_image21_") for key in qwen))

    def test_plan_shares_one_download_and_adds_drama3_steps(self):
        with TemporaryDirectory() as directory:
            labels = [label for label, _ in h3_prepare.plan(("h3",), None, Path(directory))]
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0].startswith("MiniMax H3 model files"))
            labels = [label for label, _ in h3_prepare.plan(("qwen_image", "drama3"), None, Path(directory))]
        self.assertTrue(labels[0].startswith("Qwen Image 2.1 + Drama 3 ASR model files"))
        self.assertEqual(labels[1:], ["SGLang-Omni environment", "Drama 3 checkpoint", "Qwen VoiceDesign / Stable Audio"])

    def test_files_and_other_steps_can_run_apart(self):
        with TemporaryDirectory() as directory:
            files = [label for label, _ in h3_prepare.plan(("h3", "drama3"), None, Path(directory), others=False)]
            rest = [label for label, _ in h3_prepare.plan(("h3", "drama3"), None, Path(directory), files=False)]
            self.assertEqual(h3_prepare.plan(("h3",), None, Path(directory), files=False), [])
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(")") and "model files" in files[0])
        self.assertEqual(rest, ["SGLang-Omni environment", "Drama 3 checkpoint", "Qwen VoiceDesign / Stable Audio"])

    def test_catalog_only_sync_never_waits_for_a_background_download(self):
        import h3_models

        with TemporaryDirectory() as directory, patch.object(h3_models, "model_manifest_lock") as lock:
            config = h3_models.sync_models(root=Path(directory), manifest_path=Path(directory) / "manifest.json",
                                           model_keys=())
        lock.assert_not_called()
        self.assertEqual(config["manifest"], "manifest.json")


class MainTests(unittest.TestCase):
    def run_main(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = h3_prepare.main(list(args))
        return status, output.getvalue()

    def test_none_skips_everything(self):
        with patch.object(h3_prepare, "plan") as plan:
            status, output = self.run_main("--groups", "none")
        self.assertEqual(status, 0)
        plan.assert_not_called()
        self.assertIn("off", output)

    def test_failed_step_warns_without_stopping_the_others(self):
        ran = []

        def broken():
            raise RuntimeError("Hub unreachable\ndetails")

        steps = [("broken", broken), ("fine", lambda: ran.append("fine"))]
        with TemporaryDirectory() as directory, patch.object(h3_prepare, "plan", return_value=steps):
            status, output = self.run_main("--groups", "h3", "--install-dir", directory)
        self.assertEqual(status, 1)
        self.assertEqual(ran, ["fine"])
        self.assertIn("WARNING: broken not prepared (Hub unreachable)", output)
        self.assertIn("Ready: fine", output)


class StartupStatsTests(unittest.TestCase):
    def test_startup_stats_is_a_no_op_without_linux_proc(self):
        import h3_startup_stats

        with patch.object(h3_startup_stats.Path, "is_file", return_value=False):
            self.assertEqual(h3_startup_stats.main(["--max-seconds", "0"]), 0)


class LauncherOptionTests(unittest.TestCase):
    def setUp(self):
        self.bash = bash_path()
        if not self.bash.is_file():
            self.skipTest("Bash unavailable")

    def run_launcher(self, *args, env=None):
        return subprocess.run(
            [str(self.bash), (ROOT / "run_h3.sh").as_posix(), *args],
            env=dict(os.environ, **(env or {})), capture_output=True, text=True, timeout=30,
        )

    def test_help_lists_groups(self):
        result = self.run_launcher("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--prepare GROUPS", result.stdout)
        self.assertIn("qwen_image", result.stdout)

    def test_unknown_group_or_option_stops_before_setup(self):
        for args, env in ((("--prepare", "h3,ltx"), None), ((), {"H3_PREPARE_MODELS": "nope"}), (("--bogus",), None)):
            result = self.run_launcher(*args, env=env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("[h3-run error]", result.stderr)
            self.assertNotIn("Runtime Python", result.stdout)

    def test_launcher_starts_preparation_before_the_servers(self):
        source = (ROOT / "run_h3.sh").read_text(encoding="utf-8")
        self.assertIn('PREPARE_MODELS="${H3_PREPARE_MODELS-h3}"', source)
        self.assertLess(source.index('h3_prepare.py'), source.index('starting ComfyUI at'))
        # Model files start before setup; Gradio starts beside ComfyUI, not after it.
        self.assertLess(source.index('PREPARE_FILES_EARLY=1'), source.index('running automatic setup'))
        self.assertLess(source.index('log "Starting Gradio"'), source.index('log "Waiting for ComfyUI"'))
        # The resource sampler watches the early download and can be turned off.
        self.assertLess(source.index('PREPARE_FILES_EARLY=1'), source.index('h3_startup_stats.py'))
        self.assertIn('H3_STARTUP_STATS:-1', source)
        # The servers decide when the launcher exits; preparation never does.
        self.assertIn('wait -n "$COMFY_PID" "$GRADIO_PID"', source)


if __name__ == "__main__":
    unittest.main()
