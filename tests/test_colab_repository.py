"""Check notebook repository selection using local Git fixtures."""

import ast
import contextlib
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class ColabRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="colab repo ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.notebook_dir = self.root / "notebook directory"
        self.notebook_dir.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.upstream)], check=True)
        for name in (
            "run_h3.sh", "setup_h3.py", "gradio_app.py", "h3_models.py",
            "h3_app/__init__.py", "h3_ui/application.py", "README.md",
        ):
            target = self.upstream / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n", encoding="utf-8")
        self.git(self.upstream, "add", ".")
        self.git(
            self.upstream, "-c", "user.name=Notebook Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "Fixture repository",
        )

    def git(self, directory, *args):
        return subprocess.run(
            ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True,
        )

    def clone(self, target):
        subprocess.run(
            ["git", "clone", "-q", self.upstream.as_uri(), str(target)],
            check=True, capture_output=True,
        )

    def run_setup(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8")
        )
        source = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if "def is_complete_repository(" in "".join(cell["source"])
        ).split("os.chdir(WORKSPACE_DIR)", 1)[0]
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "REPO_URL" for target in node.targets
            ):
                node.value = ast.Constant(self.upstream.as_uri())
        namespace = {"NOTEBOOK_DIR": self.notebook_dir}
        with patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(tree), "<colab setup>", "exec"), namespace)
        return namespace

    def test_notebook_only_clones_into_minimax_h3_and_reuses_it_on_rerun(self):
        target = self.notebook_dir / "minimax-h3"
        namespace = self.run_setup()
        self.assertEqual(Path(namespace["WORKSPACE_DIR"]), target)
        self.assertTrue(namespace["is_complete_repository"](target))
        self.assertEqual(Path(self.run_setup()["WORKSPACE_DIR"]), target)
        self.assertEqual(list(self.notebook_dir.glob("minimax-h3-incomplete-*")), [])

    def test_fresh_clone_skips_redundant_fetch_but_existing_clone_updates(self):
        with patch.object(subprocess, "run", wraps=subprocess.run) as run:
            self.run_setup()
        self.assertFalse(any("fetch" in call.args[0] for call in run.call_args_list))
        with patch.object(subprocess, "run", wraps=subprocess.run) as run:
            self.run_setup()
        self.assertEqual(sum("fetch" in call.args[0] for call in run.call_args_list), 1)

    def test_partial_files_beside_notebook_are_preserved_and_do_not_count_as_repository(self):
        launcher = self.notebook_dir / "run_h3.sh"
        launcher.write_text("local launcher", encoding="utf-8")
        namespace = self.run_setup()
        self.assertEqual(Path(namespace["WORKSPACE_DIR"]), self.notebook_dir / "minimax-h3")
        self.assertEqual(launcher.read_text(encoding="utf-8"), "local launcher")
        self.assertFalse((self.notebook_dir / ".git").exists())

    def test_complete_repository_beside_notebook_is_reused(self):
        self.clone(self.notebook_dir)
        namespace = self.run_setup()
        self.assertEqual(Path(namespace["WORKSPACE_DIR"]), self.notebook_dir)
        self.assertFalse((self.notebook_dir / "minimax-h3").exists())

    def test_complete_child_checkout_keeps_local_files(self):
        target = self.notebook_dir / "minimax-h3"
        self.clone(target)
        marker = target / "local.txt"
        marker.write_text("keep", encoding="utf-8")
        namespace = self.run_setup()
        self.assertEqual(Path(namespace["WORKSPACE_DIR"]), target)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(self.notebook_dir.glob("minimax-h3-incomplete-*")), [])

    def test_git_history_missing_core_module_is_replaced_even_if_file_exists(self):
        target = self.notebook_dir / "minimax-h3"
        self.clone(target)
        self.git(target, "rm", "--cached", "h3_models.py")
        self.git(
            target, "-c", "user.name=Notebook Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "Remove core module from history",
        )
        namespace = self.run_setup()
        self.assertTrue(namespace["is_complete_repository"](target))
        backups = list(self.notebook_dir.glob("minimax-h3-incomplete-*"))
        self.assertEqual(len(backups), 1)
        self.assertTrue((backups[0] / "h3_models.py").is_file())

    def test_missing_tracked_file_preserves_incomplete_checkout_and_clones_fresh(self):
        target = self.notebook_dir / "minimax-h3"
        self.clone(target)
        (target / "README.md").unlink()
        (target / "local.txt").write_text("preserve", encoding="utf-8")
        namespace = self.run_setup()
        self.assertEqual(Path(namespace["WORKSPACE_DIR"]), target)
        self.assertTrue(namespace["is_complete_repository"](target))
        backups = list(self.notebook_dir.glob("minimax-h3-incomplete-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "local.txt").read_text(encoding="utf-8"), "preserve")
        self.assertFalse((backups[0] / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
