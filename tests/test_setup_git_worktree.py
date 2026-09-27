"""Regression checks for nested dependency checkouts."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from setup_h3 import _git_worktree_is_valid


class GitWorktreeOwnershipTests(unittest.TestCase):
    def test_parent_repository_does_not_own_nested_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            child = root / "h3" / "ComfyUI"
            child.mkdir(parents=True)

            self.assertTrue(_git_worktree_is_valid(root))
            self.assertFalse(_git_worktree_is_valid(child))

            subprocess.run(["git", "init", "-q", str(child)], check=True)
            self.assertTrue(_git_worktree_is_valid(child))


if __name__ == "__main__":
    unittest.main()
