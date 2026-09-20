"""Runtime contract for the high-resolution refinement compiler guard."""

import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest


def load_guard_helpers():
    path = (
        Path(__file__).resolve().parents[1]
        / "custom_nodes/H3Acceleration/__init__.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {
        "_H3_REFINEMENT_COMPILER_LOCK",
        "_h3_refinement_video_volume",
        "_make_h3_refinement_compiler_wrapper",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    compiler_args = SimpleNamespace(disable_comfy_compiler=False)
    namespace = {
        "comfy": SimpleNamespace(cli_args=SimpleNamespace(args=compiler_args)),
        "logging": logging,
        "math": math,
        "threading": threading,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace, compiler_args


class RefinementCompilerGuardTests(unittest.TestCase):
    def setUp(self):
        self.namespace, self.compiler_args = load_guard_helpers()
        self.wrapper = self.namespace["_make_h3_refinement_compiler_wrapper"](
            1_500_000
        )

    @staticmethod
    def latent(shape):
        video = SimpleNamespace(shape=shape)
        audio = SimpleNamespace(shape=(1, 32, 2, 1200))
        return SimpleNamespace(tensors=(video, audio))

    def test_small_refinement_keeps_compiler_enabled(self):
        observed = []

        def executor(value):
            observed.append(self.compiler_args.disable_comfy_compiler)
            return value

        latent = self.latent((1, 24, 40, 44, 80))
        self.assertIs(self.wrapper(executor, latent), latent)
        self.assertEqual(observed, [False])
        self.assertFalse(self.compiler_args.disable_comfy_compiler)

    def test_two_k_ten_second_refinement_keeps_compiler_enabled(self):
        observed = []

        def executor(value):
            observed.append(self.compiler_args.disable_comfy_compiler)
            return value

        latent = self.latent((1, 24, 72, 90, 160))
        self.assertIs(self.wrapper(executor, latent), latent)
        self.assertEqual(observed, [False])
        self.assertFalse(self.compiler_args.disable_comfy_compiler)

    def test_large_refinement_bypasses_and_restores_compiler(self):
        observed = []

        def executor(value):
            observed.append(self.compiler_args.disable_comfy_compiler)
            return value

        latent = self.latent((1, 24, 107, 124, 124))
        self.assertIs(self.wrapper(executor, latent), latent)
        self.assertEqual(observed, [True])
        self.assertFalse(self.compiler_args.disable_comfy_compiler)

    def test_compiler_setting_is_restored_after_failure(self):
        latent = self.latent((1, 24, 107, 124, 124))

        def fail(_value):
            self.assertTrue(self.compiler_args.disable_comfy_compiler)
            raise RuntimeError("sampling failed")

        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            self.wrapper(fail, latent)
        self.assertFalse(self.compiler_args.disable_comfy_compiler)


if __name__ == "__main__":
    unittest.main()
