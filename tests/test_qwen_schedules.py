"""CPU regression values from Qwen 2.1's official Diffusers schedule.

Pipeline revision: 80c7ed262aeffbeb43ef13ae04baeb9b84515a69.
Base config: 256/8192 tokens, 0.5/0.9 shifts, terminal 0.02.
No ComfyUI or GPU dependencies are imported.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


class QwenBaseScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        names = {"_qwen21_target_tokens", "H3Qwen21Sigmas"}
        nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        namespace = {"math": math, "torch": SimpleNamespace(
            tensor=lambda values, dtype: values, float32="float32",
        )}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        cls.scheduler = namespace["H3Qwen21Sigmas"]()

    def test_official_reference_values_at_square_rectangular_and_four_mp_sizes(self):
        # Frozen values at indices 1 and steps//2, calculated by the official
        # np.linspace -> exponential time shift -> stretch_shift_to_terminal.
        cases = (
            (64, 64, 25, .9783411785501731, .6648185221629315),
            (64, 64, 40, .9869636894187671, .6566663244001787),
            (48, 64, 25, .9773067548821420, .6542131056845337),
            (48, 64, 40, .9863160486844901, .6456232612386207),
            (128, 128, 25, .9874535438006040, .7748470054331775),
            (128, 128, 40, .9926459799383787, .7724376042030116),
        )
        for height, width, steps, second, middle in cases:
            with self.subTest(size=(height, width), steps=steps):
                native = {"samples": SimpleNamespace(shape=(1, 64, height, width))}
                empty = {"samples": SimpleNamespace(shape=(1, 4, height * 2, width * 2)),
                         "downscale_ratio_spacial": 8}
                schedule = self.scheduler.calculate(native, steps)[0]
                self.assertEqual(schedule, self.scheduler.calculate(empty, steps)[0])
                self.assertEqual(len(schedule), steps + 1)
                self.assertEqual(schedule[0], 1.)
                self.assertAlmostEqual(schedule[1], second, places=14)
                self.assertAlmostEqual(schedule[steps // 2], middle, places=14)
                self.assertAlmostEqual(schedule[-2], .02, places=14)
                self.assertEqual(schedule[-1], 0.)
                self.assertTrue(all(a > b for a, b in zip(schedule, schedule[1:])))

    def test_one_step_and_invalid_inputs(self):
        latent = {"samples": SimpleNamespace(shape=(1, 64, 64, 64))}
        self.assertEqual(self.scheduler.calculate(latent, 1)[0], [1., 0.])
        for steps in (0, 101):
            with self.assertRaisesRegex(ValueError, "between 1 and 100"):
                self.scheduler.calculate(latent, steps)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.scheduler.calculate({"samples": SimpleNamespace(shape=(1, 64, 0, 64))}, 40)


if __name__ == "__main__":
    unittest.main()
