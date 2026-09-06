"""Numerical node tests run with CPU PyTorch; no ComfyUI installation needed."""

import ast
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def load_node():
    path = (
        Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "H3SemanticBridge"
    )
    namespace = {"torch": torch, "F": F, "math": math}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["H3SemanticBridge"]


@unittest.skipIf(torch is None, "CPU PyTorch is required for numerical adapter tests")
class SemanticBridgeNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node_type = load_node()
        generator = torch.Generator().manual_seed(42)
        cls.weights = {}
        for name, rows, cols in (
            ("fc1", 512, 5120),
            ("fc2", 512, 512),
            ("fc3", 5120, 512),
        ):
            cls.weights[name + ".weight"] = (
                torch.randn(rows, cols, generator=generator) * 0.01
            )
            cls.weights[name + ".bias"] = torch.randn(rows, generator=generator) * 0.01

    def test_zero_is_exact_identity_without_loading(self):
        conditioning = [[torch.zeros(1, 2, 5120), {"keyframes": []}]]
        with patch.object(self.node_type, "_load_weights") as load:
            self.assertIs(self.node_type().apply(conditioning, 0)[0], conditioning)
            load.assert_not_called()

    def test_dtype_metadata_and_cached_inputs_are_preserved(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            native = torch.linspace(-1, 2, 10240).reshape(1, 2, 5120).to(dtype)
            original = native.clone()
            keyframes = [{"resolved_frame_index": 0}]
            metadata = {"minimax_keyframes": keyframes, "minimax_frame_count": 124}
            conditioning = [[native, metadata]]
            # Independent nn.Module formulation of the released architecture.
            layers = torch.nn.Sequential(
                torch.nn.Linear(5120, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 5120),
            )
            with torch.no_grad():
                for index, name in ((0, "fc1"), (2, "fc2"), (4, "fc3")):
                    layers[index].weight.copy_(self.weights[name + ".weight"])
                    layers[index].bias.copy_(self.weights[name + ".bias"])
                h = native.float()
                projected = layers(
                    h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
                )
                projected *= torch.sqrt(
                    h.pow(2).mean(-1, keepdim=True) + 1e-8
                ) / torch.sqrt(projected.pow(2).mean(-1, keepdim=True) + 1e-8)
                expected = (h + 0.1 * (projected - h)).to(dtype)
            with patch.object(
                self.node_type, "_load_weights", return_value=self.weights
            ):
                result = self.node_type().apply(conditioning, 0.1)[0]
            torch.testing.assert_close(result[0][0], expected, rtol=0, atol=0)
            torch.testing.assert_close(native, original, rtol=0, atol=0)
            self.assertEqual(result[0][0].dtype, dtype)
            self.assertIs(result[0][1]["minimax_keyframes"], keyframes)
            self.assertNotIn("semantic_bridge_alpha", metadata)
            self.assertEqual(result[0][1]["semantic_bridge_alpha"], 0.1)

    def test_invalid_inputs_fail_before_loading(self):
        with patch.object(self.node_type, "_load_weights") as load:
            for alpha in (-1, 2, float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    self.node_type().apply([], alpha)
            with self.assertRaises(ValueError):
                self.node_type().apply([[torch.zeros(1, 2, 128), {}]], 0.1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
