"""CPU numerical checks for curve AdaLN bypass, without importing ComfyUI or torch."""

import ast
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class Tensor(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    @property
    def device(self):
        return "cpu"

    def to(self, device=None, dtype=None):
        return Tensor(self, dtype=dtype)

    def clamp(self, min=None, max=None):
        return np.clip(self, min, max).view(Tensor)

    def floor(self):
        return np.floor(self).view(Tensor)

    def long(self):
        return self.astype(np.int64)

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

    def add_(self, update, alpha):
        self += update * alpha
        return self


class BaseAdapter:
    def __init__(self, loaded_keys, weights):
        self.loaded_keys, self.weights = loaded_keys, weights


def helpers():
    path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    names = {"H3InplaceLoRAAdapter", "H3CurveInplaceLoRAAdapter",
             "_lightx_curve_time_embedding", "_lightx_curve_wrapper"}
    tree = ast.parse(path.read_text("utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    namespace = {
        "comfy": SimpleNamespace(weight_adapter=SimpleNamespace(LoRAAdapter=BaseAdapter)),
        "F": SimpleNamespace(linear=lambda x, weight: Tensor(x @ weight.T)),
        "torch": SimpleNamespace(
            float32=np.float32,
            tensor=lambda x, device, dtype: Tensor(x, dtype),
            lerp=lambda a, b, weight: Tensor(a + (b - a) * weight),
        ),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class LightXCurveTests(unittest.TestCase):
    def setUp(self):
        self.ns = helpers()
        self.context = ContextVar("curve_test", default=None)

    def test_exact_model_rows_are_interpolated_and_base_result_is_preserved(self):
        grid = Tensor([[0, 2], [4, 6], [8, 10]], np.float32)
        calls = []
        base = object()

        def original(*args):
            calls.append(args)
            return base

        embedding = self.ns["_lightx_curve_time_embedding"](original, grid, self.context)
        # Includes distinct audio, video and conditioning rows, plus clamped endpoints.
        times = [-0.1, 0.25, 0.5, 0.999, 1.1]

        def forward():
            self.assertIs(embedding(times, "cpu", np.float16), base)
            np.testing.assert_allclose(self.context.get()["dense"],
                                       [[0, 2], [2, 4], [4, 6], [7.992, 9.992], [8, 10]])
            self.assertEqual(calls, [(times, "cpu", np.float16)])

        self.ns["_lightx_curve_wrapper"](self.context)(forward)
        self.assertIsNone(self.context.get())

    def test_curve_adapter_applies_dense_delta_and_preserves_base_curve_input(self):
        adapter = self.ns["H3CurveInplaceLoRAAdapter"](
            set(), (Tensor([[3], [5]], np.float32), Tensor([[1, 2, 3]], np.float32),
                    2.0, None, None, None))
        adapter.curve_context = self.context
        adapter.multiplier = 0.5
        curve = Tensor(np.zeros((2, 8)), np.float32)
        dense = Tensor([[1, 2, 3], [4, 5, 6]], np.float32)
        base = Tensor([[7, 8], [9, 10]], np.float32)

        def original(value):
            self.assertIs(value, curve)
            return base.copy()

        def forward():
            self.context.get()["dense"] = dense
            result = adapter.bypass_forward(original, curve)
            np.testing.assert_allclose(result, [[49, 78], [105, 170]])

        self.ns["_lightx_curve_wrapper"](self.context)(forward)

    def test_wrapper_restores_context_after_failure(self):
        outer = {"dense": "outer"}
        token = self.context.set(outer)
        try:
            def failing():
                raise ValueError("sampling failed")
            with self.assertRaisesRegex(ValueError, "sampling failed"):
                self.ns["_lightx_curve_wrapper"](self.context)(failing)
            self.assertIs(self.context.get(), outer)
        finally:
            self.context.reset(token)

    def test_missing_or_mismatched_rows_fail_instead_of_skipping_adapters(self):
        adapter = self.ns["H3CurveInplaceLoRAAdapter"](
            set(), (Tensor([[1]], np.float32), Tensor([[1]], np.float32), None, None, None, None))
        adapter.curve_context = self.context
        curve = Tensor(np.zeros((2, 8)), np.float32)
        with self.assertRaisesRegex(RuntimeError, "without dense timestep"):
            adapter.lora_input(curve)
        token = self.context.set({"dense": Tensor([[1]], np.float32)})
        try:
            with self.assertRaisesRegex(RuntimeError, "rows do not match"):
                adapter.lora_input(curve)
        finally:
            self.context.reset(token)


if __name__ == "__main__":
    unittest.main()
