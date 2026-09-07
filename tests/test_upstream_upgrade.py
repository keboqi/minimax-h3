"""CPU contract tests against downloaded, pinned upstream source.

Set H3_UPSTREAM_SOURCE_DIR to a directory containing minimax_trt_node.py and
comfy-requirements.txt from the pins in h3_node_patches/h3_requirements.
The audit cache is used by default. No upstream module or GPU code is imported.
"""
import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from h3_node_patches import _patch_trt_vae_source
from h3_requirements import COMFY_FRONTEND_VERSION, COMFY_KITCHEN_VERSION

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path(os.environ.get("H3_UPSTREAM_SOURCE_DIR", ROOT / ".cache/upstream-upgrade"))


class Tensor(np.ndarray):
    """Only the shape/array operations needed by temporal chunk assembly."""
    def repeat(self, *sizes):
        return np.tile(np.asarray(self), sizes).view(Tensor)

    def contiguous(self):
        return self.copy()

    def to(self, other):
        return self


def tensor(value):
    return np.asarray(value, dtype=np.float32).view(Tensor)


def extracted_class(source, name, methods):
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    node = ast.ClassDef(name=name, bases=[], keywords=[], decorator_list=[],
                        body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods])
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(cat=lambda arrays, dim: np.concatenate(arrays, axis=dim).view(Tensor)),
                 "math": math, "os": os, "logger": SimpleNamespace(warning=lambda *a: None)}
    exec(compile(tree, "<upstream contract>", "exec"), namespace)
    return namespace[name]


@unittest.skipUnless((SOURCES / "minimax_trt_node.py").is_file(), "pinned upstream source not downloaded")
class TensorRTUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (SOURCES / "minimax_trt_node.py").read_text(encoding="utf-8")
        cls.patched, cls.changed = _patch_trt_vae_source(cls.original)

    def test_patch_compiles_and_reapplication_is_noop(self):
        self.assertTrue(self.changed)
        compile(self.patched, "upstream", "exec")
        self.assertEqual(_patch_trt_vae_source(self.patched), (self.patched, False))
        with self.assertRaises(RuntimeError):
            _patch_trt_vae_source(self.original.replace("    return torch.cat(dec_chunks, dim=2)", "    return unexpected"))

    def decoder(self):
        cls = extracted_class(self.patched, "MiniMaxH3TRTVAE", {
            "decode", "decode_temporal", "decode_output_shape", "_decode_temporal_chunks",
            "_decode_temporal_frame_plan", "_decode_temporal_pad_frames",
        })
        obj = cls()
        obj.decoder_runner = object()
        obj.latents_mean, obj.latents_std = tensor(0), tensor(1)
        obj.vae_ratio, obj.vae_ratio_t = 16, 4
        obj.clip_length, obj.token_drop = 17, 3
        obj.frame_pre_padding, obj.tokens_chunk_size = 3, 5
        obj.token_overlap, obj.frame_overlap = 2, 5
        obj._finalize_pixels = lambda x: x
        obj.blend = lambda a, b, *args, **kwargs: b
        def tiled_decode(z):
            self.assertEqual(z.shape[2], 7, "every TRT call must use its fixed seven-token profile")
            return tensor(np.arange(28).reshape(1, 1, 28, 1, 1))
        obj.tiled_decode = tiled_decode
        return obj

    def test_temporal_output_matches_frame_plan_including_partial_tails(self):
        obj = self.decoder()
        for length in range(1, 101):
            with self.subTest(latent_frames=length):
                z = tensor(np.zeros((1, 24, length, 1, 1)))
                actual = obj.decode(z)
                self.assertEqual(actual.shape[2], obj.decode_output_shape(z.shape)[2])

    def test_single_frame_keeps_first_clip_frame_not_last_padded_frame(self):
        decoded = self.decoder().decode(tensor(np.zeros((1, 24, 1, 1, 1))))
        self.assertEqual(decoded.shape[2], 1)
        self.assertEqual(decoded.item(), 3)  # first real frame after three pre-padding frames

    def test_quantization_detection_reads_graph_with_neutral_filename(self):
        cls = extracted_class(self.patched, "MiniMaxH3TRTCompilerNode", {"_is_onnx_quantized"})
        graph = SimpleNamespace(initializer=[], node=[SimpleNamespace(op_type="DequantizeLinear")])
        with patch.dict("sys.modules", {"onnx": SimpleNamespace(load=lambda *a, **kw: SimpleNamespace(graph=graph))}):
            self.assertTrue(cls._is_onnx_quantized("decoder.onnx"))
            graph.node = []
            self.assertFalse(cls._is_onnx_quantized("decoder.onnx"))


@unittest.skipUnless((SOURCES / "comfy-requirements.txt").is_file(), "pinned Comfy requirements not downloaded")
class ComfyRequirementsTests(unittest.TestCase):
    def test_shared_versions_match_actual_upstream_requirements(self):
        lines = (SOURCES / "comfy-requirements.txt").read_text().splitlines()
        self.assertIn(f"comfyui-frontend-package=={COMFY_FRONTEND_VERSION}", lines)
        self.assertIn(f"comfy-kitchen=={COMFY_KITCHEN_VERSION}", lines)
        self.assertIn("comfy-aimdo==0.5.2", lines)


if __name__ == "__main__":
    unittest.main()
