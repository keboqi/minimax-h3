"""CPU loading contracts and optional exact-source codec characterization."""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("H3_COMPONENT_UPGRADE_SOURCE_DIR",
                               ROOT / ".cache/component-upgrade-2026-10-17"))


def extract_class(source, name, namespace, methods=None):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.bases = []
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<component contract>", "exec"), namespace)
    return namespace[name]


class LightVAELoadingTests(unittest.TestCase):
    def test_components_pruning_quantization_and_original_fallback(self):
        for encoder, decoder, quantized in ((True, False, False), (False, True, False),
                                             (True, True, False), (False, True, True),
                                             (True, True, True)):
            with self.subTest(encoder=encoder, decoder=decoder, quantized=quantized):
                constructed = []
                kept = [0, 3, 8, 11]

                class Native:
                    def __init__(self, **kwargs):
                        self.options = kwargs
                        self.latents_mean, self.latents_std = object(), object()
                        constructed.append(self)

                    def load_state_dict(self, state, strict, assign):
                        # Strict component loading must never see unselected weights.
                        self.state = state
                        self.strict, self.assign = strict, assign

                    def eval(self):
                        return self

                    def requires_grad_(self, value):
                        return self

                    def to(self, **kwargs):
                        return self

                class VAE:
                    def model_size(self):
                        return 0

                    def encode(self, value):
                        return ("lite encode", value)

                    def encode_tiled(self, value, **kwargs):
                        return ("lite tiled encode", value, kwargs)

                    def decode(self, value, options):
                        return ("lite decode", value, options)

                    def decode_tiled(self, value, **kwargs):
                        return ("lite tiled decode", value, kwargs)

                class Patcher:
                    def __init__(self, model, **kwargs):
                        self.model = model
                        self.__dict__.update(kwargs)

                base = Mock(first_stage_model=Native(), vae_dtype="bf16", device="cuda",
                            patcher=Patcher(None, offload_device="cpu"))
                quantization = object() if quantized else None
                detect = Mock(return_value=quantization)
                mixed = Mock(return_value=object())
                plain = object()
                comfy = SimpleNamespace(sd=SimpleNamespace(VAE=VAE),
                    ops=SimpleNamespace(mixed_precision_ops=mixed, disable_weight_init=plain),
                    utils=SimpleNamespace(detect_layer_quantization=detect))
                def load(path, component):
                    return {component + ".weight": path}, kept

                cls = extract_class((ROOT / "custom_nodes/H3LightVAE/h3_vae.py").read_text(),
                    "LightVAE", {"comfy": comfy, "MiniMaxH3VideoVAE": Native,
                                 "model_management": SimpleNamespace(archive_model_dtypes=Mock()),
                                 "load_component": load})
                vae = cls(base, "decoder" if decoder else None, "encoder" if encoder else None)
                model = constructed[-1]
                self.assertEqual(model.options["encoder"], encoder)
                self.assertEqual(model.options["decoder"], decoder)
                self.assertEqual(model.options["encoder_kept_blocks"],
                                 {(0, 0), (1, 1), (4, 0), (5, 1)} if encoder else None)
                self.assertEqual(model.options["num_layers"], len(kept) if decoder else 36)
                self.assertIs(model.options["operations"], mixed.return_value if quantized else plain)
                self.assertEqual(set(model.state), {"latents_mean", "latents_std"}
                                 | ({"encoder.weight"} if encoder else set())
                                 | ({"decoder.weight"} if decoder else set()))
                self.assertTrue(model.strict and model.assign)
                self.assertIs(vae.patcher.model, model)
                self.assertEqual(vae.patcher.offload_device, "cpu")
                for use, method, args, kwargs in (
                    (encoder, "encode", ("pixels",), {}),
                    (encoder, "encode_tiled", ("pixels",), {"tile_x": 128}),
                    (decoder, "decode", ("latents", {"option": True}), {}),
                    (decoder, "decode_tiled", ("latents",), {"tile_x": 128}),
                ):
                    result = getattr(vae, method)(*args, **kwargs)
                    if use:
                        self.assertTrue(result[0].startswith("lite"))
                    else:
                        getattr(base, method).assert_called_once_with(*args, **kwargs)
                        self.assertIs(result, getattr(base, method).return_value)
                self.assertIsInstance(base.first_stage_model, Native)
                self.assertIsNot(base.first_stage_model, model)


@unittest.skipUnless((UPSTREAM / "core-vae.py").is_file(), "exact pinned H3 VAE not downloaded")
class NativeEncoderPruningTests(unittest.TestCase):
    def test_native_pairs_select_the_original_flat_residual_indices(self):
        source = (UPSTREAM / "core-vae.py").read_text()
        def residual(**kwargs):
            return SimpleNamespace(kind="residual", **kwargs)

        def shortcut(**kwargs):
            return SimpleNamespace(kind="shortcut", **kwargs)
        cls = extract_class(source, "EncoderFCN3D", {
            "nn": SimpleNamespace(Module=SimpleNamespace, ModuleList=list),
            "CausalConv3d": Mock(), "Downsample3D": Mock(), "group_norm_3d": Mock(),
            "ResnetBlock3D": residual, "ResnetShortcut3D": shortcut,
        }, {"__init__"})
        for kept in ([], [0, 3, 8, 11], list(range(12))):
            with self.subTest(kept=kept):
                encoder = cls(ch=128, ch_mult=(1, 2, 2, 4, 4, 8),
                    space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1),
                    num_res_blocks=2, in_channels=3, z_channels=24,
                    kept_blocks={divmod(index, 2) for index in kept})
                blocks = [block for stage in encoder.down for block in stage.block]
                self.assertEqual([i for i, block in enumerate(blocks) if block.kind == "residual"], kept)
                # Pruned blocks retain the channel-changing shortcut dimensions.
                self.assertEqual([(b.in_channels, b.out_channels) for b in blocks],
                    [(128, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 256),
                     (256, 512), (512, 512), (512, 512), (512, 512), (512, 1024), (1024, 1024)])


@unittest.skipUnless((UPSTREAM / "drama-codec.py").is_file(), "exact pinned codec not downloaded")
class DramaCodecWindowTests(unittest.TestCase):
    def test_windows_preserve_samples_and_bound_decode_size(self):
        source = (UPSTREAM / "drama-codec.py").read_text()
        tree = ast.parse(source)
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {"DECODE_WINDOW", "DECODE_CONTEXT", "FRAME_LENGTH"}:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        owner = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                     and any(isinstance(m, ast.FunctionDef) and m.name == "decode"
                             and any(isinstance(ref, ast.Name) and ref.id == "DECODE_WINDOW"
                                     for ref in ast.walk(m)) for m in n.body))
        cls = extract_class(source, owner,
            {**constants, "torch": SimpleNamespace(cat=lambda pieces, dim: np.concatenate(pieces, axis=dim))},
            {"decode"})
        for frames in (1, 255, 256, 257, 575, 576, 577, 1800):
            with self.subTest(frames=frames):
                lengths = []
                codec = cls()
                def dequantize(codes, dtype):
                    lengths.append(codes.shape[-1])
                    return codes
                codec.quantizer = SimpleNamespace(decode=dequantize)
                codec.decoder = lambda values: np.repeat(values, constants["FRAME_LENGTH"], axis=-1)
                codes = np.arange(frames).reshape(1, 1, frames)
                actual = codec.decode(codes, "float32")
                np.testing.assert_array_equal(actual, np.repeat(codes[:, 0], constants["FRAME_LENGTH"], axis=-1))
                self.assertLessEqual(max(lengths), constants["DECODE_WINDOW"] + constants["DECODE_CONTEXT"])

    def test_new_code_dictionary_decodes_empty_and_multiple_takes(self):
        source = (UPSTREAM / "drama-nodes.py").read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "decode")

        class Tensor(np.ndarray):
            def split(self, sizes, dim):
                return np.split(self, np.cumsum(sizes)[:-1], axis=dim)

            def to(self, device):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

        def tensor(values):
            return np.asarray(values, dtype=np.float32).view(Tensor)

        load = Mock()
        namespace = {
            "FRAME_LENGTH": 2048, "SAMPLE_RATE": 44100,
            "DECODE_CONTEXT": 320, "DECODE_WINDOW": 256, "DECODE_FRAME_BYTES": 100,
            "torch": SimpleNamespace(float32="float32", zeros=lambda shape: tensor(np.zeros(shape)),
                                     cat=lambda values, dim: tensor(np.concatenate(values, axis=dim))),
            "comfy": SimpleNamespace(model_management=SimpleNamespace(
                load_models_gpu=load, cuda_device_context=lambda device: nullcontext())),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<upstream decode>", "exec"), namespace)
        codec = SimpleNamespace(patcher=SimpleNamespace(load_device="cuda"),
            model=SimpleNamespace(decode=lambda codes, dtype: tensor(codes[:, 0])))
        for sizes in ([0], [2, 0, 3], [600]):
            with self.subTest(sizes=sizes):
                load.reset_mock()
                codes = {"codes": tensor(np.arange(sum(sizes)).reshape(1, -1)),
                         "takes": sizes, "lines": []}
                actual = namespace["decode"](codec, codes, gap=1 / 44100)
                if not sum(sizes):
                    self.assertEqual(actual.shape, (1, 2048))
                    load.assert_not_called()
                else:
                    expected = np.arange(sum(sizes))
                    if sizes == [2, 0, 3]:
                        expected = np.insert(expected, 2, 0)
                    np.testing.assert_array_equal(actual[0], expected)
                    self.assertEqual(load.call_args.kwargs["memory_required"], min(max(sizes), 576) * 100)


if __name__ == "__main__":
    unittest.main()
