"""LightX2V bypass key validation for the legacy and v1.2 (AdaLN) layouts."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape
        self.ndim = len(shape)

    def numel(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


HIDDEN, T_DIM = 5376, 2688
# (out, in, rank) per target, taken from the official v1.2 ComfyUI checkpoint header.
TARGETS = {
    "attn.qkv_proj": (21504, HIDDEN, 384),
    "attn.out_proj": (HIDDEN, 7168, 128),
    "mlp.fc1": (28672, HIDDEN, 128),
    "mlp.fc2": (HIDDEN, 14336, 128),
}
ADALN = {
    **{f"diffusion_model.blocks.{i}.adaln_proj.linear": (96768, T_DIM, 16) for i in range(50)},
    "diffusion_model.final_layer.adaln_proj.linear": (10752, T_DIM, 16),
}


def core_modules(core_rank=128):
    modules = {}
    for prefix, count in (("diffusion_model.blocks", 50), ("diffusion_model.token_refiner.blocks", 2)):
        for i in range(count):
            for suffix, (out, inp, _rank) in TARGETS.items():
                rank = core_rank * 3 if suffix == "attn.qkv_proj" else core_rank
                modules[f"{prefix}.{i}.{suffix}"] = (out, inp, rank)
    return modules


def adaln_modules(rank=16):
    return {
        **{f"diffusion_model.blocks.{i}.adaln_proj.linear": (96768, T_DIM, rank) for i in range(50)},
        "diffusion_model.final_layer.adaln_proj.linear": (10752, T_DIM, rank),
    }


def lora_for(modules):
    lora = {}
    for module, (out, inp, rank) in modules.items():
        lora[module + ".lora_A.weight"] = FakeTensor(rank, inp)
        lora[module + ".lora_B.weight"] = FakeTensor(out, rank)
        lora[module + ".alpha"] = FakeTensor()
    return lora


def load_key_map():
    path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("LIGHTX_") for t in node.targets
        ):
            wanted.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_logical_weight_shape", "_lightx_key_map", "_lightx_curve_adaln",
        }:
            wanted.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fake_model(modules, *, curves=False):
    weights = {module: FakeTensor(out, inp) for module, (out, inp, _rank) in modules.items()}
    native = {module: module + ".weight" for module in modules}
    comfy = SimpleNamespace(
        lora=SimpleNamespace(model_lora_keys_unet=lambda model, _: dict(native)),
        utils=SimpleNamespace(get_attr=lambda model, key: weights[key[: -len(".weight")]]),
    )
    return SimpleNamespace(model=SimpleNamespace(
        diffusion_model=SimpleNamespace(use_adaln_curves=curves))), comfy


class LightXLoRALayoutTests(unittest.TestCase):
    def run_map(self, lora_modules, model_modules=None):
        ns = load_key_map()
        model, ns["comfy"] = fake_model(model_modules or {**core_modules(), **ADALN})
        return ns["_lightx_key_map"](model, lora_for(lora_modules))

    def test_legacy_layout_still_loads(self):
        self.assertEqual(len(self.run_map(core_modules())), 208)

    def test_v12_adaln_layout_loads(self):
        key_map = self.run_map({**core_modules(), **ADALN})
        self.assertEqual(len(key_map), 259)
        self.assertIn("diffusion_model.final_layer.adaln_proj.linear", key_map)

    def test_partial_adaln_layout_is_rejected(self):
        partial = {**core_modules(), "diffusion_model.blocks.0.adaln_proj.linear": ADALN["diffusion_model.blocks.0.adaln_proj.linear"]}
        with self.assertRaisesRegex(ValueError, "259-module"):
            self.run_map(partial)

    def test_v12_adaln_layout_loads_on_pruned_curve_models(self):
        modules = {**core_modules(), **ADALN}
        pruned = {**core_modules(), **{name: (out, 8, rank) for name, (out, _, rank) in ADALN.items()}}
        ns = load_key_map()
        model, ns["comfy"] = fake_model(pruned, curves=True)
        self.assertEqual(len(ns["_lightx_key_map"](model, lora_for(modules))), 259)

    def test_mismatched_dense_adaln_is_still_rejected(self):
        modules = {**core_modules(), **ADALN}
        for width in (8, 64):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "base=.*down=.*up="):
                self.run_map(modules, {**core_modules(), **{
                    name: (out, width, rank) for name, (out, _, rank) in ADALN.items()
                }})

    def test_curve_support_does_not_accept_wrong_output_shapes(self):
        modules = {**core_modules(), **ADALN}
        pruned = {**core_modules(), **{name: (out + 1, 8, rank) for name, (out, _, rank) in ADALN.items()}}
        ns = load_key_map()
        model, ns["comfy"] = fake_model(pruned, curves=True)
        with self.assertRaisesRegex(ValueError, "tensor shapes do not match"):
            ns["_lightx_key_map"](model, lora_for(modules))

    def test_unknown_key_is_rejected(self):
        ns = load_key_map()
        model, ns["comfy"] = fake_model(core_modules())
        lora = lora_for(core_modules())
        lora["diffusion_model.final_layer.video_out.alpha"] = FakeTensor()
        with self.assertRaisesRegex(ValueError, "Unexpected LightX2V LoRA key"):
            ns["_lightx_key_map"](model, lora)

    def test_turbo_fusion_full_layout_loads_on_dense_and_pruned_models(self):
        modules = {**core_modules(320), **adaln_modules(32)}
        key_map = self.run_map(modules)
        self.assertEqual(len(key_map), 259)
        self.assertIn("diffusion_model.final_layer.adaln_proj.linear", key_map)
        pruned = {
            **core_modules(320),
            **{name: (out, 8, rank) for name, (out, _, rank) in adaln_modules(32).items()},
        }
        ns = load_key_map()
        model, ns["comfy"] = fake_model(pruned, curves=True)
        self.assertEqual(len(ns["_lightx_key_map"](model, lora_for(modules))), 259)

    def test_unsupported_or_mixed_rank_profiles_are_rejected(self):
        for core_rank, adaln_rank in ((64, 32), (32, 32)):
            with self.subTest(core_rank=core_rank, adaln_rank=adaln_rank):
                with self.assertRaisesRegex(ValueError, "Unexpected LightX2V rank profile"):
                    self.run_map({**core_modules(core_rank), **adaln_modules(adaln_rank)})
        mixed = {**core_modules(320), **adaln_modules(32)}
        mixed["diffusion_model.blocks.1.mlp.fc1"] = (28672, HIDDEN, 128)
        with self.assertRaisesRegex(ValueError, "Unexpected LightX2V tensor layout"):
            self.run_map(mixed)
        bad_qkv = {**core_modules(320), **adaln_modules(32)}
        bad_qkv["diffusion_model.blocks.0.attn.qkv_proj"] = (21504, HIDDEN, 320)
        with self.assertRaisesRegex(ValueError, "Unexpected LightX2V tensor layout"):
            self.run_map(bad_qkv)


if __name__ == "__main__":
    unittest.main()
