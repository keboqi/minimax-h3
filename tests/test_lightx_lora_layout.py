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


def core_modules():
    modules = {}
    for prefix, count in (("diffusion_model.blocks", 50), ("diffusion_model.token_refiner.blocks", 2)):
        for i in range(count):
            for suffix, dims in TARGETS.items():
                modules[f"{prefix}.{i}.{suffix}"] = dims
    return modules


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
        elif isinstance(node, ast.FunctionDef) and node.name in {"_logical_weight_shape", "_lightx_key_map"}:
            wanted.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def fake_model(modules):
    weights = {module: FakeTensor(out, inp) for module, (out, inp, _rank) in modules.items()}
    native = {module: module + ".weight" for module in modules}
    comfy = SimpleNamespace(
        lora=SimpleNamespace(model_lora_keys_unet=lambda model, _: dict(native)),
        utils=SimpleNamespace(get_attr=lambda model, key: weights[key[: -len(".weight")]]),
    )
    return SimpleNamespace(model=object()), comfy


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

    def test_unknown_key_is_rejected(self):
        ns = load_key_map()
        model, ns["comfy"] = fake_model(core_modules())
        lora = lora_for(core_modules())
        lora["diffusion_model.final_layer.video_out.alpha"] = FakeTensor()
        with self.assertRaisesRegex(ValueError, "Unexpected LightX2V LoRA key"):
            ns["_lightx_key_map"](model, lora)


if __name__ == "__main__":
    unittest.main()
