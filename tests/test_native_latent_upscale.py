"""Native upscaler routing and precision-preserving Core model loading."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app.catalog import H3_LATENT_UPSCALER_NODE, H3_LATENT_UPSCALER_LOADER_NODE
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.workflows import h3
from h3_app.workflows.latent_upscale import add_h3_latent_upscale
from tests.workflow_cases import H3_DEFAULTS


class NativeUpscalerTests(unittest.TestCase):
    def test_refinement_uses_aligned_final_canvas_and_preserves_audio(self):
        for build in (h3.build_fl2va_graph, h3.build_ref2va_graph):
            for precision in ("fp32", "bf16", "fp16"):
                args = {**H3_DEFAULTS, "width": 864, "height": 480,
                        "latent_upscale_model_name": "upscale.pth",
                        "latent_upscale_precision": precision}
                if build is h3.build_ref2va_graph:
                    args.pop("first_image")
                    args.pop("last_image")
                    args.pop("semantic_bridge", None)
                    args.pop("semantic_bridge_alpha", None)
                    args.pop("voice_reference_audios", None)
                    args.update(reference_images=[], reference_videos=[], reference_audios=[],
                                ref_image_size="Original")
                graph = build(**args)
                native = next(n["inputs"] for n in graph.values() if n["class_type"] == H3_LATENT_UPSCALER_NODE)
                self.assertEqual((native["width"], native["height"]), (896, 512))
                self.assertEqual(graph[native["model"][0]]["inputs"], {
                    "model_name": "upscale.pth", "precision": precision})
                source = graph[native["samples"][0]]
                self.assertEqual(source["class_type"], "H3SeparateAVLatent")
                combined = next(n["inputs"] for n in graph.values() if n["class_type"] == "H3CombineAVLatent")
                self.assertEqual(combined["audio_latent"], [native["samples"][0], 1])
                self.assertNotIn("MinimaxH3LatentUpscaler3D", {n["class_type"] for n in graph.values()})
        required = h3.required_nodes_for("Text to video", False, "Off", latent_upscale=True)
        self.assertTrue({H3_LATENT_UPSCALER_NODE, H3_LATENT_UPSCALER_LOADER_NODE} <= required)

    def test_invalid_canvas_or_precision_fails_before_submission(self):
        for width, precision in ((865, "bf16"), (None, "bf16"), (1024, "int8")):
            with self.assertRaises(H3Error):
                add_h3_latent_upscale(Graph(), ["samples", 0], model_name="upscale",
                                      precision=precision, width=width, height=1024)

    def test_loader_keeps_explicit_precision_with_static_and_dynamic_loading(self):
        source = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == H3_LATENT_UPSCALER_LOADER_NODE)
        modules = {name: ModuleType(name) for name in (
            "comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.latent_upscaler",
            "comfy.model_patcher", "comfy.ops", "comfy.storage")}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        sd = {
            "conv_in.weight": SimpleNamespace(shape=(512, 24, 3, 3, 3)),
            "in_blocks.0.emb_layers.1.weight": object(),
            "in_blocks.0.in_layers.0.weight": object(),
            "out_blocks.0.in_layers.0.weight": object(),
            "in_blocks.1.dwconv.weight": SimpleNamespace(shape=(512, 1, 5, 1, 1)),
        }
        native = Mock()
        constructor = Mock(return_value=native)
        modules["comfy.ldm.minimax.latent_upscaler"].MiniMaxH3LatentUpscaler = constructor
        core = modules["comfy"]
        core.utils = SimpleNamespace(load_torch_file=Mock(return_value=sd))
        core.model_management = SimpleNamespace(archive_model_dtypes=Mock(),
            get_torch_device=lambda: "cuda", unet_offload_device=lambda: "cpu")
        core.ops.disable_weight_init = object()
        core.storage.state_dict_fast_disk = Mock(return_value="disk")
        paths = SimpleNamespace(get_full_path_or_raise=Mock(return_value="checkpoint"))
        namespace = {"folder_paths": paths,
                     "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        loader = namespace[H3_LATENT_UPSCALER_LOADER_NODE]()
        with patch.dict(sys.modules, modules):
            for precision in ("bf16", "fp16", "fp32"):
                for dynamic in (True, False):
                    patcher = SimpleNamespace(is_dynamic=lambda: dynamic)
                    core.model_patcher.CoreModelPatcher = Mock(return_value=patcher)
                    self.assertEqual(loader.load_model("upscale", precision), (patcher,))
                    self.assertEqual(constructor.call_args.kwargs["dtype"], precision)
                    native.load_state_dict.assert_called_with(sd, assign=dynamic)
                    core.model_management.archive_model_dtypes.assert_called_with(native)
                    self.assertEqual(core.model_patcher.CoreModelPatcher.call_args.kwargs,
                                     {"load_device": "cuda", "offload_device": "cpu", "fast_disk": "disk"})
            core.utils.load_torch_file.return_value = {}
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                loader.load_model("wrong", "bf16")


if __name__ == "__main__":
    unittest.main()
