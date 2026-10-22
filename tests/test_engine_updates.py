"""Focused contracts for the October 18 Qwen/Drama component adaptations."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.generation.drama3 import transcribe_drama3_audio
from h3_app.generation.flux3 import validate_flux3 as validate_request
from h3_app.generation.qwen import generate_qwen_image21
from h3_app.generation.requests import QwenImage21Request, Flux3Request
from h3_app.workflows.drama3 import build_drama3_asr_graph
from h3_app.workflows.flux3 import build_flux3_graph, flux3_model_keys
from h3_app.workflows.flux3_layout import parse_flux3_layout
from h3_app.workflows.qwen import build_qwen_image21_graph, required_qwen_image21_nodes
from h3_models import MODEL_SPECS
from h3_requirements import COMFY_KITCHEN_VERSION

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".cache/remaining-updates-2026-10-18"


def qwen_arguments(**changes):
    return dict(model_choice="BF16", text_encoder_choice="BF16", prompt="a fox", negative_prompt="",
                reference_images=(), width=1024, height=1024, reference_resolution=0, match_input_size=False,
                seed=1, steps=40, cfg=1, sampler_name="euler", scheduler="simple", cache_device="auto",
                cache_dtype="default", attention_backend="pytorch attention", accelerator="Off",
                output_stamp="1", output_nonce="test", **changes)


class ModelSelectionTests(unittest.TestCase):
    def test_flux_base_defaults_use_true_empty_conditioning_and_cfg_four(self):
        graph = build_flux3_graph(model_choice="Base INT8 ConvRot", prompt="a fox", expand_prompt=False)
        sampler = next(n["inputs"] for n in graph.values() if n["class_type"] == "KSampler")
        self.assertEqual((sampler["steps"], sampler["cfg"]), (50, 4.0))
        negative = graph[sampler["negative"][0]]
        self.assertEqual(negative["class_type"], "CLIPTextEncode")
        self.assertEqual(negative["inputs"]["text"], "")
        self.assertIn("flux3_base_int8", flux3_model_keys("Base INT8 ConvRot", False))
        with self.assertRaisesRegex(ValueError, "CFG 1"):
            build_flux3_graph(model_choice="BF16", prompt="a fox", cfg=4)

    def test_small_asr_downloads_and_loads_only_the_small_model(self):
        graph, preview = build_drama3_asr_graph(audio="ref.wav", model_choice="0.6B (smaller)")
        models = Mock()
        services = SimpleNamespace(models=models, execution=SimpleNamespace(
            object_info=lambda: {n["class_type"]: {} for n in graph.values()},
            submit_prompt=Mock(return_value="job"), poll_comfy_progress=lambda *a: iter(()),
            wait_for_history=lambda *a: {"outputs": {preview: {"text": ["Hello there."]}}}))
        self.assertEqual(transcribe_drama3_audio("ref.wav", services, model_choice="0.6B (smaller)"), "Hello there.")
        models.ensure_model_keys.assert_called_once_with(("drama3_asr_small",))
        loaded = next(n["inputs"]["model_name"] for n in graph.values() if n["class_type"] == "Qwen3ASRLoader")
        self.assertEqual(loaded, MODEL_SPECS["drama3_asr_small"].local_name)

    def test_flux_small_upsampler_is_optional_and_matches_graph(self):
        choice = "W4A8 (lower VRAM)"
        self.assertEqual(flux3_model_keys("BF16", True, choice)[-1], "flux3_upsampler_w4a8")
        self.assertNotIn("flux3_upsampler_w4a8", flux3_model_keys("BF16", False, choice))
        graph = build_flux3_graph(model_choice="BF16", prompt="a fox", upsampler_choice=choice)
        captions = [n["inputs"]["clip_name"] for n in graph.values() if n["class_type"] == "CLIPLoader"]
        self.assertIn(MODEL_SPECS["flux3_upsampler_w4a8"].local_name, captions)
        self.assertNotIn(MODEL_SPECS["flux3_upsampler"].local_name, captions)


class QwenNativeEnhancerTests(unittest.TestCase):
    def test_t2i_and_edit_select_separate_models_and_keep_alpha_conditioning(self):
        for choice, suffix in (("INT8 ConvRot", "int8"), ("W4A8 (lower VRAM)", "w4a8")):
            for references in ((), ("portrait.png", "landscape.png")):
                args = qwen_arguments(prompt_enhancer=choice)
                args["reference_images"] = references
                graph = build_qwen_image21_graph(**args)
                editor_id, editor = next((k, n) for k, n in graph.items() if n["class_type"] == "H3Qwen21PromptEnhance")
                encoder = next(n["inputs"] for n in graph.values() if n["class_type"] == "TextEncodeQwenImage21")
                self.assertEqual(encoder["prompt"], [editor_id, 0])
                writer = graph[editor["inputs"]["clip"][0]]["inputs"]
                key = f"qwen_image21_pe_{'i2i' if references else 't2i'}_{suffix}"
                self.assertEqual(writer["clip_name"], MODEL_SPECS[key].local_name)
                self.assertIn("H3Qwen21PromptEnhance", required_qwen_image21_nodes(editing=bool(references), prompt_enhancer=choice))
                for i, name in enumerate(references):
                    loaded_id = editor["inputs"][f"image{i}"][0]
                    self.assertEqual(graph[loaded_id]["inputs"]["image"], name)
                    rgba = graph[encoder[f"images.image_{i + 1}"][0]]
                    self.assertEqual(rgba["class_type"], "JoinImageWithAlpha")
                    self.assertEqual(rgba["inputs"]["image"], [loaded_id, 0])

    def test_generation_downloads_only_its_selected_native_enhancer(self):
        args = qwen_arguments(prompt_enhancer="W4A8 (lower VRAM)")
        graph = build_qwen_image21_graph(**args)
        args.pop("output_stamp")
        args.pop("output_nonce")
        request = QwenImage21Request(mode="Text to image", **args)
        services = SimpleNamespace(models=Mock(), execution=SimpleNamespace(
            object_info=lambda: {n["class_type"]: {} for n in graph.values()},
            submit_prompt=Mock(return_value="job"), poll_comfy_progress=lambda *a: iter(()),
            wait_for_history=lambda *a: {}),
            media=SimpleNamespace(resolve_image_outputs=lambda *a: [Path("image.png")]))
        services.models.missing_qwen_image21_model_names.return_value = []
        with patch("h3_app.generation.qwen.write_snapshot"):
            updates = list(generate_qwen_image21(request, services, SimpleNamespace(input_dir=Path("."))))
        self.assertNotIn("Error:", updates[-1].status)
        services.models.ensure_model_keys.assert_called_once_with(("qwen_image21_pe_t2i_w4a8",))

    def test_thinking_completion_and_unequal_size_images_use_native_qwen_settings(self):
        source = ROOT / "custom_nodes/H3Acceleration/__init__.py"
        node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "H3Qwen21PromptEnhance")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        enhancer = namespace["H3Qwen21PromptEnhance"]()
        clip = Mock()
        clip.decode.return_value = "reasoning</think> Final instruction."
        images = [np.zeros((1, 48, 96, 4)), np.ones((1, 120, 60, 3))]
        self.assertEqual(enhancer.generate(clip, "edit", "rules", 4096, 0, image0=images[0], image1=images[1]),
                         ("Final instruction.",))
        tokens = clip.tokenize.call_args.kwargs
        self.assertTrue(tokens["thinking"])
        self.assertEqual([v.shape for v in tokens["images"]], [(1, 48, 96, 3), (1, 120, 60, 3)])
        self.assertEqual(clip.generate.call_args.kwargs["presence_penalty"], 0)
        enhancer.generate(clip, "create", "rules", 4096, 0)
        self.assertEqual(clip.generate.call_args.kwargs["presence_penalty"], 1.5)
        for invalid in ("thinking without close", "reasoning</think> "):
            clip.decode.return_value = invalid
            with self.assertRaisesRegex(ValueError, "did not finish"):
                enhancer.generate(clip, "edit", "rules", 4096, 0)


class FluxLayoutTests(unittest.TestCase):
    def test_layout_reaches_native_node_after_generation_or_edit_caption_expansion(self):
        layout = '[{"bbox": [100, 200, 600, 700], "desc": "red bicycle"}]'
        for refs in ((), ("ref.png",)):
            graph = build_flux3_graph(model_choice="BF16", prompt="a bicycle", reference_images=refs, layout=layout)
            layout_id, node = next((i, n) for i, n in graph.items() if n["class_type"] == "Flux3LayoutPrompt")
            self.assertEqual(graph[node["inputs"]["prompt"][0]]["class_type"],
                             "H3Flux3EditPrompt" if refs else "TextGenerate")
            elements = graph[node["inputs"]["element"][0]]
            self.assertEqual(elements["class_type"], "H3Flux3LayoutElements")
            self.assertEqual(json.loads(elements["inputs"]["layout"]), parse_flux3_layout(layout))
            encoded = next(n["inputs"] for n in graph.values() if n["class_type"] == "CLIPTextEncode")
            self.assertEqual(encoded["text"], [layout_id, 0])

    def test_invalid_layout_is_rejected_before_generation(self):
        invalid = ("{}", "not json", '[{"bbox":[0,0,0,10],"desc":"x"}]',
                   '[{"bbox":[0,0,10,1001],"desc":"x"}]', '[{"bbox":[0,0,10,10]}]',
                   '[{"bbox":[0,0,NaN,10],"desc":"x"}]', '[{"bbox":[false,0,10,10],"desc":"x"}]',
                   '[{"bbox":[10,20,5,30],"desc":"x"}]', '[{"bbox":[0,0,10,10],"desc":"x","id":"bad"}]')
        for layout in invalid:
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                validate_request(Flux3Request("a fox", layout=layout))

    def test_native_layout_serialization_preserves_text_and_coordinate_order(self):
        # Characterize the exact native table format consumed by FLUX 3.
        source = ROOT / ".cache/flux-official-update-2026-10-18/core-flux-nodes.py"
        if not source.is_file():
            self.skipTest("native FLUX node source not downloaded")
        owner = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                     if isinstance(n, ast.ClassDef) and n.name == "Flux3LayoutPrompt")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        method.decorator_list = []
        namespace = {"json": json, "io": SimpleNamespace(NodeOutput=lambda *args: args)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
        layout = '[{"bbox":[100,200,600,700],"desc":"red sign","type":"text","text":"你好"}]'
        prompt = namespace["execute"](None, "A street scene", parse_flux3_layout(layout))[0]
        self.assertTrue(prompt.startswith("A street scene "))
        row = json.loads(prompt[len("A street scene "):])[0]
        self.assertEqual(row, {"id": "obj_1", "bbox": [100, 200, 600, 700], "desc": 'red sign "你好"'})


@unittest.skipUnless((UPSTREAM / "core-textgen.py").is_file(), "audited upstream sources not downloaded")
class UpstreamCompatibilityTests(unittest.TestCase):
    def test_core_dependency_and_new_drama_decode_apis_match(self):
        requirements = (UPSTREAM / "core-requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn(f"comfy-kitchen=={COMFY_KITCHEN_VERSION}", requirements)
        sources = {name: ast.parse((UPSTREAM / name).read_text(encoding="utf-8"))
                   for name in ("core-llama.py", "core-qwen35.py", "drama-model.py")}
        classes = {n.name: n for tree in sources.values() for n in tree.body if isinstance(n, ast.ClassDef)}
        def methods(name):
            node = classes[name]
            result = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    result |= methods(base.id)
            return result
        self.assertTrue({"write", "attend", "zeros", "shared"} <= methods("FixedKVCache"))
        self.assertIn("deferred_tracker", methods("LinearKV"))
        assignments = [n for n in sources["drama-model.py"].body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "DEFERRED_DECODE" for t in n.targets)]
        namespace = {"LinearKV": SimpleNamespace(deferred_tracker=object())}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "<drama decode detection>", "exec"), namespace)
        self.assertTrue(namespace["DEFERRED_DECODE"])

    def test_native_textgenerate_does_not_emit_unclosed_thinking_as_a_caption(self):
        tree = ast.parse((UPSTREAM / "core-textgen.py").read_text(encoding="utf-8"))
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TextGenerate")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        method.decorator_list = []
        namespace = {"io": SimpleNamespace(NodeOutput=lambda *args: args)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "<native textgen>", "exec"), namespace)
        clip = Mock()
        clip.tokenize.return_value = {"qwen35": [[(1, 1.0), (2, 1.0)]]}
        clip.decode.side_effect = ["unfinished reasoning", "assistant\n<think>\n"]
        result = namespace["execute"](None, clip, "a fox", 4096, {"sampling_mode": "on"}, thinking=True)
        self.assertEqual(result, ("", "unfinished reasoning"))


if __name__ == "__main__":
    unittest.main()
