"""MOSS-Image SFT presets, Turbo LoRA selection and native ComfyUI graphs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.catalog import DEFAULT_MOSS_IMAGE_PRESET, MOSS_IMAGE_PRESETS
from h3_app.errors import H3Error
from h3_app.generation.optional_engines import generate_optional_engine, validate_request
from h3_app.generation.requests import MossImageRequest
from h3_app.workflows.moss_image import build_moss_image_graph, moss_image_model_keys
from h3_models import (
    LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS, MOSS_IMAGE_DMD_8STEP, MOSS_IMAGE_MODEL_KEYS,
    MOSS_IMAGE_TURBO_MODES, PRELOAD_MODEL_KEYS,
)
from h3_ui.optional_engine_views import moss_image_preset_values, moss_image_turbo_defaults

ENCODER = "INT8 ConvRot (recommended)"


def graph_for(preset, **changes):
    turbo, steps, cfg = MOSS_IMAGE_PRESETS[preset]
    arguments = dict(model_choice="BF16", text_encoder_choice=ENCODER, prompt="a fox",
                     turbo_variant=turbo, steps=steps, cfg=cfg, seed=7)
    return build_moss_image_graph(**{**arguments, **changes})


def nodes(graph, class_type):
    return [n["inputs"] for n in graph.values() if n["class_type"] == class_type]


class MossImagePresetTests(unittest.TestCase):
    def test_presets_match_the_requested_recipes(self):
        self.assertEqual(MOSS_IMAGE_PRESETS, {
            "Fast": (MOSS_IMAGE_DMD_8STEP, 8, 1.0),
            "Balanced": ("Off", 25, 3.0),
            "Quality": ("Off", 50, 3.0),
        })
        self.assertEqual(DEFAULT_MOSS_IMAGE_PRESET, "Fast")
        self.assertEqual(MOSS_IMAGE_TURBO_MODES[MOSS_IMAGE_DMD_8STEP], ("moss_image_dmd_8step_lora", 8))

    def test_fast_applies_the_dmd_lora_at_full_strength_without_guidance(self):
        graph = graph_for("Fast")
        (lora,) = nodes(graph, "LoraLoaderModelOnly")
        self.assertEqual(lora["lora_name"], "moss_image_sft_dmd_8step_lora.safetensors")
        self.assertEqual(lora["strength_model"], 1.0)
        self.assertFalse(nodes(graph, "SkipLayerGuidanceDiTSimple"))
        (sampler,) = nodes(graph, "KSampler")
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"]), (8, 1.0, "euler"))
        self.assertEqual(graph[sampler["model"][0]]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(graph[sampler["negative"][0]]["class_type"], "ConditioningZeroOut")

    def test_base_presets_skip_layer_22_in_the_empty_negative_pass_only(self):
        for preset, steps in (("Balanced", 25), ("Quality", 50)):
            with self.subTest(preset=preset):
                graph = graph_for(preset)
                self.assertFalse(nodes(graph, "LoraLoaderModelOnly"))
                (slg,) = nodes(graph, "SkipLayerGuidanceDiTSimple")
                self.assertEqual((slg["single_layers"], slg["double_layers"]), ("22", ""))
                self.assertEqual((slg["start_percent"], slg["end_percent"]), (0.0, 1.0))
                (sampler,) = nodes(graph, "KSampler")
                self.assertEqual((sampler["steps"], sampler["cfg"]), (steps, 3.0))
                negative = graph[sampler["negative"][0]]
                self.assertEqual((negative["class_type"], negative["inputs"]["text"]), ("CLIPTextEncode", ""))

    def test_native_loaders_use_the_selected_files(self):
        graph = graph_for("Quality", model_choice="INT8 ConvRot (lower VRAM)", text_encoder_choice="W4A8 (lowest VRAM)")
        self.assertEqual(nodes(graph, "UNETLoader")[0]["unet_name"], "moss_image_sft_int8_convrot.safetensors")
        clip = nodes(graph, "CLIPLoader")[0]
        self.assertEqual((clip["clip_name"], clip["type"]), ("qwen3.5_9b_w4a8_gptq.safetensors", "moss_image"))
        self.assertEqual(nodes(graph, "VAELoader")[0]["vae_name"], "flux2-vae_small.safetensors")
        self.assertTrue(nodes(graph, "EmptyFlux2LatentImage"))

    def test_turbo_requires_its_step_count_and_cfg_one(self):
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            graph_for("Fast", steps=25)
        with self.assertRaisesRegex(ValueError, "CFG 1"):
            graph_for("Fast", cfg=3.0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            graph_for("Fast", turbo_variant="Imaginary Turbo")

    def test_model_keys_download_only_the_selection(self):
        self.assertEqual(moss_image_model_keys("BF16", ENCODER, "Off"),
                         ("moss_image_sft_bf16", "moss_image_text_int8", "moss_image_vae"))
        self.assertEqual(moss_image_model_keys("BF16", "BF16", MOSS_IMAGE_DMD_8STEP)[-1], "moss_image_dmd_8step_lora")
        for key in MOSS_IMAGE_MODEL_KEYS:
            self.assertIn(key, MODEL_SPECS)
            self.assertIn(key, LAZY_OPTIONAL_MODEL_KEYS)
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            self.assertTrue(MODEL_SPECS[key].expected_sha256)

    def test_layout_boxes_become_a_native_layout_prompt_on_the_output_canvas(self):
        graph = graph_for("Balanced", width=1536, height=1024,
                          layout='[{"bbox": [0, 0, 500, 500], "desc": "a red sign", "type": "text", "text": "OPEN"}]')
        (layout,) = nodes(graph, "MOSSImageLayoutPrompt")
        self.assertEqual((layout["width"], layout["height"]), (1536, 1024))
        encode = next(n for n in nodes(graph, "CLIPTextEncode") if n["text"] != "")
        self.assertEqual(graph[encode["text"][0]]["class_type"], "MOSSImageLayoutPrompt")

    def test_image_is_saved_where_the_shared_resolver_and_gallery_look(self):
        from h3_app.gallery_store import generated_image_family

        graph = graph_for("Fast", output_stamp="123", output_nonce="abcd")
        self.assertEqual(nodes(graph, "SaveImage")[0]["filename_prefix"], "h3/image_staging/moss_image_123_abcd")
        with tempfile.TemporaryDirectory() as root:
            runtime = SimpleNamespace(output_dir=Path(root), outputs_dir=Path(root) / "outputs")
            image = Path(root) / "h3/image_staging/moss_image_123_abcd_00001_.png"
            self.assertEqual(generated_image_family(image, runtime=runtime), "MOSS-Image")


class MossImageControlTests(unittest.TestCase):
    def test_preset_sets_turbo_steps_and_cfg(self):
        self.assertEqual(moss_image_preset_values("Balanced"), ("Off", 25, 3.0))
        self.assertEqual(moss_image_preset_values("Fast"), (MOSS_IMAGE_DMD_8STEP, 8, 1.0))

    def test_turbo_change_fixes_steps_and_off_restores_the_base_preset(self):
        self.assertEqual(moss_image_turbo_defaults(MOSS_IMAGE_DMD_8STEP, "Quality"), (8, 1.0))
        self.assertEqual(moss_image_turbo_defaults("Off", "Quality"), (50, 3.0))
        self.assertEqual(moss_image_turbo_defaults("Off", "Fast"), (25, 3.0))


class MossImageRequestTests(unittest.TestCase):
    def test_validation_rejects_bad_requests_before_downloads(self):
        request = MossImageRequest("a fox")
        validate_request(request)
        for changes, message in (
            ({"prompt": " "}, "prompt"),
            ({"width": 1000}, "Width"),
            ({"steps": 25}, "exactly 8"),
            ({"cfg": 3.0}, "CFG 1"),
            ({"turbo_variant": "Off", "steps": 25, "cfg": 11.0}, "CFG"),
            ({"layout": "not json"}, "JSON"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(H3Error, message):
                validate_request(replace(request, **changes))

    def test_generation_downloads_the_selection_and_records_the_preset(self):
        request = MossImageRequest("a fox", preset="Quality", turbo_variant="Off", steps=50, cfg=3.0, seed=5)
        models, submitted = Mock(), {}

        def submit(graph, _client_id):
            submitted.update(graph)
            return "job"

        info = {n["class_type"]: {} for n in graph_for("Quality").values()}
        info["CLIPLoader"] = {"input": {"required": {"type": [["flux3", "moss_image"]]}}}
        execution = SimpleNamespace(
            object_info=lambda: info,
            submit_prompt=submit, poll_comfy_progress=lambda *args: iter(()), wait_for_history=lambda *args: {})
        services = SimpleNamespace(models=models, execution=execution, media=SimpleNamespace(
            resolve_image_outputs=lambda *args: [Path("moss_image_1_a.png")]))
        with patch("h3_app.generation.optional_engines.write_snapshot") as snapshot:
            results = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertNotIn("Error:", results[-1].status)
        self.assertIn("MOSS-Image complete", results[-1].status)
        models.ensure_model_keys.assert_called_once_with(("moss_image_sft_bf16", "moss_image_text_int8", "moss_image_vae"))
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["preset"], settings["turbo_variant"], settings["steps"]), ("Quality", "Off", 50))
        self.assertEqual(snapshot.call_args.args[1]["family"], "MOSS-Image")


CUP_LAYOUT = {
    "width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
    "summary": {"caption": "A warm studio campaign for a ceramic cup."},
    "items": [
        {"label": "object", "caption": "An ivory ceramic cup.", "bbox": [176, 224, 800, 864]},
        {"label": "text", "caption": "Dark-green title.", "text": "一杯温暖\n一刻从容", "bbox": [928, 512, 1376, 656]},
    ],
}


class MossImagePromptCompilerTests(unittest.TestCase):
    def test_layout_compiles_to_centered_yxyx_like_the_official_skill(self):
        import json
        from h3_app.workflows.moss_image_prompt import compile_writer_draft, is_layout_prompt

        prompt, layout = compile_writer_draft(json.dumps(CUP_LAYOUT, ensure_ascii=False), 1536, 1024)
        self.assertTrue(layout and is_layout_prompt(prompt))
        compiled = json.loads(prompt)
        self.assertEqual(compiled["items"][0]["bbox"], [-288, -592, 352, 32])
        self.assertEqual((compiled["bbox_coord_space"], compiled["bbox_format"]), ("abs_centralized", "yxyx"))
        self.assertNotIn("width", compiled)
        self.assertEqual(compiled["items"][1]["text"], "一杯温暖\n一刻从容")
        self.assertNotIn("\n", prompt)

    def test_caption_replies_may_arrive_fenced(self):
        from h3_app.workflows.moss_image_prompt import compile_writer_draft, is_layout_prompt

        prompt, layout = compile_writer_draft('```json\n{"caption": "A fox in snow."}\n```', 1024, 1024)
        self.assertEqual((prompt, layout), ("A fox in snow.", False))
        self.assertFalse(is_layout_prompt(prompt))

    def test_drafts_must_match_the_canvas_mode_and_schema(self):
        import json
        from h3_app.workflows.moss_image_prompt import compile_writer_draft

        draft = json.dumps(CUP_LAYOUT, ensure_ascii=False)
        for reply, width, mode, message in (
            (draft, 1024, "Auto", "canvas must be 1024x1024"),
            (draft, 1536, "Caption", "caption was requested"),
            ('{"caption": "x"}', 1536, "Layout", "Layout was requested"),
            ("A fox.", 1536, "Auto", "JSON"),
            ('{"caption": "x", "caption": "y"}', 1536, "Auto", "Duplicate"),
            (json.dumps({**CUP_LAYOUT, "items": [{**CUP_LAYOUT["items"][0], "id": 1}]}), 1536, "Auto", "unknown"),
            (json.dumps({**CUP_LAYOUT, "items": [{**CUP_LAYOUT["items"][0], "bbox": [0, 0, 1600, 10]}]}),
             1536, "Auto", "exceeds"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compile_writer_draft(reply, width, 1024, mode)

    def test_rewrite_sends_canvas_and_deliverable_then_compiles_the_draft(self):
        import json
        from h3_app import prompt_service

        runtime = SimpleNamespace(prompt_systems={"MOSS-Image": Path(__file__).parents[1] / "prompt_moss_image.txt"})
        reply = (json.dumps(CUP_LAYOUT, ensure_ascii=False), "Enhanced MOSS-Image prompt with m using 0 image(s).")
        with patch.object(prompt_service, "_enhance_prompt_from_media", return_value=reply) as writer:
            prompt, status = prompt_service.enhance_moss_image_prompt(
                "a cup poster", "m", "", 1536, 1024, "Layout", runtime=runtime)
        request = writer.call_args.kwargs["request_text"]
        self.assertIn("TARGET CANVAS: 1536x1024", request)
        self.assertIn("DELIVERABLE: Layout", request)
        self.assertEqual(writer.call_args.kwargs["system_path"].name, "prompt_moss_image.txt")
        self.assertEqual(json.loads(prompt)["items"][0]["bbox"], [-288, -592, 352, 32])
        self.assertEqual(status, "Enhanced MOSS-Image prompt with m as a Layout for 1536x1024.")
        with patch.object(prompt_service, "_enhance_prompt_from_media", return_value=("not json", "Enhanced x")):
            prompt, status = prompt_service.enhance_moss_image_prompt("a cup", "m", "", 1024, 1024, runtime=runtime)
        self.assertEqual(prompt, "a cup")
        self.assertTrue(status.startswith("Prompt enhancement failed:"))

    def test_a_layout_prompt_cannot_also_take_manual_boxes(self):
        import json
        from h3_app.workflows.moss_image_prompt import compile_writer_draft

        prompt, _ = compile_writer_draft(json.dumps(CUP_LAYOUT), 1536, 1024)
        request = MossImageRequest(prompt, width=1536, height=1024)
        validate_request(request)
        with self.assertRaisesRegex(H3Error, "Clear the layout boxes"):
            validate_request(replace(request, layout='[{"bbox": [0, 0, 500, 500], "desc": "a cat"}]'))


if __name__ == "__main__":
    unittest.main()
