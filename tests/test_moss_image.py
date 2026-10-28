"""MOSS-Image presets, model/VAE/Turbo LoRA selection and native ComfyUI graphs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.catalog import DEFAULT_MOSS_IMAGE_PRESET, MOSS_IMAGE_PRESETS
from h3_app.errors import H3Error
from h3_app.generation.flux3 import validate_flux3
from h3_app.generation.moss_image import generate_moss_image as generate_optional_engine, validate_moss_image
from h3_app.generation.requests import MossImageRequest
from h3_app.workflows.moss_image import build_moss_image_graph, moss_image_model_keys
from h3_models import (
    LAZY_OPTIONAL_MODEL_KEYS, MODEL_SPECS, MOSS_IMAGE_4STEP_V2, MOSS_IMAGE_DMD_8STEP, MOSS_IMAGE_MODEL_KEYS,
    MOSS_IMAGE_SFT_BF16, MOSS_IMAGE_SFT_INT8, MOSS_IMAGE_TURBO_MODES, MOSS_IMAGE_VAE_FULL,
    MOSS_IMAGE_VAE_SMALL, PRELOAD_MODEL_KEYS,
)
from h3_ui.engines.moss_image import (
    moss_image_preset_values, moss_image_turbo_defaults, moss_image_turbo_size_steps,
)

ENCODER = "INT8 ConvRot (recommended)"


def validate_request(request):
    (validate_moss_image if isinstance(request, MossImageRequest) else validate_flux3)(request)


def graph_for(preset, **changes):
    turbo, steps, cfg, model, vae = MOSS_IMAGE_PRESETS[preset]
    arguments = dict(model_choice=model, text_encoder_choice=ENCODER, vae_choice=vae, prompt="a fox",
                     turbo_variant=turbo, steps=steps, cfg=cfg, seed=7)
    return build_moss_image_graph(**{**arguments, **changes})


def nodes(graph, class_type):
    return [n["inputs"] for n in graph.values() if n["class_type"] == class_type]


class MossImagePresetTests(unittest.TestCase):
    def test_edit_uses_reference_conditioning_and_separate_lora(self):
        graph = graph_for("Balanced", reference_images=("a.png", "b.png"), steps=30, cfg=4)
        lora, = nodes(graph, "LoraLoaderModelOnly")
        self.assertEqual(lora["lora_name"], "moss_image_edit_lora_rank_64_bf16.safetensors")
        positive, negative = nodes(graph, "TextEncodeMOSSImageEdit")
        self.assertEqual(positive["edit_format"], "Moss-Image-Edit")
        self.assertEqual(negative["prompt"], "")
        self.assertEqual(positive["images.image_1"], negative["images.image_1"])
        self.assertEqual(positive["images.image_2"], negative["images.image_2"])
        self.assertEqual(len(nodes(graph, "H3MossEditReference")), 2)
        self.assertFalse(nodes(graph, "SpectrumMOSSImageAdaptive"))
        self.assertEqual(nodes(graph, "KSampler")[0]["scheduler"], "simple")
        with self.assertRaisesRegex(H3Error, "Turbo Off"):
            validate_moss_image(MossImageRequest("edit", reference_images=("a.png",)))
        with self.assertRaisesRegex(H3Error, "four"):
            validate_moss_image(MossImageRequest("edit", turbo_variant="Off", reference_images=("a",) * 5))

    def test_spectrum_defaults_on_after_guidance_and_can_be_disabled(self):
        graph = graph_for("Quality")
        spectrum, = nodes(graph, "SpectrumMOSSImageAdaptive")
        self.assertEqual(graph[spectrum["model"][0]]["class_type"], "SkipLayerGuidanceDiTSimple")
        sampler, = nodes(graph, "KSampler")
        self.assertEqual(graph[sampler["model"][0]]["class_type"], "SpectrumMOSSImageAdaptive")
        self.assertEqual(sampler["scheduler"], "sgm_uniform")
        original = graph_for("Quality", accelerate=False)
        self.assertFalse(nodes(original, "SpectrumMOSSImageAdaptive"))
        self.assertEqual(nodes(original, "KSampler")[0]["scheduler"], "simple")

    def test_spectrum_bypasses_step_distilled_and_short_runs(self):
        for graph in (graph_for("Fast"), graph_for("Quality", steps=10),
                      graph_for("Fast", turbo_variant=MOSS_IMAGE_4STEP_V2, steps=4)):
            self.assertFalse(nodes(graph, "SpectrumMOSSImageAdaptive"))
            self.assertEqual(nodes(graph, "KSampler")[0]["scheduler"], "simple")

    def test_presets_match_the_requested_recipes(self):
        self.assertEqual(MOSS_IMAGE_PRESETS, {
            "Fast": (MOSS_IMAGE_DMD_8STEP, 8, 1.0, MOSS_IMAGE_SFT_INT8, MOSS_IMAGE_VAE_SMALL),
            "Balanced": ("Off", 25, 3.0, MOSS_IMAGE_SFT_BF16, MOSS_IMAGE_VAE_FULL),
            "Quality": ("Off", 50, 3.0, MOSS_IMAGE_SFT_BF16, MOSS_IMAGE_VAE_FULL),
        })
        self.assertEqual(DEFAULT_MOSS_IMAGE_PRESET, "Fast")
        self.assertEqual(MOSS_IMAGE_TURBO_MODES[MOSS_IMAGE_DMD_8STEP], ("moss_image_dmd_8step_lora", ((None, 8),)))

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

    def test_4step_lora_takes_more_steps_on_larger_canvases(self):
        self.assertEqual(MODEL_SPECS["moss_image_4step_v2_lora"].local_name, "moss_image_sft_4step_v2.0.safetensors")
        for (width, height), steps in (((1024, 1024), 4), ((1248, 832), 4), ((1536, 1024), 6),
                                       ((1440, 1440), 8), ((2048, 2048), 8)):
            with self.subTest(size=(width, height)):
                graph = graph_for("Fast", turbo_variant=MOSS_IMAGE_4STEP_V2, steps=steps, width=width, height=height)
                (lora,) = nodes(graph, "LoraLoaderModelOnly")
                self.assertEqual((lora["lora_name"], lora["strength_model"]),
                                 ("moss_image_sft_4step_v2.0.safetensors", 1.0))
                (sampler,) = nodes(graph, "KSampler")
                self.assertEqual((sampler["steps"], sampler["cfg"], sampler["scheduler"]), (steps, 1.0, "simple"))
        with self.assertRaisesRegex(ValueError, "exactly 6 sampling steps at 1536x1024"):
            graph_for("Fast", turbo_variant=MOSS_IMAGE_4STEP_V2, steps=4, width=1536, height=1024)

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
        self.assertEqual(nodes(graph, "VAELoader")[0]["vae_name"], "flux2-vae.safetensors")
        self.assertTrue(nodes(graph, "EmptyFlux2LatentImage"))

    def test_presets_load_their_model_and_vae(self):
        fast, quality = graph_for("Fast"), graph_for("Quality")
        self.assertEqual(nodes(fast, "UNETLoader")[0]["unet_name"], "moss_image_sft_int8_convrot.safetensors")
        self.assertEqual(nodes(fast, "VAELoader")[0]["vae_name"], "flux2-vae_small.safetensors")
        self.assertEqual(nodes(quality, "UNETLoader")[0]["unet_name"], "moss_image_sft_bf16.safetensors")
        self.assertEqual(nodes(quality, "VAELoader")[0]["vae_name"], "flux2-vae.safetensors")

    def test_base_checkpoints_are_selectable_without_the_sft_turbo_lora(self):
        for choice, filename in (("Base BF16", "moss_image_base_bf16.safetensors"),
                                 ("Base INT8 ConvRot (lower VRAM)", "moss_image_base_int8_convrot.safetensors")):
            with self.subTest(choice=choice):
                graph = graph_for("Balanced", model_choice=choice)
                self.assertEqual(nodes(graph, "UNETLoader")[0]["unet_name"], filename)
                with self.assertRaisesRegex(ValueError, "SFT"):
                    graph_for("Fast", model_choice=choice)

    def test_full_vae_is_the_shared_flux3_file(self):
        self.assertEqual(MODEL_SPECS["flux3_vae"].expected_sha256,
                         "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3")
        with self.assertRaisesRegex(ValueError, "VAE"):
            graph_for("Fast", vae_choice="Imaginary VAE")

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
        self.assertEqual(moss_image_model_keys("Base BF16", ENCODER, "Off", MOSS_IMAGE_VAE_FULL),
                         ("moss_image_base_bf16", "moss_image_text_int8", "flux3_vae"))
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
        self.assertEqual(moss_image_preset_values("Balanced"),
                         ("Off", 25, 3.0, MOSS_IMAGE_SFT_BF16, MOSS_IMAGE_VAE_FULL))
        self.assertEqual(moss_image_preset_values("Fast"),
                         (MOSS_IMAGE_DMD_8STEP, 8, 1.0, MOSS_IMAGE_SFT_INT8, MOSS_IMAGE_VAE_SMALL))

    def test_turbo_change_fixes_steps_and_off_restores_the_base_preset(self):
        self.assertEqual(moss_image_turbo_defaults(MOSS_IMAGE_DMD_8STEP, "Quality"), (8, 1.0))
        self.assertEqual(moss_image_turbo_defaults("Off", "Quality"), (50, 3.0))
        self.assertEqual(moss_image_turbo_defaults("Off", "Fast"), (25, 3.0))
        self.assertEqual(moss_image_turbo_defaults(MOSS_IMAGE_4STEP_V2, "Fast", 1024, 1024), (4, 1.0))
        self.assertEqual(moss_image_turbo_defaults(MOSS_IMAGE_4STEP_V2, "Fast", 2048, 2048), (8, 1.0))

    def test_resizing_keeps_turbo_steps_matched_and_leaves_base_steps_alone(self):
        self.assertEqual(moss_image_turbo_size_steps(MOSS_IMAGE_4STEP_V2, 4, 1536, 1024), 6)
        self.assertEqual(moss_image_turbo_size_steps(MOSS_IMAGE_4STEP_V2, 6, 1024, 1024), 4)
        self.assertEqual(moss_image_turbo_size_steps(MOSS_IMAGE_DMD_8STEP, 8, 2048, 2048), 8)
        self.assertEqual(moss_image_turbo_size_steps("Off", 37, 2048, 2048), 37)


class ResolutionPresetTests(unittest.TestCase):
    def test_every_preset_is_a_valid_request_size_for_its_engine(self):
        from h3_app.generation.requests import Flux3Request
        from h3_ui.engines import flux3, moss_image
        from h3_ui.engines.controls import resolution_preset_values as qwen_resolution_preset_values

        for presets, request, megapixels in (
            (flux3.FLUX3_1K_RESOLUTION_PRESETS, Flux3Request("a fox"), 1),
            (flux3.FLUX3_2K_RESOLUTION_PRESETS, Flux3Request("a fox"), 4),
            (moss_image.MOSS_IMAGE_1M_RESOLUTION_PRESETS, MossImageRequest("a fox"), 1),
            (moss_image.MOSS_IMAGE_2M_RESOLUTION_PRESETS, MossImageRequest("a fox"), 2),
        ):
            for name in presets:
                with self.subTest(preset=name):
                    width, height = qwen_resolution_preset_values(name)
                    validate_request(replace(request, width=width, height=height))
                    self.assertLessEqual(width * height, 4_194_304)
                    self.assertAlmostEqual(width * height / 2**20, megapixels, delta=0.08 * megapixels)
                    ratio = name.split(" · ")[1].split(":")
                    self.assertLess(abs(width / height * int(ratio[1]) / int(ratio[0]) - 1), 0.0075)

    def test_presets_set_the_size_clear_each_other_and_flux_switches_to_manual_size(self):
        import gradio as gr
        from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
        from h3_ui.engines.controls import bind_resolution_presets
        from h3_ui.engines.flux3 import build_flux3_view
        from h3_ui.engines.moss_image import build_moss_image_view

        with gr.Blocks() as demo:
            flux, moss = build_flux3_view(gr.Group()), build_moss_image_view(gr.Group())
            bind_resolution_presets(flux, extra_outputs=(flux.edit_size,), extra_values=(QWEN_EDIT_SIZE_MANUAL,))
            bind_resolution_presets(moss)

        def handler(trigger, outputs):
            ids = [component._id for component in outputs]
            return next(fn.fn for fn in demo.fns.values()
                        if [block._id for block in fn.inputs] == ([trigger._id] if trigger else [])
                        and [block._id for block in fn.outputs] == ids)

        choose = handler(flux.resolution_large, [flux.width, flux.height, flux.resolution_small, flux.edit_size])
        self.assertEqual(choose("2K · 16:9 · 2720×1536"), (2720, 1536, None, QWEN_EDIT_SIZE_MANUAL))
        choose = handler(moss.resolution_small, [moss.width, moss.height, moss.resolution_large])
        self.assertEqual(choose("1M · 3:2 · 1248×832"), (1248, 832, None))
        clear = handler(None, [moss.resolution_small, moss.resolution_large])
        self.assertEqual(clear(), (None, None))


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
            ({"vae_choice": "Imaginary VAE"}, "VAE"),
            ({"model_choice": "Base BF16"}, "SFT"),
            ({"turbo_variant": MOSS_IMAGE_4STEP_V2, "steps": 4, "width": 2048, "height": 2048}, "exactly 8"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(H3Error, message):
                validate_request(replace(request, **changes))

    def test_generation_downloads_the_selection_and_records_the_preset(self):
        request = MossImageRequest("a fox", preset="Quality", turbo_variant="Off", model_choice=MOSS_IMAGE_SFT_BF16,
                                   vae_choice=MOSS_IMAGE_VAE_FULL, steps=50, cfg=3.0, seed=5)
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
        with patch("h3_app.generation.image_jobs.write_snapshot") as snapshot:
            results = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertNotIn("Error:", results[-1].status)
        self.assertIn("MOSS-Image complete", results[-1].status)
        models.ensure_model_keys.assert_called_once_with(("moss_image_sft_bf16", "moss_image_text_int8", "flux3_vae"))
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["preset"], settings["turbo_variant"], settings["steps"]), ("Quality", "Off", 50))
        self.assertEqual(snapshot.call_args.args[1]["family"], "MOSS-Image")
        self.assertEqual((settings["spectrum_applied"], settings["scheduler"]), (True, "sgm_uniform"))
        self.assertTrue(nodes(submitted, "SpectrumMOSSImageAdaptive"))

    def test_missing_spectrum_node_falls_back_to_the_original_graph(self):
        request = MossImageRequest("a fox", preset="Quality", turbo_variant="Off", model_choice=MOSS_IMAGE_SFT_BF16,
                                   vae_choice=MOSS_IMAGE_VAE_FULL, steps=50, cfg=3.0, seed=5)
        submitted = {}

        def submit(graph, _client_id):
            submitted.update(graph)
            return "job"

        info = {n["class_type"]: {} for n in graph_for("Quality", accelerate=False).values()}
        info["CLIPLoader"] = {"input": {"required": {"type": [["flux3", "moss_image"]]}}}
        execution = SimpleNamespace(
            object_info=lambda: info,
            submit_prompt=submit, poll_comfy_progress=lambda *args: iter(()), wait_for_history=lambda *args: {})
        services = SimpleNamespace(models=Mock(), execution=execution, media=SimpleNamespace(
            resolve_image_outputs=lambda *args: [Path("moss_image_1_a.png")]))
        with patch("h3_app.generation.image_jobs.write_snapshot") as snapshot:
            results = list(generate_optional_engine(request, services, SimpleNamespace()))
        self.assertIn("MOSS-Image complete", results[-1].status)
        self.assertTrue(any("Spectrum-MOSS is not installed" in update.status for update in results))
        self.assertFalse(nodes(submitted, "SpectrumMOSSImageAdaptive"))
        self.assertEqual(nodes(submitted, "KSampler")[0]["scheduler"], "simple")
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["accelerate"], settings["spectrum_applied"], settings["scheduler"]),
                         (True, False, "simple"))


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
        from h3_app.prompt_writers import common as writers_common, moss_image as moss_image_writer

        runtime = SimpleNamespace(prompt_systems={"MOSS-Image": Path(__file__).parents[1] / "prompts" / "prompt_moss_image.txt"})
        reply = (json.dumps(CUP_LAYOUT, ensure_ascii=False), "Enhanced MOSS-Image prompt with m using 0 image(s).")
        with patch.object(writers_common, "write_prompt", return_value=reply) as writer:
            prompt, status = moss_image_writer.enhance_moss_image_prompt(
                "a cup poster", "m", "", 1536, 1024, "Layout", runtime=runtime)
        request = writer.call_args.kwargs["request_text"]
        self.assertIn("TARGET CANVAS: 1536x1024", request)
        self.assertIn("DELIVERABLE: Layout", request)
        self.assertEqual(writer.call_args.kwargs["system_path"].name, "prompt_moss_image.txt")
        self.assertEqual(json.loads(prompt)["items"][0]["bbox"], [-288, -592, 352, 32])
        self.assertEqual(status, "Enhanced MOSS-Image prompt with m as a Layout for 1536x1024.")
        with patch.object(writers_common, "write_prompt", return_value=("not json", "Enhanced x")):
            prompt, status = moss_image_writer.enhance_moss_image_prompt("a cup", "m", "", 1024, 1024, runtime=runtime)
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
