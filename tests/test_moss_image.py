"""MOSS-Image SFT presets, Turbo LoRA selection and native ComfyUI graphs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from h3_app.catalog import DEFAULT_MOSS_IMAGE_PRESET, MOSS_IMAGE_PRESETS, QWEN_EDIT_SIZE_MANUAL
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


class ResolutionPresetTests(unittest.TestCase):
    def test_every_preset_is_a_valid_request_size_for_its_engine(self):
        from h3_app.generation.requests import Flux3Request
        from h3_ui import optional_engine_views as views
        from h3_ui.qwen_bindings import qwen_resolution_preset_values

        for presets, request, megapixels in (
            (views.FLUX3_1K_RESOLUTION_PRESETS, Flux3Request("a fox"), 1),
            (views.FLUX3_2K_RESOLUTION_PRESETS, Flux3Request("a fox"), 4),
            (views.MOSS_IMAGE_1M_RESOLUTION_PRESETS, MossImageRequest("a fox"), 1),
            (views.MOSS_IMAGE_2M_RESOLUTION_PRESETS, MossImageRequest("a fox"), 2),
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
        from h3_ui.optional_engine_views import (
            bind_resolution_presets, build_flux3_view, build_moss_image_view,
        )

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


EDIT_LORA = "cyclops_edit.safetensors"


def edit_graph(preset="Balanced", **changes):
    edit = dict(reference_images=("h3_gradio/moss_image_references/a.png", "h3_gradio/moss_image_references/b.png"),
                edit_lora=EDIT_LORA)
    return graph_for(preset, **{**edit, **changes})


class MossImageEditGraphTests(unittest.TestCase):
    def test_edit_lora_and_ostris_patch_sit_between_the_loaders_and_guidance(self):
        graph = edit_graph(edit_lora_strength=0.8, kv_cache=False)
        (sampler,) = nodes(graph, "KSampler")
        slg = graph[sampler["model"][0]]
        patch_node = graph[slg["inputs"]["model"][0]]
        lora = graph[patch_node["inputs"]["model"][0]]
        self.assertEqual(slg["class_type"], "SkipLayerGuidanceDiTSimple")
        self.assertEqual((patch_node["class_type"], patch_node["inputs"]["kv_cache"]),
                         ("MOSSImageOstrisEditModelPatch", False))
        self.assertEqual((lora["inputs"]["lora_name"], lora["inputs"]["strength_model"]), (EDIT_LORA, 0.8))
        self.assertEqual(graph[lora["inputs"]["model"][0]]["class_type"], "UNETLoader")

    def test_turbo_lora_loads_before_the_edit_lora(self):
        graph = edit_graph("Fast")
        names = [n["lora_name"] for n in nodes(graph, "LoraLoaderModelOnly")]
        self.assertEqual(names, ["moss_image_sft_dmd_8step_lora.safetensors", EDIT_LORA])
        (sampler,) = nodes(graph, "KSampler")
        self.assertEqual(graph[sampler["model"][0]]["class_type"], "MOSSImageOstrisEditModelPatch")
        self.assertEqual(graph[sampler["negative"][0]]["class_type"], "ConditioningZeroOut")

    def test_both_prompts_see_every_scaled_reference_and_the_vae(self):
        graph = edit_graph()
        self.assertFalse(nodes(graph, "CLIPTextEncode"))
        encodes = nodes(graph, "TextEncodeMOSSImageOstrisEdit")
        self.assertEqual(sorted(n["prompt"] for n in encodes), ["", "a fox"])
        loads = [n["image"] for n in nodes(graph, "LoadImage")]
        self.assertEqual(loads, ["h3_gradio/moss_image_references/a.png", "h3_gradio/moss_image_references/b.png"])
        for encode in encodes:
            self.assertEqual(set(encode) - {"clip", "prompt", "vae"}, {"image1", "image2"})
            self.assertEqual(graph[encode["vae"][0]]["class_type"], "VAELoader")
            scale = graph[encode["image1"][0]]
            self.assertEqual(scale["class_type"], "FluxKontextImageScale")
            self.assertEqual(graph[scale["inputs"]["image"][0]]["inputs"]["image"], loads[0])

    def test_text_to_image_graph_is_unchanged_without_references(self):
        graph = graph_for("Balanced")
        for class_type in ("MOSSImageOstrisEditModelPatch", "TextEncodeMOSSImageOstrisEdit", "LoadImage"):
            self.assertFalse(nodes(graph, class_type))

    def test_edit_inputs_are_validated(self):
        for changes, message in (
            ({"edit_lora": ""}, "edit LoRA"),
            ({"reference_images": ("a.png",) * 4}, "at most 3"),
            ({"edit_lora": "../evil.safetensors"}, "file name"),
            ({"edit_lora": "lora.ckpt"}, "safetensors"),
            ({"edit_lora_strength": 0}, "strength"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                edit_graph(**changes)
        with self.assertRaisesRegex(ValueError, "reference image"):
            graph_for("Balanced", edit_lora=EDIT_LORA)


class MossImageEditRequestTests(unittest.TestCase):
    def request(self, **changes):
        return replace(MossImageRequest("make it a cyclops", preset="Balanced", turbo_variant="Off", steps=25, cfg=3.0,
                                        seed=5, reference_images=("refs/portrait.png",), edit_lora=EDIT_LORA), **changes)

    def test_validation_names_the_missing_lora_and_bad_size(self):
        validate_request(self.request())
        with self.assertRaisesRegex(H3Error, "edit LoRA"):
            validate_request(self.request(edit_lora=""))
        with self.assertRaisesRegex(H3Error, "edit output size"):
            validate_request(self.request(edit_size="Max"))

    def test_match_size_fits_the_first_reference_at_one_megapixel(self):
        from h3_app.generation.optional_engines import resolve_moss_image_dimensions

        with patch("h3_app.generation.optional_engines.first_reference_dimensions", return_value=(3000, 2000)):
            width, height = resolve_moss_image_dimensions(self.request(), input_dir=Path("."))
        self.assertEqual((width % 16, height % 16), (0, 0))
        self.assertLessEqual(width * height, 1_048_576)
        self.assertGreater(width * height, 0.85 * 1_048_576)
        self.assertLess(abs(width / height / 1.5 - 1), 0.01)
        self.assertEqual(resolve_moss_image_dimensions(self.request(edit_size=QWEN_EDIT_SIZE_MANUAL, width=768, height=512)),
                         (768, 512))

    def test_generation_checks_comfy_sees_the_lora_and_records_the_references(self):
        request = self.request()
        info = {n["class_type"]: {} for n in edit_graph().values()}
        info["CLIPLoader"] = {"input": {"required": {"type": [["moss_image"]]}}}
        info["LoraLoaderModelOnly"] = {"input": {"required": {"lora_name": [["other.safetensors"]]}}}
        submitted = {}
        execution = SimpleNamespace(
            object_info=lambda: info, submit_prompt=lambda graph, _id: submitted.update(graph) or "job",
            poll_comfy_progress=lambda *args: iter(()), wait_for_history=lambda *args: {})
        services = SimpleNamespace(models=Mock(), execution=execution, media=SimpleNamespace(
            resolve_image_outputs=lambda *args: [Path("moss_image_1_a.png")]))
        runtime = SimpleNamespace(input_dir=Path("."))
        sizes = patch("h3_app.generation.optional_engines.first_reference_dimensions", return_value=(1024, 1024))
        with sizes, patch("h3_app.generation.optional_engines.write_snapshot"):
            results = list(generate_optional_engine(request, services, runtime))
        self.assertIn("cannot see the edit LoRA", results[-1].status)
        self.assertFalse(submitted)
        info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0].append(EDIT_LORA)
        with sizes, patch("h3_app.generation.optional_engines.write_snapshot") as snapshot:
            results = list(generate_optional_engine(request, services, runtime))
        self.assertIn("MOSS-Image complete", results[-1].status)
        self.assertEqual(nodes(submitted, "EmptyFlux2LatentImage")[0]["width"], 1024)
        settings = snapshot.call_args.args[1]["settings"]
        self.assertEqual((settings["reference_count"], settings["edit_lora"]), (1, EDIT_LORA))
        self.assertNotIn("reference_images", settings)


class MossImageEditLoraStoreTests(unittest.TestCase):
    def test_uploads_keep_their_name_reuse_identical_files_and_hide_managed_loras(self):
        from h3_app.moss_image_loras import list_edit_loras, store_edit_lora

        with tempfile.TemporaryDirectory() as root:
            runtime = SimpleNamespace(comfy_dir=Path(root) / "ComfyUI")
            upload = Path(root) / "My Cyclops Edit.safetensors"
            upload.write_bytes(b"one")
            self.assertEqual(list_edit_loras(runtime), [])
            self.assertEqual(store_edit_lora(upload, runtime), "My_Cyclops_Edit.safetensors")
            self.assertEqual(store_edit_lora(upload, runtime), "My_Cyclops_Edit.safetensors")
            upload.write_bytes(b"two")
            renamed = store_edit_lora(upload, runtime)
            self.assertRegex(renamed, r"^My_Cyclops_Edit_[0-9a-f]{8}\.safetensors$")
            loras = runtime.comfy_dir / "models" / "loras"
            (loras / MODEL_SPECS["moss_image_dmd_8step_lora"].local_name).write_bytes(b"turbo")
            self.assertEqual(list_edit_loras(runtime), ["My_Cyclops_Edit.safetensors", renamed])
            with self.assertRaisesRegex(H3Error, "safetensors"):
                store_edit_lora(Path(root) / "missing.ckpt", runtime)
            managed = Path(root) / MODEL_SPECS["moss_image_dmd_8step_lora"].local_name
            managed.write_bytes(b"x")
            with self.assertRaisesRegex(H3Error, "managed"):
                store_edit_lora(managed, runtime)


class MossImageEditViewTests(unittest.TestCase):
    def test_edit_controls_are_generation_inputs_and_the_lora_list_refreshes(self):
        import gradio as gr
        from h3_ui.optional_engine_views import OptionalEngineActions, bind_moss_image, build_moss_image_view

        names = ["a.safetensors"]
        def stub(*_args):
            return None

        actions = OptionalEngineActions(*(stub for _ in range(15)), generate_moss_image=stub,
                                        enhance_moss_image_prompt=stub, moss_image_edit_loras=lambda: list(names),
                                        upload_moss_image_edit_lora=lambda path: "b.safetensors")
        with gr.Blocks() as demo:
            view = build_moss_image_view(gr.Group(), names)
            bind_moss_image(view, actions)
        self.assertEqual([choice[1] for choice in view.edit_lora.choices], names)
        generate = next(fn for fn in demo.fns.values() if fn.api_name == "generate_moss_image")
        ids = [block._id for block in generate.inputs]
        for component in (view.reference_images, view.edit_lora, view.edit_lora_strength, view.kv_cache, view.edit_size):
            self.assertIn(component._id, ids)

        def handler(trigger):
            return next(fn.fn for fn in demo.fns.values() if trigger._id in [target[0] for target in fn.targets]
                        and fn.outputs and fn.outputs[0]._id == view.edit_lora._id)

        names.append("b.safetensors")
        refreshed = handler(view.refresh_edit_loras)("gone.safetensors")
        self.assertEqual((refreshed["choices"], refreshed["value"]), (names, None))
        uploaded, cleared = handler(view.upload_edit_lora)("/tmp/b.safetensors")
        self.assertEqual((uploaded["value"], cleared), ("b.safetensors", None))


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
