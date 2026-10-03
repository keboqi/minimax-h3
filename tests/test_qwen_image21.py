from __future__ import annotations

import tempfile
import ast
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from h3_app import prompt_service
from h3_app.errors import H3Error
from h3_app.generation.qwen import (
    generate_qwen_image21,
    max_qwen_edit_dimensions,
    resolve_qwen_output_dimensions,
)
from h3_app.generation.requests import QwenImage21Request
from h3_app.workflows.qwen import (
    build_qwen_image21_graph,
    required_qwen_image21_nodes,
)
from h3_models import (
    DEFAULT_QWEN_IMAGE21_MODEL,
    DEFAULT_QWEN_IMAGE21_TEXT_ENCODER,
    MODEL_SPECS,
    QWEN_IMAGE21_MODEL_CHOICES,
    QWEN_IMAGE21_TEXT_ENCODER_CHOICES,
)
from h3_ui.qwen_bindings import qwen_preset_values, qwen_resolution_preset_values, qwen_turbo_defaults


class QwenImage21WorkflowTests(unittest.TestCase):
    def test_separate_edit_inputs_generate_per_image_with_individual_size(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = [Path(directory) / f"source-{index}.png" for index in range(3)]
            source_sizes = [(640, 480), (800, 800), (1024, 512)]
            for source, size in zip(sources, source_sizes):
                Image.new("RGB", size).save(source)
            request = QwenImage21Request(
                mode="Image edit", model_choice="BF16", text_encoder_choice="BF16",
                prompt="Make it blue", negative_prompt="",
                reference_images=tuple(map(str, sources)),
                width=1024, height=1024, reference_resolution=0,
                match_input_size=False, seed=42, steps=40, cfg=1.0,
                sampler_name="euler", scheduler="simple", cache_device="auto",
                cache_dtype="default", attention_backend="pytorch attention",
                batch_count=2, batch_edit_inputs=True, max_resolution=True,
            )
            graphs = []
            execution = SimpleNamespace(
                object_info=lambda: required_qwen_image21_nodes(editing=True),
                submit_prompt=lambda graph, client_id: (
                    graphs.append(graph) or f"job-{len(graphs)}"
                ),
                poll_comfy_progress=lambda prompt_id, graph: iter(()),
                wait_for_history=lambda prompt_id: {},
            )
            services = SimpleNamespace(
                models=SimpleNamespace(
                    unload_prompt_rewriter=lambda: None,
                    missing_qwen_image21_model_names=lambda *args: [],
                    ensure_qwen_image21_models=lambda *args: None,
                ),
                execution=execution,
                media=SimpleNamespace(resolve_image_outputs=lambda *args: [
                    Path(f"image-{len(graphs)}.png")
                ]),
            )
            with patch("h3_app.generation.qwen.write_snapshot") as snapshot:
                updates = list(generate_qwen_image21(
                    request, services, SimpleNamespace(input_dir=Path(directory))
                ))
            self.assertEqual(len(graphs), 6)
            self.assertEqual(
                [self._by_type(graph, "LoadImage")[0][1]["inputs"]["image"] for graph in graphs],
                [str(source) for source in sources for _ in range(2)],
            )
            self.assertTrue(
                all(len(self._by_type(graph, "LoadImage")) == 1 for graph in graphs)
            )
            self.assertEqual(
                [self._by_type(graph, "KSampler")[0][1]["inputs"]["seed"] for graph in graphs],
                [42, 43, 44, 45, 46, 47],
            )
            self.assertEqual(
                [
                    (call.args[1]["settings"]["resolved_width"],
                     call.args[1]["settings"]["resolved_height"])
                    for call in snapshot.call_args_list
                ],
                [
                    max_qwen_edit_dimensions(*size)
                    for size in source_sizes for _ in range(2)
                ],
            )
            self.assertEqual(len(updates[-1].output), 6)

            graphs.clear()
            with patch("h3_app.generation.qwen.write_snapshot"):
                updates = list(generate_qwen_image21(
                    replace(request, batch_edit_inputs=False), services,
                    SimpleNamespace(input_dir=Path(directory)),
                ))
            self.assertEqual(len(graphs), 2)
            self.assertTrue(
                all(len(self._by_type(graph, "LoadImage")) == 3 for graph in graphs)
            )
            self.assertEqual(len(updates[-1].output), 2)

    def test_batch_reuses_inputs_and_collects_each_image(self):
        request = QwenImage21Request(
            mode="Text to image", model_choice="BF16", text_encoder_choice="BF16",
            prompt="A red fox", negative_prompt="blurry", reference_images=(),
            width=1024, height=1024, reference_resolution=0,
            match_input_size=False, seed=42, steps=40, cfg=1.0,
            sampler_name="euler", scheduler="simple", cache_device="auto",
            cache_dtype="default", attention_backend="pytorch attention",
            batch_count=3,
        )
        graphs = []
        snapshots = []
        execution = SimpleNamespace(
            object_info=lambda: required_qwen_image21_nodes(editing=False),
            submit_prompt=lambda graph, client_id: (
                graphs.append(graph) or f"job-{len(graphs)}"
            ),
            poll_comfy_progress=lambda prompt_id, graph: iter(()),
            wait_for_history=lambda prompt_id: {},
        )
        services = SimpleNamespace(
            models=SimpleNamespace(
                unload_prompt_rewriter=lambda: None,
                missing_qwen_image21_model_names=lambda *args: [],
                ensure_qwen_image21_models=lambda *args: None,
            ),
            execution=execution,
            media=SimpleNamespace(resolve_image_outputs=lambda history, queued_at, count: [
                Path(f"image-{len(graphs)}.png")
            ]),
        )
        with patch("h3_app.generation.qwen.write_snapshot", side_effect=lambda path, data: snapshots.append((path, data))):
            updates = list(generate_qwen_image21(
                request, services, SimpleNamespace(input_dir=Path("."))
            ))
        self.assertEqual(len(graphs), 3)
        samplers = [self._by_type(graph, "KSampler")[0][1] for graph in graphs]
        self.assertEqual([node["inputs"]["seed"] for node in samplers], [42, 43, 44])
        self.assertEqual(
            [snapshot[1]["settings"]["seed"] for snapshot in snapshots], [42, 43, 44]
        )
        self.assertEqual(updates[-1].output, [
            "image-1.png", "image-2.png", "image-3.png"
        ])

        # Exercise the new default through validation, batching and provenance.
        from h3_app.catalog import QWEN_IMAGE21_DYNAMIC_SCHEDULER
        graphs.clear()
        snapshots.clear()
        dynamic_request = replace(request, scheduler=QWEN_IMAGE21_DYNAMIC_SCHEDULER)
        with patch("h3_app.generation.qwen.write_snapshot", side_effect=lambda path, data: snapshots.append((path, data))):
            updates = list(generate_qwen_image21(dynamic_request, services, SimpleNamespace(input_dir=Path("."))))
        self.assertEqual(len(graphs), 3)
        self.assertEqual([
            self._by_type(graph, "RandomNoise")[0][1]["inputs"]["noise_seed"] for graph in graphs
        ], [42, 43, 44])
        self.assertTrue(all(snapshot[1]["settings"]["scheduler"] == QWEN_IMAGE21_DYNAMIC_SCHEDULER
                            for snapshot in snapshots))
        self.assertEqual(len(updates[-1].output), 3)
        graphs.clear()
        execution.object_info = lambda: required_qwen_image21_nodes(editing=False) - {"H3Qwen21Sigmas"}
        updates = list(generate_qwen_image21(dynamic_request, services, SimpleNamespace(input_dir=Path("."))))
        self.assertFalse(graphs)
        self.assertIn("H3Qwen21Sigmas", updates[-1].status)

    def _build(self, references=(), match_input_size=True, *, steps=25, turbo_variant="Off",
               scheduler="simple"):
        return build_qwen_image21_graph(
            model_choice="INT8 ConvRot (lower VRAM)",
            text_encoder_choice="INT8 ConvRot (recommended)",
            prompt="A red fox reading a book",
            negative_prompt="blurry",
            reference_images=references,
            width=1024,
            height=768,
            reference_resolution=0,
            match_input_size=match_input_size,
            seed=123,
            steps=steps,
            cfg=1.0,
            sampler_name="euler",
            scheduler=scheduler,
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            accelerator="Off",
            output_stamp="1234",
            output_nonce="abcd",
            turbo_variant=turbo_variant,
        )

    @staticmethod
    def _by_type(graph, class_type):
        return [
            (node_id, node)
            for node_id, node in graph.items()
            if node["class_type"] == class_type
        ]

    def test_text_to_image_uses_official_native_nodes(self):
        graph = self._build()
        self.assertFalse(self._by_type(graph, "LoadImage"))
        self.assertFalse(self._by_type(graph, "QwenImage21Cache"))
        encoder = self._by_type(graph, "TextEncodeQwenImage21")[0][1]
        self.assertNotIn("vae", encoder["inputs"])
        latent_id, latent = self._by_type(graph, "EmptyLatentImage")[0]
        self.assertEqual((latent["inputs"]["width"], latent["inputs"]["height"]), (1024, 768))
        sampler = self._by_type(graph, "KSampler")[0][1]
        backend_id, backend = self._by_type(graph, "ModelAttentionBackend")[0]
        self.assertEqual(backend["inputs"]["attention"], "pytorch attention")
        self.assertEqual(sampler["inputs"]["model"], [backend_id, 0])
        self.assertEqual(sampler["inputs"]["latent_image"], [latent_id, 0])
        self.assertEqual(sampler["inputs"]["cfg"], 1.0)
        saved = self._by_type(graph, "SaveImage")[0][1]
        self.assertTrue(
            saved["inputs"]["filename_prefix"].startswith(
                "h3/image_staging/qwen_image21_"
            )
        )

    def test_edit_uses_references_conditioner_latent_and_cache(self):
        graph = self._build(("target.png", "style.png"))
        loads = self._by_type(graph, "LoadImage")
        self.assertEqual(
            [node["inputs"]["image"] for _, node in loads],
            ["target.png", "style.png"],
        )
        conditioner_id, conditioner = self._by_type(
            graph, "TextEncodeQwenImage21"
        )[0]
        self.assertIn("vae", conditioner["inputs"])
        joins = self._by_type(graph, "JoinImageWithAlpha")
        self.assertEqual(len(joins), len(loads))
        for index, ((loaded_id, _), (join_id, join)) in enumerate(zip(loads, joins), 1):
            self.assertEqual(join["inputs"], {"image": [loaded_id, 0], "alpha": [loaded_id, 1]})
            self.assertEqual(conditioner["inputs"][f"images.image_{index}"], [join_id, 0])
        cache_id, cache = self._by_type(graph, "QwenImage21Cache")[0]
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["model"], [cache_id, 0])
        self.assertEqual(sampler["inputs"]["latent_image"], [conditioner_id, 2])
        self.assertEqual(cache["inputs"]["device"], "auto")

    def test_max_edit_resolution_uses_custom_latent(self):
        graph = self._build(("target.png",), match_input_size=False)
        latent_id, latent = self._by_type(graph, "EmptyLatentImage")[0]
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["latent_image"], [latent_id, 0])
        self.assertEqual(
            (latent["inputs"]["width"], latent["inputs"]["height"]),
            (1024, 768),
        )

    def test_resolution_aware_base_sampling_uses_actual_output_latent(self):
        from h3_app.catalog import QWEN_IMAGE21_DEFAULTS, QWEN_IMAGE21_DYNAMIC_SCHEDULER

        self.assertEqual(QWEN_IMAGE21_DEFAULTS["scheduler"], QWEN_IMAGE21_DYNAMIC_SCHEDULER)
        for references, match in (((), False), (("rgba.png",), True), (("rgba.png",), False)):
            with self.subTest(references=references, match=match):
                graph = self._build(references, match, scheduler=QWEN_IMAGE21_DYNAMIC_SCHEDULER)
                self.assertFalse(self._by_type(graph, "KSampler"))
                self.assertFalse(self._by_type(graph, "H3Qwen21TurboSigmas"))
                sigma_id, sigma = self._by_type(graph, "H3Qwen21Sigmas")[0]
                sampler_id, sampler = self._by_type(graph, "SamplerCustomAdvanced")[0]
                expected_latent = (
                    [self._by_type(graph, "TextEncodeQwenImage21")[0][0], 2]
                    if references and match else [self._by_type(graph, "EmptyLatentImage")[0][0], 0]
                )
                self.assertEqual(sigma["inputs"], {"latent_image": expected_latent, "steps": 25})
                self.assertEqual(sampler["inputs"]["latent_image"], expected_latent)
                self.assertEqual(sampler["inputs"]["sigmas"], [sigma_id, 0])
                self.assertEqual(self._by_type(graph, "VAEDecode")[0][1]["inputs"]["samples"], [sampler_id, 0])
                guider = self._by_type(graph, "CFGGuider")[0][1]["inputs"]
                conditioner_id = self._by_type(graph, "TextEncodeQwenImage21")[0][0]
                self.assertEqual(guider["positive"], [conditioner_id, 0])
                self.assertEqual(guider["negative"], [conditioner_id, 1])
                self.assertEqual(self._by_type(graph, "RandomNoise")[0][1]["inputs"]["noise_seed"], 123)
                self.assertTrue({n["class_type"] for n in graph.values()} <= required_qwen_image21_nodes(
                    editing=bool(references), scheduler=QWEN_IMAGE21_DYNAMIC_SCHEDULER,
                ))
        legacy = required_qwen_image21_nodes(editing=False, scheduler="simple")
        self.assertNotIn("H3Qwen21Sigmas", legacy)
        self.assertNotIn("SamplerCustomAdvanced", legacy)

    def test_max_edit_resolution_stays_under_four_megapixels(self):
        self.assertEqual(max_qwen_edit_dimensions(1024, 1024), (1984, 1984))
        self.assertEqual(max_qwen_edit_dimensions(1920, 1080), (2560, 1440))
        self.assertEqual(max_qwen_edit_dimensions(1080, 1920), (1440, 2560))
        self.assertEqual(max_qwen_edit_dimensions(640, 480), (2304, 1728))
        for source in ((1, 10000), (10000, 1), (640, 480), (8192, 8192)):
            width, height = max_qwen_edit_dimensions(*source)
            self.assertTrue(256 <= width <= 2752 and width % 32 == 0)
            self.assertTrue(256 <= height <= 2752 and height % 32 == 0)
            self.assertLessEqual(width * height, 4_000_000)

    def test_prompt_writer_receives_max_edit_dimensions(self):
        from h3_ui import application as ui_app

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "target.png"
            Image.new("RGB", (1920, 1080)).save(source)
            with (
                patch.object(ui_app, "_runtime_config", return_value=object()),
                patch.object(
                    ui_app.prompt_service,
                    "enhance_qwen_image21_prompt",
                    return_value=("enhanced", "ok"),
                ) as enhance,
            ):
                ui_app.enhance_qwen_image21_prompt(
                    "Change the sky", "gemini", "", "Image edit",
                    [str(source)], 1024, 768,
                    edit_size=ui_app.QWEN_EDIT_SIZE_MAX,
                )
            self.assertEqual(enhance.call_args.args[5:7], (2560, 1440))

    def test_edit_size_radio_maps_to_exclusive_backend_flags(self):
        from h3_ui import application as ui_app

        self.assertEqual(
            ui_app.qwen_edit_size_flags(ui_app.QWEN_EDIT_SIZE_MATCH),
            (True, False),
        )
        self.assertEqual(
            ui_app.qwen_edit_size_flags(ui_app.QWEN_EDIT_SIZE_MAX),
            (False, True),
        )
        self.assertEqual(
            ui_app.qwen_edit_size_flags(ui_app.QWEN_EDIT_SIZE_MANUAL),
            (False, False),
        )
        with self.assertRaises(H3Error):
            ui_app.qwen_edit_size_flags("Invalid")

    def test_generation_receives_radio_edit_size_flags(self):
        from h3_ui import application as ui_app

        for edit_size, expected in (
            (ui_app.QWEN_EDIT_SIZE_MATCH, (True, False)),
            (ui_app.QWEN_EDIT_SIZE_MAX, (False, True)),
            (ui_app.QWEN_EDIT_SIZE_MANUAL, (False, False)),
        ):
            with self.subTest(edit_size=edit_size):
                with (
                    patch.object(ui_app, "_generation_services", return_value=object()),
                    patch.object(ui_app, "_runtime_config", return_value=object()),
                    patch.object(
                        ui_app.qwen_generation,
                        "generate_qwen_image21",
                        return_value=iter(()),
                    ) as generate,
                ):
                    list(
                        ui_app.generate_qwen_image21(
                            "Image edit", "BF16", "BF16", "Edit", "", (),
                            1024, 768, 0, edit_size, -1, 40, 1.0, "euler",
                            "simple", "auto", "default",
                        )
                    )
                request = generate.call_args.args[0]
                self.assertEqual(
                    (request.match_input_size, request.max_resolution), expected
                )

    def test_prompt_writer_matches_first_edit_image_dimensions(self):
        from h3_ui import application as ui_app

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "target.png"
            Image.new("RGB", (1920, 1080)).save(source)
            with (
                patch.object(ui_app, "_runtime_config", return_value=object()),
                patch.object(
                    ui_app.prompt_service,
                    "enhance_qwen_image21_prompt",
                    return_value=("enhanced", "ok"),
                ) as enhance,
            ):
                ui_app.enhance_qwen_image21_prompt(
                    "Change the sky", "gemini", "", "Image edit",
                    [str(source)], 1024, 768,
                    edit_size=ui_app.QWEN_EDIT_SIZE_MATCH,
                )
            self.assertEqual(enhance.call_args.args[5:7], (1920, 1080))

    def test_max_resolution_overrides_match_input_size_for_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "target.png"
            Image.new("RGB", (1920, 1080)).save(source)
            request = SimpleNamespace(
                width=1024,
                height=768,
                match_input_size=True,
                max_resolution=True,
            )
            self.assertEqual(
                resolve_qwen_output_dimensions(request, (str(source),), True),
                (2560, 1440, False),
            )
            self.assertEqual(
                resolve_qwen_output_dimensions(request, (), False),
                (1024, 768, True),
            )
            request.max_resolution = False
            self.assertEqual(
                resolve_qwen_output_dimensions(request, (str(source),), True),
                (1024, 768, True),
            )

    def test_model_registry_matches_published_repository_layout(self):
        selected = {
            *QWEN_IMAGE21_MODEL_CHOICES.values(),
            *QWEN_IMAGE21_TEXT_ENCODER_CHOICES.values(),
            "qwen_image21_vae",
        }
        self.assertTrue(selected)
        for key in selected:
            self.assertEqual(MODEL_SPECS[key].repo_id, "Comfy-Org/Qwen-Image-2.1")
        self.assertEqual(
            MODEL_SPECS["qwen_image21_vae"].filename,
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        )

    def test_required_nodes_add_edit_only_nodes(self):
        generate = required_qwen_image21_nodes(editing=False)
        edit = required_qwen_image21_nodes(editing=True)
        self.assertNotIn("LoadImage", generate)
        self.assertNotIn("QwenImage21Cache", generate)
        self.assertIn("ModelAttentionBackend", generate)
        self.assertTrue({"LoadImage", "QwenImage21Cache"} <= edit)

    def test_bf16_is_the_quality_default(self):
        self.assertEqual(DEFAULT_QWEN_IMAGE21_MODEL, "BF16")
        self.assertEqual(DEFAULT_QWEN_IMAGE21_TEXT_ENCODER, "BF16")

    def test_spectrum_quality_is_the_default_accelerator(self):
        from h3_app.catalog import QWEN_IMAGE21_DEFAULTS

        self.assertEqual(
            QWEN_IMAGE21_DEFAULTS["accelerator"],
            "Spectrum (Quality)",
        )

    def test_native_resolution_presets(self):
        from h3_ui.qwen_view import QWEN_1K_RESOLUTION_PRESETS, QWEN_2K_RESOLUTION_PRESETS

        self.assertEqual(len(QWEN_1K_RESOLUTION_PRESETS), 7)
        self.assertEqual(len(QWEN_2K_RESOLUTION_PRESETS), 7)
        for preset in QWEN_1K_RESOLUTION_PRESETS:
            self.assertTrue(preset.startswith("1K · "))
            w, h = qwen_resolution_preset_values(preset)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
        for preset in QWEN_2K_RESOLUTION_PRESETS:
            self.assertTrue(preset.startswith("2K · "))
            w, h = qwen_resolution_preset_values(preset)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

        self.assertEqual(
            qwen_resolution_preset_values("1K · 1:1 · 1024×1024"),
            (1024, 1024),
        )
        self.assertEqual(
            qwen_resolution_preset_values("1K · 16:9 · 1824×1024"),
            (1824, 1024),
        )
        self.assertEqual(
            qwen_resolution_preset_values("2K · 9:16 · 1536×2752"),
            (1536, 2752),
        )

    def test_generate_qwen_image21_accepts_1k_and_2k_presets(self):
        from h3_ui import application as ui_app

        with (
            patch.object(ui_app, "_generation_services", return_value=object()),
            patch.object(ui_app, "_runtime_config", return_value=object()),
            patch.object(
                ui_app.qwen_generation,
                "generate_qwen_image21",
                return_value=iter(()),
            ) as generate,
        ):
            # Test 1K preset
            list(
                ui_app.generate_qwen_image21(
                    "Image edit", "BF16", "BF16", "Edit", "", (),
                    512, 512, 0, ui_app.QWEN_EDIT_SIZE_MATCH, -1, 40, 1.0, "euler",
                    "simple", "auto", "default",
                    output_resolution_1k="1K · 16:9 · 1824×1024",
                )
            )
            req = generate.call_args.args[0]
            self.assertEqual((req.width, req.height), (1824, 1024))
            self.assertFalse(req.match_input_size)

            # Test 2K preset
            list(
                ui_app.generate_qwen_image21(
                    "Image edit", "BF16", "BF16", "Edit", "", (),
                    512, 512, 0, ui_app.QWEN_EDIT_SIZE_MATCH, -1, 40, 1.0, "euler",
                    "simple", "auto", "default",
                    output_resolution_2k="2K · 3:2 · 2528×1696",
                )
            )
            req = generate.call_args.args[0]
            self.assertEqual((req.width, req.height), (2528, 1696))
            self.assertFalse(req.match_input_size)

    def test_optional_kitchen_attention_wraps_the_model(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="Transparent glass sculpture",
            negative_prompt="",
            reference_images=(),
            width=2048,
            height=2048,
            reference_resolution=0,
            match_input_size=True,
            seed=42,
            steps=40,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="comfy kitchen attention",
            accelerator="Off",
            output_stamp="1234",
            output_nonce="abcd",
        )
        backend_id, backend = self._by_type(graph, "ModelAttentionBackend")[0]
        self.assertEqual(
            backend["inputs"]["attention"], "comfy kitchen attention"
        )
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(sampler["inputs"]["model"], [backend_id, 0])

    def test_optional_spectrum_wraps_the_final_patched_model(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="Transparent glass sculpture",
            negative_prompt="",
            reference_images=("target.png",),
            width=1024,
            height=1024,
            reference_resolution=0,
            match_input_size=True,
            seed=42,
            steps=40,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            accelerator="Spectrum",
            output_stamp="1234",
            output_nonce="abcd",
        )
        cache_id, _cache = self._by_type(graph, "QwenImage21Cache")[0]
        spectrum_id, spectrum = self._by_type(
            graph, "QwenSpectrumModelPatcher"
        )[0]
        sampler = self._by_type(graph, "KSampler")[0][1]
        self.assertEqual(spectrum["inputs"]["model"], [cache_id, 0])
        # The legacy value now resolves to the quality profile. Edits reserve
        # two extra exact tail steps to preserve source texture and identity.
        self.assertEqual(spectrum["inputs"]["warmup_steps"], 10)
        self.assertEqual(spectrum["inputs"]["tail_actual_steps"], 10)
        self.assertEqual(spectrum["inputs"]["max_consecutive_forecasts"], 1)
        self.assertEqual(sampler["inputs"]["model"], [spectrum_id, 0])
        self.assertIn(
            "QwenSpectrumModelPatcher",
            required_qwen_image21_nodes(editing=False, use_spectrum=True),
        )

    def test_spectrum_preview_retains_the_faster_schedule(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="Transparent glass sculpture",
            negative_prompt="",
            reference_images=(),
            width=1024,
            height=1024,
            reference_resolution=0,
            match_input_size=True,
            seed=42,
            steps=40,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            accelerator="Spectrum (Preview)",
            output_stamp="1234",
            output_nonce="abcd",
        )
        spectrum = self._by_type(graph, "QwenSpectrumModelPatcher")[0][1]
        self.assertEqual(spectrum["inputs"]["warmup_steps"], 5)
        self.assertEqual(spectrum["inputs"]["tail_actual_steps"], 2)

    def test_spectrum_quality_uses_conservative_generation_schedule(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="Transparent glass sculpture",
            negative_prompt="",
            reference_images=(),
            width=1024,
            height=1024,
            reference_resolution=0,
            match_input_size=True,
            seed=42,
            steps=40,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            accelerator="Spectrum (Quality)",
            output_stamp="1234",
            output_nonce="abcd",
        )
        spectrum = self._by_type(graph, "QwenSpectrumModelPatcher")[0][1]
        self.assertEqual(spectrum["inputs"]["warmup_steps"], 10)
        self.assertEqual(spectrum["inputs"]["tail_actual_steps"], 8)

    def test_viggle_turbo_uses_lora_and_custom_sigmas(self):
        graph = build_qwen_image21_graph(
            model_choice="BF16",
            text_encoder_choice="BF16",
            prompt="A red fox reading a book",
            negative_prompt="",
            reference_images=(),
            width=1024,
            height=1024,
            reference_resolution=0,
            match_input_size=True,
            seed=123,
            steps=6,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            cache_device="auto",
            cache_dtype="default",
            attention_backend="pytorch attention",
            accelerator="Off",
            output_stamp="1234",
            output_nonce="abcd",
            turbo_variant="Viggle Turbo v0.2.1 (6-step)",
        )
        self.assertFalse(self._by_type(graph, "KSampler"))
        lora_id, lora = self._by_type(graph, "H3Qwen21ViggleLora")[0]
        self.assertEqual(
            lora["inputs"]["lora_name"],
            MODEL_SPECS["qwen_image21_viggle_v02_lora"].local_name,
        )
        backend_id = self._by_type(graph, "ModelAttentionBackend")[0][0]
        self.assertEqual(lora["inputs"]["model"], [backend_id, 0])
        sigmas_id, sigmas = self._by_type(graph, "H3Qwen21TurboSigmas")[0]
        self.assertEqual(sigmas["inputs"]["steps"], 6)
        sampler = self._by_type(graph, "SamplerCustomAdvanced")[0][1]
        self.assertEqual(sampler["inputs"]["sigmas"], [sigmas_id, 0])
        self.assertTrue(
            {"H3Qwen21ViggleLora", "H3Qwen21TurboSigmas", "CFGGuider"}
            <= required_qwen_image21_nodes(editing=False, turbo=True, viggle=True)
        )

    def test_qwen_presets_select_the_requested_controls(self):
        self.assertEqual(
            qwen_preset_values("Fast"),
            ("INT8 ConvRot (lower VRAM)", "Viggle Turbo v0.2.1 (6-step)", 6, "Off"),
        )
        self.assertEqual(
            qwen_preset_values("Normal"),
            ("BF16", "Off", 25, "Spectrum (Quality)"),
        )
        self.assertEqual(
            qwen_preset_values("Quality"),
            ("BF16", "Off", 40, "Spectrum (Quality)"),
        )

    def test_v03_six_step_uses_shared_adapter_without_base_tail(self):
        from h3_app.model_service import qwen_image21_model_keys

        variant = "Viggle Turbo v0.3 (6-step)"
        self.assertEqual(qwen_turbo_defaults(variant), (6, 1.0, "euler", "Off"))
        self.assertEqual(
            qwen_image21_model_keys("BF16", "BF16", variant),
            qwen_image21_model_keys("BF16", "BF16", "Viggle Turbo v0.3 (9-step)"),
        )
        for references in ((), ("subject.png",)):
            with self.subTest(references=references):
                graph = self._build(references=references, steps=6, turbo_variant=variant)
                lora_id, lora = self._by_type(graph, "H3Qwen21ViggleLora")[0]
                self.assertEqual(lora["inputs"]["lora_name"],
                                 MODEL_SPECS["qwen_image21_viggle_v03_lora"].local_name)
                self.assertEqual(len(self._by_type(graph, "SamplerCustomAdvanced")), 1)
                self.assertFalse(self._by_type(graph, "SplitSigmas"))
                self.assertFalse(self._by_type(graph, "DisableNoise"))
                guider = self._by_type(graph, "CFGGuider")[0][1]
                self.assertEqual(guider["inputs"]["model"], [lora_id, 0])
                sigma_id, sigma = self._by_type(graph, "H3Qwen21TurboSigmas")[0]
                self.assertEqual(sigma["inputs"]["steps"], 6)
                sampler = self._by_type(graph, "SamplerCustomAdvanced")[0][1]
                self.assertEqual(sampler["inputs"]["sigmas"], [sigma_id, 0])
        with self.assertRaisesRegex(ValueError, "requires 6 steps"):
            self._build(steps=9, turbo_variant=variant)

    def test_nine_step_handoff_uses_base_model_and_preserves_noisy_output(self):
        for references in ((), ("subject.png", "style.png")):
            with self.subTest(references=references):
                graph = self._build(
                    references=references, steps=9,
                    turbo_variant="Viggle Turbo v0.3 (9-step)",
                )
                lora_id, lora = self._by_type(graph, "H3Qwen21ViggleLora")[0]
                self.assertEqual(lora["inputs"]["lora_name"],
                                 MODEL_SPECS["qwen_image21_viggle_v03_lora"].local_name)
                split_id, split = self._by_type(graph, "SplitSigmas")[0]
                self.assertEqual(split["inputs"]["step"], 7)
                sigma_id, sigma = self._by_type(graph, "H3Qwen21TurboSigmas")[0]
                self.assertEqual(sigma["inputs"]["steps"], 9)
                self.assertEqual(split["inputs"]["sigmas"], [sigma_id, 0])
                (first_id, first), (tail_id, tail) = self._by_type(graph, "SamplerCustomAdvanced")
                self.assertEqual(first["inputs"]["sigmas"], [split_id, 0])
                self.assertEqual(tail["inputs"]["sigmas"], [split_id, 1])
                self.assertEqual(tail["inputs"]["latent_image"], [first_id, 0])
                self.assertEqual(tail["inputs"]["sampler"], first["inputs"]["sampler"])
                noise_id = self._by_type(graph, "DisableNoise")[0][0]
                self.assertEqual(tail["inputs"]["noise"], [noise_id, 0])
                guiders = self._by_type(graph, "CFGGuider")
                self.assertEqual(guiders[0][1]["inputs"]["model"], [lora_id, 0])
                self.assertEqual(guiders[1][1]["inputs"]["model"], lora["inputs"]["model"])
                self.assertEqual(guiders[1][1]["inputs"]["cfg"], 1.0)
                self.assertEqual(self._by_type(graph, "VAEDecode")[0][1]["inputs"]["samples"],
                                 [tail_id, 0])
                if references:
                    cache_id = self._by_type(graph, "QwenImage21Cache")[0][0]
                    self.assertEqual(lora["inputs"]["model"], [cache_id, 0])
                available = required_qwen_image21_nodes(
                    editing=bool(references), turbo=True, viggle=True, nine_step=True,
                )
                self.assertTrue({node["class_type"] for node in graph.values()} <= available)

    def test_nine_step_download_defaults_and_validation(self):
        from h3_app.model_service import qwen_image21_model_keys
        from h3_models import LAZY_OPTIONAL_MODEL_KEYS, PRELOAD_MODEL_KEYS

        variant = "Viggle Turbo v0.3 (9-step)"
        self.assertEqual(qwen_turbo_defaults(variant), (9, 1.0, "euler", "Off"))
        keys = qwen_image21_model_keys("BF16", "BF16", variant)
        self.assertEqual(keys[:-1], qwen_image21_model_keys("BF16", "BF16"))
        self.assertEqual(keys[-1], "qwen_image21_viggle_v03_lora")
        self.assertIn(keys[-1], LAZY_OPTIONAL_MODEL_KEYS)
        self.assertNotIn(keys[-1], PRELOAD_MODEL_KEYS)
        with self.assertRaisesRegex(ValueError, "requires 9 steps"):
            self._build(steps=6, turbo_variant=variant)

    def test_published_sigmas_and_flow_handoff_are_continuous(self):
        # Load only the scheduler class; these tests need no ComfyUI/GPU runtime.
        path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in {"H3Qwen21TurboSigmas", "_qwen21_target_tokens"}]
        namespace = {"math": math, "torch": SimpleNamespace(
            tensor=lambda values, dtype: values, float32="float32",
        )}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        scheduler = namespace["H3Qwen21TurboSigmas"]()
        for height, width, mu in ((64, 64, .6935483870967742),
                                  (48, 64, .6419354838709678),
                                  (128, 128, 1.3129032258064517)):
            latent = {"samples": SimpleNamespace(shape=(1, 64, height, width))}
            sigmas = scheduler.calculate(latent, 9)[0]
            raw = [1, .9583, .9167, .875, .75, .5, .25, 1 / 6, 1 / 12]
            expected = [math.exp(mu) / (math.exp(mu) + 1 / s - 1) for s in raw] + [0]
            for actual, target in zip(sigmas, expected):
                self.assertAlmostEqual(actual, target, places=14)
            self.assertEqual(len(sigmas[:8]) - 1, 7)
            self.assertEqual(len(sigmas[7:]) - 1, 2)
            # Comfy flow inverse_noise_scaling at stage end and noise_scaling
            # with zero fresh noise at the next start must cancel exactly.
            boundary = sigmas[7]
            noisy = 0.42
            exported = noisy / (1 - boundary)
            resumed = (1 - boundary) * exported
            self.assertAlmostEqual(resumed, noisy)
            six = scheduler.calculate(latent, 6)[0]
            self.assertEqual(len(six), 7)
            self.assertEqual(six[-1], 0)
            expected_six = [math.exp(mu) / (math.exp(mu) + 1 / s - 1)
                            for s in (1, .9375, .875, .75, .5, .25)] + [0]
            for actual, target in zip(six, expected_six):
                self.assertAlmostEqual(actual, target, places=14)
            empty = {"samples": SimpleNamespace(shape=(1, 4, height * 2, width * 2)),
                     "downscale_ratio_spacial": 8}
            self.assertEqual(scheduler.calculate(empty, 9)[0], sigmas)
            self.assertEqual(scheduler.calculate(empty, 6)[0], six)
        with self.assertRaisesRegex(ValueError, "6 or 9"):
            scheduler.calculate(latent, 8)

    def test_nine_step_requests_execute_and_reject_invalid_settings_before_download(self):
        request = QwenImage21Request(
            mode="Text to image", model_choice="BF16", text_encoder_choice="BF16",
            prompt="A sign reading OPEN", negative_prompt="", reference_images=(),
            width=1024, height=1024, reference_resolution=0, match_input_size=False,
            seed=42, steps=9, cfg=1.0, sampler_name="euler", scheduler="simple",
            cache_device="auto", cache_dtype="default", attention_backend="pytorch attention",
            turbo_variant="Viggle Turbo v0.3 (9-step)",
        )
        graphs = []
        models = SimpleNamespace(
            unload_prompt_rewriter=Mock(),
            missing_qwen_image21_model_names=Mock(return_value=[]),
            ensure_qwen_image21_models=Mock(),
        )
        services = SimpleNamespace(
            models=models,
            execution=SimpleNamespace(
                object_info=lambda: required_qwen_image21_nodes(
                    editing=False, turbo=True, viggle=True, nine_step=True,
                ),
                submit_prompt=lambda graph, client_id: (graphs.append(graph) or "job-9"),
                poll_comfy_progress=lambda prompt_id, graph: iter(()),
                wait_for_history=lambda prompt_id: {},
            ),
            media=SimpleNamespace(resolve_image_outputs=lambda *args: [Path("nine.png")]),
        )
        runtime = SimpleNamespace(input_dir=Path("."))
        with patch("h3_app.generation.qwen.write_snapshot"):
            updates = list(generate_qwen_image21(request, services, runtime))
        self.assertEqual(updates[-1].output, ["nine.png"])
        self.assertEqual(len(self._by_type(graphs[0], "SamplerCustomAdvanced")), 2)
        models.ensure_qwen_image21_models.assert_called_once_with(
            "BF16", "BF16", request.turbo_variant,
        )
        for settings, message in (
            ({"steps": 6}, "exactly 9"),
            ({"cfg": 2}, "CFG 1"),
            ({"sampler_name": "heun"}, "Euler"),
            ({"accelerator": "Spectrum (Quality)"}, "unavailable"),
        ):
            with self.subTest(settings=settings):
                models.ensure_qwen_image21_models.reset_mock()
                updates = list(generate_qwen_image21(
                    replace(request, **settings), services, runtime,
                ))
                self.assertIn("Error:", updates[-1].status)
                self.assertIn(message, updates[-1].status)
                models.ensure_qwen_image21_models.assert_not_called()

    def test_viggle_schedule_is_fixed_and_model_download_is_optional(self):
        from h3_app.model_service import qwen_image21_model_keys

        self.assertEqual(qwen_turbo_defaults("Viggle Turbo v0.2.1 (6-step)")[0], 6)
        self.assertEqual(qwen_turbo_defaults("Off", "Normal")[0], 25)
        self.assertEqual(qwen_turbo_defaults("Off", "Quality")[0], 40)
        with self.assertRaisesRegex(ValueError, "requires 6 steps"):
            self._build(steps=7, turbo_variant="Viggle Turbo v0.2.1 (6-step)")
        base = qwen_image21_model_keys("BF16", "BF16")
        turbo = qwen_image21_model_keys("BF16", "BF16", "Viggle Turbo v0.2.1 (6-step)")
        self.assertEqual(turbo[:-1], base)
        self.assertEqual(turbo[-1], "qwen_image21_viggle_v02_lora")
        self.assertEqual(
            MODEL_SPECS[turbo[-1]].repo_id,
            "Viggle/Qwen-Image-2.1-viggle-turbo",
        )

    def test_retired_turbo_variants_are_unavailable(self):
        from h3_app.model_service import qwen_image21_model_keys

        for variant in (
            "Viggle 3-pass (configurable)",
            "Alibaba PAI PDD 4-step",
            "Pruna 8-step",
            "Pruna 5-step",
        ):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, "Unknown Qwen Image 2.1 Turbo"):
                    self._build(turbo_variant=variant)
                with self.assertRaisesRegex(H3Error, "Unknown Qwen Image 2.1 Turbo"):
                    qwen_image21_model_keys("BF16", "BF16", variant)

    def test_prompt_writer_uses_natural_single_image_reference(self):
        runtime = SimpleNamespace(
            prompt_systems={"Qwen Image 2.1": Path("prompt_qwen_image21.txt")}
        )
        with patch.object(
            prompt_service,
            "_enhance_prompt_from_media",
            return_value=("enhanced", "ok"),
        ) as enhance:
            result = prompt_service.enhance_qwen_image21_prompt(
                "change the sky",
                "gemini",
                "",
                "Image edit",
                ["target.png"],
                1024,
                1024,
                runtime=runtime,
            )
        self.assertEqual(result, ("enhanced", "ok"))
        arguments = enhance.call_args.kwargs
        self.assertEqual(
            arguments["media_values"],
            (("Input image (edit target)", "target.png"),),
        )
        self.assertIn("do not use an <image1> tag", arguments["context"])

    def test_prompt_writer_tags_every_multi_image_input(self):
        runtime = SimpleNamespace(
            prompt_systems={"Qwen Image 2.1": Path("prompt_qwen_image21.txt")}
        )
        with patch.object(
            prompt_service,
            "_enhance_prompt_from_media",
            return_value=("enhanced", "ok"),
        ) as enhance:
            prompt_service.enhance_qwen_image21_prompt(
                "put the shirt on the person",
                "gemini",
                "",
                "Image edit",
                ["target.png", "shirt.png"],
                1024,
                1024,
                runtime=runtime,
            )
        arguments = enhance.call_args.kwargs
        self.assertEqual(
            arguments["media_values"],
            (("<image1>", "target.png"), ("<image2>", "shirt.png")),
        )
        self.assertIn("Use the numbered image tags verbatim", arguments["context"])

    def test_prompt_rules_include_official_transparency_wrapper(self):
        rules = (Path(__file__).resolve().parents[1] / "prompt_qwen_image21.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("do not use an <image1> tag", rules)
        self.assertIn(
            "This is an RGBA image with transparency. <description>. "
            "The image has alpha channel and the background is transparent.",
            rules,
        )


if __name__ == "__main__":
    unittest.main()
