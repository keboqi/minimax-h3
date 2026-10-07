"""Focused CPU contracts retained from the former launcher self-test."""

import unittest
import unittest.mock

import h3_app.catalog as _catalog
import h3_app.errors as _errors
import h3_app.graph as _graph
import h3_app.policy as _policy
import h3_app.workflows.h3 as _h3_workflow
from tests.workflow_cases import MODELS as fake
from tests.workflow_cases import model_stack


class WorkflowContractsTests(unittest.TestCase):
    def test_trt_decode_preserves_conditioning_vae(self):
        hybrid_graph = _graph.Graph()
        trt_decode_ref = ["trt-vae", 0]
        self.assertEqual(
            _h3_workflow.h3_conditioning_video_vae(
                hybrid_graph,
                fake,
                trt_decode_ref,
                use_trt_vae=True,
                has_visual_conditioning=False,
            ),
            trt_decode_ref,
        )
        regular_encode_ref = _h3_workflow.h3_conditioning_video_vae(
            hybrid_graph,
            fake,
            trt_decode_ref,
            use_trt_vae=True,
            has_visual_conditioning=True,
        )
        self.assertNotEqual(regular_encode_ref, trt_decode_ref)
        hybrid_loader = next(iter(hybrid_graph.nodes.values()))
        self.assertEqual(hybrid_loader["class_type"], "VAELoader")
        self.assertEqual(hybrid_loader["inputs"], {"vae_name": fake.video_vae})
        trt_graph = _graph.Graph()
        model_stack(
            trt_graph,
            fake.profile("speed").fl2va,
            fake,
            turbo_variant=_catalog.LIGHTX2V_4STEP_TURBO,
            sol_dense_steps=1,
            available_nodes={"MiniMaxH3TRTVAELoader"},
            use_trt_vae=True,
        )
        trt_loader = next(
            node
            for node in trt_graph.nodes.values()
            if node["class_type"] == "MiniMaxH3TRTVAELoader"
        )
        self.assertEqual(
            trt_loader["inputs"],
            {"decoder": "minimax_h3_vae_decoder.engine", "encoder": "None"},
        )

    def test_conditioning_prompt_changes_cache_identity(self):
        expected_cache_key = _h3_workflow.h3_conditioning_cache_key(
            "fl2va",
            "test",
            "qwen3vl_32b_minimax_h3_bf16.safetensors",
            [],
            encoder_settings={"encoder_small_input": True},
        )
        self.assertNotEqual(
            expected_cache_key,
            _h3_workflow.h3_conditioning_cache_key(
                "fl2va",
                "changed",
                "qwen3vl_32b_minimax_h3_bf16.safetensors",
                [],
                encoder_settings={"encoder_small_input": True},
            ),
        )

    def test_image_and_audio_output_preflight(self):
        self.assertEqual(_policy.normalize_result_format("image"), "Image")
        self.assertEqual(_policy.image_sampling_length(1), 5)
        self.assertEqual(_policy.image_sampling_length(5), 5)
        self.assertEqual(_policy.image_sampling_length(6), 22)
        self.assertEqual(_policy.image_sampling_length(20), 22)
        self.assertEqual(_policy.single_frame_image_sampling_length(1), 5)
        self.assertEqual(
            _policy.selected_image_sampling_length(20, _catalog.OFFICIAL_IMAGE_VAE), 22
        )
        try:
            _policy.selected_image_sampling_length(2, _catalog.SINGLE_FRAME_IMAGE_VAE)
        except _errors.H3Error as exc:
            self.assertIn("exactly one output image", str(exc))
        else:
            raise AssertionError("Single-frame 500K accepted multiple output images")
        self.assertLessEqual(
            {"VAEDecode", "ImageFromBatch", "SaveImage"},
            _h3_workflow.required_nodes_for(
                "Text to video", False, "Off", result_format="Image"
            ),
        )
        self.assertLessEqual(
            {"VAEDecodeAudio", "SaveAudioMP3"},
            _h3_workflow.required_nodes_for(
                "Text to video", False, "Off", result_format="Audio"
            ),
        )
        self.assertLessEqual(
            {_catalog.H3_SINGLE_FRAME_VAE_LOADER_NODE, _catalog.H3_IMAGE_SLICES_NODE},
            _h3_workflow.required_nodes_for(
                "Text to video",
                False,
                "Off",
                result_format="Image",
                image_vae=_catalog.SINGLE_FRAME_IMAGE_VAE,
            ),
        )

    def test_image_and_audio_finish_branches(self):
        image_result_graph = _graph.Graph()
        _h3_workflow.finish_sampling(
            image_result_graph,
            model_ref=["model", 0],
            conditioning_ref=["conditioning", 0],
            latent_ref=["latent", 0],
            video_vae_ref=["video_vae", 0],
            audio_vae_ref=["audio_vae", 0],
            seed=1,
            steps=4,
            scheduler="simple",
            turbo_variant=None,
            filename_prefix="h3/image_staging/selftest",
            result_format="Image",
            image_frames=3,
        )
        image_result_classes = {
            node["class_type"] for node in image_result_graph.nodes.values()
        }
        self.assertLessEqual(
            {"VAEDecode", "ImageFromBatch", "SaveImage"}, image_result_classes
        )
        self.assertFalse(
            image_result_classes & {"VAEDecodeAudio", "CreateVideo", "SaveVideo"}
        )
        image_slice = next(
            node
            for node in image_result_graph.nodes.values()
            if node["class_type"] == "ImageFromBatch"
        )
        self.assertEqual(image_slice["inputs"]["length"], 3)
        audio_result_graph = _graph.Graph()
        _h3_workflow.finish_sampling(
            audio_result_graph,
            model_ref=["model", 0],
            conditioning_ref=["conditioning", 0],
            latent_ref=["latent", 0],
            video_vae_ref=["video_vae", 0],
            audio_vae_ref=["audio_vae", 0],
            seed=1,
            steps=4,
            scheduler="simple",
            turbo_variant=None,
            filename_prefix="audio/h3_selftest",
            result_format="Audio",
        )
        audio_result_classes = {
            node["class_type"] for node in audio_result_graph.nodes.values()
        }
        self.assertLessEqual({"VAEDecodeAudio", "SaveAudioMP3"}, audio_result_classes)
        self.assertFalse(
            audio_result_classes & {"VAEDecode", "CreateVideo", "SaveVideo"}
        )

    def test_split_refinement_configuration_and_wiring(self):
        self.assertEqual(
            _policy.h3_latent_upscale_dimensions(1024, 1024), (512, 512, 1024, 1024)
        )
        self.assertEqual(
            _policy.h3_latent_upscale_dimensions(864, 480), (448, 256, 896, 512)
        )
        split_config = _policy.resolve_h3_split_upscale_config(
            _catalog.H3_LATENT_UPSCALE_SPLIT,
            tile_width=512,
            tile_height=640,
            overlap_ratio=0.25,
            fade_ratio=0.50,
            chunk_frames=73,
            temporal_overlap_frames=22,
            seam_denoise=0.75,
            seam_polish="auto",
        )
        self.assertEqual(
            split_config,
            _policy.H3SplitUpscaleConfig(
                tile_width=512,
                tile_height=640,
                overlap_ratio=0.25,
                fade_ratio=0.5,
                chunk_frames=73,
                temporal_overlap_frames=22,
                seam_denoise=0.75,
                seam_polish="auto",
            ),
        )
        self.assertIs(
            _policy.resolve_h3_split_upscale_config(
                _catalog.H3_LATENT_UPSCALE_STANDARD,
                tile_width=1,
                tile_height=1,
                overlap_ratio=2,
                fade_ratio=2,
                chunk_frames=1,
                temporal_overlap_frames=2,
                seam_denoise=2,
                seam_polish="invalid",
            ),
            None,
        )
        self.assertLessEqual(
            {
                _catalog.H3_SPLIT_TEMPORAL_PARAMS_NODE,
                _catalog.H3_SPLIT_SPATIAL_PARAMS_NODE,
                _catalog.H3_SPLIT_UPSCALE_NODE,
            },
            _h3_workflow.required_nodes_for(
                "Text to video",
                False,
                "Off",
                latent_upscale=True,
                latent_upscale_method=_catalog.H3_LATENT_UPSCALE_SPLIT,
            ),
        )
        split_graph_builder = _graph.Graph()
        _h3_workflow.finish_sampling(
            split_graph_builder,
            model_ref=["model", 0],
            conditioning_ref=["target-conditioning", 0],
            latent_ref=["target-latent", 0],
            video_vae_ref=["video-vae", 0],
            audio_vae_ref=["audio-vae", 0],
            seed=11,
            steps=8,
            scheduler="beta",
            turbo_variant=None,
            filename_prefix="h3/split-selftest",
            initial_conditioning_ref=["initial-conditioning", 0],
            initial_latent_ref=["initial-latent", 0],
            latent_upscale_model_name=(
                "minimax_h3_latent_upscaler_3d_bf16.safetensors"
            ),
            latent_upscale_precision="bf16",
            latent_upscale_refine_steps=2,
            latent_split_config=split_config,
        )
        split_graph = split_graph_builder.nodes
        split_temporal_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == _catalog.H3_SPLIT_TEMPORAL_PARAMS_NODE
        )
        split_spatial_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == _catalog.H3_SPLIT_SPATIAL_PARAMS_NODE
        )
        split_upscale_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == _catalog.H3_SPLIT_UPSCALE_NODE
        )
        split_sigmas_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == "SplitSigmas"
        )
        split_noise_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == "RandomNoise"
        )
        split_sampler_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == _catalog.CORE_SAMPLER_NODE
        )
        split_combine_id = next(
            node_id
            for node_id, node in split_graph.items()
            if node["class_type"] == _catalog.H3_COMBINE_AV_LATENT_NODE
        )
        self.assertEqual(
            sum(
                (
                    node["class_type"] == "SamplerCustomAdvanced"
                    for node in split_graph.values()
                )
            ),
            1,
        )
        self.assertEqual(
            split_graph[split_temporal_id]["inputs"],
            {
                "chunk_frames": 73,
                "temporal_overlap_frames": 22,
                "anchor_strength": 0.999,
                "motion_anchor_frames": "22",
                "identity_anchor_frames": 24,
            },
        )
        self.assertEqual(
            split_graph[split_spatial_id]["inputs"],
            {
                "tile_width": 512,
                "tile_height": 640,
                "overlap_ratio": 0.25,
                "fade_ratio": 0.5,
                "min_tile_size": 256,
                "seam_denoise": 0.75,
            },
        )
        self.assertEqual(
            split_graph[split_upscale_id]["inputs"],
            {
                "model": ["model", 0],
                "conditioning": ["target-conditioning", 0],
                "latent": _graph.Graph.out(split_combine_id),
                "noise": _graph.Graph.out(split_noise_id),
                "sampler": _graph.Graph.out(split_sampler_id),
                "sigmas": _graph.Graph.out(split_sigmas_id, 1),
                "cfg": 1.0,
                "temporal_split_param": _graph.Graph.out(split_temporal_id),
                "spatial_split_param": _graph.Graph.out(split_spatial_id),
                "seam_polish": "auto",
                "color_match": True,
            },
        )

    def test_sage_model_override(self):
        sage_graph = _graph.Graph()
        model_stack(
            sage_graph,
            fake.profile("speed").fl2va,
            fake,
            turbo_variant=_catalog.LIGHTX2V_4STEP_TURBO,
            sol_dense_steps=1,
            use_sage=True,
        )
        sage_nodes = [
            node
            for node in sage_graph.nodes.values()
            if node["class_type"] == _catalog.SAGE_ATTENTION_NODE
        ]
        self.assertEqual(len(sage_nodes), 1)
        self.assertEqual(sage_nodes[0]["inputs"]["sage_attention"], "auto")
        self.assertIs(sage_nodes[0]["inputs"]["allow_compile"], False)
        self.assertFalse(
            any(
                (
                    node["class_type"] == _catalog.SOL_ATTENTION_NODE
                    for node in sage_graph.nodes.values()
                )
            )
        )

    def test_sla_presets_and_model_override(self):
        self.assertEqual(
            _policy.resolve_sla_preset("Fast"),
            ("Fast", _catalog.SLA_PRESET_INPUTS["Fast"]),
        )
        self.assertEqual(
            _policy.resolve_sla_preset("Balance"),
            ("Balanced", _catalog.SLA_PRESET_INPUTS["Balanced"]),
        )
        self.assertEqual(
            _policy.resolve_sla_preset("Quality"),
            ("Quality", _catalog.SLA_PRESET_INPUTS["Quality"]),
        )
        sla_graph = _graph.Graph()
        model_stack(
            sla_graph,
            fake.profile("speed").fl2va,
            fake,
            turbo_variant=_catalog.LIGHTX2V_4STEP_TURBO,
            sol_dense_steps=1,
            use_sla=True,
            sla_preset="Quality",
        )
        sla_nodes = [
            node
            for node in sla_graph.nodes.values()
            if node["class_type"] == _catalog.SLA_ATTENTION_NODE
        ]
        self.assertEqual(len(sla_nodes), 1)
        self.assertEqual(
            sla_nodes[0]["inputs"],
            {
                "model": sla_nodes[0]["inputs"]["model"],
                "sparsity_ratio": 0.85,
                "block_size": "64",
                "min_seq_len": 8192,
                "dense_last_steps": 1,
                "protect_audio": True,
                "engine": "triton",
                "use_int8_qk": False,
                "tail_correction": False,
                "dense_steps": "0",
                "enabled": True,
            },
        )
        self.assertFalse(
            any(
                (
                    node["class_type"]
                    in {_catalog.SOL_ATTENTION_NODE, _catalog.SAGE_ATTENTION_NODE}
                    for node in sla_graph.nodes.values()
                )
            )
        )

    def test_spectrum_follows_sol_and_convrot(self):
        spectrum_graph = _graph.Graph()
        model_stack(
            spectrum_graph,
            fake.profile("quality").fl2va,
            fake,
            turbo_variant=_catalog.LIGHTX2V_4STEP_TURBO,
            use_sol=True,
            sol_dense_steps=1,
            cache_mode="Spectrum",
        )
        spectrum_id = next(
            node_id
            for node_id, node in spectrum_graph.nodes.items()
            if node["class_type"] == "SpectrumApplyMiniMaxH3"
        )
        spectrum_sol_id = next(
            node_id
            for node_id, node in spectrum_graph.nodes.items()
            if node["class_type"] == _catalog.SOL_ATTENTION_NODE
        )
        spectrum_chunk_id = next(
            node_id
            for node_id, node in spectrum_graph.nodes.items()
            if node["class_type"] == _catalog.CHUNK_FEED_FORWARD_NODE
        )
        spectrum_inputs = spectrum_graph.nodes[spectrum_id]["inputs"]
        self.assertEqual(spectrum_inputs["model"], [spectrum_chunk_id, 0])
        self.assertEqual(
            spectrum_graph.nodes[spectrum_chunk_id]["inputs"]["model"],
            [spectrum_sol_id, 0],
        )
        self.assertIs(spectrum_inputs["offline_smoothing_replay"], True)
        self.assertEqual(spectrum_inputs["audio_blend_weight"], 0.0)
        self.assertEqual(spectrum_inputs["offline_archive_storage"], "system_ram")
        self.assertEqual(spectrum_inputs["model_aware_mode"], "off")
        self.assertEqual(spectrum_inputs["model_aware_risk_threshold"], 0.65)

    def test_larry_runtime_loading_and_sampler(self):
        larry_graph = _graph.Graph()
        larry_model, _, larry_video_vae, larry_audio_vae = model_stack(
            larry_graph,
            fake.profile("speed").fl2va,
            fake,
            turbo_lora_name=fake.larry_turbo_lora,
            turbo_variant=_catalog.LARRY_TURBO,
            sol_dense_steps=1,
            cache_mode="Spectrum",
        )
        _h3_workflow.finish_sampling(
            larry_graph,
            model_ref=larry_model,
            conditioning_ref=["conditioning", 0],
            latent_ref=["latent", 0],
            video_vae_ref=larry_video_vae,
            audio_vae_ref=larry_audio_vae,
            seed=3,
            steps=6,
            scheduler="simple",
            turbo_variant=_catalog.LARRY_TURBO,
            filename_prefix="h3/larry_test",
        )
        larry_loader = next(
            node
            for node in larry_graph.nodes.values()
            if node["class_type"] == _catalog.LARRY_TURBO_LORA_NODE
        )
        self.assertEqual(larry_loader["inputs"]["strength"], 1.0)
        self.assertIs(larry_loader["inputs"]["low_vram"], False)
        larry_loader_id = next(
            node_id
            for node_id, node in larry_graph.nodes.items()
            if node["class_type"] == _catalog.LARRY_TURBO_LORA_NODE
        )
        larry_spectrum_id = next(
            node_id
            for node_id, node in larry_graph.nodes.items()
            if node["class_type"] == "SpectrumApplyMiniMaxH3"
        )
        self.assertEqual(
            larry_graph.nodes[larry_spectrum_id]["inputs"]["model"],
            [larry_loader_id, 0],
        )
        self.assertEqual(larry_model, [larry_spectrum_id, 0])
        self.assertEqual(
            larry_graph.nodes[larry_spectrum_id]["inputs"]["offline_archive_storage"],
            "system_ram",
        )
        self.assertNotIn(
            _catalog.FUSED_MODULATION_NODE,
            {node["class_type"] for node in larry_graph.nodes.values()},
        )
        self.assertNotIn(
            _catalog.FUSED_MODULATION_NODE,
            _h3_workflow.turbo_required_nodes(_catalog.LARRY_TURBO),
        )
        self.assertIn(
            _catalog.FUSED_MODULATION_NODE,
            _h3_workflow.turbo_required_nodes(_catalog.LIGHTX2V_4STEP_TURBO),
        )
        self.assertIn(
            _catalog.FUSED_MODULATION_NODE,
            _h3_workflow.turbo_required_nodes(_catalog.LIGHTX2V_8STEP_TURBO),
        )
        self.assertIn(
            _catalog.H3_SIGMA_SHIFT_NODE,
            _h3_workflow.turbo_required_nodes(
                _catalog.LIGHTX2V_4STEP_TURBO, fake.turbo_lora
            ),
        )
        self.assertNotIn(
            _catalog.H3_SIGMA_SHIFT_NODE,
            _h3_workflow.turbo_required_nodes(
                _catalog.LIGHTX2V_4STEP_TURBO, fake.turbo_ref_lora
            ),
        )
        self.assertIn(
            _catalog.H3_SIGMA_SHIFT_NODE,
            _h3_workflow.turbo_required_nodes(
                _catalog.LIGHTX2V_8STEP_TURBO, fake.turbo_8step_lora
            ),
        )
        self.assertEqual(
            _policy.turbo_sampler_name(_catalog.LIGHTX2V_4STEP_TURBO, fake.turbo_lora),
            "euler",
        )
        self.assertEqual(
            _policy.turbo_sampler_name(
                _catalog.LIGHTX2V_4STEP_TURBO, fake.turbo_ref_lora
            ),
            "euler",
        )
        self.assertEqual(
            _policy.turbo_sampler_name(
                _catalog.LIGHTX2V_8STEP_TURBO, fake.turbo_8step_lora
            ),
            "euler",
        )
        self.assertEqual(
            _policy.turbo_sampler_name(_catalog.LIGHTX2V_8STEP_TURBO, None),
            "res_multistep",
        )
        self.assertIs(_policy.turbo_uses_custom_nodes(_catalog.LARRY_TURBO), True)
        self.assertIs(
            _policy.turbo_uses_custom_nodes(_catalog.LIGHTX2V_4STEP_TURBO), False
        )
        self.assertIs(
            _policy.turbo_uses_custom_nodes(_catalog.LIGHTX2V_8STEP_TURBO), False
        )
        self.assertTrue(
            any(
                (
                    node["class_type"] == _catalog.LARRY_TURBO_SAMPLER_NODE
                    for node in larry_graph.nodes.values()
                )
            )
        )
        self.assertFalse(
            any(
                (
                    node["class_type"] == _catalog.CORE_SAMPLER_NODE
                    for node in larry_graph.nodes.values()
                )
            )
        )

        def turbo_route_graph(profile_name: str, variant: str) -> _graph.Graph:
            route_graph = _graph.Graph()
            profile = fake.profile(profile_name)
            model_stack(
                route_graph,
                profile.fl2va,
                fake,
                turbo_lora_name=fake.turbo_lora_for("Text to video", variant),
                turbo_variant=variant,
                turbo_strength=_policy.turbo_strength_for(variant),
                sol_dense_steps=1,
            )
            return route_graph

        for profile_name in ("speed", "quality", "original", "singularity"):
            larry_route = turbo_route_graph(profile_name, _catalog.LARRY_TURBO)
            larry_route_loader = next(
                node
                for node in larry_route.nodes.values()
                if node["class_type"] == _catalog.LARRY_TURBO_LORA_NODE
            )
            self.assertIs(larry_route_loader["inputs"]["low_vram"], False)

            for lightx_variant in (
                _catalog.LIGHTX2V_4STEP_TURBO,
                _catalog.LIGHTX2V_8STEP_TURBO,
            ):
                lightx_route = turbo_route_graph(profile_name, lightx_variant)
                route_classes = {
                    node["class_type"] for node in lightx_route.nodes.values()
                }
                self.assertIn(_catalog.LIGHTX2V_BYPASS_LORA_NODE, route_classes)
                self.assertNotIn(_catalog.CORE_LORA_LOADER_NODE, route_classes)
                self.assertIn(_catalog.FUSED_MODULATION_NODE, route_classes)
                if lightx_variant == _catalog.LIGHTX2V_8STEP_TURBO:
                    shift_node = next(
                        node
                        for node in lightx_route.nodes.values()
                        if node["class_type"] == _catalog.H3_SIGMA_SHIFT_NODE
                    )
                    self.assertEqual(shift_node["inputs"]["shift_video"], 6.0)
                    self.assertEqual(shift_node["inputs"]["shift_audio"], 3.0)
        self.assertIn(
            _catalog.LIGHTX2V_BYPASS_LORA_NODE,
            _h3_workflow.turbo_required_nodes(_catalog.LIGHTX2V_4STEP_TURBO),
        )
        self.assertNotIn(
            _catalog.CORE_LORA_LOADER_NODE,
            _h3_workflow.turbo_required_nodes(_catalog.LIGHTX2V_4STEP_TURBO),
        )

    def test_reference_turbo_uses_dedicated_adapter(self):
        ref_turbo_graph = _graph.Graph()
        model_stack(
            ref_turbo_graph,
            fake.profile("quality").ref2va,
            fake,
            turbo_lora_name=fake.turbo_ref_lora,
            turbo_variant=_catalog.LIGHTX2V_4STEP_TURBO,
            sol_dense_steps=1,
            cache_mode="EasyCache",
        )
        ref_unet = next(
            node
            for node in ref_turbo_graph.nodes.values()
            if node["class_type"] == "UNETLoader"
        )
        ref_lora = next(
            node
            for node in ref_turbo_graph.nodes.values()
            if node["class_type"] == _catalog.LIGHTX2V_BYPASS_LORA_NODE
        )
        self.assertEqual(
            ref_unet["inputs"]["unet_name"], fake.profile("quality").ref2va
        )
        self.assertEqual(ref_lora["inputs"]["lora_name"], fake.turbo_ref_lora)
        self.assertEqual(ref_lora["inputs"]["strength"], 1.0)
        self.assertNotIn(
            _catalog.H3_SIGMA_SHIFT_NODE,
            {node["class_type"] for node in ref_turbo_graph.nodes.values()},
        )
        ref_easycache = next(
            node
            for node in ref_turbo_graph.nodes.values()
            if node["class_type"] == "EasyCache"
        )
        self.assertEqual(ref_easycache["inputs"]["reuse_threshold"], 0.1)
