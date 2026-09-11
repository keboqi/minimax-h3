from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import unittest
from unittest import mock

import gradio_app
from h3_ui.presentation import (
    backend_status_html,
    generation_readiness,
    mode_presentation,
    result_format_presentation,
)
from h3_ui.styles import H3_SETUP_CSS, H3_UI_CSS


class UiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event_loops = []
        event_loop_policy = asyncio.get_event_loop_policy()
        new_event_loop = event_loop_policy.new_event_loop

        def tracked_event_loop():
            loop = new_event_loop()
            cls.event_loops.append(loop)
            return loop

        with (
            mock.patch.object(
                gradio_app, "backend_status", return_value="Connected UI test"
            ),
            mock.patch.object(
                event_loop_policy,
                "new_event_loop",
                side_effect=tracked_event_loop,
            ),
        ):
            cls.demo = gradio_app.build_ui()
        cls.config = cls.demo.get_config_file()
        cls.components = {
            component["id"]: component for component in cls.config["components"]
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.demo.close()
        for loop in cls.event_loops:
            if not loop.is_closed() and not loop.is_running():
                loop.close()

    @classmethod
    def find_layout_node(cls, component_id: int):
        def visit(node):
            if node.get("id") == component_id:
                return node
            for child in node.get("children", []):
                result = visit(child)
                if result is not None:
                    return result
            return None

        return visit(cls.config["layout"])

    def test_fl2va_voice_inputs_live_under_frames_and_have_separate_api_fields(self):
        controls = {c.get("props", {}).get("label"): c for c in self.config["components"]}
        first_id = controls["First frame (auto resolution)"]["id"]
        voice_ids = [controls[f"FL2VA voice {i} · <Audio {i}>"]["id"] for i in range(1, 4)]
        ref_ids = [controls[f"Audio {i}"]["id"] for i in range(1, 4)]

        def descendants(node):
            return {node["id"]} | set().union(*(descendants(c) for c in node.get("children", [])))

        groups = [self.find_layout_node(c["id"]) for c in self.config["components"] if c["type"] == "group"]
        frame_group = min((descendants(g) for g in groups if g and first_id in descendants(g)), key=len)
        self.assertTrue(set(voice_ids) <= frame_group)
        self.assertFalse(set(ref_ids) & frame_group)
        advanced = next(d for d in self.config["dependencies"] if d.get("api_name") == "generate_video_advanced")
        self.assertEqual(advanced["inputs"][-3:], voice_ids)
        self.assertTrue(set(ref_ids) <= set(advanced["inputs"]))
        parameters = self.demo.get_api_info()["named_endpoints"]["/generate_video_advanced"]["parameters"]
        for parameter in parameters[-3:]:
            self.assertTrue(parameter["parameter_has_default"])
            self.assertIsNone(parameter["parameter_default"])


    def test_real_tabs_own_each_view(self) -> None:
        tabs = next(
            component
            for component in self.config["components"]
            if component["type"] == "tabs"
            and component.get("props", {}).get("elem_id") == "h3-main-tabs"
        )
        layout = self.find_layout_node(tabs["id"])
        self.assertIsNotNone(layout)
        tab_nodes = layout["children"]
        self.assertEqual(
            [self.components[node["id"]]["props"]["label"] for node in tab_nodes],
            ["MiniMax H3", "LTX 2.5", "MiniMax Music 3", "Gallery", "API"],
        )
        self.assertEqual(
            [self.components[node["children"][0]["id"]]["type"] for node in tab_nodes],
            ["row", "group", "group", "group", "group"],
        )

    def test_custom_server_mount_receives_ui_styles(self) -> None:
        with (
            mock.patch.object(gradio_app.httpx, "AsyncClient"),
            mock.patch.object(
                gradio_app.gr,
                "mount_gradio_app",
                return_value=mock.sentinel.mounted_app,
            ) as mount,
        ):
            result = gradio_app.build_server(mock.Mock(), [])
        self.assertIs(result, mock.sentinel.mounted_app)
        mounted_css = mount.call_args.kwargs["css"]
        self.assertEqual(mounted_css, H3_SETUP_CSS)
        self.assertNotIn(".gradio-container", mounted_css)

    def test_generation_settings_use_native_browser_state(self) -> None:
        state = next(
            component
            for component in self.config["components"]
            if component["type"] == "browserstate"
            and component.get("props", {}).get("storage_key")
            == "minimax-h3:settings:v3"
        )
        controls = {
            component.get("props", {}).get("label"): component
            for component in self.config["components"]
        }
        base_model = controls["Base model"]
        sampling_preset = controls["Generation preset"]

        restore = next(
            dependency
            for dependency in self.config["dependencies"]
            if dependency["inputs"] == [state["id"]]
            and base_model["id"] in dependency["outputs"]
        )
        self.assertIn(sampling_preset["id"], restore["outputs"])

        save = next(
            dependency
            for dependency in self.config["dependencies"]
            if dependency["outputs"] == [state["id"]]
            and base_model["id"] in dependency["inputs"]
        )
        self.assertIn(sampling_preset["id"], save["inputs"])
        # Presets save after their atomic transition, never from a change cascade.
        self.assertTrue(
            save["trigger_after"] is not None
            or all(
                event_name == "input" for _component_id, event_name in save["targets"]
            )
        )

    def test_settings_transition_owns_summary_and_bypasses_gpu_queue(self) -> None:
        controls = {
            c.get("props", {}).get("label"): c for c in self.config["components"]
        }
        summary = next(
            c
            for c in self.config["components"]
            if "h3-settings-summary" in c.get("props", {}).get("elem_classes", [])
        )
        dependency = next(
            d
            for d in self.config["dependencies"]
            if (controls["Base model"]["id"], "input") in d["targets"]
        )
        self.assertIn(summary["id"], dependency["outputs"])
        self.assertNotIn(
            (controls["Generation preset"]["id"], "change"), dependency["targets"]
        )
        fn = self.demo.fns[dependency["id"]]
        self.assertEqual(fn.concurrency_id, "h3-settings")

    def test_resolution_presets_apply_latent_alignment_atomically(self) -> None:
        self.assertEqual(
            gradio_app.resolution_choice_updates(
                "16:9 · 1920×1088", "fast", True, "Video"
            )[:2],
            (1920, 1088),
        )
        self.assertEqual(
            gradio_app.resolution_choice_updates(
                "16:9 · 1920×1088", "fast", False, "Video"
            )[:2],
            (1920, 1088),
        )

        controls = {
            component.get("props", {}).get("label"): component
            for component in self.config["components"]
        }
        preset = controls["1080p"]
        dependencies = [
            dependency
            for dependency in self.config["dependencies"]
            if preset["id"] in dependency["inputs"]
            and {"Width", "Height"}.issubset(
                {
                    self.components[output_id].get("props", {}).get("label")
                    for output_id in dependency["outputs"]
                }
            )
        ]
        self.assertEqual(len(dependencies), 1)

    def test_registered_resolution_callbacks_apply_each_preset(self) -> None:
        callbacks = [
            fn for fn in self.demo.fns.values()
            if inspect.isfunction(fn.fn)
            and "resolution_choice_updates" in fn.fn.__code__.co_names
        ]
        self.assertEqual(len(callbacks), 3)
        for callback in callbacks:
            for choice in callback.inputs[0].choices:
                name = choice[1] if isinstance(choice, (tuple, list)) else choice
                for latent_upscale in (False, True):
                    with self.subTest(name=name, latent_upscale=latent_upscale):
                        width, height, info = callback.fn(name, latent_upscale, "Video")
                        alignment = 64 if latent_upscale else 32
                        self.assertGreater(width, 0)
                        self.assertGreater(height, 0)
                        self.assertEqual(width % alignment, 0)
                        self.assertEqual(height % alignment, 0)
                        self.assertTrue(info)

    def test_first_frame_and_auto_megapixels_resolution_bindings(self) -> None:
        controls = {
            component.get("props", {}).get("label"): component
            for component in self.config["components"]
        }
        first_frame = controls["First frame (auto resolution)"]
        auto_cap = controls["Start-frame auto cap"]
        summary = next(
            c
            for c in self.config["components"]
            if "h3-settings-summary" in c.get("props", {}).get("elem_classes", [])
        )

        # 1. First frame upload has JS instant update, then server auto-resolution, then summary refresh
        upload_dep = next(
            d for d in self.config["dependencies"]
            if (first_frame["id"], "upload") in d.get("targets", [])
        )
        self.assertTrue(upload_dep.get("js"))
        h3_width_id = upload_dep["outputs"][0]
        h3_height_id = upload_dep["outputs"][1]
        self.assertEqual(self.components[h3_width_id].get("props", {}).get("label"), "Width")
        self.assertEqual(self.components[h3_height_id].get("props", {}).get("label"), "Height")

        server_auto_dep = next(
            d for d in self.config["dependencies"]
            if d.get("trigger_after") == upload_dep["id"]
        )
        self.assertIn(h3_width_id, server_auto_dep["outputs"])
        self.assertIn(h3_height_id, server_auto_dep["outputs"])

        upload_refresh_dep = next(
            d for d in self.config["dependencies"]
            if d.get("trigger_after") == server_auto_dep["id"]
        )
        self.assertIn(summary["id"], upload_refresh_dep["outputs"])

        # 2. First frame clear refreshes summary
        clear_dep = next(
            d for d in self.config["dependencies"]
            if (first_frame["id"], "clear") in d.get("targets", [])
        )
        self.assertIn(summary["id"], clear_dep["outputs"])

        # 3. Start-frame auto cap triggers on .change (so preset changes trigger auto resolution),
        # then refreshes summary
        auto_cap_change_dep = next(
            d for d in self.config["dependencies"]
            if (auto_cap["id"], "change") in d.get("targets", [])
            and h3_width_id in d.get("outputs", [])
        )
        self.assertIsNotNone(auto_cap_change_dep)
        auto_cap_refresh_dep = next(
            d for d in self.config["dependencies"]
            if d.get("trigger_after") == auto_cap_change_dep["id"]
        )
        self.assertIn(summary["id"], auto_cap_refresh_dep["outputs"])

    def test_auto_resolution_pipeline_updates_next_run_and_preset_change_applies_cap(self) -> None:
        import os
        import tempfile
        from PIL import Image

        temp_img = os.path.join(tempfile.gettempdir(), "test_ui_contract_768_1152.png")
        im = Image.new("RGB", (768, 1152), color=(255, 0, 0))
        im.save(temp_img)

        # 1. auto_resolution_from_start_frame calculates 768x1152
        w, h, info = gradio_app.auto_resolution_from_start_frame(
            temp_img, 1408, 768, "Video", False, "1 MP"
        )
        self.assertEqual((w, h), (768, 1152))

        # Find controller and run refresh with the new resolution
        refresh_fn = next(
            fn for fn in self.demo.fns.values()
            if hasattr(fn.fn, "__self__") and type(fn.fn.__self__).__name__ == "SettingsController"
        )
        controller = refresh_fn.fn.__self__

        memory = {"active": "Turbo", "modes": {}, "values": {name: getattr(controller.components[name], "value", None) for name in controller.names}}
        memory["values"]["width"] = 1408
        memory["values"]["height"] = 768

        # Stale inputs produce 1408x768
        input_values_stale = [
            1408 if controller.ids.get(getattr(comp, "_id", None)) == "width"
            else 768 if controller.ids.get(getattr(comp, "_id", None)) == "height"
            else getattr(comp, "value", None)
            for comp in controller.inputs
        ]
        out_stale = controller.refresh(memory, *input_values_stale)
        summary_stale = out_stale[-3]
        self.assertIn("1408×768", summary_stale)
        self.assertNotIn("768×1152", summary_stale)

        # Updated inputs (from upload/change pipeline) produce 768x1152
        input_values_new = [
            w if controller.ids.get(getattr(comp, "_id", None)) == "width"
            else h if controller.ids.get(getattr(comp, "_id", None)) == "height"
            else temp_img if controller.ids.get(getattr(comp, "_id", None)) == "first"
            else getattr(comp, "value", None)
            for comp in controller.inputs
        ]
        out_new = controller.refresh(memory, *input_values_new)
        summary_new = out_new[-3]
        self.assertIn("768×1152", summary_new)
        self.assertNotIn("1408×768", summary_new)

        # 2. When preset changes auto_megapixels cap (e.g. from 1 MP to 2 MP on a large image),
        # auto_resolution_from_start_frame applies the new cap
        large_img = os.path.join(tempfile.gettempdir(), "test_ui_contract_large.png")
        im_large = Image.new("RGB", (3000, 2000), color=(0, 255, 0))
        im_large.save(large_img)

        w_1mp, h_1mp, _ = gradio_app.auto_resolution_from_start_frame(
            large_img, 1408, 768, "Video", False, "1 MP"
        )
        w_2mp, h_2mp, _ = gradio_app.auto_resolution_from_start_frame(
            large_img, w_1mp, h_1mp, "Video", False, "2 MP"
        )
        self.assertLess(w_1mp * h_1mp, 1_000_000)
        self.assertGreater(w_2mp * h_2mp, w_1mp * h_1mp)
        self.assertLess(w_2mp * h_2mp, 2_000_000)



    def test_tensorrt_vae_defaults_on_and_compiles_only_when_needed(self) -> None:
        trt_vae = next(
            component
            for component in self.config["components"]
            if component.get("props", {}).get("label")
            == "Experimental TensorRT video VAE"
        )
        self.assertTrue(trt_vae["props"]["value"])

        models = mock.sentinel.models
        progress = mock.sentinel.progress
        with (
            mock.patch.object(gradio_app.model_service, "ensure_trt_video_vae") as provision,
            mock.patch.object(
                gradio_app.model_service, "trt_vae_engine_is_current", return_value=False
            ),
            mock.patch.object(gradio_app.model_service, "_build_trt_video_vae_engine") as build,
        ):
            self.assertTrue(
                gradio_app.ensure_trt_video_vae_engine(models, progress=progress)
            )
        provision.assert_has_calls(
            [mock.call(models, require_engine=False, runtime=gradio_app._runtime_config()), mock.call(models, runtime=gradio_app._runtime_config())]
        )
        build.assert_called_once_with(models, progress, runtime=gradio_app._runtime_config(), release_backend=gradio_app.unload_comfy_models)

        with (
            mock.patch.object(gradio_app.model_service, "ensure_trt_video_vae") as provision,
            mock.patch.object(
                gradio_app.model_service, "trt_vae_engine_is_current", return_value=True
            ),
            mock.patch.object(gradio_app.model_service, "_build_trt_video_vae_engine") as build,
        ):
            self.assertFalse(
                gradio_app.ensure_trt_video_vae_engine(models, progress=progress)
            )
            self.assertTrue(
                gradio_app.ensure_trt_video_vae_engine(
                    models, force=True, progress=progress
                )
            )
        provision.assert_has_calls(
            [
                mock.call(models, require_engine=False, runtime=gradio_app._runtime_config()),
                mock.call(models, require_engine=False, runtime=gradio_app._runtime_config()),
                mock.call(models, runtime=gradio_app._runtime_config()),
            ]
        )
        build.assert_called_once_with(models, progress, runtime=gradio_app._runtime_config(), release_backend=gradio_app.unload_comfy_models)

        with (
            mock.patch.object(gradio_app, "load_model_config", return_value=models),
            mock.patch.object(
                gradio_app, "ensure_trt_video_vae_engine", return_value=True
            ) as ensure_engine,
        ):
            status = gradio_app.compile_trt_video_vae(progress=progress)
            self.assertIn("compiled and ready to use", status)
            ensure_engine.assert_called_once_with(models, force=True, progress=progress)

    def test_tensorrt_vae_currency_checks_fingerprint_and_loadability(self) -> None:
        models = mock.sentinel.models
        with (
            mock.patch.object(
                gradio_app.model_service,
                "trt_vae_decoder_paths",
                return_value=(
                    Path("/fake/decoder.onnx"),
                    Path("/fake/decoder.engine"),
                    Path("/fake/marker"),
                ),
            ),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(
                gradio_app.model_service, "trt_vae_runtime_fingerprint", return_value="v4:trt_10.9:sm_89"
            ),
        ):
            # Mismatched marker (e.g. from an older TRT version 243)
            with mock.patch.object(Path, "read_text", return_value="v4:trt_10.8:sm_89"):
                self.assertFalse(gradio_app.trt_vae_engine_is_current(models))

            # Matching marker but deserialization fails (returns None)
            with (
                mock.patch.object(Path, "read_text", return_value="v4:trt_10.9:sm_89"),
                mock.patch.object(gradio_app.model_service, "is_trt_engine_loadable", return_value=False),
            ):
                self.assertFalse(gradio_app.trt_vae_engine_is_current(models))

            # Matching marker and loadable engine
            with (
                mock.patch.object(Path, "read_text", return_value="v4:trt_10.9:sm_89"),
                mock.patch.object(gradio_app.model_service, "is_trt_engine_loadable", return_value=True),
            ):
                self.assertTrue(gradio_app.trt_vae_engine_is_current(models))

    def test_gpu_actions_share_the_application_queue(self):
        expected = {"compile_trt_video_vae", "generate_for_ui", "generate_ltx25", "generate_music3", "enhance_h3_prompt", "unload_all_models"}
        found = set()
        for event in self.demo.fns.values():
            name = getattr(event.fn, "__name__", "")
            if name in expected:
                found.add(name)
                self.assertEqual(event.concurrency_id, "h3-gpu", name)
                self.assertEqual(event.concurrency_limit, 1, name)
        self.assertEqual(found, expected)

    def test_h3_progressive_section_order(self) -> None:
        tabs = next(
            component
            for component in self.config["components"]
            if component.get("props", {}).get("elem_id") == "h3-main-tabs"
        )
        tabs_layout = self.find_layout_node(tabs["id"])
        h3_row = tabs_layout["children"][0]["children"][0]
        right_column = h3_row["children"][1]
        direct_components = [
            self.components[node["id"]] for node in right_column["children"]
        ]
        labels = [
            component.get("props", {}).get("label") for component in direct_components
        ]
        essentials = labels.index("Output essentials")
        performance = labels.index("Performance & sampling (advanced)")
        finishing = labels.index("Upscaling & finishing (advanced)")
        summary = next(
            index
            for index, component in enumerate(direct_components)
            if "h3-settings-summary"
            in component.get("props", {}).get("elem_classes", [])
        )
        action = next(
            index
            for index, component in enumerate(direct_components)
            if "h3-action-dock" in component.get("props", {}).get("elem_classes", [])
        )
        self.assertLess(essentials, performance)
        self.assertLess(performance, finishing)
        self.assertLess(finishing, summary)
        self.assertLess(summary, action)
        self.assertFalse(
            direct_components[essentials].get("props", {}).get("open", True)
        )
        html_values = [
            component.get("props", {}).get("value", "")
            for component in direct_components
            if component.get("type") == "html"
        ]
        self.assertFalse(any("Review & run" in val for val in html_values))

    def test_presentation_state_is_pure_and_semantic(self) -> None:
        self.assertTrue(mode_presentation("First / last frame").show_frames)
        self.assertTrue(mode_presentation("Reference media").show_references)
        audio = result_format_presentation("audio")
        self.assertTrue(audio.is_audio)
        self.assertEqual(audio.action_label, "Generate audio")
        blocked = generation_readiness("Text to video", "", None, None)
        self.assertFalse(blocked.ready)
        self.assertIn('role="alert"', blocked.html)
        self.assertIn("&lt;offline&gt;", backend_status_html("<offline>"))

    def test_mmh3_split_upscale_controls_are_explicit_and_conditional(self) -> None:
        labels = {
            component.get("props", {}).get("label")
            for component in self.config["components"]
        }
        for label in (
            "High-resolution refinement method",
            "Tile width (pixels)",
            "Tile height (pixels)",
            "Spatial overlap",
            "Overlap fade",
            "Temporal chunk length (frames)",
            "Temporal overlap (frames)",
            "Seam denoise cap",
            "Seam polish",
        ):
            self.assertIn(label, labels)
        method = next(
            component
            for component in self.config["components"]
            if component.get("props", {}).get("label")
            == "High-resolution refinement method"
        )
        choices = method["props"]["choices"]
        self.assertIn(
            (
                gradio_app.H3_LATENT_UPSCALE_SPLIT,
                gradio_app.H3_LATENT_UPSCALE_SPLIT,
            ),
            choices,
        )
        self.assertFalse(
            gradio_app.latent_upscale_method_layout_update(
                gradio_app.H3_LATENT_UPSCALE_STANDARD
            )["visible"]
        )
        self.assertTrue(
            gradio_app.latent_upscale_method_layout_update(
                gradio_app.H3_LATENT_UPSCALE_SPLIT
            )["visible"]
        )
        advanced = next(
            dependency
            for dependency in self.config["dependencies"]
            if dependency.get("api_name") == "generate_video_advanced"
        )
        self.assertEqual(
            len(advanced["inputs"]),
            len(inspect.signature(gradio_app.generate).parameters),
        )

    def test_settings_summary_is_compact_disclosure_with_escaped_values(self) -> None:
        summary = gradio_app.compact_settings_summary(
            "Text to video",
            "Original <unsafe>",
            "BF16",
            True,
            True,
            False,
            "Turbo",
            gradio_app.LIGHTX2V_8STEP_TURBO,
            5,
            1344,
            768,
            8,
            "simple",
            "SLA",
            "Quality",
            "Spectrum",
            True,
            "Quality (FP32)",
            2,
            "None",
            "unused",
            "unused",
            False,
            False,
            5,
        )
        self.assertIn('<details class="h3-setup-disclosure">', summary)
        self.assertNotIn("<details open", summary)
        self.assertIn("Execution details", summary)
        self.assertIn('class="h3-setup-metrics"', summary)
        self.assertIn('class="h3-setup-context"', summary)
        self.assertIn("Next run", summary)
        self.assertNotIn("Ready to generate", summary)
        self.assertIn("Acceleration: Off", summary)
        self.assertIn("LightX2V v1.0 / 8-step 768p", summary)
        self.assertIn("Original &lt;unsafe&gt;", summary)
        self.assertNotIn("Original <unsafe>", summary)
        split_summary = gradio_app.compact_settings_summary(
            "Text to video",
            "Original",
            "BF16",
            True,
            True,
            False,
            "Normal",
            gradio_app.LIGHTX2V_8STEP_TURBO,
            5,
            1024,
            1024,
            20,
            "beta",
            "SLA",
            "Quality",
            "Off",
            True,
            "Balanced (BF16)",
            2,
            "None",
            "unused",
            "unused",
            False,
            False,
            5,
            latent_upscale_method=gradio_app.H3_LATENT_UPSCALE_SPLIT,
            latent_split_tile_width=512,
            latent_split_tile_height=640,
            latent_split_chunk_frames=73,
            latent_split_temporal_overlap_frames=22,
            latent_split_seam_polish="auto",
        )
        self.assertIn("MMH3 Split Upscale (experimental)", split_summary)
        self.assertIn("512×640px tiles", split_summary)
        self.assertIn("73f chunks+22f", split_summary)
        self.assertIn("polish auto", split_summary)

    def test_scoped_setup_css_contract(self) -> None:
        for rule in (
            "@container (max-width: 520px)",
            ".h3-settings-summary",
            ".h3-setup-disclosure",
            ".h3-setup-detail-grid",
            ".h3-setup-metrics",
            ".h3-setup-context",
            "grid-template-columns: repeat(3, minmax(0, 1fr))",
        ):
            self.assertIn(rule, H3_SETUP_CSS)
        self.assertNotIn(".gradio-container", H3_SETUP_CSS)


if __name__ == "__main__":
    unittest.main()
