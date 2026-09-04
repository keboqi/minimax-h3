from __future__ import annotations

import asyncio
import inspect
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
        sampling_preset = controls["Sampling preset"]

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

    def test_resolution_presets_apply_latent_alignment_atomically(self) -> None:
        self.assertEqual(
            gradio_app.resolution_choice_updates(
                "16:9 · 864×480", "fast", True, "Video"
            )[:2],
            (896, 512),
        )
        self.assertEqual(
            gradio_app.resolution_choice_updates(
                "16:9 · 864×480", "fast", False, "Video"
            )[:2],
            (864, 480),
        )

        controls = {
            component.get("props", {}).get("label"): component
            for component in self.config["components"]
        }
        preset = controls["Recommended size"]
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
            mock.patch.object(gradio_app, "ensure_trt_video_vae") as provision,
            mock.patch.object(
                gradio_app, "trt_vae_engine_is_current", return_value=False
            ),
            mock.patch.object(gradio_app, "_build_trt_video_vae_engine") as build,
        ):
            self.assertTrue(
                gradio_app.ensure_trt_video_vae_engine(models, progress=progress)
            )
        provision.assert_has_calls(
            [mock.call(models, require_engine=False), mock.call(models)]
        )
        build.assert_called_once_with(models, progress)

        with (
            mock.patch.object(gradio_app, "ensure_trt_video_vae") as provision,
            mock.patch.object(
                gradio_app, "trt_vae_engine_is_current", return_value=True
            ),
            mock.patch.object(gradio_app, "_build_trt_video_vae_engine") as build,
        ):
            self.assertFalse(
                gradio_app.ensure_trt_video_vae_engine(models, progress=progress)
            )
        provision.assert_called_once_with(models, require_engine=False)
        build.assert_not_called()

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
        self.assertFalse(direct_components[essentials].get("props", {}).get("open", True))
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
        self.assertIn("View all settings", summary)
        self.assertIn('class="h3-setup-metrics"', summary)
        self.assertIn('class="h3-setup-pills"', summary)
        self.assertIn("Ready to generate", summary)
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
            "prefers-reduced-motion",
            "@container (max-width: 430px)",
            ".h3-settings-summary",
            ".h3-setup-disclosure",
            ".h3-setup-detail-grid",
            ".h3-setup-metric-icon",
            ".h3-setup-pill",
            "linear-gradient(118deg",
        ):
            self.assertIn(rule, H3_SETUP_CSS)
        self.assertNotIn(".gradio-container", H3_SETUP_CSS)


if __name__ == "__main__":
    unittest.main()
