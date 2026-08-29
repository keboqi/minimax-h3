from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import gradio_app
from h3_ui.presentation import (
    backend_status_html,
    generation_readiness,
    mode_presentation,
    result_format_presentation,
)
from h3_ui.styles import H3_UI_CSS


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

    def test_responsive_and_accessibility_css_contract(self) -> None:
        for rule in (
            "button:focus-visible",
            "@media (max-width: 900px)",
            "@media (max-width: 600px)",
            "prefers-reduced-motion",
            "bottom: .35rem",
            ".h3-advanced-block",
            ".h3-setup-disclosure",
            ".h3-setup-detail-grid",
            ".h3-setup-metric-icon",
            ".h3-setup-pill",
            "linear-gradient(118deg",
        ):
            self.assertIn(rule, H3_UI_CSS)


if __name__ == "__main__":
    unittest.main()
