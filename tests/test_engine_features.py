"""Registered Create engines line up with navigation, layout tabs and saved settings."""

import dataclasses
import unittest

from h3_app.capabilities import ENGINES
from h3_ui.engines import FEATURES
from h3_ui.layout import AppViews
from h3_ui.persistence import EXTRA_FIELDS


class EngineFeatureTests(unittest.TestCase):
    def test_every_feature_is_a_navigable_engine_with_a_tab_and_saved_settings(self):
        engine_ids = {engine.id for engine in ENGINES}
        views = {field.name for field in dataclasses.fields(AppViews)}
        self.assertEqual(len({feature.id for feature in FEATURES}), len(FEATURES))
        for feature in FEATURES:
            with self.subTest(feature=feature.id):
                self.assertIn(feature.id, engine_ids)
                self.assertIn(feature.key, views)
                self.assertIn(feature.key, EXTRA_FIELDS)

    def test_features_build_and_bind_with_explicit_dependencies(self):
        from unittest.mock import Mock

        import gradio as gr
        from h3_ui.engines import EngineDeps

        models = Mock()
        models.render_ltx25_workflow_details.return_value = "Workflow"
        models.render_ltx25_official_model_inventory.return_value = "Inventory"
        deps = EngineDeps(runtime_config=Mock(), generation_services=Mock(), stage_file=Mock(), interrupt=Mock(),
                          models=lambda: models)
        with gr.Blocks() as demo:
            for feature in FEATURES:
                view = feature.build(gr.Group(), deps)
                handlers = feature.bind(view, deps)
                self.assertIs(handlers.deps, deps)
                for component in feature.library_inputs(view):
                    self.assertTrue(hasattr(component, "h3_library_picker"))
        api_names = {fn.api_name for fn in demo.fns.values()}
        for name in ("generate_flux3", "generate_moss_image", "generate_fibo2", "generate_ltx25_video",
                     "generate_qwen_image21", "generate_music3", "generate_yue2"):
            self.assertIn(name, api_names)


if __name__ == "__main__":
    unittest.main()
