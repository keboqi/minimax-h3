"""FL2VA adapter routing, raw-cache identity, settings, and lazy provisioning."""

import inspect
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import gradio_app as app
from h3_app.contracts import GenerationArguments, GENERATION_FIELDS
from h3_app.settings import GenerationRequest, resolve_settings
from h3_models import PRELOAD_MODEL_KEYS


class SemanticBridgeTests(unittest.TestCase):
    def test_ref2va_disables_bridge_without_losing_preference(self):
        request = GenerationRequest(mode="Reference media", semantic_bridge=True)
        plan = resolve_settings(request)
        self.assertFalse(plan.effective.semantic_bridge)
        self.assertTrue(plan.requested.semantic_bridge)
        self.assertIn("semantic_bridge_alpha", plan.inactive)
        self.assertTrue(any(a.field == "semantic_bridge" for a in plan.adjustments))
        self.assertTrue(
            resolve_settings(
                GenerationRequest(semantic_bridge=True)
            ).effective.semantic_bridge
        )

    def test_invalid_strength_is_rejected_before_downloads(self):
        for alpha in (-0.1, 1.1, float("nan"), float("inf"), "0.1", True):
            with self.subTest(alpha=alpha):
                plan = resolve_settings(
                    GenerationRequest(semantic_bridge=True, semantic_bridge_alpha=alpha)
                )
                self.assertTrue(
                    any("Semantic Bridge strength" in issue for issue in plan.issues)
                )

    def graph(self, enabled=False, alpha=0.1, upscale=False):
        signature = inspect.signature(app.build_fl2va_graph)
        args = {
            name: app.UI_DEFAULTS.get(name, 0)
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        args.update(
            prompt="A glass bottle",
            first_image=None,
            last_image=None,
            width=864,
            height=480,
            duration=5,
            steps=4,
            seed=7,
            models=SimpleNamespace(text_encoder="encoder.safetensors"),
            available_nodes={app.H3_SEMANTIC_BRIDGE_NODE},
            semantic_bridge=enabled,
            semantic_bridge_alpha=alpha,
            latent_upscale_model_name="upscaler.pth" if upscale else None,
        )
        with (
            patch.object(
                app,
                "add_model_stack",
                return_value=(["model", 0], ["clip", 0], ["vae", 0], ["audio", 0]),
            ),
            patch.object(app, "finish_sampling") as finish,
        ):
            graph = app.build_fl2va_graph(**args)
        return graph, finish.call_args.kwargs

    def test_bridge_runs_after_native_nodes_for_both_upscale_stages(self):
        graph, finish = self.graph(enabled=True, upscale=True)
        bridges = {
            key: n
            for key, n in graph.items()
            if n["class_type"] == app.H3_SEMANTIC_BRIDGE_NODE
        }
        self.assertEqual(len(bridges), 2)
        for name in ("conditioning_ref", "initial_conditioning_ref"):
            bridge = bridges[finish[name][0]]
            native = graph[bridge["inputs"]["conditioning"][0]]
            self.assertEqual(native["class_type"], "MiniMaxH3ImageToVideo")
            self.assertEqual(bridge["inputs"]["alpha"], 0.1)
        self.assertNotEqual(
            finish["conditioning_ref"], finish["initial_conditioning_ref"]
        )
        self.assertEqual(
            graph[finish["latent_ref"][0]]["class_type"], "MiniMaxH3ImageToVideo"
        )

    def test_single_stage_reuses_bridge_and_strength_keeps_encoder_cache_key(self):
        graph, finish = self.graph(enabled=True)
        self.assertEqual(finish["conditioning_ref"], finish["initial_conditioning_ref"])
        self.assertEqual(
            sum(n["class_type"] == app.H3_SEMANTIC_BRIDGE_NODE for n in graph.values()),
            1,
        )

        def cache_inputs(graph):
            return next(
                n["inputs"]
                for n in graph.values()
                if n["class_type"] == app.H3_CONDITIONING_CACHE_NODE
            )

        stronger, _ = self.graph(enabled=True, alpha=0.15)
        native, _ = self.graph()
        self.assertEqual(cache_inputs(graph), cache_inputs(stronger))
        self.assertEqual(cache_inputs(graph), cache_inputs(native))
        zero, zero_finish = self.graph(enabled=True, alpha=0)
        self.assertEqual(native, zero)
        self.assertEqual(
            zero_finish["conditioning_ref"], zero_finish["initial_conditioning_ref"]
        )

    def test_adapter_is_on_demand_and_download_is_skipped_when_ready(self):
        self.assertNotIn("semantic_bridge_v1", PRELOAD_MODEL_KEYS)
        with (
            patch.object(app, "stale_model_keys", return_value=[]),
            patch.object(app, "sync_models") as sync,
            patch.object(app, "model_file_is_ready", return_value=True),
        ):
            app.ensure_h3_semantic_bridge()
            sync.assert_not_called()
        with (
            patch.object(app, "stale_model_keys", return_value=["semantic_bridge_v1"]),
            patch.object(app, "sync_models") as sync,
            patch.object(app, "model_file_is_ready", return_value=True),
            patch.object(app, "resolve_hf_token", return_value=None),
        ):
            app.ensure_h3_semantic_bridge()
            self.assertEqual(
                sync.call_args.kwargs["model_keys"], ("semantic_bridge_v1",)
            )

    def test_ui_boundary_appends_optional_fields(self):
        parameters = list(inspect.signature(app.generate).parameters)
        self.assertEqual(parameters[:-1], list(GENERATION_FIELDS))
        legacy = GenerationArguments.from_positional(
            [None] * (len(GENERATION_FIELDS) - 5)
        )
        self.assertFalse(legacy.values["semantic_bridge"])
        self.assertEqual(legacy.values["semantic_bridge_alpha"], 0.1)
        self.assertFalse(app.UI_DEFAULTS["semantic_bridge"])
        self.assertEqual(app.UI_DEFAULTS["semantic_bridge_alpha"], 0.1)


if __name__ == "__main__":
    unittest.main()
