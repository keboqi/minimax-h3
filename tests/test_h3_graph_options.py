"""Every H3 graph option reaches both builders from one place."""

import dataclasses
import inspect
import unittest
from unittest.mock import MagicMock, patch

from h3_app.generation.construct_graph import graph_options
from h3_app import model_service
from h3_app.workflows import h3
from h3_models import MODEL_SPECS, PRELOAD_MODEL_KEYS, QWEN35_H3_ADAPTER_KEYS, QWEN35_H3_ENCODERS
from tests.workflow_cases import H3_DEFAULTS, MODELS


class H3GraphOptionTests(unittest.TestCase):
    def test_qwen35_choices_prepare_and_apply_the_h3_adapters(self):
        for label, key in QWEN35_H3_ENCODERS.items():
            with self.subTest(label=label), patch.object(model_service, "ensure_models") as ensure:
                filename, offload = model_service.ensure_h3_text_encoder(MODELS, label, runtime=MagicMock())
                self.assertEqual(filename, MODEL_SPECS[key].local_name)
                self.assertFalse(offload)
                keys = ensure.call_args.args[0]
                self.assertEqual(keys, (key, *QWEN35_H3_ADAPTER_KEYS))
                self.assertTrue(set(keys).isdisjoint(PRELOAD_MODEL_KEYS))
                graph = h3.build_fl2va_graph(**(H3_DEFAULTS | {"text_encoder_name": filename}))
                cache = next(n for n in graph.values() if n["class_type"] == "H3ConditioningCache")
                adapter = graph[cache["inputs"]["clip"][0]]
                self.assertEqual(adapter["class_type"], "Qwen35H3Apply")
                self.assertTrue(adapter["inputs"]["use_lora"])
                self.assertEqual(graph[adapter["inputs"]["clip"][0]]["inputs"]["clip_name"], filename)
                self.assertIn("Qwen35H3Apply", h3.required_nodes_for("Text to video", False, "Off",
                                                                    text_encoder_name=filename))

    def test_construct_graph_supplies_every_shared_option(self):
        options = {field.name for field in dataclasses.fields(h3.H3GraphOptions)}
        supplied = set(graph_options(MagicMock(), MagicMock()))
        self.assertEqual(supplied, options - h3.FL2VA_ONLY_OPTIONS)

    def test_builder_signatures_list_every_option_by_name(self):
        options = {field.name for field in dataclasses.fields(h3.H3GraphOptions)}
        fl2va = set(inspect.signature(h3.build_fl2va_graph).parameters)
        ref2va = set(inspect.signature(h3.build_ref2va_graph).parameters)
        self.assertLessEqual(options, fl2va)
        self.assertLessEqual(options - h3.FL2VA_ONLY_OPTIONS, ref2va)
        self.assertFalse(h3.FL2VA_ONLY_OPTIONS & ref2va)
        with self.assertRaises(TypeError):
            h3.build_ref2va_graph(prompt="x", reference_images=[], reference_videos=[], reference_audios=[],
                                  ref_image_size="match", semantic_bridge=True)


if __name__ == "__main__":
    unittest.main()
