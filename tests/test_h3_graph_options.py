"""Every H3 graph option reaches both builders from one place."""

import dataclasses
import inspect
import unittest
from unittest.mock import MagicMock

from h3_app.generation.construct_graph import graph_options
from h3_app.workflows import h3


class H3GraphOptionTests(unittest.TestCase):
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
