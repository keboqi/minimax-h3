"""Keep remaining standalone service regressions in normal discovery."""

import importlib
import unittest


class LegacyServiceTests(unittest.TestCase):
    def test_requirements(self):
        importlib.import_module("h3_requirements").selftest()

    def test_models(self):
        importlib.import_module("h3_models").selftest()

    def test_node_patches(self):
        importlib.import_module("h3_node_patches").selftest()

    def test_attention(self):
        importlib.import_module("h3_attention").selftest()

    def test_prompt_rewriter(self):
        importlib.import_module("h3_prompt_rewriter").selftest()
