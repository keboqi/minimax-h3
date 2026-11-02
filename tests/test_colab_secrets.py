"""Verify runtime secret loading without a Colab account or real credentials."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class ColabSecretsTests(unittest.TestCase):
    names = ("HF_TOKEN", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")

    def setUp(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "minimax_h3_colab.ipynb").read_text(encoding="utf-8"))
        self.cell = next(cell for cell in notebook["cells"]
                         if "def configure_key(" in "".join(cell["source"]))
        self.source = "".join(self.cell["source"])
        self.errors = [type(name, (Exception,), {}) for name in
                       ("SecretNotFoundError", "NotebookAccessError", "TimeoutException")]
        self.get = Mock()
        google, colab = ModuleType("google"), ModuleType("google.colab")
        colab.userdata = SimpleNamespace(get=self.get, **dict(zip(
            ("SecretNotFoundError", "NotebookAccessError", "TimeoutException"), self.errors)))
        google.colab = colab
        self.modules = {"google": google, "google.colab": colab}

    def run_cell(self):
        output = io.StringIO()
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(output):
            exec(compile(self.source, "<Colab secrets>", "exec"), {})
        return output.getvalue()

    def test_secrets_are_loaded_without_printing_values(self):
        values = {name: f"private-value-{index}" for index, name in enumerate(self.names)}
        self.get.side_effect = lambda name: f"  {values[name]}  "
        with patch.dict(os.environ, {}, clear=True):
            output = self.run_cell()
            self.assertEqual(dict(os.environ), values)
        self.assertEqual([call.args[0] for call in self.get.call_args_list], list(self.names))
        for value in values.values():
            self.assertNotIn(value, output)

    def test_unavailable_secrets_clear_stale_values_and_allow_setup(self):
        for error in self.errors:
            with self.subTest(error=error.__name__):
                self.get.side_effect = error("private exception detail")
                with patch.dict(os.environ, {name: "stale-private-value" for name in self.names}, clear=True):
                    output = self.run_cell()
                    self.assertTrue(all(name not in os.environ for name in self.names))
                self.assertIn("enable Notebook access", output)
                self.assertIn("official OpenAI API is used", output)
                self.assertNotIn("private exception detail", output)
                self.assertNotIn("stale-private-value", output)

    def test_optional_missing_secrets_do_not_block_available_token(self):
        self.get.side_effect = ["hf-private-value", None, "", "  "]
        with patch.dict(os.environ, {}, clear=True):
            output = self.run_cell()
            self.assertEqual(dict(os.environ), {"HF_TOKEN": "hf-private-value"})
        self.assertIn("Hugging Face token configured", output)
        self.assertIn("official OpenAI API is used", output)

    def test_no_credential_form_fields_or_saved_output(self):
        self.assertNotIn("@param", self.source)
        self.assertEqual(self.cell.get("outputs"), [])
        self.assertIsNone(self.cell.get("execution_count"))


if __name__ == "__main__":
    unittest.main()
