"""The 500K overlay must retain direct still decoding on native H3 VAEs."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class SingleFrameVAETests(unittest.TestCase):
    def test_overlay_sets_direct_decode_metadata_without_mutating_base_metadata(self):
        source = (Path(__file__).resolve().parents[1]
                  / "custom_nodes/H3Acceleration/__init__.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.ClassDef) and n.name == "H3SingleFrameVAELoader")
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                     and n.name == "load_vae"]
        for metadata in (None, {"architecture": "h3", "h3_t1_direct": "false"}):
            with self.subTest(metadata=metadata):
                original = dict(metadata) if metadata is not None else None
                base = {"decoder.weight": SimpleNamespace(shape=(2, 2)),
                        "encoder.weight": object()}
                decoder = {"decoder.weight": SimpleNamespace(shape=(2, 2))}
                vae = Mock()
                comfy = SimpleNamespace(
                    utils=SimpleNamespace(load_torch_file=Mock(
                        side_effect=[(base, metadata), decoder])),
                    sd=SimpleNamespace(VAE=vae))
                namespace = {"comfy": comfy, "logging": logging}
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "<single-frame loader>", "exec"), namespace)
                loader = namespace["H3SingleFrameVAELoader"]()
                loader._vae_path = lambda name: name
                loader._convert_diffusers_decoder = lambda value: value
                loader.EXPECTED_DECODER_TENSORS = 1
                loader.DECODER_PREFIXES = ("decoder.",)
                self.assertEqual(loader.load_vae("base", "500k"), (vae.return_value,))
                self.assertEqual(vae.call_args.kwargs["metadata"], {
                    **(original or {}), "h3_t1_direct": "true", "h3_t1_output_slice": "3"})
                self.assertEqual(metadata, original)
                self.assertIs(vae.call_args.kwargs["sd"]["decoder.weight"],
                              decoder["decoder.weight"])


if __name__ == "__main__":
    unittest.main()
