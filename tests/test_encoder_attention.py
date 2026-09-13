"""Qwen dispatch isolation, cache invalidation, and end-to-end graph settings."""
import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio_app as app
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.generation.requests import H3Request
from h3_app.settings import GenerationRequest, resolve_settings
from h3_ui.persistence import restore_preferences


def load_encoder_nodes():
    # Load the real node helpers without GPU/ComfyUI/AV imports, as in the
    # numerical node tests. Mock only the external attention selector modules.
    path = Path(__file__).resolve().parents[1] / "custom_nodes/H3Acceleration/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_H3_ENCODER_SMALL_INPUT" for t in n.targets))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == "H3ConditioningCache")
    modules = {
        name: SimpleNamespace(optimized_attention_for_device=lambda device, mask=False, small_input=False: (device, mask, small_input))
        for name in ("comfy.text_encoders.llama", "comfy.text_encoders.qwen35")
    }
    namespace = dict(OrderedDict=OrderedDict, contextmanager=contextmanager,
                     ContextVar=ContextVar, wraps=wraps, logging=logging,
                     threading=threading,
                     importlib=SimpleNamespace(import_module=modules.__getitem__))
    exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(path), "exec"), namespace)
    return namespace, modules


class EncoderRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.ns, self.modules = load_encoder_nodes()

    def test_both_selectors_preserve_masks_and_restore_on_error(self):
        context = self.ns["_h3_encoder_attention"]
        with self.assertRaisesRegex(RuntimeError, "encode failed"):
            with context(False):
                for module in self.modules.values():
                    self.assertEqual(module.optimized_attention_for_device("cuda", mask=True, small_input=True), ("cuda", True, False))
                with context(True):
                    self.assertTrue(self.modules["comfy.text_encoders.llama"].optimized_attention_for_device("cuda")[-1])
                self.assertFalse(self.modules["comfy.text_encoders.llama"].optimized_attention_for_device("cuda", small_input=True)[-1])
                raise RuntimeError("encode failed")
        for module in self.modules.values():
            self.assertEqual(module.optimized_attention_for_device("cpu", mask=True, small_input=True), ("cpu", True, True))
            self.assertFalse(module.optimized_attention_for_device("cuda", small_input=False)[-1])

    def test_other_threads_keep_original_routing_and_install_is_idempotent(self):
        with self.ns["_h3_encoder_attention"](False):
            selector = self.modules["comfy.text_encoders.llama"].optimized_attention_for_device
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertTrue(pool.submit(selector, "cuda", False, True).result()[-1])
            self.assertFalse(selector("cuda", small_input=True)[-1])
        with self.ns["_h3_encoder_attention"](True):
            self.assertIs(selector, self.modules["comfy.text_encoders.llama"].optimized_attention_for_device)

    def test_clone_reuse_and_every_attention_transition_reencodes(self):
        node = self.ns["H3ConditioningCache"]()
        routes = []
        def encode(tokens):
            routes.append(tuple(m.optimized_attention_for_device("cuda", mask=True, small_input=True)[-1] for m in self.modules.values()))
            return object()
        clip = SimpleNamespace(encode_from_tokens_scheduled=encode)
        clip.clone = lambda: clip
        revisions = []
        for small_input in (True, False, True):
            revisions.append(node.IS_CHANGED(clip, "same-inputs", small_input))
            proxy = node.wrap(clip, "same-inputs", small_input)[0].clone()
            first = proxy.encode_from_tokens_scheduled("tokens")
            self.assertIs(first, proxy.encode_from_tokens_scheduled("tokens"))
            self.assertEqual(node.IS_CHANGED(clip, "same-inputs", small_input), revisions[-1])
        self.assertEqual(routes, [(True, True), (False, False), (True, True)])
        self.assertEqual(len(set(revisions)), 3)
        self.assertTrue(node.INPUT_TYPES()["optional"]["encoder_small_input"][1]["default"])
        self.assertIsNone(self.ns["_H3_ENCODER_SMALL_INPUT"].get())

    def test_failed_encoding_is_not_cached(self):
        cache = self.ns["_H3_CONDITIONING_REUSE_CACHE"]
        encode = Mock(side_effect=[RuntimeError("failure"), "ok"])
        with self.assertRaises(RuntimeError):
            cache.encode("input", encode, False)
        self.assertEqual(cache.encode("input", encode, False), "ok")
        self.assertEqual(encode.call_count, 2)

    def test_inflight_old_route_does_not_repopulate_cache(self):
        cache = self.ns["_H3_CONDITIONING_REUSE_CACHE"]
        def encode_old():
            cache.select_attention(False)
            return "old"
        self.assertEqual(cache.encode("input", encode_old, True), "old")
        fresh = Mock(return_value="new")
        self.assertEqual(cache.encode("input", fresh, False), "new")
        fresh.assert_called_once()


class EncoderOptionTests(unittest.TestCase):
    def graph(self, family, small_input, seed=7):
        fn = getattr(app, "build_" + family + "_graph")
        args = {name: app.UI_DEFAULTS.get(name, 0) for name, p in inspect.signature(fn).parameters.items() if p.default is inspect.Parameter.empty}
        args.update(prompt="A quiet lake", width=864, height=480, duration=5,
                    steps=4, seed=seed, encoder_small_input=small_input,
                    models=SimpleNamespace(text_encoder="encoder.safetensors"),
                    available_nodes=set(), latent_upscale_model_name="upscaler.pth")
        if family == "fl2va":
            args.update(first_image="start.png", last_image=None)
        else:
            args.update(reference_images=["ref.png"], reference_videos=[], reference_audios=[], ref_image_size="match")
        with patch.object(app, "stage_file", side_effect=lambda path, *a, **kw: path), patch.object(app.h3_workflow, "add_model_stack", return_value=(["model", 0], ["clip", 0], ["vae", 0], ["audio", 0])), patch.object(app.h3_workflow, "finish_sampling"):
            return fn(**args)

    def test_both_workflows_pass_option_and_change_encoder_cache_identity(self):
        for family in ("fl2va", "ref2va"):
            keys = []
            for small_input, seed in ((True, 7), (False, 7), (False, 8)):
                graph = self.graph(family, small_input, seed)
                cache_id, node = next((i, n) for i, n in graph.items() if n["class_type"] == app.H3_CONDITIONING_CACHE_NODE)
                self.assertIs(node["inputs"]["encoder_small_input"], small_input)
                keys.append(node["inputs"]["cache_key"])
                stages = [n for n in graph.values() if n["class_type"] in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")]
                self.assertEqual(len(stages), 2)
                self.assertTrue(all(n["inputs"]["clip"] == [cache_id, 0] for n in stages))
            self.assertNotEqual(keys[0], keys[1])
            self.assertEqual(keys[1], keys[2])

    def test_default_legacy_api_and_saved_false(self):
        self.assertTrue(app.UI_DEFAULTS["encoder_small_input"])
        for boundary in ("semantic_bridge", "fl2va_audio_1", "encoder_small_input"):
            values = GenerationArguments.from_positional([None] * GENERATION_FIELDS.index(boundary)).values
            self.assertTrue(values["encoder_small_input"])
        self.assertFalse(resolve_settings(GenerationRequest.from_values({"encoder_small_input": False})).effective.sampling.encoder_small_input)
        values = dict(app.UI_DEFAULTS)
        # Legacy named requests omit the new field and retain the default.
        for name in GENERATION_FIELDS:
            values.setdefault(name, None)
        values.pop("encoder_small_input")
        self.assertTrue(H3Request.from_values(values).sampling.encoder_small_input)
        preferences, _ = restore_preferences({"values": {"h3.encoder_small_input": False}}, {"h3.encoder_small_input": SimpleNamespace(value=True)})
        self.assertIs(preferences["h3.encoder_small_input"], False)


if __name__ == "__main__":
    unittest.main()
