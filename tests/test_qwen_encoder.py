"""Encoder patch lifecycle and graph routing without a GPU or ComfyUI import."""
import ast
import inspect
import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio_app as app
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_ui.persistence import PERSISTED_NAMES


def load_proxy(compiler):
    source = Path('custom_nodes/H3Acceleration/__init__.py').read_text(encoding='utf-8')
    names = {'_h3_encoder_forward', '_H3AcceleratedCLIPProxy', 'H3QwenEncoderAcceleration'}
    nodes = [n for n in ast.parse(source).body if getattr(n, 'name', None) in names]
    ns = dict(inspect=inspect, logging=logging, time=time, torch=SimpleNamespace(compile=compiler), _H3_ENCODER_LOCK=threading.RLock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<encoder>', 'exec'), ns)
    return ns['_H3AcceleratedCLIPProxy']


class Block:
    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, **kwargs):
        return optimized_attention(x, mask=attention_mask, enable_gqa=True)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Mock(side_effect=lambda fn, **kwargs: fn)
        self.proxy = load_proxy(self.compiler)
        self.native = Mock(return_value='native')
        self.sage = Mock(return_value='sage')
        self.blocks = [Block(), Block()]
        model = type('MiniMaxQwen3VL', (), {})()
        model.model = SimpleNamespace(layers=self.blocks[:1])
        model.visual = SimpleNamespace(blocks=self.blocks[1:])
        self.clip = SimpleNamespace(cond_stage_model=SimpleNamespace(qwen3vl_32b=SimpleNamespace(transformer=model)))
        self.clip.clone = lambda: self.clip
        self.mask = object()
        self.clip.encode_from_tokens_scheduled = lambda tokens: [b.forward(tokens, self.mask, None, self.native) for b in self.blocks]
        backend = SimpleNamespace(SAGE_ATTENTION_IS_AVAILABLE=True, SAGE_ATTENTION_SUPPORTS_MASK=True, attention_sage=self.sage)
        self.modules = patch.dict(sys.modules, {'comfy.ldm.modules': SimpleNamespace(attention=backend)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_modes_restore_baseline_and_reuse_compiled_variants(self):
        for sage, compile_encoder in ((False, False), (True, False), (False, True), (True, True)):
            wrapper = self.proxy(self.clip, sage, compile_encoder).clone()
            for _ in range(2):
                self.assertEqual(wrapper.encode_from_tokens_scheduled('tokens'), ['sage' if sage else 'native'] * 2)
                self.assertTrue(all('forward' not in b.__dict__ for b in self.blocks))
            if sage:
                self.assertIs(self.sage.call_args.kwargs['mask'], self.mask)
                self.assertTrue(self.sage.call_args.kwargs['enable_gqa'])
        self.assertEqual(self.compiler.call_count, 4)
        self.assertEqual(self.proxy(self.clip).encode_from_tokens_scheduled('x'), ['native'] * 2)

    def test_failure_restores_all_shared_blocks(self):
        self.clip.encode_from_tokens_scheduled = Mock(side_effect=RuntimeError('test failure'))
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            self.proxy(self.clip, True, True).encode_from_tokens_scheduled('x')
        self.assertTrue(all('forward' not in b.__dict__ for b in self.blocks))

    def test_partial_compile_failure_restores_previous_blocks(self):
        self.compiler.side_effect = [lambda *a, **k: None, RuntimeError('compile failure')]
        with self.assertRaisesRegex(RuntimeError, 'compile failure'):
            self.proxy(self.clip, False, True).encode_from_tokens_scheduled('x')
        self.assertTrue(all('forward' not in b.__dict__ for b in self.blocks))

    def graph(self, builder, sage=False, compile_encoder=False, nodes=True):
        args = {n: app.UI_DEFAULTS.get(n, 0) for n, p in inspect.signature(builder).parameters.items() if p.default is inspect.Parameter.empty}
        args.update(prompt='test', first_image=None, last_image=None, reference_images=[], reference_videos=[], reference_audios=[], width=864, height=480, duration=5, steps=4, seed=7, models=SimpleNamespace(text_encoder='encoder.safetensors'), available_nodes={'H3QwenEncoderAcceleration'} if nodes else set(), qwen_sage=sage, qwen_compile=compile_encoder)
        args = {k: v for k, v in args.items() if k in inspect.signature(builder).parameters}
        with patch.object(app, 'add_model_stack', return_value=(["model", 0], ["clip", 0], ["vae", 0], ["audio", 0])), patch.object(app, 'finish_sampling'):
            return builder(**args)

    def test_both_graphs_separate_cache_by_encoder_options(self):
        for builder in (app.build_fl2va_graph, app.build_ref2va_graph):
            keys = []
            for sage, compiled in ((False, False), (True, False), (False, True), (True, True)):
                graph = self.graph(builder, sage, compiled)
                cache = next(n for n in graph.values() if n['class_type'] == app.H3_CONDITIONING_CACHE_NODE)
                accel = graph[cache['inputs']['clip'][0]]
                self.assertEqual(accel['inputs']['sage'], sage)
                self.assertEqual(accel['inputs']['compile_encoder'], compiled)
                keys.append(cache['inputs']['cache_key'])
            self.assertEqual(len(set(keys)), 4)
            with self.assertRaisesRegex(app.H3Error, 'Update provisioning'):
                self.graph(builder, True, nodes=False)

    def test_old_api_defaults_and_preferences(self):
        old = GenerationArguments.from_positional([None] * GENERATION_FIELDS.index('qwen_sage'))
        for field in ('qwen_sage', 'qwen_compile'):
            self.assertFalse(old.values[field])
            self.assertFalse(app.UI_DEFAULTS[field])
            self.assertIn('h3.' + field, PERSISTED_NAMES)


if __name__ == '__main__':
    unittest.main()
