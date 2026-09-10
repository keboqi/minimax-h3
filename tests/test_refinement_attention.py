"""Stage-specific SLA graph composition and pinned runtime scheduling."""
import ast
from pathlib import Path
import unittest

import gradio_app as app


class RefinementAttentionTests(unittest.TestCase):
    def stack(self, preset='Quality'):
        graph = app.Graph()
        base = graph.add('ModelLoader')
        sla = graph.add(app.SLA_ATTENTION_NODE, model=app.Graph.out(base), **app.SLA_PRESET_INPUTS[preset], dense_steps='0', engine='triton', enabled=True)
        chunk = graph.add(app.CHUNK_FEED_FORWARD_NODE, model=app.Graph.out(sla), chunks=2)
        cache = graph.add('SpectrumApplyMiniMaxH3', model=app.Graph.out(chunk), degree=1)
        shift = graph.add(app.H3_SIGMA_SHIFT_NODE, model=app.Graph.out(cache), shift_video=6.0, shift_audio=3.0)
        return graph, app.Graph.out(shift), sla

    def test_branch_preserves_patches_and_does_not_stack_sla(self):
        for preset in app.SLA_PRESET_INPUTS:
            graph, original, sla_id = self.stack(preset)
            refined = app.h3_refinement_attention_model(graph, original)
            for expected in (app.H3_SIGMA_SHIFT_NODE, 'SpectrumApplyMiniMaxH3', app.CHUNK_FEED_FORWARD_NODE, app.SLA_ATTENTION_NODE):
                old, new = graph.nodes[original[0]], graph.nodes[refined[0]]
                self.assertNotEqual(original, refined)
                self.assertEqual(new['class_type'], expected)
                for key, value in old['inputs'].items():
                    if key not in ('model', 'dense_steps'):
                        self.assertEqual(new['inputs'][key], value)
                if expected == app.SLA_ATTENTION_NODE:
                    self.assertEqual(new['inputs']['dense_steps'], '')
                    self.assertEqual(new['inputs']['model'], old['inputs']['model'])
                original, refined = old['inputs']['model'], new['inputs']['model']
            self.assertEqual(graph.nodes[sla_id]['inputs']['dense_steps'], '0')

    def test_other_attention_is_unchanged(self):
        graph = app.Graph()
        ref = app.Graph.out(graph.add('SageAttention', model=['external', 0]))
        self.assertEqual(app.h3_refinement_attention_model(graph, ref), ref)
        self.assertEqual(len(graph.nodes), 1)

    def test_sampler_and_split_refinement_use_branched_model(self):
        for upscale, split in ((False, False), (True, False), (True, True)):
            graph, model, _ = self.stack()
            config = app.H3SplitUpscaleConfig(512, 512, .25, .5, 73, 22, .75, 'off') if split else None
            app.finish_sampling(graph, model_ref=model, conditioning_ref=['target', 0], latent_ref=['latent', 0], video_vae_ref=['vae', 0], audio_vae_ref=['audio', 0], seed=7, steps=8, scheduler='simple', turbo_variant=None, filename_prefix='test', initial_conditioning_ref=['initial', 0], initial_latent_ref=['initial_latent', 0], latent_upscale_model_name='upscaler.pth' if upscale else None, latent_split_config=config)
            guiders = [n['inputs'] for n in graph.nodes.values() if n['class_type'] == 'BasicGuider']
            target = next(g for g in guiders if g['conditioning'] == ['target', 0])
            if upscale:
                self.assertNotEqual(target['model'], model)
                initial = next(g for g in guiders if g['conditioning'] == ['initial', 0])
                self.assertEqual(initial['model'], model)
                if split:
                    node = next(n for n in graph.nodes.values() if n['class_type'] == app.H3_SPLIT_UPSCALE_NODE)
                    self.assertEqual(node['inputs']['model'], target['model'])
            else:
                self.assertEqual(target['model'], model)
                self.assertEqual(sum(n['class_type'] == app.SLA_ATTENTION_NODE for n in graph.nodes.values()), 1)

    @unittest.skipUnless(Path('.cache/upstream-upgrade/sla_patch.py').is_file(), 'pinned SLA audit source unavailable')
    def test_pinned_sla_two_step_schedule(self):
        source = ast.parse(Path('.cache/upstream-upgrade/sla_patch.py').read_text(encoding='utf-8'))
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_make_wrapper')
        def prepare(state, options, *unused):
            state['step'] = options['index'] + 1
            return options['count']
        namespace = {'_prepare_run_state': prepare, '_call_next_wrapper': lambda executor, *a, **k: executor(k['transformer_options']['_h3sla_dense']), '_summarise': lambda *a: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned SLA wrapper>', 'exec'), namespace)
        for preset, count, expected in (('Fast', 2, [False, False]), ('Balanced', 2, [False, False]), ('Quality', 2, [False, True]), ('Quality', 1, [True])):
            state = dict(step=0, summarized=False, prev_lut={})
            wrapper = namespace['_make_wrapper'](state, .85, 64, 64, app.SLA_PRESET_INPUTS[preset]['dense_last_steps'], dense_steps=frozenset())
            actual = [wrapper(lambda dense: dense, None, None, None, transformer_options={'index': i, 'count': count}) for i in range(count)]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
