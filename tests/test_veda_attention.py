"""Veda routing, independent sampling stages and lazy predictor provisioning."""

from dataclasses import replace
import inspect
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from h3_app import catalog, model_service
from h3_app.errors import H3Error
from h3_app.generation.construct_graph import construct_graph
from h3_app.generation import h3 as generation
from h3_app.generation.preparation import prepare_h3
from h3_app.generation.requests import H3Request
from h3_app.graph import Graph
from h3_app.policy import H3SplitUpscaleConfig
from h3_app.settings import GenerationRequest, SamplingSettings, resolve_settings
from h3_app.workflows import h3
from h3_models import MODEL_SPECS, PRELOAD_MODEL_KEYS
from h3_ui import application as app
from tests.workflow_cases import H3_DEFAULTS, MODELS, AVAILABLE_NODES, model_stack


def chain(nodes, model):
    result = []
    while model and model[0] in nodes:
        node = nodes[model[0]]
        result.append(node)
        model = node['inputs'].get('model')
    return result


class VedaAttentionTests(unittest.TestCase):
    def build(self, mode='Text to video', preset='Fast', refine_steps=2,
              upscale=True, split=False, cache='Off', refinement_variant=None):
        kwargs = dict(
            H3_DEFAULTS, use_veda=True, sla_preset=preset, steps=8,
            latent_upscale_model_name='upscaler.pth' if upscale else None,
            latent_upscale_refine_steps=refine_steps, cache_mode=cache,
            available_nodes=AVAILABLE_NODES | {catalog.VEDA_ATTENTION_NODE},
            model_name=MODELS.profile('quality').fl2va,
        )
        if refinement_variant:
            kwargs.update(refinement_variant=refinement_variant,
                          refinement_lora_name=MODELS.turbo_8step_lora)
        if split:
            kwargs['latent_split_config'] = H3SplitUpscaleConfig(
                512, 512, .25, .5, 73, 22, .75, 'off'
            )
        if mode == 'Reference media':
            for key in ('first_image', 'last_image', 'semantic_bridge',
                        'semantic_bridge_alpha', 'voice_reference_audios'):
                kwargs.pop(key, None)
            kwargs.update(reference_images=['reference.png'], reference_videos=[],
                          reference_audios=[], ref_image_size='Original')
            return h3.build_ref2va_graph(**kwargs)
        kwargs['first_image'] = 'first.png' if mode == 'First / last frame' else None
        return h3.build_fl2va_graph(**kwargs)

    def test_each_stage_gets_one_independent_override_and_its_own_schedule(self):
        for mode in ('Text to video', 'First / last frame', 'Reference media'):
            for preset in catalog.SLA_PRESET_INPUTS:
                for count in (1, 2, 3):
                    for split in (False, True):
                        with self.subTest(mode=mode, preset=preset, count=count, split=split):
                            nodes = self.build(mode, preset, count, split=split)
                            overrides = [n for n in nodes.values()
                                         if n['class_type'] == catalog.VEDA_ATTENTION_NODE]
                            self.assertEqual(len(overrides), 2)
                            base, refine = [n['inputs'] for n in overrides]
                            self.assertEqual(base['model'], refine['model'])
                            self.assertEqual(base['full_attention_steps'],
                                             '0, 7' if preset == 'Quality' else '0')
                            self.assertEqual(refine['full_attention_steps'],
                                             str(count - 1) if preset == 'Quality' else '')
                            key = 'veda_r2va' if mode == 'Reference media' else 'veda_t2va'
                            for inputs in (base, refine):
                                self.assertEqual(inputs['predictor'], MODEL_SPECS[key].local_name)
                                self.assertEqual(inputs['generated_sparsity'], '32')
                                self.assertEqual(inputs['reference_sparsity'], '0%')
                            for guider in (n for n in nodes.values() if n['class_type'] == 'BasicGuider'):
                                branch = chain(nodes, guider['inputs']['model'])
                                self.assertEqual(sum(n['class_type'] == catalog.VEDA_ATTENTION_NODE for n in branch), 1)
                                self.assertIn(catalog.CHUNK_FEED_FORWARD_NODE,
                                              [n['class_type'] for n in branch])
                            self.assertFalse(any(n['class_type'] in {
                                catalog.SLA_ATTENTION_NODE, catalog.SOL_ATTENTION_NODE,
                                catalog.SAGE_ATTENTION_NODE,
                            } for n in nodes.values()))

    def test_single_stage_and_spectrum_preserve_the_selected_attention(self):
        for cache in ('Off', 'Spectrum', 'EasyCache', 'FirstBlockCache'):
            nodes = self.build(upscale=False, cache=cache)
            self.assertEqual(sum(n['class_type'] == catalog.VEDA_ATTENTION_NODE for n in nodes.values()), 1)
            guider = next(n for n in nodes.values() if n['class_type'] == 'BasicGuider')
            self.assertIn(catalog.VEDA_ATTENTION_NODE,
                          [n['class_type'] for n in chain(nodes, guider['inputs']['model'])])

    def test_refinement_adapter_keeps_veda_on_both_active_branches(self):
        nodes = self.build(preset='Quality', refinement_variant=catalog.LIGHTX2V_8STEP_TURBO)
        schedules = []
        for guider in (n for n in nodes.values() if n['class_type'] == 'BasicGuider'):
            overrides = [n for n in chain(nodes, guider['inputs']['model'])
                         if n['class_type'] == catalog.VEDA_ATTENTION_NODE]
            self.assertEqual(len(overrides), 1)
            schedules.append(overrides[0]['inputs']['full_attention_steps'])
        self.assertCountEqual(schedules, ['0, 7', '1'])

    def test_cannot_stack_veda_with_another_attention_backend(self):
        for other in ('use_sla', 'use_sol', 'use_sage'):
            with self.subTest(other=other), self.assertRaisesRegex(H3Error, 'only attention override'):
                model_stack(Graph(), 'test.safetensors', MODELS, use_veda=True, **{other: True})

    def test_missing_node_fails_before_workflow_execution(self):
        with self.assertRaisesRegex(H3Error, 'VedaSparseAttention is not loaded'):
            model_stack(Graph(), 'test.safetensors', MODELS, use_veda=True,
                        available_nodes=set())
        required = h3.required_nodes_for('Text to video', False, 'Off', use_veda=True)
        self.assertIn(catalog.VEDA_ATTENTION_NODE, required)
        self.assertNotIn(catalog.SLA_ATTENTION_NODE, required)
        self.assertNotIn(catalog.SOL_ATTENTION_NODE, required)

    def test_sla_remains_default_and_veda_does_not_fall_into_auto_sol(self):
        self.assertEqual(SamplingSettings().attention_mode, 'SLA')
        self.assertEqual(app.UI_DEFAULTS['attention_mode'], 'SLA')
        with patch.object(app, 'SERVER_ATTENTION_BACKEND', 'sol'):
            for mode in ('Text to video', 'Reference media'):
                use_sol, _, reason = app.resolve_sol_policy('Veda', mode, 1344, 768, 15, None, None)
                self.assertFalse(use_sol)
                self.assertEqual(reason, 'forced Veda')
        request = GenerationRequest(sampling=replace(SamplingSettings(), attention_mode='Veda'))
        self.assertNotIn('sla_preset', resolve_settings(request).inactive)

    def test_predictors_are_lazy_and_task_specific(self):
        for mode, key in (('Text to video', 'veda_t2va'),
                          ('First / last frame', 'veda_t2va'),
                          ('Reference media', 'veda_r2va')):
            self.assertNotIn(key, PRELOAD_MODEL_KEYS)
            with (patch.object(model_service, 'stale_model_keys', return_value=[key]),
                  patch.object(model_service, 'sync_models') as sync,
                  patch.object(model_service, 'resolve_hf_token', return_value=None),
                  patch.object(model_service, 'model_file_is_ready', return_value=True)):
                app.ensure_h3_veda_predictor(mode)
                self.assertEqual(sync.call_args.kwargs['model_keys'], (key,))
            with (patch.object(model_service, 'stale_model_keys', return_value=[]),
                  patch.object(model_service, 'sync_models') as sync,
                  patch.object(model_service, 'model_file_is_ready', return_value=True)):
                app.ensure_h3_veda_predictor(mode)
                sync.assert_not_called()
        with (patch.object(model_service, 'stale_model_keys', return_value=[]),
              patch.object(model_service, 'model_file_is_ready', return_value=False)):
            with self.assertRaisesRegex(H3Error, 'valid model file'):
                app.ensure_h3_veda_predictor('Text to video')

    def test_preparation_downloads_only_for_veda_and_passes_selection_to_graph(self):
        for attention in ('SLA', 'Veda'):
            for mode in ('Text to video', 'First / last frame', 'Reference media'):
                values = {
                    name: app.UI_DEFAULTS.get(name) if p.default is inspect.Parameter.empty else p.default
                    for name, p in inspect.signature(app.generate).parameters.items() if name != 'progress'
                }
                values.update(
                    mode=mode, model_profile='Speed', prompt='test', attention_mode=attention,
                    cache_mode='Off', generation_mode='Normal', steps=15, seed=123,
                    sla_preset='Quality', semantic_bridge=False, latent_upscale=False,
                    use_trt_vae=False, use_int8_vae=False, use_lynnreal_vae=False,
                    first_image='first.png' if mode == 'First / last frame' else None,
                    ref_image_1='reference.png' if mode == 'Reference media' else None,
                    sol_step_off=0.0, sol_sink_tokens=0, postprocess='None',
                )
                request = H3Request.from_values(values)
                models = SimpleNamespace(
                    load_model_config=Mock(return_value=MODELS),
                    h3_text_encoder_settings=Mock(return_value=('text_encoder', MODELS.text_encoder, False)),
                    ensure_h3_text_encoder=Mock(return_value=(MODELS.text_encoder, False)),
                    ensure_profile_model=Mock(), ensure_audio_vae=Mock(),
                    model_file_is_ready=Mock(return_value=True), ensure_h3_veda_predictor=Mock(),
                )
                available = AVAILABLE_NODES | {catalog.VEDA_ATTENTION_NODE}
                services = SimpleNamespace(
                    models=models, execution=SimpleNamespace(object_info=lambda: dict.fromkeys(available, {})),
                    policy=SimpleNamespace(resolve_request_settings=app.resolve_request_settings,
                                           resolve_sol_policy=app.resolve_sol_policy),
                    workflows=SimpleNamespace(build_fl2va_graph=Mock(return_value={}),
                                              build_ref2va_graph=Mock(return_value={})),
                )
                preparation = prepare_h3(request, services, app.RUNTIME, values, time.monotonic(), lambda *a, **k: None)
                while True:
                    try:
                        next(preparation)
                    except StopIteration as completed:
                        prepared = completed.value
                        break
                self.assertEqual(prepared.effective_veda, attention == 'Veda')
                self.assertEqual(prepared.effective_sla, attention == 'SLA')
                self.assertFalse(prepared.effective_sol)
                self.assertEqual(prepared.effective_sla_preset, 'Quality')
                if attention == 'Veda':
                    models.ensure_h3_veda_predictor.assert_called_once_with(mode)
                else:
                    models.ensure_h3_veda_predictor.assert_not_called()
                construct_graph(request, prepared, services)
                builder = services.workflows.build_ref2va_graph if mode == 'Reference media' else services.workflows.build_fl2va_graph
                self.assertEqual(builder.call_args.kwargs['use_veda'], attention == 'Veda')

                # Exercise final provenance through the orchestration boundary.
                # Graph tests above separately verify the actual stage schedules.
                if attention == 'Veda':
                    def prepared_request(*args, **kwargs):
                        yield from ()
                        return prepared

                    request.output.result_format = 'Audio'
                    request.finishing.latent_upscale = True
                    request.finishing.latent_upscale_refine_steps = 3
                    models.unload_prompt_rewriter = Mock()
                    services.execution.submit_prompt = Mock(return_value='veda-job')
                    services.execution.poll_comfy_progress = Mock(return_value=[])
                    services.execution.wait_for_history = Mock(return_value={})
                    services.media = SimpleNamespace(
                        resolve_audio_output=Mock(return_value=Path('result.wav')),
                    )
                    with (patch.object(generation, 'prepare_h3', prepared_request),
                          patch.object(generation, 'construct_graph', return_value={}),
                          patch.object(generation, 'write_snapshot') as snapshot):
                        updates = list(generation.generate(
                            request, services, app.RUNTIME, run_context={},
                        ))
                    self.assertEqual(updates[-1].output, 'result.wav', updates[-1].status)
                    saved = snapshot.call_args.args[1]['settings']
                    key = 'veda_r2va' if mode == 'Reference media' else 'veda_t2va'
                    self.assertEqual(saved['attention_mode'], 'Veda')
                    self.assertEqual(saved['veda_predictor'], MODEL_SPECS[key].local_name)
                    self.assertEqual(saved['veda_predictor_sha256'], MODEL_SPECS[key].expected_sha256)
                    self.assertEqual(saved['veda_base_dense_steps'], '0, 14')
                    self.assertEqual(saved['veda_refine_dense_steps'], '2')


if __name__ == '__main__':
    unittest.main()
