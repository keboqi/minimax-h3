"""Independent refinement adapters, sampler routing, and lazy provisioning."""
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gradio_app as app
import h3_models
from h3_app import model_service
from h3_app.catalog import (
    LARRY_TURBO, LIGHTX2V_8STEP_TURBO, PDMD_2STEP_LORA, PDMD_4STEP_LORA,
    REFINEMENT_LORA_SETTINGS, SAME_REFINEMENT_LORA,
)
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS, GenerationArguments
from h3_app.generation.requests import H3Request
from h3_app.settings import GenerationRequest, resolve_settings
from h3_ui.persistence import restore_preferences
from h3_app.workflows.h3 import add_refinement_model
from types import SimpleNamespace


class RefinementLoraTests(unittest.TestCase):
    def models(self):
        return app.ModelConfig(
            {}, 'speed', 'text.safetensors', 'video.safetensors', 'audio.safetensors',
            turbo_8step_lora='generation_8step_768p.safetensors',
            turbo_8step_ref_lora='reference_8step_768p.safetensors',
            pdmd_2step_lora=h3_models.MODEL_SPECS['pdmd_2step_lora'].local_name,
            pdmd_4step_lora=h3_models.MODEL_SPECS['pdmd_4step_lora'].local_name,
        )

    def model_chain(self, graph, ref):
        nodes = []
        while ref:
            node = graph[ref[0]]
            nodes.append(node)
            ref = node['inputs'].get('model')
        return nodes

    def test_both_workflows_replace_lora_only_on_refinement_branch(self):
        for build in (app.build_fl2va_graph, app.build_ref2va_graph):
            for variant in (PDMD_2STEP_LORA, PDMD_4STEP_LORA, LARRY_TURBO):
                for split in (False, True):
                    with self.subTest(build=build.__name__, variant=variant, split=split):
                        args = {
                            name: app.UI_DEFAULTS.get(name, 0)
                            for name, param in inspect.signature(build).parameters.items()
                            if param.default is inspect.Parameter.empty
                        }
                        models = self.models()
                        args.update(
                            prompt='A bird flies', width=864, height=480, duration=5,
                            steps=8, scheduler='simple', seed=7, model_name='base.safetensors',
                            models=models, turbo_variant=LIGHTX2V_8STEP_TURBO,
                            turbo_lora_name=models.turbo_8step_lora, turbo_strength=1.0,
                            use_sol=False, cache_mode='Off', use_sla=True,
                            latent_upscale_model_name='upscaler.pth',
                            refinement_lora_name='refinement.safetensors', refinement_variant=variant,
                            available_nodes=(app.turbo_required_nodes(LIGHTX2V_8STEP_TURBO, models.turbo_8step_lora)
                                             | app.turbo_required_nodes(variant) | {app.SLA_ATTENTION_NODE}),
                            latent_split_config=(app.H3SplitUpscaleConfig(512, 512, .25, .5, 73, 22, .75, 'off')
                                                 if split else None),
                        )
                        if build is app.build_fl2va_graph:
                            args.update(first_image=None, last_image=None)
                        else:
                            args.update(reference_images=[], reference_videos=[], reference_audios=[])
                        graph = build(**args)
                        samples = [n['inputs'] for n in graph.values() if n['class_type'] == 'SamplerCustomAdvanced']
                        initial = samples[0]
                        initial_model = graph[initial['guider'][0]]['inputs']['model']
                        if split:
                            target = next(n['inputs'] for n in graph.values() if n['class_type'] == app.H3_SPLIT_UPSCALE_NODE)
                            refine_model = target['model']
                        else:
                            target = samples[1]
                            refine_model = graph[target['guider'][0]]['inputs']['model']
                        initial_chain = self.model_chain(graph, initial_model)
                        refine_chain = self.model_chain(graph, refine_model)
                        self.assertEqual([n['inputs']['lora_name'] for n in initial_chain if 'lora_name' in n['inputs']],
                                         [models.turbo_8step_lora])
                        self.assertEqual([n['inputs']['lora_name'] for n in refine_chain if 'lora_name' in n['inputs']],
                                         ['refinement.safetensors'])
                        self.assertEqual(initial_chain[-1], refine_chain[-1])
                        self.assertEqual(next(n['inputs']['dense_steps'] for n in initial_chain if n['class_type'] == app.SLA_ATTENTION_NODE), '0')
                        self.assertEqual(next(n['inputs']['dense_steps'] for n in refine_chain if n['class_type'] == app.SLA_ATTENTION_NODE), '')
                        expected = app.LARRY_TURBO_SAMPLER_NODE if variant == LARRY_TURBO else app.CORE_SAMPLER_NODE
                        self.assertEqual(graph[target['sampler'][0]]['class_type'], expected)
                        if variant != LARRY_TURBO:
                            self.assertEqual(graph[target['sampler'][0]]['inputs']['sampler_name'], 'euler')
                        # Retain the generation schedule's last two intervals for low denoise.
                        split_node = graph[target['sigmas'][0]]
                        self.assertEqual(split_node['inputs']['step'], 6)
                        self.assertEqual(split_node['inputs']['sigmas'], initial['sigmas'])

    def test_same_has_no_extra_model_branch_and_old_api_defaults_to_same(self):
        graph = app.Graph()
        base = app.Graph.out(graph.add('UNETLoader', unet_name='base', weight_dtype='default'))
        self.assertIsNone(add_refinement_model(graph, base, lora_name=None, variant=None, available_nodes=set()))
        self.assertEqual(len(graph.nodes), 1)
        values = [app.UI_DEFAULTS.get(name) for name in GENERATION_FIELDS[:-1]]
        self.assertEqual(GenerationArguments.from_positional(values).values['latent_upscale_refine_lora'], SAME_REFINEMENT_LORA)
        request_values = {name: app.UI_DEFAULTS.get(name) for name in GENERATION_FIELDS[:-1]}
        self.assertEqual(H3Request.from_values(request_values).finishing.latent_upscale_refine_lora, SAME_REFINEMENT_LORA)

    def test_selection_survives_restore_and_is_in_effective_settings(self):
        field = 'h3.latent_upscale_refine_lora'
        restored, _ = restore_preferences(
            {field: PDMD_2STEP_LORA},
            {field: SimpleNamespace(value=SAME_REFINEMENT_LORA, choices=[SAME_REFINEMENT_LORA, *REFINEMENT_LORA_SETTINGS])},
        )
        self.assertEqual(restored[field], PDMD_2STEP_LORA)
        plan = resolve_settings(GenerationRequest.from_values({'latent_upscale_refine_lora': PDMD_2STEP_LORA}))
        self.assertEqual(plan.effective.finishing.latent_upscale_refine_lora, PDMD_2STEP_LORA)
        disabled = resolve_settings(GenerationRequest.from_values({'latent_upscale': False, 'latent_upscale_refine_lora': PDMD_2STEP_LORA}))
        self.assertIn('latent_upscale_refine_lora', disabled.inactive)

    def test_pdmd_is_lazy_and_old_model_config_gets_filenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = RuntimeConfig(root, 'http://fixture', root / 'ComfyUI', root / 'models.json', root / 'outputs')
            config = h3_models._build_config('manifest.json')
            for key in ('pdmd_2step_lora', 'pdmd_4step_lora'):
                config.pop(key)
                self.assertNotIn(key, h3_models.PRELOAD_MODEL_KEYS)
            runtime.models_config.write_text(json.dumps(config), encoding='utf-8')
            models = model_service.load_model_config(runtime=runtime)
            for variant, spec in ((PDMD_2STEP_LORA, 'pdmd_2step_lora'), (PDMD_4STEP_LORA, 'pdmd_4step_lora')):
                for mode in ('Text to video', 'Reference media'):
                    self.assertEqual(models.turbo_lora_for(mode, variant), h3_models.MODEL_SPECS[spec].local_name)
                    with (patch.object(model_service, 'stale_model_keys', return_value=[spec]),
                          patch.object(model_service, 'sync_models') as sync,
                          patch.object(model_service, 'model_file_is_ready', return_value=True),
                          patch.object(model_service, 'resolve_hf_token', return_value=None)):
                        self.assertTrue(model_service.ensure_turbo_lora(models, variant, mode, runtime=runtime))
                        self.assertEqual(sync.call_args.kwargs['model_keys'], (spec,))
                    with (patch.object(model_service, 'stale_model_keys', return_value=[]),
                          patch.object(model_service, 'sync_models') as sync):
                        self.assertFalse(model_service.ensure_turbo_lora(models, variant, mode, runtime=runtime))
                        sync.assert_not_called()
