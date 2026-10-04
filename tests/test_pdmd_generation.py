"""PDMD/DMAD generation selection, schedules, and independent refinement."""
import inspect
from dataclasses import replace
from types import SimpleNamespace
import unittest

from h3_ui import application as app
from h3_app.catalog import DMAD_4STEP_LORA, PDMD_4STEP_LORA, TURBO_SETTINGS
from h3_app.generation.preparation import _validate_sampling_steps
from h3_app.policy import normalize_turbo_variant, turbo_sampler_name, turbo_sigma_shifts, turbo_steps_for
from h3_app.settings import GenerationRequest, SamplingSettings, resolve_settings, transition_modes
from h3_ui.persistence import restore_preferences
from tests import test_refinement_lora as refinement_tests


class PdmdGenerationTests(unittest.TestCase):
    def test_defaults_validation_and_preferences(self):
        for variant, steps in ((PDMD_4STEP_LORA, 4), (DMAD_4STEP_LORA, 4)):
            with self.subTest(variant=variant):
                self.assertEqual(normalize_turbo_variant(variant), variant)
                self.assertEqual(turbo_steps_for(variant), steps)
                self.assertEqual(turbo_sampler_name(variant, 'pdmd.safetensors'), 'euler')
                self.assertEqual(turbo_sampler_name(variant, None), 'res_multistep')
                _, values = transition_modes(None, {
                    'generation_mode': 'Turbo', 'turbo_variant': variant,
                    'steps': 8, 'scheduler': 'beta', 'latent_upscale_refine_steps': 2,
                }, 'turbo_variant')
                self.assertEqual((values['steps'], values['scheduler']), (steps, 'simple'))
                self.assertEqual(values['latent_upscale_refine_steps'], 2)
                self.assertEqual(resolve_settings(GenerationRequest.from_values(values)).issues, ())
                _validate_sampling_steps('speed', True, variant, steps)
                with self.assertRaises(app.H3Error):
                    _validate_sampling_steps('speed', True, variant, steps - 1)
                request = GenerationRequest(sampling=SamplingSettings(
                    steps=steps, turbo_variant=variant, latent_upscale_refine_steps=1,
                ))
                self.assertTrue(resolve_settings(replace(request, sampling=replace(request.sampling, steps=steps - 1))).issues)
                restored, _ = restore_preferences(
                    {'h3.steps': steps, 'h3.turbo_variant': variant},
                    {'h3.steps': SimpleNamespace(value=4, minimum=4, maximum=30),
                     'h3.turbo_variant': SimpleNamespace(value=app.DEFAULT_TURBO, choices=list(TURBO_SETTINGS))},
                )
                self.assertEqual(restored['h3.steps'], steps)
                self.assertEqual(restored['h3.turbo_variant'], variant)

    def test_both_workflows_use_selected_adapter_for_generation_and_refinement(self):
        models = refinement_tests.RefinementLoraTests().models()
        for build in (app.build_fl2va_graph, app.build_ref2va_graph):
            for variant, steps in ((PDMD_4STEP_LORA, 4), (DMAD_4STEP_LORA, 4)):
                for refine_variant in (None, PDMD_4STEP_LORA, DMAD_4STEP_LORA):
                    with self.subTest(build=build.__name__, variant=variant, refinement=refine_variant):
                        args = {
                            name: app.UI_DEFAULTS.get(name, 0)
                            for name, param in inspect.signature(build).parameters.items()
                            if param.default is inspect.Parameter.empty
                        }
                        filename = models.turbo_lora_for('Text to video', variant)
                        args.update(
                            prompt='A bird flies', width=864, height=480, duration=5,
                            steps=steps, scheduler='simple', seed=7, model_name='base.safetensors',
                            models=models, turbo_variant=variant, turbo_lora_name=filename,
                            turbo_strength=1.0, use_sol=False, cache_mode='Off',
                            available_nodes=(app.turbo_required_nodes(variant)
                                             | (app.turbo_required_nodes(refine_variant) if refine_variant else set())),
                            latent_upscale_model_name='upscaler.pth', latent_upscale_refine_steps=1,
                            refinement_lora_name=models.turbo_lora_for('Text to video', refine_variant) if refine_variant else None,
                            refinement_variant=refine_variant,
                        )
                        if build is app.build_fl2va_graph:
                            args.update(first_image=None, last_image=None)
                        else:
                            args.update(reference_images=[], reference_videos=[], reference_audios=[])
                        graph = build(**args)
                        schedule = next(n['inputs'] for n in graph.values() if n['class_type'] == 'BasicScheduler')
                        self.assertEqual(schedule['steps'], steps)
                        samples = [n['inputs'] for n in graph.values() if n['class_type'] == 'SamplerCustomAdvanced']
                        initial, refined = samples
                        for sample, expected, selected in (
                            (initial, filename, variant),
                            (refined, models.turbo_lora_for('Text to video', refine_variant) if refine_variant else filename, refine_variant or variant),
                        ):
                            model = graph[sample['guider'][0]]['inputs']['model']
                            chain = refinement_tests.RefinementLoraTests().model_chain(graph, model)
                            loras = [n['inputs'] for n in chain if 'lora_name' in n['inputs']]
                            self.assertEqual([n['lora_name'] for n in loras], [expected])
                            self.assertEqual(loras[0]['strength_model'], 1.0)
                            self.assertEqual(graph[sample['sampler'][0]]['inputs']['sampler_name'], 'euler')
                            shifts = [n['inputs'] for n in chain if n['class_type'] == app.H3_SIGMA_SHIFT_NODE]
                            self.assertEqual(
                                [(n['shift_video'], n['shift_audio']) for n in shifts],
                                [(12.0, 2.0)] if selected == DMAD_4STEP_LORA else [],
                            )
                        self.assertEqual(graph[refined['sigmas'][0]]['inputs']['sigmas'], initial['sigmas'])
                        self.assertFalse(any(n['class_type'] == app.LIGHTX2V_BYPASS_LORA_NODE for n in graph.values()))

    def test_dmad_requires_shift_node_and_normal_mode_keeps_base_schedule(self):
        self.assertEqual(turbo_sigma_shifts(DMAD_4STEP_LORA, 'dmad.safetensors'), (12.0, 2.0))
        self.assertIsNone(turbo_sigma_shifts(DMAD_4STEP_LORA, None))
        self.assertIn(app.H3_SIGMA_SHIFT_NODE, app.turbo_required_nodes(DMAD_4STEP_LORA))
        graph = app.Graph()
        with self.assertRaisesRegex(app.H3Error, 'MiniMaxH3SigmaShift'):
            app.add_turbo_model_patch(
                graph, ['base', 0], lora_name='dmad.safetensors',
                turbo_variant=DMAD_4STEP_LORA, strength=1.0,
                available_nodes={app.CORE_LORA_LOADER_NODE, app.CORE_SAMPLER_NODE},
            )
