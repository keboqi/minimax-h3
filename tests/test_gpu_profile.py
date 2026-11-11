"""A100 deployment defaults and graph routing without CUDA or downloads."""
import contextlib
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import h3_gpu


class GpuProfileTests(unittest.TestCase):
    def test_a100_driver_requires_cuda13_compatible_release(self):
        for driver in ("580.65.06", "580.126.16", "590.10.1"):
            with patch.object(subprocess, "check_output", return_value=driver):
                h3_gpu.verify_a100_driver()
        for driver in ("550.54.15", "580.64.00", "580.65.05"):
            with patch.object(subprocess, "check_output", return_value=driver):
                with self.assertRaisesRegex(SystemExit, 'driver 580'):
                    h3_gpu.verify_a100_driver()
        with patch.object(subprocess, "check_output", return_value=""):
            with self.assertRaisesRegex(SystemExit, 'Cannot verify'):
                h3_gpu.verify_a100_driver()

    def test_notebook_hardware_summary_recognizes_supported_single_gpu(self):
        from tests.test_colab_runtime import cell_source
        tree = ast.parse(cell_source('COLAB_G4 ='))
        end = next(i for i, node in enumerate(tree.body)
                   if isinstance(node, ast.FunctionDef) and node.name == 'meminfo_bytes')
        tree.body = tree.body[:end]
        for gpu, supported, a100 in (
            ('NVIDIA A100-SXM4-80GB, 81920 MiB, 580.126.16', True, True),
            ('NVIDIA A100 80GB PCIe, 81159 MiB, 580.126.16', True, True),
            ('NVIDIA A100-SXM4-40GB, 40960 MiB, 580.126.16', False, False),
            ('NVIDIA RTX PRO 6000, 97887 MiB, 580.126.16', True, False),
            ('NVIDIA A100, unknown, 580.126.16', False, False),
            ('NVIDIA A100, 81920 MiB, 580\nNVIDIA A100, 81920 MiB, 580', False, False),
            ('', False, False),
        ):
            namespace = {}
            with patch.object(subprocess, 'check_output', return_value=gpu), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(tree, '<hardware cell>', 'exec'), namespace)
            self.assertEqual(namespace['COLAB_STUDIO_SUPPORTED'], supported, gpu)
            self.assertEqual(namespace['COLAB_A100'], a100, gpu)

    def test_detection_handles_missing_or_multiple_gpus(self):
        for output, expected in (
            ("NVIDIA A100-SXM4-80GB\n", "a100"),
            ("NVIDIA RTX PRO 6000 Blackwell\n", "default"),
            ("NVIDIA A100\nNVIDIA A100\n", "default"),
        ):
            with patch.object(subprocess, "check_output", return_value=output):
                self.assertEqual(h3_gpu.detect_profile(), expected)
        with patch.object(subprocess, "check_output", side_effect=FileNotFoundError):
            self.assertEqual(h3_gpu.detect_profile(), "default")

    def test_colab_detection_rejects_small_unknown_missing_and_multiple_gpus(self):
        for gpu in ("", "NVIDIA T4, 15360", "NVIDIA A100-SXM4-40GB, 40960",
                    "NVIDIA A100, unknown", "NVIDIA RTX A1000, 81920",
                    "NVIDIA A100, 81920\nNVIDIA A100, 81920"):
            with patch.object(subprocess, "check_output", return_value=gpu):
                with self.assertRaisesRegex(ValueError, 'Studio needs'):
                    h3_gpu.detect_profile(colab=True)
        for gpu, profile in (("NVIDIA A100-SXM4-80GB, 81920", "a100"),
                             ("NVIDIA A100 80GB PCIe, 81159", "a100"),
                             ("NVIDIA RTX PRO 6000, 97887", "default")):
            with patch.object(subprocess, "check_output", return_value=gpu):
                self.assertEqual(h3_gpu.detect_profile(colab=True), profile)

    def test_a100_attention_aliases_resolve_before_generation(self):
        from h3_app.settings import GenerationRequest, SamplingSettings, resolve_settings
        with patch.dict(os.environ, H3_GPU_PROFILE="a100"):
            for mode, expected in (("sla", "SLA"), (" sparse-linear ", "SLA"),
                                   ("Auto", "Auto"), ("dense", "PyTorch"),
                                   ("Kitchen", "PyTorch"), ("pytorch", "PyTorch")):
                plan = resolve_settings(GenerationRequest(sampling=SamplingSettings(attention_mode=mode)))
                self.assertFalse(plan.issues, plan.issues)
                self.assertEqual(plan.effective.sampling.attention_mode, expected)
            for mode in ("Sol-Attn", "sage2", "Veda"):
                plan = resolve_settings(GenerationRequest(sampling=SamplingSettings(attention_mode=mode)))
                self.assertTrue(any('A100 supports' in issue for issue in plan.issues))

    def test_defaults_and_preloads_in_fresh_process_for_both_profiles(self):
        code = """
import json
from types import SimpleNamespace
from pathlib import Path
from h3_app.catalog import UI_DEFAULTS
from h3_app.settings import preset_settings
from h3_app.generation.requests import Flux3Request
from h3_app.drama3.serving.server import ServerSettings
from h3_prepare import h3_model_keys
from h3_models import PRELOAD_MODEL_KEYS
from h3_gpu import attention_choices
runtime = SimpleNamespace(script_dir=Path.cwd(), comfy_dir=Path.cwd()/'h3'/'ComfyUI')
print(json.dumps(dict(encoder=UI_DEFAULTS['text_encoder'], offload=UI_DEFAULTS['stage_model_offload'],
    preset=preset_settings('Singularity','Turbo').text_encoder,
    preset_offload=preset_settings('Fast','Turbo').stage_model_offload,
    flux=Flux3Request('test').model_choice, drama=ServerSettings.from_environment(runtime).checkpoint,
    preload=PRELOAD_MODEL_KEYS, prepare=h3_model_keys(), attention=attention_choices())))
"""
        for profile in ("default", "a100"):
            env = dict(os.environ, H3_GPU_PROFILE=profile)
            env.pop("H3_DRAMA3_SGLANG_CHECKPOINT", None)
            result = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                    capture_output=True, text=True)
            data = json.loads(result.stdout)
            a100 = profile == "a100"
            encoder = "INT8 ConvRot" if a100 else "NVFP4 / AWQ"
            self.assertEqual(data['encoder'], encoder)
            self.assertEqual(data['preset'], encoder)
            self.assertEqual(data['offload'], a100)
            self.assertEqual(data['preset_offload'], a100)
            self.assertEqual(data['flux'], "INT8 ConvRot (lower VRAM)" if a100 else "NVFP4 (Blackwell)")
            self.assertEqual(data['drama'], "int4" if a100 else "fp8")
            key, other = ("text_encoder_int8", "text_encoder") if a100 else ("text_encoder", "text_encoder_int8")
            for keys in (data['preload'], data['prepare']):
                self.assertIn(key, keys)
                self.assertNotIn(other, keys)
            if a100:
                self.assertEqual(data['attention'], ["PyTorch", "SLA", "Auto"])

    def test_a100_skips_blackwell_wheel_and_blocks_nemotron(self):
        import setup_h3
        from h3_app.dubbing import nemotron
        with patch.dict(os.environ, H3_GPU_PROFILE="a100"), patch.object(setup_h3, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(setup_h3.ensure_sageattention(Path("unused")))
            run.assert_not_called()
            self.assertFalse(nemotron.supported())

    def test_a100_auto_never_selects_sol_and_sla_uses_pytorch_dense(self):
        from h3_app import catalog, policy
        from h3_app.workflows import h3
        from h3_app.errors import H3Error
        from tests.workflow_cases import H3_DEFAULTS, AVAILABLE_NODES
        with patch.dict(os.environ, H3_GPU_PROFILE="a100"):
            selected, _, _ = policy.resolve_sol_policy(
                "Auto", "Reference media", 1376, 768, 10, None, None)
            self.assertFalse(selected)
            graph = h3.build_fl2va_graph(**dict(H3_DEFAULTS, use_sla=True, available_nodes=AVAILABLE_NODES))
            sla = [node for node in graph.values() if node['class_type'] == catalog.SLA_ATTENTION_NODE]
            self.assertTrue(sla)
            self.assertTrue(all(node['inputs']['dense_backend'] == 'pytorch' for node in sla))
            for backend in ('use_sol', 'use_sage', 'use_veda'):
                with self.assertRaisesRegex(H3Error, 'A100 supports'):
                    h3.build_fl2va_graph(**dict(H3_DEFAULTS, **{backend: True}, available_nodes=AVAILABLE_NODES))

    def test_a100_restores_blackwell_preferences_to_compatible_choices(self):
        from types import SimpleNamespace
        from h3_ui.persistence import restore_preferences
        choices = {
            'h3.text_encoder': ['NVFP4 / AWQ', 'INT8 ConvRot', 'BF16'],
            'h3.attention_mode': ['SLA', 'PyTorch', 'Auto'],
            'flux3.model': ['NVFP4 (Blackwell)', 'INT8 ConvRot (lower VRAM)', 'Base INT8 ConvRot'],
        }
        components = {key: SimpleNamespace(value=values[-1], choices=[(v, v) for v in values])
                      for key, values in choices.items()}
        saved = {'h3.text_encoder': 'NVFP4 / AWQ', 'h3.attention_mode': 'Sol-Attn',
                 'flux3.model': 'NVFP4 (Blackwell)'}
        with patch.dict(os.environ, H3_GPU_PROFILE='a100'):
            restored, _ = restore_preferences(saved, components)
        self.assertEqual(restored['h3.text_encoder'], 'INT8 ConvRot')
        self.assertEqual(restored['h3.attention_mode'], 'SLA')
        self.assertEqual(restored['flux3.model'], 'INT8 ConvRot (lower VRAM)')
        with patch.dict(os.environ, H3_GPU_PROFILE='a100'):
            for invalid in ([], {}, ['Sol-Attn']):
                restored, _ = restore_preferences({key: invalid for key in choices}, components)
                self.assertEqual(restored, {key: component.value for key, component in components.items()})


if __name__ == '__main__':
    unittest.main()
