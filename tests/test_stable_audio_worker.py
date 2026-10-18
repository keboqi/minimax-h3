"""The Stable Audio worker against a stand-in for the pinned stable-audio-tools API."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import soundfile
    import torch
except ImportError:  # pragma: no cover - the isolated runtime has both
    torch = None

from h3_app import stable_audio_worker

RATE = 1000
PADDING_SECONDS = 6  # upstream adapts the generated length to seconds_total + 6 s
# Keywords generate_diffusion_cond_inpaint forwards through **sampler_kwargs to the
# sampler (pinned commit 3241adba). Anything else reaches the transformer and crashes.
SAMPLER_KWARGS = {"sampler_type", "sigma_min", "sigma_max", "rho", "callback", "cfg_interval",
                  "scale_phi", "cfg_norm_threshold", "lora_configs", "inversion_params"}


def generate_diffusion_cond_inpaint(
        model, steps=250, cfg_scale=6, conditioning=None, conditioning_tensors=None,
        negative_conditioning=None, negative_conditioning_tensors=None, batch_size=1,
        sample_size=2097152, seed=-1, device="cuda", init_audio=None, init_noise_level=1.0,
        inpaint_audio=None, inpaint_mask=None, inpaint_mask_start_seconds=None,
        inpaint_mask_end_seconds=None, return_latents=False, adapt_duration_to_conditioning=None,
        duration_padding_sec=6.0, use_effective_length_for_schedule=None, mask_padding_attention=None,
        apg_scale=1.0, dist_shift=None, **sampler_kwargs):
    unknown = set(sampler_kwargs) - SAMPLER_KWARGS
    if unknown:
        raise TypeError(f"TransformerBlock.forward() got an unexpected keyword argument {sorted(unknown)[0]!r}")
    seconds = conditioning[0]["seconds_total"]
    audio = torch.full((1, 2, (seconds + PADDING_SECONDS) * RATE), 0.25)
    audio[..., seconds * RATE:] = 0.9  # louder padding must not survive or set the gain
    return audio


class FakeModel:
    pretransform = None

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, *_args):
        return self


@unittest.skipIf(torch is None, "torch and soundfile are needed")
class StableAudioWorkerTests(unittest.TestCase):
    def test_clips_are_cut_to_the_requested_duration(self):
        modules = {
            "stable_audio_tools": types.ModuleType("stable_audio_tools"),
            "stable_audio_tools.inference": types.ModuleType("stable_audio_tools.inference"),
            "stable_audio_tools.inference.generation": types.SimpleNamespace(
                generate_diffusion_cond_inpaint=generate_diffusion_cond_inpaint),
            "stable_audio_tools.models": types.ModuleType("stable_audio_tools.models"),
            "stable_audio_tools.models.factory": types.SimpleNamespace(
                create_model_from_config=lambda config: FakeModel()),
            "stable_audio_tools.models.utils": types.SimpleNamespace(
                copy_state_dict=lambda model, state: None, load_ckpt_state_dict=lambda path: {}),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_config.json").write_text(json.dumps({"sample_rate": RATE, "sample_size": 60 * RATE}))
            outputs = [root / "a.wav", root / "b.wav"]
            request = {"cache_dir": str(root), "items": [
                {"prompt": "rain", "duration": 3, "output": str(outputs[0])},
                {"prompt": "door", "duration": 1, "output": str(outputs[1])},
            ]}
            with patch.dict(sys.modules, modules), \
                    patch.object(stable_audio_worker, "_resolve_local_model_dirs", return_value=(root, None)), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.is_bf16_supported", return_value=True), \
                    patch.dict("os.environ", {}):
                stable_audio_worker.synthesize(request)
            for path, seconds in zip(outputs, (3, 1)):
                samples, rate = soundfile.read(path)
                self.assertEqual((rate, samples.shape), (RATE, (seconds * RATE,)))
                self.assertAlmostEqual(float(np.max(np.abs(samples))), 1.0, places=3)  # normalized after the cut


if __name__ == "__main__":
    unittest.main()
