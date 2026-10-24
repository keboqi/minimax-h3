"""The resident audio tools worker against stand-ins for qwen-tts and the pinned stable-audio-tools API."""

from __future__ import annotations

from contextlib import nullcontext
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

try:
    import soundfile
    import torch
except ImportError:  # pragma: no cover - the isolated runtime has both
    torch = None

from h3_app import audio_tools_worker as worker_module

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


def stable_audio_modules(create_model):
    return {
        "stable_audio_tools": types.ModuleType("stable_audio_tools"),
        "stable_audio_tools.inference": types.ModuleType("stable_audio_tools.inference"),
        "stable_audio_tools.inference.generation": types.SimpleNamespace(
            generate_diffusion_cond_inpaint=generate_diffusion_cond_inpaint),
        "stable_audio_tools.models": types.ModuleType("stable_audio_tools.models"),
        "stable_audio_tools.models.factory": types.SimpleNamespace(create_model_from_config=create_model),
        "stable_audio_tools.models.utils": types.SimpleNamespace(
            copy_state_dict=lambda model, state: None, load_ckpt_state_dict=lambda path: {}),
    }


@unittest.skipIf(torch is None, "torch and soundfile are needed")
class StableAudioWorkerTests(unittest.TestCase):
    def test_clips_are_cut_to_the_requested_duration_and_the_model_stays_loaded(self):
        loads = []
        modules = stable_audio_modules(lambda config: loads.append(config) or FakeModel())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = worker_module.stable_audio_dir(root)
            model_dir.mkdir(parents=True)
            (model_dir / "model_config.json").write_text(json.dumps({"sample_rate": RATE, "sample_size": 60 * RATE}))
            outputs = [root / "a.wav", root / "b.wav", root / "c.wav"]
            worker = worker_module.Worker(root)
            with patch.dict(sys.modules, modules), \
                    patch.object(worker, "flash_attention", return_value=False), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.is_bf16_supported", return_value=True):
                worker.handle({"op": "stable_audio", "items": [
                    {"prompt": "rain", "duration": 3, "output": str(outputs[0])},
                    {"prompt": "door", "duration": 1, "output": str(outputs[1])},
                ]})
                worker.handle({"op": "stable_audio", "items": [
                    {"prompt": "wind", "duration": 2, "output": str(outputs[2])},
                ]})
            self.assertEqual(len(loads), 1)
            self.assertEqual(worker.loaded, ["stable_audio"])
            for path, seconds in zip(outputs, (3, 1, 2)):
                samples, rate = soundfile.read(path)
                self.assertEqual((rate, samples.shape), (RATE, (seconds * RATE, 2)))  # stereo kept
                self.assertAlmostEqual(float(np.max(np.abs(samples))), 1.0, places=3)  # normalized after the cut


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_voice_design_loads_offline_once_with_flash_attention_and_passes_text_verbatim(self):
        model = Mock()
        model.generate_voice_design.return_value = ([np.ones(2400, dtype=np.float32) * 0.1], 24000)
        qwen = SimpleNamespace(Qwen3TTSModel=Mock())
        qwen.Qwen3TTSModel.from_pretrained.return_value = model
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                                     bfloat16="bf16", float16="fp16", inference_mode=nullcontext)
        sound = Mock()
        hub = SimpleNamespace(snapshot_download=Mock(return_value="persistent/snapshot"))
        worker = worker_module.Worker(self.root)
        request = {"op": "voice_design", "text": "你好，出发吧。", "description": "温暖自然的女声。",
                   "language": "Chinese", "output": str(self.root / "voice.wav")}
        with patch.dict(sys.modules, {"qwen_tts": qwen, "torch": fake_torch, "soundfile": sound,
                                      "huggingface_hub": hub, "flash_attn": types.ModuleType("flash_attn")}):
            worker.handle(request)
            worker.handle({**request, "output": str(self.root / "voice2.wav")})
        qwen.Qwen3TTSModel.from_pretrained.assert_called_once_with(
            "persistent/snapshot", device_map="cuda:0", dtype="bf16", attn_implementation="flash_attention_2")
        self.assertEqual(hub.snapshot_download.call_args.kwargs,
                         {"cache_dir": str(worker_module.voice_design_cache(self.root)), "local_files_only": True})
        model.generate_voice_design.assert_called_with(text=request["text"], language="Chinese",
                                                       instruct=request["description"], max_new_tokens=4096)
        self.assertEqual(sound.write.call_args_list[0].args[0], request["output"])
        self.assertEqual(sound.write.call_args_list[0].args[2], 24000)
        self.assertEqual(worker.loaded, ["voice_design"])

    def test_voice_batch_keeps_each_failure_with_its_item(self):
        worker = worker_module.Worker(self.root)

        def design(item):
            if item["text"] == "bad":
                raise RuntimeError("empty audio")
            return item["output"]

        items = [{"text": "ok", "output": "a.wav"}, {"text": "bad", "output": "b.wav"},
                 {"text": "ok", "output": "c.wav", "items": ["ignored"]}]
        with patch.object(worker, "design_one_voice", side_effect=design), patch("traceback.print_exc"):
            reply = worker.handle({"op": "voice_design", "items": items})
        self.assertEqual(reply["outputs"], ["a.wav", None, "c.wav"])
        self.assertEqual(reply["errors"][0], "")
        self.assertIn("empty audio", reply["errors"][1])

    def test_each_item_raises_unless_asked_to_keep_going(self):
        def make(item):
            raise RuntimeError(item)

        with self.assertRaises(RuntimeError):
            worker_module.Worker.each_item(["x"], make, keep_going=False)
        with patch("traceback.print_exc"):
            self.assertEqual(worker_module.Worker.each_item(["x"], make, keep_going=True),
                             {"outputs": [None], "errors": ["RuntimeError: x"]})

    def test_load_request_preloads_the_named_model_without_generating(self):
        worker = worker_module.Worker(self.root)
        with patch.object(worker, "load_stable_audio") as load_sound, \
                patch.object(worker, "load_voice_design") as load_voice:
            self.assertEqual(worker.handle({"op": "load", "model": "stable_audio"}), {})
            load_sound.assert_called_once_with()
            load_voice.assert_not_called()
        with self.assertRaises(KeyError):
            worker.handle({"op": "load", "model": "drama3"})

    def test_broken_flash_attention_falls_back_to_sdpa(self):
        worker = worker_module.Worker(self.root)
        with patch.dict(sys.modules, {"flash_attn": None}), patch("builtins.print"):
            self.assertFalse(worker.flash_attention())

    def test_serve_answers_each_request_and_survives_a_failure(self):
        worker = Mock()
        worker.loaded = ["stable_audio"]
        worker.handle.side_effect = [{"outputs": ["a.wav"]}, RuntimeError("CUDA out of memory")]
        replies = io.StringIO()
        lines = [json.dumps({"id": 1, "op": "stable_audio"}).encode(), b"\n",
                 json.dumps({"id": 2, "op": "voice_design"}).encode()]
        with patch.object(worker_module, "Worker", return_value=worker), patch("traceback.print_exc"):
            worker_module.serve(str(self.root), requests=lines, replies=replies)
        answers = [json.loads(line) for line in replies.getvalue().splitlines()]
        self.assertEqual(answers[0], {"id": 1, "loaded": ["stable_audio"], "ok": True, "outputs": ["a.wav"]})
        self.assertEqual((answers[1]["id"], answers[1]["ok"]), (2, False))
        self.assertIn("CUDA out of memory", answers[1]["error"])

    def test_weights_are_ready_only_when_complete(self):
        self.assertFalse(worker_module.weights_ready("voice_design", self.root))
        cache = worker_module.voice_design_cache(self.root)
        cache.mkdir(parents=True)
        (cache / worker_module.VOICE_DESIGN_MARKER).write_text("done")
        self.assertTrue(worker_module.weights_ready("voice_design", self.root))
        model_dir = worker_module.stable_audio_dir(self.root)
        for name in worker_module.STABLE_AUDIO_FILES[:-1]:
            (model_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (model_dir / name).write_text("x")
        self.assertFalse(worker_module.weights_ready("stable_audio", self.root))
        (model_dir / worker_module.STABLE_AUDIO_FILES[-1]).write_text("x")
        self.assertTrue(worker_module.weights_ready("stable_audio", self.root))

    def test_config_points_hub_references_at_the_local_snapshot(self):
        model_dir = self.root / "medium"
        (model_dir / worker_module.T5GEMMA_SUBFOLDER).mkdir(parents=True)
        (model_dir / "vae.ckpt").write_text("x")
        config = {"model": {"conditioning": {"configs": [
            {"type": "t5gemma", "config": {"model_name": worker_module.T5GEMMA_ID,
                                           "repo_id": worker_module.STABLE_AUDIO_ID,
                                           "subfolder": worker_module.T5GEMMA_SUBFOLDER}},
            {"type": "t5gemma", "config": {"model_name": worker_module.T5GEMMA_ID}},
        ]}, "pretransform": {"pretransform_ckpt_path": "weights/vae.ckpt", "ckpt_path": "missing.ckpt"}}}
        patched = worker_module.patch_stable_audio_config(config, model_dir)
        first, second = (entry["config"] for entry in patched["model"]["conditioning"]["configs"])
        self.assertEqual(first, {"model_name": worker_module.T5GEMMA_ID, "model_path": str(model_dir),
                                 "subfolder": worker_module.T5GEMMA_SUBFOLDER})
        self.assertEqual(second["model_path"], str(model_dir / worker_module.T5GEMMA_SUBFOLDER))
        self.assertEqual(patched["model"]["pretransform"], {"pretransform_ckpt_path": str(model_dir / "vae.ckpt")})


if __name__ == "__main__":
    unittest.main()
