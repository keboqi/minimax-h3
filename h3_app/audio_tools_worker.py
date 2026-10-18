"""Standalone entry point: run only with the isolated audio tools interpreter.

``serve`` keeps Qwen VoiceDesign and Stable Audio 3 Medium resident: each model
loads on its first request and stays in VRAM until the parent stops the process.
Requests arrive one JSON line at a time on stdin; replies go to the original
stdout, and everything the libraries print goes to stderr (the log). Serving runs
with HF_HUB_OFFLINE=1; ``download`` fetches missing weights once beforehand.

The parent imports this module for the weight locations, so keep heavy imports
inside functions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

VOICE_DESIGN_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
STABLE_AUDIO_ID = "stabilityai/stable-audio-3-medium"
T5GEMMA_ID = "google/t5gemma-b-b-ul2"
# The Stable Audio 3 Medium repo bundles its T5Gemma text encoder in this folder.
T5GEMMA_SUBFOLDER = "t5gemma-b-b-ul2"
VOICE_DESIGN_MARKER = ".h3-complete-qwen3-tts-voice-design"
STABLE_AUDIO_FILES = ("model_config.json", "model.safetensors",
                      f"{T5GEMMA_SUBFOLDER}/config.json", f"{T5GEMMA_SUBFOLDER}/model.safetensors",
                      f"{T5GEMMA_SUBFOLDER}/tokenizer.json")
MAX_SOUND_SECONDS = 47


def voice_design_cache(models_dir: Path) -> Path:
    """Hugging Face cache layout, so earlier per-clip worker downloads are reused."""
    return Path(models_dir) / "qwen3_tts"


def stable_audio_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "stable_audio3" / "medium"


def weights_ready(model: str, models_dir: Path) -> bool:
    if model == "voice_design":
        return (voice_design_cache(models_dir) / VOICE_DESIGN_MARKER).is_file()
    if model == "stable_audio":
        root = stable_audio_dir(models_dir)
        return all((root / name).is_file() for name in STABLE_AUDIO_FILES)
    raise ValueError(f"Unknown audio tools model: {model}")


def download(model: str, models_dir: str) -> None:
    """Fetch one model's weights (hf_xet when the repo supports it) and mark them complete."""
    from huggingface_hub import snapshot_download

    models = Path(models_dir)
    if model == "voice_design":
        cache = voice_design_cache(models)
        snapshot_download(VOICE_DESIGN_ID, cache_dir=str(cache))
        (cache / VOICE_DESIGN_MARKER).write_text(VOICE_DESIGN_ID, encoding="utf-8")
    elif model == "stable_audio":
        # Gated: the parent passes HF_TOKEN. Files land atomically, so their presence marks completion.
        snapshot_download(STABLE_AUDIO_ID, local_dir=str(stable_audio_dir(models)))
    else:
        raise ValueError(f"Unknown audio tools model: {model}")
    if not weights_ready(model, models):
        raise RuntimeError(f"{model} download finished without its model files.")


def patch_stable_audio_config(model_config: dict, model_dir: Path) -> dict:
    """Point Hub references in the shipped config at the local snapshot."""
    text_encoder = model_dir / T5GEMMA_SUBFOLDER

    def patch(node):
        if isinstance(node, dict):
            subfolder = node.get("subfolder")
            if node.get("repo_id") == STABLE_AUDIO_ID and subfolder and (model_dir / subfolder).exists():
                node["model_path"] = str(model_dir)
                node.pop("repo_id", None)
            elif node.get("model_name") == T5GEMMA_ID and not node.get("model_path") and text_encoder.is_dir():
                node.pop("repo_id", None)
                if subfolder and (model_dir / subfolder).exists():
                    node["model_path"] = str(model_dir)
                else:
                    node["model_path"] = str(text_encoder)
                    node.pop("subfolder", None)
            for key in ("ckpt_path", "pretransform_ckpt_path"):
                value = node.get(key)
                if isinstance(value, str):
                    candidate = model_dir / Path(value).name
                    if candidate.is_file():
                        node[key] = str(candidate)
                    else:
                        del node[key]
            for value in node.values():
                patch(value)
        elif isinstance(node, list):
            for entry in node:
                patch(entry)

    patch(model_config)
    return model_config


class Worker:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.voice_design = None
        self.stable_audio = None
        self._flash = None

    @property
    def loaded(self) -> list[str]:
        return [name for name in ("voice_design", "stable_audio") if getattr(self, name) is not None]

    def torch(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen VoiceDesign and Stable Audio 3 Medium need an available CUDA GPU.")
        return torch

    def dtype(self):
        torch = self.torch()
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def flash_attention(self) -> bool:
        if self._flash is None:
            try:
                import flash_attn  # noqa: F401  (loads the CUDA extension)

                self._flash = True
            except Exception as exc:  # a broken wheel must not block generation
                print(f"[audio-tools] FlashAttention unavailable, using SDPA: {exc}", file=sys.stderr, flush=True)
                self._flash = False
        return self._flash

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "voice_design":
            return self.design_voice(request)
        if op == "stable_audio":
            return self.generate_sounds(request)
        if op == "ping":
            return {}
        raise ValueError(f"Unknown audio tools request: {op!r}")

    def load_voice_design(self):
        if self.voice_design is None:
            from huggingface_hub import snapshot_download
            from qwen_tts import Qwen3TTSModel

            self.torch()
            model_dir = snapshot_download(VOICE_DESIGN_ID, cache_dir=str(voice_design_cache(self.models_dir)),
                                          local_files_only=True)
            self.voice_design = Qwen3TTSModel.from_pretrained(
                model_dir, device_map="cuda:0", dtype=self.dtype(),
                attn_implementation="flash_attention_2" if self.flash_attention() else "sdpa",
            )
        return self.voice_design

    def design_voice(self, request: dict) -> dict:
        import numpy as np
        import soundfile as sf

        model = self.load_voice_design()
        with self.torch().inference_mode():
            wavs, rate = model.generate_voice_design(
                text=request["text"], language=request["language"], instruct=request["description"],
                max_new_tokens=4096,
            )
        samples = np.asarray(wavs[0])
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("Qwen VoiceDesign returned empty or invalid audio.")
        sf.write(request["output"], samples, rate, subtype="PCM_16")
        return {"outputs": [request["output"]]}

    def load_stable_audio(self):
        if self.stable_audio is None:
            from stable_audio_tools.models.factory import create_model_from_config
            from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

            torch = self.torch()
            self.flash_attention()  # report a broken wheel; stable-audio-tools picks it up on import
            model_dir = stable_audio_dir(self.models_dir)
            model_config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
            model_config = patch_stable_audio_config(model_config, model_dir)

            model = create_model_from_config(model_config)
            state_dict = load_ckpt_state_dict(str(model_dir / "model.safetensors"))
            ema_prefix = "diffusion_ema.ema_model."
            ema_state = {key[len(ema_prefix):]: value for key, value in state_dict.items()
                         if key.startswith(ema_prefix)}
            if ema_state and hasattr(model, "model"):
                copy_state_dict(model.model, ema_state)
            else:
                copy_state_dict(model, state_dict)
            del state_dict, ema_state

            model.to(torch.device("cuda:0")).eval().requires_grad_(False)
            model.to(self.dtype())
            if getattr(model, "pretransform", None) is not None and not getattr(model.pretransform, "enable_half", False):
                model.pretransform.to(torch.float32)
            self.stable_audio = (model, int(model_config["sample_rate"]), int(model_config["sample_size"]))
        return self.stable_audio

    def generate_sounds(self, request: dict) -> dict:
        import numpy as np
        import soundfile as sf
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

        torch = self.torch()
        model, sample_rate, sample_size = self.load_stable_audio()
        outputs = []
        with torch.inference_mode():
            for item in request["items"]:
                prompt = str(item["prompt"]).strip()
                duration = max(1, min(int(item.get("duration", 5)), MAX_SOUND_SECONDS))
                conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration}]
                # Unknown keywords pass through **sampler_kwargs down to the transformer blocks,
                # so only the function's own parameters and the sampler's go here.
                output = generate_diffusion_cond_inpaint(
                    model,
                    steps=int(item.get("steps", 8)),
                    cfg_scale=float(item.get("cfg_scale", 1.0)),
                    conditioning=conditioning,
                    sample_size=sample_size,
                    sigma_max=1.0,
                    sampler_type="pingpong",
                    device="cuda:0",
                    seed=int(item.get("seed", -1)),
                    apg_scale=1.0,
                )
                # Generation runs about six seconds past seconds_total; keep only the requested length.
                output = output[..., : duration * sample_rate].squeeze(0).to(torch.float32)
                peak = torch.max(torch.abs(output))
                if peak > 0:
                    output = output.div(peak).clamp(-1.0, 1.0)
                audio = output.cpu().numpy()
                if audio.ndim == 2:
                    # Channels-first (C, N) -> mono (N,)
                    audio = np.mean(audio, axis=0)
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not audio.size or not np.isfinite(audio).all():
                    raise RuntimeError(f"Stable Audio 3 Medium returned empty or invalid audio for '{prompt}'.")
                sf.write(item["output"], audio, sample_rate, subtype="PCM_16")
                outputs.append(item["output"])
        return {"outputs": outputs}


def serve(models_dir: str, requests=None, replies=None) -> None:
    if replies is None:
        # Keep the protocol on its own descriptor; stray prints (including C-level) go to the log.
        replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
    worker = Worker(Path(models_dir))
    for line in requests if requests is not None else sys.stdin.buffer:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            reply = {"ok": True, **worker.handle(request)}
        except Exception as exc:
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stderr.flush()
        replies.write(json.dumps({"id": request.get("id"), "loaded": worker.loaded, **reply}) + "\n")
        replies.flush()


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "serve":
        serve(sys.argv[2])
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")
