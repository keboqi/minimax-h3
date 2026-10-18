"""Standalone entry point: run only with the isolated VoiceDesign interpreter."""

import json
import os
from pathlib import Path
import sys


def synthesize(request):
    # Set this before importing Hub/Transformers: nested tokenizer/processor loads
    # must use the persistent models volume too.
    os.environ["HF_HUB_CACHE"] = request["cache_dir"]
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen VoiceDesign needs an available CUDA GPU.")
    model_dir = snapshot_download(request["model"], cache_dir=request["cache_dir"])
    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="cuda:0",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa",
    )
    with torch.inference_mode():
        wavs, rate = model.generate_voice_design(
            text=request["text"], language=request["language"], instruct=request["description"],
            max_new_tokens=4096,
        )
    samples = np.asarray(wavs[0])
    if not samples.size or not np.isfinite(samples).all():
        raise RuntimeError("Qwen VoiceDesign returned empty or invalid audio.")
    sf.write(request["output"], samples, rate, subtype="PCM_16")
    # Process exit releases the model and all CUDA allocations.


if __name__ == "__main__":
    synthesize(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
