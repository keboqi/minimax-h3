"""Standalone entry point for OpenMOSS's Transformers inference API."""

import os
from pathlib import Path
import sys


def transcribe(checkpoint: str, audio: str, max_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS (Python) requires a CUDA GPU.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Loading MOSS-Transcribe-Diarize weights", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)

    print("Finding speakers and transcribing", flush=True)
    result = generate_transcription(
        model, processor, build_transcription_messages(audio),
        max_new_tokens=max_tokens, do_sample=False, device=device, dtype=dtype,
    )
    if result["generated_tokens"] >= max_tokens:
        raise RuntimeError("MOSS reached its output token limit. Trim the source or use SGLang-Omni.")
    return result["text"]


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    checkpoint, audio, output, max_tokens = sys.argv[1:]
    Path(output).write_text(transcribe(checkpoint, audio, int(max_tokens)), encoding="utf-8")
