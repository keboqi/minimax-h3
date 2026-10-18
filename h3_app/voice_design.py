"""Qwen VoiceDesign references from the shared resident audio tools worker.

The model runs in :mod:`h3_app.audio_tools`' isolated environment and stays loaded
between clips; never install qwen-tts in ComfyUI's environment.
"""

from __future__ import annotations

import uuid

from h3_app import audio_tools
from h3_app.errors import H3Error

MODEL_ID = audio_tools.layout.VOICE_DESIGN_ID
LANGUAGES = ("Auto", "Chinese", "English", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian")


def validate_inputs(description, text, language):
    description, text = str(description or "").strip(), str(text or "").strip()
    if not description or len(description) > 2000:
        raise H3Error("Describe the voice in 1–2000 characters.")
    if not text or len(text) > 1000:
        raise H3Error("Add 1–1000 characters of preview speech for the reference clip.")
    if any(char in text for char in "<>[]"):
        raise H3Error("Preview text must contain spoken words without speaker tags or delivery controls.")
    if language not in LANGUAGES:
        raise H3Error("Choose a supported VoiceDesign language.")
    return description, text, language


def generate_reference(description, text, language, runtime, *, free_gpu, hf_token=None):
    description, text, language = validate_inputs(description, text, language)
    output = runtime.output_dir / "drama3_voices"
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"QwenVoiceDesign_{uuid.uuid4().hex}.wav"
    try:
        audio_tools.run("voice_design", {"description": description, "text": text, "language": language,
                                         "output": str(result.resolve())},
                        runtime, free_gpu=free_gpu, hf_token=hf_token)
        if not result.is_file() or result.stat().st_size <= 44:
            raise H3Error("Qwen VoiceDesign returned no audio.")
    except BaseException:
        result.unlink(missing_ok=True)
        raise
    return result, text
