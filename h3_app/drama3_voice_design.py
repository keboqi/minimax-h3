"""Drama 3 voice design: a description cue casts one invented voice for a reference take.

The take runs on SGLang-Omni with the tab's sampling and no references, so the
model creates the voice from the cue. The preview speech, without the cue, is
the clip's exact transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import uuid

from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.voice_design import validate_inputs

ENGINE_DRAMA3 = "Drama 3"
ENGINE_QWEN = "Qwen3-TTS VoiceDesign"
VOICE_DESIGN_ENGINES = (ENGINE_DRAMA3, ENGINE_QWEN)
# Drama 3's reliability rules: speech opens the line; the first cue follows the opening phrase.
_OPENING = re.compile(r"[,;:.!?，、；：。！？](?=\s*\S)")


def design_script(description: str, text: str) -> str:
    """``Opening phrase, [description] rest of the preview.`` Drama 3 gets only this text."""
    description, text, _language = validate_inputs(description, text, "Auto")
    description = " ".join(description.split())
    if any(char in description for char in "<>[]|"):
        raise H3Error("The voice description is a Drama 3 cue; remove brackets, angle brackets and pipes.")
    text = " ".join(text.split())
    match = _OPENING.search(text)
    if match:
        cut = match.end()
    else:
        space = text.find(" ")
        # Unspaced scripts (Chinese, Japanese) without punctuation: after the first two characters.
        cut = space if space > 0 else min(2, len(text))
    head, tail = text[:cut].strip(), text[cut:].strip()
    return f"{head} [{description}] {tail}".rstrip()


def _sampling(request) -> dict:
    return {"temperature": float(request.temperature), "top_p": float(request.top_p), "top_k": int(request.top_k)}


def generate_drama3_reference(description, text, request, runtime, services, *, free_gpu, hf_token=None):
    """Yield status updates, then a final update whose output is the clip path.

    ``request`` is the tab's ``Drama3Request`` (sampling and frame cap); its text and
    voices are ignored. Raises ``H3Error`` on failure.
    """
    from h3_app.drama3_serving.generate import design_voice_served

    script = design_script(description, text)
    transcript = " ".join(str(text).split())
    target = runtime.output_dir / "drama3_voices" / f"Drama3VoiceDesign_{uuid.uuid4().hex}.wav"
    services.models.unload_prompt_rewriter()
    yield from design_voice_served(script, target, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token,
                                   sampling=_sampling(request), max_frames=int(request.max_frames))
    yield GenerationUpdate(str(target), transcript)


@dataclass(frozen=True)
class DesignResult:
    key: object
    audio: str | None
    transcript: str
    error: str = ""


def generate_drama3_references(briefs, request, runtime, services, *, free_gpu, hf_token=None):
    """Design ``{key: (description, preview)}``; yield status updates and one ``DesignResult`` each.

    SGLang-Omni sends the takes concurrently, up to ``request.parallel``. Failures stay per voice.
    """
    from h3_app.drama3_serving.generate import DesignedVoice, design_voices_served

    scripts = {}
    for key, (description, text) in briefs.items():
        try:
            scripts[key] = (design_script(description, text), " ".join(str(text).split()))
        except H3Error as exc:
            yield DesignResult(key, None, "", str(exc))
    if not scripts:
        return
    folder = runtime.output_dir / "drama3_voices"
    takes = {key: (script, folder / f"Drama3VoiceDesign_{uuid.uuid4().hex}.wav")
             for key, (script, _transcript) in scripts.items()}
    services.models.unload_prompt_rewriter()
    for update in design_voices_served(takes, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token,
                                       sampling=_sampling(request), max_frames=int(request.max_frames),
                                       parallel=int(request.parallel)):
        if isinstance(update, DesignedVoice):
            transcript = scripts[update.key][1]
            yield (DesignResult(update.key, str(takes[update.key][1]), transcript) if update.error is None
                   else DesignResult(update.key, None, transcript, str(update.error)))
        else:
            yield update
