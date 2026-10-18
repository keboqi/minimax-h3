"""Drama 3 voice design: a description cue casts one invented voice for a reference take.

The take runs on the Drama 3 backend selected for the tab, without references, so
the model creates the voice from the cue. The preview speech, without the cue, is
the clip's exact transcript.
"""

from __future__ import annotations

from dataclasses import replace
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
    """``<|speaker:0|> Opening phrase, [description] rest of the preview.``"""
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
    return f"<|speaker:0|> {head} [{description}] {tail}".rstrip()


def generate_drama3_reference(description, text, request, runtime, services, *, free_gpu, hf_token=None,
                              progress=None):
    """Yield status updates, then a final update whose output is the clip path.

    ``request`` is the tab's ``Drama3Request`` (backend, model and sampling); its
    text and voices are replaced. Raises ``H3Error`` on failure.
    """
    from h3_app.drama3_serving.generate import design_voice_served, frame_budget
    from h3_app.generation.optional_engines import DRAMA3_SERVED_BACKENDS, generate_optional_engine

    script = design_script(description, text)
    transcript = " ".join(str(text).split())
    if request.backend in DRAMA3_SERVED_BACKENDS:
        target = runtime.output_dir / "drama3_voices" / f"Drama3VoiceDesign_{uuid.uuid4().hex}.wav"
        services.models.unload_prompt_rewriter()
        sampling = {"temperature": float(request.temperature), "top_p": float(request.top_p),
                    "top_k": int(request.top_k)}
        yield from design_voice_served(script, target, runtime=runtime, free_gpu=free_gpu, hf_token=hf_token,
                                       sampling=sampling, max_frames=int(request.max_frames),
                                       label=request.backend)
        yield GenerationUpdate(str(target), transcript)
        return
    # One take on ComfyUI: no splitting, and a frame cap sized to the preview.
    request = replace(request, text=script, voices=(), voice_names=(), split_long_text=False,
                      max_frames=frame_budget(script, int(request.max_frames)))
    kwargs = {"progress": progress} if progress is not None else {}
    result = None
    for update in generate_optional_engine(request, services, runtime, free_gpu=free_gpu, **kwargs):
        if isinstance(update.output, dict):
            result = update.output.get("value")
        elif update.output is None and update.status.startswith("Error: "):
            raise H3Error(update.status.removeprefix("Error: ").split("\n\n", 1)[0])
        else:
            yield GenerationUpdate(None, update.status)
    if not result:
        raise H3Error("Drama 3 returned no voice clip.")
    yield GenerationUpdate(str(result), transcript)
