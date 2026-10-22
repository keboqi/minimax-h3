"""Drama 3 requests: validation, generation on SGLang-Omni and Qwen3-ASR transcription."""

import time
import uuid

from h3_app.errors import H3Error
from h3_app.jobs import record_failure
from h3_app.progress import no_progress
from h3_app.status import StageTimings
from h3_app.workflows.drama3 import (
    MAX_FRAMES, MAX_SPEAKER, build_drama3_asr_graph, drama3_asr_model_keys, request_cast, speaker_label,
    unknown_speaker_names, validate_drama3_text,
)
from h3_app.drama3_serving.tree import script_speakers
from h3_models import DEFAULT_DRAMA3_ASR

from .checks import whole_number
from .results import GenerationUpdate

DRAMA3_BACKEND = "SGLang-Omni"


def validate_drama3(request):
    """SGLang-Omni takes no per-request seed; every spoken speaker needs a voice."""
    validate_drama3_text(request.text)
    speakers = [speaker for speaker, _audio, _transcript in request.voices]
    if len(set(speakers)) != len(speakers):
        raise H3Error("Each speaker can have only one reference voice.")
    for speaker, audio, transcript in request.voices:
        whole_number(speaker, 0, MAX_SPEAKER, "Speaker number")
        if not audio or not str(transcript).strip():
            raise H3Error(f"Speaker {speaker} needs both a reference clip and its exact transcript.")
        validate_drama3_text(transcript, f"Speaker {speaker}'s transcript")
    from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_MODES

    if request.script_mode not in DRAMA3_SCRIPT_MODES:
        raise H3Error("Choose Basic or Advanced (sound enhanced) Drama 3 script mode.")
    whole_number(request.parallel, 1, 32, "Parallel requests")
    whole_number(request.retries, 0, 4, "Retries per chunk")
    # Each chunk is sent with its speaker's own clip, so every voice must be given.
    names, default_speaker, resolve = request_cast(request)
    unknown = unknown_speaker_names(request.text, names)
    if unknown:
        raise H3Error("No speaker in Cast is named " + ", ".join(f'"{name}"' for name in unknown)
                      + '. Give a Cast slot that name, or start the line with its slot, such as "Speaker 2:".')
    voiced = {speaker_label(speaker, names) for speaker, _a, _t in request.voices}
    spoken = script_speakers(request.text, default_speaker=default_speaker, resolve=resolve)
    missing = [speaker for speaker in spoken if speaker not in voiced]
    if missing:
        raise H3Error(
            "Drama 3 sends each speaker's lines with that speaker's reference voice. Add a clip, "
            "saved speaker or generated voice for " + ", ".join(missing) + "."
        )
    for name, low, high in (("temperature", 0, 2), ("top_p", 0.05, 1)):
        if not low <= float(getattr(request, name)) <= high:
            raise H3Error(f"{name} must be between {low} and {high}.")
    whole_number(request.top_k, 0, 4096, "Top-k")
    whole_number(request.max_frames, 1, MAX_FRAMES, "Maximum audio frames")


def generate_drama3(request, services, runtime, *, progress=no_progress, free_gpu=None):
    """``free_gpu`` releases ComfyUI's VRAM before Drama 3 starts or wakes SGLang-Omni."""
    timings = StageTimings("Drama 3", time.monotonic(), "Validating request")
    try:
        validate_drama3(request)
        from h3_app.drama3_serving.generate import generate_drama3_served
        from h3_models import resolve_hf_token

        services.models.unload_prompt_rewriter()
        timings.transition(f"{DRAMA3_BACKEND} Drama 3")
        yield from generate_drama3_served(
            request, runtime=runtime, free_gpu=free_gpu or (lambda: None),
            hf_token=resolve_hf_token(), progress=progress,
        )
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()


def transcribe_drama3_audio(audio, services, *, language="auto", model_choice=DEFAULT_DRAMA3_ASR):
    """Transcribe a staged clip with Qwen3-ASR through ComfyUI; returns the text."""
    graph, preview = build_drama3_asr_graph(audio=audio, language=language, model_choice=model_choice)
    missing = {node["class_type"] for node in graph.values()} - set(services.execution.object_info())
    if missing:
        raise H3Error("Update setup and restart ComfyUI (the pinned ComfyUI-Drama3 Qwen3-ASR nodes are required). "
                      f"Missing nodes: {', '.join(sorted(missing))}")
    services.models.ensure_model_keys(drama3_asr_model_keys(model_choice))
    prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
    for _update in services.execution.poll_comfy_progress(prompt_id, graph):
        pass
    history = services.execution.wait_for_history(prompt_id)
    texts = history.get("outputs", {}).get(preview, {}).get("text") or []
    transcript = " ".join(str(text).strip() for text in texts if str(text).strip())
    if not transcript or transcript == "None":
        raise H3Error("Qwen3-ASR heard no speech in this clip. Use a few seconds of clean speech.")
    return transcript
