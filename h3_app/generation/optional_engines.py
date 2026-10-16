"""FLUX 3 and Drama 3 requests over the shared, owned ComfyUI queue."""

from dataclasses import asdict
import math
import random
import time
import uuid

from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX
from h3_app.errors import H3Error
from h3_app.jobs import CURRENT_JOB, record_failure, variant_seed
from h3_app.progress import no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import StageTimings, progress_status
from h3_app.workflows.flux3 import build_flux3_graph, flux3_model_keys
from h3_app.workflows.drama3 import (
    MAX_FRAMES, MAX_SPEAKER, build_drama3_asr_graph, build_drama3_graph, text_speakers, validate_drama3_text,
)
from h3_models import DRAMA3_ASR_MODEL_KEYS, DRAMA3_MODEL_KEYS
from .qwen import first_reference_dimensions
from .requests import Flux3Request, Drama3Request
from .results import GenerationUpdate, fresh_audio


DRAMA3_BACKENDS = ("ComfyUI", "vLLM-Omni", "SGLang-Omni")
DRAMA3_SERVED_BACKENDS = DRAMA3_BACKENDS[1:]
FLUX3_EDIT_SIZES = (QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL)
FLUX3_MAX_PIXELS, FLUX3_MIN_SIDE, FLUX3_MAX_SIDE, FLUX3_GRID = 4_194_304, 256, 4096, 16


def _integer(value, low, high, name, multiple=1):
    if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value:
        raise H3Error(f"{name} must be a whole number.")
    value = int(value)
    if not low <= value <= high or value % multiple:
        raise H3Error(f"{name} must be {low}–{high}, in increments of {multiple}.")
    return value


def flux3_edit_dimensions(source_width, source_height, *, maximize=False):
    """Fit the first reference's aspect ratio on FLUX's 16-pixel grid.

    Match keeps the native area (downscaled to the 4 MP cap); Max scales toward 4 MP.
    """
    source_width, source_height = int(source_width), int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise H3Error("The first reference image has invalid dimensions.")
    lo, hi, grid = FLUX3_MIN_SIDE, FLUX3_MAX_SIDE, FLUX3_GRID
    source_area = source_width * source_height
    target_area = FLUX3_MAX_PIXELS if maximize else min(source_area, FLUX3_MAX_PIXELS)
    scale = min(math.sqrt(target_area / source_area), hi / source_width, hi / source_height)
    target_area = max(lo * lo, min(hi, max(lo, source_width * scale)) * min(hi, max(lo, source_height * scale)))
    source_ratio = source_width / source_height
    # Search both sides jointly so common ratios stay exact where the grid allows.
    candidates = [
        (width, height)
        for width in range(lo, hi + 1, grid)
        for height in range(lo, hi + 1, grid)
        if target_area * 0.85 <= width * height <= min(target_area, FLUX3_MAX_PIXELS)
    ]
    if not candidates:
        raise H3Error("The first reference image's aspect ratio is outside FLUX 3's supported canvas.")
    def drift(size):
        return abs(math.log((size[0] / size[1]) / source_ratio))
    # Up to 1% ratio drift is invisible; prefer the larger canvas within it (1920×1088, not 1792×1008).
    close = [size for size in candidates if drift(size) <= 0.01]
    if close:
        return max(close, key=lambda size: (size[0] * size[1], -drift(size)))
    return min(candidates, key=lambda size: (drift(size), -(size[0] * size[1])))


def resolve_flux3_dimensions(request, *, input_dir=None):
    """Edits default to the first reference's size; manual size applies otherwise."""
    if not request.reference_images or request.edit_size == QWEN_EDIT_SIZE_MANUAL:
        return int(request.width), int(request.height)
    size = first_reference_dimensions(request.reference_images[0], input_dir)
    return flux3_edit_dimensions(*size, maximize=request.edit_size == QWEN_EDIT_SIZE_MAX)


def validate_request(request):
    _integer(request.seed, -1, 2**63 - 2, "Seed")
    if isinstance(request, Flux3Request):
        if not request.prompt.strip():
            raise H3Error("Write an image prompt or complete edit caption.")
        flux3_model_keys(request.model_choice, request.expand_prompt)
        _integer(request.width, 256, 4096, "Width", 16)
        _integer(request.height, 256, 4096, "Height", 16)
        _integer(request.steps, 1, 100, "Steps")
        if request.edit_size not in FLUX3_EDIT_SIZES:
            raise H3Error("Choose a valid FLUX 3 edit output size.")
        if len(request.reference_images) > 10:
            raise H3Error("FLUX 3 supports at most ten reference images.")
        if request.reference_images and request.expand_prompt:
            raise H3Error("Disable prompt expansion for edits and write a full scene caption including your changes.")
    elif isinstance(request, Drama3Request):
        validate_drama3_text(request.text)
        speakers = [speaker for speaker, _audio, _transcript in request.voices]
        if len(set(speakers)) != len(speakers):
            raise H3Error("Each speaker can have only one reference voice.")
        for speaker, audio, transcript in request.voices:
            _integer(speaker, 0, MAX_SPEAKER, "Speaker number")
            if not audio or not str(transcript).strip():
                raise H3Error(f"Speaker {speaker} needs both a reference clip and its exact transcript.")
            validate_drama3_text(transcript, f"Speaker {speaker}'s transcript")
        if request.experts not in {"in RAM", "in VRAM"}:
            raise H3Error("Choose experts in RAM or in VRAM.")
        if request.backend not in DRAMA3_BACKENDS:
            raise H3Error("Choose the ComfyUI, vLLM-Omni or SGLang-Omni Drama 3 backend.")
        _integer(request.parallel, 1, 32, "Parallel requests")
        _integer(request.retries, 0, 4, "Retries per chunk")
        if request.backend in DRAMA3_SERVED_BACKENDS:
            # Each chunk is sent with its speaker's own clip, so every voice must be given.
            missing = sorted(text_speakers(request.text) - {speaker for speaker, _a, _t in request.voices})
            if missing:
                raise H3Error(
                    f"The {request.backend} backend sends each speaker's lines with that speaker's reference voice. "
                    "Add a clip or saved speaker for " + ", ".join(f"<|speaker:{n}|>" for n in missing)
                    + ", or use the ComfyUI backend to let Drama 3 create voices in one take."
                )
        for name, low, high in (("temperature", 0, 2), ("top_p", 0.05, 1), ("expert_cache_gb", 0, 256)):
            if not low <= float(getattr(request, name)) <= high:
                raise H3Error(f"{name} must be between {low} and {high}.")
        _integer(request.top_k, 1, 4096, "Top-k")
        _integer(request.max_frames, 1, MAX_FRAMES, "Maximum audio frames")
    else:
        raise TypeError("Unsupported generation request.")


def generate_optional_engine(request, services, runtime, *, progress=no_progress, free_gpu=None):
    """Validate before downloads, retain retry seeds, and resolve only this job's output.

    ``free_gpu`` releases ComfyUI's VRAM before the optional vLLM-Omni Drama 3 route runs.
    """
    image = isinstance(request, Flux3Request)
    family = "FLUX 3 Image Klein" if image else "Drama 3"
    started = time.monotonic()
    timings = StageTimings(family, started, "Validating request")
    try:
        validate_request(request)
        job = CURRENT_JOB.get()
        seed = variant_seed(job.variant if job else 0,
                            lambda: random.randrange(2**63 - 1) if request.seed < 0 else int(request.seed))
        values = {**asdict(request), "seed": seed}
        if not image and request.backend in DRAMA3_SERVED_BACKENDS:
            from h3_app.drama3_serving.generate import generate_drama3_served
            from h3_models import resolve_hf_token

            services.models.unload_prompt_rewriter()
            timings.transition(f"{request.backend} Drama 3")
            yield from generate_drama3_served(
                request, runtime=runtime, free_gpu=free_gpu or (lambda: None),
                hf_token=resolve_hf_token(), seed=seed, progress=progress,
            )
            return
        if image:
            values["width"], values["height"] = resolve_flux3_dimensions(request, input_dir=runtime.input_dir)
            # The shared image resolver only reads the managed staging directory.
            graph = build_flux3_graph(
                **{k: v for k, v in values.items() if k != "edit_size"},
                output_stamp=str(int(time.time())), output_nonce=uuid.uuid4().hex[:8],
                system_prompt=runtime.prompt_systems["FLUX 3 Image Klein"].read_text(encoding="utf-8").strip(),
            )
        else:
            from h3_app.drama3_serving.server import release_all

            release_all()  # the ComfyUI route needs the VRAM a served backend holds
            graph = build_drama3_graph(**{key: value for key, value in values.items()
                                          if key not in {"backend", "parallel", "retries", "voice_names"}})
        info = services.execution.object_info()
        missing = {node["class_type"] for node in graph.values()} - set(info)
        if missing:
            raise H3Error(f"Update setup and restart ComfyUI (v0.41.0 and pinned Drama 3 nodes required). Missing nodes: {', '.join(sorted(missing))}")
        if image:
            clip_types = info.get("CLIPLoader", {}).get("input", {}).get("required", {}).get("type", [[]])[0]
            if "flux3" not in clip_types:
                raise H3Error("This backend lacks the FLUX 3 text encoder. Update setup and restart ComfyUI.")
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(None, f"Preparing {family} models; missing weights download on first use.")
        keys = flux3_model_keys(request.model_choice, request.expand_prompt) if image else DRAMA3_MODEL_KEYS
        services.models.ensure_model_keys(keys)
        queued_at = time.time()
        prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
        size = f" · {values['width']}×{values['height']}" if image else ""
        yield GenerationUpdate(None, f"Queued {family} job {prompt_id} · seed {seed}{size}")
        for stage, done, total, step, steps in services.execution.poll_comfy_progress(prompt_id, graph):
            timings.transition(stage)
            if step is not None and steps:
                progress((step, steps), desc=stage)
            elif total:
                progress((done, total), desc=stage)
            yield GenerationUpdate(None, progress_status(stage, started=started,
                completed_nodes=done, total_nodes=total, step=step, step_total=steps))
        history = services.execution.wait_for_history(prompt_id)
        result = (services.media.resolve_image_outputs(history, queued_at, 1)[0] if image
                  else services.media.resolve_audio_output(history, queued_at))
        settings = {key: value for key, value in values.items()
                    if key not in {"prompt", "text", "voices", "reference_images"}}
        if image:
            settings["reference_count"] = len(request.reference_images)
        else:
            settings["reference_count"] = len(request.voices)
            settings["cloned_speakers"] = sorted(speaker for speaker, _a, _t in request.voices)
            settings["text_speakers"] = sorted(text_speakers(request.text))
        write_snapshot(result, {"job_id": prompt_id, "family": family, "settings": settings})
        progress(1, desc="Complete")
        yield GenerationUpdate([str(result)] if image else fresh_audio(result),
                               f"{family} complete · seed {seed}\n\n{timings.summary()}")
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()


def transcribe_drama3_audio(audio, services, *, language="auto"):
    """Transcribe a staged clip with Qwen3-ASR through ComfyUI; returns the text."""
    graph, preview = build_drama3_asr_graph(audio=audio, language=language)
    missing = {node["class_type"] for node in graph.values()} - set(services.execution.object_info())
    if missing:
        raise H3Error("Update setup and restart ComfyUI (pinned Drama 3 nodes with Qwen3-ASR required). "
                      f"Missing nodes: {', '.join(sorted(missing))}")
    services.models.ensure_model_keys(DRAMA3_ASR_MODEL_KEYS)
    prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
    for _update in services.execution.poll_comfy_progress(prompt_id, graph):
        pass
    history = services.execution.wait_for_history(prompt_id)
    texts = history.get("outputs", {}).get(preview, {}).get("text") or []
    transcript = " ".join(str(text).strip() for text in texts if str(text).strip())
    if not transcript or transcript == "None":
        raise H3Error("Qwen3-ASR heard no speech in this clip. Use a few seconds of clean speech.")
    return transcript
