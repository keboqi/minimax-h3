"""FLUX 3 and MOSS-Image over the shared, owned ComfyUI queue; Drama 3 on SGLang-Omni."""

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
from h3_app.workflows.flux3_layout import parse_flux3_layout
from h3_app.workflows.moss_image import (
    build_moss_image_graph, moss_image_model_keys, validate_moss_image_sampling, moss_spectrum_enabled,
    MOSS_SPECTRUM_NODE,
)
from h3_app.workflows.fibo2 import build_fibo2_graph, fibo2_model_keys, validate_fibo2_inputs
from h3_app.workflows.moss_image_prompt import is_layout_prompt
from h3_models import FLUX3_BASE_MODEL_CHOICES, MODEL_SPECS
from h3_app.workflows.drama3 import (
    MAX_FRAMES, MAX_SPEAKER, build_drama3_asr_graph, request_cast, speaker_label, unknown_speaker_names,
    validate_drama3_text, drama3_asr_model_keys,
)
from h3_app.drama3_serving.tree import script_speakers
from h3_models import DEFAULT_DRAMA3_ASR
from .qwen import first_reference_dimensions
from .requests import Flux3Request, Drama3Request, MossImageRequest, Fibo2Request
from .results import GenerationUpdate


DRAMA3_BACKEND = "SGLang-Omni"
FLUX3_EDIT_SIZES = (QWEN_EDIT_SIZE_MATCH, QWEN_EDIT_SIZE_MAX, QWEN_EDIT_SIZE_MANUAL)
MOSS_IMAGE_FAMILY = "MOSS-Image"
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
    if not isinstance(request, Drama3Request):  # SGLang-Omni takes no per-request seed
        _integer(request.seed, -1, 2**63 - 2, "Seed")
    if isinstance(request, Flux3Request):
        if not request.prompt.strip():
            raise H3Error("Write an image prompt or edit request.")
        flux3_model_keys(request.model_choice, request.expand_prompt, request.upsampler_choice)
        _integer(request.width, 256, 4096, "Width", 16)
        _integer(request.height, 256, 4096, "Height", 16)
        _integer(request.steps, 1, 100, "Steps")
        if request.cfg is not None:
            if isinstance(request.cfg, bool) or not math.isfinite(request.cfg) or not 1 <= request.cfg <= 20:
                raise H3Error("FLUX CFG must be between 1 and 20.")
            if request.model_choice not in FLUX3_BASE_MODEL_CHOICES and request.cfg != 1:
                raise H3Error("Distilled FLUX 3 requires CFG 1.")
        parse_flux3_layout(request.layout)
        if request.edit_size not in FLUX3_EDIT_SIZES:
            raise H3Error("Choose a valid FLUX 3 edit output size.")
        if len(request.reference_images) > 10:
            raise H3Error("FLUX 3 supports at most ten reference images.")
    elif isinstance(request, Fibo2Request):
        _integer(request.width, 256, 2048, "Width", 16)
        _integer(request.height, 256, 2048, "Height", 16)
        _integer(request.steps, 1, 100, "Steps")
        if isinstance(request.cfg, bool) or not math.isfinite(float(request.cfg)) or not 1 <= request.cfg <= 20:
            raise H3Error("Fibo CFG must be between 1 and 20.")
        try:
            validate_fibo2_inputs(request.model_choice, request.prompt, request.reference_images,
                                  request.steps, request.cfg, request.precision)
        except ValueError as exc:
            raise H3Error(str(exc)) from exc
    elif isinstance(request, MossImageRequest):
        if not request.prompt.strip():
            raise H3Error("Write an image prompt.")
        _integer(request.width, 256, 2048, "Width", 16)
        _integer(request.height, 256, 2048, "Height", 16)
        _integer(request.steps, 1, 100, "Steps")
        if isinstance(request.cfg, bool) or not math.isfinite(float(request.cfg)) or not 1 <= request.cfg <= 10:
            raise H3Error("MOSS-Image CFG must be between 1 and 10.")
        try:
            moss_image_model_keys(request.model_choice, request.text_encoder_choice, request.turbo_variant,
                                  request.vae_choice)
            validate_moss_image_sampling(request.turbo_variant, request.steps, request.cfg,
                                         request.width, request.height)
            parse_flux3_layout(request.layout)
        except ValueError as exc:
            raise H3Error(str(exc)) from exc
        if request.layout.strip() and is_layout_prompt(request.prompt):
            raise H3Error("The prompt is already a MOSS-Image Layout with its own boxes. Clear the layout boxes.")
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
        from h3_app.prompt_service import DRAMA3_SCRIPT_MODES

        if request.script_mode not in DRAMA3_SCRIPT_MODES:
            raise H3Error("Choose Basic or Advanced (sound enhanced) Drama 3 script mode.")
        _integer(request.parallel, 1, 32, "Parallel requests")
        _integer(request.retries, 0, 4, "Retries per chunk")
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
        _integer(request.top_k, 0, 4096, "Top-k")
        _integer(request.max_frames, 1, MAX_FRAMES, "Maximum audio frames")
    else:
        raise TypeError("Unsupported generation request.")


def generate_optional_engine(request, services, runtime, *, progress=no_progress, free_gpu=None):
    """Validate before downloads, retain retry seeds, and resolve only this job's output.

    ``free_gpu`` releases ComfyUI's VRAM before Drama 3 starts or wakes SGLang-Omni.
    """
    moss = isinstance(request, MossImageRequest)
    fibo = isinstance(request, Fibo2Request)
    image = moss or fibo or isinstance(request, Flux3Request)
    family = "Fibo 2.0" if fibo else MOSS_IMAGE_FAMILY if moss else "FLUX 3 Image Klein" if image else "Drama 3"
    started = time.monotonic()
    timings = StageTimings(family, started, "Validating request")
    try:
        validate_request(request)
        if not image:
            from h3_app.drama3_serving.generate import generate_drama3_served
            from h3_models import resolve_hf_token

            services.models.unload_prompt_rewriter()
            timings.transition(f"{DRAMA3_BACKEND} Drama 3")
            yield from generate_drama3_served(
                request, runtime=runtime, free_gpu=free_gpu or (lambda: None),
                hf_token=resolve_hf_token(), progress=progress,
            )
            return
        job = CURRENT_JOB.get()
        seed = variant_seed(job.variant if job else 0,
                            lambda: random.randrange(2**63 - 1) if request.seed < 0 else int(request.seed))
        values = {**asdict(request), "seed": seed}
        info = services.execution.object_info()
        if fibo:
            graph = build_fibo2_graph(**values, output_stamp=str(int(time.time())),
                                      output_nonce=uuid.uuid4().hex[:8])
        elif moss:
            # Installs that predate the Spectrum-MOSS pin keep the original graph.
            wanted = moss_spectrum_enabled(request.accelerate, request.turbo_variant, request.steps)
            spectrum = wanted and MOSS_SPECTRUM_NODE in info
            if wanted and not spectrum:
                yield GenerationUpdate(None, "Spectrum-MOSS is not installed; using the original sampler. "
                                             "Rerun setup and restart ComfyUI to enable it.")
            graph = build_moss_image_graph(
                **{k: v for k, v in values.items() if k not in {"preset", "accelerate"}}, accelerate=spectrum,
                output_stamp=str(int(time.time())), output_nonce=uuid.uuid4().hex[:8],
            )
        else:
            values["width"], values["height"] = resolve_flux3_dimensions(request, input_dir=runtime.input_dir)
            # The shared image resolver only reads the managed staging directory.
            graph = build_flux3_graph(
                **{k: v for k, v in values.items() if k != "edit_size"},
                output_stamp=str(int(time.time())), output_nonce=uuid.uuid4().hex[:8],
                system_prompt=runtime.prompt_systems[
                    "FLUX 3 Image Klein edit" if request.reference_images else "FLUX 3 Image Klein"
                ].read_text(encoding="utf-8").strip(),
            )
        missing = {node["class_type"] for node in graph.values()} - set(info)
        if missing:
            raise H3Error(f"Update setup and restart ComfyUI (the pinned Core nodes are required). Missing nodes: {', '.join(sorted(missing))}")
        clip_types = info.get("CLIPLoader", {}).get("input", {}).get("required", {}).get("type", [[]])[0]
        clip_type, label = (("fibo2", "Fibo 2.0") if fibo else ("moss_image", "MOSS-Image") if moss
                            else ("flux3", "FLUX 3"))
        if clip_type not in clip_types:
            raise H3Error(f"This backend lacks the {label} text encoder. Update setup and restart ComfyUI.")
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(None, f"Preparing {family} models; missing weights download on first use.")
        keys = (fibo2_model_keys(request.model_choice, request.precision, bool(request.reference_images))
                if fibo else
                moss_image_model_keys(request.model_choice, request.text_encoder_choice,
                                      request.turbo_variant, request.vae_choice)
                if moss else
                flux3_model_keys(request.model_choice, request.expand_prompt, request.upsampler_choice))
        services.models.ensure_model_keys(keys)
        queued_at = time.time()
        prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
        auto_size = fibo and request.auto_size and bool(request.reference_images)
        size = " · auto size" if auto_size else f" · {values['width']}×{values['height']}"
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
        result = services.media.resolve_image_outputs(history, queued_at, 1)[0]
        settings = {key: value for key, value in values.items()
                    if key not in {"prompt", "text", "voices", "reference_images", "layout"}}
        if moss:
            settings["layout_box_count"] = len(parse_flux3_layout(request.layout))
            settings["spectrum_applied"] = spectrum
            settings["scheduler"] = "sgm_uniform" if settings["spectrum_applied"] else "simple"
        elif fibo:
            settings["reference_count"] = len(request.reference_images)
            settings["text_encoder"] = MODEL_SPECS[keys[1]].local_name
        else:
            settings["reference_count"] = len(request.reference_images)
            settings["layout_box_count"] = len(parse_flux3_layout(request.layout))
        write_snapshot(result, {"job_id": prompt_id, "family": family, "settings": settings})
        progress(1, desc="Complete")
        yield GenerationUpdate([str(result)], f"{family} complete · seed {seed}\n\n{timings.summary()}")
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
