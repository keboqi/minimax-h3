"""One ComfyUI job for the single-image engines: FLUX 3, MOSS-Image and Fibo 2.0.

Each engine supplies its validation and graph; this module checks the backend
before any download, retains retry seeds, and resolves only this job's output.
"""

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Callable, Iterator
import uuid

from h3_app.errors import H3Error
from h3_app.jobs import CURRENT_JOB, record_failure, variant_seed
from h3_app.progress import no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import StageTimings, progress_status

from .results import GenerationUpdate

# Request fields left out of the recorded settings: private text and media.
PRIVATE_FIELDS = frozenset({"prompt", "text", "voices", "reference_images", "layout"})


@dataclass(frozen=True)
class PreparedImage:
    graph: dict[str, Any]
    model_keys: tuple[str, ...]
    settings: dict[str, Any]  # recorded beside the request's own values
    size: str  # the queued status's size note, such as " · 1024×1024"


@dataclass(frozen=True)
class ImageEngine:
    family: str  # status and settings family
    clip_type: str  # CLIPLoader type the backend must offer
    label: str  # the engine's name in the missing-encoder message
    validate: Callable[[Any], None]
    # (request, values, object_info, runtime) -> PreparedImage, yielding notes first.
    # ``values`` is the request with its seed; set the resolved size on it.
    prepare: Callable[..., Iterator[GenerationUpdate]]


def output_naming() -> dict[str, str]:
    return {"output_stamp": str(int(time.time())), "output_nonce": uuid.uuid4().hex[:8]}


def size_note(values) -> str:
    return f" · {values['width']}×{values['height']}"


def aspect_edit_dimensions(source_width, source_height, *, maximize, max_pixels, min_side, max_side, grid,
                           label):
    """Fit the first reference's aspect ratio on a model's grid.

    Match keeps the native area (downscaled to the cap); Max scales toward the cap.
    """
    source_width, source_height = int(source_width), int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise H3Error("The first reference image has invalid dimensions.")
    lo, hi = min_side, max_side
    source_area = source_width * source_height
    target_area = max_pixels if maximize else min(source_area, max_pixels)
    scale = min(math.sqrt(target_area / source_area), hi / source_width, hi / source_height)
    target_area = max(lo * lo, min(hi, max(lo, source_width * scale)) * min(hi, max(lo, source_height * scale)))
    source_ratio = source_width / source_height
    # Search both sides jointly so common ratios stay exact where the grid allows.
    candidates = [
        (width, height)
        for width in range(lo, hi + 1, grid)
        for height in range(lo, hi + 1, grid)
        if target_area * 0.85 <= width * height <= min(target_area, max_pixels)
    ]
    if not candidates:
        raise H3Error(f"The first reference image's aspect ratio is outside {label}'s supported canvas.")
    def drift(size):
        return abs(math.log((size[0] / size[1]) / source_ratio))
    # Up to 1% ratio drift is invisible; prefer the larger canvas within it (1920×1088, not 1792×1008).
    close = [size for size in candidates if drift(size) <= 0.01]
    if close:
        return max(close, key=lambda size: (size[0] * size[1], -drift(size)))
    return min(candidates, key=lambda size: (drift(size), -(size[0] * size[1])))


def run_image_job(engine: ImageEngine, request, services, runtime, *, progress=no_progress):
    family = engine.family
    started = time.monotonic()
    timings = StageTimings(family, started, "Validating request")
    try:
        engine.validate(request)
        job = CURRENT_JOB.get()
        seed = variant_seed(job.variant if job else 0,
                            lambda: random.randrange(2**63 - 1) if request.seed < 0 else int(request.seed))
        values = {**asdict(request), "seed": seed}
        info = services.execution.object_info()
        prepared = yield from engine.prepare(request, values, info, runtime)
        graph = prepared.graph
        missing = {node["class_type"] for node in graph.values()} - set(info)
        if missing:
            raise H3Error(f"Update setup and restart ComfyUI (the pinned Core nodes are required). Missing nodes: {', '.join(sorted(missing))}")
        clip_types = info.get("CLIPLoader", {}).get("input", {}).get("required", {}).get("type", [[]])[0]
        if engine.clip_type not in clip_types:
            raise H3Error(f"This backend lacks the {engine.label} text encoder. Update setup and restart ComfyUI.")
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(None, f"Preparing {family} models; missing weights download on first use.")
        services.models.ensure_model_keys(prepared.model_keys)
        queued_at = time.time()
        prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
        yield GenerationUpdate(None, f"Queued {family} job {prompt_id} · seed {seed}{prepared.size}")
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
        settings = {key: value for key, value in values.items() if key not in PRIVATE_FIELDS}
        settings.update(prepared.settings)
        write_snapshot(result, {"job_id": prompt_id, "family": family, "settings": settings})
        progress(1, desc="Complete")
        yield GenerationUpdate([str(result)], f"{family} complete · seed {seed}\n\n{timings.summary()}")
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()
