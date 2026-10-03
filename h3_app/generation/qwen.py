"""Qwen Image 2.1 request orchestration."""

from __future__ import annotations

from h3_app.jobs import record_failure, variant_seed, variant_indices

import math
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from PIL import Image
from h3_models import QWEN_IMAGE21_TURBO_MODES

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.progress import ProgressCallback, no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import StageTimings, progress_status
from h3_app.workflows.qwen import (
    build_qwen_image21_graph,
    required_qwen_image21_nodes,
)

from .requests import QwenImage21Request
from .results import GenerationUpdate
from .services import GenerationServices


def _image_dimension(value: int, label: str) -> int:
    resolved = int(value)
    if not 256 <= resolved <= 2752 or resolved % 32:
        raise H3Error(f"{label} must be a multiple of 32 between 256 and 2752.")
    return resolved


def max_qwen_edit_dimensions(source_width: int, source_height: int) -> tuple[int, int]:
    """Fit the source aspect ratio near 4 MP on Qwen's 32-pixel grid."""
    source_width, source_height = int(source_width), int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise H3Error("The first reference image has invalid dimensions.")
    max_pixels, max_side, grid = 4_000_000, 2752, 32
    scale = min(
        math.sqrt(max_pixels / (source_width * source_height)),
        max_side / source_width,
        max_side / source_height,
    )

    target_width = min(max_side, max(256, source_width * scale))
    target_height = min(max_side, max(256, source_height * scale))
    target_area = target_width * target_height
    source_ratio = source_width / source_height
    # Search the grid jointly: rounding the two sides independently can distort
    # common ratios such as 16:9 even when an exact aligned size is available.
    candidates = (
        (width, height)
        for width in range(256, max_side + 1, grid)
        for height in range(256, max_side + 1, grid)
        if width * height <= max_pixels and width * height >= target_area * 0.85
    )
    return min(
        candidates,
        key=lambda size: (
            abs(math.log((size[0] / size[1]) / source_ratio)),
            -(size[0] * size[1]),
        ),
    )


def first_reference_dimensions(
    path: str, input_dir: Path | None = None
) -> tuple[int, int]:
    try:
        image_path = Path(path)
        if not image_path.is_absolute() and input_dir is not None:
            image_path = input_dir / image_path
        with Image.open(image_path) as image:
            width, height = image.size
            if image.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
            return width, height
    except (OSError, ValueError) as exc:
        raise H3Error("Could not read the first reference image size.") from exc


def resolve_qwen_output_dimensions(
    request: QwenImage21Request,
    references: tuple[str, ...],
    editing: bool,
    *,
    input_dir: Path | None = None,
) -> tuple[int, int, bool]:
    """Resolve edit sizing, with Max resolution taking priority."""
    max_edit_resolution = editing and bool(request.max_resolution)
    if max_edit_resolution:
        source_width, source_height = first_reference_dimensions(
            references[0], input_dir
        )
        width, height = max_qwen_edit_dimensions(source_width, source_height)
    else:
        width = _image_dimension(request.width, "Width")
        height = _image_dimension(request.height, "Height")
    if width * height > 4_400_000:
        raise H3Error(
            "Output resolution must stay within the model's native "
            "approximately 4.3 MP canvas."
        )
    match_input_size = bool(request.match_input_size) and not max_edit_resolution
    return width, height, match_input_size


def generate_qwen_image21(
    request: QwenImage21Request,
    services: GenerationServices,
    runtime: RuntimeConfig,
    *,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    """Run Qwen Image 2.1 through the shared ComfyUI queue."""
    started = time.monotonic()
    timings = StageTimings("Qwen Image 2.1", started, "Preparing request")
    outputs: list[str] = []
    snapshot_values = {
        key: value
        for key, value in asdict(request).items()
        if key not in {"prompt", "negative_prompt", "reference_images"}
    }
    try:
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(
            None, progress_status("Validating Qwen Image 2.1 request", started=started)
        )
        prompt = str(request.prompt).strip()
        if not prompt:
            raise H3Error("Prompt or edit instruction is required.")
        batch_count = int(request.batch_count)
        if batch_count != request.batch_count or not 1 <= batch_count <= 4:
            raise H3Error("Images per batch must be between 1 and 4.")
        editing = str(request.mode).strip().lower() == "image edit"
        references = tuple(path for path in request.reference_images if path)
        if editing and not references:
            raise H3Error("Image edit mode requires at least one reference image.")
        if not editing and references:
            raise H3Error(
                "Reference images are only used in Image edit mode. "
                "Switch the mode or remove the uploaded images."
            )
        turbo = request.turbo_variant != "Off"
        if (
            request.turbo_variant != "Off"
            and request.turbo_variant not in QWEN_IMAGE21_TURBO_MODES
        ):
            raise H3Error(f"Unknown Qwen Turbo variant: {request.turbo_variant}")
        reference_limit = 3 if turbo else 10
        separate_inputs = editing and bool(request.batch_edit_inputs)
        if not separate_inputs and len(references) > reference_limit:
            raise H3Error(
                f"Qwen Image 2.1 {request.turbo_variant} supports at most "
                f"{reference_limit} reference images."
            )

        input_groups = (
            tuple((reference,) for reference in references)
            if separate_inputs
            else (references,)
        )
        dimensions = tuple(
            resolve_qwen_output_dimensions(
                request, group, editing, input_dir=runtime.input_dir
            )
            for group in input_groups
        )
        total_jobs = len(input_groups) * batch_count
        max_edit_resolution = editing and bool(request.max_resolution)
        reference_resolution = int(request.reference_resolution)
        if reference_resolution and (
            not 256 <= reference_resolution <= 2048 or reference_resolution % 32
        ):
            raise H3Error(
                "Reference resolution must be 0 or a multiple of 32 from 256 to 2048."
            )
        steps = int(request.steps)
        if not 1 <= steps <= 100:
            raise H3Error("Sampling steps must be between 1 and 100.")
        viggle = request.turbo_variant in QWEN_IMAGE21_TURBO_MODES
        nine_step = viggle and QWEN_IMAGE21_TURBO_MODES[request.turbo_variant][1] == 9
        if viggle and steps != QWEN_IMAGE21_TURBO_MODES[request.turbo_variant][1]:
            raise H3Error(
                f"{request.turbo_variant} requires exactly "
                f"{QWEN_IMAGE21_TURBO_MODES[request.turbo_variant][1]} sampling steps."
            )
        cfg = float(request.cfg)
        if not 0 <= cfg <= 20:
            raise H3Error("CFG must be between 0 and 20.")
        if viggle and cfg != 1.0:
            raise H3Error("Viggle Turbo requires CFG 1.")
        if nine_step and request.sampler_name != "euler":
            raise H3Error("Viggle nine-step mode requires Euler sampling.")
        attention_backend = str(request.attention_backend)
        if attention_backend not in {
            "pytorch attention",
            "comfy kitchen attention",
        }:
            raise H3Error("Unsupported Qwen attention backend.")
        accelerator = str(request.accelerator or "Off").strip()
        if accelerator.lower() not in {
            "off",
            "spectrum",
            "spectrum (quality)",
            "spectrum (preview)",
        }:
            raise H3Error(
                "Unsupported Qwen accelerator. Choose Off, Spectrum (Quality), "
                "or Spectrum (Preview)."
            )
        if turbo and accelerator.lower() != "off":
            raise H3Error("Spectrum acceleration is unavailable with Qwen Turbo.")
        base_seed = int(request.seed)
        if base_seed >= 0 and base_seed + total_jobs - 1 >= 2**63 - 1:
            raise H3Error("Seed is too large for this batch count.")

        missing_files = services.models.missing_qwen_image21_model_names(
            request.model_choice, request.text_encoder_choice, request.turbo_variant
        )
        if missing_files:
            yield GenerationUpdate(
                None,
                progress_status(
                    "Downloading Qwen Image 2.1 models on demand",
                    started=started,
                    detail=", ".join(missing_files),
                ),
            )
        services.models.ensure_qwen_image21_models(
            request.model_choice, request.text_encoder_choice, request.turbo_variant
        )

        available = set(services.execution.object_info())
        missing_nodes = (
            required_qwen_image21_nodes(
                editing=editing,
                use_spectrum=accelerator.lower() != "off",
                turbo=turbo,
                viggle=viggle,
                nine_step=nine_step,
                scheduler=request.scheduler,
            )
            - available
        )
        if missing_nodes:
            raise H3Error(
                "Qwen Image 2.1 requires the latest ComfyUI; missing nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        sampling_detail = f"{steps} steps"
        for index in variant_indices(total_jobs):
            input_index, _ = divmod(index, batch_count)
            job_references = input_groups[input_index]
            width, height, match_input_size = dimensions[input_index]
            job_size = (
                f"edit at {width}×{height}"
                if max_edit_resolution
                else "edit" if editing else f"{width}×{height} generation"
            )
            actual_seed = variant_seed(
                index,
                lambda: (
                    random.randrange(0, 2**63 - 1)
                    if base_seed < 0
                    else base_seed + index
                ),
            )
            queued_at = time.time()
            graph = build_qwen_image21_graph(
                model_choice=request.model_choice,
                text_encoder_choice=request.text_encoder_choice,
                prompt=prompt,
                negative_prompt=str(request.negative_prompt or ""),
                reference_images=job_references,
                width=width,
                height=height,
                reference_resolution=reference_resolution,
                match_input_size=match_input_size,
                seed=actual_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=request.sampler_name,
                scheduler=request.scheduler,
                cache_device=request.cache_device,
                cache_dtype=request.cache_dtype,
                attention_backend=attention_backend,
                accelerator=accelerator,
                output_stamp=str(int(queued_at)),
                output_nonce=uuid.uuid4().hex[:8],
                turbo_variant=request.turbo_variant,
            )
            client_id = str(uuid.uuid4())
            prompt_id = services.execution.submit_prompt(graph, client_id)
            timings.label = f"Qwen Image 2.1 job {prompt_id}"
            timings.transition("Waiting for ComfyUI")
            yield GenerationUpdate(
                outputs.copy(),
                f"Queued Qwen Image 2.1 image {index + 1}/{total_jobs} · "
                f"job `{prompt_id}` · seed {actual_seed} · {job_size} · "
                f"{request.model_choice} · {request.turbo_variant} · "
                f"{sampling_detail} · accelerator {accelerator}",
            )
            for (
                stage,
                completed_nodes,
                total_nodes,
                step,
                step_total,
            ) in services.execution.poll_comfy_progress(prompt_id, graph):
                timings.transition(stage)
                if step is not None and step_total:
                    progress((step, step_total), desc=stage)
                elif total_nodes:
                    progress((completed_nodes, total_nodes), desc=stage)
                yield GenerationUpdate(
                    outputs.copy(),
                    progress_status(
                        stage,
                        started=started,
                        completed_nodes=completed_nodes,
                        total_nodes=total_nodes,
                        step=step,
                        step_total=step_total,
                        configured_steps=steps,
                        detail=f"Image {index + 1}/{total_jobs} · Qwen job `{prompt_id}`",
                    ),
                )

            timings.transition("Locating generated image")
            history = services.execution.wait_for_history(prompt_id)
            result = services.media.resolve_image_outputs(history, queued_at, 1)[0]
            write_snapshot(
                result,
                {
                    "job_id": prompt_id,
                    "family": "Qwen Image 2.1",
                    "settings": {
                        **snapshot_values,
                        "seed": actual_seed,
                        "editing": editing,
                        "resolved_width": width,
                        "resolved_height": height,
                        "effective_match_input_size": match_input_size,
                        "input_index": input_index + 1 if separate_inputs else None,
                    },
                },
            )
            outputs.append(str(result))
            elapsed = time.monotonic() - started
            yield GenerationUpdate(
                outputs.copy(),
                f"Qwen Image 2.1 image {index + 1}/{total_jobs} completed in "
                f"{elapsed:.1f}s · output {result.name} · seed {actual_seed}\n\n"
                f"{timings.summary()}",
            )
        progress(1, desc="Complete")
    except Exception as exc:
        record_failure(exc)
        yield GenerationUpdate(outputs.copy(), f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()
