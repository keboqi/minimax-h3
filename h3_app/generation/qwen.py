"""Qwen Image 2.1 request orchestration."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict
from typing import Iterator

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
    queued_at = time.time()
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
        editing = str(request.mode).strip().lower() == "image edit"
        references = tuple(path for path in request.reference_images if path)
        if editing and not references:
            raise H3Error("Image edit mode requires at least one reference image.")
        if not editing and references:
            raise H3Error(
                "Reference images are only used in Image edit mode. "
                "Switch the mode or remove the uploaded images."
            )
        if len(references) > 10:
            raise H3Error("Qwen Image 2.1 supports at most 10 reference images.")

        width = _image_dimension(request.width, "Width")
        height = _image_dimension(request.height, "Height")
        if width * height > 4_400_000:
            raise H3Error(
                "Output resolution must stay within the model's native "
                "approximately 4.3 MP canvas."
            )
        reference_resolution = int(request.reference_resolution)
        if reference_resolution and (
            not 256 <= reference_resolution <= 2048
            or reference_resolution % 32
        ):
            raise H3Error(
                "Reference resolution must be 0 or a multiple of 32 from 256 to 2048."
            )
        steps = int(request.steps)
        if not 1 <= steps <= 100:
            raise H3Error("Sampling steps must be between 1 and 100.")
        cfg = float(request.cfg)
        if not 0 <= cfg <= 20:
            raise H3Error("CFG must be between 0 and 20.")
        attention_backend = str(request.attention_backend)
        if attention_backend not in {
            "pytorch attention",
            "comfy kitchen attention",
        }:
            raise H3Error("Unsupported Qwen attention backend.")
        accelerator = str(request.accelerator or "Off").strip()
        if accelerator.lower() not in {"off", "spectrum"}:
            raise H3Error("Unsupported Qwen accelerator. Choose Off or Spectrum.")
        actual_seed = (
            random.randrange(0, 2**63 - 1)
            if int(request.seed) < 0
            else int(request.seed)
        )

        missing_files = services.models.missing_qwen_image21_model_names(
            request.model_choice, request.text_encoder_choice
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
            request.model_choice, request.text_encoder_choice
        )

        available = set(services.execution.object_info())
        missing_nodes = required_qwen_image21_nodes(
            editing=editing,
            use_spectrum=accelerator.lower() == "spectrum",
        ) - available
        if missing_nodes:
            raise H3Error(
                "Qwen Image 2.1 requires the latest ComfyUI; missing nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        graph = build_qwen_image21_graph(
            model_choice=request.model_choice,
            text_encoder_choice=request.text_encoder_choice,
            prompt=prompt,
            negative_prompt=str(request.negative_prompt or ""),
            reference_images=references,
            width=width,
            height=height,
            reference_resolution=reference_resolution,
            match_input_size=bool(request.match_input_size),
            seed=actual_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=request.sampler_name,
            scheduler=request.scheduler,
            cache_device=request.cache_device,
            cache_dtype=request.cache_dtype,
            attention_backend=attention_backend,
            accelerator=accelerator,
            output_stamp=str(int(time.time())),
            output_nonce=uuid.uuid4().hex[:8],
        )
        client_id = str(uuid.uuid4())
        prompt_id = services.execution.submit_prompt(graph, client_id)
        timings.label = f"Qwen Image 2.1 job {prompt_id}"
        timings.transition("Waiting for ComfyUI")
        yield GenerationUpdate(
            None,
            f"Queued Qwen Image 2.1 job `{prompt_id}` · seed {actual_seed} · "
            f"{'edit' if editing else f'{width}×{height} generation'} · "
            f"{request.model_choice} · accelerator {accelerator}",
        )
        for stage, completed_nodes, total_nodes, step, step_total in (
            services.execution.poll_comfy_progress(prompt_id, graph)
        ):
            timings.transition(stage)
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total_nodes:
                progress((completed_nodes, total_nodes), desc=stage)
            yield GenerationUpdate(
                None,
                progress_status(
                    stage,
                    started=started,
                    completed_nodes=completed_nodes,
                    total_nodes=total_nodes,
                    step=step,
                    step_total=step_total,
                    configured_steps=steps,
                    detail=f"Qwen Image 2.1 job `{prompt_id}`",
                ),
            )

        timings.transition("Locating generated image")
        history = services.execution.wait_for_history(prompt_id)
        result = services.media.resolve_image_outputs(history, queued_at, 1)[0]
        snapshot_values.update(seed=actual_seed, editing=editing)
        write_snapshot(
            result,
            {
                "job_id": prompt_id,
                "family": "Qwen Image 2.1",
                "settings": snapshot_values,
            },
        )
        elapsed = time.monotonic() - started
        progress(1, desc="Complete")
        yield GenerationUpdate(
            str(result),
            f"Qwen Image 2.1 completed in {elapsed:.1f}s · output {result.name} · "
            f"seed {actual_seed}\n\n{timings.summary()}",
        )
    except Exception as exc:
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()
