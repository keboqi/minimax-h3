"""H3 finish video phase."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

from h3_app.catalog import (
    COMFY_UPSCALE_OPTIONS,
    LTX25_UPSCALE,
    SEEDVR2_UPSCALE,
    SWIFTVR_UPSCALE,
)
from h3_app.media_types import UpscaleClipBatch
from h3_app.policy import upscale_target_dimensions
from h3_app.progress import ProgressCallback
from h3_app.status import StageTimings, progress_status

from .preparation import PreparedH3
from .requests import H3Request
from .results import GenerationUpdate
from .services import GenerationServices


def finish_video(
    request: H3Request,
    prepared: PreparedH3,
    services: GenerationServices,
    source: Path,
    client_id: str,
    started: float,
    timings: StageTimings,
    progress: ProgressCallback,
) -> Generator[GenerationUpdate, None, Path]:
    clip_batch: UpscaleClipBatch | None = None
    clip_outputs: list[Path] = []
    try:
        if request.finishing.postprocess == SWIFTVR_UPSCALE:
            metadata = services.media.probe_video_metadata(source)
            target_width, target_height = upscale_target_dimensions(
                metadata.width, metadata.height, request.finishing.upscale_resolution
            )
            if request.finishing.upscale_force_offload:
                services.models.unload_comfy_models()
            result = services.media.postprocess_swiftvr_video(
                source,
                fps=metadata.fps,
                target_width=target_width,
                target_height=target_height,
            )
        elif request.finishing.postprocess in COMFY_UPSCALE_OPTIONS:
            if request.finishing.postprocess == SEEDVR2_UPSCALE:
                model_status = f"SeedVR2 {request.finishing.seedvr2_model}"
                stage_bucket = "seedvr2_upscale"
                yield GenerationUpdate(
                    None,
                    progress_status(f"Checking {model_status} models", started=started),
                )
                downloaded = services.models.ensure_seedvr2_upscale_models(
                    prepared.models, request.finishing.seedvr2_model
                )
            else:
                model_status = f"LTX-2.5 {request.finishing.ltx25_model} and 2x IC-LoRA"
                stage_bucket = "ltx25_upscale"
                yield GenerationUpdate(
                    None,
                    progress_status(f"Checking {model_status} models", started=started),
                )
                downloaded = services.models.ensure_ltx25_upscale_models(
                    request.finishing.ltx25_model
                )

            if downloaded:
                yield GenerationUpdate(
                    None,
                    progress_status(
                        f"{request.finishing.postprocess} models downloaded",
                        started=started,
                    ),
                )
            metadata = services.media.probe_video_metadata(source)
            target_width, target_height = upscale_target_dimensions(
                metadata.width, metadata.height, request.finishing.upscale_resolution
            )
            use_split = request.finishing.postprocess == LTX25_UPSCALE and bool(
                request.finishing.upscale_split_enabled
            )
            clip_batch = services.media.prepare_upscale_clip_batch(
                source,
                category=stage_bucket,
                split_enabled=use_split,
                split_seconds=request.finishing.upscale_split_seconds,
                metadata=metadata,
            )
            clip_count = len(clip_batch.sources)
            if use_split:
                yield GenerationUpdate(
                    None,
                    progress_status(
                        f"Split source into {clip_count} LTX-safe clips",
                        started=started,
                        detail=f"Target clip length {float(request.finishing.upscale_split_seconds):g}s",
                    ),
                )
            staged_source = clip_batch.sources[0]
            unload_h3_before_upscale = bool(request.finishing.upscale_force_offload)
            if unload_h3_before_upscale:
                progress(0, desc="Unloading H3 models")
                yield GenerationUpdate(
                    None,
                    progress_status(
                        f"Unloading H3 models before {request.finishing.postprocess}",
                        started=started,
                    ),
                )
                services.models.unload_comfy_models()

            upscale_graph, configured_upscale_steps = (
                services.workflows.build_upscale_graph(
                    option=request.finishing.postprocess,
                    source_video=staged_source,
                    seed=prepared.actual_seed,
                    models=prepared.models,
                    seedvr2_model=request.finishing.seedvr2_model,
                    ltx25_model=request.finishing.ltx25_model,
                    prompt=request.media.prompt,
                    width=prepared.resolved_width,
                    height=prepared.resolved_height,
                    target_width=target_width,
                    target_height=target_height,
                    fps=metadata.fps,
                )
            )

            upscale_queued_at = time.time()
            upscale_prompt_id = services.execution.submit_prompt(
                upscale_graph, client_id
            )
            unload_note = (
                "H3 unload enabled"
                if unload_h3_before_upscale
                else "automatic residency"
            )
            yield GenerationUpdate(
                None,
                progress_status(
                    f"{request.finishing.postprocess} queued",
                    started=started,
                    detail=(
                        f"Job `{upscale_prompt_id}` · {prepared.resolved_width * 2}×"
                        f"{prepared.resolved_height * 2} · {unload_note}"
                    ),
                ),
            )
            upscale_updates = services.execution.poll_comfy_progress(
                upscale_prompt_id, upscale_graph
            )
            for (
                stage,
                completed_nodes,
                total_nodes,
                step,
                step_total,
            ) in upscale_updates:
                if stage == "Generating video and audio":
                    stage = f"Upscaling with {request.finishing.postprocess}"
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
                        configured_steps=configured_upscale_steps
                        if step is not None
                        else None,
                        detail=f"Upscale job `{upscale_prompt_id}`",
                    ),
                )
            upscale_history = services.execution.wait_for_history(upscale_prompt_id)
            result = services.media.resolve_output(upscale_history, upscale_queued_at)
            clip_outputs.append(result)
            for clip_index, staged_source in enumerate(clip_batch.sources[1:], start=1):
                clip_seed = (prepared.actual_seed + clip_index) % (2**63 - 1)
                upscale_graph, configured_upscale_steps = (
                    services.workflows.build_upscale_graph(
                        option=request.finishing.postprocess,
                        source_video=staged_source,
                        seed=clip_seed,
                        models=prepared.models,
                        seedvr2_model=request.finishing.seedvr2_model,
                        ltx25_model=request.finishing.ltx25_model,
                        prompt=request.media.prompt,
                        width=prepared.resolved_width,
                        height=prepared.resolved_height,
                        target_width=target_width,
                        target_height=target_height,
                        fps=metadata.fps,
                    )
                )
                upscale_queued_at = time.time()
                upscale_prompt_id = services.execution.submit_prompt(
                    upscale_graph, client_id
                )
                clip_label = f"Clip {clip_index + 1}/{clip_count}"
                yield GenerationUpdate(
                    None,
                    progress_status(
                        f"{request.finishing.postprocess} queued",
                        started=started,
                        detail=(
                            f"{clip_label} 路 job `{upscale_prompt_id}` 路 "
                            f"seed {clip_seed} 路 {unload_note}"
                        ),
                    ),
                )
                upscale_updates = services.execution.poll_comfy_progress(
                    upscale_prompt_id, upscale_graph
                )
                for (
                    stage,
                    completed_nodes,
                    total_nodes,
                    step,
                    step_total,
                ) in upscale_updates:
                    if stage == "Generating video and audio":
                        stage = f"Upscaling with {request.finishing.postprocess}"
                    timings.transition(f"{clip_label}: {stage}")
                    if step is not None and step_total:
                        progress(
                            (clip_index * step_total + step, clip_count * step_total),
                            desc=f"{clip_label}: {stage}",
                        )
                    elif total_nodes:
                        progress(
                            (
                                clip_index * total_nodes + completed_nodes,
                                clip_count * total_nodes,
                            ),
                            desc=f"{clip_label}: {stage}",
                        )
                    yield GenerationUpdate(
                        None,
                        progress_status(
                            f"{clip_label}: {stage}",
                            started=started,
                            completed_nodes=completed_nodes,
                            total_nodes=total_nodes,
                            step=step,
                            step_total=step_total,
                            configured_steps=(
                                configured_upscale_steps if step is not None else None
                            ),
                            detail=f"Upscale job `{upscale_prompt_id}`",
                        ),
                    )
                clip_outputs.append(
                    services.media.resolve_output(
                        services.execution.wait_for_history(upscale_prompt_id),
                        upscale_queued_at,
                    )
                )
            if use_split:
                timings.transition("Concatenating upscaled clips")
                yield GenerationUpdate(
                    None,
                    progress_status(
                        "Concatenating upscaled clips and restoring source audio",
                        started=started,
                        detail=f"{clip_count} clips",
                    ),
                )
                result = services.media.concat_upscaled_clips(
                    source,
                    clip_outputs,
                    option=request.finishing.postprocess,
                    duration=metadata.duration,
                    frame_count=metadata.frame_count,
                )
        else:
            if request.finishing.postprocess != "None":
                timings.transition(f"Applying {request.finishing.postprocess}")
            result = services.media.postprocess_video(
                source, request.finishing.postprocess
            )

        return result
    finally:
        services.media.cleanup_upscale_clip_batch(
            clip_batch,
            clip_outputs if clip_batch and clip_batch.temporary_inputs else (),
        )
