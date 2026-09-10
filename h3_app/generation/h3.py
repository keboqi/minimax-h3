"""H3 request orchestration."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS
from h3_app.progress import ProgressCallback, no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import StageTimings, progress_status
from h3_models import MODEL_SPECS

from .construct_graph import construct_graph
from .finish_video import finish_video
from .preparation import prepare_h3
from .requests import H3Request
from .results import GenerationUpdate
from .services import GenerationServices


def generate(
    request: H3Request,
    services: GenerationServices,
    runtime: RuntimeConfig,
    *,
    run_context: Mapping[str, Any],
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    requested_values = request.values()
    requested_values.update(run_context)
    request = request.copy()
    started = time.monotonic()
    timings = StageTimings("H3 generation", started, "Preparing request")
    queued_at = time.time()
    fallback_video: Path | None = None
    try:
        services.models.unload_prompt_rewriter()
        progress(0, desc="Validating request")
        yield GenerationUpdate(
            None, progress_status("Validating request", started=started)
        )
        prepared = yield from prepare_h3(
            request, services, runtime, requested_values, started, progress
        )

        progress(0, desc="Building ComfyUI workflow")
        yield GenerationUpdate(
            None,
            progress_status("Preparing inputs and building workflow", started=started),
        )
        graph = construct_graph(request, prepared, services)

        client_id = str(uuid.uuid4())
        websocket_note = None
        prompt_id = services.execution.submit_prompt(graph, client_id)
        websocket_note = getattr(prompt_id, "notice", None)
        execution_snapshot = {
            "job_id": prompt_id,
            "preset": requested_values.get("preset", "Singularity"),
            "settings": {
                key: value
                for key, value in requested_values.items()
                if key in GENERATION_FIELDS
                and key not in {"prompt", "first_image", "last_image"}
                and not key.startswith(("ref_", "fl2va_audio_"))
            },
            "changes_from_preset": prepared.plan.differences()
            if requested_values.get("preset")
            else {},
            "adjustments": [asdict(item) for item in prepared.plan.adjustments],
        }
        execution_snapshot["settings"].update(
            seed=prepared.actual_seed,
            width=prepared.resolved_width,
            height=prepared.resolved_height,
            steps=prepared.effective_steps,
            scheduler=prepared.effective_scheduler,
            cache_mode=prepared.effective_cache_mode,
            stage_model_offload=prepared.effective_stage_offload,
            postprocess=request.finishing.postprocess,
            latent_upscale=request.finishing.latent_upscale,
            model_filename=prepared.selected_model,
            text_encoder_filename=prepared.selected_text_encoder,
            attention_mode="SLA"
            if prepared.effective_sla
            else "Sage 2"
            if prepared.effective_sage
            else "Sol-Attn"
            if prepared.effective_sol
            else "Kitchen",
            fl2va_voice_reference_count=len(prepared.voice_refs),
            fl2va_voice_reference_mode="Hybrid/native" if prepared.voice_refs else None,
            semantic_bridge=request.sampling.semantic_bridge,
            semantic_bridge_alpha=request.sampling.semantic_bridge_alpha
            if request.sampling.semantic_bridge
            else 0.0,
            semantic_bridge_adapter=MODEL_SPECS["semantic_bridge_v1"].local_name
            if request.sampling.semantic_bridge
            else None,
            semantic_bridge_sha256=MODEL_SPECS["semantic_bridge_v1"].expected_sha256
            if request.sampling.semantic_bridge
            else None,
            semantic_bridge_magnitude_match="per_token"
            if request.sampling.semantic_bridge
            else None,
            sampled_frames=prepared.generation_frames,
            image_frames=prepared.requested_image_frames,
            use_trt_vae=request.output.use_trt_vae,
            use_int8_vae=request.output.use_int8_vae,
        )
        timings.label = f"H3 job {prompt_id}"
        timings.transition("Waiting for ComfyUI")
        attention_status = (
            f"SLA {prepared.effective_sla_preset} "
            f"(sparsity={prepared.effective_sla_inputs['sparsity_ratio']:.2f}, "
            f"block={prepared.effective_sla_inputs['block_size']}, "
            f"dense-last={prepared.effective_sla_inputs['dense_last_steps']}, audio protected)"
            if prepared.effective_sla
            else "Sage 2"
            if prepared.effective_sage
            else f"zero-copy on ({request.sampling.sol_thresh_type}, τ={float(request.sampling.sol_tau):.1f}, "
            f"{request.sampling.sol_exact_mode}, dense-tail-blocks={int(request.sampling.sol_dense_steps)})"
            if prepared.effective_sol
            else "Comfy Kitchen"
        )
        if prepared.effective_cache_mode.lower() == "firstblockcache":
            cache_status = (
                f"FirstBlockCache {request.sampling.fbcache_preset} "
                f"(threshold={float(request.sampling.fbcache_threshold):.3f}, "
                f"window={float(request.sampling.fbcache_start):.2f}–{float(request.sampling.fbcache_end):.2f}, "
                f"max-hits={int(request.sampling.fbcache_max_hits)}, "
                f"temporal-guard={'on' if request.sampling.fbcache_temporal_guard else 'off'})"
            )
        elif prepared.effective_cache_mode.lower() == "spectrum":
            cache_status = (
                "Spectrum (degree=1, audio-isolated offline replay, "
                "audio-blend=0, history/archive=system RAM)"
            )
        elif prepared.effective_cache_mode.lower() == "easycache":
            cache_status = (
                f"EasyCache (threshold={float(request.sampling.easycache_threshold):.2f}, "
                f"{float(request.sampling.easycache_start):.2f}–{float(request.sampling.easycache_end):.2f})"
            )
        else:
            cache_status = "off"
        queued_status = (
            f"Queued `{prompt_id}` · seed {prepared.actual_seed} · "
            f"{prepared.resolved_width}×{prepared.resolved_height} · {prepared.generation_frames} sampled frames · "
            f"result {request.output.result_format.lower()}"
            + (
                f" ({prepared.requested_image_frames} decoded frame(s)) · "
                if request.output.result_format == "Image"
                else " · "
            )
            + f"model {prepared.selected_label} · {prepared.effective_steps} steps/{prepared.effective_scheduler} · "
            f"attention {attention_status} ({prepared.sol_reason}; ~{prepared.packed_tokens:,} target tokens) · "
            f"dense-backend {runtime.dense_attention_backend} · "
            f"cache {cache_status} · unchanged-input reuse "
            f"{'on' if request.media.reuse_unchanged_inputs else 'off'}"
        )
        if request.finishing.latent_upscale:
            queued_status += (
                f"\n\nH3 latent upscale: {request.finishing.latent_upscaler_model} · "
                f"full {prepared.effective_steps}-step generation at "
                f"{prepared.latent_source_width}×{prepared.latent_source_height} → "
                f"{prepared.resolved_width}×{prepared.resolved_height} · "
                f"{int(request.finishing.latent_upscale_refine_steps)} low-denoise refinement steps · "
                f"{prepared.resolved_latent_upscale_method}."
            )
            if prepared.latent_split_config is not None:
                queued_status += (
                    f"\n\nMMH3 split settings: "
                    f"{prepared.latent_split_config.tile_width}×"
                    f"{prepared.latent_split_config.tile_height}px tiles · "
                    f"{prepared.latent_split_config.overlap_ratio:.0%} overlap · "
                    f"{prepared.latent_split_config.chunk_frames}-frame chunks · "
                    f"seam cap {prepared.latent_split_config.seam_denoise:.2f} · "
                    f"polish {prepared.latent_split_config.seam_polish}."
                )
        if prepared.cache_note:
            queued_status += f"\n\nAcceleration notice: {prepared.cache_note}"
        if prepared.generation_note:
            queued_status += f"\n\nGeneration notice: {prepared.generation_note}"
        if websocket_note:
            queued_status += f"\n\nProgress notice: {websocket_note}"
        progress(0, desc="Queued in ComfyUI")
        yield GenerationUpdate(None, queued_status)

        updates = services.execution.poll_comfy_progress(prompt_id, graph)
        for stage, completed_nodes, total_nodes, step, step_total in updates:
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
                    configured_steps=(
                        prepared.effective_steps
                        if stage == "Generating video and audio"
                        else None
                    ),
                    detail=f"Job `{prompt_id}`",
                ),
            )

        timings.transition("Locating generated output")
        progress(1, desc="Resolving generated output")
        yield GenerationUpdate(
            None,
            progress_status(
                "Generation complete; locating output",
                started=started,
                completed_nodes=len(graph),
                total_nodes=len(graph),
            ),
        )
        history = services.execution.wait_for_history(prompt_id)
        if request.output.result_format == "Image":
            results = services.media.resolve_image_outputs(
                history, queued_at, prepared.requested_image_frames
            )
            for path in results:
                write_snapshot(path, execution_snapshot)
            finished_at = time.monotonic()
            elapsed = finished_at - started
            timing_summary = timings.summary(now=finished_at)
            progress(1, desc="Complete")
            yield GenerationUpdate(
                [str(path) for path in results],
                f"Completed in {elapsed:.1f}s · {len(results)} image frame(s) ready "
                f"for selection · seed {prepared.actual_seed}\n\n{timing_summary}",
            )
            return
        if request.output.result_format == "Audio":
            result = services.media.resolve_audio_output(history, queued_at)
            write_snapshot(result, execution_snapshot)
            finished_at = time.monotonic()
            elapsed = finished_at - started
            timing_summary = timings.summary(now=finished_at)
            progress(1, desc="Complete")
            yield GenerationUpdate(
                str(result),
                f"Completed in {elapsed:.1f}s · audio {result.name} · "
                f"seed {prepared.actual_seed} · "
                f"{elapsed / float(request.output.duration):.1f}s compute per output second"
                f"\n\n{timing_summary}",
            )
            return
        source = services.media.resolve_output(history, queued_at)
        fallback_video = source
        write_snapshot(
            source, {**execution_snapshot, "stage": "H3 output before post-processing"}
        )
        if request.finishing.postprocess != "None":
            timings.transition(f"Post-processing: {request.finishing.postprocess}")
            progress(0, desc="Post-processing video")
            yield GenerationUpdate(
                None,
                progress_status(
                    f"Post-processing: {request.finishing.postprocess}", started=started
                ),
            )
        result = yield from finish_video(
            request, prepared, services, source, client_id, started, timings, progress
        )
        write_snapshot(result, execution_snapshot)
        finished_at = time.monotonic()
        elapsed = finished_at - started
        timing_summary = timings.summary(now=finished_at)
        progress(1, desc="Complete")
        yield GenerationUpdate(
            str(result),
            f"Completed in {elapsed:.1f}s · output {result.name} · seed {prepared.actual_seed} · "
            f"{elapsed / float(request.output.duration):.1f}s compute per output second"
            f"\n\n{timing_summary}",
        )
    except Exception as exc:
        fallback = str(fallback_video) if fallback_video is not None else None
        suffix = " The completed H3 video is still available." if fallback else ""
        if "Failed to deserialize TensorRT engine" in str(
            exc
        ) or "deserializeCudaEngine" in str(exc):
            try:
                _, _, marker_path = services.models.trt_vae_decoder_paths(
                    services.models.load_model_config()
                )
                marker_path.unlink(missing_ok=True)
            except Exception:
                pass
        yield GenerationUpdate(fallback, f"Error: {exc}{suffix}\n\n{timings.summary()}")
    finally:
        timings.finish()
