"""MUSIC request orchestration."""

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
from h3_app.workflows.music import build_music3_graph, required_music3_nodes

from .requests import MusicRequest
from .results import GenerationUpdate
from .services import GenerationServices


def generate_music3(
    request: MusicRequest,
    services: GenerationServices,
    runtime: RuntimeConfig,
    *,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    """Run MiniMax Music 3 through the same ComfyUI queue as video jobs."""
    snapshot_values = {
        key: value
        for key, value in asdict(request).items()
        if key
        not in {
            "prompt",
            "negative_prompt",
            "caption",
            "lyrics",
            "first_image",
            "middle_image",
            "end_image",
        }
    }
    started = time.monotonic()
    timings = StageTimings("Music 3 generation", started, "Preparing request")
    queued_at = time.time()
    try:
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(
            None, progress_status("Validating Music 3 request", started=started)
        )
        if not str(request.caption).strip():
            raise H3Error("A music caption is required.")
        if not 1 <= float(request.max_duration) <= 300:
            raise H3Error("Maximum duration must be between 1 and 300 seconds.")
        if not 1 <= int(request.steps) <= 100:
            raise H3Error("Sampling steps must be between 1 and 100.")
        if not 0 <= float(request.cfg) <= 100 or not 0 <= float(request.ar_cfg) <= 100:
            raise H3Error("CFG values must be between 0 and 100.")
        if not 1 <= int(request.top_k) <= 8192:
            raise H3Error("Top K must be between 1 and 8192.")
        actual_seed = (
            random.randrange(0, 2**63 - 1)
            if int(request.seed) < 0
            else int(request.seed)
        )

        missing_files = services.models.missing_music3_model_names(request.model_choice)
        if missing_files:
            yield GenerationUpdate(
                None,
                progress_status(
                    "Downloading MiniMax Music 3 models on demand",
                    started=started,
                    detail=", ".join(missing_files),
                ),
            )
        services.models.ensure_music3_models(request.model_choice)
        available = set(services.execution.object_info())
        missing_nodes = required_music3_nodes(bool(request.tiled_decode)) - available
        if missing_nodes:
            raise H3Error(
                "MiniMax Music 3 requires ComfyUI 0.33.0 or newer; missing nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        graph = build_music3_graph(
            model_choice=request.model_choice,
            caption=str(request.caption).strip(),
            lyrics=str(request.lyrics or "").strip(),
            max_duration=float(request.max_duration),
            seed=actual_seed,
            steps=int(request.steps),
            cfg=float(request.cfg),
            ar_cfg=float(request.ar_cfg),
            top_k=int(request.top_k),
            tiled_decode=bool(request.tiled_decode),
        )
        client_id = str(uuid.uuid4())
        prompt_id = services.execution.submit_prompt(graph, client_id)
        timings.label = f"Music 3 job {prompt_id}"
        timings.transition("Waiting for ComfyUI")
        yield GenerationUpdate(
            None,
            f"Queued Music 3 job `{prompt_id}` · seed {actual_seed} · "
            f"up to {float(request.max_duration):g}s · {request.model_choice}",
        )
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
                    configured_steps=int(request.steps)
                    if stage == "Generating music"
                    else None,
                    detail=f"Music 3 job `{prompt_id}`",
                ),
            )
        timings.transition("Locating generated output")
        result = services.media.resolve_audio_output(
            services.execution.wait_for_history(prompt_id), queued_at
        )
        snapshot_values.update(seed=actual_seed)
        write_snapshot(
            result,
            {"job_id": prompt_id, "family": "Music 3", "settings": snapshot_values},
        )
        finished_at = time.monotonic()
        elapsed = finished_at - started
        timing_summary = timings.summary(now=finished_at)
        progress(1, desc="Complete")
        yield GenerationUpdate(
            str(result),
            f"Music 3 completed in {elapsed:.1f}s · output {result.name} · "
            f"seed {actual_seed}\n\n{timing_summary}",
        )
    except Exception as exc:
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()
