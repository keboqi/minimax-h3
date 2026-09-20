"""YuE2 generation orchestration over the shared ComfyUI queue."""

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
from h3_app.workflows.yue2 import build_yue2_graph, required_yue2_nodes

from .requests import YuE2Request
from .results import GenerationUpdate
from .services import GenerationServices


def generate_yue2(
    request: YuE2Request,
    services: GenerationServices,
    runtime: RuntimeConfig,
    *,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    """Run YuE2 through ComfyUI and resolve its saved audio output."""
    snapshot_values = {
        key: value
        for key, value in asdict(request).items()
        if key not in {"style", "lyrics", "abc"}
    }
    started = time.monotonic()
    timings = StageTimings("YuE2 generation", started, "Preparing request")
    queued_at = time.time()
    try:
        services.models.unload_prompt_rewriter()
        yield GenerationUpdate(
            None, progress_status("Validating YuE2 request", started=started)
        )
        if not str(request.style).strip():
            raise H3Error("A YuE2 style prompt is required.")
        if not str(request.lyrics).strip():
            raise H3Error("Lyrics or section tags are required.")
        if request.mode not in {"full", "melody", "off"}:
            raise H3Error("YuE2 mode must be full, melody, or off.")
        if not 0.04 <= float(request.max_duration) <= 900:
            raise H3Error("Maximum duration must be between 0.04 and 900 seconds.")
        if not 1 <= int(request.steps) <= 100:
            raise H3Error("Sampling steps must be between 1 and 100.")
        if not 0 <= float(request.cfg) <= 20:
            raise H3Error("Diffusion CFG must be between 0 and 20.")
        if not 0 <= float(request.temperature) <= 5 or not 0 < float(request.top_p) <= 1:
            raise H3Error("Acoustic temperature must be 0–5 and top-p must be in (0, 1].")
        if not 1 <= int(request.top_k) <= 32768:
            raise H3Error("Acoustic top-k must be between 1 and 32768.")
        if not 0.01 <= float(request.repetition_penalty) <= 10:
            raise H3Error("Acoustic repetition penalty must be between 0.01 and 10.")
        if not 1 <= int(request.max_abc_tokens) <= 20000:
            raise H3Error("Maximum ABC tokens must be between 1 and 20000.")
        if not 0 <= float(request.abc_temperature) <= 5 or not 0.01 <= float(request.abc_top_p) <= 1:
            raise H3Error("ABC temperature must be 0–5 and top-p must be in [0.01, 1].")
        if not 1 <= int(request.abc_top_k) <= 32768:
            raise H3Error("ABC top-k must be between 1 and 32768.")
        if not 0.01 <= float(request.abc_repetition_penalty) <= 10:
            raise H3Error("ABC repetition penalty must be between 0.01 and 10.")
        if not 1 <= int(request.abc_penalty_window) <= 20000:
            raise H3Error("ABC penalty window must be between 1 and 20000.")
        actual_seed = (
            random.randrange(0, 2**63 - 1)
            if int(request.seed) < 0
            else int(request.seed)
        )

        missing_files = services.models.missing_yue2_model_names(request.model_choice)
        if missing_files:
            yield GenerationUpdate(
                None,
                progress_status(
                    "Downloading YuE2 checkpoint on demand",
                    started=started,
                    detail=", ".join(missing_files),
                ),
            )
        services.models.ensure_yue2_models(request.model_choice)
        generate_abc = request.mode in {"full", "melody"} and not str(
            request.abc or ""
        ).strip()
        available = set(services.execution.object_info())
        missing_nodes = required_yue2_nodes(
            tiled_decode=bool(request.tiled_decode), generate_abc=generate_abc
        ) - available
        if missing_nodes:
            raise H3Error(
                "YuE2 requires ComfyUI v0.36.0 or newer; missing nodes: "
                + ", ".join(sorted(missing_nodes))
            )
        graph = build_yue2_graph(
            model_choice=request.model_choice,
            style=str(request.style).strip(),
            lyrics=str(request.lyrics).strip(),
            abc=str(request.abc or "").strip(),
            mode=str(request.mode),
            max_duration=float(request.max_duration),
            seed=actual_seed,
            steps=int(request.steps),
            cfg=float(request.cfg),
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            top_k=int(request.top_k),
            repetition_penalty=float(request.repetition_penalty),
            max_abc_tokens=int(request.max_abc_tokens),
            abc_temperature=float(request.abc_temperature),
            abc_top_p=float(request.abc_top_p),
            abc_top_k=int(request.abc_top_k),
            abc_repetition_penalty=float(request.abc_repetition_penalty),
            abc_penalty_window=int(request.abc_penalty_window),
            tiled_decode=bool(request.tiled_decode),
        )
        prompt_id = services.execution.submit_prompt(graph, str(uuid.uuid4()))
        timings.label = f"YuE2 job {prompt_id}"
        timings.transition("Waiting for ComfyUI")
        yield GenerationUpdate(
            None,
            f"Queued YuE2 job {prompt_id} · seed {actual_seed} · "
            f"up to {float(request.max_duration):g}s · {request.model_choice}",
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
                    configured_steps=(
                        int(request.steps) if stage == "Generating YuE2 audio" else None
                    ),
                    detail=f"YuE2 job {prompt_id}",
                ),
            )
        timings.transition("Locating generated output")
        result = services.media.resolve_audio_output(
            services.execution.wait_for_history(prompt_id), queued_at
        )
        snapshot_values.update(seed=actual_seed)
        write_snapshot(
            result,
            {"job_id": prompt_id, "family": "YuE2", "settings": snapshot_values},
        )
        finished_at = time.monotonic()
        elapsed = finished_at - started
        timing_summary = timings.summary(now=finished_at)
        progress(1, desc="Complete")
        yield GenerationUpdate(
            str(result),
            f"YuE2 completed in {elapsed:.1f}s · output {result.name} · "
            f"seed {actual_seed}\n\n{timing_summary}",
        )
    except Exception as exc:
        yield GenerationUpdate(None, f"Error: {exc}\n\n{timings.summary()}")
    finally:
        timings.finish()
