"""Sequence (H3 Continuum) orchestration: one backend submission per group.

Every operation records intent in the ledger, submits once, then reconciles
upstream's committed catalog. "Finish remaining" repeats ordinary one-group
Continue operations under one GPU job, checking cancellation and committing a
checkpoint between groups, so no separate full-run recovery path exists.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from h3_app.config import RuntimeConfig
from h3_app.continuum_policy import parse_run_storage
from h3_app.continuum_store import ContinuumStore, UpstreamProject, run_lock
from h3_app.errors import H3Error
from h3_app.jobs import CURRENT_JOB, check_cancelled
from h3_app.progress import ProgressCallback, no_progress
from h3_app.provenance import write_snapshot
from h3_app.status import progress_status
from h3_app.workflows.continuum import (
    REPORT_ID,
    STATUS_ID,
    build_continuum_graph,
    history_text,
)

from .continuum_requests import (
    FINISH,
    NEXT,
    REBUILD,
    USE_TAKE,
    ContinuumAction,
    ContinuumRequest,
)
from .results import GenerationUpdate
from .services import GenerationServices


@dataclass(frozen=True)
class ContinuumServices:
    store: ContinuumStore
    generation: GenerationServices
    runtime: RuntimeConfig
    # Copy a retained source into ComfyUI input/ under a content-stable name.
    stage: Callable[[str], str]
    # Make app and backend writes durable (e.g. Modal volume commit).
    checkpoint: Callable[[str], None] = lambda operation_id: None


@dataclass(frozen=True)
class OperationResult:
    operation_id: str
    output: str | None
    project: UpstreamProject | None
    reused: int | None
    generated: int | None
    warnings: tuple[str, ...]
    status: str


def _prepare_models(request: ContinuumRequest, services: ContinuumServices) -> dict[str, Any]:
    models_service = services.generation.models
    models = models_service.load_model_config()
    profile_key = models.profile_key(request.model_profile)
    profile = models.profiles[profile_key]
    mode = "Reference media" if request.mode == "Reference media" else "First / last frame"
    models_service.ensure_profile_model(profile_key, profile, mode)
    text_encoder, _ = models_service.ensure_h3_text_encoder(models, request.text_encoder)
    models_service.ensure_base_video_vae(models)
    models_service.ensure_audio_vae(models)
    return {
        "model_name": profile.ref2va if request.mode == "Reference media" else profile.fl2va,
        "text_encoder_name": text_encoder,
        "video_vae_name": models.video_vae,
        "audio_vae_name": models.audio_vae,
        "label": profile.label,
    }


def _expected_reuse(action: ContinuumAction, before: UpstreamProject | None) -> int:
    """Chunks the operation must reuse; fewer means upstream resampled them."""
    accepted = before.accepted_chunks if before else 0
    if action.kind in {NEXT, FINISH}:
        return accepted
    if action.kind in {USE_TAKE, REBUILD}:
        take = next(
            (t for t in (before.takes if before else ()) if t.revision_id == action.take_revision_id),
            None,
        )
        return take.end if take else 0
    if action.kind == "retry" and before and before.review_unit:
        return before.review_unit[0] - 1
    if action.kind == "continue_from_take":
        take = next(
            (t for t in (before.takes if before else ()) if t.revision_id == action.take_revision_id),
            None,
        )
        return take.end if take else 0
    if action.kind == "regenerate_from":
        return max(0, action.regenerate_from - 1)
    return 0


def _projected_chunks(action: ContinuumAction, before: UpstreamProject | None) -> int:
    """Saved chunks after the operation, to route complete results to Media."""
    accepted = before.accepted_chunks if before else 0
    take = next(
        (t for t in (before.takes if before else ()) if t.revision_id == action.take_revision_id),
        None,
    )
    if action.kind in {NEXT, FINISH}:
        return accepted + 1
    if action.kind in {USE_TAKE, REBUILD}:
        return take.end if take else accepted
    if action.kind == "continue_from_take":
        return (take.end if take else accepted) + 1
    if action.kind == "regenerate_from":
        return action.regenerate_from
    return accepted


def preview_directory(run_id: str) -> str:
    """Partial previews stay out of Media (the gallery skips h3_continuum/)."""
    return f"h3_continuum/previews/{run_id}"


def run_operation(
    owner: str,
    run_id: str,
    action: ContinuumAction,
    services: ContinuumServices,
    *,
    expected_epoch: int,
    key: str | None = None,
    parent_id: str | None = None,
    reuse_cap: int | None = None,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate | OperationResult]:
    """Run exactly one upstream review operation; yields progress, then a result."""
    store = services.store
    started = time.monotonic()
    run = store.run(owner, run_id)
    version, request = store.request(owner, run_id)
    operation = store.reserve(
        owner,
        run_id,
        action,
        expected_epoch=expected_epoch,
        key=key,
        parent_id=parent_id,
        request_version=version,
    )
    if operation["state"] != "reserved":
        raise H3Error("This Sequence action was already submitted; refresh the sequence.")
    before = store.project(run)
    try:
        if action.kind in {USE_TAKE, REBUILD} and before is None:
            raise H3Error("This sequence has no saved takes yet.")
        progress(0, desc="Preparing Sequence models")
        yield GenerationUpdate(None, progress_status("Preparing Sequence models", started=started))
        models = _prepare_models(request, services)
        available = set(services.generation.execution.object_info())
        staged = {
            path: services.stage(path)
            for path in (request.first_image, *request.reference_images)
            if path
        }
        graph = build_continuum_graph(
            request,
            action,
            project_id=str(run["project_uuid"]),
            model_name=models["model_name"],
            text_encoder_name=models["text_encoder_name"],
            video_vae_name=models["video_vae_name"],
            audio_vae_name=models["audio_vae_name"],
            staged_inputs=staged,
            available_nodes=available,
            filename_prefix=(
                f"h3/sequence/{run_id[:12]}"
                if _projected_chunks(action, before) >= request.chunk_count
                else f"{preview_directory(run_id)}/preview"
            ),
        )
        check_cancelled()
        store.update_operation(operation["id"], state="submitting")
        queued_at = time.time()
        prompt_id = services.generation.execution.submit_prompt(graph, uuid.uuid4().hex)
        store.update_operation(operation["id"], state="submitted", prompt_id=str(prompt_id))
        store.set_state(run_id, "sampling" if action.samples else "exporting")
        for stage, done, total, step, step_total in services.generation.execution.poll_comfy_progress(prompt_id, graph):
            if step is not None and step_total:
                progress((step, step_total), desc=stage)
            elif total:
                progress((done, total), desc=stage)
            yield GenerationUpdate(
                None,
                progress_status(
                    stage,
                    started=started,
                    completed_nodes=done,
                    total_nodes=total,
                    step=step,
                    step_total=step_total,
                    detail=f"Sequence job `{prompt_id}`",
                ),
            )
        history = services.generation.execution.wait_for_history(prompt_id)
        status = history_text(history, STATUS_ID)
        report = history_text(history, REPORT_ID)
        output = services.generation.media.resolve_output(history, queued_at)
    except BaseException as exc:
        cancelled = CURRENT_JOB.get() is not None and CURRENT_JOB.get().cancelled.is_set()
        store.update_operation(
            operation["id"],
            state="cancelled" if cancelled else "failed",
            receipt={"error": type(exc).__name__},
        )
        # Committed groups survive any failure; the next view rereads them.
        store.set_state(run_id, "interrupted" if cancelled else "failed")
        raise
    after = store.project(run)
    storage = parse_run_storage(status)
    warnings = []
    expected = _expected_reuse(action, before)
    if reuse_cap is not None:
        # The request was just edited: chunks from the first change regenerate.
        expected = min(expected, max(0, reuse_cap))
    if storage is None:
        warnings.append("Upstream did not report saved-chunk reuse; verify the result.")
    else:
        if storage.reused < expected:
            warnings.append(
                f"Expected to reuse {expected} saved chunk(s) but upstream reused "
                f"{storage.reused} and regenerated the rest. Settings may have "
                "changed the reuse contract."
            )
        if storage.reuse_disabled:
            warnings.append(f"Upstream disabled saved-chunk reuse: {storage.reuse_disabled}.")
        if not action.samples and storage.generated:
            warnings.append(
                f"Take selection unexpectedly sampled {storage.generated} chunk(s)."
            )
    receipt = {
        "reused": storage.reused if storage else None,
        "generated": storage.generated if storage else None,
        "expected_reuse": expected,
        "head": after.head_revision if after else None,
        "accepted": after.accepted_chunks if after else 0,
        "warnings": warnings,
    }
    complete = bool(after and after.accepted_chunks >= request.chunk_count)
    store.update_operation(operation["id"], state="completed", receipt=receipt)
    store.set_state(
        run_id,
        "complete" if complete else "review",
        head_revision=receipt["head"],
        output=str(output),
        report={"status": status[-4000:], "report": report[-4000:], "warnings": warnings},
    )
    previews = (Path(services.runtime.output_dir) / "h3_continuum").resolve()
    if complete and not Path(output).resolve().is_relative_to(previews):
        # Only a completed sequence enters Media; its snapshot registers it.
        write_snapshot(
            Path(output),
            {
                "job_id": str(prompt_id),
                "stage": "H3 Sequence (Continuum)",
                "settings": {
                    "mode": request.mode,
                    "chunks": request.chunk_count,
                    "chunk_seconds": request.chunk_seconds,
                    "continuity": request.continuity,
                    "seed": request.seed,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "scheduler": request.scheduler,
                    "model_filename": models["model_name"],
                    "text_encoder_filename": models["text_encoder_name"],
                    "sequence_action": action.kind,
                    "reused_chunks": receipt["reused"],
                    "generated_chunks": receipt["generated"],
                },
            },
        )
    services.checkpoint(operation["id"])
    yield OperationResult(
        operation_id=operation["id"],
        output=str(output),
        project=after,
        reused=receipt["reused"],
        generated=receipt["generated"],
        warnings=tuple(warnings),
        status=status,
    )


def execute(
    owner: str,
    run_id: str,
    action: ContinuumAction,
    services: ContinuumServices,
    *,
    expected_epoch: int,
    key: str | None = None,
    reuse_cap: int | None = None,
    progress: ProgressCallback = no_progress,
) -> Iterator[GenerationUpdate]:
    """Run an app action. Finish repeats one-group Continue until complete."""
    started = time.monotonic()
    with run_lock(run_id):
        request = services.store.request(owner, run_id)[1]
        epoch = int(expected_epoch)
        parent = None
        result: OperationResult | None = None
        while True:
            step = ContinuumAction(NEXT) if action.kind == FINISH else action
            for update in run_operation(
                owner,
                run_id,
                step,
                services,
                expected_epoch=epoch,
                key=key if parent is None else None,
                parent_id=parent,
                reuse_cap=reuse_cap if parent is None else None,
                progress=progress,
            ):
                if isinstance(update, OperationResult):
                    result = update
                else:
                    yield update
            assert result is not None
            accepted = result.project.accepted_chunks if result.project else 0
            yield GenerationUpdate(
                result.output,
                summarize(result, request.chunk_count, started=started),
            )
            if action.kind != FINISH or accepted >= request.chunk_count or result.warnings:
                return
            check_cancelled()
            # Later groups are children of the first reservation; only the
            # first operation compares the epoch captured by the browser.
            epoch = int(services.store.run(owner, run_id)["epoch"])
            parent = parent or result.operation_id


def summarize(result: OperationResult, chunk_count: int, *, started: float) -> str:
    accepted = result.project.accepted_chunks if result.project else 0
    lines = [
        f"Sequence: {accepted}/{chunk_count} chunk(s) saved · "
        f"{time.monotonic() - started:.1f}s"
    ]
    if result.reused is not None:
        lines.append(f"Reused {result.reused} saved chunk(s), generated {result.generated}.")
    if accepted < chunk_count:
        lines.append("Preview shows the sequence so far. Continue, retry, or pick a take.")
    else:
        lines.append("All chunks are saved; the preview is the assembled sequence.")
    for warning in result.warnings:
        lines.append(f"⚠ {warning}")
    return "\n\n".join(lines)
