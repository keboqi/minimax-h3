"""Session-owned Comfy jobs and process-wide GPU preparation coordination."""

from __future__ import annotations

import uuid
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable


class JobCancelled(RuntimeError):
    pass


@dataclass
class Job:
    owner: str
    family: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt_id: str | None = None
    output_token: str | None = None
    cancelled: Event = field(default_factory=Event)
    has_gpu: bool = False
    submissions: list[Any] = field(default_factory=list)
    state: str = "queued"
    stage: str = "Waiting for execution"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    variant: int = 0
    variant_seeds: dict[int, int] = field(default_factory=dict)
    failed_variants: set[int] = field(default_factory=set)
    retry_of: str | None = None
    replay_seeds: tuple[int, ...] = ()
    replay_indices: tuple[int, ...] = ()
    replay_ledger: list[dict] = field(default_factory=list, repr=False)
    replay_counts: dict[int, int] = field(default_factory=dict)
    idempotency_key: str | None = None
    request_digest: str | None = None
    snapshot_json: str | None = None
    callback: Any = field(default=None, repr=False)
    media_indices: tuple[int, ...] = ()
    output_indices: tuple[int, ...] = ()
    input_lease: str | None = None
    input_bytes: int = 0
    claimed: bool = False
    recoverable_source: str | None = None
    finishing_request: dict | None = None
    finishing_callbacks: dict[int, Any] = field(default_factory=dict, repr=False)
    finishing_offsets: dict[int, int] = field(default_factory=dict)
    recoverable_sources: dict[int, str] = field(default_factory=dict)

    def retry_finishing(self, index):
        callback = self.finishing_callbacks.get(index)
        source = self.recoverable_sources.get(index)
        if callback is None or not source or not Path(source).is_file():
            raise ValueError("The completed source is unavailable for finishing.")
        return callback, source

    def check(self):
        if self.cancelled.is_set():
            raise JobCancelled("Generation interrupted.")

    def values(self):
        if self.snapshot_json is None:
            raise ValueError("This job has no retained request to replay.")
        return json.loads(self.snapshot_json)

    def fail(self, error):
        self.error = str(error)
        self.failed_variants.add(self.variant)
        self.state = (
            "cancelled"
            if self.cancelled.is_set() or isinstance(error, JobCancelled)
            else "failed"
        )
        for entry in reversed(self.ledger):
            if entry["variant"] == self.variant and entry["state"] in {
                "submitting",
                "submitted",
            }:
                entry["state"] = self.state
                entry["error"] = str(error)
                break

    def observe_submission(self, prompt_id, token, graph):
        seeds = {
            f"{node_id}.{key}": value
            for node_id, node in graph.items()
            for key, value in node.get("inputs", {}).items()
            if key in {"seed", "noise_seed"} and isinstance(value, int)
        }
        self.ledger.append(
            {
                "variant": self.variant,
                "stage": self.stage,
                "prompt_id": str(prompt_id) if prompt_id is not None else None,
                "output_token": token,
                "seeds": seeds,
                "graph_json": json.dumps(graph),
                "state": "submitted" if prompt_id is not None else "submitting",
                "submitted_at": time.time(),
            }
        )
        return self.ledger[-1]


def record_failure(error):
    job = CURRENT_JOB.get()
    if job is not None:
        job.fail(error)


def variant_seed(index, resolve):
    """Resolve once, or replay the recorded seed for this exact variant."""
    job = CURRENT_JOB.get()
    if job is None:
        return resolve()
    job.variant = index
    seed = job.variant_seeds.get(index)
    if seed is None:
        seed = resolve()
        job.variant_seeds[index] = seed
    return seed


def variant_indices(count):
    job = CURRENT_JOB.get()
    return (
        job.replay_indices if job is not None and job.replay_indices else range(count)
    )


CURRENT_JOB: ContextVar[Job | None] = ContextVar("h3_job", default=None)


class JobCoordinator:
    def __init__(self):
        self.gpu = Lock()
        self.lock = RLock()
        self.active: dict[str, Job] = {}
        self.records: dict[str, Job] = {}
        self.idempotency: dict[tuple[str, str], str] = {}
        self.retention_seconds = 1800
        self.max_records = 128
        self._stopped = Event()
        self._reaper = None

    def close(self):
        """Release temporary input copies at normal process shutdown."""
        self._stopped.set()
        with self.lock:
            for job in self.records.values():
                if job.input_lease:
                    shutil.rmtree(job.input_lease, ignore_errors=True)
            self.records.clear()
            self.active.clear()
            self.idempotency.clear()

    def _start_retention_cleanup(self):
        if self._reaper is not None:
            return

        def reap():
            while not self._stopped.wait(30):
                with self.lock:
                    self._prune()

        self._reaper = Thread(target=reap, name="h3-input-retention", daemon=True)
        self._reaper.start()

    def _prune(self):
        now = time.time()
        expired = [
            job
            for job in self.records.values()
            if job.finished_at is not None
            and now - job.finished_at > self.retention_seconds
        ]
        terminals = sorted(
            (
                job
                for job in self.records.values()
                if job.finished_at is not None and job not in expired
            ),
            key=lambda job: job.finished_at,
        )
        expired.extend(
            terminals[: max(0, len(self.records) - len(expired) - self.max_records + 1)]
        )
        for job in expired:
            self.records.pop(job.id, None)
            if job.idempotency_key:
                self.idempotency.pop((job.owner, job.idempotency_key), None)
            if job.input_lease:
                shutil.rmtree(job.input_lease, ignore_errors=True)

    def accept(
        self,
        owner,
        family,
        values,
        *,
        callback=None,
        media_indices=(),
        output_indices=(),
        key=None,
        retry_of=None,
    ):
        """Capture once before queueing; this registry does not dispatch work."""
        encoded = json.dumps(values, sort_keys=True, default=str)
        if len(encoded.encode()) > 2 * 1024**2:
            raise ValueError(
                "The request snapshot exceeds 2 MiB. Shorten the prompt or reduce input entries."
            )
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.lock:
            if self._stopped.is_set():
                raise ValueError("The job registry is shutting down.")
            self._start_retention_cleanup()
            self._prune()
            if key and (owner, key) in self.idempotency:
                old = self.records[self.idempotency[(owner, key)]]
                if old.request_digest != digest:
                    raise ValueError(
                        "Idempotency key already belongs to another request."
                    )
                return old
            if (
                sum(
                    job.owner == owner and job.finished_at is None
                    for job in self.records.values()
                )
                >= 8
            ):
                raise ValueError(
                    "This session already has eight waiting or active jobs."
                )
            if len(self.records) >= self.max_records:
                raise ValueError(
                    "The job registry is full. Cancel waiting work before submitting more."
                )
            job = Job(
                owner,
                family,
                callback=callback,
                media_indices=tuple(media_indices),
                output_indices=tuple(output_indices),
                retry_of=retry_of,
                idempotency_key=key,
                request_digest=digest,
            )
            pinned = copy.deepcopy(list(values))
            sources = {}
            size = 0
            max_bytes = int(os.getenv("H3_JOB_MAX_INPUT_BYTES", str(8 * 1024**3)))

            def pin(value):
                nonlocal size
                if value is None:
                    return None
                if isinstance(value, (tuple, list)):
                    return [pin(item) for item in value]
                if isinstance(value, dict):
                    result = dict(value)
                    for name in ("path", "name", "video", "audio"):
                        if result.get(name):
                            result[name] = pin(result[name])
                    return result
                source = Path(value)
                if not source.is_file():
                    raise ValueError(
                        "An input file is missing. Upload it again before submitting."
                    )
                if str(source) not in sources:
                    size += source.stat().st_size
                    if (
                        size + sum(item.input_bytes for item in self.records.values())
                        > max_bytes
                    ):
                        raise ValueError("Queued inputs exceed H3_JOB_MAX_INPUT_BYTES.")
                    if job.input_lease is None:
                        job.input_lease = tempfile.mkdtemp(prefix="h3-job-")
                    destination = Path(job.input_lease) / (
                        uuid.uuid4().hex + source.suffix
                    )
                    shutil.copy2(source, destination)
                    sources[str(source)] = str(destination)
                return sources[str(source)]

            try:
                for index in media_indices:
                    pinned[index] = pin(pinned[index])
                job.snapshot_json = json.dumps(pinned)
                job.input_bytes = size
            except BaseException:
                if job.input_lease:
                    shutil.rmtree(job.input_lease, ignore_errors=True)
                raise
            self.records[job.id] = job
            self.active[job.id] = job
            if key:
                self.idempotency[(owner, key)] = job.id
            return job

    def owned(self, owner, job_id):
        with self.lock:
            self._prune()
            job = self.records.get(job_id)
            if job is None or job.owner != owner:
                raise ValueError("This job is unavailable in the current session.")
            return job

    def references_path(self, path):
        """Protect managed sources used by accepted maintenance requests."""
        target = str(Path(path).resolve())

        def contains(value):
            if isinstance(value, dict):
                return any(contains(item) for item in value.values())
            if isinstance(value, list):
                return any(contains(item) for item in value)
            return (
                isinstance(value, str)
                and value == str(path)
                or isinstance(value, str)
                and value == target
            )

        with self.lock:
            return any(
                job.snapshot_json and contains(job.values())
                for job in self.active.values()
            )

    def list_owned(self, owner):
        with self.lock:
            self._prune()
            return tuple(
                sorted(
                    (job for job in self.records.values() if job.owner == owner),
                    key=lambda job: job.created_at,
                    reverse=True,
                )
            )

    def retry(self, owner, job_id, *, variants=None):
        source = self.owned(owner, job_id)
        if any(entry["state"] == "submission_unknown" for entry in source.ledger):
            raise ValueError(
                "A backend submission has an unknown outcome. Inspect the ComfyUI queue before explicitly submitting new work."
            )
        if source.finished_at is None or source.callback is None:
            raise ValueError("Only a finished job with retained inputs can be retried.")
        values = source.values()
        selected = sorted(source.failed_variants if variants is None else variants)
        if not selected:
            raise ValueError("No failed variants to retry.")
        if any(index not in source.failed_variants for index in selected):
            raise ValueError(
                "Select failed variants only; completed outputs are preserved."
            )
        if any(index not in source.variant_seeds for index in selected):
            raise ValueError(
                "A variant has no recorded seed. Submit a new request explicitly."
            )
        job = self.accept(
            owner,
            source.family,
            values,
            callback=source.callback,
            media_indices=source.media_indices,
            output_indices=source.output_indices,
            key=uuid.uuid4().hex,
            retry_of=source.id,
        )
        job.replay_seeds = (
            tuple(source.variant_seeds[index] for index in selected)
            if source.family == "h3"
            else ()
        )
        job.replay_indices = tuple(selected)
        job.variant = selected[0]
        job.variant_seeds = {index: source.variant_seeds[index] for index in selected}
        job.replay_ledger = copy.deepcopy(
            [entry for entry in source.ledger if entry["variant"] in selected]
        )
        return job

    @contextmanager
    def run(self, owner: str, family: str, *, accepted: Job | None = None):
        job = accepted or Job(owner, family)
        with self.lock:
            self._prune()
            if job.owner != owner or job.family != family:
                raise ValueError("Job ownership mismatch.")
            if job.claimed:
                raise ValueError("This job has already been claimed.")
            job.claimed = True
            self.active[job.id] = job
            self.records[job.id] = job

        try:
            job.check()
            with self.gpu:
                job.check()
                job.state = "preparing"
                job.stage = "Preparing inputs and models"
                job.has_gpu = True
                try:
                    yield job
                finally:
                    for submission in job.submissions:
                        submission.close()
                    job.has_gpu = False
        except BaseException as exc:
            job.fail(exc)
            raise
        finally:
            with self.lock:
                if job.cancelled.is_set():
                    job.state = "cancelled"
                elif job.failed_variants:
                    job.state = "failed"
                elif job.state not in {"failed", "cancelled"}:
                    job.state = "completed"
                job.finished_at = time.time()
                self.active.pop(job.id, None)

    @contextmanager
    def maintenance(self, family: str):
        """Own standalone GPU work; reuse a generation lease without relocking."""
        current = CURRENT_JOB.get()
        with self.lock:
            owned = (
                current is not None
                and current.has_gpu
                and self.active.get(current.id) is current
            )
        if owned:
            current.check()
            yield current
            return
        with self.run(uuid.uuid4().hex, family) as job:
            token = CURRENT_JOB.set(job)
            try:
                yield job
            finally:
                CURRENT_JOB.reset(token)

    def cancel(
        self, owner: str, family: str, get: Callable, post: Callable, *, job_id=None
    ) -> str:
        with self.lock:
            jobs = [
                job
                for job in self.active.values()
                if job.owner == owner
                and (job_id is None or job.id == job_id)
                and job.family in ({family, "h3-input"} if family == "h3" else {family})
            ]
            for job in jobs:
                job.cancelled.set()
                job.state = "cancel_requested"
                if not job.claimed:
                    job.state = "cancelled"
                    job.finished_at = time.time()
                    self.active.pop(job.id, None)
            prompt_ids = {job.prompt_id for job in jobs if job.prompt_id}
            if not jobs:
                return "No active job in this session and tab."
            if not prompt_ids:
                return "Cancellation requested during preparation."
            # ComfyUI interrupt is global: only send it if the executing ID is ours.
            queue = get("/queue").json()
            running = {
                str(row[1]) for row in queue.get("queue_running", []) if len(row) > 1
            }
            pending = [
                str(row[1])
                for row in queue.get("queue_pending", [])
                if len(row) > 1 and str(row[1]) in prompt_ids
            ]
            if pending:
                post("/queue", json={"delete": pending})
            if running & prompt_ids:
                # A final check avoids interrupting an unrelated externally queued job.
                latest = get("/queue").json()
                if any(
                    len(row) > 1 and str(row[1]) in prompt_ids
                    for row in latest.get("queue_running", [])
                ):
                    post("/interrupt", json={})
            return "Cancellation requested for this session and tab."


JOBS = JobCoordinator()
atexit.register(JOBS.close)


def scoped_graph(graph: dict[str, Any], token: str) -> None:
    """Keep family folders and add an unguessable submission prefix to saves."""
    for node in graph.values():
        inputs = node.get("inputs", {})
        prefix = inputs.get("filename_prefix")
        if isinstance(prefix, str):
            inputs["filename_prefix"] = f"{prefix}_{token}"


def check_cancelled() -> None:
    job = CURRENT_JOB.get()
    if job:
        job.check()


def gpu_maintenance(family: str):
    """Keep synchronous local GPU callbacks serialized outside the UI too."""
    from functools import wraps

    def decorate(callback):
        @wraps(callback)
        def run(*args, **kwargs):
            with JOBS.maintenance(family):
                return callback(*args, **kwargs)

        return run

    return decorate
