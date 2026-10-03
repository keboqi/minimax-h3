"""Owner-scoped Comfy jobs and process-wide GPU preparation coordination."""

from __future__ import annotations

import uuid
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable
from .input_leases import cleanup_orphans, create_lease, release_lease


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
    lease_lock: Any = field(default=None, repr=False)
    input_bytes: int = 0
    claimed: bool = False
    recoverable_source: str | None = None
    finishing_request: dict | None = None
    finishing_callbacks: dict[int, Any] = field(default_factory=dict, repr=False)
    finishing_offsets: dict[int, int] = field(default_factory=dict)
    recoverable_sources: dict[int, str] = field(default_factory=dict)
    checkpoint: Any = field(default=None, repr=False)
    canvas_request: dict | None = None
    source_asset_ids: list[str] = field(default_factory=list)
    recovered: bool = False
    project_id: str | None = None

    def persist(self):
        if self.checkpoint is not None:
            self.checkpoint(self)

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
        self.persist()

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
        self.persist()  # Write intent before sending anything to ComfyUI.
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
        job.persist()
    return seed


def variant_indices(count):
    job = CURRENT_JOB.get()
    return (
        job.replay_indices if job is not None and job.replay_indices else range(count)
    )


CURRENT_JOB: ContextVar[Job | None] = ContextVar("h3_job", default=None)


class JobCoordinator:
    def __init__(self):
        cleanup_orphans()
        self.gpu = Lock()
        self.admission = Lock()
        self.lock = RLock()
        self.active: dict[str, Job] = {}
        self.records: dict[str, Job] = {}
        self.idempotency: dict[tuple[str, str], str] = {}
        self.retention_seconds = 1800
        self.max_records = 128
        self._stopped = Event()
        self._reaper = None
        self.store = None
        self.callbacks = {}
        self.finishing_factory = None
        self.history_outputs = None

    def configure(self, store):
        """Restore technical records; never dispatch work during recovery."""
        with self.lock:
            if self.store is not None:
                return
            self.store = store
            for payload in store.jobs():
                job = self._restore(payload)
                self.records.setdefault(job.id, job)
                if job.idempotency_key:
                    self.idempotency[(job.owner, job.idempotency_key)] = job.id

    def _restore(self, payload):
        payload = dict(payload)
        for field_name in ("variant_seeds", "recoverable_sources", "finishing_offsets"):
            payload[field_name] = {
                int(k): v for k, v in payload.get(field_name, {}).items()
            }
        payload["failed_variants"] = set(payload.get("failed_variants", []))
        job = Job(**payload, checkpoint=self.store.save_job)
        job.claimed = True
        job.recovered = True
        if job.state == "cancel_requested":
            job.cancelled.set()
        if job.finished_at is None:
            job.state = "recovering"
            job.stage = "Reconnect to recorded ComfyUI submissions; no automatic replay"
            for entry in job.ledger:
                if entry["state"] == "submitting":
                    entry["state"] = "submission_unknown"
            job.claimed = True
            self.active[job.id] = job
        self._restore_project(job)
        return job

    def register_callback(self, family, callback):
        self.callbacks[family] = callback
        for job in self.records.values():
            if job.family == family:
                self._restore_project(job)

    def _restore_project(self, job):
        if self.store is None:
            return
        project = self.store.project_for_job(job.owner, job.id)
        if project:
            job.project_id = project["id"]
            job.snapshot_json = json.dumps(project["values"])
            job.callback = self.callbacks.get(job.family)
            job.ledger = project["ledger"]
            job.finishing_request = project.get("finishing")
            if self.finishing_factory and job.finishing_request:
                self.finishing_factory(job)

    def reconcile(self, owner, get):
        """Observe history/queue only. Missing/ambiguous submissions require review."""
        recovering = [
            job
            for job in self.list_owned(owner)
            if job.recovered and job.state in {"recovering", "cancel_requested"}
        ]
        if not recovering:
            return
        queue = get("/queue").json()
        pending = {
            str(item[1])
            for key in ("queue_pending", "queue_running")
            for item in queue.get(key, [])
            if len(item) > 1
        }
        for job in recovering[:8]:
            waiting = False
            uncertain = False
            for entry in job.ledger:
                prompt_id = entry.get("prompt_id")
                if not prompt_id:
                    if entry["state"] == "submitting":
                        entry["state"] = "submission_unknown"
                    uncertain |= entry["state"] in {"submitting", "submission_unknown"}
                    continue
                if prompt_id in pending:
                    waiting = True
                    continue
                history = get(f"/history/{prompt_id}").json().get(prompt_id)
                if history and (
                    history.get("status", {}).get("completed") or history.get("outputs")
                ):
                    entry["state"] = (
                        "completed"
                        if history.get("status", {}).get("status_str") != "error"
                        else "failed"
                    )
                    job.failed_variants.add(entry["variant"])
                    if self.history_outputs and entry["state"] == "completed":
                        recovered = self.history_outputs(history, entry)
                        job.outputs = list(dict.fromkeys([*job.outputs, *recovered]))
                        entry["recovered_outputs"] = recovered
                    # Finishing/delivery was interrupted even if this backend stage completed.
                elif entry["state"] != "completed":
                    entry["state"] = "submission_unknown"
                    uncertain = True
            if waiting:
                job.stage = "Recorded workflow remains in ComfyUI; reconnect to observe"
            else:
                job.state = (
                    "cancelled"
                    if job.cancelled.is_set()
                    else ("needs_review" if uncertain else "interrupted")
                )
                job.stage = "Recovery requires explicit action; no work was replayed"
                job.finished_at = time.time()
                self.active.pop(job.id, None)
            job.persist()

    def close(self):
        """Release temporary input copies at normal process shutdown."""
        self._stopped.set()
        with self.lock:
            for job in self.records.values():
                if job.input_lease:
                    release_lease(job.input_lease, job.lease_lock)
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
                release_lease(job.input_lease, job.lease_lock)

    def accept(self, owner, family, values, **options):
        # Serial admission protects the input quota and idempotency reservation;
        # unlike the registry lock it never blocks monitoring or cancellation.
        with self.admission:
            return self._accept(owner, family, values, **options)

    def _accept(
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
            if key and self.store and (owner, key) not in self.idempotency:
                saved = self.store.job_for_key(owner, key)
                if saved:
                    old = self._restore(saved)
                    self.records[old.id] = old
                    self.idempotency[(owner, key)] = old.id
            if key and (owner, key) in self.idempotency:
                old = self.records[self.idempotency[(owner, key)]]
                if old.request_digest != digest or old.family != family:
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
        with self.lock:
            retained_bytes = sum(item.input_bytes for item in self.records.values())

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
                job.source_asset_ids.append(
                    uuid.uuid5(uuid.NAMESPACE_URL, source.resolve().as_uri()).hex
                )
                size += source.stat().st_size
                if size + retained_bytes > max_bytes:
                    raise ValueError("Queued inputs exceed H3_JOB_MAX_INPUT_BYTES.")
                if job.input_lease is None:
                    job.input_lease, job.lease_lock = create_lease()
                destination = Path(job.input_lease) / (uuid.uuid4().hex + source.suffix)
                before = (source.stat().st_size, source.stat().st_mtime_ns)
                shutil.copy2(source, destination)
                if before != (source.stat().st_size, source.stat().st_mtime_ns):
                    raise ValueError(
                        "An input changed during acceptance. Upload it again."
                    )
                sources[str(source)] = str(destination)
            return sources[str(source)]

        try:
            for index in media_indices:
                pinned[index] = pin(pinned[index])
            job.snapshot_json = json.dumps(pinned)
            job.input_bytes = size
        except BaseException:
            if job.input_lease:
                release_lease(job.input_lease, job.lease_lock)
            raise
        job.checkpoint = self.store.save_job if self.store else None
        job.persist()
        with self.lock:
            self.records[job.id] = job
            self.active[job.id] = job
            if key:
                self.idempotency[(owner, key)] = job.id
        return job

    def owned(self, owner, job_id):
        with self.lock:
            self._prune()
            job = self.records.get(job_id)
            if job is None and self.store:
                job = self._restore(self.store.job(owner, job_id))
                self.records[job.id] = job
            if job is None or job.owner != owner:
                raise ValueError("This job is unavailable in the current session.")
            return job

    def forget_project(self, owner, project_id):
        """Clear private replay state loaded from a deleted saved project."""
        root = (self.store.root / "projects" / project_id).resolve()

        def contains(value):
            if isinstance(value, dict):
                return any(contains(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(contains(item) for item in value)
            if isinstance(value, str):
                try:
                    return Path(value).is_absolute() and Path(
                        value
                    ).resolve().is_relative_to(root)
                except (ValueError, OSError):
                    return False
            return False

        with self.lock:
            for job in self.records.values():
                if job.owner == owner and (
                    job.project_id == project_id
                    or job.retry_of == project_id
                    or job.snapshot_json
                    and contains(job.values())
                ):
                    job.snapshot_json = None
                    job.callback = None
                    job.project_id = None
                    job.finishing_callbacks.clear()
                    job.finishing_request = None
                    job.replay_ledger.clear()
                    job.ledger = [
                        {
                            key: value
                            for key, value in entry.items()
                            if key not in {"graph_json", "error", "stage"}
                        }
                        for entry in job.ledger
                    ]

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
            if self.store:
                for payload in self.store.jobs(owner):
                    if payload["id"] not in self.records:
                        self.records[payload["id"]] = self._restore(payload)
            return tuple(
                sorted(
                    (job for job in self.records.values() if job.owner == owner),
                    key=lambda job: job.created_at,
                    reverse=True,
                )
            )

    def retry(self, owner, job_id, *, variants=None):
        source = self.owned(owner, job_id)
        if any(
            entry["state"] in {"submitting", "submission_unknown"}
            for entry in source.ledger
        ):
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
        job.canvas_request = source.canvas_request
        job.source_asset_ids = source.source_asset_ids
        job.persist()
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
            job.checkpoint = self.store.save_job if self.store else None
            job.persist()
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
                job.persist()

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
                job.persist()
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
