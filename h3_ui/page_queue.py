"""Server-side queues that start one page run after another, for unattended batches.

Storyboard's auto render and Soundstage's batch produce share this: one queue per owner, run in
a thread of its own, each item started as the same page run a person would start by hand and
waited on before the next. Stopping interrupts the job the current run is waiting on and skips
the rest. A queue lives in this process only: a restart ends it, and nothing resumes on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock, Thread
import time
from typing import Any, Callable
import uuid

from h3_app.errors import H3Error
from h3_app.workspace_activity import workspace_task

from .page_runs import Run

WAITING, FAILED, STOPPED, RECOVERED, BLOCKED = "waiting", "failed", "stopped", "recovered", "blocked"


@dataclass
class QueueItem:
    key: str  # What the item works on: a storyboard id, a project name.
    name: str
    state: str = WAITING
    detail: str = ""
    job_id: str | None = None
    input_revision: str | None = None
    recovered: bool = False
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at is not None and self.finished_at is not None and self.finished_at >= self.started_at:
            return round(self.finished_at - self.started_at, 2)
        return None

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.key, "name": self.name, "state": self.state, "detail": self.detail}
        if self.job_id:
            out["job_id"] = self.job_id
        if self.input_revision:
            out["input_revision"] = self.input_revision
        if self.recovered:
            out["recovered"] = True
        if self.duration_s is not None:
            out["duration_s"] = self.duration_s
        return out

    def serialize(self) -> dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "state": self.state, "detail": self.detail,
            "job_id": self.job_id, "input_revision": self.input_revision, "recovered": self.recovered,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "QueueItem":
        return cls(
            key=str(data.get("key") or data.get("id") or ""),
            name=str(data.get("name") or ""),
            state=str(data.get("state") or WAITING),
            detail=str(data.get("detail") or ""),
            job_id=data.get("job_id") or None,
            input_revision=data.get("input_revision") or None,
            recovered=bool(data.get("recovered", False)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )


@dataclass
class RunQueue:
    owner: str
    items: list[QueueItem]
    session: Any
    options: dict = field(default_factory=dict)  # What every item's run needs, such as the writer service.
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    current: QueueItem | None = None
    run: Run | None = None  # The current item's run.
    stopping: bool = False
    interrupted: set[str] = field(default_factory=set)  # Jobs already asked to stop.
    finished_at: float | None = None
    error: str | None = None
    resumable: bool = False


@dataclass(frozen=True)
class QueueMessages:
    busy: str  # Starting while a queue runs.
    idle: str  # Stopping when none runs.
    interrupting: str  # Stopping the current item's job.
    pending: str  # Stopping before the current item's job was accepted.
    after_step: str  # Stopping work that can't be interrupted.
    not_started: str  # The detail of an item a stop skipped.


class RunQueues:
    """One queue per owner; subclasses say how one item is worked through in ``process``.

    ``interrupt(session, run)`` interrupts the job ``run`` waits on; without it a stop takes
    effect once the current item's run ends. When ``store`` and ``kind`` are provided, queue
    state persists across restarts and reconciles accepted jobs without auto-replaying ambiguous ones.
    """

    working: tuple[str, ...] = ()  # States, besides waiting, of an item not finished yet.
    thread_name = "page-queue"

    def __init__(self, messages: QueueMessages, interrupt: Callable[[Any, Run], str] | None = None, *,
                 store=None, kind: str | None = None, job_lookup: Callable[[str], dict | None] | None = None):
        self.messages = messages
        self.interrupt = interrupt
        self.store = store
        self.kind = kind
        self.job_lookup = job_lookup
        self.lock = RLock()
        self.queues: dict[str, RunQueue] = {}
        if self.store is not None and self.kind:
            self._restore_from_store()

    # Persistence & recovery ------------------------------------------------
    def _lookup_job(self, job_id: str) -> dict | None:
        if self.job_lookup is not None:
            return self.job_lookup(job_id)
        if self.store is not None and hasattr(self.store, "load_job_payload"):
            return self.store.load_job_payload(job_id)
        return None

    def _checkpoint(self, queue: RunQueue) -> None:
        if self.store is None or not self.kind or not hasattr(self.store, "save_run_queue"):
            return
        try:
            serializable_opts = {k: v for k, v in queue.options.items() if not callable(v)}
            payload = {
                "id": queue.id,
                "owner": queue.owner,
                "options": serializable_opts,
                "stopping": queue.stopping,
                "finished_at": queue.finished_at,
                "error": queue.error,
                "resumable": queue.resumable,
                "items": [item.serialize() for item in queue.items],
            }
            self.store.save_run_queue(self.kind, queue.owner, payload)
        except Exception:
            pass  # Persistence must not crash an in-flight queue.

    def _reconcile_item(self, item: QueueItem) -> None:
        if item.state not in self.working:
            return
        if item.job_id:
            job = self._lookup_job(item.job_id)
            if isinstance(job, dict):
                state = str(job.get("state") or "")
                ledger = job.get("ledger") or []
                has_unknown = any(entry.get("state") == "submission_unknown" for entry in ledger if isinstance(entry, dict))
                if state in ("completed", "done") and job.get("finished_at") is not None and not job.get("error"):
                    item.state = RECOVERED
                    item.recovered = True
                    item.detail = "Recovered confirmed output from previous run."
                    return
                if has_unknown or state in ("recovering", "cancel_requested", "running", "submitting"):
                    item.state = BLOCKED
                    item.detail = "Submission status unknown after restart; reconcile before retrying."
                    return
            else:
                item.state = BLOCKED
                item.detail = "Accepted job record unavailable after restart; reconcile before retrying."
                return
        item.state = WAITING
        item.detail = "Interrupted before job submission; ready to resume."

    def _restore_from_store(self) -> None:
        if not hasattr(self.store, "load_run_queues"):
            return
        for owner, data in self.store.load_run_queues(self.kind).items():
            items = [QueueItem.deserialize(row) for row in (data.get("items") or []) if isinstance(row, dict)]
            if not items:
                continue
            was_stopped = bool(data.get("stopping"))
            was_finished = data.get("finished_at") is not None
            queue = RunQueue(
                owner=owner,
                items=items,
                session=None,
                options=dict(data.get("options") or {}),
                id=str(data.get("id") or uuid.uuid4().hex),
                stopping=was_stopped,
                finished_at=data.get("finished_at"),
                error=data.get("error"),
                resumable=bool(data.get("resumable", False)),
            )
            if not was_finished:
                if was_stopped:
                    for item in queue.items:
                        if item.state in (WAITING, *self.working):
                            item.state, item.detail = STOPPED, item.detail or self.messages.not_started
                    queue.resumable = False
                else:
                    for item in queue.items:
                        self._reconcile_item(item)
                    queue.resumable = any(item.state == WAITING for item in queue.items)
                queue.finished_at = time.time()
                self._checkpoint(queue)
            self.queues[owner] = queue

    # Hooks -----------------------------------------------------------------
    def process(self, queue: RunQueue, item: QueueItem) -> None:
        raise NotImplementedError

    def interruptible(self, run: Run) -> bool:
        """Whether a stop interrupts ``run``'s job, or lets it finish."""
        return True

    def stopping(self, run: Run) -> None:
        """Tell ``run`` itself that the queue is stopping, for work between its jobs."""

    # The queues ------------------------------------------------------------
    def latest(self, owner) -> RunQueue | None:
        return self.queues.get(owner)

    def active(self, owner) -> RunQueue | None:
        queue = self.queues.get(owner)
        return queue if queue is not None and queue.finished_at is None else None

    def payload(self, owner) -> dict[str, Any] | None:
        """The owner's latest queue as the page reads it, or None."""
        queue = self.queues.get(owner)
        if queue is None:
            return None
        finished = sum(item.state not in (WAITING, BLOCKED, *self.working) for item in queue.items)
        live = queue.finished_at is None and queue.run is not None and queue.run.finished_at is None
        blocked = sum(item.state == BLOCKED for item in queue.items)
        out = {
            "id": queue.id, "active": queue.finished_at is None, "stopping": queue.stopping, "error": queue.error,
            "current": queue.current.key if queue.current and queue.finished_at is None else None,
            "status": queue.run.status if live else "", "stage": getattr(queue.run, "stage", None) if live else None,
            "finished": finished, "items": [item.public() for item in queue.items],
        }
        if queue.resumable or blocked:
            out["resumable"] = bool(queue.resumable and not blocked and not queue.stopping and queue.finished_at is not None)
            out["blocked"] = blocked
        if queue.finished_at is None:
            durations = sorted(d for d in (item.duration_s for item in queue.items) if d is not None and d > 0)
            remaining = sum(item.state in (WAITING, *self.working) for item in queue.items)
            if durations and remaining:
                p50 = durations[len(durations) // 2]
                out["eta_seconds"] = round(p50 * remaining, 1)
        return out

    def start(self, owner, items: list[QueueItem], session, **options) -> RunQueue:
        with self.lock:
            if self.active(owner) is not None:
                raise H3Error(self.messages.busy)
            queue = self.queues[owner] = RunQueue(owner, items, session, options)
            self._checkpoint(queue)
        Thread(target=workspace_task(self._execute), args=(queue,), name=self.thread_name, daemon=True).start()
        return queue

    def resume(self, owner, session) -> RunQueue:
        """Resume a persisted interrupted queue from its remaining WAITING items without duplicating finished ones."""
        with self.lock:
            if self.active(owner) is not None:
                raise H3Error(self.messages.busy)
            queue = self.queues.get(owner)
            if queue is None or not queue.resumable or queue.stopping:
                raise H3Error("No interrupted batch is waiting to resume.")
            if any(item.state == BLOCKED for item in queue.items):
                raise H3Error("Reconcile ambiguous submissions before resuming this batch.")
            queue.session = session
            queue.finished_at = None
            queue.resumable = False
            queue.error = None
            self._checkpoint(queue)
        Thread(target=workspace_task(self._execute), args=(queue,), name=self.thread_name, daemon=True).start()
        return queue

    def stop(self, owner, session) -> str:
        queue = self.active(owner)
        if queue is None:
            return self.messages.idle
        queue.stopping = True
        queue.resumable = False
        self._checkpoint(queue)
        run = queue.run
        if run is None or run.finished_at is not None or self.interrupt is None or not self.interruptible(run):
            if run is not None and run.finished_at is None:
                self.stopping(run)
            return self.messages.after_step
        self.stopping(run)
        if run.job_id is None:
            return self.messages.pending
        self._interrupt(queue, run, session)
        return self.messages.interrupting

    # The worker ------------------------------------------------------------
    def _interrupt(self, queue: RunQueue, run: Run, session) -> None:
        with self.lock:
            if run.job_id in queue.interrupted:
                return
            queue.interrupted.add(run.job_id)
        self.interrupt(session, run)

    def _execute(self, queue: RunQueue):
        try:
            for item in queue.items:
                if item.state != WAITING:
                    continue
                if queue.stopping:
                    item.state, item.detail = STOPPED, self.messages.not_started
                    self._checkpoint(queue)
                    continue
                queue.current = item
                item.started_at = time.time()
                self._checkpoint(queue)
                self.process(queue, item)
                item.finished_at = time.time()
                self._checkpoint(queue)
        except Exception as exc:  # The page shows every failure; nothing else reads it.
            queue.error = str(exc)
        finally:
            for item in queue.items:
                if item.state in (WAITING, *self.working):
                    item.state, item.detail = STOPPED, item.detail or "Not finished."
            queue.run = None
            queue.finished_at = time.time()
            self._checkpoint(queue)
            chain_next = queue.options.get("chain_next")
            if (callable(chain_next) and not queue.stopping and queue.error is None
                    and all(item.state not in (WAITING, FAILED, STOPPED, BLOCKED, *self.working)
                            for item in queue.items)):
                try:
                    chain_next(queue)
                except Exception as exc:
                    queue.error = f"chain_next failed: {exc}"
                    self._checkpoint(queue)

    def wait(self, queue: RunQueue, run: Run) -> Run:
        """Wait for ``run`` to end; a stop interrupts each job it starts meanwhile, as soon as it has one."""
        queue.run = run
        while not run.done.wait(0.5):
            if run.job_id and queue.current is not None and queue.current.job_id != run.job_id:
                queue.current.job_id = run.job_id
                self._checkpoint(queue)
            if queue.stopping:
                self.stopping(run)
                if run.job_id and self.interrupt is not None and self.interruptible(run):
                    self._interrupt(queue, run, queue.session)
        if run.job_id and queue.current is not None and queue.current.job_id != run.job_id:
            queue.current.job_id = run.job_id
            self._checkpoint(queue)
        return run
