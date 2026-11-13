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

from .page_runs import Run

WAITING, FAILED, STOPPED = "waiting", "failed", "stopped"


@dataclass
class QueueItem:
    key: str  # What the item works on: a storyboard id, a project name.
    name: str
    state: str = WAITING
    detail: str = ""

    def public(self) -> dict[str, Any]:
        return {"id": self.key, "name": self.name, "state": self.state, "detail": self.detail}


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
    effect once the current item's run ends.
    """

    working: tuple[str, ...] = ()  # States, besides waiting, of an item not finished yet.
    thread_name = "page-queue"

    def __init__(self, messages: QueueMessages, interrupt: Callable[[Any, Run], str] | None = None):
        self.messages = messages
        self.interrupt = interrupt
        self.lock = RLock()
        self.queues: dict[str, RunQueue] = {}

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
        finished = sum(item.state not in (WAITING, *self.working) for item in queue.items)
        live = queue.finished_at is None and queue.run is not None and queue.run.finished_at is None
        return {
            "id": queue.id, "active": queue.finished_at is None, "stopping": queue.stopping, "error": queue.error,
            "current": queue.current.key if queue.current and queue.finished_at is None else None,
            "status": queue.run.status if live else "", "stage": getattr(queue.run, "stage", None) if live else None,
            "finished": finished, "items": [item.public() for item in queue.items],
        }

    def start(self, owner, items: list[QueueItem], session, **options) -> RunQueue:
        with self.lock:
            if self.active(owner) is not None:
                raise H3Error(self.messages.busy)
            queue = self.queues[owner] = RunQueue(owner, items, session, options)
        Thread(target=self._execute, args=(queue,), name=self.thread_name, daemon=True).start()
        return queue

    def stop(self, owner, session) -> str:
        queue = self.active(owner)
        if queue is None:
            return self.messages.idle
        queue.stopping = True
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
                if queue.stopping:
                    item.state, item.detail = STOPPED, self.messages.not_started
                    continue
                queue.current = item
                self.process(queue, item)
        except Exception as exc:  # The page shows every failure; nothing else reads it.
            queue.error = str(exc)
        finally:
            for item in queue.items:
                if item.state in (WAITING, *self.working):
                    item.state, item.detail = STOPPED, item.detail or "Not finished."
            queue.run = None
            queue.finished_at = time.time()

    def wait(self, queue: RunQueue, run: Run) -> Run:
        """Wait for ``run`` to end; a stop interrupts each job it starts meanwhile, as soon as it has one."""
        queue.run = run
        while not run.done.wait(0.5):
            if queue.stopping:
                self.stopping(run)
                if run.job_id and self.interrupt is not None and self.interruptible(run):
                    self._interrupt(queue, run, queue.session)
        return run
