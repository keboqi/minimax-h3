"""Session-owned Comfy jobs and process-wide GPU preparation coordination."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, Lock, Event
from typing import Any, Callable
import uuid


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

    def check(self):
        if self.cancelled.is_set():
            raise JobCancelled("Generation interrupted.")


CURRENT_JOB: ContextVar[Job | None] = ContextVar("h3_job", default=None)


class JobCoordinator:
    def __init__(self):
        self.gpu = Lock()
        self.lock = RLock()
        self.active: dict[str, Job] = {}

    @contextmanager
    def run(self, owner: str, family: str):
        job = Job(owner, family)
        with self.lock:
            self.active[job.id] = job

        try:
            with self.gpu:
                job.check()
                yield job
        finally:
            with self.lock:
                self.active.pop(job.id, None)

    def cancel(self, owner: str, family: str, get: Callable, post: Callable) -> str:
        with self.lock:
            jobs = [
                job
                for job in self.active.values()
                if job.owner == owner
                and job.family in ({family, "h3-input"} if family == "h3" else {family})
            ]
            for job in jobs:
                job.cancelled.set()
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
