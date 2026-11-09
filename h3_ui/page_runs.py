"""Process-local page runs shared by standalone workspaces.

GPU admission and history remain in JOBS. These short-lived records support page
polling; they do not survive server restarts or coordinate separate processes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import functools
from threading import RLock, Thread
import time
from typing import Any, Callable
import uuid
from fastapi.responses import JSONResponse
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled

# A finished run stays readable this long, for a page that polls or reopens afterwards.
RUN_RETENTION_SECONDS = 600


@dataclass
class Run:
    """Long work on a storyboard: GPU generation, or joining the takes."""

    owner: str
    storyboard_id: str
    kind: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str | None = None  # The generation job, once accepted.
    project: str | None = None  # The Remake project it renders, once it has one.
    status: str = "Waiting for the GPU…"
    # Rises each time the storyboard changed, so the page knows to read it again.
    revision: int = 0
    finished_at: float | None = None
    error: str | None = None
    result_url: str | None = None
    message: str | None = None

    def public(self, scene_id) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status, "revision": self.revision,
                "active": self.finished_at is None, "error": self.error, "scene_id": scene_id,
                "project": self.project, "result_url": self.result_url, "message": self.message}


class Runs:
    def __init__(self, busy="This storyboard is already doing that.", *, lock=None, blocked=None):
        self.lock = lock if lock is not None else RLock()
        self.blocked = blocked
        self.busy = busy
        self.items: dict[str, Run] = {}

    def start(self, run: Run, work: Callable[[Run], None]) -> Run:
        with self.lock:
            if self.blocked is not None and (reason := self.blocked(run)):
                raise H3Error(reason)
            cutoff = time.time() - RUN_RETENTION_SECONDS
            for key in [key for key, item in self.items.items()
                        if item.finished_at is not None and item.finished_at < cutoff]:
                del self.items[key]
            if self.active(run.owner, run.storyboard_id, run.kind):
                raise H3Error(self.busy)
            self.items[run.id] = run

        def execute():
            try:
                work(run)
            except JobCancelled:
                run.status = "Interrupted."
            except Exception as exc:  # The page shows every failure; nothing else reads it.
                run.error = str(exc)
                run.status = "Stopped."
            finally:
                run.revision += 1
                run.finished_at = time.time()

        Thread(target=execute, name=f"page-{run.kind}", daemon=True).start()
        return run

    def active(self, owner, storyboard_id, kind=None) -> Run | None:
        return next((run for run in list(self.items.values())
                     if run.owner == owner and run.storyboard_id == storyboard_id and run.finished_at is None
                     and (kind is None or run.kind == kind)), None)

    def latest(self, owner, storyboard_id) -> Run | None:
        with self.lock:
            active = self.active(owner, storyboard_id)
            if active is not None:
                return active
            return next((run for run in reversed(self.items.values())
                         if run.owner == owner and run.storyboard_id == storyboard_id), None)


def guarded(endpoint):
    """Every refusal reaches the page as ``{"error": message}``."""

    @functools.wraps(endpoint)
    def run(*args, **kwargs):
        try:
            return endpoint(*args, **kwargs)
        except (H3Error, ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return run

