"""Auto render: generate every selected storyboard's remaining scenes, export each to Media, in turn.

Each storyboard runs as the same generate and export runs the page starts by hand, so a
storyboard opened during the queue shows its progress as usual. The queue lives in this
process only: a restart ends it, and the storyboards keep what was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock, Thread
import time
from typing import Any, Callable
import uuid

from h3_app import storyboard as sb
from h3_app.errors import H3Error

from ..page_runs import Run

WAITING, RENDERING, EXPORTING = "waiting", "rendering", "exporting"
EXPORTED, INCOMPLETE, FAILED, STOPPED = "exported", "incomplete", "failed", "stopped"


@dataclass
class Item:
    storyboard_id: str
    name: str
    state: str = WAITING
    detail: str = ""

    def public(self) -> dict[str, Any]:
        return {"id": self.storyboard_id, "name": self.name, "state": self.state, "detail": self.detail}


@dataclass
class AutoRender:
    owner: str
    items: list[Item]
    session: Any
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    current: Item | None = None
    run: Run | None = None  # The current storyboard's generate or export run.
    stopping: bool = False
    interrupted: set[str] = field(default_factory=set)  # Runs already asked to stop.
    finished_at: float | None = None
    started_at: float = field(default_factory=time.monotonic)
    error: str | None = None

    def public(self) -> dict[str, Any]:
        finished = sum(item.state not in (WAITING, RENDERING, EXPORTING) for item in self.items)
        return {
            "id": self.id, "active": self.finished_at is None, "stopping": self.stopping, "error": self.error,
            "current": self.current.storyboard_id if self.current and self.finished_at is None else None,
            "status": self.run.status if self.run is not None and self.run.finished_at is None else "",
            "finished": finished, "items": [item.public() for item in self.items],
        }


class AutoRenders:
    """One auto-render queue per owner, run in a thread of its own."""

    def __init__(self, service: Callable[[], sb.StoryboardService], *,
                 start_generation: Callable[[str, str, Any], Run], start_export: Callable[[str, str], Run],
                 interrupt: Callable[[Any, Run], str] | None):
        self.service = service
        self.start_generation = start_generation
        self.start_export = start_export
        self.interrupt = interrupt
        self.lock = RLock()
        self.queues: dict[str, AutoRender] = {}

    def latest(self, owner) -> AutoRender | None:
        return self.queues.get(owner)

    def start(self, owner, storyboard_ids: list[str], session) -> AutoRender:
        ids = list(dict.fromkeys(str(value) for value in storyboard_ids or [] if value))
        if not ids:
            raise H3Error("Select at least one storyboard to render.")
        items = [Item(storyboard_id, self.service().load(owner, storyboard_id)["name"]) for storyboard_id in ids]
        with self.lock:
            current = self.queues.get(owner)
            if current is not None and current.finished_at is None:
                raise H3Error("Auto render is already running. Stop it before starting another.")
            queue = self.queues[owner] = AutoRender(owner, items, session)
        Thread(target=self._execute, args=(queue,), name="storyboard-auto-render", daemon=True).start()
        return queue

    def stop(self, owner, session) -> str:
        queue = self.queues.get(owner)
        if queue is None or queue.finished_at is not None:
            return "Auto render isn't running."
        queue.stopping = True
        run = queue.run
        if run is not None and run.finished_at is None and run.kind == "generate" and self.interrupt is not None:
            if run.job_id is None:
                return "Stopping once the current storyboard's run starts."
            queue.interrupted.add(run.id)
            self.interrupt(session, run)
            return "Stopping the current storyboard; the rest won't start."
        return "Stopping after the current step; the rest won't start."

    # The queue -------------------------------------------------------------
    def _execute(self, queue: AutoRender):
        try:
            for item in queue.items:
                if queue.stopping:
                    item.state, item.detail = STOPPED, "Not started: auto render was stopped."
                    continue
                queue.current = item
                self._render(queue, item)
        except Exception as exc:  # The panel shows every failure; nothing else reads it.
            queue.error = str(exc)
        finally:
            for item in queue.items:
                if item.state in (WAITING, RENDERING, EXPORTING):
                    item.state, item.detail = STOPPED, item.detail or "Not finished."
            queue.run = None
            queue.finished_at = time.time()

    def _wait(self, queue: AutoRender, run: Run) -> Run:
        queue.run = run
        # A stop that arrived before the job was accepted interrupts it once it has one.
        while not run.done.wait(0.5):
            if (queue.stopping and run.kind == "generate" and run.job_id and self.interrupt is not None
                    and run.id not in queue.interrupted):
                queue.interrupted.add(run.id)
                self.interrupt(queue.session, run)
        return run

    def _render(self, queue: AutoRender, item: Item):
        owner, storyboard_id = queue.owner, item.storyboard_id
        try:
            board = self.service().load(owner, storyboard_id)
        except (H3Error, ValueError, KeyError) as exc:
            item.state, item.detail = FAILED, str(exc) or "This storyboard is gone."
            return
        item.name = board["name"]
        # Only when a scene can be made: one that needs input would take a GPU turn for nothing.
        if sb.runnable_scenes(board):
            item.state, item.detail = RENDERING, ""
            try:
                run = self._wait(queue, self.start_generation(owner, storyboard_id, queue.session))
            except H3Error as exc:
                item.state, item.detail = FAILED, str(exc)
                return
            if queue.stopping or run.status == "Interrupted.":
                queue.stopping = True
                item.state, item.detail = STOPPED, "Interrupted; the scenes made so far are kept."
                return
            if run.error:
                item.state, item.detail = FAILED, run.error
                return
            board = self.service().load(owner, storyboard_id)
        left = sb.pending_scenes(board)
        if left:
            numbers = ", ".join(str(index + 1) for index in left)
            item.state = INCOMPLETE
            item.detail = (f"Not exported: scene {numbers} still need{'s' if len(left) == 1 else ''} input "
                           "or wait for one that does.")
            return
        item.state, item.detail = EXPORTING, ""
        try:
            run = self._wait(queue, self.start_export(owner, storyboard_id))
        except H3Error as exc:
            item.state, item.detail = FAILED, str(exc)
            return
        if run.error:
            item.state, item.detail = FAILED, run.error
        else:
            item.state, item.detail = EXPORTED, run.message or "Exported to Media."
