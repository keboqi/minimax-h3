"""Auto render: generate every selected storyboard's remaining scenes, export each to Media, in turn.

Each storyboard runs as the same generate and export runs the page starts by hand, so a
storyboard opened during the queue shows its progress as usual. The queue lives in this
process only: a restart ends it, and the storyboards keep what was made.
"""

from __future__ import annotations

from typing import Any, Callable

from h3_app import storyboard as sb
from h3_app.errors import H3Error

from ..page_queue import FAILED, STOPPED, WAITING, QueueItem, QueueMessages, RunQueue, RunQueues
from ..page_runs import Run

RENDERING, EXPORTING = "rendering", "exporting"
EXPORTED, INCOMPLETE = "exported", "incomplete"
__all__ = ["AutoRenders", "WAITING", "RENDERING", "EXPORTING", "EXPORTED", "INCOMPLETE", "FAILED", "STOPPED"]

MESSAGES = QueueMessages(
    busy="Auto render is already running. Stop it before starting another.",
    idle="Auto render isn't running.",
    interrupting="Stopping the current storyboard; the rest won't start.",
    pending="Stopping once the current storyboard's run starts.",
    after_step="Stopping after the current step; the rest won't start.",
    not_started="Not started: auto render was stopped.",
)


class AutoRenders(RunQueues):
    """One auto-render queue per owner: each storyboard generates its remaining scenes, then exports."""

    working = (RENDERING, EXPORTING)
    thread_name = "storyboard-auto-render"

    def __init__(self, service: Callable[[], sb.StoryboardService], *,
                 start_generation: Callable[[str, str, Any], Run], start_export: Callable[[str, str], Run],
                 interrupt: Callable[[Any, Run], str] | None):
        super().__init__(MESSAGES, interrupt)
        self.service = service
        self.start_generation = start_generation
        self.start_export = start_export

    def start(self, owner, storyboard_ids: list[str], session) -> RunQueue:
        ids = list(dict.fromkeys(str(value) for value in storyboard_ids or [] if value))
        if not ids:
            raise H3Error("Select at least one storyboard to render.")
        items = [QueueItem(storyboard_id, self.service().load(owner, storyboard_id)["name"]) for storyboard_id in ids]
        return super().start(owner, items, session)

    def interruptible(self, run: Run) -> bool:
        return run.kind == "generate"  # An export finishes; it's quick and leaves nothing half made.

    def process(self, queue: RunQueue, item: QueueItem):
        owner, storyboard_id = queue.owner, item.key
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
                run = self.wait(queue, self.start_generation(owner, storyboard_id, queue.session))
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
            run = self.wait(queue, self.start_export(owner, storyboard_id))
        except H3Error as exc:
            item.state, item.detail = FAILED, str(exc)
            return
        if run.error:
            item.state, item.detail = FAILED, run.error
        else:
            item.state, item.detail = EXPORTED, run.message or "Exported to Media."
