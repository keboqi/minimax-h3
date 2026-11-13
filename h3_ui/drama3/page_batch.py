"""Batch produce: run Auto-produce on every selected Soundstage project, one after another.

Projects are prepared anywhere (the CPU Workshop included) and produced on the GPU endpoint.
Each project runs as the same Auto run its page starts by hand, so a project opened during the
queue shows its progress as usual. The queue lives in this process only: a restart ends it, and
each project keeps what was made.
"""

from __future__ import annotations

from typing import Any, Callable

from h3_app.errors import H3Error

from ..page_queue import FAILED, STOPPED, QueueItem, QueueMessages, RunQueue, RunQueues
from ..page_runs import Run

PRODUCING, DONE = "producing", "done"
INTERRUPTED = ("Auto interrupted.", "Interrupted.")

MESSAGES = QueueMessages(
    busy="A batch is already running. Stop it before starting another.",
    idle="No batch is running.",
    interrupting="Stopping the current project; the rest won't start.",
    pending="Stopping once the current step answers; the rest won't start.",
    after_step="Stopping once the current step answers; the rest won't start.",
    not_started="Not started: the batch was stopped.",
)


class Batches(RunQueues):
    """One batch per owner. ``start_auto(owner, name, session, writer)`` starts a project's Auto run
    (or refuses it), with ``writer`` the writer service unprepared scripts are rewritten with."""

    working = (PRODUCING,)
    thread_name = "soundstage-batch"

    def __init__(self, start_auto: Callable[[str, str, Any, dict], Run],
                 interrupt: Callable[[Any, Run], str] | None):
        super().__init__(MESSAGES, interrupt)
        self.start_auto = start_auto

    def start(self, owner, names: list[str], session, writer: dict | None = None) -> RunQueue:
        names = list(dict.fromkeys(str(value) for value in names or [] if value))
        if not names:
            raise H3Error("Select at least one project to produce.")
        return super().start(owner, [QueueItem(name, name) for name in names], session, writer=dict(writer or {}))

    def stopping(self, run: Run) -> None:
        # Auto checks this between its steps, so a stop during the rewrite ends it before the GPU.
        run.stop_requested = True

    def process(self, queue: RunQueue, item: QueueItem):
        item.state = PRODUCING
        try:
            run = self.wait(queue, self.start_auto(queue.owner, item.key, queue.session, queue.options["writer"]))
        except (H3Error, ValueError, KeyError) as exc:
            item.state, item.detail = FAILED, str(exc) or "This project can't be produced."
            return
        if run.error is None and run.status not in INTERRUPTED:
            item.state, item.detail = DONE, " ".join(filter(None, ["Spoken.", run.message]))
        elif queue.stopping or run.status in INTERRUPTED:
            # Interrupting the project from its own page stops the batch too, as auto render does.
            queue.stopping = True
            item.state, item.detail = STOPPED, run.error or "Interrupted; what was made so far is kept."
        else:
            item.state, item.detail = FAILED, run.error
