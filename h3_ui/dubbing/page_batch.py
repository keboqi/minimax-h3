"""Run saved dubbing projects in order using the page's Auto run."""
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from ..page_queue import FAILED, STOPPED, QueueItem, QueueMessages, RunQueues


class Batches(RunQueues):
    working = ("producing",)
    thread_name = "dubbing-batch"

    def __init__(self, start_auto, interrupt):
        super().__init__(QueueMessages(
            busy="A batch is already running. Stop it before starting another.",
            idle="No batch is running.", interrupting="Stopping the current dub and the batch.",
            pending="Stopping before the next step.", after_step="Stopping before the next step.",
            not_started="Not started: the batch was stopped."), interrupt)
        self.start_auto = start_auto

    def start(self, owner, names, session, api_key=""):
        if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
            raise H3Error("Select at least one saved project to generate.")
        return super().start(owner, [QueueItem(name, name) for name in dict.fromkeys(names)], session, api_key=api_key)

    def stopping(self, run):
        run.stop_requested = True

    def process(self, queue, item):
        item.state = "producing"
        try:
            run = self.wait(queue, self.start_auto(queue.owner, item.key, queue.session, queue.options["api_key"]))
            if queue.stopping or run.stop_requested or run.status == "Interrupted.":
                queue.stopping = True
                item.state, item.detail = STOPPED, "Interrupted; completed work is kept."
            elif run.error:
                item.state, item.detail = FAILED, run.error
            else:
                item.state, item.detail = "done", run.status
        except JobCancelled:
            queue.stopping = True
            item.state, item.detail = STOPPED, "Interrupted; completed work is kept."
        except Exception as exc:  # A broken project must not discard the remaining queue.
            item.state, item.detail = FAILED, str(exc)
