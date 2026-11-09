"""Concurrency contracts shared by the Soundstage page and its adapters."""
from threading import Barrier, Event, RLock, Thread
import unittest
from unittest.mock import patch

from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_ui.page_runs import Run, Runs
from h3_ui.drama3.page_common import PageRun, run_job


class SoundstageRunTests(unittest.TestCase):
    def test_cross_registry_admission_is_atomic(self):
        lock, ready, finish = RLock(), Barrier(3), Event()
        speech = Runs(lock=lock, blocked=lambda run: "Cast active" if cast.active("owner", "project") else None)
        cast = Runs(lock=lock, blocked=lambda run: "Auto active" if speech.active("owner", "project") else None)
        accepted, refused = [], []

        def start(registry, kind):
            ready.wait(timeout=5)
            try:
                accepted.append(registry.start(Run("owner", "project", kind), lambda run: finish.wait(timeout=5)))
            except H3Error as exc:
                refused.append(str(exc))

        workers = [Thread(target=start, args=(speech, "auto")), Thread(target=start, args=(cast, "cast"))]
        for worker in workers:
            worker.start()
        try:
            ready.wait(timeout=5)
            for worker in workers:
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(refused), 1)
        finally:
            finish.set()

    def test_cancelled_run_does_not_accept_a_gpu_job(self):
        run = PageRun("owner", "project", "cast", stop_requested=True)
        with patch("h3_ui.drama3.page_common.JOBS.accept") as accept:
            with self.assertRaises(JobCancelled):
                list(run_job(run, "owner", None, "drama3_voice", [], lambda: None))
        accept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
