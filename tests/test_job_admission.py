"""Generate-button busy state after a job ends."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from h3_ui import job_admission


def job(family="qwen_image21", finished_at=None, recovered=False):
    return SimpleNamespace(family=family, finished_at=finished_at, recovered=recovered)


class FamilyBusyTests(unittest.TestCase):
    def busy(self, *jobs):
        registry = SimpleNamespace(list_owned=lambda owner: jobs)
        with patch.object(job_admission, "JOBS", registry):
            return job_admission.family_busy("owner", "qwen_image21")

    def test_live_unfinished_job_of_the_family_keeps_generate_busy(self):
        self.assertTrue(self.busy(job()))

    def test_finished_and_other_family_jobs_release_generate(self):
        self.assertFalse(self.busy(job(finished_at=1.0), job(family="flux3")))

    def test_restored_unreconciled_job_does_not_pin_generating(self):
        # A job restored after a restart or redeploy stays unfinished until reconciled.
        self.assertFalse(self.busy(job(recovered=True), job(finished_at=1.0)))


if __name__ == "__main__":
    unittest.main()
