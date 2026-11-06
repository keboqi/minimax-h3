"""GPU-button busy state while a job runs and after it ends."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from h3_ui import job_admission


def job(family="qwen_image21", finished_at=None, recovered=False):
    return SimpleNamespace(family=family, finished_at=finished_at, recovered=recovered)


class FamilyBusyTests(unittest.TestCase):
    def busy(self, *jobs):
        registry = SimpleNamespace(list_owned=lambda owner: jobs)
        with patch.object(job_admission, "JOBS", registry):
            return job_admission.family_busy("owner", "qwen_image21")

    def test_live_unfinished_job_of_the_family_keeps_interrupt_enabled(self):
        self.assertTrue(self.busy(job()))

    def test_finished_and_other_family_jobs_release_interrupt(self):
        self.assertFalse(self.busy(job(finished_at=1.0), job(family="flux3")))

    def test_restored_unreconciled_job_does_not_pin_interrupt(self):
        # A job restored after a restart or redeploy stays unfinished until reconciled.
        self.assertFalse(self.busy(job(recovered=True), job(finished_at=1.0)))


class BusyUpdatesTests(unittest.TestCase):
    def updates(self, label, **options):
        with gr.Blocks():
            return job_admission.busy_updates(gr.Button(label, **options))

    def test_analyze_reads_analyzing_and_greys_out_until_idle(self):
        busy, idle = self.updates("Analyze source", elem_classes=["h3-primary-action"])
        self.assertEqual(busy["value"], "Analyzing…")
        self.assertEqual(busy["elem_classes"], ["h3-primary-action", job_admission.BUSY_CLASS])
        self.assertEqual(idle["value"], "Analyze source")
        self.assertEqual(idle["elem_classes"], ["h3-primary-action"])

    def test_busy_state_leaves_interactive_to_readiness(self):
        busy, idle = self.updates("Generate video")
        self.assertEqual(busy["value"], "Generating…")
        self.assertNotIn("interactive", busy)
        self.assertNotIn("interactive", idle)

    def test_regenerate_is_not_read_as_generate(self):
        self.assertEqual(self.updates("Regenerate line")[0]["value"], "Regenerating…")

    def test_unlisted_label_only_greys_out(self):
        busy, idle = self.updates("Reconnect")
        self.assertNotIn("value", busy)
        self.assertEqual(busy["elem_classes"], [job_admission.BUSY_CLASS])
        self.assertEqual(idle["elem_classes"], [])

    def test_hidden_buttons_and_event_triggers_get_no_busy_state(self):
        self.assertIsNone(self.updates("Generate", visible=False))
        self.assertIsNone(job_admission.busy_updates(None))

    def test_dynamic_image_action_label_is_owned_by_mode_not_job_completion(self):
        with gr.Blocks():
            button = gr.Button("Generate image")
            button.h3_dynamic_label = True
            busy, idle = job_admission.busy_updates(button)
        self.assertNotIn("value", busy)
        self.assertNotIn("value", idle)


class ResultMetadataTests(unittest.TestCase):
    def test_metadata_uses_current_video_image_audio_positions(self):
        from h3_app.jobs import CURRENT_JOB
        from h3_ui.job_bindings import owned_generation

        current = SimpleNamespace(owner="owner", family="image", has_gpu=True, check=lambda: None)
        update = ("video.mp4", gr.skip(), gr.skip(), gr.skip(), gr.update(visible=True),
                  gr.update(value="image.png"), gr.update(value="audio.wav"), "Completed")

        def generate(prompt):
            yield update

        run = owned_generation(generate, "image", ("prompt",), metadata_output=True)
        token = CURRENT_JOB.set(current)
        try:
            with (patch.object(job_admission, "require_owner", return_value="owner"),
                  patch("h3_ui.job_bindings.render_snapshot", return_value="metadata") as snapshot):
                self.assertEqual(list(run("fox", None)), [(*update, "metadata")])
                snapshot.assert_called_once_with(["video.mp4", "image.png", "audio.wav"])
        finally:
            CURRENT_JOB.reset(token)


if __name__ == "__main__":
    unittest.main()
