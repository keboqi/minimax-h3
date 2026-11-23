"""Offline tests for SQLite-persisted RunQueues recovery and reconciliation invariants."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from h3_app.errors import H3Error
from h3_app.workspace_store import WorkspaceStore
from h3_ui.page_queue import BLOCKED, RECOVERED, STOPPED, WAITING, QueueItem, QueueMessages, RunQueue, RunQueues

MESSAGES = QueueMessages(
    busy="Busy.",
    idle="Idle.",
    interrupting="Stopping.",
    pending="Pending stop.",
    after_step="Stopping after step.",
    not_started="Not started.",
)


class _DemoQueues(RunQueues):
    working = ("running",)

    def __init__(self, store, ran, job_lookup=None):
        self.ran = ran
        super().__init__(MESSAGES, store=store, kind="demo_batch", job_lookup=job_lookup)

    def process(self, queue: RunQueue, item: QueueItem) -> None:
        item.state = "running"
        self.ran.append(item.key)
        item.state = "done"
        item.detail = "Finished."


class PageQueueRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WorkspaceStore(Path(self.tmp.name))

    def test_restart_recovers_completed_job_and_resumes_only_waiting_items(self):
        # Simulate an active queue persisted right before a crash:
        # item-1 done, item-2 in-flight with a completed backend job, item-3 waiting.
        self.store.save_run_queue("demo_batch", "owner-1", {
            "id": "q1",
            "owner": "owner-1",
            "options": {},
            "stopping": False,
            "finished_at": None,
            "error": None,
            "resumable": False,
            "items": [
                {"key": "item-1", "name": "One", "state": "done", "detail": "Finished."},
                {"key": "item-2", "name": "Two", "state": "running", "detail": "", "job_id": "job-done"},
                {"key": "item-3", "name": "Three", "state": "waiting", "detail": ""},
            ],
        })
        jobs = {"job-done": {"id": "job-done", "state": "completed", "finished_at": 100.0, "error": None, "ledger": []}}
        ran = []
        queues = _DemoQueues(self.store, ran, job_lookup=jobs.get)

        payload = queues.payload("owner-1")
        self.assertFalse(payload["active"])
        self.assertTrue(payload["resumable"])
        self.assertEqual(payload["blocked"], 0)
        states = {row["id"]: row["state"] for row in payload["items"]}
        self.assertEqual(states, {"item-1": "done", "item-2": RECOVERED, "item-3": WAITING})

        # Explicit resume runs ONLY item-3 (item-1 and recovered item-2 are never duplicated).
        queue = queues.resume("owner-1", session=None)
        deadline = time.time() + 5.0
        while queue.finished_at is None and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(ran, ["item-3"])
        self.assertEqual(queues.payload("owner-1")["items"][2]["state"], "done")

    def test_ambiguous_submission_is_blocked_and_never_auto_replayed(self):
        self.store.save_run_queue("demo_batch", "owner-2", {
            "id": "q2",
            "owner": "owner-2",
            "options": {},
            "stopping": False,
            "finished_at": None,
            "error": None,
            "resumable": False,
            "items": [
                {"key": "item-1", "name": "One", "state": "running", "detail": "", "job_id": "job-unknown"},
                {"key": "item-2", "name": "Two", "state": "waiting", "detail": ""},
            ],
        })
        jobs = {
            "job-unknown": {
                "id": "job-unknown",
                "state": "recovering",
                "finished_at": None,
                "ledger": [{"state": "submission_unknown"}],
            }
        }
        ran = []
        queues = _DemoQueues(self.store, ran, job_lookup=jobs.get)
        payload = queues.payload("owner-2")
        self.assertEqual(payload["items"][0]["state"], BLOCKED)
        self.assertEqual(payload["blocked"], 1)
        self.assertFalse(payload["resumable"])
        with self.assertRaisesRegex(H3Error, "Reconcile ambiguous"):
            queues.resume("owner-2", session=None)
        self.assertEqual(ran, [])

    def test_stopped_queue_stays_stopped_after_restart(self):
        self.store.save_run_queue("demo_batch", "owner-3", {
            "id": "q3",
            "owner": "owner-3",
            "options": {},
            "stopping": True,
            "finished_at": None,
            "error": None,
            "resumable": False,
            "items": [
                {"key": "item-1", "name": "One", "state": "done", "detail": "Finished."},
                {"key": "item-2", "name": "Two", "state": "waiting", "detail": ""},
            ],
        })
        ran = []
        queues = _DemoQueues(self.store, ran)
        payload = queues.payload("owner-3")
        self.assertEqual(payload["items"][1]["state"], STOPPED)
        self.assertTrue(payload["stopping"])
        with self.assertRaisesRegex(H3Error, "No interrupted batch"):
            queues.resume("owner-3", session=None)
        self.assertEqual(ran, [])

    def test_chain_next_runs_on_clean_completion_and_asr_take_check_catches_mismatch(self):
        import numpy as np
        from h3_app.drama3.serving.audio import check_take, check_take_transcript

        chained = []
        ran = []
        queues = _DemoQueues(self.store, ran)
        q = queues.start("owner-4", [QueueItem("a", "A"), QueueItem("b", "B")], session=None,
                         chain_next=lambda done_q: chained.append(done_q.id))
        deadline = time.time() + 5.0
        while q.finished_at is None and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(ran, ["a", "b"])
        self.assertEqual(chained, [q.id])
        self.assertIsNotNone(q.items[0].duration_s)

        # ASR transcript check catches garbled speech even when RMS activity looks valid.
        ok, sim, _ = check_take_transcript("有人吗？我回来了。", "有人吗我回来了")
        self.assertTrue(ok)
        self.assertGreater(sim, 0.9)

        samples = np.sin(2 * np.pi * 220.0 * np.linspace(0, 1.5, 24000, dtype=np.float32)) * 0.2
        res_bad = check_take(samples, 16000, 1.5, expected_text="雨下了一整夜",
                             asr_transcribe=lambda _s, _r: "完全无关的乱码内容")
        self.assertFalse(res_bad.ok)
        self.assertIn("ASR transcript mismatch", res_bad.reason)

    def test_mode_aware_storyboard_planner_and_soundstage_bridge(self):
        from h3_app.storyboard.bridge import storyboard_spec_from_soundstage
        from h3_app.storyboard.model import FRAMES, REFERENCES, TEXT
        from h3_app.storyboard.writer import plan_scenes_with_modes

        planned = plan_scenes_with_modes(
            [
                {"prompt": "Storm clouds gather over the empty lighthouse.", "seconds": 5},
                {"prompt": "Mia opens the heavy brass latch and looks outside.", "seconds": 7},
                {"prompt": "She steps onto the rocks.", "seconds": 6,
                 "entry_state": "doorway", "exit_state": "wet stone ledge"},
            ],
            cast_names=["Mia"],
            cast_descriptions={"Mia": "young woman in yellow oilskin"},
        )
        self.assertEqual([s["mode"] for s in planned], [TEXT, REFERENCES, FRAMES])
        self.assertIn("subject_definitions:", planned[1]["prompt"])
        self.assertIn("<Picture 1> is Mia", planned[1]["prompt"])

        spec = storyboard_spec_from_soundstage(
            "Storm · ch01",
            {"title": "Storm", "voice_0_save_name": "narrator", "voice_1_save_name": "Mia",
             "voice_1_description": "warm female voice"},
            visual_plan={"plan": {
                "characters": {"Mia": "young woman in yellow coat"},
                "shots": [
                    {"beat": "Establish", "prompt": "Dark ocean waves at dusk.", "transition": "cut"},
                    {"beat": "Reveal", "prompt": "Mia holds up the lantern in the rain.", "transition": "dissolve"},
                ],
            }},
        )
        self.assertEqual([c["name"] for c in spec["cast"]], ["Mia"])
        self.assertEqual([s["mode"] for s in spec["scenes"]], [TEXT, REFERENCES])
        self.assertEqual([s["link"] for s in spec["scenes"]], ["cut", "flow"])


if __name__ == "__main__":
    unittest.main()
