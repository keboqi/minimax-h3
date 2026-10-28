"""Worker progress messages update the UI without being mistaken for final results."""

import io
import itertools
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app.dubbing import tools_worker
from h3_app.workers.resident import ResidentWorker, RequestFailed


class WorkerProgressTests(unittest.TestCase):
    def worker(self, replies):
        worker = object.__new__(ResidentWorker)
        worker._ids = itertools.count(1)
        worker._replies = queue.Queue()
        for reply in replies:
            worker._replies.put(reply)
        worker.process = SimpleNamespace(stdin=io.BytesIO())
        worker.label, worker.log_path = "Tools", Path("worker.log")
        worker.loaded = set()
        return worker

    def test_progress_updates_do_not_complete_the_request(self):
        worker = self.worker([
            {"id": 0, "event": "progress", "status": "old request"},
            {"id": 1, "event": "progress", "status": "Detecting speech with OmniVAD"},
            {"id": 1, "event": "progress", "status": "Transcribing speech chunks 1-16 of 30"},
            {"id": 1, "ok": True, "segments": ["done"], "loaded": ["asr+aligner"]},
        ])
        updates = []
        result = worker.request({"op": "transcribe"}, 2, report=updates.append)
        self.assertEqual(updates, ["Detecting speech with OmniVAD", "Transcribing speech chunks 1-16 of 30"])
        self.assertEqual(result["segments"], ["done"])
        self.assertEqual(worker.loaded, {"asr+aligner"})
        self.assertTrue(json.loads(worker.process.stdin.getvalue())["progress"])

    def test_failure_after_progress_still_raises(self):
        worker = self.worker([{"id": 1, "event": "progress", "status": "Loading model"},
                              {"id": 1, "ok": False, "error": "out of memory"}])
        with self.assertRaisesRegex(RequestFailed, "out of memory"):
            worker.request({"op": "transcribe"}, 2, report=lambda message: None)

    def test_existing_workers_do_not_need_progress_support(self):
        worker = self.worker([{"id": 1, "ok": True, "loaded": []}])
        worker.request({"op": "translate"}, 2)
        self.assertNotIn("progress", json.loads(worker.process.stdin.getvalue()))

    def test_worker_emits_progress_and_then_one_final_reply(self):
        output = io.StringIO()

        def work(worker, request):
            worker.progress("Separating vocals from the background")
            return {"vocals": "vocals.wav", "backing": "backing.wav"}

        with TemporaryDirectory() as directory, patch.object(tools_worker.Worker, "handle", work):
            tools_worker.serve(directory, requests=[json.dumps({"id": 7, "op": "separate", "progress": True})],
                               replies=output)
        progress, final = map(json.loads, output.getvalue().splitlines())
        self.assertEqual(progress, {"id": 7, "event": "progress", "status": "Separating vocals from the background"})
        self.assertTrue(final["ok"])
        self.assertEqual(final["id"], 7)
        self.assertEqual(final["vocals"], "vocals.wav")
