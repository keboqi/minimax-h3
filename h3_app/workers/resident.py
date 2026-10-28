"""A resident worker process answering one JSON request per line, and the slot that owns it."""

from __future__ import annotations

import atexit
import itertools
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled

from .env import popen_flags, reap, worker_env


class RequestFailed(H3Error):
    """The worker reported a failed request and is still healthy."""


class ResidentWorker:
    """A ``<script> serve <models>`` process. Replies echo the request id and the loaded models."""

    def __init__(self, python: Path, script: Path, models: Path, log: Path, *, env: dict[str, str], label: str):
        self.python, self.models, self.log_path, self.label = python, models, log, label
        self.loaded: set[str] = set()
        self._ids = itertools.count(1)
        self._replies: queue.Queue = queue.Queue()
        self._log = log.open("ab")
        self.process = subprocess.Popen([str(python), str(script), "serve", str(models)],
                                        env={**worker_env(), **env},
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        **popen_flags())
        threading.Thread(target=self._read, name=f"{Path(script).stem}-replies", daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                self._replies.put(json.loads(line))
            except ValueError:
                continue
        self._replies.put(None)

    def running(self) -> bool:
        return self.process.poll() is None

    def request(self, payload: dict, timeout: float, *, report=None) -> dict:
        request_id = next(self._ids)
        try:
            self.process.stdin.write((json.dumps({**payload, "id": request_id,
                **({"progress": True} if report is not None else {})}) + "\n").encode("utf-8"))
            self.process.stdin.flush()
        except OSError as exc:
            raise H3Error(f"{self.label} worker is not accepting requests. See {self.log_path}.") from exc
        deadline = time.monotonic() + timeout
        while True:
            check_cancelled()
            if time.monotonic() >= deadline:
                raise H3Error(f"{self.label} timed out. See {self.log_path}.")
            try:
                reply = self._replies.get(timeout=0.25)
            except queue.Empty:
                continue
            if reply is None:
                self.process.wait()
                raise H3Error(f"{self.label} worker exited (code {self.process.returncode}). See {self.log_path}.")
            if reply.get("id") != request_id:
                continue
            if reply.get("event") == "progress":
                if report is not None:
                    report(str(reply.get("status") or ""))
                continue
            self.loaded = set(reply.get("loaded") or ())
            if not reply.get("ok"):
                raise RequestFailed(f"{self.label} failed: {reply.get('error')}. See {self.log_path}.")
            return reply

    def stop(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        reap(self.process)
        self._log.close()


class WorkerSlot:
    """Holds at most one resident worker; stopping it returns all of its VRAM.

    Hold ``lock`` while starting or using ``worker``. The worker is stopped at exit.
    """

    def __init__(self):
        self.worker: ResidentWorker | None = None
        self.lock = threading.RLock()
        atexit.register(self.stop)

    def running(self) -> ResidentWorker | None:
        return self.worker if self.worker is not None and self.worker.running() else None

    def stop(self) -> bool:
        """Exit the worker. True if one was running."""
        with self.lock:
            worker, self.worker = self.worker, None
        if worker is None:
            return False
        worker.stop()
        return True

    def request(self, worker: ResidentWorker, payload: dict, timeout: float) -> dict:
        """Send ``payload`` to ``worker``, reaping it as :meth:`reaping` does."""
        return self.reaping(worker, lambda: worker.request(payload, timeout=timeout))

    def reaping(self, worker: ResidentWorker, action):
        """Run ``action``; cancellation, timeouts and crashes reap ``worker``.

        After those its GPU state is unknown, so it cannot be reused. A failed
        request leaves a healthy worker running.
        """
        try:
            return action()
        except RequestFailed:
            raise
        except BaseException:
            if self.worker is worker:
                self.stop()
            raise
