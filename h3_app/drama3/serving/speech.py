"""Speak single lines in reference voices with Drama 3: the interface other features use.

Each line is one SGLang-Omni request with its speaker's reference clip and
transcript. A take that fails its check is generated again and the best take
is kept. Dub & Translate voices its lines through this module; Drama 3's own
script generation shares the same primitives in :mod:`.generate`.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Iterator

import numpy as np

from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import CURRENT_JOB
from h3_app.workflows.drama3 import MAX_FRAMES

from . import generate
from .server import ServerSettings, SpeechRequestError, persist_caches, shared_server

__all__ = ["LineSpeaker", "SpokenLine", "persist_caches", "run_blocking"]


def run_blocking(label: str, work: Callable[[Callable[[str], None]], Any]) -> Iterator[GenerationUpdate]:
    """Run ``work(report)`` in a thread, yielding its latest progress; returns its result.

    Use as ``result = yield from run_blocking(...)``. The thread sees this job's Interrupt.
    """
    return generate._with_heartbeat(label, work)


@dataclass(frozen=True)
class SpokenLine:
    key: Any
    wav: bytes
    samples: np.ndarray
    rate: int
    problem: str  # empty when the take passed its check, else why the kept take failed
    no_audio: bool = False  # Drama 3 voiced nothing for the line (SpeechRequestError.no_audio)


class LineSpeaker:
    """The shared SGLang-Omni server, used for one batch of lines."""

    def __init__(self, runtime):
        self.settings = ServerSettings.from_environment(runtime)
        self.server = shared_server(self.settings)

    def reference(self, path: str | Path) -> Path:
        """A copy of the clip the server may read; its stem identifies the clip's content."""
        return generate._staged_reference(str(path), self.settings.media_dir)

    @staticmethod
    def budget(text: str) -> int:
        """The frame limit for one line of ``text``."""
        return generate.frame_budget(text, MAX_FRAMES)

    def ready(self, free_gpu: Callable[[], None], hf_token: str | None = None) -> Iterator[GenerationUpdate]:
        """Prepare, download and start (or wake) the server."""
        yield from generate._ready_server(self.settings, self.server, generate.LABEL, free_gpu, hf_token)

    def speak(self, lines: list[tuple[Any, str, Path, str]], *, sampling: dict, parallel: int,
              thread_prefix: str = "drama3-lines", keep_going: bool = False) -> Iterator[list[SpokenLine]]:
        """Voice ``(key, text, reference, transcript)`` lines, ``parallel`` at a time.

        Yields the lines finished in each heartbeat (possibly none). An Interrupt
        cancels the pending requests. With ``keep_going`` a line the server rejects
        (an HTTP 4xx, no audio frames for it, or audio it cannot decode) comes back without samples and with
        its error as the problem, so the other lines still finish; a server that is
        down or failing every request still stops the batch.
        """

        def synthesize(line, stopped: threading.Event) -> SpokenLine:
            key, text, reference, transcript = line
            try:
                attempt = generate.best_take(
                    self.server, text=text, reference=reference, transcript=transcript,
                    max_new_tokens=self.budget(text), sampling=sampling,
                    attempts=generate.AUTOMATIC_RETRIES + 1, stopped=stopped)
            except (SpeechRequestError, ValueError) as exc:
                if not keep_going or (isinstance(exc, SpeechRequestError) and not exc.rejected_line):
                    raise
                return SpokenLine(key, b"", np.zeros(0, dtype=np.float32), 0, f"generation failed: {exc}",
                                  no_audio=isinstance(exc, SpeechRequestError) and exc.no_audio)
            problem = "" if attempt.check.ok else attempt.check.reason
            return SpokenLine(key, attempt.wav, attempt.samples, attempt.rate, problem)

        workers = max(1, min(int(parallel), len(lines)))
        with generate._request_pool(workers, thread_prefix) as (pool, stopped):
            pending = {pool.submit(synthesize, line, stopped): line for line in lines}
            while pending:
                finished, _ = wait(pending, timeout=generate.HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
                job = CURRENT_JOB.get()
                if job is not None and job.cancelled.is_set():
                    for future in pending:
                        future.cancel()
                    job.check()
                for future in finished:
                    pending.pop(future)
                yield [future.result() for future in finished]
