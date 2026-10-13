"""Bounded binary video pipes with cancellation and job-owned process cleanup."""

from __future__ import annotations

import os
import logging
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

from .errors import H3Error
from .jobs import check_cancelled


@dataclass(frozen=True)
class PipeProgress:
    frames: int
    diagnostics: tuple[str, ...] = ()


def stop_process(process):
    """Terminate the process group even when its original parent has exited."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def stream_pipeline(
    commands: list[list[str]], *, input_size: int, output_size: int, timeout: float,
    worker_env: dict | None = None, worker_cwd=None,
    source_bytes: bytes | None = None, capture: bytearray | None = None,
) -> Iterator[PipeProgress]:
    """Decode -> worker -> encode, or feed/capture a tiny runtime probe.

    Each relay holds at most one frame. stderr is drained continuously with a
    fixed memory limit. Binary payloads never enter the text diagnostic buffers.
    """
    processes = []
    threads = []
    errors = deque(maxlen=4)
    logs = deque(maxlen=96)
    worker_logs = deque(maxlen=64)
    headers = {}  # Keep runtime evidence after progress lines rotate out.
    log_lock = threading.Lock()
    counts = [0, 0]
    deadline = time.monotonic() + timeout
    failed = False
    worker_index = 0 if source_bytes is not None else 1
    check_cancelled()

    def relay(reader, writer, frame_size, slot, data=None, output=None):
        try:
            while True:
                if data is not None:
                    block, data = data[:frame_size], data[frame_size:]
                else:
                    block = bytearray()
                    while len(block) < frame_size:
                        piece = reader.read(frame_size - len(block))
                        if not piece:
                            break
                        block.extend(piece)
                if not block:
                    break
                if len(block) != frame_size:
                    raise H3Error("Worker pipeline returned a truncated video frame.")
                if writer is not None:
                    remaining = memoryview(block)
                    while remaining:
                        written = writer.write(remaining)
                        if not written:
                            raise H3Error("Worker pipe stopped accepting video frames.")
                        remaining = remaining[written:]
                    writer.flush()
                if output is not None:
                    expected = len(source_bytes) // input_size * output_size
                    if len(output) + len(block) > expected:
                        raise H3Error("Runtime probe returned excess video frames.")
                    output.extend(block)
                counts[slot] += 1
        except (OSError, ValueError, H3Error) as exc:
            errors.append(str(exc))
        finally:
            if reader is not None:
                reader.close()
            if writer is not None:
                try:
                    writer.close()
                except OSError:
                    pass

    def log_line(line, label):
        decoded = line[-2048:].decode("utf-8", "replace")
        with log_lock:
            for marker in ("H3VFX ", "feature ready; streaming", "optical flow:"):
                if marker in decoded:
                    headers[label + marker] = label + decoded
            logs.append(label + decoded)
            if label == f"[{worker_index}] ":
                worker_logs.append(label + decoded)

    def diagnostics():
        with log_lock:
            return tuple(headers.values()) + tuple(logs)

    def drain(stream, label):
        # Read bounded chunks, rather than readline on an untrusted long line.
        pending = b""
        try:
            while True:
                block = stream.read(1024)
                if not block:
                    break
                pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    log_line(line, label)
                pending = pending[-4096:]
            if pending:
                log_line(pending, label)
        finally:
            stream.close()

    def start_thread(fn, *args):
        thread = threading.Thread(target=fn, args=args, daemon=True)
        threads.append(thread)
        thread.start()

    try:
        for index, command in enumerate(commands):
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
                env=worker_env if index == worker_index else None,
                cwd=worker_cwd if index == worker_index else None,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            processes.append(process)
            start_thread(drain, process.stderr, f"[{index}] ")
        if source_bytes is None:
            decoder, worker, encoder = processes
            decoder.stdin.close()
            encoder.stdout.close()
            start_thread(relay, decoder.stdout, worker.stdin, input_size, 0)
            start_thread(relay, worker.stdout, encoder.stdin, output_size, 1)
        else:
            worker = processes[0]
            start_thread(relay, None, worker.stdin, input_size, 0, source_bytes)
            start_thread(relay, worker.stdout, None, output_size, 1, None, capture)
        while any(p.poll() is None for p in processes) or any(t.is_alive() for t in threads):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise H3Error("NVIDIA processing timed out; worker processes were stopped.")
            if errors or any(p.poll() not in (None, 0) for p in processes):
                failed = True
                # A closed input pipe can precede the worker's traceback. Let
                # it finish reporting the cause before stopping the pipeline.
                try:
                    worker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                break
            yield PipeProgress(counts[1], diagnostics())
            time.sleep(0.1)
        failed = failed or bool(errors) or any(p.poll() not in (None, 0) for p in processes)
        if not failed and (not counts[0] or counts[0] != counts[1]):
            raise H3Error(f"Video worker lost frames ({counts[0]} input, {counts[1]} output).")
        if not failed:
            yield PipeProgress(counts[1], diagnostics())
    finally:
        for process in reversed(processes):
            stop_process(process)
        for thread in threads:
            thread.join(timeout=5)
        for process in processes:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
    if failed:
        # Construct the error after stderr drain threads have reached EOF.
        # Decoder/encoder teardown can be noisy. Keep the neural worker's final
        # exception independently and place it last so truncation preserves it.
        exits = [f"[{index}] process exited with code {process.returncode}"
                 for index, process in enumerate(processes)]
        with log_lock:
            other_logs = [line for line in logs if not line.startswith(f"[{worker_index}] ")]
            detail = "\n".join((*exits, *errors, *other_logs, *worker_logs))[-6000:]
        logging.getLogger(__name__).error("NVIDIA processing failed:\n%s", detail)
        raise H3Error("Video worker failed: " + detail)
