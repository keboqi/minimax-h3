"""Cancellable child processes with bounded diagnostics and output cleanup."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence


def run_process(
    command: Sequence[str],
    *,
    timeout: float,
    check_cancelled: Callable[[], None],
    partial_outputs: Sequence[Path] = (),
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Reap every child and retain only the last 64 KiB of each output stream."""
    check_cancelled()
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    buffers = [bytearray(), bytearray()]

    def drain(stream, buffer):
        try:
            for block in iter(lambda: stream.read(4096), b""):
                buffer.extend(block)
                del buffer[:-65536]
        finally:
            stream.close()

    readers = [
        threading.Thread(target=drain, args=(stream, buffer), daemon=True)
        for stream, buffer in zip((process.stdout, process.stderr), buffers)
    ]
    for reader in readers:
        reader.start()
    succeeded = False
    try:
        while process.poll() is None:
            check_cancelled()
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(list(command), timeout)
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        check_cancelled()
        succeeded = process.returncode == 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join()
        if not succeeded:
            for path in partial_outputs:
                path.unlink(missing_ok=True)
    stdout, stderr = (bytes(value) for value in buffers)
    if text:
        stdout, stderr = (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    return subprocess.CompletedProcess(
        list(command), process.returncode, stdout, stderr
    )


def run_media_process(cmd, *, timeout=600, partial_outputs=(), **kwargs):
    from .jobs import check_cancelled

    return run_process(
        cmd,
        timeout=timeout,
        check_cancelled=check_cancelled,
        partial_outputs=partial_outputs,
        **kwargs,
    )
