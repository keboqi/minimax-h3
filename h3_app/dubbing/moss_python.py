"""Run MOSS with Transformers in a disposable, single-recording process."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from h3_app import audio_tools
from h3_app.drama3.serving import server
from h3_app.workers import env as worker_env
from h3_requirements import MOSS_PYTHON_READY_MARKER, moss_python_fingerprint, moss_python_install_commands

from . import moss, tools

LABEL = "MOSS (Python)"
WORKER = Path(__file__).with_name("moss_python_worker.py")


def _make_room(duration: float, free_gpu) -> None:
    # Weights and activations plus the growing KV cache (~112 KiB per token).
    needed = 6 * 1024 + 3 * duration

    def short() -> bool:
        memory = server.gpu_memory()
        return memory is not None and memory[0] < needed

    for release in (moss.release_all, tools.stop, free_gpu, audio_tools.stop, server.release_all):
        if not short():
            break
        release()


def transcribe(runtime, audio: Path, *, duration: float, free_gpu, hf_token, report) -> str:
    venv = Path(os.getenv("H3_MOSS_PYTHON_VENV") or runtime.script_dir / "h3" / "moss-python-venv")
    logs = tools.log_dir(runtime)
    with TemporaryDirectory(prefix="moss-python-", dir=logs) as temp:
        work = Path(temp).resolve()
        log = logs / f"{work.name}.log"
        python = worker_env.prepare_venv(
            venv, log, name=LABEL, label=LABEL, fingerprint=moss_python_fingerprint(),
            ready_marker=MOSS_PYTHON_READY_MARKER, install_commands=moss_python_install_commands,
            timeout=2400, report=report)
        report("Downloading MOSS-Transcribe-Diarize (first use only)")
        checkpoint = moss.ensure_model(moss.model_dir(moss.model_root(runtime)), token=hf_token)
        _make_room(duration, free_gpu)
        result = work / "transcript.txt"
        # Keep the checkpoint's usual budget for short clips and allow more for long transcripts.
        max_tokens = max(5120, min(60000, int(16 * duration)))
        report(f"Starting {LABEL}")
        worker_env.run_process(
            [str(python), str(WORKER), str(checkpoint.resolve()), str(audio.resolve()), str(result), str(max_tokens)],
            log, timeout=runtime.generation_timeout, env={"PYTHONIOENCODING": "utf-8"}, label=LABEL, report=report)
        return result.read_text(encoding="utf-8")
