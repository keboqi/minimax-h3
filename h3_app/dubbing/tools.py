"""Dub & Translate's isolated tools worker: separation, transcription and Hy-MT translation.

Prepared on first use like the VoiceDesign/Stable Audio environment. The worker
stays resident between requests of one dubbing session; every other GPU job,
and the Drama 3 speech step of a dub, stop it so its VRAM returns at once.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import threading
import time

from h3_app import audio_tools
from h3_app.dubbing import tools_worker as layout
from h3_app.drama3_serving.environment import _setup_lock, compiler_env, uv_command
from h3_app.errors import H3Error
from h3_requirements import (
    DUB_TOOLS_READY_MARKER as READY_MARKER,
    dub_tools_fingerprint as fingerprint,
    dub_tools_install_commands as install_commands,
)

LABEL = "Dub & Translate tools"
# Job families of Dub & Translate's steps; separate families let each Interrupt stop only its step.
FAMILY_ANALYZE, FAMILY_LINES, FAMILY_DUB = "dubbing_analyze", "dubbing_lines", "dubbing"
FAMILIES = frozenset({FAMILY_ANALYZE, FAMILY_LINES, FAMILY_DUB})
WORKER = Path(__file__).with_name("tools_worker.py")
# Free VRAM (MiB) wanted before a request's models load: weights plus working room.
LOAD_HEADROOM_MIB = {"separate": 4 * 1024, "transcribe": 7 * 1024, "translate": 6 * 1024}
# The worker's loaded-model names start with these, per request.
OP_MODELS = {"separate": "separator:", "transcribe": "asr", "translate": "translator"}
_SETUP_LOCK = threading.Lock()


def log_dir(runtime) -> Path:
    path = runtime.output_dir / "dub_tools"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(runtime) -> Path:
    return (runtime.comfy_dir / "models").resolve()


def prepare(runtime, log: Path) -> Path:
    venv = Path(os.getenv("H3_DUB_TOOLS_VENV") or runtime.script_dir / "h3" / "dub-tools-venv")
    with _SETUP_LOCK, _setup_lock(venv.parent / f".{venv.name}.lock"):
        marker = venv / READY_MARKER
        python = audio_tools.python_path(venv)
        if python.is_file() and marker.is_file() and marker.read_text().strip() == fingerprint():
            return python
        uv = uv_command()
        if not uv:
            raise H3Error("Install uv (`pip install uv`) to prepare the isolated Dub & Translate runtime.")
        marker.unlink(missing_ok=True)
        commands = install_commands(venv, uv)
        if python.is_file():
            commands = commands[1:]
        for command in commands:
            # audio-separator's diffq ships only as source and compiles a C extension.
            audio_tools.run_process(command, log, timeout=2400, env=compiler_env(), label=f"{LABEL} setup")
        marker.write_text(fingerprint(), encoding="utf-8")
    return python


_WORKER: audio_tools.ResidentWorker | None = None
_WORKER_LOCK = threading.RLock()


def stop() -> bool:
    """Exit the resident worker; process exit returns all of its VRAM. True if one was running."""
    global _WORKER
    with _WORKER_LOCK:
        worker, _WORKER = _WORKER, None
    if worker is None:
        return False
    worker.stop()
    return True


def release_for_job(family: str) -> None:
    """Every other engine reclaims the GPU; Dub & Translate's own steps keep the worker resident."""
    if family not in FAMILIES:
        stop()


atexit.register(stop)


def _make_room(op: str, worker, free_gpu) -> None:
    """Room for ``op``'s models: the worker's idle models go first, then ComfyUI, VoiceDesign and Drama 3.

    Nothing is freed when the op's model is already loaded or the GPU has room.
    """
    from h3_app.drama3_serving.server import gpu_memory, release_all

    needed, prefix = LOAD_HEADROOM_MIB.get(op, 0), OP_MODELS.get(op, "")
    if not needed or any(name.startswith(prefix) for name in worker.loaded):
        return

    def short() -> bool:
        memory = gpu_memory()
        return memory is not None and memory[0] < needed

    if not short():
        return
    for name in sorted(worker.loaded):
        worker.request({"op": "unload", "model": name}, 300)
    if not short():
        return
    free_gpu()
    audio_tools.stop()
    if short():
        release_all()


def run(op: str, payload: dict, runtime, *, free_gpu, hf_token: str | None = None,
        report=lambda _message: None) -> dict:
    """Send one request to the resident worker, preparing the environment and GPU room first."""
    from h3_app.drama3_serving.server import persist_caches
    from h3_app.jobs import JOBS

    global _WORKER
    JOBS.add_gpu_claim_hook(release_for_job)
    with _WORKER_LOCK:
        logs = log_dir(runtime)
        if _WORKER is None or not _WORKER.running():
            report(f"Preparing the {LABEL} environment (first use installs it)")
            python = prepare(runtime, logs / f"setup_{time.strftime('%Y%m%d-%H%M%S')}.log")
            models = models_dir(runtime)
            env = {"HF_HOME": str(layout.models_root(models) / "hf"), "HF_XET_HIGH_PERFORMANCE": "1",
                   **({"HF_TOKEN": hf_token} if hf_token else {})}
            _WORKER = audio_tools.ResidentWorker(
                python, models, logs / f"worker_{time.strftime('%Y%m%d-%H%M%S')}.log",
                script=WORKER, env=env, label=LABEL)
        worker = _WORKER
        try:
            _make_room(op, worker, free_gpu)
            reply = worker.request({"op": op, **payload}, runtime.generation_timeout)
        except audio_tools.RequestFailed:
            raise
        except BaseException:
            # Cancellation, timeouts and crashes leave GPU state unknown: reap the process.
            if _WORKER is worker:
                stop()
            raise
    persist_caches()  # Commit downloaded weights when running on Modal's volume.
    return reply
