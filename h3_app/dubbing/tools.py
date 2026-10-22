"""Dub & Translate's isolated tools worker: separation, transcription and Hy-MT translation.

Prepared on first use like the VoiceDesign/Stable Audio environment. The worker
stays resident between requests of one dubbing session; every other GPU job,
and the Drama 3 speech step of a dub, stop it so its VRAM returns at once.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from h3_app import audio_tools
from h3_app.dubbing import tools_worker as layout
from h3_app.drama3_serving.environment import _setup_lock, uv_command
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled
from h3_requirements import (
    DUB_TOOLS_READY_MARKER as READY_MARKER,
    dub_tools_fingerprint as fingerprint,
    dub_tools_install_commands as install_commands,
)

LABEL = "Dub & Translate tools"
FAMILY = "dubbing"
WORKER = Path(__file__).with_name("tools_worker.py")
# Free VRAM (MiB) wanted before a request's models load: weights plus working room.
LOAD_HEADROOM_MIB = {"separate": 4 * 1024, "transcribe": 7 * 1024, "translate": 6 * 1024, "unload": 0}
_SETUP_LOCK = threading.Lock()


class RequestFailed(H3Error):
    """The worker reported a failed request and is still healthy."""


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
            audio_tools.run_process(command, log, timeout=2400, label=f"{LABEL} setup")
        marker.write_text(fingerprint(), encoding="utf-8")
    return python


class ResidentWorker:
    def __init__(self, python: Path, models: Path, log: Path, *, hf_token: str | None = None):
        self.python, self.log_path = python, log
        self.loaded: set[str] = set()
        self._ids = itertools.count(1)
        self._replies: queue.Queue = queue.Queue()
        self._log = log.open("ab")
        env = audio_tools.worker_env()
        env.update(HF_HOME=str(layout.models_root(models) / "hf"), HF_XET_HIGH_PERFORMANCE="1")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        self.process = subprocess.Popen([str(python), str(WORKER), "serve", str(models)], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        **audio_tools._popen_flags())
        threading.Thread(target=self._read, name="dub-tools-replies", daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                self._replies.put(json.loads(line))
            except ValueError:
                continue
        self._replies.put(None)

    def running(self) -> bool:
        return self.process.poll() is None

    def request(self, payload: dict, timeout: float) -> dict:
        request_id = next(self._ids)
        try:
            self.process.stdin.write((json.dumps({**payload, "id": request_id}) + "\n").encode("utf-8"))
            self.process.stdin.flush()
        except OSError as exc:
            raise H3Error(f"{LABEL} worker is not accepting requests. See {self.log_path}.") from exc
        deadline = time.monotonic() + timeout
        while True:
            check_cancelled()
            if time.monotonic() >= deadline:
                raise H3Error(f"{LABEL} timed out. See {self.log_path}.")
            try:
                reply = self._replies.get(timeout=0.25)
            except queue.Empty:
                continue
            if reply is None:
                self.process.wait()
                raise H3Error(f"{LABEL} worker exited (code {self.process.returncode}). See {self.log_path}.")
            if reply.get("id") != request_id:
                continue
            self.loaded = set(reply.get("loaded") or ())
            if not reply.get("ok"):
                raise RequestFailed(f"{LABEL} failed: {reply.get('error')}. See {self.log_path}.")
            return reply

    def stop(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        audio_tools._reap(self.process)
        self._log.close()


_WORKER: ResidentWorker | None = None
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
    """Every other engine reclaims the GPU; dubbing jobs keep the worker resident."""
    if family != FAMILY:
        stop()


atexit.register(stop)


def _make_room(op: str, free_gpu) -> None:
    """Free ComfyUI, the VoiceDesign worker and then a loaded Drama 3 server, only as needed."""
    from h3_app.drama3_serving.server import gpu_memory, release_all

    needed = LOAD_HEADROOM_MIB.get(op, 0)
    memory = gpu_memory()
    if not needed or memory is None or memory[0] >= needed:
        return
    free_gpu()
    audio_tools.stop()
    memory = gpu_memory()
    if memory is not None and memory[0] < needed:
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
            _WORKER = ResidentWorker(python, models_dir(runtime),
                                     logs / f"worker_{time.strftime('%Y%m%d-%H%M%S')}.log", hf_token=hf_token)
        _make_room(op, free_gpu)
        worker = _WORKER
        try:
            reply = worker.request({"op": op, **payload}, runtime.generation_timeout)
        except RequestFailed:
            raise
        except BaseException:
            # Cancellation, timeouts and crashes leave GPU state unknown: reap the process.
            if _WORKER is worker:
                stop()
            raise
    persist_caches()  # Commit downloaded weights when running on Modal's volume.
    return reply
