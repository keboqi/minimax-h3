"""Qwen VoiceDesign and Stable Audio 3 Medium in one resident worker with its own environment.

The official qwen-tts package pins Transformers 4.57.3. Never install it in
ComfyUI's environment. Weights download once (hf_xet) and the worker loads them
offline. Each model loads on first use and stays in VRAM beside Drama 3; other
engines' GPU jobs, Unload, and a Drama 3 start that lacks room stop the worker.
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

from h3_app import audio_tools_worker as layout
from h3_app.drama3_serving.environment import _setup_lock, uv_command
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled
from h3_requirements import (
    AUDIO_TOOLS_READY_MARKER as READY_MARKER,
    audio_tools_fingerprint as fingerprint,
    audio_tools_install_commands as install_commands,
)

LABEL = "Qwen VoiceDesign / Stable Audio"
# Job families that share the GPU with the resident worker; any other GPU job stops it.
COEXISTING_FAMILIES = {"drama3", "drama3_voice", "drama3_sound"}
# Free VRAM (MiB) wanted before a model's first load: weights plus generation headroom.
LOAD_HEADROOM_MIB = {"voice_design": 6 * 1024, "stable_audio": 8 * 1024}
WORKER = Path(__file__).with_name("audio_tools_worker.py")
_SETUP_LOCK = threading.Lock()


class RequestFailed(H3Error):
    """The worker reported a failed request and is still healthy."""


def python_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT", "UV_CONSTRAINT"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    return env


def _popen_flags() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _reap(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_process(command: list[str], log: Path, *, timeout: float, env: dict[str, str] | None = None,
                label: str = LABEL) -> None:
    """Drain output to disk; cancellation/timeout always reaps the process.

    ``env`` adds variables (such as a Hugging Face token) on top of :func:`worker_env`;
    ``label`` names the step in failure messages.
    """
    check_cancelled()
    with log.open("ab") as stream:
        process = subprocess.Popen(command, env={**worker_env(), **(env or {})}, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, **_popen_flags())
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise H3Error(f"{label} timed out. See {log}.")
                time.sleep(0.25)
            check_cancelled()
            if process.returncode:
                raise H3Error(f"{label} failed (exit {process.returncode}). See {log}.")
        finally:
            _reap(process)


def log_dir(runtime) -> Path:
    path = runtime.output_dir / "audio_tools"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(runtime) -> Path:
    return (runtime.comfy_dir / "models").resolve()


def prepare(runtime, log: Path) -> Path:
    venv = Path(os.getenv("H3_AUDIO_TOOLS_VENV") or runtime.script_dir / "h3" / "audio-tools-venv")
    with _SETUP_LOCK, _setup_lock(venv.parent / f".{venv.name}.lock"):
        marker = venv / READY_MARKER
        if python_path(venv).is_file() and marker.is_file() and marker.read_text().strip() == fingerprint():
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error("Install uv (`pip install uv`) to prepare the isolated VoiceDesign/Stable Audio runtime.")
        marker.unlink(missing_ok=True)
        commands = install_commands(venv, uv)
        if python_path(venv).is_file():
            commands = commands[1:]
        for command in commands:
            run_process(command, log, timeout=1800, label=f"{LABEL} setup")
        marker.write_text(fingerprint(), encoding="utf-8")
    return python_path(venv)


def ensure_weights(model: str, runtime, python: Path, log: Path, *, hf_token: str | None = None) -> None:
    """Download once, online; the resident worker itself never touches the Hub."""
    models = models_dir(runtime)
    if layout.weights_ready(model, models):
        return
    env = {"HF_XET_HIGH_PERFORMANCE": "1", **({"HF_TOKEN": hf_token} if hf_token else {})}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env[key] = "0"
    run_process([str(python), str(WORKER), "download", model, str(models)], log,
                timeout=runtime.generation_timeout, env=env, label=f"{LABEL} download")
    from h3_app.drama3_serving.server import persist_caches

    persist_caches()  # Commit the downloaded weights when running on Modal's volume.


class ResidentWorker:
    def __init__(self, python: Path, models: Path, log: Path):
        self.python, self.models, self.log_path = python, models, log
        self.loaded: set[str] = set()
        self._ids = itertools.count(1)
        self._replies: queue.Queue = queue.Queue()
        self._log = log.open("ab")
        env = worker_env()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_CACHE=str(layout.voice_design_cache(models)))
        self.process = subprocess.Popen([str(python), str(WORKER), "serve", str(models)], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        **_popen_flags())
        threading.Thread(target=self._read, name="audio-tools-replies", daemon=True).start()

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
        _reap(self.process)
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


def loaded_models() -> set[str]:
    with _WORKER_LOCK:
        return set(_WORKER.loaded) if _WORKER is not None and _WORKER.running() else set()


def release_for_job(family: str) -> None:
    """Other engines reclaim the GPU; Drama 3 jobs keep the worker resident."""
    if family not in COEXISTING_FAMILIES:
        stop()


atexit.register(stop)


def _make_room(model: str, free_gpu) -> None:
    """Keep Drama 3 and ComfyUI loaded when the GPU has room; free them only when it does not."""
    from h3_app.drama3_serving.server import gpu_memory, release_all

    needed = LOAD_HEADROOM_MIB[model]
    memory = gpu_memory()
    if memory is None or memory[0] >= needed:
        return
    free_gpu()
    memory = gpu_memory()
    if memory is not None and memory[0] < needed:
        release_all()


def run(model: str, payload: dict, runtime, *, free_gpu, hf_token: str | None = None) -> dict:
    """Send one request to the resident worker, starting it and fetching weights as needed."""
    global _WORKER
    from h3_app.jobs import JOBS
    from h3_app.drama3_serving.server import persist_caches

    JOBS.add_gpu_claim_hook(release_for_job)
    logs = log_dir(runtime)
    with _WORKER_LOCK:
        if _WORKER is None or not _WORKER.running():
            setup_log = logs / f"setup_{time.strftime('%Y%m%d-%H%M%S')}.log"
            python = prepare(runtime, setup_log)
            ensure_weights(model, runtime, python, setup_log, hf_token=hf_token)
            if _WORKER is not None:
                _WORKER.stop()
            _make_room(model, free_gpu)
            _WORKER = ResidentWorker(python, models_dir(runtime),
                                     logs / f"worker_{time.strftime('%Y%m%d-%H%M%S')}.log")
        elif model not in _WORKER.loaded:
            ensure_weights(model, runtime, _WORKER.python, _WORKER.log_path, hf_token=hf_token)
            _make_room(model, free_gpu)
        worker = _WORKER
        try:
            reply = worker.request({"op": model, **payload}, timeout=runtime.generation_timeout)
        except RequestFailed:
            raise
        except BaseException:
            # Cancellation, timeouts and crashes leave GPU state unknown: reap the process.
            if _WORKER is worker:
                stop()
            raise
    persist_caches()  # Commit the clip when running on Modal's volume.
    return reply
