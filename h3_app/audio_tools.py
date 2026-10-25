"""Qwen VoiceDesign and Stable Audio 3 Medium in one resident worker with its own environment.

The official qwen-tts package pins Transformers 4.57.3. Never install it in
ComfyUI's environment. Weights download once (hf_xet) and the worker loads them
offline. Each model loads on first use and stays in VRAM beside Drama 3; other
engines' GPU jobs, Unload, and a Drama 3 start that lacks room stop the worker.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from h3_app import audio_tools_worker as layout
from h3_app.workers import env as worker_env
from h3_app.workers.resident import RequestFailed, ResidentWorker as _ResidentWorker, WorkerSlot
from h3_requirements import (
    AUDIO_TOOLS_READY_MARKER as READY_MARKER,
    audio_tools_fingerprint as fingerprint,
    audio_tools_install_commands as install_commands,
)

LABEL = "Qwen VoiceDesign / Stable Audio"
# Job families that share the GPU with the resident worker; any other GPU job stops it.
COEXISTING_FAMILIES = {"drama3", "drama3_voice", "drama3_sound", "drama3_preload"}
# Free VRAM (MiB) wanted before a model's first load: weights plus generation headroom.
LOAD_HEADROOM_MIB = {"voice_design": 6 * 1024, "stable_audio": 8 * 1024}
WORKER = Path(__file__).with_name("audio_tools_worker.py")
__all__ = ["RequestFailed", "ResidentWorker", "loaded_models", "preload", "release_for_job", "run", "stop"]


def log_dir(runtime) -> Path:
    # A Modal snapshot keeps the worker's log open, so it must not live on the volume then.
    path = Path(os.getenv("H3_AUDIO_TOOLS_LOG_DIR") or runtime.output_dir / "audio_tools")
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(runtime) -> Path:
    return (runtime.comfy_dir / "models").resolve()


def prepare(runtime, log: Path) -> Path:
    venv = Path(os.getenv("H3_AUDIO_TOOLS_VENV") or runtime.script_dir / "h3" / "audio-tools-venv")
    return worker_env.prepare_venv(
        venv, log, name="VoiceDesign/Stable Audio", label=LABEL, fingerprint=fingerprint(),
        ready_marker=READY_MARKER, install_commands=install_commands, timeout=1800)


def ensure_weights(model: str, runtime, python: Path, log: Path, *, hf_token: str | None = None) -> None:
    """Download once, online; the resident worker itself never touches the Hub."""
    models = models_dir(runtime)
    if layout.weights_ready(model, models):
        return
    env = {"HF_XET_HIGH_PERFORMANCE": "1", **({"HF_TOKEN": hf_token} if hf_token else {})}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env[key] = "0"
    worker_env.run_process([str(python), str(WORKER), "download", model, str(models)], log,
                           timeout=runtime.generation_timeout, env=env, label=f"{LABEL} download")
    from h3_app.drama3.serving.server import persist_caches

    persist_caches()  # Commit the downloaded weights when running on Modal's volume.


class ResidentWorker(_ResidentWorker):
    """The offline VoiceDesign/Stable Audio worker."""

    def __init__(self, python: Path, models: Path, log: Path):
        super().__init__(python, WORKER, models, log, label=LABEL, env={
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": str(layout.voice_design_cache(models))})


_SLOT = WorkerSlot()


def stop() -> bool:
    """Exit the resident worker; process exit returns all of its VRAM. True if one was running."""
    return _SLOT.stop()


def loaded_models() -> set[str]:
    with _SLOT.lock:
        worker = _SLOT.running()
        return set(worker.loaded) if worker is not None else set()


def release_for_job(family: str) -> None:
    """Other engines reclaim the GPU; Drama 3 jobs keep the worker resident."""
    if family not in COEXISTING_FAMILIES:
        stop()


def _make_room(model: str, free_gpu, *, evict: bool = True) -> bool:
    """Keep Drama 3 and ComfyUI loaded when the GPU has room; free them only when it does not.

    Without ``evict``, SGLang-Omni stays loaded and False reports that the GPU still lacks room.
    """
    from h3_app.drama3.serving.server import gpu_memory, release_all

    needed = LOAD_HEADROOM_MIB[model]
    memory = gpu_memory()
    if memory is None or memory[0] >= needed:
        return True
    free_gpu()
    memory = gpu_memory()
    if memory is not None and memory[0] < needed:
        if not evict:
            return False
        release_all()
    return True


def _worker_for(model: str, runtime, *, free_gpu, hf_token: str | None, evict: bool = True):
    """The running worker with ``model``'s weights on disk and room to load it; hold ``_SLOT.lock``.

    None when ``evict`` is off and the GPU has no room without sleeping SGLang-Omni.
    """
    logs = log_dir(runtime)
    worker = _SLOT.running()
    if worker is None:
        setup_log = logs / f"setup_{time.strftime('%Y%m%d-%H%M%S')}.log"
        python = prepare(runtime, setup_log)
        ensure_weights(model, runtime, python, setup_log, hf_token=hf_token)
        _SLOT.stop()
        if not _make_room(model, free_gpu, evict=evict):
            return None
        _SLOT.worker = ResidentWorker(python, models_dir(runtime),
                                      logs / f"worker_{time.strftime('%Y%m%d-%H%M%S')}.log")
    elif model not in worker.loaded:
        ensure_weights(model, runtime, worker.python, worker.log_path, hf_token=hf_token)
        if not _make_room(model, free_gpu, evict=evict):
            return None
    return _SLOT.worker


def run(model: str, payload: dict, runtime, *, free_gpu, hf_token: str | None = None) -> dict:
    """Send one request to the resident worker, starting it and fetching weights as needed."""
    from h3_app.jobs import JOBS
    from h3_app.drama3.serving.server import persist_caches

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SLOT.lock:
        worker = _worker_for(model, runtime, free_gpu=free_gpu, hf_token=hf_token)
        reply = _SLOT.request(worker, {"op": model, **payload}, runtime.generation_timeout)
    persist_caches()  # Commit the clip when running on Modal's volume.
    return reply


def preload(model: str, runtime, *, free_gpu, hf_token: str | None = None) -> bool:
    """Load ``model`` before its first request, never sleeping SGLang-Omni to make room.

    True once it is loaded (or already was); False when the GPU has no room for it,
    leaving the first request to make room as usual.
    """
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SLOT.lock:
        if model in loaded_models():
            return True
        worker = _worker_for(model, runtime, free_gpu=free_gpu, hf_token=hf_token, evict=False)
        if worker is None:
            return False
        _SLOT.request(worker, {"op": "load", "model": model}, runtime.generation_timeout)
    return True
