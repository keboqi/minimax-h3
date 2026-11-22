"""MuseTalk 1.5 and LatentSync 1.6 resident worker and isolated venv manager.

Both engines share one diffusers 0.32.2 / CUDA 13 environment (separate from
ComfyUI's >=0.36 stack). Models load lazily on first use and stay in VRAM
between windows until another GPU job runs or another model needs room.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from h3_app import audio_tools
from h3_app.dubbing import lip_sync_worker as layout
from h3_app.errors import H3Error
from h3_app.workers import env as worker_env
from h3_app.workers.resident import ResidentWorker, WorkerSlot
from h3_gpu import CPU_ONLY_MESSAGE, is_cpu_only
from h3_requirements import (
    LIP_SYNC_READY_MARKER as READY_MARKER,
    lip_sync_fingerprint as fingerprint,
    lip_sync_install_commands as install_commands,
    lip_sync_src_dirs,
)

LABEL = "Dubbing lip sync"
FAMILY_LIP_SYNC = "dubbing_lip_sync"
FAMILIES = frozenset({FAMILY_LIP_SYNC})
WORKER = Path(__file__).with_name("lip_sync_worker.py")
LOAD_HEADROOM_MIB = {
    "render_musetalk": 6 * 1024,
    "render_latentsync": 20 * 1024,
}
OP_MODELS = {
    "render_musetalk": "musetalk_v15",
    "render_latentsync": "latentsync_16",
}


def log_dir(runtime) -> Path:
    path = runtime.output_dir / "lip_sync_worker"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(runtime) -> Path:
    return (runtime.comfy_dir / "models").resolve()


def venv_dir(runtime) -> Path:
    return Path(os.getenv("H3_LIP_SYNC_VENV") or runtime.script_dir / "h3" / "lip-sync-venv")


def prepare(runtime, log: Path, *, report=None) -> Path:
    """Create or reuse the isolated MuseTalk / LatentSync venv."""
    return worker_env.prepare_venv(
        venv_dir(runtime),
        log,
        name="Lip sync",
        label=LABEL,
        fingerprint=fingerprint(),
        ready_marker=READY_MARKER,
        install_commands=install_commands,
        timeout=2400,
        env=worker_env.compiler_env(),
        report=report,
    )


_SLOT = WorkerSlot()


def stop() -> bool:
    """Exit the resident worker; process exit returns all of its VRAM."""
    return _SLOT.stop()


def release_for_job(family: str) -> None:
    """Reclaim VRAM whenever another engine's GPU job claims the device."""
    if family not in FAMILIES:
        stop()


def _make_room(op: str, worker, free_gpu) -> None:
    """Free other loaded lip-sync models, then other GPU engines if VRAM is short."""
    from h3_app.drama3.serving.server import gpu_memory, release_all
    from h3_app.dubbing import moss, tools as dubbing_tools

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
    dubbing_tools.stop()
    audio_tools.stop()
    if short():
        release_all()
    if short():
        moss.release_all()


def run(
    op: str,
    payload: dict,
    runtime,
    *,
    free_gpu,
    hf_token: str | None = None,
    report=lambda _message: None,
) -> dict:
    """Send one lip-sync render request to the resident worker."""
    if is_cpu_only():
        raise H3Error(CPU_ONLY_MESSAGE)
    from h3_app.drama3.serving.server import persist_caches
    from h3_app.jobs import JOBS

    JOBS.add_gpu_claim_hook(release_for_job)
    with _SLOT.lock:
        logs = log_dir(runtime)
        if _SLOT.running() is None:
            venv = venv_dir(runtime)
            python = prepare(runtime, logs / f"setup_{time.strftime('%Y%m%d-%H%M%S')}.log", report=report)
            models = models_dir(runtime)
            src_root, musetalk_dir, latentsync_dir = lip_sync_src_dirs(venv)
            env = {
                "HF_HOME": str(layout.models_root(models) / "hf"),
                "HF_XET_HIGH_PERFORMANCE": "1",
                "PYTHONPATH": os.pathsep.join((str(musetalk_dir), str(latentsync_dir))),
                "H3_LIP_SYNC_SRC_DIR": str(src_root),
                **({"HF_TOKEN": hf_token} if hf_token else {}),
            }
            _SLOT.worker = ResidentWorker(
                python,
                WORKER,
                models,
                logs / f"worker_{time.strftime('%Y%m%d-%H%M%S')}.log",
                env=env,
                label=LABEL,
            )
        worker = _SLOT.worker

        def make_room_and_request():
            report("Making GPU room for lip sync")
            _make_room(op, worker, free_gpu)
            step = {
                "render_musetalk": "Rendering lip sync with MuseTalk 1.5",
                "render_latentsync": "Rendering lip sync with LatentSync 1.6",
            }.get(op, op)
            report(step)
            print(f"[{LABEL}] {step}; worker log: {worker.log_path}", flush=True)

            def worker_progress(message):
                print(f"[{LABEL}] {message}", flush=True)
                report(message)

            return worker.request({"op": op, **payload}, 1800, report=worker_progress)

        reply = _SLOT.reaping(worker, make_room_and_request)
    persist_caches()
    return reply
