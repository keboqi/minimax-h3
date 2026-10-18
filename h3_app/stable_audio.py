"""Stable Audio 3 Medium in a disposable worker with its own dependency environment.

Model downloads persist under `<comfy_dir>/models/stable_audio3`; the GPU worker
exits per single/batch sound generation request so VRAM is returned cleanly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from h3_app.drama3_serving.environment import _setup_lock, uv_command
from h3_app.errors import H3Error
from h3_app.voice_design import python_path, run_process
from h3_requirements import (
    STABLE_AUDIO_READY_MARKER as READY_MARKER,
    stable_audio_fingerprint as fingerprint,
    stable_audio_install_commands as install_commands,
)

MODEL_ID = "stabilityai/stable-audio-3-medium"
DEFAULT_DURATION_SECONDS = 5
DEFAULT_LOOP_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = 47
_LOCK = threading.Lock()


def prepare(runtime, log: Path) -> Path:
    venv = Path(os.getenv("H3_STABLE_AUDIO_VENV") or runtime.script_dir / "h3" / "stable-audio-venv")
    with _LOCK, _setup_lock(venv.parent / f".{venv.name}.lock"):
        marker = venv / READY_MARKER
        if python_path(venv).is_file() and marker.is_file() and marker.read_text().strip() == fingerprint():
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error("Install uv (`pip install uv`) to prepare the isolated Stable Audio 3 Medium runtime.")
        marker.unlink(missing_ok=True)
        commands = install_commands(venv, uv)
        if python_path(venv).is_file():
            commands = commands[1:]
        for command in commands:
            run_process(command, log, timeout=1800, label="Stable Audio 3 Medium setup")
        marker.write_text(fingerprint(), encoding="utf-8")
    return python_path(venv)


def validate_sound_inputs(description: str, duration: int | float = DEFAULT_DURATION_SECONDS) -> tuple[str, int]:
    cleaned = str(description or "").strip()
    if not cleaned or len(cleaned) > 2000:
        raise H3Error("Describe the sound effect or music in 1–2000 characters.")
    try:
        seconds = int(round(float(duration or DEFAULT_DURATION_SECONDS)))
    except (TypeError, ValueError) as exc:
        raise H3Error("Sound duration must be a number of seconds between 1 and 47.") from exc
    if seconds < 1 or seconds > MAX_DURATION_SECONDS:
        raise H3Error(f"Sound duration must be between 1 and {MAX_DURATION_SECONDS} seconds.")
    return cleaned, seconds


def cached_sound_path(output_dir: Path, description: str, duration: int) -> Path:
    digest = hashlib.sha256(f"{description.strip().lower()}|{int(duration)}".encode("utf-8")).hexdigest()[:16]
    return output_dir / f"StableAudio3_cache_{digest}.wav"


def generate_sound_references(
    items: list[tuple[str, int | float] | dict],
    runtime,
    *,
    free_gpu,
    use_cache: bool = False,
    hf_token: str | None = None,
) -> list[Path]:
    """Generate one or more sound clips in a single Stable Audio 3 Medium worker run."""
    if not items:
        return []
    validated = [
        validate_sound_inputs(
            item.get("description") or item.get("prompt") or "",
            item.get("duration", 5),
        )
        if isinstance(item, dict)
        else validate_sound_inputs(item[0], item[1])
        for item in items
    ]
    output_dir = runtime.output_dir / "drama3_sounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path | None] = [None] * len(validated)
    worker_items: list[dict] = []
    worker_indices: list[int] = []
    created_paths: list[Path] = []

    for idx, (desc, dur) in enumerate(validated):
        if use_cache:
            cached = cached_sound_path(output_dir, desc, dur)
            if cached.is_file() and cached.stat().st_size > 44:
                results[idx] = cached
                continue
            target = cached
        else:
            token = uuid.uuid4().hex
            target = output_dir / f"StableAudio3_{token}.wav"
        created_paths.append(target)
        worker_indices.append(idx)
        worker_items.append(
            {
                "prompt": desc,
                "duration": dur,
                "seed": -1,
                "steps": 8,
                "cfg_scale": 1.0,
                "output": str(target.resolve()),
            }
        )

    if not worker_items:
        return [path for path in results if path is not None]

    batch_token = uuid.uuid4().hex
    log = output_dir / f"StableAudio3_{batch_token}.log"
    python = prepare(runtime, log)
    from h3_app.drama3_serving.server import persist_caches, release_all

    release_all()
    free_gpu()
    worker = Path(__file__).with_name("stable_audio_worker.py")
    with tempfile.TemporaryDirectory(prefix="h3-stable-audio-") as directory:
        request_file = Path(directory) / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "cache_dir": str((runtime.comfy_dir / "models" / "stable_audio3").resolve()),
                    "items": worker_items,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            run_process(
                [str(python), str(worker), str(request_file)],
                log,
                timeout=runtime.generation_timeout,
                # The model and its T5Gemma encoder are gated; the worker downloads them on first use.
                env={"HF_TOKEN": hf_token} if hf_token else None,
                label="Stable Audio 3 Medium",
            )
            for idx, item in zip(worker_indices, worker_items):
                out_path = Path(item["output"])
                if not out_path.is_file() or out_path.stat().st_size <= 44:
                    raise H3Error(f"Stable Audio 3 Medium returned no audio. See {log}.")
                results[idx] = out_path
        except BaseException:
            for path in created_paths:
                path.unlink(missing_ok=True)
            raise
    persist_caches()
    return [path for path in results if path is not None]


def generate_sound_reference(
    description: str,
    duration: int | float,
    runtime,
    *,
    free_gpu,
    use_cache: bool = False,
    hf_token: str | None = None,
) -> Path:
    """Generate a single sound clip with Stable Audio 3 Medium."""
    paths = generate_sound_references(
        [(description, duration)],
        runtime,
        free_gpu=free_gpu,
        use_cache=use_cache,
        hf_token=hf_token,
    )
    return paths[0]
