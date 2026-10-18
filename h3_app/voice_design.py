"""Qwen VoiceDesign in a disposable worker with its own dependency environment.

The official qwen-tts package pins Transformers 4.57.3. Never install it in
ComfyUI's environment. Model downloads persist; the GPU worker exits per clip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from h3_app.drama3_serving.environment import _setup_lock, uv_command
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled
from h3_requirements import (
    VOICE_DESIGN_READY_MARKER as READY_MARKER,
    voice_design_fingerprint as fingerprint,
    voice_design_install_commands as install_commands,
)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LANGUAGES = ("Auto", "Chinese", "English", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian")
_LOCK = threading.Lock()


def python_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT", "UV_CONSTRAINT"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    return env


def run_process(command: list[str], log: Path, *, timeout: float, env: dict[str, str] | None = None,
                label: str = "Qwen VoiceDesign") -> None:
    """Drain output to disk; cancellation/timeout always reaps the GPU process.

    ``env`` adds variables (such as a Hugging Face token) on top of :func:`worker_env`;
    ``label`` names the worker in failure messages.
    """
    check_cancelled()
    with log.open("ab") as stream:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        process = subprocess.Popen(command, env={**worker_env(), **(env or {})}, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, **kwargs)
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
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def prepare(runtime, log: Path) -> Path:
    venv = Path(os.getenv("H3_VOICE_DESIGN_VENV") or runtime.script_dir / "h3" / "voice-design-venv")
    with _LOCK, _setup_lock(venv.parent / f".{venv.name}.lock"):
        marker = venv / READY_MARKER
        if python_path(venv).is_file() and marker.is_file() and marker.read_text().strip() == fingerprint():
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error("Install uv (`pip install uv`) to prepare the isolated Qwen VoiceDesign runtime.")
        marker.unlink(missing_ok=True)
        commands = install_commands(venv, uv)
        if python_path(venv).is_file():
            commands = commands[1:]
        for command in commands:
            run_process(command, log, timeout=1800)
        marker.write_text(fingerprint(), encoding="utf-8")
    return python_path(venv)


def validate_inputs(description, text, language):
    description, text = str(description or "").strip(), str(text or "").strip()
    if not description or len(description) > 2000:
        raise H3Error("Describe the voice in 1–2000 characters.")
    if not text or len(text) > 1000:
        raise H3Error("Add 1–1000 characters of preview speech for the reference clip.")
    if any(char in text for char in "<>[]"):
        raise H3Error("Preview text must contain spoken words without speaker tags or delivery controls.")
    if language not in LANGUAGES:
        raise H3Error("Choose a supported VoiceDesign language.")
    return description, text, language


def generate_reference(description, text, language, runtime, *, free_gpu) -> tuple[Path, str]:
    description, text, language = validate_inputs(description, text, language)
    output = runtime.output_dir / "drama3_voices"
    output.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    result = output / f"QwenVoiceDesign_{token}.wav"
    log = output / f"QwenVoiceDesign_{token}.log"
    python = prepare(runtime, log)
    from h3_app.drama3_serving.server import release_all, persist_caches

    release_all()
    free_gpu()
    worker = Path(__file__).with_name("voice_design_worker.py")
    # The request stays off the command line; it is also the source of the exact transcript.
    with tempfile.TemporaryDirectory(prefix="h3-voice-design-") as directory:
        request = Path(directory) / "request.json"
        request.write_text(json.dumps({"model": MODEL_ID, "description": description, "text": text,
                                       "language": language, "output": str(result.resolve()),
                                       "cache_dir": str((runtime.comfy_dir / "models" / "qwen3_tts").resolve())},
                                      ensure_ascii=False), encoding="utf-8")
        try:
            run_process([str(python), str(worker), str(request)], log,
                        timeout=runtime.generation_timeout)
            if not result.is_file() or result.stat().st_size <= 44:
                raise H3Error(f"Qwen VoiceDesign returned no audio. See {log}.")
        except BaseException:
            result.unlink(missing_ok=True)
            raise
    persist_caches()  # Commit downloaded models and the clip when running on Modal's volume.
    return result, text
