"""Provision isolated uv environments and run setup steps in them.

Tools whose pins conflict with ComfyUI's stack (SGLang-Omni, qwen-tts, the
dubbing tools) each get their own venv. A venv is ready only once its marker
holds the current fingerprint, written after every install step succeeded.
"""

from __future__ import annotations

import contextlib
import codecs
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled

# Inherited interpreter settings would leak ComfyUI's packages into the venv.
SCRUBBED_ENV = ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "UV_CONSTRAINT", "VIRTUAL_ENV")

_VENV_LOCKS: dict[Path, threading.Lock] = {}
_VENV_LOCKS_GUARD = threading.Lock()


def uv_command() -> list[str] | None:
    """The uv executable, or uv run as a module of this interpreter (Colab, setup_h3.py)."""
    found = shutil.which("uv")
    if found:
        return [found]
    return [sys.executable, "-m", "uv"] if importlib.util.find_spec("uv") else None


def compiler_env(env=None) -> dict[str, str]:
    """``CC``/``CXX`` for uv installs that compile C extensions (cdifflib, diffq).

    Standalone Python builds name clang as their C compiler; fall back to gcc when
    clang is absent and no compiler was chosen.
    """
    env = os.environ if env is None else env
    if "CC" not in env and not shutil.which("clang") and shutil.which("gcc"):
        return {"CC": "gcc", "CXX": "g++"}
    return {}


@contextlib.contextmanager
def setup_lock(path: Path):
    """Serialize setup of one venv across processes (POSIX file lock)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def python_path(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker_env() -> dict[str, str]:
    """This process's environment without the interpreter settings that would leak into a venv."""
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    return env


def popen_flags() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def reap(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_process(command: list[str], log: Path, *, timeout: float, env: dict[str, str] | None = None,
                label: str, report: Callable[[str], None] | None = None) -> None:
    """Drain output to disk; cancellation/timeout always reaps the process.

    ``env`` adds variables (such as a Hugging Face token) on top of :func:`worker_env`;
    ``label`` names the step in failure messages.
    """
    check_cancelled()
    with log.open("ab") as stream:
        reader = log.open("rb") if report is not None else None
        if reader is not None:
            reader.seek(0, os.SEEK_END)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        last_report = 0.0
        latest, dirty = "", False

        def progress(final=False):
            nonlocal last_report, latest, dirty
            if reader is None:
                return
            while block := reader.read(16384):
                output = decoder.decode(block)
                print(output, end="", flush=True)
                lines = [line.strip() for line in output.replace("\r", "\n").splitlines() if line.strip()]
                if lines:
                    latest = lines[-1].strip()
                    dirty = bool(latest)
            now = time.monotonic()
            if dirty and (final or now - last_report >= 1):
                report(f"{label}: {latest[-300:]}")
                last_report = now
                dirty = False

        if report is not None:
            print(f"[{label}] Output log: {log}", flush=True)
        process = None
        try:
            process = subprocess.Popen(command, env={**worker_env(), **(env or {})}, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, **popen_flags())
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                check_cancelled()
                progress()
                if time.monotonic() >= deadline:
                    raise H3Error(f"{label} timed out. See {log}.")
                time.sleep(0.25)
            progress(final=True)
            check_cancelled()
            if process.returncode:
                raise H3Error(f"{label} failed (exit {process.returncode}). See {log}.")
        finally:
            if process is not None:
                reap(process)
            if reader is not None:
                reader.close()


def _venv_lock(venv: Path) -> threading.Lock:
    with _VENV_LOCKS_GUARD:
        return _VENV_LOCKS.setdefault(venv.resolve(), threading.Lock())


def prepare_venv(venv: Path, log: Path, *, name: str, label: str, fingerprint: str, ready_marker: str,
                 install_commands: Callable[[Path, list[str]], list[list[str]]], timeout: float,
                 env: dict[str, str] | None = None, report: Callable[[str], None] | None = None) -> Path:
    """Create or reuse ``venv``; its interpreter once every install step has succeeded.

    ``name`` appears in the missing-uv message and ``label`` in step failures. An
    existing interpreter is kept and the pins are reinstalled into it.
    """
    if report is not None:
        report(f"Checking {label} environment (waiting if another setup is running)")
        print(f"[{label}] Checking environment: {venv}", flush=True)
    with _venv_lock(venv), setup_lock(venv.parent / f".{venv.name}.lock"):
        marker = venv / ready_marker
        python = python_path(venv)
        if python.is_file() and marker.is_file() and marker.read_text().strip() == fingerprint:
            if report is not None:
                report(f"Using the prepared {label} environment")
                print(f"[{label}] Environment is ready; skipping installation", flush=True)
            return python
        uv = uv_command()
        if not uv:
            raise H3Error(f"Install uv (`pip install uv`) to prepare the isolated {name} runtime.")
        marker.unlink(missing_ok=True)
        commands = install_commands(venv, uv)
        if python.is_file():
            commands = commands[1:]
        for index, command in enumerate(commands, 1):
            step = ("Installing Python packages" if "install" in command else
                    "Checking installed package imports" if "-c" in command else "Creating Python environment")
            if report is not None:
                message = f"{label}: {step} ({index}/{len(commands)}); log: {log}"
                report(message)
                print(message, flush=True)
            progress_args = {"report": report} if report is not None else {}
            run_process(command, log, timeout=timeout, env=env, label=f"{label} setup", **progress_args)
        marker.write_text(fingerprint, encoding="utf-8")
    return python
