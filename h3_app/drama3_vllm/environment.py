"""Isolated vLLM-Omni environment for Drama 3; ready only after validation.

vLLM-Omni needs Torch 2.13 and Transformers 5.13, which conflict with ComfyUI's
stack, so it lives in its own uv environment. Modal bakes it into the image with
``install_commands``; Linux installs prepare it on first use with ``prepare``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from h3_requirements import (
    DRAMA3_VLLM_READY_MARKER as READY_MARKER,
    drama3_vllm_fingerprint as fingerprint,
    drama3_vllm_install_commands,
)

from h3_app.errors import H3Error

# Inherited interpreter settings would leak ComfyUI's packages into the venv.
SCRUBBED_ENV = ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "UV_CONSTRAINT", "VIRTUAL_ENV")


def default_venv(script_dir: Path) -> Path:
    configured = os.getenv("H3_DRAMA3_VLLM_VENV")
    return Path(configured) if configured else Path(script_dir) / "h3" / "drama3-vllm-venv"


def python_path(venv: Path) -> Path:
    return Path(venv) / "bin" / "python"


def vllm_executable(venv: Path) -> Path:
    return Path(venv) / "bin" / "vllm"


def is_ready(venv: Path) -> bool:
    marker = Path(venv) / READY_MARKER
    try:
        return python_path(venv).is_file() and marker.read_text(encoding="utf-8").strip() == fingerprint()
    except OSError:
        return False


def uv_command() -> list[str] | None:
    """The uv executable, or uv run as a module of this interpreter (Colab, setup_h3.py)."""
    executable = shutil.which("uv")
    if executable:
        return [executable]
    import importlib.util

    return [sys.executable, "-m", "uv"] if importlib.util.find_spec("uv") else None


def install_commands(venv: Path, uv: str | list[str] = "uv") -> list[list[str]]:
    """The install steps, shared with the Modal image build."""
    return drama3_vllm_install_commands(venv, uv)


def install_env(venv: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env.update(
        # vllm-omni's setup.py picks platform requirements from this, not the GPU.
        VLLM_OMNI_TARGET_DEVICE="cuda",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PATH=str(Path(venv) / "bin") + os.pathsep + os.environ.get("PATH", ""),
    )
    return env


@contextlib.contextmanager
def _setup_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def prepare(venv: Path, *, force: bool = False, run=None, log=print) -> Path:
    """Create or reuse the environment; publish readiness only after the smoke test."""
    venv = Path(venv)
    if os.name != "posix":
        raise H3Error("The vLLM-Omni Drama 3 backend runs on Linux only; use the ComfyUI backend here.")
    run = run or (lambda command, **kwargs: subprocess.run(command, check=True, **kwargs))
    with _setup_lock(venv.parent / f".{venv.name}.lock"):
        if not force and is_ready(venv):
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error("uv is required to set up the vLLM-Omni Drama 3 backend; "
                          "run `pip install uv` and retry.")
        marker = venv / READY_MARKER
        marker.unlink(missing_ok=True)
        log("[drama3-vllm] Preparing the isolated vLLM-Omni environment (first use takes several minutes).")
        env = install_env(venv)
        commands = install_commands(venv, uv)
        if python_path(venv).is_file():
            commands = commands[1:]  # keep a usable interpreter; reinstall pins into it
        for command in commands:
            run(command, env=env)
        marker.write_text(fingerprint(), encoding="utf-8")
        log("[drama3-vllm] Environment ready.")
    return python_path(venv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fingerprint", action="store_true", help="print the readiness fingerprint and exit")
    args = parser.parse_args()
    if args.fingerprint:
        print(fingerprint())
        return
    print(prepare(args.venv, force=args.force))


if __name__ == "__main__":
    main()
