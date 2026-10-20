"""Isolated SGLang-Omni environment for Drama 3; ready only after validation.

SGLang-Omni needs Torch 2.13, which conflicts with ComfyUI's stack.
Modal bakes its uv environment into the image; Linux installs prepare it on first use.
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
    DRAMA3_SGLANG_READY_MARKER as READY_MARKER,
    drama3_sglang_fingerprint,
    drama3_sglang_install_commands,
)

from h3_app.errors import H3Error

# Inherited interpreter settings would leak ComfyUI's packages into the venv.
SCRUBBED_ENV = ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "UV_CONSTRAINT", "VIRTUAL_ENV")


def fingerprint() -> str:
    return drama3_sglang_fingerprint()


def default_venv(script_dir: Path) -> Path:
    configured = os.getenv("H3_DRAMA3_SGLANG_VENV")
    return Path(configured) if configured else Path(script_dir) / "h3" / "drama3-sglang-venv"


def python_path(venv: Path) -> Path:
    return Path(venv) / "bin" / "python"


def executable(venv: Path, name: str) -> Path:
    return Path(venv) / "bin" / name


def is_ready(venv: Path) -> bool:
    marker = Path(venv) / READY_MARKER
    try:
        return python_path(venv).is_file() and marker.read_text(encoding="utf-8").strip() == fingerprint()
    except OSError:
        return False


def uv_command() -> list[str] | None:
    """The uv executable, or uv run as a module of this interpreter (Colab, setup_h3.py)."""
    found = shutil.which("uv")
    if found:
        return [found]
    import importlib.util

    return [sys.executable, "-m", "uv"] if importlib.util.find_spec("uv") else None


def install_commands(venv: Path, uv: str | list[str] = "uv") -> list[list[str]]:
    """The install steps, shared with the Modal image build."""
    return drama3_sglang_install_commands(venv, uv)


def install_env(venv: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED_ENV}
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PATH=str(Path(venv) / "bin") + os.pathsep + os.environ.get("PATH", ""),
    )
    # Standalone Python builds name clang as their C compiler; fall back to gcc
    # for dependencies that compile extensions (cdifflib) when clang is absent.
    if "CC" not in env and not shutil.which("clang") and shutil.which("gcc"):
        env.update(CC="gcc", CXX="g++")
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
    """Create or reuse an environment; publish readiness only after the smoke test."""
    venv = Path(venv)
    label = "SGLang-Omni"
    if os.name != "posix":
        raise H3Error(f"Drama 3 runs on {label}, which requires Linux.")
    run = run or (lambda command, **kwargs: subprocess.run(command, check=True, **kwargs))
    with _setup_lock(venv.parent / f".{venv.name}.lock"):
        if not force and is_ready(venv):
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error(f"uv is required to set up the {label} Drama 3 backend; run `pip install uv` and retry.")
        marker = venv / READY_MARKER
        marker.unlink(missing_ok=True)
        log(f"[drama3-serving] Preparing the isolated {label} environment (first use takes several minutes).")
        env = install_env(venv)
        commands = install_commands(venv, uv)
        if python_path(venv).is_file():
            commands = commands[1:]  # keep a usable interpreter; reinstall pins into it
        for command in commands:
            run(command, env=env)
        marker.write_text(fingerprint(), encoding="utf-8")
        log(f"[drama3-serving] {label} environment ready.")
    return python_path(venv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(prepare(args.venv, force=args.force))


if __name__ == "__main__":
    main()
