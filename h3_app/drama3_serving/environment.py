"""Isolated serving environments for Drama 3; ready only after validation.

vLLM-Omni and SGLang-Omni each need Torch 2.13 and their own Transformers pin,
which conflict with ComfyUI's stack, so each lives in its own uv environment.
Modal bakes them into the image; other Linux installs prepare them on first use.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from h3_requirements import (
    DRAMA3_VLLM_READY_MARKER as READY_MARKER,
    drama3_sglang_fingerprint,
    drama3_sglang_install_commands,
    drama3_vllm_fingerprint,
    drama3_vllm_install_commands,
)

from h3_app.errors import H3Error

# Inherited interpreter settings would leak ComfyUI's packages into the venv.
SCRUBBED_ENV = ("PYTHONPATH", "PYTHONHOME", "PIP_CONSTRAINT", "UV_CONSTRAINT", "VIRTUAL_ENV")


@dataclass(frozen=True)
class Runtime:
    kind: str  # "vllm" or "sglang"
    label: str
    venv_env: str  # environment variable that overrides the venv location
    default_dir: str
    fingerprint: Callable[[], str]
    commands: Callable[..., list[list[str]]]


RUNTIMES = {
    "vllm": Runtime("vllm", "vLLM-Omni", "H3_DRAMA3_VLLM_VENV", "drama3-vllm-venv",
                    drama3_vllm_fingerprint, drama3_vllm_install_commands),
    "sglang": Runtime("sglang", "SGLang-Omni", "H3_DRAMA3_SGLANG_VENV", "drama3-sglang-venv",
                      drama3_sglang_fingerprint, drama3_sglang_install_commands),
}


def fingerprint(kind: str = "vllm") -> str:
    return RUNTIMES[kind].fingerprint()


def default_venv(script_dir: Path, kind: str = "vllm") -> Path:
    runtime = RUNTIMES[kind]
    configured = os.getenv(runtime.venv_env)
    return Path(configured) if configured else Path(script_dir) / "h3" / runtime.default_dir


def python_path(venv: Path) -> Path:
    return Path(venv) / "bin" / "python"


def executable(venv: Path, name: str) -> Path:
    return Path(venv) / "bin" / name


def is_ready(venv: Path, kind: str = "vllm") -> bool:
    marker = Path(venv) / READY_MARKER
    try:
        return python_path(venv).is_file() and marker.read_text(encoding="utf-8").strip() == fingerprint(kind)
    except OSError:
        return False


def uv_command() -> list[str] | None:
    """The uv executable, or uv run as a module of this interpreter (Colab, setup_h3.py)."""
    found = shutil.which("uv")
    if found:
        return [found]
    import importlib.util

    return [sys.executable, "-m", "uv"] if importlib.util.find_spec("uv") else None


def install_commands(venv: Path, uv: str | list[str] = "uv", kind: str = "vllm") -> list[list[str]]:
    """The install steps, shared with the Modal image build."""
    return RUNTIMES[kind].commands(venv, uv)


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


def prepare(venv: Path, *, kind: str = "vllm", force: bool = False, run=None, log=print) -> Path:
    """Create or reuse an environment; publish readiness only after the smoke test."""
    venv = Path(venv)
    label = RUNTIMES[kind].label
    if os.name != "posix":
        raise H3Error(f"The {label} Drama 3 backend runs on Linux only; use the ComfyUI backend here.")
    run = run or (lambda command, **kwargs: subprocess.run(command, check=True, **kwargs))
    with _setup_lock(venv.parent / f".{venv.name}.lock"):
        if not force and is_ready(venv, kind):
            return python_path(venv)
        uv = uv_command()
        if not uv:
            raise H3Error(f"uv is required to set up the {label} Drama 3 backend; run `pip install uv` and retry.")
        marker = venv / READY_MARKER
        marker.unlink(missing_ok=True)
        log(f"[drama3-serving] Preparing the isolated {label} environment (first use takes several minutes).")
        env = install_env(venv)
        commands = install_commands(venv, uv, kind)
        if python_path(venv).is_file():
            commands = commands[1:]  # keep a usable interpreter; reinstall pins into it
        for command in commands:
            run(command, env=env)
        marker.write_text(fingerprint(kind), encoding="utf-8")
        log(f"[drama3-serving] {label} environment ready.")
    return python_path(venv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(RUNTIMES), default="vllm")
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(prepare(args.venv, kind=args.kind, force=args.force))


if __name__ == "__main__":
    main()
