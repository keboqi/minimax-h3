"""Select Python 3.12 before installing the standalone GPU runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def python_version(executable: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(executable), "-I", "-c", "import sys; print('%s.%s' % sys.version_info[:2])"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run(*args: str | Path) -> None:
    # stdout contains only the selected interpreter for the Bash launcher.
    subprocess.run([str(arg) for arg in args], check=True, stdout=sys.stderr)


def ensure_python312(install_dir: Path) -> Path:
    if sys.version_info[:2] == (3, 12):
        return Path(sys.executable)

    environment = install_dir / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    if environment.exists():
        if python_version(python) != "3.12":
            raise RuntimeError(
                f"Existing environment {environment} is not a working Python 3.12 "
                "environment. Move it aside before rerunning; it has been preserved."
            )
        print(f"[h3-run] Reusing Python 3.12 environment at {environment}", file=sys.stderr)
        return python

    install_dir.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if not uv:
        bootstrap = install_dir / ".bootstrap"
        cached_uv = bootstrap / "uv" / ("uv.exe" if os.name == "nt" else "uv")
        if not cached_uv.is_file():
            run(sys.executable, "-m", "pip", "install", "--upgrade", "--target", bootstrap, "uv>=0.8")
        if not cached_uv.is_file():
            raise RuntimeError("uv bootstrap did not produce an executable; install uv and rerun.")
        uv = str(cached_uv)

    print(
        f"[h3-run] Python {sys.version_info.major}.{sys.version_info.minor} is incompatible "
        f"with the pinned GPU wheels; creating Python 3.12 environment at {environment}",
        file=sys.stderr,
    )
    run(uv, "venv", "--python", "3.12", "--seed", environment)
    if python_version(python) != "3.12":
        raise RuntimeError(f"Failed to provision a working Python 3.12 interpreter at {python}.")
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", required=True)
    args = parser.parse_args()
    try:
        python = ensure_python312(Path(args.install_dir).expanduser().resolve())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[h3-run error] {error}\n")
    print(python)


if __name__ == "__main__":
    main()
