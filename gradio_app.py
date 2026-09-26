#!/usr/bin/env python3
"""Launch H3, or expose its temporary compatibility API to existing callers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory and any nested repository checkout are on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NESTED = _ROOT / "minimax-h3"
if not (_ROOT / "h3_ui").is_dir() and (_NESTED / "h3_ui").is_dir() and str(_NESTED) not in sys.path:
    sys.path.insert(0, str(_NESTED))

try:
    from h3_ui import application
except ModuleNotFoundError:
    # Attempt automatic recovery from Git if h3_ui is missing from the working tree
    try:
        import subprocess

        print("[h3-run] h3_ui missing; restoring repository from GitHub...", flush=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(_ROOT),
                "fetch",
                "--depth",
                "1",
                "https://github.com/keboqi/minimax-h3.git",
                "HEAD",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(_ROOT),
                "checkout",
                "FETCH_HEAD",
                "--",
                ".",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )
    except Exception as exc:
        print(f"[h3-run] automatic Git recovery failed: {exc}", flush=True)

    from h3_ui import application

if __name__ == "__main__":
    application.main()
else:
    # Preserve module identity so legacy integrations patch the same callbacks
    # that composition injects into services. New code imports h3_app directly.
    sys.modules[__name__] = application
