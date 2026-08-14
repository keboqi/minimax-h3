#!/usr/bin/env python3
"""Shared dependency compatibility policy for MiniMax H3 deployments."""
from __future__ import annotations

import re
from collections.abc import Iterable


TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
NUMPY_VERSION = "1.26.4"
SCIPY_VERSION = "1.15.3"
# ComfyUI-LTXVideo ac4d998 imports ``pad`` from Kornia's pyramid module.
# Kornia 0.8.2+ removed that module-level compatibility export.
KORNIA_VERSION = "0.8.1"
# Keep the ComfyUI source and its pinned comfy-kitchen dependency in lockstep.
# This revision includes native LTX 2.5 NVFP4 checkpoint loading and enables
# Dynamic VRAM automatically on supported NVIDIA WSL installations.
COMFY_REF = "2220d111c8b036f094eb465400fdf962626e4afa"
COMFY_KITCHEN_VERSION = "0.2.30"

ABI_CONSTRAINTS = (
    f"torch=={TORCH_VERSION}",
    f"torchvision=={TORCHVISION_VERSION}",
    f"torchaudio=={TORCHAUDIO_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    f"scipy=={SCIPY_VERSION}",
)

PINNED_REQUIREMENTS = frozenset(
    {"torch", "torchvision", "torchaudio", "numpy", "scipy"}
)


def requirement_name(line: str) -> str | None:
    """Return a normalized package name for a requirement-file line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Keep options, includes, URLs, and editable requirements untouched. They
    # are not safely comparable to one of our pinned package names.
    if stripped.startswith(("-", "http://", "https://", "git+")):
        return None

    name = re.split(r"[<>=!~;\[\s]", stripped, maxsplit=1)[0]
    return name.lower().replace("_", "-") or None


def filter_pinned_requirements(
    lines: Iterable[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Remove ABI-sensitive packages and report the skipped entries."""
    filtered: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in lines:
        package = requirement_name(line)
        if package in PINNED_REQUIREMENTS:
            skipped.append((package, line.strip()))
        else:
            filtered.append(line)
    return filtered, skipped


def selftest() -> None:
    source = [
        "torch>=2.0",
        "NumPy==2.0; python_version >= '3.12'",
        "scipy[extra]~=1.14",
        "requests>=2.32",
        "-r optional.txt",
        "# torch is intentionally pinned elsewhere",
        "",
    ]
    filtered, skipped = filter_pinned_requirements(source)
    assert [package for package, _ in skipped] == ["torch", "numpy", "scipy"]
    assert filtered == source[3:]
    assert ABI_CONSTRAINTS == (
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
        "numpy==1.26.4",
        "scipy==1.15.3",
    )
    assert KORNIA_VERSION == "0.8.1"
    print("h3_requirements selftest OK")


if __name__ == "__main__":
    selftest()
