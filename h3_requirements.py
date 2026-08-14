#!/usr/bin/env python3
"""Shared dependency compatibility policy for MiniMax H3 deployments."""
from __future__ import annotations

import re
from collections.abc import Iterable
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen


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
COMFY_FRONTEND_VERSION = "1.48.7"

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


def comfy_frontend_static_references(content: str) -> list[str]:
    """Return immutable browser assets referenced by a ComfyUI index."""
    candidates = re.findall(
        r"(?i)\b(?:href|src)\s*=\s*[\"']([^\"']+)",
        content,
    )
    references = []
    for candidate in candidates:
        parsed = urlsplit(candidate)
        if candidate.startswith(("//", "#")) or parsed.scheme or not parsed.path:
            continue
        normalized = parsed.path.lstrip("/").removeprefix("./")
        # user.css and api/userdata/user.css are optional, runtime-served files.
        # Hashed assets and the icon stylesheet must exist in the frontend wheel.
        if normalized.startswith("assets/") or normalized == (
            "materialdesignicons.min.css"
        ):
            references.append(candidate)
    return references


def comfy_frontend_package_is_ready() -> bool:
    """Validate the pinned frontend package and its immutable index assets."""
    try:
        if version("comfyui-frontend-package") != COMFY_FRONTEND_VERSION:
            return False
        import comfyui_frontend_package
    except (ImportError, PackageNotFoundError):
        return False

    root = Path(resources.files(comfyui_frontend_package) / "static")
    index = root / "index.html"
    if not index.is_file():
        return False
    try:
        content = index.read_text(encoding="utf-8")
    except OSError:
        return False

    references = comfy_frontend_static_references(content)
    return bool(references) and all(
        (root / urlsplit(reference).path.lstrip("/").removeprefix("./")).is_file()
        for reference in references
    )


def probe_comfy_frontend(base_url: str, timeout: float = 5) -> int:
    """Fetch a ComfyUI index and all immutable assets; return asset count."""
    index_url = base_url.rstrip("/") + "/"
    with urlopen(index_url, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"ComfyUI index returned HTTP {response.status}")
        content = response.read().decode(
            response.headers.get_content_charset() or "utf-8",
            errors="replace",
        )

    references = comfy_frontend_static_references(content)
    if not references:
        raise RuntimeError("ComfyUI index did not reference any static assets")
    for reference in references:
        asset_url = urljoin(index_url, reference)
        with urlopen(asset_url, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"ComfyUI asset returned HTTP {response.status}: {asset_url}"
                )
    return len(references)


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
    assert COMFY_FRONTEND_VERSION == "1.48.7"
    assert comfy_frontend_static_references(
        '<link href="user.css"><link href="materialdesignicons.min.css">'
        '<script src="./assets/index-abc.js"></script>'
    ) == ["materialdesignicons.min.css", "./assets/index-abc.js"]
    print("h3_requirements selftest OK")


if __name__ == "__main__":
    selftest()
