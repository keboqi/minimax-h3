#!/usr/bin/env python3
"""Install/update local MiniMax H3 dependencies, custom nodes, and models."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from h3_models import sync_models, write_json_atomic
from h3_requirements import (
    NUMPY_VERSION,
    SCIPY_VERSION,
    TORCH_INDEX,
    TORCH_VERSION,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    filter_pinned_requirements,
)


COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
SOL_REPO = "https://github.com/Saganaki22/ComfyUI-sol-attn.git"
SOL_REF = "90467f1c633ce53af7d77e4a1cc243b5001d89b0"
SPECTRUM_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"
SPECTRUM_REF = "6a37360e3e785d2b9b5ad58190af380a8de8ec1a"  # v0.2.2
SAGE_WHEEL_URL = "https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
SAGE_WHEEL_NAME = "sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLED_ACCEL_NODE = (
    SCRIPT_DIR / "custom_nodes" / "H3Acceleration" / "__init__.py"
)

_UV_CMD: list[str] | None = None


def run(
    *args,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("[h3-setup]", *args, flush=True)
    subprocess.run(
        [str(arg) for arg in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def ensure_uv() -> list[str]:
    """Use system uv when available; bootstrap it once otherwise."""
    global _UV_CMD
    if _UV_CMD is not None:
        return _UV_CMD.copy()

    executable = shutil.which("uv")
    if executable:
        _UV_CMD = [executable]
    else:
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "uv>=0.8",
        )
        executable = shutil.which("uv")
        _UV_CMD = (
            [executable]
            if executable
            else [sys.executable, "-m", "uv"]
        )
    return _UV_CMD.copy()


def uv_pip(
    *packages: str,
    index: str | None = None,
    no_deps: bool = False,
) -> None:
    """Install into the current Python/Conda environment without a new venv."""
    command = ensure_uv() + [
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
    ]
    if index:
        command += ["--index-url", index]
    if no_deps:
        command += ["--no-deps"]
    command += packages
    run(*command)


def sync_git_repo(
    url: str,
    dest: Path,
    *,
    ref: str | None = None,
) -> None:
    """Clone/update a repo; optionally pin it to an exact commit/ref."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    fetch_ref = ref or "HEAD"
    if (dest / ".git").is_dir():
        run("git", "-C", dest, "remote", "set-url", "origin", url)
        run(
            "git", "-C", dest, "fetch", "--depth", "1",
            "origin", fetch_ref,
        )
        run("git", "-C", dest, "reset", "--hard", "FETCH_HEAD")
        run("git", "-C", dest, "clean", "-ffd")
    else:
        if dest.exists():
            incomplete = dest.with_name(dest.name + ".incomplete")
            if incomplete.exists():
                shutil.rmtree(incomplete)
            dest.rename(incomplete)
        if ref:
            run("git", "init", dest)
            run("git", "-C", dest, "remote", "add", "origin", url)
            run(
                "git", "-C", dest, "fetch", "--depth", "1",
                "origin", ref,
            )
            run("git", "-C", dest, "checkout", "--detach", "FETCH_HEAD")
        else:
            run("git", "clone", "--depth", "1", url, dest)

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "--short=12", "HEAD"],
            text=True,
        ).strip()
        print(f"[h3-setup] node revision {dest.name}: {revision}")
    except Exception:
        pass




def current_torch_version() -> str | None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
        capture_output=True,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def torch_stack_matches() -> bool:
    version = current_torch_version()
    return bool(version and version.startswith(TORCH_VERSION + "+cu130"))


def install_pinned_torch_stack() -> None:
    uv_pip(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index=TORCH_INDEX,
    )


def current_numpy_version() -> str | None:
    probe = subprocess.run(
        [sys.executable, "-c", "import numpy; print(numpy.__version__)"],
        text=True,
        capture_output=True,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def numpy_stack_matches() -> bool:
    return current_numpy_version() == NUMPY_VERSION


def install_pinned_numpy_stack() -> None:
    uv_pip(
        f"numpy=={NUMPY_VERSION}",
        f"scipy=={SCIPY_VERSION}",
    )


def install_comfy_requirements(comfy: Path) -> None:
    """Install ComfyUI deps without allowing Torch/NumPy ABI drift."""
    requirements = comfy / "requirements.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    filtered, skipped = filter_pinned_requirements(lines)
    for package, requirement in skipped:
        print(
            f"[h3-setup] Keeping pinned {package}; "
            f"skipping ComfyUI entry: {requirement}",
            flush=True,
        )

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(filtered) + "\n")
        filtered_path = Path(handle.name)

    try:
        uv_pip("-r", str(filtered_path))
    finally:
        filtered_path.unlink(missing_ok=True)

    if not torch_stack_matches():
        found = current_torch_version()
        print(
            f"[h3-setup] Torch drift detected ({found}); restoring "
            f"{TORCH_VERSION}+cu130",
            flush=True,
        )
        install_pinned_torch_stack()

    if not numpy_stack_matches():
        found = current_numpy_version()
        print(
            f"[h3-setup] NumPy drift detected ({found}); restoring "
            f"{NUMPY_VERSION}",
            flush=True,
        )
        install_pinned_numpy_stack()


def sageattention_importable() -> bool:
    """Check Sage in a brand-new interpreter."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sageattention; "
                "print(getattr(sageattention, '__version__', 'unknown')); "
                "print(sageattention.__file__)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0:
        detail = probe.stdout.strip().replace("\\n", " | ")
        print(f"[h3-setup] SageAttention fresh-import OK: {detail}", flush=True)
        return True
    error = (probe.stderr or probe.stdout).strip()
    print(f"[h3-setup] SageAttention fresh-import failed: {error}", flush=True)
    return False


def ensure_sageattention(install_dir: Path) -> bool:
    """Require the prebuilt SageAttention wheel; never compile Sage source."""
    if sageattention_importable():
        print("[h3-setup] Reusing working SageAttention installation", flush=True)
        return True

    print(
        f"[h3-setup] Installing prebuilt SageAttention wheel: "
        f"{SAGE_WHEEL_NAME}",
        flush=True,
    )
    command = ensure_uv() + [
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
        "--no-deps",
        SAGE_WHEEL_URL,
    ]
    run(*command)

    if not sageattention_importable():
        raise RuntimeError(
            "The prebuilt SageAttention wheel installed but still failed a "
            "fresh-process import check. Source compilation is disabled by "
            "design; inspect the fresh-import error above."
        )

    print("[h3-setup] Prebuilt SageAttention wheel is ready", flush=True)
    return True

def install_environment(comfy: Path) -> None:
    ensure_uv()
    uv_pip("setuptools<82", "wheel")
    install_pinned_torch_stack()
    install_pinned_numpy_stack()

    sync_git_repo(COMFY_REPO, comfy)
    install_comfy_requirements(comfy)
    uv_pip(
        "gradio>=5,<7",
        "huggingface_hub>=0.34",
        "requests>=2.32",
        "websocket-client>=1.8",
        "aiohttp>=3.11,<4",
        "httpx>=0.27",
        "uvicorn>=0.30",
        "Pillow>=10",
    )
    install_pinned_numpy_stack()


def sync_external_nodes(
    comfy: Path,
    *,
    install_requirements: bool,
) -> None:
    sol = comfy / "custom_nodes" / "ComfyUI_sol-attn_Blackwell"
    sync_git_repo(SOL_REPO, sol, ref=SOL_REF)
    if install_requirements and (sol / "requirements.txt").is_file():
        uv_pip("-r", str(sol / "requirements.txt"), no_deps=True)

    spectrum = comfy / "custom_nodes" / "ComfyUI-Spectrum-MiniMax-H3"
    sync_git_repo(SPECTRUM_REPO, spectrum, ref=SPECTRUM_REF)
    if install_requirements and (spectrum / "requirements.txt").is_file():
        uv_pip("-r", str(spectrum / "requirements.txt"), no_deps=True)



def install_bundled_nodes(comfy: Path) -> None:
    if not BUNDLED_ACCEL_NODE.is_file():
        raise RuntimeError(
            f"Missing bundled H3 acceleration node: {BUNDLED_ACCEL_NODE}"
        )

    destination = (
        comfy / "custom_nodes" / "H3Acceleration" / "__init__.py"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLED_ACCEL_NODE, destination)
    print(f"[h3-setup] synced {destination}")


def sync_model_inventory(install_dir: Path, comfy: Path) -> None:
    manifest_path = install_dir / "h3_model_manifest.json"
    config_path = install_dir / "h3_models.json"

    config = sync_models(
        root=comfy / "models",
        manifest_path=manifest_path,
        log_prefix="[h3-setup]",
    )
    write_json_atomic(config_path, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", default="h3")
    parser.add_argument("--skip-env", action="store_true")
    args = parser.parse_args()

    install_dir = Path(args.install_dir).expanduser().resolve()
    comfy = install_dir / "ComfyUI"
    install_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_env:
        install_environment(comfy)
    elif not (comfy / "main.py").is_file():
        raise RuntimeError(
            "--skip-env requires an existing ComfyUI installation"
        )
    elif not torch_stack_matches():
        found = current_torch_version()
        print(
            f"[h3-setup] Existing Torch {found} is incompatible with the "
            f"prebuilt SageAttention wheel; restoring "
            f"{TORCH_VERSION}+cu130",
            flush=True,
        )
        install_pinned_torch_stack()

    if not numpy_stack_matches():
        found = current_numpy_version()
        print(
            f"[h3-setup] Existing NumPy {found} is incompatible with "
            f"compiled extensions; restoring {NUMPY_VERSION}",
            flush=True,
        )
        install_pinned_numpy_stack()

    ensure_sageattention(install_dir)
    sync_external_nodes(
        comfy,
        install_requirements=not args.skip_env,
    )
    install_bundled_nodes(comfy)
    sync_model_inventory(install_dir, comfy)


if __name__ == "__main__":
    main()
