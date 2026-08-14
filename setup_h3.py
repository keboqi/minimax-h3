#!/usr/bin/env python3
"""Install/update local MiniMax H3 dependencies, custom nodes, and models."""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from h3_models import PRELOAD_MODEL_KEYS, sync_models, write_json_atomic
from h3_node_patches import patch_larry_turbo_node
from h3_requirements import (
    ABI_CONSTRAINTS,
    COMFY_FRONTEND_VERSION,
    COMFY_REF,
    KORNIA_VERSION,
    NUMPY_VERSION,
    SCIPY_VERSION,
    TORCH_INDEX,
    TORCH_VERSION,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    WSPROTO_VERSION,
    comfy_frontend_package_is_ready,
    filter_pinned_requirements,
)


COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
SOL_REPO = "https://github.com/Saganaki22/ComfyUI-sol-attn.git"
SOL_REF = "e1d211026583064d33dc4326207c6502e2442208"  # v0.6.1
SPECTRUM_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"
SPECTRUM_REF = "7911ec7827921de599492f21eade181211266029"  # v0.2.7
LARRY_TURBO_REPO = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
LARRY_TURBO_REF = "55fee864dd7b2976b1c4ce3c3d5f7968f181409f"
LTXVIDEO_REPO = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
LTXVIDEO_REF = "ac4d99839020b983e956a8ab67ec38aec1b6e65a"
KJNODES_REPO = "https://github.com/kijai/ComfyUI-KJNodes.git"
KJNODES_REF = "6ab7e8130e449ed2c0037589bcf84146ceb7fc9c"
CONTROLNET_AUX_REPO = "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
CONTROLNET_AUX_REF = "e8b689a513c3e6b63edc44066560ca5919c0576e"
VIDEO_DEPTH_REPO = "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
VIDEO_DEPTH_REF = "a0db08e63d1ea571601c45cde4aaee0acdd0544d"
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
    timeout: float | None = None,
) -> None:
    print("[h3-setup]", *args, flush=True)
    subprocess.run(
        [str(arg) for arg in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        timeout=timeout,
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


def _fresh_git_checkout(url: str, dest: Path, ref: str | None) -> None:
    """Create a complete checkout without reusing destination Git metadata."""
    run("git", "init", dest, timeout=30)
    run("git", "-C", dest, "remote", "add", "origin", url, timeout=30)
    run(
        "git", "-C", dest, "fetch", "--depth", "1",
        "origin", ref or "HEAD",
        timeout=300,
    )
    run(
        "git", "-C", dest, "checkout", "--detach", "FETCH_HEAD",
        timeout=120,
    )


def _remove_tree(path: Path) -> None:
    """Remove a tree, making copied read-only Git objects writable as needed."""
    def make_writable(function, failed_path, _error) -> None:
        mode = os.stat(failed_path, follow_symlinks=False).st_mode
        os.chmod(failed_path, mode | stat.S_IWUSR)
        function(failed_path)

    shutil.rmtree(path, onexc=make_writable)


def _git_worktree_is_valid(dest: Path) -> bool:
    """Return whether Git recognizes dest, not merely whether .git exists."""
    probe = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "--is-inside-work-tree"],
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _requires_staged_git_update(dest: Path) -> bool:
    """Teamspace mounts can block indefinitely during in-place Git resets."""
    return dest.as_posix().startswith("/teamspace/")


def _restore_tracked_files(
    url: str,
    dest: Path,
    ref: str | None,
) -> None:
    """Restore a full tracked tree while preserving untracked models/nodes."""
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dest.name}.restore-", dir=dest.parent)
    )
    try:
        _fresh_git_checkout(url, staging, ref)
        git_metadata = dest / ".git"
        if git_metadata.exists() or git_metadata.is_symlink():
            if git_metadata.is_dir() and not git_metadata.is_symlink():
                _remove_tree(git_metadata)
            else:
                git_metadata.unlink()
        shutil.copytree(
            staging,
            dest,
            dirs_exist_ok=True,
            symlinks=True,
        )
    finally:
        try:
            _remove_tree(staging)
        except FileNotFoundError:
            pass


def _missing_required_files(
    dest: Path,
    required_paths: tuple[str, ...],
) -> list[str]:
    return [path for path in required_paths if not (dest / path).is_file()]


def sync_git_repo(
    url: str,
    dest: Path,
    *,
    ref: str | None = None,
    required_paths: tuple[str, ...] = (),
    clean_untracked: bool = True,
) -> None:
    """Synchronize a repo and repair incomplete mounted worktrees."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    fetch_ref = ref or "HEAD"
    valid_worktree = _git_worktree_is_valid(dest)
    if valid_worktree and _requires_staged_git_update(dest):
        print(
            f"[h3-setup] Using staged Git update for mounted checkout "
            f"{dest.name}",
            flush=True,
        )
        _restore_tracked_files(url, dest, ref)
    elif valid_worktree:
        try:
            run(
                "git", "-C", dest, "remote", "set-url", "origin", url,
                timeout=30,
            )
            run(
                "git", "-C", dest, "fetch", "--depth", "1",
                "origin", fetch_ref,
                timeout=300,
            )
            # Mounted worktrees can have stale sparse/index metadata. Bound
            # both operations so network filesystems cannot block setup.
            run(
                "git", "-C", dest, "sparse-checkout", "disable",
                timeout=60,
            )
            run(
                "git", "-C", dest, "reset", "--hard", "FETCH_HEAD",
                timeout=90,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(
                f"[h3-setup] In-place Git update failed for {dest.name} "
                f"({exc}); rebuilding its checkout metadata",
                flush=True,
            )
            _restore_tracked_files(url, dest, ref)
        else:
            if clean_untracked:
                try:
                    run("git", "-C", dest, "clean", "-ffd", timeout=90)
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    print(
                        f"[h3-setup] WARNING: could not clean untracked "
                        f"files in {dest.name}: {exc}",
                        flush=True,
                    )
    else:
        if dest.exists():
            _restore_tracked_files(url, dest, ref)
        else:
            _fresh_git_checkout(url, dest, ref)

    missing = _missing_required_files(dest, required_paths)
    if missing:
        print(
            f"[h3-setup] Restoring incomplete checkout {dest.name}; "
            f"missing tracked files: {', '.join(missing)}",
            flush=True,
        )
        _restore_tracked_files(url, dest, ref)
        missing = _missing_required_files(dest, required_paths)
    if missing:
        raise RuntimeError(
            f"Repository checkout {dest} is missing required files after a "
            f"fresh clone: {', '.join(missing)}"
        )

    revision_probe = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "--short=12", "HEAD"],
        text=True,
        capture_output=True,
    )
    if revision_probe.returncode == 0:
        revision = revision_probe.stdout.strip()
        print(f"[h3-setup] node revision {dest.name}: {revision}")


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


def controlnet_aux_dependencies_importable() -> bool:
    """Check the dependency chains exercised by ControlNet Aux at startup."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lazy_loader, matplotlib, pyparsing, skimage",
        ],
        text=True,
        capture_output=True,
    )
    return probe.returncode == 0


def install_controlnet_aux_requirements(requirements: Path) -> None:
    """Resolve ControlNet Aux dependencies without drifting the CUDA ABI."""
    filtered, skipped = filter_pinned_requirements(
        requirements.read_text(encoding="utf-8").splitlines()
    )
    for package, requirement in skipped:
        print(
            f"[h3-setup] Keeping pinned {package}; "
            f"skipping ControlNet Aux entry: {requirement}",
            flush=True,
        )

    requirement_path: Path | None = None
    constraint_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as handle:
            handle.write("\n".join(filtered) + "\n")
            requirement_path = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as handle:
            handle.write("\n".join(ABI_CONSTRAINTS) + "\n")
            constraint_path = Path(handle.name)
        uv_pip(
            "-r",
            str(requirement_path),
            "--constraint",
            str(constraint_path),
        )
    finally:
        if requirement_path is not None:
            requirement_path.unlink(missing_ok=True)
        if constraint_path is not None:
            constraint_path.unlink(missing_ok=True)


def ensure_controlnet_aux_runtime_dependencies(requirements: Path) -> None:
    """Repair incomplete prior installs, then verify startup imports."""
    if controlnet_aux_dependencies_importable():
        return

    print(
        "[h3-setup] Repairing ControlNet Aux dependency graph",
        flush=True,
    )
    install_controlnet_aux_requirements(requirements)
    if not controlnet_aux_dependencies_importable():
        raise RuntimeError(
            "ControlNet Aux dependencies installed but matplotlib/scikit-image "
            "still fail to import"
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

    if not comfy_frontend_package_is_ready():
        print(
            "[h3-setup] Repairing incomplete ComfyUI frontend package",
            flush=True,
        )
        uv_pip(
            "--force-reinstall",
            "--no-deps",
            f"comfyui-frontend-package=={COMFY_FRONTEND_VERSION}",
        )
    if not comfy_frontend_package_is_ready():
        raise RuntimeError(
            "ComfyUI frontend package is still incomplete after reinstall"
        )

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

    sync_git_repo(
        COMFY_REPO,
        comfy,
        ref=COMFY_REF,
        required_paths=("main.py", "requirements.txt"),
        clean_untracked=False,
    )
    install_comfy_requirements(comfy)
    uv_pip(
        "gradio>=5,<7",
        "huggingface_hub>=0.34",
        "requests>=2.32",
        "websocket-client>=1.8",
        "aiohttp>=3.11,<4",
        "httpx>=0.27",
        "uvicorn>=0.30",
        f"wsproto=={WSPROTO_VERSION}",
        "Pillow>=10",
    )
    install_pinned_numpy_stack()


def sync_external_nodes(
    comfy: Path,
    *,
    install_requirements: bool,
) -> None:
    sol = comfy / "custom_nodes" / "ComfyUI_sol-attn_Blackwell"
    sync_git_repo(
        SOL_REPO,
        sol,
        ref=SOL_REF,
        required_paths=("__init__.py",),
    )
    if install_requirements and (sol / "requirements.txt").is_file():
        uv_pip("-r", str(sol / "requirements.txt"), no_deps=True)

    spectrum = comfy / "custom_nodes" / "ComfyUI-Spectrum-MiniMax-H3"
    sync_git_repo(
        SPECTRUM_REPO,
        spectrum,
        ref=SPECTRUM_REF,
        required_paths=("__init__.py",),
    )
    if install_requirements and (spectrum / "requirements.txt").is_file():
        uv_pip("-r", str(spectrum / "requirements.txt"), no_deps=True)

    larry_turbo = comfy / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
    sync_git_repo(
        LARRY_TURBO_REPO,
        larry_turbo,
        ref=LARRY_TURBO_REF,
        required_paths=("__init__.py",),
    )
    patch_larry_turbo_node(larry_turbo)
    if install_requirements and (larry_turbo / "requirements.txt").is_file():
        uv_pip("-r", str(larry_turbo / "requirements.txt"), no_deps=True)

    official_nodes = (
        (LTXVIDEO_REPO, LTXVIDEO_REF, "ComfyUI-LTXVideo"),
        (KJNODES_REPO, KJNODES_REF, "ComfyUI-KJNodes"),
        (CONTROLNET_AUX_REPO, CONTROLNET_AUX_REF, "comfyui_controlnet_aux"),
        (VIDEO_DEPTH_REPO, VIDEO_DEPTH_REF, "ComfyUI-Video-Depth-Anything"),
    )
    installed: dict[str, Path] = {}
    for repo, ref, directory_name in official_nodes:
        destination = comfy / "custom_nodes" / directory_name
        required_paths = ("__init__.py",)
        if directory_name == "ComfyUI-LTXVideo":
            required_paths += (
                "example_workflows/2.5/"
                "LTX-2.5_T2A_Single_Stage_Distilled.json",
            )
        sync_git_repo(
            repo,
            destination,
            ref=ref,
            required_paths=required_paths,
        )
        installed[directory_name] = destination
        requirements = destination / "requirements.txt"
        if requirements.is_file():
            if directory_name == "comfyui_controlnet_aux":
                if install_requirements:
                    install_controlnet_aux_requirements(requirements)
            else:
                uv_pip("-r", str(requirements), no_deps=True)
    ensure_controlnet_aux_runtime_dependencies(
        installed["comfyui_controlnet_aux"] / "requirements.txt"
    )
    # --no-deps deliberately protects the pinned CUDA/Torch stack, so install
    # the one dependency expressed only through transformers' `timm` extra.
    uv_pip("timm>=0.9.16,<2", no_deps=True)
    # ComfyUI-LTXVideo imports ``pad`` from the pyramid module; that compatibility
    # export was removed after Kornia 0.8.1.
    uv_pip(f"kornia=={KORNIA_VERSION}", no_deps=True)

    workflow_source = (
        installed["ComfyUI-LTXVideo"] / "example_workflows" / "2.5"
    )
    workflow_destination = (
        comfy / "user" / "default" / "workflows" / "LTX 2.5"
    )
    workflow_destination.mkdir(parents=True, exist_ok=True)
    workflows = list(workflow_source.glob("*.json"))
    if len(workflows) != 9:
        raise RuntimeError(
            f"Expected 9 official LTX 2.5 workflows, found {len(workflows)}"
        )
    for workflow in workflows:
        shutil.copy2(workflow, workflow_destination / workflow.name)
    print(
        f"[h3-setup] synced official LTX 2.5 workflows to {workflow_destination}",
        flush=True,
    )


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
        model_keys=PRELOAD_MODEL_KEYS,
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
    if not torch_stack_matches():
        install_pinned_torch_stack()
    if not numpy_stack_matches():
        install_pinned_numpy_stack()
    install_bundled_nodes(comfy)
    sync_model_inventory(install_dir, comfy)


if __name__ == "__main__":
    main()
