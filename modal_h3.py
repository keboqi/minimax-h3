#!/usr/bin/env python3
"""Self-contained MiniMax H3 Modal deployment."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import modal


IS_LOCAL = modal.is_local()
LOCAL = Path(__file__).resolve().parent

ROOT = PurePosixPath("/opt/h3")
COMFY = ROOT / "ComfyUI"
UI = ROOT / "gradio_app.py"
SHARED_MODELS = ROOT / "h3_models.py"
SHARED_REQUIREMENTS = ROOT / "h3_requirements.py"
ATTENTION_HELPER = ROOT / "h3_attention.py"
ACCEL_DEST = COMFY / "custom_nodes" / "H3Acceleration" / "__init__.py"

LOCAL_UI = LOCAL / "gradio_app.py"
LOCAL_ACCEL = LOCAL / "custom_nodes" / "H3Acceleration" / "__init__.py"
LOCAL_SHARED_MODELS = LOCAL / "h3_models.py"
LOCAL_SHARED_REQUIREMENTS = LOCAL / "h3_requirements.py"
LOCAL_ATTENTION_HELPER = LOCAL / "h3_attention.py"

DATA = PurePosixPath("/data")
MODELS = DATA / "models"
INPUT = DATA / "input"
OUTPUT = DATA / "output"
LOGS = DATA / "logs"
CONFIG = DATA / "h3_models.json"
MANIFEST = DATA / "h3_model_manifest.json"

COMFY_PORT = 8188
UI_PORT = 7860

COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
SOL_REPO = "https://github.com/Saganaki22/ComfyUI-sol-attn.git"
SOL_REF = "bd28909e6a152ba5db4e3590d2d9df1c249943f2"  # v0.6.0
SPECTRUM_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"
SPECTRUM_REF = "6a37360e3e785d2b9b5ad58190af380a8de8ec1a"  # v0.2.2
LARRY_TURBO_REPO = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
LARRY_TURBO_REF = "55fee864dd7b2976b1c4ce3c3d5f7968f181409f"
SAGE_WHEEL_URL = "https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
SAGE_WHEEL_NAME = "sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
APP = os.getenv("H3_MODAL_APP_NAME", "minimax-h3")
VOL = os.getenv("H3_MODAL_VOLUME", "minimax-h3-data")
GPU = "RTX-PRO-6000"
MIN_CONTAINERS = int(os.getenv("H3_MODAL_MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(
    os.getenv("H3_MODAL_SCALEDOWN_WINDOW", "1200")
)
PROXY_AUTH = os.getenv("H3_MODAL_PROXY_AUTH", "0") != "0"


def _shared_import_path() -> Path:
    return LOCAL if IS_LOCAL else Path(ROOT)


sys.path.insert(0, str(_shared_import_path()))
# Only build inputs may be imported while Modal constructs the image. Helpers
# mounted after run_function() must be imported lazily inside runtime functions.
from h3_requirements import (  # noqa: E402
    NUMPY_VERSION,
    SCIPY_VERSION,
    TORCH_INDEX,
    TORCH_VERSION,
    TORCHAUDIO_VERSION,
    TORCHVISION_VERSION,
    filter_pinned_requirements,
)


_BUILD_LOCAL_MOUNTS = (
    (LOCAL_SHARED_REQUIREMENTS, SHARED_REQUIREMENTS),
)
_RUNTIME_LOCAL_MOUNTS = (
    (LOCAL_UI, UI),
    (LOCAL_ACCEL, ACCEL_DEST),
    (LOCAL_SHARED_MODELS, SHARED_MODELS),
    (LOCAL_ATTENTION_HELPER, ATTENTION_HELPER),
)
_BUILD_LOCAL_FILES = tuple(local for local, _ in _BUILD_LOCAL_MOUNTS)
_RUNTIME_LOCAL_FILES = tuple(local for local, _ in _RUNTIME_LOCAL_MOUNTS)
_REQUIRED_LOCAL_FILES = _BUILD_LOCAL_FILES + _RUNTIME_LOCAL_FILES
if IS_LOCAL:
    missing = [str(path) for path in _REQUIRED_LOCAL_FILES if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Keep these files beside modal_h3.py: " + ", ".join(missing)
        )


def _revision() -> str:
    if not IS_LOCAL:
        return "remote-import"

    digest = hashlib.sha256()
    # Runtime-mounted files must not invalidate the expensive image build.
    for path in _BUILD_LOCAL_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


REVISION = _revision()


def _run(
    *args,
    cwd: Path | PurePosixPath | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        [str(arg) for arg in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def _clone(url: str, destination: Path, *, ref: str | None = None) -> None:
    if ref is None:
        _run("git", "clone", "--depth", "1", url, destination)
        return

    _run("git", "init", destination)
    _run("git", "-C", destination, "remote", "add", "origin", url)
    _run("git", "-C", destination, "fetch", "--depth", "1", "origin", ref)
    _run("git", "-C", destination, "checkout", "--detach", "FETCH_HEAD")



def _print_git_revision(directory: Path) -> None:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "--short=12", "HEAD"],
            text=True,
        ).strip()
        print(
            f"[modal-h3] node revision {directory.name}: {revision}",
            flush=True,
        )
    except Exception:
        pass


def build(revision: str) -> None:
    """Build-time clone/install step baked into the Modal image."""
    print("[modal-h3] runtime", revision, flush=True)

    if not Path(SHARED_REQUIREMENTS).is_file():
        raise RuntimeError(
            f"Missing shared requirements module: {SHARED_REQUIREMENTS}"
        )

    _clone(COMFY_REPO, Path(COMFY))

    sol_dir = Path(COMFY) / "custom_nodes" / "ComfyUI_sol-attn_Blackwell"
    _clone(SOL_REPO, sol_dir, ref=SOL_REF)
    _print_git_revision(sol_dir)

    spectrum_dir = (
        Path(COMFY) / "custom_nodes" / "ComfyUI-Spectrum-MiniMax-H3"
    )
    _clone(SPECTRUM_REPO, spectrum_dir, ref=SPECTRUM_REF)
    _print_git_revision(spectrum_dir)

    larry_turbo_dir = (
        Path(COMFY) / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
    )
    _clone(LARRY_TURBO_REPO, larry_turbo_dir, ref=LARRY_TURBO_REF)
    _print_git_revision(larry_turbo_dir)

    # ComfyUI currently lists torch/torchvision/torchaudio unpinned.
    # Filter those entries so image build cannot replace the pinned cu130 ABI.
    comfy_requirements = Path(COMFY) / "requirements.txt"
    filtered_requirements = Path("/tmp/comfy-requirements-no-torch.txt")
    filtered_lines, skipped = filter_pinned_requirements(
        comfy_requirements.read_text(encoding="utf-8").splitlines()
    )
    for package, requirement in skipped:
        print(
            f"[modal-h3] Keeping pinned {package}; "
            f"skipping ComfyUI entry: {requirement}",
            flush=True,
        )

    filtered_requirements.write_text(
        "\n".join(filtered_lines) + "\n",
        encoding="utf-8",
    )
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        "-r",
        filtered_requirements,
    )

    # Reassert the ABI trio after all Comfy requirements.
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        "--index-url",
        TORCH_INDEX,
    )
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        f"numpy=={NUMPY_VERSION}",
        f"scipy=={SCIPY_VERSION}",
    )
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        "gradio>=5,<7",
        "huggingface_hub>=0.34",
        "requests>=2.32",
        "websocket-client>=1.8",
        "aiohttp>=3.11,<4",
        "httpx>=0.27",
        "uvicorn>=0.30",
        "Pillow>=10",
        f"numpy=={NUMPY_VERSION}",
        f"scipy=={SCIPY_VERSION}",
        "setuptools<82",
    )

    import torch as _torch
    if not str(_torch.__version__).startswith(TORCH_VERSION + "+cu130"):
        raise RuntimeError(
            f"Unexpected Torch ABI before Sage install: "
            f"{_torch.__version__}; expected {TORCH_VERSION}+cu130"
        )

    print(
        f"[modal-h3] Installing prebuilt SageAttention wheel: "
        f"{SAGE_WHEEL_NAME}",
        flush=True,
    )
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        "--no-deps",
        SAGE_WHEEL_URL,
    )
    sage_import = subprocess.run(
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
    if sage_import.returncode != 0:
        raise RuntimeError(
            "Prebuilt SageAttention wheel failed fresh-process import: "
            + (sage_import.stderr or sage_import.stdout).strip()
        )
    print(
        "[modal-h3] SageAttention fresh-import OK: "
        + sage_import.stdout.strip().replace("\\n", " | "),
        flush=True,
    )

    sol_requirements = sol_dir / "requirements.txt"
    if sol_requirements.is_file():
        _run(
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "-r",
            sol_requirements,
        )


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsndfile1",
        "pkg-config",
    )
    .uv_pip_install("uv>=0.8", "setuptools<82", "wheel")
    .uv_pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index_url=TORCH_INDEX,
    )
)

if IS_LOCAL:
    for local_path, remote_path in _BUILD_LOCAL_MOUNTS:
        image = image.add_local_file(
            local_path,
            remote_path=remote_path.as_posix(),
            copy=True,
        )

image = image.run_function(
    build,
    timeout=7200,
    kwargs={"revision": REVISION},
)

if IS_LOCAL:
    # Runtime-only changes are mounted when containers start. Keeping these
    # after all build steps prevents normal code iteration from rebuilding
    # ComfyUI and its dependency stack.
    for local_path, remote_path in _RUNTIME_LOCAL_MOUNTS:
        image = image.add_local_file(
            local_path,
            remote_path=remote_path.as_posix(),
            copy=False,
        )

volume = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App(APP, image=image)


def layout() -> None:
    for path in (MODELS, INPUT, OUTPUT, LOGS):
        Path(path).mkdir(parents=True, exist_ok=True)

    mappings = (
        (COMFY / "models", MODELS),
        (COMFY / "input", INPUT),
        (COMFY / "output", OUTPUT),
    )
    for destination, source in mappings:
        dest = Path(destination)
        src = Path(source)

        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src, target_is_directory=True)


def _provision_unlocked() -> dict:
    from h3_models import sync_models, validate_config_files, write_json_atomic

    layout()

    config = sync_models(
        root=Path(MODELS),
        manifest_path=Path(MANIFEST),
        token=os.getenv("HF_TOKEN") or None,
        log_prefix="[modal-h3]",
    )

    missing = validate_config_files(Path(MODELS), config)
    if missing:
        raise RuntimeError(
            "Provisioning incomplete: " + ", ".join(missing)
        )

    write_json_atomic(Path(CONFIG), config)
    volume.commit()

    print(
        "[modal-h3] Model provisioning and remote version check complete",
        flush=True,
    )
    return config


def provision() -> dict:
    """Serialize writes to the shared persistent model volume."""
    layout()
    lock_path = Path(DATA / ".model-provision.lock")
    lock_path.touch(exist_ok=True)

    import fcntl

    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _provision_unlocked()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def service_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "COMFY_DIR": COMFY.as_posix(),
            "COMFY_URL": f"http://127.0.0.1:{COMFY_PORT}",
            "MODELS_CONFIG": CONFIG.as_posix(),
            "GRADIO_OUTPUT_DIR": OUTPUT.as_posix(),
            "SERVER_ATTENTION_BACKEND": "sol",
            "SERVER_DENSE_ATTENTION_BACKEND": "pytorch",
            "SERVER_MEMORY_PROFILE": "gpu-only",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/tmp/hf",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    return env


def wait_for_service(
    url: str,
    process: subprocess.Popen,
    *,
    timeout: int,
    label: str,
) -> None:
    print(f"[modal-h3] Waiting for {label}: {url}", flush=True)
    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{label} exited before becoming ready: "
                f"{process.returncode}"
            )
        try:
            with urlopen(url, timeout=5) as response:
                if response.status < 500:
                    print(f"[modal-h3] {label} is ready", flush=True)
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(f"{label} did not start: {last_error}")


@app.function(
    timeout=21600,
    volumes={DATA.as_posix(): volume},
    max_containers=1,
)
def provision_models():
    return provision()


@app.function(
    gpu=GPU,
    timeout=86400,
    startup_timeout=3600,
    volumes={DATA.as_posix(): volume},
    min_containers=MIN_CONTAINERS,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(
    UI_PORT,
    startup_timeout=3600,
    label="h3",
    requires_proxy_auth=PROXY_AUTH,
)
def serve():
    from h3_attention import probe_sageattention

    print("[modal-h3] Starting MiniMax H3 service", flush=True)
    provision()

    env = service_env()
    env["GRADIO_SERVER_NAME"] = "0.0.0.0"
    env["GRADIO_SERVER_PORT"] = str(UI_PORT)

    sage = probe_sageattention()
    comfy_args = [
        "python",
        "-u",
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(COMFY_PORT),
        "--gpu-only",
    ]
    if sage.available:
        env["SERVER_DENSE_ATTENTION_BACKEND"] = "sage"
        comfy_args.append("--use-sage-attention")
        print(
            f"[modal-h3] Dense/fallback attention: SageAttention "
            f"({sage.message})",
            flush=True,
        )
    else:
        env["SERVER_DENSE_ATTENTION_BACKEND"] = "pytorch"
        print(
            f"[modal-h3] Dense/fallback attention: PyTorch "
            f"({sage.message})",
            flush=True,
        )
    comfy_args += ["--enable-cors-header", "*"]

    print("[modal-h3] Launching ComfyUI", flush=True)
    comfy_process = subprocess.Popen(
        comfy_args,
        cwd=COMFY.as_posix(),
        env=env,
    )
    wait_for_service(
        f"http://127.0.0.1:{COMFY_PORT}/system_stats",
        comfy_process,
        timeout=20 * 60,
        label="ComfyUI",
    )

    print("[modal-h3] Launching Gradio", flush=True)
    gradio_process = subprocess.Popen(
        ["python", "-u", UI.as_posix()],
        env=env,
    )
    wait_for_service(
        f"http://127.0.0.1:{UI_PORT}/",
        gradio_process,
        timeout=10 * 60,
        label="Gradio",
    )

    print(
        f"[modal-h3] Public Gradio server is listening on port {UI_PORT}",
        flush=True,
    )


@app.local_entrypoint()
def main():
    print(json.dumps(provision_models.remote(), indent=2))
