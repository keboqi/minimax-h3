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
UI_PACKAGE = ROOT / "h3_ui"
SHARED_MODELS = ROOT / "h3_models.py"
SHARED_REQUIREMENTS = ROOT / "h3_requirements.py"
NODE_PATCHES = ROOT / "h3_node_patches.py"
ATTENTION_HELPER = ROOT / "h3_attention.py"
PROMPT_REWRITER = ROOT / "h3_prompt_rewriter.py"
PROMPT_ENHANCER = ROOT / "prompt.txt"
PROMPT_MUSIC3 = ROOT / "prompt_music3.txt"
PROMPT_LTX25 = ROOT / "prompt_ltx25.txt"
ACCEL_DEST = COMFY / "custom_nodes" / "H3Acceleration" / "__init__.py"

LOCAL_UI = LOCAL / "gradio_app.py"
LOCAL_UI_PACKAGE = LOCAL / "h3_ui"
LOCAL_ACCEL = LOCAL / "custom_nodes" / "H3Acceleration" / "__init__.py"
LOCAL_SHARED_MODELS = LOCAL / "h3_models.py"
LOCAL_SHARED_REQUIREMENTS = LOCAL / "h3_requirements.py"
LOCAL_NODE_PATCHES = LOCAL / "h3_node_patches.py"
LOCAL_ATTENTION_HELPER = LOCAL / "h3_attention.py"
LOCAL_PROMPT_REWRITER = LOCAL / "h3_prompt_rewriter.py"
LOCAL_PROMPT_ENHANCER = LOCAL / "prompt.txt"
LOCAL_PROMPT_MUSIC3 = LOCAL / "prompt_music3.txt"
LOCAL_PROMPT_LTX25 = LOCAL / "prompt_ltx25.txt"

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
SOL_REF = "930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf"  # v0.6.2, SM86 support
SLA_REPO = "https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git"
SLA_REF = "aaec055cd642b3292df18e69824c012d345ebfe8"  # v1.4.4
SPECTRUM_REPO = "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"
SPECTRUM_REF = "beb32dd210ef9e95520453107f158241d4f2ecf3"
LARRY_TURBO_REPO = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
LARRY_TURBO_REF = "4274783a23afcfdbea3b4876cb79effd6c510785"  # v1.2.3+ audio/reference fixes
H3_LATENT_UPSCALER_NODE_REPO = (
    "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git"
)
H3_LATENT_UPSCALER_NODE_REF = "d7c01b9011f2e8439493f6c02c29995a27df276f"
LTXVIDEO_REPO = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
LTXVIDEO_REF = "15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d"
KJNODES_REPO = "https://github.com/kijai/ComfyUI-KJNodes.git"
KJNODES_REF = "e8e88f7c88e3f6205b122f5de87e69a09fbce5ac"
CONTROLNET_AUX_REPO = "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
CONTROLNET_AUX_REF = "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
VIDEO_DEPTH_REPO = "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
VIDEO_DEPTH_REF = "a0db08e63d1ea571601c45cde4aaee0acdd0544d"
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
HF_SECRET_NAME = os.getenv("H3_MODAL_HF_SECRET", "custom-secret")


def _shared_import_path() -> Path:
    return LOCAL if IS_LOCAL else Path(ROOT)


sys.path.insert(0, str(_shared_import_path()))
# Only build inputs may be imported while Modal constructs the image. Helpers
# mounted after run_function() must be imported lazily inside runtime functions.
from h3_requirements import (  # noqa: E402
    ABI_CONSTRAINTS,
    COMFY_FRONTEND_VERSION,
    COMFY_REF,
    KERNELS_VERSION,
    KORNIA_VERSION,
    NUMPY_VERSION,
    SCIPY_VERSION,
    TORCH_INDEX,
    TORCH_VERSION,
    TORCHAUDIO_VERSION,
    TENSORRT_PACKAGE,
    TORCHVISION_VERSION,
    WSPROTO_VERSION,
    SWIFTVR_REF,
    SWIFTVR_REPO,
    comfy_frontend_package_is_ready,
    filter_pinned_requirements,
    probe_comfy_frontend,
    probe_comfy_workflow,
    sync_ltx25_workflows,
)
from h3_node_patches import (  # noqa: E402
    TRT_VAE_NODE_REF,
    TRT_VAE_NODE_REPO,
    patch_larry_turbo_node,
    patch_trt_vae_node,
)

_BUILD_LOCAL_MOUNTS = (
    (LOCAL_SHARED_REQUIREMENTS, SHARED_REQUIREMENTS),
    (LOCAL_NODE_PATCHES, NODE_PATCHES),
)
_RUNTIME_LOCAL_MOUNTS = (
    (LOCAL_UI, UI),
    (LOCAL_ACCEL, ACCEL_DEST),
    (LOCAL_SHARED_MODELS, SHARED_MODELS),
    (LOCAL_ATTENTION_HELPER, ATTENTION_HELPER),
    (LOCAL_PROMPT_REWRITER, PROMPT_REWRITER),
    (LOCAL_PROMPT_ENHANCER, PROMPT_ENHANCER),
    (LOCAL_PROMPT_MUSIC3, PROMPT_MUSIC3),
    (LOCAL_PROMPT_LTX25, PROMPT_LTX25),
)
_BUILD_LOCAL_FILES = tuple(local for local, _ in _BUILD_LOCAL_MOUNTS)
_RUNTIME_LOCAL_FILES = tuple(local for local, _ in _RUNTIME_LOCAL_MOUNTS)
_REQUIRED_LOCAL_FILES = _BUILD_LOCAL_FILES + _RUNTIME_LOCAL_FILES
if IS_LOCAL:
    missing = [str(path) for path in _REQUIRED_LOCAL_FILES if not path.is_file()]
    if not LOCAL_UI_PACKAGE.is_dir():
        missing.append(str(LOCAL_UI_PACKAGE))
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

    _clone(COMFY_REPO, Path(COMFY), ref=COMFY_REF)
    _print_git_revision(Path(COMFY))

    swiftvr_dir = Path(ROOT) / "SwiftVR"
    _clone(SWIFTVR_REPO, swiftvr_dir, ref=SWIFTVR_REF)
    _print_git_revision(swiftvr_dir)

    sol_dir = Path(COMFY) / "custom_nodes" / "ComfyUI_sol-attn_Blackwell"
    _clone(SOL_REPO, sol_dir, ref=SOL_REF)
    _print_git_revision(sol_dir)

    sla_dir = Path(COMFY) / "custom_nodes" / "ComfyUI-PlagueKind-Nodes"
    _clone(SLA_REPO, sla_dir, ref=SLA_REF)
    _print_git_revision(sla_dir)

    spectrum_dir = (
        Path(COMFY) / "custom_nodes" / "ComfyUI-Spectrum-MiniMax-H3"
    )
    _clone(SPECTRUM_REPO, spectrum_dir, ref=SPECTRUM_REF)
    _print_git_revision(spectrum_dir)

    trt_vae_dir = Path(COMFY) / "custom_nodes" / "ComfyUI-H3VAE_TRT"
    _clone(TRT_VAE_NODE_REPO, trt_vae_dir, ref=TRT_VAE_NODE_REF)
    patch_trt_vae_node(trt_vae_dir)
    _print_git_revision(trt_vae_dir)

    larry_turbo_dir = (
        Path(COMFY) / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
    )
    _clone(LARRY_TURBO_REPO, larry_turbo_dir, ref=LARRY_TURBO_REF)
    patch_larry_turbo_node(larry_turbo_dir)
    _print_git_revision(larry_turbo_dir)

    official_nodes = (
        (
            H3_LATENT_UPSCALER_NODE_REPO,
            H3_LATENT_UPSCALER_NODE_REF,
            "Comfyui_Minimax_h3_latent_Upscaler",
        ),
        (LTXVIDEO_REPO, LTXVIDEO_REF, "ComfyUI-LTXVideo"),
        (KJNODES_REPO, KJNODES_REF, "ComfyUI-KJNodes"),
        (CONTROLNET_AUX_REPO, CONTROLNET_AUX_REF, "comfyui_controlnet_aux"),
        (VIDEO_DEPTH_REPO, VIDEO_DEPTH_REF, "ComfyUI-Video-Depth-Anything"),
    )
    installed_nodes: dict[str, Path] = {}
    for repo, ref, directory_name in official_nodes:
        destination = Path(COMFY) / "custom_nodes" / directory_name
        _clone(repo, destination, ref=ref)
        _print_git_revision(destination)
        installed_nodes[directory_name] = destination

    workflow_source = (
        installed_nodes["ComfyUI-LTXVideo"] / "example_workflows" / "2.5"
    )
    workflow_destination = (
        Path(COMFY) / "user" / "default" / "workflows" / "LTX 2.5"
    )
    workflows = sync_ltx25_workflows(workflow_source, workflow_destination)
    print(
        f"[modal-h3] baked {len(workflows)} official LTX 2.5 workflows",
        flush=True,
    )

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
    abi_constraints = Path("/tmp/h3-abi-constraints.txt")
    abi_constraints.write_text(
        "\n".join(ABI_CONSTRAINTS) + "\n",
        encoding="utf-8",
    )
    _run(
        "uv",
        "pip",
        "install",
        "--system",
        "--upgrade",
        "--constraint",
        abi_constraints,
        "-r",
        filtered_requirements,
    )
    if not comfy_frontend_package_is_ready():
        print(
            "[modal-h3] Reinstalling ComfyUI frontend as contained files",
            flush=True,
        )
        _run(
            "uv",
            "pip",
            "install",
            "--system",
            "--force-reinstall",
            "--no-deps",
            "--link-mode",
            "copy",
            f"comfyui-frontend-package=={COMFY_FRONTEND_VERSION}",
        )
    if not comfy_frontend_package_is_ready():
        raise RuntimeError(
            "ComfyUI frontend package is incomplete after reinstall"
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
        "transformers>=4.57.1",
        "diffusers>=0.36,<0.37",
        f"kernels=={KERNELS_VERSION}",
        "accelerate>=1.12",
        "peft>=0.18",
        "safetensors>=0.7",
        "einops>=0.8.2",
        "decord==0.6.0",
        "imageio>=2.37.2",
        "imageio-ffmpeg>=0.6",
        "requests>=2.32",
        "openai>=1.109,<3",
        "websocket-client>=1.8",
        "aiohttp>=3.11,<4",
        "httpx>=0.27",
        "uvicorn>=0.30",
        f"wsproto=={WSPROTO_VERSION}",
        "Pillow>=10",
        f"numpy=={NUMPY_VERSION}",
        f"scipy=={SCIPY_VERSION}",
        TENSORRT_PACKAGE,
        "setuptools<82",
        "--constraint",
        abi_constraints,
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
        "--force-reinstall",
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

    custom_requirements = [
        sol_dir / "requirements.txt",
        sla_dir / "requirements.txt",
        spectrum_dir / "requirements.txt",
        larry_turbo_dir / "requirements.txt",
    ]
    custom_requirements.extend(
        directory / "requirements.txt"
        for directory in installed_nodes.values()
    )
    for requirements in custom_requirements:
        if not requirements.is_file():
            continue
        if requirements.parent.name == "comfyui_controlnet_aux":
            filtered, skipped = filter_pinned_requirements(
                requirements.read_text(encoding="utf-8").splitlines()
            )
            for package, requirement in skipped:
                print(
                    f"[modal-h3] Keeping pinned {package}; skipping "
                    f"ControlNet Aux entry: {requirement}",
                    flush=True,
                )
            filtered_controlnet = Path(
                "/tmp/controlnet-aux-requirements-constrained.txt"
            )
            filtered_controlnet.write_text(
                "\n".join(filtered) + "\n",
                encoding="utf-8",
            )
            _run(
                "uv", "pip", "install", "--system", "--upgrade",
                "--constraint", abi_constraints,
                "-r", filtered_controlnet,
            )
            continue
        _run(
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "-r",
            requirements,
        )
    # The pinned ComfyUI-LTXVideo revision imports a compatibility symbol that
    # Kornia removed in 0.8.2. Its requirement is unbounded, so restore the
    # known-compatible version after all custom-node requirements are applied.
    _run(
        "uv", "pip", "install", "--system", "--upgrade", "--no-deps",
        f"kornia=={KORNIA_VERSION}",
    )
    kornia_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import kornia; "
                "from kornia.geometry.transform.pyramid import pad; "
                "print(kornia.__version__)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    if kornia_import.returncode != 0:
        raise RuntimeError(
            "LTX Kornia compatibility verification failed: "
            + (kornia_import.stderr or kornia_import.stdout).strip()
        )
    print(
        "[modal-h3] LTX Kornia compatibility OK: "
        + kornia_import.stdout.strip(),
        flush=True,
    )
    _run(
        "uv", "pip", "install", "--system", "--upgrade", "--no-deps",
        "timm>=0.9.16,<2",
    )
    controlnet_import = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lazy_loader, matplotlib, pyparsing, skimage",
        ],
        text=True,
        capture_output=True,
    )
    if controlnet_import.returncode != 0:
        raise RuntimeError(
            "ControlNet Aux dependency verification failed: "
            + (controlnet_import.stderr or controlnet_import.stdout).strip()
        )
    # Custom-node requirements must not drift the pinned Torch/NumPy ABI.
    _run(
        "uv", "pip", "install", "--system", "--upgrade",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        "--index-url", TORCH_INDEX,
    )
    _run(
        "uv", "pip", "install", "--system", "--upgrade",
        f"numpy=={NUMPY_VERSION}",
        f"scipy=={SCIPY_VERSION}",
    )
    trt_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tensorrt as trt; "
                "assert hasattr(trt, 'Builder'); "
                "print(trt.__version__)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    if trt_import.returncode != 0:
        raise RuntimeError(
            "TensorRT builder failed fresh-process import: "
            + (trt_import.stderr or trt_import.stdout).strip()
        )
    print(
        "[modal-h3] TensorRT builder fresh-import OK: "
        + trt_import.stdout.strip(),
        flush=True,
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
    image = image.add_local_dir(
        LOCAL_UI_PACKAGE,
        remote_path=UI_PACKAGE.as_posix(),
        copy=False,
    )

if IS_LOCAL:
    image = image.add_local_dir(LOCAL / "h3_app", remote_path=(ROOT / "h3_app").as_posix(), copy=False)

volume = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App(APP, image=image)
hf_secret = modal.Secret.from_name(
    HF_SECRET_NAME,
    required_keys=["HF_TOKEN"],
)


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
    from h3_models import (
        PRELOAD_MODEL_KEYS,
        sync_models,
        validate_config_files,
        write_json_atomic,
    )

    layout()

    config = sync_models(
        root=Path(MODELS),
        manifest_path=Path(MANIFEST),
        token=os.getenv("HF_TOKEN") or None,
        log_prefix="[modal-h3]",
        model_keys=PRELOAD_MODEL_KEYS,
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
            "SERVER_DENSE_ATTENTION_BACKEND": "comfy-kitchen",
            # Modal already provides the public endpoint. Avoid asking Gradio
            # to create a second, unauthenticated share tunnel at startup.
            "GRADIO_SHARE": "false",
            "SERVER_MEMORY_PROFILE": "dynamic",
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


def wait_for_comfy_frontend(
    url: str,
    process: subprocess.Popen,
    *,
    timeout: int,
    label: str,
) -> None:
    print(f"[modal-h3] Checking {label}: {url}", flush=True)
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{label} exited before its frontend became ready: "
                f"{process.returncode}"
            )
        try:
            count = probe_comfy_frontend(url)
            workflow_size = probe_comfy_workflow(url)
            print(
                f"[modal-h3] {label} served {count} frontend assets and "
                f"a {workflow_size}-byte nested workflow",
                flush=True,
            )
            return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"{label} frontend did not start: {last_error}")


@app.function(
    timeout=21600,
    volumes={DATA.as_posix(): volume},
    secrets=[hf_secret],
    max_containers=1,
)
def provision_models():
    return provision()


@app.function(
    gpu=GPU,
    timeout=86400,
    startup_timeout=3600,
    volumes={DATA.as_posix(): volume},
    secrets=[hf_secret],
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
    print("[modal-h3] Starting MiniMax H3 service", flush=True)
    print(
        "[modal-h3] Hugging Face token injected: "
        + ("yes" if os.getenv("HF_TOKEN") else "no"),
        flush=True,
    )
    provision()

    # User workflow files live in the image filesystem rather than the model
    # volume. Re-sync them on every cold start so a reused image layer can never
    # expose newer UI mappings without their corresponding JSON templates.
    workflow_source = (
        Path(COMFY) / "custom_nodes" / "ComfyUI-LTXVideo"
        / "example_workflows" / "2.5"
    )
    workflow_destination = (
        Path(COMFY) / "user" / "default" / "workflows" / "LTX 2.5"
    )
    workflows = sync_ltx25_workflows(workflow_source, workflow_destination)
    print(
        f"[modal-h3] runtime-synced {len(workflows)} official LTX 2.5 "
        f"workflows to {workflow_destination}",
        flush=True,
    )

    env = service_env()
    env["GRADIO_SERVER_NAME"] = "0.0.0.0"
    env["GRADIO_SERVER_PORT"] = str(UI_PORT)

    comfy_args = [
        "python",
        "-u",
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(COMFY_PORT),
        "--use-ck-attention",
    ]
    print("[modal-h3] Dense/fallback attention: Comfy Kitchen", flush=True)
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
    try:
        wait_for_comfy_frontend(
            f"http://127.0.0.1:{UI_PORT}/comfyui/",
            gradio_process,
            timeout=30,
            label="ComfyUI proxy",
        )
    except Exception as exc:
        print(
            "[modal-h3] WARNING: Main UI is running, but "
            f"/comfyui asset validation failed: {exc}",
            flush=True,
        )


@app.local_entrypoint()
def main():
    print(json.dumps(provision_models.remote(), indent=2))
