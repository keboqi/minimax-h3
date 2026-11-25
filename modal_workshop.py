#!/usr/bin/env python3
"""Minimal CPU-only Modal deployment for Prism Workshop (project review and backup).

Uses a bare Debian slim Python 3.12 image with only the core Python packages
required to serve the Workshop UI, Media library, Models manager, and project
export/import/transfer routes — skipping CUDA, ComfyUI, FFmpeg, Playwright,
and URL downloaders so image builds and redeploys stay fast.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path, PurePosixPath

import modal


IS_LOCAL = modal.is_local()
LOCAL = Path(__file__).resolve().parent

ROOT = PurePosixPath("/opt/h3")
COMFY = ROOT / "ComfyUI"
DATA = PurePosixPath("/data")
MODELS = DATA / "models"
INPUT = DATA / "input"
OUTPUT = DATA / "output"
LOGS = DATA / "logs"
CONFIG = DATA / "h3_models.json"
MANIFEST = DATA / "h3_model_manifest.json"

COMFY_PORT = 8188
UI_PORT = 7860

APP = os.getenv("H3_MODAL_WORKSHOP_APP_NAME", "minimax-h3-workshop")
VOL = os.getenv("H3_MODAL_VOLUME", "minimax-h3-data")
LABEL = os.getenv("H3_MODAL_WORKSHOP_LABEL", "workshop-cpu")
SCALEDOWN_WINDOW = int(os.getenv("H3_MODAL_SCALEDOWN_WINDOW", "300"))
PROXY_AUTH = os.getenv("H3_MODAL_PROXY_AUTH", "0") != "0"
HF_SECRET_NAME = os.getenv("H3_MODAL_HF_SECRET", "custom-secret").strip()

_BYTECODE_IGNORE = ["**/__pycache__", "**/*.pyc"]
_LOCAL_FILES = (
    "gradio_app.py",
    "h3_models.py",
    "h3_gpu.py",
    "h3_workshop.py",
    "h3_prompt_rewriter.py",
    "h3_requirements.py",
    "h3_workshop_requirements.py",
)
_LOCAL_DIRS = ("h3_app", "h3_ui", "prompts")

sys.path.insert(0, str(LOCAL if IS_LOCAL else Path(ROOT)))

if IS_LOCAL:
    missing = [str(LOCAL / name) for name in _LOCAL_FILES if not (LOCAL / name).is_file()]
    missing.extend(str(LOCAL / name) for name in _LOCAL_DIRS if not (LOCAL / name).is_dir())
    if missing:
        raise RuntimeError(
            "Keep these files beside modal_workshop.py: " + ", ".join(missing)
        )

# Minimal CPU base image: Python 3.12 slim + core Workshop Python packages only.
image = modal.Image.debian_slim(python_version="3.12")
if IS_LOCAL:
    from h3_workshop_requirements import WORKSHOP_MINIMAL_REQUIREMENTS

    image = image.uv_pip_install(*WORKSHOP_MINIMAL_REQUIREMENTS)
    for filename in _LOCAL_FILES:
        image = image.add_local_file(
            LOCAL / filename,
            remote_path=(ROOT / filename).as_posix(),
            copy=False,
        )
    for dirname in _LOCAL_DIRS:
        image = image.add_local_dir(
            LOCAL / dirname,
            remote_path=(ROOT / dirname).as_posix(),
            copy=False,
            ignore=_BYTECODE_IGNORE,
        )

volume = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App(APP, image=image)
secrets = [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []


def layout() -> None:
    """Create shared volume directories and ComfyUI path aliases for saved projects."""
    for path in (MODELS, INPUT, OUTPUT, LOGS):
        Path(path).mkdir(parents=True, exist_ok=True)

    for destination, source in (
        (COMFY / "models", MODELS),
        (COMFY / "input", INPUT),
        (COMFY / "output", OUTPUT),
    ):
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


def prepare_runtime_models() -> None:
    """Ensure shared volume layout and model config exist before serving Workshop."""
    from h3_models import build_model_config, write_json_atomic

    layout()
    write_json_atomic(Path(CONFIG), build_model_config(Path(MANIFEST).name))
    volume.commit()


def cpu_studio_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "COMFY_DIR": COMFY.as_posix(),
            "COMFY_URL": f"http://127.0.0.1:{COMFY_PORT}",
            "MODELS_CONFIG": CONFIG.as_posix(),
            "GRADIO_OUTPUT_DIR": OUTPUT.as_posix(),
            "H3_WORKSPACE_DIR": (DATA / "h3-workspace").as_posix(),
            "H3_DRAMA3_SGLANG_MODELS": (MODELS / "drama3_serving").as_posix(),
            "H3_MODAL_COMMIT_VOLUME": VOL,
            "ENABLE_CLOUDFLARE_TUNNEL": "false",
            "GRADIO_SHARE": "false",
            "GRADIO_ANALYTICS_ENABLED": "False",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/tmp/hf",
            "XDG_CACHE_HOME": "/tmp/cache",
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": str(UI_PORT),
            "H3_CPU_ONLY": "1",
        }
    )
    return env


@app.function(
    image=image,
    timeout=21600,
    volumes={DATA.as_posix(): volume},
    secrets=secrets,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(
    UI_PORT,
    startup_timeout=180,
    label=LABEL,
    requires_proxy_auth=PROXY_AUTH,
)
def workshop():
    from h3_workshop import serve

    prepare_runtime_models()
    os.environ.update(cpu_studio_env())
    serve(Path(DATA), volume, port=UI_PORT, script_dir=Path(ROOT))
