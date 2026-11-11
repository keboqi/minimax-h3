"""Lightweight GPU deployment policy, usable before Torch is installed.

Launchers detect the assigned GPU and export H3_GPU_PROFILE to setup,
background downloaders, ComfyUI and the UI. Imports never initialize CUDA.
"""
from __future__ import annotations

import os
import re
import subprocess


def gpu_profile() -> str:
    """Policy selected by the launcher; no hardware queries during imports."""
    return "a100" if os.environ.get("H3_GPU_PROFILE", "").strip().lower() == "a100" else "default"


def is_a100() -> bool:
    return gpu_profile() == "a100"


def default_text_encoder() -> str:
    return "INT8 ConvRot" if is_a100() else "NVFP4 / AWQ"


def default_flux3_model() -> str:
    return "INT8 ConvRot (lower VRAM)" if is_a100() else "NVFP4 (Blackwell)"


def default_drama3_checkpoint() -> str:
    return "int4" if is_a100() else "fp8"


def attention_choices() -> list[str]:
    # The pinned Sol kernels exclude SM80; the Sage wheel targets Blackwell.
    # Veda's FP8 predictor has not been validated on this profile.
    return (["PyTorch", "SLA", "Auto"] if is_a100() else
            ["Sage 2", "Kitchen", "SLA", "Veda", "Sol-Attn", "Auto"])


def normalize_attention_mode(mode: str) -> str:
    """Keep API aliases and browser choices consistent with the dense backend."""
    label = {
        "dense": "Kitchen", "kitchen": "Kitchen", "comfy-kitchen": "Kitchen",
        "sage": "Sage 2", "sage 2": "Sage 2", "sage2": "Sage 2",
        "sla": "SLA", "sla attention": "SLA", "sparse-linear": "SLA",
        "sol-attn": "Sol-Attn", "sol": "Sol-Attn", "sparse": "Sol-Attn",
        "auto": "Auto", "veda": "Veda", "pytorch": "PyTorch",
    }.get(str(mode).strip().lower(), str(mode).strip())
    if label in {"Kitchen", "PyTorch"}:
        return "PyTorch" if is_a100() else "Kitchen"
    return label


def validate_attention_mode(mode: str) -> str:
    label = normalize_attention_mode(mode)
    if is_a100() and label not in attention_choices():
        raise ValueError("A100 supports SLA, PyTorch or Auto attention. Select one of those modes.")
    return label


def detect_profile(*, colab: bool = False) -> str:
    """Validate Colab assignments using the same detector as Terminal launches."""
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=15, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        rows = []
    if len(rows) == 1:
        name, _, memory = rows[0].partition(",")
        if re.search(r"\bA100\b", name.upper()):
            try:
                enough_memory = int(memory.strip().split()[0]) >= 75000
            except (ValueError, IndexError):
                enough_memory = False
            if enough_memory or not colab:
                return "a100"
        elif "RTX PRO 6000" in name.upper():
            return "default"
    if colab:
        raise ValueError("The Studio needs RTX PRO 6000 or A100 80 GB. Select a supported GPU or use Workshop.")
    return "default"


def verify_a100_driver() -> None:
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=15,
        ).strip().splitlines()[0]
        version = tuple(int(part) for part in driver.split("."))
        if (version + (0, 0, 0))[:3] < (580, 65, 6):
            raise SystemExit("A100 needs NVIDIA driver 580.65.06 or newer for this CUDA 13 runtime. Reconnect to a newer Colab runtime.")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise SystemExit(f"Cannot verify A100 CUDA 13 driver compatibility: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab", action="store_true", help="Require a supported single-GPU Colab assignment")
    args = parser.parse_args()
    try:
        profile = detect_profile(colab=args.colab)
        if profile == "a100":
            verify_a100_driver()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(profile)
