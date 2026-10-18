"""User-supplied MOSS-Image edit LoRAs in ComfyUI's top-level loras folder.

No edit LoRA is published, so the UI lists what the user dropped in or uploaded.
Managed LoRAs (Turbo, LightX2V, ...) are hidden; they are not edit LoRAs.
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from h3_app.errors import H3Error
from h3_app.staging import file_content_sha256
from h3_models import MODEL_SPECS


def loras_dir(runtime) -> Path:
    return runtime.comfy_dir / "models" / "loras"


def _managed_names() -> set[str]:
    return {spec.local_name for spec in MODEL_SPECS.values() if spec.folder == "loras"}


def list_edit_loras(runtime) -> list[str]:
    managed = _managed_names()
    directory = loras_dir(runtime)
    if not directory.is_dir():
        return []
    return sorted((path.name for path in directory.glob("*.safetensors")
                   if path.is_file() and path.name not in managed), key=str.lower)


def store_edit_lora(path, runtime) -> str:
    """Copy an uploaded LoRA in under its own name; an identical file is reused."""
    source = Path(path)
    if source.suffix.lower() != ".safetensors" or not source.is_file():
        raise H3Error("Upload the edit LoRA as a .safetensors file.")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "moss_image_edit"
    directory = loras_dir(runtime)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stem}.safetensors"
    if target.name in _managed_names():
        raise H3Error("That file name belongs to a managed LoRA. Rename the edit LoRA and upload it again.")
    if target.is_file():
        digest = file_content_sha256(source)
        if file_content_sha256(target) == digest:
            return target.name
        target = directory / f"{stem}_{digest[:8]}.safetensors"
        if target.is_file():
            return target.name
    partial = directory / f".{target.name}.{uuid.uuid4().hex}.partial"
    try:
        shutil.copyfile(source, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target.name
