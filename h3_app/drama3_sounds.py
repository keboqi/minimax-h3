"""Named Drama 3 sound presets: a sound effect or music clip plus its description and duration.

Presets live beside the workspace database, outside the served output roots, so
the media library never lists them and the browser cannot fetch them directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm"})
MAX_NAME_LENGTH = 64
MAX_PRESETS = 200
_LOCK = threading.Lock()


@dataclass(frozen=True)
class SoundPreset:
    name: str
    audio: Path
    description: str = ""
    duration: int = 5


def sounds_root(runtime: RuntimeConfig) -> Path:
    workspace = os.getenv("H3_WORKSPACE_DIR")
    outputs_dir = (
        getattr(runtime, "outputs_dir", None)
        or getattr(runtime, "output_dir", None)
        or getattr(runtime, "script_dir", Path("."))
    )
    base = Path(workspace) if workspace else Path(outputs_dir) / ".h3-workspace"
    return base.resolve() / "drama3_sounds"


def normalize_sound_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        raise H3Error("Name the sound before saving it.")
    if len(cleaned) > MAX_NAME_LENGTH or any(ord(char) < 32 for char in cleaned):
        raise H3Error(f"Sound names are 1–{MAX_NAME_LENGTH} printable characters.")
    return cleaned


def _folder(root: Path, name: str) -> Path:
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:32] or "sound"
    return root / f"{slug}-{digest}"


def _read(folder: Path) -> SoundPreset | None:
    try:
        data = json.loads((folder / "sound.json").read_text(encoding="utf-8"))
        audio = folder / str(data["audio"])
        if audio.parent != folder or not audio.is_file():
            return None
        duration = max(1, min(int(data.get("duration", 5)), 47))
        return SoundPreset(
            str(data["name"]),
            audio,
            str(data.get("description", "")),
            duration,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_sounds(runtime: RuntimeConfig) -> list[SoundPreset]:
    root = sounds_root(runtime)
    if not root.is_dir():
        return []
    presets = (_read(folder) for folder in root.iterdir() if folder.is_dir())
    return sorted((preset for preset in presets if preset), key=lambda preset: preset.name.casefold())


def sound_names(runtime: RuntimeConfig) -> list[str]:
    return [preset.name for preset in list_sounds(runtime)]


def load_sound(name: str, runtime: RuntimeConfig) -> SoundPreset:
    preset = _read(_folder(sounds_root(runtime), normalize_sound_name(name)))
    if preset is None:
        raise H3Error(f'The saved sound "{name}" is no longer available.')
    return preset


def find_sound(name: str, runtime: RuntimeConfig) -> SoundPreset | None:
    try:
        cleaned = normalize_sound_name(name)
    except H3Error:
        return None
    return _read(_folder(sounds_root(runtime), cleaned))


def save_sound(
    name: str,
    audio: str | Path,
    description: str,
    duration_or_runtime: int | float | RuntimeConfig = 5,
    runtime: RuntimeConfig | int | float | None = None,
) -> SoundPreset:
    """Create or replace a sound preset from an uploaded or generated clip."""
    if isinstance(duration_or_runtime, (int, float, str)):
        duration = duration_or_runtime
        resolved_runtime = runtime
    else:
        resolved_runtime = duration_or_runtime
        duration = runtime if isinstance(runtime, (int, float, str)) else 5
    name = normalize_sound_name(name)
    description = str(description or "").strip()
    source = Path(str(audio or ""))
    if not str(audio or "") or not source.is_file():
        raise H3Error("Upload or generate a sound clip before saving the sound.")
    suffix = source.suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise H3Error("Save a WAV, FLAC, MP3, OGG, M4A, AAC, Opus or WebM clip.")
    clamped_duration = max(1, min(int(round(float(duration or 5))), 47))
    root = sounds_root(resolved_runtime)
    folder = _folder(root, name)
    with _LOCK:
        if not folder.is_dir() and len(list_sounds(resolved_runtime)) >= MAX_PRESETS:
            raise H3Error(f"Delete a saved sound first; at most {MAX_PRESETS} are kept.")
        staging = root / f".saving-{folder.name}-{time.time_ns()}"
        staging.mkdir(parents=True)
        try:
            shutil.copyfile(source, staging / f"sound{suffix}")
            (staging / "sound.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "audio": f"sound{suffix}",
                        "description": description,
                        "duration": clamped_duration,
                        "saved": time.time(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if folder.exists():
                shutil.rmtree(folder)
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return load_sound(name, resolved_runtime)


def delete_sound(name: str, runtime: RuntimeConfig) -> None:
    folder = _folder(sounds_root(runtime), normalize_sound_name(name))
    with _LOCK:
        if not folder.is_dir():
            raise H3Error(f'The saved sound "{name}" does not exist.')
        shutil.rmtree(folder)
