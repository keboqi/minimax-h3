"""Named Drama 3 speaker presets: a reference clip plus its exact transcript.

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
class SpeakerPreset:
    name: str
    audio: Path
    transcript: str
    description: str = ""


def speakers_root(runtime: RuntimeConfig) -> Path:
    workspace = os.getenv("H3_WORKSPACE_DIR")
    base = Path(workspace) if workspace else runtime.outputs_dir / ".h3-workspace"
    return base.resolve() / "drama3_speakers"


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
    if not cleaned:
        raise H3Error("Name the speaker before saving it.")
    if len(cleaned) > MAX_NAME_LENGTH or any(ord(char) < 32 for char in cleaned):
        raise H3Error(f"Speaker names are 1–{MAX_NAME_LENGTH} printable characters.")
    return cleaned


def _folder(root: Path, name: str) -> Path:
    # Case-insensitive identity keeps "Alice" and "alice" from becoming two presets.
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:32] or "speaker"
    return root / f"{slug}-{digest}"


def _read(folder: Path) -> SpeakerPreset | None:
    try:
        data = json.loads((folder / "speaker.json").read_text(encoding="utf-8"))
        audio = folder / str(data["audio"])
        if audio.parent != folder or not audio.is_file():
            return None
        return SpeakerPreset(str(data["name"]), audio, str(data["transcript"]),
                             str(data.get("description", "")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_speakers(runtime: RuntimeConfig) -> list[SpeakerPreset]:
    root = speakers_root(runtime)
    if not root.is_dir():
        return []
    presets = (_read(folder) for folder in root.iterdir() if folder.is_dir())
    return sorted((preset for preset in presets if preset), key=lambda preset: preset.name.casefold())


def speaker_names(runtime: RuntimeConfig) -> list[str]:
    return [preset.name for preset in list_speakers(runtime)]


def load_speaker(name: str, runtime: RuntimeConfig) -> SpeakerPreset:
    preset = _read(_folder(speakers_root(runtime), normalize_name(name)))
    if preset is None:
        raise H3Error(f'The saved speaker "{name}" is no longer available.')
    return preset


def save_speaker(name: str, audio: str | Path, transcript: str, runtime: RuntimeConfig,
                 description: str = "") -> SpeakerPreset:
    """Create or replace a preset from an uploaded or designed reference clip."""
    name = normalize_name(name)
    transcript = str(transcript or "").strip()
    if not transcript:
        raise H3Error("Add the clip's exact transcript (or auto-transcribe it) before saving.")
    source = Path(str(audio or ""))
    if not str(audio or "") or not source.is_file():
        raise H3Error("Upload or generate a reference clip before saving the speaker.")
    suffix = source.suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise H3Error("Save a WAV, FLAC, MP3, OGG, M4A, AAC, Opus or WebM clip.")
    root = speakers_root(runtime)
    folder = _folder(root, name)
    with _LOCK:
        if not folder.is_dir() and len(list_speakers(runtime)) >= MAX_PRESETS:
            raise H3Error(f"Delete a saved speaker first; at most {MAX_PRESETS} are kept.")
        staging = root / f".saving-{folder.name}-{time.time_ns()}"
        staging.mkdir(parents=True)
        try:
            shutil.copyfile(source, staging / f"voice{suffix}")
            (staging / "speaker.json").write_text(json.dumps(
                {"name": name, "audio": f"voice{suffix}", "transcript": transcript,
                 "description": str(description or "").strip(), "saved": time.time()},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            if folder.exists():
                shutil.rmtree(folder)
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return load_speaker(name, runtime)


def delete_speaker(name: str, runtime: RuntimeConfig) -> None:
    folder = _folder(speakers_root(runtime), normalize_name(name))
    with _LOCK:
        if not folder.is_dir():
            raise H3Error(f'The saved speaker "{name}" does not exist.')
        shutil.rmtree(folder)
