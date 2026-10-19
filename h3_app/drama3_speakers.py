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
    group: str = "Global"

    @property
    def key(self):
        return self.name if self.group == "Global" else "group:" + json.dumps([self.group, self.name], ensure_ascii=False)


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
                             str(data.get("description", "")), str(data.get("group", "Global")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_speakers(runtime: RuntimeConfig, group="Global") -> list[SpeakerPreset]:
    root = group_root(runtime, group)
    if not root.is_dir():
        return []
    presets = (_read(folder) for folder in root.iterdir() if folder.is_dir())
    return sorted((preset for preset in presets if preset), key=lambda preset: preset.name.casefold())


def speaker_names(runtime: RuntimeConfig) -> list[str | tuple[str, str]]:
    return [(f"{preset.group} / {preset.name}", preset.key) if preset.group != "Global" else preset.name
            for group in list_groups(runtime) for preset in list_speakers(runtime, group)]


def load_speaker(name: str, runtime: RuntimeConfig) -> SpeakerPreset:
    group, name = split_key(name)
    preset = _read(_folder(group_root(runtime, group), normalize_name(name)))
    if preset is None:
        raise H3Error(f'The saved speaker "{name}" is no longer available.')
    return preset


def save_speaker(name: str, audio: str | Path, transcript: str, runtime: RuntimeConfig,
                 description: str = "", group: str = "Global") -> SpeakerPreset:
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
    group = create_group(group, runtime)
    root = group_root(runtime, group)
    folder = _folder(root, name)
    with _LOCK:
        if not folder.is_dir() and len(list_speakers(runtime, group)) >= MAX_PRESETS:
            raise H3Error(f"Delete a saved speaker first; at most {MAX_PRESETS} are kept.")
        staging = root / f".saving-{folder.name}-{time.time_ns()}"
        staging.mkdir(parents=True)
        try:
            shutil.copyfile(source, staging / f"voice{suffix}")
            (staging / "speaker.json").write_text(json.dumps(
                {"name": name, "group": group, "audio": f"voice{suffix}", "transcript": transcript,
                 "description": str(description or "").strip(), "saved": time.time()},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            if folder.exists():
                shutil.rmtree(folder)
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return _read(folder)


def delete_speaker(name: str, runtime: RuntimeConfig) -> None:
    group, name = split_key(name)
    folder = _folder(group_root(runtime, group), normalize_name(name))
    with _LOCK:
        if not folder.is_dir():
            raise H3Error(f'The saved speaker "{name}" does not exist.')
        shutil.rmtree(folder)


def split_key(key):
    if str(key).startswith("group:"):
        try:
            group, name = json.loads(key[6:])
            return normalize_name(group), normalize_name(name)
        except (ValueError, TypeError):
            pass
    return "Global", key


def group_root(runtime, group="Global"):
    group = normalize_name(group or "Global")
    root = speakers_root(runtime)
    return root if group == "Global" else _folder(root / "groups", group)


def list_groups(runtime):
    root = speakers_root(runtime) / "groups"
    groups = []
    if root.is_dir():
        for path in root.glob("*/group.json"):
            try:
                groups.append(json.loads(path.read_text(encoding="utf-8"))["name"])
            except (OSError, ValueError, KeyError):
                continue
    return ["Global", *sorted(set(groups), key=str.casefold)]


def create_group(name, runtime):
    name = normalize_name(name)
    for existing in list_groups(runtime):
        if existing.casefold() == name.casefold():
            return existing
    root = group_root(runtime, name)
    root.mkdir(parents=True, exist_ok=True)
    (root / "group.json").write_text(json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8")
    return name


def delete_group(name, runtime):
    """Delete a custom group and its owned reference clips and metadata."""
    name = normalize_name(name)
    if name.casefold() == "global":
        raise H3Error("Global is the default group and cannot be deleted.")
    parent = (speakers_root(runtime) / "groups").resolve()
    folder = group_root(runtime, name).resolve()
    if folder.parent != parent:
        raise H3Error("The speaker group is outside the saved groups directory.")
    with _LOCK:
        if not folder.is_dir():
            raise H3Error(f'The speaker group "{name}" does not exist.')
        removed = list_speakers(runtime, name)
        shutil.rmtree(folder)
    return removed


def identity(name):
    name = normalize_name(name).casefold()
    return "narrator" if name in {"narrator", "narration", "旁白", "story teller", "storyteller"} else name


def match_speaker(name, runtime, group="Global"):
    if not str(name or "").strip():
        return None
    for scope in dict.fromkeys([group or "Global", "Global"]):
        for preset in list_speakers(runtime, scope):
            if identity(preset.name) == identity(name):
                return preset
    return None


def update_speaker(key, name, transcript, description, runtime):
    """Edit a preset, rejecting renames that would overwrite another voice."""
    source = load_speaker(key, runtime)
    name = normalize_name(name)
    destination = _folder(group_root(runtime, source.group), name)
    if destination != source.audio.parent and destination.exists():
        raise H3Error(f'A speaker named "{name}" already exists in "{source.group}".')
    saved = save_speaker(name, source.audio, transcript, runtime, description, source.group)
    if destination != source.audio.parent:
        delete_speaker(key, runtime)
    return saved


def copy_speaker(key, group, runtime):
    """Copy a voice into another group without replacing an existing preset."""
    source = load_speaker(key, runtime)
    group = create_group(group or "Global", runtime)
    if group == source.group:
        raise H3Error("Choose a different destination group.")
    if _folder(group_root(runtime, group), source.name).exists():
        raise H3Error(f'A speaker named "{source.name}" already exists in "{group}".')
    return save_speaker(source.name, source.audio, source.transcript, runtime, source.description, group)
