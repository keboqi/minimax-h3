"""API-only access to saved speakers, saved sounds and result details, for scripted clients.

Soundstage's own library controls are browser-only. These handlers let a client keep
its own copy of a story's voices and sounds: list a group, save a clip into it, and
download a saved clip with its exact transcript. Clips are returned as copies in the
system temp directory, which Gradio serves, so the private workspace stays unserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

import gradio as gr

from h3_app.errors import H3Error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _speaker_entry(preset) -> dict:
    return {"name": preset.name, "group": preset.group, "key": preset.key, "description": preset.description,
            "transcript": preset.transcript, "audio": preset.audio.name, "sha256": _sha256(preset.audio)}


def _sound_entry(preset) -> dict:
    return {"name": preset.name, "description": preset.description, "duration": int(preset.duration),
            "audio": preset.audio.name, "sha256": _sha256(preset.audio)}


def _speaker_key(group, name) -> str:
    group = str(group or "").strip() or "Global"
    if group.casefold() == "global":
        return str(name or "")
    return "group:" + json.dumps([group, str(name or "")], ensure_ascii=False)


EXPORT_PREFIXES = ("h3-drama3-export-", "h3-drama3-library-")
EXPORT_MAX_AGE_SECONDS = 6 * 3600


def _prune_exports():
    """Drop old served copies; Gradio has copied each one into its own cache by the time it is sent."""
    cutoff = time.time() - EXPORT_MAX_AGE_SECONDS
    for prefix in EXPORT_PREFIXES:
        for folder in Path(tempfile.gettempdir()).glob(prefix + "*"):
            try:
                if folder.is_dir() and not folder.is_symlink() and folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue


def _export(audio: Path, name: str) -> str:
    """A served copy of a private clip, named after its preset."""
    _prune_exports()
    slug = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")[:48] or "clip"
    folder = Path(tempfile.mkdtemp(prefix=EXPORT_PREFIXES[0]))
    target = folder / f"{slug}{audio.suffix.lower()}"
    shutil.copyfile(audio, target)
    return str(target)


class LibraryApiHandlers:
    """Saved speakers and sounds by group and name, and a result's per-line details."""

    def drama3_library_groups(self):
        """Every speaker group, Global first."""
        from h3_app.drama3 import speakers
        return speakers.list_groups(self._runtime_config())

    def drama3_library_speakers(self, group="Global"):
        """The speakers saved in ``group`` itself (Global voices are not repeated), with clip digests."""
        from h3_app.drama3 import speakers
        try:
            return [_speaker_entry(preset)
                    for preset in speakers.list_speakers(self._runtime_config(), group or "Global")]
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    def drama3_library_save_speaker(self, group, name, audio, transcript, description=""):
        """Create or replace a speaker in ``group`` (created if missing) from a clip and its exact transcript."""
        from h3_app.drama3 import speakers
        try:
            saved = speakers.save_speaker(name, audio, transcript, self._runtime_config(), description,
                                          group or "Global")
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        return _speaker_entry(saved)

    def drama3_library_speaker_clip(self, group, name):
        """A saved speaker's clip and its metadata."""
        from h3_app.drama3 import speakers
        try:
            preset = speakers.load_speaker(_speaker_key(group, name), self._runtime_config())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        return _export(preset.audio, preset.name), _speaker_entry(preset)

    def drama3_library_sounds(self):
        """Every saved sound (sounds have no groups), with clip digests."""
        from h3_app.drama3 import sounds
        return [_sound_entry(preset) for preset in sounds.list_sounds(self._runtime_config())]

    def drama3_library_save_sound(self, name, audio, description, duration=5):
        """Create or replace a saved sound from a clip."""
        from h3_app.drama3 import sounds
        try:
            saved = sounds.save_sound(name, audio, description, duration, self._runtime_config())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        return _sound_entry(saved)

    def drama3_library_sound_clip(self, name):
        """A saved sound's clip and its metadata."""
        from h3_app.drama3 import sounds
        try:
            preset = sounds.load_sound(name, self._runtime_config())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        return _export(preset.audio, preset.name), _sound_entry(preset)

    def drama3_result_details(self, result):
        """A result's settings snapshot: title, script mode, and every take with its key, check and timings.

        ``result`` is the result's file name or URL as the client received it; its 32-hex
        id finds the original under the output folders.
        """
        name = Path(str(result or "").split("?", 1)[0]).name
        if not re.search(r"[0-9a-f]{32}", name):
            raise gr.Error("Pass the result file name returned by /generate_drama3.")
        settings = self._result_settings(name)
        if not settings:
            raise gr.Error(f'No Drama 3 result named "{name}" was found.')
        return settings
