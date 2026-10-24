"""API-only batch endpoints for scripted clients: many voices or sounds per GPU job, and a model preload.

Soundstage's own batch buttons take one input per slot, so a script cannot call them. These
handlers take a JSON list instead, run the whole list as one job on the resident audio worker
(one queue slot, one model load, no per-item round trips) and return the clips with a listing
that says which item each clip belongs to and why any item failed.
"""

from __future__ import annotations

import json

import gradio as gr

from h3_app.errors import H3Error

from .library_api import _export


def _items(value, label: str) -> list[dict]:
    try:
        items = json.loads(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise gr.Error(f"{label} must be a JSON list: {exc}") from exc
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) for item in items):
        raise gr.Error(f"{label} must be a non-empty JSON list of objects.")
    return items


class BatchApiHandlers:
    """Batch voice design, batch sounds and preload, for clients that cannot drive the page."""

    def generate_drama3_voices_batch(self, briefs, language="Auto"):
        """Design many Qwen VoiceDesign references in one job.

        ``briefs`` is a JSON list of ``{"name", "description", "preview_text"[, "language"]}``.
        Returns the clips (in the order of the successful entries) and a listing with one
        ``{"name", "ok", "file", "transcript", "error"}`` per brief; ``file`` is the clip's
        position in the returned list. The preview speech is each clip's exact transcript.
        """
        from h3_app import voice_design
        from h3_models import resolve_hf_token

        items = _items(briefs, "briefs")
        language = language or "Auto"
        try:
            results = voice_design.generate_references(
                [(item.get("description"), item.get("preview_text"), item.get("language") or language)
                 for item in items],
                self._runtime_config(), free_gpu=self.free_comfy_vram, hf_token=resolve_hf_token())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        files, listing = [], []
        for item, (audio, transcript, error) in zip(items, results):
            name = str(item.get("name") or f"voice_{len(listing)}")
            entry = {"name": name, "ok": audio is not None, "file": None, "transcript": transcript, "error": error}
            if audio is not None:
                entry["file"] = len(files)
                files.append(_export(audio, name))
            listing.append(entry)
        return files, listing

    def generate_drama3_sounds_batch(self, sounds):
        """Generate many Stable Audio 3 Medium clips in one worker request.

        ``sounds`` is a JSON list of ``{"name", "description", "duration"}``; the listing has
        one ``{"name", "ok", "file", "error"}`` per entry, ``file`` being the clip's position
        in the returned list. An invalid or failed entry fails alone; a worker failure fails the job.
        """
        from h3_app import stable_audio
        from h3_models import resolve_hf_token

        items = _items(sounds, "sounds")
        valid, listing = [], []
        for item in items:
            name = str(item.get("name") or f"sound_{len(listing)}")
            entry = {"name": name, "ok": False, "file": None, "error": ""}
            try:
                valid.append((len(listing), stable_audio.validate_sound_inputs(
                    item.get("description"), item.get("duration", stable_audio.DEFAULT_DURATION_SECONDS))))
            except H3Error as exc:
                entry["error"] = str(exc)
            listing.append(entry)
        files = []
        if valid:
            try:
                results = stable_audio.generate_sound_results(
                    [pair for _index, pair in valid], self._runtime_config(),
                    free_gpu=self.free_comfy_vram, hf_token=resolve_hf_token())
            except H3Error as exc:
                raise gr.Error(str(exc)) from exc
            for (index, _pair), (path, error) in zip(valid, results):
                if path is None:
                    listing[index]["error"] = error
                    continue
                listing[index].update(ok=True, file=len(files))
                files.append(_export(path, listing[index]["name"]))
        return files, listing
