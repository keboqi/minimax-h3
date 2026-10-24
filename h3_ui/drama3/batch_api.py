"""API-only batch endpoints for scripted clients: many voices, sounds or renders per GPU job, and a model preload.

Soundstage's own batch buttons take one input per slot, so a script cannot call them. These
handlers take a JSON list instead, run the whole list as one job on the resident audio worker
(one queue slot, one model load, no per-item round trips) and return the clips with a listing
that says which item each clip belongs to and why any item failed. The batch render takes
saved speakers and sounds by name and renders a whole book's parts back to back.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    """Batch voice design, batch sounds, batch renders and preload, for clients that cannot drive the page."""

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

    def _batch_render_request(self, item: dict):
        """A Drama 3 request for one batch job, its voices and sounds taken from the saved library."""
        from h3_app.drama3 import sounds as drama3_sounds
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_app.drama3.cast import clean_title
        from h3_app.generation import requests as generation_requests

        from .library_api import _speaker_key

        runtime = self._runtime_config()
        input_dir = runtime.input_dir
        voices, names = [], []
        for speaker in item.get("speakers") or []:
            if not isinstance(speaker, dict) or "slot" not in speaker:
                raise H3Error('Each speaker is {"slot", "name", "group", "voice"}.')
            slot = int(speaker["slot"])
            preset = drama3_speakers.load_speaker(
                _speaker_key(speaker.get("group"), speaker.get("voice") or speaker.get("name")), runtime)
            # SGLang-Omni reads staged clips by absolute path.
            voices.append((slot, str(input_dir / self.stage_file(str(preset.audio), "drama3_voices", reuse=True)),
                           preset.transcript))
            names.append((slot, " ".join(str(speaker.get("name") or preset.name).split())))
        sounds = []
        for sound in item.get("sounds") or []:
            preset = drama3_sounds.load_sound(sound.get("name") if isinstance(sound, dict) else sound, runtime)
            sounds.append((preset.name, str(input_dir / self.stage_file(str(preset.audio), "drama3_sounds", reuse=True)),
                           preset.description, int(preset.duration)))
        return generation_requests.Drama3Request(
            str(item.get("text") or ""), tuple(voices), float(item.get("temperature", 1.0)),
            float(item.get("top_p", 0.9)), int(item.get("top_k", 0)), int(item.get("max_frames", 8192)),
            int(item.get("parallel", 16)), int(item.get("retries", 0)), tuple(sorted(names)),
            script_mode=str(item.get("script_mode") or "Basic"), sounds=tuple(sounds),
            title=clean_title(item.get("title")))

    def _render_batch_job(self, request, listing, label):
        """Run one render, yielding ``(listing, status)`` progress; returns (result path, final status)."""
        from h3_app.generation.drama3 import generate_drama3 as run_drama3
        from h3_app.progress import no_progress

        output, status = None, ""
        for update in run_drama3(request, self._generation_services(), self._runtime_config(),
                                 progress=no_progress, free_gpu=self.free_comfy_vram):
            status = str(update.status or "")
            output = update.output or output
            yield listing, f"{label}: {(status.splitlines() or [''])[0]}"
        if status.startswith("Error") or not output:
            raise H3Error((status.splitlines() or ["no result"])[0].removeprefix("Error: "))
        if isinstance(output, dict):
            output = output.get("value") or output.get("path")
        return str(output), status

    def generate_drama3_batch(self, jobs, fix_rounds=1):
        """Render many Drama 3 scripts back to back in one GPU job, from saved speakers and sounds.

        ``jobs`` is a JSON list of ``{"id", "text", "title", "script_mode", "temperature", "top_p",
        "top_k", "max_frames", "parallel", "retries", "speakers": [{"slot", "name", "group",
        "voice"}], "sounds": [{"name"}]}``: each speaker names a saved speaker (``voice`` in
        ``group``) and the Cast name its lines use; each sound names a saved sound. Nothing is
        uploaded, so a client renders a whole book with the voices it saved once.

        Takes that still fail their check are generated again up to ``fix_rounds`` more times
        (0-3) before the next job starts. Yields ``(listing, status)``: one ``{"id", "ok",
        "result", "status", "failed", "rounds", "error"}`` per finished job, ``result`` being
        the file name /drama3_result_details and /drama3_result_audio take, so a client can
        fetch one job while the next renders. A failed job fails alone.
        """
        from dataclasses import replace

        from h3_app.jobs import JobCancelled, check_cancelled

        items = _items(jobs, "jobs")
        try:
            rounds = int(fix_rounds if fix_rounds not in (None, "") else 1)
        except (TypeError, ValueError):
            rounds = -1
        if not 0 <= rounds <= 3:
            raise gr.Error("fix_rounds is a whole number from 0 to 3.")
        listing = []
        for number, item in enumerate(items, 1):
            job_id = str(item.get("id") or f"job_{number}")
            label = f"Job {number}/{len(items)} ({job_id})"
            entry = {"id": job_id, "ok": False, "result": "", "status": "", "failed": [], "rounds": 0, "error": ""}
            yield listing, f"{label}: starting"
            try:
                request = self._batch_render_request(item)
                for attempt in range(rounds + 1):
                    output, status = yield from self._render_batch_job(request, listing, label)
                    failed = [chunk["take"] for chunk in self._result_settings(output).get("chunks") or []
                              if chunk.get("take") and chunk.get("ok") is False]
                    entry.update(ok=True, result=Path(output).name, status=status, failed=failed, rounds=attempt)
                    if not failed or attempt == rounds:
                        break
                    request = replace(request, regenerate=tuple(failed))
                    yield listing, f"{label}: regenerating {len(failed)} failed take(s)"
            except JobCancelled:
                raise
            except Exception as exc:  # one broken script must not cost the rest of the book
                # A failed repair round keeps the render it started from.
                entry.update(ok=bool(entry["result"]), error=str(exc))
            # A render reports an interrupt as its error status; the batch stops there too.
            check_cancelled()
            listing.append(entry)
        done = sum(entry["ok"] for entry in listing)
        yield listing, f"Batch finished: {done}/{len(listing)} job(s) rendered."
