"""Speech generation, regeneration, remix, preload, takes and subtitles."""

from __future__ import annotations

from dataclasses import replace
import html
from pathlib import Path
import re

import gradio as gr

from h3_app.prompt_writers import drama3 as drama3_writer
from h3_app.errors import H3Error
from h3_app.generation import requests as generation_requests
from h3_app.jobs import JobCancelled

_CUE = re.compile(r"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]")


class GenerationHandlers:
    """Soundstage handlers for this area; runtime services come from :class:`Drama3Handlers`."""

    def _drama3_voices(self, slots, saved_presets=None):
        """Resolve (preset, clip, transcript) slot triples to (speaker, clip, transcript).

        ``saved_presets``, when given, collects each loaded saved speaker by slot.
        """
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_ui.drama3.controls import UPLOAD_VOICE

        voices, names = [], []
        for speaker in range(len(slots) // 3):
            preset, audio, transcript = slots[3 * speaker:3 * speaker + 3]
            if preset and preset != UPLOAD_VOICE:
                saved = drama3_speakers.load_speaker(preset, self._runtime_config())
                if saved_presets is not None:
                    saved_presets[speaker] = saved
                voices.append((speaker, str(saved.audio), saved.transcript))
                names.append((speaker, saved.name))
            elif audio or str(transcript or "").strip():
                if not audio:
                    raise H3Error(f"Speaker {speaker} has a transcript but no reference clip.")
                voices.append((speaker, str(audio), str(transcript or "")))
                names.append((speaker, ""))
        return tuple(voices), names

    def _drama3_sounds(self, slots):
        """Resolve (preset, audio, save_name, description, duration) sound slot 5-tuples."""
        from h3_app.drama3 import sounds as drama3_sounds
        from h3_app.drama3.serving.tree import default_sound_name
        from h3_app.stable_audio import MAX_DURATION_SECONDS as MAX_SOUND_SECONDS
        from h3_ui.drama3.controls import UPLOAD_SOUND

        sounds = []
        for index in range(len(slots) // 5):
            preset, audio, save_name, description, duration = slots[5 * index:5 * index + 5]
            if preset and preset != UPLOAD_SOUND:
                saved = drama3_sounds.load_sound(preset, self._runtime_config())
                sounds.append((saved.name, str(saved.audio), saved.description, int(saved.duration)))
            elif audio or str(save_name or "").strip() or str(description or "").strip():
                desc = str(description or "").strip()
                # An unnamed slot takes the name an unnamed <sound> tag with this description gets.
                name = str(save_name or "").strip() or (default_sound_name(desc) if desc else f"sound_{index}")
                desc = desc or name
                try:
                    dur = max(1, min(MAX_SOUND_SECONDS, int(round(float(duration or 5)))))
                except (TypeError, ValueError):
                    dur = 5
                sounds.append((name, str(audio or ""), desc, dur))
        return tuple(sounds)

    def generate_drama3(self, *values, progress=gr.Progress(track_tqdm=False)):
        """Inputs: text, then (voice, audio, transcript) per speaker slot, then settings and sound slots.

        The parameter list is generated from the slot count; see ``_drama3_signature``.
        Lines with a saved take for the same text, voice and sampling are reused.
        """
        yield from self._drama3_job(values, progress)

    def regenerate_drama3_line(self, line, *values, progress=gr.Progress(track_tqdm=False)):
        """Generate one line's take again (``line`` is its take key from the takes table), then remix."""
        if not line:
            raise gr.Error("Choose a line from the takes table first.")
        yield from self._drama3_job(values, progress, regenerate=(str(line),))

    def remix_drama3(self, *values, progress=gr.Progress(track_tqdm=False)):
        """Rebuild the mix from saved takes, with the current sounds and effects; nothing is generated."""
        yield from self._drama3_job(values, progress, remix_only=True)

    def preload_drama3_models(self, voice_engine, script_mode="Basic"):
        """Load what Auto needs while the writer works: the voice design engine, Stable Audio, then SGLang-Omni.

        Stable Audio loads only for Advanced scripts. Loaded models are kept and never
        unloaded to make room; a model that fails or does not fit is left to the step
        that needs it, which loads it as usual. Yields the preload status.
        """
        from h3_app import audio_tools
        from h3_app.drama3.serving.generate import preload_served
        from h3_app.drama3.voice_design import ENGINE_QWEN
        from h3_models import resolve_hf_token

        runtime, hf_token = self._runtime_config(), resolve_hf_token()

        def audio_tool(model):
            if not audio_tools.preload(model, runtime, free_gpu=self.free_comfy_vram, hf_token=hf_token):
                raise H3Error("no free GPU memory beside the loaded models")
            yield from ()

        def sglang():
            for update in preload_served(runtime=runtime, free_gpu=self.free_comfy_vram, hf_token=hf_token):
                yield update.status

        qwen = voice_engine == ENGINE_QWEN
        # Drama 3 voice design runs on SGLang-Omni, so the server comes first then.
        steps = [("Qwen VoiceDesign", lambda: audio_tool("voice_design")) if qwen else ("SGLang-Omni", sglang)]
        if script_mode == drama3_writer.DRAMA3_SCRIPT_ADVANCED:
            steps.append(("Stable Audio 3 Medium", lambda: audio_tool("stable_audio")))
        if qwen:
            steps.append(("SGLang-Omni", sglang))
        ready, skipped = [], []

        def status(current=""):
            parts = [f"Preloading {current}" if current else "Preload finished."]
            if ready:
                parts.append("Ready: " + ", ".join(ready) + ".")
            if skipped:
                parts.append("Not preloaded, its step loads it: " + "; ".join(skipped) + ".")
            return " ".join(parts)

        for name, load in steps:
            yield status(f"{name}…")
            try:
                for message in load():
                    yield status(f"{name}: {message}")
            except JobCancelled:
                raise
            except Exception as exc:
                reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
                skipped.append(f"{name} ({reason})")
                continue
            ready.append(name)
        yield status()

    def _result_settings(self, output):
        """A result's settings snapshot, or {}; ``output`` may be Gradio's cached copy."""
        from h3_app.provenance import read_snapshot

        if isinstance(output, dict):
            output = output.get("path") or output.get("value")
        snapshot = read_snapshot(self.services.resolve_result(output)) if output else None
        return (snapshot or {}).get("settings") or {}

    def _take_chunks(self, output):
        """The chunks with a saved take, in script order, from a result's settings snapshot."""
        return [chunk for chunk in self._result_settings(output).get("chunks") or [] if chunk.get("take")]

    def drama3_subtitles(self, output):
        """Subtitle lines for a result, timed to the mix; the browser highlights the one playing.

        Results from before line timings were recorded have none, and the panel stays hidden.
        """
        if isinstance(output, dict):
            output = output.get("path") or output.get("value")
        settings = self._result_settings(output)
        lines = []
        for chunk in settings.get("chunks") or []:
            if chunk.get("start") is None:
                continue
            speaker = chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}"
            # A take can hold several script lines; results from before line timings show it whole.
            for line in chunk.get("lines") or [chunk]:
                text = " ".join(_CUE.sub(" ", str(line.get("text", ""))).split())
                if not text or line.get("start") is None:
                    continue
                lines.append(
                    f'<li data-start="{float(line["start"]):.3f}" data-end="{float(line.get("end", line["start"])):.3f}">'
                    f'<span class="h3-subtitle-speaker">{html.escape(str(speaker))}</span>'
                    f'<span class="h3-subtitle-text">{html.escape(text)}</span></li>')
        if not lines:
            return gr.update(value="", visible=False)
        # The result's 32-hex id is in every copy's file name, so the browser can find its player.
        media = re.search(r"[0-9a-f]{32}", Path(str(output)).name)
        return gr.update(visible=True, value=(
            f'<div class="h3-drama3-subtitles" data-media="{media.group(0) if media else ""}" '
            f'data-duration="{float(settings.get("duration") or 0):.3f}">'
            f'<ol>{"".join(lines)}</ol></div>'))

    def drama3_take_selected(self, output, evt: gr.SelectData):
        """The take key of the takes-table row the user clicked."""
        chunks = self._take_chunks(output)
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(row, int) or not 0 <= row < len(chunks):
            return gr.skip()
        return chunks[row]["take"]

    def drama3_take_audio(self, key):
        """One line's saved take, before staging and speaker effects, for the line player."""
        from h3_app.drama3.serving.takes import load_take, takes_root

        key = str(key or "")
        # Keys are hex digests; anything else must not become a path.
        loaded = load_take(takes_root(self._runtime_config()), key) if key.isalnum() else None
        if loaded is None:
            return None
        samples, rate = loaded
        return rate, samples

    def drama3_takes(self, output):
        """Takes table rows and line choices for a generated result, from its settings snapshot."""
        chunks = self._take_chunks(output)
        rows, choices = [], []
        for number, chunk in enumerate(chunks, 1):
            speaker = chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}"
            text = " ".join(str(chunk.get("text", "")).split())
            take = "reused" if chunk.get("reused") else chunk.get("check") or ""
            if take == "skipped":
                take = "new"
            rows.append([number, speaker, text, chunk.get("seconds", ""), take])
            short = text if len(text) <= 48 else text[:47] + "…"
            choices.append((f"{number} · {speaker} · {short}", chunk["take"]))
        return (gr.update(value=rows, visible=bool(rows)),
                gr.update(choices=choices, value=choices[0][1] if choices else None, visible=bool(rows)),
                gr.update(visible=bool(rows)))

    def _cast_names(self, names, slot_names):
        """Every Cast name by slot: saved speakers' names (``names``) win over typed ones."""
        cast = {speaker: " ".join(str(name or "").split()) for speaker, name in enumerate(slot_names)}
        cast.update((speaker, name) for speaker, name in names if name)
        return sorted((speaker, name) for speaker, name in cast.items() if name)

    def _drama3_job(self, values, progress, **options):
        from h3_app.generation.drama3 import generate_drama3 as run_drama3, validate_drama3
        from h3_app.drama3.cast import clean_title
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_SOUND_SLOT_INPUT_NAMES
        slot_count = len(DRAMA3_SLOT_INPUT_NAMES)
        text, slots, settings = values[0], values[1:1 + slot_count], list(values[1 + slot_count:])
        defaults = [1.0, 0.9, 0, 8192, 8, 0]
        if len(settings) < len(defaults):
            settings += defaults[len(settings):]
        temperature, top_p, top_k, max_frames, parallel, retries = settings[:6]
        script_mode = settings[6] if len(settings) > 6 and settings[6] else "Basic"
        # The sound slots, then the title, which names the output file.
        sound_end = 7 + len(DRAMA3_SOUND_SLOT_INPUT_NAMES)
        sound_slots = settings[7:sound_end]
        title = clean_title(settings[sound_end]) if len(settings) > sound_end else ""
        voices, names = self._drama3_voices(slots)
        # Lines start with these names; each line is sent with its named speaker's clip.
        names = self._cast_names(names, settings[sound_end + 1:])
        sounds = self._drama3_sounds(sound_slots)
        request = generation_requests.Drama3Request(
            text, voices, temperature, top_p, top_k, max_frames, parallel, retries, tuple(names),
            script_mode=script_mode, sounds=sounds, title=title, **options,
        )
        validate_drama3(request)
        # SGLang-Omni reads staged clips by absolute path.
        input_dir = self._runtime_config().input_dir
        request = replace(request, voices=tuple(
            (speaker, str(input_dir / self.stage_file(audio, "drama3_voices", reuse=True)), transcript)
            for speaker, audio, transcript in request.voices
        ))
        if request.sounds:
            request = replace(request, sounds=tuple(
                (name, str(input_dir / self.stage_file(audio, "drama3_sounds", reuse=True)) if audio else "",
                 description, duration)
                for name, audio, description, duration in request.sounds
            ))
        yield from run_drama3(request, self._generation_services(), self._runtime_config(),
                              progress=progress, free_gpu=self.free_comfy_vram)
