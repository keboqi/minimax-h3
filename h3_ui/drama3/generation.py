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
from h3_ui.subtitle_words import word_spans_html

_CUE = re.compile(r"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]|\{[^{}\n]*\}")
# The script box's name colors: the narrator has its own, characters take eight in turn.
_NAME_TONES = 8


def _stage_tones(speakers, pans=None):
    """Each speaker's color and place in the stereo image, by name, in order of first line.

    Places follow the mix (``speaker_positions``): the narrator is centred, and characters
    spread to both sides only when there are two or more of them. ``pans`` holds where the
    mix placed each speaker's first line, which a scene's ``stage`` can move; results from
    before it was recorded have none.
    """
    from h3_app.drama3.serving.generate import CHARACTER_POSITIONS, NARRATOR_NAMES

    ordered = list(dict.fromkeys(speakers))
    characters = [name for name in ordered if " ".join(name.split()).casefold() not in NARRATOR_NAMES]
    spread = len(characters) >= 2
    tones = {name: ("narrator", 0.0) for name in ordered}
    tones.update((name, (str(index % _NAME_TONES),
                         CHARACTER_POSITIONS[index % len(CHARACTER_POSITIONS)] if spread else 0.0))
                 for index, name in enumerate(characters))
    for name, pan in (pans or {}).items():
        if name in tones:
            tones[name] = (tones[name][0], float(pan))
    return tones


def _rows(spans):
    """A row for each (start, end) so clips that overlap in time sit on different rows, and the count."""
    ends, rows = [], [0] * len(spans)
    for index in sorted(range(len(spans)), key=lambda index: spans[index][0]):
        start, end = spans[index]
        # Back-to-back clips share a row; a small overlap from rounding does not open a new one.
        row = next((row for row, last in enumerate(ends) if last <= start + 0.01), len(ends))
        if row == len(ends):
            ends.append(end)
        else:
            ends[row] = end
        rows[index] = row
    return rows, max(len(ends), 1)


def _place(start, end, duration, row):
    return (f'data-start="{start:.3f}" data-end="{end:.3f}" style="left: {min(start / duration, 1) * 100:.2f}%; '
            f'width: {max(end - start, 0) / duration * 100:.2f}%; --row: {row}"')


def _timeline_html(entries, tones, duration, sounds=()):
    """The result's lines and sounds on lanes, placed by time; clicking one seeks.

    Voices are colored by speaker and sounds by kind (looped beds, one-shot cues). Clips that
    play at the same time, like overlapping speech or music under an effect, stack in rows.
    """
    duration = duration or max(end for *_, end in [*entries, *sounds])
    if duration <= 0:
        return ""
    rows, count = _rows([(start, end) for *_, start, end in entries])
    lanes = [("Voices", count, "".join(
        f'<button type="button" class="h3-drama3-segment h3-drama3-tone-{tones[speaker][0]}" '
        f'data-line="{index}" {_place(start, end, duration, row)} '
        f'title="{html.escape(f"{speaker}: {text}")}" aria-label="{html.escape(f"Play from {speaker}: {text}")}">'
        f'</button>'
        for index, ((speaker, text, start, end), row) in enumerate(zip(entries, rows))))]
    if sounds:
        rows, count = _rows([(start, end) for *_, start, end in sounds])
        lanes.append(("Sounds", count, "".join(
            f'<button type="button" class="h3-drama3-segment h3-drama3-sound h3-drama3-tone-{"bed" if loop else "cue"}" '
            f'{_place(start, end, duration, row)} '
            f'title="{html.escape(name + (" (looped)" if loop else " (one-shot)"))}" '
            f'aria-label="{html.escape(f"Play from sound: {name}")}"><span>{html.escape(name)}</span></button>'
            for (name, loop, start, end), row in zip(sounds, rows))))
    names = "".join(f'<span style="--rows: {count}">{label}</span>' for label, count, _ in lanes)
    tracks = "".join(f'<div class="h3-drama3-lane" style="--rows: {count}">{segments}</div>'
                     for _, count, segments in lanes)
    return (f'<div class="h3-drama3-timeline" role="group" aria-label="Voices and sounds on the timeline">'
            f'<div class="h3-drama3-lane-names" aria-hidden="true">{names}</div><div class="h3-drama3-lanes">'
            f'{tracks}<span class="h3-drama3-playhead" aria-hidden="true" hidden></span></div></div>')


def _stage_html(tones):
    """Where each voice stands, left to right, as the mix places it."""
    voices = "".join(
        f'<span role="listitem" class="h3-drama3-stage-voice h3-drama3-tone-{tone}" '
        f'data-row="{row % 2}" style="left: {(position + 1) * 50:.1f}%">{html.escape(name)}</span>'
        for row, (name, (tone, position)) in enumerate(tones.items()))
    return ('<div class="h3-drama3-stage-map"><span class="h3-drama3-stage-label">Stage</span>'
            f'<div class="h3-drama3-stage-field" role="list" aria-label="Where each voice stands, left to right">'
            f'{voices}<span class="h3-drama3-stage-side" aria-hidden="true">L</span>'
            '<span class="h3-drama3-stage-side" aria-hidden="true">C</span>'
            '<span class="h3-drama3-stage-side" aria-hidden="true">R</span></div></div>')


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

    def repair_drama3_words(self, output):
        """Refresh subtitle metadata for the existing audio; speech and mix remain unchanged."""
        from h3_app.drama3.serving.alignment import repair_result
        from h3_app.dubbing import tools
        from h3_models import resolve_hf_token

        if isinstance(output, dict):
            output = output.get("path") or output.get("value")
        try:
            if not output:
                raise H3Error("Generate speech first.")
            source = self.services.resolve_result(output)
            for message in repair_result(source, self._runtime_config(), free_gpu=self.free_comfy_vram,
                                         hf_token=resolve_hf_token()):
                yield gr.skip(), message
            yield self.drama3_subtitles(output), message
        except JobCancelled:
            raise
        except Exception as exc:
            yield gr.skip(), f"Word alignment unavailable: {exc}"
        finally:
            tools.stop()

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
        """A result's lines and sounds on a timeline, its lines as subtitles and its voices on the stage.

        The browser highlights the line playing and moves the timeline's playhead. Results
        from before line timings were recorded have none, and the panel stays hidden.
        """
        if isinstance(output, dict):
            output = output.get("path") or output.get("value")
        settings = self._result_settings(output)
        entries, pans, words = [], {}, []
        for chunk in settings.get("chunks") or []:
            if chunk.get("start") is None:
                continue
            speaker = str(chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}")
            if chunk.get("pan") is not None:
                pans.setdefault(speaker, chunk["pan"])
            # A take can hold several script lines; results from before line timings show it whole.
            for line in chunk.get("lines") or [chunk]:
                text = " ".join(_CUE.sub(" ", str(line.get("text", ""))).split())
                if not text or line.get("start") is None:
                    continue
                start = float(line["start"])
                entries.append((speaker, text, start, float(line.get("end", start))))
                words.append(line.get("words"))
        if not entries:
            return gr.update(value="", visible=False)
        tones = _stage_tones((speaker for speaker, *_ in entries), pans)
        sounds = [(str(sound.get("name") or "Sound"), bool(sound.get("loop")), float(sound["start"]),
                   float(sound.get("end", sound["start"])))
                  for sound in settings.get("sounds") or [] if sound.get("start") is not None]
        lines = "".join(
            f'<li data-start="{start:.3f}" data-end="{end:.3f}">'
            f'<span class="h3-subtitle-speaker h3-drama3-tone-{tones[speaker][0]}">{html.escape(speaker)}</span>'
            f'<span class="h3-subtitle-text">{word_spans_html(text, line_words)}'
            + ('<small> · Word timing unavailable; use Generate / repair word alignment.</small>'
               if settings.get("word_timing") and not line_words else '') + '</span></li>'
            for (speaker, text, start, end), line_words in zip(entries, words))
        duration = float(settings.get("duration") or 0)
        # The result's 32-hex id is in every copy's file name, so the browser can find its player.
        media = re.search(r"[0-9a-f]{32}", Path(str(output)).name)
        return gr.update(visible=True, value=(
            f'<div class="h3-drama3-subtitles" data-media="{media.group(0) if media else ""}" '
            f'data-duration="{duration:.3f}">'
            f'{_timeline_html(entries, tones, duration, sounds)}<ol>{lines}</ol>{_stage_html(tones)}</div>'))

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
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_SOUND_SLOT_INPUT_NAMES, DRAMA3_VOICE_SLOTS
        slot_count = len(DRAMA3_SLOT_INPUT_NAMES)
        text, slots, settings = values[0], values[1:1 + slot_count], list(values[1 + slot_count:])
        defaults = [1.0, 0.9, 0, 8192, 16, 0]
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
        names_end = sound_end + 1 + DRAMA3_VOICE_SLOTS
        names = self._cast_names(names, settings[sound_end + 1:names_end])
        # Word-by-word subtitles come last.
        word_timing = bool(settings[names_end]) if len(settings) > names_end else False
        sounds = self._drama3_sounds(sound_slots)
        request = generation_requests.Drama3Request(
            text, voices, temperature, top_p, top_k, max_frames, parallel, retries, tuple(names),
            script_mode=script_mode, sounds=sounds, title=title, word_timing=word_timing, **options,
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
