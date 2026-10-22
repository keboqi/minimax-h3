"""Voice transcription and voice design, one voice or a whole cast."""

from __future__ import annotations


import gradio as gr

from h3_app.errors import H3Error
from h3_app.generation import requests as generation_requests
from h3_app.jobs import JobCancelled

DRAMA3_VOICE_DESIGN_SETTINGS = ("engine", "temperature", "top_p", "top_k", "max_frames")


class VoiceHandlers:
    """Soundstage handlers for this area; runtime services come from :class:`Drama3Handlers`."""

    def transcribe_drama3_voice(self, audio, model_choice="1.7B (quality)"):
        """Fill a voice slot's transcript with Qwen3-ASR running in ComfyUI."""
        from h3_app.generation.drama3 import transcribe_drama3_audio
        if not audio:
            return gr.skip(), "Upload a reference clip first."
        try:
            staged = self.stage_file(audio, "drama3_asr")
            transcript = transcribe_drama3_audio(staged, self._generation_services(), model_choice=model_choice)
        except Exception as exc:
            return gr.skip(), f"Transcription failed: {exc}"
        return transcript, "Transcribed with Qwen3-ASR. Check names and punctuation before generating."

    def generate_drama3_voice(self, description, preview_text, language, engine="Qwen3-TTS VoiceDesign",
                              temperature=1.0, top_p=0.9, top_k=0, max_frames=8192):
        """A generated reference and its synthesis text always update the same slot together.

        Drama 3 designs on SGLang-Omni with the tab's sampling; Qwen uses its isolated runtime.
        """
        from h3_app.drama3.voice_design import ENGINE_DRAMA3, ENGINE_QWEN, generate_drama3_reference
        from h3_app.voice_design import generate_reference

        try:
            if engine == ENGINE_QWEN:
                yield gr.skip(), gr.skip(), "Preparing Qwen VoiceDesign; runtime and weights download on first use."
                audio, transcript = generate_reference(description, preview_text, language, self._runtime_config(),
                                                       free_gpu=self.free_comfy_vram)
            elif engine == ENGINE_DRAMA3:
                from h3_models import resolve_hf_token

                request = self._drama3_design_request(temperature, top_p, top_k, max_frames)
                yield gr.skip(), gr.skip(), "Preparing Drama 3 voice design on SGLang-Omni."
                audio = transcript = None
                for update in generate_drama3_reference(description, preview_text, request, self._runtime_config(),
                                                        self._generation_services(), free_gpu=self.free_comfy_vram,
                                                        hf_token=resolve_hf_token()):
                    if update.output:
                        audio, transcript = update.output, update.status
                    else:
                        yield gr.skip(), gr.skip(), update.status
            else:
                raise H3Error("Choose Drama 3 or Qwen3-TTS VoiceDesign to generate a voice.")
        except Exception as exc:
            yield gr.skip(), gr.skip(), f"Voice generation failed: {exc}"
            return
        yield str(audio), gr.update(value=transcript, interactive=True), "Voice generated. Preview speech copied as its exact transcript; ready to save."

    def _drama3_design_request(self, temperature, top_p, top_k, max_frames, parallel=8):
        """The tab's sampling for voice design; text and voices are filled per take."""
        return generation_requests.Drama3Request(
            "", (), temperature, top_p, top_k, max_frames, int(parallel or 1))

    def generate_drama3_voices(self, *values):
        """Generate each shown upload slot that has a design brief and no clip yet.

        Inputs: the voice-design settings, parallel requests, the shown slot count,
        whether to skip speakers that have a clip, then (preset, has_clip,
        description, preview_text, language) per slot. Outputs:
        (audio, transcript, status) per slot, then the overall status. Drama 3 on
        SGLang-Omni generates up to ``parallel`` voices at once; Qwen runs them in turn.
        """
        from h3_app.drama3.voice_design import ENGINE_DRAMA3
        from h3_ui.drama3.controls import DRAMA3_VOICE_SLOTS, UPLOAD_VOICE

        settings = values[:len(DRAMA3_VOICE_DESIGN_SETTINGS)]
        parallel = values[len(settings)]
        shown = int(values[len(settings) + 1] or 0)
        skip_existing = values[len(settings) + 2] is not False
        slots = values[len(settings) + 3:]
        updates = [gr.skip()] * (3 * DRAMA3_VOICE_SLOTS)

        def frame(status):
            # Transcripts and statuses are sent once, but a clip repeats in every later
            # frame: Gradio's client applies streamed diffs in place to the value it gave
            # the Audio, so a skip after a clip turns that value into {"__type__": "update"}
            # and the next request reading the slot fails validation. Shared with helpers.
            current = list(updates)
            updates[:] = [value if index % 3 == 0 else gr.skip() for index, value in enumerate(current)]
            return (*current, status)

        targets = []
        for speaker in range(max(0, min(shown, DRAMA3_VOICE_SLOTS))):
            preset, has_clip, description, preview, language = slots[5 * speaker:5 * speaker + 5]
            if preset and preset != UPLOAD_VOICE:
                continue
            if has_clip and skip_existing:
                updates[3 * speaker + 2] = ("Skipped: this speaker already has a clip. Uncheck "
                                            "Skip speakers with a voice to regenerate it.")
            elif not str(description or "").strip() or not str(preview or "").strip():
                updates[3 * speaker + 2] = "Skipped: add a voice description and preview speech first."
            else:
                targets.append((speaker, description, preview, language))
        if not targets:
            yield frame("No speaker voices to generate. Shown speakers without a clip need a voice description "
                        "and preview speech.")
            return
        failed = []
        if settings[0] == ENGINE_DRAMA3:
            yield from self._generate_drama3_designs(targets, settings, parallel, updates, frame, failed)
        else:
            for index, (speaker, description, preview, language) in enumerate(targets, 1):
                progress = f"Generating speaker voices: {index}/{len(targets)} (speaker {speaker})"
                # No frame of its own: the voice's first status follows at once and carries
                # the progress line, and every frame resends each finished clip.
                status = ""
                for audio, transcript, status in self.generate_drama3_voice(description, preview, language, *settings):
                    if isinstance(audio, str):
                        updates[3 * speaker:3 * speaker + 2] = [audio, transcript]
                    updates[3 * speaker + 2] = status
                    yield frame(progress)
                if status.startswith("Voice generation failed"):
                    failed.append(speaker)
        done = len(targets) - len(failed)
        summary = f"Generated {done} of {len(targets)} speaker voices."
        if failed:
            summary += " Failed: " + ", ".join(f"speaker {speaker}" for speaker in failed) + "; see each speaker's panel."
        yield frame(summary)

    def _generate_drama3_designs(self, targets, settings, parallel, updates, frame, failed):
        from h3_app.drama3 import voice_design as drama3_voice_design
        from h3_models import resolve_hf_token

        _engine, *request_settings = settings
        request = self._drama3_design_request(*request_settings, parallel=parallel)
        for speaker, *_brief in targets:
            updates[3 * speaker + 2] = "Waiting for Drama 3 voice design on SGLang-Omni…"
        yield frame(f"Preparing Drama 3 voice design for {len(targets)} speaker(s) on SGLang-Omni.")
        pending = {speaker for speaker, *_brief in targets}
        try:
            for result in drama3_voice_design.generate_drama3_references(
                    {speaker: (description, preview) for speaker, description, preview, _language in targets},
                    request, self._runtime_config(), self._generation_services(), free_gpu=self.free_comfy_vram,
                    hf_token=resolve_hf_token()):
                if isinstance(result, drama3_voice_design.DesignResult):
                    speaker = result.key
                    pending.discard(speaker)
                    if result.audio:
                        updates[3 * speaker:3 * speaker + 3] = [
                            result.audio, gr.update(value=result.transcript, interactive=True),
                            "Voice generated. Preview speech copied as its exact transcript; ready to save."]
                    else:
                        failed.append(speaker)
                        updates[3 * speaker + 2] = f"Voice generation failed: {result.error}"
                    yield frame(f"Designed {len(targets) - len(pending)}/{len(targets)} speaker voices on SGLang-Omni.")
                else:
                    yield frame(result.status)
        except JobCancelled:
            raise
        except Exception as exc:
            for speaker in sorted(pending):
                failed.append(speaker)
                updates[3 * speaker + 2] = f"Voice generation failed: {exc}"
            yield frame(f"Voice generation failed: {exc}")

    def generate_drama3_voice_takes(self, key, count, description, transcript, engine="Qwen3-TTS VoiceDesign",
                                    temperature=1.0, top_p=0.9, top_k=0, max_frames=8192, parallel=8):
        """Design up to four new voices for a saved speaker without replacing its clip.

        Outputs: one clip per try, the voice choice, the brief the tries were made from
        (saved with the chosen clip) and a status. Every frame resends every clip; see
        ``generate_drama3_voices`` for why a skip after a clip breaks the next request.
        """
        from h3_app.drama3 import voice_design as drama3_voice_design
        from h3_app.voice_design import generate_reference
        from h3_ui.drama3.speaker_library import KEEP_SAVED, MAX_TAKES, take_label

        count = max(1, min(int(count or 1), MAX_TAKES))
        clips = [None] * MAX_TAKES
        errors = []

        def frame(status, brief=gr.skip(), choice=gr.skip()):
            return (*[gr.update(value=clip, visible=index < count) for index, clip in enumerate(clips)],
                    choice, brief, status)

        if not key:
            yield frame("Select a saved speaker first.")
            return
        if not str(description or "").strip() or not str(transcript or "").strip():
            yield frame("Add a voice description and an exact clip transcript first.")
            return
        brief = {"key": key, "description": str(description).strip(), "transcript": " ".join(str(transcript).split())}
        yield frame(f"Generating {count} new voice(s)…", brief=None,
                    choice=gr.update(choices=[KEEP_SAVED], value=KEEP_SAVED))
        try:
            if engine == drama3_voice_design.ENGINE_DRAMA3:
                from h3_models import resolve_hf_token

                request = self._drama3_design_request(temperature, top_p, top_k, max_frames, parallel)
                for result in drama3_voice_design.generate_drama3_references(
                        {index: (brief["description"], brief["transcript"]) for index in range(count)},
                        request, self._runtime_config(), self._generation_services(), free_gpu=self.free_comfy_vram,
                        hf_token=resolve_hf_token()):
                    if isinstance(result, drama3_voice_design.DesignResult):
                        if result.audio:
                            clips[result.key] = result.audio
                            brief["transcript"] = result.transcript
                        else:
                            errors.append(result.error)
                        yield frame(f"Designed {sum(clip is not None for clip in clips) + len(errors)}/{count} "
                                    "voices on SGLang-Omni.")
                    else:
                        yield frame(result.status)
            elif engine == drama3_voice_design.ENGINE_QWEN:
                for index in range(count):
                    yield frame(f"Generating voice {index + 1}/{count} with Qwen VoiceDesign…")
                    try:
                        audio, brief["transcript"] = generate_reference(
                            brief["description"], brief["transcript"], "Auto", self._runtime_config(),
                            free_gpu=self.free_comfy_vram)
                        clips[index] = str(audio)
                    except JobCancelled:
                        raise
                    except Exception as exc:
                        errors.append(str(exc))
            else:
                raise H3Error("Choose Drama 3 or Qwen3-TTS VoiceDesign to generate a voice.")
        except JobCancelled:
            raise
        except Exception as exc:
            errors.append(str(exc))
        made = [take_label(index) for index, clip in enumerate(clips) if clip]
        status = f"Generated {len(made)} of {count} new voices. Listen, then pick one to keep or try again."
        if errors:
            status += " Failed: " + "; ".join(dict.fromkeys(errors))
        yield frame(status, brief=brief if made else None,
                    choice=gr.update(choices=[KEEP_SAVED, *made], value=KEEP_SAVED))
