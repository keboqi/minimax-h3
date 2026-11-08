"""The saved-sound library and Stable Audio sound generation."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from h3_app.jobs import JobCancelled

class SoundHandlers:
    """Soundstage handlers for this area; runtime services come from :class:`Drama3Handlers`."""

    def generate_drama3_sound(self, description, duration=5):
        """Generate a single sound effect or music clip with Stable Audio 3 Medium."""
        from h3_app import stable_audio
        from h3_models import resolve_hf_token

        try:
            yield gr.skip(), "Preparing Stable Audio 3 Medium; runtime and weights download on first use."
            audio = stable_audio.generate_sound_reference(
                description, duration, self._runtime_config(),
                free_gpu=self.free_comfy_vram, hf_token=resolve_hf_token(),
            )
        except Exception as exc:
            yield gr.skip(), f"Sound generation failed: {exc}"
            return
        yield str(audio), "Sound generated with Stable Audio 3 Medium; ready to save."

    def generate_drama3_sounds(self, *values):
        """Batch-generate shown upload sound slots that have a description and no clip yet."""
        from h3_app import stable_audio
        from h3_models import resolve_hf_token
        from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, UPLOAD_SOUND

        shown = int(values[0] or 0)
        skip_existing = values[1] is not False
        slots = values[2:]
        updates = [gr.skip()] * (2 * DRAMA3_SOUND_SLOTS)

        def frame(status):
            current = list(updates)
            updates[:] = [gr.skip()] * len(updates)
            return (*current, status)

        targets = []
        for index in range(max(0, min(shown, DRAMA3_SOUND_SLOTS))):
            preset, has_clip, save_name, description, duration = slots[5 * index:5 * index + 5]
            if preset and preset != UPLOAD_SOUND:
                continue
            if has_clip and skip_existing:
                updates[2 * index + 1] = (
                    "Skipped: this sound already has a clip. Uncheck "
                    "Skip sounds with a clip to regenerate it."
                )
            elif not str(description or "").strip():
                updates[2 * index + 1] = "Skipped: add a sound description first."
            else:
                targets.append((index, str(save_name or f"sound_{index}"), str(description), duration))

        if not targets:
            yield frame("No sounds to generate. Shown sound slots without a clip need a sound description.")
            return

        for index, *_rest in targets:
            updates[2 * index + 1] = "Waiting for Stable Audio 3 Medium…"
        yield frame(f"Generating {len(targets)} sound(s) with Stable Audio 3 Medium…")

        try:
            items = [
                (description, duration)
                for _index, _name, description, duration in targets
            ]
            results = list(stable_audio.generate_sound_references(
                items, self._runtime_config(), free_gpu=self.free_comfy_vram, hf_token=resolve_hf_token(),
            ))
        except JobCancelled:
            raise
        except Exception as exc:
            for index, *_rest in targets:
                updates[2 * index + 1] = f"Sound generation failed: {exc}"
            yield frame(f"Sound generation failed: {exc}")
            return

        failed = []
        for position, (index, _name, _description, _duration) in enumerate(targets):
            audio = results[position] if position < len(results) else None
            if audio and Path(str(audio)).is_file():
                updates[2 * index:2 * index + 2] = [
                    str(audio), "Sound generated with Stable Audio 3 Medium; ready to save."]
            else:
                failed.append(index)
                updates[2 * index + 1] = "Sound generation failed: Stable Audio returned no clip; generate again."
        summary = f"Generated {len(targets) - len(failed)} of {len(targets)} sound clips."
        if failed:
            summary += " Failed: " + ", ".join(f"sound {index}" for index in failed) + "; see each sound's panel."
        yield frame(summary)
