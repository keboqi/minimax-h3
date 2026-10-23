"""Script writer and cast: rewriting, pasted replies, script modes and sound cues."""

from __future__ import annotations


import gradio as gr

from h3_app.prompt_writers import drama3 as drama3_writer
from h3_app.errors import H3Error

class WriterHandlers:
    """Soundstage handlers for this area; runtime services come from :class:`Drama3Handlers`."""

    def _drama3_writer_slots(self, slots):
        """Split writer inputs into voice triples, cast metadata and the shown slot count.

        Returns (names, existing, cast_context, shown): reference speaker names for the
        voice context, known slot identities by ID, and the JSON slot context line.
        """
        import json
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VOICE_SLOTS, DRAMA3_VISIBLE_SLOTS
        from h3_app.drama3.speakers import normalize_name

        count = len(DRAMA3_SLOT_INPUT_NAMES)
        voice_values = slots[:count]
        metadata = list(slots[count:count * 2])
        metadata += [""] * (count - len(metadata))
        shown = slots[count * 2] if len(slots) > count * 2 else DRAMA3_VISIBLE_SLOTS
        saved_presets = {}
        voices, names = self._drama3_voices(voice_values, saved_presets)
        references = {speaker for speaker, _audio, _transcript in voices}
        saved_names = dict(names)
        context = []
        for speaker in range(DRAMA3_VOICE_SLOTS):
            name, description, _preview = metadata[3 * speaker:3 * speaker + 3]
            if speaker in saved_presets:
                description = saved_presets[speaker].description or description
            name = saved_names.get(speaker) or name or ""
            name = normalize_name(name) if str(name).strip() else ""
            if speaker in references or name or description:
                context.append({"speaker_id": speaker, "name": name,
                                "description": description or "", "has_reference": speaker in references})
        existing = {entry["speaker_id"]: entry for entry in context}
        names = [(speaker, existing[speaker]["name"]) for speaker in sorted(references)]
        cast_context = ("Existing slot identities (preserve IDs and known names/descriptions; "
                        "has_reference=false is a design brief, not a cloned voice):\n"
                        + json.dumps(context, ensure_ascii=False))
        group = self._drama3_writer_group(slots)
        if group is not None:
            from h3_app.drama3 import speakers as drama3_speakers
            available = {}
            for scope in dict.fromkeys([group or "Global", "Global"]):
                for preset in drama3_speakers.list_speakers(self._runtime_config(), scope):
                    available.setdefault(drama3_speakers.identity(preset.name),
                                         {"name": preset.name, "description": preset.description})
            cast_context += ("\nReusable speaker voices (match canonical names when the character appears; "
                             "do not add unused characters; these voices will be attached automatically):\n"
                             + json.dumps(list(available.values()), ensure_ascii=False))
        return names, existing, cast_context, shown

    def _slot_names(self, existing):
        """Speaker ID to Cast name, from the writer's slot identities."""
        return {speaker: entry["name"] for speaker, entry in existing.items() if entry.get("name")}

    def _drama3_writer_mode_and_sounds(self, slots):
        """Extract optional script_mode and saved sound preset context for writer prompts."""
        import json
        from h3_app.drama3 import sounds as drama3_sounds
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES

        count = len(DRAMA3_SLOT_INPUT_NAMES)
        script_mode = (
            slots[count * 2 + 1]
            if len(slots) > count * 2 + 1 and slots[count * 2 + 1]
            else drama3_writer.DRAMA3_SCRIPT_BASIC
        )
        sound_context = ""
        if script_mode == drama3_writer.DRAMA3_SCRIPT_ADVANCED:
            saved = drama3_sounds.list_sounds(self._runtime_config())
            if saved:
                items = [
                    {"name": preset.name, "description": preset.description, "duration": preset.duration}
                    for preset in saved
                ]
                sound_context = (
                    "Existing saved sound presets (reuse matching names/descriptions when fitting):\n"
                    + json.dumps(items, ensure_ascii=False)
                )
        return script_mode, sound_context

    def _drama3_writer_group(self, slots):
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES
        index = 2 * len(DRAMA3_SLOT_INPUT_NAMES) + 2
        return (slots[index] or "Global") if len(slots) > index else None

    def _drama3_unchanged_outputs(self, group=None):
        from h3_ui.drama3.controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VOICE_SLOTS
        # Slot metadata, visibility, Add/Remove, presets with a group, then the title.
        return [gr.skip()] * (len(DRAMA3_SLOT_INPUT_NAMES) + DRAMA3_VOICE_SLOTS + 3
                              + (DRAMA3_VOICE_SLOTS if group is not None else 0) + 1)

    def _drama3_apply_cast(self, script, status, cast, existing, shown, group=None, title=""):
        """Script, status, slot metadata, visibility, presets and title for a validated cast.

        Raises on reassignment. A reply without a title keeps the one in the box.
        """
        title_update = title if title else gr.skip()
        return (*self._drama3_cast_updates(script, status, cast, existing, shown, group), title_update)

    def _drama3_cast_updates(self, script, status, cast, existing, shown, group=None):
        from h3_ui.drama3.controls import (
            DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VISIBLE_SLOTS, DRAMA3_VOICE_SLOTS, _slot_visibility
        )

        from h3_app.drama3 import speakers as drama3_speakers
        from h3_app.workflows.drama3 import format_script, tags_to_names
        presets = [gr.skip()] * DRAMA3_VOICE_SLOTS if group is not None else []
        updates = [gr.skip()] * len(DRAMA3_SLOT_INPUT_NAMES)
        # The names the slots end up with, which the script's lines show.
        final_names = {speaker: entry.get("name", "") for speaker, entry in existing.items()}
        match = drama3_speakers.speaker_matcher(self._runtime_config(), group) if group is not None and cast else None
        for entry in cast:
            speaker = entry["speaker_id"]
            previous = existing.get(speaker, {})
            if previous.get("name") and drama3_speakers.identity(previous["name"]) != drama3_speakers.identity(entry["name"]):
                raise H3Error(f"The writer reassigned speaker {speaker} from {previous['name']} to {entry['name']}.")
            # A reference's authored metadata remains authoritative; the writer has not heard the clip.
            name = previous.get("name") or entry["name"]
            description = previous.get("description") or entry["description"]
            if group is not None and not previous.get("has_reference"):
                saved = match(name)
                if saved:
                    presets[speaker] = gr.update(value=saved.key)
                    name, description = saved.name, saved.description
            updates[3 * speaker:3 * speaker + 3] = [name, description, entry["preview_text"]]
            final_names[speaker] = name
        script = format_script(tags_to_names(script, final_names))
        if not cast:
            return (script, status, *updates, *[gr.skip()] * (len(_slot_visibility(shown))), *presets)
        previous = int(shown or DRAMA3_VISIBLE_SLOTS)
        shown = max(previous, max(entry["speaker_id"] for entry in cast) + 1)
        return (script, status + f" Filled {len(cast)} matching speaker slots; "
                "open Speaker voices to upload or generate references.", *updates,
                *_slot_visibility(shown, previous), *presets)

    def enhance_drama3_text(self, text, speaker_mode, model, temporary_api_key, backend, openai_api_key,
                            openai_model, openai_base_url, *slots):
        """Apply indexed cast metadata and slot visibility only after validating the complete result."""
        group = self._drama3_writer_group(slots)
        try:
            names, existing, cast_context, shown = self._drama3_writer_slots(slots)
            script_mode, sound_context = self._drama3_writer_mode_and_sounds(slots)
            script, status, cast, title = drama3_writer.enhance_drama3_text(
                text, model, temporary_api_key, names, backend=backend,
                openai_api_key=openai_api_key, openai_model=openai_model, openai_base_url=openai_base_url,
                speaker_mode=speaker_mode, runtime=self._runtime_config(), return_cast=True,
                cast_context=cast_context, script_mode=script_mode, sound_context=sound_context,
                return_title=True, speaker_names=self._slot_names(existing),
            )
            if status.startswith("Prompt enhancement failed:"):
                return text, status, *self._drama3_unchanged_outputs(group)
            return self._drama3_apply_cast(script, status, cast, existing, shown, group, title)
        except H3Error as exc:
            return text, f"Prompt enhancement failed: {exc}", *self._drama3_unchanged_outputs(group)

    def copy_drama3_writer_prompt(self, text, speaker_mode, *slots):
        """The full writer prompt (system prompt, slot context and speech text) for an external chat model.

        It only fills the browser's clipboard, so a running rewrite and its status are left alone.
        """
        try:
            names, _existing, cast_context, _shown = self._drama3_writer_slots(slots)
            script_mode, sound_context = self._drama3_writer_mode_and_sounds(slots)
            return drama3_writer.drama3_full_prompt(
                text, names, speaker_mode, runtime=self._runtime_config(),
                cast_context=cast_context, script_mode=script_mode, sound_context=sound_context,
            )
        except H3Error as exc:
            raise gr.Error(f"Could not build the writer prompt: {exc}") from exc

    def apply_drama3_writer_reply(self, reply, text, speaker_mode, *slots):
        """Validate a pasted reply like the built-in writer, then fill the script and speaker slots."""
        group = self._drama3_writer_group(slots)
        try:
            names, existing, _cast_context, shown = self._drama3_writer_slots(slots)
            script_mode, _sound_context = self._drama3_writer_mode_and_sounds(slots)
            script, status, cast, title = drama3_writer.apply_drama3_reply(
                reply, text, names, speaker_mode, script_mode=script_mode, return_title=True,
                speaker_names=self._slot_names(existing),
            )
            if status.startswith("Prompt enhancement failed:"):
                return text, status, *self._drama3_unchanged_outputs(group)
            return self._drama3_apply_cast(script, status, cast, existing, shown, group, title)
        except H3Error as exc:
            return text, f"Prompt enhancement failed: {exc}", *self._drama3_unchanged_outputs(group)

    def switch_drama3_script_mode(self, script_mode, text):
        """Show the sounds panel for Advanced; Basic keeps but ignores the script's sound tags."""
        from h3_app.drama3.serving.tree import has_drama3_sound_features

        if script_mode == drama3_writer.DRAMA3_SCRIPT_ADVANCED:
            return (gr.update(visible=True),
                    "Advanced sound-enhanced mode enabled (sounds use Stable Audio 3 Medium).")
        # The script is left as authored, so switching back to Advanced restores its staging.
        status = "Basic script mode enabled."
        if has_drama3_sound_features(str(text or "")):
            status += (" The script's sound, staging, pause and vocal effect tags are kept but ignored "
                       "in Basic mode; switch back to Advanced to use them.")
        return gr.update(visible=False), status

    def sync_drama3_script_sounds(self, text, script_mode, shown_sound_slots, *sound_meta_values):
        """Populate sound slot names, descriptions, and durations from <sound> tags in the script."""
        from h3_app.drama3.serving.tree import extract_script_sounds
        from h3_ui.drama3.controls import DRAMA3_SOUND_SLOTS, DRAMA3_VISIBLE_SOUND_SLOTS, _sound_slot_visibility

        total_meta = 3 * DRAMA3_SOUND_SLOTS
        updates = [gr.skip()] * total_meta
        shown = int(shown_sound_slots or DRAMA3_VISIBLE_SOUND_SLOTS)
        unchanged = (*updates, *_sound_slot_visibility(shown, shown))
        if script_mode != drama3_writer.DRAMA3_SCRIPT_ADVANCED:
            return unchanged
        extracted = extract_script_sounds(str(text or ""))
        if not extracted:
            return unchanged

        existing_by_name = {}
        used_slots = set()
        for idx in range(DRAMA3_SOUND_SLOTS):
            triple = sound_meta_values[3 * idx:3 * idx + 3] if len(sound_meta_values) >= 3 * (idx + 1) else ("", "", 5)
            s_name = str(triple[0] or "").strip()
            s_desc = str(triple[1] or "").strip()
            s_dur = triple[2]
            if s_name:
                existing_by_name[s_name.casefold()] = (idx, s_name, s_desc, s_dur)
                used_slots.add(idx)

        max_slot = shown - 1
        next_free = 0
        for item in extracted:
            key = item["name"].casefold()
            if key in existing_by_name:
                idx, prev_name, prev_desc, prev_dur = existing_by_name[key]
                name = prev_name or item["name"]
                desc = prev_desc or item["description"]
                dur = prev_dur or item["duration"]
            else:
                while next_free < DRAMA3_SOUND_SLOTS and next_free in used_slots:
                    next_free += 1
                if next_free >= DRAMA3_SOUND_SLOTS:
                    break
                idx = next_free
                used_slots.add(idx)
                name = item["name"]
                desc = item["description"]
                dur = item["duration"]
            max_slot = max(max_slot, idx)
            updates[3 * idx:3 * idx + 3] = [name, desc, int(dur)]

        return (*updates, *_sound_slot_visibility(max_slot + 1, shown))
