"""The saved-speaker library and its groups."""

from __future__ import annotations


import gradio as gr

from h3_app.errors import H3Error

class SpeakerHandlers:
    """Soundstage handlers for this area; runtime services come from :class:`Drama3Handlers`."""

    def drama3_speaker_groups(self):
        from h3_app.drama3 import speakers as drama3_speakers
        return drama3_speakers.list_groups(self._runtime_config())

    def drama3_speaker_group_size(self, group):
        """Number of speakers saved in ``group`` itself (Global voices are not counted twice)."""
        from h3_app.drama3 import speakers as drama3_speakers
        return len(drama3_speakers.list_speakers(self._runtime_config(), group or "Global"))

    def create_drama3_speaker_group(self, name):
        from h3_app.drama3 import speakers as drama3_speakers
        try:
            name = drama3_speakers.create_group(name, self._runtime_config())
            return gr.update(choices=self.drama3_speaker_groups(), value=name), f'Using speaker group "{name}".'
        except H3Error as exc:
            return gr.skip(), str(exc)

    def match_drama3_speaker(self, group, name, preset, audio, match=None):
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_ui.drama3.controls import UPLOAD_VOICE
        if audio and (not preset or preset == UPLOAD_VOICE):
            return gr.skip()
        saved = (match or drama3_speakers.speaker_matcher(self._runtime_config(), group))(name)
        return gr.update(value=saved.key) if saved else gr.skip()

    def match_drama3_speakers(self, group, *slots):
        from h3_app.drama3 import speakers as drama3_speakers
        match = drama3_speakers.speaker_matcher(self._runtime_config(), group)
        return [self.match_drama3_speaker(group, *slots[i:i + 3], match=match) for i in range(0, len(slots), 3)]

    def save_all_drama3_speakers(self, group, shown, *slots):
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_ui.drama3.controls import UPLOAD_VOICE, drama3_voice_choices
        group = group or "Global"
        selected, errors, saved_count = {}, [], 0
        for i in range(min(int(shown or 1), len(slots) // 5)):
            preset, name, audio, transcript, description = slots[i * 5:i * 5 + 5]
            try:
                if preset and preset != UPLOAD_VOICE:
                    source = drama3_speakers.load_speaker(preset, self._runtime_config())
                    name, audio, transcript, description = source.name, source.audio, source.transcript, source.description
                if not audio:
                    continue
                saved = drama3_speakers.save_speaker(name, audio, transcript, self._runtime_config(), description, group)
                selected[i] = saved.key
                saved_count += 1
            except H3Error as exc:
                errors.append(f"Speaker {i}: {exc}")
        choices = drama3_voice_choices(self.drama3_speaker_names())
        return [f'Saved {saved_count} speakers to "{group}". ' + " ".join(errors),
                *[gr.update(choices=choices, **({"value": selected[i]} if i in selected else {}))
                  for i in range(len(slots) // 5)]]

    def drama3_speaker_names(self):
        from h3_app.drama3 import speakers as drama3_speakers
        return drama3_speakers.speaker_names(self._runtime_config())

    def _drama3_preset_updates(self, count, selected_slot=None, selected=None):
        from h3_ui.drama3.controls import drama3_voice_choices
        choices = drama3_voice_choices(self.drama3_speaker_names())
        return [
            gr.update(choices=choices, value=selected) if slot == selected_slot else gr.update(choices=choices)
            for slot in range(count)
        ]

    def select_drama3_speaker(self, name):
        """Show the upload controls, or a saved speaker's transcript in their place."""
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_ui.drama3.controls import UPLOAD_VOICE
        if not name or name == UPLOAD_VOICE:
            return (gr.update(visible=True), gr.update(value="", interactive=True),
                    gr.update(visible=True), gr.update(visible=False), "", gr.skip(),
                    gr.update(interactive=True), gr.skip(), None)
        try:
            saved = drama3_speakers.load_speaker(name, self._runtime_config())
        except H3Error as exc:
            return (gr.update(visible=True), gr.update(interactive=True),
                    gr.update(visible=True), gr.update(visible=False), str(exc), gr.skip(),
                    gr.update(interactive=True), gr.skip(), gr.skip())
        return (gr.update(visible=False), gr.update(value=saved.transcript, interactive=False),
                gr.update(visible=False), gr.update(visible=True), f'Using saved speaker "{saved.name}".',
                saved.name, gr.update(value=saved.description, interactive=False), saved.transcript, None)

    def save_drama3_speaker(self, slot, name, audio, transcript, count, description="", group="Global"):
        from h3_app.drama3 import speakers as drama3_speakers
        try:
            saved = drama3_speakers.save_speaker(name, audio, transcript, self._runtime_config(), description, group or "Global")
        except H3Error as exc:
            return [str(exc), *[gr.skip()] * count]
        return [f'Saved speaker "{saved.name}".', *self._drama3_preset_updates(count, slot, saved.key)]

    def delete_drama3_speaker(self, slot, name, count):
        from h3_app.drama3 import speakers as drama3_speakers
        from h3_ui.drama3.controls import UPLOAD_VOICE
        try:
            drama3_speakers.delete_speaker(name, self._runtime_config())
        except H3Error as exc:
            return [str(exc), *[gr.skip()] * count]
        return [f'Deleted saved speaker "{name}".', *self._drama3_preset_updates(count, slot, UPLOAD_VOICE)]
