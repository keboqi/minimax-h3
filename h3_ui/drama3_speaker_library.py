"""Saved-speaker library controls and callbacks, independent of generation jobs."""

from dataclasses import dataclass

import gradio as gr

from h3_app import drama3_speakers as speakers
from h3_app.errors import H3Error


@dataclass(frozen=True)
class SpeakerLibraryView:
    open: gr.Button
    panel: gr.Group
    selected: gr.Dropdown
    audio: gr.Audio
    name: gr.Textbox
    description: gr.Textbox
    transcript: gr.Textbox
    save: gr.Button
    destination: gr.Dropdown
    copy: gr.Button
    delete: gr.Button
    close: gr.Button
    status: gr.Markdown
    delete_group: gr.Button


def build_speaker_library(groups):
    open_button = gr.Button("Manage saved speakers", size="sm")
    with gr.Group(visible=False) as panel:
        gr.Markdown("### Saved speaker library\nBrowse the selected group above. Preview a voice, edit its details, "
                    "or copy it to another group for use in other stories.")
        with gr.Row():
            with gr.Column():
                selected = gr.Dropdown([], label="Saved speaker", filterable=True,
                                       info="Type to find a speaker in the selected group.")
                audio = gr.Audio(label="Saved voice preview", interactive=False, type="filepath")
                destination = gr.Dropdown(list(groups), value="Global", label="Copy to group")
                copy = gr.Button("Copy speaker to group", size="sm")
            with gr.Column():
                name = gr.Textbox(label="Speaker name")
                description = gr.Textbox(label="Voice description", lines=3)
                transcript = gr.Textbox(label="Exact clip transcript", lines=3,
                                       info="Keep this identical to the words spoken in the preview.")
                save = gr.Button("Save changes", variant="primary", size="sm")
        with gr.Row():
            delete = gr.Button("Delete selected preset", variant="stop", size="sm")
            delete_group = gr.Button("Delete group and all its speakers", variant="stop", size="sm")
            close = gr.Button("Close library", size="sm")
        status = gr.Markdown("")
    return SpeakerLibraryView(open_button, panel, selected, audio, name, description, transcript,
                              save, destination, copy, delete, close, status, delete_group)


class SpeakerLibraryActions:
    def __init__(self, runtime):
        self.runtime = runtime

    def choices(self, group):
        return [(preset.name, preset.key) for preset in speakers.list_speakers(self.runtime(), group or "Global")]

    def refresh(self, group):
        choices = self.choices(group)
        return (gr.update(choices=choices, value=None),
                gr.update(choices=speakers.list_groups(self.runtime())),
                f'{len(choices)} saved speakers in "{group or "Global"}".')

    def select(self, key):
        if not key:
            return None, "", "", ""
        try:
            preset = speakers.load_speaker(key, self.runtime())
            return str(preset.audio), preset.name, preset.description, preset.transcript
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    def delete_group(self, group, *selected):
        from h3_ui.optional_engine_views import UPLOAD_VOICE, drama3_voice_choices

        try:
            removed = speakers.delete_group(group or "Global", self.runtime())
        except H3Error as exc:
            return [gr.skip(), gr.skip(), gr.skip(), str(exc), *[gr.skip() for _ in selected]]
        choices = drama3_voice_choices(speakers.speaker_names(self.runtime()))
        groups = speakers.list_groups(self.runtime())
        return [gr.update(choices=groups, value="Global"),
                gr.update(choices=self.choices("Global"), value=None),
                gr.update(choices=groups, value="Global"),
                f'Deleted group "{group}" and its {len(removed)} saved speakers.',
                *[gr.update(choices=choices, value=UPLOAD_VOICE
                            if speakers.split_key(value)[0].casefold() == group.casefold() else value)
                  for value in selected]]

    def mutate(self, operation, group, key, name, description, transcript, destination, *selected):
        from h3_ui.optional_engine_views import UPLOAD_VOICE, drama3_voice_choices

        replacement = key
        try:
            if not key:
                raise H3Error("Select a saved speaker first.")
            if operation == "save":
                saved = speakers.update_speaker(key, name, transcript, description, self.runtime())
                replacement = saved.key
                status = f'Saved changes to "{saved.name}" in "{saved.group}".'
            elif operation == "copy":
                saved = speakers.copy_speaker(key, destination, self.runtime())
                status = f'Copied "{saved.name}" to "{saved.group}".'
            elif operation == "delete":
                source = speakers.load_speaker(key, self.runtime())
                speakers.delete_speaker(key, self.runtime())
                replacement = None
                status = f'Deleted "{source.name}" from "{source.group}".'
            else:
                raise H3Error("Unknown speaker library action.")
        except H3Error as exc:
            return [gr.skip(), str(exc), *[gr.skip() for _ in selected]]
        choices = drama3_voice_choices(speakers.speaker_names(self.runtime()))
        return [gr.update(choices=self.choices(group), value=replacement), status,
                *[gr.update(choices=choices, value=(replacement or UPLOAD_VOICE) if value == key else value)
                  for value in selected]]


def bind_speaker_library(view, group, slots, actions, group_status):
    quiet = {"queue": False, "api_name": False, "show_progress": "hidden"}
    presets = [slot.preset for slot in slots]

    def open_library(group):
        return gr.update(visible=True), *actions.refresh(group)

    view.open.click(open_library, inputs=group,
                    outputs=[view.panel, view.selected, view.destination, view.status], **quiet)
    group.change(actions.refresh, inputs=group,
                 outputs=[view.selected, view.destination, view.status], **quiet)
    view.close.click(fn=None, js="() => ({__type__: 'update', visible: false})", outputs=view.panel, **quiet)
    view.selected.change(actions.select, inputs=view.selected,
                         outputs=[view.audio, view.name, view.description, view.transcript], **quiet)
    view.delete_group.click(actions.delete_group, inputs=[group, *presets],
                            outputs=[group, view.selected, view.destination, group_status, *presets], **quiet)
    for button, operation in ((view.save, "save"), (view.copy, "copy"), (view.delete, "delete")):
        def mutate(group, key, name, description, transcript, destination, *selected, operation=operation):
            return actions.mutate(operation, group, key, name, description, transcript, destination, *selected)

        button.click(mutate, inputs=[group, view.selected, view.name, view.description,
                                    view.transcript, view.destination, *presets],
                     outputs=[view.selected, view.status, *presets], **quiet)
