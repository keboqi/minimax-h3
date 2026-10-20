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
    shown: gr.State
    take_count: gr.Slider
    try_takes: gr.Button
    stop_takes: gr.Button
    takes: tuple
    take_choice: gr.Radio
    use_take: gr.Button
    take_brief: gr.State
    take_status: gr.Markdown


MANAGE_LABEL = "Manage saved speakers"
HIDE_LABEL = "Hide saved speakers"
KEEP_SAVED = "Keep saved voice"
MAX_TAKES = 4


def take_label(index):
    return f"New voice {index + 1}"


def build_speaker_library(groups, open_button=None):
    """``open_button`` lets the caller place the button that reveals the library."""
    open_button = open_button or gr.Button(MANAGE_LABEL, size="sm")
    shown = gr.State(False)
    with gr.Group(visible=False, elem_classes=["h3-drama3-toolbar"]) as panel:
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
        gr.Markdown("#### Try new voices\nDesign fresh voices from the description and transcript above with the "
                    "Voice generator below. The saved voice stays until you pick a new one.")
        with gr.Row(equal_height=True):
            take_count = gr.Slider(1, MAX_TAKES, value=1, step=1, label="Voices per batch")
            with gr.Row(elem_classes=["h3-drama3-inline-actions"]):
                try_takes = gr.Button("Generate new voices", size="sm", variant="primary")
                stop_takes = gr.Button("Interrupt", size="sm", interactive=False)
                try_takes.h3_stop = stop_takes
        with gr.Row():
            takes = tuple(gr.Audio(label=take_label(index), interactive=False, type="filepath", visible=False)
                          for index in range(MAX_TAKES))
        with gr.Row(equal_height=True):
            take_choice = gr.Radio([KEEP_SAVED], value=KEEP_SAVED, label="Voice to keep")
            use_take = gr.Button("Use selected voice", size="sm", variant="primary")
        take_brief = gr.State(None)
        take_status = gr.Markdown("")
        with gr.Row():
            delete = gr.Button("Delete selected preset", variant="stop", size="sm")
            delete_group = gr.Button("Delete group and all its speakers", variant="stop", size="sm")
            close = gr.Button("Close library", size="sm")
        status = gr.Markdown("")
    return SpeakerLibraryView(open_button, panel, selected, audio, name, description, transcript,
                              save, destination, copy, delete, close, status, delete_group, shown,
                              take_count, try_takes, stop_takes, takes, take_choice, use_take, take_brief,
                              take_status)


def cleared_takes():
    """Hide earlier tries, e.g. when another speaker is selected."""
    return (*[gr.update(value=None, visible=False) for _ in range(MAX_TAKES)],
            gr.update(choices=[KEEP_SAVED], value=KEEP_SAVED), None, "")


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

    def refresh_saved(self, group, selected):
        """After a save elsewhere, list the group's speakers again and keep the selection when it still exists."""
        choices = self.choices(group)
        keep = selected if selected in {key for _name, key in choices} else None
        return (gr.update(choices=choices, value=keep),
                gr.update(choices=speakers.list_groups(self.runtime())))

    def select(self, key):
        if not key:
            return None, "", "", ""
        try:
            preset = speakers.load_speaker(key, self.runtime())
            return str(preset.audio), preset.name, preset.description, preset.transcript
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    def toggle(self, shown, group):
        if shown:
            return False, gr.update(visible=False), gr.update(value=MANAGE_LABEL), gr.skip(), gr.skip(), gr.skip()
        return True, gr.update(visible=True), gr.update(value=HIDE_LABEL), *self.refresh(group)

    def use_take(self, key, choice, brief, *takes):
        """Replace the saved clip with the chosen try; outputs preview, description, transcript, choice, status."""
        skip = (gr.skip(),) * 4
        if not choice or choice == KEEP_SAVED:
            return (*skip, "Kept the saved voice.")
        try:
            if not key:
                raise H3Error("Select a saved speaker first.")
            if not brief or brief.get("key") != key:
                raise H3Error("These voices were made for another speaker; generate new ones first.")
            index = [take_label(i) for i in range(MAX_TAKES)].index(choice)
            audio = takes[index] if index < len(takes) else None
            if not audio:
                raise H3Error("That voice is no longer available; generate new ones.")
            source = speakers.load_speaker(key, self.runtime())
            saved = speakers.save_speaker(source.name, audio, brief["transcript"], self.runtime(),
                                          brief["description"], source.group)
        except (H3Error, ValueError) as exc:
            return (*skip, str(exc))
        return (str(saved.audio), saved.description, saved.transcript,
                gr.update(value=KEEP_SAVED), f'"{saved.name}" now uses {choice.lower()}.')

    def delete_group(self, group, *selected):
        from h3_ui.drama3_controls import UPLOAD_VOICE, drama3_voice_choices

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
        from h3_ui.drama3_controls import UPLOAD_VOICE, drama3_voice_choices

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


def bind_speaker_library(view, group, slots, actions, group_status, store_events=()):
    """``store_events`` are Drama 3 events that save or delete presets; the library list follows them.

    Returns the library's own save, copy and delete events.
    """
    quiet = {"queue": False, "api_name": False, "show_progress": "hidden"}
    presets = [slot.preset for slot in slots]
    for event in store_events:
        event.then(actions.refresh_saved, inputs=[group, view.selected],
                   outputs=[view.selected, view.destination], **quiet)

    view.open.click(actions.toggle, inputs=[view.shown, group],
                    outputs=[view.shown, view.panel, view.open, view.selected, view.destination, view.status],
                    **quiet)
    group.change(actions.refresh, inputs=group,
                 outputs=[view.selected, view.destination, view.status], **quiet)
    view.close.click(lambda: (False, gr.update(visible=False), gr.update(value=MANAGE_LABEL)),
                     outputs=[view.shown, view.panel, view.open], **quiet)
    view.selected.change(actions.select, inputs=view.selected,
                         outputs=[view.audio, view.name, view.description, view.transcript], **quiet)
    view.selected.change(cleared_takes, outputs=[*view.takes, view.take_choice, view.take_brief,
                                                 view.take_status], **quiet)
    view.use_take.click(actions.use_take, inputs=[view.selected, view.take_choice, view.take_brief, *view.takes],
                        outputs=[view.audio, view.description, view.transcript, view.take_choice,
                                 view.take_status], **quiet)
    view.delete_group.click(actions.delete_group, inputs=[group, *presets],
                            outputs=[group, view.selected, view.destination, group_status, *presets], **quiet)
    mutations = []
    for button, operation in ((view.save, "save"), (view.copy, "copy"), (view.delete, "delete")):
        def mutate(group, key, name, description, transcript, destination, *selected, operation=operation):
            return actions.mutate(operation, group, key, name, description, transcript, destination, *selected)

        mutations.append(button.click(mutate, inputs=[group, view.selected, view.name, view.description,
                                                     view.transcript, view.destination, *presets],
                                      outputs=[view.selected, view.status, *presets], **quiet))
    return mutations


def bind_voice_takes(view, generate, settings, interrupt):
    """Generate tries for the selected speaker with the Cast's voice generator and sampling."""
    from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt

    generated = bind_gpu_action(
        view.try_takes.click, owned_generation(generate, "drama3_voice"),
        inputs=[view.selected, view.take_count, view.description, view.transcript, *settings],
        outputs=[*view.takes, view.take_choice, view.take_brief, view.take_status],
        show_progress="minimal", api_name=False,
    )
    stopped = view.stop_takes.click(owned_interrupt(interrupt, "drama3_voice"), outputs=view.take_status,
                                    queue=False, api_name=False)
    stopped.then(fn=None, cancels=[generated], queue=False, api_name=False)
    return generated
