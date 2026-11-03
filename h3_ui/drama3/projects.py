"""Named Sound Stage projects shared between the CPU Workshop and Studio."""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from uuid import uuid4

import gradio as gr

from h3_app.storyboard.service import file_digest
from .controls import _slot_visibility, _sound_slot_visibility


@dataclass(frozen=True)
class ProjectControls:
    name: gr.Textbox
    saved: gr.Dropdown
    save: gr.Button
    load: gr.Button
    active: gr.State
    status: gr.Textbox

    @classmethod
    def build(cls):
        with gr.Group():
            with gr.Row():
                name = gr.Textbox(label="Project name", max_lines=1)
                saved = gr.Dropdown([], label="Saved Sound Stage projects", value=None)
            with gr.Row():
                save = gr.Button("Save project", size="sm")
                load = gr.Button("Load project", size="sm")
            status = gr.Textbox(label="Current project", value="Unsaved draft", interactive=False)
        return cls(name, saved, save, load, gr.State(""), status)


def _fields(view):
    keys = ("text", "title", "script_mode", "speaker_mode", "writer_workflow", "max_batch_lines",
            "strict_source", "smart_budget_profile", "speaker_group", "voice_engine", "skip_existing",
            "skip_existing_sounds", "temperature", "top_p", "top_k", "max_frames", "parallel",
            "retries", "word_timing", "auto_save_speakers", "shown_slots", "shown_sound_slots")
    fields = {key: getattr(view, key) for key in keys}
    for kind, slots, attrs in (
        ("voice", view.voices, ("preset", "save_name", "description", "preview_text", "language", "audio", "transcript")),
        ("sound", view.sounds, ("preset", "save_name", "description", "duration", "audio")),
    ):
        for i, slot in enumerate(slots):
            for attr in attrs:
                fields[f"{kind}_{i}_{attr}"] = getattr(slot, attr)
    return fields


def _keep_clip(path, directory):
    source = Path(path)
    target = directory / (file_digest(source) + source.suffix.lower())
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / uuid4().hex
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target.name


def bind_projects(view, store, output_dir, load_trigger, *, gpu=True):
    controls = view.project
    fields = _fields(view)
    clips = Path(output_dir) / "h3" / ".soundstage" / "clips"

    def choices():
        return [(name + (" · Generated" if generated else ""), name)
                for name, generated in store.soundstage_projects()]

    def refresh(active):
        options = choices()
        label = next((label for label, name in options if name == active), "Unsaved draft")
        return gr.update(choices=options), label

    quiet = dict(queue=False, api_name=False, show_progress="hidden")
    picker_outputs = [controls.saved, controls.status]
    load_trigger(refresh, inputs=controls.active, outputs=picker_outputs, **quiet)

    def save(name, marker, *values):
        name = (name or "").strip()
        if not name:
            raise gr.Error("Enter a project name to save this draft.")
        data = dict(zip(fields, values, strict=True))
        for key, value in data.items():
            if key.endswith("_audio") and value:
                data[key] = _keep_clip(value, clips)
        data["prepared"] = marker == (data["text"], data["script_mode"])
        store.save_soundstage_project(name, data)
        picker, label = refresh(name)
        return name, name, {**picker, "value": name}, label

    controls.save.click(save, [controls.name, view.prepared_script, *fields.values()],
                        [controls.active, controls.name, controls.saved, controls.status], api_name=False)

    def load(name):
        try:
            data, result = store.soundstage_project(name)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        values = []
        missing = []
        for key, component in fields.items():
            value = data.get(key, component.value)
            if key.endswith("_audio") and value:
                clip = clips / Path(value).name
                value = str(clip) if clip.is_file() else None
                if value is None:
                    missing.append(key)
            values.append(value)
        if missing:
            gr.Warning("Reattach missing project clips: " + ", ".join(missing))
        marker = (data["text"], data["script_mode"]) if data.get("prepared") else None
        voice_layout = _slot_visibility(data["shown_slots"])[1:]
        sound_layout = _sound_slot_visibility(data["shown_sound_slots"])[1:]
        audio = result if result and Path(result).is_file() else None
        return (name, name, f"{name} · Generated" if result else f"{name} · Draft", marker,
                *values, *voice_layout, *sound_layout, *([audio] if gpu else []))

    layout = [*[slot.group for slot in view.voices], view.add_speaker, view.remove_speaker,
              *[slot.group for slot in view.sounds], view.add_sound, view.remove_sound]
    controls.load.click(load, controls.saved, [controls.active, controls.name, controls.status,
                        view.prepared_script, *fields.values(), *layout, *([view.output] if gpu else [])],
                        api_name=False)
    if gpu:
        view.output.change(refresh, inputs=controls.active, outputs=picker_outputs, **quiet)
