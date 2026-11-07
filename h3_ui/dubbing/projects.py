"""The same project controls and settings in Studio and the CPU Workshop."""
from dataclasses import dataclass

import gradio as gr

from h3_app.dubbing import projects, segments
from h3_app.errors import H3Error
from . import presenters


@dataclass(frozen=True)
class ProjectControls:
    name: gr.Textbox
    saved: gr.Dropdown
    save: gr.Button
    load: gr.Button
    active: gr.State
    settings: gr.State
    status: gr.Markdown

    @classmethod
    def build(cls):
        with gr.Group():
            with gr.Row():
                name = gr.Textbox(label="Project name", max_lines=1)
                saved = gr.Dropdown([], label="Saved dubbing projects", value=None)
            with gr.Row():
                save = gr.Button("Save project", size="sm")
                load = gr.Button("Load project", size="sm")
            status = gr.Markdown("Unsaved draft")
            active = gr.State(None)
            settings = gr.State(None)
        return cls(name, saved, save, load, active, settings, status)


def bind_projects(view, actions, load_trigger):
    runtime = actions.services.runtime_config
    controls = view.project
    # Never persist credentials. Speaker choices and edited lines live in the session itself.
    keys = ("source_language", "target_language", "style", "notes", "pipeline", "translate", "translator",
            "gemini_model", "separate", "separator_model", "reuse", "aligner", "merge_gap",
            "refine_timestamps", "translation_context", "voice_mode", "preset", "timing", "max_speed",
            "speech_volume", "background_volume", "keep_original", "regenerate", "parallel", "word_timing")
    fields = [getattr(view, key) for key in keys]
    quiet = dict(queue=False, api_name=False, show_progress="hidden")

    def refresh():
        return gr.update(choices=projects.names(runtime()))

    load_trigger(refresh, outputs=controls.saved, **quiet)
    controls.saved.focus(refresh, outputs=controls.saved, **quiet)

    def save(name, session_id, source, library, *values):
        settings = dict(zip(keys, values, strict=True))
        try:
            session = projects.save(runtime(), name, session_id, source or library, settings)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        name = name.strip()
        settings = {"session": session.id, **settings}
        return ({"name": name, "session": session.id}, settings, name,
                gr.update(choices=projects.names(runtime()), value=name),
                f"Saved **{name}**. Line edits save automatically.", session.id,
                str(session.path("source")), None, presenters.board_value(session),
                segments.to_rows(session.segments))

    controls.save.click(save, [controls.name, view.session, view.source, view.library, *fields],
                        [controls.active, controls.settings, controls.name, controls.saved, controls.status, view.session,
                         view.source, view.library, view.board, view.lines], api_name=False)

    def load(name):
        try:
            data, session = projects.load(runtime(), name)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        values = [data["settings"].get(key, component.value) for key, component in zip(keys, fields)]
        return ({"name": name, "session": session.id}, {"session": session.id, **data["settings"]},
                name, f"Opened **{name}**.", session.id, str(session.path("source")), None,
                presenters.board_value(session), segments.to_rows(session.segments), "", *values)

    loaded = controls.load.click(load, controls.saved,
                                 [controls.active, controls.settings, controls.name, controls.status, view.session, view.source,
                                  view.library, view.board, view.lines, view.api_key, *fields], api_name=False)

    def refresh_settings(separate, separator, reuse, pipeline, translate, translator, target, aligner,
                         merge_gap, timing, parallel, regenerate):
        from .view import advanced_label, gemini_visibility, local_visibility, manual_visibility, speed_visibility

        return (actions.analysis_plan(separate, separator, reuse, pipeline, translate, translator, target),
                actions.options_label(separate, separator, reuse, pipeline, aligner, merge_gap),
                gemini_visibility(pipeline, translator), local_visibility(pipeline), local_visibility(pipeline),
                speed_visibility(timing), advanced_label(parallel, regenerate), gr.update(visible=bool(separate)),
                manual_visibility(pipeline))

    loaded.then(refresh_settings,
                inputs=[view.separate, view.separator_model, view.reuse, view.pipeline, view.translate,
                        view.translator, view.target_language, view.aligner, view.merge_gap, view.timing,
                        view.parallel, view.regenerate],
                outputs=[view.plan, view.options, view.gemini, view.local, view.refine_timestamps,
                         view.max_speed, view.advanced, view.separator_model, view.manual], **quiet)
