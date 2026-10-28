"""Dub & Translate event bindings: GPU steps run as owned jobs; edits save directly."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from .view import (
    DubbingView, advanced_label, gemini_visibility, local_visibility, opened_session_js, speed_visibility,
)
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB, FAMILY_LINES

from ..job_bindings import bind_gpu_action, owned_generation, owned_interrupt

_QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}


def bind_dubbing(view: DubbingView, *, task, create_tab, interrupt, actions) -> None:
    """``task`` is Create's task picker; Dub & Translate is its "Dub" task."""
    _bind_source(view, task=task, create_tab=create_tab, actions=actions)
    _bind_review(view, actions=actions)
    _bind_dub(view, actions=actions)

    analyzed = bind_gpu_action(
        view.analyze.click, owned_generation(actions.analyze_source, FAMILY_ANALYZE),
        inputs=[view.source, view.library, view.separate, view.separator_model, view.pipeline,
                view.source_language, view.target_language, view.translate, view.translator, view.aligner,
                view.merge_gap, view.gemini_model, view.api_key, view.notes, view.reuse],
        outputs=[view.session, view.board, view.lines, view.analyze_status, view.plan], show_progress="minimal",
        api_name="dub_analyze_source",
    )
    translated = bind_gpu_action(
        view.retranslate.click, owned_generation(actions.translate_lines, FAMILY_LINES),
        inputs=[view.session, view.board, view.target_language, view.translator, view.gemini_model,
                view.api_key, view.notes],
        outputs=[view.board, view.lines, view.line_status], show_progress="minimal", api_name=False,
    )
    translated.then(actions.focus_line, inputs=[view.session, view.board], outputs=inspector_outputs(view),
                    **_QUIET)
    line_inputs = [view.session, view.line_numbers, view.voice_mode, view.preset, view.timing, view.max_speed,
                   view.parallel]
    previewed = bind_gpu_action(
        view.preview.click, owned_generation(actions.preview_line, FAMILY_LINES), inputs=line_inputs,
        outputs=[view.preview_audio, view.line_status], show_progress="minimal", api_name=False,
    )
    retaken = bind_gpu_action(
        view.new_take.click, owned_generation(actions.preview_new_take, FAMILY_LINES), inputs=line_inputs,
        outputs=[view.preview_audio, view.line_status], show_progress="minimal", api_name=False,
    )
    dubbed = bind_gpu_action(
        view.run.click, owned_generation(actions.generate_dub, FAMILY_DUB),
        inputs=[view.session, view.voice_mode, view.preset, view.timing, view.max_speed, view.parallel,
                view.speech_volume, view.background_volume, view.keep_original,
                view.regenerate],
        outputs=[view.player, view.video, view.audio, view.files, view.status, view.export_source,
                 view.export_subtitles, view.exported_video, view.export_status], show_progress="minimal",
        api_name="dub_generate",
    )
    view.export_subtitles.click(
        actions.export_subtitles, inputs=view.export_source, outputs=[view.exported_video, view.export_status],
        concurrency_limit=1, show_progress="minimal", api_name=False,
    )
    # Each Interrupt stops only its own step's family, so stopping a preview leaves a running dub alone.
    for stop, status, family, events in (
        (view.stop_analyze, view.analyze_status, FAMILY_ANALYZE, [analyzed]),
        (view.stop_preview, view.line_status, FAMILY_LINES, [translated, previewed, retaken]),
        (view.stop, view.status, FAMILY_DUB, [dubbed]),
    ):
        stopped = stop.click(owned_interrupt(interrupt, family), outputs=status, queue=False, api_name=False)
        stopped.then(fn=None, cancels=events, queue=False, api_name=False)


def _bind_source(view: DubbingView, *, task, create_tab, actions) -> None:
    view.refresh_library.click(actions.library_choices, inputs=view.library, outputs=view.library, **_QUIET)

    def refresh_for_dub(selected, current):
        # Engine tabs are switched by the task picker, so their own select event never fires.
        return actions.library_choices(current) if selected == "Dub" else gr.skip()

    for trigger in (task.input, create_tab.select):
        trigger(refresh_for_dub, inputs=[task, view.library], outputs=view.library, **_QUIET)
    view.media_tab.select(actions.library_choices, inputs=view.library, outputs=view.library, **_QUIET)
    # One source at a time: an upload replaces the Media pick and the other way round.
    view.source.upload(lambda: None, outputs=view.library, **_QUIET)
    view.library.input(lambda chosen: None if chosen else gr.skip(), inputs=view.library, outputs=view.source,
                       **_QUIET)
    for event in (view.source.change, view.library.change):
        event(actions.source_card, inputs=[view.source, view.library], outputs=view.source_card, **_QUIET)
    view.sessions_tab.select(actions.recent_sessions, outputs=view.recent, **_QUIET)
    view.recent.open(actions.open_recent, inputs=view.recent,
                     outputs=[view.session, view.board, view.lines, view.analyze_status, view.target_language],
                     **_QUIET)

    # The analysis plan and the More options title follow the settings.
    plan = {"fn": actions.analysis_plan, "outputs": view.plan, "inputs": [
        view.separate, view.separator_model, view.reuse, view.pipeline, view.translate, view.translator,
        view.target_language], **_QUIET}
    options = {"fn": actions.options_label, "outputs": view.options, "inputs": [
        view.separate, view.separator_model, view.reuse, view.pipeline, view.aligner, view.merge_gap], **_QUIET}
    for control in (view.separate, view.separator_model, view.reuse, view.translate, view.translator,
                    view.target_language):
        control.change(**plan)
    for control in (view.separate, view.separator_model, view.reuse, view.aligner, view.merge_gap):
        control.change(**options)
    view.pipeline.choose(**plan)
    view.pipeline.choose(**options)
    view.pipeline.choose(gemini_visibility, inputs=[view.pipeline, view.translator], outputs=view.gemini,
                         **_QUIET)
    view.translator.change(gemini_visibility, inputs=[view.pipeline, view.translator], outputs=view.gemini,
                           **_QUIET)
    view.pipeline.choose(local_visibility, inputs=view.pipeline, outputs=view.local, **_QUIET)
    view.separate.change(lambda on: gr.update(visible=bool(on)), inputs=view.separate,
                         outputs=view.separator_model, **_QUIET)


def inspector_outputs(view: DubbingView) -> list:
    """What :meth:`DubbingHandlers.focus_line` fills: the inspector, the line numbers and the shown line's key."""
    return [view.inspector, view.start, view.end, view.speaker, view.original, view.translation, view.line_dub,
            view.fit, view.line_numbers, view.inspected]


def _bind_review(view: DubbingView, *, actions) -> None:
    focus = {"fn": actions.focus_line, "inputs": [view.session, view.board], "outputs": inspector_outputs(view),
             **_QUIET}
    # A new session (analyzed or reopened) clears the last one's results and opens Review lines.
    view.session.change(actions.clear_results,
                        outputs=[view.line_numbers, view.line_status, view.preview_audio, view.regenerate,
                                 view.player, view.video, view.audio, view.files, view.status, view.export_source,
                                 view.export_subtitles, view.exported_video, view.export_status], **_QUIET)
    view.session.change(None, inputs=view.session, js=opened_session_js(view.steps), **_QUIET)
    view.session.change(**focus)

    view.board.focus(**focus)
    view.board.pick(actions.pick_lines, inputs=view.board, outputs=view.line_numbers, **_QUIET)
    for button, step in ((view.previous_line, -1), (view.next_line, 1)):
        button.click(lambda session, board, step=step: actions.step_line(session, board, step),
                     inputs=[view.session, view.board], outputs=view.board, **_QUIET).then(**focus)
    # Switching Dub, adding or deleting a line changes the list; the inspector then follows the focus.
    changed = {"inputs": [view.session, view.board], "outputs": [view.board, view.lines, view.line_status],
               **_QUIET}
    view.board.dub(actions.toggle_dub, **changed).then(**focus)
    view.add_line.click(actions.add_line, **changed).then(**focus)
    view.delete_line.click(actions.delete_line, **changed).then(**focus)

    # The inspector saves a field once it is left (times also on Enter), so a half-typed time
    # does not move the line; the speaker and Dub save when changed.
    save = {"fn": actions.save_focused, "outputs": [view.board, view.lines, view.inspector, view.fit,
                                                    view.line_status, view.inspected],
            "inputs": [view.session, view.inspected, view.board, view.start, view.end, view.speaker, view.original,
                       view.translation, view.line_dub], **_QUIET}
    for event in (view.start.blur, view.start.submit, view.end.blur, view.end.submit, view.original.blur,
                  view.translation.blur, view.speaker.input, view.line_dub.input):
        event(**save)

    # The table saves as it is edited; the board and inspector follow.
    view.lines.input(actions.save_lines, inputs=[view.session, view.lines],
                     outputs=[view.lines, view.line_status], **_QUIET).then(
        actions.board_for, inputs=[view.session, view.board], outputs=view.board, **_QUIET).then(**focus)


def _bind_dub(view: DubbingView, *, actions) -> None:
    # Sessions that told speakers apart show a card per speaker; opening the Dub step again
    # picks up newly saved Drama 3 voices.
    speakers = {"fn": actions.speakers, "inputs": view.session, "outputs": [view.speakers_panel, view.speakers],
                **_QUIET}
    view.session.change(**speakers)
    view.steps.tabs.select(**speakers)
    view.speakers.save(actions.save_speaker, inputs=[view.session, view.speakers], outputs=view.speakers_status,
                       **_QUIET)
    readiness = {"fn": actions.readiness, "inputs": view.session, "outputs": view.readiness, **_QUIET}
    # Refreshed when the session arrives and whenever a step opens, after any edits to its lines.
    view.session.change(**readiness)
    view.steps.tabs.select(**readiness)
    view.timing.choose(speed_visibility, inputs=view.timing, outputs=view.max_speed, **_QUIET)
    for control in (view.parallel, view.regenerate):
        control.change(advanced_label, inputs=[view.parallel, view.regenerate], outputs=view.advanced, **_QUIET)


def bind_media_downloads(controls: SimpleNamespace, *, on_downloaded, actions) -> None:
    """Download from a URL and manage saved cookies inside the Media library.

    ``on_downloaded(event)`` chains the library refresh after a finished download.
    """
    controls.info.click(actions.video_info, inputs=controls.url, outputs=controls.status,
                        queue=True, api_name=False, concurrency_id="h3-download", concurrency_limit=2)
    downloaded = controls.download.click(
        actions.download_video, inputs=[controls.url, controls.quality],
        outputs=[controls.status, controls.path], api_name=False, show_progress="minimal",
        concurrency_id="h3-download", concurrency_limit=2,
    )
    # Signal the yt-dlp thread first: cancelling the event alone would leave it downloading.
    controls.stop.click(actions.stop_downloads, outputs=controls.status, queue=False, api_name=False,
                        show_progress="hidden").then(fn=None, cancels=[downloaded], queue=False, api_name=False)
    on_downloaded(downloaded)
    controls.cookies.expand(actions.cookie_sites, outputs=controls.sites, **_QUIET)
    controls.cookies.expand(actions.cookie_site_choices, outputs=controls.saved_site, **_QUIET)
    for event in (
        controls.import_curl.click(actions.import_curl_cookies, inputs=[controls.curl, controls.curl_domain],
                                   outputs=[controls.cookie_status, controls.sites, controls.curl], **_QUIET),
        controls.import_file.click(actions.import_cookie_file, inputs=[controls.cookie_file, controls.file_domain],
                                   outputs=[controls.cookie_status, controls.sites], **_QUIET),
        controls.delete.click(actions.delete_cookies, inputs=controls.saved_site,
                              outputs=[controls.cookie_status, controls.sites], **_QUIET),
    ):
        event.then(actions.cookie_site_choices, outputs=controls.saved_site, **_QUIET)
