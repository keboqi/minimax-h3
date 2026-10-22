"""Dub & Translate event bindings: GPU steps run as owned jobs; edits save directly."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from .view import DubbingView, gemini_visibility, local_visibility, opened_session_js, speed_visibility
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB, FAMILY_LINES

from ..job_bindings import bind_gpu_action, owned_generation, owned_interrupt

_QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}


def bind_dubbing(view: DubbingView, *, task, create_tab, interrupt, actions) -> None:
    """``task`` is Create's task picker; Dub & Translate is its "Dub" task."""
    view.refresh_library.click(actions.library_choices, inputs=view.library, outputs=view.library, **_QUIET)

    def refresh_for_dub(selected, current):
        # Engine tabs are switched by the task picker, so their own select event never fires.
        return actions.library_choices(current) if selected == "Dub" else gr.skip()

    for trigger in (task.input, create_tab.select):
        trigger(refresh_for_dub, inputs=[task, view.library], outputs=view.library, **_QUIET)
    # One source at a time: an upload replaces the Media pick and the other way round.
    view.source.upload(lambda: None, outputs=view.library, **_QUIET)
    view.library.input(lambda chosen: None if chosen else gr.skip(), inputs=view.library, outputs=view.source,
                       **_QUIET)
    view.reopen.expand(actions.session_choices, outputs=view.sessions, **_QUIET)
    view.sessions.input(actions.open_session, inputs=view.sessions,
                        outputs=[view.session, view.lines, view.analyze_status, view.summary,
                                 view.target_language], **_QUIET)
    for control in (view.pipeline, view.translator):
        control.change(gemini_visibility, inputs=[view.pipeline, view.translator], outputs=view.gemini, **_QUIET)
    view.pipeline.change(local_visibility, inputs=view.pipeline, outputs=view.local, **_QUIET)
    view.separate.change(lambda on: gr.update(visible=bool(on)), inputs=view.separate,
                         outputs=view.separator_model, **_QUIET)
    view.timing.change(speed_visibility, inputs=view.timing, outputs=view.max_speed, **_QUIET)
    # A new session (analyzed or reopened) clears the last one's results and opens Review lines.
    view.session.change(actions.clear_results,
                        outputs=[view.line_numbers, view.line_status, view.preview_audio, view.regenerate,
                                 view.video, view.audio, view.files, view.status], **_QUIET)
    view.session.change(None, inputs=view.session, js=opened_session_js(view.steps), **_QUIET)
    view.summary.change(lambda text: text, inputs=view.summary, outputs=view.dub_summary, **_QUIET)
    # Edits are saved to the session as they happen; GPU steps read the saved lines.
    view.lines.input(actions.save_lines, inputs=[view.session, view.lines],
                     outputs=[view.lines, view.summary], **_QUIET)
    view.lines.select(actions.select_line, outputs=view.line_numbers, **_QUIET)

    analyzed = bind_gpu_action(
        view.analyze.click, owned_generation(actions.analyze_source, FAMILY_ANALYZE),
        inputs=[view.source, view.library, view.separate, view.separator_model, view.pipeline,
                view.source_language, view.target_language, view.translate, view.translator, view.aligner,
                view.merge_gap, view.gemini_model, view.api_key, view.notes],
        outputs=[view.session, view.lines, view.analyze_status, view.summary], show_progress="minimal",
        api_name="dub_analyze_source",
    )
    translated = bind_gpu_action(
        view.retranslate.click, owned_generation(actions.translate_lines, FAMILY_LINES),
        inputs=[view.session, view.line_numbers, view.target_language, view.translator, view.gemini_model,
                view.api_key, view.notes],
        outputs=[view.lines, view.line_status, view.summary], show_progress="minimal", api_name=False,
    )
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
                view.speech_volume,
                view.background_volume, view.keep_original, view.burn_subtitles, view.regenerate],
        outputs=[view.video, view.audio, view.files, view.status], show_progress="minimal",
        api_name="dub_generate",
    )
    # Each Interrupt stops only its own step's family, so stopping a preview leaves a running dub alone.
    for stop, status, family, events in (
        (view.stop_analyze, view.analyze_status, FAMILY_ANALYZE, [analyzed]),
        (view.stop_preview, view.line_status, FAMILY_LINES, [translated, previewed, retaken]),
        (view.stop, view.status, FAMILY_DUB, [dubbed]),
    ):
        stopped = stop.click(owned_interrupt(interrupt, family), outputs=status, queue=False, api_name=False)
        stopped.then(fn=None, cancels=events, queue=False, api_name=False)


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
