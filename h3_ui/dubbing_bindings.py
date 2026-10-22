"""Dub & Translate event bindings: GPU steps run as owned jobs; edits save directly."""

from __future__ import annotations

from types import SimpleNamespace

from . import dubbing_handlers as handlers
from .dubbing_view import DubbingView, gemini_visibility, local_visibility
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB, FAMILY_LINES

from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt

_QUIET = {"queue": False, "api_name": False, "show_progress": "hidden"}


def bind_dubbing(view: DubbingView, *, tab, interrupt, actions=handlers) -> None:
    for trigger in (tab.select, view.refresh_library.click):
        trigger(actions.library_choices, outputs=view.library, **_QUIET)
    for control in (view.pipeline, view.translator):
        control.change(gemini_visibility, inputs=[view.pipeline, view.translator], outputs=view.gemini, **_QUIET)
    view.pipeline.change(local_visibility, inputs=view.pipeline, outputs=view.local, **_QUIET)
    # Edits are saved to the session as they happen; GPU steps read the saved lines.
    view.lines.input(actions.save_lines, inputs=[view.session, view.lines],
                     outputs=[view.lines, view.table_status], **_QUIET)

    analyzed = bind_gpu_action(
        view.analyze.click, owned_generation(actions.analyze_source, FAMILY_ANALYZE),
        inputs=[view.source, view.library, view.separate, view.separator_model, view.pipeline,
                view.source_language, view.target_language, view.translate, view.translator, view.aligner,
                view.merge_gap, view.gemini_model, view.api_key, view.notes],
        outputs=[view.session, view.lines, view.analyze_status], show_progress="minimal",
        api_name="dub_analyze_source",
    )
    translated = bind_gpu_action(
        view.retranslate.click, owned_generation(actions.translate_lines, FAMILY_LINES),
        inputs=[view.session, view.line_numbers, view.target_language, view.translator, view.gemini_model,
                view.api_key, view.notes],
        outputs=[view.lines, view.table_status], show_progress="minimal", api_name=False,
    )
    previewed = bind_gpu_action(
        view.preview.click, owned_generation(actions.preview_line, FAMILY_LINES),
        inputs=[view.session, view.line_numbers, view.voice_mode, view.preset, view.max_speed, view.parallel],
        outputs=[view.preview_audio, view.table_status], show_progress="minimal", api_name=False,
    )
    dubbed = bind_gpu_action(
        view.run.click, owned_generation(actions.generate_dub, FAMILY_DUB),
        inputs=[view.session, view.voice_mode, view.preset, view.max_speed, view.parallel, view.speech_volume,
                view.background_volume, view.keep_original, view.burn_subtitles, view.regenerate],
        outputs=[view.video, view.audio, view.files, view.status], show_progress="minimal",
        api_name="dub_generate",
    )
    # Each Interrupt stops only its own step's family, so stopping a preview leaves a running dub alone.
    for stop, status, family, events in (
        (view.stop_analyze, view.analyze_status, FAMILY_ANALYZE, [analyzed]),
        (view.stop_preview, view.table_status, FAMILY_LINES, [translated, previewed]),
        (view.stop, view.status, FAMILY_DUB, [dubbed]),
    ):
        stopped = stop.click(owned_interrupt(interrupt, family), outputs=status, queue=False, api_name=False)
        stopped.then(fn=None, cancels=events, queue=False, api_name=False)


def bind_media_downloads(controls: SimpleNamespace, *, on_downloaded, actions=handlers) -> None:
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
