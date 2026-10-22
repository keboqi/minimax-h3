"""Dub & Translate and Media download event handlers.

The application installs its runtime services once at import; handlers read them on
each call, so tests can patch the accessors below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

import gradio as gr

from h3_app import video_download
from h3_app.dubbing import pipeline, segments as segment_tools, session as sessions
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled, record_failure

LIBRARY_LIMIT = 200


@dataclass(frozen=True)
class DubbingServices:
    runtime_config: Callable[[], Any]
    free_gpu: Callable[[], None]
    # (mode) -> newest-first managed media paths for "Video" or "Audio".
    list_media: Callable[[str], list]


_SERVICES: DubbingServices | None = None


def install(services: DubbingServices) -> None:
    global _SERVICES
    _SERVICES = services


def _services() -> DubbingServices:
    if _SERVICES is None:
        raise RuntimeError("Dub & Translate handlers need install() before use.")
    return _SERVICES


def _runtime():
    return _services().runtime_config()


def _hf_token():
    from h3_models import resolve_hf_token

    return resolve_hf_token()


def _gpu(run):
    """Shared keyword arguments for pipeline steps."""
    return run(runtime=_runtime(), free_gpu=_services().free_gpu, hf_token=_hf_token())


def _failed(exc: BaseException) -> str:
    if isinstance(exc, JobCancelled):
        raise exc
    record_failure(exc)
    return f"**Error:** {exc}"


# ---------------------------------------------------------------------------
# Source and line table
# ---------------------------------------------------------------------------


def library_choices(current=None):
    """Media videos and audio, newest first, as (label, path) choices; a still-listed pick stays chosen."""
    choices = []
    for mode in ("Video", "Audio"):
        try:
            paths = _services().list_media(mode)
        except Exception:
            paths = []
        choices += [(f"{mode} · {Path(path).name}", str(path)) for path in paths[:LIBRARY_LIMIT]]
    return gr.update(choices=choices, value=current if current in {path for _label, path in choices} else None)


def session_choices():
    """Analyzed sessions to reopen, most recent first."""
    choices = []
    for data in sessions.recent(_runtime()):
        lines = data.get("segments") or []
        when = time.strftime("%b %d, %H:%M", time.localtime(data.get("updated") or data.get("created") or 0))
        target = data.get("target_language") or "untranslated"
        choices.append((f"{data.get('name') or 'Untitled'} · {len(lines)} line(s) · {target} · {when}", data["id"]))
    return gr.update(choices=choices, value=None)


def summary(session: sessions.Session) -> str:
    """One line on what the next dub will voice."""
    lines = session.segments
    if not lines:
        return "No lines yet."
    ready = sum(bool(s["translation"]) and s.get("dub", True) for s in lines)
    untranslated = sum(not s["translation"] for s in lines)
    kept = sum(bool(s["translation"]) and not s.get("dub", True) for s in lines)
    speakers = len({s["speaker"] for s in lines})
    target = session.data.get("target_language") or "the target language"
    parts = [f"**{ready} of {len(lines)} line(s) ready** to dub into {target}", f"{speakers} speaker(s)"]
    if untranslated:
        parts.append(f"{untranslated} untranslated")
    if kept:
        parts.append(f"{kept} keep the original voice")
    return " · ".join(parts)


def save_lines(session_id, rows):
    """Save the editor's rows as the session's lines (runs on every edit).

    Outputs: line rows, summary. Saving sorts lines by time and drops empty ones; when
    that changes the order or numbering of the finished rows, the table is redrawn so
    the line numbers used by the line actions match what is saved. A row still being
    filled in (no times or text yet) stays in the editor until it is complete.
    """
    if not str(session_id or "").strip():
        return gr.skip(), "Analyze a source first; edits apply to an analyzed session."
    rows = segment_tools.table_rows(rows)
    try:
        session = sessions.update_segments(_runtime(), session_id, segment_tools.from_rows(rows))
    except (H3Error, ValueError) as exc:
        return gr.skip(), f"**Not saved:** {exc}"
    saved = segment_tools.to_rows(session.segments)
    finished = [row for row in rows if segment_tools.row_segment(row) is not None]

    def layout(table):
        return [(row[0] if row else None, segment_tools.parse_time(row[1]) if len(row) > 1 else None)
                for row in table]

    redraw = layout(finished) != layout(saved)
    unfinished = 0 if redraw else len(rows) - len(finished)
    note = (f" · {unfinished} unfinished row(s) not saved yet: each needs a start, an end and text"
            if unfinished else "")
    return (saved if redraw else gr.skip()), summary(session) + note


def open_session(session_id):
    """Outputs: session id, line rows, status, summary, target language."""
    if not session_id:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    try:
        session = sessions.load(_runtime(), session_id)
    except H3Error as exc:
        return gr.skip(), gr.skip(), f"**Error:** {exc}", gr.skip(), gr.skip()
    return (session.id, segment_tools.to_rows(session.segments),
            f"Opened **{session.data.get('name') or 'the session'}** with {len(session.segments)} line(s).",
            summary(session), session.data.get("target_language") or gr.skip())


def clear_results():
    """A new session makes the previous one's line picks, preview and dub stale.

    Outputs: line numbers, line status, preview, fresh takes, video, audio, subtitles, dub status.
    """
    return "", "", None, "", gr.update(value=None, visible=True), None, None, ""


def select_line(event: gr.SelectData):
    """Clicking a row picks its line for the line actions; an unsaved row has no number yet."""
    row = getattr(event, "row_value", None)
    if row is not None:
        number = row[0] if row else None
    else:
        index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
        number = None if index is None else int(index) + 1
    try:
        return str(int(number))
    except (TypeError, ValueError):
        return gr.skip()


def _analysis_values(result: pipeline.Analysis):
    return (result.session.id, segment_tools.to_rows(result.session.segments), result.message,
            summary(result.session))


def analyze_source(source, library, separate, separator_model, transcription, source_language, target_language,
                   translate, translator, aligner, merge_gap, gemini_model, api_key, notes):
    """Outputs: session id, line rows, status, summary."""
    chosen = source or library
    try:
        if not chosen:
            raise H3Error("Upload a video or audio file, or pick one from Media.")
        request = pipeline.AnalyzeRequest(
            str(chosen), transcription, str(target_language or "").strip(), source_language, bool(translate),
            translator, bool(separate), separator_model, bool(aligner), float(merge_gap), gemini_model,
            str(api_key or ""), True, str(notes or ""))
        for update in _gpu(lambda **kw: pipeline.analyze(request, **kw)):
            if isinstance(update, pipeline.Analysis):
                yield _analysis_values(update)
            else:
                yield gr.skip(), gr.skip(), update, gr.skip()
    except Exception as exc:
        yield gr.skip(), gr.skip(), _failed(exc), gr.skip()


def translate_lines(session_id, line_numbers, target_language, translator, gemini_model, api_key, notes):
    """Outputs: line rows, line status, summary."""
    try:
        session = sessions.load(_runtime(), session_id)
        indices = segment_tools.parse_line_numbers(line_numbers, len(session.segments)) or None
        request = pipeline.AnalyzeRequest("", target_language=str(target_language or "").strip(),
                                          translator=translator, gemini_model=gemini_model,
                                          api_key=str(api_key or ""), notes=str(notes or ""))
        for update in _gpu(lambda **kw: pipeline.retranslate(session_id, indices, request, **kw)):
            if isinstance(update, pipeline.Analysis):
                yield segment_tools.to_rows(update.session.segments), update.message, summary(update.session)
            else:
                yield gr.skip(), update, gr.skip()
    except Exception as exc:
        yield gr.skip(), _failed(exc), gr.skip()


def _dub_request(session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume=100,
                 background_volume=100, keep_original=True, burn_subtitles=False, regenerate=""):
    session = sessions.load(_runtime(), session_id)
    fresh = tuple(segment_tools.parse_line_numbers(regenerate, len(session.segments)))
    return pipeline.DubRequest(
        session_id, voice_mode=voice_mode, preset=preset, timing=timing, parallel=int(parallel),
        max_speed=float(max_speed), speech_volume=float(speech_volume),
        background_volume=float(background_volume), keep_original_track=bool(keep_original),
        burn_subtitles=bool(burn_subtitles), regenerate=fresh)


def _preview(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, *, fresh: bool):
    try:
        session = sessions.load(_runtime(), session_id)
        indices = segment_tools.parse_line_numbers(line_numbers, len(session.segments))
        if not indices:
            raise H3Error("Click a line in the table, or enter its number, to preview it.")
        request = _dub_request(session_id, voice_mode, preset, timing, max_speed, parallel,
                               regenerate=str(indices[0] + 1) if fresh else "")
        for update in _gpu(lambda **kw: pipeline.preview_line(request, indices[0], **kw)):
            if isinstance(update, pipeline.LinePreview):
                yield (update.rate, update.samples), update.message
            else:
                yield gr.skip(), update
    except Exception as exc:
        yield gr.skip(), _failed(exc)


def preview_line(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel):
    """Outputs: preview audio, line status. Voices the first listed line, reusing its saved take."""
    yield from _preview(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, fresh=False)


def preview_new_take(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel):
    """Outputs: preview audio, line status. A new take of the first listed line, which the dub then uses."""
    yield from _preview(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, fresh=True)


def generate_dub(session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume, background_volume,
                 keep_original, burn_subtitles, regenerate):
    """Outputs: video, audio, subtitle files, status."""
    try:
        request = _dub_request(session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume,
                               background_volume, keep_original, burn_subtitles, regenerate)
        for update in _gpu(lambda **kw: pipeline.dub(request, **kw)):
            if isinstance(update, pipeline.Dub):
                video = str(update.video) if update.video else None
                yield (gr.update(value=video, visible=video is not None), str(update.audio),
                       [str(path) for path in update.subtitles], update.message)
            else:
                yield gr.skip(), gr.skip(), gr.skip(), update
    except Exception as exc:
        yield gr.skip(), gr.skip(), gr.skip(), _failed(exc)


# ---------------------------------------------------------------------------
# Media: downloads and cookies
# ---------------------------------------------------------------------------


def _cookie_store():
    return video_download.CookieStore(video_download.cookies_dir(_runtime()))


def cookie_sites() -> str:
    sites = _cookie_store().sites()
    if not sites:
        return "No saved cookies. Public videos need none."
    return "Saved cookies: " + ", ".join(f"**{site}** ({count})" for site, count in sites.items())


def cookie_site_choices():
    return gr.update(choices=list(_cookie_store().sites()), value=None)


def import_curl_cookies(curl_text, domain):
    """Outputs: status, saved sites, cURL box (cleared on success)."""
    try:
        site, count = video_download.import_curl(_cookie_store(), curl_text, domain)
    except H3Error as exc:
        return f"**Not saved:** {exc}", gr.skip(), gr.skip()
    return f"Saved {count} cookie(s) for {site}.", cookie_sites(), ""


def import_cookie_file(path, domain):
    """Outputs: status, saved sites."""
    try:
        if not path:
            raise H3Error("Choose a cookies.txt file.")
        site, count = video_download.import_cookie_file(_cookie_store(), path, domain)
    except (H3Error, OSError) as exc:
        return f"**Not saved:** {exc}", gr.skip()
    return f"Saved {count} cookie(s) for {site}.", cookie_sites()


def delete_cookies(domain):
    """Outputs: status, saved sites."""
    try:
        deleted = _cookie_store().delete(domain) if domain else False
    except H3Error as exc:
        return f"**Not deleted:** {exc}", gr.skip()
    return (f"Deleted the cookies for {domain}." if deleted else "Choose a saved site."), cookie_sites()


def video_info(url):
    try:
        return video_download.video_info(url, _runtime())
    except H3Error as exc:
        return f"**Error:** {exc}"


# Running downloads' stop flags by browser session, so Interrupt reaches the yt-dlp thread.
_DOWNLOAD_STOPS: dict[str, set[threading.Event]] = {}
_DOWNLOAD_LOCK = threading.Lock()


def _session_key(request) -> str:
    return str(getattr(request, "session_hash", None) or "")


def stop_downloads(request: gr.Request = None) -> str:
    """Interrupt this browser session's downloads at their next progress update."""
    with _DOWNLOAD_LOCK:
        running = list(_DOWNLOAD_STOPS.get(_session_key(request), ()))
    for stop in running:
        stop.set()
    return "Interrupting the download…" if running else "No download is running."


def download_video(url, quality, request: gr.Request = None):
    """Outputs: status, downloaded path (empty until the file is published).

    Downloads hold no GPU, so they run outside the job queue; ``stop_downloads``
    stops yt-dlp at its next progress update and the staged file is discarded.
    """
    from h3_app.dubbing.pipeline import _blocking

    key, stop = _session_key(request), threading.Event()
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_STOPS.setdefault(key, set()).add(stop)
    try:
        work = _blocking("Starting the download", lambda report: video_download.download_video(
            url, quality, _runtime(), report=report, stopped=stop.is_set))
        while True:
            try:
                yield next(work), ""
            except StopIteration as done:
                path = done.value
                break
        yield f"Downloaded **{path.name}** to Media.", str(path)
    except H3Error as exc:
        yield f"**Error:** {exc}", ""
    finally:
        stop.set()
        with _DOWNLOAD_LOCK:
            flags = _DOWNLOAD_STOPS.get(key, set())
            flags.discard(stop)
            if not flags:
                _DOWNLOAD_STOPS.pop(key, None)
