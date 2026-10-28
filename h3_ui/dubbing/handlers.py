"""Dub & Translate and Media download event handlers, built from explicit runtime services.

The application constructs one :class:`DubbingHandlers`; handlers read the services on
each call, so tests can construct their own with fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

import gradio as gr

from h3_app import video_download
from h3_app.drama3.serving.audio import encode_wav
from h3_app.dubbing import pipeline, segments as segment_tools, session as sessions
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled, record_failure

from . import presenters
from .player import player_html

LIBRARY_LIMIT = 200
_DOWNLOAD_STOPS: dict[str, set[threading.Event]] = {}
_DOWNLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class DubbingServices:
    runtime_config: Callable[[], Any]
    free_gpu: Callable[[], None]
    # (mode) -> newest-first managed media paths for "Video" or "Audio".
    list_media: Callable[[str], list]


def _line_key(index: int, line: dict) -> dict:
    """What names a line across saves: its place, start and original text."""
    return {"index": index, "start": line["start"], "source": line["source"]}


def _find_line(lines: list[dict], key) -> int | None:
    """The index of the line ``key`` names, preferring its old place when several match; None if gone."""
    if not isinstance(key, dict) or key.get("start") is None:
        return None
    start, source = round(float(key["start"]), 3), str(key.get("source") or "").strip()
    found = [index for index, line in enumerate(lines)
             if round(float(line["start"]), 3) == start and str(line["source"]).strip() == source]
    if not found:
        return None
    return key.get("index") if key.get("index") in found else found[0]


class DubbingHandlers:
    """Every Dub & Translate and Media download handler, bound to one set of services."""

    def __init__(self, services: DubbingServices):
        self.services = services

    def _runtime(self):
        return self.services.runtime_config()

    def _hf_token(self):
        from h3_models import resolve_hf_token

        return resolve_hf_token()

    def _gpu(self, run):
        """Shared keyword arguments for pipeline steps."""
        return run(runtime=self._runtime(), free_gpu=self.services.free_gpu, hf_token=self._hf_token())

    def _failed(self, exc: BaseException) -> str:
        if isinstance(exc, JobCancelled):
            raise exc
        record_failure(exc)
        return f"**Error:** {exc}"

    def library_choices(self, current=None):
        """Media videos and audio, newest first, as (label, path) choices; a still-listed pick stays chosen."""
        choices = []
        for mode in ("Video", "Audio"):
            try:
                paths = self.services.list_media(mode)
            except Exception:
                paths = []
            choices += [(f"{mode} · {Path(path).name}", str(path)) for path in paths[:LIBRARY_LIMIT]]
        return gr.update(choices=choices, value=current if current in {path for _label, path in choices} else None)

    # -- Source -------------------------------------------------------------

    def recent_sessions(self):
        """Analyzed sessions to reopen, most recent first."""
        return presenters.recent_value(self._runtime())

    def open_recent(self, recent):
        """Outputs: as :meth:`open_session`, for the session chosen in the earlier-sessions list."""
        recent = recent if isinstance(recent, dict) else {}
        listed, chosen = recent.get("sessions") or [], recent.get("chosen")
        if not isinstance(chosen, int) or not 0 <= chosen < len(listed):
            return self.open_session(None)
        return self.open_session(listed[chosen]["id"])

    def source_card(self, source, library=None):
        """The picked file: an upload wins over a Media pick."""
        return presenters.source_card(source or library)

    def analysis_plan(self, separate, separator_model, reuse, transcription, translate, translator,
                      target_language):
        return presenters.plan_html(self._plan_steps(separate, separator_model, reuse, transcription, translate,
                                                     translator, target_language))

    @staticmethod
    def _plan_steps(separate, separator_model, reuse, transcription, translate, translator, target_language):
        return presenters.plan_steps(
            separate=bool(separate), separator_model=str(separator_model or ""), reuse=bool(reuse),
            transcription=str(transcription or pipeline.PIPELINE_LOCAL), translate=bool(translate),
            translator=str(translator or pipeline.TRANSLATOR_HYMT),
            target_language=str(target_language or "").strip())

    def options_label(self, separate, separator_model, reuse, transcription, aligner, merge_gap):
        return gr.update(label=presenters.options_label(
            separate=bool(separate), separator_model=str(separator_model or ""), reuse=bool(reuse),
            transcription=str(transcription or ""), aligner=bool(aligner), merge_gap=float(merge_gap or 0)))

    # -- Review lines -------------------------------------------------------

    def _session(self, session_id) -> sessions.Session | None:
        try:
            return sessions.load(self._runtime(), session_id) if session_id else None
        except H3Error:
            return None

    def summary(self, session: sessions.Session) -> str:
        """One line on what the next dub will voice."""
        return presenters.summary_html(session) if session.segments else "No lines yet."

    def board_for(self, session_id, board=None):
        session = self._session(session_id)
        return presenters.board_value(session, board) if session is not None else presenters.empty_board()

    def _inspector(self, session: sessions.Session | None, board) -> tuple:
        """The focused line in the inspector, its key, and the line numbers the line actions use.

        Outputs: heading, start, end, speaker, original, translation, dub, fit, line numbers, key.
        The key names the line the inspector shows, so its saves reach that line even after
        the board's focus has moved on or the lines were re-sorted.
        """
        focus = presenters.board_focus(board)
        lines = session.segments if session is not None else []
        if focus is None or not 0 <= focus < len(lines):
            return (presenters.inspector_heading(session, None), "", "", gr.update(choices=[], value=None), "", "",
                    True, "", "", None)
        line = lines[focus]
        speakers = pipeline.session_speakers(session)
        return (presenters.inspector_heading(session, focus), segment_tools.format_time(line["start"]),
                segment_tools.format_time(line["end"]),
                gr.update(choices=[(presenters.speaker_label(s), s) for s in speakers], value=line["speaker"]),
                line["source"], line["translation"], bool(line.get("dub", True)), presenters.fit_note(line),
                presenters.line_numbers(presenters.board_picks(board) or [focus]), _line_key(focus, line))

    def focus_line(self, session_id, board):
        """Outputs: as :meth:`_inspector`."""
        return self._inspector(self._session(session_id), board)

    def step_line(self, session_id, board, step: int):
        """The board with its focus moved to the previous or next line."""
        session = self._session(session_id)
        if session is None:
            return gr.skip()
        focus = presenters.board_focus(board)
        return presenters.board_value(session, board, focus=0 if focus is None else focus + step)

    def pick_lines(self, board):
        """The picked lines, or the focused one, as the line actions' numbers."""
        picks = presenters.board_picks(board)
        focus = presenters.board_focus(board)
        return presenters.line_numbers(picks or ([focus] if focus is not None else []))

    def _change_lines(self, session_id, board, change):
        """Apply ``change(lines)`` to the saved lines; it returns the line to focus afterwards, or None.

        Returns the session and the board, or the status saying why nothing was saved.
        """
        session = self._session(session_id)
        if session is None:
            return None, None, "Analyze a source first; edits apply to an analyzed session."
        lines = [dict(line) for line in session.segments]
        try:
            target = change(lines)
            session = sessions.update_segments(self._runtime(), session.id, lines)
        except (H3Error, ValueError) as exc:
            return None, None, f"**Not saved:** {exc}"
        # Saving sorts the lines by time: follow the changed line to its new place.
        focus = presenters.board_focus(board) if target is None else _find_line(session.segments, target)
        return session, presenters.board_value(session, board, focus=focus), ""

    def save_focused(self, session_id, key, board, start, end, speaker, original, translation, dub):
        """Save the inspector's fields to the line it shows, found by ``key``.

        Outputs: board, table rows, inspector heading, fit, status, key.
        """
        begin, finish = segment_tools.parse_time(start), segment_tools.parse_time(end)

        def change(lines):
            index = _find_line(lines, key)
            if index is None:
                raise H3Error("That line has changed; click it on the board again.")
            if begin is None or finish is None:
                raise H3Error("Start and end need times such as 01:13.5.")
            if not str(original or "").strip() and not str(translation or "").strip():
                raise H3Error("A line needs its original text or a translation.")
            lines[index].update(start=begin, end=max(finish, begin + segment_tools.MIN_SEGMENT_SECONDS),
                                speaker=segment_tools.speaker_id(speaker), source=str(original or "").strip(),
                                translation=str(translation or "").strip(), dub=bool(dub))
            return lines[index]

        session, value, status = self._change_lines(session_id, board, change)
        if session is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), status, gr.skip()
        focus = value["focus"]
        line = session.segments[focus]
        # The viewer may have clicked another line while this saved; only a line that moved
        # takes the focus with it.
        value["keepFocus"] = focus == key.get("index")
        return (value, segment_tools.to_rows(session.segments), presenters.inspector_heading(session, focus),
                presenters.fit_note(line), "", _line_key(focus, line))

    def _edit_lines(self, session_id, board, change, done: str):
        """Outputs: board, table rows, status. The inspector follows the board's focus afterwards."""
        session, value, status = self._change_lines(session_id, board, change)
        if session is None:
            return gr.skip(), gr.skip(), status
        return value, segment_tools.to_rows(session.segments), done

    def toggle_dub(self, session_id, board):
        """A Dub switch on the board; it focuses that line. Outputs: board, table rows, status."""
        changed = board.get("changed") if isinstance(board, dict) else None
        index = changed.get("line") if isinstance(changed, dict) else None

        def change(lines):
            if not isinstance(index, int) or not 0 <= index < len(lines):
                raise H3Error("That line is no longer there.")
            lines[index]["dub"] = bool(changed.get("dub"))
            return lines[index]

        return self._edit_lines(session_id, board, change, "")

    def add_line(self, session_id, board):
        """A new line after the focused one, filling the gap before the next line. Outputs: board, rows, status."""
        focus = presenters.board_focus(board)

        def change(lines):
            after = lines[focus] if focus is not None and 0 <= focus < len(lines) else None
            start = round(after["end"] + 0.05, 3) if after else 0.0
            later = [line["start"] for line in lines if line["start"] > start]
            end = round(min(later) - 0.05, 3) if later and min(later) - 0.05 > start + 0.5 else start + 1.5
            speaker = after["speaker"] if after else segment_tools.DEFAULT_SPEAKER
            line = {"start": start, "end": end, "speaker": speaker, "source": "…", "translation": "", "dub": True}
            lines.append(line)
            return line

        return self._edit_lines(session_id, board, change, "Added a line: type its original text and translation.")

    def delete_line(self, session_id, board):
        """Remove the focused line. Outputs: board, table rows, status."""
        focus = presenters.board_focus(board)

        def change(lines):
            if focus is None or not 0 <= focus < len(lines):
                raise H3Error("Pick a line on the board first.")
            if len(lines) == 1:
                raise H3Error("A session keeps at least one line.")
            lines.pop(focus)
            return None

        return self._edit_lines(session_id, board, change, "Line deleted.")

    def save_lines(self, session_id, rows):
        """Save the table's rows as the session's lines (runs on every edit).

        Outputs: line rows, status. Saving sorts lines by time and drops empty ones; when
        that changes the order or numbering of the finished rows, the table is redrawn so
        the line numbers used by the line actions match what is saved. A row still being
        filled in (no times or text yet) stays in the editor until it is complete.
        """
        if not str(session_id or "").strip():
            return gr.skip(), "Analyze a source first; edits apply to an analyzed session."
        rows = segment_tools.table_rows(rows)
        try:
            session = sessions.update_segments(self._runtime(), session_id, segment_tools.from_rows(rows))
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
        return (saved if redraw else gr.skip()), self.summary(session) + note

    def open_session(self, session_id):
        """Outputs: session id, board, table rows, status, target language."""
        if not session_id:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        try:
            session = sessions.load(self._runtime(), session_id)
        except H3Error as exc:
            return gr.skip(), gr.skip(), gr.skip(), f"**Error:** {exc}", gr.skip()
        return (session.id, presenters.board_value(session), segment_tools.to_rows(session.segments),
                f"Opened **{session.data.get('name') or 'the session'}** with {len(session.segments)} line(s).",
                session.data.get("target_language") or gr.skip())

    # -- Dub ----------------------------------------------------------------

    def _voice_choices(self):
        """The original voice, then every saved Drama 3 speaker."""
        from h3_app.drama3 import speakers as drama3_speakers

        choices = [("Cloned from the original", pipeline.ORIGINAL_VOICE)]
        for entry in drama3_speakers.speaker_names(self._runtime()):
            choices.append(entry if isinstance(entry, tuple) else (entry, entry))
        return choices

    def speakers(self, session_id):
        """Outputs: speakers panel, speaker cards. Shown only when the transcription told speakers apart."""
        session = self._session(session_id)
        if session is None or not pipeline.speaker_settings_available(session):
            return gr.update(visible=False), gr.skip()
        return gr.update(visible=True), presenters.speakers_value(session, self._voice_choices())

    def save_speaker(self, session_id, cards):
        """Save the changed card's voice and volume for the next preview and dub. Outputs: status."""
        session = self._session(session_id)
        cards = cards if isinstance(cards, dict) else {}
        index = cards.get("changed")
        entries = cards.get("speakers") or []
        if session is None or not isinstance(index, int) or not 0 <= index < len(entries):
            return gr.skip()
        speakers = pipeline.session_speakers(session)
        if index >= len(speakers):
            return "**Not saved:** the speakers changed; open the Dub step again."
        entry = entries[index]
        try:
            pipeline.set_speaker(self._runtime(), session.id, speakers[index], voice=str(entry.get("voice") or ""),
                                 volume=float(entry.get("volume", 100)))
        except (H3Error, TypeError, ValueError) as exc:
            return f"**Not saved:** {exc}"
        return ""

    def readiness(self, session_id):
        return presenters.readiness_html(self._session(session_id))

    def clear_results(self):
        """A new session makes the previous one's line picks, preview and dub stale.

        Outputs: line controls, result files/status, export source/button/file/status.
        """
        return ("", "", None, "", gr.update(value="", visible=False), gr.update(value=None, visible=True), None,
                None, "", None, gr.update(interactive=False), None, "")

    def analyze_source(self, source, library, separate, separator_model, transcription, source_language,
                       target_language, translate, translator, aligner, merge_gap, gemini_model, api_key, notes,
                       reuse=True, refine_timestamps=False):
        """Outputs: session id, board, table rows, status, plan. The plan marks the step running."""
        chosen = source or library
        steps = self._plan_steps(separate, separator_model, reuse, transcription, translate, translator,
                                 target_language)
        active = None
        try:
            if not chosen:
                raise H3Error("Upload a video or audio file, or pick one from Media.")
            request = pipeline.AnalyzeRequest(
                str(chosen), transcription, str(target_language or "").strip(), source_language, bool(translate),
                translator, bool(separate), separator_model, bool(aligner), float(merge_gap), gemini_model,
                str(api_key or ""), True, str(notes or ""), bool(reuse), bool(refine_timestamps))
            for update in self._gpu(lambda **kw: pipeline.analyze(request, **kw)):
                if isinstance(update, pipeline.Analysis):
                    session = update.session
                    yield (session.id, presenters.board_value(session), segment_tools.to_rows(session.segments),
                           update.message, presenters.plan_html(steps, finished=True))
                else:
                    active = presenters.step_for_status(update) or active
                    yield gr.skip(), gr.skip(), gr.skip(), update, presenters.plan_html(steps, active=active)
        except Exception as exc:
            yield gr.skip(), gr.skip(), gr.skip(), self._failed(exc), presenters.plan_html(steps)

    def translate_lines(self, session_id, board, target_language, translator, gemini_model, api_key, notes):
        """Translate the picked lines again; with none picked, fill the untranslated ones.

        Outputs: board, table rows, line status.
        """
        try:
            session = sessions.load(self._runtime(), session_id)
            indices = [index for index in presenters.board_picks(board) if index < len(session.segments)] or None
            request = pipeline.AnalyzeRequest("", target_language=str(target_language or "").strip(),
                                              translator=translator, gemini_model=gemini_model,
                                              api_key=str(api_key or ""), notes=str(notes or ""))
            for update in self._gpu(lambda **kw: pipeline.retranslate(session_id, indices, request, **kw)):
                if isinstance(update, pipeline.Analysis):
                    yield (presenters.board_value(update.session, board),
                           segment_tools.to_rows(update.session.segments), update.message)
                else:
                    yield gr.skip(), gr.skip(), update
        except Exception as exc:
            yield gr.skip(), gr.skip(), self._failed(exc)

    def _dub_request(self, session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume=100,
                     background_volume=100, keep_original=True, regenerate=""):
        session = sessions.load(self._runtime(), session_id)
        fresh = tuple(segment_tools.parse_line_numbers(regenerate, len(session.segments)))
        return pipeline.DubRequest(
            session_id, voice_mode=voice_mode, preset=preset, timing=timing, parallel=int(parallel),
            max_speed=float(max_speed), speech_volume=float(speech_volume),
            background_volume=float(background_volume), keep_original_track=bool(keep_original),
            regenerate=fresh)

    def _preview(self, session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, *, fresh: bool):
        try:
            session = sessions.load(self._runtime(), session_id)
            indices = segment_tools.parse_line_numbers(line_numbers, len(session.segments))
            if not indices:
                raise H3Error("Click a line in the table, or enter its number, to preview it.")
            request = self._dub_request(session_id, voice_mode, preset, timing, max_speed, parallel,
                                   regenerate=str(indices[0] + 1) if fresh else "")
            for update in self._gpu(lambda **kw: pipeline.preview_line(request, indices[0], **kw)):
                if isinstance(update, pipeline.LinePreview):
                    # Gradio peak-normalizes float arrays, undoing the speaker's volume setting.
                    yield encode_wav(update.samples, update.rate), update.message
                else:
                    yield gr.skip(), update
        except Exception as exc:
            yield gr.skip(), self._failed(exc)

    def preview_line(self, session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel):
        """Outputs: preview audio, line status. Voices the first listed line, reusing its saved take."""
        yield from self._preview(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, fresh=False)

    def preview_new_take(self, session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel):
        """Outputs: preview audio, line status. A new take of the first listed line, which the dub then uses."""
        yield from self._preview(session_id, line_numbers, voice_mode, preset, timing, max_speed, parallel, fresh=True)

    def generate_dub(self, session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume, background_volume,
                     keep_original, regenerate):
        """Outputs: player, files, status, and a frozen export source for this result."""
        yield (gr.skip(), gr.skip(), gr.skip(), gr.skip(), "Generating dub…", None,
               gr.update(interactive=False), None, "")
        try:
            request = self._dub_request(session_id, voice_mode, preset, timing, max_speed, parallel, speech_volume,
                                   background_volume, keep_original, regenerate)
            for update in self._gpu(lambda **kw: pipeline.dub(request, **kw)):
                if isinstance(update, pipeline.Dub):
                    video = str(update.video) if update.video else None
                    shown = (gr.update(value=player_html(update.player, key=update.audio.stem), visible=True)
                             if update.player is not None else gr.update(value="", visible=False))
                    export = ({"video": video, "subtitles": str(update.subtitles[0])}
                              if video and update.subtitles and not update.subtitles[0].name.endswith(".original.srt")
                              else None)
                    yield (shown, gr.update(value=video, visible=video is not None), str(update.audio),
                           [str(path) for path in update.subtitles], update.message, export,
                           gr.update(interactive=export is not None), None, "")
                else:
                    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), update, *([gr.skip()] * 4)
        except Exception as exc:
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), self._failed(exc), *([gr.skip()] * 4)

    def export_subtitles(self, source):
        """Export the completed dub without regenerating voices or replacing the player."""
        try:
            if not source:
                raise H3Error("Generate a video dub first.")
            yield None, "Exporting video with burned-in translated subtitles…"
            path = pipeline.export_subtitled_video(source["video"], source["subtitles"], runtime=self._runtime())
            yield str(path), "Export saved to Media."
        except Exception as exc:
            yield gr.skip(), self._failed(exc)

    def _cookie_store(self):
        return video_download.CookieStore(video_download.cookies_dir(self._runtime()))

    def cookie_sites(self) -> str:
        sites = self._cookie_store().sites()
        if not sites:
            return "No saved cookies. Public videos need none."
        return "Saved cookies: " + ", ".join(f"**{site}** ({count})" for site, count in sites.items())

    def cookie_site_choices(self):
        return gr.update(choices=list(self._cookie_store().sites()), value=None)

    def import_curl_cookies(self, curl_text, domain):
        """Outputs: status, saved sites, cURL box (cleared on success)."""
        try:
            site, count = video_download.import_curl(self._cookie_store(), curl_text, domain)
        except H3Error as exc:
            return f"**Not saved:** {exc}", gr.skip(), gr.skip()
        return f"Saved {count} cookie(s) for {site}.", self.cookie_sites(), ""

    def import_cookie_file(self, path, domain):
        """Outputs: status, saved sites."""
        try:
            if not path:
                raise H3Error("Choose a cookies.txt file.")
            site, count = video_download.import_cookie_file(self._cookie_store(), path, domain)
        except (H3Error, OSError) as exc:
            return f"**Not saved:** {exc}", gr.skip()
        return f"Saved {count} cookie(s) for {site}.", self.cookie_sites()

    def delete_cookies(self, domain):
        """Outputs: status, saved sites."""
        try:
            deleted = self._cookie_store().delete(domain) if domain else False
        except H3Error as exc:
            return f"**Not deleted:** {exc}", gr.skip()
        return (f"Deleted the cookies for {domain}." if deleted else "Choose a saved site."), self.cookie_sites()

    def video_info(self, url):
        try:
            return video_download.video_info(url, self._runtime())
        except H3Error as exc:
            return f"**Error:** {exc}"

    def _session_key(self, request) -> str:
        return str(getattr(request, "session_hash", None) or "")

    def stop_downloads(self, request: gr.Request = None) -> str:
        """Interrupt this browser session's downloads at their next progress update."""
        with _DOWNLOAD_LOCK:
            running = list(_DOWNLOAD_STOPS.get(self._session_key(request), ()))
        for stop in running:
            stop.set()
        return "Interrupting the download…" if running else "No download is running."

    def download_video(self, url, quality, request: gr.Request = None):
        """Outputs: status, downloaded path (empty until the file is published).

        Downloads hold no GPU, so they run outside the job queue; ``stop_downloads``
        stops yt-dlp at its next progress update and the staged file is discarded.
        """
        from h3_app.dubbing.pipeline import blocking

        key, stop = self._session_key(request), threading.Event()
        with _DOWNLOAD_LOCK:
            _DOWNLOAD_STOPS.setdefault(key, set()).add(stop)
        try:
            work = blocking("Starting the download", lambda report: video_download.download_video(
                url, quality, self._runtime(), report=report, stopped=stop.is_set))
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
