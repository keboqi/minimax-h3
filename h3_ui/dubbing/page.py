"""The Dubbing page and its JSON API, outside Gradio, as Soundstage's is.

The page works on dubbing sessions and saved projects. Every action goes through the same
:class:`DubbingHandlers` methods the published API uses: analysis, line actions, the dub and word
alignment run as owned jobs on the usual job path (in Workshop, which has no GPU, the steps that need
none run directly), each in a page run the page polls, so a run outlives the page that started it.
Line edits save at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response

from h3_app.catalog import AUDIO_EXTENSIONS, DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, VIDEO_EXTENSIONS
from h3_app.dubbing import nemotron, pipeline, projects, review_checks, segments as segment_tools, session as sessions, shares
from h3_app.dubbing.results import TRACKS, load_result
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB, FAMILY_LINES, FAMILY_WORDS
from h3_app.drama3.serving.audio import encode_wav
from h3_app.errors import H3Error
from h3_app.jobs import JOBS, JobCancelled
from h3_app.workspace_store import request_owner

from ..drama3.page_common import SKIP, PageRun, file_response, found, not_found, run_job, value
from ..job_bindings import owned_generation
from ..page_runs import Runs, guarded
from . import presenters
from .handlers import DubbingHandlers
from .player import PLAYER_JS, player_html
from .share import share_path

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {name: "application/javascript"
                for name in ("app.js", "api.js", "source.js", "review.js", "dub.js", "result.js", "ui.js")}
STATIC_TYPES["app.css"] = "text/css"
ROUTE = "/dubbing"
API = f"{ROUTE}/api"
UPLOAD_SUFFIXES = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | {".m4v"}
DEFAULT_SEPARATOR = "Balanced (BS-Roformer)"
UPLOAD_DAYS = 14
# Never kept with a project: API keys. Edited lines and speaker voices live in the session itself.
PROJECT_KEYS = ("source_language", "target_language", "style", "notes", "pipeline", "translate", "translator",
                "gemini_model", "separate", "separator_model", "reuse", "aligner", "merge_gap",
                "refine_timestamps", "translation_context", "voice_mode", "preset", "timing", "max_speed",
                "speech_volume", "background_volume", "keep_original", "regenerate", "parallel", "word_timing",
                "rewrite", "rewrite_style")
TRANSCRIPTIONS = (
    (pipeline.PIPELINE_LOCAL, "Qwen3-ASR", "One speaker, with word timing."),
    (pipeline.PIPELINE_MOSS, "MOSS", "Speakers take turns. Up to 90 minutes."),
    (pipeline.PIPELINE_NEMOTRON, "Nemotron", "Overlapping speakers, with word timing. Up to 60 minutes."),
    (pipeline.PIPELINE_GEMINI, "Gemini (cloud)", "Uploads the vocals. Needs an API key."),
    (pipeline.PIPELINE_MANUAL, "Gemini by hand", "Paste a chat model's reply. No key."),
)
TIMINGS = (
    (pipeline.TIMING_SMART, "Smart timing", "Uses nearby silence and shorter pauses before speeding up."),
    (pipeline.TIMING_MATCH, "Match the original", "Starts and ends with each line. Best for lip-sync."),
    (pipeline.TIMING_NATURAL, "Natural pace", "Keeps each take's own pace; long takes may overlap."),
)
ANALYZE, LINES, DUB, WORDS, EXPORT = "analyze", "lines", "dub", "words", "export"
FAMILIES = {ANALYZE: FAMILY_ANALYZE, LINES: FAMILY_LINES, DUB: FAMILY_DUB, WORDS: FAMILY_WORDS}
BUSY = {ANALYZE: "An analysis is already running.", LINES: "A line action is already running.",
        DUB: "This session is already dubbing.", WORDS: "Words are already being aligned.",
        EXPORT: "An export is already running."}
# Analysis has no session until it finishes: one runs per browser at a time.
SOURCE_KEY = "source"
STUDIO_ONLY = "Open this project in Studio to {} on the GPU."


@dataclass
class DubRun(PageRun):
    """A step of the page: what it is doing, and what it left for the page to fetch."""

    task: str = ""
    session: str | None = None
    step: str | None = None
    audio: bytes | None = None
    file: str | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "task": self.task, "session": self.session, "step": self.step,
                "audio": self.audio is not None, "file": Path(self.file).name if self.file else None}


def _plain(text: str) -> str:
    return unescape(str(text or ""))


def _error(status) -> str | None:
    """A handler's status line when it says nothing was done."""
    text = status if isinstance(status, str) else ""
    for prefix in ("**Error:**", "**Not saved:**", "**Not shared:**", "**Not deleted:**"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return None


def _status(status) -> str:
    text = value(status)
    return "" if text is SKIP or not isinstance(text, str) else text


def source_info(path) -> dict | None:
    """The picked file: its name, kind, size and length."""
    if not path:
        return None
    source = Path(path)
    info = {"path": str(source), "name": source.name, "kind": "Video" if pipeline.is_video(source) else "Audio",
            "size": None, "seconds": None}
    try:
        info["size"] = source.stat().st_size
    except OSError:
        pass
    try:
        from h3_app.media_tools import probe_audio_seconds

        info["seconds"] = probe_audio_seconds(source)
    except Exception:
        pass
    return info


def lines_payload(session: sessions.Session) -> dict:
    """The board's lines with their flags, as plain text for the page to show."""
    board = presenters.board_value(session)
    for item, line in zip(board["lines"], session.segments):
        item.update(source=line["source"], translation=line["translation"], speakerId=line["speaker"],
                    label=presenters.speaker_label(line["speaker"]), problem=_plain(item["problem"]),
                    check=_plain(item["check"]),
                    timing=_plain(item["timing"]), fit=presenters.fit_note(line),
                    edited=not pipeline.unchanged(line))
    for speaker in board["speakers"]:
        speaker["label"] = _plain(speaker["label"])
    board.pop("summary", None)
    return board


def readiness_payload(session: sessions.Session) -> dict:
    lines = session.segments
    return {
        "voiced": sum(bool(s["translation"]) and s.get("dub", True) for s in lines),
        "kept": sum(not s.get("dub", True) for s in lines),
        "untranslated": sum(not s["translation"] and s.get("dub", True) for s in lines),
        "speakers": len({s["speaker"] for s in lines}),
        "failed": sorted(pipeline.take_problems(session)),
        "rushed": sorted(pipeline.timing_problems(session)),
        "long": pipeline.long_lines(lines),
        "issues": presenters.translation_flags(session),
    }


def _version(path) -> str:
    try:
        return str(Path(path).stat().st_mtime_ns)
    except OSError:
        return "0"


def result_payload(session: sessions.Session) -> dict | None:
    """The latest finished dub: its player, files and what can still be done to it."""
    from h3_app.provenance import read_snapshot

    output = session.data.get("last_output")
    result = load_result(output) if output else None
    if result is None:
        return None
    sid = session.id
    files = []
    for kind, path in (("video", result.video), ("audio", result.audio),
                       *(("subtitles", path) for path in result.subtitles)):
        if path is None:
            continue
        files.append({"kind": kind, "name": path.name, "size": path.stat().st_size,
                      "label": "Original subtitles" if path.name.endswith(".original.srt") else
                      {"video": "Dubbed video", "audio": "Dub mix", "subtitles": "Subtitles"}[kind],
                      "url": f"{API}/output?session={sid}&name={quote(path.name)}&v={_version(path)}"})
    player = result.player
    html, cues, timed, failed = "", 0, 0, []
    if player is not None:
        urls = {getattr(player, track): f"{API}/track?session={sid}&name={track}&v={_version(getattr(player, track))}"
                for track in TRACKS if getattr(player, track)}
        html = player_html(player, key=Path(output).stem, url=urls.__getitem__)
        cues = len(player.lines)
        timed = sum(bool(cue.get("translation_words")) for cue in player.lines)
        failed = [cue.get("line_number") for cue in player.lines if cue.get("take_problem")]
    snapshot = read_snapshot(result.audio or result.video) or {}
    settings = snapshot.get("settings") or {}
    return {"html": html, "files": files, "label": getattr(player, "target_language", "") or "",
            "when": time.strftime("%b %d, %H:%M", time.localtime(Path(output).stat().st_mtime))
            if Path(output).exists() else "",
            "duration": getattr(player, "duration", 0.0), "separated": getattr(player, "music", None) is not None,
            "word_timing": bool(getattr(player, "word_timing", False)), "cues": cues, "timed": timed,
            "failed": failed, "export": bool(result.video and result.translated_subtitles),
            "settings": {key: settings.get(key) for key in ("timing", "preset", "voice_mode", "word_timing")}}


def plan_steps(handlers: DubbingHandlers, body: dict) -> list[dict]:
    """The analysis steps for these settings, as the Analyze card lists them."""
    target = str(body.get("target_language") or "").strip()
    same = target == pipeline.SAME_LANGUAGE
    translate = bool(body.get("translate", True))
    steps = handlers._plan_steps(bool(body.get("separate", True)), body.get("separator_model") or DEFAULT_SEPARATOR,
                                 bool(body.get("reuse", True)), body.get("pipeline") or pipeline.PIPELINE_LOCAL,
                                 translate and not same, body.get("translator") or pipeline.TRANSLATOR_HYMT, target)
    if same:
        rewriter = "Gemini" if body.get("translator") == pipeline.TRANSLATOR_GEMINI else "Hy-MT2 (experimental)"
        steps.append((presenters.TRANSLATION,
                      f"Rewrite for {body.get('rewrite_style') or pipeline.DEFAULT_REWRITE_STYLE}"
                      if body.get("rewrite") else "Copy the lines as said",
                      rewriter if body.get("rewrite") else "No translation"))
    return [{"key": key, "title": title, "detail": detail} for key, title, detail in steps]


def _step(status: str) -> str | None:
    text = status.lower()
    if "rewrit" in text or "copying the lines" in text:
        return presenters.TRANSLATION
    return presenters.step_for_status(status)


def install_dubbing_page(app: FastAPI, handlers: DubbingHandlers, *, gpu: bool = True, interrupt=None) -> None:
    """Serve the page at ``/dubbing/`` and its API under ``/dubbing/api``."""
    runs = {kind: Runs(busy=message) for kind, message in BUSY.items()}

    def runtime():
        return handlers._runtime()

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        """What outlives ``request`` and names its owner to the job path."""
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None, headers=dict(request.headers))

    def uploads() -> Path:
        folder = sessions.sessions_dir(runtime()).parent / "dubbing-uploads"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def prune_uploads() -> None:
        """Uploads are copied into a session or project when used; ones untouched for 14 days go."""
        cutoff = time.time() - UPLOAD_DAYS * 86400
        for entry in uploads().iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue

    def media() -> list[dict]:
        found = []
        for mode in ("Video", "Audio"):
            try:
                paths = handlers.services.list_media(mode)
            except Exception:
                paths = []
            found += [{"label": f"{mode} · {Path(path).name}", "path": str(Path(path).resolve())}
                      for path in paths[:200]]
        return found

    def checked_source(path) -> str | None:
        """A source the page may use: an upload, a Media file or a session's own copy; never any other path."""
        if not path:
            return None
        chosen = Path(str(path)).resolve()
        allowed = (uploads().resolve(), sessions.sessions_dir(runtime()).resolve())
        if chosen.is_file() and (any(chosen.is_relative_to(root) for root in allowed)
                                 or str(chosen) in {item["path"] for item in media()}):
            return str(chosen)
        raise H3Error("Pick the source again: that file is no longer available.")

    def load(session_id) -> sessions.Session:
        if not session_id:
            raise H3Error("Analyze a source first.")
        return sessions.load(runtime(), str(session_id))

    def origin(request: Request) -> str:
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        return f"{request.headers.get('x-forwarded-proto') or request.url.scheme}://{host}" if host else ""

    def share_url(request: Request, session: sessions.Session) -> str | None:
        output = session.data.get("last_output")
        token = shares.find(runtime(), output) if output and not gpu else None
        return origin(request) + share_path(token) if token else None

    def payload(request: Request, session: sessions.Session) -> dict:
        speakers = None
        if pipeline.speaker_settings_available(session):
            cards = presenters.speakers_value(session, handlers._voice_choices())
            speakers = {"cards": [{**card, "label": _plain(card["label"]), "detail": _plain(card["detail"]),
                                   "initials": _plain(card["initials"]), "voice": _plain(card["voice"])}
                                  for card in cards["speakers"]],
                        "voices": [[_plain(label), _plain(key)] for label, key in cards["voices"]]}
        data = session.data
        source = source_info(session.path("source"))
        if source and data.get("name"):
            # A session keeps its copy as "source.<ext>"; show the name it was analyzed under.
            source["name"] = str(data["name"]) + Path(source["name"]).suffix
        return {"id": session.id, "name": data.get("name") or "", "target_language": data.get("target_language") or "",
                "source_language": data.get("source_language") or "", "spoken": pipeline.spoken_language(session),
                "same": pipeline.same_language(session), "rewrite": bool(data.get("rewrite")),
                "rewrite_style": data.get("rewrite_style") or "", "pipeline": data.get("pipeline") or "",
                "voice_default": pipeline.default_voice_mode(data.get("pipeline")),
                "source": source, "board": lines_payload(session),
                "speakers": speakers, "readiness": readiness_payload(session), "result": result_payload(session),
                "share": share_url(request, session), "label": pipeline.dub_label(session)}

    def session_answer(request: Request, session_id, **extra) -> dict:
        return {"session": payload(request, load(session_id)), **extra}

    # The page --------------------------------------------------------------
    @app.get(ROUTE, include_in_schema=False)
    def dubbing_redirect():
        return RedirectResponse(f"{ROUTE}/")

    @app.get(f"{ROUTE}/", include_in_schema=False)
    def dubbing_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get(f"{ROUTE}/player.js", include_in_schema=False)
    def player_script():
        """The result player's script, shared with the share page, as a module the page calls."""
        return Response("export function bindPlayers(element) {\n" + PLAYER_JS + "\n}\n",
                        media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    @app.get(ROUTE + "/{name}", include_in_schema=False)
    def dubbing_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner_of(request)
        return {
            "gpu": gpu, "same": pipeline.SAME_LANGUAGE,
            "source_languages": list(pipeline.SOURCE_LANGUAGES), "target_languages": list(pipeline.TARGET_LANGUAGES),
            "styles": list(pipeline.TRANSLATION_STYLES), "default_style": pipeline.DEFAULT_STYLE,
            "rewrite_styles": list(pipeline.REWRITE_STYLES), "default_rewrite_style": pipeline.DEFAULT_REWRITE_STYLE,
            "pipelines": [{"key": key, "label": label, "desc": desc, "cloud": key in pipeline.CLOUD_PIPELINES}
                          for key, label, desc in TRANSCRIPTIONS
                          if (gpu or key in pipeline.CLOUD_PIPELINES)
                          and (key != pipeline.PIPELINE_NEMOTRON or nemotron.supported())],
            "translators": [{"key": pipeline.TRANSLATOR_HYMT, "label": "Hy-MT2 1.8B"},
                            {"key": pipeline.TRANSLATOR_GEMINI, "label": "Gemini"}],
            "separators": list(pipeline.SEPARATOR_MODELS), "default_separator": DEFAULT_SEPARATOR,
            "gemini_models": list(GEMINI_PROMPT_MODELS), "default_gemini_model": DEFAULT_GEMINI_PROMPT_MODEL,
            "voice_modes": list(pipeline.VOICE_MODES), "presets": list(pipeline.VOICE_PRESETS),
            "timings": [{"key": key, "label": label, "desc": desc} for key, label, desc in TIMINGS],
            "projects": projects.names(runtime()),
        }

    # Sources ---------------------------------------------------------------
    @app.get(f"{API}/media", include_in_schema=False)
    @guarded
    def list_media(request: Request):
        owner_of(request)
        return {"media": media()}

    @app.get(f"{API}/recent", include_in_schema=False)
    @guarded
    def recent(request: Request):
        owner_of(request)
        return {"sessions": [{**item, "name": _plain(item["name"]), "detail": _plain(item["detail"]),
                              "when": _plain(item["when"])}
                             for item in presenters.recent_value(runtime())["sessions"]]}

    @app.post(f"{API}/upload", include_in_schema=False)
    @guarded
    def upload(request: Request, file: UploadFile = File(...)):
        owner_of(request)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise H3Error(f"{file.filename} isn't a video or audio file.")
        prune_uploads()
        folder = uploads() / uuid.uuid4().hex
        folder.mkdir()
        path = folder / Path(file.filename).name
        with path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        return {"source": source_info(path)}

    @app.get(f"{API}/source", include_in_schema=False)
    @guarded
    def source_details(path: str, request: Request):
        owner_of(request)
        return {"source": source_info(checked_source(path))}

    # Projects and sessions -------------------------------------------------
    @app.get(f"{API}/projects", include_in_schema=False)
    @guarded
    def project_names(request: Request):
        owner_of(request)
        return {"projects": projects.names(runtime())}

    @app.post(f"{API}/project/load", include_in_schema=False)
    @guarded
    def load_project(request: Request, body: dict = Body(...)):
        owner_of(request)
        name = str(body.get("name") or "")
        data, session = projects.load(runtime(), name)
        return {"project": {"name": name, "session": session.id}, "settings": data["settings"],
                "session": payload(request, session)}

    @app.post(f"{API}/project/save", include_in_schema=False)
    @guarded
    def save_project(request: Request, body: dict = Body(...)):
        owner_of(request)
        name = str(body.get("name") or "").strip()
        settings = {key: value for key, value in (body.get("settings") or {}).items() if key in PROJECT_KEYS}
        session = projects.save(runtime(), name, body.get("session") or None,
                                checked_source(body.get("source")), settings)
        return {"project": {"name": name, "session": session.id}, "projects": projects.names(runtime()),
                "session": payload(request, session)}

    @app.get(f"{API}/session", include_in_schema=False)
    @guarded
    def read_session(id: str, request: Request):
        owner_of(request)
        return session_answer(request, id)

    # Lines -----------------------------------------------------------------
    def edited(request, session_id, answer, *, status_index, board_index=0):
        status = _status(answer[status_index])
        problem = _error(status)
        if problem:
            raise H3Error(problem)
        board = value(answer[board_index])
        focus = board.get("focus") if isinstance(board, dict) else None
        return session_answer(request, session_id, focus=focus, status=status)

    @app.patch(f"{API}/line", include_in_schema=False)
    @guarded
    def save_line(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id, key = body.get("session"), body.get("key") or {}
        board = {"session": session_id, "focus": key.get("index")}
        answer = handlers.save_focused(session_id, key, board, body.get("start"), body.get("end"),
                                       body.get("speaker"), body.get("original"), body.get("translation"),
                                       bool(body.get("dub", True)))
        return edited(request, session_id, answer, status_index=4)

    @app.post(f"{API}/line/dub", include_in_schema=False)
    @guarded
    def switch_dub(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id, line = body.get("session"), body.get("line")
        board = {"session": session_id, "rev": body.get("rev"), "focus": line, "changed": {"line": line, "dub": bool(body.get("dub"))}}
        return edited(request, session_id, handlers.toggle_dub(session_id, board), status_index=2)

    @app.post(f"{API}/line/add", include_in_schema=False)
    @guarded
    def add_line(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        board = {"session": session_id, "rev": body.get("rev"), "focus": body.get("focus")}
        return edited(request, session_id, handlers.add_line(session_id, board), status_index=2)

    @app.post(f"{API}/line/delete", include_in_schema=False)
    @guarded
    def delete_line(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        board = {"session": session_id, "rev": body.get("rev"), "focus": body.get("focus")}
        return edited(request, session_id, handlers.delete_line(session_id, board), status_index=2)

    @app.post(f"{API}/line/merge", include_in_schema=False)
    @guarded
    def merge_lines(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        picked = body.get("picked") or []
        board = {"session": session_id, "rev": body.get("rev"), "focus": min(picked) if picked and all(isinstance(n, int) for n in picked)
                 else None}
        return edited(request, session_id, handlers.merge_lines(session_id, board, picked), status_index=2)

    @app.post(f"{API}/line/merge-groups", include_in_schema=False)
    @guarded
    def merge_groups(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        groups = body.get("groups") or []
        board = {"session": session_id, "rev": body.get("rev"), "focus": None}
        answer = handlers.merge_groups(session_id, board, groups, relabel=bool(body.get("relabel")))
        return edited(request, session_id, answer, status_index=2)

    @app.post(f"{API}/combine-suggestions", include_in_schema=False)
    @guarded
    def combine_suggestions(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = str(body.get("session") or "")
        load(session_id)
        groups, by_context, status = review_checks.detect_combine(
            session_id, runtime=runtime(), context=bool(body.get("context", True)),
            api_key=str(body.get("api_key") or ""))
        return {"groups": groups, "context": by_context, "status": status}

    @app.post(f"{API}/check-translations", include_in_schema=False)
    @guarded
    def check_translations(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = str(body.get("session") or "")
        picked = [n for n in body.get("picked") or [] if isinstance(n, int) and not isinstance(n, bool)]
        _session, status = review_checks.check_translations(session_id, picked or None, runtime=runtime(),
                                                            api_key=str(body.get("api_key") or ""))
        return session_answer(request, session_id, status=status)

    @app.put(f"{API}/lines", include_in_schema=False)
    @guarded
    def save_table(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        rows = body.get("rows") or []
        _rows, status = handlers.save_lines(session_id, rows)
        problem = _error(status)
        if problem:
            raise H3Error(problem)
        unfinished = [row for row in rows if row and row[0] is None and segment_tools.row_segment(row) is None]
        return session_answer(request, session_id, status=_plain(status), unfinished=unfinished)

    @app.post(f"{API}/speaker", include_in_schema=False)
    @guarded
    def save_speaker(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id, index = body.get("session"), body.get("index")
        if not isinstance(index, int) or index < 0:
            raise H3Error("Pick a speaker.")
        entries = [{} for _ in range(index)] + [{"voice": body.get("voice") or "",
                                                 "volume": body.get("volume", 100)}]
        problem = _error(handlers.save_speaker(session_id, {"changed": index, "speakers": entries}))
        if problem:
            raise H3Error(problem)
        return session_answer(request, session_id)

    @app.get(f"{API}/original", include_in_schema=False)
    def original_audio(session: str, line: int, request: Request):
        with found():
            owner_of(request)
            clip = pipeline.original_line(session, int(line), runtime=runtime())
        return Response(encode_wav(clip.samples, clip.rate), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    # By hand in a chat model ------------------------------------------------
    def gradio_errors(call: Callable):
        try:
            return call()
        except H3Error:
            raise
        except Exception as exc:  # gr.Error carries the message to show.
            if type(exc).__name__ == "Error":
                raise H3Error(str(getattr(exc, "message", exc))) from exc
            raise

    @app.post(f"{API}/source-prompt", include_in_schema=False)
    @guarded
    def source_prompt(request: Request, body: dict = Body(...)):
        owner_of(request)
        prompt = gradio_errors(lambda: handlers.source_prompt(
            body.get("target_language"), bool(body.get("translate", True)), body.get("notes"),
            body.get("style") or pipeline.DEFAULT_STYLE))
        return {"prompt": prompt}

    @app.post(f"{API}/lines-prompt", include_in_schema=False)
    @guarded
    def lines_prompt(request: Request, body: dict = Body(...)):
        owner_of(request)
        prompt, record, status = gradio_errors(lambda: handlers.lines_prompt(
            body.get("session"), {"picked": body.get("picked") or []}, body.get("target_language"),
            body.get("notes"), body.get("style") or pipeline.DEFAULT_STYLE))
        return {"prompt": prompt, "request": record, "status": status}

    @app.post(f"{API}/lines-reply", include_in_schema=False)
    @guarded
    def lines_reply(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = body.get("session")
        answer = handlers.apply_lines_reply(session_id, body.get("request"), body.get("reply"),
                                            {"session": session_id})
        return edited(request, session_id, answer, status_index=2)

    @app.post(f"{API}/plan", include_in_schema=False)
    @guarded
    def plan(request: Request, body: dict = Body(...)):
        owner_of(request)
        return {"steps": plan_steps(handlers, body)}

    # Runs ------------------------------------------------------------------
    def frames(run: DubRun, owner, request_session, kind: str, callback: Callable, values: list, *, job=True):
        """The handler's output frames: as an owned job with a GPU, else directly in this run."""
        if job and gpu:
            yield from run_job(run, owner, request_session, FAMILIES[kind],
                               values, owned_generation(callback, FAMILIES[kind]))
            return
        updates = callback(*values)
        try:
            for update in updates:
                if run.stop_requested:
                    raise JobCancelled("Interrupted.")
                yield update
        finally:
            updates.close()

    def start(request: Request, kind: str, key: str, task: str, callback: Callable, values: list,
              read: Callable[[DubRun, tuple], None], *, job=True, session_id=None) -> dict:
        owner = owner_of(request)
        request_session = session_of(request)
        run = DubRun(owner, key, kind, task=task, session=session_id)
        run.status = "Starting…"

        def work(run: DubRun):
            for update in frames(run, owner, request_session, kind, callback, values, job=job):
                read(run, update)
            if run.error is None and not run.message:
                run.status = run.status if run.status and run.status != "Starting…" else "Done."

        runs[kind].start(run, work)
        return {"run": run.public()}

    def reader(status_index: int, *, session_index: int | None = None, audio_index: int | None = None,
               file_index: int | None = None) -> Callable[[DubRun, tuple], None]:
        def read(run: DubRun, update: tuple) -> None:
            status = _status(update[status_index])
            if status:
                problem = _error(status)
                if problem:
                    run.error = problem
                    run.status = "Stopped."
                else:
                    run.status = _plain(status).replace("**", "").strip()
                    run.step = _step(status) or run.step
            if session_index is not None and (found := value(update[session_index])) not in (SKIP, None, ""):
                run.session = str(found)
            if audio_index is not None and isinstance(found := value(update[audio_index]), bytes):
                run.audio = found
            if file_index is not None and isinstance(found := value(update[file_index]), str) and found:
                run.file = found
            run.revision += 1

        return read

    def picked(body: dict) -> dict:
        return {"picked": [int(index) for index in body.get("picked") or [] if isinstance(index, int)]}

    @app.post(f"{API}/analyze", include_in_schema=False)
    @guarded
    def analyze(request: Request, body: dict = Body(...)):
        project = body.get("project")
        if not isinstance(project, dict) or not project.get("name"):
            project = None
        values = [checked_source(body.get("source")), None, bool(body.get("separate", True)),
                  body.get("separator_model") or DEFAULT_SEPARATOR, body.get("pipeline") or pipeline.PIPELINE_LOCAL,
                  body.get("source_language") or "Auto-detect", body.get("target_language") or "",
                  bool(body.get("translate", True)), body.get("translator") or pipeline.TRANSLATOR_HYMT,
                  bool(body.get("aligner", True)), float(body.get("merge_gap", 0.35)),
                  body.get("gemini_model") or DEFAULT_GEMINI_PROMPT_MODEL, body.get("api_key") or "",
                  body.get("notes") or "", bool(body.get("reuse", True)), bool(body.get("refine_timestamps", True)),
                  bool(body.get("translation_context", False)),
                  project,
                  body.get("manual_reply") or "", body.get("style") or pipeline.DEFAULT_STYLE,
                  bool(body.get("rewrite", False))]
        if not values[0]:
            raise H3Error("Upload a video or audio file, or pick one from Media.")
        read_frame = reader(3, session_index=0)

        def read_analysis(run, update):
            read_frame(run, update)
            if run.session and handlers.project_selection(project, run.session):
                run.project = project["name"]

        return start(request, ANALYZE, SOURCE_KEY, "analyze", handlers.analyze_source, values, read_analysis)

    @app.post(f"{API}/prepare-audio", include_in_schema=False)
    @guarded
    def prepare_audio(request: Request, body: dict = Body(...)):
        values = [checked_source(body.get("source")), None, bool(body.get("separate", True)),
                  body.get("separator_model") or DEFAULT_SEPARATOR, bool(body.get("reuse", True)),
                  body.get("target_language") or ""]
        if not values[0]:
            raise H3Error("Upload a video or audio file, or pick one from Media.")
        return start(request, ANALYZE, SOURCE_KEY, "prepare", handlers.prepare_speech, values,
                     reader(1, file_index=0))

    @app.post(f"{API}/translate", include_in_schema=False)
    @guarded
    def translate(request: Request, body: dict = Body(...)):
        session_id = str(body.get("session") or "")
        load(session_id)
        values = [session_id, picked(body), body.get("target_language") or "",
                  body.get("translator") or pipeline.TRANSLATOR_HYMT,
                  body.get("gemini_model") or DEFAULT_GEMINI_PROMPT_MODEL, body.get("api_key") or "",
                  body.get("notes") or "", bool(body.get("translation_context", False)),
                  body.get("style") or pipeline.DEFAULT_STYLE, bool(body.get("rewrite", False))]
        return start(request, LINES, session_id, "translate", handlers.translate_lines, values, reader(2),
                     session_id=session_id)

    def dub_settings(body: dict) -> list:
        return [body.get("voice_mode") or pipeline.VOICE_LINE, body.get("preset") or "Balance",
                body.get("timing") or pipeline.TIMING_SMART, float(body.get("max_speed", 1.35)),
                int(body.get("parallel", 16))]

    @app.post(f"{API}/preview", include_in_schema=False)
    @guarded
    def preview(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("preview a line"))
        session_id = str(body.get("session") or "")
        session = load(session_id)
        line = body.get("line")
        if not isinstance(line, int) or not 0 <= line < len(session.segments):
            raise H3Error("Pick a line first.")
        callback = handlers.preview_new_take if body.get("fresh") else handlers.preview_line
        values = [session_id, str(line + 1), *dub_settings(body)]
        return start(request, LINES, session_id, "take" if body.get("fresh") else "preview", callback, values,
                     reader(1, audio_index=0), session_id=session_id)

    @app.post(f"{API}/dub", include_in_schema=False)
    @guarded
    def dub(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("generate the dub"))
        session_id = str(body.get("session") or "")
        load(session_id)
        values = [session_id, *dub_settings(body), float(body.get("speech_volume", 100)),
                  float(body.get("background_volume", 100)), bool(body.get("keep_original", True)),
                  str(body.get("regenerate") or ""), bool(body.get("word_timing", True))]
        return start(request, DUB, session_id, "dub", handlers.generate_dub, values, reader(4),
                     session_id=session_id)

    @app.post(f"{API}/align", include_in_schema=False)
    @guarded
    def align(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("align words"))
        session_id = str(body.get("session") or "")
        load(session_id)
        return start(request, WORDS, session_id, "words", handlers.align_words, [session_id], reader(1),
                     session_id=session_id)

    @app.post(f"{API}/export", include_in_schema=False)
    @guarded
    def export(request: Request, body: dict = Body(...)):
        session_id = str(body.get("session") or "")
        session = load(session_id)
        result = load_result(session.data.get("last_output")) if session.data.get("last_output") else None
        if result is None or not result.video or not result.translated_subtitles:
            raise H3Error("Generate a video dub first.")
        source = {"video": str(result.video), "subtitles": str(result.translated_subtitles)}
        return start(request, EXPORT, session_id, "export", handlers.export_subtitles, [source],
                     reader(1, file_index=0), job=False, session_id=session_id)

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, session: str = ""):
        owner = owner_of(request)
        found = {}
        for kind, items in runs.items():
            key = SOURCE_KEY if kind == ANALYZE else session
            run = items.latest(owner, key) if key else None
            found[kind] = run.public() if run is not None else None
        return {"runs": found}

    def owned_run(request: Request, run_id: str) -> DubRun:
        owner = owner_of(request)
        for items in runs.values():
            run = items.items.get(run_id)
            if run is not None and run.owner == owner:
                return run
        raise not_found()

    @app.get(f"{API}/run-audio", include_in_schema=False)
    def run_audio(id: str, request: Request):
        with found():
            run = owned_run(request, id)
        if run.audio is None:
            raise not_found()
        return Response(run.audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get(f"{API}/run-file", include_in_schema=False)
    def run_file(id: str, request: Request):
        with found():
            run = owned_run(request, id)
        if not run.file:
            raise not_found()
        return file_response(run.file, cache="no-store", filename=Path(run.file).name)

    @app.post(f"{API}/interrupt", include_in_schema=False)
    @guarded
    def stop(request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        kind = body.get("kind")
        if kind not in runs:
            raise H3Error("Unknown step.")
        run = runs[kind].active(owner, SOURCE_KEY if kind == ANALYZE else str(body.get("session") or ""))
        if run is None:
            return {"message": "Nothing is running.", "run": None}
        run.stop_requested = True
        if run.job_id is not None and interrupt is not None:
            message = interrupt(session_of(request), family=run.family or FAMILIES.get(kind), job_id=run.job_id)
        else:
            message = "Interrupting…"
        return {"message": message, "run": run.public()}

    # Share (Workshop serves the links) ------------------------------------
    @app.post(f"{API}/share", include_in_schema=False)
    @guarded
    def create_share(request: Request, body: dict = Body(...)):
        owner_of(request)
        if gpu:
            raise H3Error("Share links are made in Workshop. Open this project there.")
        session = load(body.get("session"))
        output = session.data.get("last_output")
        if not output:
            raise H3Error("Generate the dub in Studio first.")
        shares.create(runtime(), Path(output), str(session.data.get("name") or ""))
        return {"session": payload(request, session)}

    @app.delete(f"{API}/share", include_in_schema=False)
    @guarded
    def stop_share(request: Request, session: str):
        owner_of(request)
        loaded = load(session)
        output = loaded.data.get("last_output")
        if output:
            shares.revoke(runtime(), Path(output))
        return {"session": payload(request, loaded)}

    # Files -----------------------------------------------------------------
    def latest_result(session_id: str):
        session = load(session_id)
        output = session.data.get("last_output")
        result = load_result(output) if output else None
        if result is None:
            raise not_found()
        return result

    @app.get(f"{API}/track", include_in_schema=False)
    def track(session: str, name: str, request: Request):
        with found():
            owner_of(request)
            player = latest_result(session).player
        path = getattr(player, name, None) if player is not None and name in TRACKS else None
        return file_response(path)

    @app.get(f"{API}/output", include_in_schema=False)
    def output(session: str, name: str, request: Request):
        with found():
            owner_of(request)
            result = latest_result(session)
        files = {path.name: path for path in (result.video, result.audio, *result.subtitles) if path is not None}
        if name not in files:
            raise not_found()
        return file_response(files[name], filename=name)
