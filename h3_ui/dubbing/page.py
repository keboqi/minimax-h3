"""The Dubbing page and its JSON API, outside Gradio, as Soundstage's is.

The page works on dubbing sessions and saved projects. Every action goes through the same
:class:`DubbingHandlers` methods the published API uses: analysis, line actions, the dub and word
alignment run as owned jobs on the usual job path (in Workshop, which has no GPU, the steps that need
none run directly), each in a page run the page polls, so a run outlives the page that started it.
Line edits save at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import unescape
import json
import math
from pathlib import Path
from threading import RLock
import shutil
import time
from typing import Any, Callable
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.responses import Response

from h3_app.catalog import AUDIO_EXTENSIONS, DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, VIDEO_EXTENSIONS
from h3_app.dubbing import nemotron, pipeline, projects, review_checks, segments as segment_tools, session as sessions, shares
from h3_app.dubbing.alignment import open_alignment
from h3_app.dubbing.lip_sync import (
    REPORT_FILE, SCAN_CACHE_DIR, build_lip_sync_plan_payload, load_lip_sync_plan,
    load_manual_overrides, plan_lip_sync_windows, save_manual_overrides, _save_lip_sync_plan,
)
from h3_app.dubbing.lip_sync_match import assign_speakers_to_shots, load_source_scan, video_fingerprint
from h3_app.dubbing.results import PLAYER_DIR, TRACKS, load_player, load_result, player_folder
from h3_app.dubbing.tools import FAMILY_ANALYZE, FAMILY_DUB, FAMILY_LINES, FAMILY_WORDS
from h3_app.drama3.serving.audio import encode_wav
from h3_app.errors import H3Error
from h3_app.processes import run_media_process
from h3_app.jobs import JOBS, JobCancelled

from ..drama3.page_common import SKIP, PageRun, file_response, found, not_found, run_job, value
from ..job_bindings import owned_generation
from ..page_routes import install_static_page, owner_of, session_of
from ..page_runs import Runs, guarded
from ..player_fullscreen import PLAYER_FULLSCREEN_CSS
from .page_batch import Batches
from . import presenters
from .handlers import DubbingHandlers
from .player import PLAYER_JS, player_html
from .share import share_path

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {name: "application/javascript"
                for name in ("app.js", "api.js", "source.js", "review.js", "dub.js", "lipsync.js", "result.js", "ui.js", "batch.js")}
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
                "rewrite", "rewrite_style", "rewrite_by_hand", "shorten_lines", "readings")
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
ANALYZE, LINES, DUB, WORDS, EXPORT, AUTO = "analyze", "lines", "dub", "words", "export", "auto"
LIP_SYNC, LIP_SYNC_DETECT = "lip-sync", "lip-sync-detect"
FAMILIES = {ANALYZE: FAMILY_ANALYZE, LINES: FAMILY_LINES, DUB: FAMILY_DUB, WORDS: FAMILY_WORDS,
            LIP_SYNC: "dubbing_lip_sync", LIP_SYNC_DETECT: "dubbing_lip_sync_detect"}
BUSY = {ANALYZE: "An analysis is already running.", LINES: "A line action is already running.",
        DUB: "This session is already dubbing.", WORDS: "Words are already being aligned.",
        EXPORT: "An export is already running.", AUTO: "Auto mode is already running.",
        LIP_SYNC: "This project is already running lip sync.",
        LIP_SYNC_DETECT: "Face and speaker detection is already running."}
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
    batch: bool = False
    dub_video: str | None = None

    def public(self, scene_id=None) -> dict[str, Any]:
        return {**super().public(scene_id), "task": self.task, "session": self.session, "step": self.step,
                "audio": self.audio is not None, "file": Path(self.file).name if self.file else None,
                "batch": self.batch, "dub_video": self.dub_video}


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


@lru_cache(maxsize=128)
def _source_duration(source: Path, _size: int, _modified: int, _changed: int) -> float:
    from h3_app.media_tools import probe_audio_seconds

    duration = probe_audio_seconds(source)
    if duration is None:
        raise ValueError("Source duration unavailable.")
    return duration


def source_info(path) -> dict | None:
    """The picked file: its name, kind, size and length."""
    if not path:
        return None
    source = Path(path)
    info = {"path": str(source), "name": source.name, "kind": "Video" if pipeline.is_video(source) else "Audio",
            "size": None, "seconds": None}
    try:
        stat = source.stat()
    except OSError:
        return info
    info["size"] = stat.st_size
    try:
        info["seconds"] = _source_duration(source, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
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


def _result_player_html(session_id, output, player) -> str:
    if player is None:
        return ""
    urls = {getattr(player, track): f"{API}/track?session={session_id}&name={track}&v={_version(getattr(player, track))}"
            for track in TRACKS if getattr(player, track)}
    return player_html(player, key=Path(output).stem, url=urls.__getitem__)


def _lip_sync_payload(sid: str, video: Path | None) -> dict | None:
    """Lip-sync availability, latest output, and any uncertain shots for optional review."""
    if video is None or not video.is_file():
        return None
    folder = player_folder(video)
    ready = (folder / "alignment.json").is_file() and (folder / "alignment.npz").is_file()
    out = video.with_name(f"{video.stem}.lipsync.mp4")
    output = None
    if out.is_file():
        ver = _version(out)
        output = {
            "name": out.name,
            "size": out.stat().st_size,
            "url": f"{API}/output?session={sid}&name={quote(out.name)}&v={ver}",
            "inline_url": f"{API}/output?session={sid}&name={quote(out.name)}&inline=true&v={ver}",
        }
    report = None
    report_path = folder / REPORT_FILE
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
    review_shots = []
    if isinstance(report, dict) and report.get("uncertain_count"):
        scan = None
        try:
            scan = load_source_scan(video.parent / PLAYER_DIR / SCAN_CACHE_DIR, video_fingerprint(video))
        except OSError:
            scan = None
        shots_by_index = {shot.index: shot for shot in (scan.shots if scan else ())}
        for item in report.get("assignments") or ():
            if item.get("status") != "uncertain":
                continue
            shot_index = int(item.get("shot_index", 0))
            shot = shots_by_index.get(shot_index)
            ref_time = item.get("reference_time")
            if ref_time is None and shot is not None:
                ref_time = 0.5 * (shot.start + shot.end)
            tracks = []
            if shot is not None:
                for track in shot.tracks:
                    box = track.representative_box()
                    if box is not None:
                        tracks.append({"track_id": track.track_id, "box": [round(v, 4) for v in box]})
            review_shots.append({
                "shot_index": shot_index,
                "speaker": item.get("speaker", ""),
                "speaker_label": presenters.speaker_label(item.get("speaker", "")),
                "time": round(float(ref_time or 0.0), 3),
                "reason": item.get("reason", ""),
                "frame_url": f"{API}/lip-sync-frame?session={sid}&time={float(ref_time or 0.0):.3f}",
                "tracks": tracks,
            })
    return {
        "video": video.name,
        "ready": ready,
        "output": output,
        "report": {
            "total_windows": report.get("total_windows", 0),
            "cached_windows": report.get("cached_windows", 0),
            "rendered_windows": report.get("rendered_windows", 0),
            "uncertain_count": report.get("uncertain_count", 0),
            "backend": report.get("backend", "none"),
        } if isinstance(report, dict) else None,
        "review_shots": review_shots,
    }


def result_payload(session: sessions.Session, *, include_player: bool = True) -> dict | None:
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
    html = _result_player_html(sid, output, player) if include_player or player is None else None
    cues, timed, failed = 0, 0, []
    if player is not None:
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
            "lip_sync": _lip_sync_payload(sid, result.video),
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
    admission_lock = RLock()
    lip_sync_available = bool(gpu and handlers.services.generation_services is not None)

    def active_runs(owner):
        return [item for collection in runs.values() for item in collection.items.values()
                if item.owner == owner and item.finished_at is None]

    def blocked(run):
        if run.kind == AUTO:
            if any(item.kind != AUTO for item in active_runs(run.owner)):
                return "Wait for the current dubbing action before starting Auto mode."
        elif runs[AUTO].active(run.owner, SOURCE_KEY):
            return "Auto mode is running. Stop it before starting another action."
        if not run.batch and batches.active(run.owner):
            return "A batch is running. Stop it before starting another action."

    runs = {kind: Runs(busy=message, lock=admission_lock, blocked=blocked) for kind, message in BUSY.items()}

    def runtime():
        return handlers._runtime()

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

    def payload(request: Request, session: sessions.Session, *, include_player: bool = True) -> dict:
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
                "speakers": speakers, "readiness": readiness_payload(session),
                "result": result_payload(session, include_player=include_player),
                "share": share_url(request, session), "label": pipeline.dub_label(session)}

    def session_answer(request: Request, session_id, *, include_player: bool = True, **extra) -> dict:
        return {"session": payload(request, load(session_id), include_player=include_player), **extra}

    # The page --------------------------------------------------------------
    @app.get(f"{ROUTE}/player.js", include_in_schema=False)
    def player_script():
        """The result player's script, shared with the share page, as a module the page calls."""
        return Response("export function bindPlayers(element) {\n" + PLAYER_JS + "\n}\n",
                        media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    @app.get(f"{ROUTE}/app.css", include_in_schema=False)
    def dubbing_styles():
        return Response((STATIC / "app.css").read_text(encoding="utf-8") + PLAYER_FULLSCREEN_CSS,
                        media_type="text/css", headers={"Cache-Control": "no-cache"})

    install_static_page(app, ROUTE, STATIC, STATIC_TYPES)

    @app.get(f"{API}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner_of(request)
        return {
            "gpu": gpu, "same": pipeline.SAME_LANGUAGE, "lip_sync": lip_sync_available,
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
                "session": payload(request, session, include_player=body.get("include_player") is not False)}

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
    def read_session(id: str, request: Request, include_player: bool = True):
        owner_of(request)
        return session_answer(request, id, include_player=include_player)

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
            body.get("style") or pipeline.DEFAULT_STYLE, bool(body.get("shorten_lines", False)),
            bool(body.get("readings", False))))
        return {"prompt": prompt}

    @app.post(f"{API}/lines-prompt", include_in_schema=False)
    @guarded
    def lines_prompt(request: Request, body: dict = Body(...)):
        owner_of(request)
        prompt, record, status = gradio_errors(lambda: handlers.lines_prompt(
            body.get("session"), {"picked": body.get("picked") or []}, body.get("target_language"),
            body.get("notes"), body.get("style") or pipeline.DEFAULT_STYLE, bool(body.get("shorten_lines", False)),
            bool(body.get("readings", False))))
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
              read: Callable[[DubRun, tuple], None], *, job=True, session_id=None, dub_video=None) -> dict:
        owner = owner_of(request)
        request_session = session_of(request)
        run = DubRun(owner, key, kind, task=task, session=session_id, dub_video=dub_video)
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
                if status.strip() in ("Interrupted.", "Auto interrupted."):
                    run.stop_requested = True
                    raise JobCancelled("Interrupted.")
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

    def analysis_inputs(body: dict):
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
                  bool(body.get("rewrite", False)), bool(body.get("shorten_lines", False)),
                  bool(body.get("readings", False))]
        if not values[0]:
            raise H3Error("Upload a video or audio file, or pick one from Media.")
        read_frame = reader(3, session_index=0)

        def read_analysis(run, update):
            read_frame(run, update)
            if run.session and handlers.project_selection(project, run.session):
                run.project = project["name"]

        return values, read_analysis

    @app.post(f"{API}/analyze", include_in_schema=False)
    @guarded
    def analyze(request: Request, body: dict = Body(...)):
        values, read_analysis = analysis_inputs(body)
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
                  body.get("style") or pipeline.DEFAULT_STYLE, bool(body.get("rewrite", False)),
                  bool(body.get("shorten_lines", False)), bool(body.get("readings", False))]
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
        fresh = bool(body.get("fresh"))
        indices = body.get("lines") if fresh and "lines" in body else [body.get("line")]
        if (not isinstance(indices, list) or not indices
                or any(not isinstance(index, int) or isinstance(index, bool)
                       or not 0 <= index < len(session.segments) for index in indices)):
            raise H3Error("Pick a line first.")
        callback = handlers.preview_new_take if fresh else handlers.preview_line
        values = [session_id, ",".join(str(index + 1) for index in sorted(set(indices))), *dub_settings(body)]
        return start(request, LINES, session_id, "take" if fresh else "preview", callback, values,
                     reader(1, audio_index=0), session_id=session_id)

    def dub_inputs(session_id, body):
        return [session_id, *dub_settings(body), float(body.get("speech_volume", 100)),
                float(body.get("background_volume", 100)), bool(body.get("keep_original", True)),
                str(body.get("regenerate") or ""), bool(body.get("word_timing", True))]

    @app.post(f"{API}/dub", include_in_schema=False)
    @guarded
    def dub(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("generate the dub"))
        session_id = str(body.get("session") or "")
        load(session_id)
        values = dub_inputs(session_id, body)
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

    def _lip_sync_plan_response(sid: str, loaded: sessions.Session, dub: Path) -> dict[str, Any]:
        ver = _version(dub)
        plan = load_lip_sync_plan(dub)
        overrides = load_manual_overrides(dub)
        out = dub.with_name(f"{dub.stem}.lipsync.mp4")
        output = None
        if out.is_file():
            over = _version(out)
            output = {
                "name": out.name,
                "size": out.stat().st_size,
                "url": f"{API}/output?session={sid}&name={quote(out.name)}&v={over}",
                "inline_url": f"{API}/output?session={sid}&name={quote(out.name)}&inline=true&v={over}",
            }
        speakers = {}
        for line in loaded.segments:
            spk = str(line.get("speaker") or "")
            if spk and spk not in speakers:
                speakers[spk] = presenters.speaker_label(spk)
        return {
            "session": sid,
            "video": dub.name,
            "video_url": f"{API}/output?session={sid}&name={quote(dub.name)}&inline=true&v={ver}",
            "dub_audio_url": f"{API}/track?session={sid}&name=dub&v={ver}",
            "speakers": speakers,
            "plan": plan,
            "overrides": {f"{s}:{spk}": tid for (s, spk), tid in overrides.items()},
            "output": output,
        }

    @app.get(f"{API}/lip-sync-plan", include_in_schema=False)
    @guarded
    def lip_sync_plan(session: str, request: Request):
        owner_of(request)
        loaded = load(session)
        output = loaded.data.get("last_output")
        result = load_result(output) if output else None
        dub = result.video if result else None
        if dub is None or not dub.is_file():
            raise H3Error("Generate a video dub first.")
        return _lip_sync_plan_response(session, loaded, dub)

    @app.post(f"{API}/lip-sync-detect", include_in_schema=False)
    @guarded
    def lip_sync_detect(request: Request, body: dict = Body(...)):
        session_id = str(body.get("session") or "")
        loaded = load(session_id)
        output = loaded.data.get("last_output")
        result = load_result(output) if output else None
        dub = result.video if result else None
        if dub is None or not dub.is_file():
            raise H3Error("Generate a video dub first.")
        force = bool(body.get("force", False))
        return start(
            request, LIP_SYNC_DETECT, session_id, LIP_SYNC_DETECT, handlers.detect_lip_sync,
            [session_id, force], reader(1), session_id=session_id, dub_video=dub.name,
        )

    @app.post(f"{API}/lip-sync-overrides", include_in_schema=False)
    @guarded
    def lip_sync_overrides_save(request: Request, body: dict = Body(...)):
        owner_of(request)
        session_id = str(body.get("session") or "")
        loaded = load(session_id)
        output = loaded.data.get("last_output")
        result = load_result(output) if output else None
        dub = result.video if result else None
        if dub is None or not dub.is_file():
            raise H3Error("Generate a video dub first.")
        merged = load_manual_overrides(dub)
        if body.get("clear"):
            merged = {}
        raw_overrides = body.get("overrides") or []
        if isinstance(raw_overrides, list):
            for item in raw_overrides:
                if not isinstance(item, dict) or "shot_index" not in item or not item.get("speaker"):
                    raise H3Error("Each face override needs shot_index and speaker.")
                key = (int(item["shot_index"]), str(item["speaker"]))
                if item.get("reset"):
                    merged.pop(key, None)
                else:
                    track_id = item.get("track_id")
                    merged[key] = None if track_id is None else int(track_id)
        save_manual_overrides(dub, merged)
        # Recompute cached plan assignments immediately without re-scanning video frames.
        scan = load_source_scan(dub.parent / PLAYER_DIR / SCAN_CACHE_DIR, video_fingerprint(dub))
        if scan is not None:
            try:
                with open_alignment(dub) as (saved, _clips):
                    saved_lines = list(saved.get("lines") or ())
                    saved_spoken = dict(saved.get("spoken") or {})
                    saved_vs = list(saved.get("visual_speakers") or ())
                eff_vs = saved_vs if saved_vs else list((loaded.data.get("transcript") or {}).get("visual_speakers") or ())
                assignments = assign_speakers_to_shots(
                    scan, saved_lines, visual_speakers=eff_vs or None, manual_overrides=merged or None,
                )
                windows = plan_lip_sync_windows(
                    dub, scan.duration, video_size=(scan.width, scan.height), shot_cuts=scan.cuts,
                    assignments=assignments, visual_speakers=eff_vs or None,
                )
                plan = build_lip_sync_plan_payload(
                    dub=dub, scan=scan, lines=saved_lines, spoken=saved_spoken,
                    assignments=assignments, windows=windows,
                )
                _save_lip_sync_plan(dub, plan)
            except (OSError, ValueError, KeyError, TypeError, H3Error):
                pass
        return _lip_sync_plan_response(session_id, loaded, dub)

    @app.post(f"{API}/lip-sync", include_in_schema=False)
    @guarded
    def lip_sync(request: Request, body: dict = Body(...)):
        if not lip_sync_available:
            raise H3Error(STUDIO_ONLY.format("run lip sync"))
        session_id = str(body.get("session") or "")
        loaded = load(session_id)
        output = loaded.data.get("last_output")
        result = load_result(output) if output else None
        dub = result.video if result else None
        if dub is None or not dub.is_file():
            raise H3Error("Generate a video dub first.")
        raw_overrides = body.get("overrides")
        overrides: dict[tuple[int, str], int | None] | None = None
        if isinstance(raw_overrides, list) and raw_overrides:
            overrides = {}
            for item in raw_overrides:
                if not isinstance(item, dict) or "shot_index" not in item or not item.get("speaker"):
                    raise H3Error("Each face override needs shot_index and speaker.")
                track_id = item.get("track_id")
                overrides[(int(item["shot_index"]), str(item["speaker"]))] = (
                    None if track_id is None else int(track_id)
                )
        strength = body.get("strength")
        feather = body.get("feather")
        return start(
            request, LIP_SYNC, session_id, LIP_SYNC, handlers.lip_sync_video,
            [session_id, overrides, strength, feather], reader(1, file_index=0),
            session_id=session_id, dub_video=dub.name,
        )

    @app.get(f"{API}/lip-sync-frame", include_in_schema=False)
    def lip_sync_frame(session: str, time: float, request: Request):
        with found():
            owner_of(request)
            result = latest_result(session)
        dub = result.video
        if dub is None or not dub.is_file() or not math.isfinite(time) or time < 0:
            raise not_found()
        proc = run_media_process(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{float(time):.3f}",
             "-i", str(dub), "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "-q:v", "4", "-"],
            capture_output=True, timeout=15,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise not_found()
        return Response(proc.stdout, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

    def start_auto(owner, request_session, body, *, batch=False):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("run Auto mode"))
        if not batch and batches.active(owner):
            raise H3Error("A batch is running. Stop it before starting another Auto run.")
        body = dict(body)
        existing = load(body["session"]) if body.get("session") else None
        prepared = bool(existing and existing.segments)
        analysis = None
        if not prepared:
            same = body.get("target_language") == pipeline.SAME_LANGUAGE
            if same and body.get("rewrite") and body.get("rewrite_by_hand"):
                raise H3Error("Apply the rewrite by hand in Review lines before starting Auto mode.")
            if same and body.get("rewrite"):
                body["style"] = body.get("rewrite_style") or pipeline.DEFAULT_REWRITE_STYLE
            body["source"] = body.get("source") or (str(existing.path("source")) if existing else None)
            if body.get("pipeline") == pipeline.PIPELINE_MANUAL and not body.get("manual_reply"):
                raise H3Error("Paste the transcription reply or analyze this project before starting Auto mode.")
            analysis = analysis_inputs(body)
        # Validate numeric settings before accepting a run.
        dub_inputs("", body)
        if prepared and body.get("target_language") and existing.data.get("target_language"):
            if body["target_language"] != existing.data["target_language"]:
                raise H3Error("The dub language changed. Use Translate again in Review lines before running Auto mode.")
        run = DubRun(owner, SOURCE_KEY, AUTO, task=AUTO, session=existing.id if existing else None, batch=batch)
        run.project = (body.get("project") or {}).get("name") if prepared else None

        def work(run):
            if analysis:
                run.step = "analyze"
                values, read = analysis
                # A draft's source session is not a completed analysis.
                run.session = None
                for update in frames(run, owner, request_session, ANALYZE, handlers.analyze_source, values):
                    read(run, update)
                if run.error:
                    return
                if not run.session or not load(run.session).segments:
                    raise H3Error("Analysis returned no lines to dub.")
            if run.stop_requested:
                raise JobCancelled("Interrupted.")
            session = load(run.session)
            if any(line.get("dub", True) and not line.get("translation") for line in session.segments):
                raise H3Error("Some lines have no new text. Translate or edit them in Review lines, then run Auto again.")
            settings = dict(body)
            settings.setdefault("voice_mode", pipeline.default_voice_mode(session.data.get("pipeline")))
            if pipeline.same_language(session) and not session.data.get("rewrite"):
                settings.setdefault("timing", pipeline.TIMING_MATCH)
            run.step, run.status = "dub", "Generating dub…"
            read = reader(4, file_index=2)
            for update in frames(run, owner, request_session, DUB, handlers.generate_dub,
                                 dub_inputs(run.session, settings)):
                read(run, update)
            if not run.error and not run.file:
                raise H3Error("Generation finished without a dub. Review the session and try again.")

        return runs[AUTO].start(run, work)

    @app.post(f"{API}/auto", include_in_schema=False)
    @guarded
    def auto(request: Request, body: dict = Body(...)):
        return {"run": start_auto(owner_of(request), session_of(request), body).public()}

    def auto_project(owner, name, request_session, api_key):
        data, session = projects.load(runtime(), name)
        return start_auto(owner, request_session, {**data["settings"], "session": session.id,
                          "project": {"name": name, "session": session.id}, "api_key": api_key}, batch=True)

    batches = Batches(auto_project, (lambda session, run: interrupt(
        session, family=run.family, job_id=run.job_id)) if interrupt else None)

    @app.get(f"{API}/batch", include_in_schema=False)
    @guarded
    def read_batch(request: Request):
        return {"batch": batches.payload(owner_of(request)), "projects": projects.names(runtime())}

    @app.post(f"{API}/batch", include_in_schema=False)
    @guarded
    def start_batch(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error(STUDIO_ONLY.format("batch generate dubs"))
        owner = owner_of(request)
        with admission_lock:
            if active_runs(owner):
                raise H3Error("Wait for the current dubbing action before starting a batch.")
            batches.start(owner, body.get("names"), session_of(request), str(body.get("api_key") or ""))
        return {"batch": batches.payload(owner)}

    @app.post(f"{API}/batch/stop", include_in_schema=False)
    @guarded
    def stop_batch(request: Request):
        owner = owner_of(request)
        return {"message": batches.stop(owner, session_of(request)), "batch": batches.payload(owner)}

    @app.post(f"{API}/export", include_in_schema=False)
    @guarded
    def export(request: Request, body: dict = Body(...)):
        session_id = str(body.get("session") or "")
        session = load(session_id)
        result = load_result(session.data.get("last_output")) if session.data.get("last_output") else None
        if result is None or not result.video or not result.translated_subtitles:
            raise H3Error("Generate a video dub first.")
        mode = str(body.get("subtitles") or "translation")
        if mode not in pipeline.SUBTITLE_MODES:
            raise H3Error("Choose which subtitles to burn in: translation, original or both.")
        source = {"video": str(result.video), "subtitles": str(result.translated_subtitles), "mode": mode}
        return start(request, EXPORT, session_id, "export", handlers.export_subtitles, [source],
                     reader(1, file_index=0), job=False, session_id=session_id)

    @app.get(f"{API}/run", include_in_schema=False)
    @guarded
    def read_runs(request: Request, session: str = ""):
        owner = owner_of(request)
        found = {}
        for kind, items in runs.items():
            key = SOURCE_KEY if kind in (ANALYZE, AUTO) else session
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
        run = runs[kind].active(owner, SOURCE_KEY if kind in (ANALYZE, AUTO) else str(body.get("session") or ""))
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
    @app.get(f"{API}/result-player", include_in_schema=False)
    @guarded
    def result_player(session: str, request: Request):
        owner_of(request)
        loaded = load(session)
        output = loaded.data.get("last_output")
        player = load_player(output) if output else None
        return {"html": _result_player_html(loaded.id, output, player)}

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
    def output(session: str, name: str, request: Request, inline: bool = False):
        with found():
            owner_of(request)
            result = latest_result(session)
        lip_sync_out = (
            result.video.with_name(f"{result.video.stem}.lipsync.mp4")
            if result.video is not None else None
        )
        candidates = [result.video, result.audio, *result.subtitles]
        if lip_sync_out is not None and lip_sync_out.is_file():
            candidates.append(lip_sync_out)
        files = {path.name: path for path in candidates if path is not None}
        if name not in files:
            raise not_found()
        return file_response(files[name], filename=None if inline else name)
