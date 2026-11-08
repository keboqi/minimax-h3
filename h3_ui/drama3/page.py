"""The Soundstage page and its JSON API, outside Gradio, as Remake's and Storyboard's are.

It works on saved Soundstage projects: the page reads a project, plays its mix with the
timeline and lines in the browser, and speaks, remixes or regenerates a line from the project
as last saved. A run takes the same job path as every other generation, in a thread here, and
the page polls it, so it outlives the page that started it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from h3_app.drama3.results import result_settings
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.prompt_writers.drama3 import DRAMA3_PROMPT_FAILED
from h3_app.workspace_store import request_owner

from ..job_admission import execute_accepted
from ..job_bindings import owned_generation
from ..storyboard.page import Run, Runs, guarded
from ..subtitle_words import word_pieces
from .controls import DRAMA3_SAMPLING_PRESETS, DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND, UPLOAD_VOICE
from .page_cast import boot_payload as cast_boot_payload, cast_settings, install_cast_routes, sounds_payload, speakers_payload
from .page_script import boot_payload, install_script_routes, rewrite_project, script_payload
from .result_view import clean_line, timeline_data

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
GENERATE, REMIX, REGENERATE = "generate", "remix", "regenerate"
# The Speak step's defaults, for a project saved before a field existed.
DEFAULTS = {"temperature": 2.0, "top_p": 1.0, "top_k": 30, "max_frames": 8192, "parallel": 16, "retries": 0,
            "word_timing": False, "script_mode": "Basic", "text": "", "title": ""}
SETTING_FIELDS = {"temperature": float, "top_p": float, "top_k": int, "word_timing": bool, "parallel": int,
                  "retries": int, "max_frames": int, "auto_save_speakers": bool}
# The Speak step's slider ranges, which the page draws and saved values are kept inside.
SETTING_RANGES = {"temperature": (0.0, 2.0, 0.05), "top_p": (0.05, 1.0, 0.05), "top_k": (0, 4096, 1),
                  "parallel": (1, 32, 1), "retries": (0, 4, 1), "max_frames": (1, 8192, 1)}
RUN_KIND = "speak"


@dataclass
class SpeakRun(Run):
    """Speaking, or Auto, which rewrites, makes the cast and speaks in turn (``stage``)."""

    auto: bool = False
    stage: str | None = None
    family: str | None = None  # The running job's family, which Interrupt names.
    slots: dict[str, str] = field(default_factory=dict)
    stop_requested: bool = False

    def public(self, scene_id) -> dict[str, Any]:
        return {**super().public(scene_id), "auto": self.auto, "stage": self.stage, "slots": dict(self.slots)}


def _value(data: dict, key: str):
    value = data.get(key)
    return DEFAULTS.get(key) if value is None else value


def _clip(clips: Path, value) -> str | None:
    """A project clip's path, or None when it has none or the file is gone."""
    if not value:
        return None
    clip = clips / Path(str(value)).name
    return str(clip) if clip.is_file() else None


def generation_values(name: str, data: dict, clips: Path) -> list[Any]:
    """A saved project as ``generate_drama3``'s inputs, in the order the Speak step sends them."""
    values: list[Any] = [_value(data, "text")]
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_preset") or UPLOAD_VOICE, _clip(clips, data.get(f"voice_{slot}_audio")),
                   data.get(f"voice_{slot}_transcript") or ""]
    values += [_value(data, key) for key in ("temperature", "top_p", "top_k", "max_frames", "parallel", "retries",
                                             "script_mode")]
    for slot in range(DRAMA3_SOUND_SLOTS):
        values += [data.get(f"sound_{slot}_preset") or UPLOAD_SOUND, _clip(clips, data.get(f"sound_{slot}_audio")),
                   data.get(f"sound_{slot}_save_name") or "", data.get(f"sound_{slot}_description") or "",
                   data.get(f"sound_{slot}_duration") or 5]
    values.append(_value(data, "title"))
    values += [data.get(f"voice_{slot}_save_name") or "" for slot in range(DRAMA3_VOICE_SLOTS)]
    values += [bool(_value(data, "word_timing")), name]
    return values


def _take_status(chunk: dict) -> str:
    if not chunk.get("ok", True) or chunk.get("warnings"):
        return "flagged"
    return "reused" if chunk.get("reused") else "new"


def timeline_payload(settings: dict) -> dict:
    """A snapshot's timed lines (with their aligned words), speakers and sounds, as the page draws them."""
    entries, words, tones, sounds = timeline_data(settings)
    lines = []
    for (speaker, text, start, end), line_words in zip(entries, words):
        pieces = word_pieces(text, line_words)
        lines.append({"speaker": speaker, "text": text, "start": start, "end": end,
                      "words": None if pieces is None else {
                          "lead": pieces[0], "items": [{"text": piece, "start": s, "end": e}
                                                       for piece, s, e in pieces[1]]}})
    return {
        "duration": float(settings.get("duration") or 0),
        "speakers": [{"name": name, "tone": tone, "pan": pan} for name, (tone, pan) in tones.items()],
        "lines": lines,
        "sounds": [{"name": name, "loop": loop, "start": start, "end": end} for name, loop, start, end in sounds],
    }


def result_payload(path: str | Path | None) -> dict | None:
    """What the page draws for a mix: its lines, sounds and takes on the timeline."""
    if not path or not Path(path).is_file():
        return None
    settings = result_settings(path)
    takes = []
    for chunk in settings.get("chunks") or []:
        if not chunk.get("take"):
            continue
        takes.append({"key": chunk["take"], "number": len(takes) + 1,
                      "speaker": str(chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}"),
                      "text": clean_line(chunk.get("text", "")), "seconds": chunk.get("seconds"),
                      "status": _take_status(chunk), "warnings": list(chunk.get("warnings") or []),
                      "start": chunk.get("start"), "end": chunk.get("end")})
    file = Path(path)
    return {**timeline_payload(settings), "file": file.name, "version": file.stat().st_mtime_ns,
            "title": settings.get("title") or "", "word_timing": bool(settings.get("word_timing")), "takes": takes}


def _preset(data: dict) -> str | None:
    current = tuple(float(_value(data, key)) for key in ("temperature", "top_p", "top_k"))
    return next((name for name, values in DRAMA3_SAMPLING_PRESETS.items()
                 if tuple(float(value) for value in values) == current), None)


def project_payload(name: str, data: dict, result, clips: Path, runtime=None) -> dict:
    text = str(_value(data, "text"))
    speakers = speakers_payload(name, data, lambda value: _clip(clips, value), runtime)
    return {
        "name": name,
        "title": _value(data, "title"),
        "script_mode": _value(data, "script_mode"),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        # The speakers who are someone: named, or with a voice.
        "cast": [speaker for speaker in speakers if speaker["save_name"] or speaker["saved_name"] or speaker["ready"]],
        "speakers": speakers,
        "sounds": sounds_payload(name, data, lambda value: _clip(clips, value)),
        "cast_settings": cast_settings(data),
        "settings": {"temperature": float(_value(data, "temperature")), "top_p": float(_value(data, "top_p")),
                     "top_k": int(_value(data, "top_k")), "word_timing": bool(_value(data, "word_timing")),
                     "parallel": int(_value(data, "parallel")), "retries": int(_value(data, "retries")),
                     "max_frames": int(_value(data, "max_frames")),
                     "auto_save_speakers": bool(data.get("auto_save_speakers")), "preset": _preset(data)},
        "result": result_payload(result),
        "script": script_payload(data),
    }


def install_soundstage_page(app: FastAPI, actions, *, store, output_dir, takes_dir: Callable[[], Path],
                            runtime_config: Callable[[], Any] | None = None, gpu: bool = True,
                            interrupt=None) -> None:
    """Serve the page at ``/soundstage/`` and its API under ``/soundstage/api``.

    ``actions`` holds Soundstage's speech actions; ``store`` keeps the saved projects and
    ``output_dir`` their clips, as the Soundstage tab saves them; ``takes_dir`` is where line
    takes are kept.
    """
    runs = Runs(busy="This project is already speaking.")
    clips = Path(output_dir) / "h3" / ".soundstage" / "clips"
    callbacks = {
        GENERATE: owned_generation(actions.generate_drama3, "drama3"),
        REMIX: owned_generation(actions.remix_drama3, "drama3"),
        REGENERATE: owned_generation(actions.regenerate_drama3_line, "drama3"),
    }
    api = "/soundstage/api"

    def owner_of(request: Request) -> str:
        return request_owner(request, JOBS.store)

    def session_of(request: Request):
        """What outlives ``request`` and names its owner to the job path."""
        return SimpleNamespace(cookies=dict(request.cookies), session_hash=None)

    def project(name: str) -> tuple[dict, str | None]:
        data, result = store.soundstage_project(name)
        return data, result if result and Path(result).is_file() else None

    def run_payload(owner, name):
        run = runs.latest(owner, name)
        return run.public(None) if run is not None else None

    def payload(owner, name):
        data, result = project(name)
        cast_run = cast_jobs.runs.latest(owner, name) if cast_jobs is not None else None
        return {**project_payload(name, data, result, clips, runtime_config), "run": run_payload(owner, name),
                "cast_run": cast_run.public(None) if cast_run is not None else None}

    def auto_running(owner, name) -> str | None:
        run = runs.active(owner, name, RUN_KIND)
        return "Auto is running; it makes the voices and sounds itself." if getattr(run, "auto", False) else None

    cast_jobs = None

    # The page --------------------------------------------------------------
    @app.get("/soundstage", include_in_schema=False)
    def soundstage_redirect():
        return RedirectResponse("/soundstage/")

    @app.get("/soundstage/", include_in_schema=False)
    def soundstage_page():
        return FileResponse(STATIC / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/soundstage/{name}", include_in_schema=False)
    def soundstage_asset(name: str):
        if name not in STATIC_TYPES:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC / name, media_type=STATIC_TYPES[name], headers={"Cache-Control": "no-cache"})

    @app.get(f"{api}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner_of(request)
        return {"gpu": gpu,
                "projects": [{"name": name, "generated": generated} for name, generated in store.soundstage_projects()],
                "presets": {name: list(values) for name, values in DRAMA3_SAMPLING_PRESETS.items()},
                "ranges": {key: list(bounds) for key, bounds in SETTING_RANGES.items()},
                **boot_payload(), **cast_boot_payload()}

    install_script_routes(app, actions, api=api, store=store, owner_of=owner_of, payload=payload,
                          clip=lambda value: _clip(clips, value))
    cast_jobs = install_cast_routes(app, actions, api=api, store=store, owner_of=owner_of, session_of=session_of,
                                    payload=payload, clips=clips, clip=lambda value: _clip(clips, value),
                                    runtime=runtime_config, gpu=gpu, interrupt=interrupt, blocked=auto_running)

    # Projects --------------------------------------------------------------
    @app.get(f"{api}/project", include_in_schema=False)
    @guarded
    def read_project(name: str, request: Request):
        return payload(owner_of(request), name)

    @app.patch(f"{api}/project/settings", include_in_schema=False)
    @guarded
    def save_settings(name: str, request: Request, body: dict = Body(...)):
        """Keep the speech settings with the project, as Save project in the tab does."""
        owner = owner_of(request)
        data, _ = store.soundstage_project(name)
        for key, kind in SETTING_FIELDS.items():
            if key in body:
                data[key] = kind(body[key])
                if key in SETTING_RANGES:
                    low, high, _step = SETTING_RANGES[key]
                    data[key] = kind(min(high, max(low, data[key])))
        store.save_soundstage_project(name, data)
        return payload(owner, name)

    # Files -----------------------------------------------------------------
    @app.get(f"{api}/audio", include_in_schema=False)
    def project_audio(name: str, request: Request):
        try:
            owner_of(request)
            _, result = project(name)
        except (H3Error, ValueError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(result, headers={"Cache-Control": "private, max-age=3600"})

    @app.get(f"{api}/take", include_in_schema=False)
    def take_audio(name: str, key: str, request: Request):
        """One line's saved take, before staging and effects; only takes of the project's mix."""
        try:
            owner_of(request)
            _, result = project(name)
        except (H3Error, ValueError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        keys = {chunk.get("take") for chunk in result_settings(result).get("chunks") or []} if result else set()
        file = Path(takes_dir()) / f"{key}.wav"
        if not key.isalnum() or key not in keys or not file.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})

    # Speaking --------------------------------------------------------------
    @app.get(f"{api}/run", include_in_schema=False)
    @guarded
    def read_run(name: str, request: Request):
        return {"run": run_payload(owner_of(request), name)}

    def speak_job(run, owner: str, session, name: str, action: str, line=None) -> None:
        """Speak, remix or regenerate a line from the project as it is saved now."""
        data, _ = project(name)
        values = generation_values(name, data, clips)
        if action == REGENERATE:
            values = [line, *values]
        # Accepted here, once this is the project's only run: a refused run leaves no job.
        job = JOBS.accept(owner, "drama3", values, callback=callbacks[action], output_indices=(0,),
                          key=uuid.uuid4().hex)
        run.job_id, run.family = job.id, "drama3"
        for update in execute_accepted(job.id, session):
            status = str(update[1] or "") if len(update) > 1 else ""
            if status:
                run.status = status.splitlines()[0]
            if status.startswith("Error"):
                # The reason, without the timing summary that follows it.
                run.error = status.removeprefix("Error:").strip().split("\n\n")[0]
            if update[0]:
                run.revision += 1
        if run.error is None:
            run.status = "Done."

    @app.post(f"{api}/speak", include_in_schema=False)
    @guarded
    def speak(name: str, request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Open Soundstage in Studio to speak on the GPU.")
        owner = owner_of(request)
        action = body.get("action")
        if action not in callbacks:
            raise H3Error("Unknown speech request.")
        _, result = project(name)
        if action == REGENERATE:
            keys = {take["key"] for take in (result_payload(result) or {}).get("takes", [])}
            if body.get("line") not in keys:
                raise H3Error("Choose a line of this mix first.")
        elif action == REMIX and result is None:
            raise H3Error("Speak the project once before remixing it.")
        session = session_of(request)
        runs.start(SpeakRun(owner, name, RUN_KIND),
                   lambda run: speak_job(run, owner, session, name, action, body.get("line")))
        return {"run": run_payload(owner, name)}

    @app.post(f"{api}/auto", include_in_schema=False)
    @guarded
    def auto(name: str, request: Request, body: dict = Body(default={})):
        """Auto: rewrite the script (unless it is prepared), make every missing voice and sound, then speak.

        Each step starts only after the one before it worked, as the tab's Auto does; ``body``
        carries the writer service for the rewrite.
        """
        if not gpu:
            raise H3Error("Open Soundstage in Studio to run Auto on the GPU.")
        owner = owner_of(request)
        if cast_jobs.runs.active(owner, name, "cast") is not None:
            raise H3Error("Wait for the voices and sounds being made, then run Auto.")
        data, _ = project(name)
        if not (data.get("text") or "").strip():
            raise H3Error("Write or paste some text on Script first.")
        session = session_of(request)
        clip = lambda value: _clip(clips, value)  # noqa: E731

        def stopped(run) -> bool:
            if run.stop_requested:
                run.status = "Auto interrupted."
            return run.stop_requested

        def work(run):
            if project(name)[0].get("prepared"):
                run.status = "Using the prepared script."
            else:
                run.status = "Rewriting the script…"
                _, status, refused = rewrite_project(actions, store, name, body, clip)
                run.revision += 1
                if refused:
                    run.error = "The writer stopped: " + status.removeprefix(DRAMA3_PROMPT_FAILED).strip()
                    return
            if stopped(run):
                return
            run.stage, run.status = "cast", "Making the missing voices and sounds…"
            data, _ = project(name)
            _, family, callback, values = cast_jobs.job_for(data, "prepare")
            cast_jobs.execute(run, owner, session, name, "prepare", None, family, callback, values)
            if run.error or stopped(run):
                return
            # The cast job's own summary is in the slots' notes; Auto's message is only what it saved.
            run.message = cast_jobs.save_all(name) if data.get("auto_save_speakers") else None
            run.stage, run.status, run.job_id = "speak", "Speaking…", None
            speak_job(run, owner, session, name, GENERATE)

        runs.start(SpeakRun(owner, name, RUN_KIND, auto=True, stage="script"), work)
        return {"run": run_payload(owner, name)}

    @app.post(f"{api}/interrupt", include_in_schema=False)
    @guarded
    def stop(name: str, request: Request):
        if interrupt is None:
            raise H3Error("Nothing can run here.")
        owner = owner_of(request)
        run = runs.active(owner, name, RUN_KIND)
        if run is None:
            message = "This project isn't speaking."
        else:
            run.stop_requested = True
            if run.job_id is None:
                message = ("Auto stops when the writer answers." if run.auto
                           else "The run is still starting. Try again in a moment.")
            else:
                message = interrupt(session_of(request), family=run.family or "drama3", job_id=run.job_id)
        return {"message": message, "run": run_payload(owner, name)}
