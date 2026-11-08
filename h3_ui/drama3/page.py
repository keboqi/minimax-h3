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

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from h3_app.drama3.results import result_settings
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.prompt_writers.drama3 import DRAMA3_PROMPT_FAILED
from h3_app.workspace_store import request_owner

from ..job_bindings import owned_generation
from ..storyboard.page import Runs, guarded
from .controls import DRAMA3_SAMPLING_PRESETS, DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND
from .page_cast import boot_payload as cast_boot_payload, cast_settings, install_cast_routes, sounds_payload, speakers_payload
from .page_common import PageRun, apply_fields, file_response, found, not_found, run_job, speech_value, timeline_payload
from .page_script import boot_payload, install_script_routes, rewrite_project, script_payload, voice_inputs
from .page_library import install_library_routes
from .page_store import editing
from .page_video import boot_payload as video_boot_payload, install_video_routes, video_payload
from .result_view import clean_line

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
GENERATE, REMIX, REGENERATE = "generate", "remix", "regenerate"
SETTING_FIELDS = {"temperature": float, "top_p": float, "top_k": int, "word_timing": bool, "parallel": int,
                  "retries": int, "max_frames": int, "auto_save_speakers": bool}
# The Speak step's slider ranges, which the page draws and saved values are kept inside.
SETTING_RANGES = {"temperature": (0.0, 2.0, 0.05), "top_p": (0.05, 1.0, 0.05), "top_k": (0, 4096, 1),
                  "parallel": (1, 32, 1), "retries": (0, 4, 1), "max_frames": (1, 8192, 1)}
RUN_KIND = "speak"


@dataclass
class SpeakRun(PageRun):
    """Speaking, or Auto, which rewrites, makes the cast and speaks in turn (``stage``)."""

    auto: bool = False
    task: str = "speak"  # "speak", or "words" for word alignment
    stage: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    stop_requested: bool = False

    def public(self, scene_id) -> dict[str, Any]:
        return {**super().public(scene_id), "auto": self.auto, "task": self.task, "stage": self.stage,
                "slots": dict(self.slots)}


def _clip(clips: Path, value) -> str | None:
    """A project clip's path, or None when it has none or the file is gone."""
    if not value:
        return None
    clip = clips / Path(str(value)).name
    return str(clip) if clip.is_file() else None


def generation_values(name: str, data: dict, clips: Path) -> list[Any]:
    """A saved project as ``generate_drama3``'s inputs, in the order the Speak step sends them."""
    values: list[Any] = [speech_value(data, "text"), *voice_inputs(data, lambda value: _clip(clips, value))]
    values += [speech_value(data, key) for key in ("temperature", "top_p", "top_k", "max_frames", "parallel",
                                                   "retries", "script_mode")]
    for slot in range(DRAMA3_SOUND_SLOTS):
        values += [data.get(f"sound_{slot}_preset") or UPLOAD_SOUND, _clip(clips, data.get(f"sound_{slot}_audio")),
                   data.get(f"sound_{slot}_save_name") or "", data.get(f"sound_{slot}_description") or "",
                   data.get(f"sound_{slot}_duration") or 5]
    values.append(speech_value(data, "title"))
    values += [data.get(f"voice_{slot}_save_name") or "" for slot in range(DRAMA3_VOICE_SLOTS)]
    values += [bool(speech_value(data, "word_timing")), name]
    return values


def _take_status(chunk: dict) -> str:
    if not chunk.get("ok", True) or chunk.get("warnings"):
        return "flagged"
    return "reused" if chunk.get("reused") else "new"


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
    current = tuple(float(speech_value(data, key)) for key in ("temperature", "top_p", "top_k"))
    return next((name for name, values in DRAMA3_SAMPLING_PRESETS.items()
                 if tuple(float(value) for value in values) == current), None)


def project_payload(name: str, data: dict, result, clips: Path, runtime=None) -> dict:
    text = str(speech_value(data, "text"))
    speakers = speakers_payload(name, data, lambda value: _clip(clips, value), runtime)
    return {
        "name": name,
        "title": speech_value(data, "title"),
        "script_mode": speech_value(data, "script_mode"),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        # The speakers who are someone: named, or with a voice.
        "cast": [speaker for speaker in speakers if speaker["save_name"] or speaker["saved_name"] or speaker["ready"]],
        "speakers": speakers,
        "sounds": sounds_payload(name, data, lambda value: _clip(clips, value)),
        "cast_settings": cast_settings(data),
        "settings": {**{key: kind(speech_value(data, key)) for key, kind in SETTING_FIELDS.items()},
                     "preset": _preset(data)},
        "result": result_payload(result),
        "script": script_payload(data),
    }


def install_soundstage_page(app: FastAPI, actions, *, store, output_dir, takes_dir: Callable[[], Path],
                            runtime_config: Callable[[], Any] | None = None, gpu: bool = True,
                            interrupt=None, video_export=None) -> None:
    """Serve the page at ``/soundstage/`` and its API under ``/soundstage/api``.

    ``actions`` holds Soundstage's speech actions; ``store`` keeps the saved projects and
    ``output_dir`` their clips, as the Soundstage tab saves them; ``takes_dir`` is where line
    takes are kept. ``video_export`` is the tab's illustrated video export; without it the page has none.
    """
    runs = Runs(busy="This project is already speaking.")
    clips = Path(output_dir) / "h3" / ".soundstage" / "clips"

    def clip(value) -> str | None:
        return _clip(clips, value)

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

    def latest(source: Runs | None, owner, name):
        run = source.latest(owner, name) if source is not None else None
        return run.public(None) if run is not None else None

    def run_payload(owner, name):
        return latest(runs, owner, name)

    def payload(owner, name):
        data, result = project(name)
        return {**project_payload(name, data, result, clips, runtime_config), "run": run_payload(owner, name),
                "cast_run": latest(cast_jobs.runs if cast_jobs is not None else None, owner, name),
                "video": video_payload(name, data, result, video_export) if video_export else None,
                "video_run": latest(video_runs, owner, name)}

    def voice_settings(data: dict) -> list:
        """The Cast's voice generator with the speech sampling, as the tab's voice tries use them."""
        from h3_app.drama3.voice_design import ENGINE_QWEN

        return [data.get("voice_engine") or ENGINE_QWEN,
                *(speech_value(data, key) for key in ("temperature", "top_p", "top_k", "max_frames", "parallel"))]

    def auto_running(owner, name) -> str | None:
        run = runs.active(owner, name, RUN_KIND)
        return "Auto is running; it makes the voices and sounds itself." if getattr(run, "auto", False) else None

    cast_jobs = None
    video_runs = None

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
                "video": video_boot_payload(video_export.engines) if video_export else None,
                **boot_payload(), **cast_boot_payload()}

    install_script_routes(app, actions, api=api, store=store, owner_of=owner_of, payload=payload, clip=clip)
    cast_jobs = install_cast_routes(app, actions, api=api, store=store, owner_of=owner_of, session_of=session_of,
                                    payload=payload, clips=clips, clip=clip,
                                    runtime=runtime_config, gpu=gpu, interrupt=interrupt, blocked=auto_running)
    install_library_routes(app, actions, api=api, store=store, owner_of=owner_of, session_of=session_of,
                           payload=payload, runtime=runtime_config, voice_settings=voice_settings, gpu=gpu,
                           interrupt=interrupt)
    if video_export is not None:
        video_runs = install_video_routes(app, video_export, api=api, store=store, owner_of=owner_of,
                                          session_of=session_of, payload=payload, project=project, gpu=gpu,
                                          interrupt=interrupt)

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
        with editing(store, name) as data:
            apply_fields(data, body, SETTING_FIELDS, SETTING_RANGES)
        return payload(owner, name)

    # Files -----------------------------------------------------------------
    @app.get(f"{api}/audio", include_in_schema=False)
    def project_audio(name: str, request: Request):
        with found():
            owner_of(request)
            _, result = project(name)
        return file_response(result)

    @app.get(f"{api}/take", include_in_schema=False)
    def take_audio(name: str, key: str, request: Request):
        """One line's saved take, before staging and effects; only takes of the project's mix."""
        with found():
            owner_of(request)
            _, result = project(name)
        keys = {chunk.get("take") for chunk in result_settings(result).get("chunks") or []} if result else set()
        if not key.isalnum() or key not in keys:
            raise not_found()
        return file_response(Path(takes_dir()) / f"{key}.wav", media_type="audio/wav")

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
        for update in run_job(run, owner, session, "drama3", values, callbacks[action], output_indices=(0,)):
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

    @app.post(f"{api}/words", include_in_schema=False)
    @guarded
    def repair_words(name: str, request: Request):
        """Time every word of the mix from its saved takes, without speaking or remixing again."""
        if not gpu:
            raise H3Error("Open Soundstage in Studio to align words on the GPU.")
        if actions.repair_drama3_words is None:
            raise H3Error("Word alignment isn't available here.")
        owner = owner_of(request)
        _, result = project(name)
        if result is None:
            raise H3Error("Speak the project first.")
        session = session_of(request)
        callback = owned_generation(actions.repair_drama3_words, "drama3_words")

        def work(run):
            for update in run_job(run, owner, session, "drama3_words", [result], callback):
                status = str(update[1] or "") if len(update) > 1 else ""
                if status:
                    run.status = status.splitlines()[0]
                if status.startswith("Word alignment unavailable"):
                    run.error = status.removeprefix("Word alignment unavailable:").strip()
            run.revision += 1

        run = SpeakRun(owner, name, RUN_KIND, task="words")
        run.status = "Aligning words…"
        runs.start(run, work)
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
