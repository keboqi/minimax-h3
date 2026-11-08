"""The Soundstage page and its JSON API, outside Gradio, as Remake's and Storyboard's are.

It works on saved Soundstage projects: the page reads a project, plays its mix with the
timeline and lines in the browser, and speaks, remixes or regenerates a line from the project
as last saved. A run takes the same job path as every other generation, in a thread here, and
the page polls it, so it outlives the page that started it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from h3_app.drama3.results import result_settings
from h3_app.errors import H3Error
from h3_app.jobs import JOBS
from h3_app.workspace_store import request_owner

from ..job_admission import execute_accepted
from ..job_bindings import owned_generation
from ..storyboard.page import Run, Runs, guarded
from ..subtitle_words import word_pieces
from .controls import DRAMA3_SAMPLING_PRESETS, DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND, UPLOAD_VOICE
from .page_script import boot_payload, install_script_routes, script_payload
from .result_view import clean_line, timeline_data

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {"app.js": "application/javascript", "app.css": "text/css"}
GENERATE, REMIX, REGENERATE = "generate", "remix", "regenerate"
# The Speak step's defaults, for a project saved before a field existed.
DEFAULTS = {"temperature": 2.0, "top_p": 1.0, "top_k": 30, "max_frames": 8192, "parallel": 16, "retries": 0,
            "word_timing": False, "script_mode": "Basic", "text": "", "title": ""}
SETTING_FIELDS = {"temperature": float, "top_p": float, "top_k": int, "word_timing": bool}
RUN_KIND = "speak"


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


def _cast(data: dict, clips: Path) -> list[dict]:
    """The project's speakers and whether each has a voice, in slot order."""
    shown = int(data.get("shown_slots") or 1)
    cast = []
    for slot in range(min(shown, DRAMA3_VOICE_SLOTS)):
        preset = data.get(f"voice_{slot}_preset")
        saved = bool(preset) and preset != UPLOAD_VOICE
        ready = saved or _clip(clips, data.get(f"voice_{slot}_audio")) is not None
        name = data.get(f"voice_{slot}_save_name") or (preset if saved else "")
        if not name and not ready:
            continue  # An empty slot is no speaker yet.
        cast.append({"slot": slot, "name": name or f"Speaker {slot}", "ready": ready})
    return cast


def _sounds(data: dict, clips: Path) -> list[dict]:
    shown = int(data.get("shown_sound_slots") or 1)
    sounds = []
    for slot in range(min(shown, DRAMA3_SOUND_SLOTS)):
        preset = data.get(f"sound_{slot}_preset")
        saved = bool(preset) and preset != UPLOAD_SOUND
        name = data.get(f"sound_{slot}_save_name") or (preset if saved else "") or data.get(f"sound_{slot}_description")
        if not name:
            continue
        sounds.append({"slot": slot, "name": name,
                       "ready": saved or _clip(clips, data.get(f"sound_{slot}_audio")) is not None})
    return sounds


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


def project_payload(name: str, data: dict, result, clips: Path) -> dict:
    text = str(_value(data, "text"))
    return {
        "name": name,
        "title": _value(data, "title"),
        "script_mode": _value(data, "script_mode"),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        "cast": _cast(data, clips),
        "sounds": _sounds(data, clips),
        "settings": {"temperature": float(_value(data, "temperature")), "top_p": float(_value(data, "top_p")),
                     "top_k": int(_value(data, "top_k")), "word_timing": bool(_value(data, "word_timing")),
                     "preset": _preset(data)},
        "result": result_payload(result),
        "script": script_payload(data),
    }


def install_soundstage_page(app: FastAPI, actions, *, store, output_dir, takes_dir: Callable[[], Path],
                            gpu: bool = True, interrupt=None) -> None:
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
        return {**project_payload(name, data, result, clips), "run": run_payload(owner, name)}

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
                **boot_payload()}

    install_script_routes(app, actions, api=api, store=store, owner_of=owner_of, payload=payload,
                          clip=lambda value: _clip(clips, value))

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

    @app.post(f"{api}/speak", include_in_schema=False)
    @guarded
    def speak(name: str, request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Open Soundstage in Studio to speak on the GPU.")
        owner = owner_of(request)
        action = body.get("action")
        if action not in callbacks:
            raise H3Error("Unknown speech request.")
        data, result = project(name)
        values = generation_values(name, data, clips)
        if action == REGENERATE:
            keys = {take["key"] for take in (result_payload(result) or {}).get("takes", [])}
            if body.get("line") not in keys:
                raise H3Error("Choose a line of this mix first.")
            values = [body["line"], *values]
        elif action == REMIX and result is None:
            raise H3Error("Speak the project once before remixing it.")
        session = session_of(request)

        def work(run):
            # Accepted here, once this is the project's only run: a refused run leaves no job.
            job = JOBS.accept(owner, "drama3", values, callback=callbacks[action], output_indices=(0,),
                              key=uuid.uuid4().hex)
            run.job_id = job.id
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

        runs.start(Run(owner, name, RUN_KIND), work)
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
        elif run.job_id is None:
            message = "The run is still starting. Try again in a moment."
        else:
            message = interrupt(session_of(request), family="drama3", job_id=run.job_id)
        return {"message": message, "run": run_payload(owner, name)}
