"""The Soundstage page's voice library: manage saved speakers and sounds, and try new voices.

These are the tab's library operations (``speakers``, ``sounds``, ``generate_drama3_voice_takes``).
A change to a saved speaker also points the open project's speaker slots at it again, as the tab
updates its slots, so a renamed voice stays in use and a deleted one is let go.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import Body, FastAPI, Request

from h3_app.errors import H3Error

from ..job_bindings import owned_generation
from ..storyboard.page import Runs, guarded
from .controls import DRAMA3_VOICE_SLOTS, UPLOAD_VOICE
from .page_common import SKIP, PageRun, file_response, found, run_job, value
from .page_store import editing

MAX_TAKES = 4


@dataclass
class TakesRun(PageRun):
    """New voices tried for a saved speaker: their clips, and the brief they were made from."""

    key: str | None = None
    takes: list[str | None] = field(default_factory=list)
    brief: dict | None = None

    def public(self, scene_id) -> dict[str, Any]:
        return {**super().public(scene_id), "key": self.key,
                "takes": [f"/soundstage/api/library/take?run={self.id}&index={index}" if take else None
                          for index, take in enumerate(self.takes)]}


def install_library_routes(app: FastAPI, actions, *, api: str, store, owner_of, session_of, payload,
                           runtime: Callable[[], Any], voice_settings: Callable[[dict], list], gpu: bool,
                           interrupt) -> Runs:
    """The library routes. ``voice_settings`` gives a project's voice generator and sampling. Returns the runs."""
    from h3_app.drama3 import sounds as drama3_sounds
    from h3_app.drama3 import speakers as drama3_speakers

    runs = Runs(busy="New voices are already being made for this speaker.")

    def repoint(project: str | None, old: str, new: str | None) -> None:
        """Point the project's slots that used ``old`` at ``new``, or let them go when it was deleted."""
        if not project:
            return
        with editing(store, project) as data:
            for slot in range(DRAMA3_VOICE_SLOTS):
                if data.get(f"voice_{slot}_preset") == old:
                    data[f"voice_{slot}_preset"] = new or UPLOAD_VOICE

    def answer(request: Request, project: str | None, message: str, **extra):
        reply = {"message": message, **extra}
        if project:
            reply["project"] = payload(owner_of(request), project)
        return reply

    def speaker_row(preset) -> dict:
        return {"key": preset.key, "name": preset.name, "group": preset.group, "description": preset.description,
                "transcript": preset.transcript}

    @app.get(f"{api}/library/manage", include_in_schema=False)
    @guarded
    def manage(request: Request, group: str = "Global"):
        """A group's own saved speakers (Global's are listed under Global), and every saved sound."""
        owner_of(request)
        config = runtime()
        return {"groups": drama3_speakers.list_groups(config),
                "speakers": [speaker_row(preset) for preset in drama3_speakers.list_speakers(config, group or "Global")],
                "sounds": [{"name": preset.name, "description": preset.description, "duration": preset.duration}
                           for preset in drama3_sounds.list_sounds(config)]}

    @app.get(f"{api}/library/clip", include_in_schema=False)
    def library_clip(kind: str, key: str, request: Request):
        with found():
            owner_of(request)
            saved = (drama3_speakers.load_speaker(key, runtime()) if kind == "voice"
                     else drama3_sounds.load_sound(key, runtime()))
        return file_response(saved.audio, cache="no-cache")

    @app.post(f"{api}/library/speaker/update", include_in_schema=False)
    @guarded
    def update_speaker(request: Request, body: dict = Body(...)):
        owner_of(request)
        key = str(body.get("key") or "")
        saved = drama3_speakers.update_speaker(key, body.get("name") or "", body.get("transcript") or "",
                                               body.get("description") or "", runtime())
        if saved.key != key:
            repoint(body.get("project"), key, saved.key)
        return answer(request, body.get("project"), f'Saved changes to "{saved.name}" in "{saved.group}".',
                      key=saved.key)

    @app.post(f"{api}/library/speaker/copy", include_in_schema=False)
    @guarded
    def copy_speaker(request: Request, body: dict = Body(...)):
        owner_of(request)
        saved = drama3_speakers.copy_speaker(str(body.get("key") or ""), body.get("group") or "Global", runtime())
        return answer(request, None, f'Copied "{saved.name}" to "{saved.group}".', key=saved.key)

    @app.post(f"{api}/library/speaker/delete", include_in_schema=False)
    @guarded
    def delete_speaker(request: Request, body: dict = Body(...)):
        owner_of(request)
        key = str(body.get("key") or "")
        source = drama3_speakers.load_speaker(key, runtime())
        drama3_speakers.delete_speaker(key, runtime())
        repoint(body.get("project"), key, None)
        return answer(request, body.get("project"), f'Deleted "{source.name}" from "{source.group}".')

    @app.post(f"{api}/library/group/delete", include_in_schema=False)
    @guarded
    def delete_group(request: Request, body: dict = Body(...)):
        owner_of(request)
        group = str(body.get("group") or "")
        removed = drama3_speakers.delete_group(group, runtime())
        project = body.get("project")
        if project:
            for preset in removed:
                repoint(project, preset.key, None)
            with editing(store, project) as data:
                if (data.get("speaker_group") or "Global").casefold() == group.casefold():
                    data["speaker_group"] = "Global"
        return answer(request, project, f'Deleted group "{group}" and its {len(removed)} saved speakers.')

    @app.post(f"{api}/library/sound/delete", include_in_schema=False)
    @guarded
    def delete_sound(request: Request, body: dict = Body(...)):
        owner_of(request)
        name = str(body.get("name") or "")
        drama3_sounds.delete_sound(name, runtime())
        project = body.get("project")
        if project:
            with editing(store, project) as data:
                for key in [key for key, item in data.items() if key.startswith("sound_") and key.endswith("_preset") and item == name]:
                    data[key] = None
        return answer(request, project, f'Deleted saved sound "{name}".')

    # Try new voices ----------------------------------------------------------
    @app.post(f"{api}/library/speaker/takes", include_in_schema=False)
    @guarded
    def try_voices(request: Request, body: dict = Body(...)):
        """Design one to four new voices for a saved speaker, without touching its clip."""
        if not gpu:
            raise H3Error("Open Soundstage in Studio to make voices on the GPU.")
        owner = owner_of(request)
        key = str(body.get("key") or "")
        drama3_speakers.load_speaker(key, runtime())
        count = max(1, min(MAX_TAKES, int(body.get("count") or 1)))
        description, transcript = str(body.get("description") or ""), str(body.get("transcript") or "")
        if not description.strip() or not transcript.strip():
            raise H3Error("Add a voice description and an exact clip transcript first.")
        data = store.soundstage_project(body["project"])[0] if body.get("project") else {}
        engine, temperature, top_p, top_k, max_frames, parallel = voice_settings(data)
        values = [key, count, description, transcript, engine, temperature, top_p, top_k, max_frames, parallel]
        session = session_of(request)
        callback = owned_generation(actions.generate_drama3_voice_takes, "drama3_voice")

        def work(run):
            for update in run_job(run, owner, session, "drama3_voice", values, callback):
                update = list(update)
                clips, brief, status = update[:MAX_TAKES], value(update[MAX_TAKES + 1]), value(update[MAX_TAKES + 2])
                run.takes = [value(clip) if isinstance(value(clip), str) else None for clip in clips][:count]
                if brief is not SKIP:
                    run.brief = brief
                if isinstance(status, str):
                    run.status = status
                run.revision += 1
            if not any(run.takes):
                run.error = run.status

        run = TakesRun(owner, key, "takes", key=key)
        run.status = f"Generating {count} new voice{'s' if count > 1 else ''}…"
        runs.start(run, work)
        return {"takes_run": run.public(None)}

    def owned_run(request: Request, run_id: str) -> TakesRun:
        owner = owner_of(request)
        run = runs.items.get(run_id)
        if run is None or run.owner != owner:
            raise H3Error("Those voices are gone; generate new ones.")
        return run

    @app.get(f"{api}/library/takes", include_in_schema=False)
    @guarded
    def read_takes(request: Request, run: str):
        return {"takes_run": owned_run(request, run).public(None)}

    @app.get(f"{api}/library/take", include_in_schema=False)
    def take_clip(request: Request, run: str, index: int):
        with found():
            file = owned_run(request, run).takes[index]
        return file_response(file)

    @app.post(f"{api}/library/speaker/use-take", include_in_schema=False)
    @guarded
    def use_take(request: Request, body: dict = Body(...)):
        """Replace the saved clip with the chosen new voice, with the brief it was made from."""
        run = owned_run(request, str(body.get("run") or ""))
        index = int(body.get("index", -1))
        if not 0 <= index < len(run.takes) or not run.takes[index] or not run.brief:
            raise H3Error("That voice is no longer available; generate new ones.")
        source = drama3_speakers.load_speaker(run.key, runtime())
        saved = drama3_speakers.save_speaker(source.name, run.takes[index], run.brief["transcript"], runtime(),
                                             run.brief["description"], source.group)
        return answer(request, body.get("project"), f'"{saved.name}" now uses voice {index + 1}.', key=saved.key)

    @app.post(f"{api}/library/takes/interrupt", include_in_schema=False)
    @guarded
    def stop_takes(request: Request, body: dict = Body(...)):
        run = owned_run(request, str(body.get("run") or ""))
        if run.finished_at is not None or run.job_id is None or interrupt is None:
            return {"message": "Nothing to stop."}
        return {"message": interrupt(session_of(request), family=run.family, job_id=run.job_id)}

    return runs
