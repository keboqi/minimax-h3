"""The Soundstage page and its JSON API, outside Gradio, as Remake's and Storyboard's are.

It works on saved Soundstage projects: the page reads a project, plays its mix with the
timeline and lines in the browser, and speaks, remixes or regenerates a line from the project
as last saved. A run takes the same job path as every other generation, in a thread here, and
the page polls it, so it outlives the page that started it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import re
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from fastapi import Body, FastAPI, Request

from h3_app.drama3.results import result_settings
from h3_app.errors import H3Error
from h3_app.prompt_writers.drama3 import DRAMA3_PROMPT_FAILED, DRAMA3_SCRIPT_ADVANCED
from h3_gpu import is_cpu_only

from ..job_bindings import owned_generation
from ..page_routes import install_static_page, owner_of, session_of
from ..page_runs import Runs, guarded
from .controls import DRAMA3_SAMPLING_PRESETS
from .page_cast import boot_payload as cast_boot_payload, cast_settings, install_cast_routes, sounds_payload, speakers_payload
from .page_common import PageRun, apply_fields, file_response, found, not_found, run_job, speech_value, timeline_payload, value
from .page_script import boot_payload, install_script_routes, rewrite_project, script_payload
from .page_library import install_library_routes
from .page_music import (MUSIC_FIELD, applied_music, desired_music, generate_track, mixed_audio, music_clip,
                         music_mix_id, music_settings, music_track, needs_music_clip, publish_mix, remember_applied_music)
from .page_store import editing, project_lock
from .page_video import boot_payload as video_boot_payload, install_video_routes, video_payload
from .result_view import clean_line
from .page_adapter import generation_values as generation_values, SpeechProgress
from .page_batch import Batches
from .project_visuals import can_reuse_script, carry_project_stills, planning_mix

STATIC = Path(__file__).with_name("page")
STATIC_TYPES = {name: "application/javascript" for name in ("app.js", "workspace.js", "saves.js", "transport.js",
                  "script.js", "scriptview.js", "jump.js", "cast.js", "mix.js", "video.js", "effects.js", "effects_webgpu.js",
                  "effects_picker.js", "effects_pass.js",
                  "library.js", "project.js", "batch.js", "fullscreen.js", "layout.js", "stage.js")}
STATIC_TYPES["app.css"] = "text/css"
STATIC_TYPES.update({f"effect-demo-{kind}.webp": "image/webp" for kind in ("camera", "atmosphere", "lighting")})
GENERATE, REMIX, REGENERATE = "generate", "remix", "regenerate"
SETTING_FIELDS = {"temperature": float, "top_p": float, "top_k": int, "word_timing": bool, "parallel": int,
                  "retries": int, "max_frames": int, "auto_save_speakers": bool, "auto_illustrate": bool}
# The Speak step's slider ranges, which the page draws and saved values are kept inside.
SETTING_RANGES = {"temperature": (0.0, 2.0, 0.05), "top_p": (0.05, 1.0, 0.05), "top_k": (0, 4096, 1),
                  "parallel": (1, 32, 1), "retries": (0, 4, 1), "max_frames": (1, 8192, 1)}
RUN_KIND = "speak"


@dataclass
class SpeakRun(PageRun):
    """Speaking, or Auto, which rewrites, makes the cast and speaks in turn (``stage``)."""

    auto: bool = False
    task: str = "speak"
    stage: str | None = None
    slots: dict[str, str] = field(default_factory=dict)

    def public(self, scene_id) -> dict[str, Any]:
        return {**super().public(scene_id), "auto": self.auto, "task": self.task, "stage": self.stage,
                "slots": dict(self.slots)}


def _clip(clips: Path, value) -> str | None:
    """A project clip's path, or None when it has none or the file is gone."""
    if not value:
        return None
    clip = clips / Path(str(value)).name
    return str(clip) if clip.is_file() else None


def _take_status(chunk: dict) -> str:
    if not chunk.get("ok", True) or chunk.get("warnings"):
        return "flagged"
    return "reused" if chunk.get("reused") else "new"


def result_payload(path: str | Path | None, settings: dict | None = None) -> dict | None:
    """What the page draws for a mix: its lines, sounds and takes on the timeline."""
    if not path or not Path(path).is_file():
        return None
    if settings is None:
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


def project_payload(name: str, data: dict, result, clips: Path, runtime=None, *, mix_settings=None) -> dict:
    text = str(speech_value(data, "text"))
    speakers = speakers_payload(name, data, lambda value: _clip(clips, value), runtime)
    track = music_track(data)
    desired = desired_music(data)
    mix = result_payload(result, mix_settings)
    mix_id = music_mix_id(applied_music(data))
    if mix and mix_id:
        mix["version"] = f"{mix['version']}-{mix_id}"
        mix["music"] = {"start": 0, "end": mix["duration"]}
    return {
        "name": name,
        "title": speech_value(data, "title"),
        "script_mode": speech_value(data, "script_mode"),
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        # The speakers who are someone: named, or with a voice.
        "cast": [speaker for speaker in speakers if speaker["save_name"] or speaker["saved_name"] or speaker["ready"]],
        "speakers": speakers,
        "sounds": sounds_payload(name, data, lambda value: _clip(clips, value), runtime),
        "cast_settings": cast_settings(data),
        "settings": {**{key: kind(speech_value(data, key)) for key, kind in SETTING_FIELDS.items()},
                     "preset": _preset(data)},
        "result": mix,
        "background_music": bool(track),
        "music_settings": music_settings(data),
        "music_pending": bool(mix) and (mix_id != music_mix_id(desired)
                                         or needs_music_clip(clips, desired)),
        "script": script_payload(data),
    }


def natural_key(name: str) -> list:
    """Sort "Chapter 2" before "Chapter 10": digit runs compare as numbers, the rest ignoring case."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
            for part in re.split(r"(\d+)", str(name)) if part]


def batch_row(name: str, generated: bool, data: dict, clip, runtime, *, busy: bool = False) -> dict:
    """What Auto would do for one project, for the batch list."""
    text = str(speech_value(data, "text"))
    speakers = speakers_payload(name, data, clip, runtime)
    advanced = speech_value(data, "script_mode") == DRAMA3_SCRIPT_ADVANCED
    return {
        "name": name, "generated": bool(generated), "busy": busy,
        "lines": sum(1 for line in text.splitlines() if line.strip()),
        "prepared": bool(data.get("prepared")),
        # The speakers who are someone without a voice yet, and sounds without a clip (Advanced only).
        "voices": sum(1 for speaker in speakers if (speaker["save_name"] or speaker["saved_name"]) and not speaker["ready"]),
        "sounds": sum(1 for sound in sounds_payload(name, data, clip, runtime) if not sound["ready"]) if advanced else 0,
    }


def install_soundstage_page(app: FastAPI, actions, *, store, output_dir, takes_dir: Callable[[], Path],
                            runtime_config: Callable[[], Any] | None = None, gpu: bool = True,
                            interrupt=None, video_export=None, music=None) -> None:
    """Serve the page at ``/soundstage/`` and its API under ``/soundstage/api``.

    ``actions`` holds Soundstage's speech actions; ``store`` keeps the saved projects and
    ``output_dir`` their clips, as the Soundstage tab saves them; ``takes_dir`` is where line
    takes are kept. ``video_export`` is the tab's illustrated video export; without it the page has none.
    """
    admission_lock = RLock()

    def has_gpu() -> bool:
        return bool(gpu and not is_cpu_only())

    if is_cpu_only():
        gpu = False
    raw_interrupt = interrupt
    if raw_interrupt is not None:
        from ..api_images import interrupt_api_image

        def safe_interrupt(session, *, family="image", job_id=None):
            if not has_gpu():
                return interrupt_api_image(session, family=family, job_id=job_id)
            return raw_interrupt(session, family=family, job_id=job_id)

        interrupt = safe_interrupt

    def speak_blocked(run):
        if video_runs is not None and video_runs.active(run.owner, run.storyboard_id):
            return "Wait for illustration planning, drawing or saving to finish."
        if run.auto and cast_jobs is not None and cast_jobs.runs.active(run.owner, run.storyboard_id, "cast"):
            return "Wait for the voices and sounds being made, then run Auto."
        return None

    runs = Runs(busy="This project is already speaking.", lock=admission_lock, blocked=speak_blocked)
    clips = Path(output_dir) / "h3" / ".soundstage" / "clips"
    generate_music = owned_generation(music.generate_music3, "music") if gpu and music is not None else None

    def clip(value) -> str | None:
        return _clip(clips, value)

    callbacks = {
        GENERATE: actions.generate_drama3,
        REMIX: actions.remix_drama3,
        REGENERATE: actions.regenerate_drama3_line,
    }
    api = "/soundstage/api"

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
        mix_settings = result_settings(result) if result else None
        return {**project_payload(name, data, result, clips, runtime_config, mix_settings=mix_settings),
                "run": run_payload(owner, name),
                "cast_run": latest(cast_jobs.runs if cast_jobs is not None else None, owner, name),
                "video": video_payload(name, data, result, video_export, store=store, result_snapshot=mix_settings)
                if video_export else None,
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
    install_static_page(app, "/soundstage", STATIC, STATIC_TYPES)

    @app.get(f"{api}/boot", include_in_schema=False)
    @guarded
    def boot(request: Request):
        owner_of(request)
        return {"gpu": has_gpu(),
                "background_music": has_gpu() and generate_music is not None,
                "capabilities": {"prepare": True, "library_upload": True, "batch_resume": True},
                "projects": [{"name": name, "generated": generated} for name, generated in store.soundstage_projects()],
                "presets": {name: list(values) for name, values in DRAMA3_SAMPLING_PRESETS.items()},
                "ranges": {key: list(bounds) for key, bounds in SETTING_RANGES.items()},
                "video": video_boot_payload(video_export.engines) if video_export else None,
                **boot_payload(), **cast_boot_payload()}

    @app.get(f"{api}/capabilities", include_in_schema=False)
    @guarded
    def capabilities_route(request: Request):
        """Authoritative Soundstage schema & presets so skills never hardcode drifting catalogs."""
        from h3_app.drama3.visual_effects import EFFECT_PRESETS
        from . import visuals

        owner_of(request)
        effect_kinds = {
            preset["id"]: preset["kind"]
            for preset in EFFECT_PRESETS
            if isinstance(preset, dict) and preset.get("id") and preset.get("kind")
        }
        return {
            "gpu": has_gpu(),
            "features": {"prepare": True, "library_upload": True, "batch_resume": True},
            "effect_kinds": effect_kinds,
            "style_presets": list(visuals.STYLE_PRESETS.keys()),
            "setting_ranges": {key: list(bounds) for key, bounds in SETTING_RANGES.items()},
        }

    def example_blocked(owner, name):
        return (runs.active(owner, name)
                or (cast_jobs is not None and cast_jobs.runs.active(owner, name))
                or (video_runs is not None and video_runs.active(owner, name)))

    install_script_routes(app, actions, api=api, store=store, owner_of=owner_of, payload=payload, clip=clip,
                          runtime=runtime_config, admission_lock=admission_lock, example_blocked=example_blocked)
    cast_jobs = install_cast_routes(app, actions, api=api, store=store, owner_of=owner_of, session_of=session_of,
                                    payload=payload, clips=clips, clip=clip,
                                    runtime=runtime_config, gpu=gpu, interrupt=interrupt, blocked=auto_running, admission_lock=admission_lock)
    install_library_routes(app, actions, api=api, store=store, owner_of=owner_of, session_of=session_of,
                           payload=payload, runtime=runtime_config, voice_settings=voice_settings, gpu=gpu,
                           interrupt=interrupt)
    if video_export is not None:
        video_runs = install_video_routes(app, video_export, api=api, store=store, owner_of=owner_of,
                                          session_of=session_of, payload=payload, project=project, gpu=gpu,
                                          interrupt=interrupt, admission_lock=admission_lock, music_clips=clips,
                                          blocked=lambda owner, name: "Wait for speech or Auto to finish."
                                          if runs.active(owner, name) else None)

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

    @app.post(f"{api}/project/prepare", include_in_schema=False)
    @guarded
    def prepare_project(name: str, request: Request, body: dict = Body(...)):
        """Atomic CPU-safe validate-and-apply for Workshop handoffs with digest and revision checks."""
        from .controls import DRAMA3_VOICE_SLOTS
        from .page_script import _apply_cast, _setting, failed as reply_failed, sync_sounds, writer_slots
        from .page_video import VIDEO_FIELDS, VIDEO_RANGES, video_settings
        from h3_app.drama3.visual_plan import check_plan_reply

        owner = owner_of(request)
        input_digest = str(body.get("input_digest") or "").strip() or None
        expected_revision = body.get("expected_revision")
        reply_raw = str(body.get("reply") or "")
        if not reply_raw.strip():
            raise H3Error("Prepare requires a writer reply.")

        existing_names = {proj_name for proj_name, _ in store.soundstage_projects()}
        created_new = name not in existing_names
        if created_new:
            store.create_soundstage_project(name)
        else:
            current_data, current_result = project(name)
            if current_result is not None or example_blocked(owner, name):
                raise H3Error(f"Workshop project {name!r} has production work; prepare revisions in a new project/title.")
            saved_digest = current_data.get("_prepare_digest")
            saved_rev = current_data.get("_prepare_revision")
            if input_digest and saved_digest == input_digest:
                proj = payload(owner, name)
                return {"project": proj, "prepare_revision": saved_rev, "prepare_digest": saved_digest,
                        "reused": True}
            if expected_revision is not None and saved_rev != expected_revision:
                raise H3Error(f"Workshop project {name!r} changed since handoff (revision {saved_rev} != {expected_revision}).")

        try:
            with editing(store, name) as data:
                if body.get("source") is not None:
                    data["text"] = str(body.get("source") or "")
                if body.get("script_mode"):
                    data["script_mode"] = str(body["script_mode"])
                if body.get("title") is not None:
                    data["title"] = str(body.get("title") or "")
                if body.get("group"):
                    data["speaker_group"] = str(body["group"])
                if body.get("engine"):
                    data["voice_engine"] = str(body["engine"])
                data["skip_existing"] = True
                data["skip_existing_sounds"] = True
                strict_source = bool(body.get("strict_source", False))
                data["strict_source"] = strict_source

                before = data.get("text") or ""
                outputs = list(actions.apply_drama3_writer_reply(
                    reply_raw, before, _setting(data, "speaker_mode"),
                    *writer_slots(data, clip), strict_source=strict_source))
                status = str(outputs[1])
                if reply_failed(status):
                    raise H3Error(status)
                data.pop("_writer_proposal", None)
                _apply_cast(data, outputs)
                data["prepared"] = True
                sync_sounds(actions, data)

                language = body.get("language")
                if language:
                    for slot in range(DRAMA3_VOICE_SLOTS):
                        if data.get(f"voice_{slot}_save_name") or data.get(f"voice_{slot}_description"):
                            data[f"voice_{slot}_language"] = str(language)

                sampling = body.get("sampling")
                if isinstance(sampling, dict):
                    apply_fields(data, sampling, SETTING_FIELDS, SETTING_RANGES)

                visuals_body = body.get("visuals")
                if isinstance(visuals_body, dict) and video_export is not None:
                    vsettings = visuals_body.get("settings")
                    if isinstance(vsettings, dict):
                        apply_fields(data, vsettings, VIDEO_FIELDS, VIDEO_RANGES)
                    vreply = str(visuals_body.get("reply") or "")
                    if vreply.strip():
                        _, mix_settings = planning_mix(name, data, None, video_export, store=store)
                        parsed_plan, _ = check_plan_reply(
                            vreply, mix_settings, runtime=runtime_config() if runtime_config else None)
                        resolved = video_settings(data, video_export.engines)
                        data["video_limit"] = len(parsed_plan["shots"])
                        data["visual_plan"] = {"style": resolved["style"], "limit": len(parsed_plan["shots"]),
                                               "plan": parsed_plan}
                    data["auto_illustrate"] = True
                    data["auto_save_speakers"] = True

                next_rev = int(data.get("_prepare_revision") or 0) + 1
                data["_prepare_revision"] = next_rev
                if input_digest:
                    data["_prepare_digest"] = input_digest
        except Exception:
            if created_new and hasattr(store, "delete_soundstage_project"):
                try:
                    store.delete_soundstage_project(name)
                except Exception:
                    pass
            raise

        proj = payload(owner, name)
        return {"project": proj, "prepare_revision": next_rev, "prepare_digest": input_digest, "reused": False}

    # Files -----------------------------------------------------------------
    @app.get(f"{api}/audio", include_in_schema=False)
    def project_audio(name: str, request: Request):
        with found():
            owner_of(request)
            with project_lock(name):
                data, result = project(name)
                if result:
                    soundtrack = applied_music(data)
                    try:
                        path = mixed_audio(result, soundtrack, clips)
                    except H3Error:
                        if music_clip(clips, soundtrack) is None:
                            path = Path(result)
                        else:
                            raise
                else:
                    path = None
        return file_response(path, cache="private, no-cache")

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

    def require_music() -> None:
        if generate_music is None:
            raise H3Error("Open Soundstage in Studio to generate background music."
                          if not gpu else "Music 3 is unavailable here.")

    @app.post(f"{api}/project/music", include_in_schema=False)
    @guarded
    def start_music(name: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        require_music()
        data, _ = project(name)
        if not (data.get("text") or "").strip():
            raise H3Error("Write or paste some text on Script first.")
        session = session_of(request)
        runs.start(SpeakRun(owner, name, RUN_KIND, task="music"),
                   lambda run: generate_track(run, owner, session, name, data, body,
                                              write=music.enhance_music3_prompt, generate=generate_music,
                                              store=store, clips=clips))
        return {"run": run_payload(owner, name)}

    @app.patch(f"{api}/project/music", include_in_schema=False)
    @guarded
    def save_music_settings(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        with editing(store, name) as data:
            settings = music_settings(data)
            if any(key in body and body[key] is None for key in settings):
                raise H3Error("Set a volume from 0 to 100 and choose whether to duck the music.")
            apply_fields(settings, body, {"music_enabled": bool, "music_volume": int, "music_ducking": bool},
                         {"music_volume": (0, 100)})
            remember_applied_music(data)
            data.update(settings)
        return payload(owner, name)

    @app.delete(f"{api}/project/music", include_in_schema=False)
    @guarded
    def remove_music(name: str, request: Request):
        owner = owner_of(request)
        with admission_lock:
            if runs.active(owner, name) or (video_runs is not None and video_runs.active(owner, name)):
                raise H3Error("Wait for the project’s current run to finish.")
            with editing(store, name) as data:
                remember_applied_music(data)
                data.pop(MUSIC_FIELD, None)
                data["music_enabled"] = False
        return payload(owner, name)

    def speak_job(run, owner: str, session, name: str, action: str, line=None, body=None) -> None:
        """Speak, remix or regenerate a line from the project as it is saved now."""
        data, previous_result = project(name)
        soundtrack = desired_music(data) if action != REGENERATE else applied_music(data)
        if action != REGENERATE and needs_music_clip(clips, soundtrack):
            require_music()
            run.stage, run.job_id = "music", None
            track = generate_track(run, owner, session, name, data, body or {}, write=music.enhance_music3_prompt,
                                   generate=generate_music, store=store, clips=clips)
            if track is None or run.stop_requested:
                return
            soundtrack[MUSIC_FIELD] = track
            run.stage, run.job_id = "speak", None
        if video_export is not None and previous_result:
            # Upgrade an older timestamp plan against its own audio before replacing the mix.
            planning_mix(name, data, previous_result, video_export, store=store)
        values = generation_values(name, data, clips)
        if action == REGENERATE:
            values = [line, *values]
        # The page publishes after both speech and music have finished. Scripted callers
        # keep the generation handler's normal project publication.
        callback = callbacks[action]

        @wraps(callback)
        def render(*values):
            return callback(*values, publish=False)

        wrapped = owned_generation(render, "drama3")
        result = None
        for update in run_job(run, owner, session, "drama3", values, wrapped, output_indices=(0,)):
            progress = SpeechProgress.read(update)
            if progress.status:
                run.status = progress.status.splitlines()[0]
            if progress.error:
                run.error = progress.error
            output = value(progress.output)
            if isinstance(output, (str, Path)) and output:
                result = output
        if run.error is not None:
            return
        if run.stop_requested:
            run.status = "Interrupted."
            return
        if result is None:
            raise H3Error("Speech returned no audio.")
        if music_mix_id(soundtrack):
            run.status = "Mixing background music…"
        publish_mix(store, name, result, soundtrack, clips, stopped=lambda: run.stop_requested)
        run.revision += 1
        if video_export is not None:
            try:
                carry_project_stills(store, name, previous_result, result, video_export)
            except H3Error as exc:
                run.message = f"Speech ready. Illustrations need review: {exc}"
        run.status = "Done."

    @app.post(f"{api}/speak", include_in_schema=False)
    @guarded
    def speak(name: str, request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Open Soundstage in Studio to speak on the GPU.")
        owner = owner_of(request)
        action = body.get("action")
        if not callbacks.get(action):
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
                   lambda run: speak_job(run, owner, session, name, action, body.get("line"), body))
        return {"run": run_payload(owner, name)}

    def start_auto(owner: str, name: str, session, body: dict | None) -> SpeakRun:
        """Auto: rewrite the script (unless it is prepared), make every missing voice and sound, then speak.

        Each step starts only after the one before it worked, as the tab's Auto does; ``body``
        carries the writer service for the rewrite, as the page's request does.
        """
        if not gpu:
            raise H3Error("Open Soundstage in Studio to run Auto on the GPU.")
        if cast_jobs.runs.active(owner, name, "cast") is not None:
            raise H3Error("Wait for the voices and sounds being made, then run Auto.")
        data, _ = project(name)
        if not (data.get("text") or "").strip():
            raise H3Error("Write or paste some text on Script first.")
        body = dict(body or {})
        illustrate = body.get("illustrate")
        illustrate = data.get("auto_illustrate", False) if illustrate is None else bool(illustrate)
        if illustrate and (video_export is None or not video_export.engines):
            raise H3Error("No image model is available for automatic illustrations.")

        def stopped(run) -> bool:
            if run.stop_requested:
                run.status = "Auto interrupted."
            return run.stop_requested

        def work(run):
            data, previous = project(name)
            planned = can_reuse_script(data, previous, runtime_config)
            if data.get("prepared") or planned:
                run.status = "Using the saved script with its wording corrections." if planned else "Using the prepared script."
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
            # Speakers the group already has a voice for (saved by an earlier chapter, say) reuse it.
            cast_jobs.match_all(name)
            data, _ = project(name)
            _, family, callback, values = cast_jobs.job_for(data, "prepare")
            cast_jobs.execute(run, owner, session, name, "prepare", None, family, callback, values, expected=data)
            if run.error or stopped(run):
                return
            # The cast job's own summary is in the slots' notes; Auto's message is only what it saved.
            run.message = cast_jobs.save_all(name) if data.get("auto_save_speakers") else None
            if stopped(run):
                return
            run.stage, run.status, run.job_id = "speak", "Speaking…", None
            speak_job(run, owner, session, name, GENERATE, body=body)
            if run.error or stopped(run):
                return
            if illustrate:
                video_runs.produce(run, owner, name, session, body)
                if run.error or stopped(run):
                    return
                run.message = " ".join(filter(None, [run.message, "Illustrations ready."]))
            run.status = "Done."

        return runs.start(SpeakRun(owner, name, RUN_KIND, auto=True, stage="script"), work)

    @app.post(f"{api}/auto", include_in_schema=False)
    @guarded
    def auto(name: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        start_auto(owner, name, session_of(request), body)
        return {"run": run_payload(owner, name)}

    # Batch produce ---------------------------------------------------------
    batches = Batches(
        start_auto,
        interrupt=(None if interrupt is None else
                   lambda session, run: interrupt(session, family=run.family or "drama3", job_id=run.job_id)),
        store=store)
    batch_payload = batches.payload

    @app.get(f"{api}/batch", include_in_schema=False)
    @guarded
    def read_batch(request: Request, projects: bool = True):
        """The batch queue, and with ``projects`` what Auto would do for each project."""
        owner = owner_of(request)
        if not projects:
            return {"batch": batch_payload(owner)}
        rows = []
        # In name order, so a series' chapters run in turn and later ones reuse earlier ones' voices.
        for name, generated in sorted(store.soundstage_projects(), key=lambda row: natural_key(row[0])):
            try:
                data, _ = project(name)
            except (H3Error, ValueError, KeyError):
                continue
            rows.append(batch_row(name, generated, data, clip, runtime_config,
                                  busy=runs.active(owner, name) is not None
                                  or cast_jobs.runs.active(owner, name) is not None
                                  or (video_runs is not None and video_runs.active(owner, name) is not None)))
        return {"batch": batch_payload(owner), "projects": rows}

    @app.post(f"{api}/batch", include_in_schema=False)
    @guarded
    def start_batch(request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Open Soundstage in Studio to produce projects on the GPU.")
        owner = owner_of(request)
        names = body.get("names")
        if not isinstance(names, list):
            raise H3Error("Select at least one project to produce.")
        writer = body.get("writer") if isinstance(body.get("writer"), dict) else {}
        batches.start(owner, names, session_of(request), writer, illustrate=body.get("illustrate"))
        return {"batch": batch_payload(owner)}

    @app.post(f"{api}/batch/stop", include_in_schema=False)
    @guarded
    def stop_batch(request: Request):
        owner = owner_of(request)
        message = batches.stop(owner, session_of(request))
        return {"message": message, "batch": batch_payload(owner)}

    @app.post(f"{api}/batch/resume", include_in_schema=False)
    @guarded
    def resume_batch(request: Request):
        if not gpu:
            raise H3Error("Open Soundstage in Studio to resume batch production on the GPU.")
        owner = owner_of(request)
        batches.resume(owner, session_of(request))
        return {"batch": batch_payload(owner)}

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
                progress = SpeechProgress.read(update, alignment=True)
                if progress.status:
                    run.status = progress.status.splitlines()[0]
                if progress.error:
                    run.error = progress.error
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
                if run.auto:
                    message = "Auto stops when the writer answers."
                elif run.task == "music" or run.stage == "music":
                    message = "Music generation stops when the writer answers."
                else:
                    message = "The run is still starting. Try again in a moment."
            else:
                message = interrupt(session_of(request), family=run.family or "drama3", job_id=run.job_id)
        return {"message": message, "run": run_payload(owner, name)}
