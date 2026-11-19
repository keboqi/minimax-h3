"""The Soundstage page's illustrated video: plan shots, check a pasted plan, draw, play and save.

The plan is saved in the shared project and beside its mix. Workshop plans use script anchors
that resolve against generated speech. Draw images draws the plan for the current Art direction and
Minimum shots with the Image tab's model settings and keeps the images as the mix's stills, which
the page slides along the mix. Save to Media renders the video from the stills, on request only,
beside the mix, where the newest one for the mix is found again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
import json
from functools import partial
from pathlib import Path
import time
from typing import Callable
from urllib.parse import quote

from fastapi import Body, FastAPI, Request

from h3_app.drama3 import voice_looks
from h3_app.drama3.visual_effects import EFFECT_PRESETS
from h3_app.drama3.visual_plan import (check_plan_reply, edit_shot, edit_subject, full_plan_prompt, mentions,
                                       plan_shots, subjects)
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot

from ..job_bindings import owned_generation
from ..page_runs import Runs, guarded
from .page_common import PageRun, apply_fields, file_response, found, not_found, run_job, value, writer_service
from .page_store import editing
from .stage_video import latest_export
from . import visuals
from .project_visuals import planning_mix, save_project_plan

VIDEO_FIELDS = {"video_engine": str, "video_style": str, "video_limit": int, "video_continuity": bool,
                "video_orientation": str, "video_subtitles": bool, "video_subtitle_size": int, "video_effects": bool,
                "video_effects_override_reduced_motion": bool}
LIMITS = (1, 64)
# The subtitles' size on the video, in percent of their usual size: lowest, highest, step.
SUBTITLE_SIZES = (50, 200, 10)
VIDEO_RANGES = {"video_limit": LIMITS, "video_subtitle_size": SUBTITLE_SIZES}
PLAN, DRAW, SAVE, PREVIEW = "plan", "draw", "save", "preview"
VideoRun = PageRun


class VideoRuns(Runs):
    """Video activity plus the synchronous illustration step used by Auto."""

    def __init__(self, produce, **options):
        super().__init__(**options)
        self.produce = produce


def drawing_job(export, run, owner, session, generate, input_names, values):
    """Share GPU admission and progress handling between Auto and manual drawing."""
    callback = owned_generation(generate, "drama3_visuals", input_names=(*input_names, *export.controller.fields))
    for update in run_job(run, owner, session, "drama3_visuals", values, callback):
        status = value(update[1]) if len(update) > 1 else None
        if isinstance(status, str) and status:
            run.status = status
        if isinstance(value(update[0]), str):
            run.revision += 1


def video_settings(data: dict, engines: list) -> dict:
    style = data.get("video_style")
    style = visuals.STYLE_PRESETS[visuals.DEFAULT_STYLE] if style is None else style
    engine_ids = [engine_id for _label, engine_id in engines]
    engine = data.get("video_engine")
    if engine not in engine_ids:
        engine = engine_ids[0] if engine_ids else None
    limit = int(data.get("video_limit") or 16)
    return {"engine": engine, "style": style, "preset": visuals.preset_for(style), "limit": limit,
            "orientation": visuals.orientation_of(data.get("video_orientation")),
            "continuity": data.get("video_continuity") is True,
            "subtitles": data.get("video_subtitles") is not False,
            "effects_enabled": data.get("video_effects") is not False,
            "effects_override_reduced_motion": data.get("video_effects_override_reduced_motion") is True,
            "subtitle_size": int(data.get("video_subtitle_size") or 100)}


def stills_payload(name: str, record: dict | None) -> dict | None:
    """The stills the page slides along the mix: each image's time and the seconds it dissolves in."""
    if not record:
        return None
    fps = float(record.get("fps") or visuals.FPS)
    shots = []
    for item in record["shots"]:
        first, frames = int(item["first_frame"]), int(item["frames"])
        fade = min(12, frames - 1) / fps if item.get("dissolve") and frames > 1 else 0.0
        shots.append({"start": round(first / fps, 3), "end": round((first + frames) / fps, 3),
                      "fade": round(fade, 3), "beat": item.get("beat") or "",
                      "url": still_url(name, item["image"])})
    return {"revision": record["revision"], "shots": shots,
            "orientation": visuals.orientation_of(record.get("orientation"))}


def still_url(name: str, file: str) -> str:
    return f"/soundstage/api/video/still?name={quote(name)}&file={quote(file)}"


def _voice_for(key: str, anchor: str, voices: dict[str, str]) -> tuple[str, str] | None:
    """The Cast speaker a character is: by key ("old_tom" for "Old Tom") or the name its looks start with."""
    named = anchor.split(",", 1)[0].strip().casefold()
    for speaker, voice in voices.items():
        folded = speaker.casefold()
        if key.casefold() in (folded, folded.replace(" ", "_")) or named == folded:
            return speaker, voice
    return None


def plan_payload(name: str, current: dict | None, duration: float, record: dict | None, voices: dict[str, str],
                 root, audio, settings: dict) -> dict:
    """The plan drawing would use, in the chosen style and frame: each shot with its status and image,
    and the characters, props and places it mentions with their looks (and, for a character, their
    Cast voice). A shot drawn in another style or frame shows as changed."""
    if not current or not duration:
        return {"rows": [], "subjects": [], "art_style": "", "counts": {}}
    try:
        shots = visuals.plan_shots_for(current, duration, settings["style"], settings["orientation"])
    except H3Error:
        return {"rows": [], "subjects": [], "art_style": "", "counts": {}}
    plan = current["plan"]
    keys = mentions(plan)
    rows = []
    for index, (shot, still, row, mentioned) in enumerate(zip(shots, visuals.match_stills(shots, record),
                                                              plan["shots"], keys)):
        status = visuals.shot_status(shot, still)
        rows.append({"index": index, "start": round(shot.first / visuals.FPS, 3),
                     "end": round((shot.first + shot.frames) / visuals.FPS, 3),
                     "transition": "Dissolve" if shot.dissolve else "Cut", "beat": shot.beat,
                     "prompt": row["prompt"], "subjects": list(mentioned), "status": status,
                     "effects": row.get("effects", []),
                     "image": still_url(name, still["image"]) if still else None})
    portraits = visuals.kept_portraits(record, root, audio)
    current_keys = visuals.portrait_keys(current, settings["style"])
    found = []
    for key, (kind, anchor) in subjects(plan).items():
        entry = {"kind": kind, "key": key, "anchor": anchor,
                 "uses": sum(key in mentioned for mentioned in keys), "portrait": None}
        if kind == "characters":
            voice = _voice_for(key, anchor, voices)
            if voice:
                entry.update(speaker=voice[0], voice=voice[1], conflicts=voice_looks.conflicts(anchor, voice[1]))
            kept = portraits.get(key)
            if kept:
                entry.update(portrait=still_url(name, Path(kept[0]).name),
                             portrait_current=kept[1] in current_keys.get(key, ()))
        found.append(entry)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("drawn", "changed", "todo")}
    return {"rows": rows, "subjects": found, "counts": counts,
            "art_style": str(plan.get("art_style") or plan.get("style") or "")}


def video_payload(name: str, data: dict, result, export, *, store=None, result_snapshot=None) -> dict:
    """The video settings, the shot plan drawing would use, the mix's stills and its newest saved video."""
    settings = video_settings(data, export.engines)
    root = export.deps.runtime_config().output_dir
    plan = {"rows": [], "subjects": [], "art_style": "", "counts": {}, "stale": False, "saved": False, "saved_settings": None}
    latest = None
    stills = None
    timing_error = None
    if result or data.get("visual_plan"):
        try:
            audio, mix_settings = planning_mix(name, data, result, export, store=store, result_snapshot=result_snapshot)
        except H3Error as exc:
            # A cleared script must remain editable even when it still has an old plan.
            return {**settings, "timing_pending": not bool(result),
                    "plan": {**plan, "saved": bool(data.get("visual_plan")), "error": str(exc)},
                    "stills": None, "latest": None, "previews": {}}
        timing_error = mix_settings.get("visual_timing_error")
        saved = visuals.read_plan(audio)
        current = saved if saved is not None and saved.get("limit") == settings["limit"] else None
        duration = float(mix_settings.get("duration") or 0)
        record = visuals.read_stills(audio, root)
        voices = mix_settings["voices"] if current else {}
        stale = saved is not None and current is None
        plan = {**plan_payload(name, current, duration, record, voices, root, audio, settings),
                "stale": stale, "saved": saved is not None,
                # The Minimum shots the saved plan was made for, so the page can set it back and show it again.
                "saved_settings": {"limit": saved.get("limit")} if stale else None}
        if timing_error:
            plan.update(rows=[], subjects=[], counts={}, error=timing_error)
        stills = stills_payload(name, record)
        video = latest_export(audio)
        if video is not None:
            made = (read_snapshot(video) or {}).get("settings") or {}
            latest = {"url": f"/soundstage/api/video?name={quote(name)}&v={video.stat().st_mtime_ns}",
                      "file": video.name,
                      # Saved from the stills shown now, so saving again would make the same video,
                      # unless the subtitles changed: the page compares ``subtitles`` (size, 0 for none).
                      "current": bool(record) and made.get("stills") == record["revision"],
                      "subtitles": int(made.get("subtitles") or 0)}
    previews = {preset: preview_url(settings["engine"], file)
                for preset, file in visuals.preview_images(root, settings["engine"]).items()}
    return {**settings, **({"timing_pending": True} if not result else {}), "plan": plan, "stills": stills, "latest": latest, "previews": previews}


def preview_url(engine: str, file: str) -> str:
    return f"/soundstage/api/video/preview?engine={quote(engine)}&file={quote(file)}"


def boot_payload(engines: list) -> dict:
    return {"engines": [{"id": engine_id, "label": label} for label, engine_id in engines],
            "effect_presets": EFFECT_PRESETS,
            "styles": visuals.STYLE_PRESETS, "style_groups": visuals.STYLE_GROUPS,
            "style_uses": visuals.STYLE_USES, "default_style": visuals.DEFAULT_STYLE,
            "custom_style": visuals.CUSTOM_STYLE, "limits": list(LIMITS),
            "subtitle_sizes": list(SUBTITLE_SIZES), "plan_timeout": visuals.PLAN_TIMEOUT}


def install_video_routes(app: FastAPI, export, *, api: str, store, owner_of, session_of, payload,
                         project: Callable[[str], tuple[dict, str | None]], gpu: bool, interrupt,
                         blocked=None, admission_lock=None) -> Runs:
    """The video routes; ``export`` is the tab's :class:`visuals.VideoExport`. Returns the video runs."""
    busy = "This mix is already being planned, drawn or saved."
    # One run per mix, of any kind: drawing uses the plan and saving uses the stills.

    def check_available(owner, name, *, allow_planning=False):
        if blocked and (reason := blocked(owner, name)):
            raise H3Error(reason)
        active = runs.active(owner, name)
        if active and not (allow_planning and active.kind == PLAN):
            raise H3Error(busy)

    def mix_of(name: str, *, strict=False):
        data, result = project(name)
        audio, settings = planning_mix(name, data, result, export, strict=strict, store=store)
        return data, audio, settings, video_settings(data, export.engines)

    def produce(run, owner, name, session, body):
        """Auto's illustration step, sharing its run, cancellation and GPU job identity."""
        _, audio, mix_settings, settings = mix_of(name, strict=True)
        saved = visuals.read_plan(audio)
        if saved is None:
            run.stage, run.status, run.job_id = "plan", "Planning illustrations…", None
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-auto-shot-plan")
            future = pool.submit(plan_shots, mix_settings, settings["limit"], *writer_service(body),
                                 runtime=export.deps.runtime_config())
            pool.shutdown(wait=False)
            started = time.monotonic()
            while True:
                if run.stop_requested:
                    run.status = "Auto interrupted."
                    return
                try:
                    plan_data, _ = future.result(timeout=visuals.PLAN_TICK)
                    break
                except FuturesTimeout:
                    if time.monotonic() - started >= visuals.PLAN_TIMEOUT:
                        raise H3Error("The illustration planner timed out. Try another writer or paste a plan.")
            save_project_plan(store, name, audio, settings["style"], settings["limit"], plan_data,
                              runtime=export.deps.runtime_config)
        else:
            # An explicitly prepared plan wins over the default minimum in Auto.
            settings["limit"] = saved["limit"]
            with editing(store, name) as data:
                data["video_limit"] = saved["limit"]
        if run.stop_requested:
            run.status = "Auto interrupted."
            return
        current = video_payload(name, project(name)[0], str(audio), export, store=store)
        if current["plan"]["rows"] and not any(current["plan"]["counts"].get(k) for k in ("changed", "todo")):
            return
        run.stage, run.status, run.job_id = "illustrate", "Drawing illustrations…", None
        values = [audio.name, settings["engine"], settings["style"], settings["limit"], settings["continuity"],
                  settings["orientation"], json.dumps("missing"), *export.values(owner)]
        drawing_job(export, run, owner, session, export.generate, visuals.VIDEO_INPUT_NAMES, values)

    runs = VideoRuns(produce, busy=busy, lock=admission_lock,
                     blocked=lambda run: blocked(run.owner, run.storyboard_id) if blocked else None)

    @app.patch(f"{api}/project/video", include_in_schema=False)
    @guarded
    def save_video_settings(name: str, request: Request, body: dict = Body(...)):
        with editing(store, name) as data:
            apply_fields(data, body, VIDEO_FIELDS, VIDEO_RANGES,
                         choices={"video_engine": [key for _, key in export.engines],
                                  "video_orientation": list(visuals.ORIENTATIONS)})
        return payload(owner_of(request), name)

    def start_drawing(request: Request, owner: str, name: str, kind: str, generate, input_names, values):
        """Run ``generate`` on the GPU job path as the mix's ``kind`` run, following its status."""
        session = session_of(request)
        def work(run):
            drawing_job(export, run, owner, session, generate, input_names, values)

        run = VideoRun(owner, name, kind)
        run.status = "Waiting for image generation…"
        with runs.lock:
            check_available(owner, name)
            runs.start(run, work)
        return {"video_run": run.public(None)}

    def edit_plan(name: str, change):
        """Change the plan drawing would use, unless the mix is being planned, drawn or saved."""
        def apply(request: Request):
            owner = owner_of(request)
            with runs.lock:
                check_available(owner, name)
                _, audio, mix_settings, settings = mix_of(name, strict=True)
                saved = visuals.read_plan(audio, settings["limit"])
                if saved is None:
                    raise H3Error("Plan shots first: there is no shot plan for the current Minimum shots.")
                plan = change(saved["plan"])
                shots = visuals.plan_shots_for({**saved, "plan": plan}, float(mix_settings["duration"]),  # still valid
                                               settings["style"], settings["orientation"])
                save_project_plan(store, name, audio, saved["style"], saved["limit"], plan,
                                  runtime=export.deps.runtime_config)
                # A cut or dissolve applies to the drawn images straight away; a prompt waits for a redraw.
                visuals.retime_stills(audio, export.deps.runtime_config().output_dir, shots)
                return payload(owner, name)
        return apply

    @app.patch(f"{api}/project/video/shot", include_in_schema=False)
    @guarded
    def edit_plan_shot(name: str, request: Request, body: dict = Body(...)):
        """Edit a shot; only image prompt changes require redrawing."""
        transition = body.get("transition")
        return edit_plan(name, lambda plan: edit_shot(
            plan, body.get("index"), prompt=body.get("prompt"),
            transition=str(transition).lower() if transition is not None else None,
            effects=body.get("effects")))(request)

    @app.patch(f"{api}/project/video/subject", include_in_schema=False)
    @guarded
    def edit_plan_subject(name: str, request: Request, body: dict = Body(...)):
        """Change how a character, prop or place looks in every shot that mentions it."""
        return edit_plan(name, lambda plan: edit_subject(plan, str(body.get("kind") or ""), str(body.get("key") or ""),
                                                         body.get("anchor")))(request)

    @app.post(f"{api}/project/video/plan", include_in_schema=False)
    @guarded
    def plan(name: str, request: Request, body: dict = Body(default={})):
        """Plan shots with the writer in the background; the page follows its elapsed time."""
        owner = owner_of(request)
        _, audio, mix_settings, settings = mix_of(name)
        writer = writer_service(body)
        model = writer[4] if writer[2] == "OpenAI-compatible" else writer[0]
        run_id = None

        def save_reply(plan_data):
            return save_project_plan(store, name, audio, settings["style"], settings["limit"], plan_data, run=run_id,
                                     runtime=export.deps.runtime_config)

        def work(run):
            started = time.monotonic()
            # No ``with``: leaving it would wait for a writer that never answers.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-shot-plan")
            future = pool.submit(plan_shots, mix_settings, settings["limit"], *writer,
                                 runtime=export.deps.runtime_config())
            pool.shutdown(wait=False)
            while True:
                try:
                    plan_data, shots = future.result(timeout=visuals.PLAN_TICK)
                    break
                except FuturesTimeout:
                    elapsed = int(time.monotonic() - started)
                    if elapsed >= visuals.PLAN_TIMEOUT:
                        # A reply that still arrives later is saved, unless a newer plan was made by then.
                        future.add_done_callback(lambda done: done.exception() is None and save_reply(done.result()[0]))
                        raise H3Error(f"{model} hasn't returned a shot plan after {elapsed // 60} minutes, so "
                                      "planning stopped. Try again, pick another writer model, or copy the "
                                      "planner prompt into any chat model and paste its reply.") from None
                    run.status = f"Planning shots with {model}… {elapsed // 60}m{elapsed % 60:02d}s"
            if not save_reply(plan_data):
                raise H3Error("A newer shot plan was started or pasted while this one was planning, so this "
                              "reply was dropped.")
            run.status = f"Planned {len(shots)} shots in {int(time.monotonic() - started)}s."
            run.revision += 1

        run = VideoRun(owner, name, PLAN)
        run.status = f"Planning shots with {model}…"
        with runs.lock:
            check_available(owner, name)
            run_id = visuals.claim_plan(audio)
            runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.post(f"{api}/project/video/prompt", include_in_schema=False)
    @guarded
    def planner_prompt(name: str, request: Request):
        owner_of(request)
        _, _audio, mix_settings, settings = mix_of(name)
        return {"prompt": full_plan_prompt(mix_settings, settings["limit"], runtime=export.deps.runtime_config())}

    @app.post(f"{api}/project/video/reply", include_in_schema=False)
    @guarded
    def apply_reply(name: str, request: Request, body: dict = Body(...)):
        """Check a plan from another chat model like the built-in planner's, then save it with the mix."""
        reply = str(body.get("reply") or "")
        if not reply.strip():
            raise H3Error("Paste the chat model's JSON reply first.")
        with runs.lock:
            check_available(owner_of(request), name, allow_planning=True)
            _, audio, mix_settings, settings = mix_of(name)
            plan_data, shots = check_plan_reply(reply, mix_settings, settings["limit"])
            # Claiming a run drops the reply of any planning still running for this mix.
            save_project_plan(store, name, audio, settings["style"], settings["limit"], plan_data,
                              run=visuals.claim_plan(audio), runtime=export.deps.runtime_config)
        return {**payload(owner_of(request), name), "message": f"Applied the pasted {len(shots)}-shot plan."}

    @app.post(f"{api}/project/video/draw", include_in_schema=False)
    @guarded
    def draw(name: str, request: Request, body: dict = Body(default={})):
        """Draw the plan for the current settings and keep the images as the mix's stills, on the usual job path.

        ``only`` is ``all`` (the default), ``missing`` (shots not drawn from their current prompt) or
        a list of shot indexes; the other shots keep their images.
        """
        owner = owner_of(request)
        _, audio, _mix_settings, settings = mix_of(name, strict=True)
        result = project(name)[1]
        if not result and settings["engine"] != "gpt_image":
            raise H3Error("Speak the project before drawing images.")
        if not gpu and settings["engine"] != "gpt_image":
            raise H3Error("Choose GPT Image to draw images in Workshop.")
        if visuals.read_plan(audio, settings["limit"]) is None:
            raise H3Error("This mix has no shot plan for the current Minimum shots yet. "
                          "Plan shots, or paste a plan from another chat model, then draw.")
        if not settings["engine"]:
            raise H3Error("No image model is available here.")
        only = body.get("only", "all")
        if not (only in ("all", "missing") or (isinstance(only, list) and only and all(type(i) is int for i in only))):
            raise H3Error("Draw all shots, the missing ones, or a list of shots.")
        values = [audio.name, settings["engine"], settings["style"], settings["limit"], settings["continuity"],
                  settings["orientation"], json.dumps(only), *export.values(owner)]
        generate = (partial(export.generate, mix=(audio, _mix_settings))
                    if settings["engine"] == "gpt_image" else export.generate)
        return start_drawing(request, owner, name, DRAW, generate, visuals.VIDEO_INPUT_NAMES, values)

    @app.post(f"{api}/project/video/preview", include_in_schema=False)
    @guarded
    def draw_previews(name: str, request: Request, body: dict = Body(default={})):
        """Draw style previews with the project's image model and keep them for every project.

        ``presets`` is ``missing`` (the default: presets with no current preview for the model) or a
        list of preset names.
        """
        if export.preview is None:
            raise H3Error("Style previews aren't available here.")
        owner = owner_of(request)
        data, _result = project(name)
        settings = video_settings(data, export.engines)
        if not gpu and settings["engine"] != "gpt_image":
            raise H3Error("Choose GPT Image to draw style previews in Workshop.")
        if not settings["engine"]:
            raise H3Error("No image model is available here.")
        presets = body.get("presets", "missing")
        # Checked now, so a bad request or nothing to draw says so before waiting for the GPU.
        if not visuals.chosen_previews(export.deps.runtime_config().output_dir, settings["engine"], presets):
            raise H3Error("Every style preset already has a preview for this image model.")
        values = [settings["engine"], json.dumps(presets), *export.values(owner)]
        return start_drawing(request, owner, name, PREVIEW, export.preview, visuals.PREVIEW_INPUT_NAMES, values)

    @app.get(f"{api}/video/preview", include_in_schema=False)
    def preview_file(engine: str, file: str, request: Request):
        """A kept style preview; only files the model's previews record names."""
        owner_of(request)
        root = export.deps.runtime_config().output_dir
        if file not in visuals.preview_images(root, engine).values():
            raise not_found()
        # Each preview gets a new name when drawn again, so a file never changes under its name.
        return file_response(visuals.previews_dir(root, engine) / file, media_type="image/jpeg",
                             cache="private, max-age=31536000, immutable")

    @app.post(f"{api}/project/video/save", include_in_schema=False)
    @guarded
    def save(name: str, request: Request):
        """Render the video from the mix's stills and add it to Media; it needs FFmpeg, not the GPU.

        The subtitles are burned in as the page shows them: on or off, at the chosen size.
        """
        owner = owner_of(request)
        data, result = project(name)
        if not result:
            raise H3Error("Speak the project first: the video is drawn from its mix.")
        root = export.deps.runtime_config().output_dir
        if visuals.read_stills(result, root) is None:
            raise H3Error("Draw the images first: the video is rendered from them.")
        settings = video_settings(data, export.engines)
        subtitles = settings["subtitle_size"] if settings["subtitles"] else 0

        def work(run):
            video = visuals.render_stills(result, root, subtitles=subtitles)
            run.status = f"Saved {video.name} to Media."
            run.revision += 1

        run = VideoRun(owner, name, SAVE)
        run.status = "Rendering the video for Media…"
        with runs.lock:
            check_available(owner, name)
            runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.get(f"{api}/video/still", include_in_schema=False)
    def still(name: str, file: str, request: Request):
        """One of the mix's stills; only images its stills record names."""
        with found():
            owner_of(request)
            data, result = project(name)
            audio, _ = planning_mix(name, data, result, export, store=store)
        root = export.deps.runtime_config().output_dir
        record = visuals.read_stills(audio, root)
        names = {item["image"] for item in record["shots"]} | {
            item.get("image") for item in (record.get("portraits") or {}).values()} if record else set()
        if file not in names:
            raise not_found()
        # Each drawing names its images afresh, so a still never changes under its name.
        media = "image/png" if file.endswith(".png") else "image/jpeg"
        return file_response(visuals.stills_dir(audio, root) / file, media_type=media,
                             cache="private, max-age=31536000, immutable")

    @app.post(f"{api}/project/video/interrupt", include_in_schema=False)
    @guarded
    def stop(name: str, request: Request):
        owner = owner_of(request)
        run = runs.active(owner, name)
        if run is None:
            return {"message": "Nothing is being planned or drawn."}
        if run.kind == PLAN:
            return {"message": "Planning can't be stopped; its reply is dropped if you plan again or paste a plan."}
        if run.kind == SAVE:
            return {"message": "Saving to Media can't be stopped; it finishes in a moment."}
        if interrupt is None or run.job_id is None:
            return {"message": "The run is still starting. Try again in a moment."}
        return {"message": interrupt(session_of(request), family=run.family, job_id=run.job_id)}

    @app.get(f"{api}/video", include_in_schema=False)
    def video_file(name: str, request: Request):
        """The newest illustrated video made from the project's mix."""
        with found():
            owner_of(request)
            _, result = project(name)
        return file_response(latest_export(result) if result else None, media_type="video/mp4")

    return runs
