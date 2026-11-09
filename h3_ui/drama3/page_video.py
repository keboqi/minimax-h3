"""The Soundstage page's illustrated video: plan shots, check a pasted plan, draw, play and save.

The plan is saved beside the mix. Draw images draws the plan for the current Art direction and
Minimum shots with the Image tab's model settings and keeps the images as the mix's stills, which
the page slides along the mix. Save to Media renders the video from the stills, on request only,
beside the mix, where the newest one for the mix is found again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
import time
from typing import Callable
from urllib.parse import quote

from fastapi import Body, FastAPI, Request

from h3_app.drama3.visual_plan import check_plan_reply, full_plan_prompt, plan_shots
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot

from ..job_bindings import owned_generation
from ..page_runs import Runs, guarded
from .page_common import PageRun, apply_fields, file_response, found, not_found, run_job, value, writer_service
from .page_store import editing
from .stage_video import latest_export
from . import visuals

VIDEO_FIELDS = {"video_engine": str, "video_style": str, "video_limit": int, "video_continuity": bool,
                "video_subtitles": bool, "video_subtitle_size": int}
LIMITS = (1, 64)
# The subtitles' size on the video, in percent of their usual size: lowest, highest, step.
SUBTITLE_SIZES = (50, 200, 10)
VIDEO_RANGES = {"video_limit": LIMITS, "video_subtitle_size": SUBTITLE_SIZES}
PLAN, DRAW, SAVE = "plan", "draw", "save"
VideoRun = PageRun


def video_settings(data: dict, engines: list) -> dict:
    style = data.get("video_style")
    style = visuals.STYLE_PRESETS[visuals.DEFAULT_STYLE] if style is None else style
    engine_ids = [engine_id for _label, engine_id in engines]
    engine = data.get("video_engine")
    if engine not in engine_ids:
        engine = engine_ids[0] if engine_ids else None
    limit = int(data.get("video_limit") or 16)
    return {"engine": engine, "style": style, "preset": visuals.preset_for(style), "limit": limit,
            "continuity": data.get("video_continuity") is not False,
            "subtitles": data.get("video_subtitles") is not False,
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
                      "url": f"/soundstage/api/video/still?name={quote(name)}&file={quote(item['image'])}"})
    return {"revision": record["revision"], "shots": shots}


def video_payload(name: str, data: dict, result, export) -> dict:
    """The video settings, the shot plan drawing would use, the mix's stills and its newest saved video."""
    settings = video_settings(data, export.engines)
    plan = {"rows": [], "stale": False, "saved": False}
    latest = None
    stills = None
    if result:
        audio = Path(result)
        saved = visuals.read_plan(audio)
        current = visuals.read_plan(audio, settings["style"], settings["limit"])
        duration = float((visuals.read_snapshot(audio) or {}).get("settings", {}).get("duration") or 0)
        plan = {"rows": [{"start": row[0], "end": row[1], "transition": row[2], "beat": row[3], "prompt": row[4]}
                         for row in (visuals.plan_rows(current, duration) if current and duration else [])],
                "stale": saved is not None and current is None, "saved": saved is not None}
        record = visuals.read_stills(audio, export.deps.runtime_config().output_dir)
        stills = stills_payload(name, record)
        video = latest_export(audio)
        if video is not None:
            made_from = ((read_snapshot(video) or {}).get("settings") or {}).get("stills")
            latest = {"url": f"/soundstage/api/video?name={quote(name)}&v={video.stat().st_mtime_ns}",
                      "file": video.name,
                      # Saved from the stills shown now, so saving again would make the same video.
                      "current": bool(record) and made_from == record["revision"]}
    return {**settings, "plan": plan, "stills": stills, "latest": latest}


def boot_payload(engines: list) -> dict:
    return {"engines": [{"id": engine_id, "label": label} for label, engine_id in engines],
            "styles": visuals.STYLE_PRESETS, "default_style": visuals.DEFAULT_STYLE,
            "custom_style": visuals.CUSTOM_STYLE, "limits": list(LIMITS),
            "subtitle_sizes": list(SUBTITLE_SIZES), "plan_timeout": visuals.PLAN_TIMEOUT}


def install_video_routes(app: FastAPI, export, *, api: str, store, owner_of, session_of, payload,
                         project: Callable[[str], tuple[dict, str | None]], gpu: bool, interrupt) -> Runs:
    """The video routes; ``export`` is the tab's :class:`visuals.VideoExport`. Returns the video runs."""
    runs = Runs(busy="This mix is already being planned, drawn or saved.")

    def mix_of(name: str):
        data, result = project(name)
        if not result:
            raise H3Error("Speak the project first: the video is drawn from its mix.")
        audio, settings = visuals.load_mix(export.deps, Path(result).name)
        return data, audio, settings, video_settings(data, export.engines)

    @app.patch(f"{api}/project/video", include_in_schema=False)
    @guarded
    def save_video_settings(name: str, request: Request, body: dict = Body(...)):
        with editing(store, name) as data:
            apply_fields(data, body, VIDEO_FIELDS, VIDEO_RANGES,
                         choices={"video_engine": [key for _, key in export.engines]})
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/video/plan", include_in_schema=False)
    @guarded
    def plan(name: str, request: Request, body: dict = Body(default={})):
        """Plan shots with the writer in the background; the page follows its elapsed time."""
        owner = owner_of(request)
        _, audio, mix_settings, settings = mix_of(name)
        writer = writer_service(body)
        model = writer[4] if writer[2] == "OpenAI-compatible" else writer[0]
        run_id = visuals.claim_plan(audio)

        def work(run):
            started = time.monotonic()
            # No ``with``: leaving it would wait for a writer that never answers.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-shot-plan")
            future = pool.submit(plan_shots, mix_settings, settings["style"], settings["limit"], *writer,
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
                        future.add_done_callback(lambda done: done.exception() is None and visuals.save_plan(
                            audio, settings["style"], settings["limit"], done.result()[0], run_id))
                        raise H3Error(f"{model} hasn't returned a shot plan after {elapsed // 60} minutes, so "
                                      "planning stopped. Try again, pick another writer model, or copy the "
                                      "planner prompt into any chat model and paste its reply.") from None
                    run.status = f"Planning shots with {model}… {elapsed // 60}m{elapsed % 60:02d}s"
            if not visuals.save_plan(audio, settings["style"], settings["limit"], plan_data, run_id):
                raise H3Error("A newer shot plan was started or pasted while this one was planning, so this "
                              "reply was dropped.")
            run.status = f"Planned {len(shots)} shots in {int(time.monotonic() - started)}s."
            run.revision += 1

        run = VideoRun(owner, name, PLAN)
        run.status = f"Planning shots with {model}…"
        runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.post(f"{api}/project/video/prompt", include_in_schema=False)
    @guarded
    def planner_prompt(name: str, request: Request):
        owner_of(request)
        _, _audio, mix_settings, settings = mix_of(name)
        return {"prompt": full_plan_prompt(mix_settings, settings["style"], settings["limit"],
                                                   runtime=export.deps.runtime_config())}

    @app.post(f"{api}/project/video/reply", include_in_schema=False)
    @guarded
    def apply_reply(name: str, request: Request, body: dict = Body(...)):
        """Check a plan from another chat model like the built-in planner's, then save it with the mix."""
        reply = str(body.get("reply") or "")
        if not reply.strip():
            raise H3Error("Paste the chat model's JSON reply first.")
        _, audio, mix_settings, settings = mix_of(name)
        plan_data, shots = check_plan_reply(reply, mix_settings, settings["style"], settings["limit"])
        # Claiming a run drops the reply of any planning still running for this mix.
        visuals.save_plan(audio, settings["style"], settings["limit"], plan_data, visuals.claim_plan(audio))
        return {**payload(owner_of(request), name), "message": f"Applied the pasted {len(shots)}-shot plan."}

    @app.post(f"{api}/project/video/draw", include_in_schema=False)
    @guarded
    def draw(name: str, request: Request):
        """Draw the plan for the current settings and keep the images as the mix's stills, on the usual job path."""
        if not gpu:
            raise H3Error("Open Soundstage in Studio to draw the images on the GPU.")
        owner = owner_of(request)
        _, audio, _mix_settings, settings = mix_of(name)
        if visuals.read_plan(audio, settings["style"], settings["limit"]) is None:
            raise H3Error("This mix has no shot plan for the current Art direction and Minimum shots yet. "
                          "Plan shots, or paste a plan from another chat model, then draw.")
        if not settings["engine"]:
            raise H3Error("No image model is available here.")
        values = [audio.name, settings["engine"], settings["style"], settings["limit"], settings["continuity"],
                  *export.values(owner)]
        session = session_of(request)
        callback = owned_generation(export.generate, "drama3_visuals",
                                    input_names=(*visuals.VIDEO_INPUT_NAMES, *export.controller.fields))

        def work(run):
            for update in run_job(run, owner, session, "drama3_visuals", values, callback):
                status = value(update[1]) if len(update) > 1 else None
                if isinstance(status, str) and status:
                    run.status = status
                if isinstance(value(update[0]), str):
                    run.revision += 1

        run = VideoRun(owner, name, DRAW)
        run.status = "Waiting for the GPU…"
        runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.post(f"{api}/project/video/save", include_in_schema=False)
    @guarded
    def save(name: str, request: Request):
        """Render the video from the mix's stills and add it to Media; it needs FFmpeg, not the GPU."""
        owner = owner_of(request)
        _, result = project(name)
        if not result:
            raise H3Error("Speak the project first: the video is drawn from its mix.")
        root = export.deps.runtime_config().output_dir
        if visuals.read_stills(result, root) is None:
            raise H3Error("Draw the images first: the video is rendered from them.")

        def work(run):
            video = visuals.render_stills(result, root)
            run.status = f"Saved {video.name} to Media."
            run.revision += 1

        run = VideoRun(owner, name, SAVE)
        run.status = "Rendering the video for Media…"
        runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.get(f"{api}/video/still", include_in_schema=False)
    def still(name: str, file: str, request: Request):
        """One of the mix's stills; only images its stills record names."""
        with found():
            owner_of(request)
            _, result = project(name)
        root = export.deps.runtime_config().output_dir
        record = visuals.read_stills(result, root) if result else None
        if record is None or file not in {item["image"] for item in record["shots"]}:
            raise not_found()
        # Each drawing names its images afresh, so a still never changes under its name.
        return file_response(visuals.stills_dir(result, root) / file, media_type="image/jpeg",
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
