"""The Soundstage page's illustrated video: plan shots, check a pasted plan, draw and render, and play.

It is the tab's own export (``visuals``): the plan is saved beside the mix, Generate draws the plan
for the current Art direction and Minimum shots with the Image tab's model settings, and the video
lands beside the mix, where the newest one for the mix is found again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, OPENAI_PROMPT_MODELS
from h3_app.drama3.visual_plan import check_plan_reply, full_plan_prompt, plan_shots
from h3_app.errors import H3Error
from h3_app.jobs import JOBS

from ..job_admission import execute_accepted
from ..job_bindings import owned_generation
from ..storyboard.page import Run, Runs, guarded
from .page_script import value
from .stage_video import latest_export
from . import visuals

VIDEO_FIELDS = {"video_engine": str, "video_style": str, "video_limit": int, "video_continuity": bool,
                "video_subtitles": bool}
LIMITS = (1, 64)
PLAN, RENDER = "plan", "render"


@dataclass
class VideoRun(Run):
    family: str | None = None


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
            "subtitles": data.get("video_subtitles") is not False}


def video_payload(name: str, data: dict, result, engines: list) -> dict:
    """The video settings, the shot plan Generate would use with them, and the newest video for the mix."""
    settings = video_settings(data, engines)
    plan = {"rows": [], "stale": False, "saved": False}
    latest = None
    if result:
        audio = Path(result)
        saved = visuals.read_plan(audio)
        current = visuals.read_plan(audio, settings["style"], settings["limit"])
        duration = float((visuals.read_snapshot(audio) or {}).get("settings", {}).get("duration") or 0)
        plan = {"rows": [{"start": row[0], "end": row[1], "transition": row[2], "beat": row[3], "prompt": row[4]}
                         for row in (visuals.plan_rows(current, duration) if current and duration else [])],
                "stale": saved is not None and current is None, "saved": saved is not None}
        video = latest_export(audio)
        if video is not None:
            latest = {"url": f"/soundstage/api/video?name={quote(name)}&v={video.stat().st_mtime_ns}",
                      "file": video.name}
    return {**settings, "plan": plan, "latest": latest}


def boot_payload(engines: list) -> dict:
    return {"engines": [{"id": engine_id, "label": label} for label, engine_id in engines],
            "styles": visuals.STYLE_PRESETS, "default_style": visuals.DEFAULT_STYLE,
            "custom_style": visuals.CUSTOM_STYLE, "limits": list(LIMITS), "plan_timeout": visuals.PLAN_TIMEOUT}


def _writer(body: dict) -> list[Any]:
    """The planner's writer inputs, as the Script step's writer service sends them."""
    service = body.get("writer") or {}
    return [service.get("gemini_prompt_model") or DEFAULT_GEMINI_PROMPT_MODEL, service.get("gemini_api_key") or "",
            service.get("prompt_writer_backend") or "Gemini", service.get("openai_api_key") or "",
            service.get("openai_model") or OPENAI_PROMPT_MODELS[0], service.get("openai_base_url") or ""]


def install_video_routes(app: FastAPI, export, *, api: str, store, owner_of, session_of, payload,
                         project: Callable[[str], tuple[dict, str | None]], gpu: bool, interrupt) -> Runs:
    """The video routes; ``export`` is the tab's :class:`visuals.VideoExport`. Returns the video runs."""
    runs = Runs(busy="This mix is already being planned or drawn.")

    def mix_of(name: str):
        data, result = project(name)
        if not result:
            raise H3Error("Speak the project first: the video is drawn from its mix.")
        audio, settings = visuals.load_mix(export.deps, Path(result).name)
        return data, audio, settings, video_settings(data, export.engines)

    @app.patch(f"{api}/project/video", include_in_schema=False)
    @guarded
    def save_video_settings(name: str, request: Request, body: dict = Body(...)):
        data, _ = store.soundstage_project(name)
        for key, kind in VIDEO_FIELDS.items():
            if key in body:
                data[key] = kind(body[key])
        if "video_limit" in data:
            data["video_limit"] = max(LIMITS[0], min(LIMITS[1], int(data["video_limit"])))
        store.save_soundstage_project(name, data)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/video/plan", include_in_schema=False)
    @guarded
    def plan(name: str, request: Request, body: dict = Body(default={})):
        """Plan shots with the writer in the background; the page follows its elapsed time."""
        owner = owner_of(request)
        _, audio, mix_settings, settings = mix_of(name)
        writer = _writer(body)
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

    @app.post(f"{api}/project/video/render", include_in_schema=False)
    @guarded
    def render(name: str, request: Request):
        """Draw the plan for the current settings and render the video, on the usual job path."""
        if not gpu:
            raise H3Error("Open Soundstage in Studio to draw the video on the GPU.")
        owner = owner_of(request)
        _, audio, _mix_settings, settings = mix_of(name)
        if visuals.read_plan(audio, settings["style"], settings["limit"]) is None:
            raise H3Error("This mix has no shot plan for the current Art direction and Minimum shots yet. "
                          "Plan shots, or paste a plan from another chat model, then generate.")
        if not settings["engine"]:
            raise H3Error("No image model is available here.")
        values = [audio.name, settings["engine"], settings["style"], settings["limit"], settings["continuity"],
                  *export.values(owner)]
        session = session_of(request)
        callback = owned_generation(export.generate, "drama3_visuals",
                                    input_names=(*visuals.VIDEO_INPUT_NAMES, *export.controller.fields))

        def work(run):
            job = JOBS.accept(owner, "drama3_visuals", values, callback=callback, output_indices=(0,),
                              key=uuid.uuid4().hex)
            run.job_id, run.family = job.id, "drama3_visuals"
            for update in execute_accepted(job.id, session):
                status = value(update[1]) if len(update) > 1 else None
                if isinstance(status, str) and status:
                    run.status = status
                if isinstance(value(update[0]), str):
                    run.revision += 1

        run = VideoRun(owner, name, RENDER)
        run.status = "Waiting for the GPU…"
        runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.post(f"{api}/project/video/interrupt", include_in_schema=False)
    @guarded
    def stop(name: str, request: Request):
        owner = owner_of(request)
        run = runs.active(owner, name)
        if run is None:
            return {"message": "Nothing is being planned or drawn."}
        if run.kind == PLAN:
            return {"message": "Planning can't be stopped; its reply is dropped if you plan again or paste a plan."}
        if interrupt is None or run.job_id is None:
            return {"message": "The run is still starting. Try again in a moment."}
        return {"message": interrupt(session_of(request), family=run.family, job_id=run.job_id)}

    @app.get(f"{api}/video", include_in_schema=False)
    def video_file(name: str, request: Request):
        """The newest illustrated video made from the project's mix."""
        try:
            owner_of(request)
            _, result = project(name)
        except (H3Error, ValueError) as exc:
            raise HTTPException(status_code=404, detail="File not found") from exc
        video = latest_export(result) if result else None
        if video is None:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(video, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})

    return runs
