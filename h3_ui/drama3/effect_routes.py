"""Sound Stage's optional post-draw segmentation and effect editing routes."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import re
import time
from uuid import uuid4

from fastapi import Body, Request

from h3_app.drama3 import effect_enhancement
from h3_app.drama3.visual_timing import shot_speech_cues
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled

from ..job_bindings import owned_generation
from ..page_runs import guarded
from . import effect_regions, effects_pass, visuals
from .page_common import PageRun, file_response, found, not_found, run_job, timeline_payload, writer_service
from .page_store import editing
from .project_visuals import planning_mix

SEGMENT, ENHANCE, REFINE_MASK = "segment_effects", "enhance_effects", "refine_effect_mask"
EFFECT_RUNS = (SEGMENT, ENHANCE, REFINE_MASK)
WRITER_BATCH_SIZE = 24


def install_effect_routes(app, *, api, export, store, gpu, runs, owner_of, session_of, project, payload,
                          mix_of, check_available, video_settings, plan_payload):
    def effect_inputs(name, editing_data=None):
        if editing_data is None:
            data, audio, mix, settings = mix_of(name, strict=True)
        else:
            # The caller already has the project transaction. Bind against its
            # snapshot without opening a second transaction to normalize the plan.
            _, result = project(name)
            data = editing_data
            audio, mix = planning_mix(name, data, result, export, strict=True)
            settings = video_settings(data, export.engines)
        root = export.deps.runtime_config().output_dir
        saved = visuals.read_plan(audio, settings["limit"])
        record = visuals.read_stills(audio, root)
        plan = plan_payload(name, saved, float(mix.get("duration") or 0), record, mix.get("voices", {}),
                            root, audio, settings)
        items = effects_pass.inventory(audio, root, plan["rows"], plan["subjects"], data,
                                       speech=shot_speech_cues(mix))
        if not items:
            raise H3Error("Draw the shots before enhancing their effects.")
        return visuals.stills_dir(audio, root), items, timeline_payload(mix)["sounds"]

    def find_effect_shot(items, body):
        item = next((item for item in items if item["shot_id"] == body.get("shot_id")), None)
        if item is None or item["image"] != body.get("image"):
            raise H3Error("This shot changed. Open its current effects and try again.")
        return item

    def start_effect_job(request, name, kind, status, work):
        owner = owner_of(request)
        run = PageRun(owner, name, kind)
        run.status = status
        with runs.lock:
            check_available(owner, name)
            runs.start(run, work)
        return {"video_run": run.public(None)}

    @app.post(f"{api}/project/video/effects/segment", include_in_schema=False)
    @guarded
    def segment_effects(name: str, request: Request):
        if not gpu:
            raise H3Error("Open Sound Stage on the GPU server to run SAM 3.1.")
        folder, items, _ = effect_inputs(name)
        pending = {(item["image"], tuple(item["labels"])) for item in items if item["regions"] is None}
        if not pending:
            return {**payload(owner_of(request), name), "message": "Shot regions are already available."}
        owner, session = owner_of(request), session_of(request)
        runtime = export.deps.runtime_config()

        def work(run):
            failures = []

            def generate():
                services = export.deps.generation_services()
                effect_regions.prepare(services)
                for index, (image, labels) in enumerate(sorted(pending), 1):
                    run.status = f"Segmenting shot {index}/{len(pending)} with SAM 3.1…"
                    check_cancelled()
                    try:
                        effect_regions.segment(folder / image, labels, services, runtime)
                    except (H3Error, OSError) as exc:
                        check_cancelled()
                        failures.append(str(exc))
                    yield run.status

            callback = owned_generation(generate, "drama3_visuals")
            for _ in run_job(run, owner, session, "drama3_visuals", [], callback):
                pass
            if len(failures) == len(pending):
                raise H3Error(f"SAM 3.1 could not segment these shots. {failures[0]}")
            run.status = f"Processed {len(pending) - len(failures)} shots."
            if failures:
                run.status += f" {len(failures)} failed and can be retried."

        return start_effect_job(request, name, SEGMENT, "Preparing SAM 3.1…", work)

    def current_prompt(name, include_manual, editing_data=None):
        _, items, sounds = effect_inputs(name, editing_data)
        shots, token = effects_pass.prompt_input(items, sounds, include_manual)
        if not shots:
            raise H3Error("No segmented shots are available for enhancement with these options.")
        return shots, token

    def store_effect_proposal(name, include_manual, *, reply=None, decisions=None, expected=None):
        with editing(store, name) as data:
            shots, token = current_prompt(name, include_manual, data)
            if expected is not None and token != expected:
                raise H3Error("The shots changed while the writer was working. Generate a new proposal.")
            if decisions is None:
                decisions = effect_enhancement.check_reply(reply, shots, token)
            data["video_effect_proposal"] = {"id": uuid4().hex, "revision": token,
                                             "include_manual": include_manual, "decisions": decisions}

    @app.post(f"{api}/project/video/effects/prompt", include_in_schema=False)
    @guarded
    def effects_prompt(name: str, request: Request, body: dict = Body(default={})):
        owner_of(request)
        shots, token = current_prompt(name, body.get("include_manual") is True)
        return {"prompt": effect_enhancement.copy_prompt(shots, token, export.deps.runtime_config())}

    @app.post(f"{api}/project/video/effects/generate", include_in_schema=False)
    @guarded
    def generate_effects(name: str, request: Request, body: dict = Body(...)):
        include_manual = body.get("include_manual") is True
        shots, token = current_prompt(name, include_manual)
        writer = writer_service(body)
        runtime = export.deps.runtime_config()

        def work(run):
            batches = [shots[first:first + WRITER_BATCH_SIZE] for first in range(0, len(shots), WRITER_BATCH_SIZE)]
            decisions = []
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-effects-writer")
            try:
                for index, batch in enumerate(batches, 1):
                    if run.stop_requested:
                        return
                    run.status = f"Writing visual effects · batch {index}/{len(batches)}…"
                    future = pool.submit(effect_enhancement.generate_reply, batch, token, writer, runtime)
                    deadline = time.monotonic() + visuals.PLAN_TIMEOUT
                    while not run.stop_requested:
                        try:
                            reply = future.result(timeout=visuals.PLAN_TICK)
                            decisions.extend(effect_enhancement.check_reply(reply, batch, token))
                            break
                        except FuturesTimeout:
                            if time.monotonic() >= deadline:
                                raise H3Error("The effects writer timed out. Try again or paste a reply.") from None
                with runs.lock:
                    if not run.stop_requested:
                        store_effect_proposal(name, include_manual, decisions=decisions, expected=token)
                        run.status = "Effects proposal ready to preview."
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        return start_effect_job(request, name, ENHANCE, "Writing visual effects…", work)

    @app.post(f"{api}/project/video/effects/reply", include_in_schema=False)
    @guarded
    def paste_effects(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        reply = body.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise H3Error("Paste the writer's JSON reply.")
        with runs.lock:
            check_available(owner, name)
            store_effect_proposal(name, body.get("include_manual") is True, reply=reply)
        return payload(owner, name)

    @app.post(f"{api}/project/video/effects/apply", include_in_schema=False)
    @guarded
    def apply_effects(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        with runs.lock, editing(store, name) as data:
            check_available(owner, name)
            proposal = data.get("video_effect_proposal")
            if not proposal or proposal["id"] != body.get("id"):
                raise H3Error("This proposal changed. Review the current proposal before applying.")
            _, items, sounds = effect_inputs(name, data)
            _, token = effects_pass.prompt_input(items, sounds, proposal["include_manual"])
            if token != proposal["revision"]:
                raise H3Error("The shots changed. Generate a new proposal.")
            by_id = {item["shot_id"]: item for item in items}
            stacks = data.setdefault("video_effect_stacks", {})
            for decision in proposal["decisions"]:
                if decision["action"] == "replace":
                    shot_id = decision["shot_id"]
                    stacks[shot_id] = {"image": by_id[shot_id]["image"], "effects": decision["effects"]}
            data.pop("video_effect_proposal")
        return payload(owner, name)

    @app.post(f"{api}/project/video/effects/shot", include_in_schema=False)
    @guarded
    def save_effect_stack(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        with runs.lock, editing(store, name) as data:
            check_available(owner, name)
            _, items, _ = effect_inputs(name, data)
            item = find_effect_shot(items, body)
            stacks = data.setdefault("video_effect_stacks", {})
            if body.get("reset") is True:
                stacks.pop(item["shot_id"], None)
            else:
                regions = {region["id"] for region in item["regions"] or ()}
                stack = effect_enhancement.check_stack(body.get("effects"), regions)
                stacks[item["shot_id"]] = {"image": item["image"], "effects": stack, "manual": True}
        return payload(owner, name)

    @app.post(f"{api}/project/video/effects/refine", include_in_schema=False)
    @guarded
    def refine_effect_mask(name: str, request: Request, body: dict = Body(...)):
        from PIL import Image

        if not gpu:
            raise H3Error("Open Sound Stage on the GPU server to refine a mask.")
        owner, session = owner_of(request), session_of(request)
        folder, items, _ = effect_inputs(name)
        item = find_effect_shot(items, body)
        region = next((r for r in item["regions"] or () if r["id"] == body.get("region_id")), None)
        if region is None:
            raise H3Error("Choose a region to refine.")
        runtime = export.deps.runtime_config()
        image = folder / item["image"]
        with Image.open(image) as source:
            width, height = source.size
        clicks = {}
        for key in ("include", "exclude"):
            points = body.get(key) or []
            if not isinstance(points, list) or len(points) > 32 or any(
                not isinstance(p, list) or len(p) != 2 or
                any(type(n) not in (int, float) or not 0 <= n <= 1 for n in p) for p in points
            ):
                raise H3Error("Mask points must be inside the image.")
            clicks[key] = [{"x": min(width - 1, int(x * width)), "y": min(height - 1, int(y * height))}
                           for x, y in points]
        if not clicks["include"]:
            raise H3Error("Add at least one Include point inside the object.")

        def work(run):
            def generate():
                services = export.deps.generation_services()
                effect_regions.prepare(services)
                refined = effect_regions.segment(image, [region["label"]], services, runtime, clicks=clicks)
                if not refined:
                    raise H3Error("SAM 3.1 found no mask for those points.")
                check_cancelled()
                with editing(store, name) as data:
                    _, current, _ = effect_inputs(name, data)
                    find_effect_shot(current, body)
                    fixes = data.setdefault("video_effect_regions", {})
                    entry = fixes.get(item["shot_id"]) or {}
                    regions = entry.get("regions", {}) if entry.get("image") == item["image"] else {}
                    fixes[item["shot_id"]] = {"image": item["image"], "regions": {
                        **regions, region["id"]: {**refined[0], "id": region["id"]}}}
                yield "Updated mask."

            callback = owned_generation(generate, "drama3_visuals")
            for _ in run_job(run, owner, session, "drama3_visuals", [], callback):
                pass
            run.status = "Updated mask."

        return start_effect_job(request, name, REFINE_MASK, "Refining the mask…", work)

    @app.get(f"{api}/video/effect-mask", include_in_schema=False)
    def effect_mask(name: str, file: str, request: Request):
        with found():
            owner_of(request)
            data, result = project(name)
            audio, _ = planning_mix(name, data, result, export, store=store)
        match = re.fullmatch(r"([\w-]+\.jpg)\.[0-9a-f]{32}\.region_\d+\.png", file)
        root = export.deps.runtime_config().output_dir
        record = visuals.read_stills(audio, root)
        if not match or not record or not any(shot["image"] == match[1] for shot in record["shots"]):
            raise not_found()
        return file_response(visuals.stills_dir(audio, root) / effect_regions.MASK_FOLDER / file,
                             media_type="image/png", cache="private, max-age=31536000, immutable")
