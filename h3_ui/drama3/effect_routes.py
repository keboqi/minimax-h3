"""Sound Stage's optional post-draw segmentation and effect editing routes."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import contextvars
import re
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from fastapi import Body, Request

from h3_app.drama3 import effect_enhancement
from h3_app.drama3.visual_timing import shot_speech_cues
from h3_app.errors import H3Error
from h3_app.jobs import check_cancelled
from h3_gpu import is_cpu_only

from ..job_bindings import owned_generation
from ..page_runs import guarded
from . import effect_depth, effect_regions, effects_pass, visuals
from .page_common import PageRun, file_response, found, not_found, run_job, timeline_payload, writer_service
from .page_store import editing
from .project_visuals import planning_mix

SEGMENT, ENHANCE, REFINE_MASK = "segment_effects", "enhance_effects", "refine_effect_mask"
EFFECT_RUNS = (SEGMENT, ENHANCE, REFINE_MASK)
WRITER_BATCH_SIZE = 24
MAX_TOTAL_REGIONS = 24


def install_effect_routes(app, *, api, export, store, gpu, runs, owner_of, session_of, project, payload,
                          mix_of, check_available, video_settings, plan_payload):
    def has_gpu() -> bool:
        return bool(gpu and not is_cpu_only())

    def effect_context(name, editing_data=None):
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
                                       speech=shot_speech_cues(mix), depth_model=settings["depth_model"])
        if not items:
            raise H3Error("Draw the shots before enhancing their effects.")
        return (
            visuals.stills_dir(audio, root),
            items,
            timeline_payload(mix)["sounds"],
            settings,
            plan.get("art_style", ""),
        )

    def effect_inputs(name, editing_data=None):
        folder, items, sounds, _, _ = effect_context(name, editing_data)
        return folder, items, sounds

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

    def run_visual_job(run, owner, session, generate):
        callback = owned_generation(generate, "drama3_visuals")
        try:
            for _ in run_job(run, owner, session, "drama3_visuals", [], callback):
                pass
        finally:
            run.job_id, run.family = None, None

    def await_writer_reply(pool, run, fn, *args, timeout_message: str, **kwargs):
        ctx = contextvars.copy_context()
        future = pool.submit(ctx.run, fn, *args, **kwargs)
        deadline = time.monotonic() + visuals.PLAN_TIMEOUT
        while not run.stop_requested:
            try:
                return future.result(timeout=visuals.PLAN_TICK)
            except FuturesTimeout:
                if time.monotonic() >= deadline:
                    raise H3Error(timeout_message) from None
        return None

    def saved_shot_regions(data: dict, shot_id: str, image: str) -> dict:
        entry = (data.get("video_effect_regions") or {}).get(shot_id) or {}
        return dict(entry.get("regions", {})) if entry.get("image") == image else {}

    def save_shot_regions(data: dict, shot_id: str, image: str, updates: dict) -> None:
        fixes = data.setdefault("video_effect_regions", {})
        fixes[shot_id] = {"image": image, "regions": {**saved_shot_regions(data, shot_id, image), **updates}}

    @app.post(f"{api}/project/video/effects/segment", include_in_schema=False)
    @guarded
    def segment_effects(name: str, request: Request, body: dict = Body(default={})):
        if not has_gpu():
            raise H3Error("Open Sound Stage on the GPU server to run SAM 3.1.")
        folder, items, _, settings, _ = effect_context(name)
        depth_model = settings["depth_model"]
        target_items = [find_effect_shot(items, body)] if body.get("shot_id") else items
        pending = {(item["image"], tuple(item["labels"])) for item in target_items if item["regions"] is None}
        get_services = getattr(export.deps, "generation_services", None)
        has_depth_services = callable(get_services) and get_services() is not None
        depth_images = (
            effect_depth.missing_stills(folder, [item["image"] for item in target_items], depth_model)
            if has_depth_services else []
        )
        if not pending and not depth_images:
            return {**payload(owner_of(request), name), "message": "Shot regions are already available."}
        owner, session = owner_of(request), session_of(request)
        runtime = export.deps.runtime_config()

        def work(run):
            failures = []

            def generate():
                services = export.deps.generation_services()
                if pending:
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
                if depth_images:
                    def on_depth_progress(index, total):
                        run.status = f"Estimating depth for shot {index}/{total} with Depth Pro 2…"

                    effect_depth.ensure_stills_depth(
                        folder, depth_images, depth_model, services, runtime, on_progress=on_depth_progress
                    )
                    yield run.status

            run_visual_job(run, owner, session, generate)
            if pending and len(failures) == len(pending):
                raise H3Error(f"SAM 3.1 could not segment these shots. {failures[0]}")
            if pending:
                run.status = f"Processed {len(pending) - len(failures)} shots."
                if failures:
                    run.status += f" {len(failures)} failed and can be retried."
            else:
                run.status = "Depth maps ready for visual effects."

        status = "Preparing SAM 3.1…" if pending else "Estimating shot depth with Depth Pro 2…"
        return start_effect_job(request, name, SEGMENT, status, work)

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
                    reply = await_writer_reply(
                        pool,
                        run,
                        effect_enhancement.generate_reply,
                        batch,
                        token,
                        writer,
                        runtime,
                        timeout_message="The effects writer timed out. Try again or paste a reply.",
                    )
                    if reply is None:
                        return
                    decisions.extend(effect_enhancement.check_reply(reply, batch, token))
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
            shot_proposals = data.get("video_effect_shot_proposals")
            for decision in proposal["decisions"]:
                if decision["action"] == "replace":
                    shot_id = decision["shot_id"]
                    stacks[shot_id] = {"image": by_id[shot_id]["image"], "effects": decision["effects"]}
                    if isinstance(shot_proposals, dict):
                        shot_proposals.pop(shot_id, None)
            data.pop("video_effect_proposal")
        return payload(owner, name)

    def parse_shot_enhance_options(item, body):
        raw_dir = body.get("direction")
        if raw_dir is not None and not isinstance(raw_dir, str):
            raise H3Error("Enhancement direction must be text.")
        direction = " ".join((raw_dir or "").split())[:400]
        draft_raw = body.get("draft_effects")
        draft_effects = None
        if draft_raw is not None:
            region_ids = {region["id"] for region in item["regions"] or ()}
            draft_effects = effect_enhancement.check_stack(draft_raw, region_ids)
        return draft_effects, direction

    def build_shot_prompt_state(items, sounds, settings, art_style, item, draft_effects, direction):
        shot_input, token = effects_pass.shot_prompt_input(
            items,
            sounds,
            item["shot_id"],
            style_preset=settings["preset"],
            art_style=art_style,
            orientation=settings["orientation"],
            draft_effects=draft_effects,
            user_direction=direction,
        )
        can_add = bool(has_gpu() and len(item["regions"] or ()) < MAX_TOTAL_REGIONS)
        return shot_input, token, can_add

    def materialize_new_shot_regions(run, owner, session, name, body, folder, item, new_reqs, runtime):
        if not new_reqs:
            return {}, []
        if not has_gpu():
            return {}, ["Skipped new object segments because GPU segmentation is unavailable here."]
        id_map = {}
        warnings = []
        run.status = f"Segmenting {len(new_reqs)} new object(s) for shot {item['index'] + 1} with SAM 3.1…"

        def generate():
            services = export.deps.generation_services()
            effect_regions.prepare(services)
            created, seg_map, skipped = effect_regions.segment_new_regions(
                folder / item["image"],
                new_reqs,
                item["regions"],
                services,
                runtime,
                max_total=MAX_TOTAL_REGIONS,
            )
            check_cancelled()
            if created:
                with editing(store, name) as data:
                    _, current_items, _ = effect_inputs(name, data)
                    current_item = find_effect_shot(current_items, body)
                    saved_regions = saved_shot_regions(data, item["shot_id"], item["image"])
                    used_ids = {r["id"] for r in current_item["regions"] or ()} | set(saved_regions)
                    remapped_created = {}
                    next_num = 1
                    for old_cid, cspec in created.items():
                        final_cid = old_cid
                        if final_cid in used_ids:
                            final_cid, next_num = effect_regions.next_custom_id(used_ids, next_num)
                        used_ids.add(final_cid)
                        remapped_created[final_cid] = {**cspec, "id": final_cid}
                        for new_key, target_cid in list(seg_map.items()):
                            if target_cid == old_cid:
                                seg_map[new_key] = final_cid
                    save_shot_regions(data, item["shot_id"], item["image"], remapped_created)
            id_map.update(seg_map)
            if skipped:
                warnings.append(f"SAM 3.1 found no mask for: {', '.join(skipped)}.")
            yield run.status

        try:
            run_visual_job(run, owner, session, generate)
        except H3Error as exc:
            if run.stop_requested:
                raise
            warnings.append(f"Skipped new object segments ({exc}).")
        return id_map, warnings

    def commit_shot_proposal(name, body, shot_input, decision, id_map, warnings):
        with editing(store, name) as data:
            _, latest_items, _ = effect_inputs(name, data)
            latest_item = find_effect_shot(latest_items, body)
            resolved = effect_enhancement.resolve_shot_stack(
                decision, shot_input, id_map, extra_warnings=warnings
            )
            proposals = data.setdefault("video_effect_shot_proposals", {})
            proposals[latest_item["shot_id"]] = {
                "id": uuid4().hex,
                "image": latest_item["image"],
                **resolved,
            }

    @app.post(f"{api}/project/video/effects/shot/prompt", include_in_schema=False)
    @guarded
    def shot_effects_prompt(name: str, request: Request, body: dict = Body(...)):
        owner_of(request)
        _, items, sounds, settings, art_style = effect_context(name)
        item = find_effect_shot(items, body)
        draft_effects, direction = parse_shot_enhance_options(item, body)
        shot_input, token, can_add = build_shot_prompt_state(
            items, sounds, settings, art_style, item, draft_effects, direction
        )
        return {
            "prompt": effect_enhancement.copy_shot_prompt(
                shot_input,
                token,
                export.deps.runtime_config(),
                can_add_regions=can_add,
                user_direction=direction,
            )
        }

    @app.post(f"{api}/project/video/effects/shot/generate", include_in_schema=False)
    @guarded
    def generate_shot_effects(name: str, request: Request, body: dict = Body(...)):
        owner, session = owner_of(request), session_of(request)
        folder, items, sounds, settings, art_style = effect_context(name)
        initial_item = find_effect_shot(items, body)
        draft_effects, direction = parse_shot_enhance_options(initial_item, body)
        writer = writer_service(body)
        runtime = export.deps.runtime_config()
        model_label = writer[4] if writer[2] == "OpenAI-compatible" else writer[0]

        def work(run):
            cur_folder, cur_items, cur_sounds, cur_settings, cur_art = folder, items, sounds, settings, art_style
            cur_item = initial_item
            prep_warnings = []
            if has_gpu() and not run.stop_requested:
                try:
                    need_sam = cur_item["regions"] is None
                    get_services = getattr(export.deps, "generation_services", None)
                    has_depth_services = callable(get_services) and get_services() is not None
                    need_depth = (
                        effect_depth.missing_stills(cur_folder, [cur_item["image"]], cur_settings["depth_model"])
                        if has_depth_services else []
                    )
                    if need_sam or need_depth:
                        def prep_visuals():
                            services = export.deps.generation_services()
                            if need_sam:
                                run.status = f"Segmenting shot {cur_item['index'] + 1} with SAM 3.1…"
                                effect_regions.prepare(services)
                                check_cancelled()
                                effect_regions.segment(cur_folder / cur_item["image"], cur_item["labels"], services, runtime)
                                yield run.status
                            if need_depth:
                                run.status = f"Estimating depth for shot {cur_item['index'] + 1} with Depth Pro 2…"
                                effect_depth.ensure_stills_depth(
                                    cur_folder, need_depth, cur_settings["depth_model"], services, runtime
                                )
                                yield run.status

                        run_visual_job(run, owner, session, prep_visuals)
                        cur_folder, cur_items, cur_sounds, cur_settings, cur_art = effect_context(name)
                        cur_item = find_effect_shot(cur_items, body)
                except H3Error as exc:
                    if run.stop_requested:
                        raise
                    prep_warnings.append(f"Skipped GPU depth/segmentation preparation ({exc}).")

            if run.stop_requested:
                return

            shot_input, token, can_add = build_shot_prompt_state(
                cur_items, cur_sounds, cur_settings, cur_art, cur_item, draft_effects, direction
            )
            run.status = f"Enhancing shot {cur_item['index'] + 1} effects with {model_label}…"
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-shot-effects-writer")
            try:
                with TemporaryDirectory() as scratch:
                    media_values = effects_pass.shot_media_inputs(cur_folder, cur_item, scratch)
                    reply = await_writer_reply(
                        pool,
                        run,
                        effect_enhancement.generate_shot_reply,
                        shot_input,
                        token,
                        writer,
                        runtime,
                        timeout_message="The shot effects writer timed out. Try again or paste a reply.",
                        media_values=media_values,
                        can_add_regions=can_add,
                        user_direction=direction,
                    )
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            if run.stop_requested or reply is None:
                return

            decision = effect_enhancement.check_shot_reply(reply, shot_input, token)
            id_map, warnings = materialize_new_shot_regions(
                run, owner, session, name, body, cur_folder, cur_item, decision["new_regions"], runtime
            )
            with runs.lock:
                if not run.stop_requested:
                    commit_shot_proposal(name, body, shot_input, decision, id_map, prep_warnings + warnings)
                    run.status = f"Shot {cur_item['index'] + 1} effects ready to preview."

        return start_effect_job(
            request, name, ENHANCE, f"Enhancing shot {initial_item['index'] + 1} effects…", work
        )

    @app.post(f"{api}/project/video/effects/shot/reply", include_in_schema=False)
    @guarded
    def paste_shot_effects(name: str, request: Request, body: dict = Body(...)):
        owner, session = owner_of(request), session_of(request)
        reply = body.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise H3Error("Paste the writer's JSON reply.")
        folder, items, sounds, settings, art_style = effect_context(name)
        item = find_effect_shot(items, body)
        draft_effects, direction = parse_shot_enhance_options(item, body)
        shot_input, token, _ = build_shot_prompt_state(
            items, sounds, settings, art_style, item, draft_effects, direction
        )
        decision = effect_enhancement.check_shot_reply(reply, shot_input, token)
        if decision["new_regions"] and has_gpu():
            runtime = export.deps.runtime_config()

            def work(run):
                id_map, warnings = materialize_new_shot_regions(
                    run, owner, session, name, body, folder, item, decision["new_regions"], runtime
                )
                with runs.lock:
                    if not run.stop_requested:
                        commit_shot_proposal(name, body, shot_input, decision, id_map, warnings)
                        run.status = f"Shot {item['index'] + 1} effects ready to preview."

            return start_effect_job(
                request, name, ENHANCE, f"Segmenting new objects for shot {item['index'] + 1}…", work
            )

        warnings = (
            ["Skipped new object segments because GPU segmentation is unavailable here."]
            if decision["new_regions"] else []
        )
        with runs.lock:
            check_available(owner, name)
            commit_shot_proposal(name, body, shot_input, decision, {}, warnings)
        return payload(owner, name)

    @app.post(f"{api}/project/video/effects/shot", include_in_schema=False)
    @guarded
    def save_effect_stack(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        with runs.lock, editing(store, name) as data:
            check_available(owner, name)
            _, items, _ = effect_inputs(name, data)
            item = find_effect_shot(items, body)
            shot_proposals = data.get("video_effect_shot_proposals")
            if isinstance(shot_proposals, dict):
                shot_proposals.pop(item["shot_id"], None)
            if body.get("dismiss_proposal") is not True:
                stacks = data.setdefault("video_effect_stacks", {})
                if body.get("reset") is True:
                    stacks.pop(item["shot_id"], None)
                else:
                    regions = {region["id"] for region in item["regions"] or ()}
                    stack = effect_enhancement.check_stack(body.get("effects"), regions)
                    stacks[item["shot_id"]] = {"image": item["image"], "effects": stack, "manual": True}
        return payload(owner, name)

    def parse_mask_clicks(image_path, body):
        from PIL import Image

        with Image.open(image_path) as source:
            width, height = source.size
        clicks = {}
        for key in ("include", "exclude"):
            raw = body.get(key)
            points = [] if raw is None else raw
            if not isinstance(points, list) or len(points) > 32 or any(
                not isinstance(p, list) or len(p) != 2 or
                any(type(n) not in (int, float) or not 0 <= n <= 1 for n in p) for p in points
            ):
                raise H3Error("Mask points must be inside the image.")
            clicks[key] = [{"x": min(width - 1, int(x * width)), "y": min(height - 1, int(y * height))}
                           for x, y in points]
        if not clicks["include"]:
            raise H3Error("Add at least one Include point inside the object.")
        return clicks

    @app.post(f"{api}/project/video/effects/refine", include_in_schema=False)
    @guarded
    def refine_effect_mask(name: str, request: Request, body: dict = Body(...)):
        if not has_gpu():
            raise H3Error("Open Sound Stage on the GPU server to refine a mask.")
        owner, session = owner_of(request), session_of(request)
        folder, items, _ = effect_inputs(name)
        item = find_effect_shot(items, body)
        region = next((r for r in item["regions"] or () if r["id"] == body.get("region_id")), None)
        if region is None:
            raise H3Error("Choose a region to refine.")
        runtime = export.deps.runtime_config()
        image = folder / item["image"]
        clicks = parse_mask_clicks(image, body)

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
                    save_shot_regions(data, item["shot_id"], item["image"], {
                        region["id"]: {**refined[0], "id": region["id"]}
                    })
                yield "Updated mask."

            run_visual_job(run, owner, session, generate)
            run.status = "Updated mask."

        return start_effect_job(request, name, REFINE_MASK, "Refining the mask…", work)

    @app.post(f"{api}/project/video/effects/region", include_in_schema=False)
    @guarded
    def add_effect_region(name: str, request: Request, body: dict = Body(...)):
        if not gpu:
            raise H3Error("Open Sound Stage on the GPU server to add a segment.")
        owner, session = owner_of(request), session_of(request)
        folder, items, _ = effect_inputs(name)
        item = find_effect_shot(items, body)
        if item["regions"] is None:
            raise H3Error("Segment the shot before adding a custom segment.")
        if len(item["regions"]) >= MAX_TOTAL_REGIONS:
            raise H3Error(f"A shot can have at most {MAX_TOTAL_REGIONS} segments.")
        raw_label = body.get("label")
        if raw_label is not None and not isinstance(raw_label, str):
            raise H3Error("Segment label must be text.")
        label = " ".join((raw_label or "").split())[:48]
        runtime = export.deps.runtime_config()
        image = folder / item["image"]
        clicks = parse_mask_clicks(image, body)

        def work(run):
            def generate():
                services = export.deps.generation_services()
                effect_regions.prepare(services)
                created = effect_regions.segment(image, [label or "segment"], services, runtime, clicks=clicks)
                if not created:
                    raise H3Error("SAM 3.1 found no mask for those points.")
                check_cancelled()
                with editing(store, name) as data:
                    _, current, _ = effect_inputs(name, data)
                    current_item = find_effect_shot(current, body)
                    if current_item["regions"] is None:
                        raise H3Error("Segment the shot before adding a custom segment.")
                    regions = saved_shot_regions(data, item["shot_id"], item["image"])
                    used_ids = {r["id"] for r in current_item["regions"]} | set(regions)
                    new_id, next_after = effect_regions.next_custom_id(used_ids)
                    final_label = label or f"segment {next_after - 1}"
                    save_shot_regions(data, item["shot_id"], item["image"], {
                        new_id: {**created[0], "id": new_id, "label": final_label}
                    })
                yield "Added segment."

            run_visual_job(run, owner, session, generate)
            run.status = "Added segment."

        return start_effect_job(request, name, REFINE_MASK, "Creating the mask…", work)

    def serve_sidecar_png(name: str, file: str, request: Request, folder_name: str, pattern: str):
        with found():
            owner_of(request)
            data, result = project(name)
            audio, _ = planning_mix(name, data, result, export, store=store)
        match = re.fullmatch(pattern, file)
        root = export.deps.runtime_config().output_dir
        record = visuals.read_stills(audio, root)
        if not match or not record or not any(shot["image"] == match[1] for shot in record["shots"]):
            raise not_found()
        return file_response(
            visuals.stills_dir(audio, root) / folder_name / file,
            media_type="image/png",
            cache="private, max-age=31536000, immutable",
        )

    @app.get(f"{api}/video/effect-mask", include_in_schema=False)
    def effect_mask(name: str, file: str, request: Request):
        return serve_sidecar_png(
            name, file, request, effect_regions.MASK_FOLDER,
            r"([\w-]+\.(?:jpg|jpeg|png))\.[0-9a-f]{32}\.region_\d+\.png",
        )

    @app.get(f"{api}/video/effect-depth", include_in_schema=False)
    def effect_depth_map(name: str, file: str, request: Request):
        return serve_sidecar_png(
            name, file, request, effect_depth.DEPTH_FOLDER,
            r"([\w-]+\.(?:jpg|jpeg|png))\.[0-9a-f]{16}\.depth\.png",
        )
