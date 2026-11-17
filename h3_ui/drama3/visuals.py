"""The illustrated video from a Soundstage mix: plan shots, draw them, then play or render them.

Drawing keeps a web copy of each image with its time on the mix (the *stills*), which the page
slides along the mix as it plays. The video itself is rendered from those stills only when it is
saved to Media, since encoding takes a while and the file is large to load.
"""

from dataclasses import dataclass, field
import hashlib
import inspect
from typing import Any, Callable
import json
import re
from collections import OrderedDict
from itertools import count
from pathlib import Path
from math import ceil
from shutil import copyfile, which
from threading import Lock
from uuid import uuid4

import gradio as gr

from h3_app.capabilities import engines_for_task
from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL, QWEN_EDIT_SIZE_MATCH
from h3_app.burned_subtitles import Line, ass_subtitles, word_pieces
from h3_app.drama3.visual_plan import (ORIENTATIONS, compile_plan, plan_art_style, sheet_prompt,
                                      shot_cap, subjects)
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.drama3.result_view import timeline_data
from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY
from h3_ui.drama3.visual_styles import (CUSTOM_STYLE, DEFAULT_STYLE, SETTING_DETAIL, SLIDE_STYLES,
                                       STYLE_GROUPS, STYLE_PRESETS, STYLE_PREVIEWS, STYLE_USES, _PRESETS,
                                       preset_for, reference_direction)
from h3_ui.image_controller import GENERATION_OPTIONS, reference_limit, reference_tag


# Images finished by an export that failed or was stopped, by export fingerprint:
# {"portraits": {key: path}, "shots": [path, ...]}. A retry with the same mix, plan, model
# and settings resumes after them.
_FINISHED: OrderedDict[str, dict] = OrderedDict()
_FINISHED_LIMIT = 8
PLAN_TICK = 3  # Seconds between planning status updates.
PLAN_TIMEOUT = 300  # Seconds before planning gives up on the writer.
# Widescreen sizes on each model's grid, so stills fill the 1280x720 video without bars.
_WIDESCREEN = {"qwen": (1664, 928), "h3": (1664, 928), "flux3": (1344, 768),
               "moss_image": (1344, 768), "fibo2": (1344, 768), "gpt_image": (1920, 1088)}


PORTRAIT_SIZE = 1024


def export_key(result_name, plan, engine, use_references, captured):
    data = json.dumps([result_name, plan, engine, bool(use_references), captured],
                      sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def finished_images(key):
    images = _FINISHED.setdefault(key, {"portraits": {}, "shots": []})
    _FINISHED.move_to_end(key)
    while len(_FINISHED) > _FINISHED_LIMIT:
        _FINISHED.popitem(last=False)
    return images


def _size(engine, captured, width, height):
    """Draw at exactly this size; edits too, instead of following a reference's shape."""
    if engine == "h3":
        captured.update(image__width=width, image__height=height, image__edit_size=QWEN_EDIT_SIZE_MANUAL)
    elif engine == "gpt_image":
        gpt_width, gpt_height = (1088, 1920) if height > width else (1920, 1088)
        captured.update(gpt_image__width=gpt_width, gpt_image__height=gpt_height)
    else:
        captured.update({f"{engine}__width": width, f"{engine}__height": height})
        if engine in ("qwen", "flux3"):
            captured[f"{engine}__edit_size"] = QWEN_EDIT_SIZE_MANUAL


def widescreen(engine, captured, orientation="landscape"):
    """One image per request at a 16:9 size (9:16 for ``portrait``), whatever the Image tab is set to."""
    captured.update(qwen__batch_count=1, qwen__batch_edit_inputs=False, moss_image__layout="", flux3__layout="",
                    qwen__output_resolution_1k=None, qwen__output_resolution_2k=None, gpt_image__batch_count=1)
    width, height = _WIDESCREEN.get(engine, (1344, 768))
    _size(engine, captured, *((height, width) if orientation == "portrait" else (width, height)))


# The rendered video's size in each frame, so its stills fill it without bars.
VIDEO_SIZES = {"landscape": (1280, 720), "portrait": (720, 1280)}


def orientation_of(value):
    """A known frame (``landscape`` or ``portrait``); anything else is the original landscape."""
    return value if value in ORIENTATIONS else "landscape"


def character_match_request(engine, scene, portraits, shot, captured):
    """A bounded identity edit, without re-captioning the scene or asking for a style render.

    Qwen's matching canvas must track the first reference, including its latent dimensions.
    A native enhancer is useful for an open-ended draft, but must not expand this local edit.
    Other sampling settings remain exactly those selected on the Image page.
    """
    count = len(portraits) + 1
    scene_tag = reference_tag(engine, 1, count) or "the first image"
    lines = [f"In {scene_tag}, change only the depicted characters' facial identity, hairstyle and outfit "
             "design to match the following portrait references. Keep each person's current pose, "
             "expression, gaze and position."]
    for position, key in enumerate(portraits, 2):
        tag = reference_tag(engine, position, count) or f"(reference image {position})"
        # Use only this person's anchor to identify the edit target. Full shot text also describes
        # walls, rain, light and materials, turning preservation into a second scene-generation brief.
        anchor = re.search(rf"{re.escape(key)}⟦{re.escape(key)}⟧\s*\(([^;)]+)", shot.tagged or "")
        target = anchor.group(1).strip() if anchor else key
        lines.append(f"For {target}, use {tag} for facial identity, hairstyle and outfit design only.")
    lines += [f"Preserve {scene_tag}'s background, objects, composition, lighting, colors, rendering style "
              "and level of surface detail unchanged. Render the adjusted characters in that same style "
              "and lighting; do not copy the portraits' background or shading.",
              "Keep the existing number of people. No added figures, panels or insets."]
    options = dict(captured)
    if engine == "qwen":
        options.update(qwen__edit_size=QWEN_EDIT_SIZE_MATCH, qwen__reference_resolution=0,
                       qwen__output_resolution_1k=None, qwen__output_resolution_2k=None,
                       qwen__prompt_enhancer="Off")
    return "\n".join(lines), [scene, *portraits.values()], options


def match_characters(controller, engine, scene, portraits, shot, captured, *, record=None):
    """Repaint a finished scene's characters from their portraits, keeping everything else.

    The scene is the first input, so an edit model that leans toward its input keeps the scene's
    composition; the portraits only steer faces, hair and outfits. Each portrait's tag (<image2>,
    <Picture 2>...) is bound to its character's anchor, without re-describing the rest of the scene.
    """
    prompt, references, options = character_match_request(engine, scene, portraits, shot, captured)
    if record is not None:
        option_names = (
            ("models", "use_failback", "quality", "width", "height", "seed")
            if engine == "gpt_image"
            else tuple(name for name, _ in GENERATION_OPTIONS.get(engine, ()))
        )
        record.update(
            prompt=prompt,
            references=[Path(path).name for path in references],
            settings={name: options[f"{engine}__{name}"] for name in option_names if f"{engine}__{name}" in options},
        )
    return controller.generate(engine, "Image edit", prompt, references, options)


def _stream(updates, label):
    """Relay one image request's progress; return the image path it produced."""
    path, message = None, "No image returned."
    for result, message in updates:
        if message.startswith("Error:"):
            raise H3Error(f"{label} failed. {message}")
        yield gr.skip(), f"{label} · {message}"
        if result:
            path = result[0] if isinstance(result, list) else result
    if not path or not Path(path).is_file():
        raise H3Error(f"{label} failed. {message}")
    return path


def _parallel_images(tasks, captured):
    """Stream independent API jobs; publish results only on the calling thread."""
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from queue import Queue, Empty
    from threading import Event
    from h3_app.gpt_image import (DEFAULT_GPT_IMAGE_MODEL, gpt_image_credentials,
                                  gpt_image_max_concurrency, normalize_gpt_image_models)
    from h3_app.jobs import check_cancelled

    models = normalize_gpt_image_models(captured.get("gpt_image__models", (DEFAULT_GPT_IMAGE_MODEL,)))
    workers = gpt_image_max_concurrency(models, bool(captured.get("gpt_image__use_failback")),
                                        credentials=gpt_image_credentials())
    events, stop = Queue(), Event()

    def run(index, task):
        iterator = None
        try:
            options = {**captured, "gpt_image__models": (models[index % len(models)],),
                       "gpt_image__batch_count": 1}
            iterator = task(options)
            while not stop.is_set():
                check_cancelled()
                try:
                    update = next(iterator)
                except StopIteration as done:
                    events.put((index, True, done.value))
                    return
                events.put((index, False, update))
        except BaseException as exc:
            events.put((index, True, exc))
        finally:
            if iterator is not None:
                iterator.close()

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="soundstage-image")
    try:
        for index, task in enumerate(tasks):
            pool.submit(copy_context().run, run, index, task)
        pending, errors = len(tasks), []
        while pending:
            check_cancelled()
            try:
                index, done, value = events.get(timeout=0.25)
            except Empty:
                continue
            if isinstance(value, BaseException):
                errors.append(value)
                pending -= 1
                continue
            if done:
                pending -= 1
            yield index, done, value
        if errors:
            raise errors[0]
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)


def illustrate_shots(controller, engine, shots, captured, use_references, finished=None, portraits=None,
                     orientation="landscape", *, on_shot=None):
    if engine != "gpt_image":
        return (yield from _illustrate_shots_serial(controller, engine, shots, captured, use_references,
                                                    finished, portraits, orientation, on_shot=on_shot))
    finished = {"portraits": {}, "shots": []} if finished is None else finished
    drawn = finished.setdefault("portraits", {})
    details = finished.setdefault("shot_details", {})
    done = finished.setdefault("shots", [])
    portraits = portraits or {}
    options = dict(captured)
    widescreen(engine, options, orientation)
    cast = list(dict.fromkeys(key for shot in shots for key in shot.references if key in portraits)) if use_references else []
    missing = [key for key in cast if not drawn.get(key) or not Path(drawn[key]).is_file()]

    def portrait_task(key):
        def task(values):
            _size(engine, values, PORTRAIT_SIZE, PORTRAIT_SIZE)
            return (yield from _stream(controller.generate(engine, "Text to image", portraits[key], [], values),
                                       f"Portrait · {key}"))
        return task

    for index, complete, value in _parallel_images([portrait_task(key) for key in missing], options):
        if complete:
            drawn[missing[index]] = value
        else:
            yield value

    def shot_task(index):
        def task(values):
            local = {"portraits": dict(drawn), "shots": [], "shot_details": {}}
            if index < len(done) and done[index] and Path(done[index]).is_file():
                local["shots"] = [done[index]]
                local["shot_details"] = {0: details.get(index, {})}
            return (yield from _illustrate_shots_serial(controller, engine, [shots[index]], values,
                                                        use_references, local, portraits, orientation))
        return task

    images, records = [None] * len(shots), [None] * len(shots)
    for index, complete, value in _parallel_images([shot_task(i) for i in range(len(shots))], options):
        if not complete:
            yield value[0], str(value[1]).replace("Shot 1 of 1", f"Shot {index + 1} of {len(shots)}")
            continue
        paths, shot_records, _ = value
        images[index], records[index] = paths[0], shot_records[0]
        while len(done) <= index:
            done.append(None)
        done[index] = paths[0]
        details[index] = {key: item for key, item in records[index].items() if key in STILL_METADATA_FIELDS}
        if on_shot is not None:
            record = on_shot(index, paths[0], records[index])
            yield str(record), f"Shot {index + 1} of {len(shots)} · Ready"
    return images, records, {key: Path(drawn[key]).name for key in cast}


def _illustrate_shots_serial(controller, engine, shots, captured, use_references, finished=None, portraits=None,
                     orientation="landscape", *, on_shot=None):
    """Draw every scene from its text, then match its characters to their portraits.

    ``portraits`` maps character keys to portrait prompts; only characters some shot shows are drawn,
    once, before any scene. Locations and props stay in the scene text: an edit model handed a location
    or a multi-view sheet as a starting point copies it instead of composing a new scene.
    ``finished`` holds images from an earlier attempt at the same export; existing files are reused,
    and each new image is recorded so a later retry can resume after it.
    ``on_shot`` publishes each completed shot before drawing the next, returning its record path.
    """
    finished = {"portraits": {}, "shots": []} if finished is None else finished
    finished.setdefault("portraits", {})
    shot_details = finished.setdefault("shot_details", {})
    widescreen(engine, captured, orientation)
    limit = reference_limit(engine, captured.get("qwen__turbo_variant", "Off")) if use_references else 0
    slots = max(0, limit - 1)  # The scene itself takes the first input.
    portraits = portraits or {}
    cast = list(dict.fromkeys(key for shot in shots for key in shot.references if key in portraits))
    drawn = {}
    if slots:
        square = {**captured}
        _size(engine, square, PORTRAIT_SIZE, PORTRAIT_SIZE)
        for number, key in enumerate(cast, 1):
            label = f"Portrait {number} of {len(cast)} · {key}"
            saved = finished["portraits"].get(key)
            if saved and Path(saved).is_file():
                drawn[key] = saved
                yield gr.skip(), f"{label} · Reused"
                continue
            yield gr.skip(), label
            drawn[key] = yield from _stream(
                controller.generate(engine, "Text to image", portraits[key], [], square), label)
            finished["portraits"][key] = drawn[key]
    shot_records = []
    versions = {name: portrait_version(path) for name, path in drawn.items()}
    images = []
    done = finished["shots"]
    for index, shot in enumerate(shots):
        who = {key: drawn[key] for key in shot.references if key in drawn}
        who = dict(list(who.items())[:slots])
        label = f"Shot {index + 1} of {len(shots)}"
        details = {}
        if index < len(done) and Path(done[index]).is_file():
            path = done[index]
            details = shot_details.get(index, {})
            yield gr.skip(), f"{label} · Reused from the interrupted export"
        else:
            del done[index:]
            for old in [i for i in shot_details if i >= index]:
                del shot_details[old]
            yield gr.skip(), label
            path = yield from _stream(controller.generate(engine, "Text to image", shot.prompt, [], captured), label)
            if who:
                draft, label = path, f"{label} · Matching " + ", ".join(who)
                details = {"draft": str(draft), "matching": {}}
                path = yield from _stream(
                    match_characters(controller, engine, draft, who, shot, captured,
                                     record=details["matching"]), label)
                details["matching"]["output"] = Path(path).name
                snapshot = read_snapshot(path) if engine == "qwen" else None
                if snapshot:
                    details["matching"].update(job_id=snapshot.get("job_id"),
                                               effective_settings=snapshot.get("settings", {}))
            done.append(path)
            shot_details[index] = details
        images.append(path)
        shot_records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                             "image": Path(path).name, "draft": None, "characters": list(who),
                             "portrait_versions": {name: versions[name] for name in who}, **details})
        if on_shot is not None:
            record = on_shot(index, path, shot_records[-1])
            yield str(record), f"Shot {index + 1} of {len(shots)} · Ready"
    portrait_records = {key: Path(path).name for key, path in drawn.items()}
    return images, shot_records, portrait_records


def plan_path(audio):
    """The shot plan saved beside a Sound Stage mix."""
    audio = Path(audio)
    return audio.with_name(f".{audio.stem}.visual_plan.json")


def write_plan(audio, style, limit, plan):
    """Save the plan with the art direction chosen when it was made, which only an older plan's own
    planner-written style depends on, and the Minimum shots it was made for."""
    path = plan_path(audio)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"style": style, "limit": int(limit), "plan": plan}, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(path)


def read_plan(audio, limit=None):
    """The plan saved with this mix, or None; with ``limit``, only a plan made for that Minimum shots.
    The art direction and frame are added when drawing, so any of them use the same plan."""
    try:
        saved = json.loads(plan_path(audio).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or not isinstance(saved.get("plan"), dict):
        return None
    if limit is not None and saved.get("limit") != int(limit):
        return None
    return saved


# The newest planning run per mix. A run only saves its plan while it is still the newest, so a slow
# writer's late reply never replaces a plan made or pasted after that run started.
_PLAN_RUNS: dict[str, int] = {}
_PLAN_LOCK = Lock()
_RUN_IDS = count(1)


def claim_plan(audio):
    """Start a planning run for this mix; runs started before it can no longer save."""
    with _PLAN_LOCK:
        run = next(_RUN_IDS)
        _PLAN_RUNS[str(Path(audio))] = run
    return run


def save_plan(audio, style, limit, plan, run):
    """Save a run's plan unless a newer run started since; True when it was saved."""
    with _PLAN_LOCK:
        if _PLAN_RUNS.get(str(Path(audio))) != run:
            return False
        write_plan(audio, style, limit, plan)
        return True


def load_mix(deps, result_name):
    """A Sound Stage mix by its file name, and its settings snapshot."""
    if not result_name:
        raise H3Error("Generate speech before exporting an illustrated video.")
    audio = Path(deps.resolve_result(Path(result_name).name))
    snapshot = read_snapshot(audio) or {}
    if snapshot.get("family") != DRAMA3_FAMILY or not audio.is_file():
        raise H3Error("The saved Sound Stage mix could not be found.")
    return audio, snapshot["settings"]


STILLS_RECORD = "stills.json"
STILLS_QUALITY = 92  # JPEG quality of the web copies: close to the drawn image, a fraction of its size.
STILL_METADATA_FIELDS = ("draft", "characters", "matching", "portrait_versions", "portrait_stale")


def stills_dir(audio, root):
    """Where a mix's stills are kept: under ``root``'s ``.soundstage`` folder, which Media leaves alone."""
    return Path(root) / ".soundstage" / "stills" / Path(audio).stem


def prompt_key(prompt):
    """A short fingerprint of a portrait prompt: a kept portrait is reused while its prompt is unchanged."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def portrait_version(path):
    """Identity of the pixels, independent of PNG metadata or re-encoding."""
    from PIL import Image
    with Image.open(path) as image:
        pixels = image.convert("RGBA")
        return hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()[:16]


def _keep_portraits(folder, portraits):
    """Publish immutable portraits, preserving existing names while their pixels match."""
    from PIL import Image

    kept = {}
    for key, (image, fingerprint) in (portraits or {}).items():
        image = Path(image)
        version = portrait_version(image)
        name = f"portrait-{fingerprint}-{version}.png"
        if image.name.startswith("portrait-") and image.suffix.lower() == ".png":
            target = folder / image.name
            if image.resolve() == target.resolve() or not target.exists() or portrait_version(target) == version:
                name = image.name
        target = folder / name
        if not target.exists():
            with Image.open(image) as picture:
                picture.save(target, "PNG")
        kept[key] = {"image": name, "prompt": fingerprint, "version": version}
    return kept


def _still_metadata(item, *, folder=None, previous_versions=None):
    """Keep drawing provenance through checkpoints, partial redraws and mix retiming."""
    extra = {key: item[key] for key in STILL_METADATA_FIELDS if item.get(key) is not None}
    if folder is not None and extra.get("draft"):
        extra["draft"] = str(folder / extra["draft"])
    if previous_versions is not None and "portrait_versions" not in extra:
        extra["portrait_versions"] = {name: previous_versions[name] for name in item.get("characters") or []
                                       if name in previous_versions}
    return extra


def _write_stills_record(folder, record):
    """Replace the stills record in one step, so a reader never sees half of it."""
    temporary = folder / f".{record['revision']}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(folder / STILLS_RECORD)
    return folder / STILLS_RECORD


def keep_stills(audio, root, shots, images, details, *, extras=None, total_frames=None, portraits=None,
                orientation="landscape"):
    """Keep a web copy of each drawn image with its shot's time on the mix; returns the stills record.

    The record replaces the mix's earlier stills, but an image already among them (a shot drawn
    before and kept) is copied as it is. It may hold only some of the plan's shots: each image then
    holds until the next. ``extras`` adds fields to each shot's entry, ``total_frames`` is the mix's
    length in frames, and ``portraits`` maps character keys to ``(image, prompt fingerprint)`` so a
    later drawing of some shots matches its characters to the same portraits. ``details`` is what a
    video rendered from the stills records in its settings snapshot.
    """
    from PIL import Image

    folder = stills_dir(audio, root)
    folder.mkdir(parents=True, exist_ok=True)
    revision = uuid4().hex[:12]
    extras = extras or [{} for _ in shots]
    entries = sorted(zip(shots, images, extras, strict=True), key=lambda entry: entry[0].first)
    records = []
    for index, (shot, image, extra) in enumerate(entries):
        name = f"{revision}-{index + 1:03d}.jpg"
        image = Path(image)
        if image.parent.resolve() == folder.resolve():
            name = image.name  # Reuse immutable web copies while publishing subsequent shots.
        else:
            with Image.open(image) as picture:
                picture.convert("RGB").save(folder / name, "JPEG", quality=STILLS_QUALITY, optimize=True)
        extra = dict(extra)
        if extra.get("draft"):
            draft = Path(extra["draft"])
            if not draft.is_absolute():
                draft = folder / draft
            if draft.is_file():
                draft_name = f"{revision}-{index + 1:03d}-draft.png"
                if draft.parent.resolve() == folder.resolve():
                    draft_name = draft.name
                elif draft.suffix.lower() == ".png":
                    copyfile(draft, folder / draft_name)
                else:
                    with Image.open(draft) as picture:
                        picture.save(folder / draft_name, "PNG")
                extra["draft"] = draft_name
        records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                        "beat": shot.beat, "prompt": shot.prompt, "image": name,
                        **({"shot_id": shot.id} if shot.id else {}), **extra})
    kept = _keep_portraits(folder, portraits)
    for item in records:
        versions = item.get("portrait_versions") or {}
        if versions:
            item["portrait_stale"] = any((kept.get(name) or {}).get("version") != version
                                         for name, version in versions.items())
    record = _write_stills_record(folder, {"revision": revision, "source": Path(audio).name, "fps": FPS,
                                           "total_frames": total_frames, "shots": records, "portraits": kept,
                                           "orientation": orientation_of(orientation), "details": details})
    keep = {item["image"] for item in kept.values()}
    keep.update(item["image"] for item in records)
    keep.update(item["draft"] for item in records if item.get("draft"))
    for old in [*folder.glob("*.jpg"), *folder.glob("*.png")]:
        if not old.name.startswith(f"{revision}-") and old.name not in keep:
            old.unlink(missing_ok=True)
    return record


def kept_portraits(record, root, audio):
    """The record's portraits as ``{key: (path, prompt fingerprint)}``, those whose file still exists."""
    folder = stills_dir(audio, root)
    found = {}
    for key, item in ((record or {}).get("portraits") or {}).items():
        path = folder / str(item.get("image") or "")
        if item.get("image") and path.is_file():
            found[key] = (path, item.get("prompt"))
    return found


def match_stills(shots, record):
    """Match stable shot identity; only older records may fall back to their clock."""
    items = (record or {}).get("shots") or []
    by_time = {(int(item["first_frame"]), int(item["frames"])): item for item in items}
    legacy = {key: item for key, item in by_time.items() if not item.get("shot_id")}
    by_id = {item["shot_id"]: item for item in items if item.get("shot_id")}
    return [(by_id.get(shot.id) or legacy.get((shot.first, shot.frames))) if shot.id
            else by_time.get((shot.first, shot.frames)) for shot in shots]


def rebind_stills(audio, root, shots, *, previous_audio=None, previous_shots=None):
    """Move existing drawings to a new clock, including a regenerated mix's new file.

    Stable shot IDs preserve partial drawings. Older records are matched against their old
    plan's times once, before their timestamps are changed. Prompts remain those actually drawn.
    """
    previous_audio = previous_audio or audio
    record = read_stills(previous_audio, root)
    if record is None:
        return
    old_shots = previous_shots if previous_shots is not None else shots
    old_items = match_stills(old_shots, record)
    by_id = {shot.id: item for shot, item in zip(old_shots, old_items) if shot.id and item}
    if len(old_shots) == len(shots):
        for old_shot, shot, item in zip(old_shots, shots, old_items):
            if item and not old_shot.id and old_shot.beat == shot.beat:
                by_id.setdefault(shot.id, item)
    pairs = []
    for shot, item in zip(shots, match_stills(shots, record)):
        item = by_id.get(shot.id) or item
        if item is not None:
            pairs.append((shot, item))
    if not pairs:
        return
    total = shots[-1].first + shots[-1].frames
    same = Path(previous_audio) == Path(audio)
    if same and record.get("total_frames") == total and all(
            (item["first_frame"], item["frames"], bool(item["dissolve"]), item.get("shot_id", "")) ==
            (shot.first, shot.frames, shot.dissolve, shot.id) for shot, item in pairs):
        return
    folder = stills_dir(previous_audio, root)
    images = [folder / item["image"] for _, item in pairs]
    if not same:
        destination = stills_dir(audio, root)
        destination.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            target = destination / image.name
            copyfile(image, target)
            images[index] = target
    drawings = [Shot(shot.first, shot.frames, shot.dissolve, item.get("prompt", ""),
                     shot.beat, id=shot.id) for shot, item in pairs]
    extras = [_still_metadata(item, folder=folder) for _, item in pairs]
    keep_stills(audio, root, drawings, images, record.get("details") or {},
                extras=extras, total_frames=total, portraits=kept_portraits(record, root, previous_audio),
                orientation=record.get("orientation", "landscape"))


def retime_stills(audio, root, shots):
    """Give each kept still its plan shot's cut or dissolve, which needs no redrawing.

    A change gives the record a new revision, so a saved video made before it is no longer current.
    """
    record = read_stills(audio, root)
    if record is None:
        return
    changed = False
    for shot, still in zip(shots, match_stills(shots, record)):
        if still is not None and bool(still.get("dissolve")) != shot.dissolve:
            still["dissolve"] = shot.dissolve
            changed = True
    if changed:
        _write_stills_record(stills_dir(audio, root), {**record, "revision": uuid4().hex[:12]})


def shot_status(shot, still):
    """``drawn``, ``changed`` (drawn from another prompt) or ``todo`` (nothing drawn for its time)."""
    if still is None:
        return "todo"
    return "drawn" if still.get("prompt") == shot.prompt and not still.get("portrait_stale") else "changed"


def read_stills(audio, root):
    """The mix's stills record, or None when it has none or an image is missing."""
    folder = stills_dir(audio, root)
    try:
        record = json.loads((folder / STILLS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shots = record.get("shots") if isinstance(record, dict) else None
    if (not shots or record.get("source") != Path(audio).name
            or not all((folder / str(shot.get("image") or "")).is_file() for shot in shots)):
        return None
    return record


def stills_shots(record):
    """A stills record as the shots and image file names the renderer takes.

    A record may hold only some of the plan's shots: each image holds until the next one, the first
    from the start of the mix and the last to its end.
    """
    items = sorted(record["shots"], key=lambda item: int(item["first_frame"]))
    starts = [0, *(int(item["first_frame"]) for item in items[1:])]
    last = int(items[-1]["first_frame"]) + int(items[-1]["frames"])
    ends = [*starts[1:], max(last, int(record.get("total_frames") or 0))]
    shots = [Shot(start, end - start, bool(item["dissolve"]) and index > 0, item.get("prompt") or "",
                  item.get("beat") or "")
             for index, (item, start, end) in enumerate(zip(items, starts, ends))]
    return shots, [item["image"] for item in items]


def burned_subtitles(audio, size, orientation="landscape") -> str | None:
    """The mix's lines as an ASS script at ``size`` percent, as the page's video preview shows them,
    each aligned word lighting up as it is spoken; None when the mix has no timed lines."""
    entries, words, _tones, _sounds = timeline_data((read_snapshot(Path(audio)) or {}).get("settings") or {})
    lines = [Line(speaker, text, start, end, word_pieces(text, line_words))
             for (speaker, text, start, end), line_words in zip(entries, words)]
    width, height = VIDEO_SIZES[orientation_of(orientation)]
    return ass_subtitles(lines, width=width, height=height, size=size) if lines else None


def render_stills(audio, root, subtitles: int = 0):
    """Render the video from the mix's stills, beside the mix, and record it so it shows in Media.

    ``subtitles`` is the burned-in subtitles' size in percent, or 0 for none. The video's record
    keeps it, so the page can tell a video made with other subtitles from the one asked for now.
    """
    audio = Path(audio)
    record = read_stills(audio, root)
    if record is None:
        raise H3Error("Draw the images first: the video is rendered from them.")
    if which("ffmpeg") is None:
        raise H3Error("Install FFmpeg to render an illustrated video.")
    shots, names = stills_shots(record)
    folder = stills_dir(audio, root)
    orientation = orientation_of(record.get("orientation"))
    script = burned_subtitles(audio, subtitles, orientation) if subtitles else None
    video = render_video(audio, shots, [folder / name for name in names], audio.parent, subtitles=script,
                         size=VIDEO_SIZES[orientation])
    write_snapshot(video, {"family": ILLUSTRATED_FAMILY, "settings": {
        **(record.get("details") or {}), "source": audio.name, "stills": record["revision"],
        "subtitles": subtitles}})
    return video


def legacy_art_style(saved, style):
    """An older plan's planner-written style, while the art direction it was written for is chosen;
    with another style it would fight it, and new plans have none."""
    same = (saved.get("style") or "").strip() == (style or "").strip()
    return plan_art_style(saved["plan"]) if same else None


def plan_shots_for(saved, duration, style, orientation="landscape"):
    """The saved plan's shots as drawing uses them: in the chosen frame and art direction."""
    return compile_plan(saved["plan"], duration, style, shot_cap(duration, saved["limit"]),
                        orientation=orientation_of(orientation), art_style=legacy_art_style(saved, style))


def portrait_prompts(saved, style, *, with_setting=False):
    """Each character's portrait prompt, from the saved plan's looks in the chosen art direction.
    ``with_setting`` gives the prompt portraits were drawn from before SETTING_DETAIL was left out."""
    art_style = legacy_art_style(saved, style)
    direction = style if with_setting else reference_direction(style)
    return {name: sheet_prompt(kind, name, anchor, art_style, direction)
            for name, (kind, anchor) in subjects(saved["plan"]).items() if kind == "characters"}


def portrait_keys(saved, style):
    """The prompt fingerprints a kept portrait may carry and still be current, by character.
    Portraits drawn while the prompt still had SETTING_DETAIL stay current, so redrawing one shot
    of an older project keeps matching the faces already drawn."""
    old = portrait_prompts(saved, style, with_setting=True)
    keys = {name: {prompt_key(prompt), prompt_key(old[name])}
            for name, prompt in portrait_prompts(saved, style).items()}
    # Before style moved to the front, the sheet body came first and the style line
    # was unconditional (including "Art style: None" for style-neutral plans).
    art_style = legacy_art_style(saved, style)
    for name, (kind, anchor) in subjects(saved["plan"]).items():
        if kind == "characters":
            legacy = "\n".join(filter(None, [sheet_prompt(kind, name, anchor, None, ""),
                                             f"Art style: {art_style}", style]))
            keys[name].add(prompt_key(legacy))
    return keys


def chosen_shots(shots, stills, only):
    """The shot indexes a drawing draws: ``all``, ``missing`` (not drawn, or drawn from another
    prompt), or a list of indexes."""
    if only in (None, "all"):
        return list(range(len(shots)))
    if only == "missing":
        return [i for i, (shot, still) in enumerate(zip(shots, stills)) if shot_status(shot, still) != "drawn"]
    if not isinstance(only, list) or not only or not all(type(i) is int and 0 <= i < len(shots) for i in only):
        raise H3Error("Those shots are not in the current shot plan. Reload the page and try again.")
    return sorted(set(only))


def draw_stills(controller, deps, result_name, engine, style, limit, use_references, orientation, *values,
                only=None, mix=None):
    """Draw a mix's saved shot plan in ``style`` and ``orientation`` and keep the images as its stills;
    yields (record or skip, status).

    ``only`` limits the drawing to some shots (see ``chosen_shots``). The others keep the image
    already drawn for their time on the mix, and kept portraits drawn from the same prompt are
    reused, so a redrawn shot's characters match the rest.
    """
    audio, settings = mix if mix is not None else load_mix(deps, result_name)
    saved = read_plan(audio, limit)
    if saved is None:
        raise H3Error("This mix has no shot plan for the current Minimum shots yet. "
                      "Click Plan shots, or paste a plan from another chat model, then draw.")
    orientation = orientation_of(orientation)
    duration = settings["duration"]
    root = deps.runtime_config().output_dir
    shots = plan_shots_for(saved, duration, style, orientation)
    record = read_stills(audio, root)
    stills = match_stills(shots, record)
    picked = chosen_shots(shots, stills, only)
    if not picked:
        raise H3Error("Every shot is already drawn from its current prompt.")
    captured = dict(zip(controller.fields, values, strict=True))
    key = export_key(result_name, [saved["plan"], style, orientation], engine, use_references, captured) + ":" \
        + ",".join(map(str, picked))
    prompts = portrait_prompts(saved, style)
    current = portrait_keys(saved, style)
    finished = finished_images(key)
    # Drawing some shots matches their characters to the kept portraits (drawn from the same looks);
    # drawing them all starts afresh, portraits too.
    previous_portraits = kept_portraits(record, root, audio)
    previous_versions = {name: portrait_version(path) for name, (path, _) in previous_portraits.items()}
    kept = {} if only in (None, "all") else dict(previous_portraits)
    for name, (path, _fingerprint) in kept.items():
        if _fingerprint in current.get(name, ()):
            finished["portraits"].setdefault(name, str(path))
    folder = stills_dir(audio, root)

    def publish(index, image, details):
        # Read the latest checkpoint: earlier publishes can replace the images' record, and
        # untouched shots must retain both their original prompt and their immutable files.
        current = match_stills(shots, read_stills(audio, root))
        selected = picked[index]
        kept_shots, kept_images, extras = [], [], []
        for position, (shot, still) in enumerate(zip(shots, current)):
            if position == selected:
                path = image
                extra = _still_metadata(details)
            elif still is not None:
                path = folder / still["image"]
                extra = _still_metadata(still, previous_versions=previous_versions)
                shot = Shot(shot.first, shot.frames, shot.dissolve, still.get("prompt") or "", shot.beat, id=shot.id)
            else:
                continue
            kept_shots.append(shot)
            kept_images.append(path)
            extras.append(extra)
        for name, path in finished["portraits"].items():
            if name in prompts and Path(path).is_file():
                # A reused legacy portrait keeps its fingerprint and file name: later
                # shots in this run still reference that same path.
                if name not in kept or Path(kept[name][0]) != Path(path):
                    kept[name] = (path, prompt_key(prompts[name]))
        return keep_stills(audio, root, kept_shots, kept_images, {
            "model": engine, "art_direction": style, "orientation": orientation, "visual_plan": saved["plan"],
            "keep_characters_consistent": use_references,
            "portraits": {name: Path(path).name for name, path in finished["portraits"].items()}},
            extras=extras, total_frames=ceil(duration * FPS), portraits=kept, orientation=orientation)

    yield from illustrate_shots(controller, engine, [shots[i] for i in picked], captured, use_references,
                                finished, prompts, orientation, on_shot=publish)
    stills_record = folder / STILLS_RECORD
    _FINISHED.pop(key, None)  # A fresh drawing of the same settings draws new illustrations.
    drew = f"{len(picked)} image{'s' if len(picked) != 1 else ''}"
    yield str(stills_record), f"Drew {drew}. They play with the mix here; Save to Media renders the video."


VIDEO_INPUT_NAMES = ("result_name", "engine", "style", "max_shots", "use_visual_references", "orientation",
                     "only_shots")
# Previews keep the frame they were first drawn in, so changing a shot's frame line never redraws them.
PREVIEW_LINE = "Single widescreen 16:9 illustration of one moment. No text, captions, speech bubbles or lettering."
PREVIEW_INPUT_NAMES = ("engine", "presets")
PREVIEWS_RECORD = "previews.json"
_PREVIEWS_LOCK = Lock()


def preview_prompt(name):
    """What a style preview draws: the preset's own scene in the preset's rendering."""
    return "\n".join([PREVIEW_LINE, STYLE_PRESETS[name], STYLE_PREVIEWS[name]])


def previews_dir(root, engine):
    """Where an image model's style previews are kept, shared by every project."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(engine or "")):
        raise H3Error("Unknown image model.")
    return Path(root) / ".soundstage" / "style_previews" / engine


def _read_previews(folder):
    try:
        record = json.loads((folder / PREVIEWS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def preview_images(root, engine):
    """The image model's preview file for each preset, while drawn from the preset's current prompt."""
    try:
        folder = previews_dir(root, engine)
    except H3Error:
        return {}
    found = {}
    for name, item in _read_previews(folder).items():
        if (name in STYLE_PRESETS and isinstance(item, dict) and item.get("prompt") == prompt_key(preview_prompt(name))
                and (folder / str(item.get("image") or "")).is_file()):
            found[name] = item["image"]
    return found


def keep_preview(root, engine, name, image):
    """Keep a web copy of a drawn preview for the model, replacing the preset's earlier one."""
    from PIL import Image

    folder = previews_dir(root, engine)
    folder.mkdir(parents=True, exist_ok=True)
    file = f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}-{uuid4().hex[:8]}.jpg"
    with Image.open(image) as picture:
        picture.convert("RGB").save(folder / file, "JPEG", quality=STILLS_QUALITY, optimize=True)
    with _PREVIEWS_LOCK:  # Previews for one model can be drawn from several projects.
        record = _read_previews(folder)
        old = record.get(name)
        record[name] = {"image": file, "prompt": prompt_key(preview_prompt(name))}
        temporary = folder / f".{uuid4().hex[:8]}.tmp"
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        temporary.replace(folder / PREVIEWS_RECORD)
    if isinstance(old, dict) and old.get("image"):
        (folder / str(old["image"])).unlink(missing_ok=True)
    return file


def chosen_previews(root, engine, presets):
    """The presets a preview drawing draws: ``missing`` (no current preview for the model) or a list of names."""
    if presets == "missing":
        drawn = preview_images(root, engine)
        return [name for name in STYLE_PRESETS if name not in drawn]
    if not isinstance(presets, list) or not presets or not all(name in STYLE_PRESETS for name in presets):
        raise H3Error("Pick style presets to preview.")
    return list(dict.fromkeys(presets))


def draw_previews(controller, deps, engine, presets, *values):
    """Draw a preview of each chosen preset with the model and keep it for later; yields (file or skip, status).

    Each preview is kept as soon as it is drawn, so an interrupted run keeps the ones it finished.
    """
    root = deps.runtime_config().output_dir
    names = chosen_previews(root, engine, presets)
    if not names:
        raise H3Error("Every style preset already has a preview for this image model.")
    captured = dict(zip(controller.fields, values, strict=True))
    widescreen(engine, captured)
    file = None
    if engine == "gpt_image":
        completed = 0
        def task(name):
            def draw(options):
                return (yield from _stream(controller.generate(
                    engine, "Text to image", preview_prompt(name), [], options), f"Preview · {name}"))
            return draw

        for index, complete, value in _parallel_images([task(name) for name in names], captured):
            if complete:
                completed += 1
                file = keep_preview(root, engine, names[index], value)
                yield file, f"Previews {completed}/{len(names)} · {names[index]} · Kept"
            else:
                yield value[0], f"Previews {completed}/{len(names)} · {value[1]}"
        yield file, f"Drew {len(names)} style previews."
        return
    for number, name in enumerate(names, 1):
        label = f"Preview {number} of {len(names)} · {name}"
        yield gr.skip(), label
        path = yield from _stream(controller.generate(engine, "Text to image", preview_prompt(name), [], captured), label)
        file = keep_preview(root, engine, name, path)
        yield file, f"{label} · Kept"
    yield file, f"Drew {len(names)} style preview{'s' if len(names) != 1 else ''}."


@dataclass
class VideoExport:
    """What the Soundstage page needs to draw an illustrated video's images.

    ``settings`` holds each owner's Image tab values, captured when the studio loads or switches
    task; an owner the studio hasn't shown yet gets the Image tab's defaults.
    """

    controller: Any
    deps: Any
    generate: Callable
    engines: list
    settings: dict = field(default_factory=dict)
    preview: Callable | None = None  # Draws style previews: (engine, presets JSON, *Image tab values).

    def values(self, owner):
        captured = self.settings.get(owner) or {}
        return [captured.get(name, getattr(component, "value", None))
                for name, component in self.controller.fields.items()]


def video_export(controller, deps, triggers=()):
    """The drawing the Soundstage page runs for an illustrated video, with the Image tab's settings.

    ``triggers`` are the studio events (opening it, switching task) on which each owner's Image
    tab values are handed to the export, as Remake and Storyboard take the MiniMax H3 settings.
    """
    def generate(result_name, engine, style, limit, use_references, orientation, only, *values, mix=None):
        try:
            yield from draw_stills(controller, deps, result_name, engine, style, limit, use_references, orientation,
                                   *values, only=json.loads(only) if only else None, mix=mix)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    def preview(engine, presets, *values):
        try:
            yield from draw_previews(controller, deps, engine, json.loads(presets) if presets else "missing", *values)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    export = VideoExport(controller, deps, generate, [(item.label, item.id) for item in engines_for_task("Image")],
                         preview=preview)
    if triggers:
        names = list(controller.fields)

        def capture(*args):
            *values, request = args
            from h3_ui.job_admission import require_owner

            export.settings[require_owner(request)] = dict(zip(names, values, strict=True))

        capture.__signature__ = inspect.Signature([
            *(inspect.Parameter(f"value_{index}", inspect.Parameter.POSITIONAL_OR_KEYWORD) for index in range(len(names))),
            inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=gr.Request, default=None)])
        # Gradio injects the request from annotations; without this it passes None.
        capture.__annotations__ = {"request": gr.Request}
        gr.on(triggers=list(triggers), fn=capture, inputs=list(controller.fields.values()), outputs=None,
              queue=False, api_name=False, show_progress="hidden")
    return export
