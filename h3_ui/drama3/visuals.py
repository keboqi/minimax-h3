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
from shutil import which
from threading import Lock
from uuid import uuid4

import gradio as gr

from h3_app.capabilities import engines_for_task
from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
from h3_app.drama3.subtitles import Line, ass_subtitles
from h3_app.drama3.visual_plan import compile_plan, fill_tags, sheet_prompt, shot_cap, subjects
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.drama3.result_view import timeline_data
from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY
from h3_ui.image_controller import reference_limit, reference_tag
from h3_ui.subtitle_words import word_pieces


# Images finished by an export that failed or was stopped, by export fingerprint:
# {"portraits": {key: path}, "shots": [path, ...]}. A retry with the same mix, plan, model
# and settings resumes after them.
_FINISHED: OrderedDict[str, dict] = OrderedDict()
_FINISHED_LIMIT = 8
PLAN_TICK = 3  # Seconds between planning status updates.
PLAN_TIMEOUT = 300  # Seconds before planning gives up on the writer.
# Widescreen sizes on each model's grid, so stills fill the 1280x720 video without bars.
_WIDESCREEN = {"qwen": (1664, 928), "h3": (1664, 928), "flux3": (1344, 768),
               "moss_image": (1344, 768), "fibo2": (1344, 768)}
# Describe rendering, not story content: these apply to portraits and every scene.
STYLE_PRESETS = {
    "Digital painting": "Full-color narrative digital painting with believable anatomy, distinct faces and natural "
                        "expressions. Solid forms and materials, restrained visible brushwork, sharper "
                        "detail around faces and hands, and coherent local colors. Keep gestures readable.",
    "Realistic photo": "Full-color live-action film still with natural human proportions, individual facial "
                       "features, subtle skin texture and believable fabric and surfaces. Physically "
                       "plausible light, restrained color grading and fine film grain. Avoid fashion-ad "
                       "posing and beauty retouching.",
    "Anime": "Full-color 2D dramatic animation still with controlled fine outlines, consistent character proportions "
             "and clean two- or three-tone cel shading. Individually designed faces with restrained, "
             "readable expressions. Avoid chibi proportions, exaggerated reaction symbols and decorative bloom.",
    "3D animation": "Full-color stylized 3D animated feature still with appealing, consistent character proportions "
                    "and expressive but restrained faces. Rounded modeled forms, tactile cloth, "
                    "convincing material differences and clear silhouettes. Keep eyes and skin natural "
                    "within the stylization, without a glossy plastic finish.",
    "2D cartoon": "Full-color 2D cartoon illustration with clean outlines, simple rounded shapes, "
                  "bold flat colors and minimal cel shading. Consistent character proportions, distinct "
                  "silhouettes and expressive faces, serious when the story is. One continuous image "
                  "without speech bubbles, lettering or comic panels.",
    "American comic": "American comic-book illustration with bold brush-ink contours, varied line "
                      "weight, sculpted spot blacks and selective cross-hatching. Clearly modeled "
                      "anatomy, distinct faces and expressive gestures; rich flat colors with crisp "
                      "shadow shapes and restrained halftone texture. Keep the story's clothing and "
                      "physique without adding superhero costumes or exaggerated muscles. One continuous "
                      "image, no panel borders, speech balloons, captions or lettering.",
    "Japanese comic": "Full-color Japanese manga illustration with precise pen lines, delicate "
                      "facial detail and controlled variation in line weight. Rich local colors, "
                      "clean cel-shaded shadows and selective color gradients. Consistent stylized "
                      "proportions, distinct character designs and expressive but believable acting, "
                      "with legible eyes and gestures. Full color throughout. One continuous image, "
                      "no panels, speech balloons, lettering or decorative reaction symbols.",
    "Watercolor": "Full-color watercolor illustration with transparent layered washes, soft pigment "
                  "edges and selective fine pencil contours. Preserve facial features, hands and "
                  "important objects with controlled edges. Subtle paper grain, believable proportions "
                  "and consistent pigment density; avoid splatter or unfinished white gaps across "
                  "focal subjects.",
    "Oil painting": "Full-color oil painting with visible bristle strokes, layered opaque pigments "
                    "and subtle canvas texture. Model faces, hands and objects with believable "
                    "proportions and clear light and shadow, and natural material colors. Show the "
                    "painting itself, without a picture frame, gallery wall or added antique costumes.",
    "Children's book": "Full-color hand-painted gouache and colored-pencil storybook illustration. Matte opaque "
                       "color shapes, subtle pencil edges and gentle paper texture; simplified but "
                       "consistent anatomy, distinct character silhouettes and legible expressions. "
                       "Keep tension or sadness when the story has it instead of making everything cheerful.",
    "Pixel art": "Full-color narrative pixel art with a consistent coarse pixel grid, deliberate pixel clusters "
                 "and a limited shared palette. Crisp stepped edges without smoothing or mixed pixel "
                 "sizes; selective dithering for shading. Simplify forms into readable silhouettes "
                 "while preserving each character's hair, outfit and key facial cues; no game "
                 "interface or decorative border.",
    "Flat illustration": "Full-color flat business illustration with clean geometric shapes, solid "
                         "color fills and minimal shading. Use simplified people and recognizable "
                         "objects to explain one idea at a time, with consistent proportions and a "
                         "small coordinated color palette. Clear visual hierarchy, generous spacing "
                         "and a plain light background; leave an uncluttered area for slide text "
                         "to be added later. Avoid decorative clutter, lettering, logos and invented data.",
    "Isometric illustration": "Full-color isometric business illustration with parallel projection, "
                              "consistent object scale and clean geometric volumes. Use restrained "
                              "shading and coordinated colors to distinguish components. Show the "
                              "requested system, process or workspace through clearly arranged objects "
                              "and meaningful spatial relationships. Keep one coherent scene on a "
                              "simple light background, with generous margins for slide text added "
                              "later. No labels, logos, invented metrics or decorative dashboards.",
    "Minimal vector": "Full-color minimal vector-style illustration with crisp edges, simple shapes "
                      "and consistent stroke widths. Reduce the requested idea to a few recognizable "
                      "elements with strong visual hierarchy, using a small coordinated palette "
                      "on a plain light background. Keep ample negative space for slide text added "
                      "later and maintain clarity at presentation size. No texture, ornamental "
                      "detail, lettering, logos or invented charts and numbers.",
    "Editorial illustration": "Full-color editorial business illustration built around one clear "
                               "visual metaphor for the requested idea. Use simplified recognizable "
                               "forms, confident silhouettes, restrained texture and a coordinated "
                               "palette. Make the relationship between subjects easy to understand "
                               "without labels; keep the focal concept distinct from a simple "
                               "background and leave breathing room for slide text added later. "
                               "Avoid unrelated symbols, stock handshake imagery, lettering, logos "
                               "and invented data.",
}
DEFAULT_STYLE = "Digital painting"
CUSTOM_STYLE = "Custom"


def preset_for(text):
    """The preset an Art direction text matches, or Custom once it's been edited."""
    text = (text or "").strip()
    return next((name for name, value in STYLE_PRESETS.items() if value == text), CUSTOM_STYLE)


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
    else:
        captured.update({f"{engine}__width": width, f"{engine}__height": height})
        if engine in ("qwen", "flux3"):
            captured[f"{engine}__edit_size"] = QWEN_EDIT_SIZE_MANUAL


def widescreen(engine, captured):
    """One image per request at a 16:9 size, whatever the Image tab is set to."""
    captured.update(qwen__batch_count=1, qwen__batch_edit_inputs=False, moss_image__layout="", flux3__layout="",
                    qwen__output_resolution_1k=None, qwen__output_resolution_2k=None)
    _size(engine, captured, *_WIDESCREEN.get(engine, (1344, 768)))


def match_characters(controller, engine, scene, portraits, shot, captured):
    """Repaint a finished scene's characters from their portraits, keeping everything else.

    The scene is the first input, so an edit model that leans toward its input keeps the scene's
    composition; the portraits only steer faces, hair and outfits. Each portrait's tag (<image2>,
    <Picture 2>...) also sits right after that character's name in the scene text, the way the
    Image tab's clickable reference names do, so the model knows which input is which person.
    """
    count = len(portraits) + 1
    scene_tag = reference_tag(engine, 1, count) or "the first image"
    tags = {key: reference_tag(engine, position, count) or f"(reference image {position})"
            for position, key in enumerate(portraits, 2)}
    lines = [f"{scene_tag} is the finished scene. Keep its composition, camera, poses, setting, lighting and "
             "art style.", "Repaint only these people so each matches their reference:"]
    lines += [f"{key} {tag}: the same face, hairstyle and outfit." for key, tag in tags.items()]
    lines.append("Change nothing else. Never add the references as extra figures, panels or insets.")
    scene_text = re.sub(r"[ \t]+", " ", fill_tags(shot.tagged, tags)).strip() if shot.tagged else shot.beat
    if scene_text:
        lines.append(f"Scene: {scene_text}")
    return controller.generate(engine, "Image edit", "\n".join(lines), [scene, *portraits.values()], captured)


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


def illustrate_shots(controller, engine, shots, captured, use_references, finished=None, portraits=None):
    """Draw every scene from its text, then match its characters to their portraits.

    ``portraits`` maps character keys to portrait prompts; only characters some shot shows are drawn,
    once, before any scene. Locations and props stay in the scene text: an edit model handed a location
    or a multi-view sheet as a starting point copies it instead of composing a new scene.
    ``finished`` holds images from an earlier attempt at the same export; existing files are reused,
    and each new image is recorded so a later retry can resume after it.
    """
    finished = {"portraits": {}, "shots": []} if finished is None else finished
    finished.setdefault("portraits", {})
    widescreen(engine, captured)
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
                yield gr.skip(), f"{label} · Reused from the interrupted export"
                continue
            yield gr.skip(), label
            drawn[key] = yield from _stream(
                controller.generate(engine, "Text to image", portraits[key], [], square), label)
            finished["portraits"][key] = drawn[key]
    shot_records = []
    images = []
    done = finished["shots"]
    for index, shot in enumerate(shots):
        who = {key: drawn[key] for key in shot.references if key in drawn}
        who = dict(list(who.items())[:slots])
        label = f"Shot {index + 1} of {len(shots)}"
        draft = None
        if index < len(done) and Path(done[index]).is_file():
            path = done[index]
            yield gr.skip(), f"{label} · Reused from the interrupted export"
        else:
            del done[index:]
            yield gr.skip(), label
            path = yield from _stream(controller.generate(engine, "Text to image", shot.prompt, [], captured), label)
            if who:
                draft, label = path, f"{label} · Matching " + ", ".join(who)
                path = yield from _stream(
                    match_characters(controller, engine, draft, who, shot, captured), label)
            done.append(path)
        images.append(path)
        shot_records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                             "image": Path(path).name, "draft": Path(draft).name if draft else None,
                             "characters": list(who)})
    portrait_records = {key: Path(path).name for key, path in drawn.items()}
    return images, shot_records, portrait_records


def plan_path(audio):
    """The shot plan saved beside a Sound Stage mix."""
    audio = Path(audio)
    return audio.with_name(f".{audio.stem}.visual_plan.json")


def write_plan(audio, style, limit, plan):
    path = plan_path(audio)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"style": style, "limit": int(limit), "plan": plan}, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(path)


def read_plan(audio, style=None, limit=None):
    """The plan saved with this mix, or None; with ``style`` and ``limit``, only a plan made for them."""
    try:
        saved = json.loads(plan_path(audio).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or not isinstance(saved.get("plan"), dict):
        return None
    if style is not None and (saved.get("style") or "").strip() != (style or "").strip():
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


def plan_rows(saved, duration, shots=None):
    """Shot plan table rows from compiled shots, or from a saved plan."""
    if shots is None:
        try:
            shots = compile_plan(saved["plan"], duration, saved["style"], shot_cap(duration, saved["limit"]))
        except (H3Error, KeyError, TypeError):
            return []
    return [[round(s.first / FPS, 3), round((s.first + s.frames) / FPS, 3),
             "Dissolve" if s.dissolve else "Cut", s.beat, s.prompt] for s in shots]


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


def stills_dir(audio, root):
    """Where a mix's stills are kept: under ``root``'s ``.soundstage`` folder, which Media leaves alone."""
    return Path(root) / ".soundstage" / "stills" / Path(audio).stem


def keep_stills(audio, root, shots, images, details):
    """Keep a web copy of each drawn image with its shot's time on the mix; returns the stills record.

    A new drawing replaces the mix's earlier stills. ``details`` is what a video rendered from
    them records in its settings snapshot.
    """
    from PIL import Image

    folder = stills_dir(audio, root)
    folder.mkdir(parents=True, exist_ok=True)
    revision = uuid4().hex[:12]
    records = []
    for index, (shot, image) in enumerate(zip(shots, images, strict=True)):
        name = f"{revision}-{index + 1:03d}.jpg"
        with Image.open(image) as picture:
            picture.convert("RGB").save(folder / name, "JPEG", quality=STILLS_QUALITY, optimize=True)
        records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                        "beat": shot.beat, "prompt": shot.prompt, "image": name})
    record = folder / STILLS_RECORD
    temporary = folder / f".{revision}.tmp"
    temporary.write_text(json.dumps({"revision": revision, "source": Path(audio).name, "fps": FPS,
                                     "shots": records, "details": details}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(record)
    for old in folder.glob("*.jpg"):
        if not old.name.startswith(f"{revision}-"):
            old.unlink(missing_ok=True)
    return record


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
    """A stills record as the shots and image file names the renderer takes."""
    return ([Shot(int(item["first_frame"]), int(item["frames"]), bool(item["dissolve"]), item.get("prompt") or "",
                  item.get("beat") or "") for item in record["shots"]], [item["image"] for item in record["shots"]])


def burned_subtitles(audio, size) -> str | None:
    """The mix's lines as an ASS script at ``size`` percent, as the page's video preview shows them,
    each aligned word lighting up as it is spoken; None when the mix has no timed lines."""
    entries, words, _tones, _sounds = timeline_data((read_snapshot(Path(audio)) or {}).get("settings") or {})
    lines = [Line(speaker, text, start, end, word_pieces(text, line_words))
             for (speaker, text, start, end), line_words in zip(entries, words)]
    return ass_subtitles(lines, size=size) if lines else None


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
    script = burned_subtitles(audio, subtitles) if subtitles else None
    video = render_video(audio, shots, [folder / name for name in names], audio.parent, subtitles=script)
    write_snapshot(video, {"family": ILLUSTRATED_FAMILY, "settings": {
        **(record.get("details") or {}), "source": audio.name, "stills": record["revision"],
        "subtitles": subtitles}})
    return video


def draw_stills(controller, deps, result_name, engine, style, limit, use_references, *values):
    """Draw a mix's saved shot plan and keep the images as its stills; yields (record or skip, status)."""
    audio, settings = load_mix(deps, result_name)
    saved = read_plan(audio, style, limit)
    if saved is None:
        raise H3Error("This mix has no shot plan for the current Art direction and Minimum shots yet. "
                      "Click Plan shots, or paste a plan from another chat model, then draw.")
    duration = settings["duration"]
    shots = compile_plan(saved["plan"], duration, saved["style"], shot_cap(duration, saved["limit"]))
    captured = dict(zip(controller.fields, values, strict=True))
    key = export_key(result_name, saved["plan"], engine, use_references, captured)
    art_style = saved["plan"].get("art_style") or saved["plan"].get("style")
    portraits = {name: sheet_prompt(kind, name, anchor, art_style, saved["style"])
                 for name, (kind, anchor) in subjects(saved["plan"]).items() if kind == "characters"}
    images, shot_records, portrait_records = yield from illustrate_shots(
        controller, engine, shots, captured, use_references, finished_images(key), portraits)
    yield gr.skip(), "Keeping the images with the mix…"
    record = keep_stills(audio, deps.runtime_config().output_dir, shots, images, {
        "model": engine, "art_direction": style, "visual_plan": saved["plan"],
        "keep_characters_consistent": use_references, "portraits": portrait_records, "shots": shot_records})
    _FINISHED.pop(key, None)  # A fresh drawing of the same settings draws new illustrations.
    yield str(record), f"Drew {len(shots)} images. They play with the mix here; Save to Media renders the video."


VIDEO_INPUT_NAMES = ("result_name", "engine", "style", "max_shots", "use_visual_references")


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

    def values(self, owner):
        captured = self.settings.get(owner) or {}
        return [captured.get(name, getattr(component, "value", None))
                for name, component in self.controller.fields.items()]


def video_export(controller, deps, triggers=()):
    """The drawing the Soundstage page runs for an illustrated video, with the Image tab's settings.

    ``triggers`` are the studio events (opening it, switching task) on which each owner's Image
    tab values are handed to the export, as Remake and Storyboard take the MiniMax H3 settings.
    """
    def generate(result_name, engine, style, limit, use_references, *values):
        try:
            yield from draw_stills(controller, deps, result_name, engine, style, limit, use_references, *values)
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc

    export = VideoExport(controller, deps, generate, [(item.label, item.id) for item in engines_for_task("Image")])
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
