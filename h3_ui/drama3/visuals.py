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
from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
from h3_app.burned_subtitles import Line, ass_subtitles, word_pieces
from h3_app.drama3.visual_plan import SHOT_LINE, compile_plan, fill_tags, sheet_prompt, shot_cap, subjects
from h3_app.drama3.visuals import FPS, Shot, render_video
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.drama3.result_view import timeline_data
from h3_ui.drama3.stage_video import ILLUSTRATED_FAMILY
from h3_ui.image_controller import reference_limit, reference_tag


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
# Each preset: (use-case group, name, what it's for, preview scene, rendering text). The text describes
# rendering, not story content: it applies to portraits and every scene. The preview scene shows the
# style on a subject typical of its use, with faces, hands and a finished setting to judge it by.
_PRESETS = (
    ("Drama & fiction", "Digital painting",
     "Any genre for teens and adults; the all-round default",
     "A medium-wide shot at dusk: a weathered sea captain in a wool coat grips the wheel of a fishing "
     "boat while a young deckhand points past him at a lighthouse beam cutting through the storm; rain "
     "streaks the wheelhouse windows and waves break over the bow.",
     "Full-color narrative digital painting with believable anatomy, distinct faces and natural "
     "expressions. Solid forms and materials, restrained visible brushwork, clear "
     "detail on faces and hands, and coherent local colors. Keep gestures readable."),
    ("Drama & fiction", "Realistic photo",
     "Grounded contemporary drama, crime and documentary reenactment",
     "A medium shot in a small-town diner at night: a tired detective in her forties sits in a red vinyl "
     "booth, coffee in both hands, studying a spread of photographs; neon from the window falls across "
     "the chrome counter where a waitress refills a coffee pot.",
     "Full-color live-action film still with natural human proportions, individual facial "
     "features, subtle skin texture and believable fabric and surfaces. Physically "
     "plausible light, restrained color grading and fine film grain. Avoid fashion-ad "
     "posing and beauty retouching."),
    ("Drama & fiction", "American comic",
     "Action, thriller and mystery with bold, punchy staging",
     "A dynamic low-angle shot on a rain-soaked rooftop: a courier in a torn jacket leaps the gap between "
     "two brick buildings, a satchel clutched to her chest, water towers and fire escapes behind her and "
     "a searchlight sweeping the alley below.",
     "American comic-book illustration with bold brush-ink contours, varied line "
     "weight, sculpted spot blacks and selective cross-hatching. Clearly modeled "
     "anatomy, distinct faces and expressive gestures; rich flat colors with crisp "
     "shadow shapes and restrained halftone texture. Keep the story's clothing and "
     "physique without adding superhero costumes or exaggerated muscles. One continuous "
     "image, no panel borders, speech balloons, captions or lettering."),
    ("Drama & fiction", "Oil painting",
     "Historical, period and literary stories",
     "A candlelit study: an elderly scholar reads a letter at a heavy oak desk crowded with books, a quill "
     "and an hourglass, while his granddaughter waits in the doorway holding a lantern; tall windows show "
     "a snowy courtyard.",
     "Full-color oil painting with visible bristle strokes, layered opaque pigments "
     "and subtle canvas texture. Model faces, hands and objects with believable "
     "proportions and clear light and shadow, and natural material colors. Show the "
     "painting itself, without a picture frame, gallery wall or added antique costumes."),
    ("Drama & fiction", "Chinese ink painting",
     "Wuxia, xianxia, Chinese myth and history",
     "A wide shot of a mountain pass at dawn: a lone swordswoman in layered robes stands on a stone bridge "
     "above a waterfall, her sleeves lifted by the wind, pines clinging to the cliffs and a temple roof "
     "rising from the valley below.",
     "Full-color traditional Chinese ink-and-color painting on fine rice paper. Confident brush contours "
     "with varied pressure, finely drawn faces and hands, layered mineral colors in transparent washes and "
     "ink gradations that model rock, water, foliage and fabric. Distinct figures with believable "
     "proportions and readable expressions. Keep the story's own era and clothing; no seals, calligraphy, "
     "scroll borders or blank paper standing in for the setting."),
    ("Drama & fiction", "Watercolor",
     "Gentle slice of life, memoir and poetry",
     "A quiet morning kitchen: a grandmother and a boy share breakfast at a table by an open window, steam "
     "rising from their bowls, potted herbs on the sill and a bicycle leaning against the garden fence "
     "outside.",
     "Full-color watercolor illustration with transparent layered washes, soft pigment "
     "edges and selective fine pencil contours. Preserve facial features, hands and "
     "important objects with controlled edges. Subtle paper grain, believable proportions "
     "and consistent pigment density; avoid splatter or unfinished white gaps across "
     "focal subjects."),
    ("Anime, manga & games", "Anime",
     "Youth, romance, fantasy and light-novel stories",
     "A medium shot on a train platform at sunset: one classmate holds out a folded letter while the "
     "other hesitates, school bags over their shoulders, a commuter train pulling in and power lines "
     "against an orange sky.",
     "Full-color 2D dramatic animation still with controlled fine outlines, consistent character proportions "
     "and clean two- or three-tone cel shading. Individually designed faces with restrained, "
     "readable expressions. Avoid chibi proportions, exaggerated reaction symbols and decorative bloom."),
    ("Anime, manga & games", "Japanese comic",
     "Dramatic manga-style action and character stories",
     "A festival night: a young swordsman kneels to mend a girl's broken sandal strap on a crowded shrine "
     "path, paper lanterns glowing overhead, food stalls on both sides and fireworks opening above the "
     "torii gate.",
     "Full-color Japanese manga illustration with precise pen lines, delicate "
     "facial detail and controlled variation in line weight. Rich local colors, "
     "clean cel-shaded shadows and selective color gradients. Consistent stylized "
     "proportions, distinct character designs and expressive but believable acting, "
     "with legible eyes and gestures. One continuous image, "
     "no panels, speech balloons, lettering or decorative reaction symbols."),
    ("Anime, manga & games", "Pixel art",
     "Game lore, retro sci-fi and gaming podcasts",
     "A side view of a torchlit dungeon: an adventurer with a backpack and lantern faces a small dragon "
     "curled on a pile of gold, stone pillars, a locked iron gate and glowing mushrooms around them.",
     "Full-color narrative pixel art with a consistent coarse pixel grid, deliberate pixel clusters "
     "and a limited shared palette. Crisp stepped edges without smoothing or mixed pixel "
     "sizes; selective dithering for shading. Simplify forms into readable silhouettes "
     "while preserving each character's hair, outfit and key facial cues; no game "
     "interface or decorative border."),
    ("Kids & family", "Children's book",
     "Bedtime and picture-book stories for young children",
     "A cozy bedtime scene: a little girl in pajamas reads a picture book to her stuffed rabbit and a "
     "sleepy cat inside a blanket fort, fairy lights strung across the bedroom wall and a full moon in "
     "the window.",
     "Full-color hand-painted gouache and colored-pencil storybook illustration. Matte opaque "
     "color shapes, subtle pencil edges and gentle paper texture; simplified but "
     "consistent anatomy, distinct character silhouettes and legible expressions. "
     "Keep tension or sadness when the story has it instead of making everything cheerful."),
    ("Kids & family", "2D cartoon",
     "Comedy and lively stories for kids",
     "A backyard mishap: a boy and his overexcited dog are tangled in a garden hose spraying water "
     "everywhere while his older sister holds a soaked birthday cake with a deadpan look; a picket fence "
     "and a treehouse behind them.",
     "Full-color 2D cartoon illustration with clean outlines, simple rounded shapes, "
     "bold flat colors and minimal cel shading. Consistent character proportions, distinct "
     "silhouettes and expressive faces, serious when the story is. One continuous image "
     "without speech bubbles, lettering or comic panels."),
    ("Kids & family", "3D animation",
     "Family adventure with feature-animation appeal",
     "A wide adventure shot: a young inventor and her small round robot ride a homemade glider over a "
     "seaside village of colorful rooftops, laundry lines and a windmill, the inventor laughing while "
     "the robot holds on tight.",
     "Full-color stylized 3D animated feature still with appealing, consistent character proportions "
     "and expressive but restrained faces. Rounded modeled forms, tactile cloth, "
     "convincing material differences and clear silhouettes. Keep eyes and skin natural "
     "within the stylization, without a glossy plastic finish."),
    ("Explainers & slides", "Flat illustration",
     "Training, onboarding and how-to explainers",
     "A team onboarding moment: a mentor points at simple icons on a laptop screen while three new "
     "colleagues lean in around a desk with a plant, coffee mugs and a blank whiteboard.",
     "Full-color flat business illustration with clean geometric shapes, solid "
     "color fills and minimal shading. Use simplified people and recognizable "
     "objects to explain one idea at a time, with consistent proportions and a "
     "small coordinated color palette. Clear visual hierarchy, generous spacing "
     "and a plain light background; leave an uncluttered area for slide text "
     "to be added later. Avoid decorative clutter, lettering, logos and invented data."),
    ("Explainers & slides", "Isometric illustration",
     "Systems, processes and product walkthroughs",
     "How an online order travels: a stocked warehouse shelf, a packing table with boxes, a delivery van "
     "and a doorstep with a parcel, linked by a clear path on one compact platform.",
     "Full-color isometric business illustration with parallel projection, "
     "consistent object scale and clean geometric volumes. Use restrained "
     "shading and coordinated colors to distinguish components. Show the "
     "requested system, process or workspace through clearly arranged objects "
     "and meaningful spatial relationships. Keep one coherent scene on a "
     "simple light background, with generous margins for slide text added "
     "later. No labels, logos, invented metrics or decorative dashboards."),
    ("Explainers & slides", "Minimal vector",
     "Keynotes and one-idea concept slides",
     "A product launch as one idea: a single rocket lifting off from an open laptop, a few small stars "
     "around it.",
     "Full-color minimal vector-style illustration with crisp edges, simple shapes "
     "and consistent stroke widths. Reduce the requested idea to a few recognizable "
     "elements with strong visual hierarchy, using a small coordinated palette "
     "on a plain light background. Keep ample negative space for slide text added "
     "later and maintain clarity at presentation size. No texture, ornamental "
     "detail, lettering, logos or invented charts and numbers."),
    ("Explainers & slides", "Editorial illustration",
     "Commentary, news analysis and idea podcasts",
     "Information overload as a metaphor: a person reads calmly at a small desk while a towering wave of "
     "envelopes and notification bubbles curls above them.",
     "Full-color editorial business illustration built around one clear "
     "visual metaphor for the requested idea. Use simplified recognizable "
     "forms, confident silhouettes, restrained texture and a coordinated "
     "palette. Make the relationship between subjects easy to understand "
     "without labels; keep the focal concept distinct from a simple "
     "background and leave breathing room for slide text added later. "
     "Avoid unrelated symbols, stock handshake imagery, lettering, logos "
     "and invented data."),
)
_BARE_PRESETS = {name: text for _group, name, _use, _preview, text in _PRESETS}
# Detail asked only of faces and hands left settings rough and blurry, so each story preset also asks
# for a finished background. The slide presets keep their deliberately plain backgrounds.
SETTING_DETAIL = (" Render the setting with the same finish and focus as the figures: legible architecture, "
                  "objects and landscape, never a blurred, smeared or sketchy backdrop.")
SLIDE_STYLES = tuple(name for group, name, *_ in _PRESETS if group == "Explainers & slides")
# Projects saved with earlier preset text keep it as their Art direction (and their shot plan is keyed
# by it), so those texts still count as the preset rather than as Custom.
_EARLIER_DETAIL = (" Paint the setting with the same finish and focus as the figures: legible architecture, "
                   "objects and landscape, never a blurred, smeared or sketchy backdrop.")
_LEGACY_PRESETS = {text: name for name, text in _BARE_PRESETS.items()}
_LEGACY_PRESETS[_BARE_PRESETS["Digital painting"].replace(
    "clear detail on faces and hands", "sharper detail around faces and hands")] = "Digital painting"
_LEGACY_PRESETS[_BARE_PRESETS["Japanese comic"].replace(
    "with legible eyes and gestures. One", "with legible eyes and gestures. Full color throughout. One")] = \
    "Japanese comic"
_LEGACY_PRESETS.update({text + _EARLIER_DETAIL: name for text, name in list(_LEGACY_PRESETS.items())
                        if name not in SLIDE_STYLES})
STYLE_PRESETS = {name: text if name in SLIDE_STYLES else text + SETTING_DETAIL
                 for name, text in _BARE_PRESETS.items()}
STYLE_GROUPS = [{"name": group, "presets": [name for g, name, *_ in _PRESETS if g == group]}
                for group in dict.fromkeys(group for group, *_ in _PRESETS)]
STYLE_USES = {name: use for _group, name, use, _preview, _text in _PRESETS}
STYLE_PREVIEWS = {name: preview for _group, name, _use, preview, _text in _PRESETS}
DEFAULT_STYLE = "Digital painting"
CUSTOM_STYLE = "Custom"


def preset_for(text):
    """The preset an Art direction text matches, or Custom once it's been edited."""
    text = (text or "").strip()
    return next((name for name, value in STYLE_PRESETS.items() if value == text), None) \
        or _LEGACY_PRESETS.get(text, CUSTOM_STYLE)


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
                yield gr.skip(), f"{label} · Reused"
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


def prompt_key(prompt):
    """A short fingerprint of a portrait prompt: a kept portrait is reused while its prompt is unchanged."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _write_stills_record(folder, record):
    """Replace the stills record in one step, so a reader never sees half of it."""
    temporary = folder / f".{record['revision']}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(folder / STILLS_RECORD)
    return folder / STILLS_RECORD


def keep_stills(audio, root, shots, images, details, *, extras=None, total_frames=None, portraits=None):
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
            copyfile(image, folder / name)  # already a web copy: no second JPEG pass
        else:
            with Image.open(image) as picture:
                picture.convert("RGB").save(folder / name, "JPEG", quality=STILLS_QUALITY, optimize=True)
        records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                        "beat": shot.beat, "prompt": shot.prompt, "image": name, **extra})
    kept = {}
    for key, (image, fingerprint) in (portraits or {}).items():
        name = f"portrait-{fingerprint}.png"
        if Path(image).resolve() != (folder / name).resolve():
            with Image.open(image) as picture:
                picture.save(folder / name, "PNG")
        kept[key] = {"image": name, "prompt": fingerprint}
    record = _write_stills_record(folder, {"revision": revision, "source": Path(audio).name, "fps": FPS,
                                           "total_frames": total_frames, "shots": records, "portraits": kept,
                                           "details": details})
    keep = {item["image"] for item in kept.values()}
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
    """Each plan shot's still, drawn for the same time on the mix, or None."""
    by_time = {(int(item["first_frame"]), int(item["frames"])): item for item in (record or {}).get("shots") or []}
    return [by_time.get((shot.first, shot.frames)) for shot in shots]


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
    return "drawn" if still.get("prompt") == shot.prompt else "changed"


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


def plan_shots_for(saved, duration):
    """The saved plan's shots, compiled as drawing uses them."""
    return compile_plan(saved["plan"], duration, saved["style"], shot_cap(duration, saved["limit"]))


def reference_direction(style):
    """The Art direction for portraits: without SETTING_DETAIL, which asks for scenery behind a
    portrait meant to stand on a plain background."""
    style = style or ""
    for detail in (SETTING_DETAIL, _EARLIER_DETAIL):
        style = style.replace(detail.strip(), "")
    return re.sub(r"[ \t]{2,}", " ", style).strip()


def portrait_prompts(saved, *, with_setting=False):
    """Each character's portrait prompt, from the saved plan. ``with_setting`` gives the prompt
    portraits were drawn from before SETTING_DETAIL was left out of them."""
    art_style = saved["plan"].get("art_style") or saved["plan"].get("style")
    direction = saved["style"] if with_setting else reference_direction(saved["style"])
    return {name: sheet_prompt(kind, name, anchor, art_style, direction)
            for name, (kind, anchor) in subjects(saved["plan"]).items() if kind == "characters"}


def portrait_keys(saved):
    """The prompt fingerprints a kept portrait may carry and still be current, by character.
    Portraits drawn while the prompt still had SETTING_DETAIL stay current, so redrawing one shot
    of an older project keeps matching the faces already drawn."""
    old = portrait_prompts(saved, with_setting=True)
    return {name: {prompt_key(prompt), prompt_key(old[name])} for name, prompt in portrait_prompts(saved).items()}


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


def draw_stills(controller, deps, result_name, engine, style, limit, use_references, *values, only=None):
    """Draw a mix's saved shot plan and keep the images as its stills; yields (record or skip, status).

    ``only`` limits the drawing to some shots (see ``chosen_shots``). The others keep the image
    already drawn for their time on the mix, and kept portraits drawn from the same prompt are
    reused, so a redrawn shot's characters match the rest.
    """
    audio, settings = load_mix(deps, result_name)
    saved = read_plan(audio, style, limit)
    if saved is None:
        raise H3Error("This mix has no shot plan for the current Art direction and Minimum shots yet. "
                      "Click Plan shots, or paste a plan from another chat model, then draw.")
    duration = settings["duration"]
    root = deps.runtime_config().output_dir
    shots = plan_shots_for(saved, duration)
    record = read_stills(audio, root)
    stills = match_stills(shots, record)
    picked = chosen_shots(shots, stills, only)
    if not picked:
        raise H3Error("Every shot is already drawn from its current prompt.")
    captured = dict(zip(controller.fields, values, strict=True))
    key = export_key(result_name, saved["plan"], engine, use_references, captured) + ":" + ",".join(map(str, picked))
    prompts = portrait_prompts(saved)
    current = portrait_keys(saved)
    finished = finished_images(key)
    # Drawing some shots matches their characters to the kept portraits (drawn from the same looks);
    # drawing them all starts afresh, portraits too.
    kept = {} if only in (None, "all") else {name: item for name, item in kept_portraits(record, root, audio).items()
                                  if item[1] in current.get(name, ())}
    for name, (path, _fingerprint) in kept.items():
        finished["portraits"].setdefault(name, str(path))
    images, shot_records, portrait_records = yield from illustrate_shots(
        controller, engine, [shots[i] for i in picked], captured, use_references, finished, prompts)
    yield gr.skip(), "Keeping the images with the mix…"
    folder = stills_dir(audio, root)
    drawn = dict(zip(picked, zip(images, shot_records)))
    kept_shots, kept_images, extras = [], [], []
    for index, (shot, still) in enumerate(zip(shots, stills)):
        if index in drawn:
            image, details = drawn[index]
            extra = {"draft": details["draft"], "characters": details["characters"]}
        elif still is not None:
            # Not redrawn: the image already drawn for this time stays, with the prompt it was drawn from.
            image = folder / still["image"]
            extra = {name: still[name] for name in ("draft", "characters") if name in still}
            shot = Shot(shot.first, shot.frames, shot.dissolve, still.get("prompt") or "", shot.beat)
        else:
            continue
        kept_shots.append(shot)
        kept_images.append(image)
        extras.append(extra)
    for name, path in finished["portraits"].items():
        if name in prompts and Path(path).is_file():
            kept[name] = (path, prompt_key(prompts[name]))
    stills_record = keep_stills(audio, root, kept_shots, kept_images, {
        "model": engine, "art_direction": style, "visual_plan": saved["plan"],
        "keep_characters_consistent": use_references, "portraits": portrait_records},
        extras=extras, total_frames=ceil(duration * FPS), portraits=kept)
    _FINISHED.pop(key, None)  # A fresh drawing of the same settings draws new illustrations.
    drew = f"{len(picked)} image{'s' if len(picked) != 1 else ''}"
    yield str(stills_record), f"Drew {drew}. They play with the mix here; Save to Media renders the video."


VIDEO_INPUT_NAMES = ("result_name", "engine", "style", "max_shots", "use_visual_references", "only_shots")
PREVIEW_INPUT_NAMES = ("engine", "presets")
PREVIEWS_RECORD = "previews.json"
_PREVIEWS_LOCK = Lock()


def preview_prompt(name):
    """What a style preview draws: the preset's own scene in the preset's rendering."""
    return "\n".join([SHOT_LINE, STYLE_PREVIEWS[name], STYLE_PRESETS[name]])


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
    def generate(result_name, engine, style, limit, use_references, only, *values):
        try:
            yield from draw_stills(controller, deps, result_name, engine, style, limit, use_references, *values,
                                   only=json.loads(only) if only else None)
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
