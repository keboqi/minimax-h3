"""Optional illustrated video export from the Speak result."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from shutil import which

import gradio as gr

from h3_app.capabilities import engines_for_task
from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.catalog import QWEN_EDIT_SIZE_MANUAL
from h3_app.drama3.visual_plan import compile_plan, plan_shots, sheet_prompt, shot_cap, subjects
from h3_app.drama3.visuals import FPS, render_video
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.engines.controls import QUIET, generation_actions, writer_inputs
from h3_ui.image_controller import reference_limit, reference_tag
from h3_ui.job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt


# Images finished by an export that failed or was stopped, by export fingerprint:
# {"sheets": {key: path}, "shots": [path, ...]}. A retry with the same mix, plan, model
# and settings resumes after them.
_FINISHED: OrderedDict[str, dict] = OrderedDict()
_FINISHED_LIMIT = 8
# Widescreen sizes on each model's grid, so stills fill the 1280x720 video without bars.
_WIDESCREEN = {"qwen": (1664, 928), "h3": (1664, 928), "flux3": (1344, 768),
               "moss_image": (1344, 768), "fibo2": (1344, 768)}
# Starting points for Art direction; the planner builds its full production style on top.
STYLE_PRESETS = {
    "Ink and watercolor": "Cinematic ink and watercolor illustration: confident brush-pen linework, loose "
                          "transparent washes, paper texture, muted palette with one warm accent per scene.",
    "Graphic novel": "Modern graphic novel art: precise inked contours, heavy spot blacks and cross-hatching, "
                     "flat digital color with dramatic rim light, grounded realistic anatomy.",
    "Film noir": "Black-and-white film noir: hard single-source light, deep shadows, venetian-blind patterns, "
                 "rain-slick streets, high contrast with fine film grain.",
    "Anime": "Hand-painted Japanese animation look: clean cel-shaded characters, lush painted backgrounds, "
             "soft bloom, saturated skies and expressive faces.",
    "Chinese ink wash": "Traditional Chinese ink wash painting: expressive brushstrokes on rice paper, "
                        "graded grey ink, generous empty space, small touches of cinnabar and indigo.",
    "Picture book": "Warm children's picture book illustration: gouache and colored pencil, rounded friendly "
                    "shapes, soft daylight, cozy textured details.",
    "Painterly realism": "Cinematic painterly realism like film concept art: realistic proportions, "
                         "atmospheric depth, volumetric light, rich color grading.",
    "Woodcut": "Bold woodcut print: carved black lines, rough grain texture, two-color palette of black and "
               "deep red on cream paper, strong silhouettes.",
    "Retro pulp": "1950s pulp magazine art: dynamic poses, saturated halftone color, painted gouache "
                  "rendering, dramatic low angles.",
    "Pixel art": "Detailed 16-bit pixel art: limited palette, crisp dithering, side-lit scenes, cinematic "
                 "widescreen composition.",
}
DEFAULT_STYLE = "Ink and watercolor"
CUSTOM_STYLE = "Custom"


def preset_for(text):
    """The preset an Art direction text matches, or Custom once it's been edited."""
    text = (text or "").strip()
    return next((name for name, value in STYLE_PRESETS.items() if value == text), CUSTOM_STYLE)
_SHEET_NOTES = {
    "characters": "keep the same face, hairstyle, build and outfit",
    "props": "keep the same shape, colors and markings",
    "locations": "keep the same architecture, layout and landmarks",
}


def export_key(result_name, plan, engine, use_references, captured):
    data = json.dumps([result_name, plan, engine, bool(use_references), captured],
                      sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def finished_images(key):
    images = _FINISHED.setdefault(key, {"sheets": {}, "shots": []})
    _FINISHED.move_to_end(key)
    while len(_FINISHED) > _FINISHED_LIMIT:
        _FINISHED.popitem(last=False)
    return images


def widescreen(engine, captured):
    """Override the Image tab's size with a 16:9 one; edits use it too instead of a reference's shape."""
    width, height = _WIDESCREEN.get(engine, (1344, 768))
    captured.update(qwen__batch_count=1, qwen__batch_edit_inputs=False, moss_image__layout="", flux3__layout="",
                    qwen__output_resolution_1k=None, qwen__output_resolution_2k=None)
    if engine == "h3":
        captured.update(image__width=width, image__height=height, image__edit_size=QWEN_EDIT_SIZE_MANUAL)
    else:
        captured.update({f"{engine}__width": width, f"{engine}__height": height})
        if engine in ("qwen", "flux3"):
            captured[f"{engine}__edit_size"] = QWEN_EDIT_SIZE_MANUAL


def generate_shot(controller, engine, prompt, sources, captured):
    """One scene: from text alone, or as an edit that follows the subjects' reference sheets.

    ``sources`` maps each subject key to ``(kind, sheet path)`` in the planner's priority order.
    """
    if not sources:
        return controller.generate(engine, "Text to image", prompt, [], captured)
    instructions = ["Draw the new scene described below. The supplied images are reference sheets for its "
                    "subjects, not a layout to copy."]
    for position, (key, (kind, _)) in enumerate(sources.items(), 1):
        tag = reference_tag(engine, position, len(sources)) or "the reference image"
        instructions.append(f"{tag} shows {key}: {_SHEET_NOTES[kind]}.")
    instructions.append("Do not copy the sheets' poses, plain backdrops or side-by-side views. Use the scene's "
                        "own action, setting, framing and lighting, as one single image.")
    instructions.append(prompt)
    return controller.generate(engine, "Image edit", "\n".join(instructions),
                               [path for _, path in sources.values()], captured)


def _draw(controller, engine, mode_sources, prompt, captured, label):
    """Stream one image's progress; return its path."""
    path, message = None, "No image returned."
    for result, message in generate_shot(controller, engine, prompt, mode_sources, captured):
        if message.startswith("Error:"):
            raise H3Error(f"{label} failed. {message}")
        yield gr.skip(), f"{label} · {message}"
        if result:
            path = result[0] if isinstance(result, list) else result
    if not path or not Path(path).is_file():
        raise H3Error(f"{label} failed. {message}")
    return path


def illustrate_shots(controller, engine, shots, captured, use_references, finished=None, sheets=None):
    """Draw each subject's reference sheet, then every scene from the sheets of what it shows.

    ``sheets`` maps subject keys to ``(kind, sheet prompt)``; only subjects some shot shows are drawn.
    ``finished`` holds images from an earlier attempt at the same export; existing files are reused,
    and each new image is recorded so a later retry can resume after it.
    """
    finished = {"sheets": {}, "shots": []} if finished is None else finished
    widescreen(engine, captured)
    limit = reference_limit(engine, captured.get("qwen__turbo_variant", "Off")) if use_references else 0
    shown = list(dict.fromkeys(key for shot in shots for key in shot.references if key in (sheets or {})))
    drawn = {}
    if limit:
        for number, key in enumerate(shown, 1):
            kind, prompt = sheets[key]
            label = f"Reference {number} of {len(shown)} · {key}"
            saved = finished["sheets"].get(key)
            if saved and Path(saved).is_file():
                path = saved
                yield None, f"{label} · Reused from the interrupted export"
            else:
                yield None, label
                path = yield from _draw(controller, engine, {}, prompt, captured, label)
                finished["sheets"][key] = path
            drawn[key] = (kind, path)
    shot_records = []
    images = []
    done = finished["shots"]
    for index, shot in enumerate(shots):
        sources = {key: drawn[key] for key in shot.references if key in drawn}
        sources = dict(list(sources.items())[:limit])
        label = f"Shot {index + 1} of {len(shots)}"
        if sources:
            label += " · Following " + ", ".join(sources)
        if index < len(done) and Path(done[index]).is_file():
            path = done[index]
            yield None, f"{label} · Reused from the interrupted export"
        else:
            del done[index:]
            yield None, label
            path = yield from _draw(controller, engine, sources, shot.prompt, captured, label)
            done.append(path)
        images.append(path)
        shot_records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                             "image": Path(path).name, "references": list(sources)})
    sheet_records = {key: Path(path).name for key, (_, path) in drawn.items()}
    return images, shot_records, sheet_records


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


def plan_rows(saved, duration, shots=None):
    """Shot plan table rows from compiled shots, or from a saved plan."""
    if shots is None:
        try:
            shots = compile_plan(saved["plan"], duration, saved["style"], shot_cap(duration, saved["limit"]))
        except (H3Error, KeyError, TypeError):
            return []
    return [[round(s.first / FPS, 3), round((s.first + s.frames) / FPS, 3),
             "Dissolve" if s.dissolve else "Cut", s.beat, s.prompt] for s in shots]


def bind_visual_export(view, controller, deps):
    with view.visual_root, gr.Accordion("Illustrated video", open=False):
        engine = gr.Dropdown([(item.label, item.id) for item in engines_for_task("Image")],
                             value="qwen", label="Image model", info="Uses this model’s settings from the Image tab.")
        preset = gr.Dropdown([*STYLE_PRESETS, CUSTOM_STYLE], value=DEFAULT_STYLE, label="Style preset",
                             info="Fills Art direction. Edit the text to make it your own.")
        style = gr.Textbox(label="Art direction", value=STYLE_PRESETS[DEFAULT_STYLE], lines=2,
                           placeholder="Describe the cast, setting and illustration style.")
        limit = gr.Slider(1, 64, value=16, step=1, label="Minimum shots",
                          info="One moment per image. The planner uses at least this many and adds more "
                               "where the story needs them.")
        plan_button = gr.Button("Plan shots", size="sm")
        gr.Markdown("Generate plans the shots from the finished mix if you haven't reviewed a plan. "
                    "Uses the Gemini or OpenAI-compatible writer selected on Script.")
        preview = gr.Dataframe(headers=["Start (s)", "End (s)", "Transition", "Story beat", "Image prompt"],
                               interactive=False, wrap=True, label="Shot plan")
        continuity = gr.Checkbox(value=True, label="Draw reference sheets first",
                                 info="Draws each recurring character, prop and location once, then edits every "
                                      "scene from the sheets of what it shows. MOSS-Image uses text prompts only.")
        run, stop, status = generation_actions("Generate illustrated video")
        video = gr.Video(label="Illustrated mix", interactive=False)
        source = gr.State(None)

    def select_mix(path):
        """Show the plan saved with this mix, if any, so a loaded project keeps its reviewed plan."""
        name = Path(path).name if path else None
        rows = []
        if name:
            audio = Path(deps.resolve_result(name))
            saved = read_plan(audio)
            duration = ((read_snapshot(audio) or {}).get("settings") or {}).get("duration")
            if saved is not None and duration:
                rows = plan_rows(saved, float(duration))
        return name, None, rows

    view.output.change(select_mix, inputs=view.output, outputs=[source, video, preview], **QUIET)
    preset.input(lambda name: STYLE_PRESETS.get(name, gr.skip()), inputs=preset, outputs=style, **QUIET)
    style.input(preset_for, inputs=style, outputs=preset, **QUIET)

    def load_mix(result_name):
        if not result_name:
            raise gr.Error("Generate speech before exporting an illustrated video.")
        audio = Path(deps.resolve_result(Path(result_name).name))
        snapshot = read_snapshot(audio) or {}
        if snapshot.get("family") != DRAMA3_FAMILY or not audio.is_file():
            raise gr.Error("The saved Sound Stage mix could not be found.")
        return audio, snapshot["settings"]

    def prepare(result_name, style, limit, *writer):
        audio, settings = load_mix(result_name)
        try:
            plan, shots = plan_shots(settings, style, int(limit), *writer, runtime=deps.runtime_config())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        write_plan(audio, style, limit, plan)
        return plan_rows(None, None, shots)

    bind_prompt_action(plan_button.click, prepare, inputs=[source, style, limit, *writer_inputs(view.writer)],
                       outputs=preview, api_name=False, show_progress="minimal")

    def plan_if_needed(result_name, style, limit, *writer):
        """Reuse the plan saved with this mix; otherwise plan from the finished mix before the GPU job."""
        audio, settings = load_mix(result_name)
        saved = read_plan(audio, style, limit)
        if saved is not None:
            rows = plan_rows(saved, settings["duration"])
            return rows, gr.update(interactive=True), f"Using the saved {len(rows)}-shot plan…"
        rows = prepare(result_name, style, limit, *writer)
        return rows, gr.update(interactive=True), f"Planned {len(rows)} shots. Generating images…"

    def generate(result_name, engine, style, limit, use_references, *values):
        audio, settings = load_mix(result_name)
        saved = read_plan(audio, style, limit)
        if saved is None:
            raise gr.Error("This mix has no shot plan for the current art direction. Click Plan shots.")
        if which("ffmpeg") is None:
            raise gr.Error("Install FFmpeg to export an illustrated video.")
        duration = settings["duration"]
        shots = compile_plan(saved["plan"], duration, saved["style"], shot_cap(duration, saved["limit"]))
        captured = dict(zip(controller.fields, values, strict=True))
        key = export_key(result_name, saved["plan"], engine, use_references, captured)
        art_style = saved["plan"].get("art_style") or saved["plan"].get("style")
        sheets = {name: (kind, sheet_prompt(kind, name, anchor, art_style, saved["style"]))
                  for name, (kind, anchor) in subjects(saved["plan"]).items()}
        images, shot_records, sheet_records = yield from illustrate_shots(
            controller, engine, shots, captured, use_references, finished_images(key), sheets)
        yield gr.skip(), "Rendering video with the finished mix…"
        video_path = render_video(audio, shots, images, deps.runtime_config().output_dir / "drama3")
        _FINISHED.pop(key, None)  # A fresh export of the same settings draws new illustrations.
        write_snapshot(video_path, {"family": "Sound Stage illustrated video", "settings": {
            "source": audio.name, "model": engine, "art_direction": style, "visual_plan": saved["plan"],
            "use_visual_references": use_references, "reference_sheets": sheet_records, "shots": shot_records,
        }})
        yield str(video_path), f"Exported {len(shots)} illustrated shots."

    # The writer runs in the prompt queue, so its keys never enter the GPU job's saved snapshot.
    # The plan travels as a file beside the mix, not as session state, so the GPU job always finds it.
    planning = bind_prompt_action(run.click, plan_if_needed,
                                  inputs=[source, style, limit, *writer_inputs(view.writer)],
                                  outputs=[preview, stop, status], api_name=False, show_progress="minimal")
    event = bind_gpu_action(planning.success, owned_generation(generate, "drama3_visuals",
                            input_names=("result_name", "engine", "style", "max_shots",
                                         "use_visual_references", *controller.fields)),
                            inputs=[source, engine, style, limit, continuity, *controller.fields.values()],
                            outputs=[video, status], api_name=False, show_progress="minimal")
    stop.click(owned_interrupt(deps.interrupt, "drama3_visuals"), outputs=status, **QUIET).then(
        None, cancels=[planning, event], **QUIET)
