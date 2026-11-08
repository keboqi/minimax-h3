"""Optional illustrated video export from the Speak result."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from shutil import which

import gradio as gr

from h3_app.capabilities import engines_for_task
from h3_app.drama3.results import DRAMA3_FAMILY
from h3_app.drama3.visual_plan import compile_plan, plan_shots
from h3_app.drama3.visuals import FPS, render_video
from h3_app.errors import H3Error
from h3_app.provenance import read_snapshot, write_snapshot
from h3_ui.engines.controls import QUIET, generation_actions, writer_inputs
from h3_ui.image_controller import reference_limit, reference_tag
from h3_ui.job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt


# Illustrations finished by an export that failed or was stopped, by export fingerprint.
# A retry with the same mix, plan, model and settings resumes after them.
_FINISHED: OrderedDict[str, list[str]] = OrderedDict()
_FINISHED_LIMIT = 8


def export_key(result_name, plan, engine, use_references, captured):
    data = json.dumps([result_name, plan, engine, bool(use_references), captured],
                      sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def finished_images(key):
    images = _FINISHED.setdefault(key, [])
    _FINISHED.move_to_end(key)
    while len(_FINISHED) > _FINISHED_LIMIT:
        _FINISHED.popitem(last=False)
    return images


def shot_references(keys, first_seen, limit):
    """Group visible subjects by their first illustrated shot, in the planner's priority order."""
    sources = {}
    for key in dict.fromkeys(keys):
        shot = first_seen.get(key)
        if shot is not None and (shot in sources or len(sources) < limit):
            sources.setdefault(shot, []).append(key)
    return sources


def generate_shot(controller, engine, prompt, sources, images, captured):
    if not sources:
        return controller.generate(engine, "Text to image", prompt, [], captured)

    instructions = ["Compose a new shot using the supplied illustrations as visual references."]
    for position, keys in enumerate(sources.values(), 1):
        tag = reference_tag(engine, position, len(sources)) or "the reference image"
        instructions.append(f"Preserve the identifying features of {', '.join(keys)} from {tag}.")
    instructions.append("Use the new shot's action, setting, framing and lighting. "
                        "Only include subjects called for in the new shot.")
    instructions.append(prompt)
    return controller.generate(engine, "Image edit", "\n".join(instructions),
                               [images[index] for index in sources], captured)


def illustrate_shots(controller, engine, shots, captured, use_references, finished=None):
    """Stream image progress and keep first appearances local to this export.

    ``finished`` holds images from an earlier attempt at the same export; existing files are
    reused in order, and each new image is appended so a later retry can resume after it.
    """
    finished = [] if finished is None else finished
    captured.update(qwen__batch_count=1, qwen__batch_edit_inputs=False,
                    moss_image__layout="", flux3__layout="")
    ref_limit = reference_limit(engine, captured["qwen__turbo_variant"]) if use_references else 0
    first_seen = {}
    shot_records = []
    images = []
    for index, shot in enumerate(shots):
        sources = shot_references(shot.references, first_seen, ref_limit)
        label = f"Shot {index + 1} of {len(shots)}"
        if sources:
            label += " · Using reference shot(s) " + ", ".join(str(n + 1) for n in sources)
        if index < len(finished) and Path(finished[index]).is_file():
            path = finished[index]
            yield None, f"{label} · Reused from the interrupted export"
        else:
            del finished[index:]
            yield None, label
            path = None
            message = "No image returned."
            for result, message in generate_shot(controller, engine, shot.prompt, sources, images, captured):
                if message.startswith("Error:"):
                    raise H3Error(f"{label} failed. {message}")
                yield gr.skip(), f"{label} · {message}"
                if result:
                    path = result[0] if isinstance(result, list) else result
            if not path or not Path(path).is_file():
                raise H3Error(f"{label} failed. {message}")
            finished.append(path)
        images.append(path)
        for key in shot.references:
            first_seen.setdefault(key, index)
        shot_records.append({"first_frame": shot.first, "frames": shot.frames, "dissolve": shot.dissolve,
                             "image": Path(path).name, "references": [
                                 {"shot": n + 1, "subjects": subjects} for n, subjects in sources.items()]})
    return images, shot_records


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
            shots = compile_plan(saved["plan"], duration, saved["style"], saved["limit"])
        except (H3Error, KeyError, TypeError):
            return []
    return [[round(s.first / FPS, 3), round((s.first + s.frames) / FPS, 3),
             "Dissolve" if s.dissolve else "Cut", s.beat, s.prompt] for s in shots]


def bind_visual_export(view, controller, deps):
    with view.visual_root, gr.Accordion("Illustrated video", open=False):
        engine = gr.Dropdown([(item.label, item.id) for item in engines_for_task("Image")],
                             value="qwen", label="Image model", info="Uses this model’s settings from the Image tab.")
        style = gr.Textbox(label="Art direction", value="Cinematic ink and watercolor comic panels",
                           placeholder="Describe the cast, setting and illustration style.")
        limit = gr.Slider(1, 64, value=16, step=1, label="Maximum shots")
        plan_button = gr.Button("Plan shots", size="sm")
        gr.Markdown("Generate plans the shots from the finished mix if you haven't reviewed a plan. "
                    "Uses the Gemini or OpenAI-compatible writer selected on Script.")
        preview = gr.Dataframe(headers=["Start (s)", "End (s)", "Transition", "Story beat", "Image prompt"],
                               interactive=False, wrap=True, label="Shot plan")
        continuity = gr.Checkbox(value=True, label="Use earlier illustrations as visual references",
                                 info="Edit-capable models reuse the first appearance of returning subjects. "
                                      "MOSS-Image uses the planned text prompts.")
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
        shots = compile_plan(saved["plan"], settings["duration"], saved["style"], saved["limit"])
        captured = dict(zip(controller.fields, values, strict=True))
        key = export_key(result_name, saved["plan"], engine, use_references, captured)
        images, shot_records = yield from illustrate_shots(controller, engine, shots, captured, use_references,
                                                           finished_images(key))
        yield gr.skip(), "Rendering video with the finished mix…"
        video_path = render_video(audio, shots, images, deps.runtime_config().output_dir / "drama3")
        _FINISHED.pop(key, None)  # A fresh export of the same settings draws new illustrations.
        write_snapshot(video_path, {"family": "Sound Stage illustrated video", "settings": {
            "source": audio.name, "model": engine, "art_direction": style, "visual_plan": saved["plan"],
            "use_visual_references": use_references, "shots": shot_records,
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
