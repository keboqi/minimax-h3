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


def bind_visual_export(view, controller, deps):
    with view.visual_root, gr.Accordion("Illustrated video", open=False):
        engine = gr.Dropdown([(item.label, item.id) for item in engines_for_task("Image")],
                             value="qwen", label="Image model", info="Uses this model’s settings from the Image tab.")
        style = gr.Textbox(label="Art direction", value="Cinematic ink and watercolor comic panels",
                           placeholder="Describe the cast, setting and illustration style.")
        limit = gr.Slider(1, 64, value=16, step=1, label="Maximum shots")
        plan_button = gr.Button("Plan shots", size="sm")
        gr.Markdown("Uses the Gemini or OpenAI-compatible writer selected on Script.")
        preview = gr.Dataframe(headers=["Start (s)", "End (s)", "Transition", "Story beat", "Image prompt"],
                               interactive=False, wrap=True, label="Shot plan")
        planned = gr.State(None)
        continuity = gr.Checkbox(value=True, label="Use earlier illustrations as visual references",
                                 info="Edit-capable models reuse the first appearance of returning subjects. "
                                      "MOSS-Image uses the planned text prompts.")
        run, stop, status = generation_actions("Generate illustrated video")
        video = gr.Video(label="Illustrated mix", interactive=False)
        source = gr.State(None)

    def select_mix(path):
        return Path(path).name if path else None, None, None, []

    view.output.change(select_mix, inputs=view.output, outputs=[source, video, planned, preview], **QUIET)
    gr.on([style.input, limit.input], lambda: (None, []), outputs=[planned, preview], **QUIET)

    def load_mix(result_name):
        if not result_name:
            raise gr.Error("Generate speech before exporting an illustrated video.")
        audio = Path(deps.resolve_result(Path(result_name).name))
        snapshot = read_snapshot(audio) or {}
        if snapshot.get("family") != DRAMA3_FAMILY or not audio.is_file():
            raise gr.Error("The saved Sound Stage mix could not be found.")
        return audio, snapshot["settings"]

    def prepare(result_name, style, limit, *writer):
        _, settings = load_mix(result_name)
        try:
            plan, shots = plan_shots(settings, style, int(limit), *writer, runtime=deps.runtime_config())
        except H3Error as exc:
            raise gr.Error(str(exc)) from exc
        rows = [[round(s.first / FPS, 3), round((s.first + s.frames) / FPS, 3),
                 "Dissolve" if s.dissolve else "Cut", s.beat, s.prompt] for s in shots]
        return {"source": result_name, "style": style, "limit": limit, "plan": plan}, rows

    bind_prompt_action(plan_button.click, prepare, inputs=[source, style, limit, *writer_inputs(view.writer)],
                       outputs=[planned, preview], api_name=False, show_progress="minimal")

    def generate(result_name, engine, style, limit, planned, use_references, *values):
        audio, settings = load_mix(result_name)
        if not planned or (planned["source"], planned["style"], planned["limit"]) != (result_name, style, limit):
            raise gr.Error("Plan shots for the current mix and art direction before generating images.")
        if which("ffmpeg") is None:
            raise gr.Error("Install FFmpeg to export an illustrated video.")
        shots = compile_plan(planned["plan"], settings["duration"], style, limit)
        captured = dict(zip(controller.fields, values, strict=True))
        key = export_key(result_name, planned["plan"], engine, use_references, captured)
        images, shot_records = yield from illustrate_shots(controller, engine, shots, captured, use_references,
                                                           finished_images(key))
        yield gr.skip(), "Rendering video with the finished mix…"
        video_path = render_video(audio, shots, images, deps.runtime_config().output_dir / "drama3")
        _FINISHED.pop(key, None)  # A fresh export of the same settings draws new illustrations.
        write_snapshot(video_path, {"family": "Sound Stage illustrated video", "settings": {
            "source": audio.name, "model": engine, "art_direction": style, "visual_plan": planned["plan"],
            "use_visual_references": use_references, "shots": shot_records,
        }})
        yield str(video_path), f"Exported {len(shots)} illustrated shots."

    event = bind_gpu_action(run.click, owned_generation(generate, "drama3_visuals",
                            input_names=("result_name", "engine", "style", "max_shots", "visual_plan",
                                         "use_visual_references", *controller.fields)),
                            inputs=[source, engine, style, limit, planned, continuity, *controller.fields.values()],
                            outputs=[video, status], api_name=False, show_progress="minimal")
    stop.click(owned_interrupt(deps.interrupt, "drama3_visuals"), outputs=status, **QUIET).then(
        None, cancels=[event], **QUIET)
