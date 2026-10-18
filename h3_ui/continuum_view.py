"""Sequence (H3 Continuum) controls inside the shared H3 composer and preview.

Clip keeps its own prompt, results and actions; Sequence adds a shared scene
prompt, chunk rows and review actions. Conditioning inputs and model settings
are shared components, but each mode compiles its own request at admission.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import gradio as gr

from h3_app.continuum_policy import (
    CONTINUITY_CHOICES,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_CONTINUITY,
    DEFAULT_SEQUENCE_SIZE,
    DEFAULT_SEQUENCE_STEPS,
    MAX_CHUNK_SECONDS,
    MAX_CHUNKS,
    MAX_SEQUENCE_STEPS,
    MIN_CHUNK_SECONDS,
    MIN_SEQUENCE_STEPS,
    SEQUENCE_SIZES,
)
from h3_app.generation.continuum_requests import (
    CONTINUE_FROM_TAKE,
    FINISH,
    NEXT,
    REBUILD,
    REGENERATE_FROM,
    RETRY,
    USE_TAKE,
)

from .continuum_bindings import (
    CLIP,
    DESCRIBE_INPUT_NAMES,
    SEQUENCE,
    SEQUENCE_INPUT_NAMES,
    VIDEO_MODES,
    ContinuumActions,
    NEW_SEQUENCE,
)
from .job_bindings import bind_gpu_action, owned_generation, owned_interrupt

FAMILY = "h3_continuum"


@dataclass(frozen=True)
class ContinuumView:
    video_mode: gr.Radio
    saved: gr.Dropdown
    refresh_saved: gr.Button
    delete: gr.Button
    name: gr.Textbox
    shared_prompt: gr.Textbox
    chunk_rows: tuple
    chunk_prompts: tuple
    add_chunk: gr.Button
    remove_chunk: gr.Button
    chunk_count: gr.Number
    chunk_seconds: gr.Slider
    continuity: gr.Dropdown
    size: gr.Dropdown
    steps: gr.Slider
    seed: gr.Number
    summary: gr.Markdown
    prompt_preview: gr.Markdown
    run_id: gr.Textbox
    epoch: gr.Textbox
    video: gr.Video
    info: gr.Markdown
    primary: gr.Button
    finish: gr.Button
    retry: gr.Button
    take: gr.Dropdown
    use_take: gr.Button
    continue_from_take: gr.Button
    regenerate_from: gr.Dropdown
    regenerate: gr.Button
    rebuild: gr.Button
    stop: gr.Button
    status: gr.Textbox
    action_kinds: dict


def build_continuum_view(roots) -> ContinuumView:
    with roots.mode:
        video_mode = gr.Radio(
            list(VIDEO_MODES),
            value=CLIP,
            label="Video",
            info="Clip makes one independent video (up to 15 s). Sequence stacks "
            "chunks, each with its own prompt, into one continuous video.",
            elem_id="h3-video-mode-choice",
        )
    with roots.compose:
        gr.HTML(
            '<div class="h3-panel-heading"><div><h3>Sequence</h3>'
            "<p>A shared scene plus one action per chunk. Each chunk continues "
            "from the last; retry a chunk without losing earlier ones.</p></div></div>"
        )
        saved = gr.Dropdown([NEW_SEQUENCE], value="", label="Saved sequence")
        with gr.Row():
            refresh_saved = gr.Button("Refresh list", size="sm")
            delete = gr.Button("Delete sequence", size="sm", variant="stop")
        name = gr.Textbox(label="Sequence name", placeholder="Untitled sequence")
        shared_prompt = gr.Textbox(
            label="Scene & style (all chunks)",
            lines=4,
            max_lines=16,
            placeholder="Who and where: characters, setting, look, camera style…",
        )
        rows, prompts = [], []
        for index in range(1, MAX_CHUNKS + 1):
            with gr.Group(visible=index == 1) as row:
                prompts.append(
                    gr.Textbox(
                        label=f"Chunk {index} action",
                        lines=2,
                        max_lines=8,
                        placeholder="What happens in this chunk. Leave empty to reuse the scene prompt.",
                    )
                )
            rows.append(row)
        with gr.Row():
            add_chunk = gr.Button("+ Add chunk", size="sm")
            remove_chunk = gr.Button("− Remove last chunk", size="sm")
        with gr.Row():
            chunk_seconds = gr.Slider(
                MIN_CHUNK_SECONDS,
                MAX_CHUNK_SECONDS,
                value=DEFAULT_CHUNK_SECONDS,
                step=0.5,
                label="Seconds per chunk",
            )
            continuity = gr.Dropdown(
                list(CONTINUITY_CHOICES),
                value=DEFAULT_CONTINUITY,
                label="Continuity",
                info="Frames of context carried into each next chunk.",
            )
        with gr.Row():
            size = gr.Dropdown(
                list(SEQUENCE_SIZES),
                value=DEFAULT_SEQUENCE_SIZE,
                label="Size",
                info="Sequence keeps memory-safe sizes for its decode and assembly.",
            )
            steps = gr.Slider(
                MIN_SEQUENCE_STEPS,
                MAX_SEQUENCE_STEPS,
                value=DEFAULT_SEQUENCE_STEPS,
                step=1,
                label="Steps",
                info="Normal sampling; Clip recipes and Turbo do not apply.",
            )
        seed = gr.Number(
            value=-1,
            precision=0,
            label="Seed",
            info="-1 picks one at save time; the sequence keeps it for every chunk.",
        )
        summary = gr.Markdown()
        with gr.Accordion("Effective prompts per chunk", open=False):
            prompt_preview = gr.Markdown()
        chunk_count = gr.Number(value=1, precision=0, visible=False)
        run_id = gr.Textbox(visible=False)
        epoch = gr.Textbox(value="0", visible=False)
    with roots.results:
        gr.HTML(
            '<div class="h3-panel-heading"><div><h3>Sequence preview</h3>'
            "<p>Review each chunk as it lands. Previews show the sequence so far.</p></div></div>"
        )
        video = gr.Video(label="Sequence", elem_classes=["h3-result-player"])
        info = gr.Markdown("New sequence. It is saved when you first generate.")
        with gr.Row(elem_classes=["h3-generation-buttons"]):
            primary = gr.Button(
                "Save sequence & generate",
                variant="primary",
                scale=2,
                elem_classes=["h3-primary-action"],
            )
            finish = gr.Button("Generate all remaining", scale=1)
            retry = gr.Button("Retry last chunk", scale=1)
        status = gr.Textbox(label="Sequence progress", lines=2, interactive=False, elem_classes=["h3-status"])
        with gr.Accordion("Takes & regeneration", open=False):
            take = gr.Dropdown([], label="Saved takes", info="Previewing a take never changes the sequence.")
            with gr.Row():
                use_take = gr.Button("Use this take", size="sm")
                continue_from_take = gr.Button("Continue from this take", size="sm")
                rebuild = gr.Button("Rebuild video", size="sm")
            with gr.Row():
                regenerate_from = gr.Dropdown([], label="Regenerate from chunk", scale=2)
                regenerate = gr.Button("Regenerate from here", size="sm", scale=1)
        stop = gr.Button("Interrupt", size="sm")
    primary.h3_stop = stop
    # One hidden constant per action keeps a single positional job contract.
    action_kinds = {
        kind: gr.Textbox(value=kind, visible=False)
        for kind in (NEXT, FINISH, RETRY, USE_TAKE, CONTINUE_FROM_TAKE, REBUILD, REGENERATE_FROM)
    }
    return ContinuumView(
        video_mode=video_mode,
        saved=saved,
        refresh_saved=refresh_saved,
        delete=delete,
        name=name,
        shared_prompt=shared_prompt,
        chunk_rows=tuple(rows),
        chunk_prompts=tuple(prompts),
        add_chunk=add_chunk,
        remove_chunk=remove_chunk,
        chunk_count=chunk_count,
        chunk_seconds=chunk_seconds,
        continuity=continuity,
        size=size,
        steps=steps,
        seed=seed,
        summary=summary,
        prompt_preview=prompt_preview,
        run_id=run_id,
        epoch=epoch,
        video=video,
        info=info,
        primary=primary,
        finish=finish,
        retry=retry,
        take=take,
        use_take=use_take,
        continue_from_take=continue_from_take,
        regenerate_from=regenerate_from,
        regenerate=regenerate,
        rebuild=rebuild,
        stop=stop,
        status=status,
        action_kinds=action_kinds,
    )


def _named(callback, names):
    """Expose named parameters so Gradio injects gr.Request after *args."""

    def run(*args):
        # Gradio passes the request positionally, at its signature position.
        return callback(*args[:-1], request=args[-1])

    parameters = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    parameters.append(
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=gr.Request)
    )
    run.__signature__ = inspect.Signature(parameters)
    run.__annotations__ = {"request": gr.Request}
    return run


def bind_continuum_view(view: ContinuumView, h3, roots, actions: ContinuumActions) -> None:
    quiet = {"queue": False, "api_name": False, "show_progress": "hidden"}

    clip_only = [
        *roots.clip_prompt,
        roots.clip_settings,
        roots.clip_summary,
        roots.clip_results,
        roots.clip_actions,
    ]

    def switch(mode):
        sequence = mode == SEQUENCE
        return (
            gr.update(visible=sequence),
            gr.update(visible=sequence),
            *(gr.update(visible=not sequence) for _ in clip_only),
        )

    view.video_mode.change(
        switch,
        inputs=view.video_mode,
        outputs=[roots.compose, roots.results, *clip_only],
        **quiet,
    )

    def resize(count, delta):
        count = max(1, min(MAX_CHUNKS, int(count or 1) + delta))
        return (count, *(gr.update(visible=i < count) for i in range(MAX_CHUNKS)))

    for button, delta in ((view.add_chunk, 1), (view.remove_chunk, -1)):
        button.click(
            lambda count, delta=delta: resize(count, delta),
            inputs=view.chunk_count,
            outputs=[view.chunk_count, *view.chunk_rows],
            **quiet,
        )

    shared = {name: getattr(h3, name) for name in (
        "mode", "first", "last", "model_profile", "text_encoder",
        "generation_mode", "attention_mode", "cache_mode", "latent_upscale", "use_int8_vae",
        "use_trt_vae", "use_lynnreal_vae", "lightvae_decoder", "semantic_bridge",
        *(f"ref_image_{i}" for i in range(1, 10)),
        *(f"ref_video_{i}" for i in range(1, 4)),
        *(f"ref_audio_{i}" for i in range(1, 4)),
    )}
    shared["postprocess"] = h3.generation_postprocess
    own = {
        "run_id": view.run_id,
        "epoch": view.epoch,
        "name": view.name,
        "shared_prompt": view.shared_prompt,
        "chunk_count": view.chunk_count,
        "chunk_seconds": view.chunk_seconds,
        "continuity": view.continuity,
        "size": view.size,
        "steps": view.steps,
        "seed": view.seed,
        "take": view.take,
        "regenerate_from": view.regenerate_from,
        **{f"chunk_prompt_{i}": box for i, box in enumerate(view.chunk_prompts, start=1)},
    }
    components = {**shared, **own}

    describe_inputs = [components[name] for name in DESCRIBE_INPUT_NAMES]
    describe_outputs = [view.summary, view.prompt_preview, view.primary]
    describe = _named(actions.describe, DESCRIBE_INPUT_NAMES)
    for trigger in (
        view.video_mode, view.shared_prompt, *view.chunk_prompts, view.chunk_count,
        view.chunk_seconds, view.size, view.run_id, h3.mode, h3.last,
    ):
        trigger.change(describe, inputs=describe_inputs, outputs=describe_outputs, **quiet)

    load_outputs = [
        view.run_id, view.epoch, view.name, view.shared_prompt, *view.chunk_prompts,
        view.chunk_count, *view.chunk_rows, view.chunk_seconds, view.continuity,
        view.size, view.steps, view.seed, view.video, view.info, view.take, view.regenerate_from, view.primary,
    ]
    view.saved.input(actions.load, inputs=view.saved, outputs=load_outputs, **quiet)
    for trigger in (view.refresh_saved.click, view.video_mode.change):
        trigger(actions.saved_choices, outputs=view.saved, **quiet)
    view.delete.click(
        actions.delete, inputs=view.run_id, outputs=load_outputs, **quiet
    ).then(actions.saved_choices, outputs=view.saved, **quiet).then(
        lambda: "", outputs=view.saved, **quiet
    )

    callback = owned_generation(actions.run, FAMILY, SEQUENCE_INPUT_NAMES)
    callback.job_admission = actions.admit
    job_outputs = [view.video, view.status, view.run_id, view.epoch]
    events = []
    for button, kind in (
        (view.primary, NEXT),
        (view.finish, FINISH),
        (view.retry, RETRY),
        (view.use_take, USE_TAKE),
        (view.continue_from_take, CONTINUE_FROM_TAKE),
        (view.rebuild, REBUILD),
        (view.regenerate, REGENERATE_FROM),
    ):
        inputs = [view.action_kinds[kind], *(components[name] for name in SEQUENCE_INPUT_NAMES[1:])]
        event = bind_gpu_action(
            button.click,
            callback,
            inputs=inputs,
            outputs=job_outputs,
            show_progress="minimal",
            api_name=False,
        )
        # Refresh takes, counters and labels from the committed catalog.
        event.then(actions.load, inputs=view.run_id, outputs=load_outputs, **quiet).then(
            actions.saved_choices, outputs=view.saved, **quiet
        ).then(lambda run_id: run_id, inputs=view.run_id, outputs=view.saved, **quiet)
        events.append(event)
    stopped = view.stop.click(
        owned_interrupt(actions.interrupt, FAMILY), outputs=view.status, **quiet
    )
    stopped.then(fn=None, cancels=events, queue=False, api_name=False)
