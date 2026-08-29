"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import gradio as gr

from .ltx_view import LtxView
from .views import ApiView, GalleryView, MusicView


def bind_preflight(
    controls: Sequence[gr.components.Component],
    *,
    prompt: gr.Textbox,
    callback: Callable[..., Any],
    readiness: gr.HTML,
    primary_action: gr.Button,
) -> None:
    """Keep readiness synchronized for both committed and live prompt input."""

    outputs = [readiness, primary_action]
    for control in controls:
        control.change(
            callback,
            inputs=list(controls),
            outputs=outputs,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
    prompt.input(
        callback,
        inputs=list(controls),
        outputs=outputs,
        queue=False,
        show_progress="hidden",
        api_name=False,
    )


def bind_summary(
    controls: Iterable[gr.components.Component],
    *,
    callback: Callable[..., Any],
    output: gr.Markdown,
    skip: Iterable[gr.components.Component] = (),
) -> None:
    """Refresh a derived summary when any non-specialized control changes."""

    inputs = list(controls)
    skipped = set(skip)
    for control in inputs:
        if control in skipped:
            continue
        control.change(callback, inputs=inputs, outputs=output)


def bind_interrupts(
    callback: Callable[..., Any],
    bindings: Iterable[tuple[gr.Button, gr.components.Component, Sequence[Any]]],
) -> None:
    """Apply identical cancellation semantics across generation views."""

    for button, output, events in bindings:
        button.click(callback, outputs=output, cancels=list(events))


def bind_ltx_view(
    view: LtxView,
    *,
    render_workflow: Callable[..., Any],
    prepare_workflow: Callable[..., Any],
    prepare_all_models: Callable[..., Any],
    render_inventory: Callable[..., Any],
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    view.mode.change(
        lambda value: gr.update(visible=value == "Image to video"),
        inputs=view.mode,
        outputs=view.image_group,
        queue=False,
        show_progress="hidden",
    )
    view.workflow.change(
        render_workflow,
        inputs=view.workflow,
        outputs=view.workflow_details,
        queue=False,
        show_progress="hidden",
    )
    view.prepare_workflow.click(
        prepare_workflow,
        inputs=view.workflow,
        outputs=[view.workflow_status, view.model_inventory],
        show_progress="minimal",
    )
    view.prepare_all_models.click(
        prepare_all_models,
        outputs=[view.workflow_status, view.model_inventory],
        show_progress="minimal",
    )
    view.refresh_models.click(
        render_inventory,
        outputs=view.model_inventory,
        queue=False,
        show_progress="hidden",
    )
    view.enhance.click(
        enhance_prompt,
        inputs=[
            view.prompt,
            view.prompt_model,
            view.api_key,
            view.mode,
            view.image,
            view.middle_image,
            view.end_image,
            view.duration,
            view.width,
            view.height,
        ],
        outputs=[view.prompt, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_ltx25_prompt",
    )
    return view.run.click(
        generate,
        inputs=[
            view.mode,
            view.model,
            view.prompt,
            view.negative,
            view.image,
            view.duration,
            view.fps,
            view.width,
            view.height,
            view.seed,
            view.cfg,
            view.sampler,
            view.image_strength,
            view.middle_image,
            view.middle_time,
            view.middle_strength,
            view.end_image,
            view.end_strength,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_ltx25_video",
    )


def bind_music_view(
    view: MusicView,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    view.enhance.click(
        enhance_prompt,
        inputs=[
            view.caption,
            view.prompt_model,
            view.api_key,
            view.lyrics,
            *view.reference_images,
        ],
        outputs=[view.caption, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_music3_prompt",
    )
    return view.run.click(
        generate,
        inputs=[
            view.model,
            view.caption,
            view.lyrics,
            view.duration,
            view.seed,
            view.steps,
            view.cfg,
            view.ar_cfg,
            view.top_k,
            view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_music3",
    )


def bind_api_view(view: ApiView, *, generate: Callable[..., Any]) -> Any:
    return view.run.click(
        generate,
        inputs=view.prompt,
        outputs=[view.download_url, view.status],
        show_progress="minimal",
        api_name="generate_video",
    )


def bind_gallery_view(
    view: GalleryView,
    *,
    tab: gr.Tab,
    selected_ltx_model: gr.Dropdown,
    ai_options: set[str],
    seedvr_option: str,
    ltx_option: str,
    refresh: Callable[..., Any],
    select: Callable[..., Any],
    import_video: Callable[..., Any],
    postprocess: Callable[..., Any],
    interrupt: Callable[..., Any],
    delete: Callable[..., Any],
    empty: Callable[..., Any],
) -> None:
    view.postprocess.change(
        lambda value: (
            gr.update(visible=value in ai_options),
            gr.update(visible=value == seedvr_option),
            gr.update(visible=value == ltx_option),
            gr.update(visible=value == ltx_option),
            gr.update(visible=value == ltx_option),
        ),
        inputs=view.postprocess,
        outputs=[
            view.ai_settings,
            view.seedvr2_model,
            view.ltx25_prompt,
            view.split_upscale,
            view.split_seconds,
        ],
        queue=False,
        show_progress="hidden",
    )
    opened = tab.select(
        lambda: (None, "", None, False),
        outputs=[view.player, view.download, view.selected, view.confirm_delete],
    )
    opened.then(
        refresh,
        outputs=[view.grid, view.paths, view.status],
        show_progress="hidden",
    )
    view.refresh.click(
        refresh,
        outputs=[view.grid, view.paths, view.status],
        show_progress="minimal",
    )
    view.grid.select(
        select,
        inputs=view.paths,
        outputs=[view.player, view.download, view.selected],
        show_progress="minimal",
    )
    mutation_outputs = [
        view.grid,
        view.paths,
        view.status,
        view.player,
        view.download,
        view.selected,
        view.confirm_delete,
    ]
    view.import_video.click(
        import_video,
        inputs=[view.upload_video],
        outputs=mutation_outputs,
        show_progress="minimal",
        api_name=False,
    )
    post_event = view.post_run.click(
        postprocess,
        inputs=[
            view.selected,
            view.postprocess,
            view.post_seed,
            view.seedvr2_model,
            selected_ltx_model,
            view.ltx25_prompt,
            view.force_offload,
            view.split_upscale,
            view.split_seconds,
            view.upscale_resolution,
        ],
        outputs=mutation_outputs + [view.post_status],
        show_progress="minimal",
        api_name=False,
    )
    view.post_stop.click(
        interrupt, outputs=view.post_status, cancels=[post_event], api_name=False
    )
    view.delete.click(
        delete,
        inputs=[view.selected, view.confirm_delete],
        outputs=mutation_outputs,
        show_progress="minimal",
        api_name=False,
    )
    view.empty.click(
        empty,
        inputs=[view.selected, view.confirm_delete],
        outputs=mutation_outputs,
        show_progress="minimal",
        api_name=False,
    )
