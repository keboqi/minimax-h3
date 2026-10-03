"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation


from .ltx_view import LtxView


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
    view.mode.change(
        lambda value: gr.update(visible=value == "Reference images"),
        inputs=view.mode,
        outputs=view.reference_group,
        queue=False,
        show_progress="hidden",
    )
    view.mode.change(
        lambda value: (
            (768, 448, 5, 24) if value == "Reference images" else (gr.skip(),) * 4
        ),
        inputs=view.mode,
        outputs=[view.width, view.height, view.duration, view.fps],
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
    bind_gpu_action(
        view.prepare_workflow.click,
        prepare_workflow,
        inputs=view.workflow,
        outputs=[view.workflow_status, view.model_inventory],
        show_progress="minimal",
    )
    bind_gpu_action(
        view.prepare_all_models.click,
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
    bind_prompt_action(
        view.enhance.click,
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
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.prompt, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_ltx25_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "ltx"),
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
            view.reference_images,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_ltx25_video",
    )
