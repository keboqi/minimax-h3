"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import gradio as gr
from h3_models import QWEN_IMAGE21_TURBO_MODES
from h3_app.catalog import (
    LTX25_CQ_ENHANCER,
    LTX25_DEBLUR,
    LTX25_RESTORATION_OPTIONS,
    LTX25_SAME_RESOLUTION_OPTIONS,
    LTX25_SDR_TO_HDR,
)
from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation, owned_interrupt
from .workspace_mode import workspace_enabled


from .ltx_view import LtxView
from .views import ApiView, GalleryView, MusicView, QwenImage21View, YuE2View




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
        control.change(
            callback,
            inputs=inputs,
            outputs=output,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )


def bind_interrupts(
    callback: Callable[..., Any],
    bindings: Iterable[tuple[gr.Button, gr.components.Component, Sequence[Any]]],
) -> None:
    """Apply identical cancellation semantics across generation views."""

    for button, output, events in bindings:
        button.click(callback, outputs=output, cancels=list(events))







# Legacy imports retain function identity through these compatibility exports.
from .ltx_bindings import bind_ltx_view
from .music_bindings import bind_music_view
from .yue2_bindings import bind_yue2_view
from .qwen_bindings import QWEN_IMAGE21_PRESETS, qwen_resolution_preset_values, qwen_preset_values, qwen_turbo_defaults, bind_qwen_image21_view
from .api_bindings import bind_api_view
from .media_bindings import bind_gallery_view
