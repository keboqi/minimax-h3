"""Reusable Gradio event-binding helpers."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from typing import Any
import gradio as gr


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
