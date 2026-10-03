"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation


from .yue2_view import YuE2View


def bind_yue2_view(
    view: YuE2View,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.style,
            view.prompt_model,
            view.api_key,
            view.lyrics,
            view.mode,
            view.duration,
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.style, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_yue2_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "yue2"),
        inputs=[
            view.model,
            view.style,
            view.lyrics,
            view.abc,
            view.mode,
            view.duration,
            view.seed,
            view.steps,
            view.cfg,
            view.temperature,
            view.top_p,
            view.top_k,
            view.repetition_penalty,
            view.max_abc_tokens,
            view.abc_temperature,
            view.abc_top_p,
            view.abc_top_k,
            view.abc_repetition_penalty,
            view.abc_penalty_window,
            view.tiled,
        ],
        outputs=[view.output, view.status],
        show_progress="minimal",
        api_name="generate_yue2",
    )
