"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .job_bindings import bind_gpu_action, bind_prompt_action, owned_generation


from .views import MusicView


def bind_music_view(
    view: MusicView,
    *,
    enhance_prompt: Callable[..., Any],
    generate: Callable[..., Any],
) -> Any:
    bind_prompt_action(
        view.enhance.click,
        enhance_prompt,
        inputs=[
            view.caption,
            view.prompt_model,
            view.api_key,
            view.lyrics,
            *view.reference_images,
            view.prompt_backend,
            view.lightning_api_key,
        ],
        outputs=[view.caption, view.lyrics, view.enhance_status],
        show_progress="minimal",
        api_name="enhance_music3_prompt",
    )
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "music"),
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
