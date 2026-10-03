"""Reusable Gradio event-binding helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .job_bindings import bind_gpu_action, owned_generation


from .views import ApiView


def bind_api_view(view: ApiView, *, generate: Callable[..., Any]) -> Any:
    return bind_gpu_action(
        view.run.click,
        owned_generation(generate, "api"),
        inputs=view.prompt,
        outputs=[view.download_url, view.status],
        show_progress="minimal",
        api_name="generate_video",
    )
