"""Register the default video generation API with hidden Gradio components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr

from .job_bindings import bind_gpu_action, owned_generation


def bind_default_video_api(generate: Callable[..., Any]) -> None:
    with gr.Group(visible=False):
        prompt = gr.Textbox(label="Prompt", lines=4)
        run = gr.Button("Generate with UI defaults")
        download_url = gr.Textbox(label="Download URL", interactive=False)
        status = gr.Textbox(label="Status", lines=5)
    bind_gpu_action(
        run.click,
        owned_generation(generate, "api"),
        inputs=prompt,
        outputs=[download_url, status],
        show_progress="minimal",
        api_name="generate_video",
    )
