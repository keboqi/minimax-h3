"""Typed builders for self-contained application views."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class ApiView:
    prompt: gr.Textbox
    run: gr.Button
    stop: gr.Button
    download_url: gr.Textbox
    status: gr.Textbox


def build_api_view(root: gr.Group, guide: str) -> ApiView:
    with root:
        gr.Markdown(guide)
        with gr.Accordion("Try the default API request", open=False):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="Describe the video, camera motion, dialogue, and sound.",
            )
            with gr.Row():
                run = gr.Button("Generate with UI defaults", variant="primary")
                stop = gr.Button("Interrupt")
            download_url = gr.Textbox(label="Download URL", interactive=False)
            status = gr.Textbox(label="Status", lines=5)
    return ApiView(prompt, run, stop, download_url, status)
