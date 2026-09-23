"""Shared remote prompt writer controls for model-specific generation tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from h3_app.catalog import LIGHTNING_PROMPT_MODEL


@dataclass(frozen=True)
class RemotePromptWriterControls:
    backend: gr.Radio
    model: gr.Dropdown
    gemini_api_key: gr.Textbox
    lightning_api_key: gr.Textbox


def build_remote_prompt_writer_controls(
    prompt_models: Sequence[str], default_prompt_model: str
) -> RemotePromptWriterControls:
    backend = gr.Radio(
        choices=("Lightning AI", "Gemini"),
        value="Lightning AI",
        label="Prompt writer",
    )
    with gr.Group(visible=False) as gemini_group:
        with gr.Row():
            model = gr.Dropdown(
                choices=list(prompt_models),
                value=default_prompt_model,
                label="Gemini model",
            )
            gemini_api_key = gr.Textbox(
                label="Temporary Gemini API key",
                type="password",
                placeholder="Uses GEMINI_API_KEY when blank",
            )
    with gr.Group(visible=True) as lightning_group:
        gr.Markdown(
            f"Uses `{LIGHTNING_PROMPT_MODEL}`. Set `LIGHTNING_API_KEY` on the "
            "server or enter a temporary key; UI keys are not stored."
        )
        lightning_api_key = gr.Textbox(
            label="Temporary Lightning API key",
            type="password",
            placeholder="Uses LIGHTNING_API_KEY when blank",
        )
    backend.change(
        lambda value: (
            gr.update(visible=value == "Gemini"),
            gr.update(visible=value == "Lightning AI"),
        ),
        inputs=backend,
        outputs=[gemini_group, lightning_group],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    return RemotePromptWriterControls(
        backend, model, gemini_api_key, lightning_api_key
    )
