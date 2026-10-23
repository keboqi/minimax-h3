"""Shared remote prompt writer controls for model-specific generation tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gradio as gr

from h3_app.catalog import DEFAULT_OPENAI_PROMPT_MODEL, OPENAI_PROMPT_MODELS


@dataclass(frozen=True)
class RemotePromptWriterControls:
    backend: gr.Radio
    model: gr.Dropdown
    gemini_api_key: gr.Textbox
    openai_api_key: gr.Textbox
    openai_model: gr.Dropdown
    openai_base_url: gr.Textbox


def build_openai_model_dropdown() -> gr.Dropdown:
    return gr.Dropdown(
        choices=list(OPENAI_PROMPT_MODELS),
        value=DEFAULT_OPENAI_PROMPT_MODEL,
        label="OpenAI-compatible model",
        info="Pick one or type any model your endpoint serves.",
        allow_custom_value=True,
    )


def build_openai_base_url_textbox() -> gr.Textbox:
    return gr.Textbox(
        label="Temporary OpenAI endpoint",
        placeholder="Uses OPENAI_BASE_URL, else the official OpenAI API, when blank; needs a temporary key",
    )


def build_remote_prompt_writer_controls(
    prompt_models: Sequence[str], default_prompt_model: str
) -> RemotePromptWriterControls:
    default_backend = "Gemini"
    backend = gr.Radio(
        choices=("OpenAI-compatible", "Gemini"),
        value=default_backend,
        label="Prompt writer",
    )
    with gr.Group(visible=default_backend == "Gemini") as gemini_group:
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
    with gr.Group(visible=default_backend == "OpenAI-compatible") as openai_group:
        gr.Markdown(
            "Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` on the server, or enter a temporary "
            "key and endpoint; UI values are not stored."
        )
        with gr.Row():
            openai_model = build_openai_model_dropdown()
            openai_api_key = gr.Textbox(
                label="Temporary OpenAI API key",
                type="password",
                placeholder="Uses OPENAI_API_KEY when blank",
            )
            openai_base_url = build_openai_base_url_textbox()
    backend.change(
        lambda value: (
            gr.update(visible=value == "Gemini"),
            gr.update(visible=value == "OpenAI-compatible"),
        ),
        inputs=backend,
        outputs=[gemini_group, openai_group],
        queue=False,
        show_progress="hidden",
        api_name=False,
    )
    return RemotePromptWriterControls(
        backend, model, gemini_api_key, openai_api_key, openai_model, openai_base_url
    )
