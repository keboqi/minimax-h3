"""Build the prompt section in its existing parent container."""

from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Mapping
import gradio as gr
from h3_ui.prompt_writer_controls import build_openai_base_url_textbox, build_openai_model_dropdown

if TYPE_CHECKING:
    from ..services import H3ViewServices


@dataclass(frozen=True)
class PromptSection:
    prompt_preview: gr.components.Component
    prompt_proposal: gr.components.Component
    prompt_previous: gr.components.Component
    accept_prompt: gr.components.Component
    keep_prompt: gr.components.Component
    undo_prompt: gr.components.Component
    enhance_prompt_button: gr.components.Component
    enhance_prompt_status: gr.components.Component
    gemini_api_key: gr.components.Component
    gemini_prompt_model: gr.components.Component
    gemini_prompt_writer_group: gr.components.Component
    help_text: gr.components.Component
    openai_api_key: gr.components.Component
    openai_model: gr.components.Component
    openai_base_url: gr.components.Component
    openai_prompt_writer_group: gr.components.Component
    local_prompt_base_model: gr.components.Component
    local_prompt_greedy: gr.components.Component
    local_prompt_max_tokens: gr.components.Component
    local_prompt_seed: gr.components.Component
    local_prompt_temperature: gr.components.Component
    local_prompt_top_p: gr.components.Component
    local_prompt_writer_group: gr.components.Component
    prompt: gr.components.Component
    prompt_writer_backend: gr.components.Component


def build_prompt_section(
    defaults: Mapping[str, Any],
    services: H3ViewServices,
    *,
    writer_parent=None,
    writer_link: Callable[[], Any] | None = None,
) -> PromptSection:
    """The prompt and its enhance action; the writer service goes in ``writer_parent``."""
    help_text = gr.Markdown(services.mode_help("Text to video"), visible=False)
    prompt = gr.Textbox(
        label="Prompt",
        lines=5,
        max_lines=24,
        placeholder="A quiet seaside town at sunrise. The camera drifts over the rooftops as waves break in the distance…",
        elem_classes=["h3-prompt-editor"],
    )
    with writer_parent if writer_parent is not None else nullcontext():
        gr.Markdown(
            "The local MiniMax-H3 8B writer supports T2VA, I2VA, L2VA, and FL2VA. Gemini supports all Reference media; OpenAI-compatible models support text and image enhancement."
        )
        prompt_writer_backend = gr.Radio(
            services.PROMPT_WRITER_BACKENDS,
            value=services.DEFAULT_PROMPT_WRITER_BACKEND,
            label="Prompt writer",
        )
        with gr.Group(
            visible=services.DEFAULT_PROMPT_WRITER_BACKEND == "Local MiniMax-H3 8B"
        ) as local_prompt_writer_group:
            local_prompt_base_model = gr.Dropdown(
                choices=list(services.LOCAL_PROMPT_BASE_MODELS),
                value=services.DEFAULT_LOCAL_PROMPT_BASE_MODEL,
                label="Local base model",
                info="BF16 is the default full-precision checkpoint. FP8 is available as the lower-memory alternative.",
            )
            with gr.Accordion("Local decoding settings", open=False):
                local_prompt_greedy = gr.Checkbox(value=True, label="Greedy decoding")
                local_prompt_max_tokens = gr.Slider(
                    256, 8192, value=4096, step=256, label="Max new tokens"
                )
                with gr.Row():
                    local_prompt_temperature = gr.Slider(
                        0.1, 2.0, value=0.7, step=0.1, label="Temperature (sampling)"
                    )
                    local_prompt_top_p = gr.Slider(
                        0.05, 1.0, value=0.8, step=0.05, label="Top-p (sampling)"
                    )
                local_prompt_seed = gr.Number(value=42, precision=0, label="Seed")
        with gr.Group(
            visible=services.DEFAULT_PROMPT_WRITER_BACKEND == "Gemini"
        ) as gemini_prompt_writer_group:
            gr.Markdown(
                "Uses the active inputs with `prompt.txt`. Set `GEMINI_API_KEY` on the server or enter a temporary key; the server does not store UI keys."
            )
            with gr.Row():
                gemini_prompt_model = gr.Dropdown(
                    choices=list(services.GEMINI_PROMPT_MODELS),
                    value=services.DEFAULT_GEMINI_PROMPT_MODEL,
                    label="Gemini model",
                )
                gemini_api_key = gr.Textbox(
                    label="Temporary Gemini API key",
                    type="password",
                    placeholder="Uses GEMINI_API_KEY when blank",
                )
        with gr.Group(
            visible=services.DEFAULT_PROMPT_WRITER_BACKEND == "OpenAI-compatible"
        ) as openai_prompt_writer_group:
            gr.Markdown(
                "Uses the active text and images plus `prompt.txt`. Video and audio references require Gemini. Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` on the server, or enter a temporary key and endpoint; the server does not store UI values."
            )
            with gr.Row():
                openai_model = build_openai_model_dropdown()
                openai_api_key = gr.Textbox(
                    label="Temporary OpenAI API key",
                    type="password",
                    placeholder="Uses OPENAI_API_KEY when blank",
                )
                openai_base_url = build_openai_base_url_textbox()
    with gr.Accordion("Prompt writer / enhancer", open=False):
        gr.Markdown(
            "Drafts a prompt from your idea and media, or enhances the one above. "
            "You review the suggestion before it replaces your prompt.",
            elem_classes=["h3-song-hint"],
        )
        with gr.Row(elem_classes=["h3-settings-links"]):
            enhance_prompt_button = gr.Button("Generate / enhance prompt")
            if writer_link is not None:
                writer_link()
        enhance_prompt_status = gr.Textbox(
            label="Prompt enhancer status", lines=2, interactive=False
        )
        with gr.Group(visible=True):
            prompt_preview = gr.Textbox(
                label="Suggested prompt", lines=5, interactive=False
            )
            prompt_proposal = gr.State(None)
            prompt_previous = gr.State(None)
            with gr.Row():
                accept_prompt = gr.Button("Accept suggested prompt")
                keep_prompt = gr.Button("Keep original")
                undo_prompt = gr.Button("Undo accepted prompt")
    return PromptSection(
        prompt_preview=prompt_preview,
        prompt_proposal=prompt_proposal,
        prompt_previous=prompt_previous,
        accept_prompt=accept_prompt,
        keep_prompt=keep_prompt,
        undo_prompt=undo_prompt,
        enhance_prompt_button=enhance_prompt_button,
        enhance_prompt_status=enhance_prompt_status,
        gemini_api_key=gemini_api_key,
        gemini_prompt_model=gemini_prompt_model,
        gemini_prompt_writer_group=gemini_prompt_writer_group,
        help_text=help_text,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_prompt_writer_group=openai_prompt_writer_group,
        local_prompt_base_model=local_prompt_base_model,
        local_prompt_greedy=local_prompt_greedy,
        local_prompt_max_tokens=local_prompt_max_tokens,
        local_prompt_seed=local_prompt_seed,
        local_prompt_temperature=local_prompt_temperature,
        local_prompt_top_p=local_prompt_top_p,
        local_prompt_writer_group=local_prompt_writer_group,
        prompt=prompt,
        prompt_writer_backend=prompt_writer_backend,
    )
