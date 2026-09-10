"""Build the prompt section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class PromptSection:
    enhance_prompt_button: gr.components.Component
    enhance_prompt_status: gr.components.Component
    gemini_api_key: gr.components.Component
    gemini_prompt_model: gr.components.Component
    gemini_prompt_writer_group: gr.components.Component
    help_text: gr.components.Component
    lightning_api_key: gr.components.Component
    lightning_prompt_writer_group: gr.components.Component
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
    defaults: Mapping[str, Any], services: H3ViewServices
) -> PromptSection:
    help_text = gr.Markdown(services.mode_help("Text to video"))
    prompt = gr.Textbox(
        label="Prompt",
        lines=12,
        placeholder="Describe shots, camera motion, dialogue, sound effects, ambience, music, and any tagged references.",
    )
    with gr.Accordion("Prompt writer / enhancer", open=False):
        gr.Markdown(
            "The local MiniMax-H3 8B writer supports T2VA, I2VA, "
            "L2VA, and FL2VA. Gemini supports all Reference media; "
            "Lightning AI supports text and image enhancement."
        )
        prompt_writer_backend = gr.Radio(
            services.PROMPT_WRITER_BACKENDS,
            value=services.DEFAULT_PROMPT_WRITER_BACKEND,
            label="Prompt writer",
        )
        with gr.Group(visible=False) as local_prompt_writer_group:
            local_prompt_base_model = gr.Dropdown(
                choices=list(services.LOCAL_PROMPT_BASE_MODELS),
                value=services.DEFAULT_LOCAL_PROMPT_BASE_MODEL,
                label="Local base model",
                info=(
                    "BF16 is the default full-precision checkpoint. FP8 is "
                    "available as the lower-memory alternative."
                ),
            )
            with gr.Accordion("Local decoding settings", open=False):
                local_prompt_greedy = gr.Checkbox(value=True, label="Greedy decoding")
                local_prompt_max_tokens = gr.Slider(
                    256,
                    8192,
                    value=4096,
                    step=256,
                    label="Max new tokens",
                )
                with gr.Row():
                    local_prompt_temperature = gr.Slider(
                        0.1,
                        2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (sampling)",
                    )
                    local_prompt_top_p = gr.Slider(
                        0.05,
                        1.0,
                        value=0.8,
                        step=0.05,
                        label="Top-p (sampling)",
                    )
                local_prompt_seed = gr.Number(value=42, precision=0, label="Seed")
        with gr.Group(visible=True) as gemini_prompt_writer_group:
            gr.Markdown(
                "Uses the active inputs with `prompt.txt`. Set "
                "`GEMINI_API_KEY` on the server or enter a temporary key; "
                "the server does not store UI keys."
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
        with gr.Group(visible=False) as lightning_prompt_writer_group:
            gr.Markdown(
                f"Uses `{services.LIGHTNING_PROMPT_MODEL}` with the active text "
                "and images plus `prompt.txt`. Video and audio references "
                "require Gemini. Set `LIGHTNING_API_KEY` on the server or "
                "enter a temporary key; the server does not store UI keys."
            )
            lightning_api_key = gr.Textbox(
                label="Temporary Lightning API key",
                type="password",
                placeholder="Uses LIGHTNING_API_KEY when blank",
            )
        enhance_prompt_button = gr.Button("Generate / enhance prompt")
        enhance_prompt_status = gr.Textbox(
            label="Prompt enhancer status", lines=2, interactive=False
        )

    return PromptSection(
        enhance_prompt_button=enhance_prompt_button,
        enhance_prompt_status=enhance_prompt_status,
        gemini_api_key=gemini_api_key,
        gemini_prompt_model=gemini_prompt_model,
        gemini_prompt_writer_group=gemini_prompt_writer_group,
        help_text=help_text,
        lightning_api_key=lightning_api_key,
        lightning_prompt_writer_group=lightning_prompt_writer_group,
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
