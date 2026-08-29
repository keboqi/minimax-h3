"""Typed LTX-2.5 view builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


@dataclass(frozen=True)
class LtxView:
    model: gr.Dropdown
    mode: gr.Radio
    prompt: gr.Textbox
    prompt_model: gr.Dropdown
    api_key: gr.Textbox
    enhance: gr.Button
    enhance_status: gr.Textbox
    negative: gr.Textbox
    image_group: gr.Group
    image: gr.Image
    image_strength: gr.Slider
    middle_image: gr.Image
    middle_time: gr.Slider
    middle_strength: gr.Slider
    end_image: gr.Image
    end_strength: gr.Slider
    output: gr.Video
    run: gr.Button
    stop: gr.Button
    status: gr.Textbox
    duration: gr.Slider
    fps: gr.Slider
    width: gr.Number
    height: gr.Number
    seed: gr.Number
    cfg: gr.Slider
    sampler: gr.Dropdown
    workflow: gr.Dropdown
    prepare_workflow: gr.Button
    prepare_all_models: gr.Button
    refresh_models: gr.Button
    workflow_details: gr.Markdown
    workflow_status: gr.Markdown
    model_inventory: gr.Markdown


def build_ltx_view(
    root: gr.Group,
    *,
    model_choices: Sequence[str],
    defaults: Mapping[str, Any],
    prompt_models: Sequence[str],
    default_prompt_model: str,
    workflows: Sequence[str],
    initial_workflow_details: str,
    model_inventory_text: str,
) -> LtxView:
    first_workflow = workflows[0]
    with root:
        gr.Markdown(
            "## LTX-2.5 audio-video generation\n"
            "Official single-stage distilled workflow on the shared ComfyUI backend. "
            "The gated model assets download on first use; accept the "
            "[LTX-2.5 model license](https://huggingface.co/Lightricks/LTX-2.5) "
            "before generating."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                model = gr.Dropdown(
                    choices=list(model_choices),
                    value=defaults["model"],
                    label="Transformer model",
                    info="INT8 ConvRot is the default lower-memory option.",
                )
                mode = gr.Radio(
                    ["Text to video", "Image to video"],
                    value=defaults["mode"],
                    label="Mode",
                )
                prompt = gr.Textbox(
                    label="Positive prompt",
                    lines=10,
                    placeholder=(
                        "Describe the action chronologically, then the setting, "
                        "camera movement, lighting, dialogue, sound effects, and music."
                    ),
                )
                with gr.Accordion("Gemini LTX-2.5 prompt writer", open=False):
                    gr.Markdown("Create or enhance the prompt from text and keyframes.")
                    with gr.Row():
                        prompt_model = gr.Dropdown(
                            choices=list(prompt_models),
                            value=default_prompt_model,
                            label="Gemini model",
                        )
                        api_key = gr.Textbox(
                            label="Temporary Gemini API key",
                            type="password",
                            placeholder="Uses GEMINI_API_KEY when blank",
                        )
                    enhance = gr.Button("Generate / enhance LTX-2.5 prompt")
                    enhance_status = gr.Textbox(
                        label="Prompt writer status", lines=2, interactive=False
                    )
                negative = gr.Textbox(
                    label="Negative prompt",
                    lines=3,
                    placeholder="Optional artifacts or qualities to avoid",
                )
                with gr.Group(visible=False) as image_group:
                    gr.Markdown("Choose a required start frame and optional keyframes.")
                    with gr.Row():
                        image = gr.Image(
                            type="filepath", label="Start keyframe (required)"
                        )
                        image_strength = gr.Slider(
                            0.0,
                            1.0,
                            value=defaults["image_strength"],
                            step=0.05,
                            label="Start strength",
                        )
                    with gr.Accordion("Optional middle and end keyframes", open=False):
                        with gr.Row():
                            middle_image = gr.Image(
                                type="filepath", label="Middle keyframe"
                            )
                            with gr.Column():
                                middle_time = gr.Slider(
                                    0.1,
                                    19.9,
                                    value=defaults["middle_time"],
                                    step=0.1,
                                    label="Middle position (seconds)",
                                )
                                middle_strength = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=defaults["middle_strength"],
                                    step=0.05,
                                    label="Middle strength",
                                )
                        with gr.Row():
                            end_image = gr.Image(type="filepath", label="End keyframe")
                            end_strength = gr.Slider(
                                0.0,
                                1.0,
                                value=defaults["end_strength"],
                                step=0.05,
                                label="End strength",
                            )
            with gr.Column(scale=2):
                output = gr.Video(label="Generated LTX-2.5 video")
                with gr.Row():
                    run = gr.Button("Generate with LTX-2.5", variant="primary")
                    stop = gr.Button("Interrupt")
                status = gr.Textbox(label="Status", lines=6)
                gr.Markdown("### Generation settings")
                with gr.Row():
                    duration = gr.Slider(
                        1, 20, value=defaults["duration"], step=0.5, label="Seconds"
                    )
                    fps = gr.Slider(1, 60, value=defaults["fps"], step=1, label="FPS")
                with gr.Row():
                    width = gr.Number(
                        value=defaults["width"], precision=0, label="Width"
                    )
                    height = gr.Number(
                        value=defaults["height"], precision=0, label="Height"
                    )
                gr.Markdown(
                    "Dimensions snap to multiples of 32; frames snap to `8n + 1`."
                )
                with gr.Row():
                    seed = gr.Number(
                        value=defaults["seed"], precision=0, label="Seed (-1 random)"
                    )
                    cfg = gr.Slider(
                        0.0, 3.0, value=defaults["cfg"], step=0.05, label="CFG"
                    )
                sampler = gr.Dropdown(
                    ["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_sde"],
                    value=defaults["sampler"],
                    label="Sampler",
                )
        with gr.Accordion("Official workflows and model downloads", open=True):
            gr.Markdown(
                "Download missing workflow models here first. Individual Hugging Face "
                "repositories can require separate license acceptance."
            )
            with gr.Row():
                workflow = gr.Dropdown(
                    choices=list(workflows),
                    value=first_workflow,
                    label="Official workflow",
                    scale=3,
                )
                prepare_workflow = gr.Button(
                    "Download selected workflow models", variant="primary", scale=1
                )
            with gr.Row():
                prepare_all_models = gr.Button(
                    "Download all missing models", variant="secondary"
                )
                refresh_models = gr.Button(
                    "Refresh model availability", variant="secondary"
                )
            workflow_details = gr.Markdown(initial_workflow_details)
            workflow_status = gr.Markdown()
            model_inventory = gr.Markdown(model_inventory_text)
    return LtxView(
        model,
        mode,
        prompt,
        prompt_model,
        api_key,
        enhance,
        enhance_status,
        negative,
        image_group,
        image,
        image_strength,
        middle_image,
        middle_time,
        middle_strength,
        end_image,
        end_strength,
        output,
        run,
        stop,
        status,
        duration,
        fps,
        width,
        height,
        seed,
        cfg,
        sampler,
        workflow,
        prepare_workflow,
        prepare_all_models,
        refresh_models,
        workflow_details,
        workflow_status,
        model_inventory,
    )
