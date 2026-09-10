"""Build the results section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class ResultsSection:
    audio_output: gr.components.Component
    generation_readiness: gr.components.Component
    image_clear_selection: gr.components.Component
    image_frame_paths: gr.components.Component
    image_output: gr.components.Component
    image_output_group: gr.components.Component
    image_save_selected: gr.components.Component
    image_save_status: gr.components.Component
    image_saved_files: gr.components.Component
    image_select_all: gr.components.Component
    image_selection: gr.components.Component
    output: gr.components.Component
    output_2: gr.components.Component
    output_3: gr.components.Component
    output_4: gr.components.Component
    refresh: gr.components.Component
    run: gr.components.Component
    settings_used: gr.components.Component
    status: gr.components.Component
    stop: gr.components.Component


def build_results_section(
    defaults: Mapping[str, Any], services: H3ViewServices
) -> ResultsSection:
    with gr.Group(elem_classes=["h3-action-dock"]):
        generation_readiness = gr.HTML(
            services.generation_readiness_state(defaults["mode"], "", None, None).html
        )
        with gr.Row():
            run = gr.Button(
                "Generate video",
                variant="primary",
                scale=2,
                interactive=False,
                elem_classes=["h3-primary-action"],
            )
            stop = gr.Button("Interrupt", scale=1)
            refresh = gr.Button("Refresh status", scale=1)
        status = gr.Textbox(
            label="Generation progress",
            lines=2,
            interactive=False,
            elem_classes=["h3-status"],
        )
    gr.HTML(
        '<div class="h3-section-intro"><h3>Results</h3>'
        "<p>Your latest output remains here while you adjust settings.</p></div>"
    )
    with gr.Row():
        output = gr.Video(label="Generated video 1")
        output_2 = gr.Video(label="Generated video 2", visible=False)
    with gr.Row():
        output_3 = gr.Video(label="Generated video 3", visible=False)
        output_4 = gr.Video(label="Generated video 4", visible=False)
    with gr.Group(visible=False) as image_output_group:
        image_output = gr.Gallery(
            value=[],
            label="Generated image frames",
            columns=4,
            object_fit="contain",
            allow_preview=True,
            height=520,
        )
        image_frame_paths = gr.State([])
        image_selection = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Frames to save",
        )
        with gr.Row():
            image_select_all = gr.Button("Select all")
            image_clear_selection = gr.Button("Clear selection")
            image_save_selected = gr.Button("Save selected frames", variant="primary")
        image_saved_files = gr.File(
            label="Saved image files",
            file_count="multiple",
            interactive=False,
        )
        image_save_status = gr.Markdown()
    audio_output = gr.Audio(label="Generated audio", type="filepath", visible=False)
    settings_used = gr.HTML("Settings used will appear with the generated result.")

    return ResultsSection(
        audio_output=audio_output,
        generation_readiness=generation_readiness,
        image_clear_selection=image_clear_selection,
        image_frame_paths=image_frame_paths,
        image_output=image_output,
        image_output_group=image_output_group,
        image_save_selected=image_save_selected,
        image_save_status=image_save_status,
        image_saved_files=image_saved_files,
        image_select_all=image_select_all,
        image_selection=image_selection,
        output=output,
        output_2=output_2,
        output_3=output_3,
        output_4=output_4,
        refresh=refresh,
        run=run,
        settings_used=settings_used,
        status=status,
        stop=stop,
    )
