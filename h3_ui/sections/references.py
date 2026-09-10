"""Build the references section in its existing parent container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

if TYPE_CHECKING:
    from ..h3_view import H3ViewServices


@dataclass(frozen=True)
class ReferencesSection:
    first: gr.components.Component
    fl2va_audio_1: gr.components.Component
    fl2va_audio_2: gr.components.Component
    fl2va_audio_3: gr.components.Component
    frame_group: gr.components.Component
    last: gr.components.Component
    ref_audio_1: gr.components.Component
    ref_audio_2: gr.components.Component
    ref_audio_3: gr.components.Component
    ref_image_1: gr.components.Component
    ref_image_2: gr.components.Component
    ref_image_3: gr.components.Component
    ref_image_4: gr.components.Component
    ref_image_5: gr.components.Component
    ref_image_6: gr.components.Component
    ref_image_7: gr.components.Component
    ref_image_8: gr.components.Component
    ref_image_9: gr.components.Component
    ref_size: gr.components.Component
    ref_video_1: gr.components.Component
    ref_video_2: gr.components.Component
    ref_video_3: gr.components.Component
    reference_group: gr.components.Component


def build_references_section(
    defaults: Mapping[str, Any], services: H3ViewServices
) -> ReferencesSection:
    with gr.Group(visible=False) as frame_group:
        gr.Markdown("### First / last frame inputs")
        with gr.Row():
            first = gr.Image(
                type="filepath",
                label="First frame (auto resolution)",
                elem_id="first-frame-image",
            )
            last = gr.Image(type="filepath", label="Last frame")
        with gr.Accordion("Optional voice references (experimental)", open=False):
            gr.Markdown(
                "Use short, clean voice samples for new dialogue with these frames. "
                "Start with 2–3 seconds per voice. Assign speakers in the prompt, "
                'for example: `The woman uses the voice timbre of <Audio 1> and says, "Hello."` '
                "Fill slots in order. These uploads apply only to First / last frame mode; "
                "Ref2VA has separate uploads. Voice fidelity is experimental."
            )
            with gr.Row():
                fl2va_audio_1 = gr.Audio(
                    type="filepath",
                    label="FL2VA voice 1 · <Audio 1>",
                    elem_id="fl2va-voice-1",
                )
                fl2va_audio_2 = gr.Audio(
                    type="filepath",
                    label="FL2VA voice 2 · <Audio 2>",
                    elem_id="fl2va-voice-2",
                )
                fl2va_audio_3 = gr.Audio(
                    type="filepath",
                    label="FL2VA voice 3 · <Audio 3>",
                    elem_id="fl2va-voice-3",
                )
    with gr.Group(visible=False) as reference_group:
        gr.Markdown("### Reference media")
        gr.Markdown(services.reference_prompt_help())
        with gr.Accordion("Reference images · up to 9", open=True):
            with gr.Row():
                ref_image_1 = gr.Image(type="filepath", label="Picture 1")
                ref_image_2 = gr.Image(type="filepath", label="Picture 2")
                ref_image_3 = gr.Image(type="filepath", label="Picture 3")
            with gr.Row():
                ref_image_4 = gr.Image(type="filepath", label="Picture 4")
                ref_image_5 = gr.Image(type="filepath", label="Picture 5")
                ref_image_6 = gr.Image(type="filepath", label="Picture 6")
            with gr.Row():
                ref_image_7 = gr.Image(type="filepath", label="Picture 7")
                ref_image_8 = gr.Image(type="filepath", label="Picture 8")
                ref_image_9 = gr.Image(type="filepath", label="Picture 9")
        with gr.Accordion("Reference videos · up to 3", open=False):
            with gr.Row():
                ref_video_1 = gr.Video(label="Video 1")
                ref_video_2 = gr.Video(label="Video 2")
                ref_video_3 = gr.Video(label="Video 3")
        with gr.Accordion("Reference audio · up to 3", open=False):
            with gr.Row():
                ref_audio_1 = gr.Audio(type="filepath", label="Audio 1")
                ref_audio_2 = gr.Audio(type="filepath", label="Audio 2")
                ref_audio_3 = gr.Audio(type="filepath", label="Audio 3")
        ref_size = gr.Radio(
            ["match", "max"],
            value=defaults["ref_image_size"],
            label="Reference image size",
        )

    return ReferencesSection(
        first=first,
        fl2va_audio_1=fl2va_audio_1,
        fl2va_audio_2=fl2va_audio_2,
        fl2va_audio_3=fl2va_audio_3,
        frame_group=frame_group,
        last=last,
        ref_audio_1=ref_audio_1,
        ref_audio_2=ref_audio_2,
        ref_audio_3=ref_audio_3,
        ref_image_1=ref_image_1,
        ref_image_2=ref_image_2,
        ref_image_3=ref_image_3,
        ref_image_4=ref_image_4,
        ref_image_5=ref_image_5,
        ref_image_6=ref_image_6,
        ref_image_7=ref_image_7,
        ref_image_8=ref_image_8,
        ref_image_9=ref_image_9,
        ref_size=ref_size,
        ref_video_1=ref_video_1,
        ref_video_2=ref_video_2,
        ref_video_3=ref_video_3,
        reference_group=reference_group,
    )
