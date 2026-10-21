"""Shared building blocks for the guided Song views (Music 3 and YuE2)."""

from __future__ import annotations

from html import escape

import gradio as gr


def song_intro(eyebrow: str, title: str, detail: str, badge: str) -> gr.HTML:
    return gr.HTML(
        f'<div class="h3-song-intro"><div><p class="h3-eyebrow">{eyebrow}</p>'
        f"<h2>{escape(title)}</h2><p>{detail}</p></div>"
        f'<span class="h3-song-badge">{escape(badge)}</span></div>'
    )


def panel_heading(step: str, title: str, detail: str, *, classes: str = "") -> gr.HTML:
    names = " ".join(("h3-panel-heading", classes)).strip()
    return gr.HTML(
        f'<div class="{names}"><span class="h3-step" aria-hidden="true">'
        f"{escape(step)}</span><div><h3>{escape(title)}</h3><p>{escape(detail)}</p></div></div>"
    )


def duration_presets(slider: gr.Slider, presets) -> None:
    """One-click maximum-length buttons; each sets the slider in the browser."""
    with gr.Row(elem_classes=["h3-song-duration-presets"]):
        for seconds, label in presets:
            gr.Button(label, size="sm", min_width=50).click(
                None, outputs=slider, js=f"() => {int(seconds)}", queue=False, api_name=False,
            )
