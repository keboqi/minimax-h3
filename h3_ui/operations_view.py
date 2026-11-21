"""Shared Gradio presentation for the System and Jobs tabs."""

from html import escape

import gradio as gr


def page_heading(title: str, description: str, icon_path: str):
    return gr.HTML(
        '<div class="h3-ops-title">'
        '<span class="h3-ops-icon"><svg width="24" height="24" viewBox="0 0 24 24"'
        ' fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"'
        ' stroke-linejoin="round" aria-hidden="true">'
        f'<path d="{escape(icon_path, quote=True)}"></path></svg></span>'
        f'<div><h2>{escape(title)}</h2><p>{escape(description)}</p></div></div>',
        elem_classes=["h3-ops-heading-copy"],
        min_width=0,
    )
