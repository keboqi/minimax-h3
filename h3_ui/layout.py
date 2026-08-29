"""Top-level Gradio navigation containers.

Views are declared as real tab children up front, then populated by the main
application. This keeps tab ownership explicit without coupling the shell to
the large set of generation callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class AppViews:
    tabs: gr.Tabs
    generation: gr.Row
    ltx25: gr.Group
    music3: gr.Group
    gallery: gr.Group
    api: gr.Group
    gallery_tab: gr.Tab


def create_app_views() -> AppViews:
    """Create the application navigation and empty tab-owned view roots."""

    tabs = gr.Tabs(elem_id="h3-main-tabs")
    with tabs:
        with gr.Tab("MiniMax H3"):
            generation = gr.Row(elem_classes=["h3-generator-shell"])
        with gr.Tab("LTX 2.5"):
            ltx25 = gr.Group()
        with gr.Tab("MiniMax Music 3"):
            music3 = gr.Group()
        with gr.Tab("Gallery") as gallery_tab:
            gallery = gr.Group(elem_classes=["h3-gallery-shell"])
        with gr.Tab("API"):
            api = gr.Group()
    return AppViews(tabs, generation, ltx25, music3, gallery, api, gallery_tab)
