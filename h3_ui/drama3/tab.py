"""The Soundstage tab: the page in a frame, and the API-only endpoints beside it.

Everything people do happens on the page at ``/soundstage/`` (``page.py``), outside Gradio, as
with Remake and Storyboard; the tab only shows it. The hidden endpoints keep Soundstage's
published API for scripted clients.
"""

from __future__ import annotations

import gradio as gr

from .bindings import bind_soundstage_api

FRAME = ('<iframe class="h3-soundstage-frame" src="/soundstage/" title="Soundstage" '
         'allow="fullscreen; clipboard-write"></iframe>')


def build_soundstage_tab(root, actions, *, gpu=True):
    """Fill ``root`` with the page; with a GPU, also publish the API endpoints."""
    with root:
        gr.HTML(FRAME)
        bind_soundstage_api(actions, gpu=gpu)
