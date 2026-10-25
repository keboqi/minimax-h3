"""Guided-step building blocks shared by Song and Dub & Translate: intro, card headings, step strip.

H3's Fine-tune tabs reuse the card heading and the strip's tab-opening buttons.

The CSS classes keep their original ``h3-song-*`` names, which the stylesheet targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import gradio as gr


def intro(eyebrow: str, title: str, detail: str, badge: str) -> gr.HTML:
    return gr.HTML(
        f'<div class="h3-song-intro"><div><p class="h3-eyebrow">{eyebrow}</p>'
        f"<h2>{escape(title)}</h2><p>{detail}</p></div>"
        f'<span class="h3-song-badge">{escape(badge)}</span></div>'
    )


def card_heading(title: str, detail: str = "") -> gr.HTML:
    return gr.HTML(
        f'<div class="h3-song-card-heading"><h3>{escape(title)}</h3>'
        + (f"<p>{escape(detail)}</p>" if detail else "")
        + "</div>"
    )


@dataclass(frozen=True)
class StepStrip:
    """Guided steps shown as one strip (Song: Idea → Lyrics → Generate → Listen; Dub: Source → Lines → Dub).

    Every step stays mounted (``render_children``) so a button on one step can
    press Write or Generate on another, and take audio loads before it is shown.
    Moving between steps happens in the browser; no request is sent.
    """

    tabs: gr.Tabs

    def step(self, index: int, label: str) -> gr.Tab:
        return gr.Tab(label, id=index, render_children=True, elem_classes=["h3-song-step"])

    def go(self, index: int, then: str = "", *, reveal: bool = False) -> str:
        """Browser code that opens step ``index``, then runs ``then``.

        The strip scrolls into view when its top has scrolled away, or always with ``reveal``.
        """
        scroll = "true" if reveal else "root.getBoundingClientRect().top < 0"
        return (
            "() => { "
            f"const root = document.getElementById({self.tabs.elem_id!r}); "
            "const tab = root && root.querySelectorAll(':scope > .tab-wrapper [role=\"tab\"]')"
            f"[{int(index)}]; "
            f"if (tab) {{ tab.click(); if ({scroll}) "
            "root.scrollIntoView({ block: 'start', behavior: 'smooth' }); } "
            f"{then}}}"
        )

    def follow(self, trigger: gr.Button, index: int) -> None:
        """Open step ``index`` whenever ``trigger`` is pressed."""
        trigger.click(None, js=self.go(index), queue=False, api_name=False)

    def button(self, label: str, index: int, *, variant: str = "secondary",
               press: str = "", classes: tuple[str, ...] = (), reveal: bool = False) -> gr.Button:
        """A button that opens step ``index`` and, with ``press``, clicks that element.

        The press waits two frames: the first time a step opens, its buttons are wired
        only after the tab has rendered, and an earlier click is lost.
        """
        then = (f"requestAnimationFrame(() => requestAnimationFrame(() => "
                f"document.getElementById({press!r})?.click())); ") if press else ""
        button = gr.Button(label, variant=variant, elem_classes=["h3-song-nav", *classes])
        button.click(None, js=self.go(index, then, reveal=reveal), queue=False, api_name=False)
        return button


def step_strip(elem_id: str, selected: int = 0) -> StepStrip:
    # Wrap keeps every step in the strip; CSS lays them out as a stepper.
    return StepStrip(gr.Tabs(selected=selected, elem_id=elem_id, elem_classes=["h3-song-steps"],
                             overflow_behavior="wrap"))
