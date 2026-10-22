"""Session take history for the Song views: listen, compare A/B, reuse.

Takes live in the browser session only (like an unsaved draft); every file is
also in Media. Each take remembers the draft that produced it, so a listener
can go back to the version they liked without knowing what changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

import gradio as gr

MAX_TAKES = 20
# Draft fields worth naming when two takes differ.
_FIELD_NAMES = {
    "caption": "description", "style": "style", "lyrics": "lyrics", "abc": "score",
    "mode": "composition", "duration": "length limit", "vocals": "voice",
    "language": "language", "use": "purpose",
}


@dataclass(frozen=True)
class SongTakes:
    history: gr.State
    pending: gr.State
    panel: gr.Accordion
    take_a: gr.Dropdown
    take_b: gr.Dropdown
    audio_a: gr.Audio
    audio_b: gr.Audio
    details: gr.HTML
    reuse_draft: gr.Button
    use_seed: gr.Button
    reuse_score: gr.Button | None


def build_take_history(*, with_score: bool) -> SongTakes:
    history, pending = gr.State([]), gr.State(None)
    with gr.Accordion("Your takes · compare and reuse", open=False, elem_classes=["h3-song-takes"]) as panel:
        gr.Markdown(
            "Every take from this session, with the draft that made it. Play A and B back to back "
            "and keep the one you like.",
            elem_classes=["h3-song-takes-help"],
        )
        with gr.Row(equal_height=False):
            with gr.Column(min_width=200):
                take_a = gr.Dropdown([], label="Take A", interactive=True)
                audio_a = gr.Audio(label="Take A", type="filepath", interactive=False, show_label=False)
            with gr.Column(min_width=200):
                take_b = gr.Dropdown([], label="Take B", interactive=True)
                audio_b = gr.Audio(label="Take B", type="filepath", interactive=False, show_label=False)
        details = gr.HTML(_describe([], None, None))
        with gr.Row(elem_classes=["h3-song-take-actions"]):
            reuse_draft = gr.Button("Back to take A's draft", size="sm")
            use_seed = gr.Button("Use take A's seed", size="sm")
            reuse_score = gr.Button("Reuse take A's score", size="sm") if with_score else None
    return SongTakes(history, pending, panel, take_a, take_b, audio_a, audio_b, details,
                     reuse_draft, use_seed, reuse_score)


def bind_take_history(
    takes: SongTakes,
    *,
    run: gr.Button,
    output: gr.Audio,
    drafts: Mapping[str, Any],
    seed: gr.Number,
    take_facts: Callable[[Any], Mapping[str, Any]],
    score: gr.Textbox | None = None,
    mode: gr.Dropdown | None = None,
) -> None:
    names, components = list(drafts), list(drafts.values())

    # The draft is captured when Generate is pressed, so edits made while the
    # take renders do not get attributed to it.
    run.click(
        lambda *values: dict(zip(names, values)),
        inputs=components, outputs=takes.pending, queue=False, api_name=False, show_progress="hidden",
    )

    def add(value, history, pending):
        path = value.get("path") if isinstance(value, dict) else value
        if not path or any(take["path"] == str(path) for take in history):
            return (gr.skip(),) * 7
        facts = dict(take_facts(path) or {})
        number = (history[-1]["number"] + 1) if history else 1
        take = {"number": number, "path": str(path), "draft": dict(pending or {}), **facts}
        history = [*history, take][-MAX_TAKES:]
        a, b = len(history) - 1, (len(history) - 2 if len(history) > 1 else None)
        return (
            history,
            gr.update(choices=_choices(history), value=str(a)),
            gr.update(choices=_choices(history), value=None if b is None else str(b)),
            history[a]["path"],
            None if b is None else history[b]["path"],
            _describe(history, a, b),
            gr.update(open=True, label=f"Your takes · {len(history)} · compare and reuse"),
        )

    output.change(
        add, inputs=[output, takes.history, takes.pending],
        outputs=[takes.history, takes.take_a, takes.take_b, takes.audio_a, takes.audio_b,
                 takes.details, takes.panel],
        queue=False, api_name=False, show_progress="hidden",
    )

    def select(history, a, b):
        a, b = _index(history, a), _index(history, b)
        return (
            None if a is None else history[a]["path"],
            None if b is None else history[b]["path"],
            _describe(history, a, b),
        )

    gr.on(
        triggers=[takes.take_a.input, takes.take_b.input], fn=select,
        inputs=[takes.history, takes.take_a, takes.take_b],
        outputs=[takes.audio_a, takes.audio_b, takes.details],
        queue=False, api_name=False, show_progress="hidden",
    )

    def reuse(history, a):
        index = _index(history, a)
        draft = history[index]["draft"] if index is not None else {}
        return tuple(draft[name] if name in draft else gr.skip() for name in names)

    takes.reuse_draft.click(
        reuse, inputs=[takes.history, takes.take_a], outputs=components,
        queue=False, api_name=False,
    )

    def take_seed(history, a):
        index = _index(history, a)
        value = history[index].get("seed") if index is not None else None
        return gr.skip() if value is None else int(value)

    takes.use_seed.click(
        take_seed, inputs=[takes.history, takes.take_a], outputs=seed, queue=False, api_name=False,
    )
    if takes.reuse_score is not None and score is not None and mode is not None:

        def take_score(history, a, current_mode):
            index = _index(history, a)
            value = history[index].get("score") if index is not None else ""
            if not value:
                return gr.skip(), gr.skip()
            # A supplied score needs a planning mode; keep the user's choice otherwise.
            return value, ("full" if current_mode == "off" else gr.skip())

        takes.reuse_score.click(
            take_score, inputs=[takes.history, takes.take_a, mode], outputs=[score, mode],
            queue=False, api_name=False,
        )


def _index(history, value) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < len(history) else None


def _label(take) -> str:
    parts = [f"Take {take['number']}"]
    if take.get("seconds"):
        seconds = round(take["seconds"])
        parts.append(f"{seconds // 60}:{seconds % 60:02d}")
    if take.get("seed") is not None:
        parts.append(f"seed {take['seed']}")
    return " · ".join(parts)


def _choices(history) -> list[tuple[str, str]]:
    return [(_label(take), str(index)) for index, take in reversed(list(enumerate(history)))]


def _describe(history, a, b) -> str:
    if not history or a is None:
        return '<p class="h3-song-take-details">Takes appear here after you generate.</p>'
    take = history[a]
    draft = take.get("draft") or {}
    first = next(
        (str(draft[key]).strip().splitlines()[0] for key in ("caption", "style") if str(draft.get(key) or "").strip()),
        "",
    )
    lines = [f"<strong>{escape(_label(take))}</strong>"]
    if first:
        lines.append(escape(first if len(first) <= 120 else first[:119] + "…"))
    if b is not None and b != a:
        other = history[b].get("draft") or {}
        changed = [
            label for key, label in _FIELD_NAMES.items()
            if key in draft and key in other and str(draft[key]).strip() != str(other[key]).strip()
        ]
        if history[a].get("seed") != history[b].get("seed"):
            changed.append("seed")
        lines.append(
            "Compared with " + escape(_label(history[b])) + ": "
            + (escape(", ".join(changed)) + " changed." if changed else "same draft, same seed.")
        )
    return '<p class="h3-song-take-details">' + "<br>".join(lines) + "</p>"
