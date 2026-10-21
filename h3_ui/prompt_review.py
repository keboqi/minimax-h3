"""Shared preview, accept, keep and undo for other engines' prompt writers."""

from dataclasses import dataclass
import hashlib
import json
import gradio as gr


@dataclass(frozen=True)
class PromptReview:
    preview: gr.Textbox
    proposal: gr.State
    previous: gr.State
    accept: gr.Button
    keep: gr.Button
    undo: gr.Button
    panel: gr.Group | None = None


def build_prompt_review(*, preview_label="Suggested prompt", compact=False):
    with gr.Group(visible=not compact) as panel:
        preview = gr.Textbox(label=preview_label, lines=5, interactive=False)
        proposal, previous = (gr.State(None), gr.State(None))
        with gr.Row():
            accept = gr.Button("Accept suggestion")
            keep = gr.Button("Keep original")
            if not compact:
                undo = gr.Button("Undo accepted suggestion")
    if compact:
        undo = gr.Button("Undo accepted suggestion", visible=False, size="sm")
    return PromptReview(preview, proposal, previous, accept, keep, undo, panel if compact else None)


def bind_review_action(trigger, callback, options, review, queue):
    compact = review.panel is not None
    panel_outputs = [review.panel] if compact else []
    # A pending proposal hides Undo so it cannot revert past the visible draft.
    toggle_outputs = [review.panel, review.undo] if compact else []
    inputs, outputs = (list(options["inputs"]), list(options["outputs"]))
    drafts, status = (outputs[:-1], outputs[-1])
    draft_indices = [inputs.index(component) for component in drafts]
    context_indices = [
        i
        for i, component in enumerate(inputs)
        if getattr(component, "type", None) != "password"
    ]

    def digest(values):
        return hashlib.sha256(
            json.dumps([values[i] for i in context_indices], default=str).encode()
        ).hexdigest()

    gr.Button(visible=False).click(callback, **options, **queue)

    def preview(*values):
        result = callback(*values)
        suggestions = tuple(result[:-1])
        originals = tuple((values[i] for i in draft_indices))
        if suggestions == originals:
            return ("", None, result[-1], *([gr.update(visible=False), gr.skip()] if compact else []))
        text = "\n\n".join(
            (
                f"{component.label}\n{value}"
                for component, value in zip(drafts, suggestions, strict=True)
            )
        )
        return (
            text,
            (digest(values), originals, suggestions),
            result[-1] + " Preview ready; choose Accept or Keep original.",
            *([gr.update(visible=True), gr.update(visible=False)] if compact else []),
        )

    event = trigger(
        preview,
        inputs=inputs,
        outputs=[review.preview, review.proposal, status, *toggle_outputs],
        show_progress=options.get("show_progress", "minimal"),
        api_name=False,
        **queue,
    )

    def accept(proposal, *values):
        if proposal is None or proposal[0] != digest(values):
            return (
                *(gr.skip() for _ in drafts),
                gr.skip(),
                None,
                "The draft or inputs changed. Enhance the current draft again.",
                *([gr.update(visible=False), gr.skip()] if compact else []),
            )
        return (*proposal[2], proposal[1], None, "Suggested prompt accepted.",
                *([gr.update(visible=False), gr.update(visible=True)] if compact else []))

    review.accept.click(
        accept,
        inputs=[review.proposal, *inputs],
        outputs=[*drafts, review.previous, review.proposal, status, *toggle_outputs],
        queue=False,
        api_name=False,
    )
    review.keep.click(
        lambda: ("", None, "Original prompt kept.", *([gr.update(visible=False)] if compact else [])),
        outputs=[review.preview, review.proposal, status, *panel_outputs],
        queue=False,
        api_name=False,
    )
    review.undo.click(
        lambda previous: (*(previous or [gr.skip()] * len(drafts)), None,
                          *([gr.update(visible=False)] if compact else [])),
        inputs=review.previous,
        outputs=[*drafts, review.previous, *([review.undo] if compact else [])],
        queue=False,
        api_name=False,
    )
    return event
