"""Shared preview, accept, keep and undo for other engines' prompt writers."""

from dataclasses import dataclass
import hashlib
import json
import gradio as gr
from .workspace_mode import workspace_enabled


@dataclass(frozen=True)
class PromptReview:
    preview: gr.Textbox
    proposal: gr.State
    previous: gr.State
    accept: gr.Button
    keep: gr.Button
    undo: gr.Button


def build_prompt_review():
    with gr.Group(visible=workspace_enabled()):
        preview = gr.Textbox(label="Suggested prompt", lines=5, interactive=False)
        proposal, previous = gr.State(None), gr.State(None)
        with gr.Row():
            accept = gr.Button("Accept suggestion")
            keep = gr.Button("Keep original")
            undo = gr.Button("Undo accepted suggestion")
    return PromptReview(preview, proposal, previous, accept, keep, undo)


def bind_review_action(trigger, callback, options, review, queue):
    inputs, outputs = list(options["inputs"]), list(options["outputs"])
    drafts, status = outputs[:-1], outputs[-1]
    draft_indices = [inputs.index(component) for component in drafts]
    # Passwords never enter retained proposal or undo state.
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
        originals = tuple(values[i] for i in draft_indices)
        if suggestions == originals:
            return "", None, result[-1]
        text = "\n\n".join(
            f"{component.label}\n{value}"
            for component, value in zip(drafts, suggestions, strict=True)
        )
        return (
            text,
            (digest(values), originals, suggestions),
            result[-1] + " Preview ready; choose Accept or Keep original.",
        )

    event = trigger(
        preview,
        inputs=inputs,
        outputs=[review.preview, review.proposal, status],
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
            )
        return *proposal[2], proposal[1], None, "Suggested prompt accepted."

    review.accept.click(
        accept,
        inputs=[review.proposal, *inputs],
        outputs=[*drafts, review.previous, review.proposal, status],
        queue=False,
        api_name=False,
    )
    review.keep.click(
        lambda: ("", None, "Original prompt kept."),
        outputs=[review.preview, review.proposal, status],
        queue=False,
        api_name=False,
    )
    review.undo.click(
        lambda previous: (*(previous or [gr.skip()] * len(drafts)), None),
        inputs=review.previous,
        outputs=[*drafts, review.previous],
        queue=False,
        api_name=False,
    )
    return event
