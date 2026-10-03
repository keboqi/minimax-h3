"""Preview/accept prompt proposals without overwriting a changed draft."""

from dataclasses import dataclass
import hashlib
import json

import gradio as gr
from h3_app.reference_bindings import ReferenceMap
from h3_app.errors import H3Error
from .job_bindings import bind_prompt_action


@dataclass(frozen=True)
class PromptProposal:
    fingerprint: str
    original: str
    suggestion: str
    reference_version: int = 0


def fingerprint(names, values):
    # Credentials and provider controls are not retained in proposal metadata.
    context = {
        name: value
        for name, value in zip(names, values, strict=True)
        if name
        in {
            "prompt",
            "mode",
            "first",
            "last",
            "duration",
            "width",
            "height",
            "result_format",
            "image_frames",
        }
        or name.startswith(("ref_", "fl2va_audio_"))
    }
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, default=str).encode()
    ).hexdigest()


def bind_prompt_preview(components, enhance, names, inputs):
    def preview(mapping, *values):
        original = dict(zip(names, values, strict=True))
        current = dict(original)
        try:
            mapping = (mapping or ReferenceMap()).update(current)
            if current["mode"] == "Reference media":
                current = mapping.dense_values(current)
            suggestion, status = enhance(*(current[name] for name in names))
            if original["mode"] == "Reference media":
                suggestion = mapping.restore_prompt(suggestion)
        except (H3Error, ValueError) as exc:
            return "", None, str(exc)
        if suggestion == original["prompt"]:
            return "", None, status
        proposal = PromptProposal(
            fingerprint(names, values), original["prompt"], suggestion, mapping.version
        )
        return (
            suggestion,
            proposal,
            status + " Preview ready; choose Accept or Keep original.",
        )

    bind_prompt_action(
        components.enhance_prompt_button.click,
        preview,
        inputs=[components.reference_map, *inputs],
        outputs=[
            components.prompt_preview,
            components.prompt_proposal,
            components.enhance_prompt_status,
        ],
        api_name=False,
        show_progress="minimal",
    )

    def accept(proposal, mapping, *values):
        if proposal is None:
            return gr.skip(), gr.skip(), "No suggested prompt to accept.", gr.skip()
        current = dict(zip(names, values, strict=True))
        mapping = (mapping or ReferenceMap()).update(current)
        if (
            fingerprint(names, values) != proposal.fingerprint
            or mapping.version != proposal.reference_version
        ):
            return (
                gr.skip(),
                gr.skip(),
                "The prompt or input media changed. Enhance the current draft again.",
                None,
            )
        return (
            proposal.suggestion,
            proposal.original,
            "Suggested prompt accepted.",
            None,
        )

    components.accept_prompt.click(
        accept,
        inputs=[components.prompt_proposal, components.reference_map, *inputs],
        outputs=[
            components.prompt,
            components.prompt_previous,
            components.enhance_prompt_status,
            components.prompt_proposal,
        ],
        queue=False,
        api_name=False,
        show_progress="hidden",
    )
    components.keep_prompt.click(
        lambda: ("", None, "Original prompt kept."),
        outputs=[
            components.prompt_preview,
            components.prompt_proposal,
            components.enhance_prompt_status,
        ],
        queue=False,
        api_name=False,
    )
    components.undo_prompt.click(
        lambda previous: (previous if previous is not None else gr.skip(), None),
        inputs=components.prompt_previous,
        outputs=[components.prompt, components.prompt_previous],
        queue=False,
        api_name=False,
    )
    components.insert_reference_tag.click(
        lambda prompt, tag: (
            prompt + (" " if prompt and not prompt[-1].isspace() else "") + tag
            if tag
            else gr.skip()
        ),
        inputs=[components.prompt, components.reference_tag],
        outputs=components.prompt,
        queue=False,
        api_name=False,
    )
