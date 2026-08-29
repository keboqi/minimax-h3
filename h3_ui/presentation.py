"""Pure presentation state used by the Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable


@dataclass(frozen=True)
class GenerationReadiness:
    """A display message and whether the primary action can be used."""

    html: str
    ready: bool


@dataclass(frozen=True)
class ModePresentation:
    show_frames: bool
    show_references: bool


@dataclass(frozen=True)
class ResultFormatPresentation:
    format: str
    is_video: bool
    is_image: bool
    is_audio: bool
    action_label: str


def mode_presentation(mode: str) -> ModePresentation:
    return ModePresentation(
        show_frames=mode == "First / last frame",
        show_references=mode == "Reference media",
    )


def result_format_presentation(result_format: str) -> ResultFormatPresentation:
    normalized = str(result_format or "Video").strip().title()
    if normalized not in {"Video", "Image", "Audio"}:
        normalized = "Video"
    return ResultFormatPresentation(
        format=normalized,
        is_video=normalized == "Video",
        is_image=normalized == "Image",
        is_audio=normalized == "Audio",
        action_label=f"Generate {normalized.lower()}",
    )


def backend_status_html(detail: str) -> str:
    """Render a calm summary while detailed diagnostics remain available."""

    detail = str(detail)
    if detail.startswith("Connected"):
        headline = escape(detail.split("  \n", 1)[0])
        return (
            '<div class="h3-system-ready" role="status">'
            f"<strong>Ready</strong><span>{headline}</span></div>"
        )
    return (
        '<div class="h3-system-warning" role="alert">'
        "<strong>Backend needs attention</strong>"
        f"<span>{escape(detail)}</span></div>"
    )


def generation_readiness(
    mode: str,
    prompt: str,
    first_image: Any,
    last_image: Any,
    reference_media: Iterable[Any] = (),
) -> GenerationReadiness:
    """Derive primary-action readiness from the active generation inputs."""

    missing: list[str] = []
    if not str(prompt or "").strip():
        missing.append("write a prompt")
    if mode == "First / last frame" and not (first_image or last_image):
        missing.append("upload a first or last frame")
    if mode == "Reference media" and not any(reference_media):
        missing.append("add at least one reference")

    if missing:
        message = " · ".join(missing)
        return GenerationReadiness(
            '<div class="h3-system-warning" role="alert">'
            f"<strong>Before generating</strong><span>{escape(message)}.</span></div>",
            False,
        )

    return GenerationReadiness(
        '<div class="h3-system-ready" role="status">'
        "<strong>Ready to generate</strong>"
        "<span>Review the setup summary, then start the job.</span></div>",
        True,
    )
