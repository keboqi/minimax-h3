"""Shared prompt-writer plumbing: uploaded media, system prompts and the Gemini/Lightning request."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import requests

from h3_app.catalog import GEMINI_PROMPT_MODELS, DEFAULT_LIGHTNING_PROMPT_MODEL
from h3_app.errors import H3Error
from h3_app.llm import gemini, lightning, strip_code_fence


def uploaded_media_path(value: Any) -> Path | None:
    """Resolve filepath values returned by Gradio media components."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
    elif isinstance(value, dict):
        raw = value.get("path") or value.get("name")
        candidate = Path(raw) if raw else None
    elif isinstance(value, (tuple, list)) and value:
        # Older Gradio Video versions may return (video_path, subtitles_path).
        return uploaded_media_path(value[0])
    else:
        raw = getattr(value, "path", None) or getattr(value, "name", None)
        candidate = Path(raw) if raw else None
    if candidate is None or not candidate.is_file():
        raise H3Error(f"Prompt-enhancer media file does not exist: {candidate}")
    return candidate


def active_prompt_media(
    mode: str,
    first_image: Any,
    last_image: Any,
    reference_images: Iterable[Any],
    reference_videos: Iterable[Any],
    reference_audios: Iterable[Any],
) -> list[tuple[str, Path]]:
    media: list[tuple[str, Path]] = []
    if mode == "First / last frame":
        frame_values = []
        if first_image is not None:
            frame_values.append(("<Picture 1> (first frame)", first_image))
        if last_image is not None:
            picture_number = 2 if first_image is not None else 1
            frame_values.append(
                (f"<Picture {picture_number}> (last frame)", last_image)
            )
        for label, value in frame_values:
            path = uploaded_media_path(value) if value is not None else None
            if path is not None:
                media.append((label, path))
        return media
    if mode != "Reference media":
        return media
    groups = (
        ("Picture", reference_images),
        ("Video", reference_videos),
        ("Audio", reference_audios),
    )
    for prefix, values in groups:
        for index, value in enumerate(values, 1):
            path = uploaded_media_path(value) if value is not None else None
            if path is not None:
                media.append((f"<{prefix} {index}>", path))
    return media


def read_system_prompt(system_path: Path, target: str) -> str:
    if not system_path.is_file():
        raise H3Error(f"Missing {target} system prompt: {system_path}")
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise H3Error(f"{target} system prompt is empty.")
    return system_prompt


def media_paths(media_values: Iterable[tuple[str, Any]]) -> list[tuple[str, Path]]:
    media: list[tuple[str, Path]] = []
    for label, value in media_values:
        if value is not None:
            path = uploaded_media_path(value)
            if path is not None:
                media.append((label, path))
    return media


def write_prompt(
    *,
    prompt: str,
    model: str,
    temporary_api_key: str,
    target: str,
    system_path: Path,
    media_values: Iterable[tuple[str, Any]],
    context: str,
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    temperature: float = 0.6,
    request_text: str | None = None,
) -> tuple[str, str]:
    """Generate a model-specific prompt from text and optional image inputs.

    ``request_text`` replaces the generic wrapper when a writer prompt expects
    the request in its own format, such as FLUX 3's TARGET CANVAS line.
    """
    if backend == "Lightning AI":
        return write_prompt_lightning(
            prompt=prompt,
            model=lightning_model,
            request_text=request_text,
            temporary_api_key=lightning_api_key,
            target=target,
            system_path=system_path,
            media_values=media_values,
            context=context,
        )
    if backend != "Gemini":
        return str(prompt or ""), f"Prompt enhancement failed: unsupported backend {backend!r}."
    try:
        if model not in GEMINI_PROMPT_MODELS:
            raise H3Error(f"Unsupported Gemini prompt model: {model}")
        key = gemini.api_key(temporary_api_key)
        system_prompt = read_system_prompt(system_path, target)
        media = media_paths(media_values)
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload an image before enhancing.")
        parts: list[dict[str, Any]] = [
            {
                "text": request_text if request_text is not None else (
                    f"Create the final {target} prompt from the following user input.\n"
                    f"{context}\nUser text:\n"
                    f"{rough_prompt or '(No text supplied; infer only from the images.)'}"
                )
            }
        ]
        with gemini.Session(key) as session:
            for label, path in media:
                parts.append({"text": f"The next uploaded image is {label}."})
                parts.append(session.file_part(path))
            payload = session.generate(
                model,
                parts,
                system_prompt=system_prompt,
                generation_config={"temperature": float(temperature), "maxOutputTokens": 16384},
            )
        enhanced = strip_code_fence(gemini.reply_text(payload))
        if not enhanced:
            raise H3Error(
                f"Gemini returned no enhanced prompt ({gemini.empty_reply_reason(payload)})."
            )
        return (
            enhanced,
            f"Enhanced {target} prompt with {model} using {len(media)} image(s).",
        )
    except (H3Error, requests.RequestException, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"


def write_prompt_lightning(
    *,
    prompt: str,
    model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    request_text: str | None = None,
    temporary_api_key: str,
    target: str,
    system_path: Path,
    media_values: Iterable[tuple[str, Any]],
    context: str,
) -> tuple[str, str]:
    """Use the H3 Lightning endpoint with a target-specific prompt and images."""
    try:
        model = lightning.model(model)
        key = lightning.api_key(temporary_api_key)
        system_prompt = read_system_prompt(system_path, target)
        media = media_paths(media_values)
        if not all(lightning.is_image(path) for _label, path in media):
            raise H3Error("Lightning AI prompt enhancement supports images only.")
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload an image before enhancing.")

        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": request_text if request_text is not None else (
                f"Create the final {target} prompt from the following user input.\n"
                f"{context}\nUser text:\n"
                f"{rough_prompt or '(No text supplied; infer only from the images.)'}"
            ),
        }]
        for label, path in media:
            content.extend((
                {"type": "text", "text": f"The next uploaded image is {label}."},
                lightning.image_part(path),
            ))
        enhanced = strip_code_fence(lightning.complete(key, model, system_prompt, content))
        if not enhanced:
            raise H3Error("Lightning AI returned no enhanced prompt.")
        return (
            enhanced,
            f"Enhanced {target} prompt with {model} "
            f"using {len(media)} image(s).",
        )
    except (H3Error, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"
