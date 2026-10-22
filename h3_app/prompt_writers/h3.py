"""MiniMax H3 prompt writers: local 8B, Gemini and Lightning, with FL2VA voice labels."""

from __future__ import annotations

import re
from typing import Any

import requests

from h3_app.catalog import (
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_RESULT_FORMAT,
    GEMINI_PROMPT_MODELS,
    DEFAULT_LIGHTNING_PROMPT_MODEL,
    MAX_REFERENCE_AUDIOS,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.llm import gemini, lightning, strip_code_fence
from h3_app.policy import (
    active_fl2va_voice_references,
    normalize_result_format,
    validate_image_frame_count,
)
from h3_prompt_rewriter import resolution_for_size as local_prompt_resolution
from h3_prompt_rewriter import rewrite_prompt as rewrite_local_h3_prompt
from h3_prompt_rewriter import task_for_inputs as local_prompt_task
from . import common


def _enhance_h3_prompt_with_gemini(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Generate or enhance an H3 prompt from the active text and media inputs."""
    try:
        if model not in GEMINI_PROMPT_MODELS:
            raise H3Error(f"Unsupported Gemini prompt model: {model}")
        key = gemini.api_key(temporary_api_key)
        if not runtime.prompt_system_path.is_file():
            raise H3Error(
                f"Missing prompt enhancer system prompt: {runtime.prompt_system_path}"
            )
        system_prompt = runtime.prompt_system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error("prompt.txt is empty.")
        media = common.active_prompt_media(
            mode,
            first_image,
            last_image,
            (
                ref_image_1,
                ref_image_2,
                ref_image_3,
                ref_image_4,
                ref_image_5,
                ref_image_6,
                ref_image_7,
                ref_image_8,
                ref_image_9,
            ),
            (ref_video_1, ref_video_2, ref_video_3),
            (ref_audio_1, ref_audio_2, ref_audio_3),
        )
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload media before enhancing.")

        normalized_result = normalize_result_format(result_format)
        timing_request = (
            f"Requested image frames: {validate_image_frame_count(image_frames)}"
            if normalized_result == "Image"
            else f"Requested duration: {float(duration):.2f} seconds"
        )
        requested_width = 32 if normalized_result == "Audio" else int(width)
        requested_height = 32 if normalized_result == "Audio" else int(height)
        parts: list[dict[str, Any]] = [
            {
                "text": (
                    "Create the final MiniMax H3 prompt from the following user request.\n"
                    f"UI mode: {mode}\nResult format: {normalized_result}\n"
                    f"{timing_request}\n"
                    f"Requested output: {requested_width}x{requested_height}\n"
                    f"User text:\n{rough_prompt or '(No text supplied; infer only from the media.)'}"
                )
            }
        ]
        with gemini.Session(key) as session:
            for label, path in media:
                parts.append({"text": f"The next uploaded file is {label}."})
                parts.append(session.file_part(path))
            payload = session.generate(
                model,
                parts,
                system_prompt=system_prompt,
                generation_config={"temperature": 0.6, "maxOutputTokens": 16384},
            )
        enhanced = strip_code_fence(gemini.reply_text(payload))
        if not enhanced:
            raise H3Error(
                f"Gemini returned no enhanced prompt ({gemini.empty_reply_reason(payload)})."
            )
        return (
            enhanced,
            f"Enhanced with {model} using {len(media)} media file(s).",
        )
    except (H3Error, requests.RequestException, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"


def _enhance_h3_prompt_with_lightning(
    prompt: str,
    temporary_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Generate or enhance an H3 prompt through Lightning's OpenAI API."""
    try:
        lightning_model = lightning.model(lightning_model)
        key = lightning.api_key(temporary_api_key)
        if not runtime.prompt_system_path.is_file():
            raise H3Error(
                f"Missing prompt enhancer system prompt: {runtime.prompt_system_path}"
            )
        system_prompt = runtime.prompt_system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error("prompt.txt is empty.")
        media = common.active_prompt_media(
            mode,
            first_image,
            last_image,
            (
                ref_image_1,
                ref_image_2,
                ref_image_3,
                ref_image_4,
                ref_image_5,
                ref_image_6,
                ref_image_7,
                ref_image_8,
                ref_image_9,
            ),
            (ref_video_1, ref_video_2, ref_video_3),
            (ref_audio_1, ref_audio_2, ref_audio_3),
        )
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload an image before enhancing.")

        unsupported = [label for label, path in media if not lightning.is_image(path)]
        if unsupported:
            raise H3Error(
                "Lightning AI prompt enhancement supports text and images; use "
                "Gemini for active video or audio references ("
                + ", ".join(unsupported)
                + ")."
            )

        normalized_result = normalize_result_format(result_format)
        timing_request = (
            f"Requested image frames: {validate_image_frame_count(image_frames)}"
            if normalized_result == "Image"
            else f"Requested duration: {float(duration):.2f} seconds"
        )
        requested_width = 32 if normalized_result == "Audio" else int(width)
        requested_height = 32 if normalized_result == "Audio" else int(height)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Create the final MiniMax H3 prompt from the following user request.\n"
                    f"UI mode: {mode}\nResult format: {normalized_result}\n"
                    f"{timing_request}\n"
                    f"Requested output: {requested_width}x{requested_height}\n"
                    f"User text:\n{rough_prompt or '(No text supplied; infer only from the images.)'}"
                ),
            }
        ]
        for label, path in media:
            content.extend(
                (
                    {"type": "text", "text": f"The next image is {label}."},
                    lightning.image_part(path),
                )
            )
        enhanced = strip_code_fence(
            lightning.complete(key, lightning_model, system_prompt, content)
        )
        if not enhanced:
            raise H3Error("Lightning AI returned no enhanced prompt.")
        return (
            enhanced,
            f"Enhanced with {lightning_model} using {len(media)} image(s).",
        )
    except (H3Error, OSError, ValueError) as exc:
        return str(prompt or ""), f"Lightning AI prompt enhancement failed: {exc}"


def fl2va_prompt_voice_context(prompt: str, mode: str, *slots: Any):
    """Describe voice labels to writers without exposing audio files or paths."""
    if mode != "First / last frame":
        return "", set(), set()
    references = active_fl2va_voice_references(mode, *slots)
    mentioned = set(re.findall(r"<Audio\s+(\d+)>", prompt, flags=re.IGNORECASE))
    allowed = (
        {str(i) for i in range(1, len(references) + 1)} if references else mentioned
    )
    if mentioned - allowed or any(
        not 1 <= int(i) <= MAX_REFERENCE_AUDIOS for i in mentioned
    ):
        raise H3Error(
            "The prompt names a FL2VA audio tag without a matching voice slot."
        )
    if not allowed:
        return "", set(), set()
    labels = ", ".join(f"<Audio {i}>" for i in sorted(allowed, key=int))
    context = (
        "FL2VA voice-reference context for this rewrite:\n"
        f"Available voice labels: {labels}. "
        "These are voice-timbre references for generation; their audio is not provided "
        "to the prompt writer. Do not claim to have heard them or invent their vocal "
        "qualities or transcript. Preserve every audio tag already in the user's "
        "prompt, its exact speaker assignment, and all dialogue verbatim. "
        "Do not renumber tags or introduce labels outside the available set. "
        "For unused labels, assign a speaker only when the user's intent makes that "
        "association clear. Retain exact first/last-frame anchoring and the "
        "I2VA/FL2VA/L2VA task; audio references do not change it to Ref2VA. "
        "Generate new dialogue in the referenced voice; do not request source "
        "audio playback or reuse. Return only the enhanced prompt."
    )
    return context, allowed, mentioned


def enhance_h3_prompt(
    prompt: str,
    backend: str,
    local_base_model: str,
    local_max_new_tokens: int,
    local_temperature: float,
    local_top_p: float,
    local_greedy: bool,
    local_seed: int,
    gemini_model: str,
    gemini_api_key: str,
    lightning_api_key: str,
    mode: str,
    first_image: Any,
    last_image: Any,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    ref_image_4: Any,
    ref_image_5: Any,
    ref_image_6: Any,
    ref_image_7: Any,
    ref_image_8: Any,
    ref_image_9: Any,
    ref_video_1: Any,
    ref_video_2: Any,
    ref_video_3: Any,
    ref_audio_1: Any,
    ref_audio_2: Any,
    ref_audio_3: Any,
    duration: float,
    width: int,
    height: int,
    result_format: str = DEFAULT_RESULT_FORMAT,
    image_frames: int = DEFAULT_IMAGE_FRAMES,
    fl2va_audio_1: Any = None,
    fl2va_audio_2: Any = None,
    fl2va_audio_3: Any = None,
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Dispatch H3 prompt enhancement to the selected prompt-writer backend."""
    original_prompt = str(prompt or "")
    try:
        voice_context, allowed_tags, required_tags = fl2va_prompt_voice_context(
            original_prompt, mode, fl2va_audio_1, fl2va_audio_2, fl2va_audio_3
        )
    except H3Error as exc:
        return original_prompt, f"Prompt enhancement failed: {exc}"
    if voice_context:
        prompt = voice_context + "\n\nUser prompt:\n" + original_prompt

    def finish(result):
        enhanced, status = result
        if not voice_context:
            return result
        if enhanced == prompt or "failed" in status.lower():
            return original_prompt, status
        actual_tags = set(re.findall(r"<Audio\s+(\d+)>", enhanced, flags=re.IGNORECASE))
        if required_tags - actual_tags or actual_tags - allowed_tags:
            return original_prompt, (
                "Prompt enhancement failed: the writer changed the FL2VA audio labels. "
                "Your original prompt was preserved; retry to keep speaker references intact."
            )
        return enhanced, status

    if backend == "Gemini":
        return finish(
            _enhance_h3_prompt_with_gemini(
                prompt,
                gemini_model,
                gemini_api_key,
                mode,
                first_image,
                last_image,
                ref_image_1,
                ref_image_2,
                ref_image_3,
                ref_image_4,
                ref_image_5,
                ref_image_6,
                ref_image_7,
                ref_image_8,
                ref_image_9,
                ref_video_1,
                ref_video_2,
                ref_video_3,
                ref_audio_1,
                ref_audio_2,
                ref_audio_3,
                duration,
                width,
                height,
                result_format,
                image_frames,
                runtime=runtime,
            )
        )
    if backend == "Lightning AI":
        return finish(
            _enhance_h3_prompt_with_lightning(
                prompt,
                lightning_api_key,
                mode,
                first_image,
                last_image,
                ref_image_1,
                ref_image_2,
                ref_image_3,
                ref_image_4,
                ref_image_5,
                ref_image_6,
                ref_image_7,
                ref_image_8,
                ref_image_9,
                ref_video_1,
                ref_video_2,
                ref_video_3,
                ref_audio_1,
                ref_audio_2,
                ref_audio_3,
                duration,
                width,
                height,
                result_format,
                image_frames,
                lightning_model,
                runtime=runtime,
            )
        )
    if backend != "Local MiniMax-H3 8B":
        return (
            original_prompt,
            f"Prompt enhancement failed: unsupported backend {backend!r}.",
        )
    try:
        if normalize_result_format(result_format) != "Video":
            raise H3Error("The local 8B writer supports H3 audio-video prompts only.")
        rough_prompt = str(prompt or "").strip()
        if not original_prompt.strip():
            raise H3Error("Enter a text prompt before enhancing locally.")
        numeric_duration = float(duration)
        if not numeric_duration.is_integer():
            raise H3Error(
                "The local 8B writer was trained for whole-second durations; "
                "choose an integer duration from 4 to 15 seconds."
            )
        task = local_prompt_task(mode, first_image, last_image)
        resolution = local_prompt_resolution(width, height, task)
        return finish(
            rewrite_local_h3_prompt(
                prompt=rough_prompt,
                task=task,
                resolution=resolution,
                duration=int(numeric_duration),
                first_frame=first_image,
                last_frame=last_image,
                base_model=local_base_model,
                max_new_tokens=int(local_max_new_tokens),
                temperature=float(local_temperature),
                top_p=float(local_top_p),
                greedy=bool(local_greedy),
                seed=int(local_seed),
            )
        )
    except (H3Error, OSError, RuntimeError, ValueError) as exc:
        return original_prompt, f"Local prompt enhancement failed: {exc}"
