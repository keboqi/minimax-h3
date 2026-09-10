"""Extracted prompt service boundary."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from h3_app.catalog import (
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_RESULT_FORMAT,
    GEMINI_API_ROOT,
    GEMINI_PROMPT_MODELS,
    LIGHTNING_API_ROOT,
    LIGHTNING_PROMPT_MODEL,
    MAX_REFERENCE_AUDIOS,
)
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.policy import (
    active_fl2va_voice_references,
    normalize_result_format,
    validate_image_frame_count,
)
from h3_prompt_rewriter import resolution_for_size as local_prompt_resolution
from h3_prompt_rewriter import rewrite_prompt as rewrite_local_h3_prompt
from h3_prompt_rewriter import task_for_inputs as local_prompt_task


def _gemini_api_key(temporary_key: str | None) -> str:
    key = str(temporary_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set GEMINI_API_KEY in the server environment or enter a temporary "
            "Gemini API key in Prompt enhancer."
        )
    return key


def _lightning_api_key(temporary_key: str | None) -> str:
    key = str(temporary_key or "").strip() or os.getenv("LIGHTNING_API_KEY", "").strip()
    if not key:
        raise H3Error(
            "Set LIGHTNING_API_KEY in the server environment or enter a temporary "
            "Lightning API key in Prompt enhancer."
        )
    return key


def _uploaded_media_path(value: Any) -> Path | None:
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
        return _uploaded_media_path(value[0])
    else:
        raw = getattr(value, "path", None) or getattr(value, "name", None)
        candidate = Path(raw) if raw else None
    if candidate is None or not candidate.is_file():
        raise H3Error(f"Prompt-enhancer media file does not exist: {candidate}")
    return candidate


def _gemini_mime_type(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type:
        return mime_type
    suffix_defaults = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    return suffix_defaults.get(path.suffix.lower(), "application/octet-stream")


def _gemini_error(response: requests.Response, action: str) -> H3Error:
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or response.text
    except (ValueError, AttributeError):
        detail = response.text
    detail = re.sub(r"\s+", " ", str(detail)).strip()[:800]
    return H3Error(f"Gemini {action} failed (HTTP {response.status_code}): {detail}")


def _upload_gemini_file(
    session: requests.Session,
    path: Path,
    api_key: str,
) -> dict[str, Any]:
    mime_type = _gemini_mime_type(path)
    size = path.stat().st_size
    headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    started = session.post(
        f"{GEMINI_API_ROOT}/upload/v1beta/files",
        headers=headers,
        json={"file": {"displayName": path.name[:512]}},
        timeout=60,
    )
    if not started.ok:
        raise _gemini_error(started, f"upload initialization for {path.name}")
    upload_url = started.headers.get("x-goog-upload-url")
    if not upload_url:
        raise H3Error(f"Gemini did not return an upload URL for {path.name}.")
    with path.open("rb") as source:
        uploaded = session.post(
            upload_url,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": mime_type,
            },
            data=source,
            timeout=600,
        )
    if not uploaded.ok:
        raise _gemini_error(uploaded, f"upload for {path.name}")
    file_info = uploaded.json().get("file", {})
    if not file_info.get("name") or not file_info.get("uri"):
        raise H3Error(f"Gemini returned incomplete file metadata for {path.name}.")
    return file_info


def _wait_for_gemini_file(
    session: requests.Session,
    file_info: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    name = str(file_info["name"])
    deadline = time.monotonic() + 600
    while str(file_info.get("state", "ACTIVE")).upper() == "PROCESSING":
        if time.monotonic() >= deadline:
            raise H3Error(f"Gemini timed out while processing {name}.")
        time.sleep(2)
        response = session.get(
            f"{GEMINI_API_ROOT}/v1beta/{name}",
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        if not response.ok:
            raise _gemini_error(response, f"file status check for {name}")
        file_info = response.json()
    state = str(file_info.get("state", "ACTIVE")).upper()
    if state == "FAILED":
        detail = file_info.get("error", {}).get("message", "unknown processing error")
        raise H3Error(f"Gemini could not process {name}: {detail}")
    return file_info


def _active_prompt_media(
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
            path = _uploaded_media_path(value) if value is not None else None
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
            path = _uploaded_media_path(value) if value is not None else None
            if path is not None:
                media.append((f"<{prefix} {index}>", path))
    return media


def _enhance_prompt_from_media(
    *,
    prompt: str,
    model: str,
    temporary_api_key: str,
    target: str,
    system_path: Path,
    media_values: Iterable[tuple[str, Any]],
    context: str,
) -> tuple[str, str]:
    """Generate a model-specific prompt from text and optional image inputs."""
    try:
        if model not in GEMINI_PROMPT_MODELS:
            raise H3Error(f"Unsupported Gemini prompt model: {model}")
        key = _gemini_api_key(temporary_api_key)
        if not system_path.is_file():
            raise H3Error(f"Missing {target} system prompt: {system_path}")
        system_prompt = system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error(f"{target} system prompt is empty.")
        media: list[tuple[str, Path]] = []
        for label, value in media_values:
            if value is not None:
                path = _uploaded_media_path(value)
                if path is not None:
                    media.append((label, path))
        rough_prompt = str(prompt or "").strip()
        if not rough_prompt and not media:
            raise H3Error("Enter a prompt or upload an image before enhancing.")
        parts: list[dict[str, Any]] = [
            {
                "text": (
                    f"Create the final {target} prompt from the following user input.\n"
                    f"{context}\nUser text:\n"
                    f"{rough_prompt or '(No text supplied; infer only from the images.)'}"
                )
            }
        ]
        uploaded_names: list[str] = []
        with requests.Session() as session:
            try:
                for label, path in media:
                    parts.append({"text": f"The next uploaded image is {label}."})
                    file_info = _upload_gemini_file(session, path, key)
                    uploaded_names.append(str(file_info["name"]))
                    file_info = _wait_for_gemini_file(session, file_info, key)
                    parts.append(
                        {
                            "fileData": {
                                "mimeType": file_info.get("mimeType")
                                or _gemini_mime_type(path),
                                "fileUri": file_info["uri"],
                            }
                        }
                    )
                response = session.post(
                    f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {
                            "temperature": 0.6,
                            "maxOutputTokens": 16384,
                        },
                    },
                    timeout=600,
                )
                if not response.ok:
                    raise _gemini_error(response, "prompt generation")
                payload = response.json()
                candidates = payload.get("candidates") or []
                text_parts = (
                    candidates[0].get("content", {}).get("parts", [])
                    if candidates
                    else []
                )
                enhanced = "".join(
                    str(part.get("text", "")) for part in text_parts
                ).strip()
                if enhanced.startswith("```") and enhanced.endswith("```"):
                    enhanced = re.sub(r"^```[^\n]*\n?", "", enhanced)
                    enhanced = re.sub(r"\n?```$", "", enhanced).strip()
                if not enhanced:
                    reason = (
                        candidates[0].get("finishReason")
                        if candidates
                        else payload.get("promptFeedback", {}).get(
                            "blockReason", "no candidate"
                        )
                    )
                    raise H3Error(f"Gemini returned no enhanced prompt ({reason}).")
                return (
                    enhanced,
                    f"Enhanced {target} prompt with {model} using {len(media)} image(s).",
                )
            finally:
                for name in uploaded_names:
                    try:
                        session.delete(
                            f"{GEMINI_API_ROOT}/v1beta/{name}",
                            headers={"x-goog-api-key": key},
                            timeout=30,
                        )
                    except requests.RequestException:
                        pass
    except (H3Error, requests.RequestException, OSError, ValueError) as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}"


def enhance_music3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str, str]:
    caption, status = _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="MiniMax Music 3",
        system_path=runtime.prompt_systems["MiniMax Music 3"],
        media_values=(
            ("Music reference image 1", ref_image_1),
            ("Music reference image 2", ref_image_2),
            ("Music reference image 3", ref_image_3),
        ),
        context=f"Existing lyrics (preserve them unless formatting only):\n{lyrics or '(none)'}",
    )
    # Music 3's writer returns both fields using explicit markers. Keep a
    # graceful fallback for older/custom prompt responses that return caption
    # text only.
    generated_lyrics = str(lyrics or "").strip()
    marker_match = re.search(r"(?is)^\s*CAPTION:\s*(.*?)\s*LYRICS:\s*(.*)\s*$", caption)
    if marker_match:
        caption = marker_match.group(1).strip()
        candidate_lyrics = marker_match.group(2).strip()
        if not generated_lyrics and candidate_lyrics.upper() != "N/A":
            generated_lyrics = candidate_lyrics
    return caption, generated_lyrics, status


def enhance_ltx25_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    start_image: Any,
    middle_image: Any,
    end_image: Any,
    duration: float,
    width: int,
    height: int,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    return _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="LTX-2.5",
        system_path=runtime.prompt_systems["LTX-2.5"],
        media_values=(
            ("Start keyframe", start_image),
            ("Middle keyframe", middle_image),
            ("End keyframe", end_image),
        ),
        context=f"Mode: {mode}\nDuration: {float(duration):.2f} seconds\nOutput: {int(width)}x{int(height)}",
    )


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
        key = _gemini_api_key(temporary_api_key)
        if not runtime.prompt_system_path.is_file():
            raise H3Error(
                f"Missing prompt enhancer system prompt: {runtime.prompt_system_path}"
            )
        system_prompt = runtime.prompt_system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error("prompt.txt is empty.")
        media = _active_prompt_media(
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
        uploaded_names: list[str] = []
        with requests.Session() as session:
            try:
                for label, path in media:
                    parts.append({"text": f"The next uploaded file is {label}."})
                    file_info = _upload_gemini_file(session, path, key)
                    uploaded_names.append(str(file_info["name"]))
                    file_info = _wait_for_gemini_file(session, file_info, key)
                    parts.append(
                        {
                            "fileData": {
                                "mimeType": file_info.get("mimeType")
                                or _gemini_mime_type(path),
                                "fileUri": file_info["uri"],
                            }
                        }
                    )
                response = session.post(
                    f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
                    headers={
                        "x-goog-api-key": key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": parts}],
                        "generationConfig": {
                            "temperature": 0.6,
                            "maxOutputTokens": 16384,
                        },
                    },
                    timeout=600,
                )
                if not response.ok:
                    raise _gemini_error(response, "prompt generation")
                payload = response.json()
                candidates = payload.get("candidates") or []
                text_parts = (
                    candidates[0].get("content", {}).get("parts", [])
                    if candidates
                    else []
                )
                enhanced = "".join(
                    str(part.get("text", "")) for part in text_parts
                ).strip()
                if enhanced.startswith("```") and enhanced.endswith("```"):
                    enhanced = re.sub(r"^```[^\n]*\n?", "", enhanced)
                    enhanced = re.sub(r"\n?```$", "", enhanced).strip()
                if not enhanced:
                    reason = (
                        candidates[0].get("finishReason")
                        if candidates
                        else payload.get("promptFeedback", {}).get(
                            "blockReason", "no candidate"
                        )
                    )
                    raise H3Error(f"Gemini returned no enhanced prompt ({reason}).")
                return (
                    enhanced,
                    f"Enhanced with {model} using {len(media)} media file(s).",
                )
            finally:
                for name in uploaded_names:
                    try:
                        session.delete(
                            f"{GEMINI_API_ROOT}/v1beta/{name}",
                            headers={"x-goog-api-key": key},
                            timeout=30,
                        )
                    except requests.RequestException:
                        pass
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
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Generate or enhance an H3 prompt through Lightning's OpenAI API."""
    from openai import OpenAI, OpenAIError

    try:
        key = _lightning_api_key(temporary_api_key)
        if not runtime.prompt_system_path.is_file():
            raise H3Error(
                f"Missing prompt enhancer system prompt: {runtime.prompt_system_path}"
            )
        system_prompt = runtime.prompt_system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise H3Error("prompt.txt is empty.")
        media = _active_prompt_media(
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

        unsupported = [
            label
            for label, path in media
            if not _gemini_mime_type(path).startswith("image/")
        ]
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
            mime_type = _gemini_mime_type(path)
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.extend(
                (
                    {"type": "text", "text": f"The next image is {label}."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded}",
                            "detail": "high",
                        },
                    },
                )
            )

        client = OpenAI(
            base_url=LIGHTNING_API_ROOT,
            api_key=key,
            timeout=600.0,
        )
        completion = client.chat.completions.create(
            model=LIGHTNING_PROMPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        enhanced = str(completion.choices[0].message.content or "").strip()
        if enhanced.startswith("```") and enhanced.endswith("```"):
            enhanced = re.sub(r"^```[^\n]*\n?", "", enhanced)
            enhanced = re.sub(r"\n?```$", "", enhanced).strip()
        if not enhanced:
            raise H3Error("Lightning AI returned no enhanced prompt.")
        return (
            enhanced,
            f"Enhanced with {LIGHTNING_PROMPT_MODEL} using {len(media)} image(s).",
        )
    except (H3Error, OpenAIError, OSError, ValueError, IndexError) as exc:
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
