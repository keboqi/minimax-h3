"""Prompt writers for every engine over local, Gemini and Lightning backends."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import requests

from h3_app.catalog import (
    DEFAULT_IMAGE_FRAMES,
    DEFAULT_RESULT_FORMAT,
    GEMINI_PROMPT_MODELS,
    DEFAULT_LIGHTNING_PROMPT_MODEL,
    MAX_REFERENCE_AUDIOS,
)
from h3_app import song_brief
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


def _read_system_prompt(system_path: Path, target: str) -> str:
    if not system_path.is_file():
        raise H3Error(f"Missing {target} system prompt: {system_path}")
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise H3Error(f"{target} system prompt is empty.")
    return system_prompt


def _media_paths(media_values: Iterable[tuple[str, Any]]) -> list[tuple[str, Path]]:
    media: list[tuple[str, Path]] = []
    for label, value in media_values:
        if value is not None:
            path = _uploaded_media_path(value)
            if path is not None:
                media.append((label, path))
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
        return _enhance_prompt_from_media_lightning(
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
        system_prompt = _read_system_prompt(system_path, target)
        media = _media_paths(media_values)
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


def _enhance_prompt_from_media_lightning(
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
        system_prompt = _read_system_prompt(system_path, target)
        media = _media_paths(media_values)
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


def _song_label(name: str) -> str:
    # Writers decorate labels as **STYLE:**, **STYLE**: or ## Style:.
    return rf"[ \t>#*_]*{name}[ \t*_]*:[ \t*_]*"


def _song_reply(text: str, first: str) -> tuple[str, str] | None:
    """Split a writer reply into its two marked blocks; None if they are absent."""
    text = re.sub(r"(?m)^\s*```[a-zA-Z]*\s*$", "", str(text or ""))
    # The first label opens a line, so a short preamble before it is ignored.
    match = re.search(
        rf"(?is)(?:^|\n){_song_label(first)}(.*?)\s{_song_label('LYRICS')}(.*)$", text
    )
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def _song_lyrics(
    original: str, candidate: str | None, *, policy: str
) -> str:
    """Code-enforced lyric lock: the writer never silently replaces user lyrics."""
    candidate = "" if candidate is None or candidate.strip().upper() == "N/A" else candidate.strip()
    if original and (policy != song_brief.POLISH_LYRICS or not candidate):
        return original
    return candidate


def _has_sung_words(lyrics: str) -> bool:
    return any(section.lines for section in song_brief.parse_lyrics(lyrics))


def _writer_lyrics(lyrics: str, instrumental: bool) -> str:
    """The lyrics Write should keep: bare section tags only while still instrumental.

    Tags without words are the form Write fills in for an instrumental, so once
    a voice is chosen they are not lyrics to keep.
    """
    return lyrics if instrumental or _has_sung_words(lyrics) else ""


def enhance_music3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    ref_image_1: Any,
    ref_image_2: Any,
    ref_image_3: Any,
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    duration: float = 120,
    vocals: str = song_brief.AUTO,
    language: str = song_brief.AUTO,
    use: str = song_brief.AUTO,
    lyrics_policy: str = song_brief.KEEP_LYRICS,
    notes: str = "",
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str, str]:
    original_caption = str(prompt or "").strip()
    original_lyrics = str(lyrics or "").strip()
    instrumental = song_brief.wants_instrumental(vocals, use, original_caption)
    draft_lyrics = _writer_lyrics(original_lyrics, instrumental)
    request_text = original_caption
    if not request_text and draft_lyrics:
        request_text = "Create a fitting music description for these lyrics."
    brief = song_brief.build_brief(
        engine="MiniMax Music 3", max_seconds=duration, vocals=vocals,
        language=language, use=use, lyrics_policy=lyrics_policy, notes=notes,
        idea=original_caption, lyrics=draft_lyrics,
    )
    generated, status = _enhance_prompt_from_media(
        prompt=request_text,
        model=model,
        temporary_api_key=temporary_api_key,
        target="MiniMax Music 3",
        system_path=runtime.prompt_systems["MiniMax Music 3"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=(
            ("Music reference image 1", ref_image_1),
            ("Music reference image 2", ref_image_2),
            ("Music reference image 3", ref_image_3),
        ),
        context=f"{brief}\nExisting lyrics:\n{draft_lyrics or '(none)'}",
    )
    if status.startswith("Prompt enhancement failed:"):
        return original_caption, original_lyrics, status
    reply = _song_reply(generated, "CAPTION")
    if reply is None:
        if re.search(r"(?i)\bLYRICS[ \t*_]*:", generated):
            return (
                original_caption, original_lyrics,
                "Prompt enhancement failed: the writer reply could not be split "
                "into caption and lyrics. Try again.",
            )
        # Older or custom writer prompts may return a caption only.
        reply = (str(generated).strip(), None)
    caption, candidate = reply
    if not caption:
        return original_caption, original_lyrics, (
            "Prompt enhancement failed: the writer returned an empty music description."
        )
    new_lyrics = _song_lyrics(draft_lyrics, candidate, policy=lyrics_policy)
    if instrumental and not draft_lyrics:
        # Music 3 sings any words it is given; an instrumental gets none.
        new_lyrics = ""
    return caption, new_lyrics, status


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
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    return _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="LTX-2.5",
        system_path=runtime.prompt_systems["LTX-2.5"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=(
            ("Start keyframe", start_image),
            ("Middle keyframe", middle_image),
            ("End keyframe", end_image),
        ),
        context=f"Mode: {mode}\nDuration: {float(duration):.2f} seconds\nOutput: {int(width)}x{int(height)}",
    )


def enhance_qwen_image21_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    mode: str,
    reference_images: Any,
    width: int,
    height: int,
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    references = reference_images or []
    if isinstance(references, (str, Path, dict)):
        references = [references]
    reference_count = len(references)
    if reference_count == 1:
        media_values = (("Input image (edit target)", references[0]),)
        edit_context = (
            "In Image edit mode, the uploaded image is the edit target. "
            "Refer to it naturally and do not use an <image1> tag."
        )
    else:
        media_values = tuple(
            (f"<image{index}>", image)
            for index, image in enumerate(references, 1)
        )
        edit_context = (
            "In Image edit mode with multiple inputs, <image1> is the edit "
            "target and later images are references. Use the numbered image "
            "tags verbatim."
        )
    return _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="Qwen Image 2.1",
        system_path=runtime.prompt_systems["Qwen Image 2.1"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=media_values,
        context=(
            f"Mode: {mode}\nOutput: {int(width)}x{int(height)}\n"
            f"{edit_context}"
        ),
    )


def enhance_yue2_prompt(
    style: str,
    model: str,
    temporary_api_key: str,
    lyrics: str,
    mode: str,
    duration: float,
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    vocals: str = song_brief.AUTO,
    language: str = song_brief.AUTO,
    use: str = song_brief.AUTO,
    lyrics_policy: str = song_brief.KEEP_LYRICS,
    notes: str = "",
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str, str]:
    original_style = str(style or "").strip()
    original_lyrics = str(lyrics or "").strip()
    instrumental = song_brief.wants_instrumental(vocals, use, original_style)
    draft_lyrics = _writer_lyrics(original_lyrics, instrumental)
    request_text = original_style
    if not request_text and draft_lyrics:
        request_text = "Create a fitting production style for these lyrics."
    brief = song_brief.build_brief(
        engine="YuE2", max_seconds=duration, vocals=vocals, language=language,
        use=use, lyrics_policy=lyrics_policy, notes=notes, idea=original_style,
        lyrics=draft_lyrics, mode=mode,
    )
    generated, status = _enhance_prompt_from_media(
        prompt=request_text,
        model=model,
        temporary_api_key=temporary_api_key,
        target="YuE2",
        system_path=runtime.prompt_systems["YuE2"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=(),
        context=f"{brief}\nExisting lyrics:\n{draft_lyrics or '(none)'}",
    )
    if status.startswith("Prompt enhancement failed:"):
        return original_style, original_lyrics, status
    reply = _song_reply(generated, "STYLE")
    if reply is None:
        return original_style, original_lyrics, (
            "Prompt enhancement failed: the writer reply could not be split "
            "into style and lyrics. Try again."
        )
    # YuE2 reads the style as tags: one comma-separated line.
    new_style = ", ".join(
        part.strip(" .")
        for part in re.split(r"[\n,]+", reply[0])
        if part.strip(" .")
    )
    if not new_style:
        return original_style, original_lyrics, (
            "Prompt enhancement failed: the writer returned an empty style."
        )
    new_lyrics = _song_lyrics(draft_lyrics, reply[1], policy=lyrics_policy)
    if not new_lyrics or (instrumental and not draft_lyrics and _has_sung_words(new_lyrics)):
        # YuE2 needs section tags even for an instrumental; follow the planned form.
        plan = song_brief.plan_song(
            duration, vocals=song_brief.INSTRUMENTAL, use=use,
            bpm=song_brief.detect_bpm(new_style),
        )
        new_lyrics = song_brief.instrumental_tags(plan)
    return new_style, new_lyrics, status


COMMON_CANVAS_RATIOS = ((1, 1), (5, 4), (4, 3), (3, 2), (16, 9), (2, 1), (21, 9))


def _canvas_ratio(width: int, height: int) -> str:
    """Name the canvas as a common W:H when within 2%, else the exact reduced ratio."""
    width, height = int(width), int(height)
    ratio = width / height
    candidates = [*COMMON_CANVAS_RATIOS, *((h, w) for w, h in COMMON_CANVAS_RATIOS)]
    w, h = min(candidates, key=lambda pair: abs(math.log(ratio * pair[1] / pair[0])))
    if abs(math.log(ratio * h / w)) <= 0.02:
        return f"{w}:{h}"
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"


FLUX3_REFERENCE_ORDINALS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)


def enhance_flux3_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    reference_images: Any,
    width: int,
    height: int,
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Expand a FLUX 3 request with BFL's official upsampler instructions."""
    references = reference_images or []
    if isinstance(references, (str, Path, dict)):
        references = [references]
    if len(references) > len(FLUX3_REFERENCE_ORDINALS):
        return str(prompt or ""), "Prompt enhancement failed: FLUX 3 supports at most ten reference images."
    request = str(prompt or "").strip()
    if references:
        # The official edit upsampler sees the references in order.
        target = "FLUX 3 Image Klein edit"
        request_text = request or "Make the most plausible improvement to the supplied images."
    else:
        if not request:
            return str(prompt or ""), "Prompt enhancement failed: write an image request first."
        target = "FLUX 3 Image Klein"
        request_text = f"TARGET CANVAS: {_canvas_ratio(width, height)}\n{request}"
    return _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target=target,
        system_path=runtime.prompt_systems[target],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=tuple(
            (f"the {FLUX3_REFERENCE_ORDINALS[index]} image", image)
            for index, image in enumerate(references)
        ),
        context="",
        request_text=request_text,
    )


def enhance_moss_image_prompt(
    prompt: str,
    model: str,
    temporary_api_key: str,
    width: int,
    height: int,
    rewrite_mode: str = "Auto",
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    *,
    runtime: RuntimeConfig,
) -> tuple[str, str]:
    """Expand a MOSS-Image request with OpenMOSS's prompt-enhancer recipe.

    The writer returns a caption or pixel Layout draft, which is validated and
    compiled locally exactly as the official skill's compile_prompt.py does.
    """
    from h3_app.workflows.moss_image_prompt import MOSS_IMAGE_REWRITE_MODES, compile_writer_draft

    request = str(prompt or "").strip()
    if not request:
        return str(prompt or ""), "Prompt enhancement failed: write an image request first."
    if rewrite_mode not in MOSS_IMAGE_REWRITE_MODES:
        return str(prompt or ""), "Prompt enhancement failed: choose Auto, Caption or Layout."
    reply, status = _enhance_prompt_from_media(
        prompt=prompt,
        model=model,
        temporary_api_key=temporary_api_key,
        target="MOSS-Image",
        system_path=runtime.prompt_systems["MOSS-Image"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=(),
        context="",
        request_text=(
            f"TARGET CANVAS: {int(width)}x{int(height)}\n"
            f"DELIVERABLE: {rewrite_mode}\nUSER REQUEST:\n{request}"
        ),
    )
    if status.startswith("Prompt enhancement failed:"):
        return reply, status
    try:
        compiled, layout = compile_writer_draft(reply, width, height, rewrite_mode)
    except ValueError as exc:
        return str(prompt or ""), f"Prompt enhancement failed: {exc}. Try again or choose Caption."
    kind = f"Layout for {int(width)}x{int(height)}" if layout else "caption"
    return compiled, f"{status.removesuffix(' using 0 image(s).')} as a {kind}."


def validate_drama3_script(text: str) -> str:
    """Reject scripts the Drama 3 prompt builder would refuse or speak wrongly."""
    from h3_app.workflows.drama3 import validate_drama3_text

    return validate_drama3_text(text, "The writer's script")


DRAMA3_SPEAKER_AUTO = "Auto"
DRAMA3_SPEAKER_ADAPT = "Adapt to voices"
DRAMA3_SPEAKER_SINGLE = "Single"
DRAMA3_SPEAKER_MODES = (DRAMA3_SPEAKER_AUTO, DRAMA3_SPEAKER_ADAPT, DRAMA3_SPEAKER_SINGLE)

DRAMA3_SCRIPT_BASIC = "Basic"
DRAMA3_SCRIPT_ADVANCED = "Advanced (sound enhanced)"
DRAMA3_SCRIPT_MODES = (DRAMA3_SCRIPT_BASIC, DRAMA3_SCRIPT_ADVANCED)


def drama3_voice_context(voices: Iterable[tuple[int, str]]) -> str:
    """Describe cloned-voice slots by speaker ID and saved name only."""
    lines = [
        f"- speaker_id {speaker}"
        + (f' ("{name}")' if name else "")
        + " uses a cloned reference voice; direct its delivery only."
        for speaker, name in voices
    ]
    if not lines:
        return "Voice context: no reference voices; cast every voice with a style prefix."
    return (
        "Voice context (give these speaker IDs to the matching characters; any other speaker "
        "gets an invented voice cast with a style prefix):\n" + "\n".join(lines)
    )


def drama3_speaker_plan(mode: str, voices: Iterable[tuple[int, str]]) -> tuple[tuple[int, ...] | None, str]:
    """Return the allowed speaker numbers (None for any) and the writer instruction."""
    cloned = sorted({int(speaker) for speaker, _name in voices})
    if mode == DRAMA3_SPEAKER_AUTO:
        return None, "Speaker count: decide from the text as described in your instructions."
    if mode == DRAMA3_SPEAKER_SINGLE:
        speaker = cloned[0] if cloned else 0
        return (speaker,), (
            f"Speaker count: exactly ONE speaker, speaker_id {speaker}; start every line with its name. "
            "Do not split the text between characters or add a narrator; one voice performs "
            "everything, including any quoted dialogue, changing only its delivery."
        )
    if mode == DRAMA3_SPEAKER_ADAPT:
        if not cloned:
            raise H3Error("Adapt to voices needs at least one reference voice or saved speaker; "
                          "add one or choose Auto or Single.")
        ids = ", ".join(str(speaker) for speaker in cloned)
        return tuple(cloned), (
            f"Speaker count: exactly {len(cloned)} speaker(s), one per reference voice: speaker_id {ids}. "
            "Use only these speaker IDs. Distribute the lines among them by role; when the "
            "text has more characters than voices, give the extra characters' lines to the "
            "closest-fitting voice instead of inventing new speakers."
        )
    raise H3Error(f"Unsupported Drama 3 speaker mode: {mode}")


def drama3_script_mode_instruction(script_mode: str) -> str:
    if script_mode == DRAMA3_SCRIPT_BASIC:
        return (
            "Script mode: Basic (dialogue and vocal expression only; do NOT include "
            "<parallel>, <sequential>, <sound>, <#...#>, or speaker vocal effect tags)."
        )
    if script_mode == DRAMA3_SCRIPT_ADVANCED:
        return (
            'Script mode: Advanced (sound enhanced — enrich the script with nested '
            '<parallel>/<sequential> staging, <sound [loop="true"] [volume="0.2"] name="...">...</sound> '
            "sound effects and atmosphere music, <#seconds#> pauses, and speaker vocal effect tags "
            "such as <phone_call> or <robotic_voice> where appropriate)."
        )
    raise H3Error(f"Unsupported Drama 3 script mode: {script_mode}")


def enforce_drama3_speakers(script: str, allowed: tuple[int, ...] | None) -> str:
    """Fold a single-speaker script onto its tag; reject other speakers otherwise."""
    from h3_app.drama3_serving.tree import CONTAINER_TAG_RE, PAUSE_TAG_RE, _parse_sound_tag
    from h3_app.workflows.drama3 import SPEAKER_TAG, text_speakers

    if allowed is None:
        return script
    if len(allowed) == 1:
        tag = f"<|speaker:{allowed[0]}|>"
        rebuilt: list[str] = []
        for raw_line in script.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if (
                CONTAINER_TAG_RE.fullmatch(stripped)
                or PAUSE_TAG_RE.fullmatch(stripped)
                or (stripped.lower().startswith("<sound") and _parse_sound_tag(stripped) is not None)
            ):
                rebuilt.append(stripped)
                continue
            body = SPEAKER_TAG.sub("", stripped).strip()
            if body:
                rebuilt.append(f"{tag} {body}")
        return "\n".join(rebuilt)
    extra = sorted(text_speakers(script) - set(allowed))
    if extra:
        raise H3Error("The writer used speakers without a reference voice: "
                      + ", ".join(f"Speaker {speaker}" for speaker in extra) + ".")
    return script


DRAMA3_PROMPT_FAILED = "Prompt enhancement failed:"


def drama3_writer_context(
    voices: Iterable[tuple[int, str]],
    speaker_mode: str,
    cast_context: str = "",
    *,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
) -> tuple[tuple[int, ...] | None, str]:
    """The allowed speaker IDs and the context lines sent with the user's text."""
    voices = tuple(voices)
    allowed, speaker_instruction = drama3_speaker_plan(speaker_mode, voices)
    mode_instruction = drama3_script_mode_instruction(script_mode)
    parts = [drama3_voice_context(voices), speaker_instruction, mode_instruction]
    if cast_context:
        parts.append(cast_context)
    if sound_context and script_mode == DRAMA3_SCRIPT_ADVANCED:
        parts.append(sound_context)
    return allowed, "\n".join(parts)


def drama3_full_prompt(
    text: str,
    voices: Iterable[tuple[int, str]] = (),
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    runtime: RuntimeConfig,
    cast_context: str = "",
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
) -> str:
    """The exact system prompt and request the built-in writer sends, for use in another chat model."""
    original = str(text or "").strip()
    if not original:
        raise H3Error("Write the speech text or a scene brief first.")
    _allowed, context = drama3_writer_context(
        voices,
        speaker_mode,
        cast_context,
        script_mode=script_mode,
        sound_context=sound_context,
    )
    system_prompt = runtime.prompt_systems["Drama 3"].read_text(encoding="utf-8").strip()
    # Same wording as the built-in request, so pasted replies follow the same contract.
    request = (f"Create the final Drama 3 prompt from the following user input.\n"
               f"{context}\nUser text:\n{original}")
    return f"{system_prompt}\n\n----- USER REQUEST -----\n\n{request}\n"


def _strip_reply_wrapping(reply: str) -> str:
    """Accept a pasted JSON reply wrapped in a code fence or a sentence of chat."""
    reply = str(reply or "").strip()
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", reply, flags=re.S)
    if fenced:
        reply = fenced.group(1).strip()
    if not reply.startswith("{") and "{" in reply and "}" in reply:
        reply = reply[reply.index("{"):reply.rindex("}") + 1]
    return reply


def parse_drama3_reply(
    reply: str,
    original: str,
    allowed: tuple[int, ...] | None,
    speaker_mode: str,
    *,
    require_cast: bool,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    return_sounds: bool = False,
    notes: list[str] | None = None,
    meta: dict | None = None,
    speaker_names: dict[int, str] | None = None,
    direct=None,
) -> tuple:
    """Validate a writer reply exactly as the built-in writer does; raises ``H3Error``.

    ``original`` may name its speakers ("Mia: Hello."); ``speaker_names`` maps the
    Cast's names to speaker IDs for the checks against it.

    Risky cue placements are repaired; what was repaired, and source phrases the
    script dropped, are appended to ``notes`` for the status line. The reply's
    title, if any, goes in ``meta["title"]``. ``direct(script, cast)``, when given,
    returns the validated script with more delivery direction and a note, or None.
    """
    from h3_app.drama3_cast import (
        parse_script_cast_and_sounds,
        parse_writer_source_mode,
        parse_writer_title,
        voices_without_sex,
    )
    from h3_app.drama3_serving.tree import extract_script_sounds, strip_drama3_sound_tags
    from h3_app.workflows.drama3 import names_to_tags

    original = names_to_tags(original, speaker_names)

    def review(script, source_mode=""):
        from h3_app.drama3_lint import coverage_note, lint_drama3_script

        script, messages = lint_drama3_script(script)
        if notes is not None:
            notes.extend(message[0].upper() + message[1:] + "." for message in messages)
            coverage = coverage_note(original, script, source_mode=source_mode)
            if coverage:
                notes.append(coverage)
        return script

    if require_cast or reply.lstrip().startswith("{"):
        script, cast, sounds = parse_script_cast_and_sounds(reply)
        if meta is not None:
            meta["title"] = parse_writer_title(reply)
        if script_mode == DRAMA3_SCRIPT_BASIC:
            script = strip_drama3_sound_tags(script)
            sounds = []
        if direct is not None:
            script, note = direct(script, cast)
            if note and notes is not None:
                notes.append(note)
        script = review(script, parse_writer_source_mode(reply))
        unsexed = voices_without_sex(cast)
        if unsexed and notes is not None:
            notes.append(f"The voice description for {', '.join(unsexed)} does not say male or female; "
                         "add it before generating voices, or the voice may come out wrong.")
        if speaker_mode == DRAMA3_SPEAKER_AUTO:
            from h3_app.workflows.drama3 import SPEAKER_TAG, text_speakers
            existing_ids = {int(value) for value in SPEAKER_TAG.findall(str(original or ""))}
            if not existing_ids <= text_speakers(script):
                raise H3Error("The writer removed or renumbered existing script speaker IDs.")
        if allowed is not None and {entry["speaker_id"] for entry in cast} != set(allowed):
            raise H3Error("The writer's cast does not match the selected speaker mode and reference IDs.")
        return (script, cast, sounds) if return_sounds else (script, cast)
    # Compatibility for callers with older custom, plain-script system prompts.
    reply = names_to_tags(reply, speaker_names)
    cleaned_reply = strip_drama3_sound_tags(reply) if script_mode == DRAMA3_SCRIPT_BASIC else reply
    script = review(enforce_drama3_speakers(validate_drama3_script(cleaned_reply), allowed))
    sounds = [] if script_mode == DRAMA3_SCRIPT_BASIC else extract_script_sounds(script)
    return (script, [], sounds) if return_sounds else (script, [])


def apply_drama3_reply(
    reply: str,
    text: str,
    voices: Iterable[tuple[int, str]] = (),
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    return_sounds: bool = False,
    return_title: bool = False,
    speaker_names: dict[int, str] | None = None,
) -> tuple:
    """Validate a reply pasted from another chat model; the text is kept on failure.

    ``return_title`` appends the reply's title ("" when it has none or failed).
    """
    original = str(text or "")
    extra = (([],) if return_sounds else ()) + (("",) if return_title else ())
    if not str(reply or "").strip():
        return (original, f"{DRAMA3_PROMPT_FAILED} paste the writer's JSON reply first.", [], *extra)
    notes: list[str] = []
    meta: dict = {}
    try:
        allowed, _context = drama3_writer_context(voices, speaker_mode, script_mode=script_mode)
        parsed = parse_drama3_reply(
            _strip_reply_wrapping(reply),
            original,
            allowed,
            speaker_mode,
            require_cast=True,
            script_mode=script_mode,
            return_sounds=return_sounds,
            notes=notes,
            meta=meta,
            speaker_names=speaker_names,
        )
    except H3Error as exc:
        return (original, f"{DRAMA3_PROMPT_FAILED} {exc} Your text was kept.", [], *extra)
    status = _with_writer_notes("Applied the pasted Drama 3 script.", notes)
    title = (meta.get("title", ""),) if return_title else ()
    if return_sounds:
        script, cast, sounds = parsed
        return script, status, cast, sounds, *title
    script, cast = parsed
    return script, status, cast, *title


DRAMA3_DIRECTION_PROMPT = "Drama 3 direction"


def drama3_direction_request(script: str, cast: Iterable[dict], script_mode: str) -> str:
    """The director's request: the cast for context and the script to direct."""
    people = "\n".join(f"- {entry['name']}: {' '.join(str(entry.get('description', '')).split())[:300]}"
                       for entry in cast)
    return (f"Script mode: {script_mode}\n"
            f"Cast (the voices are fixed; direct only how each line is delivered):\n{people}\n\n"
            f"Script to direct (return every line, in order, changing nothing but inserted cues and tags):\n"
            f"{script}")


def direct_drama3_script(script: str, cast: list[dict], script_mode: str, ask) -> tuple[str, str]:
    """(script, note): an under-directed tagged script with a director's cues merged in.

    ``ask(request_text)`` returns the director's (reply, status). A script already
    directed enough, a failed request, or a reply that changes words keeps its lines.
    """
    from h3_app.drama3_lint import direction_share, merge_direction, under_directed
    from h3_app.workflows.drama3 import names_to_tags, tags_to_names

    if not under_directed(script):
        return script, ""
    names = {int(entry["speaker_id"]): entry["name"] for entry in cast}
    try:
        reply, status = ask(drama3_direction_request(tags_to_names(script, names), cast, script_mode))
    except H3Error as exc:
        return script, f"The direction pass failed ({exc}); the writer's cues were kept."
    if status.startswith(DRAMA3_PROMPT_FAILED):
        return script, f"The direction pass failed ({status[len(DRAMA3_PROMPT_FAILED):].strip()}); the writer's cues were kept."
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", reply, flags=re.S)
    directed, taken, refused = merge_direction(script, names_to_tags(fenced.group(1) if fenced else reply, names))
    if not taken:
        return script, "The direction pass changed the script's words or lines, so the writer's cues were kept."
    share = direction_share(directed)
    refusal = f"; {refused} line(s) that changed words were kept as written" if refused else ""
    return directed, f"A direction pass added cues to {taken} line(s), {share[0]} of {share[1]} now directed{refusal}."


def _with_writer_notes(status: str, notes: list[str]) -> str:
    return f"{status} Writer check: {' '.join(notes)}" if notes else status


def enhance_drama3_text(
    text: str,
    model: str,
    temporary_api_key: str,
    voices: Iterable[tuple[int, str]] = (),
    backend: str = "Gemini",
    lightning_api_key: str = "",
    lightning_model: str = DEFAULT_LIGHTNING_PROMPT_MODEL,
    speaker_mode: str = DRAMA3_SPEAKER_AUTO,
    *,
    runtime: RuntimeConfig,
    return_cast: bool = False,
    cast_context: str = "",
    script_mode: str = DRAMA3_SCRIPT_BASIC,
    sound_context: str = "",
    return_sounds: bool = False,
    return_title: bool = False,
    speaker_names: dict[int, str] | None = None,
) -> tuple:
    """Rewrite a script and validate its cast as one result before updating the UI.

    ``return_title`` appends the writer's title ("" when it gave none or failed).
    """
    original = str(text or "")
    voices = tuple(voices)
    meta: dict = {}

    def result(script, status, cast=None, sounds=None):
        title = (meta.get("title", ""),) if return_title else ()
        if return_sounds:
            return (script, status, cast or [], sounds or [], *title)
        return (script, status, cast or [], *title) if return_cast else (script, status, *title)

    if not original.strip():
        return result(original, f"{DRAMA3_PROMPT_FAILED} write the speech text or a scene brief first.")
    try:
        allowed, context = drama3_writer_context(
            voices,
            speaker_mode,
            cast_context,
            script_mode=script_mode,
            sound_context=sound_context,
        )
    except H3Error as exc:
        return result(original, f"{DRAMA3_PROMPT_FAILED} {exc}")
    script, status = _enhance_prompt_from_media(
        prompt=original,
        model=model,
        temporary_api_key=temporary_api_key,
        target="Drama 3",
        system_path=runtime.prompt_systems["Drama 3"],
        backend=backend,
        lightning_api_key=lightning_api_key,
        lightning_model=lightning_model,
        media_values=(),
        context=context,
    )
    if status.startswith(DRAMA3_PROMPT_FAILED):
        return result(original, status)
    notes: list[str] = []
    direction_prompt = getattr(runtime, "prompt_systems", {}).get(DRAMA3_DIRECTION_PROMPT)

    def ask_director(request_text: str) -> tuple[str, str]:
        return _enhance_prompt_from_media(
            prompt=request_text, model=model, temporary_api_key=temporary_api_key,
            target="Drama 3 direction", system_path=direction_prompt, backend=backend,
            lightning_api_key=lightning_api_key, lightning_model=lightning_model,
            media_values=(), context="", request_text=request_text,
        )

    try:
        parsed = parse_drama3_reply(
            script,
            original,
            allowed,
            speaker_mode,
            require_cast=return_cast or return_sounds,
            script_mode=script_mode,
            return_sounds=return_sounds,
            notes=notes,
            meta=meta,
            speaker_names=speaker_names,
            # A second writer call that only adds delivery cues to an under-directed script.
            direct=(lambda script, cast: direct_drama3_script(script, cast, script_mode, ask_director))
            if direction_prompt is not None and Path(direction_prompt).is_file() else None,
        )
        status = _with_writer_notes(status.replace(" prompt ", " script ", 1), notes)
        if return_sounds:
            script, cast, sounds = parsed
            return result(script, status, cast, sounds)
        script, cast = parsed
        return result(script, status, cast)
    except H3Error as exc:
        meta.clear()  # a rejected reply's title does not apply
        return result(original, f"{DRAMA3_PROMPT_FAILED} {exc} Your text was kept; retry.")



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
