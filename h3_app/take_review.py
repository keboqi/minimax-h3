"""Gemini-only multimodal take review (review + finetuned prompt) and Gemini Decision take scoring.

Both actions consume the exact same ``TakeReviewInput`` (generated take video+audio, generation
prompt, active reference media, scene context, and optional user advice):

- ``review_take_with_gemini`` uploads the take and references to Gemini (the only backend that
  natively inspects both video and audio) and returns structured feedback plus a production-ready
  finetuned MiniMax H3 prompt following ``prompts/prompt.txt``.
- ``score_take`` / ``schedule_take_score`` pass the same inputs to ``gemini-decision-1`` via
  ``h3_app.media_review`` to produce calibrated overall and per-aspect take scores plus library tags.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import requests

from h3_app import media_review
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS
from h3_app.config import RuntimeConfig
from h3_app.errors import H3Error
from h3_app.llm import decisions, gemini, media_type, repair_json, strip_code_fence
from h3_app.prompt_writers.common import read_system_prompt

VALID_AREAS = frozenset({"prompt", "visual", "motion", "audio", "reference", "timing", "advice"})
VALID_SEVERITIES = frozenset({"major", "minor"})

TAKE_REVIEW_INSTRUCTIONS = """Take review and prompt fine-tuning task (overrides Section 31 output packaging):
You are reviewing an ALREADY-GENERATED MiniMax H3 video take (including its synchronized audio track)
against the generation prompt, any attached reference media (first/last frames, reference pictures,
reference videos, reference audios / voice-timbre clips), and optional user advice.

Watch the generated video carefully AND listen closely to its audio (speech intelligibility, wording,
voice timbre, lip-sync, sound effects, ambience, music, and audio-visual sync). Compare what actually
appeared and sounded in the take against:
1. The user's explicit advice / requested fixes (highest priority when supplied).
2. Every attached reference file (exact first/last frame matching, character/object/scene identity,
   reference video motion/style, and reference audio / voice timbre).
3. The generation prompt (subjects, actions, camera behavior, cuts, timing, dialogue, and sound).
4. Visual, physical motion, and audio quality (artifacts, anatomical glitches, identity drift, pacing,
   unwanted cuts, audio distortion, or desync).

Then produce BOTH:
- A concise, honest, specific review grounded only in what you actually observe in the take.
- A complete, production-ready `finetuned_prompt` following ALL MiniMax H3 rules above (correct mode,
  exact alignment instruction line for I2VA/FL2VA/L2VA, 6-section format for Ref2VA, `<d>[Lang] ...</d>`
  for dialogue, `overall_soundscape:`, `non_diegetic_music:`).

Prompt fine-tuning rules:
- Preserve what worked well in the take and all authoritative user requirements (dialogue wording and
  language, reference labels, duration, and mode). Do NOT invent new reference labels or renumber tags.
- For First / last frame tasks with `<Audio N>` voice references, keep the I2VA/FL2VA/L2VA frame task
  and preserve valid `<Audio N>` speaker assignments; do NOT convert the prompt to Ref2VA.
- Make targeted, concrete fixes in `finetuned_prompt` for each observed flaw and each user advice item
  (e.g., clarify motion trajectory, simplify overloaded actions, tighten camera cues, anchor identity or
  props, add a selective negative constraint against a concrete artifact you saw, or refine sound/voice
  delivery).
- Do not include Markdown fences, titles, or explanations inside `finetuned_prompt`.

Return ONLY a single JSON object (no Markdown fences, no prose outside the JSON) with these exact keys:
{
  "summary": "2-3 sentence overall assessment of visuals, motion, audio, prompt/reference fidelity, and user advice.",
  "strengths": ["Concrete aspect that worked well and should be preserved."],
  "issues": [
    {
      "area": "prompt | visual | motion | audio | reference | timing | advice",
      "severity": "major | minor",
      "timestamp": "MM:SS.m or MM:SS.m-MM:SS.m when localized, else empty string",
      "detail": "Specific observed issue in the generated take."
    }
  ],
  "prompt_changes": ["Concise note explaining one concrete adjustment made in finetuned_prompt."],
  "finetuned_prompt": "The complete revised production-ready MiniMax H3 prompt."
}
"""


@dataclass(frozen=True)
class TakeReviewInput:
    """Shared input bundle consumed by both Gemini LLM take review and Gemini Decision take scoring."""

    take_path: Path
    prompt: str
    mode: str = "Text to video"
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    references: tuple[tuple[str, Path], ...] = ()
    advice: str = ""
    scene_context: str = ""

    def context_summary(self) -> str:
        parts = [f"Mode: {self.mode}"]
        if self.duration is not None:
            parts.append(f"Duration: {float(self.duration):.2f}s")
        if self.width and self.height:
            parts.append(f"Size: {int(self.width)}x{int(self.height)}")
        if self.scene_context.strip():
            parts.append(self.scene_context.strip())
        return " | ".join(parts)


def _existing_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if not value:
        return None
    candidate = Path(str(value))
    return candidate if candidate.is_file() else None


def collect_h3_references(
    mode: str,
    *,
    first_image: Any = None,
    last_image: Any = None,
    ref_images: Iterable[Any] = (),
    ref_videos: Iterable[Any] = (),
    ref_audios: Iterable[Any] = (),
    fl2va_audios: Iterable[Any] = (),
    cast_names: Sequence[str] = (),
) -> tuple[tuple[str, Path], ...]:
    """Collect active, existing reference media files with exact MiniMax H3 labels."""
    collected: list[tuple[str, Path]] = []
    if mode == "First / last frame":
        first = _existing_path(first_image)
        last = _existing_path(last_image)
        if first is not None:
            collected.append(("<Picture 1> (first frame)", first))
        if last is not None:
            number = 2 if first is not None else 1
            collected.append((f"<Picture {number}> (last frame)", last))
        for index, raw in enumerate(fl2va_audios, 1):
            path = _existing_path(raw)
            if path is not None:
                collected.append((f"<Audio {index}> (voice reference)", path))
        return tuple(collected)
    if mode != "Reference media":
        return ()
    for index, raw in enumerate(ref_images, 1):
        path = _existing_path(raw)
        if path is None:
            continue
        if index <= len(cast_names) and str(cast_names[index - 1]).strip():
            label = f"<Picture {index}> (cast: {str(cast_names[index - 1]).strip()})"
        else:
            label = f"<Picture {index}>"
        collected.append((label, path))
    for index, raw in enumerate(ref_videos, 1):
        path = _existing_path(raw)
        if path is not None:
            collected.append((f"<Video {index}>", path))
    for index, raw in enumerate(ref_audios, 1):
        path = _existing_path(raw)
        if path is not None:
            collected.append((f"<Audio {index}>", path))
    return tuple(collected)


def h3_take_input(
    take_path: str | Path,
    values: Mapping[str, Any],
    *,
    advice: str = "",
) -> TakeReviewInput:
    """Build a ``TakeReviewInput`` from an H3 generation request dictionary."""
    mode = str(values.get("mode") or "Text to video")
    references = collect_h3_references(
        mode,
        first_image=values.get("first_image"),
        last_image=values.get("last_image"),
        ref_images=[values.get(f"ref_image_{i}") for i in range(1, 10)],
        ref_videos=[values.get(f"ref_video_{i}") for i in range(1, 4)],
        ref_audios=[values.get(f"ref_audio_{i}") for i in range(1, 4)],
        fl2va_audios=[values.get(f"fl2va_audio_{i}") for i in range(1, 4)],
    )
    duration = values.get("duration")
    width = values.get("width")
    height = values.get("height")
    return TakeReviewInput(
        take_path=Path(take_path),
        prompt=str(values.get("prompt") or "").strip(),
        mode=mode,
        duration=float(duration) if duration is not None else None,
        width=int(width) if width else None,
        height=int(height) if height else None,
        references=references,
        advice=str(advice or "").strip(),
    )


def _resolve_cast_members(
    board: Mapping[str, Any],
    raw_cast: Any,
    cast_dir: Path | None,
) -> list[tuple[str, Path]]:
    """Resolve recorded take cast entries ``[[name, filename], ...]`` or ``[name, ...]`` to paths."""
    by_name = {
        str(member.get("name")): Path(str(member.get("image")))
        for member in (board.get("cast") or ())
        if member.get("name") and member.get("image")
    }
    by_filename = {path.name: path for path in by_name.values()}
    members: list[tuple[str, Path]] = []
    for entry in raw_cast or ():
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            name, filename = str(entry[0]).strip(), str(entry[1]).strip()
        else:
            name, filename = str(entry).strip(), ""
        if not name:
            continue
        candidate: Path | None = None
        if filename and cast_dir is not None and (cast_dir / filename).is_file():
            candidate = cast_dir / filename
        elif filename and filename in by_filename and by_filename[filename].is_file():
            candidate = by_filename[filename]
        elif name in by_name and by_name[name].is_file():
            candidate = by_name[name]
        if candidate is not None:
            members.append((name, candidate))
    return members


def _storyboard_take_fields(
    board: Mapping[str, Any],
    scene: Mapping[str, Any],
    take: Mapping[str, Any],
    *,
    flow: bool,
) -> dict[str, Any]:
    from h3_app import storyboard as sb

    inputs = take.get("inputs") or {}
    media = sb.scene_media(scene)
    return {
        "mode": str(inputs.get("mode") or media["mode"]),
        "flow": bool(inputs["flow"]) if "flow" in inputs else flow,
        "overlap": inputs.get("overlap") if "overlap" in inputs else (scene.get("overlap") if flow else None),
        "seconds": int(inputs.get("seconds") or scene.get("seconds") or 5),
        "prompt": str(inputs.get("prompt") if inputs.get("prompt") is not None else scene.get("prompt") or "").strip(),
        "first_image": inputs["first_image"] if "first_image" in inputs else media["first_image"],
        "last_image": inputs["last_image"] if "last_image" in inputs else media["last_image"],
        "ref_images": list(inputs["ref_images"]) if "ref_images" in inputs else list(media["ref_images"]),
        "ref_videos": list(inputs["ref_videos"]) if "ref_videos" in inputs else list(media["ref_videos"]),
        "ref_audios": list(inputs["ref_audios"]) if "ref_audios" in inputs else list(media["ref_audios"]),
        "cast": inputs["cast"] if "cast" in inputs else scene.get("cast"),
    }


def storyboard_take_ref_count(board: Mapping[str, Any], scene: Mapping[str, Any], take: Mapping[str, Any]) -> int:
    """Count the active conditioning references recorded for a generated Storyboard take."""
    from h3_app import storyboard as sb

    if sb.is_media(scene):
        return 0
    fields = _storyboard_take_fields(board, scene, take, flow=False)
    mode = fields["mode"]
    if mode == sb.FRAMES:
        count = int(bool(not fields["flow"] and fields["first_image"])) + int(bool(fields["last_image"]))
        return count + len([item for item in fields["ref_audios"] if item])
    if mode == sb.REFERENCES:
        pictures = min(
            sb.MAX_REFERENCE_IMAGES,
            len(fields["cast"] or ()) + len([item for item in fields["ref_images"] if item]),
        )
        return (
            pictures
            + len([item for item in fields["ref_videos"] if item])
            + len([item for item in fields["ref_audios"] if item])
        )
    return 0


def storyboard_take_input(
    board: Mapping[str, Any],
    scene_index: int,
    take_id: str,
    *,
    cast_dir: Path | None = None,
    advice: str = "",
) -> tuple[TakeReviewInput, list[str]]:
    """Build ``(TakeReviewInput, cast_names)`` for a generated Storyboard take."""
    from h3_app import storyboard as sb

    scene = board["scenes"][scene_index]
    if sb.is_media(scene):
        raise H3Error("Review and scoring apply to generated takes, not Video from Media scenes.")
    take = (board.get("takes") or {}).get(take_id)
    if take is None or take_id not in scene.get("takes", ()):
        raise H3Error("That take is not in this scene.")
    video_path = Path(str(take.get("video") or ""))
    if not video_path.is_file():
        raise H3Error("This take video is no longer available on disk.")

    fields = _storyboard_take_fields(board, scene, take, flow=sb.flows_from_previous(board, scene_index))
    mode = fields["mode"]
    flow = fields["flow"]
    raw_prompt = fields["prompt"]

    cast_names: list[str] = []
    cast_images: list[Path] = []
    prompt = raw_prompt
    if mode == sb.REFERENCES:
        resolved_cast = _resolve_cast_members(board, fields["cast"], cast_dir)[:sb.MAX_REFERENCE_IMAGES]
        cast_names = [name for name, _ in resolved_cast]
        cast_images = [path for _, path in resolved_cast]
        if cast_names:
            tags = " ".join(f"{name} is <Picture {number}>." for number, name in enumerate(cast_names, 1))
            prompt = f"{tags} {raw_prompt}".strip()

    references = collect_h3_references(
        mode,
        first_image=None if flow else fields["first_image"],
        last_image=fields["last_image"],
        ref_images=[*cast_images, *fields["ref_images"]][:sb.MAX_REFERENCE_IMAGES] if mode == sb.REFERENCES else (),
        ref_videos=fields["ref_videos"] if mode == sb.REFERENCES else (),
        ref_audios=fields["ref_audios"] if mode == sb.REFERENCES else (),
        fl2va_audios=fields["ref_audios"] if mode == sb.FRAMES else (),
        cast_names=cast_names,
    )

    context_parts = [f"Storyboard scene {scene_index + 1} ({scene.get('title') or 'Untitled scene'})"]
    if flow:
        overlap_s = (int(fields["overlap"]) / 24.0) if fields["overlap"] else 1.6
        context_parts.append(
            f"Continues from scene {scene_index}; the opening {overlap_s:.2f}s repeats the end of "
            f"scene {scene_index} for continuity (do not treat this intentional overlap as a flaw "
            f"unless the handoff breaks)."
        )
    profile = board.get("profile") or {}
    return (
        TakeReviewInput(
            take_path=video_path,
            prompt=prompt,
            mode=mode,
            duration=float(fields["seconds"]),
            width=int(profile["width"]) if profile.get("width") else None,
            height=int(profile["height"]) if profile.get("height") else None,
            references=references,
            advice=str(advice or "").strip(),
            scene_context=" ".join(context_parts),
        ),
        cast_names,
    )


def strip_cast_tags(prompt: str, cast_names: Sequence[str]) -> str:
    """Remove generated ``Name is <Picture N>.`` prefixes before saving back to a Storyboard scene."""
    cleaned = str(prompt or "").strip()
    for number, name in enumerate(cast_names, 1):
        cleaned = cleaned.replace(f"{name} is <Picture {number}>.", "").strip()
    return cleaned


def score_take(
    take_input: TakeReviewInput,
    *,
    api_key: str = "",
    save: bool = True,
) -> dict[str, Any]:
    """Score a generated take with Gemini Decision using take + prompt + refs + optional advice."""
    if not take_input.take_path.is_file():
        raise H3Error("This take video is no longer available on disk.")
    if not decisions.enabled():
        raise H3Error("Gemini Decision scoring is turned off (H3_DECISION=0).")
    key = gemini.api_key(api_key)
    payload = media_review.review(
        take_input.take_path,
        take_input.prompt,
        key=key,
        references=take_input.references,
        advice=take_input.advice,
        context=take_input.context_summary(),
        detailed=True,
    )
    if save:
        media_review.save_decision(take_input.take_path, payload)
        return media_review.public(take_input.take_path) or payload
    return payload


def schedule_take_score(
    take_input: TakeReviewInput,
    *,
    api_key: str = "",
    enabled: bool = True,
) -> bool:
    """Score a newly generated take in the background using its prompt and active references."""
    return media_review.schedule(
        take_input.take_path,
        take_input.prompt,
        api_key=api_key,
        enabled=enabled,
        references=take_input.references,
        advice=take_input.advice,
        context=take_input.context_summary(),
        detailed=True,
    )


def _normalize_review_payload(raw: Any, *, model: str, advice: str, ref_count: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise H3Error("Gemini returned an unexpected review format.")
    summary = str(raw.get("summary") or "").strip()
    finetuned_prompt = strip_code_fence(str(raw.get("finetuned_prompt") or "")).strip()
    if not finetuned_prompt:
        raise H3Error("Gemini review did not include a finetuned prompt.")

    strengths = [
        str(item).strip()
        for item in (raw.get("strengths") or ())
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]
    prompt_changes = [
        str(item).strip()
        for item in (raw.get("prompt_changes") or ())
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]
    issues: list[dict[str, str]] = []
    for item in raw.get("issues") or ():
        if isinstance(item, Mapping):
            detail = str(item.get("detail") or item.get("message") or "").strip()
            if not detail:
                continue
            area = str(item.get("area") or "prompt").strip().lower()
            if area not in VALID_AREAS:
                area = "prompt"
            severity = str(item.get("severity") or "minor").strip().lower()
            if severity not in VALID_SEVERITIES:
                severity = "minor"
            timestamp = str(item.get("timestamp") or "").strip()
            issues.append({"area": area, "severity": severity, "timestamp": timestamp, "detail": detail})
        elif isinstance(item, str) and item.strip():
            issues.append({"area": "prompt", "severity": "minor", "timestamp": "", "detail": item.strip()})

    if not summary:
        summary = "Take reviewed and prompt refined by Gemini."
    return {
        "status": "done",
        "model": model,
        "advice": advice,
        "ref_count": int(ref_count),
        "summary": summary,
        "strengths": strengths,
        "issues": issues,
        "prompt_changes": prompt_changes,
        "finetuned_prompt": finetuned_prompt,
        "finished": time.time(),
    }


def _parse_review_json(reply: str) -> Any:
    cleaned = strip_code_fence(reply)
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        repaired = repair_json(cleaned)
        if repaired is not None:
            return repaired
        raise H3Error("Gemini returned an unreadable review reply. Try again.")


def _build_review_brief(take_input: TakeReviewInput, attached_refs: Sequence[tuple[str, Path]]) -> str:
    brief_lines = [
        "Review the generated MiniMax H3 video take (including its audio) and produce a finetuned prompt.",
        f"UI mode: {take_input.mode}",
    ]
    if take_input.duration is not None:
        brief_lines.append(f"Requested duration: {float(take_input.duration):.2f} seconds")
    if take_input.width and take_input.height:
        brief_lines.append(f"Requested output: {int(take_input.width)}x{int(take_input.height)}")
    if take_input.scene_context.strip():
        brief_lines.append(f"Scene context: {take_input.scene_context.strip()}")
    if attached_refs:
        ref_summary = ", ".join(label for label, _ in attached_refs)
        brief_lines.append(f"Attached references ({len(attached_refs)}): {ref_summary}")
    else:
        brief_lines.append("Attached references: none")
    brief_lines.append(
        "Generation prompt used for this take:\n"
        + (take_input.prompt or "(No prompt supplied.)")
    )
    if take_input.advice.strip():
        brief_lines.append(
            "User advice / requested improvements (prioritize these):\n"
            + take_input.advice.strip()
        )
    return "\n".join(brief_lines)


def review_take_with_gemini(
    take_input: TakeReviewInput,
    *,
    model: str = "",
    api_key: str = "",
    runtime: RuntimeConfig,
    save: bool = True,
    transform_prompt: Any = None,
) -> dict[str, Any]:
    """Review a generated take (video+audio) with Gemini and return review + finetuned prompt.

    Only Gemini is supported because multimodal video + audio review requires native audiovisual
    understanding. ``transform_prompt``, when provided, transforms ``finetuned_prompt`` (for
    example restoring sparse reference tags or stripping Storyboard cast prefixes) before saving.
    """
    selected_model = str(model or "").strip() or DEFAULT_GEMINI_PROMPT_MODEL
    if selected_model not in GEMINI_PROMPT_MODELS:
        raise H3Error(f"Unsupported Gemini review model: {selected_model}")
    if not take_input.take_path.is_file():
        raise H3Error("This take video is no longer available on disk.")
    if not media_type(take_input.take_path).startswith("video/"):
        raise H3Error(f"{take_input.take_path.name} is not a video file.")

    key = gemini.api_key(api_key)
    guidelines = read_system_prompt(runtime.prompt_system_path, "MiniMax H3")
    system_prompt = guidelines + "\n\n" + TAKE_REVIEW_INSTRUCTIONS

    attached_refs = [
        (label, ref_path)
        for label, ref_path in take_input.references
        if ref_path.is_file()
    ]
    parts: list[dict[str, Any]] = [{"text": _build_review_brief(take_input, attached_refs)}]
    try:
        with gemini.Session(key) as session:
            uploaded: dict[Path, dict[str, Any]] = {}

            def upload_part(path: Path) -> dict[str, Any]:
                resolved = path.resolve()
                part = uploaded.get(resolved)
                if part is None:
                    part = session.file_part(path)
                    uploaded[resolved] = part
                return part

            for label, ref_path in attached_refs:
                parts.append({"text": f"Reference media — {label}:"})
                parts.append(upload_part(ref_path))
            parts.append({
                "text": "Generated take to review (watch the video and listen closely to the audio):"
            })
            parts.append(upload_part(take_input.take_path))

            payload = session.generate(
                selected_model,
                parts,
                system_prompt=system_prompt,
                generation_config={
                    "temperature": 0.4,
                    "maxOutputTokens": 16384,
                    "responseMimeType": "application/json",
                },
                action="take review",
            )
    except requests.RequestException as exc:
        raise H3Error(f"Gemini take review could not be reached: {exc}") from exc

    reply = gemini.reply_text(payload)
    if not reply:
        raise H3Error(f"Gemini returned no take review ({gemini.empty_reply_reason(payload)}).")
    parsed = _parse_review_json(reply)
    result = _normalize_review_payload(
        parsed, model=selected_model, advice=take_input.advice.strip(), ref_count=len(attached_refs)
    )
    if transform_prompt is not None:
        result["finetuned_prompt"] = str(transform_prompt(result["finetuned_prompt"])).strip()
        if not result["finetuned_prompt"]:
            raise H3Error("Gemini review did not produce a valid finetuned prompt.")
    if save:
        media_review.save_llm(take_input.take_path, result)
    return result
