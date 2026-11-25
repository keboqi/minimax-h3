"""Prompt-fit scores, multi-aspect take scores, Media library tags, and persisted LLM take reviews.

After a video, image or song is saved, ``schedule`` sends the file, its prompt, and any active
reference media to Gemini Decision in a background thread, so generation never waits for it. One
request evaluates rubric scores together with a set of tag questions from a fixed vocabulary (shot,
setting, mood, style…; genre, vocals, tempo… for songs). The answer is kept beside the file as
``<name>.review.json`` and the tags are added to the file's Media library tags, so the library
search finds them.

Gemini LLM take reviews (review feedback and finetuned prompts) are persisted in the same sidecar
under the ``llm`` key so deletion and export automatically keep them in sync without overwriting
Decision scores.

Everything in background scheduling is optional: no Gemini key, H3_DECISION=0, an oversized file or
a failed request only means no score is shown.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

import requests

from h3_app.errors import H3Error
from h3_app.llm import decisions, gemini
from h3_app.workspace_activity import workspace_task

SUFFIX = ".review.json"
PENDING, DONE, FAILED = "pending", "done", "failed"
# A pending review older than this was lost with its process.
STALE_SECONDS = 15 * 60
# A choice tag is kept at or above this probability; a yes/no tag at or above YES_TAG.
CHOICE_TAG = 0.5
YES_TAG = 0.6
MAX_PROMPT_CHARS = 6000
MAX_ADVICE_CHARS = 2000
MAX_CONTEXT_CHARS = 2000
# At most this many reviews run at once; more wait their turn.
_slots = threading.BoundedSemaphore(int(os.getenv("H3_REVIEW_WORKERS", "3") or 3))
_store_lock = threading.RLock()

_DECISION_KEYS = (
    "status", "started", "finished", "fit", "label", "score", "score_label",
    "scores", "tags", "ref_count", "advice", "model", "error",
)

FIT_LEVELS = (
    "Ignores the prompt: the main subject or action is missing or wrong.",
    "Matches a little: some named elements appear, but most of the request is missing.",
    "Matches about half: the main subject is right, but important details or actions are wrong or missing.",
    "Matches most of the prompt: only minor details differ.",
    "Matches the prompt fully: subject, action, setting and style are as described.",
)
FIT_LABELS = ("Poor fit", "Weak fit", "Partial fit", "Good fit", "Close fit")

REFERENCE_LEVELS = (
    "Ignores the references: identity, anchor frames, style, motion or voice do not match at all.",
    "Weak match: only vague similarities to the references with major drift or mismatched anchors.",
    "Partial match: recognizable elements from the references, but noticeable identity, frame, or voice drift.",
    "Good match: closely preserves reference identities, frames, style, and voices with only minor drift.",
    "Faithful match: accurately preserves all reference identities, anchor frames, motion, and voice timbre.",
)

VISUAL_LEVELS = (
    "Unusable visuals: severe warping, broken anatomy, flicker, or unreadable composition.",
    "Weak visuals: distracting artifacts, stiff or unnatural motion, or noticeable visual instability.",
    "Acceptable visuals: coherent overall, with some noticeable motion or rendering artifacts.",
    "Good visuals: clean imagery, believable motion, and consistent lighting with only minor flaws.",
    "Strong visuals: sharp, stable, physically plausible motion and cohesive cinematic quality.",
)

AUDIO_LEVELS = (
    "Broken audio: severe distortion, missing required speech/sound, or jarring desynchronization.",
    "Weak audio: muffled or unnatural voice/effects, poor lip-sync, or awkward sound mismatches.",
    "Acceptable audio: intelligible and mostly aligned, with noticeable roughness or flat soundscape.",
    "Good audio: clear speech/sounds well synchronized to on-screen actions with fitting ambience.",
    "Strong audio: natural, crisp dialogue/lip-sync, expressive sound effects, and balanced mix.",
)

STYLE_LEVELS = (
    "Clashing style: wrong aesthetic, broken palette, or jarring visual inconsistencies across the shot.",
    "Weak style: inconsistent lighting, flat look, or only faint resemblance to the requested aesthetic.",
    "Acceptable style: generally coherent aesthetic with minor lighting, color, or texture drift.",
    "Good style: cohesive visual aesthetic, atmospheric lighting, and consistent color grading.",
    "Strong style: striking, cohesive art direction, lighting mood, and palette throughout.",
)

CAMERA_LEVELS = (
    "Broken camera: chaotic framing, cut-off subjects, jarring unwanted cuts, or nauseating camera jitter.",
    "Weak camera: awkward framing, stiff or erratic camera movement, or weak shot composition.",
    "Acceptable camera: readable framing and basic camera motion with minor compositional awkwardness.",
    "Good camera: well-composed framing and smooth, purposeful camera movement without jarring cuts.",
    "Strong camera: cinematic composition, deliberate framing, and fluid, well-motivated camera work.",
)

ACTION_LEVELS = (
    "Broken action: frozen subjects, wrong actions, or severe physics and motion breakdowns.",
    "Weak action: stiff, sluggish, or unnatural subject movement that misses key beats of the action.",
    "Acceptable action: main action occurs with passable pacing, though motion feels somewhat mechanical.",
    "Good action: believable, expressive subject motion and natural pacing matching the prompt.",
    "Strong action: fluid, physically grounded, expressive action and crisp timing on every beat.",
)

CONTINUITY_LEVELS = (
    "Broken continuity: severe identity morphing, teleporting props, or broken scene handoff.",
    "Weak continuity: noticeable drift in face, clothing, objects, or background across frames.",
    "Acceptable continuity: core subject remains recognizable with minor frame-to-frame prop or detail drift.",
    "Good continuity: consistent character identity, wardrobe, props, and environment across the clip.",
    "Strong continuity: rock-solid temporal stability in identity, props, background, and scene continuity.",
)

SPEECH_LEVELS = (
    "Broken speech: unintelligible garbled speech, wrong dialogue words, severe lip-sync failure, or unwanted hallucinated voices.",
    "Weak speech: slurred or robotic delivery, noticeable lip-sync lag, or awkward vocal cadence.",
    "Acceptable speech: intelligible words with minor lip-sync or vocal naturalness flaws.",
    "Good speech: clear, well-timed dialogue and convincing lip-sync (or clean, speech-free audio when no speech is requested).",
    "Strong speech: crisp, expressive, natural vocal delivery with tight lip-sync (or completely clean speech-free audio when no speech is requested).",
)

# (Decision answer key, public payload key, rubric question, ordered rubric levels)
VIDEO_SCORE_ASPECTS = (
    ("visual_quality", "visuals",
     "How clean, stable, and physically natural are the visuals and motion in the generated take?", VISUAL_LEVELS),
    ("style_consistency", "style",
     "How cohesive and faithful is the visual style, lighting mood, color palette, and art direction?", STYLE_LEVELS),
    ("camera_composition", "camera",
     "How strong are the framing, shot composition, and camera movement control (without unwanted cuts or jitter)?",
     CAMERA_LEVELS),
    ("action_motion", "action",
     "How accurate, fluid, physically plausible, and well-paced are the subject actions and motion?", ACTION_LEVELS),
    ("temporal_continuity", "continuity",
     "How consistent are character identity, wardrobe, props, background, and temporal continuity across the take?",
     CONTINUITY_LEVELS),
    ("audio_quality", "audio",
     "How natural, clear, and well-synchronized is the audio in the generated take "
     "(sound effects, ambience, music, and overall mix)?", AUDIO_LEVELS),
    ("speech_lipsync", "speech",
     "How clear, accurate, and tightly lip-synced is any requested speech or vocal delivery "
     "(score high if no speech was requested and none is hallucinated)?", SPEECH_LEVELS),
)

ADVICE_LEVELS = (
    "Fails the user's advice: the requested fix or preference is completely ignored or contradicted.",
    "Barely addresses the user's advice: small hints, but the main requested point is not met.",
    "Partially satisfies the user's advice: addresses part of it but leaves key concerns unresolved.",
    "Mostly satisfies the user's advice: meets the main request with only minor room to improve.",
    "Fully satisfies the user's advice: directly and convincingly fulfills every requested note.",
)

OVERALL_LEVELS = (
    "Unusable take: major prompt, reference, visual, or audio failures; must be regenerated.",
    "Weak take: significant issues detract from the shot; needs substantial prompt/setting changes.",
    "Usable draft: core idea is visible, but clear flaws remain in fidelity, motion, or audio.",
    "Good take: strong alignment with prompt and references, clean video and audio, minor blemishes only.",
    "Excellent take: nails prompt, references, user advice, visual fidelity, and synchronized audio.",
)
OVERALL_LABELS = ("Poor take", "Weak take", "Fair take", "Good take", "Strong take")

# Tag questions: (key, question, options or None for yes/no, the tag a yes adds). "none" options add no tag.
VISUAL_TAGS = (
    ("shot", "What is the main shot type?",
     {"close-up": "A face or object fills the frame.", "medium shot": "People from the waist up, or a mid-sized scene.",
      "wide shot": "Whole bodies or a full location.", "aerial": "Seen from high above.",
      "none": "None of these fits."}),
    ("setting", "Where does it take place?",
     {"indoors": "Inside a building or vehicle.", "city": "Outdoors in a town or city.",
      "nature": "Outdoors in nature: countryside, forest, sea, mountains or sky.",
      "studio": "A plain backdrop, studio or abstract space.", "none": "Cannot tell."}),
    ("time", "What time of day does it look like?",
     {"daytime": "", "night": "", "golden hour": "Sunrise or sunset light.", "none": "Cannot tell or not applicable."}),
    ("mood", "What is the overall mood?",
     {"calm": "", "cheerful": "", "dramatic": "Tense or intense.", "sad": "", "eerie": "Dark or unsettling.",
      "energetic": "Fast and lively.", "none": "No clear mood."}),
    ("style", "What is the visual style?",
     {"photoreal": "Looks like real camera footage or a photo.", "animation": "Cartoon or 2D animation.",
      "anime": "Japanese anime style.", "3d render": "CGI or 3D rendering.",
      "illustration": "Painting, drawing or illustration.", "none": "Something else."}),
    ("people", "How many people are shown?",
     {"no people": "", "one person": "", "two people": "", "group": "Three or more people."}),
    ("animals", "Are animals a main subject?", None, "animals"),
    ("text", "Is legible text or a logo prominent?", None, "text"),
)
VIDEO_TAGS = VISUAL_TAGS + (
    ("speech", "Can someone be heard speaking?", None, "dialogue"),
    ("music", "Is music playing?", None, "music"),
    ("motion", "How much does the camera or scene move?",
     {"static": "Almost still.", "gentle motion": "", "fast motion": "Quick action or camera moves."}),
)
SONG_TAGS = (
    ("genre", "What is the main genre?",
     {"pop": "", "rock": "", "hip hop": "Rap or hip hop.", "electronic": "EDM, house, synth music.",
      "r&b": "R&B or soul.", "folk": "Folk, acoustic or country.", "jazz": "", "classical": "Orchestral or classical.",
      "ballad": "A slow emotional song.", "ambient": "Ambient or soundtrack.", "none": "Something else."}),
    ("vocals", "What vocals are there?",
     {"male vocals": "", "female vocals": "", "duet": "Two voices.", "choir": "Many voices.",
      "instrumental": "No vocals."}),
    ("mood", "What is the mood?",
     {"happy": "", "sad": "", "romantic": "", "energetic": "", "calm": "", "dark": "", "epic": "",
      "none": "No clear mood."}),
    ("tempo", "What is the tempo?", {"slow tempo": "", "mid tempo": "", "fast tempo": ""}),
)


def review_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + SUFFIX)


def _read_raw(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(review_path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = review_path(path)
    scratch = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        scratch.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        # On Windows a page reading the review at this moment makes the replace fail; try again briefly.
        for attempt in range(20):
            try:
                os.replace(scratch, destination)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        scratch.unlink(missing_ok=True)


def _write_decision(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Replace Decision review fields while keeping any saved LLM review intact."""
    with _store_lock:
        existing = _read_raw(path)
        merged = {key: value for key, value in existing.items() if key not in _DECISION_KEYS}
        merged.update(payload)
        _write(path, merged)


def save_decision(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist a completed Decision review and sync any extracted tags to the Media library."""
    _write_decision(path, payload)
    if payload.get("status") == DONE and payload.get("tags"):
        _add_library_tags(path, list(payload["tags"]))


def save_llm(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist a Gemini LLM take review alongside the take without altering Decision scores."""
    with _store_lock:
        existing = _read_raw(path)
        existing["llm"] = dict(payload)
        _write(path, existing)


def read(path: str | Path | None) -> dict[str, Any] | None:
    """The saved review for a file, or None; a pending one that outlived its process reads as failed."""
    if not path:
        return None
    payload = _read_raw(path)
    if not payload:
        return None
    if payload.get("status") == PENDING and time.time() - float(payload.get("started") or 0) > STALE_SECONDS:
        return {**payload, "status": FAILED, "error": "The review did not finish."}
    return payload


def public(path: str | Path | None) -> dict[str, Any] | None:
    """What a page shows for a file's Decision score: status, scores, label and tags."""
    saved = read(path)
    if saved is None or not saved.get("status"):
        return None
    shown: dict[str, Any] = {"status": saved.get("status")}
    if saved.get("status") == DONE:
        shown.update(fit=saved.get("fit"), label=saved.get("label"), tags=saved.get("tags") or [])
        for key in ("score", "score_label", "scores", "ref_count", "advice", "model", "finished"):
            if key in saved and saved[key] is not None:
                shown[key] = saved[key]
    elif saved.get("status") == FAILED:
        shown["error"] = saved.get("error") or "The review failed."
    return shown


def public_llm(path: str | Path | None) -> dict[str, Any] | None:
    """What a page shows for a take's persisted Gemini LLM review, or None."""
    saved = read(path)
    if not saved:
        return None
    llm = saved.get("llm")
    return dict(llm) if isinstance(llm, dict) else None


def kind_of(path: str | Path) -> str | None:
    mime = decisions.media_type(Path(path))
    kind = mime.split("/", 1)[0]
    return kind if kind in ("video", "image", "audio") else None


def _questions(
    kind: str,
    prompt: str,
    lyrics: str,
    edited: bool,
    *,
    has_refs: bool = False,
    advice: str = "",
    detailed: bool = False,
) -> tuple[dict[str, dict], tuple]:
    if kind == "audio":
        fit = (f"How well does this song match its brief?\nbrief: {prompt}"
               + (f"\nlyrics: {lyrics[:1500]}" if lyrics else ""))
        tags = SONG_TAGS
    else:
        noun = "edited image" if edited else ("generated take" if detailed else kind)
        fit = f"How well does the {noun} match this prompt?\nprompt: {prompt}"
        if edited:
            fit += "\nJudge whether the requested edit was made, not the whole picture."
        tags = VIDEO_TAGS if kind == "video" else VISUAL_TAGS
    questions: dict[str, dict] = {"fit": decisions.score(fit, FIT_LEVELS)}
    if detailed or has_refs or advice:
        if has_refs:
            questions["reference_fidelity"] = decisions.score(
                "How faithfully does the generated take match the attached reference media "
                "(anchor frames, character/object/scene appearance, reference video motion/style, "
                "and reference audio/voice timbre)?",
                REFERENCE_LEVELS,
            )
        if kind == "video":
            for answer_key, _score_key, question, levels in VIDEO_SCORE_ASPECTS:
                questions[answer_key] = decisions.score(question, levels)
        if advice:
            questions["advice_fit"] = decisions.score(
                f"How well does the generated take satisfy this user advice?\nuser advice: {advice}",
                ADVICE_LEVELS,
            )
        overall_parts = ["its generation prompt"]
        if has_refs:
            overall_parts.append("the attached reference media")
        if advice:
            overall_parts.append("the user's advice")
        if kind == "video":
            overall_parts.append("overall visual, style, camera, action, continuity, audio, and speech quality")
        questions["overall"] = decisions.score(
            "Rate the overall quality of the generated take considering " + ", ".join(overall_parts) + ".",
            OVERALL_LEVELS,
        )
    for entry in tags:
        name, question, options = entry[0], entry[1], entry[2]
        questions[f"tag_{name}"] = (decisions.yes_no(question) if options is None
                                    else decisions.choice(question, options))
    return questions, tags


def _tags(answers: Mapping[str, Any], tags: tuple) -> list[str]:
    found = []
    for entry in tags:
        answer = answers.get(f"tag_{entry[0]}")
        if entry[2] is None:
            probability = decisions.yes_probability(answer)
            if probability is not None and probability >= YES_TAG:
                found.append(entry[3])
        elif isinstance(answer, Mapping) and answer.get("type") == "choice":
            picked = answer.get("choice")
            probability = (answer.get("probabilities") or {}).get(picked, answer.get("confidence", 0))
            if picked and picked != "none" and float(probability or 0) >= CHOICE_TAG:
                found.append(str(picked))
    return found


def _valid_references(references: Sequence[tuple[str, str | Path]]) -> list[tuple[str, Path]]:
    valid: list[tuple[str, Path]] = []
    for label, raw_path in references or ():
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.is_file() and kind_of(candidate) is not None:
            valid.append((str(label), candidate))
    return valid


def _score_percent(answer: Mapping[str, Any] | None) -> int | None:
    fraction = decisions.score_fraction(answer)
    return round(fraction * 100) if fraction is not None else None


def _label_for(percent: int, labels: tuple[str, ...]) -> str:
    fraction = max(0.0, min(1.0, percent / 100.0))
    return labels[min(len(labels) - 1, int(round(fraction * (len(labels) - 1))))]


def _build_decision_blocks(
    stack: ExitStack,
    target: Path,
    kind: str,
    *,
    key: str,
    valid_refs: Sequence[tuple[str, Path]],
    cleaned_prompt: str,
    cleaned_advice: str,
    cleaned_context: str,
    use_detailed: bool,
) -> list[dict[str, Any]]:
    if not use_detailed:
        return [decisions.text(f"A generated {'song' if kind == 'audio' else kind}."), decisions.media(target)]

    media_paths = [ref_path for _, ref_path in valid_refs] + [target]
    total_bytes = sum(item.stat().st_size for item in media_paths)
    session = stack.enter_context(gemini.Session(key)) if total_bytes > decisions.MAX_INLINE_BYTES else None
    uploaded: dict[Path, dict[str, Any]] = {}

    def block_for(media_path: Path) -> dict[str, Any]:
        if session is None:
            return decisions.media(media_path)
        resolved = media_path.resolve()
        info = uploaded.get(resolved)
        if info is None:
            info = session.wait_until_active(session.upload(media_path))
            uploaded[resolved] = info
        return decisions.media_uri(info, media_path)

    header_lines = [f"Evaluate the generated {kind} take against its prompt and any attached reference media."]
    if cleaned_context:
        header_lines.append(f"Context: {cleaned_context}")
    if cleaned_prompt:
        header_lines.append(f"Generation prompt: {cleaned_prompt}")
    if cleaned_advice:
        header_lines.append(f"User advice / requested criteria: {cleaned_advice}")
    blocks: list[dict[str, Any]] = [decisions.text("\n".join(header_lines))]
    for label, ref_path in valid_refs:
        blocks.append(decisions.text(f"Reference media — {label}:"))
        blocks.append(block_for(ref_path))
    blocks.append(decisions.text(
        "Generated take to evaluate (watch the video and listen to its audio):"
        if kind == "video" else f"Generated {kind} to evaluate:"
    ))
    blocks.append(block_for(target))
    return blocks


def _format_decision_payload(
    answers: Mapping[str, Any],
    tags: tuple,
    *,
    use_detailed: bool,
    ref_count: int,
    cleaned_advice: str,
) -> dict[str, Any]:
    fit_pct = _score_percent(answers.get("fit"))
    overall_pct = _score_percent(answers.get("overall")) if use_detailed else None
    if fit_pct is None and overall_pct is None:
        raise H3Error("Gemini Decision did not score the result.")
    if fit_pct is None:
        fit_pct = overall_pct

    payload: dict[str, Any] = {
        "status": DONE,
        "fit": fit_pct,
        "label": _label_for(fit_pct, FIT_LABELS),
        "tags": _tags(answers, tags),
        "model": decisions.MODEL,
        "finished": time.time(),
    }
    if use_detailed:
        breakdown: dict[str, int] = {"prompt": fit_pct}
        for answer_key, score_key in (
            ("reference_fidelity", "references"),
            *((ak, sk) for ak, sk, _, _ in VIDEO_SCORE_ASPECTS),
            ("advice_fit", "advice"),
        ):
            value = _score_percent(answers.get(answer_key))
            if value is not None:
                breakdown[score_key] = value
        if overall_pct is None:
            overall_pct = round(sum(breakdown.values()) / len(breakdown))
        payload.update(
            score=overall_pct,
            score_label=_label_for(overall_pct, OVERALL_LABELS),
            scores=breakdown,
            ref_count=ref_count,
            advice=cleaned_advice,
        )
    return payload


def review(
    path: str | Path,
    prompt: str,
    *,
    key: str,
    lyrics: str = "",
    edited: bool = False,
    references: Sequence[tuple[str, str | Path]] = (),
    advice: str = "",
    context: str = "",
    detailed: bool = False,
) -> dict[str, Any]:
    """Score one file against its prompt (plus optional refs & user advice) and tag it.

    Raises ``H3Error`` on invalid files or failed Decision requests.
    """
    target = Path(path)
    kind = kind_of(target)
    if kind is None:
        raise H3Error(f"{target.name} is not a video, image or audio file.")
    cleaned_prompt = " ".join(str(prompt or "").split())[:MAX_PROMPT_CHARS]
    cleaned_advice = " ".join(str(advice or "").split())[:MAX_ADVICE_CHARS]
    cleaned_context = " ".join(str(context or "").split())[:MAX_CONTEXT_CHARS]
    valid_refs = _valid_references(references)
    use_detailed = bool(detailed or valid_refs or cleaned_advice or cleaned_context)
    questions, tags = _questions(
        kind, cleaned_prompt, str(lyrics or "").strip(), edited,
        has_refs=bool(valid_refs), advice=cleaned_advice, detailed=use_detailed,
    )

    try:
        with ExitStack() as stack:
            blocks = _build_decision_blocks(
                stack, target, kind,
                key=key, valid_refs=valid_refs, cleaned_prompt=cleaned_prompt,
                cleaned_advice=cleaned_advice, cleaned_context=cleaned_context, use_detailed=use_detailed,
            )
            answers = decisions.decide(questions, blocks, key=key, timeout=180)
    except requests.RequestException as exc:
        raise H3Error(f"Gemini Decision could not be reached: {exc}") from exc

    return _format_decision_payload(
        answers, tags, use_detailed=use_detailed, ref_count=len(valid_refs), cleaned_advice=cleaned_advice,
    )


def _add_library_tags(path: str | Path, tags: list[str]) -> None:
    """Add tags to the file's Media library tags, keeping the viewer's own and its favorite."""
    if not tags:
        return
    from h3_app.media_catalog import get_media_catalog

    catalog = get_media_catalog()
    if catalog is None:
        return
    try:
        catalog.store.add_tags(Path(path), tags)
    except (OSError, ValueError, AttributeError):
        pass


def schedule(
    path: str | Path | None,
    prompt: str,
    *,
    api_key: str = "",
    enabled: bool = True,
    lyrics: str = "",
    edited: bool = False,
    references: Sequence[tuple[str, str | Path]] = (),
    advice: str = "",
    context: str = "",
    detailed: bool = False,
) -> bool:
    """Review ``path`` in the background; True when a review was started.

    Nothing starts without a prompt, a Gemini key, or when ``enabled`` is off. A pending
    marker is written at once so pages can show that a score is coming.
    """
    if not enabled or not path or not str(prompt or "").strip() or kind_of(path) is None:
        return False
    key = decisions.key_or_none(api_key)
    if key is None or not Path(path).is_file():
        return False
    target = Path(path)
    ref_list = tuple((str(label), Path(ref_path)) for label, ref_path in (references or ()) if ref_path)
    try:
        _write_decision(target, {"status": PENDING, "started": time.time()})
    except OSError:
        return False

    def run() -> None:
        with _slots:
            try:
                payload = review(
                    target, prompt, key=key, lyrics=lyrics, edited=edited,
                    references=ref_list, advice=advice, context=context, detailed=detailed,
                )
            except Exception as exc:  # noqa: BLE001 - a review never affects the generation
                payload = {"status": FAILED, "error": str(exc)[:300], "finished": time.time()}
            try:
                _write_decision(target, payload)
            except OSError:
                return
            if payload["status"] == DONE:
                _add_library_tags(target, payload["tags"])

    threading.Thread(target=workspace_task(run), name=f"review-{target.name}", daemon=True).start()
    return True
