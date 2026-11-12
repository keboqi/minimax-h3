"""Prompt-fit scores and Media library tags for generated media, from one Gemini Decision request.

After a video, image or song is saved, ``schedule`` sends the file and the prompt that made
it to Gemini Decision in a background thread, so generation never waits for it. One
request asks how well the result fits the prompt (a 1–5 rubric) together with a set of tag
questions from a fixed vocabulary (shot, setting, mood, style…; genre, vocals, tempo… for
songs). The answer is kept beside the file as ``<name>.review.json`` and the tags are added
to the file's Media library tags, so the library search finds them.

Everything here is optional: no Gemini key, H3_DECISION=0, an oversized file or a failed
request only means no score is shown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
import uuid

from h3_app.errors import H3Error
from h3_app.llm import decisions

SUFFIX = ".review.json"
PENDING, DONE, FAILED = "pending", "done", "failed"
# A pending review older than this was lost with its process.
STALE_SECONDS = 15 * 60
# A choice tag is kept at or above this probability; a yes/no tag at or above YES_TAG.
CHOICE_TAG = 0.5
YES_TAG = 0.6
MAX_PROMPT_CHARS = 6000
# At most this many reviews run at once; more wait their turn.
_slots = threading.BoundedSemaphore(int(os.getenv("H3_REVIEW_WORKERS", "3") or 3))

FIT_LEVELS = (
    "Ignores the prompt: the main subject or action is missing or wrong.",
    "Matches a little: some named elements appear, but most of the request is missing.",
    "Matches about half: the main subject is right, but important details or actions are wrong or missing.",
    "Matches most of the prompt: only minor details differ.",
    "Matches the prompt fully: subject, action, setting and style are as described.",
)
FIT_LABELS = ("Poor fit", "Weak fit", "Partial fit", "Good fit", "Close fit")

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


def read(path: str | Path | None) -> dict[str, Any] | None:
    """The saved review for a file, or None; a pending one that outlived its process reads as failed."""
    if not path:
        return None
    try:
        payload = json.loads(review_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") == PENDING and time.time() - float(payload.get("started") or 0) > STALE_SECONDS:
        return {"status": FAILED, "error": "The review did not finish."}
    return payload


def public(path: str | Path | None) -> dict[str, Any] | None:
    """What a page shows for a file: status, and once done the fit score, its label and the tags."""
    review = read(path)
    if review is None:
        return None
    shown = {"status": review.get("status")}
    if review.get("status") == DONE:
        shown.update(fit=review.get("fit"), label=review.get("label"), tags=review.get("tags") or [])
    elif review.get("status") == FAILED:
        shown["error"] = review.get("error") or "The review failed."
    return shown


def kind_of(path: str | Path) -> str | None:
    mime = decisions.media_type(Path(path))
    kind = mime.split("/", 1)[0]
    return kind if kind in ("video", "image", "audio") else None


def _questions(kind: str, prompt: str, lyrics: str, edited: bool) -> tuple[dict[str, dict], tuple]:
    if kind == "audio":
        fit = (f"How well does this song match its brief?\nbrief: {prompt}"
               + (f"\nlyrics: {lyrics[:1500]}" if lyrics else ""))
        tags = SONG_TAGS
    else:
        noun = "edited image" if edited else kind
        fit = f"How well does the {noun} match this prompt?\nprompt: {prompt}"
        if edited:
            fit += "\nJudge whether the requested edit was made, not the whole picture."
        tags = VIDEO_TAGS if kind == "video" else VISUAL_TAGS
    questions = {"fit": decisions.score(fit, FIT_LEVELS)}
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


def review(path: str | Path, prompt: str, *, key: str, lyrics: str = "", edited: bool = False) -> dict[str, Any]:
    """Score one file against its prompt and tag it; the saved payload. Raises ``H3Error``."""
    kind = kind_of(path)
    if kind is None:
        raise H3Error(f"{Path(path).name} is not a video, image or audio file.")
    prompt = " ".join(str(prompt or "").split())[:MAX_PROMPT_CHARS]
    questions, tags = _questions(kind, prompt, str(lyrics or "").strip(), edited)
    blocks = [decisions.text(f"A generated {'song' if kind == 'audio' else kind}."), decisions.media(path)]
    answers = decisions.decide(questions, blocks, key=key, timeout=180)
    fit = decisions.score_fraction(answers.get("fit"))
    if fit is None:
        raise H3Error("Gemini Decision did not score the result.")
    level = min(len(FIT_LABELS) - 1, int(round(fit * (len(FIT_LABELS) - 1))))
    return {"status": DONE, "fit": round(fit * 100), "label": FIT_LABELS[level], "tags": _tags(answers, tags),
            "model": decisions.MODEL, "finished": time.time()}


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


def schedule(path: str | Path | None, prompt: str, *, api_key: str = "", enabled: bool = True,
             lyrics: str = "", edited: bool = False) -> bool:
    """Review ``path`` in the background; True when a review was started.

    Nothing starts without a prompt, a Gemini key, or when ``enabled`` is off. A pending
    marker is written at once so pages can show that a score is coming.
    """
    if not enabled or not path or not str(prompt or "").strip() or kind_of(path) is None:
        return False
    key = decisions.key_or_none(api_key)
    if key is None or not Path(path).is_file():
        return False
    path = Path(path)
    try:
        _write(path, {"status": PENDING, "started": time.time()})
    except OSError:
        return False

    def run() -> None:
        with _slots:
            try:
                payload = review(path, prompt, key=key, lyrics=lyrics, edited=edited)
            except Exception as exc:  # noqa: BLE001 - a review never affects the generation
                payload = {"status": FAILED, "error": str(exc)[:300], "finished": time.time()}
            try:
                _write(path, payload)
            except OSError:
                return
            if payload["status"] == DONE:
                _add_library_tags(path, payload["tags"])

    threading.Thread(target=run, name=f"review-{path.name}", daemon=True).start()
    return True
