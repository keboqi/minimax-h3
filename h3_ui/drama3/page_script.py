"""The Soundstage page's Script step: the script, the writer, pasted replies, examples and the preview.

The writer and examples are the Soundstage tab's own handlers. They answer with one value per
tab component, so this module hands them a saved project as those components' values and
writes what they change back into the project.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, Request

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, OPENAI_PROMPT_MODELS
from h3_app.drama3.example_visuals import example_visual_plan
from h3_app.drama3.guided.budget import AUTO as SMART_BUDGET_AUTO, BUDGET_PROFILES
from h3_app.drama3.results import NARRATOR_NAMES, result_settings, script_fingerprint
from h3_app.drama3.visual_timing import anchor_plan, bind_plan, mix_cues
from h3_app.errors import H3Error
from h3_app.prompt_writers import drama3 as writer

from ..page_runs import guarded
from .controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND
from .page_common import apply_fields, check_proposal, proposal_token, timeline_payload, value, writer_service
from .page_music import clear_project_music
from .page_store import edit_project, editing, project_lock
from .result_view import clean_line, preview_source_lines, script_preview_settings

from .page_adapter import (
    VOICE_META, SOUND_META, WriterAnswer, voice_inputs as voice_inputs, writer_slots as legacy_writer_slots,
    _apply_cast, _apply_sounds,
)

SCRIPT_FIELDS = {"text": str, "title": str, "script_mode": str}
WRITER_FIELDS = {"speaker_mode": str, "writer_workflow": str, "max_batch_lines": int, "strict_source": bool,
                 "smart_budget_profile": str}
WRITER_DEFAULTS = {"speaker_mode": writer.DRAMA3_SPEAKER_AUTO, "writer_workflow": writer.DRAMA3_WORKFLOW_SINGLE,
                   "max_batch_lines": writer.DRAMA3_GUIDED_BATCH_DEFAULT, "strict_source": False,
                   "smart_budget_profile": SMART_BUDGET_AUTO, "speaker_group": "Global"}
WRITER_CHOICES = {"speaker_mode": writer.DRAMA3_SPEAKER_MODES,
                  "writer_workflow": writer.DRAMA3_WORKFLOWS,
                  "smart_budget_profile": BUDGET_PROFILES}
WRITER_RANGES = {"max_batch_lines": (writer.DRAMA3_GUIDED_BATCH_MIN, writer.DRAMA3_GUIDED_BATCH_MAX)}
WORKFLOW_NOTES = {
    writer.DRAMA3_WORKFLOWS[0]: "One call. Best with strong models.",
    writer.DRAMA3_WORKFLOWS[1]: "Short steps, so small, fast models can manage.",
    writer.DRAMA3_WORKFLOWS[2]: "Plans the story, then writes scene by scene.",
}
EXAMPLES = (
    ("en", "Example 1: The Beacon of Blackthorn Reef"),
    ("zh", "示例二：昆仑墟底的回响"),
    ("zh-latest", "示例三：雨巷里的末班广播"),
)

def writer_source(data: dict) -> dict:
    """Inputs owned by the writer; unrelated generated clips/settings may merge."""
    keys = {"text", "title", "script_mode", "shown_slots", *WRITER_FIELDS}
    keys.update(f"voice_{slot}_{key}" for slot in range(DRAMA3_VOICE_SLOTS)
                for key in (*VOICE_META, "preset", "transcript"))
    return {key: data.get(key) for key in keys}


# A new project starts as the tab's draft does.
NEW_PROJECT = {"text": "", "title": "", "script_mode": writer.DRAMA3_SCRIPT_ADVANCED, "shown_slots": 1,
               "shown_sound_slots": 1}


def _setting(data: dict, key: str):
    current = data.get(key)
    return WRITER_DEFAULTS.get(key) if current is None else current


def writer_slots(data: dict, clip: Callable[[Any], str | None]) -> list[Any]:
    return legacy_writer_slots(data, clip, _setting)


def sync_sounds(actions, data: dict) -> None:
    """Give each ``<sound>`` tag of an Advanced script a sound slot, as the tab does when the script changes."""
    if actions.sync_drama3_script_sounds is None:
        return
    meta = [data.get(f"sound_{slot}_{field}") or ("" if field != "duration" else 5)
            for slot in range(DRAMA3_SOUND_SLOTS) for field in SOUND_META]
    outputs = list(actions.sync_drama3_script_sounds(data.get("text") or "", data.get("script_mode"),
                                                     int(data.get("shown_sound_slots") or 1), *meta))
    _apply_sounds(data, outputs[:3 * DRAMA3_SOUND_SLOTS], outputs[3 * DRAMA3_SOUND_SLOTS:])


def clear_cast_clips(data: dict) -> None:
    """Let go of every slot's clip, transcript and saved sound, and of results kept for review.

    Loading an example replaces the whole cast; a clip left in a slot would become the voice or
    sound of whoever the example puts there.
    """
    for slot in range(DRAMA3_VOICE_SLOTS):
        data.pop(f"voice_{slot}_audio", None)
        data.pop(f"voice_{slot}_transcript", None)
    for slot in range(DRAMA3_SOUND_SLOTS):
        data.pop(f"sound_{slot}_audio", None)
        data[f"sound_{slot}_preset"] = UPLOAD_SOUND
    for key in ("_cast_proposals", "_writer_proposal", "_open_sound_slot"):
        data.pop(key, None)


def failed(status: str) -> bool:
    return str(status or "").startswith(writer.DRAMA3_PROMPT_FAILED)


def script_payload(data: dict) -> dict:
    return {"text": data.get("text") or "", "title": data.get("title") or "",
            "script_mode": data.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC,
            "prepared": bool(data.get("prepared")),
            "proposal": bool(data.get("_writer_proposal")),
            "writer": {key: _setting(data, key) for key in WRITER_FIELDS}}


def preview_payload(text: str, script_mode: str, *, settings=None, **cast) -> dict:
    """The script's estimated timeline, in the shape of a mix's lines and sounds.

    Each line and sound also carries ``line``, the script line it comes from (1-based), so the page
    can jump between the script and the timeline; it is None when the script can't be matched up.
    """
    if settings is None:
        settings = script_preview_settings(text, script_mode, **cast)
    payload = timeline_payload(settings)
    speech, sounds = preview_source_lines(text, script_mode)
    chunks = settings["chunks"]
    if len(speech) != len(chunks) or len(sounds) != len(settings["sounds"]):
        speech, sounds = [None] * len(chunks), [None] * len(settings["sounds"])
    # The timeline leaves out turns with nothing left to hear once cues are dropped.
    heard = [number for chunk, number in zip(chunks, speech) if clean_line(chunk["text"])]
    for line, number in zip(payload["lines"], heard):
        line["line"] = number
    for sound, number in zip(payload["sounds"], sounds):
        sound["line"] = number
    return payload


def boot_payload() -> dict:
    return {
        "script_modes": list(writer.DRAMA3_SCRIPT_MODES), "advanced": writer.DRAMA3_SCRIPT_ADVANCED,
        "speaker_modes": list(writer.DRAMA3_SPEAKER_MODES),
        "workflows": [{"id": name, "note": WORKFLOW_NOTES.get(name, "")} for name in writer.DRAMA3_WORKFLOWS],
        "smart_workflow": writer.DRAMA3_WORKFLOW_SMART, "budget_profiles": list(BUDGET_PROFILES),
        "batch": {"min": writer.DRAMA3_GUIDED_BATCH_MIN, "max": writer.DRAMA3_GUIDED_BATCH_MAX,
                  "default": writer.DRAMA3_GUIDED_BATCH_DEFAULT},
        "writer": {"gemini_models": list(GEMINI_PROMPT_MODELS), "gemini_model": DEFAULT_GEMINI_PROMPT_MODEL,
                   "openai_models": list(OPENAI_PROMPT_MODELS)},
        "examples": [{"id": key, "label": label} for key, label in EXAMPLES],
        "narrators": sorted(NARRATOR_NAMES),
    }


def rewrite_project(actions, store, name: str, body: dict, clip) -> tuple[str, str, bool]:
    """Rewrite a saved project's script with the writer and keep the answer.

    ``body`` may carry writer choices (kept either way) and the writer service. Returns the
    script before, the writer's status and whether it failed; a failed rewrite changes nothing
    else. Raises when there is nothing to rewrite.

    The writer can take minutes, so the project isn't held while it works: its answer is written
    into the project as saved when it arrives, keeping clips and settings changed meanwhile.
    """
    def choose(data):
        apply_fields(data, body, WRITER_FIELDS, ranges=WRITER_RANGES, choices=WRITER_CHOICES)
        return dict(data)

    data = edit_project(store, name, choose)  # The writer choices stay, whatever it answers.
    before = data.get("text") or ""
    if not before.strip():
        raise H3Error("Write or paste some text first.")
    try:
        answer = WriterAnswer.read(actions.enhance_drama3_text(
            before, _setting(data, "speaker_mode"), *writer_service(body), *writer_slots(data, clip),
            _setting(data, "writer_workflow"), int(_setting(data, "max_batch_lines")),
            bool(_setting(data, "strict_source")), _setting(data, "smart_budget_profile")))
    except Exception as exc:  # The writer calls out to hosted services and can fail any way.
        return before, f"{writer.DRAMA3_PROMPT_FAILED} {exc}", True
    outputs, status = answer.outputs, answer.status
    if answer.failed:
        return before, status, True

    def keep(current):
        if writer_source(current) != writer_source(data):
            current["_writer_proposal"] = {"id": uuid.uuid4().hex, "outputs": outputs, "source": before}
            return False
        _apply_cast(current, outputs)
        current["prepared"] = True
        current.pop("_writer_proposal", None)
        sync_sounds(actions, current)
        return True

    if not edit_project(store, name, keep):
        return before, f"{writer.DRAMA3_PROMPT_FAILED} The script or cast changed while writing. The answer is kept for review.", True
    return before, status, False


def _source_line_numbers(estimated, spoken):
    """Match split takes to source lines independently for each speaker."""
    from h3_app.drama3.serving.cue_timing import normalized
    from h3_app.drama3.serving.planner import spoken_text

    pending = {}
    for line in estimated:
        pending.setdefault(line["speaker"], deque()).append((
            normalized(spoken_text(line["text"])), line.get("line")))
    numbers = []
    for line in spoken:
        words = normalized(spoken_text(line["text"]))
        queue = pending.get(line["speaker"])
        if not queue or not words or not queue[0][0].startswith(words):
            return None
        remaining, number = queue[0]
        numbers.append(number)
        queue[0] = (remaining[len(words):], number)
        if not queue[0][0]:
            queue.popleft()
    return None if any(pending.values()) else numbers


def spoken_payload(estimate: dict, settings: dict, script_mode: str, text: str) -> dict | None:
    """The spoken mix's real timeline, numbered with the estimate's script lines, or None.

    A mix belongs to the script when its source fingerprint and script mode still match.
    Older mixes without a fingerprint use the estimate, since their staging cannot be verified.
    Split synthesis chunks retain their source line. Each speaker's lines stay in source order,
    even when overlapping voices arrive in a different order. A sound takes the first unmatched
    sound of its name, or None.
    """
    if not settings or (settings.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC) != script_mode:
        return None
    if settings.get("script_fingerprint") != script_fingerprint(text):
        return None
    actual = timeline_payload(settings)
    if not actual["lines"]:
        return None
    numbers = _source_line_numbers(estimate["lines"], actual["lines"])
    if numbers is None:
        return None
    for line, number in zip(actual["lines"], numbers, strict=True):
        line["line"] = number
    sound_lines: dict[str, list] = {}
    for sound in estimate["sounds"]:
        sound_lines.setdefault(sound["name"], []).append(sound.get("line"))
    for sound in actual["sounds"]:
        found = sound_lines.get(sound["name"])
        sound["line"] = found.pop(0) if found else None
    return {**actual, "actual": True}


def preview_effects(plan: dict, settings: dict, cues: list) -> list:
    """Resolve effect clips against the same clock as the script timeline."""
    if not plan.get("script_cues") or not any(row.get("effects") for row in plan.get("shots", [])):
        return []
    try:
        shots = bind_plan(plan, {**settings, "script_cues": cues})["shots"]
    except H3Error:
        return []  # The shot plan no longer fits this script.
    duration = settings["duration"]
    return [{"index": index, "start": shot["start"], "end": end, "effects": shot["effects"]}
            for index, (shot, end) in enumerate(zip(shots, [row["start"] for row in shots[1:]] + [duration]))
            if shot.get("effects") and 0 <= shot["start"] < end <= duration]


def install_script_routes(app: FastAPI, actions, *, api: str, store, owner_of, payload, clip, runtime=None,
                          admission_lock, example_blocked) -> None:
    """The Script step's routes; ``payload`` answers with a project's page payload, ``clip`` finds a clip."""

    def load(name: str) -> dict:
        data, _ = store.soundstage_project(name)
        return data

    def answered(owner, name, before, status, refused):
        """Answer for a writer reply that was saved; a refused one changed nothing but the writer choices."""
        answer = {**payload(owner, name), "status": status, "failed": refused}
        return answer if refused else {**answer, "previous": before}

    @app.post(f"{api}/projects", include_in_schema=False)
    @guarded
    def create(request: Request, body: dict = Body(...)):
        name = " ".join(str(body.get("name") or "").split())
        if not name:
            raise H3Error("Enter a project name.")
        with project_lock(name):
            store.create_soundstage_project(name, dict(NEW_PROJECT))
        return payload(owner_of(request), name)

    @app.patch(f"{api}/project/script", include_in_schema=False)
    @guarded
    def save_script(name: str, request: Request, body: dict = Body(...)):
        """Keep the script, title, script mode and writer choices; a changed script needs rewriting again."""
        if "script_mode" in body and body["script_mode"] not in writer.DRAMA3_SCRIPT_MODES:
            raise H3Error("Unknown script mode.")
        with editing(store, name) as data:
            before = (data.get("text"), data.get("script_mode"))
            apply_fields(data, body, {**SCRIPT_FIELDS, **WRITER_FIELDS}, ranges=WRITER_RANGES,
                         choices={**WRITER_CHOICES, "script_mode": writer.DRAMA3_SCRIPT_MODES})
            if (data.get("text"), data.get("script_mode")) != before:
                data["prepared"] = False
                sync_sounds(actions, data)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/rewrite", include_in_schema=False)
    @guarded
    def rewrite(name: str, request: Request, body: dict = Body(...)):
        owner = owner_of(request)
        before, status, refused = rewrite_project(actions, store, name, body, clip)
        return answered(owner, name, before, status, refused)

    @app.post(f"{api}/project/writer-prompt", include_in_schema=False)
    @guarded
    def writer_prompt(name: str, request: Request, body: dict = Body(default={})):
        owner_of(request)
        data = load(name)
        apply_fields(data, body, WRITER_FIELDS, ranges=WRITER_RANGES, choices=WRITER_CHOICES)
        return {"prompt": actions.copy_drama3_writer_prompt(data.get("text") or "", _setting(data, "speaker_mode"),
                                                           *writer_slots(data, clip))}

    @app.get(f"{api}/project/proposal", include_in_schema=False)
    @guarded
    def read_proposal(name: str, request: Request):
        owner_of(request)
        data = load(name)
        proposal = data.get("_writer_proposal")
        if not proposal:
            raise H3Error("There is no saved writer answer.")
        _apply_cast(data, proposal["outputs"])
        return {"id": proposal_token(proposal), "text": data.get("text") or "", "source": proposal["source"]}

    @app.post(f"{api}/project/proposal", include_in_schema=False)
    @guarded
    def apply_proposal(name: str, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        with editing(store, name) as data:
            proposal = data.get("_writer_proposal")
            if not proposal:
                raise H3Error("There is no saved writer answer.")
            check_proposal(proposal, body.get("id"))
            data.pop("_writer_proposal")
            before = data.get("text") or ""
            _apply_cast(data, proposal["outputs"])
            data["prepared"] = True
            sync_sounds(actions, data)
        return answered(owner, name, before, "Applied the saved writer answer.", False)

    @app.post(f"{api}/project/reply", include_in_schema=False)
    @guarded
    def apply_reply(name: str, request: Request, body: dict = Body(...)):
        reply = str(body.get("reply") or "")
        if not reply.strip():
            raise H3Error("Paste the chat model's reply first.")
        with editing(store, name) as data:
            apply_fields(data, body, WRITER_FIELDS, ranges=WRITER_RANGES, choices=WRITER_CHOICES)
            before = data.get("text") or ""
            # "Keep my exact words" decides, as for a rewrite: off keeps the reply and lists the changes.
            outputs = list(actions.apply_drama3_writer_reply(reply, before, _setting(data, "speaker_mode"),
                                                             *writer_slots(data, clip),
                                                             strict_source=bool(_setting(data, "strict_source"))))
            status = str(outputs[1])
            if not failed(status):
                data.pop("_writer_proposal", None)  # The pasted reply replaces any saved writer answer.
                _apply_cast(data, outputs)
                data["prepared"] = True
                sync_sounds(actions, data)
        return answered(owner_of(request), name, before, status, failed(status))

    @app.post(f"{api}/project/example", include_in_schema=False)
    @guarded
    def example(name: str, request: Request, body: dict = Body(...)):
        from . import visuals
        from .project_visuals import project_cues

        loaders = {"en": actions.load_drama3_example, "zh": actions.load_drama3_example_zh,
                   "zh-latest": actions.load_drama3_example_zh_latest}
        lang = body.get("example")
        loader = loaders.get(lang)
        if loader is None:
            raise H3Error("Unknown example.")
        owner = owner_of(request)
        with admission_lock:
            if example_blocked(owner, name):
                raise H3Error("Wait for Soundstage to finish working on this project before loading an example.")
            with editing(store, name, reset_result=True) as data:
                before = data.get("text") or ""
                outputs = list(loader(_setting(data, "speaker_group")))
                clear_cast_clips(data)
                clear_project_music(data)
                rest = _apply_cast(data, outputs)
                data["script_mode"], data["speaker_mode"] = value(rest[0]), value(rest[1])
                meta = rest[2:2 + 3 * DRAMA3_SOUND_SLOTS]
                _apply_sounds(data, meta, rest[2 + 3 * DRAMA3_SOUND_SLOTS:])
                data["prepared"] = True

                plan = anchor_plan(example_visual_plan(lang), {"script_cues": project_cues(data, runtime)})
                limit = len(plan["shots"])
                style = data.get("video_style")
                if style is None:
                    style = visuals.STYLE_PRESETS[visuals.DEFAULT_STYLE]
                data["video_limit"] = limit
                data["visual_plan"] = {"style": style, "limit": limit, "plan": plan}
        return answered(owner, name, before, f"{outputs[1]} Included {limit} shots with visual effects.", False)

    @app.post(f"{api}/preview", include_in_schema=False)
    @guarded
    def preview(request: Request, body: dict = Body(...), name: str | None = None):
        """The script's timeline: the project's spoken mix when it is of this script, else an estimate."""
        owner_of(request)
        text, script_mode = str(body.get("text") or ""), str(body.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC)
        data, result = store.soundstage_project(name) if name else ({}, None)
        from .project_visuals import project_cast
        estimated = script_preview_settings(text, script_mode, **project_cast(data, runtime))
        estimate = preview_payload(text, script_mode, settings=estimated)
        spoken = result_settings(result) if result else {}
        actual = spoken_payload(estimate, spoken, script_mode, text)
        timeline = actual or estimate
        plan = (data.get("visual_plan") or {}).get("plan") or {}
        if plan and estimate["lines"]:
            effects = preview_effects(plan, spoken if actual else estimated, mix_cues(estimated))
            if effects:
                timeline["effects"] = effects
        return timeline

