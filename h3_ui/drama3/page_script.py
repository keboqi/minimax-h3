"""The Soundstage page's Script step: the script, the writer, pasted replies, examples and the preview.

The writer and examples are the Soundstage tab's own handlers. They answer with one value per
tab component, so this module hands them a saved project as those components' values and
writes what they change back into the project.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Body, FastAPI, Request

from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, GEMINI_PROMPT_MODELS, OPENAI_PROMPT_MODELS
from h3_app.drama3.guided.budget import AUTO as SMART_BUDGET_AUTO, BUDGET_PROFILES
from h3_app.drama3.results import NARRATOR_NAMES
from h3_app.errors import H3Error
from h3_app.prompt_writers import drama3 as writer

from ..storyboard.page import guarded
from .controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_VOICE
from .page_common import SKIP, apply_fields, timeline_payload, value, writer_service
from .page_store import edit_project, editing, project_lock
from .result_view import script_preview_settings

SCRIPT_FIELDS = {"text": str, "title": str, "script_mode": str}
WRITER_FIELDS = {"speaker_mode": str, "writer_workflow": str, "max_batch_lines": int, "strict_source": bool,
                 "smart_budget_profile": str}
WRITER_DEFAULTS = {"speaker_mode": writer.DRAMA3_SPEAKER_AUTO, "writer_workflow": writer.DRAMA3_WORKFLOW_SINGLE,
                   "max_batch_lines": writer.DRAMA3_GUIDED_BATCH_DEFAULT, "strict_source": False,
                   "smart_budget_profile": SMART_BUDGET_AUTO, "speaker_group": "Global"}
WORKFLOW_NOTES = {
    writer.DRAMA3_WORKFLOWS[0]: "One call. Best with strong models.",
    writer.DRAMA3_WORKFLOWS[1]: "Short steps, so small, fast models can manage.",
    writer.DRAMA3_WORKFLOWS[2]: "Plans the story, then writes scene by scene.",
}
EXAMPLES = (("en", "Example story"), ("zh", "Chinese example (中文)"), ("zh-latest", "示例三：雨巷里的末班广播"))
VOICE_META = ("save_name", "description", "preview_text")
SOUND_META = ("save_name", "description", "duration")
# A new project starts as the tab's draft does.
NEW_PROJECT = {"text": "", "title": "", "script_mode": writer.DRAMA3_SCRIPT_ADVANCED, "shown_slots": 1,
               "shown_sound_slots": 1}


def _setting(data: dict, key: str):
    current = data.get(key)
    return WRITER_DEFAULTS.get(key) if current is None else current


def voice_inputs(data: dict, clip: Callable[[Any], str | None]) -> list[Any]:
    """Each voice slot's preset, clip and transcript, as the tab's speech and writer handlers take them."""
    values: list[Any] = []
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_preset") or UPLOAD_VOICE, clip(data.get(f"voice_{slot}_audio")),
                   data.get(f"voice_{slot}_transcript") or ""]
    return values


def writer_slots(data: dict, clip: Callable[[Any], str | None]) -> list[Any]:
    """The writer's slot inputs from a project: voices, cast details, shown slots, script mode and group."""
    values = voice_inputs(data, clip)
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_{field}") or "" for field in VOICE_META]
    return [*values, int(data.get("shown_slots") or 1), data.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC,
            _setting(data, "speaker_group")]


def _apply_cast(data: dict, outputs: list[Any]) -> list[Any]:
    """Write the script, cast details, shown slots, presets and title from a writer answer into ``data``.

    ``outputs`` is the tab's writer outputs: script, status, three details per slot, the shown slot
    count with each slot's visibility and Add/Remove, a preset per slot, then the title. Returns
    what follows them.
    """
    script, rest = outputs[0], outputs[2:]
    if value(script) is not SKIP:
        data["text"] = value(script)
    details, rest = rest[:3 * DRAMA3_VOICE_SLOTS], rest[3 * DRAMA3_VOICE_SLOTS:]
    for slot in range(DRAMA3_VOICE_SLOTS):
        for index, field in enumerate(VOICE_META):
            changed = value(details[3 * slot + index])
            if changed is not SKIP:
                data[f"voice_{slot}_{field}"] = changed
    shown, rest = value(rest[0]), rest[DRAMA3_VOICE_SLOTS + 3:]
    if shown is not SKIP:
        data["shown_slots"] = int(shown)
    presets, rest = rest[:DRAMA3_VOICE_SLOTS], rest[DRAMA3_VOICE_SLOTS:]
    for slot, preset in enumerate(presets):
        if value(preset) is not SKIP:
            data[f"voice_{slot}_preset"] = value(preset)
    title, rest = value(rest[0]), rest[1:]
    if title is not SKIP:
        data["title"] = title
    return rest


def _apply_sounds(data: dict, meta: list[Any], visibility: list[Any]) -> None:
    for slot in range(DRAMA3_SOUND_SLOTS):
        for index, field in enumerate(SOUND_META):
            changed = value(meta[3 * slot + index])
            if changed is not SKIP:
                data[f"sound_{slot}_{field}"] = changed
    shown = value(visibility[0])
    if shown is not SKIP:
        data["shown_sound_slots"] = int(shown)


def sync_sounds(actions, data: dict) -> None:
    """Give each ``<sound>`` tag of an Advanced script a sound slot, as the tab does when the script changes."""
    if actions.sync_drama3_script_sounds is None:
        return
    meta = [data.get(f"sound_{slot}_{field}") or ("" if field != "duration" else 5)
            for slot in range(DRAMA3_SOUND_SLOTS) for field in SOUND_META]
    outputs = list(actions.sync_drama3_script_sounds(data.get("text") or "", data.get("script_mode"),
                                                     int(data.get("shown_sound_slots") or 1), *meta))
    _apply_sounds(data, outputs[:3 * DRAMA3_SOUND_SLOTS], outputs[3 * DRAMA3_SOUND_SLOTS:])


def failed(status: str) -> bool:
    return str(status or "").startswith(writer.DRAMA3_PROMPT_FAILED)


def script_payload(data: dict) -> dict:
    return {"text": data.get("text") or "", "title": data.get("title") or "",
            "script_mode": data.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC,
            "prepared": bool(data.get("prepared")),
            "writer": {key: _setting(data, key) for key in WRITER_FIELDS}}


def preview_payload(text: str, script_mode: str) -> dict:
    """The script's estimated timeline, in the shape of a mix's lines and sounds."""
    return timeline_payload(script_preview_settings(text, script_mode))


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
        apply_fields(data, body, WRITER_FIELDS)
        return dict(data)

    data = edit_project(store, name, choose)  # The writer choices stay, whatever it answers.
    before = data.get("text") or ""
    if not before.strip():
        raise H3Error("Write or paste some text first.")
    try:
        outputs = list(actions.enhance_drama3_text(
            before, _setting(data, "speaker_mode"), *writer_service(body), *writer_slots(data, clip),
            _setting(data, "writer_workflow"), int(_setting(data, "max_batch_lines")),
            bool(_setting(data, "strict_source")), _setting(data, "smart_budget_profile")))
    except Exception as exc:  # The writer calls out to hosted services and can fail any way.
        return before, f"{writer.DRAMA3_PROMPT_FAILED} {exc}", True
    status = str(outputs[1])
    if failed(status):
        return before, status, True

    def keep(current):
        _apply_cast(current, outputs)
        current["prepared"] = True
        sync_sounds(actions, current)

    edit_project(store, name, keep)
    return before, status, False


def install_script_routes(app: FastAPI, actions, *, api: str, store, owner_of, payload, clip) -> None:
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
            if any(existing == name for existing, _ in store.soundstage_projects()):
                raise H3Error(f"A project named {name} already exists.")
            store.save_soundstage_project(name, dict(NEW_PROJECT))
        return payload(owner_of(request), name)

    @app.patch(f"{api}/project/script", include_in_schema=False)
    @guarded
    def save_script(name: str, request: Request, body: dict = Body(...)):
        """Keep the script, title, script mode and writer choices; a changed script needs rewriting again."""
        if "script_mode" in body and body["script_mode"] not in writer.DRAMA3_SCRIPT_MODES:
            raise H3Error("Unknown script mode.")
        with editing(store, name) as data:
            before = (data.get("text"), data.get("script_mode"))
            apply_fields(data, body, {**SCRIPT_FIELDS, **WRITER_FIELDS})
            if (data.get("text"), data.get("script_mode")) != before:
                data["prepared"] = False
                sync_sounds(actions, data)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/rewrite", include_in_schema=False)
    def rewrite(name: str, request: Request, body: dict = Body(...)):
        try:
            owner = owner_of(request)
            before, status, refused = rewrite_project(actions, store, name, body, clip)
        except (H3Error, ValueError) as exc:
            return {"error": str(exc)}
        return answered(owner, name, before, status, refused)

    @app.post(f"{api}/project/writer-prompt", include_in_schema=False)
    @guarded
    def writer_prompt(name: str, request: Request, body: dict = Body(default={})):
        owner_of(request)
        data = load(name)
        apply_fields(data, body, WRITER_FIELDS)
        return {"prompt": actions.copy_drama3_writer_prompt(data.get("text") or "", _setting(data, "speaker_mode"),
                                                           *writer_slots(data, clip))}

    @app.post(f"{api}/project/reply", include_in_schema=False)
    @guarded
    def apply_reply(name: str, request: Request, body: dict = Body(...)):
        reply = str(body.get("reply") or "")
        if not reply.strip():
            raise H3Error("Paste the chat model's reply first.")
        with editing(store, name) as data:
            apply_fields(data, body, WRITER_FIELDS)
            before = data.get("text") or ""
            # "Keep my exact words" decides, as for a rewrite: off keeps the reply and lists the changes.
            outputs = list(actions.apply_drama3_writer_reply(reply, before, _setting(data, "speaker_mode"),
                                                             *writer_slots(data, clip),
                                                             strict_source=bool(_setting(data, "strict_source"))))
            status = str(outputs[1])
            if not failed(status):
                _apply_cast(data, outputs)
                data["prepared"] = True
                sync_sounds(actions, data)
        return answered(owner_of(request), name, before, status, failed(status))

    @app.post(f"{api}/project/example", include_in_schema=False)
    @guarded
    def example(name: str, request: Request, body: dict = Body(...)):
        loaders = {"en": actions.load_drama3_example, "zh": actions.load_drama3_example_zh,
                   "zh-latest": actions.load_drama3_example_zh_latest}
        loader = loaders.get(body.get("example"))
        if loader is None:
            raise H3Error("Unknown example.")
        with editing(store, name) as data:
            before = data.get("text") or ""
            outputs = list(loader(_setting(data, "speaker_group")))
            rest = _apply_cast(data, outputs)
            data["script_mode"], data["speaker_mode"] = value(rest[0]), value(rest[1])
            meta = rest[2:2 + 3 * DRAMA3_SOUND_SLOTS]
            _apply_sounds(data, meta, rest[2 + 3 * DRAMA3_SOUND_SLOTS:])
            data["prepared"] = True
        return answered(owner_of(request), name, before, str(outputs[1]), False)

    @app.post(f"{api}/preview", include_in_schema=False)
    @guarded
    def preview(request: Request, body: dict = Body(...)):
        owner_of(request)
        return preview_payload(str(body.get("text") or ""), str(body.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC))

