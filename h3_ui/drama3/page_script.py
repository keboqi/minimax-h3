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
from .result_view import script_preview_settings

SKIP = object()
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


def value(update) -> Any:
    """What a tab handler set a component to, or SKIP when it left it alone."""
    if isinstance(update, dict) and update.get("__type__") == "update":
        return update.get("value", SKIP)
    return update


def _setting(data: dict, key: str):
    current = data.get(key)
    return WRITER_DEFAULTS.get(key) if current is None else current


def writer_slots(data: dict, clip: Callable[[Any], str | None]) -> list[Any]:
    """The writer's slot inputs from a project: voices, cast details, shown slots, script mode and group."""
    values: list[Any] = []
    for slot in range(DRAMA3_VOICE_SLOTS):
        values += [data.get(f"voice_{slot}_preset") or UPLOAD_VOICE, clip(data.get(f"voice_{slot}_audio")),
                   data.get(f"voice_{slot}_transcript") or ""]
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
    from .page import timeline_payload

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


def _writer_service(body: dict) -> list[Any]:
    """The writer service inputs: model, Gemini key, backend, OpenAI key, model and base URL."""
    service = body.get("writer") or {}
    return [service.get("gemini_prompt_model") or DEFAULT_GEMINI_PROMPT_MODEL, service.get("gemini_api_key") or "",
            service.get("prompt_writer_backend") or "Gemini", service.get("openai_api_key") or "",
            service.get("openai_model") or OPENAI_PROMPT_MODELS[0], service.get("openai_base_url") or ""]


def install_script_routes(app: FastAPI, actions, *, api: str, store, owner_of, payload, clip) -> None:
    """The Script step's routes; ``payload`` answers with a project's page payload, ``clip`` finds a clip."""

    def load(name: str) -> dict:
        data, _ = store.soundstage_project(name)
        return data

    def writer_settings(data: dict, body: dict) -> None:
        for key, kind in WRITER_FIELDS.items():
            if key in body:
                data[key] = kind(body[key])

    def answered(request, name, data, before, status):
        """Save a writer or example answer; a failed one changes nothing."""
        if failed(status):
            return {**payload(owner_of(request), name), "status": status, "failed": True}
        data["prepared"] = True
        sync_sounds(actions, data)
        store.save_soundstage_project(name, data)
        return {**payload(owner_of(request), name), "status": status, "failed": False, "previous": before}

    @app.post(f"{api}/projects", include_in_schema=False)
    @guarded
    def create(request: Request, body: dict = Body(...)):
        name = " ".join(str(body.get("name") or "").split())
        if not name:
            raise H3Error("Enter a project name.")
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
        data = load(name)
        before = (data.get("text"), data.get("script_mode"))
        for key, kind in SCRIPT_FIELDS.items():
            if key in body:
                data[key] = kind(body[key])
        writer_settings(data, body)
        if (data.get("text"), data.get("script_mode")) != before:
            data["prepared"] = False
            sync_sounds(actions, data)
        store.save_soundstage_project(name, data)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/rewrite", include_in_schema=False)
    def rewrite(name: str, request: Request, body: dict = Body(...)):
        try:
            data = load(name)
            writer_settings(data, body)
        except (H3Error, ValueError) as exc:
            return {"error": str(exc)}
        before = data.get("text") or ""
        if not before.strip():
            return {"error": "Write or paste some text first."}
        try:
            outputs = list(actions.enhance_drama3_text(
                before, _setting(data, "speaker_mode"), *_writer_service(body), *writer_slots(data, clip),
                _setting(data, "writer_workflow"), int(_setting(data, "max_batch_lines")),
                bool(_setting(data, "strict_source")), _setting(data, "smart_budget_profile")))
            status = str(outputs[1])
            if not failed(status):
                _apply_cast(data, outputs)
            else:
                store.save_soundstage_project(name, data)  # The writer choices stay.
            return answered(request, name, data, before, status)
        except Exception as exc:  # The writer calls out to hosted services and can fail any way.
            return {"error": f"Script unchanged. {exc}"}

    @app.post(f"{api}/project/writer-prompt", include_in_schema=False)
    @guarded
    def writer_prompt(name: str, request: Request, body: dict = Body(default={})):
        owner_of(request)
        data = load(name)
        writer_settings(data, body)
        return {"prompt": actions.copy_drama3_writer_prompt(data.get("text") or "", _setting(data, "speaker_mode"),
                                                           *writer_slots(data, clip))}

    @app.post(f"{api}/project/reply", include_in_schema=False)
    @guarded
    def apply_reply(name: str, request: Request, body: dict = Body(...)):
        data = load(name)
        writer_settings(data, body)
        reply = str(body.get("reply") or "")
        if not reply.strip():
            raise H3Error("Paste the chat model's reply first.")
        before = data.get("text") or ""
        outputs = list(actions.apply_drama3_writer_reply(reply, before, _setting(data, "speaker_mode"),
                                                         *writer_slots(data, clip)))
        status = str(outputs[1])
        if not failed(status):
            _apply_cast(data, outputs)
        return answered(request, name, data, before, status)

    @app.post(f"{api}/project/example", include_in_schema=False)
    @guarded
    def example(name: str, request: Request, body: dict = Body(...)):
        loaders = {"en": actions.load_drama3_example, "zh": actions.load_drama3_example_zh,
                   "zh-latest": actions.load_drama3_example_zh_latest}
        loader = loaders.get(body.get("example"))
        if loader is None:
            raise H3Error("Unknown example.")
        data = load(name)
        before = data.get("text") or ""
        outputs = list(loader(_setting(data, "speaker_group")))
        rest = _apply_cast(data, outputs)
        data["script_mode"], data["speaker_mode"] = value(rest[0]), value(rest[1])
        meta = rest[2:2 + 3 * DRAMA3_SOUND_SLOTS]
        _apply_sounds(data, meta, rest[2 + 3 * DRAMA3_SOUND_SLOTS:])
        data["prepared"] = True
        store.save_soundstage_project(name, data)
        return {**payload(owner_of(request), name), "status": str(outputs[1]), "failed": False, "previous": before}

    @app.post(f"{api}/preview", include_in_schema=False)
    @guarded
    def preview(request: Request, body: dict = Body(...)):
        owner_of(request)
        return preview_payload(str(body.get("text") or ""), str(body.get("script_mode") or writer.DRAMA3_SCRIPT_BASIC))

