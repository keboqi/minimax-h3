"""Keep Workshop shot plans in the shared project and bind them to generated speech."""

import hashlib
from pathlib import Path

from h3_app.drama3.results import result_settings
from h3_app.drama3.visual_timing import bind_plan, preview_settings, script_cues
from h3_app.errors import H3Error
from . import visuals
from .page_cast import cast_voices
from .page_store import editing, project_lock


def planning_mix(name, data, result, export, *, strict=False, store=None):
    with project_lock(name):
        if store is not None:
            data = store.soundstage_project(name)[0]
        return _planning_mix(name, data, result, export, strict=strict)


def _planning_mix(name, data, result, export, *, strict):
    voices = cast_voices(data, export.deps.runtime_config)
    if result:
        audio = Path(result)
        settings = {**result_settings(audio), "voices": voices}
    else:
        root = Path(export.deps.runtime_config().output_dir) / ".soundstage" / "plans"
        root.mkdir(parents=True, exist_ok=True)
        audio = root / (hashlib.sha256(name.encode()).hexdigest() + ".wav")
        settings = preview_settings(data.get("text") or "", data.get("title") or "", voices)
    saved = data.get("visual_plan")
    current = visuals.read_plan(audio)
    source = saved or (current if result else None)
    if source:
        try:
            if not result and source["plan"].get("script_cues") != settings["script_cues"]:
                raise H3Error("The script differs from the shot plan. Plan shots again for this script.")
            plan = bind_plan(source["plan"], settings)
        except H3Error as exc:
            if strict:
                raise
            settings["visual_timing_error"] = str(exc)
        else:
            if not current or current["plan"] != plan or current["limit"] != source["limit"]:
                visuals.write_plan(audio, source["style"], source["limit"], plan)
    return audio, settings


def has_script_plan(data):
    """Only an unchanged script can bypass Auto's rewrite using a Workshop plan."""
    cues = ((data.get("visual_plan") or {}).get("plan") or {}).get("script_cues")
    return bool(cues) and cues == script_cues(data.get("text") or "")


def save_project_plan(store, name, audio, style, limit, plan, *, run=None):
    """Publish both copies under the same lock used when a refresh resolves the plan.

    Reading the sidecar back after releasing this lock lets a refresh restore the old project
    plan before the new one reaches the database.
    """
    with project_lock(name):
        if run is None:
            visuals.write_plan(audio, style, limit, plan)
        elif not visuals.save_plan(audio, style, limit, plan, run):
            return False
        with editing(store, name) as data:
            data["visual_plan"] = {"style": style, "limit": int(limit), "plan": plan}
    return True


def planning_export(runtime):
    """Workshop exposes planning and art settings; GPU drawing remains in Studio."""
    from types import SimpleNamespace
    return SimpleNamespace(engines=[], deps=SimpleNamespace(runtime_config=lambda: runtime))
