"""Keep Workshop shot plans in the shared project and bind them to generated speech."""

import hashlib
from pathlib import Path

from h3_app.drama3.results import result_settings
from h3_app.drama3.visual_timing import (adapt_plan, anchor_plan, bind_plan, mix_cues,
                                        preview_settings, same_dialogue, script_cues)
from h3_app.errors import H3Error
from . import visuals
from .page_cast import cast_voices, speakers_payload
from .page_store import editing, project_lock


def project_cast(data, runtime=None):
    """Parser defaults shared by the script preview and illustration cue planner."""
    from h3_app.workflows.drama3 import resolve_speaker, speaker_label

    speakers = speakers_payload("", data, lambda _audio: None, runtime)
    names = {entry["slot"]: entry["name"] for entry in speakers}
    configured = any(name != f"Speaker {slot}" for slot, name in names.items())
    voiced = [entry["slot"] for entry in speakers if entry["ready"] or data.get(f"voice_{entry['slot']}_audio")]
    default = speaker_label(voiced[0] if voiced else 0, names)
    return dict(default_speaker=default, resolve=lambda name: resolve_speaker(name, names) or
                (None if configured else " ".join(name.split())))


def project_cues(data, runtime=None):
    """Use the same Cast labels for names, numeric tags and unlabelled dialogue."""
    text = data.get("text") or ""
    if (data.get("script_mode") or "Basic") == "Basic":
        from h3_app.drama3.serving.tree import strip_drama3_sound_tags
        text = strip_drama3_sound_tags(text)
    return script_cues(text, **project_cast(data, runtime))


def planning_mix(name, data, result, export, *, strict=False, store=None):
    with project_lock(name):
        if store is not None:
            data = store.soundstage_project(name)[0]
        audio, settings, canonical = _planning_mix(name, data, result, export, strict=strict)
        if canonical is not None and store is not None and canonical != data.get("visual_plan"):
            with editing(store, name) as current:
                current["visual_plan"] = canonical
        return audio, settings


def _shots(saved, duration, data):
    style = data.get("video_style")
    if style is None:
        style = visuals.STYLE_PRESETS[visuals.DEFAULT_STYLE]
    return visuals.plan_shots_for(saved, duration, style, data.get("video_orientation") or "landscape")


def _publish_mix_plan(audio, current, saved, settings, data, root):
    """Validate before replacing a sidecar, then move its existing images to the new clock."""
    shots = _shots(saved, settings["duration"], data)
    if current == saved:
        return
    previous_shots = None
    if current:
        try:
            previous_shots = _shots(current, settings["duration"], data)
        except H3Error:
            pass  # Stable shot IDs can still recover images from an older, invalid clock.
    visuals.write_plan(audio, saved["style"], saved["limit"], saved["plan"])
    visuals.rebind_stills(audio, root, shots, previous_shots=previous_shots)


def _planning_mix(name, data, result, export, *, strict):
    voices = cast_voices(data, export.deps.runtime_config)
    if result:
        audio = Path(result)
        settings = {**result_settings(audio), "voices": voices}
    else:
        root = Path(export.deps.runtime_config().output_dir) / ".soundstage" / "plans"
        root.mkdir(parents=True, exist_ok=True)
        audio = root / (hashlib.sha256(name.encode()).hexdigest() + ".wav")
        settings = preview_settings(data.get("text") or "", data.get("title") or "", voices,
                                    cues=project_cues(data, export.deps.runtime_config))
    saved = data.get("visual_plan")
    current = visuals.read_plan(audio)
    source = saved or (current if result else None)
    canonical = None
    if source:
        try:
            if result:
                actual = mix_cues(settings)
                proposed = project_cues(data, export.deps.runtime_config)
                original = source["plan"].get("script_cues") or []
                # A project may already have edits while the player still plays its older mix.
                settings["script_cues"] = (proposed if same_dialogue(proposed, actual) else
                                           original if same_dialogue(original, actual) else actual)
            anchored = anchor_plan(source["plan"], settings)
            anchored = adapt_plan(anchored, settings["script_cues"])
            plan = bind_plan(anchored, settings)
            _publish_mix_plan(audio, current, {**source, "plan": plan}, settings, data,
                              export.deps.runtime_config().output_dir)
            canonical = {"style": source["style"], "limit": source["limit"], "plan": anchored}
        except H3Error as exc:
            if strict:
                raise
            settings["visual_timing_error"] = str(exc)
    return audio, settings, canonical


def can_reuse_script(data, result=None, runtime=None):
    """Existing shots or previously spoken dialogue protect wording fixes from Auto's writer."""
    plan = (data.get("visual_plan") or {}).get("plan") or {}
    if not plan and result:
        settings = result_settings(result)
        plan = {"script_cues": mix_cues(settings), "shots": []}
    try:
        adapt_plan(plan, project_cues(data, runtime))
    except H3Error:
        return False
    return True


def carry_project_stills(store, name, previous_result, result, export):
    """Reuse illustrations after speaking/remixing changes the mix file and clock."""
    if not result or (previous_result and Path(result) == Path(previous_result)):
        return
    data = store.soundstage_project(name)[0]
    if not previous_result:
        previous_result = Path(export.deps.runtime_config().output_dir) / ".soundstage" / "plans" / (
            hashlib.sha256(name.encode()).hexdigest() + ".wav")
    audio, settings = planning_mix(name, data, result, export, strict=True, store=store)
    saved = visuals.read_plan(audio)
    old = visuals.read_plan(previous_result)
    if not saved or not old:
        return
    old_duration = result_settings(previous_result).get("duration")
    if not old_duration:
        record = visuals.read_stills(previous_result, export.deps.runtime_config().output_dir)
        if record and record.get("shots"):
            frames = record.get("total_frames") or max(
                item["first_frame"] + item["frames"] for item in record["shots"])
            old_duration = frames / visuals.FPS
    if not old_duration:
        return
    shots = _shots(saved, settings["duration"], data)
    old_shots = _shots(old, old_duration, data)
    visuals.rebind_stills(audio, export.deps.runtime_config().output_dir, shots,
                          previous_audio=previous_result, previous_shots=old_shots)


def save_project_plan(store, name, audio, style, limit, plan, *, run=None):
    """Publish both copies under the same lock used when a refresh resolves the plan.

    Reading the sidecar back after releasing this lock lets a refresh restore the old project
    plan before the new one reaches the database.
    """
    with project_lock(name):
        snapshot = result_settings(audio)
        if not snapshot.get("duration"):
            data = store.soundstage_project(name)[0]
            snapshot = preview_settings(data.get("text") or "", cues=plan.get("script_cues") or project_cues(data))
        plan = anchor_plan(plan, snapshot)
        bound = bind_plan(plan, snapshot)
        if run is None:
            visuals.write_plan(audio, style, limit, bound)
        elif not visuals.save_plan(audio, style, limit, bound, run):
            return False
        with editing(store, name) as data:
            data["visual_plan"] = {"style": style, "limit": int(limit), "plan": plan}
    return True


def planning_export(runtime):
    """Workshop exposes planning and art settings; GPU drawing remains in Studio."""
    from types import SimpleNamespace
    return SimpleNamespace(engines=[], deps=SimpleNamespace(runtime_config=lambda: runtime))
