"""The Soundstage page's Cast & sounds step: speakers, sounds, their clips and the jobs that make them.

The project keeps each speaker and sound in a slot, as the Soundstage tab does. Voices, sounds
and transcripts are made by the tab's own handlers on the usual job path; each clip they make is
kept with the project's clips and written into its slot as it arrives, so a page that polls, or
opens later, finds it there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable
import uuid

from fastapi import Body, FastAPI, File, Request, UploadFile

from h3_app.catalog import AUDIO_EXTENSIONS
from h3_app.drama3.voice_design import ENGINE_QWEN, VOICE_DESIGN_ENGINES
from h3_app.errors import H3Error
from h3_app.prompt_writers.drama3 import DRAMA3_SCRIPT_ADVANCED
from h3_app.voice_design import LANGUAGES
from h3_models import DEFAULT_DRAMA3_ASR, DRAMA3_ASR_CHOICES

from ..job_bindings import owned_generation
from ..page_runs import Runs, guarded
from .controls import DRAMA3_SOUND_SLOTS, DRAMA3_VOICE_SLOTS, UPLOAD_SOUND, UPLOAD_VOICE
from .page_common import (SKIP, PageRun, apply_fields, check_proposal, proposal_token,
                          file_response, found, run_job, speech_value)
from .page_adapter import CastProgress
from .page_store import edit_project, editing


def _keep_clip(path, directory: Path) -> str:
    """Keep a clip with the project's clips under its content digest; returns its file name."""
    import os

    from h3_app.storyboard.service import file_digest

    source = Path(path)
    target = directory / (file_digest(source) + source.suffix.lower())
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / uuid.uuid4().hex
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target.name


VOICE_FIELDS = {"save_name": str, "description": str, "preview_text": str, "language": str, "transcript": str}
SOUND_FIELDS = {"save_name": str, "description": str, "duration": int}
CAST_SETTINGS = {"voice_engine": str, "asr_model": str, "skip_existing": bool, "skip_existing_sounds": bool,
                 "speaker_group": str}
CAST_DEFAULTS = {"voice_engine": ENGINE_QWEN, "asr_model": DEFAULT_DRAMA3_ASR, "skip_existing": True,
                 "skip_existing_sounds": True, "speaker_group": "Global"}
# What a cast run makes: one voice, one sound, one transcript, or everything missing.
VOICE, SOUND, TRANSCRIBE, PREPARE = "voice", "sound", "transcribe", "prepare"


@dataclass
class CastRun(PageRun):
    """A cast job, with the latest word from each slot it works on (``voice:0``, ``sound:2``)."""

    target: str | None = None
    slots: dict[str, str] = field(default_factory=dict)

    def public(self, scene_id) -> dict[str, Any]:
        return {**super().public(scene_id), "target": self.target, "slots": dict(self.slots)}


@dataclass(frozen=True)
class CastJobs:
    """The cast runs, and the pieces Auto runs in its own order."""

    runs: Runs
    job_for: Callable
    execute: Callable
    save_all: Callable
    match_all: Callable


def _setting(data: dict, key: str):
    current = data.get(key)
    return CAST_DEFAULTS[key] if current is None else current


def _is_saved(preset, upload) -> bool:
    return bool(preset) and preset != upload


def slot_keys(kind: str, slot: int) -> tuple[str, str]:
    """A slot's key prefix in the project (``voice_0``) and the preset that means "no saved one"."""
    return (f"voice_{slot}", UPLOAD_VOICE) if kind == "voice" else (f"sound_{slot}", UPLOAD_SOUND)


def clip_url(name: str, kind: str, slot: int, version) -> str:
    from urllib.parse import quote

    return f"/soundstage/api/clip?name={quote(name)}&kind={kind}&slot={slot}&v={quote(str(version))}"


def speakers_payload(name: str, data: dict, clip: Callable[[Any], str | None], runtime) -> list[dict]:
    """Every shown speaker slot: who it is, where its voice comes from and whether it has one."""
    from h3_app.drama3 import speakers as drama3_speakers

    speakers = []
    for slot in range(min(int(data.get("shown_slots") or 1), DRAMA3_VOICE_SLOTS)):
        preset = data.get(f"voice_{slot}_preset")
        entry = {"slot": slot, "preset": None, "source": "describe", "clip": None, "saved_name": None,
                 **{key: data.get(f"voice_{slot}_{key}") or "" for key in VOICE_FIELDS}}
        entry["language"] = entry["language"] or "Auto"
        if _is_saved(preset, UPLOAD_VOICE):
            entry.update(preset=preset, source="saved")
            try:
                saved = drama3_speakers.load_speaker(preset, runtime())
                entry.update(saved_name=saved.name, transcript=saved.transcript,
                             description=saved.description or entry["description"],
                             clip=clip_url(name, "voice", slot, preset))
            except (H3Error, AttributeError, TypeError):
                entry["missing"] = True
        else:
            audio = clip(data.get(f"voice_{slot}_audio"))
            if audio:
                entry.update(source="clip", clip=clip_url(name, "voice", slot, Path(audio).name))
        entry["name"] = entry["saved_name"] or entry["save_name"] or f"Speaker {slot}"
        entry["ready"] = bool(entry["clip"])
        entry["proposal"] = f"voice:{slot}" in (data.get("_cast_proposals") or {})
        speakers.append(entry)
    return speakers


def sounds_payload(name: str, data: dict, clip: Callable[[Any], str | None]) -> list[dict]:
    """Every shown sound slot, with whether the script loops it and whether it has a clip."""
    from h3_app.drama3.serving.tree import extract_script_sounds

    loops = {}
    if data.get("script_mode") == DRAMA3_SCRIPT_ADVANCED:
        try:
            loops = {item["name"].casefold(): bool(item.get("loop")) for item in extract_script_sounds(data.get("text") or "")}
        except Exception:  # A half-typed script may not parse; the slots stay.
            loops = {}
    sounds = []
    for slot in range(min(int(data.get("shown_sound_slots") or 1), DRAMA3_SOUND_SLOTS)):
        preset = data.get(f"sound_{slot}_preset")
        entry = {"slot": slot, "preset": None, "source": "describe", "clip": None,
                 "save_name": data.get(f"sound_{slot}_save_name") or "",
                 "description": data.get(f"sound_{slot}_description") or "",
                 "duration": int(data.get(f"sound_{slot}_duration") or 5)}
        if _is_saved(preset, UPLOAD_SOUND):
            entry.update(preset=preset, source="saved", clip=clip_url(name, "sound", slot, preset))
        else:
            audio = clip(data.get(f"sound_{slot}_audio"))
            if audio:
                entry.update(source="clip", clip=clip_url(name, "sound", slot, Path(audio).name))
        entry["name"] = entry["save_name"] or (preset if entry["source"] == "saved" else "") or entry["description"]
        entry["open"] = slot == data.get("_open_sound_slot")
        if not entry["name"] and not entry["clip"] and not entry["open"]:
            continue  # An empty slot is no sound yet, unless Add sound just opened it.
        entry["loop"] = loops.get(str(entry["name"]).casefold())
        entry["ready"] = bool(entry["clip"])
        entry["proposal"] = f"sound:{slot}" in (data.get("_cast_proposals") or {})
        sounds.append(entry)
    return sounds


def cast_settings(data: dict) -> dict:
    return {key: _setting(data, key) for key in CAST_SETTINGS}


def boot_payload() -> dict:
    return {"voice_engines": list(VOICE_DESIGN_ENGINES), "qwen_engine": ENGINE_QWEN, "languages": list(LANGUAGES),
            "asr_models": list(DRAMA3_ASR_CHOICES), "slots": {"voices": DRAMA3_VOICE_SLOTS, "sounds": DRAMA3_SOUND_SLOTS}}


def install_cast_routes(app: FastAPI, actions, *, api: str, store, owner_of, session_of, payload, clips: Path,
                        clip: Callable[[Any], str | None], runtime: Callable[[], Any], gpu: bool, interrupt,
                        blocked: Callable[[str, str], str | None] | None = None, admission_lock=None) -> CastJobs:
    """The Cast & sounds routes; ``blocked`` names what keeps a cast job from starting (Auto running).

    Returns the cast runs and jobs, which the project payload and Auto use.
    """
    from h3_app.drama3 import sounds as drama3_sounds
    from h3_app.drama3 import speakers as drama3_speakers

    runs = Runs(busy="This project is already making voices or sounds.", lock=admission_lock,
                blocked=lambda run: blocked(run.owner, run.storyboard_id) if blocked else None)

    def edit(name: str, change: Callable[[dict], Any]):
        return edit_project(store, name, change)

    def match_saved(data: dict, slot: int, match: Callable | None = None) -> None:
        """Give a voiceless speaker the saved voice of the same name in the group (or Global), as the tab does.

        ``match`` is a name lookup over the project's group, made once for several slots.
        """
        preset = data.get(f"voice_{slot}_preset")
        if _is_saved(preset, UPLOAD_VOICE) or clip(data.get(f"voice_{slot}_audio")):
            return
        name = data.get(f"voice_{slot}_save_name") or ""
        if not name.strip() or runtime is None:
            return
        try:
            saved = (match or drama3_speakers.speaker_matcher(runtime(), _setting(data, "speaker_group")))(name)
        except (H3Error, OSError):
            return
        if saved:
            data[f"voice_{slot}_preset"] = saved.key
            data[f"voice_{slot}_save_name"] = saved.name
            if saved.description:
                data[f"voice_{slot}_description"] = saved.description

    def match_all(name: str) -> None:
        """Give every voiceless speaker of ``name`` the group's saved voice of the same name.

        Auto runs this before making voices, so a project prepared before an earlier one saved
        its speakers (a later chapter of a series, say) reuses them instead of designing new ones.
        """
        if runtime is None:
            return

        def change(data):
            try:
                match = drama3_speakers.speaker_matcher(runtime(), _setting(data, "speaker_group"))
            except (H3Error, OSError):
                return
            for slot in range(min(int(data.get("shown_slots") or 1), DRAMA3_VOICE_SLOTS)):
                match_saved(data, slot, match)

        edit(name, change)

    def slot_of(kind: str, slot) -> int:
        slot = int(slot)
        if not 0 <= slot < (DRAMA3_VOICE_SLOTS if kind == "voice" else DRAMA3_SOUND_SLOTS):
            raise H3Error("No such slot.")
        return slot

    def keep(audio) -> str:
        return _keep_clip(str(audio), clips)

    def spool(upload: UploadFile, directory: str) -> Path:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise H3Error(f"{upload.filename} isn't an audio file ({', '.join(sorted(AUDIO_EXTENSIONS))}).")
        path = Path(directory) / f"{uuid.uuid4().hex}{suffix}"
        with path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        return path

    # Speakers and sounds ---------------------------------------------------
    @app.patch(f"{api}/project/cast-settings", include_in_schema=False)
    @guarded
    def save_cast_settings(name: str, request: Request, body: dict = Body(...)):
        if "voice_engine" in body and body["voice_engine"] not in VOICE_DESIGN_ENGINES:
            raise H3Error("Unknown voice generator.")

        def change(data):
            group_before = _setting(data, "speaker_group")
            apply_fields(data, body, CAST_SETTINGS, choices={"voice_engine": VOICE_DESIGN_ENGINES,
                                                            "asr_model": DRAMA3_ASR_CHOICES})
            if _setting(data, "speaker_group") != group_before:
                # A new group's saved voices fill the speakers that have none, as the tab matches them.
                for slot in range(min(int(data.get("shown_slots") or 1), DRAMA3_VOICE_SLOTS)):
                    match_saved(data, slot)

        edit(name, change)
        return payload(owner_of(request), name)

    @app.patch(f"{api}/project/slot", include_in_schema=False)
    @guarded
    def save_slot(name: str, kind: str, slot: int, request: Request, body: dict = Body(...)):
        """A speaker's or sound's details; ``source`` "upload" lets go of a saved voice or sound."""
        slot = slot_of(kind, slot)
        fields = VOICE_FIELDS if kind == "voice" else SOUND_FIELDS
        prefix, upload = slot_keys(kind, slot)

        def change(data):
            values = {}
            apply_fields(values, body, fields,
                         choices={"language": LANGUAGES} if kind == "voice" else None)
            data.update({f"{prefix}_{key}": val for key, val in values.items()})
            if body.get("source") == "upload":
                data[f"{prefix}_preset"] = upload
            elif kind == "voice" and "save_name" in body:
                match_saved(data, slot)

        edit(name, change)
        return payload(owner_of(request), name)

    def set_clip(name: str, kind: str, slot: int, kept: str | None) -> None:
        """Give a slot its own clip (or none), letting go of a saved one; a voice's transcript goes with it."""
        prefix, upload = slot_keys(kind, slot)

        def change(data):
            if kept is None:
                data.pop(f"{prefix}_audio", None)
            else:
                data[f"{prefix}_audio"] = kept
            data[f"{prefix}_preset"] = upload
            if kind == "voice":
                data[f"{prefix}_transcript"] = ""

        edit(name, change)

    @app.post(f"{api}/project/slot/upload", include_in_schema=False)
    @guarded
    def upload_clip(name: str, kind: str, slot: int, request: Request, file: UploadFile = File(...)):
        """A reference clip or sound clip of your own; a new voice clip needs its own transcript."""
        slot = slot_of(kind, slot)
        with TemporaryDirectory(prefix="h3-soundstage-") as directory:
            kept = keep(spool(file, directory))
        set_clip(name, kind, slot, kept)
        return payload(owner_of(request), name)

    @app.delete(f"{api}/project/slot/clip", include_in_schema=False)
    @guarded
    def remove_clip(name: str, kind: str, slot: int, request: Request):
        set_clip(name, kind, slot_of(kind, slot), None)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/slots", include_in_schema=False)
    @guarded
    def resize(name: str, kind: str, request: Request, body: dict = Body(...)):
        """Add a slot, or remove the last one (its details stay, as the tab keeps them)."""
        key, limit = ("shown_slots", DRAMA3_VOICE_SLOTS) if kind == "voice" else ("shown_sound_slots", DRAMA3_SOUND_SLOTS)
        step = 1 if body.get("add") else -1

        def change(data):
            shown = int(data.get(key) or 1)
            if kind == "sound" and step > 0:
                # The page lists only sounds with a name or clip, so a new sound takes the first empty
                # slot, which stays listed (as open) until it is filled or another one is opened.
                data.pop("_open_sound_slot", None)
                listed = {entry["slot"] for entry in sounds_payload(name, {**data, key: limit}, clip)}
                free = next((slot for slot in range(limit) if slot not in listed), None)
                if free is None:
                    raise H3Error("Every sound slot is in use.")
                data["_open_sound_slot"] = free
                data[key] = max(shown, free + 1)
                return
            data[key] = max(1, min(limit, shown + step))
            if kind == "sound" and int(data.get("_open_sound_slot") or 0) >= data[key]:
                data.pop("_open_sound_slot", None)

        edit(name, change)
        return payload(owner_of(request), name)

    # Saved voices and sounds -----------------------------------------------
    @app.get(f"{api}/library", include_in_schema=False)
    @guarded
    def library(request: Request, group: str = "Global"):
        owner_of(request)
        config = runtime()
        groups = drama3_speakers.list_groups(config)
        seen, speakers = set(), []
        for scope in dict.fromkeys([group or "Global", "Global"]):
            for preset in drama3_speakers.list_speakers(config, scope):
                if preset.key not in seen:
                    seen.add(preset.key)
                    speakers.append({"key": preset.key, "name": preset.name, "group": preset.group,
                                     "description": preset.description})
        sounds = [{"key": preset.name, "name": preset.name, "description": preset.description,
                   "duration": preset.duration} for preset in drama3_sounds.list_sounds(config)]
        return {"groups": groups, "speakers": speakers, "sounds": sounds}

    @app.post(f"{api}/library/groups", include_in_schema=False)
    @guarded
    def create_group(request: Request, body: dict = Body(...)):
        owner_of(request)
        return {"group": drama3_speakers.create_group(str(body.get("name") or ""), runtime()),
                "groups": drama3_speakers.list_groups(runtime())}

    @app.post(f"{api}/project/slot/use-saved", include_in_schema=False)
    @guarded
    def use_saved(name: str, kind: str, slot: int, request: Request, body: dict = Body(...)):
        slot = slot_of(kind, slot)
        key = str(body.get("key") or "")
        if kind == "voice":
            saved = drama3_speakers.load_speaker(key, runtime())

            def change(data):
                data[f"voice_{slot}_preset"] = saved.key
                data[f"voice_{slot}_save_name"] = saved.name
                if saved.description:
                    data[f"voice_{slot}_description"] = saved.description
        else:
            saved = drama3_sounds.load_sound(key, runtime())

            def change(data):
                data[f"sound_{slot}_preset"] = saved.name
                data[f"sound_{slot}_save_name"] = saved.name
                data[f"sound_{slot}_description"] = saved.description
                data[f"sound_{slot}_duration"] = int(saved.duration)

        edit(name, change)
        return payload(owner_of(request), name)

    @app.post(f"{api}/project/slot/save", include_in_schema=False)
    @guarded
    def save_to_library(name: str, kind: str, slot: int, request: Request, body: dict = Body(default={})):
        """Keep a speaker in the selected group, or a sound in the sound library, and use the saved one."""
        slot = slot_of(kind, slot)
        data, _ = store.soundstage_project(name)
        if kind == "voice":
            audio = clip(data.get(f"voice_{slot}_audio"))
            saved = drama3_speakers.save_speaker(
                data.get(f"voice_{slot}_save_name") or f"Speaker {slot}", audio, data.get(f"voice_{slot}_transcript") or "",
                runtime(), data.get(f"voice_{slot}_description") or "", body.get("group") or _setting(data, "speaker_group"))
            key, message = saved.key, f'Saved speaker "{saved.name}" to {saved.group}.'
        else:
            audio = clip(data.get(f"sound_{slot}_audio"))
            saved = drama3_sounds.save_sound(data.get(f"sound_{slot}_save_name") or f"sound_{slot}", audio,
                                             data.get(f"sound_{slot}_description") or "",
                                             data.get(f"sound_{slot}_duration") or 5, runtime())
            key, message = saved.name, f'Saved sound "{saved.name}".'

        def change(current):
            current[f"{slot_keys(kind, slot)[0]}_preset"] = key

        edit(name, change)
        return {**payload(owner_of(request), name), "message": message}

    @app.get(f"{api}/clip", include_in_schema=False)
    def slot_clip(name: str, kind: str, slot: int, request: Request):
        """A slot's clip: the project's own, or the saved voice or sound it uses."""
        with found():
            owner_of(request)
            slot = slot_of(kind, slot)
            data, _ = store.soundstage_project(name)
            prefix, upload = slot_keys(kind, slot)
            preset = data.get(f"{prefix}_preset")
            if _is_saved(preset, upload):
                saved = (drama3_speakers.load_speaker(preset, runtime()) if kind == "voice"
                         else drama3_sounds.load_sound(preset, runtime()))
                file = saved.audio
            else:
                file = clip(data.get(f"{prefix}_audio"))
        return file_response(file)

    # Making voices, sounds and transcripts ---------------------------------
    def voice_values(data: dict) -> list[Any]:
        """``generate_drama3_voices``' inputs: design settings, parallel, shown slots, skip, then each slot."""
        settings = [_setting(data, "voice_engine"),
                    *(speech_value(data, key) for key in ("temperature", "top_p", "top_k", "max_frames"))]
        values = [*settings, data.get("parallel") or 16, int(data.get("shown_slots") or 1),
                  bool(_setting(data, "skip_existing"))]
        for slot in range(DRAMA3_VOICE_SLOTS):
            values += [data.get(f"voice_{slot}_preset") or UPLOAD_VOICE, clip(data.get(f"voice_{slot}_audio")) is not None,
                       data.get(f"voice_{slot}_description") or "", data.get(f"voice_{slot}_preview_text") or "",
                       data.get(f"voice_{slot}_language") or "Auto"]
        return values

    def sound_values(data: dict) -> list[Any]:
        values = [int(data.get("shown_sound_slots") or 1), bool(_setting(data, "skip_existing_sounds"))]
        for slot in range(DRAMA3_SOUND_SLOTS):
            values += [data.get(f"sound_{slot}_preset") or UPLOAD_SOUND, clip(data.get(f"sound_{slot}_audio")) is not None,
                       data.get(f"sound_{slot}_save_name") or "", data.get(f"sound_{slot}_description") or "",
                       data.get(f"sound_{slot}_duration") or 5]
        return values

    def slot_source(data, kind, slot):
        prefix = f"{kind}_{slot}_"
        count = "shown_slots" if kind == "voice" else "shown_sound_slots"
        return {"fields": {key: val for key, val in data.items() if key.startswith(prefix)},
                "shown": slot < int(data.get(count) or 1),
                "version": (data.get("_slot_versions") or {}).get(f"{kind}:{slot}", 0)}

    def save_generated(run, name, kind, slot, patch):
        key = f"{kind}:{slot}"
        def change(data):
            if slot_source(data, kind, slot) != run.expected_slots[key]:
                data.setdefault("_cast_proposals", {})[key] = {"id": uuid.uuid4().hex, "patch": patch}
                return False
            changed = any(data.get(field) != val for field, val in patch.items())
            data.update(patch)
            source = slot_source(data, kind, slot)
            if changed:
                source["version"] += 1
            return source
        source = edit(name, change)
        if source is False:
            run.error = "The slot changed while generating. The result is kept for review in Cast & sounds."
        else:
            run.expected_slots[key] = source

    def apply_voice(run, name, slot, audio, transcript):
        patch = {f"voice_{slot}_audio": keep(audio), f"voice_{slot}_preset": UPLOAD_VOICE}
        if transcript is not SKIP and transcript is not None:
            patch[f"voice_{slot}_transcript"] = str(transcript)
        save_generated(run, name, "voice", slot, patch)

    def apply_sound(run, name, slot, audio):
        save_generated(run, name, "sound", slot,
                       {f"sound_{slot}_audio": keep(audio), f"sound_{slot}_preset": UPLOAD_SOUND})

    def apply_outputs(run: CastRun, name: str, target: str, slot, update) -> None:
        """Apply named progress; the adapter owns Gradio's component layout."""
        progress = CastProgress.read(target, slot, update)
        for item in progress.slots:
            key = f"{item.kind}:{item.slot}"
            if isinstance(item.audio, str) and item.audio:
                clip_key = f"{item.kind}-clip:{item.slot}"
                if run.slots.get(clip_key) != item.audio:
                    run.slots[clip_key] = item.audio
                    if item.kind == "voice":
                        apply_voice(run, name, item.slot, item.audio, item.transcript)
                    else:
                        apply_sound(run, name, item.slot, item.audio)
            elif target == TRANSCRIBE and item.transcript is not SKIP and item.transcript is not None:
                save_generated(run, name, "voice", item.slot, {f"voice_{item.slot}_transcript": str(item.transcript)})
            if item.status is not SKIP and item.status:
                run.slots[key] = str(item.status)
                if target != PREPARE:
                    run.status = str(item.status)
        if progress.status:
            run.status = progress.status.split("\n\n")[0]
            run.message = progress.status
        if progress.error:
            run.error = progress.error
        run.revision += 1

    def job_for(data: dict, target: str, slot=None):
        """The job family, callback and inputs that make ``target``; refuses what can't be made yet."""
        if target == VOICE:
            slot = slot_of("voice", slot)
            if not (data.get(f"voice_{slot}_description") or "").strip() or not (data.get(f"voice_{slot}_preview_text") or "").strip():
                raise H3Error("Add a voice description and the line it reads first.")
            settings = voice_values(data)[:5]
            return slot, "drama3_voice", actions.generate_drama3_voice, [
                data.get(f"voice_{slot}_description"), data.get(f"voice_{slot}_preview_text"),
                data.get(f"voice_{slot}_language") or "Auto", *settings]
        if target == SOUND:
            slot = slot_of("sound", slot)
            if not (data.get(f"sound_{slot}_description") or "").strip():
                raise H3Error("Describe the sound first.")
            return slot, "drama3_sound", actions.generate_drama3_sound, [
                data.get(f"sound_{slot}_description"), data.get(f"sound_{slot}_duration") or 5]
        if target == TRANSCRIBE:
            slot = slot_of("voice", slot)
            audio = clip(data.get(f"voice_{slot}_audio"))
            if not audio:
                raise H3Error("Upload a reference clip first.")
            return slot, "drama3_voice", actions.transcribe_drama3_voice, [audio, _setting(data, "asr_model")]
        if target == PREPARE:
            if actions.prepare_drama3_cast is None:
                raise H3Error("Making the whole cast isn't available here.")
            return None, "drama3_prepare", actions.prepare_drama3_cast, [
                data.get("script_mode"), *voice_values(data), *sound_values(data)]
        raise H3Error("Unknown request.")

    def execute(run, owner: str, session, name: str, target: str, slot, family: str, callback, values, *, expected=None) -> None:
        """Run one cast job to its end on the usual job path, writing each frame into the project."""
        expected = expected if expected is not None else store.soundstage_project(name)[0]
        run.expected_slots = {f"{kind}:{index}": slot_source(expected, kind, index)
                              for kind, count in (("voice", DRAMA3_VOICE_SLOTS), ("sound", DRAMA3_SOUND_SLOTS))
                              for index in range(count)}
        for update in run_job(run, owner, session, family, values, owned_generation(callback, family)):
            apply_outputs(run, name, target, slot, update if isinstance(update, (list, tuple)) else (update,))
        if run.error is None:
            failed = [note for key, note in run.slots.items() if "-clip:" not in key and "failed" in note]
            if target in (VOICE, SOUND) and failed:
                run.error = failed[-1]
            run.status = run.status if run.error else (run.status or "Done.")

    def save_all(name: str) -> str:
        """Keep every voiced speaker in the selected group and use the saved voice, as the tab's Save all does."""
        saved, errors = 0, []
        with editing(store, name) as data:
            group = _setting(data, "speaker_group")
            for slot in range(min(int(data.get("shown_slots") or 1), DRAMA3_VOICE_SLOTS)):
                preset = data.get(f"voice_{slot}_preset")
                try:
                    if _is_saved(preset, UPLOAD_VOICE):
                        source = drama3_speakers.load_speaker(preset, runtime())
                        speaker, audio, transcript, description = (source.name, source.audio, source.transcript,
                                                                   source.description)
                    else:
                        audio = clip(data.get(f"voice_{slot}_audio"))
                        speaker = data.get(f"voice_{slot}_save_name") or f"Speaker {slot}"
                        transcript = data.get(f"voice_{slot}_transcript") or ""
                        description = data.get(f"voice_{slot}_description") or ""
                    if not audio:
                        continue
                    kept = drama3_speakers.save_speaker(speaker, audio, transcript, runtime(), description, group)
                    data[f"voice_{slot}_preset"] = kept.key
                    saved += 1
                except H3Error as exc:
                    errors.append(f"Speaker {slot}: {exc}")
        return " ".join([f'Saved {saved} speaker{"s" if saved != 1 else ""} to "{group}".', *errors])

    @app.post(f"{api}/project/make", include_in_schema=False)
    @guarded
    def make(name: str, request: Request, body: dict = Body(...)):
        """Make one speaker's voice, one sound, one transcript, or everything missing."""
        if not gpu:
            raise H3Error("Open Soundstage in Studio to make voices and sounds on the GPU.")
        owner = owner_of(request)
        if blocked is not None and (reason := blocked(owner, name)):
            raise H3Error(reason)
        target = body.get("target")
        data, _ = store.soundstage_project(name)
        slot, family, callback, values = job_for(data, target, body.get("slot"))
        session = session_of(request)
        run = CastRun(owner, name, "cast", target=f"{target}:{slot}" if slot is not None else target)
        run.status = "Waiting for the GPU…"
        runs.start(run, lambda run: execute(run, owner, session, name, target, slot, family, callback, values, expected=data))
        return {"cast_run": run.public(None)}

    @app.get(f"{api}/project/slot/proposal", include_in_schema=False)
    @guarded
    def slot_proposal(name: str, kind: str, slot: int, request: Request):
        owner_of(request)
        slot = slot_of(kind, slot)
        data, _ = store.soundstage_project(name)
        proposal = (data.get("_cast_proposals") or {}).get(f"{kind}:{slot}")
        if not proposal:
            raise H3Error("There is no saved result for this slot.")
        patch = proposal.get("patch", proposal)
        audio = patch.get(f"{kind}_{slot}_audio")
        return {"id": proposal_token(proposal), "transcript": patch.get(f"{kind}_{slot}_transcript"), "audio": bool(audio)}

    @app.get(f"{api}/project/slot/proposal/audio", include_in_schema=False)
    def slot_proposal_audio(name: str, kind: str, slot: int, request: Request, id: str):
        with found():
            owner_of(request)
            slot = slot_of(kind, slot)
            data, _ = store.soundstage_project(name)
            proposal = (data.get("_cast_proposals") or {}).get(f"{kind}:{slot}", {})
            check_proposal(proposal, id)
            patch = proposal.get("patch", proposal)
            return file_response(clip(patch.get(f"{kind}_{slot}_audio")), cache="no-cache")

    @app.post(f"{api}/project/slot/proposal", include_in_schema=False)
    @guarded
    def apply_slot_proposal(name: str, kind: str, slot: int, request: Request, body: dict = Body(default={})):
        owner = owner_of(request)
        slot = slot_of(kind, slot)
        with editing(store, name) as data:
            count = "shown_slots" if kind == "voice" else "shown_sound_slots"
            if slot >= int(data.get(count) or 1):
                raise H3Error("Add this slot again before applying its saved result.")
            proposals = data.get("_cast_proposals") or {}
            proposal = proposals.get(f"{kind}:{slot}")
            if not proposal:
                raise H3Error("There is no saved result for this slot.")
            check_proposal(proposal, body.get("id"))
            patch = proposal.get("patch", proposal)
            proposals.pop(f"{kind}:{slot}")
            data.update(patch)
        return payload(owner, name)

    @app.post(f"{api}/project/save-all", include_in_schema=False)
    @guarded
    def save_all_speakers(name: str, request: Request):
        return {**payload(owner_of(request), name), "message": save_all(name)}

    @app.post(f"{api}/project/make/interrupt", include_in_schema=False)
    @guarded
    def stop_making(name: str, request: Request):
        if interrupt is None:
            raise H3Error("Nothing can run here.")
        owner = owner_of(request)
        run = runs.active(owner, name, "cast")
        if run is not None:
            run.stop_requested = True
        if run is None:
            message = "This project isn't making anything."
        elif run.job_id is None:
            message = "The run is still starting. Try again in a moment."
        else:
            message = interrupt(session_of(request), family=run.family, job_id=run.job_id)
        return {"message": message}

    return CastJobs(runs=runs, job_for=job_for, execute=execute, save_all=save_all, match_all=match_all)
