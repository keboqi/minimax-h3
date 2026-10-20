"""Drama 3 event handlers: writer, cast, voice design, sounds and SGLang-Omni speech.

The application installs its runtime services once at import; handlers read them on
each call, so tests can patch the accessors below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import html
from pathlib import Path
import re
from typing import Any, Callable

import gradio as gr

from h3_app import prompt_service
from h3_app.errors import H3Error
from h3_app.generation import requests as generation_requests
from h3_app.jobs import JobCancelled


@dataclass(frozen=True)
class Drama3Services:
    runtime_config: Callable[[], Any]
    stage_file: Callable[..., str]
    generation_services: Callable[[], Any]
    free_gpu: Callable[[], None]
    # Maps Gradio's cached copy of a result back to the output that has its snapshot.
    resolve_result: Callable[[Any], Any] = lambda path: path


_SERVICES: Drama3Services | None = None


def install(services: Drama3Services) -> None:
    global _SERVICES
    _SERVICES = services


def _services() -> Drama3Services:
    if _SERVICES is None:
        raise RuntimeError("Drama 3 handlers need install() before use.")
    return _SERVICES


def _runtime_config():
    return _services().runtime_config()


def stage_file(path, category, transcode_video=False, reuse=False):
    return _services().stage_file(path, category, transcode_video, reuse)


def _generation_services():
    return _services().generation_services()


def free_comfy_vram():
    """Release ComfyUI's VRAM before SGLang-Omni, VoiceDesign or Stable Audio takes the GPU."""
    _services().free_gpu()


def _drama3_voices(slots, saved_presets=None):
    """Resolve (preset, clip, transcript) slot triples to (speaker, clip, transcript).

    ``saved_presets``, when given, collects each loaded saved speaker by slot.
    """
    from h3_app import drama3_speakers
    from h3_ui.drama3_controls import UPLOAD_VOICE

    voices, names = [], []
    for speaker in range(len(slots) // 3):
        preset, audio, transcript = slots[3 * speaker:3 * speaker + 3]
        if preset and preset != UPLOAD_VOICE:
            saved = drama3_speakers.load_speaker(preset, _runtime_config())
            if saved_presets is not None:
                saved_presets[speaker] = saved
            voices.append((speaker, str(saved.audio), saved.transcript))
            names.append((speaker, saved.name))
        elif audio or str(transcript or "").strip():
            if not audio:
                raise H3Error(f"Speaker {speaker} has a transcript but no reference clip.")
            voices.append((speaker, str(audio), str(transcript or "")))
            names.append((speaker, ""))
    return tuple(voices), names


def _drama3_sounds(slots):
    """Resolve (preset, audio, save_name, description, duration) sound slot 5-tuples."""
    from h3_app import drama3_sounds
    from h3_app.drama3_serving.tree import _default_sound_name
    from h3_app.stable_audio import MAX_DURATION_SECONDS as MAX_SOUND_SECONDS
    from h3_ui.drama3_controls import UPLOAD_SOUND

    sounds = []
    for index in range(len(slots) // 5):
        preset, audio, save_name, description, duration = slots[5 * index:5 * index + 5]
        if preset and preset != UPLOAD_SOUND:
            saved = drama3_sounds.load_sound(preset, _runtime_config())
            sounds.append((saved.name, str(saved.audio), saved.description, int(saved.duration)))
        elif audio or str(save_name or "").strip() or str(description or "").strip():
            desc = str(description or "").strip()
            # An unnamed slot takes the name an unnamed <sound> tag with this description gets.
            name = str(save_name or "").strip() or (_default_sound_name(desc) if desc else f"sound_{index}")
            desc = desc or name
            try:
                dur = max(1, min(MAX_SOUND_SECONDS, int(round(float(duration or 5)))))
            except (TypeError, ValueError):
                dur = 5
            sounds.append((name, str(audio or ""), desc, dur))
    return tuple(sounds)


def generate_drama3(*values, progress=gr.Progress(track_tqdm=False)):
    """Inputs: text, then (voice, audio, transcript) per speaker slot, then settings and sound slots.

    The parameter list is generated from the slot count; see ``_drama3_signature``.
    Lines with a saved take for the same text, voice and sampling are reused.
    """
    yield from _drama3_job(values, progress)


def regenerate_drama3_line(line, *values, progress=gr.Progress(track_tqdm=False)):
    """Generate one line's take again (``line`` is its take key from the takes table), then remix."""
    if not line:
        raise gr.Error("Choose a line from the takes table first.")
    yield from _drama3_job(values, progress, regenerate=(str(line),))


def remix_drama3(*values, progress=gr.Progress(track_tqdm=False)):
    """Rebuild the mix from saved takes, with the current sounds and effects; nothing is generated."""
    yield from _drama3_job(values, progress, remix_only=True)


def _result_settings(output):
    """A result's settings snapshot, or {}; ``output`` may be Gradio's cached copy."""
    from h3_app.provenance import read_snapshot

    if isinstance(output, dict):
        output = output.get("path") or output.get("value")
    snapshot = read_snapshot(_services().resolve_result(output)) if output else None
    return (snapshot or {}).get("settings") or {}


def _take_chunks(output):
    """The chunks with a saved take, in script order, from a result's settings snapshot."""
    return [chunk for chunk in _result_settings(output).get("chunks") or [] if chunk.get("take")]


_CUE = re.compile(r"<\|[^|]*\|>|<[^>]*>|\[[^\]]*\]")


def drama3_subtitles(output):
    """Subtitle lines for a result, timed to the mix; the browser highlights the one playing.

    Results from before line timings were recorded have none, and the panel stays hidden.
    """
    if isinstance(output, dict):
        output = output.get("path") or output.get("value")
    settings = _result_settings(output)
    lines = []
    for chunk in settings.get("chunks") or []:
        text = " ".join(_CUE.sub(" ", str(chunk.get("text", ""))).split())
        if not text or chunk.get("start") is None:
            continue
        speaker = chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}"
        lines.append(
            f'<li data-start="{float(chunk["start"]):.3f}" data-end="{float(chunk.get("end", chunk["start"])):.3f}">'
            f'<span class="h3-subtitle-speaker">{html.escape(str(speaker))}</span>'
            f'<span class="h3-subtitle-text">{html.escape(text)}</span></li>')
    if not lines:
        return gr.update(value="", visible=False)
    # The result's 32-hex id is in every copy's file name, so the browser can find its player.
    media = re.search(r"[0-9a-f]{32}", Path(str(output)).name)
    return gr.update(visible=True, value=(
        f'<div class="h3-drama3-subtitles" data-media="{media.group(0) if media else ""}" '
        f'data-duration="{float(settings.get("duration") or 0):.3f}">'
        f'<ol>{"".join(lines)}</ol></div>'))


def drama3_take_selected(output, evt: gr.SelectData):
    """The take key of the takes-table row the user clicked."""
    chunks = _take_chunks(output)
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not isinstance(row, int) or not 0 <= row < len(chunks):
        return gr.skip()
    return chunks[row]["take"]


def drama3_take_audio(key):
    """One line's saved take, before staging and speaker effects, for the line player."""
    from h3_app.drama3_serving.takes import load_take, takes_root

    key = str(key or "")
    # Keys are hex digests; anything else must not become a path.
    loaded = load_take(takes_root(_runtime_config()), key) if key.isalnum() else None
    if loaded is None:
        return None
    samples, rate = loaded
    return rate, samples


def drama3_takes(output):
    """Takes table rows and line choices for a generated result, from its settings snapshot."""
    chunks = _take_chunks(output)
    rows, choices = [], []
    for number, chunk in enumerate(chunks, 1):
        speaker = chunk.get("name") or f"Speaker {chunk.get('speaker', 0)}"
        text = " ".join(str(chunk.get("text", "")).split())
        take = "reused" if chunk.get("reused") else chunk.get("check") or ""
        if take == "skipped":
            take = "new"
        rows.append([number, speaker, text, chunk.get("seconds", ""), take])
        short = text if len(text) <= 48 else text[:47] + "…"
        choices.append((f"{number} · {speaker} · {short}", chunk["take"]))
    return (gr.update(value=rows, visible=bool(rows)),
            gr.update(choices=choices, value=choices[0][1] if choices else None, visible=bool(rows)),
            gr.update(visible=bool(rows)))


def _cast_names(names, slot_names):
    """Every Cast name by slot: saved speakers' names (``names``) win over typed ones."""
    cast = {speaker: " ".join(str(name or "").split()) for speaker, name in enumerate(slot_names)}
    cast.update((speaker, name) for speaker, name in names if name)
    return sorted((speaker, name) for speaker, name in cast.items() if name)


def _drama3_job(values, progress, **options):
    from h3_app.generation.optional_engines import generate_optional_engine, validate_request
    from h3_app.drama3_cast import clean_title
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_SOUND_SLOT_INPUT_NAMES
    slot_count = len(DRAMA3_SLOT_INPUT_NAMES)
    text, slots, settings = values[0], values[1:1 + slot_count], list(values[1 + slot_count:])
    defaults = [1.0, 0.9, 0, 8192, 8, 0]
    if len(settings) < len(defaults):
        settings += defaults[len(settings):]
    temperature, top_p, top_k, max_frames, parallel, retries = settings[:6]
    script_mode = settings[6] if len(settings) > 6 and settings[6] else "Basic"
    # The sound slots, then the title, which names the output file.
    sound_end = 7 + len(DRAMA3_SOUND_SLOT_INPUT_NAMES)
    sound_slots = settings[7:sound_end]
    title = clean_title(settings[sound_end]) if len(settings) > sound_end else ""
    voices, names = _drama3_voices(slots)
    # Lines start with these names; each line is sent with its named speaker's clip.
    names = _cast_names(names, settings[sound_end + 1:])
    sounds = _drama3_sounds(sound_slots)
    request = generation_requests.Drama3Request(
        text, voices, temperature, top_p, top_k, max_frames, parallel, retries, tuple(names),
        script_mode=script_mode, sounds=sounds, title=title, **options,
    )
    validate_request(request)
    # SGLang-Omni reads staged clips by absolute path.
    input_dir = _runtime_config().input_dir
    request = replace(request, voices=tuple(
        (speaker, str(input_dir / stage_file(audio, "drama3_voices", reuse=True)), transcript)
        for speaker, audio, transcript in request.voices
    ))
    if request.sounds:
        request = replace(request, sounds=tuple(
            (name, str(input_dir / stage_file(audio, "drama3_sounds", reuse=True)) if audio else "",
             description, duration)
            for name, audio, description, duration in request.sounds
        ))
    yield from generate_optional_engine(request, _generation_services(), _runtime_config(),
                                        progress=progress, free_gpu=free_comfy_vram)


def _drama3_signature(leading, trailing):
    """Named API parameters around the generated per-slot inputs."""
    import inspect
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES

    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = [inspect.Parameter(name, positional) for name in (*leading, *DRAMA3_SLOT_INPUT_NAMES)]
    parameters += [inspect.Parameter(name, positional, **({"default": default} if default is not inspect.Parameter.empty else {}))
                   for name, default in trailing]
    return parameters


def _drama3_writer_slots(slots):
    """Split writer inputs into voice triples, cast metadata and the shown slot count.

    Returns (names, existing, cast_context, shown): reference speaker names for the
    voice context, known slot identities by ID, and the JSON slot context line.
    """
    import json
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VOICE_SLOTS, DRAMA3_VISIBLE_SLOTS
    from h3_app.drama3_speakers import normalize_name

    count = len(DRAMA3_SLOT_INPUT_NAMES)
    voice_values = slots[:count]
    metadata = list(slots[count:count * 2])
    metadata += [""] * (count - len(metadata))
    shown = slots[count * 2] if len(slots) > count * 2 else DRAMA3_VISIBLE_SLOTS
    saved_presets = {}
    voices, names = _drama3_voices(voice_values, saved_presets)
    references = {speaker for speaker, _audio, _transcript in voices}
    saved_names = dict(names)
    context = []
    for speaker in range(DRAMA3_VOICE_SLOTS):
        name, description, _preview = metadata[3 * speaker:3 * speaker + 3]
        if speaker in saved_presets:
            description = saved_presets[speaker].description or description
        name = saved_names.get(speaker) or name or ""
        name = normalize_name(name) if str(name).strip() else ""
        if speaker in references or name or description:
            context.append({"speaker_id": speaker, "name": name,
                            "description": description or "", "has_reference": speaker in references})
    existing = {entry["speaker_id"]: entry for entry in context}
    names = [(speaker, existing[speaker]["name"]) for speaker in sorted(references)]
    cast_context = ("Existing slot identities (preserve IDs and known names/descriptions; "
                    "has_reference=false is a design brief, not a cloned voice):\n"
                    + json.dumps(context, ensure_ascii=False))
    group = _drama3_writer_group(slots)
    if group is not None:
        from h3_app import drama3_speakers
        available = {}
        for scope in dict.fromkeys([group or "Global", "Global"]):
            for preset in drama3_speakers.list_speakers(_runtime_config(), scope):
                available.setdefault(drama3_speakers.identity(preset.name),
                                     {"name": preset.name, "description": preset.description})
        cast_context += ("\nReusable speaker voices (match canonical names when the character appears; "
                         "do not add unused characters; these voices will be attached automatically):\n"
                         + json.dumps(list(available.values()), ensure_ascii=False))
    return names, existing, cast_context, shown


def _slot_names(existing):
    """Speaker ID to Cast name, from the writer's slot identities."""
    return {speaker: entry["name"] for speaker, entry in existing.items() if entry.get("name")}


def _drama3_writer_mode_and_sounds(slots):
    """Extract optional script_mode and saved sound preset context for writer prompts."""
    import json
    from h3_app import drama3_sounds
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES

    count = len(DRAMA3_SLOT_INPUT_NAMES)
    script_mode = (
        slots[count * 2 + 1]
        if len(slots) > count * 2 + 1 and slots[count * 2 + 1]
        else prompt_service.DRAMA3_SCRIPT_BASIC
    )
    sound_context = ""
    if script_mode == prompt_service.DRAMA3_SCRIPT_ADVANCED:
        saved = drama3_sounds.list_sounds(_runtime_config())
        if saved:
            items = [
                {"name": preset.name, "description": preset.description, "duration": preset.duration}
                for preset in saved
            ]
            sound_context = (
                "Existing saved sound presets (reuse matching names/descriptions when fitting):\n"
                + json.dumps(items, ensure_ascii=False)
            )
    return script_mode, sound_context


def _drama3_writer_group(slots):
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES
    index = 2 * len(DRAMA3_SLOT_INPUT_NAMES) + 2
    return (slots[index] or "Global") if len(slots) > index else None


def _drama3_unchanged_outputs(group=None):
    from h3_ui.drama3_controls import DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VOICE_SLOTS
    # Slot metadata, visibility, Add/Remove, presets with a group, then the title.
    return [gr.skip()] * (len(DRAMA3_SLOT_INPUT_NAMES) + DRAMA3_VOICE_SLOTS + 3
                          + (DRAMA3_VOICE_SLOTS if group is not None else 0) + 1)


def _drama3_apply_cast(script, status, cast, existing, shown, group=None, title=""):
    """Script, status, slot metadata, visibility, presets and title for a validated cast.

    Raises on reassignment. A reply without a title keeps the one in the box.
    """
    title_update = title if title else gr.skip()
    return (*_drama3_cast_updates(script, status, cast, existing, shown, group), title_update)


def _drama3_cast_updates(script, status, cast, existing, shown, group=None):
    from h3_ui.drama3_controls import (
        DRAMA3_SLOT_INPUT_NAMES, DRAMA3_VISIBLE_SLOTS, DRAMA3_VOICE_SLOTS, _slot_visibility
    )

    from h3_app import drama3_speakers
    from h3_app.workflows.drama3 import format_script, tags_to_names
    presets = [gr.skip()] * DRAMA3_VOICE_SLOTS if group is not None else []
    updates = [gr.skip()] * len(DRAMA3_SLOT_INPUT_NAMES)
    # The names the slots end up with, which the script's lines show.
    final_names = {speaker: entry.get("name", "") for speaker, entry in existing.items()}
    match = drama3_speakers.speaker_matcher(_runtime_config(), group) if group is not None and cast else None
    for entry in cast:
        speaker = entry["speaker_id"]
        previous = existing.get(speaker, {})
        if previous.get("name") and drama3_speakers.identity(previous["name"]) != drama3_speakers.identity(entry["name"]):
            raise H3Error(f"The writer reassigned speaker {speaker} from {previous['name']} to {entry['name']}.")
        # A reference's authored metadata remains authoritative; the writer has not heard the clip.
        name = previous.get("name") or entry["name"]
        description = previous.get("description") or entry["description"]
        if group is not None and not previous.get("has_reference"):
            saved = match(name)
            if saved:
                presets[speaker] = gr.update(value=saved.key)
                name, description = saved.name, saved.description
        updates[3 * speaker:3 * speaker + 3] = [name, description, entry["preview_text"]]
        final_names[speaker] = name
    script = format_script(tags_to_names(script, final_names))
    if not cast:
        return (script, status, *updates, *[gr.skip()] * (len(_slot_visibility(shown))), *presets)
    previous = int(shown or DRAMA3_VISIBLE_SLOTS)
    shown = max(previous, max(entry["speaker_id"] for entry in cast) + 1)
    return (script, status + f" Filled {len(cast)} matching speaker slots; "
            "open Speaker voices to upload or generate references.", *updates,
            *_slot_visibility(shown, previous), *presets)


def enhance_drama3_text(text, speaker_mode, model, temporary_api_key, backend, lightning_api_key,
                        lightning_model, *slots):
    """Apply indexed cast metadata and slot visibility only after validating the complete result."""
    group = _drama3_writer_group(slots)
    try:
        names, existing, cast_context, shown = _drama3_writer_slots(slots)
        script_mode, sound_context = _drama3_writer_mode_and_sounds(slots)
        script, status, cast, title = prompt_service.enhance_drama3_text(
            text, model, temporary_api_key, names, backend=backend,
            lightning_api_key=lightning_api_key, lightning_model=lightning_model,
            speaker_mode=speaker_mode, runtime=_runtime_config(), return_cast=True,
            cast_context=cast_context, script_mode=script_mode, sound_context=sound_context,
            return_title=True, speaker_names=_slot_names(existing),
        )
        if status.startswith("Prompt enhancement failed:"):
            return text, status, *_drama3_unchanged_outputs(group)
        return _drama3_apply_cast(script, status, cast, existing, shown, group, title)
    except H3Error as exc:
        return text, f"Prompt enhancement failed: {exc}", *_drama3_unchanged_outputs(group)


def copy_drama3_writer_prompt(text, speaker_mode, *slots):
    """The full writer prompt (system prompt, slot context and speech text) for an external chat model."""
    try:
        names, _existing, cast_context, _shown = _drama3_writer_slots(slots)
        script_mode, sound_context = _drama3_writer_mode_and_sounds(slots)
        prompt = prompt_service.drama3_full_prompt(
            text, names, speaker_mode, runtime=_runtime_config(),
            cast_context=cast_context, script_mode=script_mode, sound_context=sound_context,
        )
    except H3Error as exc:
        return gr.skip(), f"Could not build the writer prompt: {exc}"
    return (gr.update(value=prompt, visible=True),
            "Copy the full prompt into any chat model, then paste its JSON reply below and apply it.")


def apply_drama3_writer_reply(reply, text, speaker_mode, *slots):
    """Validate a pasted reply like the built-in writer, then fill the script and speaker slots."""
    group = _drama3_writer_group(slots)
    try:
        names, existing, _cast_context, shown = _drama3_writer_slots(slots)
        script_mode, _sound_context = _drama3_writer_mode_and_sounds(slots)
        script, status, cast, title = prompt_service.apply_drama3_reply(
            reply, text, names, speaker_mode, script_mode=script_mode, return_title=True,
            speaker_names=_slot_names(existing),
        )
        if status.startswith("Prompt enhancement failed:"):
            return text, status, *_drama3_unchanged_outputs(group)
        return _drama3_apply_cast(script, status, cast, existing, shown, group, title)
    except H3Error as exc:
        return text, f"Prompt enhancement failed: {exc}", *_drama3_unchanged_outputs(group)


def switch_drama3_script_mode(script_mode, text):
    """Show the sounds panel for Advanced; Basic keeps but ignores the script's sound tags."""
    from h3_app.drama3_serving.tree import has_drama3_sound_features

    if script_mode == prompt_service.DRAMA3_SCRIPT_ADVANCED:
        return (gr.update(visible=True),
                "Advanced sound-enhanced mode enabled (sounds use Stable Audio 3 Medium).")
    # The script is left as authored, so switching back to Advanced restores its staging.
    status = "Basic script mode enabled."
    if has_drama3_sound_features(str(text or "")):
        status += (" The script's sound, staging, pause and vocal effect tags are kept but ignored "
                   "in Basic mode; switch back to Advanced to use them.")
    return gr.update(visible=False), status


def sync_drama3_script_sounds(text, script_mode, shown_sound_slots, *sound_meta_values):
    """Populate sound slot names, descriptions, and durations from <sound> tags in the script."""
    from h3_app.drama3_serving.tree import extract_script_sounds
    from h3_ui.drama3_controls import DRAMA3_SOUND_SLOTS, DRAMA3_VISIBLE_SOUND_SLOTS, _sound_slot_visibility

    total_meta = 3 * DRAMA3_SOUND_SLOTS
    updates = [gr.skip()] * total_meta
    shown = int(shown_sound_slots or DRAMA3_VISIBLE_SOUND_SLOTS)
    unchanged = (*updates, *_sound_slot_visibility(shown, shown))
    if script_mode != prompt_service.DRAMA3_SCRIPT_ADVANCED:
        return unchanged
    extracted = extract_script_sounds(str(text or ""))
    if not extracted:
        return unchanged

    existing_by_name = {}
    used_slots = set()
    for idx in range(DRAMA3_SOUND_SLOTS):
        triple = sound_meta_values[3 * idx:3 * idx + 3] if len(sound_meta_values) >= 3 * (idx + 1) else ("", "", 5)
        s_name = str(triple[0] or "").strip()
        s_desc = str(triple[1] or "").strip()
        s_dur = triple[2]
        if s_name:
            existing_by_name[s_name.casefold()] = (idx, s_name, s_desc, s_dur)
            used_slots.add(idx)

    max_slot = shown - 1
    next_free = 0
    for item in extracted:
        key = item["name"].casefold()
        if key in existing_by_name:
            idx, prev_name, prev_desc, prev_dur = existing_by_name[key]
            name = prev_name or item["name"]
            desc = prev_desc or item["description"]
            dur = prev_dur or item["duration"]
        else:
            while next_free < DRAMA3_SOUND_SLOTS and next_free in used_slots:
                next_free += 1
            if next_free >= DRAMA3_SOUND_SLOTS:
                break
            idx = next_free
            used_slots.add(idx)
            name = item["name"]
            desc = item["description"]
            dur = item["duration"]
        max_slot = max(max_slot, idx)
        updates[3 * idx:3 * idx + 3] = [name, desc, int(dur)]

    return (*updates, *_sound_slot_visibility(max_slot + 1, shown))


def _publish_drama3_signatures():
    import inspect
    from h3_ui.drama3_controls import (
        DRAMA3_CAST_INPUT_NAMES, DRAMA3_SOUND_SLOT_INPUT_NAMES, DRAMA3_VISIBLE_SLOTS, DRAMA3_VOICE_SLOTS,
        UPLOAD_SOUND,
    )

    empty = inspect.Parameter.empty
    sound_defaults = []
    for name in DRAMA3_SOUND_SLOT_INPUT_NAMES:
        if name.endswith("_preset"):
            sound_defaults.append((name, UPLOAD_SOUND))
        elif name.endswith("_audio"):
            sound_defaults.append((name, None))
        elif name.endswith("_duration"):
            sound_defaults.append((name, 5))
        else:
            sound_defaults.append((name, ""))
    progress = inspect.Parameter("progress", inspect.Parameter.KEYWORD_ONLY, default=gr.Progress(track_tqdm=False))
    generation = _drama3_signature(("text",), (
        ("temperature", empty), ("top_p", empty), ("top_k", empty), ("max_frames", empty),
        ("parallel", 8), ("retries", 0), ("script_mode", "Basic"),
        *sound_defaults, ("title", ""),
        # Cast names, so scripts can name their speakers ("Mia: Hello.") instead of tagging them.
        *((f"speaker_{speaker}_name", "") for speaker in range(DRAMA3_VOICE_SLOTS)),
    ))
    generate_drama3.__signature__ = inspect.Signature([*generation, progress])
    remix_drama3.__signature__ = inspect.Signature([*generation, progress])
    regenerate_drama3_line.__signature__ = inspect.Signature([
        inspect.Parameter("line", inspect.Parameter.POSITIONAL_OR_KEYWORD), *generation, progress])
    enhance_drama3_text.__signature__ = inspect.Signature([*_drama3_signature(
        ("text", "speaker_mode", "model", "temporary_api_key", "backend", "lightning_api_key",
         "lightning_model"), ()),
        *[inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default="")
          for name in DRAMA3_CAST_INPUT_NAMES],
        inspect.Parameter("shown_slots", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=DRAMA3_VISIBLE_SLOTS),
        inspect.Parameter("script_mode", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="Basic"),
        inspect.Parameter("speaker_group", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="Global"),
    ])


_publish_drama3_signatures()


def transcribe_drama3_voice(audio, model_choice="1.7B (quality)"):
    """Fill a voice slot's transcript with Qwen3-ASR running in ComfyUI."""
    from h3_app.generation.optional_engines import transcribe_drama3_audio
    if not audio:
        return gr.skip(), "Upload a reference clip first."
    try:
        staged = stage_file(audio, "drama3_asr")
        transcript = transcribe_drama3_audio(staged, _generation_services(), model_choice=model_choice)
    except Exception as exc:
        return gr.skip(), f"Transcription failed: {exc}"
    return transcript, "Transcribed with Qwen3-ASR. Check names and punctuation before generating."


def generate_drama3_voice(description, preview_text, language, engine="Qwen3-TTS VoiceDesign",
                          temperature=1.0, top_p=0.9, top_k=0, max_frames=8192):
    """A generated reference and its synthesis text always update the same slot together.

    Drama 3 designs on SGLang-Omni with the tab's sampling; Qwen uses its isolated runtime.
    """
    from h3_app.drama3_voice_design import ENGINE_DRAMA3, ENGINE_QWEN, generate_drama3_reference
    from h3_app.voice_design import generate_reference

    try:
        if engine == ENGINE_QWEN:
            yield gr.skip(), gr.skip(), "Preparing Qwen VoiceDesign; runtime and weights download on first use."
            audio, transcript = generate_reference(description, preview_text, language, _runtime_config(),
                                                   free_gpu=free_comfy_vram)
        elif engine == ENGINE_DRAMA3:
            from h3_models import resolve_hf_token

            request = _drama3_design_request(temperature, top_p, top_k, max_frames)
            yield gr.skip(), gr.skip(), "Preparing Drama 3 voice design on SGLang-Omni."
            audio = transcript = None
            for update in generate_drama3_reference(description, preview_text, request, _runtime_config(),
                                                    _generation_services(), free_gpu=free_comfy_vram,
                                                    hf_token=resolve_hf_token()):
                if update.output:
                    audio, transcript = update.output, update.status
                else:
                    yield gr.skip(), gr.skip(), update.status
        else:
            raise H3Error("Choose Drama 3 or Qwen3-TTS VoiceDesign to generate a voice.")
    except Exception as exc:
        yield gr.skip(), gr.skip(), f"Voice generation failed: {exc}"
        return
    yield str(audio), gr.update(value=transcript, interactive=True), "Voice generated. Preview speech copied as its exact transcript; ready to save."


DRAMA3_VOICE_DESIGN_SETTINGS = ("engine", "temperature", "top_p", "top_k", "max_frames")


def _drama3_design_request(temperature, top_p, top_k, max_frames, parallel=8):
    """The tab's sampling for voice design; text and voices are filled per take."""
    return generation_requests.Drama3Request(
        "", (), temperature, top_p, top_k, max_frames, int(parallel or 1))


def generate_drama3_voices(*values):
    """Generate each shown upload slot that has a design brief and no clip yet.

    Inputs: the voice-design settings, parallel requests, the shown slot count,
    whether to skip speakers that have a clip, then (preset, has_clip,
    description, preview_text, language) per slot. Outputs:
    (audio, transcript, status) per slot, then the overall status. Drama 3 on
    SGLang-Omni generates up to ``parallel`` voices at once; Qwen runs them in turn.
    """
    from h3_app.drama3_voice_design import ENGINE_DRAMA3
    from h3_ui.drama3_controls import DRAMA3_VOICE_SLOTS, UPLOAD_VOICE

    settings = values[:len(DRAMA3_VOICE_DESIGN_SETTINGS)]
    parallel = values[len(settings)]
    shown = int(values[len(settings) + 1] or 0)
    skip_existing = values[len(settings) + 2] is not False
    slots = values[len(settings) + 3:]
    updates = [gr.skip()] * (3 * DRAMA3_VOICE_SLOTS)

    def frame(status):
        # Transcripts and statuses are sent once, but a clip repeats in every later
        # frame: Gradio's client applies streamed diffs in place to the value it gave
        # the Audio, so a skip after a clip turns that value into {"__type__": "update"}
        # and the next request reading the slot fails validation. Shared with helpers.
        current = list(updates)
        updates[:] = [value if index % 3 == 0 else gr.skip() for index, value in enumerate(current)]
        return (*current, status)

    targets = []
    for speaker in range(max(0, min(shown, DRAMA3_VOICE_SLOTS))):
        preset, has_clip, description, preview, language = slots[5 * speaker:5 * speaker + 5]
        if preset and preset != UPLOAD_VOICE:
            continue
        if has_clip and skip_existing:
            updates[3 * speaker + 2] = ("Skipped: this speaker already has a clip. Uncheck "
                                        "Skip speakers with a voice to regenerate it.")
        elif not str(description or "").strip() or not str(preview or "").strip():
            updates[3 * speaker + 2] = "Skipped: add a voice description and preview speech first."
        else:
            targets.append((speaker, description, preview, language))
    if not targets:
        yield frame("No speaker voices to generate. Shown speakers without a clip need a voice description "
                    "and preview speech.")
        return
    failed = []
    if settings[0] == ENGINE_DRAMA3:
        yield from _generate_drama3_designs(targets, settings, parallel, updates, frame, failed)
    else:
        for index, (speaker, description, preview, language) in enumerate(targets, 1):
            progress = f"Generating speaker voices: {index}/{len(targets)} (speaker {speaker})"
            updates[3 * speaker + 2] = "Waiting for voice generation…"
            yield frame(progress)
            status = ""
            for audio, transcript, status in generate_drama3_voice(description, preview, language, *settings):
                if isinstance(audio, str):
                    updates[3 * speaker:3 * speaker + 2] = [audio, transcript]
                updates[3 * speaker + 2] = status
                yield frame(progress)
            if status.startswith("Voice generation failed"):
                failed.append(speaker)
    done = len(targets) - len(failed)
    summary = f"Generated {done} of {len(targets)} speaker voices."
    if failed:
        summary += " Failed: " + ", ".join(f"speaker {speaker}" for speaker in failed) + "; see each speaker's panel."
    yield frame(summary)


def _generate_drama3_designs(targets, settings, parallel, updates, frame, failed):
    from h3_app import drama3_voice_design
    from h3_models import resolve_hf_token

    _engine, *request_settings = settings
    request = _drama3_design_request(*request_settings, parallel=parallel)
    for speaker, *_brief in targets:
        updates[3 * speaker + 2] = "Waiting for Drama 3 voice design on SGLang-Omni…"
    yield frame(f"Preparing Drama 3 voice design for {len(targets)} speaker(s) on SGLang-Omni.")
    pending = {speaker for speaker, *_brief in targets}
    try:
        for result in drama3_voice_design.generate_drama3_references(
                {speaker: (description, preview) for speaker, description, preview, _language in targets},
                request, _runtime_config(), _generation_services(), free_gpu=free_comfy_vram,
                hf_token=resolve_hf_token()):
            if isinstance(result, drama3_voice_design.DesignResult):
                speaker = result.key
                pending.discard(speaker)
                if result.audio:
                    updates[3 * speaker:3 * speaker + 3] = [
                        result.audio, gr.update(value=result.transcript, interactive=True),
                        "Voice generated. Preview speech copied as its exact transcript; ready to save."]
                else:
                    failed.append(speaker)
                    updates[3 * speaker + 2] = f"Voice generation failed: {result.error}"
                yield frame(f"Designed {len(targets) - len(pending)}/{len(targets)} speaker voices on SGLang-Omni.")
            else:
                yield frame(result.status)
    except JobCancelled:
        raise
    except Exception as exc:
        for speaker in sorted(pending):
            failed.append(speaker)
            updates[3 * speaker + 2] = f"Voice generation failed: {exc}"
        yield frame(f"Voice generation failed: {exc}")


def generate_drama3_voice_takes(key, count, description, transcript, engine="Qwen3-TTS VoiceDesign",
                                temperature=1.0, top_p=0.9, top_k=0, max_frames=8192, parallel=8):
    """Design up to four new voices for a saved speaker without replacing its clip.

    Outputs: one clip per try, the voice choice, the brief the tries were made from
    (saved with the chosen clip) and a status. Every frame resends every clip; see
    ``generate_drama3_voices`` for why a skip after a clip breaks the next request.
    """
    from h3_app import drama3_voice_design
    from h3_app.voice_design import generate_reference
    from h3_ui.drama3_speaker_library import KEEP_SAVED, MAX_TAKES, take_label

    count = max(1, min(int(count or 1), MAX_TAKES))
    clips = [None] * MAX_TAKES
    errors = []

    def frame(status, brief=gr.skip(), choice=gr.skip()):
        return (*[gr.update(value=clip, visible=index < count) for index, clip in enumerate(clips)],
                choice, brief, status)

    if not key:
        yield frame("Select a saved speaker first.")
        return
    if not str(description or "").strip() or not str(transcript or "").strip():
        yield frame("Add a voice description and an exact clip transcript first.")
        return
    brief = {"key": key, "description": str(description).strip(), "transcript": " ".join(str(transcript).split())}
    yield frame(f"Generating {count} new voice(s)…", brief=None,
                choice=gr.update(choices=[KEEP_SAVED], value=KEEP_SAVED))
    try:
        if engine == drama3_voice_design.ENGINE_DRAMA3:
            from h3_models import resolve_hf_token

            request = _drama3_design_request(temperature, top_p, top_k, max_frames, parallel)
            for result in drama3_voice_design.generate_drama3_references(
                    {index: (brief["description"], brief["transcript"]) for index in range(count)},
                    request, _runtime_config(), _generation_services(), free_gpu=free_comfy_vram,
                    hf_token=resolve_hf_token()):
                if isinstance(result, drama3_voice_design.DesignResult):
                    if result.audio:
                        clips[result.key] = result.audio
                        brief["transcript"] = result.transcript
                    else:
                        errors.append(result.error)
                    yield frame(f"Designed {sum(clip is not None for clip in clips) + len(errors)}/{count} "
                                "voices on SGLang-Omni.")
                else:
                    yield frame(result.status)
        elif engine == drama3_voice_design.ENGINE_QWEN:
            for index in range(count):
                yield frame(f"Generating voice {index + 1}/{count} with Qwen VoiceDesign…")
                try:
                    audio, brief["transcript"] = generate_reference(
                        brief["description"], brief["transcript"], "Auto", _runtime_config(),
                        free_gpu=free_comfy_vram)
                    clips[index] = str(audio)
                except JobCancelled:
                    raise
                except Exception as exc:
                    errors.append(str(exc))
        else:
            raise H3Error("Choose Drama 3 or Qwen3-TTS VoiceDesign to generate a voice.")
    except JobCancelled:
        raise
    except Exception as exc:
        errors.append(str(exc))
    made = [take_label(index) for index, clip in enumerate(clips) if clip]
    status = f"Generated {len(made)} of {count} new voices. Listen, then pick one to keep or try again."
    if errors:
        status += " Failed: " + "; ".join(dict.fromkeys(errors))
    yield frame(status, brief=brief if made else None,
                choice=gr.update(choices=[KEEP_SAVED, *made], value=KEEP_SAVED))


def _publish_drama3_voices_signature():
    import inspect
    from h3_ui.drama3_controls import DRAMA3_VOICE_SLOTS

    names = [*DRAMA3_VOICE_DESIGN_SETTINGS, "parallel", "shown_slots", "skip_existing",
             *[f"speaker_{speaker}_{field}" for speaker in range(DRAMA3_VOICE_SLOTS)
               for field in ("voice", "has_clip", "description", "preview_text", "language")]]
    generate_drama3_voices.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names])


_publish_drama3_voices_signature()


def drama3_speaker_groups():
    from h3_app import drama3_speakers
    return drama3_speakers.list_groups(_runtime_config())


def drama3_speaker_group_size(group):
    """Number of speakers saved in ``group`` itself (Global voices are not counted twice)."""
    from h3_app import drama3_speakers
    return len(drama3_speakers.list_speakers(_runtime_config(), group or "Global"))


def create_drama3_speaker_group(name):
    from h3_app import drama3_speakers
    try:
        name = drama3_speakers.create_group(name, _runtime_config())
        return gr.update(choices=drama3_speaker_groups(), value=name), f'Using speaker group "{name}".'
    except H3Error as exc:
        return gr.skip(), str(exc)


def match_drama3_speaker(group, name, preset, audio, match=None):
    from h3_app import drama3_speakers
    from h3_ui.drama3_controls import UPLOAD_VOICE
    if audio and (not preset or preset == UPLOAD_VOICE):
        return gr.skip()
    saved = (match or drama3_speakers.speaker_matcher(_runtime_config(), group))(name)
    return gr.update(value=saved.key) if saved else gr.skip()


def match_drama3_speakers(group, *slots):
    from h3_app import drama3_speakers
    match = drama3_speakers.speaker_matcher(_runtime_config(), group)
    return [match_drama3_speaker(group, *slots[i:i + 3], match=match) for i in range(0, len(slots), 3)]


def save_all_drama3_speakers(group, shown, *slots):
    from h3_app import drama3_speakers
    from h3_ui.drama3_controls import UPLOAD_VOICE, drama3_voice_choices
    group = group or "Global"
    selected, errors, saved_count = {}, [], 0
    for i in range(min(int(shown or 1), len(slots) // 5)):
        preset, name, audio, transcript, description = slots[i * 5:i * 5 + 5]
        try:
            if preset and preset != UPLOAD_VOICE:
                source = drama3_speakers.load_speaker(preset, _runtime_config())
                name, audio, transcript, description = source.name, source.audio, source.transcript, source.description
            if not audio:
                continue
            saved = drama3_speakers.save_speaker(name, audio, transcript, _runtime_config(), description, group)
            selected[i] = saved.key
            saved_count += 1
        except H3Error as exc:
            errors.append(f"Speaker {i}: {exc}")
    choices = drama3_voice_choices(drama3_speaker_names())
    return [f'Saved {saved_count} speakers to "{group}". ' + " ".join(errors),
            *[gr.update(choices=choices, **({"value": selected[i]} if i in selected else {}))
              for i in range(len(slots) // 5)]]


def drama3_speaker_names():
    from h3_app import drama3_speakers
    return drama3_speakers.speaker_names(_runtime_config())


def _drama3_preset_updates(count, selected_slot=None, selected=None):
    from h3_ui.drama3_controls import drama3_voice_choices
    choices = drama3_voice_choices(drama3_speaker_names())
    return [
        gr.update(choices=choices, value=selected) if slot == selected_slot else gr.update(choices=choices)
        for slot in range(count)
    ]


def select_drama3_speaker(name):
    """Show the upload controls, or a saved speaker's transcript in their place."""
    from h3_app import drama3_speakers
    from h3_ui.drama3_controls import UPLOAD_VOICE
    if not name or name == UPLOAD_VOICE:
        return (gr.update(visible=True), gr.update(value="", interactive=True),
                gr.update(visible=True), gr.update(visible=False), "", gr.skip(),
                gr.update(interactive=True), gr.skip(), None)
    try:
        saved = drama3_speakers.load_speaker(name, _runtime_config())
    except H3Error as exc:
        return (gr.update(visible=True), gr.update(interactive=True),
                gr.update(visible=True), gr.update(visible=False), str(exc), gr.skip(),
                gr.update(interactive=True), gr.skip(), gr.skip())
    return (gr.update(visible=False), gr.update(value=saved.transcript, interactive=False),
            gr.update(visible=False), gr.update(visible=True), f'Using saved speaker "{saved.name}".',
            saved.name, gr.update(value=saved.description, interactive=False), saved.transcript, None)


def save_drama3_speaker(slot, name, audio, transcript, count, description="", group="Global"):
    from h3_app import drama3_speakers
    try:
        saved = drama3_speakers.save_speaker(name, audio, transcript, _runtime_config(), description, group or "Global")
    except H3Error as exc:
        return [str(exc), *[gr.skip()] * count]
    return [f'Saved speaker "{saved.name}".', *_drama3_preset_updates(count, slot, saved.key)]


def delete_drama3_speaker(slot, name, count):
    from h3_app import drama3_speakers
    from h3_ui.drama3_controls import UPLOAD_VOICE
    try:
        drama3_speakers.delete_speaker(name, _runtime_config())
    except H3Error as exc:
        return [str(exc), *[gr.skip()] * count]
    return [f'Deleted saved speaker "{name}".', *_drama3_preset_updates(count, slot, UPLOAD_VOICE)]


def drama3_sound_names():
    from h3_app import drama3_sounds
    return drama3_sounds.sound_names(_runtime_config())


def _drama3_sound_preset_updates(count, selected_slot=None, selected=None):
    from h3_ui.drama3_controls import drama3_sound_choices
    choices = drama3_sound_choices(drama3_sound_names())
    return [
        gr.update(choices=choices, value=selected) if slot == selected_slot else gr.update(choices=choices)
        for slot in range(count)
    ]


def select_drama3_sound(name):
    """Show the sound upload controls, or a saved sound preset's metadata in their place."""
    from h3_app import drama3_sounds
    from h3_ui.drama3_controls import UPLOAD_SOUND
    if not name or name == UPLOAD_SOUND:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            gr.skip(),
            gr.update(interactive=True),
            gr.update(interactive=True),
            None,
        )
    try:
        saved = drama3_sounds.load_sound(name, _runtime_config())
    except H3Error as exc:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            str(exc),
            gr.skip(),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.skip(),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        f'Using saved sound "{saved.name}".',
        saved.name,
        gr.update(value=saved.description, interactive=False),
        gr.update(value=saved.duration, interactive=False),
        str(saved.audio),
    )


def save_drama3_sound(slot, name, audio, description, duration, count):
    from h3_app import drama3_sounds
    try:
        saved = drama3_sounds.save_sound(name, audio, description, duration, _runtime_config())
    except H3Error as exc:
        return [str(exc), *[gr.skip()] * count]
    return [f'Saved sound "{saved.name}".', *_drama3_sound_preset_updates(count, slot, saved.name)]


def delete_drama3_sound(slot, name, count):
    from h3_app import drama3_sounds
    from h3_ui.drama3_controls import UPLOAD_SOUND
    try:
        drama3_sounds.delete_sound(name, _runtime_config())
    except H3Error as exc:
        return [str(exc), *[gr.skip()] * count]
    return [f'Deleted saved sound "{name}".', *_drama3_sound_preset_updates(count, slot, UPLOAD_SOUND)]


def generate_drama3_sound(description, duration=5):
    """Generate a single sound effect or music clip with Stable Audio 3 Medium."""
    from h3_app import stable_audio
    from h3_models import resolve_hf_token

    try:
        yield gr.skip(), "Preparing Stable Audio 3 Medium; runtime and weights download on first use."
        audio = stable_audio.generate_sound_reference(
            description, duration, _runtime_config(),
            free_gpu=free_comfy_vram, hf_token=resolve_hf_token(),
        )
    except Exception as exc:
        yield gr.skip(), f"Sound generation failed: {exc}"
        return
    yield str(audio), "Sound generated with Stable Audio 3 Medium; ready to save."


def generate_drama3_sounds(*values):
    """Batch-generate shown upload sound slots that have a description and no clip yet."""
    from h3_app import stable_audio
    from h3_models import resolve_hf_token
    from h3_ui.drama3_controls import DRAMA3_SOUND_SLOTS, UPLOAD_SOUND

    shown = int(values[0] or 0)
    skip_existing = values[1] is not False
    slots = values[2:]
    updates = [gr.skip()] * (2 * DRAMA3_SOUND_SLOTS)

    def frame(status):
        current = list(updates)
        updates[:] = [gr.skip()] * len(updates)
        return (*current, status)

    targets = []
    for index in range(max(0, min(shown, DRAMA3_SOUND_SLOTS))):
        preset, has_clip, save_name, description, duration = slots[5 * index:5 * index + 5]
        if preset and preset != UPLOAD_SOUND:
            continue
        if has_clip and skip_existing:
            updates[2 * index + 1] = (
                "Skipped: this sound already has a clip. Uncheck "
                "Skip sounds with a clip to regenerate it."
            )
        elif not str(description or "").strip():
            updates[2 * index + 1] = "Skipped: add a sound description first."
        else:
            targets.append((index, str(save_name or f"sound_{index}"), str(description), duration))

    if not targets:
        yield frame("No sounds to generate. Shown sound slots without a clip need a sound description.")
        return

    for index, *_rest in targets:
        updates[2 * index + 1] = "Waiting for Stable Audio 3 Medium…"
    yield frame(f"Generating {len(targets)} sound(s) with Stable Audio 3 Medium…")

    try:
        items = [
            (description, duration)
            for _index, _name, description, duration in targets
        ]
        results = list(stable_audio.generate_sound_references(
            items, _runtime_config(), free_gpu=free_comfy_vram, hf_token=resolve_hf_token(),
        ))
    except JobCancelled:
        raise
    except Exception as exc:
        for index, *_rest in targets:
            updates[2 * index + 1] = f"Sound generation failed: {exc}"
        yield frame(f"Sound generation failed: {exc}")
        return

    failed = []
    for position, (index, _name, _description, _duration) in enumerate(targets):
        audio = results[position] if position < len(results) else None
        if audio and Path(str(audio)).is_file():
            updates[2 * index:2 * index + 2] = [
                str(audio), "Sound generated with Stable Audio 3 Medium; ready to save."]
        else:
            failed.append(index)
            updates[2 * index + 1] = "Sound generation failed: Stable Audio returned no clip; generate again."
    summary = f"Generated {len(targets) - len(failed)} of {len(targets)} sound clips."
    if failed:
        summary += " Failed: " + ", ".join(f"sound {index}" for index in failed) + "; see each sound's panel."
    yield frame(summary)


def _publish_drama3_sounds_signature():
    import inspect
    from h3_ui.drama3_controls import DRAMA3_SOUND_SLOTS

    names = [
        "shown_sound_slots",
        "skip_existing_sounds",
        *[
            f"sound_{index}_{field}"
            for index in range(DRAMA3_SOUND_SLOTS)
            for field in ("preset", "has_clip", "name", "description", "duration")
        ],
    ]
    generate_drama3_sounds.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
    )


_publish_drama3_sounds_signature()
