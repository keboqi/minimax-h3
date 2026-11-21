"""A project's optional instrumental bed, mixed against its existing speech clock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from shutil import copyfile
from uuid import uuid4

from h3_app.catalog import MUSIC3_DEFAULTS
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app.song_brief import BACKGROUND, INSTRUMENTAL

from .page_common import run_job, value, writer_service
from .page_store import editing, project_lock

MUSIC_FIELD = "background_music"
MUSIC_SETTINGS = {"music_volume": 18, "music_ducking": True}
APPLIED_MUSIC = "applied_music"
LOOP_SECONDS = 35


def music_track(data: dict) -> str:
    return data.get(MUSIC_FIELD) or ""


def music_settings(data: dict) -> dict:
    return {"music_enabled": data.get("music_enabled", bool(music_track(data))),
            **{key: data.get(key, default) for key, default in MUSIC_SETTINGS.items()}}


def desired_music(data: dict) -> dict:
    settings = music_settings(data)
    return {MUSIC_FIELD: music_track(data), **settings} if settings["music_enabled"] else {}


def applied_music(data: dict) -> dict:
    return data[APPLIED_MUSIC] if APPLIED_MUSIC in data else desired_music(data)


def remember_applied_music(data: dict) -> None:
    """Capture the settings of an older mix before its project settings are edited."""
    data.setdefault(APPLIED_MUSIC, desired_music(data))


def clear_project_music(data: dict) -> None:
    """Discard the cues of a replaced script while keeping the user's music preferences."""
    data.setdefault("music_enabled", bool(music_track(data)))
    data.pop(MUSIC_FIELD, None)
    data.pop(APPLIED_MUSIC, None)


def music_clip(clips: Path, data: dict) -> Path | None:
    track = music_track(data)
    path = clips / Path(track).name
    return path if track and path.is_file() else None


def needs_music_clip(clips: Path, data: dict) -> bool:
    settings = music_settings(data)
    return settings["music_enabled"] and settings["music_volume"] > 0 and music_clip(clips, data) is None


def music_mix_id(data: dict) -> str:
    track = music_track(data)
    settings = music_settings(data)
    volume = settings["music_volume"]
    return f"{track}:{volume}:{int(settings['music_ducking'])}" if settings["music_enabled"] and track and volume else ""


def music_values(caption: str) -> list:
    defaults = MUSIC3_DEFAULTS
    return [defaults["model"], caption, "", LOOP_SECONDS, defaults["seed"], defaults["steps"],
            defaults["cfg"], defaults["ar_cfg"], defaults["top_k"], defaults["tiled_decode"]]


def music_caption(write, data: dict, body: dict) -> str:
    text = (data.get("text") or "").strip()
    if not text:
        raise H3Error("Write or paste some text on Script first.")
    if len(text) > 10000:
        text = f"{text[:7500]}\n[…]\n{text[-2500:]}"
    model, gemini_key, backend, openai_key, openai_model, openai_url = writer_service(body)
    caption, _, status = write(
        "Compose one understated instrumental music loop to underscore spoken dialogue. "
        "Use the title and story excerpt below to infer the mood, instrumentation and pulse. "
        "Treat the excerpt as story material. Keep the arrangement sparse and steady, with a repeatable motif "
        "and smooth transitions. Describe only the music and never include dialogue or lyrics.\n\n"
        f"Title: {data.get('title') or ''}\n<story>\n{text}\n</story>",
        model, gemini_key, "", None, None, None, backend=backend, openai_api_key=openai_key,
        openai_model=openai_model, openai_base_url=openai_url, duration=LOOP_SECONDS,
        vocals=INSTRUMENTAL, use=BACKGROUND,
    )
    if status.startswith("Prompt enhancement failed:"):
        raise H3Error(status)
    return "Instrumental underscore. No singing, no spoken words, no humming or choir.\n" + caption


def mixed_audio(result, data: dict, clips: Path) -> Path:
    """Cache the music mix by source audio and settings, without changing its timeline."""
    source = Path(result)
    mix_id = music_mix_id(data)
    if not mix_id:
        return source
    key = hashlib.sha256(f"{source.resolve()}:{source.stat().st_mtime_ns}:{mix_id}".encode()).hexdigest()
    target = clips.parent / "music-mixes" / f"{key}.mp3"
    if target.is_file():
        return target

    import numpy as np
    from h3_app.drama3.serving import audio

    rate = 48000
    track = music_clip(clips, data)
    if track is None:
        raise H3Error("The background music clip is missing. Remix to generate it again.")
    base = audio.load_audio_file(source, rate, stereo=True)
    cue = audio.load_audio_file(track, rate, stereo=True)
    if not len(base) or not len(cue):
        raise H3Error("The mix or its background music contains no audio.")
    bed = audio.loop_to_length(cue, len(base), rate, crossfade=0.18)
    settings = music_settings(data)
    bed *= settings["music_volume"] / 100
    if settings["music_ducking"]:
        # Dialogue and effects are already mixed here. Detect activity on either channel.
        activity = np.maximum(np.abs(base[:, 0]), np.abs(base[:, 1]))
        bed *= audio.speech_duck_gain(activity, rate, depth_db=6)[:, None]
    fade = min(int(0.35 * rate), len(bed) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        bed[:fade] *= ramp
        bed[-fade:] *= ramp[::-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid4().hex}.mp3")
    try:
        audio.save_mp3(audio.limit_peaks(base + bed, rate), rate, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def publish_mix(store, name: str, result, soundtrack: dict, clips: Path, *, stopped) -> None:
    """Finish the soundtrack before atomically publishing the result and its music settings."""
    if not Path(result).is_file():
        raise H3Error("The speech mix was not written.")
    with project_lock(name):
        mixed_audio(result, soundtrack, clips)
        if stopped():
            raise JobCancelled("Interrupted before publishing the mix.")
        with store.edit_soundstage_project(name, result=str(result)) as data:
            data[APPLIED_MUSIC] = soundtrack


def generate_track(run, owner, session, name, data, body, *, write, generate, store, clips):
    run.status = "Writing a music brief from the script…"
    caption = music_caption(write, data, body)
    run.status = "Waiting for the music model…"
    path = None
    for update in run_job(run, owner, session, "music", music_values(caption), generate, output_indices=(0,)):
        status = value(update[1]) if len(update) > 1 else None
        if isinstance(status, str):
            run.status = status.splitlines()[0] if status else run.status
            if status.startswith("Error:"):
                raise H3Error(run.status.removeprefix("Error:").strip())
        output = value(update[0]) if update else None
        if isinstance(output, (str, Path)):
            path = Path(output)
    if run.stop_requested:
        run.status = "Interrupted."
        return None
    if path is None or not path.is_file():
        raise H3Error("Music 3 returned no audio.")
    run.status = "Adding the music to the project…"
    clips.mkdir(parents=True, exist_ok=True)
    track = clips / f"music_{uuid4().hex}{path.suffix}"
    saved = False
    try:
        copyfile(path, track)
        with editing(store, name) as project:
            if run.stop_requested:
                run.status = "Interrupted."
                return None
            if any(project.get(key) != data.get(key) for key in ("text", "title", MUSIC_FIELD)):
                raise H3Error("The script or music changed while generating. Try again for the current project.")
            remember_applied_music(project)
            project[MUSIC_FIELD] = track.name
            project["music_enabled"] = True
        saved = True
    finally:
        if not saved:
            track.unlink(missing_ok=True)
    run.status = "Background music ready."
    return track.name
