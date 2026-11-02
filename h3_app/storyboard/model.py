"""Storyboard documents and the rules that read them; no I/O.

A storyboard is plain JSON: a frozen H3 settings profile, a cast, ordered scenes
and every take. A scene's state is computed from what its take in use was made
from, never stored, so it stays right after edits, reorders and restarts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from typing import Any

from h3_app.contracts import GENERATION_FIELDS

SCENE_SECONDS = (5, 10, 15)
OVERLAP_FRAMES = (22, 39, 56)
DEFAULT_OVERLAP = 39
MAX_TAKES_PER_RUN = 4
MAX_CAST = 9
FPS = 24
FLOW, CUT = "flow", "cut"
TEXT, FRAMES, REFERENCES = "Text to video", "First / last frame", "Reference media"
MODES = (TEXT, FRAMES, REFERENCES)
MAX_REFERENCE_IMAGES, MAX_REFERENCE_VIDEOS, MAX_REFERENCE_AUDIOS = 9, 3, 3
# A scene's own conditioning media, like the MiniMax H3 composer's.
MEDIA_FIELDS = ("first_image", "last_image", "ref_images", "ref_videos", "ref_audios")
DONE, OUTDATED, DRAFT = "done", "outdated", "draft"
STATE_NAMES = {DONE: "Done", OUTDATED: "Out of date", DRAFT: "Not generated"}

# Each scene supplies these; everything else in an H3 request is the profile.
SCENE_FIELDS = frozenset({
    "mode", "prompt", "first_image", "last_image", "duration", "seed",
    "result_format", "image_frames", "fl2va_audio_1", "fl2va_audio_2", "fl2va_audio_3",
    *(f"ref_image_{index}" for index in range(1, 10)),
    *(f"ref_video_{index}" for index in range(1, 4)),
    *(f"ref_audio_{index}" for index in range(1, 4)),
})
PROFILE_FIELDS = tuple(name for name in GENERATION_FIELDS if name not in SCENE_FIELDS)
# A take's playable video must be the raw H3 output its latent decodes to.
PROFILE_OVERRIDES = {"postprocess": "None"}
# Finishing and reference-size settings that Storyboard takes from the request defaults.
DEFAULTED_FIELDS = frozenset({
    "postprocess", "upscale_force_offload", "upscale_resolution", "seedvr2_model", "ltx25_model",
    "ref_image_size",
})


def request_defaults() -> dict[str, Any]:
    """Every H3 generate value from the app's defaults, without importing the GPU application.

    The Workshop (CPU) creates storyboards from these; the GPU endpoint layers the
    MiniMax H3 tab's current settings on top.
    """
    from h3_app.catalog import DEFAULT_UPSCALE_RESOLUTION, UI_DEFAULTS
    from h3_models import DEFAULT_LTX25_MODEL

    values = {name: None for name in GENERATION_FIELDS}
    values.update(UI_DEFAULTS)
    values.update(sol_step_off=0.0, sol_sink_tokens=0, ltx25_model=DEFAULT_LTX25_MODEL,
                  upscale_resolution=DEFAULT_UPSCALE_RESOLUTION, prompt="")
    return values


def new_id() -> str:
    return uuid.uuid4().hex


def make_profile(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """The H3 settings a storyboard keeps; finishing settings come from ``defaults``."""
    merged = {**{name: defaults[name] for name in DEFAULTED_FIELDS if name in defaults}, **values}
    missing = [name for name in PROFILE_FIELDS if name not in merged]
    if missing:
        raise ValueError("The MiniMax H3 settings are incomplete: " + ", ".join(missing[:5]))
    profile = {name: merged[name] for name in PROFILE_FIELDS}
    profile.update(PROFILE_OVERRIDES)
    return profile


def profile_digest(profile: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True, default=str).encode()).hexdigest()[:16]


def profile_summary(profile: dict[str, Any]) -> str:
    parts = [
        str(profile.get("model_profile", "")),
        f"{profile.get('width')}×{profile.get('height')}",
        str(profile.get("turbo_variant", "")) if profile.get("generation_mode") == "Turbo" else "Normal",
        "latent upscale on" if profile.get("latent_upscale") else "",
    ]
    return " · ".join(part for part in parts if part and part != "None")


def new_scene(title="New scene", prompt="", seconds=5, link=FLOW, cast=(), mode=TEXT) -> dict[str, Any]:
    return {
        "id": new_id(), "title": title, "prompt": prompt, "seconds": int(seconds),
        "link": link, "cast": list(cast), "overlap": DEFAULT_OVERLAP, "seed": -1,
        "takes_per_run": 1, "takes": [], "in_use": None, "mode": mode,
        "first_image": None, "last_image": None, "ref_images": [], "ref_videos": [], "ref_audios": [],
    }


def scene_media(scene: dict[str, Any]) -> dict[str, Any]:
    """The scene's mode and media; scenes saved before per-scene media read as text to video."""
    return {
        "mode": scene.get("mode", TEXT),
        "first_image": scene.get("first_image"),
        "last_image": scene.get("last_image"),
        "ref_images": list(scene.get("ref_images", [])),
        "ref_videos": list(scene.get("ref_videos", [])),
        "ref_audios": list(scene.get("ref_audios", [])),
    }


def title_from(text: str, words: int = 6) -> str:
    """A short scene title from the start of its description."""
    parts = text.split()
    title = " ".join(parts[:words]).rstrip(".,;:")
    return title + ("…" if len(parts) > words else "")


def new_storyboard(name: str, profile: dict[str, Any], scenes=(), cast=()) -> dict[str, Any]:
    return {
        "name": (name.strip() or "Untitled storyboard")[:120],
        "profile": profile,
        "cast": list(cast),
        "scenes": list(scenes) or [new_scene("Opening scene")],
        "takes": {},
        "versions": [],
        "queue": [],
    }


def validate_scene(scene: dict[str, Any]) -> None:
    if int(scene["seconds"]) not in SCENE_SECONDS:
        raise ValueError("A scene is 5, 10 or 15 seconds long.")
    if scene["link"] not in (FLOW, CUT):
        raise ValueError("A scene either flows on or cuts to a new shot.")
    if int(scene["overlap"]) not in OVERLAP_FRAMES:
        raise ValueError("Overlap is 22, 39 or 56 frames.")
    if not 1 <= int(scene["takes_per_run"]) <= MAX_TAKES_PER_RUN:
        raise ValueError(f"Takes per run is 1 to {MAX_TAKES_PER_RUN}.")
    media = scene_media(scene)
    if media["mode"] not in MODES:
        raise ValueError(f"Unknown scene mode {media['mode']!r}.")
    if len(media["ref_images"]) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"A scene has at most {MAX_REFERENCE_IMAGES} reference pictures.")
    if len(media["ref_videos"]) > MAX_REFERENCE_VIDEOS or len(media["ref_audios"]) > MAX_REFERENCE_AUDIOS:
        raise ValueError("A scene has at most 3 reference videos and 3 reference audio clips.")


def flows_from_previous(board: dict[str, Any], index: int) -> bool:
    return index > 0 and board["scenes"][index]["link"] == FLOW


def scene_inputs(board: dict[str, Any], index: int) -> dict[str, Any]:
    """What a take of this scene must have been made from to be current."""
    scene = board["scenes"][index]
    flow = flows_from_previous(board, index)
    return {
        "prompt": scene["prompt"].strip(),
        "seconds": int(scene["seconds"]),
        "cast": sorted(scene["cast"]),
        "profile": profile_digest(board["profile"]),
        "flow": flow,
        "overlap": int(scene["overlap"]) if flow else None,
        "source_take": board["scenes"][index - 1]["in_use"] if flow else None,
        **scene_media(scene),
    }


def scene_states(board: dict[str, Any]) -> list[str]:
    """Done, outdated or draft per scene. A flow scene follows its predecessor."""
    states = []
    for index, scene in enumerate(board["scenes"]):
        take = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
        if take is None:
            states.append(DRAFT)
        elif take["inputs"] != scene_inputs(board, index) or (
            flows_from_previous(board, index) and states[index - 1] != DONE
        ):
            states.append(OUTDATED)
        else:
            states.append(DONE)
    return states


def affected_by(board: dict[str, Any], index: int) -> list[int]:
    """Later scenes that continue, through unbroken flows, from scene ``index``."""
    affected = []
    for later in range(index + 1, len(board["scenes"])):
        if board["scenes"][later]["link"] != FLOW:
            break
        affected.append(later)
    return affected


def overlap_seconds(frames: int) -> float:
    return int(frames) / FPS


def new_seconds(board: dict[str, Any], index: int) -> float:
    """Footage this scene adds to the full video; a flow scene repeats its overlap."""
    scene = board["scenes"][index]
    seconds = float(scene["seconds"])
    if flows_from_previous(board, index):
        seconds -= overlap_seconds(scene["overlap"])
    return seconds


def total_seconds(board: dict[str, Any]) -> float:
    return sum(new_seconds(board, index) for index in range(len(board["scenes"])))


def pending_scenes(board: dict[str, Any]) -> list[int]:
    states = scene_states(board)
    return [index for index, state in enumerate(states) if state != DONE]


def next_runnable(board: dict[str, Any], queue: list[str]) -> int | None:
    """First queued scene whose source is ready: cuts can run ahead of a waiting flow."""
    states = scene_states(board)
    positions = {scene["id"]: index for index, scene in enumerate(board["scenes"])}
    for scene_id in queue:
        index = positions.get(scene_id)
        if index is None:
            continue
        if not flows_from_previous(board, index) or states[index - 1] == DONE:
            return index
    return None


def cast_prompt(board: dict[str, Any], scene: dict[str, Any]) -> tuple[str, list[str]]:
    """The scene prompt with cast picture tags, and the pictures in tag order."""
    pictures = {member["name"]: member["image"] for member in board["cast"]}
    names = [name for name in scene["cast"] if name in pictures][:MAX_CAST]
    if not names:
        return scene["prompt"].strip(), []
    tags = " ".join(f"{name} is <Picture {number}>." for number, name in enumerate(names, 1))
    return f"{tags} {scene['prompt'].strip()}", [pictures[name] for name in names]


def runs_as(board: dict[str, Any], index: int) -> str:
    """The mode as the scene editor names it."""
    mode = scene_media(board["scenes"][index])["mode"]
    if mode == FRAMES and flows_from_previous(board, index):
        return "Last frame"
    return mode


def picture_tags(board: dict[str, Any], scene: dict[str, Any]) -> list[str]:
    """Which ``<Picture n>`` each cast member and scene image becomes, in order."""
    names = [name for name in scene["cast"] if name in {member["name"] for member in board["cast"]}][:MAX_CAST]
    tags = [f"{name} = <Picture {number}>" for number, name in enumerate(names, 1)]
    tags += [f"image {number} = <Picture {len(names) + number}>"
             for number in range(1, len(scene_media(scene)["ref_images"]) + 1)]
    return tags


def cast_count(board: dict[str, Any], scene: dict[str, Any]) -> int:
    names = {member["name"] for member in board["cast"]}
    return len([name for name in scene["cast"] if name in names][:MAX_CAST])


def scene_issues(board: dict[str, Any], index: int) -> list[str]:
    """What the scene still needs before it can be generated, in plain words.

    A continuing scene opens on the previous take, so its first frame is never
    used; its frames mode needs a last frame instead.
    """
    scene = board["scenes"][index]
    media = scene_media(scene)
    flow = flows_from_previous(board, index)
    issues = []
    if not scene["prompt"].strip():
        issues.append("Write a prompt")
    if media["mode"] == FRAMES:
        if flow and not media["last_image"]:
            issues.append("Add the last-frame picture")
        if not flow and not media["first_image"] and not media["last_image"]:
            issues.append("Add a first or last frame")
    if media["mode"] == REFERENCES:
        pictures = cast_count(board, scene) + len(media["ref_images"])
        if pictures > MAX_REFERENCE_IMAGES:
            issues.append(f"Use at most {MAX_REFERENCE_IMAGES} pictures (cast plus images); there are {pictures}")
        if not pictures and not media["ref_videos"] and not media["ref_audios"]:
            issues.append("Pick at least one reference")
    if flow and scene_states(board)[index - 1] != DONE:
        issues.append(f"Generate scene {index} first: this scene continues from it")
    return issues


def check_scene(board: dict[str, Any], index: int) -> None:
    """Refuse a take whose scene isn't ready, before it reaches the GPU."""
    issues = scene_issues(board, index)
    if issues:
        raise ValueError(f"Scene {index + 1}: " + "; ".join(issues) + ".")


def scene_request_values(board: dict[str, Any], index: int, defaults: dict[str, Any], seed: int) -> dict[str, Any]:
    """Every H3 generate value for one take of this scene."""
    scene = board["scenes"][index]
    media = scene_media(scene)
    mode = media["mode"]
    flow = flows_from_previous(board, index)
    prompt, pictures = cast_prompt(board, scene) if mode == REFERENCES else (scene["prompt"].strip(), [])
    values = dict(defaults)
    values.update(board["profile"])
    values.update(
        mode=mode,
        prompt=prompt,
        first_image=media["first_image"] if mode == FRAMES and not flow else None,
        last_image=media["last_image"] if mode == FRAMES else None,
        duration=float(scene["seconds"]),
        seed=int(seed),
        result_format="Video",
        **{f"ref_image_{number}": None for number in range(1, 10)},
        **{f"ref_video_{number}": None for number in range(1, 4)},
        **{f"ref_audio_{number}": None for number in range(1, 4)},
        **{f"fl2va_audio_{number}": None for number in range(1, 4)},
    )
    if mode == REFERENCES:
        for number, picture in enumerate([*pictures, *media["ref_images"]][:MAX_REFERENCE_IMAGES], 1):
            values[f"ref_image_{number}"] = picture
        for number, video in enumerate(media["ref_videos"], 1):
            values[f"ref_video_{number}"] = video
        for number, audio in enumerate(media["ref_audios"], 1):
            values[f"ref_audio_{number}"] = audio
    return values


def media_paths(scene: dict[str, Any]) -> list[str]:
    media = scene_media(scene)
    return [path for path in (media["first_image"], media["last_image"], *media["ref_images"],
                              *media["ref_videos"], *media["ref_audios"]) if path]


def snapshot(board: dict[str, Any]) -> dict[str, Any]:
    """What a saved version keeps: scene order, scene fields and each take in use."""
    return {"scenes": copy.deepcopy(board["scenes"]), "cast": copy.deepcopy(board["cast"])}
