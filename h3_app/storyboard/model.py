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
DONE, OUTDATED, DRAFT = "done", "outdated", "draft"

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


def new_scene(title="New scene", prompt="", seconds=5, link=FLOW, cast=()) -> dict[str, Any]:
    return {
        "id": new_id(), "title": title, "prompt": prompt, "seconds": int(seconds),
        "link": link, "cast": list(cast), "overlap": DEFAULT_OVERLAP, "seed": -1,
        "takes_per_run": 1, "takes": [], "in_use": None,
    }


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


def scene_request_values(board: dict[str, Any], index: int, defaults: dict[str, Any], seed: int) -> dict[str, Any]:
    """Every H3 generate value for one take of this scene."""
    scene = board["scenes"][index]
    prompt, pictures = cast_prompt(board, scene)
    values = dict(defaults)
    values.update(board["profile"])
    values.update(
        mode="Reference media" if pictures else "Text to video",
        prompt=prompt,
        first_image=None,
        last_image=None,
        duration=float(scene["seconds"]),
        seed=int(seed),
        result_format="Video",
        **{f"ref_image_{number}": None for number in range(1, 10)},
        **{f"ref_video_{number}": None for number in range(1, 4)},
        **{f"ref_audio_{number}": None for number in range(1, 4)},
        **{f"fl2va_audio_{number}": None for number in range(1, 4)},
    )
    for number, picture in enumerate(pictures, 1):
        values[f"ref_image_{number}"] = picture
    return values


def snapshot(board: dict[str, Any]) -> dict[str, Any]:
    """What a saved version keeps: scene order, scene fields and each take in use."""
    return {"scenes": copy.deepcopy(board["scenes"]), "cast": copy.deepcopy(board["cast"])}
