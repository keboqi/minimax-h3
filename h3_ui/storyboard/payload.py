"""The storyboard as JSON for the page: everything a scene needs to show without asking again."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from h3_app import media_review, storyboard as sb

MODE_NAMES = {sb.TEXT: "text only", sb.FRAMES: "picture frames", sb.REFERENCES: "references"}
TAKE_NOTE = "Every take is kept. Generate again for another version, then choose which one to use."


def seconds_label(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + " s"


def clock(value: float) -> str:
    total = int(round(value))
    return f"{total // 60}:{total % 60:02d}"


def file_url(storyboard_id: str, path: str, **extra) -> str:
    return f"/storyboard/api/boards/{storyboard_id}/file?" + urlencode({"path": path, **extra})


def thumb_url(storyboard_id: str, path: str) -> str:
    return f"/storyboard/api/boards/{storyboard_id}/thumb?" + urlencode({"path": path})


def poster_url(path: str) -> str:
    return "/video-posters?" + urlencode({"path": path})


def referenced_paths(board: dict[str, Any]) -> set[str]:
    """Every file this storyboard shows; the page's file routes serve nothing else."""
    paths = {member["image"] for member in board["cast"]}
    paths |= {take["video"] for take in board["takes"].values()}
    for scene in board["scenes"]:
        paths |= set(sb.model.media_paths(scene))
    return paths


def _media(storyboard_id: str, path: str, *, picture=False) -> dict[str, Any]:
    item = {"path": path, "name": Path(path).name, "url": file_url(storyboard_id, path)}
    if picture:
        item["thumb"] = thumb_url(storyboard_id, path)
    return item


def scene_summary(board: dict, index: int) -> str:
    start = "Opening" if index == 0 else f"Continues scene {index}" if sb.flows_from_previous(board, index) \
        else "Fresh start"
    mode = sb.scene_media(board["scenes"][index])["mode"]
    name = "last frame" if mode == sb.FRAMES and sb.flows_from_previous(board, index) else MODE_NAMES[mode]
    return f"{start} · {name}"


def _context(board, index, flow) -> str:
    if index == 0:
        return "This is the opening scene."
    title = board["scenes"][index - 1]["title"]
    if flow:
        return f"Comes after scene {index} “{title}” and continues from its last frames."
    return f"Comes after scene {index} “{title}” as a fresh start (a new shot)."


def _prompt_hint(index, scene, media, flow) -> str:
    if media["mode"] == sb.REFERENCES:
        return "Refer to references by name or tag, for example " + (
            f"“{scene['cast'][0]}” or <Picture 1>." if scene["cast"] else "<Picture 1>.")
    if flow:
        return f"Describe what happens after the hand-over from scene {index}."
    return "Describe the action, the camera and the sound."


def _added(board, index, scene, flow) -> str:
    added = sb.new_seconds(board, index)
    if flow:
        return (f"Adds about {seconds_label(added)} of new footage; the first "
                f"{seconds_label(scene['seconds'] - added)} repeats scene {index}.")
    return f"Adds {seconds_label(added)}."


def _thumb(storyboard_id, board, scene, media) -> str | None:
    """A scene's filmstrip picture: its take in use, else its own pictures."""
    take = board["takes"].get(scene["in_use"]) if scene["in_use"] else None
    if take is not None:
        return poster_url(take["video"])
    pictures = [media["first_image"], media["last_image"], *media["ref_images"]]
    picture = next((path for path in pictures if path), None)
    return thumb_url(storyboard_id, picture) if picture else None


def _continued_by(board, take_id) -> list[int]:
    """The numbers of the scenes whose take in use was made from this take."""
    return [number for number, scene in enumerate(board["scenes"], 1)
            if board["takes"].get(scene["in_use"], {}).get("inputs", {}).get("source_take") == take_id]


def _scene(storyboard_id, board, index, states) -> dict[str, Any]:
    scene = board["scenes"][index]
    media = sb.scene_media(scene)
    flow = sb.flows_from_previous(board, index)
    state = states[index]
    issues = sb.scene_issues(board, index, states)
    own = [issue for issue in issues if not issue.startswith("Generate scene")]
    takes = [
        {"id": take_id, "number": number, "seed": board["takes"][take_id]["seed"],
         "video": file_url(storyboard_id, board["takes"][take_id]["video"]),
         "poster": poster_url(board["takes"][take_id]["video"]),
         "missing": not Path(board["takes"][take_id]["video"]).is_file(),
         "continued_by": _continued_by(board, take_id),
         "review": media_review.public(board["takes"][take_id]["video"])}
        for number, take_id in enumerate(scene["takes"], 1) if take_id in board["takes"]
    ]
    pictures = []
    if media["mode"] == sb.REFERENCES:
        _, cast_images = sb.model.cast_prompt(board, scene)
        pictures = [{"tag": tag, **_media(storyboard_id, path, picture=True)}
                    for path, tag in zip([*cast_images, *media["ref_images"]], sb.picture_tags(board, scene))]
    return {
        "id": scene["id"], "number": index + 1, "title": scene["title"], "prompt": scene["prompt"],
        "seconds": scene["seconds"], "link": scene["link"], "flow": flow, "overlap": int(scene["overlap"]),
        "seed": int(scene["seed"]), "takes_per_run": int(scene["takes_per_run"]), "cast": list(scene["cast"]),
        "mode": media["mode"], "runs_as": sb.runs_as(board, index), "state": state,
        "state_label": "Needs input" if state == sb.DRAFT and own else sb.STATE_NAMES[state],
        "summary": scene_summary(board, index), "context": _context(board, index, flow),
        "issues": issues, "hint": _prompt_hint(index, scene, media, flow), "added": _added(board, index, scene, flow),
        "added_seconds": max(sb.new_seconds(board, index), 0.5),
        "tags": sb.picture_tags(board, scene), "pictures": pictures,
        "first_image": _media(storyboard_id, media["first_image"], picture=True) if media["first_image"] else None,
        "last_image": _media(storyboard_id, media["last_image"], picture=True) if media["last_image"] else None,
        "ref_images": [_media(storyboard_id, path, picture=True) for path in media["ref_images"]],
        "ref_videos": [{**_media(storyboard_id, path), "poster": poster_url(path)} for path in media["ref_videos"]],
        "ref_audios": [_media(storyboard_id, path) for path in media["ref_audios"]],
        "takes": takes, "in_use": scene["in_use"], "thumb": _thumb(storyboard_id, board, scene, media),
    }


def board_payload(storyboard_id: str, board: dict[str, Any]) -> dict[str, Any]:
    states = sb.scene_states(board)
    scenes = [_scene(storyboard_id, board, index, states) for index in range(len(board["scenes"]))]
    done = states.count(sb.DONE)
    return {
        "id": storyboard_id, "name": board["name"], "settings": sb.profile_summary(board["profile"]),
        "total": clock(sb.total_seconds(board)), "done": done, "pending": len(states) - done,
        "queue": list(board["queue"]), "scenes": scenes,
        "cast": [{**_media(storyboard_id, member["image"], picture=True), "name": member["name"]}
                 for member in board["cast"]],
        "versions": [{"id": version["id"], "label": version["label"]} for version in reversed(board["versions"])],
        "take_note": TAKE_NOTE,
    }
