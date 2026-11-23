"""Bridge Soundstage illustrated audio projects into mode-aware Storyboard specs."""
from __future__ import annotations

from typing import Any

from .model import CUT, FLOW, SCENE_SECONDS
from .writer import plan_scenes_with_modes


def _soundstage_cast(project_data: dict[str, Any], visual_plan: dict[str, Any] | None = None) -> tuple[list[str], dict[str, str]]:
    """Extract canonical non-narrator cast names and visual descriptions from a Soundstage project."""
    names: list[str] = []
    descriptions: dict[str, str] = {}
    plan = (visual_plan or {}).get("plan") if isinstance(visual_plan, dict) else None
    if isinstance(plan, dict):
        for key, desc in (plan.get("characters") or {}).items():
            clean = str(key or "").strip()
            if clean and clean.casefold() != "narrator":
                names.append(clean)
                descriptions[clean] = str(desc or "").strip()
    for slot in range(16):
        save_name = str(project_data.get(f"voice_{slot}_save_name") or "").strip()
        if not save_name or save_name.casefold() == "narrator":
            continue
        if save_name not in descriptions:
            names.append(save_name)
            descriptions[save_name] = str(project_data.get(f"voice_{slot}_description") or "").strip()
    return list(dict.fromkeys(names)), descriptions


def storyboard_spec_from_soundstage(
    name: str,
    project_data: dict[str, Any],
    *,
    visual_plan: dict[str, Any] | None = None,
    usual_seconds: int = 8,
) -> dict[str, Any]:
    """Promote a Soundstage project (cast + visual shot plan or script) into a Storyboard import spec."""
    vplan = visual_plan if visual_plan is not None else project_data.get("visual_plan")
    cast_names, cast_descs = _soundstage_cast(project_data, vplan)
    shots = (((vplan or {}).get("plan") or {}).get("shots") or []) if isinstance(vplan, dict) else []

    raw_scenes: list[dict[str, Any]] = []
    if shots:
        for idx, shot in enumerate(shots, 1):
            prompt = str(shot.get("prompt") or "").strip()
            if not prompt:
                continue
            transition = str(shot.get("transition") or "cut").strip().lower()
            raw_scenes.append({
                "key": f"shot_{idx:02d}",
                "title": str(shot.get("beat") or f"Shot {idx}"),
                "prompt": prompt,
                "seconds": min(max(int(shot.get("seconds") or usual_seconds), SCENE_SECONDS[0]), SCENE_SECONDS[-1]),
                "link": FLOW if idx > 1 and transition == "dissolve" else CUT,
            })
    else:
        script_text = str(project_data.get("text") or "").strip()
        if script_text:
            raw_scenes.append({
                "key": "shot_01",
                "title": str(project_data.get("title") or name),
                "prompt": script_text[:800],
                "seconds": min(max(int(usual_seconds), SCENE_SECONDS[0]), SCENE_SECONDS[-1]),
                "link": CUT,
            })

    planned_scenes = plan_scenes_with_modes(
        raw_scenes,
        cast_names=cast_names,
        cast_descriptions=cast_descs,
        usual_seconds=usual_seconds,
    )
    return {
        "name": str(name or project_data.get("title") or "Soundstage Storyboard")[:120],
        "source_soundstage": str(name),
        "cast": [{"name": c, "description": cast_descs.get(c, "")} for c in cast_names],
        "scenes": planned_scenes,
    }
