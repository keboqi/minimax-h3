"""Plan visual beats with the Sound Stage writer and bind them to the mix clock."""

import json
import re
from itertools import pairwise
from math import ceil, isfinite

from h3_app.errors import H3Error
from h3_app.llm import repair_json
from h3_app.prompt_writers.common import write_prompt
from .visuals import FPS, Shot


def _timeline(settings, duration):
    cues = []

    def add(item, **details):
        start, end = item.get("start"), item.get("end")
        if start is not None and end is not None and 0 <= start < min(duration, end):
            cues.append(dict(start=start, end=min(duration, end), **details))

    for chunk in settings.get("chunks") or []:
        for line in chunk.get("lines") or [chunk]:
            if line.get("text"):
                add(line, speaker=chunk.get("name") or chunk.get("speaker"), text=line["text"],
                    **({"words": [[w["start"], w["text"]] for w in line["words"]]} if line.get("words") else {}))
    for sound in settings.get("sounds") or []:
        add(sound, sound=sound.get("name"), description=sound.get("description"), loop=bool(sound.get("loop")))
    if not cues:
        raise H3Error("This mix has no timed dialogue or sounds. Generate a timed mix before planning.")
    return sorted(cues, key=lambda cue: cue["start"])


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _frame_boundaries(rows, duration):
    total = ceil(duration * FPS)
    starts = []
    for i, row in enumerate(rows, 1):
        start = row.get("start") if isinstance(row, dict) else None
        if type(start) not in (int, float) or not 0 <= start < duration:
            raise H3Error(f"Shot {i} needs a start time within the mix.")
        frame = round(start * FPS)
        if (i == 1 and start != 0) or (starts and frame <= starts[-1]) or frame >= total:
            raise H3Error("Shot boundaries must start at zero and advance by at least one frame.")
        starts.append(frame)
    return [*starts, total]


SHOT_LINE = ("Single widescreen 16:9 illustration of one moment. "
             "No text, captions, speech bubbles or lettering.")
# The kinds of recurring subjects, each drawn as a reference sheet before any scene.
KINDS = ("characters", "props", "locations")


def subjects(plan):
    """Recurring subjects as ``{key: (kind, anchor)}``; a legacy ``continuity`` entry is a character."""
    found = {}
    groups = [(kind, plan.get(kind)) for kind in KINDS]
    if all(group is None for _, group in groups) and "continuity" in plan:
        groups = [("characters", plan.get("continuity"))]
    for kind, group in groups:
        if group is None:
            continue
        if not isinstance(group, dict) or not all(_text(k) and _text(v) for k, v in group.items()):
            raise H3Error(f"The visual planner returned an invalid {kind} dictionary.")
        for key, anchor in group.items():
            found.setdefault(key.strip(), (kind, anchor.strip()))
    return found


def _mentions(keys):
    """A pattern for ``@key`` or ``@key(current state)``, longest keys first so prefixes never win."""
    names = "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True))
    return re.compile(rf"@({names})(?:\s*[(（]([^)）]*)[)）])?") if names else None


def expand_prompt(prompt, known):
    """Inline each subject's anchor where the shot mentions it, keeping the shot's current state.

    Returns the expanded text and the mentioned keys in order of first mention.
    """
    pattern = _mentions(known)
    mentioned = []
    if pattern is None:
        return prompt.replace("@", ""), ()

    def inline(match):
        key, state = match.group(1), (match.group(2) or "").strip()
        mentioned.append(key)
        # A state can mention another subject: @mara(holding @radio).
        state = pattern.sub(inline, state)
        anchor = known[key][1]
        return f"{key} ({anchor}; {state})" if state else f"{key} ({anchor})"

    text = pattern.sub(inline, prompt)
    # An @word the bible never defined is ordinary text.
    return re.sub(r"@(?=\w)", "", text), tuple(dict.fromkeys(mentioned))


def sheet_prompt(kind, key, anchor, art_style, art_direction):
    """A reference image for one recurring subject.

    A character gets one centered portrait, never a multi-view sheet: an edit model pastes extra
    views into the scene it is asked to match.
    """
    body = {
        "characters": f"Portrait of {key}: {anchor}. One single figure from the waist up, facing the viewer, "
                      "centered, calm expression. Only this one view of this one person.",
        "props": f"Prop reference sheet of {key}: {anchor}. The object alone, centered, shown from the front "
                 "and at a three-quarter angle.",
        "locations": f"Location reference of {key}: {anchor}. An empty wide establishing view that shows its "
                     "layout and landmarks, no people.",
    }[kind]
    backdrop = "" if kind == "locations" else " Plain light grey background, even soft lighting."
    return "\n".join(filter(None, [
        f"{body}{backdrop} No text, labels or captions.", f"Art style: {art_style}", art_direction]))


def compile_plan(plan, duration, art_direction, limit):
    if not isinstance(plan, dict):
        raise H3Error("The visual planner did not return a JSON object.")
    art_style = plan.get("art_style") or plan.get("style")
    if not _text(art_style):
        raise H3Error("The visual planner omitted the production style.")
    known = subjects(plan)
    rows = plan.get("shots")
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise H3Error(f"The visual planner must return between 1 and {limit} shots.")
    intervals = pairwise(_frame_boundaries(rows, duration))
    shots = []
    for i, (row, (first, end)) in enumerate(zip(rows, intervals, strict=True), 1):
        if not all(_text(row.get(key)) for key in ("prompt", "beat")):
            raise H3Error(f"Shot {i} is missing its beat or image prompt.")
        if row.get("transition") not in ("cut", "dissolve"):
            raise H3Error(f"Shot {i} has an unknown transition.")
        scene, mentioned = expand_prompt(row["prompt"].strip(), known)
        # Older plans listed references separately instead of mentioning them.
        listed = row.get("references") or []
        if not isinstance(listed, list):
            raise H3Error(f"Shot {i} has an invalid references list.")
        references = tuple(dict.fromkeys([*mentioned, *(k for k in listed if isinstance(k, str) and k in known)]))
        if not mentioned:
            scene = "\n".join([scene, *(f"{key} ({known[key][1]})" for key in references)])
        prompt = "\n".join(filter(None, [SHOT_LINE, scene, f"Art style: {art_style.strip()}", art_direction]))
        frames = end - first
        dissolve = first > 0 and frames > 1 and row["transition"] == "dissolve"
        shots.append(Shot(first, frames, dissolve, prompt, row["beat"], references))
    return shots


MAX_SHOTS = 200


def shot_cap(duration, minimum):
    """The most shots a plan may use: one per 2 seconds of mix, never below the requested minimum."""
    return max(int(minimum), min(MAX_SHOTS, ceil(float(duration) / 2)))


def plan_shots(settings, style, minimum, model, temporary_api_key, backend, openai_api_key, openai_model,
               openai_base_url, *, runtime):
    """Plan at least ``minimum`` shots; the writer adds more wherever the story needs them."""
    duration = float(settings.get("duration") or 0)
    if not isfinite(duration) or duration <= 0:
        raise H3Error("Generate a timed Sound Stage mix first.")
    limit = shot_cap(duration, minimum)
    request = json.dumps(dict(title=settings.get("title"), duration=duration, art_direction=style,
                              min_shots=int(minimum), max_shots=limit, cues=_timeline(settings, duration)),
                         ensure_ascii=False)
    reply, status = write_prompt(prompt=request, request_text=request, model=model,
                                 temporary_api_key=temporary_api_key, backend=backend,
                                 openai_api_key=openai_api_key, openai_model=openai_model,
                                 openai_base_url=openai_base_url, target="Sound Stage shot plan",
                                 system_path=runtime.prompt_systems["Drama 3 visuals"], media_values=[], context="")
    if status.startswith("Prompt enhancement failed"):
        raise H3Error(status)
    try:
        plan = json.loads(reply)
    except ValueError:
        plan = repair_json(reply)
    if plan is None:
        raise H3Error("The writer's shot plan was cut off or isn't valid JSON. "
                      "Plan again, or lower Minimum shots for a long mix.")
    shots = compile_plan(plan, duration, style, limit)
    # A mix shorter than the minimum at one frame per shot can't hold that many.
    if len(shots) < min(int(minimum), ceil(duration * FPS)):
        raise H3Error(f"The writer planned {len(shots)} shots, fewer than the minimum of {int(minimum)}. "
                      "Plan again, or lower Minimum shots.")
    return plan, shots
