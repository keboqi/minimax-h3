"""Plan visual beats with the Sound Stage writer and bind them to the mix clock."""

import json
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


def compile_plan(plan, duration, art_direction, limit):
    if not isinstance(plan, dict) or not _text(plan.get("style")):
        raise H3Error("The visual planner omitted the production style.")
    continuity = plan.get("continuity")
    if not isinstance(continuity, dict) or not all(_text(k) and _text(v) for k, v in continuity.items()):
        raise H3Error("The visual planner returned an invalid continuity dictionary.")
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
        refs = row.get("references")
        if not isinstance(refs, list) or not all(isinstance(k, str) and k in continuity for k in refs):
            raise H3Error(f"Shot {i} refers to an undefined character, place or prop.")
        references = tuple(dict.fromkeys(refs))
        prompt = "\n".join(filter(None, [
            "Single wordless widescreen illustration. No captions or lettering.", art_direction, plan["style"],
            *(f"{key}: {continuity[key]}" for key in references), row["prompt"],
        ]))
        frames = end - first
        dissolve = first > 0 and frames > 1 and row["transition"] == "dissolve"
        shots.append(Shot(first, frames, dissolve, prompt, row["beat"], references))
    return shots


def plan_shots(settings, style, limit, model, temporary_api_key, backend, openai_api_key, openai_model,
               openai_base_url, *, runtime):
    duration = float(settings.get("duration") or 0)
    if not isfinite(duration) or duration <= 0:
        raise H3Error("Generate a timed Sound Stage mix first.")
    request = json.dumps(dict(title=settings.get("title"), duration=duration, art_direction=style,
                              max_shots=limit, cues=_timeline(settings, duration)), ensure_ascii=False)
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
    shots = compile_plan(plan, duration, style, limit)
    return plan, shots
