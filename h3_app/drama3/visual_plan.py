"""Plan visual beats with the Sound Stage writer and bind them to the mix clock."""

import json
import re
from itertools import pairwise
from math import ceil, isfinite

from h3_app.errors import H3Error
from h3_app.llm import repair_json
from h3_app.prompt_writers.common import read_system_prompt, write_prompt
from .visuals import FPS, Shot
from .visual_effects import EFFECT_PRESETS, check_effects


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


# The frame each shot is drawn in. The plan itself is written for either, so it can change later.
ORIENTATIONS = ("landscape", "portrait")
FRAME_LINES = {
    "landscape": "Single widescreen 16:9 illustration of one moment, composed for a wide frame. ",
    "portrait": "Single vertical 9:16 illustration of one moment, composed for a tall frame. ",
}
NO_TEXT = "No text, captions, speech bubbles or lettering."
SHOT_LINE = FRAME_LINES["landscape"] + NO_TEXT


def frame_line(orientation="landscape"):
    """The line every shot prompt opens with: one moment in a wide or tall frame, without text."""
    return FRAME_LINES.get(orientation, FRAME_LINES["landscape"]) + NO_TEXT


def plan_art_style(plan):
    """An older plan's production style, which the planner wrote for one art direction; new plans
    leave style out so it can change after planning."""
    value = plan.get("art_style") or plan.get("style") if isinstance(plan, dict) else None
    return value.strip() if _text(value) else None
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
    return re.compile(rf"@({names})(?!\w)(?:\s*[(（]([^)）]*)[)）])?") if names else None


_SLOT = re.compile(r"⟦([^⟦⟧]+)⟧")


def fill_tags(text, tags):
    """Put each subject's reference tag in its slot after the name; drop the slots of the rest."""
    return _SLOT.sub(lambda match: f" {tags[match.group(1)]}" if match.group(1) in tags else "", text)


def expand_prompt(prompt, known, slots=False):
    """Inline each subject's anchor where the shot mentions it, keeping the shot's current state.

    With ``slots``, each name is followed by ⟦key⟧ for ``fill_tags``. Returns the expanded text and
    the mentioned keys in order of first mention.
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
        name = f"{key}⟦{key}⟧" if slots else key
        return f"{name} ({anchor}; {state})" if state else f"{name} ({anchor})"

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
        f"Art style: {art_style}" if art_style else None, art_direction,
        f"{body}{backdrop} No text, labels or captions."]))


def compile_plan(plan, duration, art_direction, limit, *, orientation="landscape", art_style=None):
    """The plan's shots on the mix clock, each prompt in the given frame and style.

    The plan holds only what each shot shows; ``art_direction`` (and an older plan's ``art_style``,
    passed only while its art direction is still the one chosen) and the frame are added here.
    """
    if not isinstance(plan, dict):
        raise H3Error("The visual planner did not return a JSON object.")
    known = subjects(plan)
    rows = plan.get("shots")
    if plan.get("script_cues") and isinstance(rows, list):
        limit = max(limit, min(MAX_SHOTS, len(rows)))
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise H3Error(f"The visual planner must return between 1 and {limit} shots.")
    intervals = pairwise(_frame_boundaries(rows, duration))
    shots = []
    for i, (row, (first, end)) in enumerate(zip(rows, intervals, strict=True), 1):
        check_effects(row.get("effects", []))
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
        tagged, _ = expand_prompt(row["prompt"].strip(), known, slots=True)
        if not mentioned:
            scene = "\n".join([scene, *(f"{key} ({known[key][1]})" for key in references)])
            tagged = "\n".join([tagged, *(f"{key}⟦{key}⟧ ({known[key][1]})" for key in references)])
        # Establish the depiction before material-rich scene text, and use the same rendering
        # direction for the whole frame rather than treating style as a trailing qualifier.
        prompt = "\n".join(filter(None, [frame_line(orientation),
                                          f"Art style: {art_style}" if art_style else None, art_direction,
                                          scene]))
        frames = end - first
        dissolve = first > 0 and frames > 1 and row["transition"] == "dissolve"
        shots.append(Shot(first, frames, dissolve, prompt, row["beat"], references, tagged, row.get("id", "")))
    return shots


def edit_shot(plan, index, *, prompt=None, transition=None, effects=None):
    """Copy a plan with one shot's image prompt, transition or playback effects changed."""
    rows = plan.get("shots") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not isinstance(index, int) or not 0 <= index < len(rows):
        raise H3Error("That shot is not in the current shot plan. Reload the page and try again.")
    row = dict(rows[index])
    if effects is not None:
        row["effects"] = check_effects(effects)
    if prompt is not None:
        if not _text(prompt):
            raise H3Error("A shot needs an image prompt.")
        row["prompt"] = " ".join(prompt.split())
    if transition is not None:
        if transition not in ("cut", "dissolve"):
            raise H3Error("A shot comes in with a cut or a dissolve.")
        row["transition"] = transition
    return {**plan, "shots": [*rows[:index], row, *rows[index + 1:]]}


def edit_subject(plan, kind, key, anchor):
    """A copy of ``plan`` with one character's, prop's or place's looks changed."""
    if not _text(anchor):
        raise H3Error("Describe how it looks.")
    if kind == "characters" and all(plan.get(name) is None for name in KINDS) and "continuity" in plan:
        kind = "continuity"  # a legacy plan keeps its characters there
    group = plan.get(kind) if kind in (*KINDS, "continuity") else None
    if not isinstance(group, dict) or key not in group:
        raise H3Error("That character, prop or place is not in the current shot plan. Reload the page.")
    return {**plan, kind: {**group, key: " ".join(anchor.split())}}


def mentions(plan):
    """The subject keys each shot of ``plan`` shows, in order of first mention."""
    known = subjects(plan)
    found = []
    for row in plan.get("shots") or []:
        _, keys = expand_prompt(str(row.get("prompt") or ""), known)
        listed = [key for key in row.get("references") or [] if isinstance(key, str) and key in known]
        found.append(tuple(dict.fromkeys([*keys, *listed])))
    return found


MAX_SHOTS = 200


def shot_cap(duration, minimum):
    """The most shots a plan may use: one per 2 seconds of mix, never below the requested minimum."""
    return max(int(minimum), min(MAX_SHOTS, ceil(float(duration) / 2)))


def _duration(settings):
    duration = float(settings.get("duration") or 0)
    if not isfinite(duration) or duration <= 0:
        raise H3Error("Generate a timed Sound Stage mix first.")
    return duration


def _speakers(settings, cues):
    """The voice description of each speaker with a line in the mix, from the Cast (``voices``)."""
    voices = {name.casefold(): text.strip() for name, text in (settings.get("voices") or {}).items() if _text(text)}
    names = dict.fromkeys(cue["speaker"] for cue in cues if _text(cue.get("speaker")))
    return [dict(name=name, voice=voices[name.casefold()]) for name in names if name.casefold() in voices]


def plan_request(settings, minimum):
    """The JSON request the planner answers: the timed mix, its speakers' voices and the shot range.
    The art direction and frame stay out, so they can change without planning again."""
    duration = _duration(settings)
    from .visual_timing import mix_cues
    cues = mix_cues(settings)
    if not cues:
        raise H3Error("This mix has no timed dialogue. Generate a mix before planning.")
    speakers = _speakers(settings, cues)
    return json.dumps(dict(title=settings.get("title"), duration=None if settings.get("untimed") else duration,
                           timing="untimed" if settings.get("untimed") else "anchored", script=settings.get("script", ""),
                           anchor_format="cue (zero-based dialogue index), offset (fraction 0 <= offset < 1)",
                           min_shots=int(minimum), max_shots=MAX_SHOTS if settings.get("untimed") else shot_cap(duration, minimum),
                           **({"speakers": speakers} if speakers else {}), cues=cues,
                           sounds=settings.get("sounds") or [], effect_presets=EFFECT_PRESETS), ensure_ascii=False)


def full_plan_prompt(settings, minimum, *, runtime):
    """The exact system prompt and request the built-in planner sends, for use in another chat model."""
    system_path = runtime.prompt_systems["Drama 3 visuals"]
    system_prompt = read_system_prompt(system_path, "Sound Stage shot plan")
    return f"{system_prompt}\n\n----- USER REQUEST -----\n\n{plan_request(settings, minimum)}\n"


def _reply_json(reply):
    """The JSON object in a reply, allowing a code fence or a sentence of chat around it."""
    text = str(reply or "").strip()
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    if "{" in text and "}" in text:
        try:
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
        except ValueError:
            pass
    return repair_json(text)


def check_plan_reply(reply, settings, minimum):
    """Validate a planner reply, from the built-in writer or pasted from another chat model."""
    duration = _duration(settings)
    plan = _reply_json(reply)
    if plan is None:
        raise H3Error("The shot plan was cut off or isn't valid JSON. "
                      "Plan again, or lower Minimum shots for a long mix.")
    from .visual_timing import anchor_plan, bind_plan
    if not isinstance(plan, dict):
        raise H3Error("The visual planner did not return a JSON object.")
    # Older pasted timestamp replies remain usable, but are stored as anchors too.
    plan = anchor_plan({**plan, "script_cues": None}, settings)
    compiled = bind_plan(plan, settings)
    shots = compile_plan(compiled, duration, "", shot_cap(duration, minimum))
    # A mix shorter than the minimum at one frame per shot can't hold that many.
    if len(shots) < min(int(minimum), ceil(duration * FPS)):
        raise H3Error(f"The plan has {len(shots)} shots, fewer than the minimum of {int(minimum)}. "
                      "Plan again, or lower Minimum shots.")
    return plan, shots


def plan_shots(settings, minimum, model, temporary_api_key, backend, openai_api_key, openai_model,
               openai_base_url, *, runtime):
    """Plan at least ``minimum`` shots; the writer adds more wherever the story needs them."""
    request = plan_request(settings, minimum)
    reply, status = write_prompt(prompt=request, request_text=request, model=model,
                                 temporary_api_key=temporary_api_key, backend=backend,
                                 openai_api_key=openai_api_key, openai_model=openai_model,
                                 openai_base_url=openai_base_url, target="Sound Stage shot plan",
                                 system_path=runtime.prompt_systems["Drama 3 visuals"], media_values=[], context="")
    if status.startswith("Prompt enhancement failed"):
        raise H3Error(status)
    return check_plan_reply(reply, settings, minimum)
