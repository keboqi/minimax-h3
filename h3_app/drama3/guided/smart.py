"""Scene-first planning and bounded content requests for the Smart Guided writer."""

from __future__ import annotations

import json
from dataclasses import replace

from h3_app.errors import H3Error
from .batching import SMART_CHARS, windows
from .model import Scene
from .schema import StageError, canonical, load_answer, parse_content_brief, parse_content_preserve


def parse_plan(reply, source=None, allowed=None):
    data = load_answer(reply)
    cast, scenes = data.get("cast"), data.get("scenes")
    if not isinstance(cast, list) or not isinstance(scenes, list) or not 1 <= len(scenes) <= 64:
        raise StageError(['provide cast and 1–64 scenes'])
    from .schema import _cast
    problems = []
    _cast(data, problems, allowed)
    if any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
           or not isinstance(c.get("role", ""), str) for c in cast):
        problems.append('each cast entry must be an object with name and a compact role')
    if not isinstance(data.get("title", ""), str) or not isinstance(data.get("summary", ""), str):
        problems.append('title and summary must be text')
    if source is not None:
        ids = [u.id for u in source.units]
        starts = [s.get("from") if isinstance(s, dict) else None for s in scenes]
        if (any(not isinstance(s, str) or s not in ids for s in starts) or starts[0] != ids[0]
                or len(set(starts)) != len(starts)
                or [ids.index(s) for s in starts] != sorted(ids.index(s) for s in starts)):
            problems.append('scene starts must be unique, ordered source IDs, beginning at the first unit')
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("change"), str):
            problems.append('each scene needs a short change summary')
        elif source is None and (not isinstance(scene.get("parts"), list)
                                 or not 1 <= len(scene["parts"]) <= 16
                                 or any(not isinstance(p, str) or not p.strip() for p in scene["parts"])):
            problems.append('each brief scene needs 1–16 short ordered part instructions')
        if isinstance(scene, dict) and not isinstance(scene.get("place", ""), str):
            problems.append('scene place must be text')
    if problems:
        raise StageError(problems)
    if len(json.dumps(data, ensure_ascii=False)) > 16000:
        raise StageError(['the story map is too long; keep only compact cast profiles and scene summaries'])
    return data


def context(plan, budget=None, *, scenes=(), names=None):
    """The shared plan is context only; later calls must not re-emit it."""
    if budget is not None and budget.compact:
        cast = plan.get("cast", [])
        if names is not None:
            cast = [c for c in cast if canonical(c["name"]) in names]
        # Keep exact relevant facts, not arbitrarily truncated character profiles.
        plan = {"title": plan.get("title", ""), "summary": plan.get("summary", ""), "cast": cast,
                "scenes": [{k: s[k] for k in ("place", "change") if k in s} for s in scenes]}
    return "Shared story plan (context only):\n" + json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def preserve_content(source, plan, calls, path, prefix, expansion, limit, allowed=None, budget=None):
    """Ask only for local attribution/annotations; code supplies global cast and scenes."""
    starts = {u.id: i for i, u in enumerate(source.units)}
    result = {"speakers": {}, "fixes": [], "pronunciations": [], "insertions": []}
    for index, scene in enumerate(plan["scenes"]):
        begin = starts[scene["from"]]
        end = starts[plan["scenes"][index + 1]["from"]] if index + 1 < len(plan["scenes"]) else len(source.units)
        shared = context(plan, budget, scenes=[scene])
        radius = budget.neighbors if budget else 2
        for chunk in windows(source.units[begin:end], lambda u: u.text, limit,
                             characters=budget.characters if budget else SMART_CHARS):
            local = replace(source, units=tuple(chunk))
            offset = starts[chunk[0].id]
            neighbors = (source.units[max(0, offset - radius):offset]
                         + source.units[offset + len(chunk):offset + len(chunk) + radius])
            listing = "\n".join(f"{u.id} [{u.kind}, label {u.label}] {u.text}" for u in chunk)
            request = (f"{prefix}\n{shared}\nCurrent scene: {json.dumps(scene, ensure_ascii=False)}\n"
                       "Neighboring source (context only; never annotate these IDs):\n"
                       + "\n".join(f"{u.id}: {u.text}" for u in neighbors)
                       + f"\nTARGET units:\n{listing}\nReturn only speakers and optional fixes, pronunciations, "
                         "insertions for TARGET units. Do not return cast, scenes, title or palette. "
                         "Use the shared cast's canonical names. Keep the answer short.")

            def parse(reply):
                data = load_answer(reply)
                data.update(cast=plan["cast"], title=plan.get("title", ""), scenes=[])
                parsed = parse_content_preserve(json.dumps(data, ensure_ascii=False), local, expansion=expansion)
                return parsed

            answer = calls.run(path, request, parse, "scene casting")
            result["speakers"].update(answer["speakers"])
            for field in ("fixes", "pronunciations", "insertions"):
                result[field].extend(answer[field])
    # Validate the merged result too: e.g. conflicting readings in different windows.
    combined = dict(result, title=plan.get("title", ""), cast=plan["cast"], scenes=plan["scenes"])
    combined["fixes"] = [dict(unit=u, **{"from": old, "to": new}) for u, old, new in result["fixes"]]
    combined["pronunciations"] = [dict(word=w, reading=r) for w, r in result["pronunciations"]]
    combined["insertions"] = [dict(after=a, speaker=s, text=t) for a, s, t in result["insertions"]]
    try:
        return parse_content_preserve(json.dumps(combined, ensure_ascii=False), source, allowed, expansion=expansion)
    except StageError as exc:
        raise H3Error(f"Smart scene results conflict: {exc}") from None


def brief_content(plan, calls, path, prefix, limit, notes, budget=None):
    utterances, scenes = [], []
    roles = {canonical(c["name"]): c.get("role", "") for c in plan["cast"] if isinstance(c, dict)}
    for scene in plan["scenes"]:
        scene_ids = []
        beats = []
        for part in scene["parts"]:
            previous_count = 2 * budget.neighbors if budget else 4
            previous = "\n".join(f"{u.speaker}: {u.text}" for u in utterances[-previous_count:])
            # The outline's other scenes are summaries in token mode, never repeated part lists.
            request = (f"{prefix}\n{context(plan, budget, scenes=plan['scenes'])}\nCurrent scene: {json.dumps(scene, ensure_ascii=False)}\n"
                       f"Write ONLY this part: {part}\nPrevious spoken lines (do not repeat):\n{previous}\n"
                       f"Return only scenes with exactly one scene, containing 1–{limit} short lines. "
                       "Do not repeat cast, title or palette; use the shared cast. Do not write other parts.")

            def parse(reply):
                data = load_answer(reply)
                raw = data.get("scenes")
                if (not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict)
                        or not isinstance(raw[0].get("lines"), list) or not 1 <= len(raw[0]["lines"]) <= limit):
                    raise StageError([f'return exactly one scene with 1–{limit} lines'])
                data.update(cast=plan["cast"], title=plan.get("title", ""))
                # Lead with this scene's previous line so an opening reaction or interrupt has its cue.
                lead = utterances[-1:] if scene_ids else []
                if lead:
                    raw[0]["lines"] = [{"speaker": lead[0].speaker, "text": lead[0].text}, *raw[0]["lines"]]
                title, parsed_lines, parsed_scenes, parsed_roles = parse_content_brief(
                    json.dumps(data, ensure_ascii=False), notes=notes)
                if lead:
                    parsed_lines = parsed_lines[1:]
                    parsed_scenes = [replace(parsed_scenes[0], utterances=parsed_scenes[0].utterances[1:])]
                parsed = title, parsed_lines, parsed_scenes, parsed_roles
                max_chars = budget.characters if budget else SMART_CHARS
                if sum(len(line.text) for line in parsed[1]) > max_chars:
                    raise StageError([f'this part is too long; keep spoken text below {max_chars} characters'])
                return parsed

            _, lines, written_scenes, _ = calls.run(path, request, parse, "scene writing")
            beats.extend(written_scenes[0].beats)
            for line in lines:
                uid = f"u{len(utterances) + 1}"
                utterances.append(replace(line, id=uid))
                scene_ids.append(uid)
        scenes.append(Scene(f"c{len(scenes) + 1}", tuple(scene_ids), scene.get("place", ""), scene["change"],
                            tuple(dict.fromkeys(beats))))
    return plan.get("title", ""), utterances, scenes, roles


