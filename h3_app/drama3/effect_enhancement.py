"""Text contract for enhancing the effects of already drawn Sound Stage shots."""

from hashlib import sha256
import json
from math import isfinite

from h3_app.errors import H3Error
from h3_app.prompt_writers.common import read_system_prompt, write_prompt
from .visual_effects import EFFECT_PRESETS
from .visual_plan import _reply_json

MAX_LAYERS = 16
MAX_NEW_REGIONS = 6
MIN_INCLUDE_POINTS = 2
MAX_REGION_POINTS = 8
FULL_FRAME = "full_frame"
MODES = {
    FULL_FRAME: "Apply to the whole shot.",
    "inside": "Apply inside the selected silhouette.",
    "outside": "Apply everywhere outside the selected silhouette.",
    "behind": "Show the effect behind the selected foreground or midground silhouette using its metric depth so closer objects occlude it.",
}
CATALOG = {
    key: {**preset, "modes": [FULL_FRAME] + (
        ["inside", "outside", "behind"] if preset["kind"] == "atmosphere" else
        ["inside", "outside"] if preset["kind"] == "lighting" else []
    )}
    for key, preset in EFFECT_PRESETS.items()
}
LAYER_FIELDS = {
    "preset_or_presets": "single preset ID (`preset`) or ordered list (`presets`) sharing the same mode, target, intensity, and feather",
    "mode": "full_frame, inside, outside, or behind",
    "region_id_or_region_ids": "required when mode != full_frame; single ID (`region_id`) or list of IDs (`region_ids`) combined into one mask",
    "intensity": "optional 0–1 for non-camera presets",
    "feather": "optional 0–32 pixels for masked non-camera presets",
}
NEW_REGION_FIELDS = {
    "id": "temporary ID such as `new_1` referenced by `region_id` or `region_ids` in `effects`",
    "label": "short display label for the segmented object (for example `brass lantern`)",
    "prompt": "concise ASCII English visual noun phrase (1–4 words) for SAM 3.1 text detection",
    "box": "normalized `[x0, y0, x1, y1]` bounding box in 0–1 coordinates framing the object in <Shot Image>",
    "include": "required 2–4 (up to 8) distinct normalized `[x, y]` interior points spread across different parts of the object in <Shot Image> so SAM segments the complete object instead of a subpart",
    "exclude": "optional normalized `[x, y]` points on adjacent background or occluders to exclude",
    "merge": "optional boolean (`true` only when combining multiple matching instances into one mask)",
}


def _normalize_text(value, limit):
    return " ".join(str(value or "").split())[:limit]


def revision(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def base_stack(effects):
    return [{"preset": preset, "mode": FULL_FRAME} for preset in effects]


def layer_region_ids(layer):
    if isinstance(layer.get("region_ids"), list):
        return [rid for rid in layer["region_ids"] if isinstance(rid, str)]
    if isinstance(layer.get("region_id"), str):
        return [layer["region_id"]]
    return []


def _layer_region_target(layer, mode, regions):
    if mode == FULL_FRAME:
        return {}
    if "region_id" in layer and "region_ids" in layer:
        raise H3Error("Specify either region_id or region_ids for an effect layer.")
    if "region_ids" in layer:
        raw_ids = layer.get("region_ids")
        if not isinstance(raw_ids, list) or not raw_ids or any(
            not isinstance(rid, str) or rid not in regions for rid in raw_ids
        ):
            raise H3Error("An effect targets a region outside this shot.")
        ids = list(dict.fromkeys(raw_ids))
    else:
        region = layer.get("region_id")
        if not isinstance(region, str) or region not in regions:
            raise H3Error("An effect targets a region outside this shot.")
        ids = [region]
    return {"region_id": ids[0]} if len(ids) == 1 else {"region_ids": ids}


def check_stack(value, regions):
    if not isinstance(value, list):
        raise H3Error(f"Use an ordered list of up to {MAX_LAYERS} effect layers.")
    stack = []
    camera = False
    for layer in value:
        if not isinstance(layer, dict):
            raise H3Error("Each effect layer must be an object.")
        if "preset" in layer and "presets" in layer:
            raise H3Error("Specify either preset or presets for an effect layer.")
        if "presets" in layer:
            raw_presets = layer.get("presets")
            if not isinstance(raw_presets, list) or not raw_presets:
                raise H3Error("An effect uses an unknown preset or unsupported application mode.")
            preset_list = raw_presets
        else:
            preset_list = [layer.get("preset")]
        mode = layer.get("mode")
        for preset in preset_list:
            if not isinstance(preset, str) or preset not in CATALOG or mode not in CATALOG[preset]["modes"]:
                raise H3Error("An effect uses an unknown preset or unsupported application mode.")
        target = _layer_region_target(layer, mode, regions)
        for preset in preset_list:
            spec = CATALOG[preset]
            if spec["kind"] == "camera":
                if camera:
                    raise H3Error("Use one camera move per shot.")
                camera = True
            clean = {"preset": preset, "mode": mode, **target}
            if spec["kind"] != "camera":
                limits = (("intensity", 1), ("feather", 32)) if mode != FULL_FRAME else (("intensity", 1),)
                for key, maximum in limits:
                    if key not in layer:
                        continue
                    number = layer[key]
                    if type(number) not in (int, float) or not 0 <= number <= maximum or not isfinite(number):
                        raise H3Error(f"Effect {key} must be between 0 and {maximum}.")
                    clean[key] = number
            stack.append(clean)
            if len(stack) > MAX_LAYERS:
                raise H3Error(f"Use an ordered list of up to {MAX_LAYERS} effect layers.")
    return stack


def request_text(shots, token):
    return json.dumps({
        "revision": token,
        "max_layers_per_shot": MAX_LAYERS,
        "application_modes": MODES,
        "presets": CATALOG,
        "layer_fields": LAYER_FIELDS,
        "shots": shots,
    }, ensure_ascii=False)


def prompt_path(runtime):
    return runtime.script_dir / "prompts" / "prompt_soundstage_effects.txt"


def shot_prompt_path(runtime):
    return runtime.script_dir / "prompts" / "prompt_soundstage_shot_effects.txt"


def copy_prompt(shots, token, runtime):
    return f"{read_system_prompt(prompt_path(runtime), 'Sound Stage effects')}\n\n{request_text(shots, token)}"


def shot_request_text(shot, token, *, can_add_regions=True, user_direction=""):
    direction = _normalize_text(user_direction, 400)
    payload = {
        "revision": token,
        "max_layers_per_shot": MAX_LAYERS,
        "can_add_regions": bool(can_add_regions),
        **({"max_new_regions": MAX_NEW_REGIONS, "new_region_fields": NEW_REGION_FIELDS} if can_add_regions else {}),
        "application_modes": MODES,
        "presets": CATALOG,
        "layer_fields": LAYER_FIELDS,
        **({"user_direction": direction} if direction else {}),
        "shot": shot,
    }
    return json.dumps(payload, ensure_ascii=False)


def copy_shot_prompt(shot, token, runtime, *, can_add_regions=True, user_direction=""):
    body = shot_request_text(shot, token, can_add_regions=can_add_regions, user_direction=user_direction)
    return f"{read_system_prompt(shot_prompt_path(runtime), 'Sound Stage shot effects')}\n\n{body}"


def _invoke_writer(request, writer, system_path, target, *, media_values=()):
    model, gemini_key, backend, openai_key, openai_model, openai_url = writer
    reply, status = write_prompt(
        prompt=request, request_text=request, model=model, temporary_api_key=gemini_key,
        backend=backend, openai_api_key=openai_key, openai_model=openai_model,
        openai_base_url=openai_url, target=target, system_path=system_path,
        media_values=list(media_values), context="", temperature=0.3,
    )
    if status.startswith("Prompt enhancement failed"):
        raise H3Error(status)
    return reply


def generate_reply(shots, token, writer, runtime):
    return _invoke_writer(request_text(shots, token), writer, prompt_path(runtime), "Sound Stage effects")


def generate_shot_reply(shot, token, writer, runtime, *, media_values=(), can_add_regions=True, user_direction=""):
    request = shot_request_text(shot, token, can_add_regions=can_add_regions, user_direction=user_direction)
    return _invoke_writer(
        request, writer, shot_prompt_path(runtime), "Sound Stage shot effects", media_values=media_values
    )


def _unit_number(value, message):
    if type(value) not in (int, float) or not isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise H3Error(message)
    return round(float(value), 4)


def _check_points(raw, field_name):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_REGION_POINTS:
        raise H3Error(f"New region {field_name} must list up to {MAX_REGION_POINTS} [x, y] points.")
    points = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise H3Error(f"New region {field_name} points must be [x, y] coordinates in 0–1.")
        points.append([
            _unit_number(item[0], f"New region {field_name} coordinates must be between 0 and 1."),
            _unit_number(item[1], f"New region {field_name} coordinates must be between 0 and 1."),
        ])
    return points


def _check_box(raw):
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 4:
        raise H3Error("New region box must be [x0, y0, x1, y1] in 0–1 coordinates.")
    x0, y0, x1, y1 = (_unit_number(v, "New region box coordinates must be between 0 and 1.") for v in raw)
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        raise H3Error("New region box must have positive width and height.")
    return [x0, y0, x1, y1]


def check_new_regions(value, existing_ids, *, max_new=MAX_NEW_REGIONS):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > max_new:
        raise H3Error(f"A shot enhancement can propose at most {max_new} new regions.")
    seen = set(existing_ids)
    cleaned = []
    for entry in value:
        if not isinstance(entry, dict):
            raise H3Error("Each new region must be an object.")
        rid = entry.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise H3Error("Each new region requires a non-empty string id.")
        rid = rid.strip()
        if rid in seen:
            raise H3Error(f"Duplicate or conflicting region id: {rid}.")
        seen.add(rid)

        raw_label = entry.get("label")
        if raw_label is not None and not isinstance(raw_label, str):
            raise H3Error("New region label must be text.")
        raw_prompt = entry.get("prompt")
        if raw_prompt is not None and not isinstance(raw_prompt, str):
            raise H3Error("New region prompt must be text.")

        label_clean = _normalize_text(raw_label, 48)
        prompt_clean = _normalize_text(raw_prompt, 64)
        if not prompt_clean and label_clean.isascii():
            prompt_clean = label_clean
        box = _check_box(entry.get("box"))
        raw_include = _check_points(entry.get("include"), "include")
        include = [list(pt) for pt in dict.fromkeys((u, v) for u, v in raw_include)]
        exclude = _check_points(entry.get("exclude"), "exclude")
        if len(include) < MIN_INCLUDE_POINTS:
            raise H3Error(
                f"New region {rid} must include at least {MIN_INCLUDE_POINTS} distinct [x, y] points spread across the object."
            )
        if prompt_clean and not prompt_clean.isascii():
            prompt_clean = ""
        merge = entry.get("merge", False)
        if not isinstance(merge, bool):
            raise H3Error("New region merge must be true or false.")
        spec = {
            "id": rid,
            "label": label_clean or prompt_clean or rid,
            "prompt": prompt_clean.lower(),
            "include": include,
            "exclude": exclude,
            "merge": merge,
        }
        if box is not None:
            spec["box"] = box
        cleaned.append(spec)
    return cleaned


def check_shot_reply(reply, shot, token):
    parsed = _reply_json(reply)
    if not isinstance(parsed, dict):
        raise H3Error("The shot effects reply must be a JSON object.")
    if parsed.get("revision") != token:
        raise H3Error("This reply is for an older effects prompt. Copy the current prompt and try again.")
    shot_id = shot["shot_id"]
    if isinstance(parsed.get("shots"), list):
        matches = [
            entry for entry in parsed["shots"]
            if isinstance(entry, dict) and entry.get("shot_id") == shot_id
        ]
        if len(matches) != 1:
            raise H3Error("Missing or duplicate decision for this shot.")
        item = matches[0]
    else:
        reply_shot = parsed.get("shot_id")
        if reply_shot is not None and reply_shot != shot_id:
            raise H3Error("This reply is for a different shot.")
        item = parsed

    raw_summary = item.get("summary")
    summary = _normalize_text(raw_summary, 240) if isinstance(raw_summary, str) else ""
    action = item.get("action")
    if action == "keep":
        return {
            "shot_id": shot_id,
            "action": "keep",
            "summary": summary,
            "new_regions": [],
            "effects": list(shot["effects"]),
        }
    if action != "replace":
        raise H3Error("Expected a keep or replace decision.")

    existing_ids = {region["id"] for region in shot["regions"]}
    new_regions = check_new_regions(item.get("new_regions"), existing_ids)
    allowed_ids = existing_ids | {region["id"] for region in new_regions}
    stack = check_stack(item.get("effects"), allowed_ids)
    referenced = {rid for layer in stack for rid in layer_region_ids(layer)}
    used_new_regions = [region for region in new_regions if region["id"] in referenced]
    return {
        "shot_id": shot_id,
        "action": "replace",
        "summary": summary,
        "new_regions": used_new_regions,
        "effects": stack,
    }


def _resolved_decision(shot_id, action, *, summary="", effects=None, added_regions=(), warnings=()):
    result = {
        "shot_id": shot_id,
        "action": action,
        "summary": summary,
        "added_regions": list(added_regions),
        "warnings": list(warnings),
    }
    if effects is not None:
        result["effects"] = effects
    return result


def resolve_shot_stack(decision, shot, region_id_map=None, *, extra_warnings=()):
    shot_id = shot["shot_id"]
    summary = decision.get("summary", "")
    warnings = [w for w in extra_warnings if w]
    if decision["action"] == "keep":
        return _resolved_decision(shot_id, "keep", summary=summary, warnings=warnings)
    existing_ids = {region["id"] for region in shot["regions"]}
    mapping = dict(region_id_map or {})
    labels_by_new = {r["id"]: r["label"] for r in decision.get("new_regions", ())}
    remapped = []
    dropped_labels = []
    for layer in decision["effects"]:
        if layer.get("mode", FULL_FRAME) == FULL_FRAME:
            remapped.append(dict(layer))
            continue
        resolved_ids = []
        for rid in layer_region_ids(layer):
            if rid in existing_ids:
                resolved_ids.append(rid)
            elif rid in mapping:
                resolved_ids.append(mapping[rid])
            else:
                label = labels_by_new.get(rid, rid)
                if label not in dropped_labels:
                    dropped_labels.append(label)
        resolved_ids = list(dict.fromkeys(resolved_ids))
        if not resolved_ids:
            continue
        target = {"region_id": resolved_ids[0]} if len(resolved_ids) == 1 else {"region_ids": resolved_ids}
        remapped.append({k: v for k, v in layer.items() if k not in ("region_id", "region_ids")} | target)

    if dropped_labels and not warnings:
        warnings.append(f"Skipped unsegmented target(s): {', '.join(dropped_labels)}.")
    clean_stack = check_stack(remapped, existing_ids | set(mapping.values()))
    added_regions = list(dict.fromkeys(
        target_id for target_id in mapping.values() if target_id not in existing_ids
    ))
    if clean_stack == shot["effects"] and not added_regions:
        return _resolved_decision(shot_id, "keep", summary=summary, warnings=warnings)
    return _resolved_decision(
        shot_id, "replace", summary=summary, effects=clean_stack, added_regions=added_regions, warnings=warnings
    )


def check_reply(reply, shots, token):
    parsed = _reply_json(reply)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("shots"), list):
        raise H3Error("The effects reply must be a JSON object with a shots list.")
    if parsed.get("revision") != token:
        raise H3Error("This reply is for an older effects prompt. Copy the current prompt and try again.")
    by_id = {}
    for item in parsed["shots"]:
        if isinstance(item, dict) and isinstance(item.get("shot_id"), str):
            by_id.setdefault(item["shot_id"], []).append(item)
    decisions = []
    for shot in shots:
        shot_id = shot["shot_id"]
        entries = by_id.get(shot_id, [])
        try:
            if len(entries) != 1:
                raise H3Error("Missing or duplicate shot decision.")
            item = entries[0]
            if item.get("action") == "keep":
                decisions.append({"shot_id": shot_id, "action": "keep"})
                continue
            if item.get("action") != "replace":
                raise H3Error("Expected a keep or replace decision.")
            stack = check_stack(item.get("effects"), {region["id"] for region in shot["regions"]})
            if stack == shot["effects"]:
                decisions.append({"shot_id": shot_id, "action": "keep"})
            else:
                decisions.append({"shot_id": shot_id, "action": "replace", "effects": stack})
        except H3Error as exc:
            decisions.append({"shot_id": shot_id, "action": "skip", "reason": str(exc)})
    return decisions
