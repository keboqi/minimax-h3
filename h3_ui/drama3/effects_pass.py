"""The optional effects pass over the current drawn shot plan."""

from pathlib import Path
from urllib.parse import quote

from h3_app.drama3.effect_enhancement import FULL_FRAME, base_stack, layer_region_ids, revision
from h3_app.drama3.serving.planner import spoken_text
from . import effect_depth, effect_regions
from .visuals import stills_dir


def _stack_regions_valid(stack, region_ids):
    for layer in stack:
        if layer.get("mode", FULL_FRAME) == FULL_FRAME:
            continue
        ids = layer_region_ids(layer)
        if not ids or any(rid not in region_ids for rid in ids):
            return False
    return True


def _has_mask_file(folder, region):
    return (
        isinstance(region, dict)
        and isinstance(region.get("mask"), str)
        and (folder / effect_regions.MASK_FOLDER / region["mask"]).is_file()
    )


def inventory(audio, root, rows, subjects, data, *, speech=(), depth_model=None):
    saved = data.get("video_effect_stacks") or {}
    fixes = data.get("video_effect_regions") or {}
    subjects = {subject["key"]: subject for subject in subjects}
    folder = stills_dir(audio, root)
    items = []
    for row in rows:
        if not row.get("image_file"):
            continue
        shot_id = row["shot_id"]
        image = row["image_file"]
        labels, person_labels = effect_regions.concepts_with_people(row, subjects)
        regions = effect_regions.read_regions(folder, image, labels)
        entry = saved.get(shot_id) or {}
        if entry.get("image") != image:
            entry = {}
        repaired = fixes.get(shot_id) or {}
        if regions is not None and repaired.get("image") == image:
            replacements = repaired.get("regions") or {}
            seen_ids = set()
            merged = []
            for region in regions:
                rid = region["id"]
                seen_ids.add(rid)
                replacement = replacements.get(rid)
                merged.append(replacement if _has_mask_file(folder, replacement) else region)
            for rid, added in replacements.items():
                if rid not in seen_ids and _has_mask_file(folder, added):
                    merged.append(added)
            regions = merged
        depth_entry = effect_depth.read_depth(folder, image, depth_model)
        if regions is not None and depth_entry is not None:
            enriched = []
            for region in regions:
                stats = effect_depth.region_depth_stats(folder, depth_entry, region.get("mask"))
                enriched.append({**region, **stats} if stats else region)
            regions = enriched
        region_ids = {region["id"] for region in regions or ()}
        stack = entry.get("effects")
        if stack is not None and not _stack_regions_valid(stack, region_ids):
            stack = None
        item = {
            "shot_id": shot_id, "index": row["index"], "image": image,
            "start": row["start"], "end": row["end"], "beat": row["beat"],
            "prompt": row.get("prompt", ""), "transition": row.get("transition", "Cut"),
            "subjects_info": [
                {"key": key, "kind": subjects[key]["kind"], "description": subjects[key]["anchor"]}
                for key in row.get("subjects", ()) if key in subjects
            ],
            "speech": [{"speaker": cue.get("speaker", ""), "text": spoken_text(cue["text"]),
                        "start_s": round(max(0.0, cue["start"] - row["start"]), 2),
                        "end_s": round(min(row["end"] - row["start"], cue["end"] - row["start"]), 2)}
                       for cue in speech if cue["start"] < row["end"] and row["start"] < cue["end"]],
            "labels": labels, "person_labels": person_labels, "regions": regions, "default": base_stack(row["effects"]),
            "applied": stack, "manual": bool(entry.get("manual") and stack is not None),
        }
        if depth_entry is not None:
            item["depth_file"] = depth_entry["png"]
            item["depth_meta"] = depth_entry["meta"]
        items.append(item)
    return items


def prompt_input(items, sounds, include_manual=False):
    shots = []
    for item in items:
        if not item["regions"] or (item["manual"] and not include_manual):
            continue
        shot = {"shot_id": item["shot_id"], "beat": item["beat"],
                "speech": [{"speaker": s["speaker"], "text": s["text"]} for s in item["speech"]],
                "effects": item["applied"] if item["applied"] is not None else item["default"],
                "sounds": [sound["name"] for sound in sounds
                           if sound["start"] is not None and item["start"] <= sound["start"] < item["end"]],
                "regions": [effect_regions.describe(region) for region in item["regions"]]}
        if item.get("depth_meta"):
            meta = item["depth_meta"]
            shot["depth_summary"] = {
                "subject_m": meta.get("subject_z"),
                "range_m": [meta.get("z_p10"), meta.get("z_p90")],
            }
        shots.append(shot)
    return shots, revision([items, sounds, include_manual])


def _neighbor_summary(item):
    return {
        "shot_id": item["shot_id"],
        "beat": item["beat"],
        "transition": str(item.get("transition") or "Cut").lower(),
        "effects": item["applied"] if item["applied"] is not None else item["default"],
    }


def shot_prompt_input(
    items,
    sounds,
    shot_id,
    *,
    style_preset="",
    art_style="",
    orientation="landscape",
    draft_effects=None,
    user_direction="",
):
    pos = next((idx for idx, entry in enumerate(items) if entry["shot_id"] == shot_id), None)
    if pos is None:
        return None, None
    item = items[pos]
    current_effects = (
        draft_effects
        if draft_effects is not None
        else (item["applied"] if item["applied"] is not None else item["default"])
    )
    shot_sounds = []
    for sound in sounds:
        s_start = sound.get("start")
        if s_start is None:
            continue
        s_end = sound.get("end") if sound.get("end") is not None else s_start
        is_loop = bool(sound.get("loop"))
        overlaps = (s_start < item["end"] and s_end > item["start"]) if is_loop else (item["start"] <= s_start < item["end"])
        if not overlaps:
            continue
        cue_entry = {
            "name": sound["name"],
            "loop": is_loop,
            "onset_s": round(max(0.0, float(s_start) - item["start"]), 2),
        }
        if sound.get("description"):
            cue_entry["description"] = sound["description"]
        shot_sounds.append(cue_entry)

    shot = {
        "shot_id": item["shot_id"],
        "shot_number": item["index"] + 1,
        "duration_s": round(max(0.05, item["end"] - item["start"]), 2),
        "transition": str(item.get("transition") or "Cut").lower(),
        "orientation": orientation,
        **({"style_preset": style_preset} if style_preset else {}),
        **({"art_style": art_style} if art_style else {}),
        "beat": item["beat"],
        "image_prompt": item.get("prompt", ""),
        **({"subjects": item["subjects_info"]} if item.get("subjects_info") else {}),
        "speech": item["speech"],
        "sounds": shot_sounds,
        "effects": current_effects,
        "segmented": item["regions"] is not None,
        "regions": [effect_regions.describe(region) for region in item["regions"] or ()],
    }
    if item.get("depth_meta"):
        meta = item["depth_meta"]
        depth_summary = {
            "subject_m": meta.get("subject_z"),
            "range_m": [meta.get("z_p10"), meta.get("z_p90")],
            "sky_fraction": meta.get("sky_fraction"),
            "auto_strength": meta.get("auto_strength"),
        }
        if meta.get("warm_anchor"):
            depth_summary["warm_anchor"] = meta["warm_anchor"]
        if meta.get("sky_dir"):
            depth_summary["sky_dir"] = meta["sky_dir"]
        shot["depth_summary"] = depth_summary

    neighbors = {}
    if pos > 0:
        neighbors["previous_shot"] = _neighbor_summary(items[pos - 1])
    if pos + 1 < len(items):
        neighbors["next_shot"] = _neighbor_summary(items[pos + 1])
    if neighbors:
        shot["neighbors"] = neighbors

    direction = " ".join(str(user_direction or "").split())[:400]
    return shot, revision([shot, direction])


def shot_media_inputs(folder, item, scratch_dir):
    folder = Path(folder)
    scratch = Path(scratch_dir)
    media = []
    still_path = folder / item["image"]
    if still_path.is_file():
        media.append(("<Shot Image>", still_path))
    if item.get("depth_file"):
        depth_vis = effect_depth.render_depth_preview(
            folder,
            {"png": item["depth_file"], "meta": item.get("depth_meta") or {}},
            scratch / "depth_map.png",
        )
        if depth_vis is not None:
            media.append((
                "<Depth Map> (bright warm yellow/orange = near foreground, crimson/purple = midground, dark indigo = far background/sky)",
                depth_vis,
            ))
    if item.get("regions"):
        seg_vis = effect_regions.render_regions_preview(
            still_path,
            folder / effect_regions.MASK_FOLDER,
            item["regions"],
            scratch / "segmented_regions.png",
        )
        if seg_vis is not None:
            media.append((
                "<Segmented Regions Overlay> (colored masks and [region_id: label] badges for existing segments)",
                seg_vis,
            ))
    return media


def region_url(project, mask):
    return f"/soundstage/api/video/effect-mask?name={quote(project)}&file={quote(mask)}"


def layer_payload(project, item, stack):
    masks = {region["id"]: region["mask"] for region in item["regions"] or ()}
    depths = {region["id"]: region["depth_m"] for region in item["regions"] or () if region.get("depth_m") is not None}
    payload = []
    for layer in stack:
        ids = layer_region_ids(layer)
        depth_vals = [depths[rid] for rid in ids if rid in depths]
        depth_extra = {"region_depth_m": min(depth_vals)} if depth_vals else {}
        if len(ids) > 1:
            payload.append({
                **layer,
                "mask_files": [masks[rid] for rid in ids],
                "mask_urls": [region_url(project, masks[rid]) for rid in ids],
                **depth_extra,
            })
        elif len(ids) == 1:
            payload.append({
                **layer,
                "mask_file": masks[ids[0]],
                "mask_url": region_url(project, masks[ids[0]]),
                **depth_extra,
            })
        else:
            payload.append(layer)
    return payload


def public_pass(project, items, sounds, data):
    proposal = data.get("video_effect_proposal")
    if proposal:
        _, token = prompt_input(items, sounds, proposal["include_manual"])
        if proposal["revision"] != token:
            proposal = None
        else:
            by_id = {item["shot_id"]: item for item in items}
            proposal = {**proposal, "decisions": [
                {**decision, "layers": layer_payload(project, by_id[decision["shot_id"]], decision["effects"])}
                if decision["action"] == "replace" else decision for decision in proposal["decisions"]
            ]}
    shot_proposals = data.get("video_effect_shot_proposals") or {}

    def format_shot_proposal(item):
        sp = shot_proposals.get(item["shot_id"])
        if not isinstance(sp, dict) or sp.get("image") != item["image"]:
            return None
        action = sp.get("action")
        if action not in ("replace", "keep"):
            return None
        base = {
            "id": sp.get("id", ""),
            "action": action,
            "summary": sp.get("summary", ""),
            "added_regions": sp.get("added_regions") or [],
            "warnings": sp.get("warnings") or [],
        }
        if action == "keep":
            return base
        effects_list = sp.get("effects")
        region_ids = {region["id"] for region in item["regions"] or ()}
        if not isinstance(effects_list, list) or not _stack_regions_valid(effects_list, region_ids):
            return None
        return {
            **base,
            "effects": effects_list,
            "layers": layer_payload(project, item, effects_list),
        }

    return {
        "shots": [{"shot_id": item["shot_id"], "index": item["index"], "image": item["image"],
                   "segmented": item["regions"] is not None, "effects": item["applied"],
                   "manual": item["manual"], "default": item["default"],
                   "proposal": format_shot_proposal(item),
                   "regions": [{**effect_regions.describe(region), "url": region_url(project, region["mask"])}
                               for region in item["regions"] or ()]} for item in items],
        "proposal": proposal,
    }
