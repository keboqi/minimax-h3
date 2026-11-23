"""The optional effects pass over the current drawn shot plan."""

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
        labels = effect_regions.concepts(row, subjects)
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
            "speech": [{"speaker": cue.get("speaker", ""), "text": spoken_text(cue["text"])}
                       for cue in speech if cue["start"] < row["end"] and row["start"] < cue["end"]],
            "labels": labels, "regions": regions, "default": base_stack(row["effects"]),
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
        shot = {"shot_id": item["shot_id"], "beat": item["beat"], "speech": item["speech"],
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
    return {
        "shots": [{"shot_id": item["shot_id"], "index": item["index"], "image": item["image"],
                   "segmented": item["regions"] is not None, "effects": item["applied"],
                   "manual": item["manual"], "default": item["default"],
                   "regions": [{**effect_regions.describe(region), "url": region_url(project, region["mask"])}
                               for region in item["regions"] or ()]} for item in items],
        "proposal": proposal,
    }
