"""The optional effects pass over the current drawn shot plan."""

from urllib.parse import quote

from h3_app.drama3.effect_enhancement import FULL_FRAME, base_stack, layer_region_ids, revision
from h3_app.drama3.serving.planner import spoken_text
from . import effect_regions
from .visuals import stills_dir


def _stack_regions_valid(stack, region_ids):
    for layer in stack:
        if layer.get("mode", FULL_FRAME) == FULL_FRAME:
            continue
        ids = layer_region_ids(layer)
        if not ids or any(rid not in region_ids for rid in ids):
            return False
    return True


def inventory(audio, root, rows, subjects, data, *, speech=()):
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
            replacements = repaired.get("regions", {})
            regions = [replacement if (replacement := replacements.get(region["id"])) and
                       (folder / effect_regions.MASK_FOLDER / replacement["mask"]).is_file() else region
                       for region in regions]
        region_ids = {region["id"] for region in regions or ()}
        stack = entry.get("effects")
        if stack is not None and not _stack_regions_valid(stack, region_ids):
            stack = None
        items.append({
            "shot_id": shot_id, "index": row["index"], "image": image,
            "start": row["start"], "end": row["end"], "beat": row["beat"],
            "speech": [{"speaker": cue.get("speaker", ""), "text": spoken_text(cue["text"])}
                       for cue in speech if cue["start"] < row["end"] and row["start"] < cue["end"]],
            "labels": labels, "regions": regions, "default": base_stack(row["effects"]),
            "applied": stack, "manual": bool(entry.get("manual") and stack is not None),
        })
    return items


def prompt_input(items, sounds, include_manual=False):
    shots = []
    for item in items:
        if not item["regions"] or (item["manual"] and not include_manual):
            continue
        shots.append({"shot_id": item["shot_id"], "beat": item["beat"], "speech": item["speech"],
                      "effects": item["applied"] if item["applied"] is not None else item["default"],
                      "sounds": [sound["name"] for sound in sounds
                                 if sound["start"] is not None and item["start"] <= sound["start"] < item["end"]],
                      "regions": [effect_regions.describe(region) for region in item["regions"]]})
    return shots, revision([items, sounds, include_manual])


def region_url(project, mask):
    return f"/soundstage/api/video/effect-mask?name={quote(project)}&file={quote(mask)}"


def layer_payload(project, item, stack):
    masks = {region["id"]: region["mask"] for region in item["regions"] or ()}
    payload = []
    for layer in stack:
        ids = layer_region_ids(layer)
        if len(ids) > 1:
            payload.append({
                **layer,
                "mask_files": [masks[rid] for rid in ids],
                "mask_urls": [region_url(project, masks[rid]) for rid in ids],
            })
        elif len(ids) == 1:
            payload.append({
                **layer,
                "mask_file": masks[ids[0]],
                "mask_url": region_url(project, masks[ids[0]]),
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
