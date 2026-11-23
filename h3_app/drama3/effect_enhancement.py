"""Text contract for enhancing the effects of already drawn Sound Stage shots."""

from hashlib import sha256
import json
from math import isfinite

from h3_app.errors import H3Error
from h3_app.prompt_writers.common import read_system_prompt, write_prompt
from .visual_effects import EFFECT_PRESETS
from .visual_plan import _reply_json

MAX_LAYERS = 16
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
        "layer_fields": {
            "preset_or_presets": "single preset ID (`preset`) or ordered list (`presets`) sharing the same mode, target, intensity, and feather",
            "mode": "full_frame, inside, outside, or behind",
            "region_id_or_region_ids": "required when mode != full_frame; single ID (`region_id`) or list of IDs (`region_ids`) combined into one mask",
            "intensity": "optional 0–1 for non-camera presets",
            "feather": "optional 0–32 pixels for masked non-camera presets",
        },
        "shots": shots,
    }, ensure_ascii=False)


def prompt_path(runtime):
    return runtime.script_dir / "prompts" / "prompt_soundstage_effects.txt"


def copy_prompt(shots, token, runtime):
    return f"{read_system_prompt(prompt_path(runtime), 'Sound Stage effects')}\n\n{request_text(shots, token)}"


def generate_reply(shots, token, writer, runtime):
    model, gemini_key, backend, openai_key, openai_model, openai_url = writer
    request = request_text(shots, token)
    reply, status = write_prompt(
        prompt=request, request_text=request, model=model, temporary_api_key=gemini_key,
        backend=backend, openai_api_key=openai_key, openai_model=openai_model,
        openai_base_url=openai_url, target="Sound Stage effects", system_path=prompt_path(runtime),
        media_values=[], context="", temperature=0.3,
    )
    if status.startswith("Prompt enhancement failed"):
        raise H3Error(status)
    return reply


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
