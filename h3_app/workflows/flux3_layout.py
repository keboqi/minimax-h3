"""Validated Comfy FLUX layout elements in top/left/bottom/right order."""

import json
import math


def parse_flux3_layout(value):
    if not value or not str(value).strip():
        return []
    try:
        elements = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("FLUX layout must be a JSON list of boxes.") from exc
    if not isinstance(elements, list) or len(elements) > 100:
        raise ValueError("FLUX layout must contain at most 100 boxes.")
    normalized = []
    for i, element in enumerate(elements, 1):
        if not isinstance(element, dict) or set(element) - {"bbox", "desc", "type", "text"}:
            raise ValueError(f"Layout box {i} accepts bbox, desc, type and text only.")
        bbox = element.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1000 for v in bbox)):
            raise ValueError(f"Layout box {i} needs bbox [top, left, bottom, right] on the 0–1000 grid.")
        top, left, bottom, right = bbox
        if top >= bottom or left >= right:
            raise ValueError(f"Layout box {i} must have positive width and height.")
        desc, kind, text = element.get("desc", ""), element.get("type", "obj"), element.get("text", "")
        if (not isinstance(desc, str) or not desc.strip() or not isinstance(kind, str)
                or kind not in {"obj", "text"} or not isinstance(text, str)):
            raise ValueError(f"Layout box {i} needs a description and type obj or text.")
        normalized.append({"bbox": bbox, "desc": desc.strip(), "type": kind, "text": text})
    return normalized
