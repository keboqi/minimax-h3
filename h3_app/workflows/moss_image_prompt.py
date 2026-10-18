"""Compile a writer's caption or pixel Layout draft into a MOSS-Image prompt.

Ported from OpenMOSS's moss-image-prompt-enhancer skill
(skills/moss-image-prompt-enhancer/scripts/compile_prompt.py at 20f5592, Apache-2.0).
The writer chooses content and absolute xyxy boxes; this module does the centering
and serialization so the model never sees hand-computed coordinates.
"""

import json
import math
import re

MOSS_IMAGE_REWRITE_MODES = ("Auto", "Caption", "Layout")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, allowed, required, location):
    require(isinstance(value, dict), f"{location} must be an object")
    unknown = value.keys() - allowed
    missing = required - value.keys()
    require(not unknown, f"{location}: unknown fields {sorted(unknown)}")
    require(not missing, f"{location}: missing fields {sorted(missing)}")


def text(value, location):
    require(isinstance(value, str) and bool(value.strip()), f"{location} must be nonempty text")


def strings(value, location):
    require(isinstance(value, list), f"{location} must be an array")
    for index, item in enumerate(value):
        text(item, f"{location}[{index}]")


def centered_box(box, width, height, location):
    require(isinstance(box, list) and len(box) == 4, f"{location} must be [x1,y1,x2,y2]")
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in box),
        f"{location} must contain four finite numbers",
    )
    x1, y1, x2, y2 = box
    require(x1 < x2 and y1 < y2, f"{location} must have positive width and height")
    require(
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
        f"{location} exceeds the {width}x{height} canvas",
    )
    # Enclose in pixel edges before centering, as in OpenMOSS's generation export.
    result = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    require(result[0] < result[2] and result[1] < result[3],
            f"{location} is too small at export precision")
    return result


def compile_prompt(draft):
    require(isinstance(draft, dict), "Draft must be a JSON object")
    if "caption" in draft:
        fields(draft, {"caption"}, {"caption"}, "Caption draft")
        text(draft["caption"], "caption")
        return draft["caption"]

    root_fields = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
    fields(draft, root_fields, root_fields, "Layout draft")
    width, height = draft["width"], draft["height"]
    for name, value in (("width", width), ("height", height)):
        require(type(value) is int and value > 0 and value % 16 == 0,
                f"{name} must be a positive integer multiple of 16")
    require(draft["bbox_coord_space"] == "abs", "Draft bbox_coord_space must be abs")
    require(draft["bbox_format"] == "xyxy", "Draft bbox_format must be xyxy")

    summary = draft["summary"]
    fields(summary, {"caption", "meta", "canvas", "palette"}, {"caption"}, "summary")
    text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        require(summary["canvas"] in ("single", "multi_panel", "collage"),
                "summary.canvas must be single, multi_panel or collage")
    for key in ("meta", "palette"):
        if key in summary:
            strings(summary[key], f"summary.{key}")

    items = draft["items"]
    require(isinstance(items, list) and bool(items), "items must be a nonempty flat array")
    compiled_items = []
    for index, item in enumerate(items):
        location = f"items[{index}]"
        fields(item, {"label", "caption", "bbox", "text", "tags", "palette"},
               {"label", "caption", "bbox"}, location)
        require(item["label"] in ("object", "character", "text"),
                f"{location}.label must be object, character or text")
        text(item["caption"], f"{location}.caption")
        if item["label"] == "text":
            require("text" in item, f"{location}: text items require literal text")
        if "text" in item:
            text(item["text"], f"{location}.text")
        for key in ("tags", "palette"):
            if key in item:
                strings(item[key], f"{location}.{key}")
        compiled_items.append({
            key: centered_box(value, width, height, f"{location}.bbox") if key == "bbox" else value
            for key, value in item.items()
        })

    result = {"summary": summary, "items": compiled_items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def compile_writer_draft(reply, width, height, mode="Auto"):
    """Validate a writer reply against the canvas and requested deliverable."""
    reply = str(reply or "").strip()
    fenced = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", reply, re.DOTALL)
    if fenced:
        reply = fenced.group(1).strip()
    try:
        draft = json.loads(reply, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the writer did not return a JSON caption or Layout draft ({exc.msg})") from exc
    layout = isinstance(draft, dict) and "caption" not in draft
    if mode == "Caption":
        require(not layout, "the writer returned a Layout when a caption was requested")
    elif mode == "Layout":
        require(layout, "the writer returned a caption when a Layout was requested")
    if layout:
        require((draft.get("width"), draft.get("height")) == (int(width), int(height)),
                f"the writer's Layout canvas must be {int(width)}x{int(height)}")
    return compile_prompt(draft), layout


def is_layout_prompt(prompt):
    """True for a compiled Layout, which already carries its own boxes."""
    value = str(prompt or "").strip()
    if not value.startswith("{"):
        return False
    try:
        parsed = json.loads(value)
    except ValueError:
        return False
    return isinstance(parsed, dict) and "items" in parsed and "summary" in parsed
