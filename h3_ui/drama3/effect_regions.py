"""Create and reuse source-sized SAM 3.1 masks for finished Sound Stage stills."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image, ImageChops

from h3_app.drama3.effect_enhancement import revision
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.jobs import check_cancelled
from h3_models import MODEL_SPECS

MODEL = "soundstage_sam31"
MASK_FOLDER = ".effect-masks"
OBJECTS = ("window", "door", "lamp", "candle", "screen", "mirror", "fire", "tree",
           "vehicle", "sign", "table", "chair", "book", "cup", "microphone", "telephone")
MAX_REGIONS = 12


def concepts(row, subjects):
    descriptions = []
    for key in row.get("subjects", []):
        subject = subjects.get(key)
        if subject:
            words = subject["anchor"].split(",")
            name = key.replace("_", " ").casefold()
            phrase = next((part.strip() for part in words if part.strip() and part.strip().casefold() != name), "")
            if phrase:
                descriptions.append(" ".join(phrase.split()[:8]).lower())
            if subject["kind"] == "props":
                descriptions.append(key.replace("_", " ").lower())
    text = " ".join((row.get("prompt", ""), row.get("beat", ""),
                     *(subjects[key]["anchor"] for key in row.get("subjects", []) if key in subjects))).lower()
    detected_words = [label for label in OBJECTS if re.search(rf"\b{label}(?:s|es)?\b", text)]
    return list(dict.fromkeys([*descriptions, "person", *detected_words, "window", "door", "lamp", "screen"]))[:12]


def manifest_path(folder, image, labels):
    key = revision([MODEL_SPECS[MODEL].expected_sha256, 1, labels])[:16]
    return folder / MASK_FOLDER / f"{image}.{key}.json"


def read_regions(folder, image, labels):
    path = manifest_path(folder, image, labels)
    try:
        regions = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(regions, list) and all((path.parent / region["mask"]).is_file() for region in regions):
            return regions
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare(services):
    required = {"SAM3_Detect", "CheckpointLoaderSimple", "MaskToImage", "SaveImage"}
    if missing := required.difference(services.execution.object_info()):
        raise H3Error(f"Update ComfyUI for SAM 3.1. Missing nodes: {', '.join(sorted(missing))}.")
    services.models.ensure_model_keys((MODEL,))


def _graph(image, labels, prefix, clicks=None):
    graph = Graph()
    source = graph.add("LoadImage", image=image)
    checkpoint = graph.add("CheckpointLoaderSimple", ckpt_name=MODEL_SPECS[MODEL].local_name)
    outputs = {}
    for index, label in enumerate(labels):
        inputs = {}
        if clicks is None:
            text = graph.add("CLIPTextEncode", clip=Graph.out(checkpoint, 1), text=label)
            inputs["conditioning"] = Graph.out(text)
        else:
            inputs["positive_coords"] = json.dumps(clicks["include"])
            inputs["negative_coords"] = json.dumps(clicks["exclude"])
        mask = graph.add("SAM3_Detect", model=Graph.out(checkpoint), image=Graph.out(source),
                         individual_masks=clicks is None, refine_iterations=2, threshold=0.5, **inputs)
        pixels = graph.add("MaskToImage", mask=Graph.out(mask))
        output = graph.add("SaveImage", images=Graph.out(pixels), filename_prefix=f"{prefix}/{index}")
        outputs[output] = label
    return graph.nodes, outputs


def _same_mask(first, second):
    overlap = ImageChops.multiply(first, second).histogram()[255]
    union = ImageChops.lighter(first, second).histogram()[255]
    return bool(union and overlap / union >= 0.8)


def segment(path, labels, services, runtime, *, clicks=None):
    """Commit a complete manifest only after all of its masks have been written."""
    folder = path.parent
    manifest = manifest_path(folder, path.name, labels)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    client = uuid4().hex
    output_folder = runtime.output_dir / "h3" / "soundstage_masks" / client
    staged = services.media.stage_file(str(path), "soundstage_masks", reuse=True)
    graph, outputs = _graph(staged, labels, f"h3/soundstage_masks/{client}", clicks)
    try:
        prompt = services.execution.submit_prompt(graph, client)
        for _ in services.execution.poll_comfy_progress(prompt, graph):
            check_cancelled()
        history = services.execution.wait_for_history(prompt)
        with Image.open(path) as source:
            width, height = source.size
        regions, seen = [], []
        with TemporaryDirectory(dir=manifest.parent) as temporary:
            scratch = Path(temporary)
            for node, label in outputs.items():
                for asset in (history.get("outputs", {}).get(node) or {}).get("images", []):
                    name = Path(asset["filename"]).name
                    with Image.open(output_folder / name) as image:
                        mask = image.convert("L")
                    box = mask.point(lambda n: 255 if n >= 128 else 0).getbbox()
                    if not box or (box[2] - box[0]) * (box[3] - box[1]) < width * height * 0.0005:
                        continue
                    small = mask.resize((96, 96)).point(lambda n: 255 if n >= 128 else 0)
                    if any(_same_mask(small, other) for other in seen):
                        continue
                    seen.append(small)
                    region_id = f"region_{len(regions) + 1}"
                    filename = f"{path.name}.{client}.{region_id}.png"
                    alpha = Image.new("RGBA", mask.size, (255, 255, 255))
                    alpha.putalpha(mask)
                    alpha.save(scratch / filename)
                    regions.append({"id": region_id, "label": label, "mask": filename,
                                    "box": [round(n / d, 4) for n, d in zip(box, (width, height, width, height))]})
                    if len(regions) >= MAX_REGIONS:
                        break
                if len(regions) >= MAX_REGIONS:
                    break
            check_cancelled()
            for region in regions:
                (scratch / region["mask"]).replace(manifest.parent / region["mask"])
            if clicks is None:
                saved = scratch / "regions.json"
                saved.write_text(json.dumps(regions), encoding="utf-8")
                saved.replace(manifest)
        return regions
    finally:
        if output_folder.is_dir():
            for asset in output_folder.glob("*.png"):
                asset.unlink(missing_ok=True)
            output_folder.rmdir()


def describe(region):
    x0, y0, x1, y1 = region["box"]
    column = ("left", "center", "right")[min(2, int((x0 + x1) * 1.5))]
    row = ("upper", "middle", "lower")[min(2, int((y0 + y1) * 1.5))]
    return {key: region[key] for key in ("id", "label", "box")} | {"position": f"{row} {column}"}
