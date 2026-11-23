"""Create and reuse source-sized SAM 3.1 masks for finished Sound Stage stills."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image, ImageChops

from h3_app.drama3 import voice_looks
from h3_app.drama3.effect_enhancement import revision
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.jobs import check_cancelled
from h3_models import MODEL_SPECS

MODEL = "soundstage_sam31"
MASK_OUTPUT_NODE = "H3SaveDetectedMasks"
MASK_FOLDER = ".effect-masks"
OBJECTS = ("window", "door", "lamp", "candle", "screen", "mirror", "fire", "tree",
           "vehicle", "sign", "table", "chair", "book", "cup", "microphone", "telephone")
_CJK_OBJECT_PATTERNS = (
    ("window", r"窗户|窗子|舷窗|窗框|落地窗|天窗|窗外|窗前|窗边|窗台"),
    ("door", r"大门|木门|铁门|舱门|房门|车门|拱门|门口|门框|门帘"),
    ("lamp", r"灯笼|台灯|路灯|壁灯|吊灯|油灯|提灯|汽灯|霓虹灯|探照灯|车灯|灯光"),
    ("candle", r"蜡烛|烛火|烛台|烛光"),
    ("screen", r"屏幕|显示器|监视器|荧光屏|电脑屏|手机屏|大屏"),
    ("mirror", r"镜子|铜镜|后视镜|穿衣镜|玻璃镜"),
    ("fire", r"火焰|篝火|炉火|炭火|火把|火苗|火光|壁炉|火盆"),
    ("tree", r"大树|古树|松树|柳树|竹林|树林|树枝|树干|树冠"),
    ("vehicle", r"汽车|轿车|卡车|列车|火车|电车|船只|渔船|木船|潜艇|飞船|马车"),
    ("sign", r"招牌|路牌|指示牌|告示牌|灯牌|站牌"),
    ("table", r"桌子|木桌|书桌|餐桌|茶几|案几|柜台|讲台|工作台"),
    ("chair", r"椅子|木椅|长椅|扶手椅|座椅|板凳"),
    ("book", r"书本|古籍|书籍|卷宗|笔记本|日记本|账本|书册"),
    ("cup", r"茶杯|酒杯|水杯|咖啡杯|瓷杯|玻璃杯|茶碗"),
    ("microphone", r"麦克风|话筒|广播话筒"),
    ("telephone", r"电话|听筒|话机|对讲机|步话机"),
)
MERGE_LABELS = frozenset({"window", "lamp", "candle", "fire", "tree"})
MAX_REGIONS = 12


def concepts(row, subjects):
    descriptions = []
    for key in row.get("subjects", []):
        subject = subjects.get(key)
        if not subject or subject.get("kind") == "locations":
            continue
        if subject.get("kind") == "props":
            descriptions.append(key.replace("_", " ").lower())
        else:
            anchor = str(subject.get("anchor") or "")
            words = anchor.split(",")
            name = key.replace("_", " ").casefold()
            phrase = next((part.strip() for part in words if part.strip() and part.strip().casefold() != name), "")
            if phrase and phrase.isascii():
                descriptions.append(phrase.lower())
            elif phrase:
                sex = voice_looks.gender(anchor)
                if sex == "female":
                    descriptions.append("woman")
                elif sex == "male":
                    descriptions.append("man")
    raw_text = " ".join((row.get("prompt", ""), row.get("beat", ""),
                         *(subjects[key]["anchor"] for key in row.get("subjects", []) if key in subjects)))
    text = raw_text.lower()
    detected_words = [
        label for label in OBJECTS
        if re.search(rf"\b{label}(?:s|es)?\b", text)
        or any(target == label and re.search(pattern, raw_text) for target, pattern in _CJK_OBJECT_PATTERNS)
    ]
    return list(dict.fromkeys([*descriptions, "person", *detected_words, "window", "door", "lamp"]))[:MAX_REGIONS]


def manifest_path(folder, image, labels):
    key = revision([MODEL_SPECS[MODEL].expected_sha256, 1, labels])[:16]
    return folder / MASK_FOLDER / f"{image}.{key}.json"


def _load_manifest(path: Path) -> list | None:
    try:
        regions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(regions, list):
        return None
    if all(
        isinstance(region, dict)
        and isinstance(region.get("mask"), str)
        and (path.parent / region["mask"]).is_file()
        for region in regions
    ):
        return regions
    return None


def _manifest_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (path.stat().st_mtime_ns, path.name)
    except OSError:
        return (-1, path.name)


def read_regions(folder, image, labels):
    exact = manifest_path(folder, image, labels)
    if (regions := _load_manifest(exact)) is not None:
        return regions
    mask_dir = Path(folder) / MASK_FOLDER
    if not mask_dir.is_dir():
        return None
    pattern = re.compile(rf"{re.escape(str(image))}\.[0-9a-f]{{16}}\.json")
    try:
        candidates = [
            entry
            for entry in mask_dir.iterdir()
            if entry != exact and pattern.fullmatch(entry.name) and entry.is_file()
        ]
    except OSError:
        return None
    for candidate in sorted(candidates, key=_manifest_sort_key, reverse=True):
        if (regions := _load_manifest(candidate)) is not None:
            return regions
    return None


def prepare(services):
    required = {"SAM3_Detect", "CheckpointLoaderSimple", MASK_OUTPUT_NODE}
    if missing := required.difference(services.execution.object_info()):
        raise H3Error(f"Update ComfyUI and the bundled H3 nodes for SAM 3.1. Missing nodes: {', '.join(sorted(missing))}.")
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
        output = graph.add(MASK_OUTPUT_NODE, masks=Graph.out(mask), filename_prefix=f"{prefix}/{index}")
        outputs[output] = label
    return graph.nodes, outputs


def _same_mask(first, second):
    overlap = ImageChops.multiply(first, second).histogram()[255]
    union = ImageChops.lighter(first, second).histogram()[255]
    return bool(union and overlap / union >= 0.8)


def _valid_mask(mask, total_pixels, filter_full_frame):
    box = mask.point(lambda n: 255 if n >= 128 else 0).getbbox()
    if not box:
        return None
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < total_pixels * 0.0005 or (filter_full_frame and area >= total_pixels * 0.96):
        return None
    return box


def _collect_masks(assets, output_folder, total_pixels, *, merge, filter_full_frame):
    masks = []
    for asset in assets:
        name = Path(asset["filename"]).name
        with Image.open(output_folder / name) as image:
            mask = image.convert("L")
        if _valid_mask(mask, total_pixels, filter_full_frame):
            masks.append(mask)
    if merge and len(masks) > 1:
        combined = masks[0]
        for extra in masks[1:]:
            combined = ImageChops.lighter(combined, extra)
        return [combined]
    return masks


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
                assets = (history.get("outputs", {}).get(node) or {}).get("images", [])
                for mask in _collect_masks(assets, output_folder, width * height,
                                           merge=clicks is None and label in MERGE_LABELS,
                                           filter_full_frame=clicks is None):
                    box = _valid_mask(mask, width * height, clicks is None)
                    if not box:
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


def describe(region, depth=None):
    x0, y0, x1, y1 = region["box"]
    column = ("left", "center", "right")[min(2, int((x0 + x1) * 1.5))]
    row = ("upper", "middle", "lower")[min(2, int((y0 + y1) * 1.5))]
    described = {key: region[key] for key in ("id", "label", "box")} | {"position": f"{row} {column}"}
    if depth is None:
        for key in ("depth_m", "depth_band"):
            if key in region:
                described[key] = region[key]
    else:
        described.update(depth)
    return described
