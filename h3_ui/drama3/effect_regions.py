"""Create and reuse source-sized SAM 3.1 masks for finished Sound Stage stills."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image, ImageChops, ImageDraw, ImageFilter

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


def concepts_with_people(row, subjects):
    descriptions = []
    people = {"person", "woman", "man"}
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
                people.add(phrase.lower())
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
    labels = list(dict.fromkeys([*descriptions, "person", *detected_words, "window", "door", "lamp"]))[:MAX_REGIONS]
    return labels, sorted(people)


def concepts(row, subjects):
    return concepts_with_people(row, subjects)[0]


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


def _norm_box(pixel_box, width: int, height: int) -> list[float]:
    return [round(n / d, 4) for n, d in zip(pixel_box, (width, height, width, height))]


def _save_rgba_mask(mask: Image.Image, dest: Path) -> None:
    alpha = Image.new("RGBA", mask.size, (255, 255, 255))
    alpha.putalpha(mask)
    alpha.save(dest)


def _read_mask_alpha(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            return (img.convert("RGBA").getchannel("A") if "A" in img.getbands() else img.convert("L")).copy()
    except OSError:
        return None


def next_custom_id(used_ids: set[str], start: int = 1) -> tuple[str, int]:
    num = max(1, start)
    while f"custom_{num}" in used_ids:
        num += 1
    return f"custom_{num}", num + 1


def _execute_mask_graph(graph_nodes: dict, client: str, output_folder: Path, services, process_outputs):
    try:
        prompt = services.execution.submit_prompt(graph_nodes, client)
        for _ in services.execution.poll_comfy_progress(prompt, graph_nodes):
            check_cancelled()
        history = services.execution.wait_for_history(prompt)
        return process_outputs(history.get("outputs") or {})
    finally:
        if output_folder.is_dir():
            for asset in output_folder.glob("*.png"):
                asset.unlink(missing_ok=True)
            try:
                output_folder.rmdir()
            except OSError:
                pass


def segment(path, labels, services, runtime, *, clicks=None):
    """Commit a complete manifest only after all of its masks have been written."""
    folder = path.parent
    manifest = manifest_path(folder, path.name, labels)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    client = uuid4().hex
    output_folder = Path(runtime.output_dir) / "h3" / "soundstage_masks" / client
    staged = services.media.stage_file(str(path), "soundstage_masks", reuse=True)
    graph, outputs = _graph(staged, labels, f"h3/soundstage_masks/{client}", clicks)

    def build_regions(history_outputs):
        with Image.open(path) as source:
            width, height = source.size
        regions, seen = [], []
        with TemporaryDirectory(dir=manifest.parent) as temporary:
            scratch = Path(temporary)
            for node, label in outputs.items():
                assets = (history_outputs.get(node) or {}).get("images", [])
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
                    _save_rgba_mask(mask, scratch / filename)
                    regions.append({"id": region_id, "label": label, "mask": filename,
                                    "box": _norm_box(box, width, height)})
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

    return _execute_mask_graph(graph, client, output_folder, services, build_regions)


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


SEGMENT_PALETTE = (
    (45, 212, 191),
    (251, 191, 36),
    (248, 113, 113),
    (56, 189, 248),
    (167, 139, 250),
    (74, 222, 128),
    (251, 113, 133),
    (34, 211, 238),
    (250, 204, 21),
    (129, 140, 248),
    (251, 146, 60),
    (232, 121, 249),
)


def _box_iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / max(1e-6, area_a + area_b - inter)


def _box_containment(mask_box_norm, hint_box) -> float:
    mx0, my0, mx1, my1 = mask_box_norm
    hx0, hy0, hx1, hy1 = hint_box
    ix0, iy0 = max(mx0, hx0), max(my0, hy0)
    ix1, iy1 = min(mx1, hx1), min(my1, hy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    mask_area = max(1e-6, (mx1 - mx0) * (my1 - my0))
    return inter / mask_area


def _mask_point_hits(mask, points, width: int, height: int) -> float:
    if not points:
        return 0.0
    hits = 0
    for u, v in points:
        px = min(width - 1, max(0, int(u * width)))
        py = min(height - 1, max(0, int(v * height)))
        if mask.getpixel((px, py)) >= 128:
            hits += 1
    return hits / len(points)


def _select_best_new_mask(text_masks, point_masks, req: dict, width: int, height: int):
    total_pixels = width * height
    include = req.get("include") or []
    exclude = req.get("exclude") or []
    hint_box = req.get("box")
    want_merge = bool(req.get("merge")) or (
        req.get("prompt") in MERGE_LABELS and not include and hint_box is None
    )

    valid_text = []
    for mask in text_masks:
        pbox = _valid_mask(mask, total_pixels, True)
        if pbox:
            valid_text.append((mask, _norm_box(pbox, width, height)))

    valid_point = []
    for mask in point_masks:
        pbox = _valid_mask(mask, total_pixels, True)
        if pbox:
            valid_point.append((mask, _norm_box(pbox, width, height)))

    if want_merge and len(valid_text) > 1:
        matching = [
            mask for mask, nbox in valid_text
            if (hint_box is None or _box_containment(nbox, hint_box) >= 0.2 or _box_iou(nbox, hint_box) >= 0.1)
            and (not exclude or _mask_point_hits(mask, exclude, width, height) < 0.5)
        ]
        pool = matching or [mask for mask, _ in valid_text]
        combined = pool[0]
        for extra in pool[1:]:
            combined = ImageChops.lighter(combined, extra)
        pbox = _valid_mask(combined, total_pixels, True)
        if pbox:
            return combined, _norm_box(pbox, width, height)

    def score_candidate(mask, nbox, *, source_bonus: float) -> float:
        score = source_bonus
        if include:
            score += 2.0 * _mask_point_hits(mask, include, width, height)
        if exclude:
            score -= 2.0 * _mask_point_hits(mask, exclude, width, height)
        if hint_box is not None:
            score += 1.5 * _box_iou(nbox, hint_box) + 0.5 * _box_containment(nbox, hint_box)
        return score

    scored = [
        (score_candidate(mask, nbox, source_bonus=0.25), mask, nbox)
        for mask, nbox in valid_text
    ] + [
        (score_candidate(mask, nbox, source_bonus=0.0), mask, nbox)
        for mask, nbox in valid_point
    ]
    if not scored:
        return None
    scored.sort(key=lambda entry: entry[0], reverse=True)
    best_score, best_mask, best_box = scored[0]
    if (include or hint_box is not None) and best_score < -0.5:
        return None
    return best_mask, best_box


def segment_new_regions(
    path: Path,
    requests: list[dict],
    existing_regions: list[dict] | None,
    services,
    runtime,
    *,
    max_total: int = 24,
) -> tuple[dict[str, dict], dict[str, str], list[str]]:
    """Segment LLM-proposed object regions via hybrid text + point SAM 3.1 in one ComfyUI pass."""
    if not requests:
        return {}, {}, []
    path = Path(path)
    mask_dir = path.parent / MASK_FOLDER
    mask_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as source_img:
        width, height = source_img.size

    existing = list(existing_regions or ())
    existing_small: list[tuple[str, Image.Image]] = []
    for region in existing:
        mask_name = region.get("mask")
        if not isinstance(mask_name, str):
            continue
        alpha = _read_mask_alpha(mask_dir / mask_name)
        if alpha is not None:
            existing_small.append((region["id"], alpha.resize((96, 96)).point(lambda n: 255 if n >= 128 else 0)))

    client = uuid4().hex
    output_folder = Path(runtime.output_dir) / "h3" / "soundstage_masks" / client
    staged = services.media.stage_file(str(path), "soundstage_masks", reuse=True)
    prefix = f"h3/soundstage_masks/{client}"

    graph = Graph()
    source_node = graph.add("LoadImage", image=staged)
    checkpoint = graph.add("CheckpointLoaderSimple", ckpt_name=MODEL_SPECS[MODEL].local_name)
    node_plan: list[tuple[dict, str | None, str | None]] = []

    for index, req in enumerate(requests):
        text_out = None
        prompt_text = str(req.get("prompt") or "").strip()
        if prompt_text and prompt_text.isascii():
            text_enc = graph.add("CLIPTextEncode", clip=Graph.out(checkpoint, 1), text=prompt_text)
            tmask = graph.add(
                "SAM3_Detect",
                model=Graph.out(checkpoint),
                image=Graph.out(source_node),
                individual_masks=True,
                refine_iterations=2,
                threshold=0.45,
                conditioning=Graph.out(text_enc),
            )
            text_out = graph.add(MASK_OUTPUT_NODE, masks=Graph.out(tmask), filename_prefix=f"{prefix}/{index}_txt")

        point_out = None
        inc = list(req.get("include") or ())
        if not inc and req.get("box"):
            bx0, by0, bx1, by1 = req["box"]
            inc = [[(bx0 + bx1) * 0.5, (by0 + by1) * 0.5]]
        if inc:
            pos_px = [
                {"x": min(width - 1, max(0, int(u * width))), "y": min(height - 1, max(0, int(v * height)))}
                for u, v in inc
            ]
            neg_px = [
                {"x": min(width - 1, max(0, int(u * width))), "y": min(height - 1, max(0, int(v * height)))}
                for u, v in (req.get("exclude") or ())
            ]
            pmask = graph.add(
                "SAM3_Detect",
                model=Graph.out(checkpoint),
                image=Graph.out(source_node),
                individual_masks=False,
                refine_iterations=2,
                threshold=0.5,
                positive_coords=json.dumps(pos_px),
                negative_coords=json.dumps(neg_px),
            )
            point_out = graph.add(MASK_OUTPUT_NODE, masks=Graph.out(pmask), filename_prefix=f"{prefix}/{index}_pt")

        node_plan.append((req, text_out, point_out))

    def collect_new_regions(outputs):
        created: dict[str, dict] = {}
        id_map: dict[str, str] = {}
        skipped: list[str] = []
        used_ids = {r["id"] for r in existing}
        next_custom = 1
        total_pixels = width * height
        with TemporaryDirectory(dir=mask_dir) as temporary:
            scratch = Path(temporary)
            staged_files: list[tuple[Path, Path]] = []
            file_seq = 1
            for req, text_out, point_out in node_plan:
                text_assets = (outputs.get(text_out) or {}).get("images", []) if text_out else []
                point_assets = (outputs.get(point_out) or {}).get("images", []) if point_out else []
                text_masks = _collect_masks(
                    text_assets, output_folder, total_pixels, merge=False, filter_full_frame=True
                )
                point_masks = _collect_masks(
                    point_assets, output_folder, total_pixels, merge=False, filter_full_frame=True
                )
                chosen = _select_best_new_mask(text_masks, point_masks, req, width, height)
                if chosen is None:
                    skipped.append(req["label"])
                    continue
                mask, nbox = chosen
                small = mask.resize((96, 96)).point(lambda n: 255 if n >= 128 else 0)
                duplicate_id = next(
                    (rid for rid, other in existing_small if _same_mask(small, other)),
                    None,
                )
                if duplicate_id is not None:
                    id_map[req["id"]] = duplicate_id
                    continue
                if len(existing) + len(created) >= max_total:
                    skipped.append(req["label"])
                    continue
                custom_id, next_custom = next_custom_id(used_ids, next_custom)
                used_ids.add(custom_id)
                filename = f"{path.name}.{client}.region_{file_seq}.png"
                file_seq += 1
                tmp_file = scratch / filename
                _save_rgba_mask(mask, tmp_file)
                staged_files.append((tmp_file, mask_dir / filename))
                created[custom_id] = {
                    "id": custom_id,
                    "label": req["label"],
                    "mask": filename,
                    "box": nbox,
                }
                existing_small.append((custom_id, small))
                id_map[req["id"]] = custom_id

            check_cancelled()
            for tmp_file, dest_file in staged_files:
                tmp_file.replace(dest_file)
        return created, id_map, skipped

    return _execute_mask_graph(graph.nodes, client, output_folder, services, collect_new_regions)


def render_regions_preview(
    still_path: Path | str,
    mask_dir: Path | str,
    regions: list[dict] | None,
    out_path: Path | str,
    *,
    max_side: int = 1024,
) -> Path | None:
    """Render an annotated composite of existing SAM regions (fill + outline + ID/label pill) for VLMs."""
    if not regions:
        return None
    still_path = Path(still_path)
    mask_dir = Path(mask_dir)
    if not still_path.is_file():
        return None
    try:
        with Image.open(still_path) as base_img:
            rgb = base_img.convert("RGB")
    except OSError:
        return None

    longest = max(rgb.size)
    if max_side > 0 and longest > max_side:
        scale = max_side / float(longest)
        size = (max(1, round(rgb.size[0] * scale)), max(1, round(rgb.size[1] * scale)))
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    else:
        size = rgb.size
    width, height = size
    canvas = rgb.point(lambda v: int(v * 0.55)).convert("RGBA")
    badges: list[tuple[int, int, str, tuple[int, int, int]]] = []

    for index, region in enumerate(regions):
        mask_name = region.get("mask")
        if not isinstance(mask_name, str):
            continue
        alpha = _read_mask_alpha(mask_dir / mask_name)
        if alpha is None:
            continue
        if alpha.size != size:
            alpha = alpha.resize(size, Image.Resampling.BILINEAR)
        binary = alpha.point(lambda v: 255 if v >= 128 else 0)
        bbox = binary.getbbox()
        if not bbox:
            continue
        color = SEGMENT_PALETTE[index % len(SEGMENT_PALETTE)]

        fill_layer = Image.new("RGBA", size, (*color, 0))
        fill_layer.putalpha(binary.point(lambda v: 92 if v else 0))
        canvas = Image.alpha_composite(canvas, fill_layer)

        dilated = binary.filter(ImageFilter.MaxFilter(5))
        eroded = binary.filter(ImageFilter.MinFilter(3))
        edge = ImageChops.subtract(dilated, eroded)
        edge_layer = Image.new("RGBA", size, (*color, 0))
        edge_layer.putalpha(edge.point(lambda v: 240 if v else 0))
        canvas = Image.alpha_composite(canvas, edge_layer)

        cx = int((bbox[0] + bbox[2]) * 0.5)
        cy = int((bbox[1] + bbox[3]) * 0.5)
        badges.append((cx, cy, f"{region['id']}: {region.get('label', '')}".strip(": "), color))

    if not badges:
        return None

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for cx, cy, text, color in badges:
        try:
            tbox = draw.textbbox((0, 0), text)
            tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        except AttributeError:
            tw, th = len(text) * 6, 11
        pad_x, pad_y, dot = 6, 4, 8
        bw = dot + 5 + tw + pad_x * 2
        bh = max(th, dot) + pad_y * 2
        bx = min(max(4, cx - bw // 2), max(4, width - bw - 4))
        by = min(max(4, cy - bh // 2), max(4, height - bh - 4))
        for _ in range(6):
            if not any(
                not (bx + bw + 2 < ox0 or bx > ox1 + 2 or by + bh + 2 < oy0 or by > oy1 + 2)
                for ox0, oy0, ox1, oy1 in placed_boxes
            ):
                break
            by = min(max(4, by + bh + 3), max(4, height - bh - 4))
        placed_boxes.append((bx, by, bx + bw, by + bh))
        draw.rectangle((bx, by, bx + bw, by + bh), fill=(12, 18, 16, 224), outline=(*color, 245), width=1)
        dy = by + (bh - dot) // 2
        draw.rectangle((bx + pad_x, dy, bx + pad_x + dot, dy + dot), fill=(*color, 255))
        draw.text((bx + pad_x + dot + 5, by + pad_y - 1), text, fill=(248, 252, 250, 255))

    final_rgb = Image.alpha_composite(canvas, overlay).convert("RGB")
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    final_rgb.save(dest, format="PNG")
    return dest
