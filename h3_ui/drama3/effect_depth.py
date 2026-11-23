"""Estimate, cache, and inspect Depth Pro 2 metric depth maps for Sound Stage stills."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from shutil import copyfile
import statistics
from tempfile import TemporaryDirectory
from urllib.parse import quote
from uuid import uuid4

from PIL import Image

from h3_app.drama3.effect_enhancement import revision
from h3_app.errors import H3Error
from h3_app.graph import Graph
from h3_app.jobs import check_cancelled
from h3_models import (
    DEFAULT_SOUNDSTAGE_DEPTH_MODEL,
    MODEL_SPECS,
    SOUNDSTAGE_DEPTH_FOCAL_KEY,
    SOUNDSTAGE_DEPTH_MODEL_KEYS,
)

DEPTH_NODE = "H3EstimateDepthPro2"
DEPTH_FOLDER = ".effect-depth"
FLAT_DEPTH_STYLES = frozenset({
    "Pixel art",
    "Flat illustration",
    "Isometric illustration",
    "Minimal vector",
    "Editorial illustration",
})
STYLIZED_DEPTH_STYLES = frozenset({
    "Anime",
    "Japanese comic",
    "American comic",
    "2D cartoon",
    "Watercolor",
    "Chinese ink painting",
    "Children's book",
})
_LOG = logging.getLogger(__name__)
_DEPTH_CACHE_VERSION = 1


def normalize_depth_model(value: str | None) -> str:
    return value if value in SOUNDSTAGE_DEPTH_MODEL_KEYS else DEFAULT_SOUNDSTAGE_DEPTH_MODEL


def depth_key(depth_model: str | None = None) -> str:
    preset = normalize_depth_model(depth_model)
    model_key = SOUNDSTAGE_DEPTH_MODEL_KEYS[preset]
    return revision([
        MODEL_SPECS[model_key].expected_sha256,
        MODEL_SPECS[SOUNDSTAGE_DEPTH_FOCAL_KEY].expected_sha256,
        _DEPTH_CACHE_VERSION,
    ])[:16]


def artifact_paths(folder: Path | str, image: str, depth_model: str | None = None) -> tuple[Path, Path]:
    key = depth_key(depth_model)
    base = Path(folder) / DEPTH_FOLDER
    return base / f"{image}.{key}.depth.png", base / f"{image}.{key}.depth.json"


def _load_depth(json_path: Path) -> dict | None:
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    png_name = meta.get("png") or f"{json_path.stem}.png"
    if not isinstance(png_name, str):
        return None
    png_path = json_path.parent / png_name
    if not png_path.is_file():
        return None
    try:
        if float(meta.get("inv_max", 0.0)) <= 0.0:
            return None
    except (TypeError, ValueError):
        return None
    return {"png": png_name, "meta": meta}


def _sort_key(path: Path) -> tuple[int, str]:
    try:
        return (path.stat().st_mtime_ns, path.name)
    except OSError:
        return (-1, path.name)


def read_depth(
    folder: Path | str,
    image: str,
    depth_model: str | None = None,
    *,
    exact: bool = False,
) -> dict | None:
    _, exact_json = artifact_paths(folder, image, depth_model)
    if (loaded := _load_depth(exact_json)) is not None:
        return loaded
    if exact:
        return None
    depth_dir = Path(folder) / DEPTH_FOLDER
    if not depth_dir.is_dir():
        return None
    pattern = re.compile(rf"{re.escape(str(image))}\.[0-9a-f]{{16}}\.depth\.json")
    try:
        candidates = [
            entry
            for entry in depth_dir.iterdir()
            if entry != exact_json and pattern.fullmatch(entry.name) and entry.is_file()
        ]
    except OSError:
        return None
    for candidate in sorted(candidates, key=_sort_key, reverse=True):
        if (loaded := _load_depth(candidate)) is not None:
            return loaded
    return None


def depth_url(project: str, filename: str) -> str:
    return f"/soundstage/api/video/effect-depth?name={quote(project)}&file={quote(filename)}"


def effective_meta(meta: dict | None, style_preset: str | None = None) -> dict | None:
    if not isinstance(meta, dict):
        return None
    out = dict(meta)
    try:
        auto_strength = float(out.get("auto_strength") if out.get("auto_strength") is not None else 1.0)
    except (TypeError, ValueError):
        auto_strength = 1.0
    if style_preset in FLAT_DEPTH_STYLES:
        out["depth_strength"] = 0.0
        out["parallax_scale"] = 0.0
    elif style_preset in STYLIZED_DEPTH_STYLES:
        out["depth_strength"] = round(min(1.0, max(0.0, auto_strength)), 3)
        out["parallax_scale"] = 0.65
    else:
        out["depth_strength"] = round(min(1.0, max(0.0, auto_strength)), 3)
        out["parallax_scale"] = 1.0
    return out


def prepare(services, depth_model: str | None = None) -> None:
    preset = normalize_depth_model(depth_model)
    model_key = SOUNDSTAGE_DEPTH_MODEL_KEYS[preset]
    if DEPTH_NODE not in services.execution.object_info():
        raise H3Error(f"Update ComfyUI and the bundled H3 nodes for Depth Pro 2 ({DEPTH_NODE} missing).")
    services.models.ensure_model_keys((model_key, SOUNDSTAGE_DEPTH_FOCAL_KEY))


def estimate_still(path: Path, depth_model: str, services, runtime) -> dict:
    """Estimate and atomically commit a still's RG8 depth PNG and anchor JSON."""
    preset = normalize_depth_model(depth_model)
    model_key = SOUNDSTAGE_DEPTH_MODEL_KEYS[preset]
    path = Path(path)
    folder = path.parent
    png_target, json_target = artifact_paths(folder, path.name, preset)
    png_target.parent.mkdir(parents=True, exist_ok=True)
    client = uuid4().hex
    output_folder = Path(runtime.output_dir) / "h3" / "soundstage_depth" / client
    staged = services.media.stage_file(str(path), "soundstage_depth", reuse=True)

    graph = Graph()
    source = graph.add("LoadImage", image=staged)
    node = graph.add(
        DEPTH_NODE,
        image=Graph.out(source),
        preset=preset,
        model_name=MODEL_SPECS[model_key].local_name,
        focal_model_name=MODEL_SPECS[SOUNDSTAGE_DEPTH_FOCAL_KEY].local_name,
        filename_prefix=f"h3/soundstage_depth/{client}/depth",
    )

    try:
        prompt = services.execution.submit_prompt(graph.nodes, client)
        for _ in services.execution.poll_comfy_progress(prompt, graph.nodes):
            check_cancelled()
        history = services.execution.wait_for_history(prompt)
        entries = (history.get("outputs", {}).get(node) or {}).get("depth_maps", [])
        if not entries:
            raise H3Error("Depth Pro 2 produced no depth output for this still.")
        entry = entries[0]
        src_png = output_folder / Path(entry["filename"]).name
        meta = entry.get("meta")
        if not src_png.is_file() or not isinstance(meta, dict):
            raise H3Error("Depth Pro 2 output files were incomplete.")
        check_cancelled()
        stored_meta = {**meta, "png": png_target.name}
        with TemporaryDirectory(dir=png_target.parent) as temporary:
            scratch = Path(temporary)
            tmp_png = scratch / png_target.name
            tmp_json = scratch / json_target.name
            copyfile(src_png, tmp_png)
            tmp_json.write_text(json.dumps(stored_meta, ensure_ascii=False), encoding="utf-8")
            tmp_png.replace(png_target)
            tmp_json.replace(json_target)
        return {"png": png_target.name, "meta": stored_meta}
    finally:
        if output_folder.is_dir():
            for asset in output_folder.glob("*.png"):
                asset.unlink(missing_ok=True)
            try:
                output_folder.rmdir()
            except OSError:
                pass


def missing_stills(folder: Path | str, images, depth_model: str | None) -> list[str]:
    """Return deduplicated existing image filenames that lack exact depth artifacts."""
    root = Path(folder)
    preset = normalize_depth_model(depth_model)
    unique = list(dict.fromkeys(str(name) for name in images if name))
    return [
        name for name in unique
        if (root / name).is_file() and read_depth(root, name, preset, exact=True) is None
    ]


def ensure_stills_depth(
    folder: Path | str,
    images,
    depth_model: str | None,
    services,
    runtime,
    *,
    on_progress=None,
) -> int:
    """Backfill Depth Pro 2 depth maps for any stills missing exact artifacts."""
    if services is None:
        return 0
    folder = Path(folder)
    preset = normalize_depth_model(depth_model)
    missing = missing_stills(folder, images, preset)
    if not missing:
        return 0
    try:
        prepare(services, preset)
    except (H3Error, OSError) as exc:
        check_cancelled()
        _LOG.warning("Skipping Soundstage Depth Pro 2 estimation: %s", exc)
        return 0

    completed = 0
    for index, name in enumerate(missing, 1):
        check_cancelled()
        if on_progress is not None:
            on_progress(index, len(missing))
        try:
            estimate_still(folder / name, preset, services, runtime)
            completed += 1
        except (H3Error, OSError, ValueError) as exc:
            check_cancelled()
            _LOG.warning("Depth Pro 2 failed for %s: %s", name, exc)
    return completed


def region_depth_stats(
    folder: Path | str,
    depth_entry: dict | None,
    mask_filename: str | None,
) -> dict | None:
    """Compute a SAM region's median metric depth and band from cached RG8 depth PNG."""
    if not depth_entry or not mask_filename:
        return None
    folder = Path(folder)
    depth_path = folder / DEPTH_FOLDER / str(depth_entry.get("png") or "")
    mask_path = folder / ".effect-masks" / str(mask_filename)
    meta = depth_entry.get("meta") or {}
    if not depth_path.is_file() or not mask_path.is_file():
        return None
    try:
        inv_min = float(meta.get("inv_min", 0.0))
        inv_max = float(meta["inv_max"])
    except (KeyError, TypeError, ValueError):
        return None
    inv_span = inv_max - inv_min
    if inv_span <= 0.0:
        return None

    size = (128, 72)
    with Image.open(depth_path) as d_img, Image.open(mask_path) as m_img:
        depth_rgb = d_img.convert("RGB").resize(size, Image.Resampling.NEAREST)
        mask_rgba = m_img.convert("RGBA").resize(size, Image.Resampling.NEAREST)
        samples = []
        sky_samples = []
        d_pixels = getattr(depth_rgb, "get_flattened_data", depth_rgb.getdata)()
        m_pixels = getattr(mask_rgba, "get_flattened_data", mask_rgba.getdata)()
        for (r, g, b), (_mr, _mg, _mb, alpha) in zip(d_pixels, m_pixels):
            if alpha < 128:
                continue
            q = (r << 8) | g
            inv = inv_min + (q / 65535.0) * inv_span
            z = min(150.0, max(0.1, 1.0 / max(inv, 1e-4)))
            if b >= 200 and inv < 1e-3:
                sky_samples.append(150.0)
            else:
                samples.append(z)
    if not samples:
        samples = sky_samples
    if not samples:
        return None

    median_z = float(statistics.median(samples))
    z_p25 = float(meta.get("z_p25", 2.0))
    z_p75 = float(meta.get("z_p75", 10.0))
    if median_z <= z_p25 * 1.1:
        band = "foreground"
    elif median_z >= z_p75 * 0.9:
        band = "background"
    else:
        band = "midground"
    return {"depth_m": round(median_z, 2), "depth_band": band}
