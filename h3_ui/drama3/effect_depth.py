"""Estimate, cache, and inspect Depth Pro 2 metric depth maps for Sound Stage stills."""

from __future__ import annotations

from functools import lru_cache
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
    try:
        subject_z = float(out.get("subject_z") or 0.0)
        rack_z = float(out.get("rack_start_z") or 0.0)
        if subject_z > 0.1 and rack_z > 0.1:
            diopter_gap = abs((1.0 / rack_z) - (1.0 / subject_z))
            if diopter_gap < 0.12:
                z_p10 = float(out.get("z_p10") or subject_z * 0.6)
                z_p90 = float(out.get("z_p90") or subject_z * 2.2)
                near_candidate = min(z_p10, 1.0 / ((1.0 / subject_z) + 0.14))
                far_candidate = max(z_p90, 1.0 / max(1e-3, (1.0 / subject_z) - 0.14))
                if subject_z - z_p10 >= subject_z * 0.28 and near_candidate >= 0.25:
                    out["rack_start_z"] = round(max(0.25, near_candidate), 3)
                else:
                    out["rack_start_z"] = round(min(150.0, far_candidate), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
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


DEPTH_BATCH_SIZE = 8


def estimate_stills_batch(paths: list[Path], depth_model: str, services, runtime) -> list[dict]:
    """Estimate and commit up to ``DEPTH_BATCH_SIZE`` same-size stills in one GPU pass."""
    if not paths:
        return []
    if len(paths) == 1:
        return [estimate_still(paths[0], depth_model, services, runtime)]
    preset = normalize_depth_model(depth_model)
    model_key = SOUNDSTAGE_DEPTH_MODEL_KEYS[preset]
    paths = [Path(p) for p in paths]
    folder = paths[0].parent
    targets = [artifact_paths(folder, p.name, preset) for p in paths]
    targets[0][0].parent.mkdir(parents=True, exist_ok=True)
    client = uuid4().hex
    output_folder = Path(runtime.output_dir) / "h3" / "soundstage_depth" / client
    staged = [services.media.stage_file(str(p), "soundstage_depth", reuse=True) for p in paths]

    graph = Graph()
    batched = graph.add("LoadImage", image=staged[0])
    for staged_img in staged[1:]:
        nxt = graph.add("LoadImage", image=staged_img)
        batched = graph.add("ImageBatch", image1=Graph.out(batched), image2=Graph.out(nxt))
    node = graph.add(
        DEPTH_NODE,
        image=Graph.out(batched),
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
        if len(entries) != len(paths):
            raise H3Error("Depth Pro 2 batch output count did not match requested stills.")
        check_cancelled()
        committed = []
        with TemporaryDirectory(dir=targets[0][0].parent) as temporary:
            scratch = Path(temporary)
            staged_commits = []
            for entry, (png_target, json_target) in zip(entries, targets, strict=True):
                src_png = output_folder / Path(entry["filename"]).name
                meta = entry.get("meta")
                if not src_png.is_file() or not isinstance(meta, dict):
                    raise H3Error("Depth Pro 2 output files were incomplete.")
                stored_meta = {**meta, "png": png_target.name}
                tmp_png = scratch / png_target.name
                tmp_json = scratch / json_target.name
                copyfile(src_png, tmp_png)
                tmp_json.write_text(json.dumps(stored_meta, ensure_ascii=False), encoding="utf-8")
                staged_commits.append((tmp_png, png_target, tmp_json, json_target, stored_meta))
            for tmp_png, png_target, tmp_json, json_target, stored_meta in staged_commits:
                tmp_png.replace(png_target)
                tmp_json.replace(json_target)
                committed.append({"png": png_target.name, "meta": stored_meta})
        return committed
    finally:
        if output_folder.is_dir():
            for asset in output_folder.glob("*.png"):
                asset.unlink(missing_ok=True)
            try:
                output_folder.rmdir()
            except OSError:
                pass


def _group_same_size_batches(folder: Path, names: list[str], batch_size: int) -> list[list[Path]]:
    groups: dict[tuple[int, int], list[Path]] = {}
    singles: list[list[Path]] = []
    for name in names:
        path = folder / name
        try:
            with Image.open(path) as im:
                dims = im.size
        except OSError:
            singles.append([path])
            continue
        groups.setdefault(dims, []).append(path)
    batches: list[list[Path]] = []
    for items in groups.values():
        for start in range(0, len(items), batch_size):
            batches.append(items[start:start + batch_size])
    batches.extend(singles)
    return batches


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

    try:
        info = services.execution.object_info()
        can_batch = isinstance(info, (dict, set, frozenset, list, tuple)) and "ImageBatch" in info
    except Exception:
        can_batch = False

    completed = 0
    processed = 0
    total = len(missing)
    if can_batch and total > 1:
        for batch in _group_same_size_batches(folder, missing, DEPTH_BATCH_SIZE):
            check_cancelled()
            if on_progress is not None:
                on_progress(min(total, processed + 1), total)
            if len(batch) > 1:
                try:
                    done = estimate_stills_batch(batch, preset, services, runtime)
                    completed += len(done)
                    processed += len(batch)
                    continue
                except (H3Error, OSError, ValueError) as exc:
                    check_cancelled()
                    _LOG.warning("Depth Pro 2 batch failed (%s); retrying individually.", exc)
            for path in batch:
                check_cancelled()
                processed += 1
                if on_progress is not None:
                    on_progress(processed, total)
                try:
                    estimate_still(path, preset, services, runtime)
                    completed += 1
                except (H3Error, OSError, ValueError) as exc:
                    check_cancelled()
                    _LOG.warning("Depth Pro 2 failed for %s: %s", path.name, exc)
        return completed

    for index, name in enumerate(missing, 1):
        check_cancelled()
        if on_progress is not None:
            on_progress(index, total)
        try:
            estimate_still(folder / name, preset, services, runtime)
            completed += 1
        except (H3Error, OSError, ValueError) as exc:
            check_cancelled()
            _LOG.warning("Depth Pro 2 failed for %s: %s", name, exc)
    return completed


_STATS_GRID = (128, 72)


@lru_cache(maxsize=256)
def _decoded_depth_grid(depth_path_str: str, mtime_ns: int, size_bytes: int, inv_min: float, inv_span: float):
    del mtime_ns, size_bytes
    with Image.open(depth_path_str) as d_img:
        depth_rgb = d_img.convert("RGB").resize(_STATS_GRID, Image.Resampling.NEAREST)
        try:
            import numpy as np

            arr = np.asarray(depth_rgb, dtype=np.uint16)
            q = (arr[..., 0] << 8) | arr[..., 1]
            inv = inv_min + (q.astype(np.float32) / 65535.0) * inv_span
            z = np.clip(1.0 / np.maximum(inv, 1e-4), 0.1, 150.0)
            sky = (arr[..., 2] >= 200) & (inv < 1e-3)
            return ("np", z, sky)
        except ImportError:
            d_pixels = getattr(depth_rgb, "get_flattened_data", depth_rgb.getdata)()
            z_list = []
            sky_list = []
            for r, g, b in d_pixels:
                q = (r << 8) | g
                inv = inv_min + (q / 65535.0) * inv_span
                z_list.append(min(150.0, max(0.1, 1.0 / max(inv, 1e-4))))
                sky_list.append(b >= 200 and inv < 1e-3)
            return ("py", tuple(z_list), tuple(sky_list))


@lru_cache(maxsize=2048)
def _cached_region_depth_stats(
    depth_path_str: str,
    depth_mtime_ns: int,
    depth_size: int,
    mask_path_str: str,
    mask_mtime_ns: int,
    mask_size: int,
    inv_min: float,
    inv_span: float,
    z_p25: float,
    z_p75: float,
) -> tuple[float, str] | None:
    del mask_mtime_ns, mask_size
    kind, z_grid, sky_grid = _decoded_depth_grid(depth_path_str, depth_mtime_ns, depth_size, inv_min, inv_span)
    with Image.open(mask_path_str) as m_img:
        mask_alpha = m_img.convert("RGBA").resize(_STATS_GRID, Image.Resampling.NEAREST).getchannel("A")
        if kind == "np":
            import numpy as np

            fg = np.asarray(mask_alpha, dtype=np.uint8) >= 128
            if not fg.any():
                return None
            valid = fg & (~sky_grid)
            if valid.any():
                median_z = float(np.median(z_grid[valid]))
            else:
                median_z = 150.0
        else:
            alphas = getattr(mask_alpha, "get_flattened_data", mask_alpha.getdata)()
            samples = []
            sky_samples = []
            for z, is_sky, alpha in zip(z_grid, sky_grid, alphas):
                if alpha < 128:
                    continue
                if is_sky:
                    sky_samples.append(150.0)
                else:
                    samples.append(z)
            if not samples:
                samples = sky_samples
            if not samples:
                return None
            median_z = float(statistics.median(samples))

    if median_z <= z_p25 * 1.1:
        band = "foreground"
    elif median_z >= z_p75 * 0.9:
        band = "background"
    else:
        band = "midground"
    return (round(median_z, 2), band)


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
    try:
        d_stat = depth_path.stat()
        m_stat = mask_path.stat()
        inv_min = round(float(meta.get("inv_min", 0.0)), 6)
        inv_max = round(float(meta["inv_max"]), 6)
        z_p25 = round(float(meta.get("z_p25", 2.0)), 4)
        z_p75 = round(float(meta.get("z_p75", 10.0)), 4)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    inv_span = round(inv_max - inv_min, 6)
    if inv_span <= 0.0:
        return None
    try:
        res = _cached_region_depth_stats(
            str(depth_path), d_stat.st_mtime_ns, d_stat.st_size,
            str(mask_path), m_stat.st_mtime_ns, m_stat.st_size,
            inv_min, inv_span, z_p25, z_p75,
        )
    except OSError:
        return None
    if res is None:
        return None
    return {"depth_m": res[0], "depth_band": res[1]}
