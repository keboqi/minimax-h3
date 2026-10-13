"""Still-image NVIDIA enhancement using one isolated RGB/RGBA worker frame."""

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image, ImageCms, ImageOps

from . import nvidia_postprocess as backend
from .catalog import DLSS5_NR, NVIDIA_POSTPROCESS_OPTIONS
from .errors import H3Error
from .jobs import check_cancelled
from .nvidia_pipeline import stream_pipeline
from .provenance import read_snapshot, snapshot_path, write_snapshot


def load_image(source):
    """Normalize EXIF orientation and ICC color; retain alpha separately."""
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    with Image.open(source) as opened:
        if min(opened.size) < 2 or max(opened.size) > 8192:
            raise H3Error("Source dimensions must be from 2 to 8192 pixels per axis.")
        if getattr(opened, "n_frames", 1) != 1:
            raise H3Error("Choose a still image; animated images are unsupported.")
        if opened.mode not in {"RGB", "RGBA", "L", "LA", "P", "CMYK", "1"}:
            raise H3Error("NVIDIA image enhancement supports 8-bit SDR images only.")
        if opened.format == "PNG":
            with Path(source).open("rb") as handle:
                header = handle.read(25)
            if len(header) < 25 or header[24] > 8 or opened.info.get("gamma") == 1:
                raise H3Error("16-bit and linear PNG images are unsupported; convert to 8-bit sRGB first.")
        bits = getattr(opened, "tag_v2", {}).get(258, (8,))
        if max(bits if isinstance(bits, tuple) else (bits,)) > 8:
            raise H3Error("High-bit-depth images are unsupported; convert to 8-bit sRGB first.")
        image = ImageOps.exif_transpose(opened)
        alpha = image.convert("RGBA").getchannel("A") if (
            "A" in image.getbands() or "transparency" in image.info
        ) else None
        icc = image.info.get("icc_profile")
        if icc:
            try:
                profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                color = image if image.mode in {"RGB", "CMYK", "L"} else image.convert("RGB")
                rgb = ImageCms.profileToProfile(color, profile, srgb, outputMode="RGB")
            except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                raise H3Error("Could not convert the image's color profile to sRGB.") from exc
        else:
            rgb = image.convert("RGB")
        return rgb, alpha, srgb.tobytes()


def postprocess_image(source, method, resolution, settings_value, *, runtime):
    if method not in NVIDIA_POSTPROCESS_OPTIONS:
        raise H3Error("Unsupported NVIDIA image enhancement method.")
    settings = backend.NvidiaSettings.parse(settings_value)
    ready, message = backend.availability(method)
    if not ready:
        raise H3Error(message)
    check_cancelled()
    yield backend.NvidiaUpdate("Preparing image and color profile", total=1)
    rgb, alpha, icc = load_image(source)
    width, height = rgb.size
    out_width, out_height = backend.target_dimensions(
        method, width, height, resolution, settings, image=True,
    )
    if method == DLSS5_NR:
        yield backend.NvidiaUpdate("Checking DLSS 5 neural execution", total=1)
        backend.probe_runtime(method, runtime)
    channels = 4
    mode = "RGBA"
    raw = rgb.convert(mode).tobytes()
    command, cwd = backend.worker_command(method, width, height, out_width, out_height, settings, frames=1, motion=False)
    result = bytearray()
    with backend.worker_context(method, runtime) as env:
        for update in stream_pipeline(
            [command], input_size=len(raw), output_size=out_width * out_height * channels,
            timeout=runtime.generation_timeout, worker_env=env, worker_cwd=cwd,
            source_bytes=raw, capture=result,
        ):
            yield backend.NvidiaUpdate("Enhancing image", update.frames, 1)
    details = {
        "method": method, **asdict(settings), "media_type": "Image", "frames": 1,
        "source_resolution": f"{width}x{height}", "target_resolution": f"{out_width}x{out_height}",
        "format": "PNG", "color_profile": "sRGB", "alpha": "preserved" if alpha is not None else "none",
        **backend.runtime_details(method, update.diagnostics,
                                  upscaled=rgb.size != (out_width, out_height), motion=False),
    }
    enhanced = Image.frombytes(mode, (out_width, out_height), bytes(result)).convert("RGB")
    if alpha is not None:
        enhanced.putalpha(alpha.resize(enhanced.size, Image.Resampling.LANCZOS))
    category = "dlss5"
    root = runtime.outputs_dir / category
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".processing-", dir=root) as directory:
        pending = Path(directory) / "output.png"
        output = root / (Path(source).stem + "_" + Path(directory).name.removeprefix(".processing-") + ".png")
        check_cancelled()
        enhanced.save(pending, format="PNG", icc_profile=icc)
        with Image.open(pending) as saved:
            if saved.size != enhanced.size or saved.mode != enhanced.mode:
                raise H3Error("Enhanced image dimensions or transparency do not match the request.")
            saved.verify()
        check_cancelled()
        source_snapshot = read_snapshot(source) or {}
        published = False
        try:
            pending.replace(output)
            published = True
            write_snapshot(output, {"family": category, "settings": details,
                "source_asset_ids": [source_snapshot["asset_id"]] if source_snapshot.get("asset_id") else []})
        except BaseException:
            if published:
                output.unlink(missing_ok=True)
                snapshot_path(output).unlink(missing_ok=True)
            raise
        yield backend.NvidiaUpdate("Image enhancement complete", 1, 1, output)
