"""Opt-in image conditioning transforms; originals are never overwritten."""

from pathlib import Path
from PIL import Image, ImageOps

MODES = ("Input-derived", "Fit / pad", "Fill / crop")


def transform_canvas(source, destination, mode, width, height):
    if mode not in MODES:
        raise ValueError("Unknown image canvas mode.")
    if mode == "Input-derived":
        return str(source)
    width, height = int(width), int(height)
    if (
        not (256 <= width <= 4096 and 256 <= height <= 4096)
        or width % 32
        or height % 32
    ):
        raise ValueError(
            "Image canvas dimensions must be 256–4096 pixels, aligned to 32 pixels."
        )
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("A canvas transform must write a derived copy.")
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if mode == "Fit / pad":
            result = ImageOps.pad(
                image, (width, height), method=Image.Resampling.LANCZOS, color=(0, 0, 0)
            )
        else:
            result = ImageOps.fit(
                image, (width, height), method=Image.Resampling.LANCZOS
            )
        result.save(destination, format="PNG")
    return str(destination)


def apply_job_canvas(job, values):
    canvas = job.canvas_request
    if not canvas or canvas["mode"] == "Input-derived":
        return values
    if job.family != "h3":
        return values
    names = job.callback.job_input_names
    result = list(values)
    if (
        result[names.index("result_format")] != "Image"
        or result[names.index("mode")] != "First / last frame"
    ):
        return result
    first_index = names.index("first_image")
    if not result[first_index]:
        raise ValueError(
            "Upload a first frame before applying an image canvas transform."
        )
    target = Path(job.input_lease) / "canvas-first.png"
    result[first_index] = transform_canvas(
        result[first_index], target, canvas["mode"], canvas["width"], canvas["height"]
    )
    last_index = names.index("last_image")
    if result[last_index]:
        result[last_index] = transform_canvas(
            result[last_index],
            Path(job.input_lease) / "canvas-last.png",
            canvas["mode"],
            canvas["width"],
            canvas["height"],
        )
    return result
