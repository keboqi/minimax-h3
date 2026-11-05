"""Measure and undo the global shift or zoom an image editor adds to a frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

WORK_EDGE = 512
# An editor's drift is small. Beyond this, the peak came from the edit and not from a real shift.
MAX_SHIFT = 0.1
MIN_CONFIDENCE = 0.02
ZOOMS = tuple(round(0.94 + 0.01 * index, 2) for index in range(13))


@dataclass(frozen=True)
class Drift:
    """Moving the edited frame by ``dx``, ``dy`` pixels and zooming it by ``zoom`` lines it up."""

    dx: int
    dy: int
    zoom: float
    confidence: float

    def plausible(self, width: int, height: int) -> bool:
        return (self.confidence >= MIN_CONFIDENCE and abs(self.dx) <= MAX_SHIFT * width
                and abs(self.dy) <= MAX_SHIFT * height)

    @property
    def visible(self) -> bool:
        return abs(self.dx) >= 2 or abs(self.dy) >= 2 or abs(self.zoom - 1) >= 0.005


def _transformed(image: Image.Image, zoom: float, dx: float, dy: float, fill: Image.Image | None = None) -> Image.Image:
    """``image`` zoomed about its centre, then moved; exposed edges come from ``fill``."""
    width, height = image.size
    x_centre, y_centre = width / 2, height / 2
    coefficients = (1 / zoom, 0, x_centre - (x_centre + dx) / zoom,
                    0, 1 / zoom, y_centre - (y_centre + dy) / zoom)
    moved = image.transform(image.size, Image.Transform.AFFINE, coefficients, Image.Resampling.BICUBIC)
    if fill is None:
        return moved
    covered = Image.new("L", image.size, 255).transform(image.size, Image.Transform.AFFINE, coefficients)
    return Image.composite(moved, fill, covered)


def _shift(reference: np.ndarray, moving: np.ndarray, window: np.ndarray) -> tuple[int, int, float]:
    """Phase correlation: the integer move that lines ``moving`` up with ``reference``."""
    spectrum = np.fft.fft2((reference - reference.mean()) * window) * np.conj(
        np.fft.fft2((moving - moving.mean()) * window)
    )
    correlation = np.fft.ifft2(spectrum / (np.abs(spectrum) + 1e-9)).real
    row, column = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    height, width = correlation.shape
    return (
        int(column - width if column > width // 2 else column),
        int(row - height if row > height // 2 else row),
        float(correlation[row, column]),
    )


def measure_drift(source: Image.Image, edited: Image.Image) -> Drift:
    source = source.convert("L")
    edited = edited.convert("L").resize(source.size, Image.Resampling.LANCZOS)
    scale = min(1.0, WORK_EDGE / max(source.size))
    size = (max(16, round(source.width * scale)), max(16, round(source.height * scale)))
    reference = np.asarray(source.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    small_edited = edited.resize(size, Image.Resampling.LANCZOS)
    window = np.outer(np.hanning(size[1]), np.hanning(size[0])).astype(np.float32)
    best = (0, 0, 1.0, -1.0)
    for zoom in ZOOMS:
        moving = np.asarray(_transformed(small_edited, zoom, 0, 0), dtype=np.float32)
        dx, dy, peak = _shift(reference, moving, window)
        if peak > best[3]:
            best = (dx, dy, zoom, peak)
    dx, dy, zoom, peak = best
    return Drift(round(dx / scale), round(dy / scale), zoom, peak)


def align_image(source: Image.Image, edited: Image.Image, drift: Drift) -> Image.Image:
    """The edited frame on the source's canvas, moved back into place."""
    source = source.convert("RGB")
    edited = edited.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    return _transformed(edited, drift.zoom, drift.dx, drift.dy, fill=source)


def comparison(source: Image.Image, edited: Image.Image, difference: bool = False) -> Image.Image:
    source = source.convert("RGB")
    edited = edited.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    return ImageChops.difference(source, edited) if difference else Image.blend(source, edited, 0.5)


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
